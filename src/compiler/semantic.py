"""
Carbon Protocol SDK - Semantic Compiler (Main Router Logic)

This module implements the main CarbonCompiler class that orchestrates
the complete "Wake-on-Meaning" processing pipeline.

Reference: Carbon Protocol Research Paper
    "Semantic Compiler Architecture" - Section IV

Pipeline Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CARBON COMPILER PIPELINE                      │
    │                                                                  │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
    │  │  INPUT   │───►│ NEURONAL │───►│  SKILL   │───►│  OUTPUT  │ │
    │  │  TEXT    │    │  GATING  │    │ DATABASE │    │ BYTECODE │ │
    │  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
    │                       │                │                        │
    │                       ▼                ▼                        │
    │                   [IDLE?]          [MATCH?]                     │
    │                       │                │                        │
    │                  No Intent         OP_MACRO                     │
    │                  OP_IDLE          (Deterministic)               │
    │                       │                │                        │
    │                       │          No Match                       │
    │                       │           OP_GEN                        │
    │                       │         (Generative)                    │
    │                       ▼                ▼                        │
    │              ┌─────────────────────────────┐                    │
    │              │     CARBON BYTECODE         │                    │
    │              │  (Instructions for Runtime) │                    │
    │              └─────────────────────────────┘                    │
    └─────────────────────────────────────────────────────────────────┘

Energy Model:
    - IDLE Path: ~0 FLOPs (no LLM needed)
    - MACRO Path: ~1K FLOPs (template expansion)
    - GEN Path: ~1T FLOPs (full LLM inference)
    
    Goal: Maximize IDLE + MACRO, minimize GEN.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum, auto

from ..c_isa import (
    CarbonOpCode,
    CarbonInstruction,
    InstructionSequence,
    CarbonBytecode,
    BytecodeBuilder,
    OP_IDLE,
    OP_GEN,
    OP_MACRO,
)
from ..ingestion import (
    IntentDetector,
    IntentResult,
    DetectorConfig,
)
from ..registry import (
    SkillDB,
    SkillMatch,
    SkillDBConfig,
    GenerativeTrace,
)


class ExecutionPath(Enum):
    """
    Classification of execution paths through the compiler.
    
    The Carbon Protocol distinguishes three fundamental paths:
    - IDLE: No significant intent detected, no action needed
    - DETERMINISTIC: Skill matched, execute macro (no LLM)
    - GENERATIVE: No match, fallback to LLM generation
    """
    IDLE = auto()           # Low intent, no action
    DETERMINISTIC = auto()  # Skill matched, execute macro
    GENERATIVE = auto()     # No match, fallback to LLM


@dataclass
class CompilationResult:
    """
    Result of compiling an input through the Carbon pipeline.
    
    Attributes:
        bytecode: The compiled Carbon bytecode.
        path: The execution path taken.
        intent: Detected intent information.
        skill_match: Skill database lookup result.
        compilation_time_ms: Time taken to compile.
        input_hash: Hash of the input for caching.
        metadata: Additional metadata.
    """
    bytecode: CarbonBytecode
    path: ExecutionPath
    intent: IntentResult
    skill_match: SkillMatch | None
    compilation_time_ms: float
    input_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_idle(self) -> bool:
        """Check if result is IDLE (no action needed)."""
        return self.path == ExecutionPath.IDLE
    
    @property
    def is_deterministic(self) -> bool:
        """Check if result is deterministic (macro execution)."""
        return self.path == ExecutionPath.DETERMINISTIC
    
    @property
    def is_generative(self) -> bool:
        """Check if result requires LLM generation."""
        return self.path == ExecutionPath.GENERATIVE
    
    @property
    def primary_opcode(self) -> CarbonOpCode:
        """Get the primary opcode for this result."""
        if self.bytecode.sequence.instructions:
            return self.bytecode.sequence.instructions[0].opcode
        return CarbonOpCode.IDLE
    
    @property
    def estimated_flops(self) -> int:
        """Estimated FLOPs for execution."""
        return self.bytecode.sequence.total_flops
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path.name,
            "intent": self.intent.to_dict(),
            "skill_match": {
                "found": self.skill_match.found if self.skill_match else False,
                "match_type": self.skill_match.match_type if self.skill_match else None,
            },
            "compilation_time_ms": self.compilation_time_ms,
            "input_hash": self.input_hash,
            "primary_opcode": self.primary_opcode.name,
            "estimated_flops": self.estimated_flops,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CompilerConfig:
    """
    Configuration for the CarbonCompiler.
    
    Attributes:
        enable_gating: Enable neuromorphic gating (can disable for testing).
        enable_skill_lookup: Enable skill database lookup.
        enable_trace_recording: Record generative traces for promotion.
        idle_threshold: Minimum intent confidence to proceed.
        fallback_max_tokens: Default max tokens for generative fallback.
    """
    enable_gating: bool = True
    enable_skill_lookup: bool = True
    enable_trace_recording: bool = True
    idle_threshold: float = 0.3
    fallback_max_tokens: int = 1024


DEFAULT_COMPILER_CONFIG = CompilerConfig()


class CarbonCompiler:
    """
    Main Carbon Protocol Compiler - orchestrates the full pipeline.
    
    The CarbonCompiler is the central "router" component that processes 
    user input through the "Wake-on-Meaning" architecture:
    
    1. **Neuromorphic Gating**: Use LIF neurons to detect intent
    2. **Skill Lookup**: Check if a deterministic skill matches (Trie/Aho-Corasick)
    3. **Path Selection**: Route to IDLE, MACRO, or GEN path
    4. **Bytecode Generation**: Produce executable instructions
    
    Reference: Carbon Protocol Research Paper
        "Semantic Compiler Pipeline" - Section IV
    
    Example:
        >>> compiler = CarbonCompiler.create_default()
        >>> 
        >>> # Process input
        >>> result = compiler.process("Write a Python script to parse JSON")
        >>> 
        >>> if result.is_deterministic:
        ...     print("Using cached skill:", result.primary_opcode)
        ... elif result.is_generative:
        ...     print("Falling back to LLM")
        ... else:
        ...     print("No action needed (IDLE)")
        >>> 
        >>> # After LLM generates response, record for promotion
        >>> if result.is_generative:
        ...     compiler.record_generation(
        ...         input_text="Write a Python script to parse JSON",
        ...         output_text="import json\\n...",
        ...         tokens_out=150,
        ...     )
    
    Complexity:
        O(L + k) where L = input length, k = number of intent neurons
        
        This is dramatically more efficient than:
        - BERT classifier: O(L * d²) where d = hidden dimension
        - Vector similarity: O(L * n) where n = corpus size
    """
    
    def __init__(
        self,
        intent_detector: IntentDetector | None = None,
        skill_db: SkillDB | None = None,
        config: CompilerConfig | None = None,
    ) -> None:
        """
        Initialize the CarbonCompiler.
        
        Args:
            intent_detector: IntentDetector for neuromorphic gating.
            skill_db: SkillDB for skill lookup (Trie/Aho-Corasick).
            config: Compiler configuration.
        """
        self._config = config or DEFAULT_COMPILER_CONFIG
        self._detector = intent_detector or IntentDetector.create_default()
        self._registry = skill_db or SkillDB()
        
        # Statistics
        self._stats = {
            "total_compilations": 0,
            "idle_count": 0,
            "deterministic_count": 0,
            "generative_count": 0,
            "total_time_ms": 0.0,
        }
    
    @classmethod
    def create_default(cls) -> CarbonCompiler:
        """
        Factory method to create a fully configured compiler.
        
        Returns:
            CarbonCompiler with default components.
        """
        return cls(
            intent_detector=IntentDetector.create_default(),
            skill_db=SkillDB(),
            config=DEFAULT_COMPILER_CONFIG,
        )
    
    @property
    def config(self) -> CompilerConfig:
        """Get compiler configuration."""
        return self._config
    
    @property
    def detector(self) -> IntentDetector:
        """Get the intent detector."""
        return self._detector
    
    @property
    def registry(self) -> SkillDB:
        """Get the skill database."""
        return self._registry
    
    @property
    def skill_db(self) -> SkillDB:
        """Get the skill database (alias for registry)."""
        return self._registry
    
    def process(self, user_input: str) -> CompilationResult:
        """
        Process user input through the Carbon pipeline.
        
        This is the main entry point for the compiler. It:
        1. Performs neuromorphic intent detection
        2. Checks the skill database for matches
        3. Generates appropriate bytecode
        
        Args:
            user_input: The user's input text.
            
        Returns:
            CompilationResult with bytecode and metadata.
        """
        start_time = time.perf_counter()
        
        # Compute input hash for caching
        input_hash = hashlib.sha256(
            user_input.lower().strip().encode('utf-8')
        ).hexdigest()[:16]
        
        # Step 1: Neuromorphic Gating
        if self._config.enable_gating:
            intent = self._detector.detect(user_input)
        else:
            # Bypass gating (for testing)
            intent = IntentResult(
                primary_intent="bypass",
                fired_intents=["bypass"],
                activation_levels={},
                signals_extracted=0,
                is_idle=False,
                confidence=1.0,
            )
        
        # Check for IDLE (no significant intent)
        if intent.is_idle and intent.confidence < self._config.idle_threshold:
            bytecode = self._build_idle_bytecode(user_input)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._update_stats(ExecutionPath.IDLE, elapsed_ms)
            
            return CompilationResult(
                bytecode=bytecode,
                path=ExecutionPath.IDLE,
                intent=intent,
                skill_match=None,
                compilation_time_ms=elapsed_ms,
                input_hash=input_hash,
            )
        
        # Step 2: Skill Database Lookup (Trie/Aho-Corasick)
        skill_match: SkillMatch | None = None
        
        if self._config.enable_skill_lookup:
            skill_match = self._registry.lookup(user_input)
        
        # Step 3: Path Selection and Bytecode Generation
        if skill_match and skill_match.found:
            # DETERMINISTIC PATH - skill matched
            bytecode = self._build_deterministic_bytecode(
                user_input, intent, skill_match
            )
            path = ExecutionPath.DETERMINISTIC
        else:
            # GENERATIVE PATH - no match, fallback to LLM
            bytecode = self._build_generative_bytecode(user_input, intent)
            path = ExecutionPath.GENERATIVE
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(path, elapsed_ms)
        
        return CompilationResult(
            bytecode=bytecode,
            path=path,
            intent=intent,
            skill_match=skill_match,
            compilation_time_ms=elapsed_ms,
            input_hash=input_hash,
            metadata={
                "intent_label": intent.primary_intent,
                "intent_confidence": intent.confidence,
            },
        )
    
    def _build_idle_bytecode(self, input_text: str) -> CarbonBytecode:
        """Build bytecode for IDLE path."""
        builder = BytecodeBuilder()
        builder.idle()
        return builder.build(input_text)
    
    def _build_deterministic_bytecode(
        self,
        input_text: str,
        intent: IntentResult,
        match: SkillMatch,
    ) -> CarbonBytecode:
        """Build bytecode for deterministic (MACRO) path."""
        builder = BytecodeBuilder()
        
        # Add the matched skill's instruction
        if match.instruction:
            builder.custom(match.instruction)
        else:
            # Fallback to generic macro
            builder.macro(
                name=match.skill.name if match.skill else "unknown",
            )
        
        return builder.with_metadata(
            "skill_id", match.skill.skill_id if match.skill else None
        ).with_metadata(
            "match_type", match.match_type
        ).with_metadata(
            "match_score", match.score
        ).build(input_text)
    
    def _build_generative_bytecode(
        self,
        input_text: str,
        intent: IntentResult,
    ) -> CarbonBytecode:
        """Build bytecode for generative (LLM) path."""
        builder = BytecodeBuilder()
        
        # Add context loading if intent suggests it
        if intent.primary_intent in ("code", "debug", "scaffold"):
            builder.load("workspace_context")
        
        # Add safety check
        builder.check("content_safety")
        
        # Generative instruction
        builder.generate(
            max_tokens=self._config.fallback_max_tokens,
            intent=intent.primary_intent or "general",
        )
        
        return builder.with_metadata(
            "intent", intent.primary_intent
        ).with_metadata(
            "confidence", intent.confidence
        ).build(input_text)
    
    def _update_stats(self, path: ExecutionPath, elapsed_ms: float) -> None:
        """Update compilation statistics."""
        self._stats["total_compilations"] += 1
        self._stats["total_time_ms"] += elapsed_ms
        
        if path == ExecutionPath.IDLE:
            self._stats["idle_count"] += 1
        elif path == ExecutionPath.DETERMINISTIC:
            self._stats["deterministic_count"] += 1
        else:
            self._stats["generative_count"] += 1
    
    def record_generation(
        self,
        input_text: str,
        output_text: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> GenerativeTrace | None:
        """
        Record a generative interaction for potential promotion.
        
        Call this after the LLM generates a response to enable
        the self-optimization feature.
        
        Args:
            input_text: Original user input.
            output_text: LLM-generated response.
            tokens_in: Input token count.
            tokens_out: Output token count.
            latency_ms: Response latency.
            success: Whether the interaction was successful.
            
        Returns:
            The recorded GenerativeTrace, or None if disabled.
        """
        if not self._config.enable_trace_recording:
            return None
        
        return self._registry.record_generative_trace(
            input_text=input_text,
            output_text=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            success=success,
        )
    
    def add_skill(
        self,
        name: str,
        pattern: str,
        template: str,
        pattern_type: str = "substring",
    ) -> None:
        """
        Add a skill to the database (convenience method).
        
        Args:
            name: Skill name.
            pattern: Trigger pattern.
            template: Output template.
            pattern_type: Pattern type.
        """
        self._registry.add_macro_skill(name, pattern, template, pattern_type)
    
    def add_scaffold_skill(
        self,
        name: str,
        pattern: str,
        lang: str,
        arch: str = "default",
        db: str | None = None,
    ) -> None:
        """
        Add a scaffolding skill (convenience method).
        
        Args:
            name: Skill name.
            pattern: Trigger pattern.
            lang: Programming language.
            arch: Architecture pattern.
            db: Optional database.
        """
        self._registry.add_scaffold_skill(name, pattern, lang, arch, db)
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get compiler statistics.
        
        Returns:
            Dictionary with compilation statistics.
        """
        total = self._stats["total_compilations"]
        
        return {
            "total_compilations": total,
            "idle_count": self._stats["idle_count"],
            "deterministic_count": self._stats["deterministic_count"],
            "generative_count": self._stats["generative_count"],
            "idle_rate": self._stats["idle_count"] / total if total > 0 else 0,
            "deterministic_rate": self._stats["deterministic_count"] / total if total > 0 else 0,
            "generative_rate": self._stats["generative_count"] / total if total > 0 else 0,
            "avg_compilation_ms": self._stats["total_time_ms"] / total if total > 0 else 0,
            "registry": self._registry.get_stats(),
            "detector": self._detector.get_stats(),
        }
    
    def get_efficiency_report(self) -> dict[str, Any]:
        """
        Generate an efficiency report showing energy savings.
        
        Returns:
            Dictionary with efficiency metrics.
        """
        stats = self.get_stats()
        total = stats["total_compilations"]
        
        if total == 0:
            return {"message": "No compilations recorded"}
        
        # Estimate FLOPs saved
        # Baseline: all generative = total * 1T FLOPs
        # Actual: idle * 0 + deterministic * 1K + generative * 1T
        baseline_flops = total * 1_000_000_000_000
        actual_flops = (
            stats["idle_count"] * 0 +
            stats["deterministic_count"] * 1_000 +
            stats["generative_count"] * 1_000_000_000_000
        )
        
        flops_saved = baseline_flops - actual_flops
        efficiency_gain = flops_saved / baseline_flops if baseline_flops > 0 else 0
        
        return {
            "total_compilations": total,
            "baseline_flops": baseline_flops,
            "actual_flops": actual_flops,
            "flops_saved": flops_saved,
            "efficiency_gain": efficiency_gain,
            "efficiency_gain_pct": efficiency_gain * 100,
            "deterministic_rate_pct": stats["deterministic_rate"] * 100,
            "idle_rate_pct": stats["idle_rate"] * 100,
        }
    
    def __repr__(self) -> str:
        stats = self._stats
        return (f"CarbonCompiler(compilations={stats['total_compilations']}, "
                f"deterministic={stats['deterministic_count']}, "
                f"generative={stats['generative_count']})")


# Convenience function for quick usage
def compile_input(text: str) -> CompilationResult:
    """
    Quick compilation of input text using default compiler.
    
    Args:
        text: Input text to compile.
        
    Returns:
        CompilationResult.
    """
    compiler = CarbonCompiler.create_default()
    return compiler.process(text)
