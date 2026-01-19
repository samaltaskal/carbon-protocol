"""
Carbon Protocol SDK - Macro Store

This module manages the storage and evolution of macros - deterministic
skill sequences that have been promoted from generative traces.

Reference: Carbon Protocol Research Paper
    "Generative-to-Deterministic Evolution" - Section 5.2

Evolution Process:
    1. Novel input triggers OP_GEN (generative path)
    2. LLM response is recorded as "GenerativeTrace"
    3. Trace metadata includes: input, output, latency, tokens
    4. After meeting PromotionCriteria, trace becomes a "Macro"
    5. Future matches execute OP_MACRO (deterministic path)

This models skill acquisition in biological systems:
    Novice (slow, conscious) → Expert (fast, automatic)
"""

from __future__ import annotations

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from ..c_isa.instruction import MacroDefinition, InstructionSequence, CarbonInstruction
from ..c_isa.opcodes import CarbonOpCode


@dataclass
class GenerativeTrace:
    """
    Record of a generative (LLM) interaction.
    
    Traces are accumulated and analyzed for patterns. Repeated
    traces with high success rates are candidates for promotion
    to deterministic macros.
    
    Attributes:
        trace_id: Unique identifier for this trace.
        input_text: Original user input.
        input_hash: Hash of normalized input for deduplication.
        output_text: LLM-generated response.
        tokens_in: Input token count.
        tokens_out: Output token count.
        latency_ms: Response latency in milliseconds.
        timestamp: Unix timestamp of the interaction.
        success: Whether the interaction was successful (user feedback).
        metadata: Additional metadata.
    """
    trace_id: str
    input_text: str
    input_hash: str
    output_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        input_text: str,
        output_text: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> GenerativeTrace:
        """
        Factory method to create a trace with auto-generated ID and hash.
        
        Args:
            input_text: Original user input.
            output_text: LLM response.
            tokens_in: Input token count.
            tokens_out: Output token count.
            latency_ms: Response latency.
            success: Whether successful.
            
        Returns:
            New GenerativeTrace instance.
        """
        # Normalize and hash input
        normalized = input_text.lower().strip()
        input_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
        
        # Generate trace ID
        trace_id = f"trace_{int(time.time() * 1000)}_{input_hash[:8]}"
        
        return cls(
            trace_id=trace_id,
            input_text=input_text,
            input_hash=input_hash,
            output_text=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            success=success,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trace_id": self.trace_id,
            "input_text": self.input_text,
            "input_hash": self.input_hash,
            "output_text": self.output_text,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "success": self.success,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerativeTrace:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass(frozen=True)
class PromotionCriteria:
    """
    Criteria for promoting a trace pattern to a macro.
    
    Attributes:
        min_occurrences: Minimum number of similar traces.
        min_success_rate: Minimum success rate (0.0-1.0).
        max_variance: Maximum variance in outputs (0.0-1.0).
        min_age_hours: Minimum age of oldest trace in hours.
    """
    min_occurrences: int = 3
    min_success_rate: float = 0.9
    max_variance: float = 0.2
    min_age_hours: float = 0.0


DEFAULT_PROMOTION_CRITERIA = PromotionCriteria()


class MacroStore:
    """
    Storage and evolution manager for Carbon macros.
    
    The MacroStore tracks generative traces, identifies patterns,
    and promotes recurring successful patterns to deterministic macros.
    
    Reference: Carbon Protocol Research Paper
        "Self-Optimizing Skill Registry" - Section 5.2
    
    Key Functions:
        1. Record generative traces
        2. Group traces by input similarity
        3. Evaluate promotion criteria
        4. Promote to macros when criteria met
        5. Persist macros for future use
    
    Example:
        >>> store = MacroStore()
        >>> 
        >>> # Record successful interactions
        >>> trace = GenerativeTrace.create(
        ...     input_text="Write hello world in Python",
        ...     output_text="print('Hello, World!')",
        ...     tokens_in=10, tokens_out=15,
        ... )
        >>> store.record_trace(trace)
        >>> 
        >>> # Check for promotion candidates
        >>> candidates = store.get_promotion_candidates()
        >>> for input_hash, traces in candidates:
        ...     macro = store.promote_to_macro(traces)
    """
    
    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        persist_path: Path | None = None,
    ) -> None:
        """
        Initialize the MacroStore.
        
        Args:
            criteria: Promotion criteria. Uses defaults if None.
            persist_path: Optional path for persistence.
        """
        self._criteria = criteria or DEFAULT_PROMOTION_CRITERIA
        self._persist_path = persist_path
        
        # Traces grouped by input hash
        self._traces: dict[str, list[GenerativeTrace]] = {}
        
        # Promoted macros
        self._macros: dict[str, MacroDefinition] = {}
        
        # Load persisted data if available
        if persist_path and persist_path.exists():
            self._load_from_disk()
    
    @property
    def criteria(self) -> PromotionCriteria:
        """Get promotion criteria."""
        return self._criteria
    
    @property
    def trace_count(self) -> int:
        """Total number of traces."""
        return sum(len(traces) for traces in self._traces.values())
    
    @property
    def macro_count(self) -> int:
        """Number of promoted macros."""
        return len(self._macros)
    
    @property
    def unique_patterns(self) -> int:
        """Number of unique input patterns."""
        return len(self._traces)
    
    def record_trace(self, trace: GenerativeTrace) -> None:
        """
        Record a generative trace.
        
        Args:
            trace: The GenerativeTrace to record.
        """
        input_hash = trace.input_hash
        
        if input_hash not in self._traces:
            self._traces[input_hash] = []
        
        self._traces[input_hash].append(trace)
        
        # Auto-persist if configured
        if self._persist_path:
            self._save_to_disk()
    
    def get_traces(self, input_hash: str) -> list[GenerativeTrace]:
        """
        Get all traces for a given input hash.
        
        Args:
            input_hash: The input pattern hash.
            
        Returns:
            List of traces (may be empty).
        """
        return self._traces.get(input_hash, [])
    
    def get_promotion_candidates(self) -> list[tuple[str, list[GenerativeTrace]]]:
        """
        Get trace groups that meet promotion criteria.
        
        Returns:
            List of (input_hash, traces) tuples that are candidates.
        """
        candidates = []
        
        for input_hash, traces in self._traces.items():
            # Skip if already promoted
            if input_hash in self._macros:
                continue
            
            # Check occurrence count
            if len(traces) < self._criteria.min_occurrences:
                continue
            
            # Check success rate
            success_count = sum(1 for t in traces if t.success)
            success_rate = success_count / len(traces)
            if success_rate < self._criteria.min_success_rate:
                continue
            
            # Check age
            oldest = min(t.timestamp for t in traces)
            age_hours = (time.time() - oldest) / 3600
            if age_hours < self._criteria.min_age_hours:
                continue
            
            candidates.append((input_hash, traces))
        
        return candidates
    
    def promote_to_macro(
        self,
        traces: list[GenerativeTrace],
        name: str | None = None,
    ) -> MacroDefinition:
        """
        Promote a trace group to a deterministic macro.
        
        This is the key "evolution" operation that converts a
        generative pattern to a deterministic one.
        
        Args:
            traces: List of similar traces to promote.
            name: Optional macro name. Auto-generated if None.
            
        Returns:
            The created MacroDefinition.
            
        Raises:
            ValueError: If traces list is empty.
        """
        if not traces:
            raise ValueError("Cannot promote empty trace list")
        
        # Use most recent successful trace as template
        successful_traces = [t for t in traces if t.success]
        if not successful_traces:
            successful_traces = traces
        
        template = successful_traces[-1]
        
        # Generate name if not provided
        if name is None:
            name = f"macro_{template.input_hash[:8]}"
        
        # Create instruction sequence
        # The macro will output the template response
        sequence = InstructionSequence([
            CarbonInstruction(
                opcode=CarbonOpCode.MACRO,
                args={"template": template.output_text},
                context=template.input_text,
            )
        ])
        
        # Create macro definition
        macro = MacroDefinition(
            name=name,
            pattern=template.input_text.lower().strip(),
            sequence=sequence,
            hit_count=len(traces),
            success_rate=sum(1 for t in traces if t.success) / len(traces),
            source_trace=template.trace_id,
            metadata={
                "input_hash": template.input_hash,
                "avg_tokens_in": sum(t.tokens_in for t in traces) / len(traces),
                "avg_tokens_out": sum(t.tokens_out for t in traces) / len(traces),
                "avg_latency_ms": sum(t.latency_ms for t in traces) / len(traces),
                "promotion_time": time.time(),
            },
        )
        
        # Store the macro
        self._macros[template.input_hash] = macro
        
        # Persist if configured
        if self._persist_path:
            self._save_to_disk()
        
        return macro
    
    def get_macro(self, input_hash: str) -> MacroDefinition | None:
        """
        Get a macro by input hash.
        
        Args:
            input_hash: The input pattern hash.
            
        Returns:
            MacroDefinition if found, None otherwise.
        """
        return self._macros.get(input_hash)
    
    def get_macro_by_name(self, name: str) -> MacroDefinition | None:
        """
        Get a macro by name.
        
        Args:
            name: The macro name.
            
        Returns:
            MacroDefinition if found, None otherwise.
        """
        for macro in self._macros.values():
            if macro.name == name:
                return macro
        return None
    
    def has_macro(self, input_hash: str) -> bool:
        """Check if a macro exists for this input hash."""
        return input_hash in self._macros
    
    def list_macros(self) -> list[MacroDefinition]:
        """Get all macros."""
        return list(self._macros.values())
    
    def delete_macro(self, input_hash: str) -> bool:
        """
        Delete a macro.
        
        Args:
            input_hash: The input hash of the macro to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        if input_hash in self._macros:
            del self._macros[input_hash]
            if self._persist_path:
                self._save_to_disk()
            return True
        return False
    
    def clear_traces(self, input_hash: str | None = None) -> int:
        """
        Clear traces.
        
        Args:
            input_hash: If provided, clear only traces for this hash.
                       If None, clear all traces.
                       
        Returns:
            Number of traces cleared.
        """
        if input_hash is not None:
            count = len(self._traces.get(input_hash, []))
            if input_hash in self._traces:
                del self._traces[input_hash]
            return count
        
        count = self.trace_count
        self._traces.clear()
        return count
    
    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        total_tokens_saved = sum(
            macro.metadata.get("avg_tokens_out", 0) * macro.hit_count
            for macro in self._macros.values()
        )
        
        return {
            "trace_count": self.trace_count,
            "macro_count": self.macro_count,
            "unique_patterns": self.unique_patterns,
            "total_tokens_saved": total_tokens_saved,
            "criteria": {
                "min_occurrences": self._criteria.min_occurrences,
                "min_success_rate": self._criteria.min_success_rate,
            },
        }
    
    def _save_to_disk(self) -> None:
        """Save state to disk."""
        if self._persist_path is None:
            return
        
        data = {
            "traces": {
                h: [t.to_dict() for t in traces]
                for h, traces in self._traces.items()
            },
            "macros": {
                h: m.to_dict()
                for h, m in self._macros.items()
            },
        }
        
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_disk(self) -> None:
        """Load state from disk."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        
        with open(self._persist_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._traces = {
            h: [GenerativeTrace.from_dict(t) for t in traces]
            for h, traces in data.get("traces", {}).items()
        }
        
        self._macros = {
            h: MacroDefinition.from_dict(m)
            for h, m in data.get("macros", {}).items()
        }
    
    def __repr__(self) -> str:
        return f"MacroStore(traces={self.trace_count}, macros={self.macro_count})"
