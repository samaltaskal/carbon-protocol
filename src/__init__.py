"""
Carbon Protocol SDK

A Neuromorphic "Semantic Compiler" that shifts LLM optimization to the
Client-Side Edge. Unlike neural prompt compression (LLMLingua-2, etc.),
Carbon uses deterministic O(L) pattern matching and biologically-inspired
Leaky Integrate-and-Fire (LIF) neurons for intent detection.

Reference: Carbon Protocol Research Paper
    "Wake-on-Meaning Architecture for Sustainable AI"

Key Innovations:
    1. Neuromorphic Gating: LIF neurons for O(L) intent classification
    2. Self-Optimizing Registry: Generative traces → Deterministic macros
    3. Carbon Instruction Set (C-ISA): Bytecode for semantic operations
    4. Client-Side Edge: No heavy neural networks in the hot path

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CARBON PROTOCOL PIPELINE                      │
    │                                                                  │
    │  Input → [Neuron Bank] → [Skill Database] → [Bytecode]         │
    │              │                  │                                │
    │          [IDLE?]           [MATCH?]                             │
    │              │                  │                                │
    │         OP_IDLE            OP_MACRO ──────► Deterministic       │
    │              │                  │                                │
    │              └──────────────────┴──► OP_GEN ──► Generative      │
    └─────────────────────────────────────────────────────────────────┘

Performance:
    - Intent Detection: O(L) with ~1K FLOPs (vs ~22B for BERT)
    - Skill Lookup: O(1) hash or O(L) Aho-Corasick
    - Token Reduction: 25.8% average, up to 97% for recurring workflows

Quick Start:
    >>> from carbon_protocol import CarbonCompiler
    >>> 
    >>> # Create compiler with default configuration
    >>> compiler = CarbonCompiler.create_default()
    >>> 
    >>> # Process input through the Wake-on-Meaning pipeline
    >>> result = compiler.process("Write a Python script to parse JSON")
    >>> 
    >>> if result.is_deterministic:
    ...     print("Using cached skill (no LLM needed)")
    ... elif result.is_generative:
    ...     print("Falling back to LLM generation")
    ...     # After LLM responds, record for promotion:
    ...     compiler.record_generation(input_text, output_text)

Legacy Compression API (still available):
    >>> from carbon_protocol import Registry, Compiler
    >>> registry = Registry()
    >>> registry.load_domain('core')
    >>> registry.build_automaton()
    >>> compiler = Compiler(registry)
    >>> result = compiler.compress("Please check the python script")
"""

__version__ = "2.0.0"
__author__ = "Taskal Samal"

# =============================================================================
# Core Carbon Compiler (NEW - Wake-on-Meaning Architecture)
# From compiler/semantic.py - Main Router Logic
# =============================================================================
from .compiler import (
    CarbonCompiler,
    CompilationResult,
    ExecutionPath,
    CompilerConfig,
    compile_input,
)

# =============================================================================
# Carbon Instruction Set (C-ISA)
# =============================================================================
from .c_isa import (
    CarbonOpCode,
    CarbonInstruction,
    InstructionSequence,
    MacroDefinition,
    CarbonBytecode,
    BytecodeBuilder,
    OP_IDLE,
    OP_LD,
    OP_RET,
    OP_CHK,
    OP_MACRO,
    OP_GEN,
    OP_SCAFFOLD,
    OP_TRANSFORM,
    OP_VALIDATE,
)

# =============================================================================
# Neuromorphic Ingestion Layer
# =============================================================================
from .ingestion import (
    CarbonNeuron,
    NeuronConfig,
    NeuronBank,
    NeuronBankConfig,
    SignalExtractor,
    SignalConfig,
    IntentDetector,
    IntentResult,
    DetectorConfig,
)

# =============================================================================
# Skill Database (Self-Optimizing) - Trie/Aho-Corasick
# From registry/skill_db.py - Section III.C
# =============================================================================
from .registry import (
    SkillDB,
    SkillDBConfig,
    SkillRegistry,  # Backward compatibility alias
    SkillEntry,
    SkillMatch,
    RegistryConfig,  # Backward compatibility alias
    MacroStore,
    GenerativeTrace,
    PromotionCriteria,
    PatternMatcher,
    MatchResult,
    MatcherConfig,
)

# =============================================================================
# Legacy Compression API (Backward Compatibility)
# =============================================================================
from .registry import Registry  # Legacy alias
from .legacy_compiler import Compiler, CompressionResult
from .metrics import MetricsCalculator, CompressionMetrics, AggregateMetrics

__all__ = [
    # Version
    "__version__",
    
    # Carbon Compiler (Main API)
    "CarbonCompiler",
    "CompilationResult",
    "ExecutionPath",
    "CompilerConfig",
    "compile_input",
    
    # C-ISA OpCodes
    "CarbonOpCode",
    "CarbonInstruction",
    "InstructionSequence",
    "MacroDefinition",
    "CarbonBytecode",
    "BytecodeBuilder",
    "OP_IDLE",
    "OP_LD",
    "OP_RET",
    "OP_CHK",
    "OP_MACRO",
    "OP_GEN",
    "OP_SCAFFOLD",
    "OP_TRANSFORM",
    "OP_VALIDATE",
    
    # Neuromorphic Components
    "CarbonNeuron",
    "NeuronConfig",
    "NeuronBank",
    "NeuronBankConfig",
    "SignalExtractor",
    "SignalConfig",
    "IntentDetector",
    "IntentResult",
    "DetectorConfig",
    
    # Skill Database (Trie/Aho-Corasick)
    "SkillDB",
    "SkillDBConfig",
    "SkillRegistry",  # Backward compatibility
    "SkillEntry",
    "SkillMatch",
    "RegistryConfig",  # Backward compatibility
    "MacroStore",
    "GenerativeTrace",
    "PromotionCriteria",
    "PatternMatcher",
    "MatchResult",
    "MatcherConfig",
    
    # Legacy API
    "Registry",
    "Compiler",
    "CompressionResult",
    "MetricsCalculator",
    "CompressionMetrics",
    "AggregateMetrics",
]
