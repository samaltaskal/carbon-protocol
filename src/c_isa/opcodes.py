"""
Carbon Protocol SDK - OpCode Definitions

This module defines the Carbon Instruction Set opcodes as a strongly-typed
enumeration. Each opcode represents a specific semantic operation in the
Carbon Protocol's execution model.

Reference: Carbon Protocol Research Paper
    "Deterministic vs Generative Paths" - Section 4.1

Instruction Categories:
    1. CONTROL FLOW: IDLE, GEN (gating decisions)
    2. DATA LOADING: LD, RET (context acquisition)
    3. VALIDATION: CHK (safety/compliance checks)
    4. EXECUTION: MACRO, SCAFFOLD, TRANSFORM, VALIDATE

Energy Model:
    - IDLE: ~0 FLOPs (no LLM invocation)
    - MACRO: O(1) lookup + template expansion
    - GEN: Full LLM inference (baseline cost)
    
    The goal is to maximize IDLE and MACRO paths, minimizing GEN.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


class CarbonOpCode(Enum):
    """
    Carbon Instruction Set OpCodes.
    
    These opcodes define the complete instruction set for the Carbon
    Protocol's neuromorphic execution model. Each opcode has specific
    semantics and energy characteristics.
    
    Attributes:
        IDLE: No-op, neuron bank did not fire (low intent detected)
        LD: Load context data (logs, manuals, documents)
        RET: Retrieve vector/semantic data (deferred to RAG)
        CHK: Run safety/compliance check (RegEx, rule-based)
        MACRO: Execute pre-compiled deterministic sequence
        GEN: Fallback to LLM generation (high entropy path)
        SCAFFOLD: Generate project/code scaffolding
        TRANSFORM: Apply deterministic transformation
        VALIDATE: Validate output against schema/rules
    
    Example:
        >>> from carbon_protocol.c_isa import CarbonOpCode
        >>> op = CarbonOpCode.MACRO
        >>> print(op.is_deterministic)  # True
        >>> print(op.estimated_flops)   # 1000 (minimal)
    """
    
    # Control Flow
    IDLE = auto()       # No action needed (gating: neurons didn't fire)
    
    # Data Operations
    LD = auto()         # Load Context (documents, logs, etc.)
    RET = auto()        # Retrieve from vector store / RAG
    
    # Validation
    CHK = auto()        # Run safety/compliance check
    
    # Execution
    MACRO = auto()      # Execute pre-compiled macro sequence
    GEN = auto()        # Generative fallback (LLM inference)
    
    # Extended Operations (Domain-Specific)
    SCAFFOLD = auto()   # Project scaffolding generation
    TRANSFORM = auto()  # Deterministic text transformation
    VALIDATE = auto()   # Output validation against schema
    
    @property
    def is_deterministic(self) -> bool:
        """
        Check if this opcode represents a deterministic (low-entropy) path.
        
        Deterministic operations can be executed without LLM inference,
        resulting in significant energy savings.
        
        Returns:
            True if the operation is deterministic, False otherwise.
        """
        return self in {
            CarbonOpCode.IDLE,
            CarbonOpCode.LD,
            CarbonOpCode.CHK,
            CarbonOpCode.MACRO,
            CarbonOpCode.SCAFFOLD,
            CarbonOpCode.TRANSFORM,
            CarbonOpCode.VALIDATE,
        }
    
    @property
    def is_generative(self) -> bool:
        """
        Check if this opcode requires LLM inference (high-entropy path).
        
        Returns:
            True if the operation requires LLM, False otherwise.
        """
        return self in {
            CarbonOpCode.GEN,
            CarbonOpCode.RET,  # RET may trigger embedding computation
        }
    
    @property
    def estimated_flops(self) -> int:
        """
        Estimated FLOPs for this operation (order of magnitude).
        
        These are rough estimates for comparison purposes:
        - IDLE: 0 (no computation)
        - MACRO/LD: ~1K (hash lookup + template expansion)
        - CHK: ~10K (regex matching)
        - RET: ~1M (embedding + vector search)
        - GEN: ~1T (full LLM forward pass)
        
        Returns:
            Estimated FLOPs as integer.
        """
        flops_map = {
            CarbonOpCode.IDLE: 0,
            CarbonOpCode.LD: 1_000,
            CarbonOpCode.CHK: 10_000,
            CarbonOpCode.MACRO: 1_000,
            CarbonOpCode.SCAFFOLD: 5_000,
            CarbonOpCode.TRANSFORM: 5_000,
            CarbonOpCode.VALIDATE: 10_000,
            CarbonOpCode.RET: 1_000_000,
            CarbonOpCode.GEN: 1_000_000_000_000,  # 1T FLOPs for LLM
        }
        return flops_map.get(self, 0)
    
    @property
    def mnemonic(self) -> str:
        """
        Get the assembly-style mnemonic for this opcode.
        
        Returns:
            String mnemonic (e.g., "OP_MACRO", "OP_GEN").
        """
        return f"OP_{self.name}"
    
    def __str__(self) -> str:
        return self.mnemonic


# Convenience aliases for common opcodes
OP_IDLE = CarbonOpCode.IDLE
OP_LD = CarbonOpCode.LD
OP_RET = CarbonOpCode.RET
OP_CHK = CarbonOpCode.CHK
OP_MACRO = CarbonOpCode.MACRO
OP_GEN = CarbonOpCode.GEN
OP_SCAFFOLD = CarbonOpCode.SCAFFOLD
OP_TRANSFORM = CarbonOpCode.TRANSFORM
OP_VALIDATE = CarbonOpCode.VALIDATE


@dataclass(frozen=True)
class OpCodeMetadata:
    """
    Metadata for an OpCode, used for documentation and tooling.
    
    Attributes:
        opcode: The CarbonOpCode enum value.
        description: Human-readable description.
        energy_class: Energy classification (ZERO, LOW, MEDIUM, HIGH).
        requires_context: Whether the op needs loaded context.
        side_effects: Whether the op has side effects.
    """
    opcode: CarbonOpCode
    description: str
    energy_class: str  # "ZERO", "LOW", "MEDIUM", "HIGH"
    requires_context: bool = False
    side_effects: bool = False


# OpCode metadata registry
OPCODE_METADATA: dict[CarbonOpCode, OpCodeMetadata] = {
    CarbonOpCode.IDLE: OpCodeMetadata(
        opcode=CarbonOpCode.IDLE,
        description="No operation - neuron bank did not fire",
        energy_class="ZERO",
    ),
    CarbonOpCode.LD: OpCodeMetadata(
        opcode=CarbonOpCode.LD,
        description="Load context data from specified source",
        energy_class="LOW",
        requires_context=False,
        side_effects=True,  # Modifies context state
    ),
    CarbonOpCode.RET: OpCodeMetadata(
        opcode=CarbonOpCode.RET,
        description="Retrieve data via vector search / RAG",
        energy_class="MEDIUM",
        requires_context=True,
    ),
    CarbonOpCode.CHK: OpCodeMetadata(
        opcode=CarbonOpCode.CHK,
        description="Run safety/compliance validation check",
        energy_class="LOW",
        requires_context=True,
    ),
    CarbonOpCode.MACRO: OpCodeMetadata(
        opcode=CarbonOpCode.MACRO,
        description="Execute pre-compiled deterministic macro",
        energy_class="LOW",
        requires_context=True,
    ),
    CarbonOpCode.GEN: OpCodeMetadata(
        opcode=CarbonOpCode.GEN,
        description="Generative fallback - invoke LLM inference",
        energy_class="HIGH",
        requires_context=True,
    ),
    CarbonOpCode.SCAFFOLD: OpCodeMetadata(
        opcode=CarbonOpCode.SCAFFOLD,
        description="Generate project scaffolding from template",
        energy_class="LOW",
        side_effects=True,
    ),
    CarbonOpCode.TRANSFORM: OpCodeMetadata(
        opcode=CarbonOpCode.TRANSFORM,
        description="Apply deterministic text transformation",
        energy_class="LOW",
        requires_context=True,
    ),
    CarbonOpCode.VALIDATE: OpCodeMetadata(
        opcode=CarbonOpCode.VALIDATE,
        description="Validate output against schema or rules",
        energy_class="LOW",
        requires_context=True,
    ),
}


def get_opcode_metadata(opcode: CarbonOpCode) -> OpCodeMetadata:
    """
    Get metadata for a given opcode.
    
    Args:
        opcode: The CarbonOpCode to look up.
        
    Returns:
        OpCodeMetadata for the given opcode.
        
    Raises:
        KeyError: If opcode has no registered metadata.
    """
    return OPCODE_METADATA[opcode]
