"""
Carbon Protocol SDK - Carbon Instruction Set Architecture (C-ISA)

This module defines the Carbon Bytecode - a deterministic instruction set
for neuromorphic prompt processing. The C-ISA represents the "compiled"
output of the Carbon Protocol's semantic analysis pipeline.

Reference: Carbon Protocol Research Paper
    "Wake-on-Meaning Architecture" - Section 3.2
    
The instruction set enables:
    1. Deterministic execution paths (O(1) lookup, O(L) pattern match)
    2. Clear separation between Generative and Deterministic workflows
    3. Composable macro sequences for recurring tasks
    4. Explicit gating for energy-efficient "idle" states

Design Philosophy:
    Unlike token-based compression (LLMLingua-2, etc.), the Carbon ISA
    represents semantic INTENT, not statistical token distributions.
    This allows client-side edge devices to make routing decisions
    without invoking heavy neural networks.
"""

from .opcodes import (
    CarbonOpCode,
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

from .instruction import (
    CarbonInstruction,
    InstructionSequence,
    MacroDefinition,
)

from .bytecode import (
    CarbonBytecode,
    BytecodeBuilder,
    serialize_bytecode,
    deserialize_bytecode,
)

__all__ = [
    # OpCodes
    "CarbonOpCode",
    "OP_IDLE",
    "OP_LD",
    "OP_RET",
    "OP_CHK",
    "OP_MACRO",
    "OP_GEN",
    "OP_SCAFFOLD",
    "OP_TRANSFORM",
    "OP_VALIDATE",
    # Instructions
    "CarbonInstruction",
    "InstructionSequence",
    "MacroDefinition",
    # Bytecode
    "CarbonBytecode",
    "BytecodeBuilder",
    "serialize_bytecode",
    "deserialize_bytecode",
]
