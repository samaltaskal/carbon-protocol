"""
Carbon Protocol SDK - Instruction Definitions

This module defines the CarbonInstruction class and related types for
representing individual instructions and instruction sequences in the
Carbon Protocol's execution model.

Reference: Carbon Protocol Research Paper
    "Carbon Bytecode Specification" - Section 4.2

Instruction Format:
    OP:<OPCODE> [--arg=value]* [context]*
    
    Examples:
        OP:MACRO --name=scaffold_python
        OP:SCAFFOLD --lang=python --db=bigquery --arch=mvc
        OP:CHK --rule=sql_injection
        OP:GEN --max_tokens=500
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Any

from .opcodes import CarbonOpCode


@dataclass
class CarbonInstruction:
    """
    A single Carbon Protocol instruction.
    
    Represents one operation in the Carbon execution model, including
    the opcode, arguments, and optional context payload.
    
    Attributes:
        opcode: The CarbonOpCode for this instruction.
        args: Dictionary of named arguments (e.g., {"lang": "python"}).
        context: Optional context payload (compressed prompt, data, etc.).
        source_span: Optional tuple of (start, end) positions in source.
        metadata: Optional metadata dictionary.
    
    Example:
        >>> instr = CarbonInstruction(
        ...     opcode=CarbonOpCode.SCAFFOLD,
        ...     args={"lang": "python", "arch": "mvc"}
        ... )
        >>> print(instr.to_string())
        "OP:SCAFFOLD --lang=python --arch=mvc"
    """
    opcode: CarbonOpCode
    args: dict[str, Any] = field(default_factory=dict)
    context: str | None = None
    source_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate instruction after initialization."""
        if not isinstance(self.opcode, CarbonOpCode):
            raise TypeError(f"opcode must be CarbonOpCode, got {type(self.opcode)}")
    
    @property
    def is_deterministic(self) -> bool:
        """Check if this instruction is deterministic."""
        return self.opcode.is_deterministic
    
    @property
    def estimated_flops(self) -> int:
        """Get estimated FLOPs for this instruction."""
        return self.opcode.estimated_flops
    
    @property
    def instruction_hash(self) -> str:
        """
        Compute a deterministic hash of this instruction.
        
        Used for caching and deduplication in the skill registry.
        
        Returns:
            SHA-256 hash of the instruction's canonical form.
        """
        canonical = self.to_string()
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]
    
    def to_string(self) -> str:
        """
        Serialize instruction to human-readable string format.
        
        Format: OP:<NAME> [--arg=value]*
        
        Returns:
            String representation of the instruction.
        """
        parts = [f"OP:{self.opcode.name}"]
        
        # Sort args for deterministic output
        for key, value in sorted(self.args.items()):
            if isinstance(value, bool):
                if value:
                    parts.append(f"--{key}")
            elif value is not None:
                parts.append(f"--{key}={value}")
        
        return " ".join(parts)
    
    @classmethod
    def from_string(cls, text: str) -> CarbonInstruction:
        """
        Parse instruction from string format.
        
        Args:
            text: String in format "OP:<NAME> [--arg=value]*"
            
        Returns:
            Parsed CarbonInstruction.
            
        Raises:
            ValueError: If the string format is invalid.
        """
        text = text.strip()
        
        # Parse opcode
        op_match = re.match(r'^OP:(\w+)', text)
        if not op_match:
            raise ValueError(f"Invalid instruction format: {text}")
        
        op_name = op_match.group(1)
        try:
            opcode = CarbonOpCode[op_name]
        except KeyError:
            raise ValueError(f"Unknown opcode: {op_name}")
        
        # Parse arguments
        args: dict[str, Any] = {}
        arg_pattern = re.compile(r'--(\w+)(?:=([^\s]+))?')
        
        for match in arg_pattern.finditer(text):
            key = match.group(1)
            value = match.group(2)
            
            if value is None:
                args[key] = True  # Flag argument
            elif value.isdigit():
                args[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                args[key] = float(value)
            elif value.lower() in ('true', 'false'):
                args[key] = value.lower() == 'true'
            else:
                args[key] = value
        
        return cls(opcode=opcode, args=args)
    
    def with_context(self, context: str) -> CarbonInstruction:
        """
        Create a copy of this instruction with added context.
        
        Args:
            context: The context string to attach.
            
        Returns:
            New CarbonInstruction with context set.
        """
        return CarbonInstruction(
            opcode=self.opcode,
            args=self.args.copy(),
            context=context,
            source_span=self.source_span,
            metadata=self.metadata.copy(),
        )
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"CarbonInstruction({self.to_string()})"


@dataclass
class InstructionSequence:
    """
    A sequence of Carbon instructions forming a complete program.
    
    Represents a compiled Carbon "program" - a series of instructions
    that can be executed to handle a user request.
    
    Attributes:
        instructions: List of CarbonInstruction objects.
        source_text: Original source text (if available).
        compilation_metadata: Metadata from compilation process.
    
    Example:
        >>> seq = InstructionSequence([
        ...     CarbonInstruction(CarbonOpCode.LD, {"source": "docs"}),
        ...     CarbonInstruction(CarbonOpCode.CHK, {"rule": "safety"}),
        ...     CarbonInstruction(CarbonOpCode.MACRO, {"name": "respond"}),
        ... ])
        >>> print(seq.total_flops)  # Sum of instruction FLOPs
    """
    instructions: list[CarbonInstruction] = field(default_factory=list)
    source_text: str | None = None
    compilation_metadata: dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.instructions)
    
    def __iter__(self):
        return iter(self.instructions)
    
    def __getitem__(self, index: int) -> CarbonInstruction:
        return self.instructions[index]
    
    @property
    def total_flops(self) -> int:
        """Calculate total estimated FLOPs for this sequence."""
        return sum(instr.estimated_flops for instr in self.instructions)
    
    @property
    def is_fully_deterministic(self) -> bool:
        """Check if all instructions in sequence are deterministic."""
        return all(instr.is_deterministic for instr in self.instructions)
    
    @property
    def primary_opcode(self) -> CarbonOpCode | None:
        """Get the primary (most significant) opcode in the sequence."""
        if not self.instructions:
            return None
        
        # Return the highest-energy opcode (determines path type)
        return max(self.instructions, key=lambda i: i.estimated_flops).opcode
    
    def append(self, instruction: CarbonInstruction) -> None:
        """Add an instruction to the sequence."""
        self.instructions.append(instruction)
    
    def to_string(self) -> str:
        """Serialize sequence to multi-line string."""
        return "\n".join(instr.to_string() for instr in self.instructions)
    
    @classmethod
    def from_string(cls, text: str) -> InstructionSequence:
        """Parse sequence from multi-line string."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        instructions = [CarbonInstruction.from_string(line) for line in lines]
        return cls(instructions=instructions, source_text=text)


@dataclass
class MacroDefinition:
    """
    Definition of a reusable Carbon macro.
    
    Macros are pre-compiled instruction sequences that can be invoked
    by name. They represent the "deterministic path" in the Carbon
    Protocol - recurring workflows that have been hardcoded.
    
    Reference: Carbon Protocol Research Paper
        "Self-Optimizing Skill Registry" - Section 5.2
    
    Attributes:
        name: Unique macro identifier (e.g., "scaffold_python_mvc").
        pattern: Trigger pattern (regex or exact match).
        sequence: The instruction sequence to execute.
        hit_count: Number of times this macro has been invoked.
        success_rate: Ratio of successful invocations.
        source_trace: Original generative trace that was promoted.
    
    Example:
        >>> macro = MacroDefinition(
        ...     name="scaffold_python",
        ...     pattern=r"scaffold.*python.*project",
        ...     sequence=InstructionSequence([
        ...         CarbonInstruction(CarbonOpCode.SCAFFOLD, 
        ...                          {"lang": "python", "arch": "mvc"})
        ...     ])
        ... )
    """
    name: str
    pattern: str
    sequence: InstructionSequence
    hit_count: int = 0
    success_rate: float = 1.0
    source_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def pattern_hash(self) -> str:
        """Compute hash of the pattern for fast lookup."""
        return hashlib.sha256(self.pattern.encode('utf-8')).hexdigest()[:16]
    
    @property
    def is_hot(self) -> bool:
        """Check if this macro is frequently used (hot path)."""
        return self.hit_count >= 10 and self.success_rate >= 0.9
    
    def increment_hit(self, success: bool = True) -> None:
        """Record a macro invocation."""
        self.hit_count += 1
        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
    
    def to_instruction(self) -> CarbonInstruction:
        """Create a MACRO instruction that invokes this macro."""
        return CarbonInstruction(
            opcode=CarbonOpCode.MACRO,
            args={"name": self.name},
            metadata={"macro_def": self},
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize macro to dictionary for persistence."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "sequence": self.sequence.to_string(),
            "hit_count": self.hit_count,
            "success_rate": self.success_rate,
            "source_trace": self.source_trace,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MacroDefinition:
        """Deserialize macro from dictionary."""
        return cls(
            name=data["name"],
            pattern=data["pattern"],
            sequence=InstructionSequence.from_string(data["sequence"]),
            hit_count=data.get("hit_count", 0),
            success_rate=data.get("success_rate", 1.0),
            source_trace=data.get("source_trace"),
            metadata=data.get("metadata", {}),
        )
