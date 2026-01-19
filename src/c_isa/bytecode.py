"""
Carbon Protocol SDK - Bytecode Serialization

This module provides serialization/deserialization utilities for
Carbon bytecode, enabling persistence and network transmission of
compiled instruction sequences.

Reference: Carbon Protocol Research Paper
    "Wire Protocol & Edge Caching" - Section 6.1

Format Specification:
    Carbon bytecode uses a compact binary format:
    
    Header (8 bytes):
        - Magic: "CBNC" (4 bytes)
        - Version: uint16 (2 bytes)
        - Flags: uint16 (2 bytes)
    
    Body:
        - Instruction count: varint
        - Instructions: [opcode: uint8, args_len: varint, args: bytes]*
        
    For JSON serialization, a human-readable format is also supported.
"""

from __future__ import annotations

import json
import struct
import hashlib
from dataclasses import dataclass, field
from typing import Any
from io import BytesIO

from .opcodes import CarbonOpCode
from .instruction import CarbonInstruction, InstructionSequence


# Magic bytes for binary format
BYTECODE_MAGIC = b"CBNC"
BYTECODE_VERSION = 1


@dataclass
class CarbonBytecode:
    """
    Container for compiled Carbon bytecode.
    
    Represents a complete compiled program with metadata, suitable
    for persistence, caching, or network transmission.
    
    Attributes:
        sequence: The instruction sequence.
        version: Bytecode format version.
        source_hash: Hash of original source for cache validation.
        compilation_time_ms: Time taken to compile (milliseconds).
        metadata: Additional metadata.
    
    Example:
        >>> bytecode = CarbonBytecode(
        ...     sequence=InstructionSequence([...]),
        ...     source_hash="abc123...",
        ... )
        >>> data = serialize_bytecode(bytecode)
        >>> restored = deserialize_bytecode(data)
    """
    sequence: InstructionSequence
    version: int = BYTECODE_VERSION
    source_hash: str = ""
    compilation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def checksum(self) -> str:
        """Compute checksum of this bytecode for integrity verification."""
        content = self.sequence.to_string()
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    @property
    def size_bytes(self) -> int:
        """Estimate serialized size in bytes."""
        return len(self.sequence.to_string().encode('utf-8'))
    
    @property
    def instruction_count(self) -> int:
        """Number of instructions in this bytecode."""
        return len(self.sequence)
    
    def is_cache_valid(self, source_text: str) -> bool:
        """Check if this bytecode is valid for the given source."""
        current_hash = hashlib.sha256(source_text.encode('utf-8')).hexdigest()[:16]
        return current_hash == self.source_hash


class BytecodeBuilder:
    """
    Builder for constructing Carbon bytecode programmatically.
    
    Provides a fluent interface for building instruction sequences
    and compiling them into bytecode.
    
    Example:
        >>> bytecode = (BytecodeBuilder()
        ...     .load("docs/manual.md")
        ...     .check("safety_rules")
        ...     .macro("format_response")
        ...     .build())
    """
    
    def __init__(self) -> None:
        """Initialize an empty builder."""
        self._instructions: list[CarbonInstruction] = []
        self._metadata: dict[str, Any] = {}
    
    def idle(self) -> BytecodeBuilder:
        """Add an IDLE instruction."""
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.IDLE))
        return self
    
    def load(self, source: str, **kwargs: Any) -> BytecodeBuilder:
        """Add a LD (load) instruction."""
        args = {"source": source, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.LD, args=args))
        return self
    
    def retrieve(self, query: str, **kwargs: Any) -> BytecodeBuilder:
        """Add a RET (retrieve) instruction."""
        args = {"query": query, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.RET, args=args))
        return self
    
    def check(self, rule: str, **kwargs: Any) -> BytecodeBuilder:
        """Add a CHK (check) instruction."""
        args = {"rule": rule, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.CHK, args=args))
        return self
    
    def macro(self, name: str, **kwargs: Any) -> BytecodeBuilder:
        """Add a MACRO instruction."""
        args = {"name": name, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.MACRO, args=args))
        return self
    
    def generate(self, max_tokens: int = 1024, **kwargs: Any) -> BytecodeBuilder:
        """Add a GEN (generate) instruction."""
        args = {"max_tokens": max_tokens, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.GEN, args=args))
        return self
    
    def scaffold(self, lang: str, arch: str = "default", **kwargs: Any) -> BytecodeBuilder:
        """Add a SCAFFOLD instruction."""
        args = {"lang": lang, "arch": arch, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.SCAFFOLD, args=args))
        return self
    
    def transform(self, transform_type: str, **kwargs: Any) -> BytecodeBuilder:
        """Add a TRANSFORM instruction."""
        args = {"type": transform_type, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.TRANSFORM, args=args))
        return self
    
    def validate(self, schema: str, **kwargs: Any) -> BytecodeBuilder:
        """Add a VALIDATE instruction."""
        args = {"schema": schema, **kwargs}
        self._instructions.append(CarbonInstruction(opcode=CarbonOpCode.VALIDATE, args=args))
        return self
    
    def custom(self, instruction: CarbonInstruction) -> BytecodeBuilder:
        """Add a custom instruction."""
        self._instructions.append(instruction)
        return self
    
    def with_metadata(self, key: str, value: Any) -> BytecodeBuilder:
        """Add metadata to the bytecode."""
        self._metadata[key] = value
        return self
    
    def build(self, source_text: str = "") -> CarbonBytecode:
        """
        Build the final bytecode.
        
        Args:
            source_text: Original source for hash computation.
            
        Returns:
            Compiled CarbonBytecode.
        """
        source_hash = ""
        if source_text:
            source_hash = hashlib.sha256(source_text.encode('utf-8')).hexdigest()[:16]
        
        sequence = InstructionSequence(
            instructions=self._instructions.copy(),
            source_text=source_text,
        )
        
        return CarbonBytecode(
            sequence=sequence,
            source_hash=source_hash,
            metadata=self._metadata.copy(),
        )
    
    def clear(self) -> BytecodeBuilder:
        """Clear all instructions and start fresh."""
        self._instructions.clear()
        self._metadata.clear()
        return self


def serialize_bytecode(bytecode: CarbonBytecode, format: str = "json") -> bytes:
    """
    Serialize Carbon bytecode to bytes.
    
    Args:
        bytecode: The CarbonBytecode to serialize.
        format: Serialization format ("json" or "binary").
        
    Returns:
        Serialized bytes.
        
    Raises:
        ValueError: If format is not supported.
    """
    if format == "json":
        return _serialize_json(bytecode)
    elif format == "binary":
        return _serialize_binary(bytecode)
    else:
        raise ValueError(f"Unsupported format: {format}")


def deserialize_bytecode(data: bytes, format: str = "json") -> CarbonBytecode:
    """
    Deserialize Carbon bytecode from bytes.
    
    Args:
        data: Serialized bytecode bytes.
        format: Serialization format ("json" or "binary").
        
    Returns:
        Deserialized CarbonBytecode.
        
    Raises:
        ValueError: If format is not supported or data is invalid.
    """
    if format == "json":
        return _deserialize_json(data)
    elif format == "binary":
        return _deserialize_binary(data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _serialize_json(bytecode: CarbonBytecode) -> bytes:
    """Serialize to JSON format."""
    data = {
        "version": bytecode.version,
        "source_hash": bytecode.source_hash,
        "compilation_time_ms": bytecode.compilation_time_ms,
        "metadata": bytecode.metadata,
        "instructions": [
            {
                "opcode": instr.opcode.name,
                "args": instr.args,
                "context": instr.context,
            }
            for instr in bytecode.sequence
        ],
    }
    return json.dumps(data, indent=2).encode('utf-8')


def _deserialize_json(data: bytes) -> CarbonBytecode:
    """Deserialize from JSON format."""
    obj = json.loads(data.decode('utf-8'))
    
    instructions = [
        CarbonInstruction(
            opcode=CarbonOpCode[instr["opcode"]],
            args=instr.get("args", {}),
            context=instr.get("context"),
        )
        for instr in obj.get("instructions", [])
    ]
    
    sequence = InstructionSequence(instructions=instructions)
    
    return CarbonBytecode(
        sequence=sequence,
        version=obj.get("version", BYTECODE_VERSION),
        source_hash=obj.get("source_hash", ""),
        compilation_time_ms=obj.get("compilation_time_ms", 0.0),
        metadata=obj.get("metadata", {}),
    )


def _serialize_binary(bytecode: CarbonBytecode) -> bytes:
    """Serialize to compact binary format."""
    buf = BytesIO()
    
    # Header
    buf.write(BYTECODE_MAGIC)
    buf.write(struct.pack("<H", bytecode.version))
    buf.write(struct.pack("<H", 0))  # Flags (reserved)
    
    # Instruction count
    _write_varint(buf, len(bytecode.sequence))
    
    # Instructions
    for instr in bytecode.sequence:
        # Opcode (1 byte)
        buf.write(struct.pack("<B", instr.opcode.value))
        
        # Args as JSON (length-prefixed)
        args_json = json.dumps(instr.args).encode('utf-8')
        _write_varint(buf, len(args_json))
        buf.write(args_json)
        
        # Context (length-prefixed, optional)
        context_bytes = (instr.context or "").encode('utf-8')
        _write_varint(buf, len(context_bytes))
        if context_bytes:
            buf.write(context_bytes)
    
    return buf.getvalue()


def _deserialize_binary(data: bytes) -> CarbonBytecode:
    """Deserialize from binary format."""
    buf = BytesIO(data)
    
    # Header
    magic = buf.read(4)
    if magic != BYTECODE_MAGIC:
        raise ValueError(f"Invalid bytecode magic: {magic}")
    
    version = struct.unpack("<H", buf.read(2))[0]
    _flags = struct.unpack("<H", buf.read(2))[0]  # Reserved
    
    # Instruction count
    instr_count = _read_varint(buf)
    
    # Instructions
    instructions = []
    for _ in range(instr_count):
        # Opcode
        opcode_val = struct.unpack("<B", buf.read(1))[0]
        opcode = CarbonOpCode(opcode_val)
        
        # Args
        args_len = _read_varint(buf)
        args_json = buf.read(args_len)
        args = json.loads(args_json.decode('utf-8')) if args_json else {}
        
        # Context
        context_len = _read_varint(buf)
        context = buf.read(context_len).decode('utf-8') if context_len > 0 else None
        
        instructions.append(CarbonInstruction(
            opcode=opcode,
            args=args,
            context=context,
        ))
    
    return CarbonBytecode(
        sequence=InstructionSequence(instructions=instructions),
        version=version,
    )


def _write_varint(buf: BytesIO, value: int) -> None:
    """Write a variable-length integer."""
    while value > 0x7F:
        buf.write(bytes([0x80 | (value & 0x7F)]))
        value >>= 7
    buf.write(bytes([value & 0x7F]))


def _read_varint(buf: BytesIO) -> int:
    """Read a variable-length integer."""
    result = 0
    shift = 0
    while True:
        byte = buf.read(1)
        if not byte:
            raise ValueError("Unexpected end of varint")
        b = byte[0]
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result
