"""
Carbon Protocol SDK - Compiler Module

This module contains the semantic compiler that serves as the main
"Router" logic for the Wake-on-Meaning architecture.

Reference: Carbon Protocol Research Paper
    "Semantic Compiler Architecture" - Section IV
"""

from .semantic import (
    CarbonCompiler,
    CompilationResult,
    ExecutionPath,
    CompilerConfig,
    DEFAULT_COMPILER_CONFIG,
    compile_input,
)

__all__ = [
    "CarbonCompiler",
    "CompilationResult",
    "ExecutionPath",
    "CompilerConfig",
    "DEFAULT_COMPILER_CONFIG",
    "compile_input",
]
