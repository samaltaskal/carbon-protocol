"""
Carbon Protocol SDK

A high-performance deterministic semantic compression library using
Aho-Corasick multi-pattern matching for O(n) text compression.

Features:
- Single-pass multi-pattern matching with <10ms latency
- Modular domain loading to minimize memory footprint
- Longest-match-first semantics for deterministic output
- Type-safe Python 3.10+ API

Quick Start:
    >>> from carbon_protocol import Registry, Compiler
    >>> 
    >>> # Initialize and load domains
    >>> registry = Registry()
    >>> registry.load_domain('core')
    >>> registry.build_automaton()
    >>> 
    >>> # Compress text
    >>> compiler = Compiler(registry)
    >>> result = compiler.compress("Please check the python script")
    >>> print(result.compressed)

Architecture:
    - Registry: Loads YAML domain files and builds Aho-Corasick automaton
    - Compiler: Performs O(n) text compression using the automaton
    - Domains: YAML files in data/ directory (core.yaml, k8s.yaml, etc.)

Performance:
    Time Complexity: O(n + z) where n = input length, z = matches
    Space Complexity: O(m) where m = total pattern characters
"""

__version__ = "1.1.0"
__author__ = "Taskal Samal"

from .registry import Registry
from .compiler import Compiler, CompressionResult
from .metrics import MetricsCalculator, CompressionMetrics, AggregateMetrics

__all__ = [
    "Registry",
    "Compiler", 
    "CompressionResult",
    "MetricsCalculator",
    "CompressionMetrics",
    "AggregateMetrics",
    "__version__",
]
