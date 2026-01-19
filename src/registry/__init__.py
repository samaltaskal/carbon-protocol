"""
Carbon Protocol SDK - Skill Registry Module

This module implements the Self-Optimizing Skill Registry, a deterministic
pattern-matching system that replaces vector database lookups with O(L)
hash/trie-based retrieval using Trie/Aho-Corasick.

Reference: Carbon Protocol Research Paper
    "Self-Optimizing Skill Registry" - Section III.C

Design Philosophy:
    Traditional RAG systems use vector similarity for skill retrieval:
    - Embedding: ~384+ dim vectors
    - Similarity: O(n) comparisons or ANN approximation
    - Energy: ~1M FLOPs per query
    
    The Carbon Skill Database uses deterministic matching:
    - Pattern: RegEx or exact hash
    - Lookup: O(1) hash or O(L) Aho-Corasick
    - Energy: <10K FLOPs per query
    
    This achieves 100x-1000x energy reduction while maintaining
    high accuracy for recurring workflows.

Evolution Model:
    Skills evolve from "Generative" to "Deterministic":
    
    1. Novel Input → OP_GEN (LLM generates response)
    2. Successful trace is recorded
    3. After N repetitions, trace is "promoted" to macro
    4. Future matches → OP_MACRO (deterministic, no LLM)
    
    This models the biological process of skill acquisition:
    conscious effort → automatic execution.
"""

# Primary exports from skill_db.py (Trie/Aho-Corasick)
from .skill_db import (
    SkillDB,
    SkillEntry,
    SkillMatch,
    SkillDBConfig,
    RegistryConfig,  # Alias for backward compatibility
)

# Backward compatibility: SkillRegistry is now an alias for SkillDB
from .skill_db import SkillRegistry

from .macro_store import (
    MacroStore,
    GenerativeTrace,
    PromotionCriteria,
)

from .pattern_matcher import (
    PatternMatcher,
    MatchResult,
    MatcherConfig,
)

# Legacy alias for backward compatibility
# The old Registry class from registry.py is now in the parent module
# This allows: from carbon_protocol.registry import Registry
try:
    from ..registry import Registry
except ImportError:
    # Fallback: create alias to SkillDB
    Registry = SkillDB

__all__ = [
    # Core skill database (primary)
    "SkillDB",
    "SkillDBConfig",
    # Backward compatibility aliases
    "SkillRegistry",
    "RegistryConfig",
    # Skill types
    "SkillEntry",
    "SkillMatch",
    # Macro store
    "MacroStore",
    "GenerativeTrace",
    "PromotionCriteria",
    # Pattern matching
    "PatternMatcher",
    "MatchResult",
    "MatcherConfig",
    # Legacy
    "Registry",
]
