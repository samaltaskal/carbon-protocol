"""
Carbon Protocol SDK - Pattern Matcher

This module implements O(L) pattern matching using Aho-Corasick automaton
and hash-based exact matching for the Skill Registry.

Reference: Carbon Protocol Research Paper
    "Deterministic Pattern Matching" - Section 5.1

Matching Strategies:
    1. Exact Hash Match: O(1) for known exact patterns
    2. Aho-Corasick: O(L + z) for multi-pattern substring matching
    3. RegEx Engine: O(L) for complex pattern rules

The matcher maintains a hierarchy:
    Exact Match (fastest) → Aho-Corasick → RegEx (slowest)
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Any, Pattern

# Use ahocorasick if available, otherwise fall back to regex-only
try:
    import ahocorasick
    HAS_AHOCORASICK = True
except ImportError:
    HAS_AHOCORASICK = False


@dataclass(frozen=True)
class MatcherConfig:
    """
    Configuration for PatternMatcher.
    
    Attributes:
        case_sensitive: Whether matching is case-sensitive.
        use_aho_corasick: Whether to use Aho-Corasick for multi-pattern.
        max_regex_patterns: Maximum regex patterns before warning.
        hash_algorithm: Hash algorithm for exact matching.
    """
    case_sensitive: bool = False
    use_aho_corasick: bool = True
    max_regex_patterns: int = 100
    hash_algorithm: str = "sha256"


DEFAULT_MATCHER_CONFIG = MatcherConfig()


@dataclass
class MatchResult:
    """
    Result of a pattern match operation.
    
    Attributes:
        matched: Whether a match was found.
        pattern_id: ID of the matched pattern (if any).
        pattern_text: The matched pattern string.
        match_type: Type of match ("exact", "aho", "regex").
        score: Match confidence score (0.0-1.0).
        span: Character span of match (start, end).
        captures: Named captures from regex (if any).
    """
    matched: bool
    pattern_id: str | None = None
    pattern_text: str | None = None
    match_type: str | None = None
    score: float = 0.0
    span: tuple[int, int] | None = None
    captures: dict[str, str] = field(default_factory=dict)
    
    @property
    def is_exact(self) -> bool:
        """Check if this was an exact match."""
        return self.match_type == "exact"


class PatternMatcher:
    """
    High-performance pattern matcher for the Skill Registry.
    
    Implements a tiered matching strategy for optimal performance:
    1. Exact hash lookup (O(1))
    2. Aho-Corasick multi-pattern matching (O(L))
    3. Regex pattern matching (O(L) per pattern)
    
    Reference: Carbon Protocol Research Paper
        "O(L) Deterministic Matching" - Section 5.1.1
    
    Attributes:
        config: Matcher configuration.
    
    Example:
        >>> matcher = PatternMatcher()
        >>> matcher.add_exact("write python script", "skill_001")
        >>> matcher.add_regex(r"create.*project", "skill_002")
        >>> matcher.build()
        >>> 
        >>> result = matcher.match("write python script")
        >>> print(result.pattern_id)  # "skill_001"
    """
    
    def __init__(self, config: MatcherConfig | None = None) -> None:
        """
        Initialize the PatternMatcher.
        
        Args:
            config: Matcher configuration.
        """
        self._config = config or DEFAULT_MATCHER_CONFIG
        
        # Exact match hash table: hash -> (pattern_id, original_pattern)
        self._exact_patterns: dict[str, tuple[str, str]] = {}
        
        # Aho-Corasick automaton for substring patterns
        self._aho_patterns: dict[str, str] = {}  # normalized_pattern -> pattern_id
        self._automaton: Any = None  # ahocorasick.Automaton or None
        
        # Regex patterns: list of (compiled_regex, pattern_id, original_pattern)
        self._regex_patterns: list[tuple[Pattern[str], str, str]] = []
        
        # Build state
        self._is_built = False
    
    @property
    def config(self) -> MatcherConfig:
        """Get the matcher configuration."""
        return self._config
    
    @property
    def pattern_count(self) -> int:
        """Total number of patterns."""
        return len(self._exact_patterns) + len(self._aho_patterns) + len(self._regex_patterns)
    
    @property
    def is_built(self) -> bool:
        """Check if the automaton is built."""
        return self._is_built
    
    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        if self._config.case_sensitive:
            return text.strip()
        return text.lower().strip()
    
    def _hash_pattern(self, pattern: str) -> str:
        """Compute hash for exact matching."""
        normalized = self._normalize(pattern)
        if self._config.hash_algorithm == "sha256":
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
        elif self._config.hash_algorithm == "md5":
            return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        else:
            # Simple hash for performance
            return str(hash(normalized))
    
    def add_exact(self, pattern: str, pattern_id: str) -> None:
        """
        Add an exact match pattern.
        
        Exact patterns are matched via hash lookup (O(1)).
        
        Args:
            pattern: The pattern text.
            pattern_id: Unique identifier for this pattern.
        """
        pattern_hash = self._hash_pattern(pattern)
        self._exact_patterns[pattern_hash] = (pattern_id, pattern)
        self._is_built = False
    
    def add_substring(self, pattern: str, pattern_id: str) -> None:
        """
        Add a substring pattern for Aho-Corasick matching.
        
        Substring patterns are matched via Aho-Corasick automaton (O(L)).
        
        Args:
            pattern: The pattern text.
            pattern_id: Unique identifier for this pattern.
        """
        normalized = self._normalize(pattern)
        self._aho_patterns[normalized] = pattern_id
        self._is_built = False
    
    def add_regex(self, pattern: str, pattern_id: str) -> None:
        """
        Add a regex pattern.
        
        Regex patterns are matched after exact and Aho-Corasick (O(L) per pattern).
        
        Args:
            pattern: The regex pattern string.
            pattern_id: Unique identifier for this pattern.
            
        Raises:
            re.error: If the pattern is invalid.
        """
        flags = 0 if self._config.case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
        self._regex_patterns.append((compiled, pattern_id, pattern))
        self._is_built = False
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove a pattern by ID.
        
        Args:
            pattern_id: The pattern ID to remove.
            
        Returns:
            True if removed, False if not found.
        """
        # Check exact patterns
        to_remove = [h for h, (pid, _) in self._exact_patterns.items() if pid == pattern_id]
        for h in to_remove:
            del self._exact_patterns[h]
        
        # Check Aho patterns
        aho_remove = [p for p, pid in self._aho_patterns.items() if pid == pattern_id]
        for p in aho_remove:
            del self._aho_patterns[p]
        
        # Check regex patterns
        self._regex_patterns = [(r, pid, orig) for r, pid, orig in self._regex_patterns if pid != pattern_id]
        
        if to_remove or aho_remove:
            self._is_built = False
            return True
        return False
    
    def build(self) -> None:
        """
        Build the Aho-Corasick automaton for substring matching.
        
        Must be called after adding patterns and before matching.
        """
        if HAS_AHOCORASICK and self._config.use_aho_corasick and self._aho_patterns:
            self._automaton = ahocorasick.Automaton()
            
            for pattern, pattern_id in self._aho_patterns.items():
                self._automaton.add_word(pattern, (pattern_id, pattern))
            
            self._automaton.make_automaton()
        else:
            self._automaton = None
        
        self._is_built = True
    
    def match(self, text: str) -> MatchResult:
        """
        Match input text against all patterns.
        
        Matching is performed in order of efficiency:
        1. Exact hash match (O(1))
        2. Aho-Corasick substring match (O(L))
        3. Regex pattern match (O(L) per pattern)
        
        Args:
            text: Input text to match.
            
        Returns:
            MatchResult indicating match status and details.
        """
        if not self._is_built:
            self.build()
        
        # 1. Try exact match first (O(1))
        text_hash = self._hash_pattern(text)
        if text_hash in self._exact_patterns:
            pattern_id, original = self._exact_patterns[text_hash]
            return MatchResult(
                matched=True,
                pattern_id=pattern_id,
                pattern_text=original,
                match_type="exact",
                score=1.0,
                span=(0, len(text)),
            )
        
        # 2. Try Aho-Corasick (O(L))
        if self._automaton is not None:
            normalized = self._normalize(text)
            matches: list[tuple[int, tuple[str, str]]] = list(self._automaton.iter(normalized))
            
            if matches:
                # Return longest match (most specific)
                best = max(matches, key=lambda m: len(m[1][1]))
                end_idx = best[0]
                pattern_id, pattern_text = best[1]
                start_idx = end_idx - len(pattern_text) + 1
                
                return MatchResult(
                    matched=True,
                    pattern_id=pattern_id,
                    pattern_text=pattern_text,
                    match_type="aho",
                    score=len(pattern_text) / len(text),
                    span=(start_idx, end_idx + 1),
                )
        
        # 3. Try regex patterns (O(L) per pattern)
        for compiled, pattern_id, original in self._regex_patterns:
            match = compiled.search(text)
            if match:
                return MatchResult(
                    matched=True,
                    pattern_id=pattern_id,
                    pattern_text=original,
                    match_type="regex",
                    score=len(match.group()) / len(text),
                    span=match.span(),
                    captures=match.groupdict(),
                )
        
        # No match found
        return MatchResult(matched=False)
    
    def match_all(self, text: str) -> list[MatchResult]:
        """
        Find all matching patterns (not just the first).
        
        Args:
            text: Input text to match.
            
        Returns:
            List of all MatchResult objects.
        """
        if not self._is_built:
            self.build()
        
        results: list[MatchResult] = []
        
        # Check exact match
        text_hash = self._hash_pattern(text)
        if text_hash in self._exact_patterns:
            pattern_id, original = self._exact_patterns[text_hash]
            results.append(MatchResult(
                matched=True,
                pattern_id=pattern_id,
                pattern_text=original,
                match_type="exact",
                score=1.0,
                span=(0, len(text)),
            ))
        
        # Check Aho-Corasick
        if self._automaton is not None:
            normalized = self._normalize(text)
            for end_idx, (pattern_id, pattern_text) in self._automaton.iter(normalized):
                start_idx = end_idx - len(pattern_text) + 1
                results.append(MatchResult(
                    matched=True,
                    pattern_id=pattern_id,
                    pattern_text=pattern_text,
                    match_type="aho",
                    score=len(pattern_text) / len(text),
                    span=(start_idx, end_idx + 1),
                ))
        
        # Check regex patterns
        for compiled, pattern_id, original in self._regex_patterns:
            for match in compiled.finditer(text):
                results.append(MatchResult(
                    matched=True,
                    pattern_id=pattern_id,
                    pattern_text=original,
                    match_type="regex",
                    score=len(match.group()) / len(text),
                    span=match.span(),
                    captures=match.groupdict(),
                ))
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get matcher statistics."""
        return {
            "exact_patterns": len(self._exact_patterns),
            "aho_patterns": len(self._aho_patterns),
            "regex_patterns": len(self._regex_patterns),
            "total_patterns": self.pattern_count,
            "is_built": self._is_built,
            "has_ahocorasick": HAS_AHOCORASICK,
        }
    
    def __repr__(self) -> str:
        return f"PatternMatcher(patterns={self.pattern_count}, built={self._is_built})"
