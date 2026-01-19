"""
Carbon Protocol SDK - Compiler Module

This module provides the Compiler class for high-performance text compression
using Aho-Corasick multi-pattern matching.

Performance Characteristics:
- Time Complexity: O(n + z) where n = input text length, z = number of matches
- Single Pass: The entire input is scanned exactly once
- No Backtracking: Aho-Corasick automaton ensures linear time matching

This is dramatically faster than naive O(m*n) sequential replacement
where m = number of patterns and n = text length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import Registry


@dataclass
class CompressionResult:
    """
    Result of a compression operation.
    
    Attributes:
        original: The original input text.
        compressed: The compressed output text.
        matches_found: Number of pattern matches found.
        compression_ratio: Ratio of compressed length to original length.
    """
    original: str
    compressed: str
    matches_found: int
    
    @property
    def compression_ratio(self) -> float:
        """Calculate the compression ratio (compressed / original)."""
        if len(self.original) == 0:
            return 1.0
        return len(self.compressed) / len(self.original)
    
    @property
    def bytes_saved(self) -> int:
        """Calculate bytes saved by compression."""
        return len(self.original.encode('utf-8')) - len(self.compressed.encode('utf-8'))


class Compiler:
    """
    High-performance text compression compiler using Aho-Corasick algorithm.
    
    The Compiler performs deterministic semantic compression by replacing
    natural language phrases with compressed tokens in a single pass through
    the input text. This achieves O(n) complexity instead of O(m*n) from
    naive sequential replacements.
    
    Key Features:
    - Single-pass multi-pattern matching via Aho-Corasick automaton
    - Longest-match-first semantics (e.g., "visual studio code" before "visual studio")
    - Non-overlapping replacements using greedy left-to-right matching
    - Case-insensitive matching with original case preservation in non-matched segments
    
    Attributes:
        registry: The Registry containing patterns and the compiled automaton.
        preserve_whitespace: Whether to preserve original whitespace in output.
    
    Example:
        >>> from carbon_sdk import Registry, Compiler
        >>> 
        >>> registry = Registry()
        >>> registry.load_domain('core')
        >>> registry.build_automaton()
        >>> 
        >>> compiler = Compiler(registry)
        >>> result = compiler.compress("Please check the python script")
        >>> print(result.compressed)  # Compressed output
    
    Complexity Analysis:
        The compress() method runs in O(n + z) time where:
        - n = length of input text
        - z = number of pattern occurrences found
        
        This is achieved through the Aho-Corasick automaton which:
        1. Processes each character exactly once (O(n))
        2. Reports each match in O(1) amortized time
        3. Avoids the O(m) cost per position of naive string matching
    """
    
    def __init__(self, registry: Registry, preserve_whitespace: bool = True) -> None:
        """
        Initialize the Compiler with a Registry.
        
        Args:
            registry: A Registry instance with loaded patterns and built automaton.
            preserve_whitespace: Whether to preserve whitespace patterns in output.
        """
        self.registry = registry
        self.preserve_whitespace = preserve_whitespace
    
    def compress(self, text: str) -> CompressionResult:
        """
        Compress text by replacing patterns with their mapped tokens.
        
        This method performs single-pass multi-pattern matching using the
        Aho-Corasick automaton from the Registry. It handles overlapping
        matches by selecting the longest match at each position (greedy
        left-to-right, longest-match-first).
        
        Algorithm:
        1. Convert input to lowercase for case-insensitive matching
        2. Find all pattern matches using Aho-Corasick automaton (O(n + z))
        3. Resolve overlapping matches by keeping longest non-overlapping set
        4. Build output by copying unmatched segments and inserting replacements
        
        Args:
            text: The input text to compress.
        
        Returns:
            CompressionResult containing original text, compressed text,
            match count, and compression statistics.
        
        Example:
            >>> result = compiler.compress("write a python script to scrape data")
            >>> print(result.compressed)
            "@LANG:PY to ACT:SCRAPE"
            >>> print(f"Saved {result.bytes_saved} bytes")
        
        Complexity:
            O(n + z) where n = len(text), z = number of matches found.
            This is achieved through single-pass Aho-Corasick matching.
        """
        if not text:
            return CompressionResult(original=text, compressed=text, matches_found=0)
        
        automaton = self.registry.get_automaton()
        
        # Lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Collect all matches: list of (start_idx, end_idx, pattern, replacement)
        matches: list[tuple[int, int, str, str]] = []
        
        # Aho-Corasick iter() yields (end_index, (pattern, output_token))
        for end_idx, (pattern, output_token) in automaton.iter(text_lower):
            start_idx = end_idx - len(pattern) + 1
            matches.append((start_idx, end_idx + 1, pattern, output_token))
        
        if not matches:
            return CompressionResult(original=text, compressed=text, matches_found=0)
        
        # Resolve overlapping matches: longest-match-first, left-to-right greedy
        selected_matches = self._select_non_overlapping_matches(matches)
        
        # Build the compressed output
        compressed = self._build_compressed_output(text, selected_matches)
        
        return CompressionResult(
            original=text,
            compressed=compressed,
            matches_found=len(selected_matches)
        )
    
    def _select_non_overlapping_matches(
        self, 
        matches: list[tuple[int, int, str, str]]
    ) -> list[tuple[int, int, str, str]]:
        """
        Select non-overlapping matches with longest-match-first semantics.
        
        When matches overlap, we prefer:
        1. Longer matches over shorter ones
        2. Earlier (leftmost) matches when lengths are equal
        
        This ensures deterministic behavior and matches user expectations
        (e.g., "visual studio code" is matched before "visual studio").
        
        Args:
            matches: List of (start, end, pattern, replacement) tuples.
        
        Returns:
            Filtered list of non-overlapping matches sorted by start position.
        
        Complexity:
            O(z log z) for sorting, O(z) for filtering, where z = number of matches.
        """
        if not matches:
            return []
        
        # Sort by: start position (ascending), then length (descending)
        # This gives us left-to-right order with longest matches first at each position
        sorted_matches = sorted(
            matches,
            key=lambda m: (m[0], -(m[1] - m[0]))
        )
        
        selected: list[tuple[int, int, str, str]] = []
        last_end = -1
        
        for match in sorted_matches:
            start, end, pattern, replacement = match
            # Only select if this match doesn't overlap with the last selected one
            if start >= last_end:
                selected.append(match)
                last_end = end
        
        return selected
    
    def _build_compressed_output(
        self,
        original_text: str,
        matches: list[tuple[int, int, str, str]]
    ) -> str:
        """
        Build the compressed output string from original text and matches.
        
        Args:
            original_text: The original input text.
            matches: List of (start, end, pattern, replacement) tuples,
                    sorted by start position.
        
        Returns:
            The compressed string with patterns replaced by their tokens.
        
        Complexity:
            O(n) where n = len(original_text), as we iterate through once.
        """
        if not matches:
            return original_text
        
        result_parts: list[str] = []
        current_pos = 0
        
        for start, end, pattern, replacement in matches:
            # Add unmatched segment before this match
            if start > current_pos:
                result_parts.append(original_text[current_pos:start])
            
            # Add the replacement token (could be empty for removals)
            if replacement:
                result_parts.append(replacement)
            
            current_pos = end
        
        # Add any remaining text after the last match
        if current_pos < len(original_text):
            result_parts.append(original_text[current_pos:])
        
        compressed = "".join(result_parts)
        
        # Clean up multiple spaces that may result from removals
        if not self.preserve_whitespace:
            import re
            compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        return compressed
    
    def compress_batch(self, texts: list[str]) -> list[CompressionResult]:
        """
        Compress multiple texts in sequence.
        
        Args:
            texts: List of input texts to compress.
        
        Returns:
            List of CompressionResult objects.
        
        Example:
            >>> results = compiler.compress_batch([
            ...     "write a python script",
            ...     "please check the output"
            ... ])
        """
        return [self.compress(text) for text in texts]
    
    def get_stats(self) -> dict[str, int | bool]:
        """
        Get compiler statistics.
        
        Returns:
            Dictionary with compiler configuration and registry stats.
        """
        return {
            "registry_patterns": len(self.registry.patterns),
            "automaton_ready": self.registry.automaton is not None,
            "preserve_whitespace": self.preserve_whitespace,
        }
