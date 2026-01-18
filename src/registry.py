"""
Carbon Protocol SDK - Registry Module

This module provides the Registry class for loading domain-specific
compression rules and building an Aho-Corasick automaton for O(n)
multi-pattern matching.

Complexity Analysis:
- Automaton Build: O(m) where m = total characters in all patterns
- Pattern Matching: O(n + z) where n = input length, z = number of matches

This avoids the naive O(m*n) approach of sequential string replacements.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ahocorasick
import yaml


class Registry:
    """
    A registry for loading and managing domain-specific compression rules.
    
    The Registry loads patterns from YAML files organized by domain (e.g., 'core',
    'k8s', 'python', 'sql') and compiles them into an Aho-Corasick automaton
    for efficient single-pass multi-pattern matching.
    
    Attributes:
        patterns: Dictionary mapping input phrases to output tokens.
        automaton: The compiled Aho-Corasick automaton for pattern matching.
        loaded_domains: Set of domain names that have been loaded.
        data_dir: Path to the directory containing domain YAML files.
    
    Example:
        >>> registry = Registry()
        >>> registry.load_domain('core')
        >>> registry.build_automaton()
        >>> # Now ready to be used by Compiler
    """
    
    def __init__(self, data_dir: str | Path | None = None) -> None:
        """
        Initialize the Registry.
        
        Args:
            data_dir: Path to the directory containing domain YAML files.
                     Defaults to 'data/' relative to this module's location.
        """
        self.patterns: dict[str, str] = {}
        self.automaton: ahocorasick.Automaton | None = None
        self.loaded_domains: set[str] = set()
        
        if data_dir is None:
            # Default to 'data/' folder relative to this module
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
    
    def load_domain(self, domain_name: str) -> int:
        """
        Load compression rules from a domain-specific YAML file.
        
        This method loads patterns from a YAML file named '{domain_name}.yaml'
        in the data directory. Multiple domains can be loaded incrementally,
        allowing users to customize memory usage by loading only needed domains.
        
        Args:
            domain_name: Name of the domain to load (e.g., 'core', 'k8s', 'python').
                        Corresponds to a YAML file in the data directory.
        
        Returns:
            Number of patterns loaded from this domain.
        
        Raises:
            FileNotFoundError: If the domain YAML file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
            ValueError: If the YAML structure is invalid.
        
        Example:
            >>> registry = Registry()
            >>> count = registry.load_domain('core')
            >>> print(f"Loaded {count} patterns from 'core' domain")
        
        Complexity:
            O(p) where p = number of patterns in the domain file.
        """
        yaml_path = self.data_dir / f"{domain_name}.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Domain file not found: {yaml_path}. "
                f"Available domains: {self._list_available_domains()}"
            )
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"Empty YAML file: {yaml_path}")
        
        patterns_data = data.get("patterns", [])
        
        if not isinstance(patterns_data, list):
            raise ValueError(
                f"Invalid YAML structure in {yaml_path}: "
                "'patterns' must be a list of dictionaries"
            )
        
        loaded_count = 0
        for entry in patterns_data:
            if not isinstance(entry, dict):
                continue
            
            input_phrase = entry.get("input")
            output_token = entry.get("output", "")  # Default to empty string (removal)
            
            if input_phrase is not None:
                # Normalize: lowercase for case-insensitive matching
                normalized_input = str(input_phrase).lower().strip()
                if normalized_input:
                    self.patterns[normalized_input] = str(output_token)
                    loaded_count += 1
        
        self.loaded_domains.add(domain_name)
        
        # Invalidate automaton since patterns changed
        self.automaton = None
        
        return loaded_count
    
    def load_domains(self, *domain_names: str) -> int:
        """
        Load multiple domains at once.
        
        Args:
            *domain_names: Variable number of domain names to load.
        
        Returns:
            Total number of patterns loaded across all domains.
        
        Example:
            >>> registry = Registry()
            >>> total = registry.load_domains('core', 'k8s', 'python')
        """
        total = 0
        for domain in domain_names:
            total += self.load_domain(domain)
        return total
    
    def build_automaton(self) -> ahocorasick.Automaton:
        """
        Compile loaded patterns into an Aho-Corasick automaton.
        
        This method builds the automaton data structure that enables O(n)
        multi-pattern matching. The automaton must be rebuilt after loading
        new domains.
        
        The patterns are sorted by length (longest first) and added to the
        automaton. During matching, we use the end position and pattern length
        to handle overlapping matches correctly, preferring longer matches.
        
        Returns:
            The compiled Aho-Corasick automaton.
        
        Raises:
            ValueError: If no patterns have been loaded.
        
        Complexity:
            O(m) where m = total characters across all patterns.
            This is a one-time cost; subsequent matching is O(n + z).
        
        Example:
            >>> registry = Registry()
            >>> registry.load_domain('core')
            >>> automaton = registry.build_automaton()
            >>> # Automaton is now ready for pattern matching
        """
        if not self.patterns:
            raise ValueError(
                "No patterns loaded. Call load_domain() before build_automaton()."
            )
        
        self.automaton = ahocorasick.Automaton()
        
        # Sort patterns by length (descending) to facilitate longest-match-first
        # The value stored is (pattern, output_token)
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for pattern, output_token in sorted_patterns:
            # Store (original_pattern, output_token) as the value
            self.automaton.add_word(pattern, (pattern, output_token))
        
        self.automaton.make_automaton()
        
        return self.automaton
    
    def get_automaton(self) -> ahocorasick.Automaton:
        """
        Get the automaton, building it if necessary.
        
        Returns:
            The compiled Aho-Corasick automaton.
        
        Raises:
            ValueError: If no patterns have been loaded.
        """
        if self.automaton is None:
            return self.build_automaton()
        return self.automaton
    
    def _list_available_domains(self) -> list[str]:
        """
        List available domain files in the data directory.
        
        Returns:
            List of domain names (without .yaml extension).
        """
        if not self.data_dir.exists():
            return []
        
        return [
            f.stem for f in self.data_dir.glob("*.yaml")
        ]
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the loaded registry.
        
        Returns:
            Dictionary containing registry statistics.
        """
        return {
            "loaded_domains": list(self.loaded_domains),
            "total_patterns": len(self.patterns),
            "automaton_built": self.automaton is not None,
            "available_domains": self._list_available_domains(),
        }
    
    def clear(self) -> None:
        """
        Clear all loaded patterns and reset the registry.
        """
        self.patterns.clear()
        self.automaton = None
        self.loaded_domains.clear()
