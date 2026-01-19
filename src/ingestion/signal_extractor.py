"""
Carbon Protocol SDK - Signal Extractor

This module implements lightweight signal extraction from input text.
It converts raw text into neuron signals using keyword matching,
without any neural network inference.

Reference: Carbon Protocol Research Paper
    "Client-Side Edge Signal Extraction" - Section 3.2

Design Goals:
    1. O(L) complexity where L = input length
    2. No external ML dependencies (no BERT, spaCy, etc.)
    3. Deterministic behavior (same input = same signals)
    4. Configurable keyword dictionaries

Signal Extraction Pipeline:
    Raw Text → Tokenize → Keyword Match → Signal Weights
    
    Example:
        "Write a Python script to scrape data"
        → ["write", "a", "python", "script", "to", "scrape", "data"]
        → ["write"→code:0.3, "python"→code:0.5, "script"→code:0.4, "scrape"→code:0.3]
        → {code: [0.3, 0.5, 0.4, 0.3]}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SignalConfig:
    """
    Configuration for the SignalExtractor.
    
    Attributes:
        min_token_length: Minimum token length to consider.
        case_sensitive: Whether matching is case-sensitive.
        strip_punctuation: Whether to strip punctuation from tokens.
        enable_bigrams: Whether to also match 2-word phrases.
    """
    min_token_length: int = 2
    case_sensitive: bool = False
    strip_punctuation: bool = True
    enable_bigrams: bool = True


# Default configuration
DEFAULT_SIGNAL_CONFIG = SignalConfig()


@dataclass
class ExtractedSignal:
    """
    A single extracted signal from text.
    
    Attributes:
        keyword: The matched keyword.
        intent: Target intent category.
        weight: Signal weight.
        position: Position in input (token index).
        span: Character span (start, end) in original text.
    """
    keyword: str
    intent: str
    weight: float
    position: int
    span: tuple[int, int] | None = None


class SignalExtractor:
    """
    Extracts neuron signals from input text via keyword matching.
    
    The SignalExtractor performs lightweight text analysis to identify
    keywords that map to intent categories. This is the "front-end"
    of the neuromorphic pipeline, converting raw text to neuron inputs.
    
    Key Features:
        - O(L) complexity (linear in input length)
        - No neural network dependencies
        - Configurable keyword dictionaries
        - Supports unigrams and bigrams
    
    Reference: Carbon Protocol Research Paper
        "Signal Extraction via Keyword Banks" - Section 3.2.1
    
    Attributes:
        keywords: Dictionary mapping keywords to (intent, weight) tuples.
        config: Extraction configuration.
    
    Example:
        >>> extractor = SignalExtractor()
        >>> extractor.add_keyword("python", "code", 0.5)
        >>> extractor.add_keyword("script", "code", 0.4)
        >>> 
        >>> signals = extractor.extract("Write a python script")
        >>> for sig in signals:
        ...     print(f"{sig.keyword} → {sig.intent}: {sig.weight}")
    """
    
    def __init__(self, config: SignalConfig | None = None) -> None:
        """
        Initialize the SignalExtractor.
        
        Args:
            config: Extraction configuration. Uses defaults if None.
        """
        self._config = config or DEFAULT_SIGNAL_CONFIG
        
        # keyword -> list of (intent, weight) tuples
        # A keyword can map to multiple intents
        self._keywords: dict[str, list[tuple[str, float]]] = {}
        
        # bigrams (2-word phrases) with higher priority
        self._bigrams: dict[str, list[tuple[str, float]]] = {}
        
        # Optional custom tokenizer
        self._tokenizer: Callable[[str], list[str]] | None = None
        
        # Punctuation pattern for stripping
        self._punct_pattern = re.compile(r'[^\w\s]')
    
    @property
    def config(self) -> SignalConfig:
        """Get the extractor configuration."""
        return self._config
    
    @property
    def keyword_count(self) -> int:
        """Number of registered keywords."""
        return len(self._keywords)
    
    @property
    def bigram_count(self) -> int:
        """Number of registered bigrams."""
        return len(self._bigrams)
    
    def set_tokenizer(self, tokenizer: Callable[[str], list[str]]) -> None:
        """
        Set a custom tokenizer function.
        
        Args:
            tokenizer: Function that takes text and returns list of tokens.
        """
        self._tokenizer = tokenizer
    
    def add_keyword(
        self,
        keyword: str,
        intent: str,
        weight: float = 0.5,
    ) -> None:
        """
        Add a keyword-to-intent mapping.
        
        Args:
            keyword: The trigger keyword.
            intent: Target intent category.
            weight: Signal weight (default: 0.5).
        """
        key = keyword if self._config.case_sensitive else keyword.lower()
        
        if key not in self._keywords:
            self._keywords[key] = []
        
        self._keywords[key].append((intent, weight))
    
    def add_bigram(
        self,
        phrase: str,
        intent: str,
        weight: float = 0.7,
    ) -> None:
        """
        Add a two-word phrase mapping.
        
        Bigrams have higher priority than individual keywords.
        
        Args:
            phrase: Two-word phrase (e.g., "python script").
            intent: Target intent category.
            weight: Signal weight (default: 0.7).
        """
        key = phrase if self._config.case_sensitive else phrase.lower()
        
        if key not in self._bigrams:
            self._bigrams[key] = []
        
        self._bigrams[key].append((intent, weight))
    
    def add_keywords_bulk(
        self,
        mappings: dict[str, list[tuple[str, float]]],
    ) -> int:
        """
        Add multiple keyword mappings at once.
        
        Args:
            mappings: Dict of keyword -> [(intent, weight), ...].
        
        Returns:
            Number of keywords added.
        """
        count = 0
        for keyword, intent_weights in mappings.items():
            for intent, weight in intent_weights:
                self.add_keyword(keyword, intent, weight)
                count += 1
        return count
    
    def remove_keyword(self, keyword: str) -> bool:
        """
        Remove a keyword mapping.
        
        Args:
            keyword: The keyword to remove.
        
        Returns:
            True if removed, False if not found.
        """
        key = keyword if self._config.case_sensitive else keyword.lower()
        if key in self._keywords:
            del self._keywords[key]
            return True
        return False
    
    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        """
        Tokenize input text into (token, start, end) tuples.
        
        Args:
            text: Input text to tokenize.
        
        Returns:
            List of (token, start_pos, end_pos) tuples.
        
        Complexity: O(L) where L = len(text).
        """
        if self._tokenizer is not None:
            # Custom tokenizer (positions not tracked)
            tokens = self._tokenizer(text)
            return [(t, -1, -1) for t in tokens]
        
        # Default whitespace tokenizer with position tracking
        tokens: list[tuple[str, int, int]] = []
        
        # Split by whitespace while tracking positions
        pattern = re.compile(r'\S+')
        for match in pattern.finditer(text):
            token = match.group()
            start = match.start()
            end = match.end()
            
            # Strip punctuation if configured
            if self._config.strip_punctuation:
                token = self._punct_pattern.sub('', token)
            
            # Skip short tokens
            if len(token) >= self._config.min_token_length:
                # Normalize case if needed
                if not self._config.case_sensitive:
                    token = token.lower()
                
                tokens.append((token, start, end))
        
        return tokens
    
    def extract(self, text: str) -> list[ExtractedSignal]:
        """
        Extract signals from input text.
        
        This is the main extraction method. It tokenizes the text,
        matches keywords and bigrams, and returns a list of signals.
        
        Args:
            text: Input text to analyze.
        
        Returns:
            List of ExtractedSignal objects.
        
        Complexity: O(L) where L = len(text).
        """
        tokens = self.tokenize(text)
        signals: list[ExtractedSignal] = []
        
        # Track which token positions have been used by bigrams
        # to avoid double-counting
        bigram_positions: set[int] = set()
        
        # First pass: match bigrams (higher priority)
        if self._config.enable_bigrams and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                token1, start1, _ = tokens[i]
                token2, _, end2 = tokens[i + 1]
                bigram = f"{token1} {token2}"
                
                mappings = self._bigrams.get(bigram, [])
                for intent, weight in mappings:
                    signals.append(ExtractedSignal(
                        keyword=bigram,
                        intent=intent,
                        weight=weight,
                        position=i,
                        span=(start1, end2) if start1 >= 0 else None,
                    ))
                    bigram_positions.add(i)
                    bigram_positions.add(i + 1)
        
        # Second pass: match unigrams (skip positions used by bigrams)
        for i, (token, start, end) in enumerate(tokens):
            if i in bigram_positions:
                continue
            
            mappings = self._keywords.get(token, [])
            for intent, weight in mappings:
                signals.append(ExtractedSignal(
                    keyword=token,
                    intent=intent,
                    weight=weight,
                    position=i,
                    span=(start, end) if start >= 0 else None,
                ))
        
        return signals
    
    def extract_grouped(self, text: str) -> dict[str, list[ExtractedSignal]]:
        """
        Extract signals grouped by intent.
        
        Args:
            text: Input text to analyze.
        
        Returns:
            Dictionary mapping intent labels to signal lists.
        """
        signals = self.extract(text)
        grouped: dict[str, list[ExtractedSignal]] = {}
        
        for sig in signals:
            if sig.intent not in grouped:
                grouped[sig.intent] = []
            grouped[sig.intent].append(sig)
        
        return grouped
    
    def extract_weights_by_intent(self, text: str) -> dict[str, float]:
        """
        Extract total signal weights per intent.
        
        Args:
            text: Input text to analyze.
        
        Returns:
            Dictionary mapping intent labels to summed weights.
        """
        signals = self.extract(text)
        weights: dict[str, float] = {}
        
        for sig in signals:
            if sig.intent not in weights:
                weights[sig.intent] = 0.0
            weights[sig.intent] += sig.weight
        
        return weights
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the extractor.
        
        Returns:
            Dictionary with extractor statistics.
        """
        return {
            "keyword_count": self.keyword_count,
            "bigram_count": self.bigram_count,
            "config": {
                "min_token_length": self._config.min_token_length,
                "case_sensitive": self._config.case_sensitive,
                "strip_punctuation": self._config.strip_punctuation,
                "enable_bigrams": self._config.enable_bigrams,
            },
        }
    
    def __repr__(self) -> str:
        return f"SignalExtractor(keywords={self.keyword_count}, bigrams={self.bigram_count})"


def create_default_extractor() -> SignalExtractor:
    """
    Create a SignalExtractor with default keyword dictionaries.
    
    Returns:
        Pre-configured SignalExtractor.
    """
    extractor = SignalExtractor()
    
    # Code-related keywords
    code_keywords = [
        ("code", 0.4), ("script", 0.4), ("function", 0.4),
        ("class", 0.4), ("implement", 0.5), ("write", 0.3),
        ("create", 0.3), ("build", 0.3), ("develop", 0.3),
    ]
    for kw, weight in code_keywords:
        extractor.add_keyword(kw, "code", weight)
    
    # Language keywords (boost code intent)
    language_keywords = [
        ("python", 0.5), ("javascript", 0.5), ("java", 0.5),
        ("typescript", 0.5), ("rust", 0.5), ("go", 0.5),
        ("c++", 0.5), ("sql", 0.5), ("ruby", 0.5),
    ]
    for kw, weight in language_keywords:
        extractor.add_keyword(kw, "code", weight)
    
    # Debug keywords
    debug_keywords = [
        ("debug", 0.5), ("fix", 0.4), ("error", 0.4),
        ("bug", 0.4), ("issue", 0.4), ("problem", 0.3),
        ("crash", 0.4), ("exception", 0.4), ("traceback", 0.5),
    ]
    for kw, weight in debug_keywords:
        extractor.add_keyword(kw, "debug", weight)
    
    # Scaffold keywords
    scaffold_keywords = [
        ("scaffold", 0.6), ("setup", 0.4), ("initialize", 0.4),
        ("boilerplate", 0.5), ("template", 0.4), ("starter", 0.4),
    ]
    for kw, weight in scaffold_keywords:
        extractor.add_keyword(kw, "scaffold", weight)
    
    # Scaffold bigrams
    extractor.add_bigram("create project", "scaffold", 0.7)
    extractor.add_bigram("new project", "scaffold", 0.7)
    extractor.add_bigram("set up", "scaffold", 0.6)
    extractor.add_bigram("python project", "scaffold", 0.6)
    
    # Query keywords
    query_keywords = [
        ("what", 0.3), ("how", 0.3), ("why", 0.3),
        ("explain", 0.4), ("describe", 0.4), ("tell", 0.3),
    ]
    for kw, weight in query_keywords:
        extractor.add_keyword(kw, "query", weight)
    
    # Summarize keywords
    summarize_keywords = [
        ("summarize", 0.5), ("summary", 0.5), ("brief", 0.3),
        ("overview", 0.4), ("tldr", 0.5), ("recap", 0.4),
    ]
    for kw, weight in summarize_keywords:
        extractor.add_keyword(kw, "summarize", weight)
    
    return extractor
