"""
Carbon Protocol SDK - Intent Detector

This module provides the high-level IntentDetector class that orchestrates
the neuromorphic intent detection pipeline: signal extraction → neuron bank → intent.

Reference: Carbon Protocol Research Paper
    "Wake-on-Meaning End-to-End Pipeline" - Section 3.3

Pipeline Overview:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Input     │───►│   Signal    │───►│   Neuron    │───► Intent
    │   Text      │    │  Extractor  │    │    Bank     │    Result
    └─────────────┘    └─────────────┘    └─────────────┘
    
    1. SignalExtractor: Text → Keyword Signals (O(L))
    2. NeuronBank: Signals → Neuron Voltages (O(k))
    3. Fire Detection: Voltages → Intent Labels (O(k))
    
    Total Complexity: O(L + k) where L = input length, k = neuron count

Energy Comparison:
    - BERT-based classifier: ~110M params, 22B FLOPs/inference
    - Carbon IntentDetector: ~0 params, <10K ops/inference
    - Efficiency gain: >2,000,000x
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .neuron import CarbonNeuron
from .neuron_bank import NeuronBank, create_default_intent_bank
from .signal_extractor import SignalExtractor, ExtractedSignal, create_default_extractor


@dataclass(frozen=True)
class DetectorConfig:
    """
    Configuration for the IntentDetector.
    
    Attributes:
        idle_threshold: Minimum total signal weight to avoid IDLE.
        tick_per_signal: Whether to tick neurons after each signal.
        return_all_activations: Include all activation levels in result.
    """
    idle_threshold: float = 0.3
    tick_per_signal: bool = False
    return_all_activations: bool = False


# Default configuration
DEFAULT_DETECTOR_CONFIG = DetectorConfig()


@dataclass
class IntentResult:
    """
    Result of intent detection.
    
    Attributes:
        primary_intent: The highest-confidence detected intent (or None/IDLE).
        fired_intents: List of all intents that fired.
        activation_levels: Dictionary of intent → activation ratio.
        signals_extracted: Number of signals extracted from input.
        is_idle: Whether the result is IDLE (no significant intent).
        confidence: Confidence score for primary intent (0.0-1.0).
        metadata: Additional metadata.
    
    Example:
        >>> result = detector.detect("Write a Python script")
        >>> print(result.primary_intent)  # "code"
        >>> print(result.confidence)      # 0.85
        >>> print(result.is_idle)         # False
    """
    primary_intent: str | None
    fired_intents: list[str]
    activation_levels: dict[str, float]
    signals_extracted: int
    is_idle: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_intent(self) -> bool:
        """Check if a valid intent was detected."""
        return self.primary_intent is not None and not self.is_idle
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_intent": self.primary_intent,
            "fired_intents": self.fired_intents,
            "activation_levels": self.activation_levels,
            "signals_extracted": self.signals_extracted,
            "is_idle": self.is_idle,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class IntentDetector:
    """
    High-level intent detector using neuromorphic processing.
    
    The IntentDetector orchestrates the full Wake-on-Meaning pipeline:
    1. Extract signals from input text (SignalExtractor)
    2. Feed signals to neuron bank (NeuronBank)
    3. Collect fired neurons as detected intents
    
    This achieves intent classification without any neural network
    inference, suitable for client-side edge deployment.
    
    Reference: Carbon Protocol Research Paper
        "Neuromorphic Intent Classification" - Section 3.3
    
    Attributes:
        extractor: SignalExtractor for keyword matching.
        bank: NeuronBank for intent neurons.
        config: Detector configuration.
    
    Example:
        >>> detector = IntentDetector.create_default()
        >>> 
        >>> result = detector.detect("Write a Python script to parse JSON")
        >>> print(result.primary_intent)  # "code"
        >>> print(result.confidence)      # 0.92
        >>> 
        >>> result = detector.detect("Hello")
        >>> print(result.is_idle)         # True (no significant intent)
    
    Complexity:
        O(L + k) where:
        - L = input text length
        - k = number of intent neurons
    """
    
    def __init__(
        self,
        extractor: SignalExtractor | None = None,
        bank: NeuronBank | None = None,
        config: DetectorConfig | None = None,
    ) -> None:
        """
        Initialize the IntentDetector.
        
        Args:
            extractor: SignalExtractor instance. Creates default if None.
            bank: NeuronBank instance. Creates default if None.
            config: Detector configuration. Uses defaults if None.
        """
        self._extractor = extractor or create_default_extractor()
        self._bank = bank or create_default_intent_bank()
        self._config = config or DEFAULT_DETECTOR_CONFIG
        
        # Sync extractor keywords to bank mappings
        self._sync_extractor_to_bank()
    
    @classmethod
    def create_default(cls) -> IntentDetector:
        """
        Factory method to create a fully configured detector.
        
        Returns:
            IntentDetector with default extractor and bank.
        """
        return cls(
            extractor=create_default_extractor(),
            bank=create_default_intent_bank(),
        )
    
    @property
    def extractor(self) -> SignalExtractor:
        """Get the signal extractor."""
        return self._extractor
    
    @property
    def bank(self) -> NeuronBank:
        """Get the neuron bank."""
        return self._bank
    
    @property
    def config(self) -> DetectorConfig:
        """Get the detector configuration."""
        return self._config
    
    def _sync_extractor_to_bank(self) -> None:
        """
        Ensure neuron bank has neurons for all extractor intents.
        
        This adds any missing neurons with default configuration.
        """
        # Collect all intents from extractor keywords
        intents: set[str] = set()
        for mappings in self._extractor._keywords.values():
            for intent, _ in mappings:
                intents.add(intent)
        for mappings in self._extractor._bigrams.values():
            for intent, _ in mappings:
                intents.add(intent)
        
        # Add missing neurons
        for intent in intents:
            if intent not in self._bank:
                self._bank.add_neuron(intent)
    
    def detect(self, text: str) -> IntentResult:
        """
        Detect intent from input text.
        
        This is the main detection method. It runs the full pipeline:
        1. Extract signals from text
        2. Feed signals to neuron bank
        3. Collect fired neurons
        4. Determine primary intent
        
        Args:
            text: Input text to analyze.
        
        Returns:
            IntentResult with detected intent information.
        
        Complexity: O(L + k)
        """
        # Reset neurons before processing
        self._bank.reset_all()
        
        # Step 1: Extract signals
        signals = self._extractor.extract(text)
        
        # Check for idle (insufficient signal strength)
        total_weight = sum(sig.weight for sig in signals)
        
        if total_weight < self._config.idle_threshold:
            return IntentResult(
                primary_intent=None,
                fired_intents=[],
                activation_levels=self._bank.get_activation_levels(),
                signals_extracted=len(signals),
                is_idle=True,
                confidence=0.0,
                metadata={"reason": "below_idle_threshold"},
            )
        
        # Step 2: Feed signals to neurons
        for sig in signals:
            self._bank.signal(sig.intent, sig.weight)
            
            if self._config.tick_per_signal:
                self._bank.tick_all()
        
        # Apply final tick
        self._bank.tick_all()
        
        # Step 3: Collect fired neurons
        fired = self._bank.collect_fires()
        
        # Step 4: Determine primary intent
        if fired:
            # Primary = first fired (highest priority)
            primary = fired[0]
            confidence = min(1.0, total_weight)
        else:
            # No neurons fired - check for highest activation
            activations = self._bank.get_top_activations(1)
            if activations and activations[0][1] > 0.5:
                primary = activations[0][0]
                confidence = activations[0][1]
            else:
                primary = None
                confidence = 0.0
        
        # Build result
        activation_levels = {}
        if self._config.return_all_activations:
            activation_levels = self._bank.get_activation_levels()
        
        return IntentResult(
            primary_intent=primary,
            fired_intents=fired,
            activation_levels=activation_levels,
            signals_extracted=len(signals),
            is_idle=(primary is None),
            confidence=confidence,
            metadata={
                "total_weight": total_weight,
                "signals": [(s.keyword, s.intent, s.weight) for s in signals],
            },
        )
    
    def detect_batch(self, texts: list[str]) -> list[IntentResult]:
        """
        Detect intents for multiple texts.
        
        Args:
            texts: List of input texts.
        
        Returns:
            List of IntentResult objects.
        """
        return [self.detect(text) for text in texts]
    
    def add_intent(
        self,
        intent: str,
        keywords: list[tuple[str, float]],
        threshold: float = 1.0,
        decay_rate: float = 0.9,
    ) -> None:
        """
        Add a new intent with its keywords.
        
        Args:
            intent: Intent label.
            keywords: List of (keyword, weight) tuples.
            threshold: Neuron firing threshold.
            decay_rate: Neuron decay rate.
        """
        # Add neuron if not exists
        if intent not in self._bank:
            self._bank.add_neuron(intent, threshold=threshold, decay_rate=decay_rate)
        
        # Add keywords
        for keyword, weight in keywords:
            self._extractor.add_keyword(keyword, intent, weight)
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with detector stats.
        """
        return {
            "extractor": self._extractor.get_stats(),
            "bank": self._bank.get_stats(),
            "config": {
                "idle_threshold": self._config.idle_threshold,
                "tick_per_signal": self._config.tick_per_signal,
                "return_all_activations": self._config.return_all_activations,
            },
        }
    
    def __repr__(self) -> str:
        return (f"IntentDetector(extractor={self._extractor!r}, "
                f"bank={self._bank!r})")
