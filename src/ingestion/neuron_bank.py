"""
Carbon Protocol SDK - Neuron Bank

This module implements a bank of LIF neurons for multi-intent detection.
Each neuron in the bank corresponds to a specific intent category,
enabling parallel intent classification with O(n) complexity.

Reference: Carbon Protocol Research Paper
    "Sparse Neuromorphic Gating" - Section 3.1.2

Architecture:
    The NeuronBank maintains multiple CarbonNeurons, each tuned to
    detect a specific intent type:
    
    ┌─────────────────────────────────────────┐
    │            NEURON BANK                   │
    │  ┌───────┐  ┌───────┐  ┌───────┐       │
    │  │ CODE  │  │ QUERY │  │SCAFFOLD│ ...   │
    │  │ 0.7v  │  │ 0.2v  │  │ 0.9v  │       │
    │  └───────┘  └───────┘  └───────┘       │
    │      │          │          │     *FIRE*  │
    └──────┼──────────┼──────────┼────────────┘
           ▼          ▼          ▼
        [CODE]     [IDLE]   [SCAFFOLD]
    
    Only neurons that fire produce output, achieving sparse activation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .neuron import CarbonNeuron, NeuronConfig


@dataclass(frozen=True)
class NeuronBankConfig:
    """
    Configuration for a NeuronBank.
    
    Attributes:
        default_threshold: Default threshold for neurons.
        default_decay_rate: Default decay rate for neurons.
        max_simultaneous_fires: Max neurons that can fire at once.
        require_explicit_tick: If True, tick() must be called manually.
    """
    default_threshold: float = 1.0
    default_decay_rate: float = 0.9
    max_simultaneous_fires: int = 3
    require_explicit_tick: bool = False


# Default bank configuration
DEFAULT_BANK_CONFIG = NeuronBankConfig()


class NeuronBank:
    """
    A bank of LIF neurons for multi-intent classification.
    
    The NeuronBank manages a collection of CarbonNeurons, each labeled
    with an intent category. Input signals are routed to appropriate
    neurons based on keyword-to-intent mappings.
    
    Key Features:
        - O(n) classification where n = number of matched keywords
        - Sparse activation: only high-confidence intents fire
        - No neural network: pure arithmetic operations
        - Configurable thresholds per intent category
    
    Reference: Carbon Protocol Research Paper
        "Multi-Intent Neuromorphic Detection" - Section 3.1.2
    
    Attributes:
        neurons: Dictionary mapping intent labels to CarbonNeuron instances.
        config: Bank configuration.
    
    Example:
        >>> bank = NeuronBank()
        >>> bank.add_neuron("code", threshold=1.0)
        >>> bank.add_neuron("query", threshold=0.8)
        >>> bank.add_neuron("scaffold", threshold=1.2)
        >>> 
        >>> # Route signals
        >>> bank.signal("code", 0.5)  # "python" keyword
        >>> bank.signal("code", 0.3)  # "script" keyword
        >>> bank.signal("scaffold", 0.6)  # "create project"
        >>> 
        >>> # Check which intents fired
        >>> fired = bank.collect_fires()
        >>> print(fired)  # ["code"] or ["scaffold"] or both
    
    Complexity:
        - add_neuron(): O(1)
        - signal(): O(1)
        - tick_all(): O(k) where k = number of neurons
        - collect_fires(): O(k)
    """
    
    def __init__(self, config: NeuronBankConfig | None = None) -> None:
        """
        Initialize a NeuronBank.
        
        Args:
            config: Optional bank configuration. Uses defaults if not provided.
        """
        self._config = config or DEFAULT_BANK_CONFIG
        self._neurons: dict[str, CarbonNeuron] = {}
        self._signal_weights: dict[str, dict[str, float]] = {}  # keyword -> {intent: weight}
    
    @property
    def config(self) -> NeuronBankConfig:
        """Get the bank configuration."""
        return self._config
    
    @property
    def neuron_count(self) -> int:
        """Number of neurons in the bank."""
        return len(self._neurons)
    
    @property
    def labels(self) -> list[str]:
        """List of all neuron labels."""
        return list(self._neurons.keys())
    
    def add_neuron(
        self,
        label: str,
        threshold: float | None = None,
        decay_rate: float | None = None,
        refractory_period: int = 0,
    ) -> CarbonNeuron:
        """
        Add a new neuron to the bank.
        
        Args:
            label: Unique label for this intent (e.g., "code", "query").
            threshold: Firing threshold. Uses config default if None.
            decay_rate: Decay rate. Uses config default if None.
            refractory_period: Ticks to wait after firing.
        
        Returns:
            The created CarbonNeuron.
        
        Raises:
            ValueError: If label already exists.
        """
        if label in self._neurons:
            raise ValueError(f"Neuron with label '{label}' already exists")
        
        neuron = CarbonNeuron(
            threshold=threshold or self._config.default_threshold,
            decay_rate=decay_rate or self._config.default_decay_rate,
            refractory_period=refractory_period,
            label=label,
        )
        
        self._neurons[label] = neuron
        return neuron
    
    def get_neuron(self, label: str) -> CarbonNeuron | None:
        """
        Get a neuron by label.
        
        Args:
            label: The neuron's label.
        
        Returns:
            The CarbonNeuron, or None if not found.
        """
        return self._neurons.get(label)
    
    def remove_neuron(self, label: str) -> bool:
        """
        Remove a neuron from the bank.
        
        Args:
            label: The neuron's label.
        
        Returns:
            True if removed, False if not found.
        """
        if label in self._neurons:
            del self._neurons[label]
            return True
        return False
    
    def register_signal_weight(
        self,
        keyword: str,
        intent: str,
        weight: float = 0.5,
    ) -> None:
        """
        Register a keyword-to-intent signal weight.
        
        When this keyword is detected, the specified weight is
        added to the intent's neuron.
        
        Args:
            keyword: The trigger keyword (lowercase).
            intent: The target intent label.
            weight: Signal weight to add (default: 0.5).
        """
        keyword_lower = keyword.lower()
        if keyword_lower not in self._signal_weights:
            self._signal_weights[keyword_lower] = {}
        self._signal_weights[keyword_lower][intent] = weight
    
    def get_signal_mappings(self) -> dict[str, dict[str, float]]:
        """Get all keyword-to-intent signal mappings."""
        return self._signal_weights.copy()
    
    def signal(self, label: str, weight: float) -> bool:
        """
        Send a signal to a specific neuron.
        
        Args:
            label: The target neuron's label.
            weight: Signal weight to add.
        
        Returns:
            True if neuron exists and received signal.
        """
        neuron = self._neurons.get(label)
        if neuron is None:
            return False
        
        neuron.input(weight)
        return True
    
    def signal_from_keyword(self, keyword: str) -> list[str]:
        """
        Route a keyword signal to all registered neurons.
        
        Args:
            keyword: The detected keyword.
        
        Returns:
            List of intent labels that received signals.
        """
        keyword_lower = keyword.lower()
        mappings = self._signal_weights.get(keyword_lower, {})
        
        signaled = []
        for intent, weight in mappings.items():
            if self.signal(intent, weight):
                signaled.append(intent)
        
        return signaled
    
    def tick_all(self) -> None:
        """
        Apply one tick of decay to all neurons.
        
        Should be called once per input token or time step.
        
        Complexity: O(k) where k = number of neurons.
        """
        for neuron in self._neurons.values():
            neuron.tick()
    
    def collect_fires(self) -> list[str]:
        """
        Collect all neurons that fire (exceed threshold).
        
        This checks each neuron and returns labels of those that
        fire. Fired neurons have their voltage reset.
        
        Returns:
            List of labels for neurons that fired.
        
        Complexity: O(k) where k = number of neurons.
        """
        fired: list[str] = []
        
        for label, neuron in self._neurons.items():
            if neuron.fire():
                fired.append(label)
                # Respect max simultaneous fires
                if len(fired) >= self._config.max_simultaneous_fires:
                    break
        
        return fired
    
    def tick_and_fire(self) -> list[str]:
        """
        Combined tick + fire collection (convenience method).
        
        Returns:
            List of labels for neurons that fired.
        """
        self.tick_all()
        return self.collect_fires()
    
    def get_activation_levels(self) -> dict[str, float]:
        """
        Get activation levels (voltage/threshold ratio) for all neurons.
        
        Returns:
            Dictionary mapping labels to activation ratios.
        """
        return {
            label: neuron.activation_ratio
            for label, neuron in self._neurons.items()
        }
    
    def get_top_activations(self, n: int = 3) -> list[tuple[str, float]]:
        """
        Get the top N most activated neurons.
        
        Args:
            n: Number of top activations to return.
        
        Returns:
            List of (label, activation_ratio) tuples, sorted descending.
        """
        activations = self.get_activation_levels()
        sorted_activations = sorted(
            activations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_activations[:n]
    
    def reset_all(self) -> None:
        """
        Reset all neurons to initial state.
        
        Preserves fire counts for historical analysis.
        """
        for neuron in self._neurons.values():
            neuron.reset()
    
    def hard_reset_all(self) -> None:
        """
        Full reset of all neurons including fire counts.
        """
        for neuron in self._neurons.values():
            neuron.hard_reset()
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the neuron bank.
        
        Returns:
            Dictionary with bank statistics.
        """
        total_fires = sum(n.fire_count for n in self._neurons.values())
        return {
            "neuron_count": self.neuron_count,
            "labels": self.labels,
            "total_fires": total_fires,
            "keyword_mappings": len(self._signal_weights),
            "config": {
                "default_threshold": self._config.default_threshold,
                "default_decay_rate": self._config.default_decay_rate,
                "max_simultaneous_fires": self._config.max_simultaneous_fires,
            },
        }
    
    def __iter__(self) -> Iterator[tuple[str, CarbonNeuron]]:
        """Iterate over (label, neuron) pairs."""
        return iter(self._neurons.items())
    
    def __len__(self) -> int:
        return len(self._neurons)
    
    def __contains__(self, label: str) -> bool:
        return label in self._neurons
    
    def __repr__(self) -> str:
        return f"NeuronBank(neurons={self.neuron_count}, labels={self.labels})"


def create_default_intent_bank() -> NeuronBank:
    """
    Create a NeuronBank with default intent categories.
    
    This factory function creates a pre-configured bank with
    common intent neurons for typical LLM interactions.
    
    Returns:
        Pre-configured NeuronBank.
    """
    bank = NeuronBank()
    
    # Code-related intents
    bank.add_neuron("code", threshold=1.0, decay_rate=0.85)
    bank.add_neuron("debug", threshold=0.9, decay_rate=0.8)
    bank.add_neuron("scaffold", threshold=1.2, decay_rate=0.9)
    
    # Query intents
    bank.add_neuron("query", threshold=0.8, decay_rate=0.85)
    bank.add_neuron("explain", threshold=0.9, decay_rate=0.85)
    
    # Task intents
    bank.add_neuron("transform", threshold=1.0, decay_rate=0.9)
    bank.add_neuron("summarize", threshold=0.9, decay_rate=0.85)
    bank.add_neuron("translate", threshold=0.9, decay_rate=0.85)
    
    # Register common keyword mappings
    code_keywords = ["code", "script", "function", "class", "implement", "write"]
    for kw in code_keywords:
        bank.register_signal_weight(kw, "code", 0.4)
    
    debug_keywords = ["debug", "fix", "error", "bug", "issue", "problem"]
    for kw in debug_keywords:
        bank.register_signal_weight(kw, "debug", 0.4)
    
    scaffold_keywords = ["scaffold", "create project", "setup", "initialize", "boilerplate"]
    for kw in scaffold_keywords:
        bank.register_signal_weight(kw, "scaffold", 0.5)
    
    query_keywords = ["what", "how", "why", "explain", "describe"]
    for kw in query_keywords:
        bank.register_signal_weight(kw, "query", 0.3)
        bank.register_signal_weight(kw, "explain", 0.3)
    
    return bank
