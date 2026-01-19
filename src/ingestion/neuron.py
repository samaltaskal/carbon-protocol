"""
Carbon Protocol SDK - Leaky Integrate-and-Fire Neuron

This module implements a minimal Leaky Integrate-and-Fire (LIF) neuron
for use in the Carbon Protocol's neuromorphic gating system.

Reference: Carbon Protocol Research Paper
    "Biologically-Inspired Sparse Activation" - Section 3.1.1

Mathematical Model:
    The LIF neuron follows the discrete-time dynamics:
    
    v(t+1) = λ * v(t) + I(t)    (integrate with leak)
    
    if v(t+1) > θ:               (threshold check)
        fire = True
        v(t+1) = 0               (reset)
    
    Where:
        v(t)  = membrane voltage at time t
        λ     = decay rate (0 < λ < 1), models "leaky" membrane
        I(t)  = input signal at time t
        θ     = firing threshold

Performance Requirements:
    - Pure Python arithmetic (no numpy/torch in hot path)
    - O(1) time per input signal
    - Minimal memory footprint (~32 bytes per neuron)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NeuronConfig:
    """
    Configuration for a CarbonNeuron.
    
    Attributes:
        threshold: Voltage threshold for firing (default: 1.0).
        decay_rate: Voltage decay per tick (default: 0.9).
        refractory_period: Ticks to wait after firing (default: 0).
    """
    threshold: float = 1.0
    decay_rate: float = 0.9
    refractory_period: int = 0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {self.decay_rate}")
        if self.threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.refractory_period < 0:
            raise ValueError(f"refractory_period must be non-negative, got {self.refractory_period}")


# Default configuration
DEFAULT_NEURON_CONFIG = NeuronConfig()


class CarbonNeuron:
    """
    A Leaky Integrate-and-Fire (LIF) neuron for intent detection.
    
    This class implements a biologically-inspired neuron model that
    accumulates input signals over time, with natural decay (leak).
    When the accumulated voltage exceeds a threshold, the neuron
    "fires", indicating strong intent signal.
    
    Key Properties:
        - Extremely lightweight: pure Python arithmetic, no dependencies
        - O(1) time complexity per operation
        - Deterministic behavior (no randomness)
        - Stateful: maintains voltage across inputs
    
    Reference: Carbon Protocol Research Paper
        "Neuromorphic Wake-on-Meaning" - Section 3.1
    
    Attributes:
        voltage: Current membrane voltage (mutable state).
        threshold: Voltage threshold for firing (immutable).
        decay_rate: Decay factor applied per tick (immutable).
        label: Optional label for this neuron (e.g., "code_intent").
    
    Example:
        >>> neuron = CarbonNeuron(threshold=1.0, decay_rate=0.9)
        >>> 
        >>> # Accumulate signals
        >>> neuron.input(0.4)  # "code" keyword detected
        >>> neuron.input(0.3)  # "python" keyword detected
        >>> neuron.tick()      # Time passes, voltage decays
        >>> neuron.input(0.5)  # "script" keyword detected
        >>> 
        >>> if neuron.fire():
        ...     print("Code intent detected!")
    
    Complexity:
        - input(): O(1)
        - tick(): O(1)
        - fire(): O(1)
        - reset(): O(1)
    """
    
    __slots__ = ('_voltage', '_threshold', '_decay_rate', '_refractory_ticks', 
                 '_refractory_period', '_fire_count', '_label')
    
    def __init__(
        self,
        threshold: float = 1.0,
        decay_rate: float = 0.9,
        refractory_period: int = 0,
        label: str | None = None,
        config: NeuronConfig | None = None,
    ) -> None:
        """
        Initialize a CarbonNeuron.
        
        Args:
            threshold: Voltage threshold for firing. When voltage exceeds
                      this value, the neuron fires. Default: 1.0.
            decay_rate: Decay factor applied to voltage each tick. Should
                       be in range (0, 1]. Lower values = faster decay.
                       Default: 0.9.
            refractory_period: Number of ticks to remain inactive after
                              firing. Default: 0 (no refractory period).
            label: Optional human-readable label for this neuron.
            config: Optional NeuronConfig object. If provided, other
                   parameters are ignored.
        
        Raises:
            ValueError: If decay_rate is not in (0, 1] or threshold <= 0.
        """
        if config is not None:
            threshold = config.threshold
            decay_rate = config.decay_rate
            refractory_period = config.refractory_period
        
        # Validate parameters (without external dependencies)
        if not (0.0 < decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if refractory_period < 0:
            raise ValueError(f"refractory_period must be non-negative")
        
        self._voltage: float = 0.0
        self._threshold: float = threshold
        self._decay_rate: float = decay_rate
        self._refractory_period: int = refractory_period
        self._refractory_ticks: int = 0
        self._fire_count: int = 0
        self._label: str | None = label
    
    @property
    def voltage(self) -> float:
        """Current membrane voltage."""
        return self._voltage
    
    @property
    def threshold(self) -> float:
        """Firing threshold (immutable)."""
        return self._threshold
    
    @property
    def decay_rate(self) -> float:
        """Voltage decay rate per tick (immutable)."""
        return self._decay_rate
    
    @property
    def label(self) -> str | None:
        """Optional label for this neuron."""
        return self._label
    
    @property
    def fire_count(self) -> int:
        """Number of times this neuron has fired."""
        return self._fire_count
    
    @property
    def is_refractory(self) -> bool:
        """Check if neuron is in refractory period."""
        return self._refractory_ticks > 0
    
    @property
    def activation_ratio(self) -> float:
        """Ratio of current voltage to threshold (0.0 to 1.0+)."""
        return self._voltage / self._threshold
    
    def input(self, signal: float) -> None:
        """
        Add an input signal to the neuron's voltage.
        
        This is the "integrate" step of the LIF model. The signal
        weight is added directly to the membrane voltage.
        
        Args:
            signal: Signal weight to add. Typically in range [0, 1]
                   but can be any float. Negative signals inhibit.
        
        Note:
            If the neuron is in a refractory period, input is ignored.
        
        Complexity: O(1)
        """
        if self._refractory_ticks > 0:
            return  # Ignore input during refractory period
        
        self._voltage += signal
    
    def tick(self) -> None:
        """
        Apply one time step of voltage decay.
        
        This is the "leak" step of the LIF model. The voltage is
        multiplied by the decay rate, causing exponential decay
        towards zero.
        
        Also decrements the refractory counter if in refractory period.
        
        Complexity: O(1)
        """
        if self._refractory_ticks > 0:
            self._refractory_ticks -= 1
        
        self._voltage *= self._decay_rate
    
    def fire(self) -> bool:
        """
        Check if the neuron should fire and reset if so.
        
        This is the "fire" step of the LIF model. If voltage exceeds
        the threshold, the neuron fires (returns True) and voltage
        is reset to zero. Otherwise, returns False.
        
        Returns:
            True if neuron fired, False otherwise.
        
        Side Effects:
            - If fired: voltage is reset to 0, fire_count incremented,
              refractory period begins.
        
        Complexity: O(1)
        """
        if self._refractory_ticks > 0:
            return False
        
        if self._voltage > self._threshold:
            self._voltage = 0.0
            self._fire_count += 1
            self._refractory_ticks = self._refractory_period
            return True
        
        return False
    
    def check_and_fire(self) -> bool:
        """
        Combined tick + fire check (convenience method).
        
        Applies decay then checks for firing. Useful when processing
        signals in a streaming fashion.
        
        Returns:
            True if neuron fired, False otherwise.
        
        Complexity: O(1)
        """
        self.tick()
        return self.fire()
    
    def reset(self) -> None:
        """
        Reset the neuron to initial state.
        
        Clears voltage and refractory state. Does NOT reset fire_count
        (historical data is preserved).
        
        Complexity: O(1)
        """
        self._voltage = 0.0
        self._refractory_ticks = 0
    
    def hard_reset(self) -> None:
        """
        Full reset including fire count.
        
        Complexity: O(1)
        """
        self._voltage = 0.0
        self._refractory_ticks = 0
        self._fire_count = 0
    
    def get_state(self) -> dict[str, Any]:
        """
        Get the current state of the neuron.
        
        Returns:
            Dictionary with voltage, threshold, decay_rate, etc.
        """
        return {
            "voltage": self._voltage,
            "threshold": self._threshold,
            "decay_rate": self._decay_rate,
            "refractory_ticks": self._refractory_ticks,
            "refractory_period": self._refractory_period,
            "fire_count": self._fire_count,
            "label": self._label,
            "is_refractory": self.is_refractory,
            "activation_ratio": self.activation_ratio,
        }
    
    def __repr__(self) -> str:
        label_str = f", label={self._label!r}" if self._label else ""
        return (f"CarbonNeuron(voltage={self._voltage:.4f}, "
                f"threshold={self._threshold}, "
                f"decay_rate={self._decay_rate}{label_str})")
    
    def __str__(self) -> str:
        pct = (self._voltage / self._threshold) * 100
        label_str = f"[{self._label}] " if self._label else ""
        return f"{label_str}Neuron: {pct:.1f}% activated"
