"""
Carbon Protocol SDK - Neuromorphic Ingestion Layer

This module implements the "Wake-on-Meaning" architecture using
biologically-inspired Leaky Integrate-and-Fire (LIF) neurons for
efficient intent detection at the client-side edge.

Reference: Carbon Protocol Research Paper
    "Neuromorphic Gating for Sparse Activation" - Section 3.1

Design Philosophy:
    Traditional NLP approaches use heavy neural networks (BERT, etc.)
    for intent classification. The Carbon Protocol replaces this with
    a lightweight neuromorphic approach:
    
    1. Each "neuron" corresponds to an intent category
    2. Input signals (keyword matches) add voltage to neurons
    3. Neurons leak over time (decay), modeling relevance decay
    4. When voltage exceeds threshold, the neuron "fires"
    5. Firing indicates high-confidence intent detection
    
    This achieves O(L) complexity (L = input length) using only
    native arithmetic operations - no matrix multiplications.

Energy Efficiency:
    - Standard BERT classifier: ~110M parameters, ~22B FLOPs/inference
    - Carbon LIF Bank: ~100 neurons, ~1K FLOPs/inference
    - Efficiency gain: ~22,000,000x fewer FLOPs
"""

from .neuron import CarbonNeuron, NeuronConfig
from .neuron_bank import NeuronBank, NeuronBankConfig
from .signal_extractor import SignalExtractor, SignalConfig
from .intent_detector import IntentDetector, IntentResult, DetectorConfig

__all__ = [
    # Core neuron
    "CarbonNeuron",
    "NeuronConfig",
    # Neuron bank
    "NeuronBank",
    "NeuronBankConfig",
    # Signal extraction
    "SignalExtractor",
    "SignalConfig",
    # Intent detection
    "IntentDetector",
    "IntentResult",
    "DetectorConfig",
]
