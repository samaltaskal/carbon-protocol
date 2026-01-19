# Carbon Protocol SDK - Architecture Overview

## Version 2.0.0: Wake-on-Meaning Architecture

The Carbon Protocol SDK v2.0.0 implements a **neuromorphic "Wake-on-Meaning"** architecture that routes user inputs through three energy-optimized paths, minimizing unnecessary LLM invocations.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Carbon Protocol SDK v2.0.0                               │
│                     "Wake-on-Meaning" Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Input Text                                                                │
│        │                                                                     │
│        ▼                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     INGESTION LAYER (Section III.A)                    │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │  │
│  │  │   Signal     │───▶│   Neuron     │───▶│     Intent Detector      │ │  │
│  │  │  Extractor   │    │    Bank      │    │   (collect_fires())      │ │  │
│  │  │ (keywords)   │    │  (LIF model) │    │                          │ │  │
│  │  └──────────────┘    └──────────────┘    └──────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┬─────────┘  │
│                                                                 │            │
│                                           ┌─────────────────────┘            │
│                                           │ Fired Intents                    │
│                                           ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     SKILL REGISTRY (Section III.C)                     │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐ │  │
│  │  │   SkillDB    │───▶│   Pattern    │───▶│     Macro Store          │ │  │
│  │  │ (Trie/AC)    │    │   Matcher    │    │ (trace promotion)        │ │  │
│  │  │  O(1)/O(L)   │    │   O(L)       │    │                          │ │  │
│  │  └──────────────┘    └──────────────┘    └──────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┬─────────┘  │
│                                                                 │            │
│                                           ┌─────────────────────┘            │
│                                           │ Skill Match?                     │
│                                           ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     C-ISA LAYER (Section III.B)                        │  │
│  │                                                                        │  │
│  │       ┌──────────┐     ┌──────────┐     ┌──────────────────────┐      │  │
│  │       │ OP_IDLE  │     │ OP_MACRO │     │      OP_GEN          │      │  │
│  │       │  (0x00)  │     │  (0x01)  │     │      (0x02)          │      │  │
│  │       │   ~0 FL  │     │  ~1K FL  │     │      ~1T FL          │      │  │
│  │       └────┬─────┘     └────┬─────┘     └──────────┬───────────┘      │  │
│  │            │                │                      │                  │  │
│  └────────────┼────────────────┼──────────────────────┼──────────────────┘  │
│               │                │                      │                     │
│               ▼                ▼                      ▼                     │
│           No Action      Template         LLM Inference                     │
│                          Expansion        (Generative)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Ingestion Layer (`src/ingestion/`)

**Responsibility**: Neuromorphic signal processing using Leaky Integrate-and-Fire (LIF) neurons.

#### CarbonNeuron (`neuron.py`)

```python
class CarbonNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    
    Parameters:
        threshold: Firing threshold (default: 1.0)
        decay: Leak rate per timestep (default: 0.1)
    
    Complexity: O(1) arithmetic operations
    """
    def input(self, signal: float) -> bool:
        """Integrate signal, return True if fired."""
        self.potential = (self.potential * (1 - self.decay)) + signal
        if self.potential >= self.threshold:
            self.potential = 0.0  # Reset after firing
            return True
        return False
```

#### NeuronBank (`neuron_bank.py`)

```python
class NeuronBank:
    """
    Collection of neurons for multi-intent detection.
    
    Methods:
        add_neuron(intent: str) -> None
        fire(intent: str, signal: float) -> bool
        collect_fires() -> List[str]
    """
```

#### SignalExtractor (`signal_extractor.py`)

```python
class SignalExtractor:
    """
    Extract keyword signals from input text.
    
    Features:
        - Keyword matching (exact)
        - Bigram detection
        - Signal strength calculation
    """
```

### 2. Skill Registry (`src/registry/`)

**Responsibility**: O(L) pattern matching using Trie and Aho-Corasick algorithms.

#### SkillDB (`skill_db.py`)

```python
class SkillDB:
    """
    Self-optimizing skill database with trace promotion.
    
    Lookup Strategy:
        1. O(1) Hash: Exact match lookup
        2. O(L) Aho-Corasick: Substring matching
        3. Miss: Return None (triggers OP_GEN)
    
    Features:
        - Trace-to-macro promotion
        - Automatic skill discovery
        - Pattern normalization
    """
```

#### PatternMatcher (`pattern_matcher.py`)

```python
class PatternMatcher:
    """
    High-performance pattern matching using pyahocorasick.
    
    Complexity: O(L + k) where L = input length, k = matches
    """
```

### 3. C-ISA Layer (`src/c_isa/`)

**Responsibility**: Define the Carbon Instruction Set for deterministic routing.

#### CarbonOpCode (`opcodes.py`)

```python
from enum import Enum

class CarbonOpCode(Enum):
    IDLE = 0x00       # No action needed (~0 FLOPs)
    MACRO = 0x01      # Execute template (~1K FLOPs)
    GEN = 0x02        # LLM generation (~1T FLOPs)
    SCAFFOLD = 0x03   # Project scaffolding
    TRANSFORM = 0x04  # Data transformation
    DEBUG = 0x05      # Debugging assistance
```

#### CarbonInstruction (`instruction.py`)

```python
@dataclass
class CarbonInstruction:
    opcode: CarbonOpCode
    payload: Optional[str] = None
    metadata: Optional[Dict] = None
```

### 4. Semantic Compiler (`src/compiler/`)

**Responsibility**: Main router logic that orchestrates the Wake-on-Meaning pipeline.

#### SemanticCompiler (`semantic.py`)

```python
class SemanticCompiler:
    """
    Main router for the Wake-on-Meaning architecture.
    
    Pipeline:
        1. Extract signals from input
        2. Feed signals to neuron bank
        3. Collect fired intents
        4. Lookup skills in SkillDB
        5. Generate bytecode (IDLE/MACRO/GEN)
    """
```

## Algorithm: Three-Path Routing

### Path Selection Logic

```
Input Text
    │
    ├── [No Intent Detected] ──────────────▶ OP_IDLE (No LLM)
    │
    ├── [Intent + Skill Match] ────────────▶ OP_MACRO (Deterministic)
    │
    └── [Intent + No Skill Match] ─────────▶ OP_GEN (Generative)
```

### Why This Matters

| Path | FLOPs | Energy | Use Case |
|------|-------|--------|----------|
| IDLE | ~0 | Minimal | Greetings, confirmations |
| DETERMINISTIC | ~1K | Low | Cached patterns, templates |
| GENERATIVE | ~1T | High | Novel, complex queries |

**Goal**: Maximize IDLE + DETERMINISTIC paths, minimize GENERATIVE.

## Data Flow

```
Input → SignalExtractor → NeuronBank → SkillDB → SemanticCompiler → Bytecode
           │                  │            │              │
        Keywords         Fire/No-Fire   Match/Miss    OP_IDLE/MACRO/GEN
```

## Performance Characteristics

| Component | Complexity | Typical Latency |
|-----------|------------|-----------------|
| SignalExtractor | O(L × k) | ~50μs |
| CarbonNeuron.input() | O(1) | ~1μs |
| NeuronBank.collect_fires() | O(n) | ~10μs |
| SkillDB.lookup() | O(L) | ~100μs |
| SemanticCompiler.process() | O(L + k) | ~200μs |

### Comparison with Traditional Approaches

| Approach | Complexity | FLOPs for 1K input |
|----------|------------|---------------------|
| BERT Classifier | O(L × d²) | ~22B |
| Vector Similarity | O(L × n) | ~1M |
| **Carbon Protocol** | O(L + k) | ~1K |

## Design Decisions

### 1. Neuromorphic Gating
LIF neurons provide energy-efficient, O(1) intent detection without heavy ML classifiers.

### 2. Self-Optimizing Registry
Successful generative responses are promoted to deterministic macros (trace promotion).

### 3. Deterministic Output
Given the same input and skills, output is always identical - essential for protocol compliance.

### 4. Backward Compatibility
Legacy API (Registry, Compiler) still supported alongside new CarbonCompiler API.

## Future Considerations

- [ ] Hardware acceleration (neuromorphic chips)
- [ ] Distributed skill registry
- [ ] Real-time trace promotion
- [ ] Multi-language support
- [ ] Federated learning for skill discovery

## References

1. Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models.
2. Aho, A. V., & Corasick, M. J. (1975). Efficient string matching: An aid to bibliographic search.
3. Patterson, D., et al. (2021). Carbon Emissions and Large Neural Network Training.
