# Carbon Protocol SDK

A Neuromorphic "Semantic Compiler" that shifts LLM optimization to the Client-Side Edge using deterministic O(L) pattern matching and biologically-inspired Leaky Integrate-and-Fire (LIF) neurons.

![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Patent%20Pending-blue)
![Paper](https://img.shields.io/badge/ICT4S-Submitted-orange)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18284932-blue)](https://doi.org/10.5281/zenodo.18284932)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-54%20passing-brightgreen)
![Version](https://img.shields.io/badge/version-2.0.0-blue)

> **"Wake-on-Meaning" Architecture:** Neuromorphic gating with O(L) intent detection replaces heavy ML classifiers.

**Paper Submission:** [Zero-Overhead Prompt Compression: A Deterministic Protocol for Energy-Efficient Generative AI](https://doi.org/10.5281/zenodo.18284932) (Submitted to ICT4S 2026)

## Overview

The Carbon Protocol SDK v2.0.0 implements a **neuromorphic "Wake-on-Meaning"** architecture that routes user inputs through three energy-optimized paths:

| Path | Trigger | FLOPs | Description |
|------|---------|-------|-------------|
| **IDLE** | No intent detected | ~0 | No LLM needed |
| **DETERMINISTIC** | Skill matched | ~1K | Template expansion via macro |
| **GENERATIVE** | No match | ~1T | Full LLM inference |

**Goal:** Maximize IDLE + DETERMINISTIC paths, minimize GENERATIVE.

## Key Innovations (v2.0.0)

### 1. Neuromorphic Gating (Section III.A)
- **CarbonNeuron**: Leaky Integrate-and-Fire (LIF) model with O(1) arithmetic
- **NeuronBank**: Multi-intent detection with signal routing
- **No ML in Hot Path**: Pure Python arithmetic, no numpy/torch dependencies

### 2. Self-Optimizing Skill Registry (Section III.C)
- **SkillDB**: Trie/Aho-Corasick pattern matching
- **O(1) Hash**: Exact match lookup
- **O(L) Aho-Corasick**: Substring matching
- **Trace Promotion**: Generative responses evolve into deterministic macros

### 3. Carbon Instruction Set (C-ISA) (Section III.B)
```python
CarbonOpCode.IDLE      # No action needed
CarbonOpCode.MACRO     # Execute deterministic template
CarbonOpCode.GEN       # Fallback to LLM
CarbonOpCode.SCAFFOLD  # Project scaffolding
CarbonOpCode.TRANSFORM # Data transformation
```

### 4. Semantic Compiler Pipeline (Section IV)
```
Input -> [Neuron Bank] -> [Skill DB] -> [Bytecode]
             |                |
         [IDLE?]         [MATCH?]
             |                |
        OP_IDLE          OP_MACRO -----> Deterministic
             |                |
             +----------------+---> OP_GEN ---> Generative
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/samaltaskal/carbon-protocol.git
cd carbon-protocol

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### New API (v2.0.0) - Wake-on-Meaning

```python
from src import CarbonCompiler

# Create compiler with default configuration
compiler = CarbonCompiler.create_default()

# Add skills to the registry
compiler.add_skill(
    name="python_hello",
    pattern="hello world python",
    template="print('Hello, World!')",
)

# Process input through the Wake-on-Meaning pipeline
result = compiler.process("Write hello world in python")

if result.is_deterministic:
    print("Using cached skill (no LLM needed)")
    print(f"OpCode: {result.primary_opcode}")
elif result.is_generative:
    print("Falling back to LLM generation")
    # After LLM responds, record for promotion:
    # compiler.record_generation(input_text, output_text)
else:
    print("No action needed (IDLE)")
```

### Legacy API (Still Supported)

```python
from src import Registry, Compiler

# Initialize and load domain patterns
registry = Registry()
registry.load_domain('core')
registry.build_automaton()

# Create compiler and compress
compiler = Compiler(registry)
result = compiler.compress("Could you please write a python script?")

print(f"Original:   {result.original}")
print(f"Compressed: {result.compressed}")
```

## Project Structure

```
carbon-protocol/
├── src/
│   ├── ingestion/           # Neuromorphic Layer (Section III.A)
│   │   ├── neuron.py        # CarbonNeuron LIF model
│   │   ├── neuron_bank.py   # Multi-intent detection
│   │   ├── signal_extractor.py  # Keyword/bigram matching
│   │   └── intent_detector.py   # Full detection pipeline
│   ├── registry/            # Skill Database (Section III.C)
│   │   ├── skill_db.py      # Trie/Aho-Corasick lookup
│   │   ├── pattern_matcher.py   # O(L) pattern matching
│   │   └── macro_store.py   # Trace-to-macro promotion
│   ├── c_isa/               # Carbon Instruction Set (Section III.B)
│   │   ├── opcodes.py       # CarbonOpCode enum
│   │   ├── instruction.py   # CarbonInstruction class
│   │   └── bytecode.py      # CarbonBytecode serialization
│   ├── compiler/            # Semantic Compiler (Section IV)
│   │   └── semantic.py      # Main router logic
│   ├── main.py              # Entry point
│   └── data/                # Domain YAML files
├── tests/
│   ├── test_neuromorphic.py # v2.0.0 tests (54 tests)
│   └── ...                  # Legacy tests
├── docs/
│   ├── api/                 # API reference
│   ├── guides/              # User guides
│   └── architecture/        # System design
└── pyproject.toml
```

## Testing

```bash
# Run neuromorphic architecture tests (54 tests)
pytest tests/test_neuromorphic.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance Characteristics

| Component | Complexity | Typical Time |
|-----------|------------|--------------|
| CarbonNeuron.input() | O(1) | ~1μs |
| NeuronBank.collect_fires() | O(k) | ~10μs |
| SkillDB.lookup() | O(L) | ~100μs |
| CarbonCompiler.process() | O(L + k) | ~200μs |

**vs Traditional Approaches:**
- BERT Classifier: O(L × d²) where d = 768+ hidden dim → ~22B FLOPs
- Vector Similarity: O(L × n) where n = corpus size → ~1M FLOPs
- **Carbon Protocol**: O(L + k) → ~1K FLOPs

## Environmental Impact

Based on validation testing:

| Scale | Requests/Year | Carbon Saved | Cost Saved |
|-------|---------------|--------------|------------|
| Small Org | 1M | 0.91 kg CO2 | $64 |
| Medium Org | 10M | 9.12 kg CO2 | $640 |
| Large Org | 100M | 91.2 kg CO2 | $6,400 |
| Enterprise | 1B | 912 kg CO2 | $64,000 |

## Documentation

- [Getting Started](docs/guides/getting-started.md) - Quick start guide
- [API Reference](docs/api/README.md) - Complete API documentation  
- [Architecture Overview](docs/architecture/overview.md) - System design
- [Validation Testing](docs/guides/validation-testing.md) - Impact assessment

## Research References

The Carbon Protocol implements concepts from:

1. **Neuromorphic Computing**: LIF neuron model (Maass, 1997)
2. **Aho-Corasick Algorithm**: O(n) multi-pattern matching (Aho & Corasick, 1975)
3. **Energy-Efficient AI**: Patterson et al., 2021 - Carbon emissions research

## Contributing

Contributions welcome! Areas of interest:
- Additional domain YAML files
- New intent detection keywords
- Performance optimizations
- Documentation improvements

## License

MIT License - See [LICENSE](LICENSE) file

## IP Notice

Copyright (c) 2026 Taskal Samal. This reference implementation is released under the MIT License to foster academic collaboration and sustainable AI development.

**Patent Pending** (USPTO Application No. 63/961,716)
