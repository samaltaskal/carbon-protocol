# Carbon Protocol SDK Documentation

Welcome to the Carbon Protocol SDK v2.0.0 documentation. This folder contains all project documentation for the **Wake-on-Meaning** neuromorphic architecture.

## What's New in v2.0.0

- **Neuromorphic Gating**: Leaky Integrate-and-Fire (LIF) neurons for O(1) intent detection
- **Self-Optimizing Skill Registry**: Trie/Aho-Corasick pattern matching with trace promotion
- **Carbon Instruction Set (C-ISA)**: Deterministic opcodes for energy-efficient routing
- **Three-Path Architecture**: IDLE → DETERMINISTIC → GENERATIVE routing

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── api/                   # API reference documentation
│   └── README.md          # Module reference (Registry, Compiler, CarbonCompiler)
├── guides/                # User guides and tutorials
│   ├── getting-started.md # Quick start guide
│   └── validation-testing.md  # Impact assessment guide
└── architecture/          # Architecture and design documents
    ├── overview.md        # System architecture overview
    └── IEEE_INDUSTRY_IMPACT_REPORT.md  # Industry impact analysis
```

## Quick Links

- [Getting Started](guides/getting-started.md) - Quick start guide for both new and legacy APIs
- [API Reference](api/README.md) - Complete API documentation
- [Architecture Overview](architecture/overview.md) - Wake-on-Meaning system design
- [Validation Testing](guides/validation-testing.md) - Environmental impact assessment

## Key Concepts

### Wake-on-Meaning Architecture

The v2.0.0 architecture routes inputs through three energy-optimized paths:

| Path | Trigger | FLOPs | Description |
|------|---------|-------|-------------|
| **IDLE** | No intent detected | ~0 | No LLM needed |
| **DETERMINISTIC** | Skill matched | ~1K | Template expansion via macro |
| **GENERATIVE** | No match | ~1T | Full LLM inference |

### Components

1. **Ingestion Layer** (`src/ingestion/`) - Neuromorphic signal processing
2. **Skill Registry** (`src/registry/`) - O(L) pattern matching
3. **C-ISA Layer** (`src/c_isa/`) - Instruction set opcodes
4. **Semantic Compiler** (`src/compiler/`) - Main router logic

## Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build HTML documentation
cd docs && sphinx-build -b html . _build/html
```

## Running Tests

```bash
# Run all 54 neuromorphic architecture tests
pytest tests/test_neuromorphic.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```
