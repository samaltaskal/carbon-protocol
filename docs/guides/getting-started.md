# Getting Started with Carbon Protocol SDK v2.0.0

This guide will help you get started with the Carbon Protocol SDK's **Wake-on-Meaning** neuromorphic architecture.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install from Source

```bash
# Clone repository
git clone https://github.com/samaltaskal/carbon-protocol.git
cd carbon-protocol

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

```bash
pip install pyahocorasick pyyaml
```

## Quick Start: New API (v2.0.0)

The v2.0.0 API uses the `CarbonCompiler` class for Wake-on-Meaning routing.

### Basic Usage

```python
from src import CarbonCompiler

# Create compiler with default configuration
compiler = CarbonCompiler.create_default()

# Add a skill to the registry
compiler.add_skill(
    name="python_hello",
    pattern="hello world python",
    template="print('Hello, World!')",
)

# Process input through the Wake-on-Meaning pipeline
result = compiler.process("Write hello world in python")

# Check the routing decision
if result.is_deterministic:
    print("✓ Using cached skill (no LLM needed)")
    print(f"  OpCode: {result.primary_opcode}")
    print(f"  Template: {result.payload}")
elif result.is_generative:
    print("→ Falling back to LLM generation")
else:
    print("○ No action needed (IDLE)")
```

### Understanding the Routing Paths

The compiler routes inputs through three energy-optimized paths:

| Path | Property | FLOPs | When |
|------|----------|-------|------|
| IDLE | `result.is_idle` | ~0 | No intent detected |
| DETERMINISTIC | `result.is_deterministic` | ~1K | Skill matched |
| GENERATIVE | `result.is_generative` | ~1T | No skill match |

### Recording Generations for Promotion

When the LLM generates a useful response, record it for future promotion:

```python
# After receiving LLM response
if result.is_generative:
    llm_response = call_your_llm(result.input_text)
    
    # Record for automatic skill promotion
    compiler.record_generation(
        input_text=result.input_text,
        output_text=llm_response
    )
```

## Quick Start: Legacy API

The legacy API is still fully supported for backward compatibility.

### Basic Usage

```python
from src import Registry, Compiler

# 1. Create a Registry and load domain patterns
registry = Registry()
registry.load_domain('core')  # Load the core domain
registry.build_automaton()    # Build the Aho-Corasick automaton

# 2. Create a Compiler with the registry
compiler = Compiler(registry)

# 3. Compress text
result = compiler.compress("Could you please write a python script to scrape data?")

print(f"Original:   {result.original}")
print(f"Compressed: {result.compressed}")
print(f"Matches:    {result.matches_found}")
print(f"Saved:      {result.bytes_saved} bytes")
```

### Output

```
Original:   Could you please write a python script to scrape data?
Compressed: @LANG:PY to @ACT:SCRAPE?
Matches:    3
Saved:      35 bytes
```

### Loading Multiple Domains

```python
registry = Registry()
registry.load_domains('core', 'lang_python', 'cloud_aws')
registry.build_automaton()
```

## Creating Custom Skills

### New API (v2.0.0)

```python
from src import CarbonCompiler

compiler = CarbonCompiler.create_default()

# Add skills programmatically
compiler.add_skill(
    name="k8s_deploy",
    pattern="deploy kubernetes",
    template="kubectl apply -f deployment.yaml",
)

compiler.add_skill(
    name="docker_build",
    pattern="build docker image",
    template="docker build -t myapp:latest .",
)
```

### Legacy API (Custom YAML)

Create a YAML file in `src/data/`:

```yaml
# src/data/custom.yaml
version: "1.0"
domain: "custom"
description: "Custom compression patterns"

patterns:
  - input: "my custom phrase"
    output: "@CUSTOM:TOKEN"
    
  - input: "remove this"
    output: ""  # Empty output = removal
```

Then load it:

```python
registry.load_domain('custom')
```

## Understanding the Architecture

### Wake-on-Meaning Pipeline

```
Input Text
    │
    ▼
┌─────────────────┐
│ Signal Extractor│──▶ Keywords, bigrams
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Neuron Bank   │──▶ LIF neurons fire/don't fire
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Skill DB     │──▶ O(L) pattern matching
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Semantic Compiler│──▶ OP_IDLE / OP_MACRO / OP_GEN
└─────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| CarbonNeuron | `src/ingestion/neuron.py` | LIF neuron model |
| NeuronBank | `src/ingestion/neuron_bank.py` | Multi-intent detection |
| SignalExtractor | `src/ingestion/signal_extractor.py` | Keyword extraction |
| SkillDB | `src/registry/skill_db.py` | Trie/Aho-Corasick lookup |
| CarbonOpCode | `src/c_isa/opcodes.py` | Instruction set |
| SemanticCompiler | `src/compiler/semantic.py` | Main router |

## Performance Considerations

### Complexity Comparison

| Approach | Complexity | 1000 patterns, 10KB text |
|----------|------------|--------------------------|
| Naive loop | O(m × n) | ~10,000,000 operations |
| **Aho-Corasick** | O(n + z) | ~10,000 operations |
| **Carbon v2.0** | O(L + k) | ~1,000 operations |

### Typical Latencies

| Operation | Time |
|-----------|------|
| Signal extraction | ~50μs |
| Neuron firing | ~1μs |
| Skill lookup | ~100μs |
| Full pipeline | ~200μs |

## Running Tests

```bash
# Run all 54 neuromorphic architecture tests
pytest tests/test_neuromorphic.py -v

# Run legacy tests
pytest tests/test_compiler.py tests/test_registry.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Next Steps

- [API Reference](../api/README.md) - Complete API documentation
- [Architecture Overview](../architecture/overview.md) - Deep dive into Wake-on-Meaning
- [Validation Testing](validation-testing.md) - Environmental impact assessment

## Troubleshooting

### Import Errors

Ensure you're in the project root and have installed in editable mode:

```bash
pip install -e .
```

### pyahocorasick Not Found

Install the C extension:

```bash
pip install pyahocorasick
```

### Tests Failing

Make sure all dependencies are installed:

```bash
pip install -e ".[dev]"
```
