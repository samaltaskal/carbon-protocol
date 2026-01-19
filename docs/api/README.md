# Carbon Protocol SDK - API Reference

This document provides the complete API reference for the Carbon Protocol SDK v2.0.0.

## Table of Contents

- [New API (v2.0.0)](#new-api-v200)
  - [CarbonCompiler](#carboncompiler)
  - [CompilerResult](#compilerresult)
  - [CarbonOpCode](#carbonopcode)
- [Ingestion Layer](#ingestion-layer)
  - [CarbonNeuron](#carbonneuron)
  - [NeuronBank](#neuronbank)
  - [SignalExtractor](#signalextractor)
- [Registry Layer](#registry-layer)
  - [SkillDB](#skilldb)
  - [PatternMatcher](#patternmatcher)
- [Legacy API](#legacy-api)
  - [Registry](#registry)
  - [Compiler](#compiler)
  - [CompressionResult](#compressionresult)

---

## New API (v2.0.0)

### CarbonCompiler

The main entry point for the Wake-on-Meaning architecture.

```python
from src import CarbonCompiler

compiler = CarbonCompiler.create_default()
```

#### Class: `CarbonCompiler`

**Constructor:**

```python
CarbonCompiler(
    neuron_bank: NeuronBank,
    skill_db: SkillDB,
    signal_extractor: SignalExtractor
)
```

**Class Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `create_default()` | Create compiler with default configuration | `CarbonCompiler` |

**Instance Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `process(input_text: str)` | Process input through Wake-on-Meaning pipeline | `CompilerResult` |
| `add_skill(name: str, pattern: str, template: str)` | Add a skill to the registry | `None` |
| `record_generation(input_text: str, output_text: str)` | Record LLM response for promotion | `None` |
| `get_stats()` | Get compiler statistics | `dict` |

**Example:**

```python
from src import CarbonCompiler

compiler = CarbonCompiler.create_default()

# Add skills
compiler.add_skill(
    name="python_hello",
    pattern="hello world python",
    template="print('Hello, World!')",
)

# Process input
result = compiler.process("Write hello world in python")

if result.is_deterministic:
    print(f"Template: {result.payload}")
```

---

### CompilerResult

Result from `CarbonCompiler.process()`.

#### Class: `CompilerResult`

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `input_text` | `str` | Original input text |
| `primary_opcode` | `CarbonOpCode` | Routing decision |
| `payload` | `Optional[str]` | Template or additional data |
| `fired_intents` | `List[str]` | Intents detected by neurons |
| `matched_skills` | `List[str]` | Skills matched in registry |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_idle` | `bool` | True if routed to IDLE path |
| `is_deterministic` | `bool` | True if routed to MACRO path |
| `is_generative` | `bool` | True if routed to GEN path |

---

### CarbonOpCode

Instruction set opcodes for routing decisions.

```python
from src.c_isa import CarbonOpCode
```

#### Enum: `CarbonOpCode`

| Value | Code | FLOPs | Description |
|-------|------|-------|-------------|
| `IDLE` | `0x00` | ~0 | No action needed |
| `MACRO` | `0x01` | ~1K | Execute deterministic template |
| `GEN` | `0x02` | ~1T | Fallback to LLM generation |
| `SCAFFOLD` | `0x03` | ~1K | Project scaffolding |
| `TRANSFORM` | `0x04` | ~1K | Data transformation |
| `DEBUG` | `0x05` | varies | Debugging assistance |

---

## Ingestion Layer

### CarbonNeuron

Leaky Integrate-and-Fire (LIF) neuron model.

```python
from src.ingestion import CarbonNeuron
```

#### Class: `CarbonNeuron`

**Constructor:**

```python
CarbonNeuron(
    intent: str,
    threshold: float = 1.0,
    decay: float = 0.1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intent` | `str` | required | Intent this neuron represents |
| `threshold` | `float` | `1.0` | Firing threshold |
| `decay` | `float` | `0.1` | Leak rate per timestep |

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `input(signal: float)` | Integrate signal, return True if fired | `bool` |
| `reset()` | Reset neuron potential to 0 | `None` |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `potential` | `float` | Current membrane potential |
| `has_fired` | `bool` | Whether neuron fired this cycle |

**Example:**

```python
from src.ingestion import CarbonNeuron

neuron = CarbonNeuron("code_gen", threshold=1.0, decay=0.1)

# Integrate signals
neuron.input(0.3)  # False
neuron.input(0.4)  # False
neuron.input(0.5)  # True (fired!)
```

---

### NeuronBank

Collection of neurons for multi-intent detection.

```python
from src.ingestion import NeuronBank
```

#### Class: `NeuronBank`

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `add_neuron(intent: str, threshold: float, decay: float)` | Add a neuron | `CarbonNeuron` |
| `fire(intent: str, signal: float)` | Send signal to specific neuron | `bool` |
| `fire_all(signals: Dict[str, float])` | Send signals to multiple neurons | `List[str]` |
| `collect_fires()` | Get list of fired intents | `List[str]` |
| `reset_all()` | Reset all neurons | `None` |

---

### SignalExtractor

Extract keyword signals from input text.

```python
from src.ingestion import SignalExtractor
```

#### Class: `SignalExtractor`

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `extract(text: str)` | Extract signals from text | `Dict[str, float]` |
| `add_keyword(keyword: str, intent: str, weight: float)` | Add keyword mapping | `None` |
| `add_bigram(bigram: Tuple[str, str], intent: str, weight: float)` | Add bigram mapping | `None` |

---

## Registry Layer

### SkillDB

Self-optimizing skill database with Trie/Aho-Corasick lookup.

```python
from src.registry import SkillDB
```

#### Class: `SkillDB`

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `add_skill(name: str, pattern: str, template: str)` | Add a skill | `None` |
| `lookup(text: str)` | Find matching skills | `Optional[Skill]` |
| `promote_trace(input_text: str, output_text: str)` | Promote LLM trace to skill | `None` |
| `build_automaton()` | Build Aho-Corasick automaton | `None` |
| `get_stats()` | Get registry statistics | `dict` |

**Complexity:**

| Operation | Complexity |
|-----------|------------|
| `lookup()` (exact match) | O(1) |
| `lookup()` (substring) | O(L) |
| `add_skill()` | O(p) |
| `build_automaton()` | O(m) |

---

### PatternMatcher

High-performance pattern matching using pyahocorasick.

```python
from src.registry import PatternMatcher
```

#### Class: `PatternMatcher`

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `add_pattern(pattern: str, value: Any)` | Add pattern to automaton | `None` |
| `build()` | Finalize automaton construction | `None` |
| `match(text: str)` | Find all matches in text | `List[Match]` |
| `match_first(text: str)` | Find first match | `Optional[Match]` |

---

## Legacy API

### Registry

The Registry class manages domain-specific compression rules (legacy API).

```python
from src import Registry
```

#### Class: `Registry`

**Constructor:**

```python
Registry(data_dir: str | Path | None = None)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `load_domain(domain_name: str)` | Load patterns from YAML domain file | `int` |
| `load_domains(*domain_names: str)` | Load multiple domains at once | `int` |
| `build_automaton()` | Compile patterns into Aho-Corasick automaton | `Automaton` |
| `get_automaton()` | Get automaton, building if necessary | `Automaton` |
| `get_stats()` | Get registry statistics | `dict` |
| `clear()` | Reset registry to initial state | `None` |

---

### Compiler

The Compiler class performs high-performance text compression (legacy API).

```python
from src import Compiler
```

#### Class: `Compiler`

**Constructor:**

```python
Compiler(registry: Registry, preserve_whitespace: bool = True)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `compress(text: str)` | Compress text using loaded patterns | `CompressionResult` |
| `compress_batch(texts: list[str])` | Compress multiple texts | `list[CompressionResult]` |
| `get_stats()` | Get compiler statistics | `dict` |

---

### CompressionResult

Dataclass containing compression output and metrics (legacy API).

#### Class: `CompressionResult`

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `original` | `str` | Original input text |
| `compressed` | `str` | Compressed output text |
| `matches_found` | `int` | Number of patterns matched |
| `compression_ratio` | `float` | Ratio of compressed/original length |
| `bytes_saved` | `int` | Bytes saved by compression |

---

## Usage Examples

### New API Example

```python
from src import CarbonCompiler

# Create and configure compiler
compiler = CarbonCompiler.create_default()

# Add custom skills
compiler.add_skill("k8s_deploy", "deploy to kubernetes", "kubectl apply -f")
compiler.add_skill("docker_build", "build docker", "docker build -t")

# Process inputs
for prompt in ["Deploy my app to kubernetes", "Build docker image"]:
    result = compiler.process(prompt)
    
    if result.is_deterministic:
        print(f"✓ {prompt}")
        print(f"  → {result.payload}")
    else:
        print(f"→ {prompt} (needs LLM)")
```

### Legacy API Example

```python
from src import Registry, Compiler

# Initialize and load domains
registry = Registry()
registry.load_domains('core', 'lang_python', 'cloud_aws')
registry.build_automaton()

# Create compiler and compress
compiler = Compiler(registry)
result = compiler.compress("Please write a python script for AWS Lambda")

print(f"Compressed: {result.compressed}")
print(f"Ratio: {result.compression_ratio:.2%}")
```
