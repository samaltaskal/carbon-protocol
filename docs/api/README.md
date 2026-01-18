# Carbon Protocol SDK - API Reference

This directory contains the API reference documentation for the Carbon Protocol SDK.

## Modules

### Registry (`src.registry`)

The Registry class manages domain-specific compression rules and builds the Aho-Corasick automaton.

#### Class: `Registry`

```python
from src import Registry

registry = Registry(data_dir: str | Path | None = None)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `load_domain(domain_name: str)` | Load patterns from a YAML domain file | `int` - count of patterns loaded |
| `load_domains(*domain_names: str)` | Load multiple domains at once | `int` - total patterns loaded |
| `build_automaton()` | Compile patterns into Aho-Corasick automaton | `Automaton` |
| `get_automaton()` | Get automaton, building if necessary | `Automaton` |
| `get_stats()` | Get registry statistics | `dict` |
| `clear()` | Reset registry to initial state | `None` |

---

### Compiler (`src.compiler`)

The Compiler class performs high-performance text compression using the Aho-Corasick algorithm.

#### Class: `Compiler`

```python
from src import Compiler

compiler = Compiler(registry: Registry, preserve_whitespace: bool = True)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `compress(text: str)` | Compress text using loaded patterns | `CompressionResult` |
| `compress_batch(texts: list[str])` | Compress multiple texts | `list[CompressionResult]` |
| `get_stats()` | Get compiler statistics | `dict` |

---

### CompressionResult (`src.compiler`)

Dataclass containing compression output and metrics.

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

## Usage Example

```python
from src import Registry, Compiler

# Initialize and load domains
registry = Registry()
registry.load_domain('core')
registry.build_automaton()

# Create compiler and compress text
compiler = Compiler(registry)
result = compiler.compress("Please write a python script to scrape data")

print(f"Compressed: {result.compressed}")
print(f"Matches: {result.matches_found}")
print(f"Compression ratio: {result.compression_ratio:.2%}")
```
