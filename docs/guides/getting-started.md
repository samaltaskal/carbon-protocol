# Getting Started with Carbon Protocol SDK

This guide will help you get started with the Carbon Protocol SDK for deterministic semantic compression.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install pyahocorasick pyyaml
```

### Install from Source

```bash
git clone <repository-url>
cd carbon-protocol
pip install -e .
```

## Quick Start

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
Compressed:  @LANG:PY to @ACT:SCRAPE?
Matches:    3
Saved:      35 bytes
```

## Loading Multiple Domains

You can load multiple domains to expand the compression vocabulary:

```python
registry = Registry()
registry.load_domains('core', 'k8s', 'python')  # Load multiple domains
registry.build_automaton()
```

## Creating Custom Domains

Create a YAML file in `src/data/` with your custom patterns:

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

## Performance Considerations

The SDK uses the Aho-Corasick algorithm for O(n) complexity:

- **Build time**: O(m) where m = total characters in patterns
- **Match time**: O(n + z) where n = input length, z = matches
- **Target latency**: < 10ms for typical prompts

## Next Steps

- [API Reference](../api/README.md) - Complete API documentation
- [Architecture Overview](../architecture/overview.md) - System design details
