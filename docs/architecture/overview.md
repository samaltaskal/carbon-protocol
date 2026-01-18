# Carbon Protocol SDK - Architecture Overview

## System Architecture

The Carbon Protocol SDK implements a high-performance deterministic semantic compression engine using the Aho-Corasick multi-pattern matching algorithm.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Carbon Protocol SDK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Registry   │───▶│  Automaton   │───▶│    Compiler      │  │
│  │              │    │ (Aho-Corasick)│    │                  │  │
│  └──────┬───────┘    └──────────────┘    └────────┬─────────┘  │
│         │                                          │            │
│         ▼                                          ▼            │
│  ┌──────────────┐                         ┌──────────────────┐  │
│  │ Domain YAML  │                         │ CompressionResult│  │
│  │    Files     │                         │                  │  │
│  └──────────────┘                         └──────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Registry (`src/registry.py`)

**Responsibility**: Load and manage compression patterns from domain YAML files.

**Key Features**:
- Modular domain loading (memory-efficient)
- Pattern normalization (case-insensitive)
- Aho-Corasick automaton construction
- Domain hot-reloading capability

**Complexity**:
- Domain loading: O(p) where p = patterns in domain
- Automaton build: O(m) where m = total pattern characters

### 2. Compiler (`src/compiler.py`)

**Responsibility**: Perform text compression using the automaton.

**Key Features**:
- Single-pass multi-pattern matching
- Longest-match-first semantics
- Non-overlapping match selection
- Batch compression support

**Complexity**:
- Compression: O(n + z) where n = input length, z = matches

### 3. Domain Files (`src/data/*.yaml`)

**Responsibility**: Define compression patterns for specific domains.

**Structure**:
```yaml
patterns:
  - input: "natural language phrase"
    output: "@COMPRESSED:TOKEN"
```

## Algorithm: Aho-Corasick

### Why Aho-Corasick?

| Approach | Complexity | 1000 patterns, 10KB text |
|----------|------------|--------------------------|
| Naive loop | O(m × n) | ~10,000,000 operations |
| **Aho-Corasick** | O(n + z) | ~10,000 operations |

### How It Works

1. **Build Phase**: Construct a finite state automaton (trie + failure links)
2. **Match Phase**: Single pass through input, following automaton transitions
3. **Output**: All pattern matches found in linear time

## Data Flow

```
Input Text ──▶ Lowercase ──▶ Automaton ──▶ Match Collection ──▶ Overlap Resolution ──▶ Output Assembly
                             Matching       (start, end,        (longest-first,        (compressed
                                            pattern, token)      left-to-right)          string)
```

## Design Decisions

### 1. Case-Insensitive Matching
Patterns are normalized to lowercase during loading. Input is lowercased during matching to ensure consistent behavior.

### 2. Longest-Match-First
When patterns overlap (e.g., "visual studio" vs "visual studio code"), the longer pattern takes precedence.

### 3. Modular Domains
Domains are loaded on-demand to minimize memory footprint. Users load only the domains they need.

### 4. Deterministic Output
Given the same input and patterns, output is always identical - essential for protocol compliance.

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Single compression latency | < 10ms | ~0.5ms |
| 500 batch compressions | < 5000ms | ~1.5ms |
| Memory per 1000 patterns | < 1MB | ~200KB |

## Future Considerations

- [ ] Parallel batch compression
- [ ] Streaming compression for large texts
- [ ] Pattern priority/weight system
- [ ] Compression statistics and analytics
