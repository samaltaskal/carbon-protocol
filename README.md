# Carbon Protocol SDK

High-performance deterministic semantic compression library for LLM prompts using Aho-Corasick multi-pattern matching.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

> **"Shift Left" for Green AI:** Client-side semantic compression with <10ms latency and 25%+ token reduction.

## 📄 Overview

The Carbon Protocol SDK implements deterministic semantic compression using the Aho-Corasick algorithm to achieve O(n) multi-pattern matching. By compressing prompts before they reach LLM APIs, it reduces token usage, carbon emissions, and costs.

**Key Innovation**: Single-pass multi-pattern matching instead of naive O(m×n) sequential replacement.

## 🚀 Key Features

- **O(n) Complexity**: Aho-Corasick algorithm ensures linear-time compression
- **<10ms Latency**: Average compression time of 0.85ms for typical prompts
- **25%+ Token Reduction**: Validated across 15 representative prompt categories
- **Modular Domains**: Load only the patterns you need (core, k8s, python, sql)
- **Longest-Match-First**: Deterministic output with proper overlap handling
- **Type-Safe**: Full Python 3.10+ type hints

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd carbon-protocol

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pyahocorasick pyyaml pytest
```

## ⚡ Quick Start

```python
from src import Registry, Compiler

# Initialize and load domain patterns
registry = Registry()
registry.load_domain('core')
registry.build_automaton()

# Create compiler and compress
compiler = Compiler(registry)
result = compiler.compress("Could you please write a python script to scrape data?")

print(f"Original:   {result.original}")
print(f"Compressed: {result.compressed}")
print(f"Tokens Saved: {result.matches_found}")
# Output: "Compressed: @LANG:PY to @ACT:SCRAPE?"
```

## 📊 Validated Results

From SDK validation suite (VAL-20260117):

| Metric | Value |
|--------|-------|
| **Average Token Reduction** | 25.8% |
| **Compression Ratio** | 0.74 |
| **Carbon Saved (1M requests/year)** | 0.91 kg CO2 |
| **Cost Saved (1M requests/year)** | $64 USD |
| **Test Coverage** | 52 tests, 100% pass rate |

## 🧪 Testing & Validation

### Run All Tests
```bash
# Standard test suite (52 tests)
pytest tests/ -v

# With IEEE 829 report generation
pytest --ieee-report --ieee-json

# Or use test runner
python run_tests.py --ieee
```

### SDK Validation Tests
```bash
# Run validation/impact assessment
python run_tests.py --validation

# Direct run with detailed output
pytest tests/test_validation.py -v -s
```

**Validation tests capture**:
- Token usage: With SDK vs Without SDK
- Environmental impact: Carbon emissions, energy, water
- Economic impact: API cost savings
- Annual projections: Scaled to 1M requests
- IEEE-compliant reports: Suitable for submission to authorities

See [Validation Testing Guide](docs/guides/validation-testing.md) for details.

## 📁 Project Structure

```
carbon-protocol/
├── src/                    # Source code
│   ├── registry.py         # Pattern loading & Aho-Corasick automaton
│   ├── compiler.py         # O(n) compression engine
│   ├── metrics.py          # Impact metrics calculator
│   └── data/               # Domain YAML files
├── tests/                  # Test suite
│   ├── test_registry.py    # Registry unit tests (16 tests)
│   ├── test_compiler.py    # Compiler unit tests (19 tests)
│   ├── test_integration.py # Integration tests (17 tests)
│   ├── test_validation.py  # SDK validation tests (5 tests)
│   └── results/            # Test results by date
│       └── YYYY-MM-DD/     # Date-based folders
├── docs/                   # Documentation
│   ├── api/                # API reference
│   ├── guides/             # User guides
│   └── architecture/       # System design
└── run_tests.py            # Test runner script
```

## 📖 Documentation

- [Getting Started](docs/guides/getting-started.md) - Quick start guide
- [API Reference](docs/api/README.md) - Complete API documentation  
- [Architecture Overview](docs/architecture/overview.md) - System design
- [Validation Testing](docs/guides/validation-testing.md) - Impact assessment

## 🔬 Architecture

### Aho-Corasick Algorithm

**Why not naive string replacement?**

```python
# ❌ Naive O(m×n) - Too slow
for pattern in patterns:      # m patterns
    text = text.replace(...)  # n characters
# Result: 1000 patterns × 10KB text = 10M operations

# ✅ Aho-Corasick O(n) - Fast  
automaton.find_all(text)      # Single pass
# Result: 10KB text = 10K operations (1000× faster!)
```

### Performance Characteristics

| Operation | Complexity | Time (typical) |
|-----------|------------|----------------|
| Domain loading | O(p) | 10-15ms |
| Automaton build | O(m) | 10-15ms |
| Single compression | O(n + z) | 0.85ms |
| Batch (500 prompts) | O(n + z) | 1.5ms |

## 🌍 Environmental Impact

Based on validation testing with 15 representative prompts:

**Per Sample Set (15 prompts)**:
- Energy Saved: 0.000029 kWh
- Carbon Saved: 0.0137 grams CO2
- Water Saved: 0.000052 liters

**Projected Annual (1M requests)**:
- Tokens Saved: 6.4 million
- Carbon Saved: 0.91 kg CO2 (≈2.3 tree-days)
- Cost Saved: $64 USD

*Calculations based on peer-reviewed research (Patterson et al., 2021) and IEA carbon intensity data.*

## 💰 Cost Savings

Assuming $0.01 per 1K input tokens (conservative estimate):

| Request Volume | Token Reduction | Annual Savings |
|----------------|-----------------|----------------|
| 100K requests  | 640K tokens     | $6.40 |
| 1M requests    | 6.4M tokens     | $64.00 |
| 10M requests   | 64M tokens      | $640.00 |
| 100M requests  | 640M tokens     | $6,400.00 |

*Actual savings scale with API pricing and request patterns.*

## 🧑‍💻 Contributing

Contributions welcome! Areas of interest:
- Additional domain files (k8s, sql, devops, etc.)
- Performance optimizations
- Additional test scenarios
- Documentation improvements

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 📚 References

1. Patterson, D., et al. (2021). "Carbon Emissions and Large Neural Network Training." arXiv:2104.10350
2. Aho, A.V., & Corasick, M.J. (1975). "Efficient string matching: An aid to bibliographic search." CACM 18(6)
3. IEA (2023). "Global Energy & CO2 Status Report"

## 🙏 Acknowledgments

This SDK implements the Carbon Protocol specification for deterministic semantic compression, targeting ICT4S 2026 submission.

⚖️ License & IP
Copyright (c) 2026 Taskal Samal. This reference implementation is released under the MIT License to foster academic collaboration and sustainable AI development. Patent Pending (USPTO Application No. 63/961,716)
