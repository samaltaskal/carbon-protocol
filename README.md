# Carbon Protocol SDK

High-performance deterministic semantic compression library for LLM prompts using Aho-Corasick multi-pattern matching.

![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Patent%20Pending-blue)
![Paper](https://img.shields.io/badge/ICT4S-Submitted-orange)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18284932-blue)](https://doi.org/10.5281/zenodo.18284932)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen)

> **"Shift Left" for Green AI:** Client-side semantic compression with <10ms latency and 25%+ token reduction.

**Paper Submission:** [Zero-Overhead Prompt Compression: A Deterministic Protocol for Energy-Efficient Generative AI](https://doi.org/10.5281/zenodo.18284932) (Submitted to ICT4S 2026)

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

### Via pip (Recommended)
```bash
pip install carbon-protocol
```

### From Source
```bash
# Clone repository
git clone https://github.com/samaltaskal/carbon-protocol.git
cd carbon-protocol

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## 🛠️ Tools & Mining

### Dictionary Builder
Expand the semantic coverage by generating new dictionaries from open sources (Github snippets, Keyword lists):
```bash
# Generate all dictionaries (K8s, Cloud, Python, SQL, Prompts)
python tools/build_dictionary.py --all

# Generate only K8s dictionary
python tools/build_dictionary.py --kubernetes
```

### Pattern Miner (New in v1.1.0)
Discover new compression rules from large-scale log datasets using the automated mining pipeline.
```bash
# Analyze logs to find repeating N-Grams and semantic clusters
python miner/discovery.py --input logs.txt --output new_rules.json
```
1. **Statistical**: Extracts frequent N-grams (CountVectorizer).
2. **Semantic**: Encodes phrases into vector embeddings (SentenceTransformers).
3. **Clustering**: Groups similar intents using DBSCAN.


## ⚡ Quick Start

```python
from carbon_protocol import Registry, Compiler

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

From SDK validation suite (VAL-20260117-230537):

| Metric | Value |
|--------|-------|
| **Average Token Reduction** | 25.8% |
| **Compression Ratio** | 0.74 |
| **Total Tests** | 61 tests, 100% pass rate |

### Industry-Scale Impact Projections

| Scale | Annual Requests | Carbon Saved | Cost Saved |
|-------|-----------------|--------------|------------|
| Small Org | 1M | 0.91 kg CO2 | $64 |
| Medium Org | 10M | 9.12 kg CO2 | $640 |
| Large Org | 100M | 91.2 kg CO2 | $6,400 |
| Enterprise | 1B | 912 kg CO2 | $64,000 |

### Global Adoption Scenarios (7T tokens/year baseline)

| Adoption | Carbon Saved | Cost Saved |
|----------|--------------|------------|
| 1% | 2.57 tonnes | $0.18M |
| 10% | 25.73 tonnes | $1.81M |
| 100% | 257.31 tonnes | $18.06M |

## 🧪 Testing & Validation

### Run All Tests
```bash
# Full test suite (61 tests)
pytest tests/ -v

# With IEEE 829 report generation
cd tests && python run_tests.py --ieee

# Run comprehensive test runner
cd tests && python run_all_tests.py
```

### SDK Validation Tests
```bash
# Run validation/impact assessment
cd tests && python run_tests.py --validation

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
│   ├── test_industry_impact.py # Industry impact tests (4 tests)
│   ├── run_all_tests.py    # Comprehensive test runner
│   ├── run_tests.py        # Flexible test runner
│   ├── compare_results.py  # Results comparison utility
│   └── results/            # Test results by date
│       └── YYYY-MM-DD/     # Date-based folders
├── docs/                   # Documentation
│   ├── api/                # API reference
│   ├── guides/             # User guides
│   └── architecture/       # System design
├── CHANGELOG.md            # Version history
├── VERSION                 # Current version
└── pyproject.toml          # Package configuration
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

## 📈 Dictionary Growth Tracker

Tracking the expansion of Carbon Protocol's semantic dictionaries over time.

| Date       | Version | New Domains Added | Patterns Added | Total Patterns | Source |
|------------|---------|-------------------|----------------|----------------|--------|
| 2026-01-18 | v1.1.0  | Prompts/Roleplay  | +4,106         | ~5,480         | `awesome-chatgpt-prompts` |
| 2026-01-18 | v1.1.0  | Tasks (Email, Data, debug, etc) | +213           | ~1,374         | Manual Heuristics |
| 2026-01-18 | v1.1.0  | Cloud (AWS, Azure, GCP) | +653           | ~1,161         | CLI Documentation |
| 2026-01-18 | v1.1.0  | Kubernetes        | +129           | ~508           | VS Code Snippets |
| 2026-01-17 | v1.0.0  | Python, SQL (Core)| +379           | 379            | Initial Release |

## 🌍 Environmental Impact

Based on validation testing with 15 representative prompts (VAL-20260117-230537):

**Per Sample Set (15 prompts)**:
- Energy Saved: 0.000029 kWh
- Carbon Saved: 0.0137 grams CO2
- Water Saved: 0.000052 liters

**Projected Annual (1M requests)**:
- Tokens Saved: 6.4 million
- Carbon Saved: 0.91 kg CO2
- Energy Saved: 1.92 kWh
- Water Saved: 3.46 liters
- Cost Saved: $64 USD

**Environmental Equivalents (1M requests/year)**:
- 114 smartphone charges
- 2 miles not driven
- 0.1 car-days off the road

*Calculations based on peer-reviewed research (Patterson et al., 2021) and IEA carbon intensity data.*

## 🏭 Industry-Scale Impact

Based on global LLM token consumption baseline of **7 trillion tokens/year**:

### Full Global Adoption Potential
| Metric | Value |
|--------|-------|
| Tokens Saved | 1.8 trillion |
| Carbon Saved | 257.31 tonnes CO2 |
| Energy Saved | 541.71 MWh |
| Cost Saved | $18.06 million |
| Water Saved | 0.98 megaliters |

## 💰 Cost Savings

Assuming $0.01 per 1K input tokens (conservative estimate):

| Scale | Request Volume | Tokens Saved | Annual Savings |
|-------|----------------|--------------|----------------|
| Small | 1M requests | 6.4M tokens | $64 |
| Medium | 10M requests | 64M tokens | $640 |
| Large | 100M requests | 640M tokens | $6,400 |
| Enterprise | 1B requests | 6.4B tokens | $64,000 |

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
