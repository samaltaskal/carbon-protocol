# Changelog

All notable changes to the Carbon Protocol SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-17

### Added

#### Core SDK
- **CarbonRegistry**: Pattern registry with YAML-based domain configuration
  - Lazy automaton building for performance optimization
  - Multi-domain support (core, development, data, operations)
  - Statistics and pattern management APIs
- **CarbonCompiler**: Aho-Corasick based prompt compression engine
  - O(n+z) time complexity for multi-pattern matching
  - Longest-match-first strategy for optimal compression
  - Non-overlapping, left-to-right greedy matching
  - Batch compression support
  - Case-insensitive matching

#### Metrics & Impact Assessment
- **PromptMetrics**: Per-prompt environmental impact calculations
- **AggregateMetrics**: Batch and industry-scale impact analysis
- Environmental constants based on peer-reviewed research:
  - Energy: 0.0003 kWh per 1K tokens
  - Carbon intensity: 0.475 kg CO2/kWh (global grid average)
  - Water usage: 1.8 liters/kWh
- Global token consumption baselines:
  - Daily: 6.2 billion tokens (conservative)
  - Annual: 7 trillion tokens (realistic estimate)
- Industry-scale impact projections:
  - Organizational scales (1M to 1B requests)
  - Global adoption scenarios (1% to 100%)
- Environmental equivalents (cars, trees, phones, miles, flights)

#### Testing Framework
- Comprehensive test suite with 61 tests:
  - Unit tests for Registry and Compiler
  - Integration tests for full pipeline
  - Validation tests with real-world prompts
  - Industry impact assessment tests
- IEEE 829 compliant test reporting
- Results comparison utility for tracking progress
- Date-based results storage under `tests/results/`

#### Documentation
- README with installation and usage instructions
- Inline documentation with docstrings
- IEEE format validation reports

### Technical Details
- Python 3.13+ support
- Dependencies: pyahocorasick, PyYAML, pytest
- Average token reduction: ~25.8% on test prompts
- Projected savings at full adoption: 257 tonnes CO2, $18M annually

---

## Version History

| Version | Date       | Description                    |
|---------|------------|--------------------------------|
| 1.0.0   | 2026-01-17 | Initial release                |

---

## Future Roadmap

### Planned for v1.1.0
- [ ] Additional domain patterns (cloud, security, ML/AI)
- [ ] Custom pattern definition API
- [ ] Real-time metrics dashboard
- [ ] Plugin architecture for custom compressors

### Planned for v1.2.0
- [ ] Multi-language support
- [ ] Integration with popular LLM APIs
- [ ] Caching layer for repeated prompts
- [ ] Configuration file support

---

## Contributing

When contributing, please update this changelog under the "Unreleased" section.
Follow the format:
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes
