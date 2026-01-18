# carbon-protocol
Reference implementation of the Carbon Protocol for deterministic prompt compression (ICT4S 2026).

# Carbon Protocol: Zero-Overhead Prompt Compression
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Patent%20Pending-blue)
![Paper](https://img.shields.io/badge/ICT4S-Submitted-orange)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18284932-blue)](https://doi.org/10.5281/zenodo.18284932)

> **"Shift Left" for Green AI:** Moving semantic parsing from the Data Center GPU to the Client Edge CPU.

## 📄 Abstract
The Carbon Protocol is a deterministic, client-side specification for compressing natural language prompts before they enter the network. By mapping high-entropy linguistic tokens to low-entropy command syntax, it reduces the token volume of enterprise AI transactions by **40-60%**, directly lowering the energy cost of LLM inference.

**Paper Submission:** [Zero-Overhead Prompt Compression: A Deterministic Protocol for Energy-Efficient Generative AI] (Submitted to ICT4S 2026)

## 🚀 Key Features
- **Deterministic:** No neural networks involved. 100% predictable output.
- **Zero-Latency:** Runs in microseconds on any client (Browser, Mobile, CLI).
- **Privacy-Preserving:** Strips PII and linguistic noise *before* the data leaves the user's device.

## 📦 Installation
```bash
pip install carbon-protocol-dev  # (Coming soon)

⚡ Usage Example (Python)
This repository contains the reference compiler logic used to generate the results in the research paper.

from carbon.compiler import CarbonCompiler

# 1. Initialize the Compiler with the v1.0 Rule Set
cc = CarbonCompiler(ruleset="v1")

# 2. Input: A standard verbose prompt
raw_prompt = "Could you please write a python script to scrape data from a website?"

# 3. Compress
optimized = cc.compress(raw_prompt)

print(f"Original: {raw_prompt}")
print(f"Carbon Syntax: {optimized}")
# Output: CMD:CODE | @PY | ACT:SCRAPE

📊 Benchmark Results (from Paper)

Category,Reduction %
Coding (Python),57.1%
K8s Infrastructure,55.5%
SQL Queries,52.3%
Average,49.8%

⚖️ License & IP
Copyright (c) 2026 Taskal Samal. This reference implementation is released under the MIT License to foster academic collaboration and sustainable AI development. Patent Pending (USPTO Application No. 63/961,716)
