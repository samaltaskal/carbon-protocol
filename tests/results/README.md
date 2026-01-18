# Test Results

This directory contains all test results organized by date for tracking progress and comparing improvements over time.

## Directory Structure

```
results/
├── YYYY-MM-DD/              # Date-based folders
│   ├── VAL-*.txt            # Validation reports (human-readable)
│   ├── VAL-*.json           # Validation reports (machine-readable)
│   ├── TSR-*.txt            # IEEE 829 Test Summary Reports
│   └── TSR-*.json           # IEEE 829 JSON reports
├── industry_impact_assessment.json  # Latest industry-scale impact assessment
└── README.md                # This file
```

## Running Tests and Generating Results

```bash
# Run all tests with comprehensive results
python run_all_tests.py

# Run validation tests only
python -m pytest tests/test_validation.py -v -s -p tests.ieee_report --ieee-report --ieee-json

# Run industry impact tests only
python -m pytest tests/test_industry_impact.py -v -s
```

## Comparing Results Over Time

```bash
# Show progress summary
python compare_results.py

# Compare two dates
python compare_results.py 2026-01-17 2026-01-18
```

## Key Metrics Tracked

- Average token reduction percentage
- Carbon emissions saved (g CO2)
- Energy conserved (kWh)
- API cost savings (USD)
- Industry-scale projections

## Baseline (2026-01-17)

- Average Reduction: 25.8%
- Carbon Saved: 0.91 kg CO2 @ 1M requests/year
- Cost Saved: $64.00 @ 1M requests/year
