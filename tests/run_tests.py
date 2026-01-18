#!/usr/bin/env python3
"""
Test Runner for Carbon Protocol SDK.

This script provides convenient commands for running tests with
IEEE 829 compliant reporting.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run unit tests only
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --validation       # Run validation tests only
    python run_tests.py --ieee             # Generate IEEE report
    python run_tests.py --coverage         # Run with coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Carbon Protocol SDK Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--unit", "-u",
        action="store_true",
        help="Run only unit tests",
    )
    parser.add_argument(
        "--integration", "-i",
        action="store_true", 
        help="Run only integration tests",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run only end-to-end tests",
    )
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Run only performance tests",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Run only validation/impact assessment tests",
    )
    parser.add_argument(
        "--ieee",
        action="store_true",
        help="Generate IEEE 829 compliant test report",
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests",
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add marker filters
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.e2e:
        markers.append("e2e")
    if args.performance:
        markers.append("performance")
    if args.validation:
        markers.append("validation")
    
    if markers:
        cmd.extend(["-m", " or ".join(markers)])
    
    # Skip slow tests if requested
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Add IEEE report plugin
    if args.ieee:
        cmd.extend([
            "-p", "ieee_report",
            "--ieee-report",
            "--ieee-json",
        ])
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
        ])
    
    # Verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
