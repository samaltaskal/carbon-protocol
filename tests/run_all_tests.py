"""
Comprehensive test runner for Carbon Protocol SDK.

This script runs all tests and generates timestamped results for comparison.
Results are saved to tests/results/ with timestamps for tracking progress.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_validation_tests():
    """Run validation tests with IEEE report generation."""
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION TESTS")
    print("=" * 80 + "\n")
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test_validation.py",
        "-v",
        "-s",
        "-p",
        "ieee_report",
        "--ieee-report",
        "--ieee-json",
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_industry_impact_tests():
    """Run industry-scale impact assessment tests."""
    print("\n" + "=" * 80)
    print("RUNNING INDUSTRY IMPACT TESTS")
    print("=" * 80 + "\n")
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test_industry_impact.py",
        "-v",
        "-s",
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_all_core_tests():
    """Run all core unit and integration tests."""
    print("\n" + "=" * 80)
    print("RUNNING CORE TESTS (UNIT + INTEGRATION)")
    print("=" * 80 + "\n")
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test_registry.py",
        "test_compiler.py",
        "test_integration.py",
        "-v",
        "--tb=short",
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def generate_comparison_report():
    """Generate comparison report from latest results."""
    results_dir = Path(__file__).parent / "results"
    
    # Find today's folder
    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = results_dir / today
    
    if not today_dir.exists():
        print(f"\n⚠️  No results found for today ({today})")
        return
    
    # Find latest validation JSON
    val_files = sorted(today_dir.glob("VAL-*.json"))
    industry_file = results_dir / "industry_impact_assessment.json"
    
    if not val_files:
        print("\n⚠️  No validation results found")
        return
    
    latest_val = val_files[-1]
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Load and display validation results
    with open(latest_val, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Handle nested structure
    agg = val_data.get("aggregate_metrics", val_data)
    summary = agg.get("summary", agg)
    env_impact = agg.get("environmental_impact", {})
    projections = agg.get("projections", {})
    
    print(f"\n[Summary] VALIDATION RESULTS ({latest_val.stem})")
    print("-" * 80)
    print(f"Samples Tested:        {summary.get('sample_count', 'N/A')}")
    print(f"Average Reduction:     {summary.get('avg_token_reduction_percent', 0):.2f}%")
    print(f"Tokens Saved:          {summary.get('total_token_reduction', 0)}")
    print(f"Carbon Saved:          {env_impact.get('total_carbon_saved_grams', 0):.4f}g CO2")
    print(f"Annual Projection:     {projections.get('annual_carbon_saved_kg', 0):.2f} kg CO2 @ 1M requests")
    
    # Load and display industry impact
    if industry_file.exists():
        with open(industry_file, "r", encoding="utf-8") as f:
            industry_data = json.load(f)
        
        print(f"\n[Industry] IMPACT ASSESSMENT")
        print("-" * 80)
        
        # Organizational scale
        print("\nOrganizational Scale (Annual):")
        org_scale = industry_data["organizational_scale"]
        print(f"  Small (1M):      {org_scale['small_org_1M_requests']['carbon_kg']} kg CO2, {org_scale['small_org_1M_requests']['cost_saved_usd']}")
        print(f"  Medium (10M):    {org_scale['medium_org_10M_requests']['carbon_kg']} kg CO2, {org_scale['medium_org_10M_requests']['cost_saved_usd']}")
        print(f"  Large (100M):    {org_scale['large_org_100M_requests']['carbon_kg']} kg CO2, {org_scale['large_org_100M_requests']['cost_saved_usd']}")
        print(f"  Enterprise (1B): {org_scale['enterprise_1B_requests']['carbon_kg']} kg CO2, {org_scale['enterprise_1B_requests']['cost_saved_usd']}")
        
        # Global adoption
        print("\nGlobal Industry Adoption:")
        global_impact = industry_data["industry_wide_impact"]
        print(f"  1% Adoption:     {global_impact['one_percent_adoption']['carbon_tonnes']} tonnes, {global_impact['one_percent_adoption']['cost_saved_millions_usd']}")
        print(f"  5% Adoption:     {global_impact['five_percent_adoption']['carbon_tonnes']} tonnes, {global_impact['five_percent_adoption']['cost_saved_millions_usd']}")
        print(f"  10% Adoption:    {global_impact['ten_percent_adoption']['carbon_tonnes']} tonnes, {global_impact['ten_percent_adoption']['cost_saved_millions_usd']}")
        print(f"  Full Adoption:   {global_impact['full_adoption']['carbon_tonnes']} tonnes, {global_impact['full_adoption']['cost_saved_millions_usd']}")
    
    print("\n" + "=" * 80)
    print("[Files] RESULTS LOCATION")
    print("=" * 80)
    print(f"Today's Results:     {today_dir}")
    print(f"Industry Impact:     {industry_file}")
    print(f"IEEE Reports:        {today_dir / 'TSR-*.txt'}")
    print(f"Validation Reports:  {today_dir / 'VAL-*.txt'}")
    print("\n")


def compare_with_previous():
    """Compare current results with previous runs."""
    results_dir = Path(__file__).parent / "results"
    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = results_dir / today
    
    if not today_dir.exists():
        return
    
    # Get all validation files from today
    today_vals = sorted(today_dir.glob("VAL-*.json"))
    
    if len(today_vals) < 2:
        print("\n[Tip] Run tests multiple times to see progress comparison")
        return
    
    print("\n" + "=" * 80)
    print("PROGRESS COMPARISON (Today's Runs)")
    print("=" * 80)
    
    first = today_vals[0]
    latest = today_vals[-1]
    
    with open(first, "r", encoding="utf-8") as f:
        first_data = json.load(f)
    
    with open(latest, "r", encoding="utf-8") as f:
        latest_data = json.load(f)
    
    print(f"\nFirst Run:  {first.stem}")
    print(f"Latest Run: {latest.stem}")
    print("-" * 80)
    
    # Handle nested structure
    first_agg = first_data.get("aggregate_metrics", first_data)
    latest_agg = latest_data.get("aggregate_metrics", latest_data)
    first_summary = first_agg.get("summary", first_agg)
    latest_summary = latest_agg.get("summary", latest_agg)
    first_env = first_agg.get("environmental_impact", {})
    latest_env = latest_agg.get("environmental_impact", {})
    
    # Compare key metrics
    first_reduction = first_summary.get('avg_token_reduction_percent', 0)
    latest_reduction = latest_summary.get('avg_token_reduction_percent', 0)
    
    first_carbon = first_env.get('total_carbon_saved_grams', 0)
    latest_carbon = latest_env.get('total_carbon_saved_grams', 0)
    
    print(f"\nAverage Token Reduction:")
    print(f"  First:   {first_reduction:.2f}%")
    print(f"  Latest:  {latest_reduction:.2f}%")
    print(f"  Change:  {latest_reduction - first_reduction:+.2f}%")
    
    print(f"\nCarbon Saved per Sample Set:")
    print(f"  First:   {first_carbon:.4f}g CO2")
    print(f"  Latest:  {latest_carbon:.4f}g CO2")
    print(f"  Change:  {latest_carbon - first_carbon:+.4f}g CO2")
    
    print("\n")


def main():
    """Run all tests and generate comprehensive results."""
    # Change to tests directory for relative paths
    import os
    os.chdir(Path(__file__).parent)
    
    print("\n" + "=" * 80)
    print("CARBON PROTOCOL SDK - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = {
        "core_tests": False,
        "validation_tests": False,
        "industry_tests": False,
    }
    
    # Run all test suites
    try:
        results["core_tests"] = run_all_core_tests()
        results["validation_tests"] = run_validation_tests()
        results["industry_tests"] = run_industry_impact_tests()
    except KeyboardInterrupt:
        print("\n\n[!] Test execution interrupted by user")
        sys.exit(1)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    for test_type, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:8} {test_type.replace('_', ' ').title()}")
    
    print("-" * 80)
    overall = "[PASS] ALL TESTS PASSED" if all_passed else "[FAIL] SOME TESTS FAILED"
    print(f"\n{overall}")
    print("=" * 80)
    
    # Generate comparison reports
    generate_comparison_report()
    compare_with_previous()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
