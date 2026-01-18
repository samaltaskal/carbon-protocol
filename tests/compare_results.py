"""
Results comparison utility for Carbon Protocol SDK.

This script compares test results across different dates to track progress
and improvements in the SDK's compression and environmental impact metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_latest_validation(date_str: str) -> dict[str, Any] | None:
    """Load the latest validation result for a specific date."""
    results_dir = Path(__file__).parent / "results"
    date_dir = results_dir / date_str
    
    if not date_dir.exists():
        return None
    
    val_files = sorted(date_dir.glob("VAL-*.json"))
    if not val_files:
        return None
    
    with open(val_files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def list_available_dates() -> list[str]:
    """List all dates with available results."""
    results_dir = Path(__file__).parent / "results"
    dates = []
    
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.count("-") == 2:  # YYYY-MM-DD format
            dates.append(item.name)
    
    return sorted(dates)


def compare_dates(date1: str, date2: str):
    """Compare results between two dates."""
    data1 = load_latest_validation(date1)
    data2 = load_latest_validation(date2)
    
    if not data1:
        print(f"[X] No results found for {date1}")
        return
    
    if not data2:
        print(f"[X] No results found for {date2}")
        return
    
    # Handle nested structure
    agg1 = data1.get("aggregate_metrics", data1)
    agg2 = data2.get("aggregate_metrics", data2)
    summary1 = agg1.get("summary", agg1)
    summary2 = agg2.get("summary", agg2)
    
    print("\n" + "=" * 80)
    print("CARBON PROTOCOL SDK - RESULTS COMPARISON")
    print("=" * 80)
    print(f"Date 1: {date1}")
    print(f"Date 2: {date2}")
    print("=" * 80)
    
    # Compare summary metrics
    print("\n[Summary] METRICS")
    print("-" * 80)
    
    metrics = [
        ("Total Samples", "sample_count", ""),
        ("Total Original Tokens", "total_original_tokens", "tokens"),
        ("Total Compressed Tokens", "total_compressed_tokens", "tokens"),
        ("Total Tokens Saved", "total_token_reduction", "tokens"),
        ("Average Reduction", "avg_token_reduction_percent", "%"),
        ("Average Compression Ratio", "avg_compression_ratio", ""),
    ]
    
    for label, key, unit in metrics:
        val1 = summary1.get(key, 0)
        val2 = summary2.get(key, 0)
        diff = val2 - val1
        pct_change = (diff / val1 * 100) if val1 != 0 else 0
        
        if unit == "%":
            print(f"{label:30} {val1:10.2f}{unit}  ->  {val2:10.2f}{unit}  ({diff:+.2f}{unit})")
        elif unit == "tokens":
            print(f"{label:30} {val1:10,.0f}  ->  {val2:10,.0f}  ({diff:+,.0f}, {pct_change:+.1f}%)")
        else:
            print(f"{label:30} {val1:10.4f}  ->  {val2:10.4f}  ({diff:+.4f})")
    
    # Compare environmental impact
    print("\n[Environment] IMPACT (Per Sample Set)")
    print("-" * 80)
    
    env1 = agg1.get("environmental_impact", {})
    env2 = agg2.get("environmental_impact", {})
    
    env_metrics = [
        ("Energy Saved", "total_energy_saved_kwh", "kWh", 6),
        ("Carbon Saved", "total_carbon_saved_grams", "g CO2", 4),
        ("Water Saved", "total_water_saved_liters", "liters", 6),
    ]
    
    for label, key, unit, decimals in env_metrics:
        val1 = env1.get(key, 0)
        val2 = env2.get(key, 0)
        diff = val2 - val1
        pct_change = (diff / val1 * 100) if val1 != 0 else 0
        
        print(f"{label:30} {val1:10.{decimals}f} {unit}  ->  {val2:10.{decimals}f} {unit}  ({diff:+.{decimals}f}, {pct_change:+.1f}%)")
    
    # Compare projections
    print("\n[Projections] ANNUAL (1M Requests)")
    print("-" * 80)
    
    proj1 = agg1.get("projections", {})
    proj2 = agg2.get("projections", {})
    
    proj_metrics = [
        ("Tokens Saved", "annual_tokens_saved", "", 0),
        ("Carbon Saved", "annual_carbon_saved_kg", "kg CO2", 2),
        ("Cost Saved", "annual_cost_saved_usd", "USD", 2),
    ]
    
    for label, key, unit, decimals in proj_metrics:
        val1 = proj1.get(key, 0)
        val2 = proj2.get(key, 0)
        diff = val2 - val1
        pct_change = (diff / val1 * 100) if val1 != 0 else 0
        
        if decimals == 0:
            print(f"{label:30} {val1:10,.0f} {unit}  ->  {val2:10,.0f} {unit}  ({diff:+,.0f}, {pct_change:+.1f}%)")
        else:
            print(f"{label:30} {val1:10.{decimals}f} {unit}  ->  {val2:10.{decimals}f} {unit}  ({diff:+.{decimals}f}, {pct_change:+.1f}%)")
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    reduction_change = summary2.get("avg_token_reduction_percent", 0) - summary1.get("avg_token_reduction_percent", 0)
    carbon_change = proj2.get("annual_carbon_saved_kg", 0) - proj1.get("annual_carbon_saved_kg", 0)
    
    if reduction_change > 0:
        print(f"[+] Compression improved by {reduction_change:+.2f}%")
    elif reduction_change < 0:
        print(f"[-] Compression decreased by {reduction_change:.2f}%")
    else:
        print(f"[=] Compression remained stable")
    
    if carbon_change > 0:
        print(f"[+] Carbon savings increased by {carbon_change:+.2f} kg CO2/year @ 1M requests")
    elif carbon_change < 0:
        print(f"[-] Carbon savings decreased by {carbon_change:.2f} kg CO2/year @ 1M requests")
    else:
        print(f"[=] Carbon savings remained stable")
    
    print("=" * 80 + "\n")


def show_progress_summary():
    """Show progress summary across all available dates."""
    dates = list_available_dates()
    
    if not dates:
        print("[X] No results found")
        return
    
    print("\n" + "=" * 80)
    print("CARBON PROTOCOL SDK - PROGRESS SUMMARY")
    print("=" * 80)
    print(f"Results available for {len(dates)} date(s)\n")
    
    for date in dates:
        data = load_latest_validation(date)
        if not data:
            continue
        
        # Handle nested structure under aggregate_metrics
        agg = data.get("aggregate_metrics", data)
        summary = agg.get("summary", agg)
        env = agg.get("environmental_impact", {})
        proj = agg.get("projections", {})
        
        print(f"[Date] {date}")
        print(f"   Average Reduction:    {summary.get('avg_token_reduction_percent', summary.get('average_reduction_percent', 0)):.2f}%")
        print(f"   Tokens Saved:         {summary.get('total_token_reduction', summary.get('total_tokens_saved', 0)):,}")
        print(f"   Carbon Saved:         {env.get('total_carbon_saved_grams', env.get('carbon_grams', 0)):.4f}g CO2")
        print(f"   Annual Projection:    {proj.get('annual_carbon_saved_kg', proj.get('annual_carbon_kg', 0)):.2f} kg CO2 @ 1M requests")
        print()
    
    # Compare first and latest
    if len(dates) >= 2:
        print("\n" + "=" * 80)
        print("FIRST vs LATEST COMPARISON")
        print("=" * 80)
        compare_dates(dates[0], dates[-1])


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - show progress summary
        show_progress_summary()
    elif len(sys.argv) == 3:
        # Two dates provided - compare them
        date1, date2 = sys.argv[1], sys.argv[2]
        compare_dates(date1, date2)
    else:
        print("Usage:")
        print("  python compare_results.py                    # Show progress summary")
        print("  python compare_results.py DATE1 DATE2        # Compare two dates")
        print("\nAvailable dates:")
        for date in list_available_dates():
            print(f"  - {date}")


if __name__ == "__main__":
    main()
