"""
Test suite for industry-scale impact assessment.

This module demonstrates the realistic environmental and economic impact
of the Carbon Protocol SDK at various scales, from small organizations
to industry-wide adoption.
"""

import json
from pathlib import Path

import pytest

from src.compiler import Compiler
from src.metrics import MetricsCalculator
from src.registry import Registry


# Real-world test prompts (same as validation)
INDUSTRY_TEST_PROMPTS = [
    "Please help me to write a python script to scrape data from the website.",
    "I need you to write a sql query to search for all users in the database and list all their records.",
    "Please help me to create a function that can read the file and parse the data from the api.",
    "Can you help me to develop a rest api using python fastapi framework?",
    "I would like you to write a python script to read the file from the database and save to file.",
    "Please help me to extract data from the csv file and convert it to json format for the api.",
    "I need you to write a python script to scrape data from the web and save to the database.",
    "Please help me to create a function that can parse json format data and write to file.",
    "Can you help me to deploy the application to kubernetes cluster using helm charts?",
    "I need you to write a shell script to automate the deployment process for the production environment.",
    "Please analyze all the data in the database and generate a comprehensive report with visualizations.",
    "Can you help me to review the code and identify potential bugs in the application?",
    "Please help me to write a python script that can scrape data from the web, parse the json format, and save to the database.",
    "I need you to create an automated testing framework for the application using pytest and generate detailed test reports.",
    "Can you help me to set up a ci/cd pipeline using github actions to automate testing, build, and deployment to production?",
]


@pytest.fixture
def compiler(registry):
    """Create a compiler instance."""
    return Compiler(registry)


@pytest.fixture
def metrics_calculator():
    """Create a metrics calculator instance."""
    return MetricsCalculator()


class TestIndustryScaleImpact:
    """Test industry-scale environmental and economic impact projections."""
    
    def test_comprehensive_industry_assessment(self, compiler, metrics_calculator):
        """
        Generate comprehensive industry-scale impact report.
        
        This test demonstrates the realistic environmental and economic
        impact at multiple scales, from small organizations to global adoption.
        """
        print("\n" + "=" * 80)
        print("CARBON PROTOCOL SDK - INDUSTRY-SCALE IMPACT ASSESSMENT")
        print("=" * 80)
        
        # Compress all test prompts
        metrics_list = []
        for prompt in INDUSTRY_TEST_PROMPTS:
            result = compiler.compress(prompt)
            metrics = metrics_calculator.calculate_compression_metrics(
                original_text=prompt,
                compressed_text=result.compressed,
                patterns_matched=result.matches_found,
            )
            metrics_list.append(metrics)
        
        # Calculate aggregate metrics
        aggregate = metrics_calculator.calculate_aggregate_metrics(
            metrics_list=metrics_list,
            annual_requests_estimate=1_000_000,  # Default scale
        )
        
        # Get industry impact assessment
        industry_impact = aggregate.get_industry_impact_assessment()
        
        # Print formatted report
        self._print_industry_report(industry_impact)
        
        # Save to JSON file
        self._save_industry_report(industry_impact)
        
        # Assertions
        assert aggregate.avg_token_reduction_percent > 0
        assert industry_impact["organizational_scale"]["small_org_1M_requests"]["carbon_kg"] > 0
        assert industry_impact["industry_wide_impact"]["full_adoption"]["carbon_tonnes"] != "0.00"
    
    def test_organizational_scale_comparison(self, compiler, metrics_calculator):
        """
        Compare impact across different organizational scales.
        
        Shows how the SDK benefits organizations of different sizes.
        """
        # Compress sample prompts
        metrics_list = []
        for prompt in INDUSTRY_TEST_PROMPTS[:5]:  # Use 5 representative prompts
            result = compiler.compress(prompt)
            metrics = metrics_calculator.calculate_compression_metrics(
                original_text=prompt,
                compressed_text=result.compressed,
                patterns_matched=result.matches_found,
            )
            metrics_list.append(metrics)
        
        aggregate = metrics_calculator.calculate_aggregate_metrics(
            metrics_list=metrics_list,
            annual_requests_estimate=1_000_000,
        )
        
        industry_impact = aggregate.get_industry_impact_assessment()
        
        print("\n" + "=" * 80)
        print("ORGANIZATIONAL SCALE COMPARISON")
        print("=" * 80)
        
        for scale_name, scale_data in industry_impact["organizational_scale"].items():
            scale_label = scale_name.replace("_", " ").title()
            print(f"\n{scale_label}:")
            print(f"  Annual Requests:  {scale_data['requests']}")
            print(f"  Tokens Saved:     {scale_data['tokens_saved']}")
            print(f"  Carbon Saved:     {scale_data['carbon_kg']} kg ({scale_data['carbon_tonnes']} tonnes)")
            print(f"  Energy Saved:     {scale_data['energy_kwh']} kWh")
            print(f"  Cost Saved:       {scale_data['cost_saved_usd']}")
            print(f"  Water Saved:      {scale_data['water_saved_liters']} liters")
    
    def test_global_adoption_scenarios(self, compiler, metrics_calculator):
        """
        Model global industry-wide adoption scenarios.
        
        Shows potential impact if the SDK achieves various levels of
        market penetration in the global LLM industry.
        """
        # Compress all test prompts
        metrics_list = []
        for prompt in INDUSTRY_TEST_PROMPTS:
            result = compiler.compress(prompt)
            metrics = metrics_calculator.calculate_compression_metrics(
                original_text=prompt,
                compressed_text=result.compressed,
                patterns_matched=result.matches_found,
            )
            metrics_list.append(metrics)
        
        aggregate = metrics_calculator.calculate_aggregate_metrics(
            metrics_list=metrics_list,
            annual_requests_estimate=1_000_000,
        )
        
        industry_impact = aggregate.get_industry_impact_assessment()
        
        print("\n" + "=" * 80)
        print("GLOBAL INDUSTRY ADOPTION SCENARIOS")
        print("=" * 80)
        print(f"\nGlobal LLM Token Consumption Baseline:")
        print(f"  Daily:                    {industry_impact['methodology']['global_baseline']['daily_tokens']}")
        print(f"  Annual (Conservative):    {industry_impact['methodology']['global_baseline']['annual_tokens_conservative']}")
        print(f"  Annual (Realistic):       {industry_impact['methodology']['global_baseline']['annual_tokens_realistic']}")
        print(f"\nSDK Average Compression:    {industry_impact['methodology']['compression_rate']}")
        
        for scenario_name, scenario_data in industry_impact["industry_wide_impact"].items():
            scenario_label = scenario_name.replace("_", " ").title()
            print(f"\n{scenario_label}:")
            print(f"  Adoption Rate:            {scenario_data['adoption_rate']}")
            print(f"  Affected Tokens:          {scenario_data['affected_tokens']}")
            print(f"  Tokens Saved:             {scenario_data['tokens_saved']}")
            print(f"  Carbon Saved:             {scenario_data['carbon_tonnes']} tonnes")
            print(f"                            ({scenario_data['carbon_megatonnes']} megatonnes)")
            print(f"  Energy Saved:             {scenario_data['energy_mwh']} MWh")
            print(f"  Cost Saved:               {scenario_data['cost_saved_millions_usd']}")
            print(f"  Water Saved:              {scenario_data['water_saved_megaliters']} megaliters")
    
    def test_environmental_equivalents(self, compiler, metrics_calculator):
        """
        Calculate relatable environmental equivalents.
        
        Translates carbon savings into everyday equivalents that
        stakeholders can easily understand.
        """
        # Compress all test prompts
        metrics_list = []
        for prompt in INDUSTRY_TEST_PROMPTS:
            result = compiler.compress(prompt)
            metrics = metrics_calculator.calculate_compression_metrics(
                original_text=prompt,
                compressed_text=result.compressed,
                patterns_matched=result.matches_found,
            )
            metrics_list.append(metrics)
        
        aggregate = metrics_calculator.calculate_aggregate_metrics(
            metrics_list=metrics_list,
            annual_requests_estimate=1_000_000,
        )
        
        industry_impact = aggregate.get_industry_impact_assessment()
        equivalents = industry_impact["environmental_equivalents"]
        
        print("\n" + "=" * 80)
        print("ENVIRONMENTAL IMPACT EQUIVALENTS (1M Requests/Year)")
        print("=" * 80)
        print(f"\nCarbon Saved: {equivalents['annual_carbon_kg']} kg CO2\n")
        print("This is equivalent to:")
        print(f"  [Cars]      Cars off road:         {equivalents['equivalent_to']['cars_off_road_days']} car-days")
        print(f"  [Trees]     Trees planted:         {equivalents['equivalent_to']['trees_planted_years']} tree-years")
        print(f"  [Phone]     Smartphone charges:    {equivalents['equivalent_to']['smartphone_charges']}")
        print(f"  [Driving]   Miles not driven:      {equivalents['equivalent_to']['miles_not_driven']}")
        print(f"  [Flight]    Flight hours avoided:  {equivalents['equivalent_to']['flight_hours_avoided']}")
        print("\nSources:")
        for source in equivalents["sources"]:
            print(f"  - {source}")
    
    def _print_industry_report(self, industry_impact: dict):
        """Print formatted industry impact report."""
        print("\n1. SAMPLE ANALYSIS")
        print("-" * 80)
        sample_metrics = industry_impact["sample_metrics"]
        print(f"Total Samples:              {sample_metrics['total_samples']}")
        print(f"Average Reduction:          {sample_metrics['average_reduction_percent']}%")
        print(f"Tokens Saved Per Sample:    {sample_metrics['tokens_per_sample']}")
        
        print("\n2. ORGANIZATIONAL SCALE IMPACT")
        print("-" * 80)
        for scale_name, scale_data in industry_impact["organizational_scale"].items():
            scale_label = scale_name.replace("_", " ").title()
            print(f"\n{scale_label}:")
            print(f"  Requests:         {scale_data['requests']}")
            print(f"  Carbon Saved:     {scale_data['carbon_kg']} kg ({scale_data['carbon_tonnes']} tonnes)")
            print(f"  Cost Saved:       {scale_data['cost_saved_usd']}")
        
        print("\n3. GLOBAL INDUSTRY ADOPTION SCENARIOS")
        print("-" * 80)
        print(f"Baseline: {industry_impact['methodology']['global_baseline']['annual_tokens_realistic']} tokens/year")
        for scenario_name, scenario_data in industry_impact["industry_wide_impact"].items():
            scenario_label = scenario_name.replace("_", " ").title()
            print(f"\n{scenario_label} ({scenario_data['adoption_rate']}):")
            print(f"  Carbon:  {scenario_data['carbon_tonnes']} tonnes")
            print(f"  Cost:    {scenario_data['cost_saved_millions_usd']}")
        
        print("\n4. ENVIRONMENTAL EQUIVALENTS")
        print("-" * 80)
        equivalents = industry_impact["environmental_equivalents"]
        print(f"Annual Carbon Saved (1M requests): {equivalents['annual_carbon_kg']} kg CO2")
        print(f"  = {equivalents['equivalent_to']['cars_off_road_days']} car-days off road")
        print(f"  = {equivalents['equivalent_to']['trees_planted_years']} tree-years of absorption")
        
        print("\n5. METHODOLOGY")
        print("-" * 80)
        methodology = industry_impact["methodology"]
        print(f"Compression Rate:       {methodology['compression_rate']}")
        print(f"Energy per 1K tokens:   {methodology['energy_per_1k_tokens']}")
        print(f"Carbon Intensity:       {methodology['carbon_intensity']}")
        print(f"API Cost:               {methodology['api_cost']}")
        
        print("\n" + "=" * 80)
    
    def _save_industry_report(self, industry_impact: dict):
        """Save industry impact report to JSON file."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / "industry_impact_assessment.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(industry_impact, f, indent=2, ensure_ascii=False)
        
        print(f"\nIndustry impact report saved to: {output_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
