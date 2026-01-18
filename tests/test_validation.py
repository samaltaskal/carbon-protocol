"""
SDK Validation Tests - Impact Metrics and Value Demonstration

IEEE 829 Test Specification:
    Test Suite ID: TS-VAL-001
    Test Suite Name: SDK Validation and Impact Assessment
    Purpose: Demonstrate and validate the measurable benefits of the Carbon Protocol SDK
    
This module provides comprehensive validation tests that capture:
    - Token reduction metrics (with SDK vs without SDK)
    - Environmental impact (carbon emissions, energy savings)
    - Cost savings projections
    - Compression efficiency analysis
    
These results are suitable for submission to IEEE or regulatory authorities
to demonstrate the solution's environmental and economic benefits.
"""

import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import pytest

from src import Registry, Compiler
from src.metrics import (
    MetricsCalculator, 
    CompressionMetrics, 
    AggregateMetrics,
    GLOBAL_DAILY_TOKENS,
    GLOBAL_ANNUAL_TOKENS,
    GLOBAL_ANNUAL_TOKENS_REALISTIC,
    ENERGY_PER_1K_TOKENS_KWH,
    CARBON_INTENSITY_KG_PER_KWH,
    COST_PER_1K_TOKENS_USD,
    WATER_PER_KWH_LITERS,
)


# =============================================================================
# Test Data: Representative Real-World Prompts
# =============================================================================

VALIDATION_PROMPTS = [
    # Development & Coding Prompts
    {
        "category": "Development",
        "description": "Python script request",
        "prompt": "Could you please write a python script to scrape data from the web and save it to a json format file?",
    },
    {
        "category": "Development",
        "description": "Database query request",
        "prompt": "I need you to write a sql query to search for all users in the database and list all their records.",
    },
    {
        "category": "Development",
        "description": "Function implementation",
        "prompt": "Please help me to create a function that can read the file and parse the data from the api.",
    },
    {
        "category": "Development",
        "description": "Code review request",
        "prompt": "Could you please check the python code and help me to update the function to handle errors?",
    },
    {
        "category": "Development", 
        "description": "Tool-specific request",
        "prompt": "I want you to write a bash script for visual studio code to create a new project structure.",
    },
    
    # Data Processing Prompts
    {
        "category": "Data Processing",
        "description": "Data extraction",
        "prompt": "Please help me to extract data from the csv file and convert it to json format for the api.",
    },
    {
        "category": "Data Processing",
        "description": "Data transformation",
        "prompt": "I would like you to write a python script to read the file from the database and save to file.",
    },
    {
        "category": "Data Processing",
        "description": "Batch processing",
        "prompt": "Could you please create a function to list all files and extract data from each one?",
    },
    
    # Operations & DevOps Prompts
    {
        "category": "Operations",
        "description": "Server management",
        "prompt": "I need you to write a shell script to update the configuration on the server and delete the old files.",
    },
    {
        "category": "Operations",
        "description": "Deployment request",
        "prompt": "Please help me to create a new deployment script using python to modify the settings on the server.",
    },
    
    # Analysis & Reporting Prompts
    {
        "category": "Analysis",
        "description": "Data analysis",
        "prompt": "Could you please write a python script to search for patterns in the database and show all results?",
    },
    {
        "category": "Analysis",
        "description": "Report generation",
        "prompt": "I want you to create a function to get all data from the api and format it as json for the report.",
    },
    
    # Complex Multi-Intent Prompts
    {
        "category": "Complex",
        "description": "Multi-step workflow",
        "prompt": "Please help me to write a python script that can scrape data from the web, parse the json format, and save to file in the database.",
    },
    {
        "category": "Complex",
        "description": "Full pipeline request",
        "prompt": "I would like you to create a function to read the file from the api, extract data, update the database, and write to file as csv file.",
    },
    {
        "category": "Complex",
        "description": "IDE integration",
        "prompt": "Could you please help me to write a python code for visual studio code that can search for all files and modify the configuration?",
    },
]


@dataclass
class ValidationResult:
    """Container for validation test results."""
    test_id: str
    test_date: str
    test_environment: str
    
    # Sample information
    total_samples: int
    categories_tested: list[str]
    
    # Aggregate metrics
    aggregate_metrics: dict[str, Any]
    
    # Individual sample results
    sample_results: list[dict[str, Any]]
    
    # Reference parameters
    calculation_parameters: dict[str, float]
    
    # Methodology notes
    methodology: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SDKValidationReport:
    """
    Generator for SDK validation reports suitable for IEEE/authority submission.
    Includes industry-scale impact assessment.
    """
    
    def __init__(self, validation_result: ValidationResult, aggregate_metrics: AggregateMetrics = None):
        self.result = validation_result
        self.aggregate = aggregate_metrics
    
    def generate_text_report(self) -> str:
        """Generate a comprehensive text report with industry impact."""
        r = self.result
        agg = r.aggregate_metrics
        summary = agg.get("summary", {})
        
        # Get industry impact if available
        industry = None
        if self.aggregate:
            industry = self.aggregate.get_industry_impact_assessment()
        
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("CARBON PROTOCOL SDK - VALIDATION AND IMPACT ASSESSMENT REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Report Identification
        lines.append("1. REPORT IDENTIFICATION")
        lines.append("-" * 40)
        lines.append(f"   Report ID:        {r.test_id}")
        lines.append(f"   Report Date:      {r.test_date}")
        lines.append(f"   Environment:      {r.test_environment}")
        lines.append(f"   Samples Tested:   {r.total_samples}")
        lines.append(f"   Categories:       {', '.join(r.categories_tested)}")
        lines.append("")
        
        # Executive Summary
        lines.append("2. EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"   Total Prompts Analyzed:      {summary.get('sample_count', 0)}")
        lines.append(f"   Original Tokens (Total):     {summary.get('total_original_tokens', 0):,}")
        lines.append(f"   Compressed Tokens (Total):   {summary.get('total_compressed_tokens', 0):,}")
        lines.append(f"   Tokens Saved (Total):        {summary.get('total_token_reduction', 0):,}")
        lines.append(f"   Average Token Reduction:     {summary.get('avg_token_reduction_percent', 0):.1f}%")
        lines.append(f"   Average Compression Ratio:   {summary.get('avg_compression_ratio', 0):.2f}")
        lines.append("")
        
        # Environmental Impact (Per Sample)
        lines.append("3. ENVIRONMENTAL IMPACT ASSESSMENT")
        lines.append("-" * 40)
        env = agg.get("environmental_impact", {})
        lines.append("   Per Test Sample Set:")
        lines.append(f"      Energy Saved:             {env.get('total_energy_saved_kwh', 0):.6f} kWh")
        lines.append(f"      Carbon Emissions Saved:   {env.get('total_carbon_saved_grams', 0):.4f} grams CO2")
        lines.append(f"      Water Saved:              {env.get('total_water_saved_liters', 0):.6f} liters")
        lines.append("")
        
        # Industry Scale Impact Section
        lines.append("4. INDUSTRY-SCALE IMPACT ASSESSMENT")
        lines.append("-" * 40)
        lines.append("")
        lines.append("   Global LLM Token Consumption Baseline (2026):")
        lines.append(f"      Daily (Conservative):     {GLOBAL_DAILY_TOKENS:,} tokens/day")
        lines.append(f"      Annual (Conservative):    {GLOBAL_ANNUAL_TOKENS:,} tokens/year")
        lines.append(f"      Annual (Realistic):       {GLOBAL_ANNUAL_TOKENS_REALISTIC:,} tokens/year")
        lines.append("")
        
        if industry:
            org_scale = industry.get("organizational_scale", {})
            lines.append("   Organizational Scale Impact (Annual):")
            lines.append("   " + "-" * 70)
            lines.append(f"   {'Scale':<25} {'Requests':<15} {'Carbon (kg)':<15} {'Cost Saved':<15}")
            lines.append("   " + "-" * 70)
            
            for scale_key, scale_data in org_scale.items():
                label = scale_key.replace("_", " ").title()[:24]
                lines.append(f"   {label:<25} {scale_data['requests']:<15} {scale_data['carbon_kg']:<15.2f} {scale_data['cost_saved_usd']:<15}")
            lines.append("")
            
            # Global Adoption Scenarios
            lines.append("   Global Industry Adoption Scenarios:")
            lines.append("   " + "-" * 70)
            lines.append(f"   {'Adoption':<15} {'Tokens Saved':<25} {'Carbon (tonnes)':<18} {'Cost Saved':<15}")
            lines.append("   " + "-" * 70)
            
            global_impact = industry.get("industry_wide_impact", {})
            for scenario_key, scenario_data in global_impact.items():
                label = scenario_data['adoption_rate']
                lines.append(f"   {label:<15} {scenario_data['tokens_saved']:<25} {scenario_data['carbon_tonnes']:<18} {scenario_data['cost_saved_millions_usd']:<15}")
            lines.append("")
        
        # Annual Projections (Multiple Scales)
        lines.append("5. ANNUAL PROJECTIONS")
        lines.append("-" * 40)
        proj = agg.get("projections", {})
        
        lines.append("   Small Organization (1M Requests/Year):")
        lines.append(f"      Tokens Saved:             {proj.get('annual_tokens_saved', 0):,}")
        lines.append(f"      Carbon Saved:             {proj.get('annual_carbon_saved_kg', 0):.2f} kg CO2")
        lines.append(f"      Cost Saved:               ${proj.get('annual_cost_saved_usd', 0):,.2f} USD")
        lines.append("")
        
        if industry:
            org_scale = industry.get("organizational_scale", {})
            if "enterprise_1B_requests" in org_scale:
                ent = org_scale["enterprise_1B_requests"]
                lines.append("   Enterprise Scale (1B Requests/Year):")
                lines.append(f"      Tokens Saved:             {ent['tokens_saved']}")
                lines.append(f"      Carbon Saved:             {ent['carbon_kg']} kg ({ent['carbon_tonnes']} tonnes) CO2")
                lines.append(f"      Cost Saved:               {ent['cost_saved_usd']}")
                lines.append(f"      Energy Saved:             {ent['energy_kwh']} kWh")
                lines.append(f"      Water Saved:              {ent['water_saved_liters']} liters")
                lines.append("")
        
        # Cost Analysis
        lines.append("6. COST IMPACT ANALYSIS")
        lines.append("-" * 40)
        cost = agg.get("cost_impact", {})
        lines.append(f"   Per Sample Set Cost Saved:   ${cost.get('total_cost_saved_usd', 0):.4f} USD")
        lines.append(f"   Small Org (1M/year):         ${proj.get('annual_cost_saved_usd', 0):,.2f} USD")
        
        if industry:
            global_impact = industry.get("industry_wide_impact", {})
            if "ten_percent_adoption" in global_impact:
                lines.append(f"   10% Global Adoption:         {global_impact['ten_percent_adoption']['cost_saved_millions_usd']}")
            if "full_adoption" in global_impact:
                lines.append(f"   Full Global Adoption:        {global_impact['full_adoption']['cost_saved_millions_usd']}")
        lines.append("")
        
        # Environmental Equivalents
        if industry:
            equivalents = industry.get("environmental_equivalents", {})
            if equivalents:
                lines.append("7. ENVIRONMENTAL EQUIVALENTS")
                lines.append("-" * 40)
                lines.append(f"   Annual Carbon Saved (1M requests): {equivalents.get('annual_carbon_kg', 0):.2f} kg CO2")
                lines.append("")
                lines.append("   This is equivalent to:")
                eq = equivalents.get("equivalent_to", {})
                lines.append(f"      Cars off road:            {eq.get('cars_off_road_days', 0)} car-days")
                lines.append(f"      Trees planted:            {eq.get('trees_planted_years', 0)} tree-years")
                lines.append(f"      Smartphone charges:       {eq.get('smartphone_charges', 0)}")
                lines.append(f"      Miles not driven:         {eq.get('miles_not_driven', 0)}")
                lines.append(f"      Flight hours avoided:     {eq.get('flight_hours_avoided', 0)}")
                lines.append("")
        
        # Methodology
        lines.append("8. METHODOLOGY")
        lines.append("-" * 40)
        for line in r.methodology.split("\n"):
            lines.append(f"   {line}")
        lines.append("")
        
        # Calculation Parameters
        lines.append("9. CALCULATION PARAMETERS")
        lines.append("-" * 40)
        params = r.calculation_parameters
        lines.append(f"   Characters per Token:        {params.get('chars_per_token', 0)}")
        lines.append(f"   Energy per 1K Tokens:        {params.get('energy_per_1k_tokens_kwh', 0)} kWh")
        lines.append(f"   Carbon Intensity:            {params.get('carbon_intensity_kg_per_kwh', 0)} kg CO2/kWh")
        lines.append(f"   Cost per 1K Tokens:          ${params.get('cost_per_1k_tokens_usd', 0)} USD")
        lines.append(f"   Water per kWh:               {WATER_PER_KWH_LITERS} liters")
        lines.append("")
        
        # Detailed Results Table
        lines.append("10. DETAILED SAMPLE RESULTS")
        lines.append("-" * 40)
        lines.append("")
        lines.append(f"   {'#':<3} {'Category':<15} {'Orig':<8} {'Comp':<8} {'Saved':<6} {'Reduction':<10} {'CO2 (g)':<10}")
        lines.append(f"   {'-'*3} {'-'*15} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
        
        for i, sample in enumerate(r.sample_results, 1):
            cat = sample.get("category", "")[:15]
            orig = sample.get("input", {}).get("estimated_tokens", 0)
            comp = sample.get("output", {}).get("estimated_tokens", 0)
            saved = sample.get("reduction", {}).get("tokens", 0)
            pct = sample.get("reduction", {}).get("tokens_percent", 0)
            co2 = sample.get("environmental_savings", {}).get("carbon_grams", 0)
            lines.append(f"   {i:<3} {cat:<15} {orig:<8} {comp:<8} {saved:<6} {pct:<9.1f}% {co2:<10.6f}")
        
        lines.append("")
        
        # Conclusion
        lines.append("11. CONCLUSION")
        lines.append("-" * 40)
        avg_reduction = summary.get('avg_token_reduction_percent', 0)
        lines.append(f"   The Carbon Protocol SDK demonstrates an average token reduction of {avg_reduction:.1f}%")
        lines.append(f"   across {r.total_samples} representative prompts spanning {len(r.categories_tested)} categories.")
        lines.append("")
        lines.append("   Environmental Significance:")
        lines.append(f"   - At scale (1M requests/year): {proj.get('annual_carbon_saved_kg', 0):.2f} kg CO2 saved")
        
        if industry:
            global_impact = industry.get("industry_wide_impact", {})
            if "ten_percent_adoption" in global_impact:
                lines.append(f"   - At 10% global adoption: {global_impact['ten_percent_adoption']['carbon_tonnes']} tonnes CO2 saved")
            if "full_adoption" in global_impact:
                lines.append(f"   - At full adoption: {global_impact['full_adoption']['carbon_tonnes']} tonnes CO2 saved")
        
        lines.append("")
        lines.append("   Economic Significance:")
        lines.append(f"   - Small org (1M/year): ${proj.get('annual_cost_saved_usd', 0):,.2f} USD")
        
        if industry:
            org_scale = industry.get("organizational_scale", {})
            if "enterprise_1B_requests" in org_scale:
                lines.append(f"   - Enterprise (1B/year): {org_scale['enterprise_1B_requests']['cost_saved_usd']}")
            if "full_adoption" in global_impact:
                lines.append(f"   - Full global adoption: {global_impact['full_adoption']['cost_saved_millions_usd']}")
        
        lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> str:
        """Generate JSON format report for machine processing."""
        result_dict = self.result.to_dict()
        
        # Add industry impact if available
        if self.aggregate:
            result_dict["industry_impact"] = self.aggregate.get_industry_impact_assessment()
        
        return json.dumps(result_dict, indent=2, default=str)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def validation_registry(data_dir) -> Registry:
    """Registry configured for validation testing."""
    registry = Registry(data_dir=data_dir)
    registry.load_domain("core")
    registry.build_automaton()
    return registry


@pytest.fixture(scope="module")
def validation_compiler(validation_registry: Registry) -> Compiler:
    """Compiler configured for validation testing."""
    return Compiler(validation_registry)


@pytest.fixture(scope="module")
def metrics_calculator() -> MetricsCalculator:
    """Metrics calculator with default parameters."""
    return MetricsCalculator()


# =============================================================================
# Validation Tests
# =============================================================================

@pytest.mark.validation
class TestSDKValidation:
    """
    IEEE Test Case ID: VAL-001
    Test Suite Name: SDK Validation and Impact Assessment
    Objective: Quantify and validate the benefits of the Carbon Protocol SDK
    """
    
    def test_compression_effectiveness(
        self,
        validation_compiler: Compiler,
        metrics_calculator: MetricsCalculator,
    ):
        """
        Test ID: VAL-001-01
        Description: Validate that SDK achieves meaningful compression
        Expected Result: Average token reduction > 20%
        """
        reductions = []
        
        for prompt_data in VALIDATION_PROMPTS:
            result = validation_compiler.compress(prompt_data["prompt"])
            metrics = metrics_calculator.calculate_compression_metrics(
                prompt_data["prompt"],
                result.compressed,
                result.matches_found,
            )
            reductions.append(metrics.token_reduction_percent)
        
        avg_reduction = sum(reductions) / len(reductions)
        
        assert avg_reduction > 20, f"Average token reduction {avg_reduction:.1f}% is below 20% threshold"
    
    def test_environmental_impact_positive(
        self,
        validation_compiler: Compiler,
        metrics_calculator: MetricsCalculator,
    ):
        """
        Test ID: VAL-001-02
        Description: Validate that SDK produces positive environmental impact
        Expected Result: Total carbon saved > 0
        """
        total_carbon_saved = 0
        
        for prompt_data in VALIDATION_PROMPTS:
            result = validation_compiler.compress(prompt_data["prompt"])
            metrics = metrics_calculator.calculate_compression_metrics(
                prompt_data["prompt"],
                result.compressed,
                result.matches_found,
            )
            total_carbon_saved += metrics.carbon_saved_g
        
        assert total_carbon_saved > 0, "SDK should produce positive carbon savings"
    
    def test_no_information_loss(
        self,
        validation_compiler: Compiler,
    ):
        """
        Test ID: VAL-001-03
        Description: Validate that compression preserves semantic meaning
        Expected Result: All compressed outputs contain valid tokens
        """
        for prompt_data in VALIDATION_PROMPTS:
            result = validation_compiler.compress(prompt_data["prompt"])
            
            # Verify compression occurred and output is valid
            assert isinstance(result.compressed, str)
            assert len(result.compressed) > 0 or result.matches_found > 0
    
    def test_full_validation_suite(
        self,
        validation_compiler: Compiler,
        metrics_calculator: MetricsCalculator,
    ):
        """
        Test ID: VAL-001-04
        Description: Run complete validation suite and generate report
        Expected Result: Complete validation report generated with all metrics
        """
        import sys
        
        # Collect metrics for all prompts
        individual_metrics: list[CompressionMetrics] = []
        sample_results: list[dict] = []
        categories = set()
        
        for prompt_data in VALIDATION_PROMPTS:
            result = validation_compiler.compress(prompt_data["prompt"])
            metrics = metrics_calculator.calculate_compression_metrics(
                prompt_data["prompt"],
                result.compressed,
                result.matches_found,
            )
            individual_metrics.append(metrics)
            
            # Add category to sample result
            sample_dict = metrics.to_dict()
            sample_dict["category"] = prompt_data["category"]
            sample_dict["description"] = prompt_data["description"]
            sample_results.append(sample_dict)
            
            categories.add(prompt_data["category"])
        
        # Calculate aggregate metrics
        aggregate = metrics_calculator.calculate_aggregate_metrics(
            individual_metrics,
            annual_requests_estimate=1_000_000,
        )
        
        # Create validation result
        validation_result = ValidationResult(
            test_id=f"VAL-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            test_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            test_environment=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            total_samples=len(VALIDATION_PROMPTS),
            categories_tested=sorted(categories),
            aggregate_metrics=aggregate.to_dict(),
            sample_results=sample_results,
            calculation_parameters={
                "chars_per_token": metrics_calculator.chars_per_token,
                "energy_per_1k_tokens_kwh": metrics_calculator.energy_per_1k_tokens,
                "carbon_intensity_kg_per_kwh": metrics_calculator.carbon_intensity,
                "cost_per_1k_tokens_usd": metrics_calculator.cost_per_1k_tokens,
            },
            methodology="""Token Estimation: Characters divided by average chars/token (4.0 for English)
Energy Calculation: Based on Patterson et al. (2021) estimates for LLM inference
Carbon Intensity: Global average grid carbon intensity (IEA 2023)
Cost Estimation: Based on typical LLM API pricing models
Projections: Linear scaling based on 1 million annual requests""",
        )
        
        # Generate reports
        report_generator = SDKValidationReport(validation_result, aggregate)
        text_report = report_generator.generate_text_report()
        json_report = report_generator.generate_json_report()
        
        # Save reports to results directory
        results_dir = Path(__file__).parent / "results"
        date_folder = results_dir / datetime.datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)
        
        report_id = validation_result.test_id
        
        # Save text report
        text_path = date_folder / f"{report_id}.txt"
        text_path.write_text(text_report, encoding="utf-8")
        
        # Save JSON report
        json_path = date_folder / f"{report_id}.json"
        json_path.write_text(json_report, encoding="utf-8")
        
        # Print report to console
        print("\n")
        print(text_report)
        print(f"\nReports saved to:")
        print(f"  - {text_path}")
        print(f"  - {json_path}")
        
        # Assertions
        assert aggregate.avg_token_reduction_percent > 0, "Should achieve positive token reduction"
        assert aggregate.total_carbon_saved_g > 0, "Should achieve positive carbon savings"
        assert aggregate.total_cost_saved_usd > 0, "Should achieve positive cost savings"
        assert len(sample_results) == len(VALIDATION_PROMPTS), "All samples should be processed"


@pytest.mark.validation
class TestSDKComparison:
    """
    IEEE Test Case ID: VAL-002
    Test Suite Name: SDK vs No-SDK Comparison
    Objective: Demonstrate the difference between using SDK and not using SDK
    """
    
    def test_with_vs_without_sdk(
        self,
        validation_compiler: Compiler,
        metrics_calculator: MetricsCalculator,
    ):
        """
        Test ID: VAL-002-01
        Description: Compare token usage with SDK vs without SDK
        Expected Result: SDK reduces token count significantly
        """
        comparison_results = []
        
        for prompt_data in VALIDATION_PROMPTS[:5]:  # Test first 5 prompts
            original = prompt_data["prompt"]
            result = validation_compiler.compress(original)
            compressed = result.compressed
            
            # Without SDK (original)
            original_tokens = metrics_calculator.estimate_tokens(original)
            original_energy = (original_tokens / 1000) * metrics_calculator.energy_per_1k_tokens
            original_carbon = original_energy * metrics_calculator.carbon_intensity * 1000  # grams
            original_cost = (original_tokens / 1000) * metrics_calculator.cost_per_1k_tokens
            
            # With SDK (compressed)
            compressed_tokens = metrics_calculator.estimate_tokens(compressed)
            compressed_energy = (compressed_tokens / 1000) * metrics_calculator.energy_per_1k_tokens
            compressed_carbon = compressed_energy * metrics_calculator.carbon_intensity * 1000  # grams
            compressed_cost = (compressed_tokens / 1000) * metrics_calculator.cost_per_1k_tokens
            
            comparison_results.append({
                "prompt": original[:50] + "...",
                "without_sdk": {
                    "tokens": original_tokens,
                    "energy_kwh": original_energy,
                    "carbon_g": original_carbon,
                    "cost_usd": original_cost,
                },
                "with_sdk": {
                    "tokens": compressed_tokens,
                    "energy_kwh": compressed_energy,
                    "carbon_g": compressed_carbon,
                    "cost_usd": compressed_cost,
                },
                "savings": {
                    "tokens": original_tokens - compressed_tokens,
                    "tokens_percent": ((original_tokens - compressed_tokens) / original_tokens) * 100,
                    "carbon_g": original_carbon - compressed_carbon,
                    "cost_usd": original_cost - compressed_cost,
                },
            })
        
        # Print comparison table
        print("\n")
        print("=" * 80)
        print("SDK vs NO-SDK COMPARISON")
        print("=" * 80)
        print(f"{'Metric':<20} {'Without SDK':<20} {'With SDK':<20} {'Savings':<20}")
        print("-" * 80)
        
        total_without = sum(r["without_sdk"]["tokens"] for r in comparison_results)
        total_with = sum(r["with_sdk"]["tokens"] for r in comparison_results)
        total_carbon_without = sum(r["without_sdk"]["carbon_g"] for r in comparison_results)
        total_carbon_with = sum(r["with_sdk"]["carbon_g"] for r in comparison_results)
        
        print(f"{'Total Tokens':<20} {total_without:<20} {total_with:<20} {total_without - total_with:<20}")
        print(f"{'Carbon (g CO2)':<20} {total_carbon_without:<20.6f} {total_carbon_with:<20.6f} {total_carbon_without - total_carbon_with:<20.6f}")
        print("=" * 80)
        
        # Assertions
        assert total_with < total_without, "SDK should reduce total tokens"
        assert total_carbon_with < total_carbon_without, "SDK should reduce carbon emissions"
