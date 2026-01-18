"""
Carbon Protocol SDK - Metrics Calculator

This module provides utilities for calculating the environmental and
computational impact of prompt compression, including:
- Token estimation (approximating LLM tokenization)
- Carbon emissions calculations
- Energy consumption estimates
- Cost savings projections
- Industry-scale impact projections

=============================================================================
GLOBAL LLM TOKEN USAGE STATISTICS (2026)
=============================================================================

Industry estimates based on public data from major LLM providers:

OpenAI (ChatGPT & API):
  - Active users: ~300 million (as of 2026)
  - Daily API requests: ~4.5 billion tokens/day
  - Annual volume: ~1.64 trillion tokens/year

Anthropic (Claude):
  - API usage accounts for significant market share
  - Estimated: ~200 million tokens/day
  - Annual volume: ~73 billion tokens/year

Google (Gemini/PaLM):
  - Enterprise deployments: ~500 million tokens/day
  - Annual volume: ~183 billion tokens/year

Microsoft (Azure OpenAI):
  - Enterprise API calls: ~1 billion tokens/day
  - Annual volume: ~365 billion tokens/year

**CONSERVATIVE GLOBAL ESTIMATE:**
  Total Daily Token Consumption: ~6.2 billion tokens/day
  Total Annual Token Consumption: ~2.3 trillion tokens/year
  
  This represents only major providers and excludes:
  - Self-hosted models (Llama, Mistral, etc.)
  - Regional providers (China, EU specific providers)
  - Enterprise internal deployments
  
  **Realistic total global estimate: 5-10 trillion tokens/year**

=============================================================================

Reference Data Sources:
- Patterson et al. (2021) "Carbon Emissions and Large Neural Network Training"
- Strubell et al. (2019) "Energy and Policy Considerations for Deep Learning"
- Luccioni et al. (2023) "Estimating the Carbon Footprint of BLOOM"
- Wu et al. (2022) "Sustainable AI: Environmental Implications"
- IEA Global Energy & CO2 Status Report (2023)
- Google Environmental Report (2023)
- Microsoft Sustainability Report (2023)
- ML CO2 Impact Calculator methodology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Constants and Reference Values
# =============================================================================

# Average characters per token (GPT-style tokenization)
# Based on OpenAI's tokenizer analysis: ~4 chars per token for English
CHARS_PER_TOKEN = 4.0

# Energy consumption per 1000 tokens (in kWh)
# Updated based on GPT-4 class models inference (not training)
# Sources: 
# - Patterson et al. (2021) inference estimates
# - Luccioni et al. (2023) BLOOM inference measurements
# - Industry benchmarks for large language model inference
ENERGY_PER_1K_TOKENS_KWH = 0.0003  # ~0.3 Wh per 1000 tokens (conservative)

# Carbon intensity of electricity (kg CO2 per kWh)
# Global average grid intensity
# Source: IEA Global Energy & CO2 Status Report (2023)
# Note: US average: 0.386, EU: 0.255, Global: 0.475
CARBON_INTENSITY_KG_PER_KWH = 0.475

# Cost per 1000 tokens (USD) - approximate API pricing
# Based on typical LLM API pricing (input tokens, 2026 rates)
# GPT-4: $0.01/1K, GPT-3.5: $0.0005/1K, Claude: $0.008/1K
COST_PER_1K_TOKENS_USD = 0.01  # $0.01 per 1K input tokens (conservative)

# Water consumption for datacenter cooling (liters per kWh)
# Sources:
# - Google Environmental Report: 1.8L/kWh
# - Microsoft Sustainability Report: 1.7L/kWh
# - AWS Water Positive initiative data
WATER_PER_KWH_LITERS = 1.8

# =============================================================================
# Industry-Scale Usage Projections
# =============================================================================

# Conservative global daily token consumption (2026)
GLOBAL_DAILY_TOKENS = 6_200_000_000  # 6.2 billion tokens/day
GLOBAL_ANNUAL_TOKENS = 2_300_000_000_000  # 2.3 trillion tokens/year

# Realistic global estimate including all providers
GLOBAL_ANNUAL_TOKENS_REALISTIC = 7_000_000_000_000  # 7 trillion tokens/year


@dataclass
class TokenMetrics:
    """Token-related metrics for a text sample."""
    char_count: int
    estimated_tokens: int
    
    @classmethod
    def from_text(cls, text: str) -> "TokenMetrics":
        """Calculate token metrics from text."""
        char_count = len(text)
        # Estimate tokens using average chars per token
        estimated_tokens = max(1, int(char_count / CHARS_PER_TOKEN))
        return cls(char_count=char_count, estimated_tokens=estimated_tokens)


@dataclass
class EnvironmentalMetrics:
    """Environmental impact metrics."""
    energy_kwh: float
    carbon_kg: float
    carbon_g: float
    water_liters: float
    
    @classmethod
    def from_tokens(cls, token_count: int) -> "EnvironmentalMetrics":
        """Calculate environmental metrics from token count."""
        energy_kwh = (token_count / 1000) * ENERGY_PER_1K_TOKENS_KWH
        carbon_kg = energy_kwh * CARBON_INTENSITY_KG_PER_KWH
        carbon_g = carbon_kg * 1000
        water_liters = energy_kwh * WATER_PER_KWH_LITERS
        
        return cls(
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            carbon_g=carbon_g,
            water_liters=water_liters,
        )


@dataclass
class CostMetrics:
    """Cost-related metrics."""
    api_cost_usd: float
    
    @classmethod
    def from_tokens(cls, token_count: int) -> "CostMetrics":
        """Calculate cost metrics from token count."""
        api_cost_usd = (token_count / 1000) * COST_PER_1K_TOKENS_USD
        return cls(api_cost_usd=api_cost_usd)


@dataclass
class CompressionMetrics:
    """Complete metrics for a compression operation."""
    # Input metrics
    original_text: str
    original_chars: int
    original_tokens: int
    
    # Output metrics
    compressed_text: str
    compressed_chars: int
    compressed_tokens: int
    
    # Reduction metrics
    char_reduction: int
    char_reduction_percent: float
    token_reduction: int
    token_reduction_percent: float
    
    # Environmental savings
    energy_saved_kwh: float
    carbon_saved_g: float
    carbon_saved_kg: float
    water_saved_liters: float
    
    # Cost savings
    cost_saved_usd: float
    
    # Efficiency metrics
    compression_ratio: float
    patterns_matched: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input": {
                "text": self.original_text[:100] + "..." if len(self.original_text) > 100 else self.original_text,
                "characters": self.original_chars,
                "estimated_tokens": self.original_tokens,
            },
            "output": {
                "text": self.compressed_text[:100] + "..." if len(self.compressed_text) > 100 else self.compressed_text,
                "characters": self.compressed_chars,
                "estimated_tokens": self.compressed_tokens,
            },
            "reduction": {
                "characters": self.char_reduction,
                "characters_percent": round(self.char_reduction_percent, 2),
                "tokens": self.token_reduction,
                "tokens_percent": round(self.token_reduction_percent, 2),
            },
            "environmental_savings": {
                "energy_kwh": self.energy_saved_kwh,
                "carbon_grams": round(self.carbon_saved_g, 6),
                "carbon_kg": round(self.carbon_saved_kg, 9),
                "water_liters": round(self.water_saved_liters, 6),
            },
            "cost_savings": {
                "api_cost_usd": round(self.cost_saved_usd, 6),
            },
            "efficiency": {
                "compression_ratio": round(self.compression_ratio, 4),
                "patterns_matched": self.patterns_matched,
            },
        }


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple compression operations."""
    sample_count: int
    
    # Totals
    total_original_chars: int
    total_compressed_chars: int
    total_original_tokens: int
    total_compressed_tokens: int
    
    # Savings
    total_char_reduction: int
    total_token_reduction: int
    total_energy_saved_kwh: float
    total_carbon_saved_g: float
    total_water_saved_liters: float
    total_cost_saved_usd: float
    
    # Averages
    avg_compression_ratio: float
    avg_token_reduction_percent: float
    avg_patterns_matched: float
    
    # Projections (annualized based on scale factor)
    projected_annual_tokens_saved: int = 0
    projected_annual_carbon_saved_kg: float = 0.0
    projected_annual_cost_saved_usd: float = 0.0
    
    individual_metrics: list[CompressionMetrics] = field(default_factory=list)
    
    def get_industry_impact_assessment(self) -> dict[str, Any]:
        """
        Calculate realistic industry-scale impact using global token consumption.
        
        Returns comprehensive impact assessment across different scales:
        - Small organization (1M requests/year)
        - Medium organization (10M requests/year)
        - Large organization (100M requests/year)
        - Industry-wide (assuming 1%, 5%, 10% adoption)
        
        Returns:
            Dictionary with multi-scale impact projections
        """
        return {
            "sample_metrics": {
                "total_samples": self.sample_count,
                "average_reduction_percent": round(self.avg_token_reduction_percent, 2),
                "tokens_per_sample": round(self.total_token_reduction / self.sample_count, 1) if self.sample_count > 0 else 0,
            },
            "organizational_scale": {
                "small_org_1M_requests": self._calculate_scale_impact(1_000_000),
                "medium_org_10M_requests": self._calculate_scale_impact(10_000_000),
                "large_org_100M_requests": self._calculate_scale_impact(100_000_000),
                "enterprise_1B_requests": self._calculate_scale_impact(1_000_000_000),
            },
            "industry_wide_impact": {
                "one_percent_adoption": self._calculate_global_adoption_impact(0.01),
                "five_percent_adoption": self._calculate_global_adoption_impact(0.05),
                "ten_percent_adoption": self._calculate_global_adoption_impact(0.10),
                "full_adoption": self._calculate_global_adoption_impact(1.0),
            },
            "environmental_equivalents": self._calculate_environmental_equivalents(),
            "methodology": {
                "global_baseline": {
                    "daily_tokens": f"{GLOBAL_DAILY_TOKENS:,}",
                    "annual_tokens_conservative": f"{GLOBAL_ANNUAL_TOKENS:,}",
                    "annual_tokens_realistic": f"{GLOBAL_ANNUAL_TOKENS_REALISTIC:,}",
                },
                "compression_rate": f"{self.avg_token_reduction_percent:.2f}%",
                "energy_per_1k_tokens": f"{ENERGY_PER_1K_TOKENS_KWH} kWh",
                "carbon_intensity": f"{CARBON_INTENSITY_KG_PER_KWH} kg CO2/kWh",
                "api_cost": f"${COST_PER_1K_TOKENS_USD}/1K tokens",
            },
        }
    
    def _calculate_scale_impact(self, request_count: int) -> dict[str, Any]:
        """Calculate impact for a specific request volume."""
        if self.sample_count == 0:
            return {}
        
        avg_tokens_saved_per_request = self.total_token_reduction / self.sample_count
        total_tokens_saved = avg_tokens_saved_per_request * request_count
        
        energy_kwh = (total_tokens_saved / 1000) * ENERGY_PER_1K_TOKENS_KWH
        carbon_kg = energy_kwh * CARBON_INTENSITY_KG_PER_KWH
        cost_usd = (total_tokens_saved / 1000) * COST_PER_1K_TOKENS_USD
        water_liters = energy_kwh * WATER_PER_KWH_LITERS
        
        return {
            "requests": f"{request_count:,}",
            "tokens_saved": f"{total_tokens_saved:,.0f}",
            "carbon_kg": round(carbon_kg, 2),
            "carbon_tonnes": round(carbon_kg / 1000, 4),
            "energy_kwh": round(energy_kwh, 2),
            "cost_saved_usd": f"${cost_usd:,.2f}",
            "water_saved_liters": round(water_liters, 2),
        }
    
    def _calculate_global_adoption_impact(self, adoption_rate: float) -> dict[str, Any]:
        """Calculate impact for industry-wide adoption at specified rate."""
        if self.avg_token_reduction_percent == 0:
            return {}
        
        # Use realistic global estimate
        annual_tokens = GLOBAL_ANNUAL_TOKENS_REALISTIC
        tokens_saved = annual_tokens * (self.avg_token_reduction_percent / 100) * adoption_rate
        
        energy_kwh = (tokens_saved / 1000) * ENERGY_PER_1K_TOKENS_KWH
        carbon_kg = energy_kwh * CARBON_INTENSITY_KG_PER_KWH
        carbon_tonnes = carbon_kg / 1000
        cost_usd = (tokens_saved / 1000) * COST_PER_1K_TOKENS_USD
        water_liters = energy_kwh * WATER_PER_KWH_LITERS
        
        return {
            "adoption_rate": f"{adoption_rate * 100:.0f}%",
            "affected_tokens": f"{annual_tokens * adoption_rate:,.0f}",
            "tokens_saved": f"{tokens_saved:,.0f}",
            "carbon_tonnes": f"{carbon_tonnes:,.2f}",
            "carbon_megatonnes": round(carbon_tonnes / 1_000_000, 6),
            "energy_mwh": round(energy_kwh / 1000, 2),
            "cost_saved_millions_usd": f"${cost_usd / 1_000_000:,.2f}M",
            "water_saved_megaliters": round(water_liters / 1_000_000, 4),
        }
    
    def _calculate_environmental_equivalents(self) -> dict[str, Any]:
        """Calculate relatable environmental equivalents for projected savings."""
        annual_carbon_kg = self.projected_annual_carbon_saved_kg
        
        # Environmental equivalents (EPA methodology)
        # Average car emits ~4.6 tonnes CO2/year
        cars_per_year = annual_carbon_kg / 4_600
        
        # Average tree absorbs ~21.77 kg CO2/year
        trees_per_year = annual_carbon_kg / 21.77
        
        # Smartphone charge emits ~0.008 kg CO2
        phone_charges = annual_carbon_kg / 0.008
        
        # Miles driven (average car: 0.404 kg CO2/mile)
        miles_driven = annual_carbon_kg / 0.404
        
        # Flights (short-haul: ~90 kg CO2/hour)
        flight_hours = annual_carbon_kg / 90
        
        return {
            "annual_carbon_kg": round(annual_carbon_kg, 2),
            "equivalent_to": {
                "cars_off_road_days": round(cars_per_year * 365, 1),
                "trees_planted_years": round(trees_per_year, 1),
                "smartphone_charges": f"{phone_charges:,.0f}",
                "miles_not_driven": f"{miles_driven:,.0f}",
                "flight_hours_avoided": round(flight_hours, 2),
            },
            "sources": [
                "EPA GHG Equivalencies Calculator",
                "European Environment Agency Carbon Footprint Data",
                "ICCT Transportation Emissions Database",
            ],
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": {
                "sample_count": self.sample_count,
                "total_original_tokens": self.total_original_tokens,
                "total_compressed_tokens": self.total_compressed_tokens,
                "total_token_reduction": self.total_token_reduction,
                "avg_compression_ratio": round(self.avg_compression_ratio, 4),
                "avg_token_reduction_percent": round(self.avg_token_reduction_percent, 2),
            },
            "environmental_impact": {
                "total_energy_saved_kwh": round(self.total_energy_saved_kwh, 6),
                "total_carbon_saved_grams": round(self.total_carbon_saved_g, 4),
                "total_water_saved_liters": round(self.total_water_saved_liters, 6),
            },
            "cost_impact": {
                "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
            },
            "projections": {
                "annual_tokens_saved": self.projected_annual_tokens_saved,
                "annual_carbon_saved_kg": round(self.projected_annual_carbon_saved_kg, 2),
                "annual_cost_saved_usd": round(self.projected_annual_cost_saved_usd, 2),
            },
        }


class MetricsCalculator:
    """
    Calculator for SDK impact metrics.
    
    This class computes comprehensive metrics comparing text before and after
    compression, including token counts, environmental impact, and cost savings.
    """
    
    def __init__(
        self,
        chars_per_token: float = CHARS_PER_TOKEN,
        energy_per_1k_tokens: float = ENERGY_PER_1K_TOKENS_KWH,
        carbon_intensity: float = CARBON_INTENSITY_KG_PER_KWH,
        cost_per_1k_tokens: float = COST_PER_1K_TOKENS_USD,
    ):
        """
        Initialize the metrics calculator with configurable parameters.
        
        Args:
            chars_per_token: Average characters per LLM token
            energy_per_1k_tokens: Energy consumption per 1000 tokens (kWh)
            carbon_intensity: Carbon intensity of electricity (kg CO2/kWh)
            cost_per_1k_tokens: API cost per 1000 tokens (USD)
        """
        self.chars_per_token = chars_per_token
        self.energy_per_1k_tokens = energy_per_1k_tokens
        self.carbon_intensity = carbon_intensity
        self.cost_per_1k_tokens = cost_per_1k_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        return max(1, int(len(text) / self.chars_per_token))
    
    def calculate_compression_metrics(
        self,
        original_text: str,
        compressed_text: str,
        patterns_matched: int = 0,
    ) -> CompressionMetrics:
        """
        Calculate comprehensive metrics for a single compression operation.
        
        Args:
            original_text: The original uncompressed text
            compressed_text: The compressed output text
            patterns_matched: Number of patterns matched during compression
            
        Returns:
            CompressionMetrics object with all calculated values
        """
        # Character counts
        original_chars = len(original_text)
        compressed_chars = len(compressed_text)
        char_reduction = original_chars - compressed_chars
        char_reduction_percent = (char_reduction / original_chars * 100) if original_chars > 0 else 0
        
        # Token estimates
        original_tokens = self.estimate_tokens(original_text)
        compressed_tokens = self.estimate_tokens(compressed_text)
        token_reduction = original_tokens - compressed_tokens
        token_reduction_percent = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
        
        # Environmental impact of saved tokens
        tokens_saved = max(0, token_reduction)
        energy_saved_kwh = (tokens_saved / 1000) * self.energy_per_1k_tokens
        carbon_saved_kg = energy_saved_kwh * self.carbon_intensity
        carbon_saved_g = carbon_saved_kg * 1000
        water_saved_liters = energy_saved_kwh * WATER_PER_KWH_LITERS
        
        # Cost savings
        cost_saved_usd = (tokens_saved / 1000) * self.cost_per_1k_tokens
        
        # Compression ratio
        compression_ratio = compressed_chars / original_chars if original_chars > 0 else 1.0
        
        return CompressionMetrics(
            original_text=original_text,
            original_chars=original_chars,
            original_tokens=original_tokens,
            compressed_text=compressed_text,
            compressed_chars=compressed_chars,
            compressed_tokens=compressed_tokens,
            char_reduction=char_reduction,
            char_reduction_percent=char_reduction_percent,
            token_reduction=token_reduction,
            token_reduction_percent=token_reduction_percent,
            energy_saved_kwh=energy_saved_kwh,
            carbon_saved_g=carbon_saved_g,
            carbon_saved_kg=carbon_saved_kg,
            water_saved_liters=water_saved_liters,
            cost_saved_usd=cost_saved_usd,
            compression_ratio=compression_ratio,
            patterns_matched=patterns_matched,
        )
    
    def calculate_aggregate_metrics(
        self,
        metrics_list: list[CompressionMetrics],
        annual_requests_estimate: int = 1_000_000,
    ) -> AggregateMetrics:
        """
        Calculate aggregated metrics across multiple compression operations.
        
        Args:
            metrics_list: List of individual CompressionMetrics
            annual_requests_estimate: Estimated annual API requests for projections
            
        Returns:
            AggregateMetrics with totals, averages, and projections
        """
        if not metrics_list:
            return AggregateMetrics(
                sample_count=0,
                total_original_chars=0,
                total_compressed_chars=0,
                total_original_tokens=0,
                total_compressed_tokens=0,
                total_char_reduction=0,
                total_token_reduction=0,
                total_energy_saved_kwh=0,
                total_carbon_saved_g=0,
                total_water_saved_liters=0,
                total_cost_saved_usd=0,
                avg_compression_ratio=1.0,
                avg_token_reduction_percent=0,
                avg_patterns_matched=0,
            )
        
        n = len(metrics_list)
        
        # Calculate totals
        total_original_chars = sum(m.original_chars for m in metrics_list)
        total_compressed_chars = sum(m.compressed_chars for m in metrics_list)
        total_original_tokens = sum(m.original_tokens for m in metrics_list)
        total_compressed_tokens = sum(m.compressed_tokens for m in metrics_list)
        total_char_reduction = sum(m.char_reduction for m in metrics_list)
        total_token_reduction = sum(m.token_reduction for m in metrics_list)
        total_energy_saved = sum(m.energy_saved_kwh for m in metrics_list)
        total_carbon_saved = sum(m.carbon_saved_g for m in metrics_list)
        total_water_saved = sum(m.water_saved_liters for m in metrics_list)
        total_cost_saved = sum(m.cost_saved_usd for m in metrics_list)
        
        # Calculate averages
        avg_compression_ratio = sum(m.compression_ratio for m in metrics_list) / n
        avg_token_reduction_percent = sum(m.token_reduction_percent for m in metrics_list) / n
        avg_patterns_matched = sum(m.patterns_matched for m in metrics_list) / n
        
        # Calculate projections based on sample average
        avg_tokens_saved_per_request = total_token_reduction / n if n > 0 else 0
        projected_annual_tokens = int(avg_tokens_saved_per_request * annual_requests_estimate)
        projected_annual_energy = (projected_annual_tokens / 1000) * self.energy_per_1k_tokens
        projected_annual_carbon_kg = projected_annual_energy * self.carbon_intensity
        projected_annual_cost = (projected_annual_tokens / 1000) * self.cost_per_1k_tokens
        
        return AggregateMetrics(
            sample_count=n,
            total_original_chars=total_original_chars,
            total_compressed_chars=total_compressed_chars,
            total_original_tokens=total_original_tokens,
            total_compressed_tokens=total_compressed_tokens,
            total_char_reduction=total_char_reduction,
            total_token_reduction=total_token_reduction,
            total_energy_saved_kwh=total_energy_saved,
            total_carbon_saved_g=total_carbon_saved,
            total_water_saved_liters=total_water_saved,
            total_cost_saved_usd=total_cost_saved,
            avg_compression_ratio=avg_compression_ratio,
            avg_token_reduction_percent=avg_token_reduction_percent,
            avg_patterns_matched=avg_patterns_matched,
            projected_annual_tokens_saved=projected_annual_tokens,
            projected_annual_carbon_saved_kg=projected_annual_carbon_kg,
            projected_annual_cost_saved_usd=projected_annual_cost,
            individual_metrics=metrics_list,
        )
