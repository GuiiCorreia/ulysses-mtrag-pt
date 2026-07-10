"""
Cost and inference efficiency metrics.

Based on Belcak & Wattenhofer (2025) - Small Language Models Are the Future of Agentic AI.
Emphasizes efficiency metrics (latency, cost, tokens/second) as first-class metrics.

References:
  - Belcak, P. & Wattenhofer, R. (2025). Small Language Models Are the Future of Agentic AI.
    arXiv:2506.02153.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Model pricing data (cost per 1M tokens)
# Based on DeepInfra and OpenRouter pricing (2025)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # DeepInfra API pricing (USD per 1M tokens) — all models via API
    # Source: DeepInfra pricing page (April 2026)
    "meta-llama/Llama-3.2-1B-Instruct": {"input": 0.013, "output": 0.013},
    "meta-llama/Llama-3.2-3B-Instruct": {"input": 0.06, "output": 0.06},
    "microsoft/phi-4":                  {"input": 0.07, "output": 0.14},
    "google/gemma-3-4b-it": {"input": 0.07, "output": 0.07},
    "google/gemma-4-26B-A4B-it": {"input": 0.20, "output": 0.20},
    "meta-llama/Llama-3.3-70B-Instruct": {"input": 0.23, "output": 0.23},

    # OpenRouter pricing (USD per 1M tokens)
    "qwen/qwen3-8b": {"input": 0.07, "output": 0.07},
    "deepseek/deepseek-r1-distill-qwen-32b": {"input": 0.29, "output": 0.29},
}


@dataclass
class CostMetrics:
    """Cost and efficiency metrics for model evaluation.

    Attributes:
        tokens_per_second: Processing speed (tokens/second)
        cost_per_1k_tokens_in: Cost per 1k input tokens (USD)
        cost_per_1k_tokens_out: Cost per 1k output tokens (USD)
        total_cost: Total cost for this evaluation (USD)
        bertscore_per_dollar: Quality (BERTScore) per dollar spent
        faithfulness_per_dollar: Quality (Faithfulness) per dollar spent
        latency_cost_ratio: Latency (seconds) per dollar spent
    """
    tokens_per_second: float
    cost_per_1k_tokens_in: float
    cost_per_1k_tokens_out: float
    total_cost: float

    # Efficiency ratios (Belcak 2025 emphasis)
    bertscore_per_dollar: float = 0.0
    faithfulness_per_dollar: float = 0.0
    latency_cost_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tokens_per_second": round(self.tokens_per_second, 2),
            "cost_per_1k_tokens_in": round(self.cost_per_1k_tokens_in, 6),
            "cost_per_1k_tokens_out": round(self.cost_per_1k_tokens_out, 6),
            "total_cost": round(self.total_cost, 6),
            "bertscore_per_dollar": round(self.bertscore_per_dollar, 2),
            "faithfulness_per_dollar": round(self.faithfulness_per_dollar, 2),
            "latency_cost_ratio": round(self.latency_cost_ratio, 2),
        }


def get_model_pricing(model: str) -> Optional[Dict[str, float]]:
    """
    Get pricing information for a model.

    Args:
        model: Model identifier

    Returns:
        Dictionary with 'input' and 'output' pricing (USD per 1M tokens),
        or None if pricing not found
    """
    # Exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Partial match (for variations)
    for model_key, pricing in MODEL_PRICING.items():
        if model in model_key or model_key in model:
            return pricing

    logger.warning(f"Pricing not found for model: {model}")
    return None


def calculate_cost_efficiency(
    bertscore: float,
    faithfulness: float,
    latency: float,
    n_tokens_in: int,
    n_tokens_out: int,
    n_samples: int,
    model: str,
    model_pricing: Optional[Dict[str, float]] = None,
) -> CostMetrics:
    """
    Calculate cost and efficiency metrics for a model evaluation.

    Args:
        bertscore: BERTScore F1 (0.0-1.0)
        faithfulness: Faithfulness score (0.0-1.0)
        latency: Average latency per sample (seconds)
        n_tokens_in: Total input tokens
        n_tokens_out: Total output tokens
        n_samples: Number of samples evaluated
        model: Model identifier
        model_pricing: Optional pricing dictionary (auto-looked up if None)

    Returns:
        CostMetrics object with all efficiency metrics

    Reference: Belcak & Wattenhofer (2025) - emphasis on efficiency metrics
    """
    # Get pricing
    if model_pricing is None:
        model_pricing = get_model_pricing(model)

    if model_pricing is None:
        logger.warning(f"No pricing for {model}, using $0.00")
        model_pricing = {"input": 0.00, "output": 0.00}

    # Calculate total cost
    cost_per_token_in = model_pricing["input"] / 1_000_000
    cost_per_token_out = model_pricing["output"] / 1_000_000

    total_cost = (n_tokens_in * cost_per_token_in) + (n_tokens_out * cost_per_token_out)

    # Calculate tokens per second (throughput)
    total_tokens = n_tokens_in + n_tokens_out
    total_latency = latency * n_samples

    if total_latency > 0:
        tokens_per_second = total_tokens / total_latency
    else:
        tokens_per_second = 0.0
        logger.warning(f"Zero total latency for {model}, setting tokens_per_second to 0")

    # Calculate efficiency ratios (avoid division by zero)
    if total_cost > 0:
        bertscore_per_dollar = bertscore / total_cost
        faithfulness_per_dollar = faithfulness / total_cost
        latency_cost_ratio = latency / total_cost
    else:
        bertscore_per_dollar = 0.0
        faithfulness_per_dollar = 0.0
        latency_cost_ratio = 0.0
        logger.warning(f"Zero total cost for {model}, efficiency ratios set to 0")

    return CostMetrics(
        tokens_per_second=tokens_per_second,
        cost_per_1k_tokens_in=model_pricing["input"] / 1000,
        cost_per_1k_tokens_out=model_pricing["output"] / 1000,
        total_cost=total_cost,
        bertscore_per_dollar=bertscore_per_dollar,
        faithfulness_per_dollar=faithfulness_per_dollar,
        latency_cost_ratio=latency_cost_ratio,
    )


def calculate_cost_efficiency_batch(
    results: Dict[str, Dict],
    model: str,
) -> CostMetrics:
    """
    Calculate cost efficiency from aggregated results dictionary.

    Args:
        results: Dictionary with aggregated metrics for a model
                 Should contain: bertscore, faithfulness, latency_avg,
                               tokens_in_avg, tokens_out_avg, n_queries
        model: Model identifier

    Returns:
        CostMetrics object
    """
    bertscore = results.get("bertscore", 0.0)
    faithfulness = results.get("faithfulness_avg", 0.0)
    latency = results.get("latency_avg", 0.0)
    n_tokens_in_avg = results.get("tokens_in_avg", 0)
    n_tokens_out_avg = results.get("tokens_out_avg", 0)
    n_queries = results.get("n_queries", 1)

    n_tokens_in = n_tokens_in_avg * n_queries
    n_tokens_out = n_tokens_out_avg * n_queries

    return calculate_cost_efficiency(
        bertscore=bertscore,
        faithfulness=faithfulness,
        latency=latency,
        n_tokens_in=n_tokens_in,
        n_tokens_out=n_tokens_out,
        n_samples=n_queries,
        model=model,
    )


def compare_cost_efficiency(
    model_a: str,
    model_b: str,
    results_a: Dict[str, Dict],
    results_b: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Compare cost efficiency between two models.

    Args:
        model_a: First model identifier
        model_b: Second model identifier
        results_a: Results for model A
        results_b: Results for model B

    Returns:
        Dictionary with comparison ratios (model_a / model_b)
    """
    metrics_a = calculate_cost_efficiency_batch(results_a, model_a)
    metrics_b = calculate_cost_efficiency_batch(results_b, model_b)

    return {
        "tokens_per_second_ratio": (
            metrics_a.tokens_per_second / metrics_b.tokens_per_second
            if metrics_b.tokens_per_second > 0 else 0.0
        ),
        "total_cost_ratio": (
            metrics_a.total_cost / metrics_b.total_cost
            if metrics_b.total_cost > 0 else 0.0
        ),
        "bertscore_per_dollar_ratio": (
            metrics_a.bertscore_per_dollar / metrics_b.bertscore_per_dollar
            if metrics_b.bertscore_per_dollar > 0 else 0.0
        ),
        "faithfulness_per_dollar_ratio": (
            metrics_a.faithfulness_per_dollar / metrics_b.faithfulness_per_dollar
            if metrics_b.faithfulness_per_dollar > 0 else 0.0
        ),
    }
