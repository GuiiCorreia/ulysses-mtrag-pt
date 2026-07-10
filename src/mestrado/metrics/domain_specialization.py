"""
Domain specialization metrics.

Measures how well models specialize in a particular domain (e.g., legislative)
compared to general domain (e.g., Wikipedia). Based on Belcak & Wattenhofer (2025)
emphasis on domain-specific task performance.

References:
  - Belcak, P. & Wattenhofer, R. (2025). Small Language Models Are the Future of Agentic AI.
    arXiv:2506.02153.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DomainSpecializationMetrics:
    """Domain specialization metrics.

    Attributes:
        specialization_score: Overall specialization score (-1 to 1)
        domain_better: Whether model performs better in specialized domain
        gap_magnitude: Magnitude of performance gap between domains
        metric_breakdown: Individual metric gaps (e.g., bertscore_gap, faithfulness_gap)
    """
    specialization_score: float
    domain_better: bool
    gap_magnitude: float
    metric_breakdown: Dict[str, float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "specialization_score": round(self.specialization_score, 4),
            "domain_better": self.domain_better,
            "gap_magnitude": round(self.gap_magnitude, 4),
            "metric_breakdown": {
                k: round(v, 4) for k, v in self.metric_breakdown.items()
            },
        }


def domain_specialization_score(
    legislative_metrics: Dict[str, float],
    general_metrics: Dict[str, float],
    metric_weights: Optional[Dict[str, float]] = None,
) -> DomainSpecializationMetrics:
    """
    Calculate domain specialization score.

    Measures if a model specializes in legislative domain vs. general domain.

    Args:
        legislative_metrics: Metrics on Ulysses (legislative domain)
            Expected keys: 'bertscore', 'faithfulness', 'rouge_l', 'relevance', 'coherence'
        general_metrics: Metrics on MIRACL or Wikipedia (general domain)
            Same keys as legislative_metrics
        metric_weights: Weights for each metric (default: equal weights)

    Returns:
        DomainSpecializationMetrics object

    Interpretation:
        - score > 0: Better on legislative than general (specialized in legislative)
        - score < 0: Better on general than legislative (not specialized)
        - score ≈ 0: No specialization (similar performance)

    Reference: Belcak & Wattenhofer (2025) - domain-specific task emphasis
    """
    # Default weights (all metrics equally important)
    if metric_weights is None:
        metric_weights = {
            "bertscore": 0.25,
            "faithfulness": 0.25,
            "rouge_l": 0.20,
            "relevance": 0.15,
            "coherence": 0.15,
        }

    # Calculate gap for each metric
    metric_breakdown = {}
    total_weighted_gap = 0.0
    total_weight = 0.0

    for metric, weight in metric_weights.items():
        if metric in legislative_metrics and metric in general_metrics:
            legislative_score = legislative_metrics[metric]
            general_score = general_metrics[metric]

            # Normalized gap: (legislative - general) / max(legislative, general)
            max_score = max(legislative_score, general_score, 0.01)  # Avoid division by zero
            gap = (legislative_score - general_score) / max_score

            metric_breakdown[metric] = gap
            total_weighted_gap += gap * weight
            total_weight += weight
        else:
            logger.warning(f"Metric {metric} not found in one or both metric dictionaries")

    # Calculate overall specialization score
    if total_weight > 0:
        specialization_score = total_weighted_gap / total_weight
    else:
        specialization_score = 0.0
        logger.warning("No valid metrics for specialization calculation")

    # Determine if model specializes in legislative domain
    domain_better = specialization_score > 0

    # Calculate gap magnitude
    gap_magnitude = abs(specialization_score)

    return DomainSpecializationMetrics(
        specialization_score=specialization_score,
        domain_better=domain_better,
        gap_magnitude=gap_magnitude,
        metric_breakdown=metric_breakdown,
    )


def cross_domain_gap_analysis(
    results_legislative: Dict[str, Dict],
    results_general: Dict[str, Dict],
    models: List[str],
) -> Dict[str, DomainSpecializationMetrics]:
    """
    Perform cross-domain gap analysis for multiple models.

    Args:
        results_legislative: Results on Ulysses (legislative domain)
            Format: {model_name: {metric: value, ...}}
        results_general: Results on MIRACL/Wikipedia (general domain)
            Same format as results_legislative
        models: List of model names to analyze

    Returns:
        Dictionary mapping model names to DomainSpecializationMetrics
    """
    specialization_results = {}

    for model in models:
        if model in results_legislative and model in results_general:
            leg_metrics = results_legislative[model]
            gen_metrics = results_general[model]

            spec = domain_specialization_score(
                legislative_metrics={
                    "bertscore": leg_metrics.get("bertscore", 0.0),
                    "faithfulness": leg_metrics.get("faithfulness", 0.0),
                    "rouge_l": leg_metrics.get("rouge_l", 0.0),
                    "relevance": leg_metrics.get("relevance", 0.0),
                    "coherence": leg_metrics.get("coherence", 0.0),
                },
                general_metrics={
                    "bertscore": gen_metrics.get("bertscore", 0.0),
                    "faithfulness": gen_metrics.get("faithfulness", 0.0),
                    "rouge_l": gen_metrics.get("rouge_l", 0.0),
                    "relevance": gen_metrics.get("relevance", 0.0),
                    "coherence": gen_metrics.get("coherence", 0.0),
                },
            )

            specialization_results[model] = spec
        else:
            logger.warning(f"Model {model} not found in one or both result sets")

    return specialization_results


def interpret_specialization(score: float) -> str:
    """
    Interpret domain specialization score.

    Args:
        score: Specialization score (-1 to 1)

    Returns:
        Interpretation string
    """
    if score > 0.3:
        return "Strongly specialized in legislative domain"
    elif score > 0.1:
        return "Moderately specialized in legislative domain"
    elif score > -0.1:
        return "No clear specialization (similar performance)"
    elif score > -0.3:
        return "Moderately specialized in general domain"
    else:
        return "Strongly specialized in general domain"


def compare_specialization(
    model_a: str,
    model_b: str,
    specialization_a: DomainSpecializationMetrics,
    specialization_b: DomainSpecializationMetrics,
) -> Dict[str, str]:
    """
    Compare domain specialization between two models.

    Args:
        model_a: First model identifier
        model_b: Second model identifier
        specialization_a: Specialization metrics for model A
        specialization_b: Specialization metrics for model B

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "model_a": model_a,
        "model_b": model_b,
        "model_a_score": round(specialization_a.specialization_score, 4),
        "model_b_score": round(specialization_b.specialization_score, 4),
        "model_a_interpretation": interpret_specialization(specialization_a.specialization_score),
        "model_b_interpretation": interpret_specialization(specialization_b.specialization_score),
    }

    # Determine which is more specialized in legislative domain
    if specialization_a.specialization_score > specialization_b.specialization_score:
        comparison["more_specialized"] = model_a
        comparison["specialization_gap"] = round(
            specialization_a.specialization_score - specialization_b.specialization_score, 4
        )
    elif specialization_b.specialization_score > specialization_a.specialization_score:
        comparison["more_specialized"] = model_b
        comparison["specialization_gap"] = round(
            specialization_b.specialization_score - specialization_a.specialization_score, 4
        )
    else:
        comparison["more_specialized"] = "none"
        comparison["specialization_gap"] = 0.0

    return comparison
