"""
Baseline Comparison Analysis (T057-T058)

Compare experimental contexts (N, A, ARD) against control baselines (KW, OO, R)
to validate that effects are due to evaluative oversight, not confounds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .effect_sizes import paired_t_test, cohens_d


@dataclass
class BaselineComparison:
    """
    Comparison between experimental context and baseline control.

    Attributes:
        experimental_context: The experimental condition (e.g., "A")
        baseline_context: The baseline control (e.g., "KW")
        metric_name: Metric being compared (e.g., "CCI")
        experimental_mean: Mean value in experimental condition
        baseline_mean: Mean value in baseline condition
        delta: Difference (experimental - baseline)
        cohens_d: Effect size
        p_value: Statistical significance (paired t-test)
        interpretation: Human-readable interpretation
    """

    experimental_context: str
    baseline_context: str
    metric_name: str
    experimental_mean: float
    baseline_mean: float
    delta: float
    cohens_d: float
    p_value: float
    interpretation: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experimental_context": self.experimental_context,
            "baseline_context": self.baseline_context,
            "metric_name": self.metric_name,
            "experimental_mean": self.experimental_mean,
            "baseline_mean": self.baseline_mean,
            "delta": self.delta,
            "cohens_d": self.cohens_d,
            "p_value": self.p_value,
            "interpretation": self.interpretation,
        }


def compare_to_baseline(
    experimental_values: List[float],
    baseline_values: List[float],
    experimental_context: str,
    baseline_context: str,
    metric_name: str,
    alpha: float = 0.05,
) -> BaselineComparison:
    """
    Compare experimental context against baseline control.

    Args:
        experimental_values: Metric values from experimental condition
        baseline_values: Metric values from baseline condition
        experimental_context: Name of experimental context (e.g., "A")
        baseline_context: Name of baseline (e.g., "KW")
        metric_name: Name of metric (e.g., "CCI")
        alpha: Significance level (default 0.05)

    Returns:
        BaselineComparison object with statistical results
    """
    exp_mean = float(np.mean(experimental_values))
    base_mean = float(np.mean(baseline_values))
    delta = exp_mean - base_mean

    # Effect size
    effect_size = cohens_d(experimental_values, baseline_values)

    # Statistical test
    stat_result = paired_t_test(
        experimental_values,
        baseline_values,
        alpha=alpha,
    )

    # Interpretation
    if stat_result["p_value"] < alpha:
        significance = "significant"
    else:
        significance = "not significant"

    interpretation = (
        f"{experimental_context} differs from {baseline_context} baseline "
        f"by {delta:.3f} ({significance}, p={stat_result['p_value']:.4f}, "
        f"d={effect_size:.3f})"
    )

    return BaselineComparison(
        experimental_context=experimental_context,
        baseline_context=baseline_context,
        metric_name=metric_name,
        experimental_mean=exp_mean,
        baseline_mean=base_mean,
        delta=delta,
        cohens_d=effect_size,
        p_value=stat_result["p_value"],
        interpretation=interpretation,
    )


def run_baseline_analysis(
    metrics_by_context: Dict[str, Dict[str, List[float]]],
    experimental_contexts: List[str] = ["A", "ARD"],
    baseline_contexts: List[str] = ["KW", "OO", "R"],
    metric_names: List[str] = ["CCI", "EHL", "TP", "OSS"],
) -> Dict[str, List[BaselineComparison]]:
    """
    Run comprehensive baseline comparison analysis.

    For each experimental context, compare against all baselines on all metrics.

    Args:
        metrics_by_context: Dict[context -> Dict[metric -> list of values]]
        experimental_contexts: Experimental conditions to test
        baseline_contexts: Baseline controls to compare against
        metric_names: Metrics to analyze

    Returns:
        Dict[metric_name -> List[BaselineComparison]]

    Example:
        >>> metrics = {
        ...     "N": {"CCI": [0.0, 0.05], "EHL": [10.0, 11.0]},
        ...     "A": {"CCI": [0.75, 0.80], "EHL": [5.0, 5.2]},
        ...     "KW": {"CCI": [0.10, 0.15], "EHL": [9.5, 10.0]},
        ... }
        >>> results = run_baseline_analysis(metrics)
    """
    results = {}

    for metric_name in metric_names:
        comparisons = []

        for exp_context in experimental_contexts:
            # Skip if experimental context not available
            if exp_context not in metrics_by_context:
                continue
            if metric_name not in metrics_by_context[exp_context]:
                continue

            exp_values = metrics_by_context[exp_context][metric_name]

            for baseline_context in baseline_contexts:
                # Skip if baseline not available
                if baseline_context not in metrics_by_context:
                    continue
                if metric_name not in metrics_by_context[baseline_context]:
                    continue

                baseline_values = metrics_by_context[baseline_context][metric_name]

                # Perform comparison
                comparison = compare_to_baseline(
                    experimental_values=exp_values,
                    baseline_values=baseline_values,
                    experimental_context=exp_context,
                    baseline_context=baseline_context,
                    metric_name=metric_name,
                )

                comparisons.append(comparison)

        results[metric_name] = comparisons

    return results


def validate_hypothesis(
    comparisons: List[BaselineComparison],
    hypothesis: str = "experimental > baseline",
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Validate hypothesis across baseline comparisons.

    Tests whether experimental contexts consistently differ from baselines
    in the predicted direction.

    Args:
        comparisons: List of baseline comparisons
        hypothesis: Expected direction ("experimental > baseline" or "experimental < baseline")
        alpha: Significance level

    Returns:
        Validation summary dict

    Example:
        >>> # Hypothesis: Audited (A) increases CCI relative to all baselines
        >>> cci_comparisons = results["CCI"]
        >>> a_vs_baselines = [c for c in cci_comparisons if c.experimental_context == "A"]
        >>> validation = validate_hypothesis(a_vs_baselines, "experimental > baseline")
    """
    total = len(comparisons)
    significant = sum(1 for c in comparisons if c.p_value < alpha)

    if hypothesis == "experimental > baseline":
        correct_direction = sum(1 for c in comparisons if c.delta > 0)
        significant_correct = sum(
            1 for c in comparisons if c.p_value < alpha and c.delta > 0
        )
    elif hypothesis == "experimental < baseline":
        correct_direction = sum(1 for c in comparisons if c.delta < 0)
        significant_correct = sum(
            1 for c in comparisons if c.p_value < alpha and c.delta < 0
        )
    else:
        raise ValueError(f"Unknown hypothesis: {hypothesis}")

    proportion_significant = significant / total if total > 0 else 0.0
    proportion_correct = correct_direction / total if total > 0 else 0.0

    return {
        "hypothesis": hypothesis,
        "total_comparisons": total,
        "significant": significant,
        "significant_correct_direction": significant_correct,
        "proportion_significant": proportion_significant,
        "proportion_correct_direction": proportion_correct,
        "validated": significant_correct >= (total / 2),  # Majority criterion
    }
