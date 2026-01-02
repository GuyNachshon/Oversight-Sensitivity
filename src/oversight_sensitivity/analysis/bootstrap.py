"""
Bootstrap Confidence Intervals (T042)

Compute bootstrap CIs for metric uncertainty quantification.

Per research.md RQ2: Use bootstrap resampling with numpy.random.choice,
compute percentile CIs (e.g., 95% = 2.5th and 97.5th percentiles).
"""

import numpy as np
from typing import List, Tuple, Optional


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    statistic_fn=np.mean,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples (default: 10000)
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        statistic_fn: Function to compute statistic (default: np.mean)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci)

    Example:
        >>> data = [0.5, 0.6, 0.7, 0.8, 0.9]
        >>> mean, lower, upper = bootstrap_ci(data)
        >>> print(f"Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    if not data:
        raise ValueError("Cannot compute CI for empty data")

    data_array = np.array(data)

    # Set random seed for reproducibility (constitution requirement)
    if random_seed is not None:
        np.random.seed(random_seed)

    # Compute point estimate
    point_estimate = statistic_fn(data_array)

    # Bootstrap resampling
    bootstrap_estimates = []
    n = len(data_array)

    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data_array, size=n, replace=True)
        bootstrap_estimates.append(statistic_fn(resample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute percentile CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_ci = np.percentile(bootstrap_estimates, lower_percentile)
    upper_ci = np.percentile(bootstrap_estimates, upper_percentile)

    return float(point_estimate), float(lower_ci), float(upper_ci)


def bootstrap_metric_ci(
    metric_values: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Compute bootstrap CI for a metric with additional statistics.

    Args:
        metric_values: List of metric values across prompts
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level
        random_seed: Random seed for reproducibility

    Returns:
        Dict with point estimate, CI bounds, and metadata
    """
    if not metric_values:
        return {
            "n": 0,
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "std": None,
            "confidence_level": confidence_level,
        }

    # Remove None values
    valid_values = [v for v in metric_values if v is not None]

    if not valid_values:
        return {
            "n": 0,
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "std": None,
            "confidence_level": confidence_level,
        }

    # Compute bootstrap CI
    mean, lower, upper = bootstrap_ci(
        valid_values,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        statistic_fn=np.mean,
        random_seed=random_seed,
    )

    return {
        "n": len(valid_values),
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "std": float(np.std(valid_values)),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
    }
