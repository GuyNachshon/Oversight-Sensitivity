"""
Power Analysis (T044)

Estimate statistical power via subsampling at different sample sizes.

Per research.md: Subsample prompts at N=20,40,60, estimate detection
probability via bootstrap to determine required sample size.
"""

import numpy as np
from typing import List, Tuple, Optional
from .effect_sizes import compute_effect_size_analysis


def estimate_power_via_subsampling(
    neutral_values: List[float],
    audited_values: List[float],
    sample_sizes: List[int] = [20, 40, 60],
    n_iterations: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Estimate statistical power at different sample sizes via subsampling.

    Args:
        neutral_values: Full dataset of neutral metric values
        audited_values: Full dataset of audited metric values
        sample_sizes: Sample sizes to test (default: [20, 40, 60])
        n_iterations: Number of subsampling iterations per size
        alpha: Significance level (default: 0.05)
        random_seed: Random seed for reproducibility

    Returns:
        Dict with power estimates for each sample size
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if len(neutral_values) != len(audited_values):
        raise ValueError("Neutral and audited must have same length (paired data)")

    n_available = len(neutral_values)

    results = {}

    for n in sample_sizes:
        if n > n_available:
            results[f"n_{n}"] = {
                "sample_size": n,
                "power": None,
                "detections": 0,
                "iterations": 0,
                "note": f"Not enough data (need {n}, have {n_available})",
            }
            continue

        detections = 0  # Count how many times p < alpha

        for _ in range(n_iterations):
            # Randomly subsample n pairs
            indices = np.random.choice(n_available, size=n, replace=False)
            subsample_neutral = [neutral_values[i] for i in indices]
            subsample_audited = [audited_values[i] for i in indices]

            # Run hypothesis test
            analysis = compute_effect_size_analysis(
                subsample_neutral,
                subsample_audited,
                paired=True,
            )

            # Check if significant
            if analysis["p_value"] < alpha:
                detections += 1

        # Power = proportion of detections
        power = detections / n_iterations

        results[f"n_{n}"] = {
            "sample_size": n,
            "power": power,
            "detections": detections,
            "iterations": n_iterations,
            "alpha": alpha,
        }

    # Add summary
    results["summary"] = {
        "available_samples": n_available,
        "alpha": alpha,
        "n_iterations": n_iterations,
        "sample_sizes_tested": sample_sizes,
    }

    return results


def required_sample_size(
    power_results: dict,
    target_power: float = 0.80,
) -> Optional[int]:
    """
    Estimate required sample size to achieve target power.

    Args:
        power_results: Results from estimate_power_via_subsampling
        target_power: Target power (default: 0.80 for 80%)

    Returns:
        Estimated sample size needed, or None if target not achievable
    """
    # Extract power estimates
    sample_sizes = []
    powers = []

    for key, value in power_results.items():
        if key.startswith("n_") and value.get("power") is not None:
            sample_sizes.append(value["sample_size"])
            powers.append(value["power"])

    if not sample_sizes:
        return None

    # Sort by sample size
    sorted_pairs = sorted(zip(sample_sizes, powers))
    sample_sizes, powers = zip(*sorted_pairs)

    # Find first sample size where power >= target
    for n, power in zip(sample_sizes, powers):
        if power >= target_power:
            return n

    # If target not reached, return None (need more data)
    return None
