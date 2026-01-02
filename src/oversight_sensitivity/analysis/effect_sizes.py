"""
Effect Size Calculation (T043)

Compute Cohen's d and hypothesis tests for metric differences.

Per research.md: Use scipy.stats for hypothesis testing with p-values.
"""

import numpy as np
from scipy import stats
from typing import List, Optional, Tuple


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d (standardized mean difference)

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    if not group1 or not group2:
        return np.nan

    group1_array = np.array(group1)
    group2_array = np.array(group2)

    # Remove NaN values
    group1_array = group1_array[~np.isnan(group1_array)]
    group2_array = group2_array[~np.isnan(group2_array)]

    if len(group1_array) == 0 or len(group2_array) == 0:
        return np.nan

    # Means
    mean1 = np.mean(group1_array)
    mean2 = np.mean(group2_array)

    # Pooled standard deviation
    n1 = len(group1_array)
    n2 = len(group2_array)
    var1 = np.var(group1_array, ddof=1)
    var2 = np.var(group2_array, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    return float(d)


def paired_t_test(
    group1: List[float], group2: List[float]
) -> Tuple[float, float]:
    """
    Perform paired t-test.

    Args:
        group1: First group of paired values
        group2: Second group of paired values

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if not group1 or not group2:
        return np.nan, np.nan

    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired t-test")

    group1_array = np.array(group1)
    group2_array = np.array(group2)

    # Remove pairs with NaN
    mask = ~(np.isnan(group1_array) | np.isnan(group2_array))
    group1_clean = group1_array[mask]
    group2_clean = group2_array[mask]

    if len(group1_clean) < 2:
        return np.nan, np.nan

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(group1_clean, group2_clean)

    return float(t_stat), float(p_value)


def independent_t_test(
    group1: List[float], group2: List[float]
) -> Tuple[float, float]:
    """
    Perform independent samples t-test.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if not group1 or not group2:
        return np.nan, np.nan

    group1_array = np.array(group1)
    group2_array = np.array(group2)

    # Remove NaN values
    group1_clean = group1_array[~np.isnan(group1_array)]
    group2_clean = group2_array[~np.isnan(group2_array)]

    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan, np.nan

    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(group1_clean, group2_clean)

    return float(t_stat), float(p_value)


def compute_effect_size_analysis(
    neutral_values: List[float],
    audited_values: List[float],
    paired: bool = True,
) -> dict:
    """
    Compute comprehensive effect size analysis.

    Args:
        neutral_values: Metric values for neutral context
        audited_values: Metric values for audited context
        paired: Whether data is paired (same prompts in both contexts)

    Returns:
        Dict with effect size, test statistics, and interpretation
    """
    # Cohen's d
    d = cohens_d(audited_values, neutral_values)

    # Interpret effect size
    if np.isnan(d):
        interpretation = "undefined"
    elif abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    # Hypothesis test
    if paired:
        t_stat, p_value = paired_t_test(audited_values, neutral_values)
        test_type = "paired_t_test"
    else:
        t_stat, p_value = independent_t_test(audited_values, neutral_values)
        test_type = "independent_t_test"

    return {
        "cohens_d": d,
        "effect_size_interpretation": interpretation,
        "test_type": test_type,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05 if not np.isnan(p_value) else False,
        "significant_at_0.01": p_value < 0.01 if not np.isnan(p_value) else False,
        "n_neutral": len([v for v in neutral_values if not np.isnan(v)]),
        "n_audited": len([v for v in audited_values if not np.isnan(v)]),
    }
