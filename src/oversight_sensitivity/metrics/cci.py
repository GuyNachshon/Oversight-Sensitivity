"""
Computation Compression Index (CCI)

Measures relative reduction in internal variability under context.

Per data-model.md formula:
CCI = 1 - (variance_A / variance_N)

where variance is mean activation_variance across tokens and layers.

Interpretation:
- Higher CCI → tighter, more constrained computation under audited context
- CCI > 0: Compression detected (A has lower variance than N)
- CCI < 0: Expansion (A has higher variance than N)
- CCI = 0: No difference
"""

import numpy as np
from typing import List, Dict


def compute_cci(
    stats_neutral: List[Dict[str, List[dict]]],
    stats_audited: List[Dict[str, List[dict]]],
) -> float:
    """
    Compute Computation Compression Index.

    Args:
        stats_neutral: Statistics from neutral context (list of layer dicts)
        stats_audited: Statistics from audited context (list of layer dicts)

    Returns:
        CCI value (dimensionless)

    Per data-model.md: CCI = 1 - (variance_A / variance_N)
    """
    # Extract activation variances from both contexts
    variances_n = _extract_activation_variances(stats_neutral)
    variances_a = _extract_activation_variances(stats_audited)

    if len(variances_n) == 0 or len(variances_a) == 0:
        raise ValueError("Cannot compute CCI with empty statistics")

    # Compute mean variance for each context
    mean_var_n = np.mean(variances_n)
    mean_var_a = np.mean(variances_a)

    if mean_var_n == 0:
        raise ValueError("Mean variance in neutral context is zero, cannot compute CCI")

    # CCI formula
    cci = 1.0 - (mean_var_a / mean_var_n)

    return float(cci)


def _extract_activation_variances(statistics_by_layer: List[Dict[str, List[dict]]]) -> List[float]:
    """
    Extract all activation_variance values from statistics.

    Args:
        statistics_by_layer: Dict mapping layer index → list of stat dicts

    Returns:
        Flat list of all activation_variance values across layers and tokens
    """
    variances = []

    for layer_stats_dict in statistics_by_layer:
        # statistics_by_layer is a dict like {"0": [...], "12": [...], "24": [...]}
        for layer_idx, stats_list in layer_stats_dict.items():
            for stat_dict in stats_list:
                if "activation_variance" in stat_dict:
                    variances.append(stat_dict["activation_variance"])

    return variances
