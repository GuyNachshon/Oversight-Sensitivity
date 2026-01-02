"""
Exploration Half-Life (EHL)

Measures how quickly uncertainty (logit entropy) collapses.

Per data-model.md formula:
EHL = token position where logit_entropy drops to 50% of initial value

Computed via interpolation on entropy time series.

Interpretation:
- Lower EHL → earlier commitment / reduced exploration
- Higher EHL → sustained uncertainty / more exploration
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.interpolate import interp1d


def compute_ehl(statistics_by_layer: Dict[str, List[dict]]) -> Optional[float]:
    """
    Compute Exploration Half-Life.

    Args:
        statistics_by_layer: Dict mapping layer index → list of stat dicts

    Returns:
        EHL value in tokens, or None if cannot be computed

    Per data-model.md: Find token position where entropy drops to 50% of initial
    """
    # Extract logit entropy time series (use first layer for output entropy)
    # Typically we'd use the final layer's entropy for output distribution
    entropies = _extract_logit_entropies(statistics_by_layer)

    if len(entropies) < 2:
        return None  # Need at least 2 tokens to compute half-life

    # Initial entropy (first token)
    initial_entropy = entropies[0]

    if initial_entropy == 0:
        return None  # Cannot compute half-life if initial entropy is zero

    # Target: 50% of initial entropy
    target_entropy = initial_entropy * 0.5

    # Find token position where entropy drops below target
    token_positions = np.arange(len(entropies))

    # Check if entropy ever drops below target
    if min(entropies) > target_entropy:
        # Entropy never drops to 50% - return max position as upper bound
        return float(len(entropies))

    # Find first crossing using interpolation
    # If entropy is monotonically decreasing, find crossing point
    try:
        # Create interpolator (piecewise linear)
        f = interp1d(token_positions, entropies, kind="linear")

        # Binary search for crossing point
        for i in range(1, len(entropies)):
            if entropies[i] <= target_entropy:
                # Interpolate between i-1 and i
                if entropies[i - 1] != entropies[i]:
                    # Linear interpolation
                    alpha = (target_entropy - entropies[i - 1]) / (entropies[i] - entropies[i - 1])
                    ehl = (i - 1) + alpha
                else:
                    ehl = float(i)
                return float(ehl)

    except Exception:
        # Fallback: find first token below target
        for i, entropy in enumerate(entropies):
            if entropy <= target_entropy:
                return float(i)

    return None


def _extract_logit_entropies(statistics_by_layer: Dict[str, List[dict]]) -> List[float]:
    """
    Extract logit entropy time series from statistics.

    Uses the first available layer (typically layer 0 or final layer for output distribution).

    Args:
        statistics_by_layer: Dict mapping layer index → list of stat dicts

    Returns:
        List of logit_entropy values over token positions
    """
    # Get first layer's statistics
    if not statistics_by_layer:
        return []

    # Use first available layer
    layer_key = sorted(statistics_by_layer.keys())[0]
    stats_list = statistics_by_layer[layer_key]

    entropies = []
    for stat_dict in sorted(stats_list, key=lambda x: x.get("token_position", 0)):
        if "logit_entropy" in stat_dict:
            entropies.append(stat_dict["logit_entropy"])

    return entropies
