"""
Trajectory Predictability (TP)

Measures predictability of internal dynamics using linear model.

Per data-model.md formula:
Fit: Δh_{t+1} ≈ W Δh_t (linear regression)
TP = R² score of fit

Interpretation:
- Higher R² → more predictable dynamics
- Lower R² → more chaotic/exploratory dynamics
- TP ∈ [0, 1]
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression


def compute_tp(statistics_by_layer: Dict[str, List[dict]]) -> Optional[float]:
    """
    Compute Trajectory Predictability.

    Args:
        statistics_by_layer: Dict mapping layer index → list of stat dicts

    Returns:
        TP score (R²) ∈ [0, 1], or None if cannot be computed

    Per data-model.md: Fit linear predictor on hidden-state deltas
    """
    # Extract hidden state proxies (use residual stream norms as proxy)
    # In full implementation, would use actual hidden states
    # Here we use residual_stream_norm as a 1D proxy
    hidden_states = _extract_hidden_state_proxies(statistics_by_layer)

    if len(hidden_states) < 3:
        return None  # Need at least 3 tokens to compute deltas and fit

    # Compute deltas: Δh_t = h_t - h_{t-1}
    deltas = np.diff(hidden_states, axis=0)

    if len(deltas) < 2:
        return None

    # Prepare training data for linear regression
    # X: Δh_t (deltas at time t)
    # y: Δh_{t+1} (deltas at time t+1)
    X = deltas[:-1].reshape(-1, 1)  # Δh_t for t=0..T-2
    y = deltas[1:].reshape(-1, 1)  # Δh_{t+1} for t=1..T-1

    try:
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Compute R² score
        r2 = model.score(X, y)

        # Ensure R² is in [0, 1]
        # In rare cases with very bad fit, R² can be negative
        r2 = max(0.0, min(1.0, r2))

        return float(r2)

    except Exception:
        # If fitting fails, return None
        return None


def _extract_hidden_state_proxies(statistics_by_layer: Dict[str, List[dict]]) -> np.ndarray:
    """
    Extract hidden state proxies from statistics.

    Uses residual_stream_norm as a 1D proxy for hidden states.
    In full implementation with access to model internals, would use actual hidden states.

    Args:
        statistics_by_layer: Dict mapping layer index → list of stat dicts

    Returns:
        Array of shape (num_tokens,) with residual stream norms
    """
    if not statistics_by_layer:
        return np.array([])

    # Use middle layer for trajectory analysis
    layer_keys = sorted(statistics_by_layer.keys())
    if len(layer_keys) == 0:
        return np.array([])

    # Use middle layer (index 1 if we have 3 layers)
    middle_idx = len(layer_keys) // 2
    layer_key = layer_keys[middle_idx]
    stats_list = statistics_by_layer[layer_key]

    # Extract residual stream norms as proxy
    norms = []
    for stat_dict in sorted(stats_list, key=lambda x: x.get("token_position", 0)):
        if "residual_stream_norm" in stat_dict:
            norms.append(stat_dict["residual_stream_norm"])

    return np.array(norms)
