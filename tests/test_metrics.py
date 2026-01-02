"""
Unit Tests for Metrics (T031)

Test CCI, EHL, TP, OSS metric formulas against known values.
"""

import pytest
import numpy as np
from oversight_sensitivity.metrics.cci import compute_cci
from oversight_sensitivity.metrics.ehl import compute_ehl
from oversight_sensitivity.metrics.tp import compute_tp
from oversight_sensitivity.metrics.oss import compute_oss


class TestCCI:
    """Test Computation Compression Index."""

    def _make_stats_by_layer(self, activation_variance: float, num_tokens: int = 10) -> dict:
        """Helper to create statistics_by_layer dict."""
        stats = {}
        for layer_idx in [0, 11, 21]:
            stats[str(layer_idx)] = [
                {
                    "run_id": "test",
                    "layer_index": layer_idx,
                    "token_position": i,
                    "logit_entropy": 1.0,
                    "top1_probability": 0.5,
                    "residual_stream_norm": 1.0,
                    "activation_variance": activation_variance,
                    "attention_entropy": 1.0,
                    "cosine_similarity_to_prev": 0.0 if i > 0 else None,
                }
                for i in range(num_tokens)
            ]
        return stats

    def test_cci_equal_variance(self):
        """CCI should be 0 when variances are equal."""
        stats_n = [self._make_stats_by_layer(activation_variance=2.0)]
        stats_a = [self._make_stats_by_layer(activation_variance=2.0)]

        cci = compute_cci(stats_n, stats_a)
        assert abs(cci - 0.0) < 1e-6, f"Expected CCI ≈ 0, got {cci}"

    def test_cci_reduced_variance(self):
        """CCI should be positive when audited variance is lower."""
        stats_n = [self._make_stats_by_layer(activation_variance=4.0)]
        stats_a = [self._make_stats_by_layer(activation_variance=2.0)]

        cci = compute_cci(stats_n, stats_a)
        # CCI = 1 - (2.0 / 4.0) = 1 - 0.5 = 0.5
        assert abs(cci - 0.5) < 1e-6, f"Expected CCI ≈ 0.5, got {cci}"

    def test_cci_increased_variance(self):
        """CCI should be negative when audited variance is higher."""
        stats_n = [self._make_stats_by_layer(activation_variance=2.0)]
        stats_a = [self._make_stats_by_layer(activation_variance=4.0)]

        cci = compute_cci(stats_n, stats_a)
        # CCI = 1 - (4.0 / 2.0) = 1 - 2.0 = -1.0
        assert abs(cci - (-1.0)) < 1e-6, f"Expected CCI ≈ -1.0, got {cci}"


class TestEHL:
    """Test Exploration Half-Life."""

    def _make_stats_by_layer(self, entropies: list) -> dict:
        """Helper to create statistics_by_layer dict with varying entropies."""
        stats = {}
        stats["0"] = [
            {
                "run_id": "test",
                "layer_index": 0,
                "token_position": i,
                "logit_entropy": entropies[i],
                "top1_probability": 0.5,
                "residual_stream_norm": 1.0,
                "activation_variance": 1.0,
                "attention_entropy": 1.0,
                "cosine_similarity_to_prev": 0.0 if i > 0 else None,
            }
            for i in range(len(entropies))
        ]
        return stats

    def test_ehl_immediate_decay(self):
        """EHL should be small when entropy decays quickly."""
        # Start high, drop to half by token 2
        entropies = [4.0, 3.0, 2.0, 1.5, 1.0]  # Drops to 2.0 (half of 4.0) at index 2
        stats = self._make_stats_by_layer(entropies)

        ehl = compute_ehl(stats)
        # Should be around 2.0 (where entropy hits half of initial)
        assert 1.5 <= ehl <= 2.5, f"Expected EHL ≈ 2.0, got {ehl}"

    def test_ehl_slow_decay(self):
        """EHL should be large when entropy decays slowly."""
        # Slow decay
        entropies = [4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0]
        stats = self._make_stats_by_layer(entropies)

        ehl = compute_ehl(stats)
        # Should be around 10.0
        assert 9.0 <= ehl <= 11.0, f"Expected EHL ≈ 10.0, got {ehl}"

    def test_ehl_no_decay(self):
        """EHL should be len(sequence) when entropy never reaches half."""
        # No decay
        entropies = [4.0, 4.0, 4.0, 4.0, 4.0]
        stats = self._make_stats_by_layer(entropies)

        ehl = compute_ehl(stats)
        # When entropy never drops to 50%, returns len(entropies) as upper bound
        assert ehl == len(entropies), f"Expected EHL = {len(entropies)}, got {ehl}"


class TestTP:
    """Test Trajectory Predictability."""

    def _make_stats_by_layer(self, residual_norms: list) -> dict:
        """Helper to create statistics_by_layer dict with varying norms."""
        stats = {}
        stats["0"] = [
            {
                "run_id": "test",
                "layer_index": 0,
                "token_position": i,
                "logit_entropy": 1.0,
                "top1_probability": 0.5,
                "residual_stream_norm": residual_norms[i],
                "activation_variance": 1.0,
                "attention_entropy": 1.0,
                "cosine_similarity_to_prev": 1.0 if i > 0 else None,
            }
            for i in range(len(residual_norms))
        ]
        return stats

    def test_tp_linear_trajectory(self):
        """TP should be high for linear trajectory."""
        # Create linearly increasing norms
        norms = [float(i) for i in range(10)]
        stats = self._make_stats_by_layer(norms)

        tp = compute_tp(stats)
        # Linear trajectory should have high R²
        assert tp >= 0.9, f"Expected TP >= 0.9 for linear, got {tp}"

    def test_tp_random_trajectory(self):
        """TP should be lower for random trajectory."""
        # Random norms
        np.random.seed(42)
        norms = np.random.uniform(0, 10, 10).tolist()
        stats = self._make_stats_by_layer(norms)

        tp = compute_tp(stats)
        assert 0.0 <= tp <= 1.0, f"TP should be in [0, 1], got {tp}"


class TestOSS:
    """Test Oversight Sensitivity Score."""

    def test_oss_all_positive(self):
        """OSS should average all metric deltas."""
        cci_neutral = 0.0
        cci_audited = 0.3
        ehl_neutral = 10.0
        ehl_audited = 15.0
        tp_neutral = 0.5
        tp_audited = 0.7

        oss = compute_oss(cci_neutral, cci_audited, ehl_neutral, ehl_audited, tp_neutral, tp_audited)

        # OSS = (ΔCCI + ΔEHL + ΔTP) / 3
        # ΔCCI = 0.3 - 0.0 = 0.3
        # ΔEHL = 15.0 - 10.0 = 5.0
        # ΔTP = 0.7 - 0.5 = 0.2
        # OSS = (0.3 + 5.0 + 0.2) / 3 = 1.833...
        expected = (0.3 + 5.0 + 0.2) / 3
        assert abs(oss - expected) < 1e-6, f"Expected OSS ≈ {expected}, got {oss}"

    def test_oss_mixed_changes(self):
        """OSS should handle mixed metric changes."""
        cci_neutral = 0.0
        cci_audited = 0.5
        ehl_neutral = 10.0
        ehl_audited = 8.0  # Decreased
        tp_neutral = 0.5
        tp_audited = 0.6

        oss = compute_oss(cci_neutral, cci_audited, ehl_neutral, ehl_audited, tp_neutral, tp_audited)

        # ΔCCI = 0.5, ΔEHL = -2.0, ΔTP = 0.1
        expected = (0.5 + (-2.0) + 0.1) / 3
        assert abs(oss - expected) < 1e-6, f"Expected OSS ≈ {expected}, got {oss}"

    def test_oss_with_none(self):
        """OSS should return None if any metric is None."""
        cci_neutral = 0.0
        cci_audited = 0.3
        ehl_neutral = None  # EHL may not be computable
        ehl_audited = None
        tp_neutral = 0.5
        tp_audited = 0.7

        oss = compute_oss(cci_neutral, cci_audited, ehl_neutral, ehl_audited, tp_neutral, tp_audited)

        # If any metric is None, OSS returns None
        assert oss is None, f"Expected OSS = None when metrics missing, got {oss}"
