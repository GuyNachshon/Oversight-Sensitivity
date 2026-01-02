"""
Reversibility Analysis (T036)

Analyzes whether ARD intervention reverses oversight-induced deformation.

Per PRD Section 3.3 Research Questions:
- RQ3: Does explicit reasoning discipline (ARD) reverse compression effects?

Computes:
- Δ(A-N): Deformation under oversight
- Δ(ARD-A): Change when adding reasoning discipline
- Δ(ARD-N): Net effect of ARD vs baseline
- Reversibility percentage: How much of A-N deformation is reversed by ARD-A
"""

from typing import Optional, Dict
from ..metrics.results import MetricResults


class ReversibilityAnalysis:
    """Analyze reversibility of oversight-induced deformation."""

    def __init__(
        self,
        metrics_a_vs_n: MetricResults,
        metrics_ard_vs_a: MetricResults,
        metrics_ard_vs_n: Optional[MetricResults] = None,
    ):
        """
        Initialize reversibility analysis.

        Args:
            metrics_a_vs_n: Metrics for Audited vs Neutral
            metrics_ard_vs_a: Metrics for ARD vs Audited
            metrics_ard_vs_n: Optional metrics for ARD vs Neutral (can be computed)
        """
        self.metrics_a_vs_n = metrics_a_vs_n
        self.metrics_ard_vs_a = metrics_ard_vs_a
        self.metrics_ard_vs_n = metrics_ard_vs_n

    def compute_cci_reversibility(self) -> Optional[float]:
        """
        Compute CCI reversibility percentage.

        Returns:
            Percentage of A-N compression reversed by ARD-A (0-100%)
            None if metrics unavailable

        Formula:
            reversibility = (ARD-A) / (A-N) * 100

        Interpretation:
            100% = ARD fully reverses compression to neutral
            0% = ARD has no effect
            >100% = ARD overcorrects (expands beyond neutral)
            <0% = ARD amplifies compression
        """
        if self.metrics_a_vs_n.cci is None or self.metrics_ard_vs_a.cci is None:
            return None

        delta_a_n = self.metrics_a_vs_n.cci
        delta_ard_a = self.metrics_ard_vs_a.cci

        if abs(delta_a_n) < 1e-9:
            # No deformation to reverse
            return None

        # Reversibility = (change from ARD) / (original deformation) * 100
        # Negative sign because we want positive % when ARD moves toward neutral
        reversibility = -(delta_ard_a / delta_a_n) * 100.0

        return float(reversibility)

    def compute_ehl_reversibility(self) -> Optional[float]:
        """
        Compute EHL reversibility percentage.

        Returns:
            Percentage of A-N change reversed by ARD-A (0-100%)
        """
        if self.metrics_a_vs_n.ehl is None or self.metrics_ard_vs_a.ehl is None:
            return None

        # For EHL, we need the actual EHL values, not the deltas
        # The MetricResults stores EHL for the "audited" context
        # So we need to track baseline EHL separately

        # This is a simplified version - in practice, you'd store
        # ehl_neutral, ehl_audited, ehl_ard separately

        # Placeholder: return None for now
        # Full implementation requires tracking base metrics
        return None

    def compute_tp_reversibility(self) -> Optional[float]:
        """
        Compute TP reversibility percentage.

        Returns:
            Percentage of A-N change reversed by ARD-A (0-100%)
        """
        if self.metrics_a_vs_n.tp_score is None or self.metrics_ard_vs_a.tp_score is None:
            return None

        # Similar to EHL - needs base metrics
        return None

    def compute_oss_reversibility(self) -> Optional[float]:
        """
        Compute OSS reversibility percentage.

        Returns:
            Percentage of A-N change reversed by ARD-A (0-100%)
        """
        if self.metrics_a_vs_n.oss is None or self.metrics_ard_vs_a.oss is None:
            return None

        delta_a_n = self.metrics_a_vs_n.oss
        delta_ard_a = self.metrics_ard_vs_a.oss

        if abs(delta_a_n) < 1e-9:
            return None

        reversibility = -(delta_ard_a / delta_a_n) * 100.0

        return float(reversibility)

    def to_dict(self) -> Dict:
        """
        Export reversibility analysis to dictionary.

        Returns:
            Dict with all reversibility metrics
        """
        return {
            "cci_reversibility_percent": self.compute_cci_reversibility(),
            "ehl_reversibility_percent": self.compute_ehl_reversibility(),
            "tp_reversibility_percent": self.compute_tp_reversibility(),
            "oss_reversibility_percent": self.compute_oss_reversibility(),
            "metrics_a_vs_n": {
                "cci": self.metrics_a_vs_n.cci,
                "ehl": self.metrics_a_vs_n.ehl,
                "tp": self.metrics_a_vs_n.tp_score,
                "oss": self.metrics_a_vs_n.oss,
            },
            "metrics_ard_vs_a": {
                "cci": self.metrics_ard_vs_a.cci,
                "ehl": self.metrics_ard_vs_a.ehl,
                "tp": self.metrics_ard_vs_a.tp_score,
                "oss": self.metrics_ard_vs_a.oss,
            },
        }


def compute_reversibility(
    metrics_a_vs_n: MetricResults,
    metrics_ard_vs_a: MetricResults,
    metrics_ard_vs_n: Optional[MetricResults] = None,
) -> Dict:
    """
    Convenience function to compute reversibility analysis.

    Args:
        metrics_a_vs_n: Metrics for Audited vs Neutral
        metrics_ard_vs_a: Metrics for ARD vs Audited
        metrics_ard_vs_n: Optional metrics for ARD vs Neutral

    Returns:
        Dictionary with reversibility percentages
    """
    analysis = ReversibilityAnalysis(metrics_a_vs_n, metrics_ard_vs_a, metrics_ard_vs_n)
    return analysis.to_dict()
