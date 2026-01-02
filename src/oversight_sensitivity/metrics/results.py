"""
Metric Results

Computed metrics derived from ExecutionStatistics.
Per data-model.md Entity 5: MetricResults
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json
import uuid
from pathlib import Path


@dataclass
class MetricResults:
    """
    Computed metrics for a prompt across context conditions.

    Per data-model.md formulas:
    - CCI (Computation Compression Index): 1 - (variance_A / variance_N)
    - EHL (Exploration Half-Life): token position where entropy drops to 50% of initial
    - TP (Trajectory Predictability): R² of linear fit Δh_{t+1} ≈ W Δh_t
    - OSS (Oversight Sensitivity Score): (ΔCCI + ΔEHL + ΔTP) / 3
    """

    metric_id: str
    experiment_id: str
    prompt_id: str
    context_pair: str  # Format: "A_vs_N", "ARD_vs_A", etc.
    cci: Optional[float] = None
    ehl: Optional[float] = None
    tp_score: Optional[float] = None
    oss: Optional[float] = None
    baseline_keyword_score: Optional[float] = None
    baseline_output_entropy: Optional[float] = None
    computed_at: datetime = None

    def __post_init__(self):
        """Validate per data-model.md rules."""
        # At least one metric must be non-null
        if all(
            x is None
            for x in [self.cci, self.ehl, self.tp_score, self.oss]
        ):
            raise ValueError("At least one of {cci, ehl, tp_score, oss} must be non-null")

        # Validate ranges
        if self.tp_score is not None and not (0 <= self.tp_score <= 1):
            raise ValueError(f"tp_score must be in [0, 1], got {self.tp_score}")

        if self.ehl is not None and self.ehl <= 0:
            raise ValueError(f"ehl must be > 0, got {self.ehl}")

        if self.computed_at is None:
            self.computed_at = datetime.now()

    @staticmethod
    def generate_id() -> str:
        """Generate unique metric ID."""
        return str(uuid.uuid4())

    @classmethod
    def from_json(cls, path: Path) -> "MetricResults":
        """Load metric results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Handle list of results
            return [cls.from_dict(item) for item in data]

        data["computed_at"] = datetime.fromisoformat(data["computed_at"])
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "MetricResults":
        """Create from dictionary."""
        data = data.copy()
        if "computed_at" in data and isinstance(data["computed_at"], str):
            data["computed_at"] = datetime.fromisoformat(data["computed_at"])
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_id": self.metric_id,
            "experiment_id": self.experiment_id,
            "prompt_id": self.prompt_id,
            "context_pair": self.context_pair,
            "cci": self.cci,
            "ehl": self.ehl,
            "tp_score": self.tp_score,
            "oss": self.oss,
            "baseline_keyword_score": self.baseline_keyword_score,
            "baseline_output_entropy": self.baseline_output_entropy,
            "computed_at": self.computed_at.isoformat(),
        }

    def to_json(self, path: Path) -> None:
        """
        Save metric results to JSON file.

        Per data-model.md storage: results/metrics/{experiment_id}_metrics.json
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
