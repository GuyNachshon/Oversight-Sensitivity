"""
Execution Run

Records a single prompt execution with environment metadata for reproducibility.
Per data-model.md Entity 3: ExecutionRun
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional
import json
import uuid
from pathlib import Path


Status = Literal["success", "failed", "timeout"]
ContextCondition = Literal["N", "EO", "IO", "KW", "REPRIME"]


@dataclass
class ExecutionRun:
    """
    Record of single prompt execution under one context condition.

    Per constitution: pytorch_version, transformers_version logged for reproducibility.
    Per data-model.md: status determines required fields (error_message if failed).
    """

    run_id: str
    experiment_id: str
    prompt_id: str
    context_condition: ContextCondition
    generated_text: str
    num_tokens_generated: int
    inference_time_seconds: float
    pytorch_version: str
    transformers_version: str
    cuda_version: Optional[str]
    gpu_model: Optional[str]
    timestamp: datetime
    status: Status
    error_message: Optional[str] = None
    statistics_by_layer: Dict[str, List[dict]] = field(default_factory=dict)
    # For IO context: which variant was used (0-2)
    io_variant_index: Optional[int] = None
    # For REPRIME: phase information (1=stage1, 2=stage2)
    reprime_phase: Optional[int] = None
    reprime_transition_token: Optional[int] = None

    @staticmethod
    def generate_id() -> str:
        """Generate unique run ID."""
        return str(uuid.uuid4())

    def __post_init__(self):
        """Validate per data-model.md rules."""
        if self.status == "success" and not self.generated_text:
            raise ValueError("status=success requires non-empty generated_text")

        if self.status == "failed" and not self.error_message:
            raise ValueError("status=failed requires error_message")

        if self.num_tokens_generated < 0:
            raise ValueError(f"num_tokens_generated must be >= 0, got {self.num_tokens_generated}")

    @classmethod
    def from_json(cls, path: Path) -> "ExecutionRun":
        """Load execution run from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)

    def to_json(self, path: Path) -> None:
        """
        Save execution run to JSON file.

        Per data-model.md storage: results/raw/{experiment_id}/{run_id}.json
        """
        data = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "prompt_id": self.prompt_id,
            "context_condition": self.context_condition,
            "generated_text": self.generated_text,
            "num_tokens_generated": self.num_tokens_generated,
            "inference_time_seconds": self.inference_time_seconds,
            "pytorch_version": self.pytorch_version,
            "transformers_version": self.transformers_version,
            "cuda_version": self.cuda_version,
            "gpu_model": self.gpu_model,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "error_message": self.error_message,
            "statistics_by_layer": self.statistics_by_layer,
            "io_variant_index": self.io_variant_index,
            "reprime_phase": self.reprime_phase,
            "reprime_transition_token": self.reprime_transition_token,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def add_statistics(self, layer_index: int, stats: List[dict]) -> None:
        """
        Add per-token statistics for a layer.

        Args:
            layer_index: Layer index (0 to num_layers-1)
            stats: List of statistic dicts, one per token
        """
        self.statistics_by_layer[str(layer_index)] = stats

    def get_statistics(self, layer_index: int) -> Optional[List[dict]]:
        """Retrieve statistics for a specific layer."""
        return self.statistics_by_layer.get(str(layer_index))
