"""
Experiment Configuration

Versioned configuration specifying all parameters for reproducible experiment runs.
Per data-model.md Entity 1: ExperimentConfig
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional
import json
from pathlib import Path


ContextCondition = Literal["N", "A", "ARD"]
ModelSizeCategory = Literal["1-3B", "7-8B"]


@dataclass
class GenerationConfig:
    """HuggingFace GenerationConfig parameters for reproducible generation."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    do_sample: bool = False
    use_cache: bool = False

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }


@dataclass
class ExperimentConfig:
    """
    Experiment configuration with all parameters for reproducible runs.

    Per constitution: random_seed must be documented (Reproducibility principle).
    Per data-model.md: experiment_id is unique identifier.
    """

    experiment_id: str
    model_identifier: str
    model_size_category: ModelSizeCategory
    random_seed: int
    layer_indices: List[int]  # Exactly 3 layers (early, mid, late)
    context_conditions: List[ContextCondition]
    prompt_dataset_path: str
    generation_config: GenerationConfig
    output_directory: str
    created_at: datetime
    created_by: Optional[str] = None

    def __post_init__(self):
        """Validate configuration per data-model.md validation rules."""
        if len(self.layer_indices) != 3:
            raise ValueError(
                f"layer_indices must have exactly 3 values, got {len(self.layer_indices)}"
            )

        if len(set(self.layer_indices)) != 3:
            raise ValueError("layer_indices must contain 3 unique values")

        if "N" not in self.context_conditions or "A" not in self.context_conditions:
            raise ValueError("context_conditions must contain at least N and A for comparison")

        if self.random_seed < 0:
            raise ValueError(f"random_seed must be >= 0, got {self.random_seed}")

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Parse datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Parse GenerationConfig
        gen_config_dict = data.pop("generation_config")
        data["generation_config"] = GenerationConfig(**gen_config_dict)

        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file for version control."""
        data = {
            "experiment_id": self.experiment_id,
            "model_identifier": self.model_identifier,
            "model_size_category": self.model_size_category,
            "random_seed": self.random_seed,
            "layer_indices": self.layer_indices,
            "context_conditions": self.context_conditions,
            "prompt_dataset_path": self.prompt_dataset_path,
            "generation_config": self.generation_config.to_dict(),
            "output_directory": self.output_directory,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
