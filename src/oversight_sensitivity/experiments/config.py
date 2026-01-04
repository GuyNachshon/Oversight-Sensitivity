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


ContextCondition = Literal["N", "EO", "IO", "KW", "REPRIME"]
ModelSizeCategory = Literal["1-3B", "7-8B", "small"]


@dataclass
class ReprimeConfig:
    """Configuration for two-stage re-prime generation."""

    stage1_tokens: int = 32
    stage1_context: str = "N"
    stage2_context: str = "EO"

    def to_dict(self) -> dict:
        return {
            "stage1_tokens": self.stage1_tokens,
            "stage1_context": self.stage1_context,
            "stage2_context": self.stage2_context,
        }


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
    reprime_config: Optional[ReprimeConfig] = None

    def __post_init__(self):
        """Validate configuration per data-model.md validation rules."""
        if len(self.layer_indices) != 3:
            raise ValueError(
                f"layer_indices must have exactly 3 values, got {len(self.layer_indices)}"
            )

        if len(set(self.layer_indices)) != 3:
            raise ValueError("layer_indices must contain 3 unique values")

        # Must have N for baseline comparison
        if "N" not in self.context_conditions:
            raise ValueError("context_conditions must contain N for baseline comparison")

        # Must have at least one oversight condition (EO, IO, or REPRIME)
        oversight_conditions = {"EO", "IO", "REPRIME"}
        if not any(c in oversight_conditions for c in self.context_conditions):
            raise ValueError("context_conditions must contain at least one of EO, IO, or REPRIME")

        if self.random_seed < 0:
            raise ValueError(f"random_seed must be >= 0, got {self.random_seed}")

        # Ensure reprime_config exists if REPRIME is in conditions
        if "REPRIME" in self.context_conditions and self.reprime_config is None:
            self.reprime_config = ReprimeConfig()

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

        # Parse ReprimeConfig if present
        if "reprime_config" in data and data["reprime_config"]:
            reprime_config_dict = data.pop("reprime_config")
            data["reprime_config"] = ReprimeConfig(**reprime_config_dict)

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
            "reprime_config": self.reprime_config.to_dict() if self.reprime_config else None,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
