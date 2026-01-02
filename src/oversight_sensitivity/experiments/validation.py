"""
Config Validation (T013)

Validates ExperimentConfig against constitution requirements.
"""

from pathlib import Path
from typing import List, Tuple

from .config import ExperimentConfig


class ValidationError(Exception):
    """Raised when config validation fails."""

    pass


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    Validate ExperimentConfig against constitution requirements.

    Returns list of validation warnings (non-fatal issues).
    Raises ValidationError for fatal issues.
    """
    warnings = []

    # CONSTITUTION REQUIREMENT: Random seed must be documented
    if config.random_seed is None:
        raise ValidationError("random_seed must be set (constitution requirement)")

    # CONSTITUTION REQUIREMENT: At least N and A contexts
    if "N" not in config.context_conditions:
        raise ValidationError("context_conditions must include 'N' (neutral)")
    if "A" not in config.context_conditions:
        raise ValidationError("context_conditions must include 'A' (audited)")

    # Layer indices: exactly 3 layers (early, middle, late)
    if len(config.layer_indices) != 3:
        raise ValidationError(f"layer_indices must have exactly 3 values, got {len(config.layer_indices)}")

    # Check layer indices are non-negative
    if any(idx < 0 for idx in config.layer_indices):
        raise ValidationError("layer_indices must be non-negative")

    # Prompt dataset path exists
    dataset_path = Path(config.prompt_dataset_path)
    if not dataset_path.exists():
        raise ValidationError(f"prompt_dataset_path does not exist: {config.prompt_dataset_path}")

    # Output directory is writable
    output_dir = Path(config.output_directory)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValidationError(f"output_directory not writable: {e}")

    # WARNINGS (non-fatal)
    if config.generation_config.temperature > 0:
        warnings.append(
            "temperature > 0 may reduce reproducibility (constitution principle I)"
        )

    if config.generation_config.do_sample:
        warnings.append(
            "do_sample=True may reduce reproducibility even with fixed seed"
        )

    if config.generation_config.use_cache:
        warnings.append("use_cache=True may interfere with hook-based statistics collection")

    return warnings


def validate_layer_indices_for_model(
    layer_indices: List[int], num_model_layers: int
) -> None:
    """
    Validate layer indices against actual model architecture.

    Should be called after model is loaded.
    """
    if max(layer_indices) >= num_model_layers:
        raise ValidationError(
            f"layer_indices contains {max(layer_indices)} but model only has {num_model_layers} layers"
        )

    # Check distribution: early, middle, late
    early_cutoff = num_model_layers // 3
    late_cutoff = 2 * num_model_layers // 3

    has_early = any(idx < early_cutoff for idx in layer_indices)
    has_middle = any(early_cutoff <= idx < late_cutoff for idx in layer_indices)
    has_late = any(idx >= late_cutoff for idx in layer_indices)

    if not (has_early and has_middle and has_late):
        # This is a warning, not an error
        pass  # Could return warning if needed
