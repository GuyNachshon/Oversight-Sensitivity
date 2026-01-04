"""
Timeseries Analysis Module

Load and aggregate per-token statistics for time-resolved analysis.
Supports stratification by prompt family.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from ..experiments.run import ExecutionRun
from ..experiments.dataset import PromptDataset


def load_runs_by_context(
    experiment_dir: Path,
    contexts: List[str],
) -> Dict[str, List[ExecutionRun]]:
    """
    Load ExecutionRuns grouped by context.

    Args:
        experiment_dir: Directory containing run JSON files
        contexts: List of context conditions to load (e.g., ["N", "EO", "IO", "KW"])

    Returns:
        Dict mapping context -> list of ExecutionRuns
    """
    runs_by_context = {ctx: [] for ctx in contexts}

    for json_file in experiment_dir.glob("*.json"):
        if json_file.name == "checkpoint.json":
            continue

        try:
            run = ExecutionRun.from_json(json_file)
            if run.context_condition in contexts:
                runs_by_context[run.context_condition].append(run)
        except Exception as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")

    return runs_by_context


def extract_entropy_timeseries_from_run(
    run: ExecutionRun,
    layer_aggregation: str = "mean",
) -> Dict[int, float]:
    """
    Extract logit_entropy by token position from a single run.

    Args:
        run: ExecutionRun with statistics_by_layer
        layer_aggregation: How to aggregate across layers ("mean", "first", "last")

    Returns:
        Dict mapping token_position -> entropy value
    """
    entropy_by_token = defaultdict(list)

    for layer_key, stats_list in run.statistics_by_layer.items():
        for stat in stats_list:
            token_pos = stat.get("token_position", 0)
            entropy = stat.get("logit_entropy")
            if entropy is not None:
                entropy_by_token[token_pos].append(entropy)

    # Aggregate across layers
    result = {}
    for token_pos, values in entropy_by_token.items():
        if layer_aggregation == "mean":
            result[token_pos] = np.mean(values)
        elif layer_aggregation == "first":
            result[token_pos] = values[0]
        elif layer_aggregation == "last":
            result[token_pos] = values[-1]
        else:
            result[token_pos] = np.mean(values)

    return result


def load_entropy_timeseries(
    experiment_dir: Path,
    contexts: List[str],
    layer_aggregation: str = "mean",
) -> Dict[str, Dict[int, List[float]]]:
    """
    Load logit_entropy by (context, token_position) aggregated across prompts.

    Args:
        experiment_dir: Directory containing run JSON files
        contexts: List of context conditions to load
        layer_aggregation: How to aggregate across layers

    Returns:
        {context: {token_pos: [entropy_values across prompts]}}
    """
    runs_by_context = load_runs_by_context(experiment_dir, contexts)

    result = {}
    for ctx, runs in runs_by_context.items():
        token_data = defaultdict(list)

        for run in runs:
            entropy_series = extract_entropy_timeseries_from_run(run, layer_aggregation)
            for token_pos, entropy in entropy_series.items():
                token_data[token_pos].append(entropy)

        result[ctx] = dict(token_data)

    return result


def aggregate_timeseries(
    data: Dict[str, Dict[int, List[float]]],
    ci_level: float = 0.95,
    ci_type: str = "sem",
) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    """
    Aggregate to mean, lower CI, and upper CI per token position.

    Args:
        data: {context: {token_pos: [values]}}
        ci_level: Confidence level (default 0.95)
        ci_type: "sem" for standard error of mean, "percentile" for percentile-based CI

    Returns:
        {context: (means, ci_lower, ci_upper)} where each is a list indexed by token position
    """
    from scipy import stats as scipy_stats

    result = {}

    for ctx, token_data in data.items():
        if not token_data:
            continue

        max_token = max(token_data.keys())
        means = []
        ci_lower = []
        ci_upper = []

        for t in range(max_token + 1):
            values = token_data.get(t, [])
            if values:
                mean = np.mean(values)

                if ci_type == "sem":
                    # Use SEM-based CI (normal approximation)
                    sem = np.std(values, ddof=1) / np.sqrt(len(values))
                    z = scipy_stats.norm.ppf(1 - (1 - ci_level) / 2)
                    lower = mean - z * sem
                    upper = mean + z * sem
                else:
                    # Use percentile-based CI
                    alpha = (1 - ci_level) / 2
                    lower = np.percentile(values, alpha * 100)
                    upper = np.percentile(values, (1 - alpha) * 100)

                means.append(mean)
                ci_lower.append(lower)
                ci_upper.append(upper)
            else:
                means.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)

        result[ctx] = (means, ci_lower, ci_upper)

    return result


def load_entropy_by_family(
    experiment_dir: Path,
    dataset: PromptDataset,
    contexts: List[str],
    layer_aggregation: str = "mean",
) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    """
    Group entropy timeseries by prompt family then context.

    Args:
        experiment_dir: Directory containing run JSON files
        dataset: PromptDataset with family labels
        contexts: List of context conditions to load
        layer_aggregation: How to aggregate across layers

    Returns:
        {family: {context: {token_pos: [values]}}}
    """
    # Build prompt_id -> family mapping
    prompt_to_family = {}
    for prompt in dataset.prompts:
        if hasattr(prompt, 'family') and prompt.family:
            prompt_to_family[prompt.prompt_id] = prompt.family

    runs_by_context = load_runs_by_context(experiment_dir, contexts)

    # Group by family -> context -> token
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ctx, runs in runs_by_context.items():
        for run in runs:
            family = prompt_to_family.get(run.prompt_id, "unknown")
            entropy_series = extract_entropy_timeseries_from_run(run, layer_aggregation)

            for token_pos, entropy in entropy_series.items():
                result[family][ctx][token_pos].append(entropy)

    # Convert to regular dicts
    return {
        family: {
            ctx: dict(token_data)
            for ctx, token_data in contexts_data.items()
        }
        for family, contexts_data in result.items()
    }


def compute_early_late_split(
    data: Dict[str, Dict[int, List[float]]],
    split_point: int = 32,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean entropy for early vs late tokens.

    Args:
        data: {context: {token_pos: [values]}}
        split_point: Token position to split at (default 32)

    Returns:
        {context: (early_mean, late_mean)}
    """
    result = {}

    for ctx, token_data in data.items():
        early_values = []
        late_values = []

        for token_pos, values in token_data.items():
            if token_pos < split_point:
                early_values.extend(values)
            else:
                late_values.extend(values)

        early_mean = np.mean(early_values) if early_values else np.nan
        late_mean = np.mean(late_values) if late_values else np.nan

        result[ctx] = (early_mean, late_mean)

    return result


def save_timeseries_data(
    data: Dict[str, Dict[int, List[float]]],
    output_path: Path,
) -> None:
    """Save timeseries data to JSON."""
    # Convert to serializable format (int keys to strings)
    serializable = {
        ctx: {str(k): v for k, v in token_data.items()}
        for ctx, token_data in data.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
