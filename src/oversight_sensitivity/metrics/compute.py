"""
Metric Computation Orchestrator

Loads ExecutionRuns and computes all metrics (CCI, EHL, TP, OSS).
Per contracts/cli-commands.md: oversee compute-metrics
"""

from pathlib import Path
from typing import List, Dict, Optional
import json

from ..experiments.run import ExecutionRun
from ..metrics.results import MetricResults
from ..metrics.cci import compute_cci
from ..metrics.ehl import compute_ehl
from ..metrics.tp import compute_tp
from ..metrics.oss import compute_oss


def compute_metrics_for_prompt(
    prompt_id: str,
    experiment_id: str,
    runs_neutral: List[ExecutionRun],
    runs_audited: List[ExecutionRun],
    context_pair: str = "A_vs_N",
) -> MetricResults:
    """
    Compute all metrics for a single prompt across contexts.

    Args:
        prompt_id: Prompt identifier
        experiment_id: Experiment identifier
        runs_neutral: ExecutionRuns for neutral context
        runs_audited: ExecutionRuns for audited context
        context_pair: Context comparison identifier

    Returns:
        MetricResults with computed values
    """
    # Extract statistics from runs
    stats_neutral = [run.statistics_by_layer for run in runs_neutral]
    stats_audited = [run.statistics_by_layer for run in runs_audited]

    # Compute CCI
    try:
        cci_value = compute_cci(stats_neutral, stats_audited)
    except Exception as e:
        print(f"Warning: CCI computation failed for {prompt_id}: {e}")
        cci_value = None

    # Compute EHL for each context
    try:
        ehl_neutral = compute_ehl(stats_neutral[0]) if stats_neutral else None
        ehl_audited = compute_ehl(stats_audited[0]) if stats_audited else None
    except Exception as e:
        print(f"Warning: EHL computation failed for {prompt_id}: {e}")
        ehl_neutral = ehl_audited = None

    # Compute TP for each context
    try:
        tp_neutral = compute_tp(stats_neutral[0]) if stats_neutral else None
        tp_audited = compute_tp(stats_audited[0]) if stats_audited else None
    except Exception as e:
        print(f"Warning: TP computation failed for {prompt_id}: {e}")
        tp_neutral = tp_audited = None

    # Compute OSS
    try:
        oss_value = compute_oss(
            cci_neutral=0.0,  # CCI is already a delta, use placeholder
            cci_audited=cci_value if cci_value else 0.0,
            ehl_neutral=ehl_neutral if ehl_neutral else 0.0,
            ehl_audited=ehl_audited if ehl_audited else 0.0,
            tp_neutral=tp_neutral if tp_neutral else 0.0,
            tp_audited=tp_audited if tp_audited else 0.0,
        )
    except Exception as e:
        print(f"Warning: OSS computation failed for {prompt_id}: {e}")
        oss_value = None

    return MetricResults(
        metric_id=MetricResults.generate_id(),
        experiment_id=experiment_id,
        prompt_id=prompt_id,
        context_pair=context_pair,
        cci=cci_value,
        ehl=ehl_audited,  # Report audited EHL
        tp_score=tp_audited,  # Report audited TP
        oss=oss_value,
    )


def load_runs_from_directory(experiment_dir: Path) -> List[ExecutionRun]:
    """
    Load all ExecutionRun JSON files from a directory.

    Args:
        experiment_dir: Directory containing run JSON files

    Returns:
        List of ExecutionRun objects
    """
    runs = []

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    for json_file in experiment_dir.glob("*.json"):
        try:
            run = ExecutionRun.from_json(json_file)
            runs.append(run)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return runs


def group_runs_by_prompt(runs: List[ExecutionRun]) -> Dict[str, Dict[str, List[ExecutionRun]]]:
    """
    Group runs by prompt_id and context_condition.

    Returns:
        Dict[prompt_id -> Dict[context -> List[runs]]]
    """
    grouped = {}

    for run in runs:
        if run.prompt_id not in grouped:
            grouped[run.prompt_id] = {}

        if run.context_condition not in grouped[run.prompt_id]:
            grouped[run.prompt_id][run.context_condition] = []

        grouped[run.prompt_id][run.context_condition].append(run)

    return grouped
