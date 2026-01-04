"""
CLI Command: plot-timeseries

Generate timeseries plots of entropy over token position.
"""

import argparse
from pathlib import Path

from ..analysis.timeseries import (
    load_entropy_timeseries,
    aggregate_timeseries,
    load_entropy_by_family,
)
from ..experiments.dataset import PromptDataset
from ..visualization.timeseries import (
    create_entropy_timeseries_with_ci,
    create_family_comparison_plot,
)


def run_plot_timeseries(args):
    """Execute plot-timeseries command."""
    parser = argparse.ArgumentParser(
        description="Generate timeseries plots of entropy over token position",
        prog="oversee plot-timeseries",
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing ExecutionRun JSON files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save plots (default: results/plots)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset JSONL (required for --by-family)",
    )

    parser.add_argument(
        "--contexts",
        type=str,
        default="N,EO,IO,KW",
        help="Comma-separated list of contexts to plot (default: N,EO,IO,KW)",
    )

    parser.add_argument(
        "--by-family",
        action="store_true",
        help="Also generate family-stratified plots",
    )

    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Hide confidence bands",
    )

    parsed_args = parser.parse_args(args)

    experiment_dir = Path(parsed_args.experiment_dir)
    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = [c.strip() for c in parsed_args.contexts.split(",")]

    print(f"Loading entropy timeseries from {experiment_dir}...")
    print(f"Contexts: {contexts}")

    # Load raw data
    raw_data = load_entropy_timeseries(experiment_dir, contexts)

    # Check data
    for ctx, token_data in raw_data.items():
        n_tokens = len(token_data)
        n_samples = len(next(iter(token_data.values()))) if token_data else 0
        print(f"  {ctx}: {n_tokens} token positions, {n_samples} prompts")

    # Aggregate to mean and CI
    print("\nAggregating to mean with 95% CI...")
    aggregated = aggregate_timeseries(raw_data)

    # Create main plot (Figure 1)
    print("\nGenerating main entropy timeseries plot...")
    output_path = output_dir / "entropy_timeseries"

    fig = create_entropy_timeseries_with_ci(
        aggregated,
        output_path=output_path,
        title="Logit Entropy Over Token Position",
        show_ci=not parsed_args.no_ci,
    )

    print(f"Saved: {output_path}.png")

    # Family stratification (Figure 2)
    if parsed_args.by_family:
        if not parsed_args.dataset:
            print("\nWarning: --dataset required for --by-family, skipping family plots")
            return 0

        print("\nLoading dataset for family stratification...")
        dataset = PromptDataset.from_jsonl(Path(parsed_args.dataset))

        print("Loading entropy by family...")
        family_data = load_entropy_by_family(experiment_dir, dataset, contexts)

        # Aggregate each family
        aggregated_by_family = {}
        for family, ctx_data in family_data.items():
            aggregated_by_family[family] = aggregate_timeseries(ctx_data)
            n_prompts = len(next(iter(next(iter(ctx_data.values())).values()))) if ctx_data else 0
            print(f"  {family}: {n_prompts} prompts")

        # Create family comparison plot
        print("\nGenerating family comparison plot...")
        family_output = output_dir / "entropy_by_family"

        families = ["reasoning", "judgment", "stance"]  # Explicit order
        available_families = [f for f in families if f in aggregated_by_family]

        fig2 = create_family_comparison_plot(
            aggregated_by_family,
            output_path=family_output,
            families=available_families,
        )

        print(f"Saved: {family_output}.png")

    print("\nDone!")
    return 0
