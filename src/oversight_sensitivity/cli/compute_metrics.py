"""
CLI Command: compute-metrics

Compute CCI, EHL, TP, OSS metrics from execution runs.
Per contracts/cli-commands.md Command 3.
"""

import argparse
from pathlib import Path
import json

from ..metrics.compute import (
    load_runs_from_directory,
    group_runs_by_prompt,
    compute_metrics_for_prompt,
)
from ..metrics.results import MetricResults


def run_compute_metrics(args):
    """Execute compute-metrics command."""
    parser = argparse.ArgumentParser(
        description="Compute metrics from execution runs",
        prog="oversee compute-metrics",
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing ExecutionRun JSON files",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Where to write MetricResults JSON"
    )

    parser.add_argument(
        "--context-pairs",
        type=str,
        required=True,
        help="Which comparisons to compute (e.g., A_vs_N,ARD_vs_A)",
    )

    parser.add_argument(
        "--skip-baselines", action="store_true", help="Skip keyword and output-only baselines"
    )

    parser.add_argument("--verbose", action="store_true", help="Detailed metric computation logs")

    parsed_args = parser.parse_args(args)

    # Load runs
    experiment_dir = Path(parsed_args.experiment_dir)
    print(f"Loading execution runs from {experiment_dir}...")

    runs = load_runs_from_directory(experiment_dir)
    print(f"Loaded {len(runs)} runs")

    # Group by prompt and context
    grouped = group_runs_by_prompt(runs)
    print(f"Found {len(grouped)} unique prompts")

    # Parse context pairs
    context_pairs = [pair.strip() for pair in parsed_args.context_pairs.split(",")]

    # Compute metrics for each prompt
    all_results = []

    for prompt_id, context_runs in grouped.items():
        if parsed_args.verbose:
            print(f"Processing prompt {prompt_id}...")

        for context_pair in context_pairs:
            # Parse context pair (e.g., "A_vs_N" -> contexts A and N)
            contexts = context_pair.split("_vs_")
            if len(contexts) != 2:
                print(f"Warning: Invalid context pair format: {context_pair}")
                continue

            context_a, context_n = contexts

            # Get runs for each context
            runs_a = context_runs.get(context_a, [])
            runs_n = context_runs.get(context_n, [])

            if not runs_a or not runs_n:
                print(
                    f"Warning: Missing runs for {prompt_id} "
                    f"({context_a}: {len(runs_a)}, {context_n}: {len(runs_n)})"
                )
                continue

            # Compute metrics
            try:
                result = compute_metrics_for_prompt(
                    prompt_id=prompt_id,
                    experiment_id=runs_a[0].experiment_id,
                    runs_neutral=runs_n,
                    runs_audited=runs_a,
                    context_pair=context_pair,
                )

                all_results.append(result)

                if parsed_args.verbose:
                    print(f"  {context_pair}: CCI={result.cci:.3f}, EHL={result.ehl:.1f}")

            except Exception as e:
                print(f"Error computing metrics for {prompt_id} {context_pair}: {e}")

    # Summary statistics
    if all_results:
        import numpy as np

        ccis = [r.cci for r in all_results if r.cci is not None]
        ehls = [r.ehl for r in all_results if r.ehl is not None]
        tps = [r.tp_score for r in all_results if r.tp_score is not None]
        osss = [r.oss for r in all_results if r.oss is not None]

        print(f"\nComputing metrics for {len(all_results)} prompts across {len(context_pairs)} context pairs...")
        if ccis:
            print(f"CCI: mean={np.mean(ccis):.3f} (n={len(ccis)})")
        if ehls:
            print(f"EHL: mean={np.mean(ehls):.1f} tokens (n={len(ehls)})")
        if tps:
            print(f"TP:  mean={np.mean(tps):.3f} (n={len(tps)})")
        if osss:
            print(f"OSS: mean={np.mean(osss):.3f} (n={len(osss)})")

    # Save results
    output_path = Path(parsed_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    print(f"\nWritten to: {output_path}")
