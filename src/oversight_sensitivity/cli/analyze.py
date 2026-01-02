"""
CLI Command: analyze (T046)

Compute statistical analysis (bootstrap CIs, effect sizes, power).
"""

import argparse
import json
from pathlib import Path
from typing import List

from ..metrics.results import MetricResults
from ..analysis.bootstrap import bootstrap_metric_ci
from ..analysis.effect_sizes import compute_effect_size_analysis
from ..analysis.power import estimate_power_via_subsampling, required_sample_size


def run_analyze(args):
    """Execute analyze command."""
    parser = argparse.ArgumentParser(
        description="Analyze metrics with bootstrap CIs and effect sizes",
        prog="oversee analyze",
    )

    parser.add_argument(
        "--metrics", type=str, required=True, help="Path to metrics JSON file"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Path to save analysis results JSON"
    )

    parser.add_argument(
        "--context-pair",
        type=str,
        default="A_vs_N",
        help="Context pair to analyze (default: A_vs_N)",
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--power-analysis",
        action="store_true",
        help="Include power analysis (requires paired data)",
    )

    parsed_args = parser.parse_args(args)

    # Load metrics
    print(f"Loading metrics from {parsed_args.metrics}...")
    with open(parsed_args.metrics, "r") as f:
        metrics_data = json.load(f)

    # Filter by context pair
    filtered_metrics = [
        MetricResults.from_dict(m)
        for m in metrics_data
        if m.get("context_pair") == parsed_args.context_pair
    ]

    print(f"Found {len(filtered_metrics)} results for context pair '{parsed_args.context_pair}'")

    if not filtered_metrics:
        print("No metrics found for specified context pair!")
        return 1

    # Extract metric values
    cci_values = [m.cci for m in filtered_metrics if m.cci is not None]
    ehl_values = [m.ehl for m in filtered_metrics if m.ehl is not None]
    tp_values = [m.tp_score for m in filtered_metrics if m.tp_score is not None]
    oss_values = [m.oss for m in filtered_metrics if m.oss is not None]

    print(f"\nMetric availability:")
    print(f"  CCI: {len(cci_values)}/{len(filtered_metrics)}")
    print(f"  EHL: {len(ehl_values)}/{len(filtered_metrics)}")
    print(f"  TP:  {len(tp_values)}/{len(filtered_metrics)}")
    print(f"  OSS: {len(oss_values)}/{len(filtered_metrics)}")

    # Compute bootstrap CIs
    print(f"\nComputing bootstrap confidence intervals ({parsed_args.bootstrap_iterations} iterations)...")

    analysis = {
        "context_pair": parsed_args.context_pair,
        "n_prompts": len(filtered_metrics),
        "confidence_level": parsed_args.confidence_level,
        "n_bootstrap": parsed_args.bootstrap_iterations,
        "random_seed": parsed_args.random_seed,
        "metrics": {
            "cci": bootstrap_metric_ci(
                cci_values,
                n_bootstrap=parsed_args.bootstrap_iterations,
                confidence_level=parsed_args.confidence_level,
                random_seed=parsed_args.random_seed,
            ),
            "ehl": bootstrap_metric_ci(
                ehl_values,
                n_bootstrap=parsed_args.bootstrap_iterations,
                confidence_level=parsed_args.confidence_level,
                random_seed=parsed_args.random_seed + 1 if parsed_args.random_seed else None,
            ),
            "tp": bootstrap_metric_ci(
                tp_values,
                n_bootstrap=parsed_args.bootstrap_iterations,
                confidence_level=parsed_args.confidence_level,
                random_seed=parsed_args.random_seed + 2 if parsed_args.random_seed else None,
            ),
            "oss": bootstrap_metric_ci(
                oss_values,
                n_bootstrap=parsed_args.bootstrap_iterations,
                confidence_level=parsed_args.confidence_level,
                random_seed=parsed_args.random_seed + 3 if parsed_args.random_seed else None,
            ),
        },
    }

    # Print results
    print("\n" + "=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)

    for metric_name, metric_data in analysis["metrics"].items():
        if metric_data["n"] > 0:
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {metric_data['mean']:.4f}")
            print(
                f"  {int(parsed_args.confidence_level * 100)}% CI: [{metric_data['ci_lower']:.4f}, {metric_data['ci_upper']:.4f}]"
            )
            print(f"  Std: {metric_data['std']:.4f}")
            print(f"  N: {metric_data['n']}")

    # Save analysis
    output_path = Path(parsed_args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n\nAnalysis saved to: {output_path}")

    return 0
