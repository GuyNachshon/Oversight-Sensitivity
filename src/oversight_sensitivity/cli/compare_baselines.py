"""
CLI Baseline Comparison Command (T058)

Compare experimental contexts against baseline controls to validate
that effects are due to evaluative oversight, not confounds.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from ..analysis.baseline_comparison import (
    run_baseline_analysis,
    validate_hypothesis,
    BaselineComparison,
)
from ..analysis.report import generate_baseline_report


def load_metrics_json(path: Path) -> Dict:
    """Load metrics JSON file."""
    with open(path) as f:
        return json.load(f)


def group_metrics_by_context(metrics_data: Dict) -> Dict[str, Dict[str, List[float]]]:
    """
    Group metrics by context for baseline comparison.

    Args:
        metrics_data: Full metrics JSON structure

    Returns:
        Dict[context -> Dict[metric -> list of values]]
    """
    by_context = defaultdict(lambda: defaultdict(list))

    for run_id, run_data in metrics_data.items():
        context = run_data.get("context", "N")

        # Extract metrics
        if "cci" in run_data and run_data["cci"] is not None:
            by_context[context]["CCI"].append(run_data["cci"])
        if "ehl" in run_data and run_data["ehl"] is not None:
            by_context[context]["EHL"].append(run_data["ehl"])
        if "tp" in run_data and run_data["tp"] is not None:
            by_context[context]["TP"].append(run_data["tp"])
        if "oss" in run_data and run_data["oss"] is not None:
            by_context[context]["OSS"].append(run_data["oss"])

    return dict(by_context)


def print_comparison_summary(
    results: Dict[str, List[BaselineComparison]],
    verbose: bool = False,
) -> None:
    """
    Print human-readable summary of baseline comparisons.

    Args:
        results: Baseline comparison results
        verbose: Show all comparisons or just summary
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON ANALYSIS")
    print("=" * 80)

    for metric_name, comparisons in results.items():
        print(f"\n{metric_name}:")
        print("-" * 80)

        if not comparisons:
            print("  No comparisons available")
            continue

        # Group by experimental context
        by_exp = defaultdict(list)
        for comp in comparisons:
            by_exp[comp.experimental_context].append(comp)

        for exp_context, exp_comparisons in by_exp.items():
            print(f"\n  {exp_context} vs Baselines:")

            for comp in exp_comparisons:
                sig_marker = "***" if comp.p_value < 0.001 else (
                    "**" if comp.p_value < 0.01 else (
                        "*" if comp.p_value < 0.05 else ""
                    )
                )

                print(
                    f"    vs {comp.baseline_context}: "
                    f"Δ={comp.delta:+.3f} (d={comp.cohens_d:.3f}, "
                    f"p={comp.p_value:.4f}) {sig_marker}"
                )

                if verbose:
                    print(f"      {comp.interpretation}")


def print_hypothesis_validation(
    results: Dict[str, List[BaselineComparison]],
    experimental_contexts: List[str],
    hypothesis: str,
) -> None:
    """
    Print hypothesis validation summary.

    Args:
        results: Baseline comparison results
        experimental_contexts: Contexts to validate
        hypothesis: Expected direction
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS VALIDATION")
    print("=" * 80)
    print(f"Hypothesis: {hypothesis}")
    print()

    for metric_name, comparisons in results.items():
        print(f"\n{metric_name}:")

        for exp_context in experimental_contexts:
            # Filter to this experimental context
            exp_comparisons = [
                c for c in comparisons if c.experimental_context == exp_context
            ]

            if not exp_comparisons:
                continue

            validation = validate_hypothesis(exp_comparisons, hypothesis)

            status = "✓ VALIDATED" if validation["validated"] else "✗ NOT VALIDATED"
            print(f"  {exp_context}: {status}")
            print(f"    {validation['significant_correct_direction']}/{validation['total_comparisons']} significant in correct direction")
            print(f"    Proportion correct: {validation['proportion_correct_direction']:.1%}")


def run_compare_baselines(argv: List[str]) -> None:
    """
    Main entry point for baseline comparison command.

    Usage:
        oversee compare-baselines --metrics metrics.json \\
                                 --output baseline_comparison.json \\
                                 --experimental A ARD \\
                                 --baselines KW OO R
    """
    parser = argparse.ArgumentParser(
        description="Compare experimental contexts against baseline controls"
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics JSON file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save baseline comparison results (JSON)",
    )

    parser.add_argument(
        "--experimental",
        nargs="+",
        default=["A", "ARD"],
        help="Experimental contexts to test (default: A ARD)",
    )

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["KW", "OO", "R"],
        help="Baseline controls to compare against (default: KW OO R)",
    )

    parser.add_argument(
        "--metrics-list",
        nargs="+",
        default=["CCI", "EHL", "TP", "OSS"],
        help="Metrics to analyze (default: CCI EHL TP OSS)",
    )

    parser.add_argument(
        "--hypothesis",
        choices=["experimental > baseline", "experimental < baseline"],
        default="experimental > baseline",
        help="Expected direction of effect (default: experimental > baseline)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed interpretations",
    )

    parser.add_argument(
        "--report",
        type=Path,
        help="Path to save markdown report (optional)",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    if not args.metrics.exists():
        print(f"Error: Metrics file not found: {args.metrics}", file=sys.stderr)
        sys.exit(1)

    # Load metrics
    print(f"Loading metrics from {args.metrics}...")
    metrics_data = load_metrics_json(args.metrics)

    # Group by context
    metrics_by_context = group_metrics_by_context(metrics_data)

    # Check for baseline availability
    available_contexts = set(metrics_by_context.keys())
    missing_baselines = set(args.baselines) - available_contexts

    if missing_baselines:
        print(f"\nWarning: Missing baseline contexts: {missing_baselines}")
        print(f"Available contexts: {sorted(available_contexts)}")
        print()

    # Run baseline analysis
    print("Running baseline comparisons...")
    results = run_baseline_analysis(
        metrics_by_context=metrics_by_context,
        experimental_contexts=args.experimental,
        baseline_contexts=args.baselines,
        metric_names=args.metrics_list,
    )

    # Print summary
    print_comparison_summary(results, verbose=args.verbose)

    # Validate hypothesis
    print_hypothesis_validation(
        results,
        experimental_contexts=args.experimental,
        hypothesis=args.hypothesis,
    )

    # Save results
    print(f"\nSaving results to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experimental_contexts": args.experimental,
        "baseline_contexts": args.baselines,
        "metrics": args.metrics_list,
        "hypothesis": args.hypothesis,
        "alpha": args.alpha,
        "comparisons": {
            metric_name: [c.to_dict() for c in comparisons]
            for metric_name, comparisons in results.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Generate markdown report if requested
    if args.report:
        print(f"\nGenerating markdown report...")
        report = generate_baseline_report(
            comparison_results=results,
            experimental_contexts=args.experimental,
            baseline_contexts=args.baselines,
            output_path=args.report,
        )
        print(f"Report saved to: {args.report}")

    print(f"\nBaseline comparison complete!")
    print(f"Results saved to: {args.output}")
