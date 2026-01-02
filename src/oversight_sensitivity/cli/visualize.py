"""
CLI Visualize Command (T051)

Generate publication-ready visualizations from computed metrics and analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..visualization.radar import create_radar_plot
from ..visualization.heatmap import create_metric_heatmap
from ..visualization.timeseries import create_entropy_timeseries
from ..visualization.intervention import create_intervention_comparison
from ..visualization.config import VisualizationConfig, DEFAULT_VIZ_CONFIG


def load_metrics_json(path: Path) -> Dict:
    """Load metrics JSON file."""
    with open(path) as f:
        return json.load(f)


def load_analysis_json(path: Path) -> Dict:
    """Load analysis JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_metrics_by_context(metrics_data: Dict) -> Dict[str, Dict[str, float]]:
    """
    Extract context-level metrics for radar plots.

    Args:
        metrics_data: Full metrics JSON structure

    Returns:
        Dict[context -> Dict[metric -> value]]

    Example:
        {
            "N": {"CCI": 0.0, "EHL": 10.0, "TP": 0.5, "OSS": 0.0},
            "A": {"CCI": 0.8, "EHL": 5.0, "TP": 0.7, "OSS": 1.5},
        }
    """
    # Group by context across all runs
    context_metrics = {}

    for run_id, run_data in metrics_data.items():
        context = run_data.get("context", "N")

        if context not in context_metrics:
            context_metrics[context] = {
                "CCI": [],
                "EHL": [],
                "TP": [],
                "OSS": [],
            }

        # Extract metric values
        context_metrics[context]["CCI"].append(run_data.get("cci", 0.0))
        context_metrics[context]["EHL"].append(run_data.get("ehl", 0.0))
        context_metrics[context]["TP"].append(run_data.get("tp", 0.0))
        context_metrics[context]["OSS"].append(run_data.get("oss", 0.0))

    # Average across runs for each context
    averaged_metrics = {}
    for context, metrics in context_metrics.items():
        averaged_metrics[context] = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in metrics.items()
        }

    return averaged_metrics


def extract_metrics_with_ci(analysis_data: Dict) -> Dict[str, Dict[str, tuple]]:
    """
    Extract metrics with confidence intervals for intervention plots.

    Args:
        analysis_data: Full analysis JSON structure

    Returns:
        Dict[context -> Dict[metric -> (mean, lower_ci, upper_ci)]]
    """
    metrics_with_ci = {}

    # Expected structure: analysis_data["metrics"][metric_name]
    if "metrics" in analysis_data:
        for metric_name, metric_data in analysis_data["metrics"].items():
            # Extract by context if available
            if "by_context" in metric_data:
                for context, context_data in metric_data["by_context"].items():
                    if context not in metrics_with_ci:
                        metrics_with_ci[context] = {}

                    mean = context_data.get("mean", 0.0)
                    ci = context_data.get("ci_95", [mean, mean])

                    metrics_with_ci[context][metric_name.upper()] = (
                        mean,
                        ci[0],
                        ci[1],
                    )

    return metrics_with_ci


def extract_entropy_timeseries(metrics_data: Dict) -> Dict[str, List[float]]:
    """
    Extract entropy timeseries data for each context.

    Args:
        metrics_data: Full metrics JSON structure

    Returns:
        Dict[context -> list of entropy values over time]
    """
    entropy_by_context = {}

    # For now, return empty - requires raw statistics data
    # This would need to load from statistics files, not metrics
    return entropy_by_context


def generate_radar_plot(
    metrics_data: Dict,
    output_dir: Path,
    config: VisualizationConfig,
) -> None:
    """Generate radar plot comparing contexts across metrics."""
    metrics_by_context = extract_metrics_by_context(metrics_data)

    if not metrics_by_context:
        print("Warning: No metrics data available for radar plot")
        return

    output_path = output_dir / "radar_plot"
    fig = create_radar_plot(
        metrics_by_context=metrics_by_context,
        output_path=output_path,
        config=config,
    )

    print(f"  - Radar plot saved to {output_dir}/radar_plot.[png|pdf]")


def generate_intervention_plot(
    analysis_data: Dict,
    output_dir: Path,
    config: VisualizationConfig,
) -> None:
    """Generate intervention comparison plot with confidence intervals."""
    metrics_with_ci = extract_metrics_with_ci(analysis_data)

    if not metrics_with_ci:
        print("Warning: No analysis data available for intervention plot")
        return

    output_path = output_dir / "intervention_comparison"
    fig = create_intervention_comparison(
        metrics_with_ci=metrics_with_ci,
        output_path=output_path,
        config=config,
    )

    print(f"  - Intervention plot saved to {output_dir}/intervention_comparison.[png|pdf]")


def generate_timeseries_plot(
    metrics_data: Dict,
    output_dir: Path,
    config: VisualizationConfig,
) -> None:
    """Generate entropy timeseries plot."""
    entropy_by_context = extract_entropy_timeseries(metrics_data)

    if not entropy_by_context:
        print("Warning: Timeseries plot requires raw statistics data (not yet implemented)")
        return

    output_path = output_dir / "entropy_timeseries"
    fig = create_entropy_timeseries(
        entropy_by_context=entropy_by_context,
        output_path=output_path,
        config=config,
    )

    print(f"  - Timeseries plot saved to {output_dir}/entropy_timeseries.[png|pdf]")


def generate_heatmap(
    metrics_data: Dict,
    output_dir: Path,
    config: VisualizationConfig,
) -> None:
    """Generate heatmap of layer × context metrics."""
    # Heatmap requires layer-level data, which needs statistics files
    print("Warning: Heatmap generation requires layer-level statistics (not yet implemented)")


def run_visualize(argv: List[str]) -> None:
    """
    Main entry point for visualize command.

    Usage:
        oversee visualize --metrics metrics.json --analysis analysis.json \\
                         --output-dir plots/ --plot-type all
    """
    parser = argparse.ArgumentParser(
        description="Generate visualizations from metrics and analysis"
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics JSON file",
    )

    parser.add_argument(
        "--analysis",
        type=Path,
        help="Path to analysis JSON file (optional, for CI plots)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save visualization outputs",
    )

    parser.add_argument(
        "--plot-type",
        choices=["radar", "intervention", "timeseries", "heatmap", "all"],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs (default: 300)",
    )

    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF output (PNG only)",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    if not args.metrics.exists():
        print(f"Error: Metrics file not found: {args.metrics}", file=sys.stderr)
        sys.exit(1)

    if args.analysis and not args.analysis.exists():
        print(f"Error: Analysis file not found: {args.analysis}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading metrics from {args.metrics}...")
    metrics_data = load_metrics_json(args.metrics)

    analysis_data = None
    if args.analysis:
        print(f"Loading analysis from {args.analysis}...")
        analysis_data = load_analysis_json(args.analysis)

    # Configure visualization
    config = VisualizationConfig(
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
    )

    # Generate requested plots
    print(f"\nGenerating visualizations in {args.output_dir}/...")

    if args.plot_type in ["radar", "all"]:
        generate_radar_plot(metrics_data, args.output_dir, config)

    if args.plot_type in ["intervention", "all"]:
        if analysis_data:
            generate_intervention_plot(analysis_data, args.output_dir, config)
        else:
            print("Warning: Skipping intervention plot (requires --analysis)")

    if args.plot_type in ["timeseries", "all"]:
        generate_timeseries_plot(metrics_data, args.output_dir, config)

    if args.plot_type in ["heatmap", "all"]:
        generate_heatmap(metrics_data, args.output_dir, config)

    print("\nVisualization complete!")
