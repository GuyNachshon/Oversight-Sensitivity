"""
Intervention Plot Generator (T050)

Create intervention comparison plots (ARD vs A vs N) with confidence regions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from .config import VisualizationConfig, DEFAULT_VIZ_CONFIG


def create_intervention_comparison(
    metrics_with_ci: Dict[str, Dict[str, tuple]],
    metric_names: List[str] = ["CCI", "EHL", "TP", "OSS"],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Intervention Analysis: ARD vs A vs N",
) -> plt.Figure:
    """
    Create bar plot comparing metrics across contexts with confidence intervals.

    Args:
        metrics_with_ci: Dict[context -> Dict[metric -> (mean, lower_ci, upper_ci)]]
        metric_names: List of metric names to plot
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title

    Returns:
        Matplotlib figure object

    Example:
        >>> metrics = {
        ...     "N": {"CCI": (0.0, -0.1, 0.1), "EHL": (10.0, 9.0, 11.0)},
        ...     "A": {"CCI": (0.8, 0.7, 0.9), "EHL": (5.0, 4.5, 5.5)},
        ...     "ARD": {"CCI": (0.4, 0.3, 0.5), "EHL": (7.0, 6.5, 7.5)},
        ... }
        >>> fig = create_intervention_comparison(metrics)
    """
    # Number of metrics and contexts
    contexts = list(metrics_with_ci.keys())
    n_metrics = len(metric_names)
    n_contexts = len(contexts)

    # Create figure with subplots
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(config.figure_width * n_metrics / 2, config.figure_height),
        dpi=config.dpi,
    )

    # Ensure axes is a list
    if n_metrics == 1:
        axes = [axes]

    # Bar width
    bar_width = 0.25
    x_positions = np.arange(n_contexts)

    # Plot each metric
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        # Extract data for this metric
        means = []
        lower_errors = []
        upper_errors = []

        for context in contexts:
            if metric_name in metrics_with_ci[context]:
                mean, lower_ci, upper_ci = metrics_with_ci[context][metric_name]
                means.append(mean)
                lower_errors.append(mean - lower_ci)
                upper_errors.append(upper_ci - mean)
            else:
                means.append(0)
                lower_errors.append(0)
                upper_errors.append(0)

        # Create bar plot
        colors = [config.get_context_color(ctx) for ctx in contexts]

        bars = ax.bar(
            x_positions,
            means,
            bar_width * 2,
            yerr=[lower_errors, upper_errors],
            capsize=config.error_bar_capsize,
            alpha=config.error_bar_alpha,
            color=colors,
            edgecolor="black",
            linewidth=1,
        )

        # Labels
        ax.set_xlabel("Context", fontsize=config.label_fontsize)
        ax.set_ylabel(metric_name, fontsize=config.label_fontsize)
        ax.set_title(metric_name, fontsize=config.title_fontsize)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(contexts, fontsize=config.tick_fontsize)

        # Apply styling
        config.apply_style(ax)

    # Overall title
    fig.suptitle(title, fontsize=config.title_fontsize + 2, y=1.02)

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.save_png:
            png_path = output_path.with_suffix(".png")
            fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight")

        if config.save_pdf:
            pdf_path = output_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")

    return fig
