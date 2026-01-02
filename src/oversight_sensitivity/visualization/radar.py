"""
Radar Plot Generator (T047)

Create radar plots showing context deformation across metrics.

Per research.md RQ4: Context deformation across CCI/EHL/TP with error bars,
colorblind palette.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from pathlib import Path

from .config import VisualizationConfig, DEFAULT_VIZ_CONFIG


def create_radar_plot(
    metrics_by_context: Dict[str, Dict[str, float]],
    metric_names: List[str] = ["CCI", "EHL", "TP", "OSS"],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Oversight Sensitivity Across Metrics",
) -> plt.Figure:
    """
    Create radar plot comparing contexts across metrics.

    Args:
        metrics_by_context: Dict[context -> Dict[metric -> value]]
        metric_names: List of metric names to plot
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title

    Returns:
        Matplotlib figure object

    Example:
        >>> metrics = {
        ...     "N": {"CCI": 0.0, "EHL": 10.0, "TP": 0.5, "OSS": 0.0},
        ...     "A": {"CCI": 0.8, "EHL": 5.0, "TP": 0.7, "OSS": 1.5},
        ... }
        >>> fig = create_radar_plot(metrics, output_path="radar.png")
    """
    # Number of variables
    num_vars = len(metric_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(
        figsize=config.get_figure_size(),
        subplot_kw=dict(projection="polar"),
        dpi=config.dpi,
    )

    # Plot each context
    for context, metrics in metrics_by_context.items():
        # Extract values in order
        values = [metrics.get(metric, 0.0) for metric in metric_names]
        values += values[:1]  # Complete the circle

        # Get color
        color = config.get_context_color(context)

        # Plot
        ax.plot(angles, values, "o-", linewidth=2, label=context, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=config.label_fontsize)

    # Add grid
    ax.grid(True, alpha=config.grid_alpha)

    # Add legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=config.legend_fontsize,
    )

    # Add title
    plt.title(title, fontsize=config.title_fontsize, pad=20)

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
