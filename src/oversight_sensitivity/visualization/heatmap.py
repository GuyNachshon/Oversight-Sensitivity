"""
Heatmap Generator (T048)

Create heatmaps showing layer × context metric values.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
from pathlib import Path

from .config import VisualizationConfig, DEFAULT_VIZ_CONFIG


def create_metric_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Metric Values Across Layers and Contexts",
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Create heatmap for metric values.

    Args:
        data: 2D array of metric values [rows × cols]
        row_labels: Labels for rows (e.g., layer indices)
        col_labels: Labels for columns (e.g., contexts)
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib figure object

    Example:
        >>> data = np.array([[0.5, 0.8, 0.6], [0.4, 0.9, 0.5]])
        >>> fig = create_metric_heatmap(
        ...     data,
        ...     row_labels=["Layer 0", "Layer 11"],
        ...     col_labels=["N", "A", "ARD"],
        ... )
    """
    # Create figure
    fig, ax = plt.subplots(figsize=config.get_figure_size(), dpi=config.dpi)

    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Metric Value"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    # Labels
    ax.set_xlabel("Context", fontsize=config.label_fontsize)
    ax.set_ylabel("Layer", fontsize=config.label_fontsize)
    ax.set_title(title, fontsize=config.title_fontsize, pad=15)

    # Tick parameters
    ax.tick_params(labelsize=config.tick_fontsize)

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
