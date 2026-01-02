"""
Timeseries Plot Generator (T049)

Create timeseries plots showing entropy-over-time for same prompt across contexts.
"""

import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from pathlib import Path

from .config import VisualizationConfig, DEFAULT_VIZ_CONFIG


def create_entropy_timeseries(
    entropy_by_context: Dict[str, List[float]],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Logit Entropy Over Time",
    xlabel: str = "Token Position",
    ylabel: str = "Logit Entropy (bits)",
) -> plt.Figure:
    """
    Create timeseries plot of entropy decay across contexts.

    Args:
        entropy_by_context: Dict[context -> list of entropy values]
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Matplotlib figure object

    Example:
        >>> entropies = {
        ...     "N": [4.0, 3.5, 3.0, 2.5, 2.0],
        ...     "A": [4.0, 3.0, 2.0, 1.5, 1.0],
        ... }
        >>> fig = create_entropy_timeseries(entropies)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=config.get_figure_size(), dpi=config.dpi)

    # Plot each context
    for context, entropies in entropy_by_context.items():
        token_positions = range(len(entropies))
        color = config.get_context_color(context)

        ax.plot(
            token_positions,
            entropies,
            marker="o",
            linewidth=2,
            markersize=4,
            label=context,
            color=color,
            alpha=0.8,
        )

    # Labels
    ax.set_xlabel(xlabel, fontsize=config.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.label_fontsize)
    ax.set_title(title, fontsize=config.title_fontsize)

    # Apply styling
    config.apply_style(ax)

    # Legend
    ax.legend(fontsize=config.legend_fontsize)

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
