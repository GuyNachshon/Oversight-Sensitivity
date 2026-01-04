"""
Timeseries Plot Generator (T049)

Create timeseries plots showing entropy-over-time for same prompt across contexts.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from .config import VisualizationConfig, DEFAULT_VIZ_CONFIG


# Context display names and order
CONTEXT_LABELS = {
    "N": "Neutral",
    "EO": "Explicit Oversight",
    "IO": "Implicit Oversight",
    "KW": "Keyword Baseline",
    "REPRIME": "Re-Prime (N→EO)",
}

CONTEXT_ORDER = ["N", "EO", "IO", "KW", "REPRIME"]


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

    # Plot each context in consistent order
    for context in CONTEXT_ORDER:
        if context not in entropy_by_context:
            continue
        entropies = entropy_by_context[context]
        token_positions = range(len(entropies))
        color = config.get_context_color(context)
        label = CONTEXT_LABELS.get(context, context)

        ax.plot(
            token_positions,
            entropies,
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
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


def create_entropy_timeseries_with_ci(
    data: Dict[str, Tuple[List[float], List[float], List[float]]],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Logit Entropy Over Time",
    xlabel: str = "Token Position",
    ylabel: str = "Logit Entropy (bits)",
    show_ci: bool = True,
) -> plt.Figure:
    """
    Create timeseries plot with confidence intervals.

    Args:
        data: {context: (means, ci_lower, ci_upper)} where each is a list
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_ci: Whether to show confidence bands

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=config.get_figure_size(), dpi=config.dpi)

    # Plot each context in consistent order
    for context in CONTEXT_ORDER:
        if context not in data:
            continue

        means, ci_lower, ci_upper = data[context]
        token_positions = np.arange(len(means))
        color = config.get_context_color(context)
        label = CONTEXT_LABELS.get(context, context)

        # Plot mean line
        ax.plot(
            token_positions,
            means,
            linewidth=2,
            label=label,
            color=color,
        )

        # Plot confidence band
        if show_ci:
            ax.fill_between(
                token_positions,
                ci_lower,
                ci_upper,
                alpha=0.2,
                color=color,
            )

    # Labels
    ax.set_xlabel(xlabel, fontsize=config.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.label_fontsize)
    ax.set_title(title, fontsize=config.title_fontsize)

    # Apply styling
    config.apply_style(ax)

    # Legend
    ax.legend(fontsize=config.legend_fontsize, loc='upper right')

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


def create_family_comparison_plot(
    data_by_family: Dict[str, Dict[str, Tuple[List[float], List[float], List[float]]]],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    families: List[str] = None,
    ylabel: str = "Logit Entropy (bits)",
) -> plt.Figure:
    """
    Create multi-panel plot comparing entropy timeseries across prompt families.

    Args:
        data_by_family: {family: {context: (means, ci_lower, ci_upper)}}
        output_path: Path to save figure (optional)
        config: Visualization configuration
        families: List of families to plot (default: all)
        ylabel: Y-axis label

    Returns:
        Matplotlib figure object
    """
    if families is None:
        families = list(data_by_family.keys())

    n_families = len(families)
    fig, axes = plt.subplots(
        1, n_families,
        figsize=(config.figure_width * n_families / 2, config.figure_height),
        dpi=config.dpi,
        sharey=True,
    )

    if n_families == 1:
        axes = [axes]

    for idx, family in enumerate(families):
        ax = axes[idx]
        family_data = data_by_family.get(family, {})

        # Plot each context
        for context in CONTEXT_ORDER:
            if context not in family_data:
                continue

            means, ci_lower, ci_upper = family_data[context]
            token_positions = np.arange(len(means))
            color = config.get_context_color(context)
            label = CONTEXT_LABELS.get(context, context) if idx == 0 else None

            ax.plot(token_positions, means, linewidth=2, label=label, color=color)
            ax.fill_between(token_positions, ci_lower, ci_upper, alpha=0.2, color=color)

        # Panel title
        ax.set_title(family.capitalize(), fontsize=config.title_fontsize)
        ax.set_xlabel("Token Position", fontsize=config.label_fontsize)

        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=config.label_fontsize)

        config.apply_style(ax)

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(4, len(labels)),
        fontsize=config.legend_fontsize,
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)  # Make room for legend

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
