"""
Baseline Comparison Visualization (T059)

Create visualizations comparing experimental contexts against baseline controls.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from .config import VisualizationConfig, DEFAULT_VIZ_CONFIG


def create_baseline_comparison_plot(
    comparison_data: Dict[str, Dict[str, float]],
    experimental_contexts: List[str] = ["A", "ARD"],
    baseline_contexts: List[str] = ["KW", "OO", "R"],
    metric_names: List[str] = ["CCI", "EHL", "TP", "OSS"],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Experimental Contexts vs Baseline Controls",
) -> plt.Figure:
    """
    Create grouped bar plot comparing experimental contexts and baselines.

    Args:
        comparison_data: Dict[context -> Dict[metric -> mean_value]]
        experimental_contexts: List of experimental contexts
        baseline_contexts: List of baseline controls
        metric_names: Metrics to plot
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title

    Returns:
        Matplotlib figure object

    Example:
        >>> data = {
        ...     "N": {"CCI": 0.0, "EHL": 10.0},
        ...     "A": {"CCI": 0.8, "EHL": 5.0},
        ...     "KW": {"CCI": 0.1, "EHL": 9.5},
        ... }
        >>> fig = create_baseline_comparison_plot(data)
    """
    n_metrics = len(metric_names)
    n_contexts = len(comparison_data)

    # Create figure with subplots
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(config.figure_width * n_metrics / 2, config.figure_height),
        dpi=config.dpi,
    )

    # Ensure axes is a list
    if n_metrics == 1:
        axes = [axes]

    # Prepare data
    contexts = list(comparison_data.keys())
    bar_width = 0.8 / n_contexts

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]

        # Extract values for this metric
        values = []
        colors = []

        for context in contexts:
            if metric_name in comparison_data[context]:
                values.append(comparison_data[context][metric_name])
            else:
                values.append(0.0)

            # Color based on context type
            colors.append(config.get_context_color(context))

        # Create positions
        x_positions = np.arange(len(contexts))

        # Create bars
        bars = ax.bar(
            x_positions,
            values,
            bar_width * n_contexts,
            color=colors,
            edgecolor="black",
            linewidth=1,
            alpha=0.8,
        )

        # Add horizontal line at neutral baseline
        if "N" in contexts:
            neutral_idx = contexts.index("N")
            neutral_value = values[neutral_idx]
            ax.axhline(
                y=neutral_value,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Neutral baseline",
            )

        # Separate experimental from baselines with vertical line
        if experimental_contexts and baseline_contexts:
            # Find boundary between experimental and baselines
            exp_end = max(
                [contexts.index(c) for c in experimental_contexts if c in contexts],
                default=-1,
            )
            if exp_end >= 0 and exp_end < len(contexts) - 1:
                ax.axvline(
                    x=exp_end + 0.5,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.3,
                )

        # Labels
        ax.set_xlabel("Context", fontsize=config.label_fontsize)
        ax.set_ylabel(metric_name, fontsize=config.label_fontsize)
        ax.set_title(metric_name, fontsize=config.title_fontsize)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(contexts, fontsize=config.tick_fontsize, rotation=0)

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


def create_effect_size_heatmap(
    effect_sizes: Dict[str, Dict[str, float]],
    experimental_contexts: List[str] = ["A", "ARD"],
    baseline_contexts: List[str] = ["KW", "OO", "R"],
    output_path: Optional[Path] = None,
    config: VisualizationConfig = DEFAULT_VIZ_CONFIG,
    title: str = "Effect Sizes: Experimental vs Baselines (Cohen's d)",
) -> plt.Figure:
    """
    Create heatmap of effect sizes (Cohen's d) comparing experimental to baselines.

    Args:
        effect_sizes: Dict[experimental_context -> Dict[baseline_context -> d]]
        experimental_contexts: Experimental contexts (rows)
        baseline_contexts: Baseline contexts (columns)
        output_path: Path to save figure (optional)
        config: Visualization configuration
        title: Plot title

    Returns:
        Matplotlib figure object

    Example:
        >>> effect_sizes = {
        ...     "A": {"KW": 1.5, "OO": 1.2, "R": 2.0},
        ...     "ARD": {"KW": 0.8, "OO": 0.6, "R": 1.1},
        ... }
        >>> fig = create_effect_size_heatmap(effect_sizes)
    """
    import seaborn as sns

    # Build matrix
    matrix = np.zeros((len(experimental_contexts), len(baseline_contexts)))

    for i, exp_ctx in enumerate(experimental_contexts):
        for j, base_ctx in enumerate(baseline_contexts):
            if exp_ctx in effect_sizes and base_ctx in effect_sizes[exp_ctx]:
                matrix[i, j] = effect_sizes[exp_ctx][base_ctx]

    # Create figure
    fig, ax = plt.subplots(figsize=config.get_figure_size(), dpi=config.dpi)

    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        xticklabels=baseline_contexts,
        yticklabels=experimental_contexts,
        cbar_kws={"label": "Effect Size (Cohen's d)"},
        ax=ax,
        linewidths=1,
        linecolor="gray",
    )

    # Labels
    ax.set_xlabel("Baseline Context", fontsize=config.label_fontsize)
    ax.set_ylabel("Experimental Context", fontsize=config.label_fontsize)
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
