"""
Visualization Configuration (T052)

Standardize plot aesthetics for publication quality.

Per research.md RQ4: Use colorblind-friendly palette, consistent DPI,
font sizes for readability.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class VisualizationConfig:
    """
    Configuration for publication-ready visualizations.

    Per research.md: 300 DPI for publication, colorblind palette.
    """

    # Figure dimensions (inches)
    figure_width: float = 8.0
    figure_height: float = 6.0

    # DPI for raster outputs
    dpi: int = 300

    # Font sizes
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10

    # Colors (colorblind-friendly palette)
    # Based on Wong 2011 "Points of view: Color blindness"
    color_neutral: str = "#0173B2"  # Blue
    color_audited: str = "#DE8F05"  # Orange
    color_ard: str = "#029E73"  # Green
    color_baseline: str = "#CC78BC"  # Purple

    # Additional colors for multi-series
    color_palette: List[str] = None

    # Grid and styling
    show_grid: bool = True
    grid_alpha: float = 0.3
    spine_width: float = 1.0

    # Error bar styling
    error_bar_capsize: float = 4.0
    error_bar_alpha: float = 0.7

    # Save formats
    save_png: bool = True
    save_pdf: bool = True
    save_svg: bool = False

    def __post_init__(self):
        """Initialize default color palette if not provided."""
        if self.color_palette is None:
            self.color_palette = [
                self.color_neutral,
                self.color_audited,
                self.color_ard,
                self.color_baseline,
                "#949494",  # Gray
                "#CC78BC",  # Purple
                "#CA9161",  # Brown
                "#ECE133",  # Yellow
            ]

    def get_context_color(self, context: str) -> str:
        """
        Get color for a specific context.

        Args:
            context: Context condition ("N", "A", "ARD", "KW", "OO", "R")

        Returns:
            Hex color code
        """
        color_map = {
            "N": self.color_neutral,
            "A": self.color_audited,
            "ARD": self.color_ard,
            # Baselines use grays/muted colors
            "KW": "#949494",  # Gray (keyword baseline)
            "OO": "#CA9161",  # Brown (output-only baseline)
            "R": "#ECE133",   # Yellow (random baseline)
        }
        return color_map.get(context, self.color_palette[0])

    def apply_style(self, ax) -> None:
        """
        Apply styling to matplotlib axes.

        Args:
            ax: Matplotlib axes object
        """
        import matplotlib.pyplot as plt

        # Grid
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)

        # Spines
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        # Font sizes
        ax.tick_params(labelsize=self.tick_fontsize)

    def get_figure_size(self) -> Tuple[float, float]:
        """Get figure size tuple."""
        return (self.figure_width, self.figure_height)


# Default configuration
DEFAULT_VIZ_CONFIG = VisualizationConfig()
