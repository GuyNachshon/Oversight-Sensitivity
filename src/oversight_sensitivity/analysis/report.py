"""
Baseline Comparison Report Generator (T061)

Generate human-readable reports summarizing baseline validation results.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .baseline_comparison import BaselineComparison


def generate_baseline_report(
    comparison_results: Dict[str, List[BaselineComparison]],
    experimental_contexts: List[str],
    baseline_contexts: List[str],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive baseline comparison report.

    Args:
        comparison_results: Dict[metric -> List[BaselineComparison]]
        experimental_contexts: Experimental conditions tested
        baseline_contexts: Baseline controls used
        output_path: Path to save markdown report (optional)

    Returns:
        Markdown-formatted report string

    Example:
        >>> results = run_baseline_analysis(metrics_by_context)
        >>> report = generate_baseline_report(
        ...     results,
        ...     experimental_contexts=["A", "ARD"],
        ...     baseline_contexts=["KW", "OO", "R"],
        ... )
    """
    lines = []

    # Header
    lines.append("# Baseline Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"This report compares experimental contexts {experimental_contexts} ")
    lines.append(f"against baseline controls {baseline_contexts} to validate that observed ")
    lines.append("effects are due to evaluative oversight, not confounding factors.")
    lines.append("")

    # Baseline descriptions
    lines.append("## Baseline Controls")
    lines.append("")
    lines.append("- **KW (Keyword)**: Same keywords as Audited but without evaluative framing")
    lines.append("- **OO (Output-Only)**: Audits final answer only, not reasoning process")
    lines.append("- **R (Random)**: Arbitrary text matched for length, no semantic content")
    lines.append("")

    # Results by metric
    lines.append("## Results by Metric")
    lines.append("")

    for metric_name, comparisons in comparison_results.items():
        lines.append(f"### {metric_name}")
        lines.append("")

        if not comparisons:
            lines.append("*No comparisons available for this metric.*")
            lines.append("")
            continue

        # Group by experimental context
        by_exp = {}
        for comp in comparisons:
            if comp.experimental_context not in by_exp:
                by_exp[comp.experimental_context] = []
            by_exp[comp.experimental_context].append(comp)

        # Create table for each experimental context
        for exp_context in experimental_contexts:
            if exp_context not in by_exp:
                continue

            lines.append(f"#### {exp_context} vs Baselines")
            lines.append("")

            # Table header
            lines.append("| Baseline | Δ | Cohen's d | p-value | Significant |")
            lines.append("|----------|---|-----------|---------|-------------|")

            # Table rows
            for comp in by_exp[exp_context]:
                sig = "Yes***" if comp.p_value < 0.001 else (
                    "Yes**" if comp.p_value < 0.01 else (
                        "Yes*" if comp.p_value < 0.05 else "No"
                    )
                )

                lines.append(
                    f"| {comp.baseline_context} | "
                    f"{comp.delta:+.3f} | "
                    f"{comp.cohens_d:.3f} | "
                    f"{comp.p_value:.4f} | "
                    f"{sig} |"
                )

            lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### Hypothesis Validation")
    lines.append("")

    for metric_name, comparisons in comparison_results.items():
        lines.append(f"**{metric_name}:**")
        lines.append("")

        for exp_context in experimental_contexts:
            exp_comparisons = [
                c for c in comparisons if c.experimental_context == exp_context
            ]

            if not exp_comparisons:
                continue

            # Count significant comparisons
            total = len(exp_comparisons)
            significant = sum(1 for c in exp_comparisons if c.p_value < 0.05)
            significant_pct = (significant / total * 100) if total > 0 else 0

            lines.append(f"- {exp_context}: {significant}/{total} ({significant_pct:.0f}%) significant comparisons vs baselines")

            # Effect size interpretation
            mean_d = sum(abs(c.cohens_d) for c in exp_comparisons) / total if total > 0 else 0
            effect_interpretation = (
                "large" if mean_d > 0.8 else (
                    "medium" if mean_d > 0.5 else (
                        "small" if mean_d > 0.2 else "negligible"
                    )
                )
            )
            lines.append(f"  - Mean |Cohen's d| = {mean_d:.3f} ({effect_interpretation} effect)")

        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    # Determine overall validation status
    overall_significant = sum(
        sum(1 for c in comparisons if c.p_value < 0.05)
        for comparisons in comparison_results.values()
    )
    overall_total = sum(len(comparisons) for comparisons in comparison_results.values())

    if overall_total > 0:
        overall_pct = overall_significant / overall_total * 100
        lines.append(
            f"Overall, {overall_significant}/{overall_total} ({overall_pct:.0f}%) "
            "of comparisons show significant differences between experimental contexts "
            "and baseline controls."
        )
    else:
        lines.append("No comparisons were performed.")

    lines.append("")

    if overall_pct > 50:
        lines.append(
            "**Conclusion:** Experimental contexts (A, ARD) show consistently different "
            "behavior from baseline controls, supporting the hypothesis that effects are "
            "due to evaluative oversight semantics, not confounding factors like keywords, "
            "prompt length, or output-only evaluation."
        )
    else:
        lines.append(
            "**Conclusion:** Limited evidence for differentiation from baseline controls. "
            "Further investigation needed to rule out confounding factors."
        )

    lines.append("")

    # Footnotes
    lines.append("---")
    lines.append("")
    lines.append("*Significance levels: *** p<0.001, ** p<0.01, * p<0.05*")
    lines.append("")

    # Join into single string
    report = "\n".join(lines)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
