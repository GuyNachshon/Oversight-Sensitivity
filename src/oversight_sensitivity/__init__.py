"""
Oversight Sensitivity Measurement System

A research measurement system to quantify how evaluative context (audit framing)
reshapes internal reasoning dynamics in LLMs.

Core capabilities:
- Execute prompts under controlled contexts (Neutral, Audited, Audited+Reasoning Discipline)
- Collect lightweight execution statistics per token (6 measurements × 3 layers)
- Compute metrics: CCI, EHL, TP, OSS
- Statistical analysis with bootstrap CIs and power analysis
- Publication-ready visualizations

Constitution principles enforced:
- Reproducibility: Fixed seeds, versioned configs, deterministic execution
- Falsifiability: Power analysis, baseline comparisons
- Measurement over claims: Observable phenomena, not anthropomorphic interpretations
- Ethics: Dual-use disclaimers in all reports
- Lightweight: Summary statistics only, no full activation storage
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
