# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oversight Sensitivity Measurement - A research system to quantify how evaluative context (audit framing) reshapes internal reasoning dynamics in LLMs. Measures 6 per-token statistics across model layers under different contexts (Neutral, Audited, Audited+Reasoning Discipline) and computes metrics (CCI, EHL, TP, OSS).

## Development Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_metrics.py

# Run specific test
uv run pytest tests/test_metrics.py::TestCCI::test_basic_cci -v

# Format and lint
uv run black src/ tests/
uv run ruff check src/ tests/

# CLI usage (after install)
oversee --help
oversee run-single --config experiments/configs/example.json --prompt-id p001 --prompt-text "What is 2+2?" --context N
oversee run-batch --config experiments/configs/example.json
oversee compute-metrics --experiment-dir results/raw/exp/ --output results/metrics/metrics.json --context-pairs A_vs_N
oversee analyze --metrics results/metrics/metrics.json --output results/analysis/stats.json
oversee visualize --metrics results/metrics/metrics.json --analysis results/analysis/stats.json --output-dir plots/
oversee compare-baselines --metrics results/metrics/metrics.json --output results/analysis/baseline.json
oversee select-layers --model google/gemma-3-270m-it  # Smart layer selection for a model

# Debug mode
oversee --debug run-single ...
OVERSEE_LOG_LEVEL=DEBUG oversee run-batch ...
```

## Architecture

```
src/oversight_sensitivity/
├── inference/          # Model loading, hooks, reproducibility, context templates
│   ├── executor.py     # PromptExecutor: orchestrates single-prompt runs
│   ├── model_loader.py # HuggingFace model loading with reproducibility
│   ├── hooks.py        # PyTorch forward hooks for statistics extraction
│   └── contexts.py     # Context templates (N, A, ARD, KW, OO, R)
├── statistics/         # Per-token stat collection (entropy, norm, variance, etc.)
│   └── collector.py    # StatisticsCollector: 6 measurements per token
├── metrics/            # Core metric implementations
│   ├── compute.py      # Orchestrator: loads runs, computes all metrics
│   ├── cci.py          # Computation Compression Index
│   ├── ehl.py          # Exploration Half-Life
│   ├── tp.py           # Trajectory Predictability
│   └── oss.py          # Oversight Sensitivity Score (composite)
├── experiments/        # Config, batch execution, checkpointing
│   ├── config.py       # ExperimentConfig dataclass and validation
│   ├── batch_run.py    # BatchExecutor with checkpointing
│   └── checkpoint.py   # Resumable run management
├── analysis/           # Statistical analysis
│   ├── bootstrap.py    # Bootstrap CIs (10k iterations)
│   ├── effect_sizes.py # Cohen's d, paired t-tests
│   ├── power.py        # Power analysis via subsampling
│   └── baseline_comparison.py  # Validate vs control baselines
├── visualization/      # Publication-ready plots (300 DPI, colorblind-safe)
│   ├── radar.py, heatmap.py, timeseries.py, intervention.py
│   └── config.py       # VisualizationConfig
└── cli/                # Command-line interface entry points
```

**Data Flow:** Config → PromptExecutor → StatisticsCollector → ExecutionRun (JSON) → MetricResults → Analysis → Visualization

## Key Concepts

- **Context Conditions:** N (neutral), A (audited), ARD (audited + reasoning discipline), KW/OO/R (baselines)
- **Layer Indices:** Sample 3 layers (early/mid/late) via `oversee select-layers --model <model>`
- **6 Statistics:** logit entropy, top-1 probability, residual stream norm, activation variance, attention entropy, cosine similarity
- **4 Metrics:** CCI (compression), EHL (exploration decay), TP (predictability), OSS (composite score)

## Important Guidelines

- Use UV for all Python execution and dependency management
- DO NOT WRITE mocks/placeholders/stubs - implement full functionality
- Fixed seeds and versioned configs ensure reproducibility
- Results stored as JSON in `results/raw/`, metrics in `results/metrics/`, plots in `results/plots/`
