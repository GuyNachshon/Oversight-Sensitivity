# Oversight Sensitivity Measurement

A research measurement system to quantify how evaluative context (audit framing) reshapes internal reasoning dynamics in LLMs.

## Overview

This project measures **Oversight Sensitivity** - how strongly a model's internal reasoning dynamics change in response to evaluative or social context. It introduces new measurement axes (CCI, EHL, TP, OSS) to characterize context-conditioned deformation of internal computation.

**Key Features**:
- Execute prompts under 3 controlled contexts: Neutral (N), Audited (A), Audited+Reasoning Discipline (ARD)
- Baseline controls: Keyword (KW), Output-Only (OO), Random (R) to validate effects
- Collect 6 per-token statistics across 3 layers: logit entropy, top-1 probability, residual stream norm, activation variance, attention entropy, cosine similarity
- Compute 4 core metrics: Computation Compression Index (CCI), Exploration Half-Life (EHL), Trajectory Predictability (TP), Oversight Sensitivity Score (OSS)
- Statistical rigor: Bootstrap CIs (10k iterations), power analysis, effect sizes, baseline comparisons
- Publication-ready visualizations (300 DPI, colorblind-safe palettes)
- Full reproducibility: Fixed seeds, versioned configs, bit-identical reruns

## Quick Start

See **[Quickstart Guide](specs/001-oversight-sensitivity-measurement/quickstart.md)** for a 30-minute walkthrough from installation to first measurement.

### Installation

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Verify installation
oversee --version
```

### Single Measurement (5 minutes)

```bash
# 1. Create experiment config
cat > experiments/configs/quickstart.json << 'EOF'
{
  "experiment_id": "quickstart",
  "model_identifier": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "random_seed": 42,
  "layer_indices": [0, 11, 21],
  "context_conditions": ["N", "A"],
  ...
}
EOF

# 2. Run single prompt
oversee run-single \
  --config experiments/configs/quickstart.json \
  --prompt "What is the capital of France?" \
  --prompt-id p_test \
  --contexts N,A \
  --output results/raw/quickstart/

# 3. Compute metrics
oversee compute-metrics \
  --experiment-dir results/raw/quickstart/ \
  --output results/metrics/quickstart_metrics.json \
  --context-pairs A_vs_N

# 4. Statistical analysis
oversee analyze \
  --metrics results/metrics/quickstart_metrics.json \
  --output results/analysis/quickstart_stats.json

# 5. Visualize results
oversee visualize \
  --metrics results/metrics/quickstart_metrics.json \
  --analysis results/analysis/quickstart_stats.json \
  --output-dir results/plots/quickstart/
```

### Full Research Pipeline

```bash
# 1. Run batch experiment with all contexts (including baselines)
oversee run-batch \
  --config experiments/configs/full_experiment.json \
  --contexts N A ARD KW OO R

# 2. Compute metrics for all context pairs
oversee compute-metrics \
  --experiment-dir results/raw/full/ \
  --output results/metrics/full_metrics.json \
  --context-pairs A_vs_N ARD_vs_A

# 3. Statistical analysis with bootstrap CIs
oversee analyze \
  --metrics results/metrics/full_metrics.json \
  --output results/analysis/full_stats.json \
  --n-bootstrap 10000

# 4. Baseline comparison & validation
oversee compare-baselines \
  --metrics results/metrics/full_metrics.json \
  --output results/analysis/baseline_comparison.json \
  --report results/reports/baseline_validation.md \
  --experimental A ARD \
  --baselines KW OO R

# 5. Generate all visualizations
oversee visualize \
  --metrics results/metrics/full_metrics.json \
  --analysis results/analysis/full_stats.json \
  --output-dir results/plots/full/ \
  --plot-type all
```

## Project Structure

```
oversight-sensitivity/
├── src/oversight_sensitivity/     # Source code
│   ├── inference/                 # Model loading, hooks, reproducibility
│   ├── statistics/                # Per-token stat collection
│   ├── metrics/                   # CCI, EHL, TP, OSS computation
│   ├── experiments/               # Config, batch execution, checkpointing
│   ├── baselines/                 # Keyword and output-only baselines
│   ├── analysis/                  # Bootstrap CIs, power analysis, effect sizes
│   ├── visualization/             # Publication-ready plots
│   └── cli/                       # Command-line interface
├── experiments/                   # Versioned experiment configs and datasets
├── results/                       # Metrics, figures, reports (tracked in git)
├── tests/                         # Unit and integration tests
└── specs/001-oversight-sensitivity-measurement/  # Design documentation
```

## CLI Commands

### Execution
- `oversee run-single`: Execute single prompt under specified contexts
- `oversee run-batch`: Batch process dataset with checkpointing and progress tracking

### Analysis
- `oversee compute-metrics`: Calculate CCI/EHL/TP/OSS from execution runs
- `oversee analyze`: Statistical analysis (bootstrap CIs, effect sizes, power)
- `oversee compare-baselines`: Validate effects against control baselines (KW/OO/R)

### Visualization & Reporting
- `oversee visualize`: Generate publication-ready plots (radar, heatmap, timeseries, intervention)
- Baseline comparison reports with hypothesis validation

### Utilities
- `oversee validate-config`: Pre-flight configuration validation

See [CLI Reference](specs/001-oversight-sensitivity-measurement/contracts/cli-commands.md) for full documentation.

## Research Integrity (Constitution)

This project follows strict research integrity principles defined in `.specify/memory/constitution.md`:

1. **Reproducibility**: All metrics defined before execution, fixed seeds, versioned configs
2. **Falsifiability**: Hypotheses with failure conditions, power analysis, baseline comparisons
3. **Measurement Over Claims**: Observable phenomena only (e.g., "variability reduction" not "deception")
4. **Ethics & Dual-Use**: Mandatory dual-use disclaimers, limitations sections in all reports
5. **Lightweight**: Summary statistics only, no full activation storage
6. **Experiment-Driven**: Clear pipeline separation (inference → statistics → metrics → analysis → visualization)

## Documentation

- **[Quickstart Guide](specs/001-oversight-sensitivity-measurement/quickstart.md)**: 30-minute tutorial
- **[Feature Spec](specs/001-oversight-sensitivity-measurement/spec.md)**: User stories and requirements
- **[Implementation Plan](specs/001-oversight-sensitivity-measurement/plan.md)**: Technical architecture
- **[Data Model](specs/001-oversight-sensitivity-measurement/data-model.md)**: Entities and schemas
- **[Research Decisions](specs/001-oversight-sensitivity-measurement/research.md)**: Technology choices with rationale
- **[Tasks](specs/001-oversight-sensitivity-measurement/tasks.md)**: Implementation task breakdown

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Configure nbstripout (for notebooks)
nbstripout --install
```

## License

[Add license information]

## Citation

[Add citation information when published]

## Acknowledgments

This project was developed for the Manipulation Hackathon to address gaps in measuring strategic behavior in LLMs under oversight conditions.
