# CLI Command Contracts

**Purpose**: Define command-line interface for the oversight sensitivity measurement system.
**Design Principle**: CLI-first, composable, follows Unix philosophy (do one thing well, text in/out).

---

## Command 1: `oversee run-single`

**Purpose**: Execute a single prompt under specified context conditions and collect statistics.

**Usage**:
```bash
oversee run-single \
  --config experiments/configs/exp_001.json \
  --prompt "What is the capital of France?" \
  --prompt-id p001 \
  --contexts N,A,ARD \
  --output results/raw/exp_001/
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | path | Yes | Path to ExperimentConfig JSON |
| `--prompt` | string | Yes | Prompt text to execute |
| `--prompt-id` | string | Yes | Unique identifier for this prompt |
| `--contexts` | string | Yes | Comma-separated context codes (N/A/ARD) |
| `--output` | path | Yes | Directory to write ExecutionRun JSON files |
| `--verbose` | flag | No | Enable detailed logging to stderr |

**Output** (stdout): JSON array of run_ids
```json
{
  "run_ids": ["uuid1", "uuid2", "uuid3"],
  "execution_time_seconds": 892.5,
  "status": "success"
}
```

**Output** (files): `{output}/{run_id}.json` for each execution

**Errors** (stderr):
- Exit code 1: Invalid config file
- Exit code 2: Model loading failed
- Exit code 3: Inference failed
- Exit code 4: Statistics extraction failed

**Example**:
```bash
oversee run-single \
  --config experiments/configs/pilot.json \
  --prompt "Explain photosynthesis" \
  --prompt-id p042 \
  --contexts N,A \
  --output results/raw/pilot/
```

---

## Command 2: `oversee run-batch`

**Purpose**: Execute all prompts in a dataset across specified contexts with progress tracking and checkpoint recovery.

**Usage**:
```bash
oversee run-batch \
  --config experiments/configs/exp_001.json \
  --dataset experiments/prompts/core_60.csv \
  --checkpoint results/checkpoints/exp_001.ckpt
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | path | Yes | Path to ExperimentConfig JSON |
| `--dataset` | path | Yes | Path to PromptDataset CSV/JSON |
| `--checkpoint` | path | No | Checkpoint file for resume capability |
| `--max-workers` | int | No | Parallel workers (default: 1, sequential) |
| `--verbose` | flag | No | Enable detailed logging |

**Output** (stdout): Progress updates and summary
```
[1/60] Processing p001 (N, A, ARD)... 3/3 ✓ (285s)
[2/60] Processing p002 (N, A, ARD)... 3/3 ✓ (291s)
...
[60/60] Processing p060 (N, A, ARD)... 3/3 ✓ (278s)

Batch complete:
  Total runs: 180
  Successful: 178
  Failed: 2 (see results/raw/exp_001/failures.log)
  Total time: 5h 42m
```

**Output** (files):
- ExecutionRun JSONs in `{config.output_directory}/{run_id}.json`
- Checkpoint file updated after each prompt
- `failures.log` if any runs failed

**Checkpoint Format** (JSON):
```json
{
  "experiment_id": "exp_001",
  "completed_prompts": ["p001", "p002", ...],
  "last_update": "2025-12-31T15:30:00Z"
}
```

**Errors**:
- Exit code 1: Config/dataset file invalid
- Exit code 2: Model loading failed
- Exit code 0 with warnings: Partial failures (logged)

**Example**:
```bash
# Initial run
oversee run-batch \
  --config experiments/configs/main.json \
  --dataset experiments/prompts/balanced_120.csv \
  --checkpoint results/checkpoints/main.ckpt

# Resume after interruption
oversee run-batch \
  --config experiments/configs/main.json \
  --dataset experiments/prompts/balanced_120.csv \
  --checkpoint results/checkpoints/main.ckpt
```

---

## Command 3: `oversee compute-metrics`

**Purpose**: Compute CCI, EHL, TP, OSS metrics from execution runs.

**Usage**:
```bash
oversee compute-metrics \
  --experiment-dir results/raw/exp_001/ \
  --output results/metrics/exp_001_metrics.json \
  --context-pairs A_vs_N,ARD_vs_A
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--experiment-dir` | path | Yes | Directory containing ExecutionRun JSONs |
| `--output` | path | Yes | Where to write MetricResults JSON |
| `--context-pairs` | string | Yes | Which comparisons to compute (e.g., `A_vs_N`) |
| `--skip-baselines` | flag | No | Skip keyword and output-only baselines |
| `--verbose` | flag | No | Detailed metric computation logs |

**Output** (stdout): Summary statistics
```
Computing metrics for 60 prompts across 2 context pairs...
CCI: mean=0.23 (95% CI: [0.18, 0.28])
EHL: mean=12.4 tokens (95% CI: [10.1, 14.7])
TP:  mean=0.67 (95% CI: [0.62, 0.72])
OSS: mean=0.31 (95% CI: [0.25, 0.37])

Written to: results/metrics/exp_001_metrics.json
```

**Output** (file): MetricResults JSON with all computed values

**Errors**:
- Exit code 1: Missing required execution runs
- Exit code 2: Metric computation failed (invalid data)
- Exit code 3: Statistics formula error

**Example**:
```bash
oversee compute-metrics \
  --experiment-dir results/raw/main/ \
  --output results/metrics/main_metrics.json \
  --context-pairs A_vs_N,ARD_vs_A,ARD_vs_N
```

---

## Command 4: `oversee analyze`

**Purpose**: Perform statistical analysis (bootstrap CIs, power analysis, effect sizes).

**Usage**:
```bash
oversee analyze \
  --metrics results/metrics/exp_001_metrics.json \
  --output results/analysis/exp_001_analysis.json \
  --alpha 0.05 \
  --bootstrap-samples 10000
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--metrics` | path | Yes | MetricResults JSON file |
| `--output` | path | Yes | Where to write analysis results |
| `--alpha` | float | No | Significance level (default: 0.05) |
| `--bootstrap-samples` | int | No | Bootstrap iterations (default: 10000) |
| `--power-analysis` | flag | No | Include power analysis with subsampling |
| `--verbose` | flag | No | Detailed statistical output |

**Output** (stdout): Statistical summary
```
Statistical Analysis Results:

Context Pair: A_vs_N
  CCI effect size (Cohen's d): 0.52 (medium)
  P-value: 0.003 **
  95% CI: [0.18, 0.28]

  EHL effect size (Cohen's d): -0.48 (medium)
  P-value: 0.007 **
  95% CI: [-18.2, -6.6] tokens

Power Analysis (if --power-analysis):
  N=20: 45% detection probability
  N=40: 68% detection probability
  N=60: 82% detection probability

Baseline Comparison:
  Internal metrics effect size: 0.52
  Keyword baseline effect size: 0.21
  Ratio: 2.48x (internal metrics superior)
```

**Output** (file): Analysis JSON with all statistics

**Errors**:
- Exit code 1: Insufficient data for analysis
- Exit code 2: Statistical test failed

**Example**:
```bash
oversee analyze \
  --metrics results/metrics/main_metrics.json \
  --output results/analysis/main_analysis.json \
  --alpha 0.01 \
  --bootstrap-samples 20000 \
  --power-analysis
```

---

## Command 5: `oversee visualize`

**Purpose**: Generate publication-ready plots.

**Usage**:
```bash
oversee visualize \
  --metrics results/metrics/exp_001_metrics.json \
  --analysis results/analysis/exp_001_analysis.json \
  --output-dir results/figures/ \
  --figures radar,heatmap,timeseries
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--metrics` | path | Yes | MetricResults JSON |
| `--analysis` | path | No | Analysis JSON (for CIs/error bars) |
| `--output-dir` | path | Yes | Directory for figure output |
| `--figures` | string | Yes | Comma-separated figure types |
| `--dpi` | int | No | PNG resolution (default: 300) |
| `--format` | string | No | Output formats (default: `png,pdf`) |

**Figure Types**:
- `radar`: Context deformation radar plot
- `heatmap`: Layer × context metric heatmap
- `timeseries`: Entropy curves over tokens
- `intervention`: ARD vs A vs N comparison
- `baseline`: Internal vs baseline comparison
- `all`: Generate all figure types

**Output** (stdout): List of generated files
```
Generated:
  results/figures/fig_radar_exp_001.png (300 DPI)
  results/figures/fig_radar_exp_001.pdf (vector)
  results/figures/fig_radar_exp_001_data.json
  results/figures/fig_heatmap_exp_001.png
  results/figures/fig_heatmap_exp_001.pdf
  results/figures/fig_heatmap_exp_001_data.json
```

**Output** (files): PNG, PDF, and source JSON for each figure

**Errors**:
- Exit code 1: Missing required data
- Exit code 2: Visualization generation failed

**Example**:
```bash
oversee visualize \
  --metrics results/metrics/main_metrics.json \
  --analysis results/analysis/main_analysis.json \
  --output-dir results/figures/ \
  --figures all \
  --dpi 600
```

---

## Command 6: `oversee report`

**Purpose**: Generate analysis report with dual-use disclaimers and limitations.

**Usage**:
```bash
oversee report \
  --experiment-id exp_001 \
  --metrics results/metrics/exp_001_metrics.json \
  --analysis results/analysis/exp_001_analysis.json \
  --figures results/figures/ \
  --output results/reports/exp_001_report.md
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--experiment-id` | string | Yes | Experiment identifier |
| `--metrics` | path | Yes | MetricResults JSON |
| `--analysis` | path | Yes | Analysis JSON |
| `--figures` | path | Yes | Directory containing figures |
| `--output` | path | Yes | Output report path (.md or .pdf) |
| `--format` | string | No | Report format (default: `markdown`) |

**Output** (file): Markdown or PDF report with sections:
1. Executive Summary
2. Methodology (references config, datasets)
3. Results (metrics, statistical tests, figures)
4. Baseline Comparisons
5. **Limitations** (mandatory per constitution FR-017)
6. **Dual-Use Risk Disclaimer** (mandatory per constitution FR-017)
7. Reproducibility Information (software versions, configs)

**Errors**:
- Exit code 1: Missing required inputs
- Exit code 2: Report generation failed

**Example**:
```bash
oversee report \
  --experiment-id main_experiment \
  --metrics results/metrics/main_metrics.json \
  --analysis results/analysis/main_analysis.json \
  --figures results/figures/ \
  --output results/reports/main_report.md
```

---

## Command 7: `oversee validate-config`

**Purpose**: Validate experiment configuration before execution.

**Usage**:
```bash
oversee validate-config --config experiments/configs/exp_001.json
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--config` | path | Yes | ExperimentConfig JSON to validate |

**Output** (stdout): Validation results
```
Validating experiments/configs/exp_001.json...

✓ Config structure valid
✓ Model identifier exists: meta-llama/Llama-2-7b-hf
✓ Layer indices valid: [0, 12, 24] (model has 32 layers)
✓ Random seed documented: 42
✓ Prompt dataset exists: experiments/prompts/core_60.csv
✓ Output directory writable: results/raw/exp_001/
✗ WARNING: temperature > 0 may reduce reproducibility

Config valid with 1 warning.
```

**Errors**:
- Exit code 1: Config invalid (validation failed)
- Exit code 0: Config valid (warnings non-fatal)

**Example**:
```bash
oversee validate-config --config experiments/configs/pilot.json
```

---

## Pipeline Composition Example

```bash
#!/bin/bash
# Full experiment pipeline

EXPERIMENT_ID="main_experiment"
CONFIG="experiments/configs/${EXPERIMENT_ID}.json"
DATASET="experiments/prompts/balanced_120.csv"

# Step 1: Validate configuration
oversee validate-config --config "$CONFIG" || exit 1

# Step 2: Run batch execution
oversee run-batch \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "results/checkpoints/${EXPERIMENT_ID}.ckpt"

# Step 3: Compute metrics
oversee compute-metrics \
  --experiment-dir "results/raw/${EXPERIMENT_ID}/" \
  --output "results/metrics/${EXPERIMENT_ID}_metrics.json" \
  --context-pairs "A_vs_N,ARD_vs_A"

# Step 4: Statistical analysis
oversee analyze \
  --metrics "results/metrics/${EXPERIMENT_ID}_metrics.json" \
  --output "results/analysis/${EXPERIMENT_ID}_analysis.json" \
  --power-analysis

# Step 5: Generate figures
oversee visualize \
  --metrics "results/metrics/${EXPERIMENT_ID}_metrics.json" \
  --analysis "results/analysis/${EXPERIMENT_ID}_analysis.json" \
  --output-dir "results/figures/" \
  --figures all

# Step 6: Create report
oversee report \
  --experiment-id "$EXPERIMENT_ID" \
  --metrics "results/metrics/${EXPERIMENT_ID}_metrics.json" \
  --analysis "results/analysis/${EXPERIMENT_ID}_analysis.json" \
  --figures "results/figures/" \
  --output "results/reports/${EXPERIMENT_ID}_report.md"

echo "Pipeline complete. Report: results/reports/${EXPERIMENT_ID}_report.md"
```

---

## Design Principles Applied

1. **CLI-First** (Constitution Principle V): Every operation accessible via command line
2. **Text In/Out**: JSON for structured data, human-readable summaries to stdout
3. **Composability**: Commands chain via files (Unix philosophy)
4. **Reproducibility** (Constitution Principle I): Configs and checkpoints enable exact reruns
5. **Error Handling**: Clear exit codes, detailed stderr messages
6. **Progress Tracking**: Verbose logging to stderr, summaries to stdout
7. **Dual-Use Ethics** (Constitution Principle IV): `report` command mandates disclaimers
