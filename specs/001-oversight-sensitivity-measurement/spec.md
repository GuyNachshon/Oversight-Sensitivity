# Feature Specification: Oversight Sensitivity Measurement

**Feature Branch**: `001-oversight-sensitivity-measurement`
**Created**: 2025-12-31
**Status**: Draft
**Input**: User description: "init project based on @docs/PRD.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Core Measurement Pipeline (Priority: P1)

A researcher wants to measure how evaluative context affects LLM internal computation by running controlled experiments that compare model behavior under neutral vs. audited conditions.

**Why this priority**: This is the fundamental capability - without the measurement pipeline, no research can be conducted. It delivers immediate scientific value by enabling the first hypothesis test (H1: oversight compresses computation).

**Independent Test**: Can be fully tested by running a single prompt through neutral and audited contexts, collecting execution statistics, computing the Computation Compression Index (CCI), and verifying the metric value is mathematically correct.

**Acceptance Scenarios**:

1. **Given** a prompt and a model configuration, **When** the researcher runs the prompt under neutral context, **Then** execution statistics (logit entropy, top-1 probability, residual stream norm, activation variance, attention entropy, cosine similarity) are logged per token for selected layers
2. **Given** the same prompt and model, **When** the researcher runs it under audited context, **Then** the same execution statistics are logged with identical structure
3. **Given** logged statistics from both contexts, **When** the researcher computes CCI, **Then** a single numerical value quantifying relative variability reduction is returned
4. **Given** experiment results, **When** the researcher exports data, **Then** all raw statistics, metrics, and configurations are saved in reproducible format with fixed random seeds

---

### User Story 2 - Intervention Analysis (Priority: P2)

A researcher wants to test if explicit reasoning discipline instructions can counteract oversight-induced computation deformation by adding a third condition (Audited + Reasoning Discipline).

**Why this priority**: This tests hypothesis H2 and demonstrates the measurement's scientific utility beyond simple observation. It requires the P1 pipeline to exist first.

**Independent Test**: Can be tested by running a prompt through all three contexts (N, A, ARD), computing metrics for each, and verifying that ARD metrics shift toward neutral values.

**Acceptance Scenarios**:

1. **Given** a prompt with neutral and audited results, **When** the researcher runs the same prompt under Audited + Reasoning Discipline (ARD) context, **Then** execution statistics are logged with the intervention instructions active
2. **Given** metrics from all three conditions (N, A, ARD), **When** the researcher compares them, **Then** the system shows whether ARD partially reverses the A-N deformation
3. **Given** ARD results, **When** computing delta metrics (ARD-A vs A-N), **Then** reversibility percentages are calculated and reported

---

### User Story 3 - Multi-Prompt Analysis (Priority: P3)

A researcher wants to run the measurement pipeline across a dataset of 60-120 prompts to establish distributional robustness and statistical power.

**Why this priority**: This enables publishable results with proper power analysis. It builds on P1+P2 but requires batch processing and statistical aggregation.

**Independent Test**: Can be tested by loading a prompt dataset, executing all prompts across conditions, and generating summary statistics (mean CCI, confidence intervals, effect sizes).

**Acceptance Scenarios**:

1. **Given** a dataset of 60-120 prompts balanced across capability/reasoning, ambiguity/caution, and opinion/stance categories, **When** the researcher initiates batch processing, **Then** all prompts are executed across all conditions with progress tracking
2. **Given** completed batch results, **When** the researcher requests statistical analysis, **Then** bootstrap confidence intervals, effect sizes, and power analysis are computed
3. **Given** aggregated results, **When** the researcher generates visualizations, **Then** publication-ready plots (context deformation radar, layer×context heatmaps, entropy-over-time curves) are created with error bars

---

### User Story 4 - Baseline Comparison (Priority: P4)

A researcher wants to compare internal deformation metrics against simple output-level baselines (keyword detection, aggregate logit entropy) to demonstrate added value.

**Why this priority**: This validates the measurement's scientific contribution beyond trivial signals. It's important for publication but not essential for initial measurement capability.

**Independent Test**: Can be tested by running baseline metrics on the same data, comparing separation quality (e.g., Cohen's d) between internal metrics vs. baselines, and verifying internal metrics show clearer context sensitivity.

**Acceptance Scenarios**:

1. **Given** experiment results with internal metrics, **When** the researcher computes keyword baselines (agreement phrases, hedging, refusal markers), **Then** baseline scores are calculated for each context
2. **Given** both internal and baseline metrics, **When** the researcher compares effect sizes, **Then** a quantitative comparison shows whether internal metrics better predict context sensitivity
3. **Given** baseline comparison results, **When** generating figures, **Then** a baseline comparison plot is created showing relative performance

---

### Edge Cases

- What happens when a model generates extremely short responses (< 5 tokens)? → Statistics are still computed but flagged for low-sample warnings
- How does the system handle out-of-memory errors when logging statistics for very long sequences (> 2048 tokens)? → Sequence is truncated or sampled at fixed intervals with documentation
- What happens when layer indices specified don't exist for a given model? → Validation error before execution with clear guidance on valid layer ranges
- How does the system handle inference failures (model crash, API timeout)? → Failed runs are logged separately, experiment can resume from checkpoint
- What happens when prompt datasets contain duplicates or empty strings? → Duplicates are flagged and optionally deduplicated; empty prompts are rejected during validation

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST execute prompts under three controlled context conditions: Neutral (N), Audited (A), and Audited + Reasoning Discipline (ARD)
- **FR-002**: System MUST log per-token execution statistics for selected layers: logit entropy, top-1 probability, residual stream norm, activation variance proxy, attention entropy, cosine similarity to previous hidden state
- **FR-003**: System MUST sample statistics from early, mid, and late layers (configurable layer indices)
- **FR-004**: System MUST compute Computation Compression Index (CCI) from logged statistics to measure relative reduction in internal variability
- **FR-005**: System MUST compute Exploration Half-Life (EHL) to measure how quickly logit entropy collapses
- **FR-006**: System MUST compute Trajectory Predictability (TP) by fitting a linear predictor on hidden-state deltas
- **FR-007**: System MUST generate Oversight Sensitivity Score (OSS) as a weighted combination of ΔCCI, ΔEHL, and ΔTP
- **FR-008**: System MUST accept experiment configurations specifying model identifiers, layer indices, random seeds, and prompt datasets
- **FR-009**: System MUST version-control all experiment configurations and archive them alongside results
- **FR-010**: System MUST export results in structured format (JSON) containing raw statistics, computed metrics, and metadata
- **FR-011**: System MUST support batch processing of 60-120 prompts with progress tracking and checkpoint recovery
- **FR-012**: System MUST compute bootstrap confidence intervals for metrics across prompt sets
- **FR-013**: System MUST generate publication-ready visualizations: context deformation radar plots, layer×context heatmaps, entropy-over-time curves, intervention comparison plots, baseline comparison plots
- **FR-014**: System MUST implement keyword baseline comparisons (agreement phrases, hedging language, refusal markers)
- **FR-015**: System MUST implement output-only baseline using aggregate logit entropy without layer statistics
- **FR-016**: System MUST perform power analysis by subsampling prompts (N=20, 40, 60) and estimating effect detection probability
- **FR-017**: System MUST document dual-use risks and limitations in all generated reports
- **FR-018**: System MUST validate prompts before execution (reject empty strings, flag duplicates)
- **FR-019**: System MUST handle model-specific configurations for different model sizes (7-8B and 1-3B parameter ranges)
- **FR-020**: System MUST archive model snapshots and inference configurations to ensure reproducibility

### Assumptions

- Models are accessed via local inference or API with sufficient quota for batch processing
- Execution statistics can be extracted from model internals (requires model architecture support)
- Researchers have Python environment with ML libraries available
- Prompt datasets are provided in structured format (CSV or JSON)
- Visualization output is consumed via static image files or notebooks
- Statistical analysis expertise exists to interpret bootstrap CIs and effect sizes
- Computing resources allow for multiple inference passes per prompt (3 contexts × N prompts)

### Key Entities

- **Experiment Configuration**: Specifies model identifier, layer indices to sample, random seed, context templates (N/A/ARD), prompt dataset path, output directory, hyperparameters
- **Execution Statistics**: Per-token measurements collected during inference: logit entropy, top-1 probability, residual stream norm, activation variance, attention entropy, hidden state cosine similarity; linked to specific prompt, context, layer, and token position
- **Metric Results**: Computed values derived from statistics: CCI (Computation Compression Index), EHL (Exploration Half-Life), TP (Trajectory Predictability), OSS (Oversight Sensitivity Score); linked to prompt and context comparison (e.g., A vs N)
- **Prompt Dataset**: Collection of prompts balanced across categories (capability/reasoning, ambiguity/caution, opinion/stance) with metadata for stratification
- **Baseline Metric**: Simple output-level measurement for comparison: keyword counts, aggregate entropy; computed on same data as internal metrics
- **Visualization Artifact**: Publication-ready plot with source data archived: radar plot, heatmap, time-series curve, comparison chart; linked to specific experiment run

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Researcher can execute a single prompt through all three contexts (N, A, ARD) and receive complete execution statistics within 5 minutes per prompt for a 7-8B parameter model
- **SC-002**: System computes CCI, EHL, TP, and OSS metrics with mathematically correct formulas verified by unit tests achieving 100% pass rate
- **SC-003**: Batch processing completes 60 prompts across 3 contexts (180 total runs) without manual intervention, with automatic checkpoint recovery if interrupted
- **SC-004**: Generated visualizations meet publication standards: axis labels with units, error bars or confidence regions, colorblind-friendly palettes, reproducible via scripts with archived source data
- **SC-005**: Statistical analysis produces bootstrap confidence intervals with configurable alpha level (default 0.05) and reports effect sizes (Cohen's d) alongside p-values
- **SC-006**: Baseline comparison demonstrates internal metrics outperform output-only baselines in context separation quality (measured by effect size ratio > 1.2)
- **SC-007**: Reproducibility validation succeeds: re-running an experiment with the same config and random seed produces bit-identical metric values
- **SC-008**: Power analysis shows 80% probability of detecting a positive effect with N=60 prompts at medium effect size (Cohen's d = 0.5)
- **SC-009**: All experiment outputs include explicit dual-use risk documentation and limitations section as verified by automated checklist
- **SC-010**: System handles model inference failures gracefully with clear error messages and preserves partial results for completed runs
- **SC-011**: Documentation enables an external researcher to reproduce results from archived configs and datasets without requiring author support

### Assumptions

- "Within 5 minutes per prompt" assumes GPU-accelerated inference; CPU-only execution may be slower but is acceptable if progress tracking works
- "Colorblind-friendly palettes" uses standard colorblind-safe color schemes (e.g., viridis, colorblind-friendly categorical palettes)
- "Mathematically correct formulas" verified against independently implemented reference implementations or published papers
- "Publication standards" defined by conference submission guidelines (e.g., NeurIPS, ICML visualization requirements)
