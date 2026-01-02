# Implementation Plan: Oversight Sensitivity Measurement

**Branch**: `001-oversight-sensitivity-measurement` | **Date**: 2025-12-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-oversight-sensitivity-measurement/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature builds a measurement system to quantify how evaluative context (audit framing) reshapes internal reasoning dynamics in LLMs. The system executes prompts under three controlled conditions (Neutral, Audited, Audited+Reasoning Discipline), logs lightweight execution statistics per token (logit entropy, top-1 probability, residual stream norm, activation variance, attention entropy, hidden state cosine similarity), and computes four core metrics: Computation Compression Index (CCI), Exploration Half-Life (EHL), Trajectory Predictability (TP), and Oversight Sensitivity Score (OSS). The technical approach emphasizes reproducibility (fixed seeds, versioned configs, model archival), statistical rigor (bootstrap CIs, power analysis, baseline comparisons), and lightweight efficiency (summary statistics, not full activations). Deliverables include a CLI-first pipeline for single/batch prompt execution, statistical analysis tools, publication-ready visualizations, and explicit dual-use risk documentation.

## Technical Context

**Language/Version**: Python 3.13 (per project pyproject.toml)
**Primary Dependencies**: transformers>=4.36.0 (model loading/hooks), torch>=2.1.0 (inference), numpy>=1.24.0, scipy>=1.11.0 (statistics), scikit-learn>=1.3.0 (linear models), matplotlib>=3.8.0, seaborn>=0.13.0 (visualization), pyyaml>=6.0 (configs)
**Storage**: File system (JSON for results/configs; CSV/JSON for prompts; PNG/PDF for figures)
**Testing**: pytest>=7.4.0 with hypothesis>=6.92.0 for property-based testing of metric formulas
**Target Platform**: Linux/macOS (local research environment with GPU access preferred)
**Project Type**: Single project (research CLI + notebooks)
**Performance Goals**: 5 minutes per prompt for 7-8B model (GPU), batch 60 prompts in < 6 hours
**Constraints**: GPU memory < 24GB VRAM, summary statistics only (no full activation storage), colorblind-safe visualizations
**Scale/Scope**: 60-120 prompts for core experiments, 2 model sizes (7-8B, 1-3B), 3 context conditions, 6 statistics × 3 layers per token

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Research integrity gates per `.specify/memory/constitution.md`:

- [x] **Reproducibility**: Are all metrics defined with exact formulas before execution? → YES (spec.md FR-004 through FR-007 define CCI, EHL, TP, OSS)
- [ ] **Reproducibility**: Are random seeds documented in experiment configs? → PENDING (will be specified in Phase 1 experiment config schema)
- [x] **Falsifiability**: Are hypotheses stated as falsifiable predictions with failure conditions? → YES (PRD section 3 defines H1, H2, H3 with operational predictions)
- [x] **Falsifiability**: Is power analysis included to justify sample sizes? → YES (spec.md FR-016, SC-008)
- [x] **Measurement Over Claims**: Are metrics focused on observable phenomena (not anthropomorphic interpretations)? → YES (CCI measures variability reduction, not "deception")
- [x] **Ethics & Dual-Use**: Is dual-use risk acknowledged in deliverables? → YES (spec.md FR-017, SC-009)
- [x] **Ethics & Dual-Use**: Is limitations section included? → YES (required in all reports per FR-017)
- [x] **Lightweight & Efficient**: Are we using summary statistics instead of full activation dumps? → YES (6 statistics per token, not full hidden states)
- [x] **Experiment-Driven**: Is there clear separation between data collection, metric computation, and visualization? → YES (src/ structure: inference/ → statistics/ → metrics/ → analysis/ → visualization/)
- [x] **Statistical Standards**: Are baseline comparisons defined that could invalidate claims? → YES (spec.md FR-014, FR-015, US4)

**Complexity Justification Required If**:
- Introducing new dependencies beyond core ML stack (transformers, numpy, matplotlib)
- Deviating from CLI-first interface design
- Storing raw activations instead of statistics

**Initial Gate Status**: 9/10 PASS (random seed documentation pending Phase 1 config design)

**Action**: Proceed to Phase 0 research to resolve NEEDS CLARIFICATION items

---

**Post-Phase 1 Re-evaluation**:

- [x] **Reproducibility**: Are random seeds documented in experiment configs? → YES (data-model.md ExperimentConfig.random_seed is required field, validated)

**Final Gate Status**: 10/10 PASS ✓

**Complexity Violations**: NONE (all dependencies within core ML stack per research.md; CLI-first design per contracts/; summary statistics only per data-model.md)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── oversight_sensitivity/
│   ├── inference/          # Model loading, execution, hook registration
│   ├── statistics/         # Per-token stat collection (entropy, norms, etc.)
│   ├── metrics/            # CCI, EHL, TP, OSS computation
│   ├── experiments/        # Config parsing, batch execution, checkpointing
│   ├── baselines/          # Keyword and output-only baseline metrics
│   ├── analysis/           # Bootstrap CIs, power analysis, effect sizes
│   ├── visualization/      # Radar plots, heatmaps, time-series curves
│   └── cli/                # Command-line interface entry points

experiments/
├── configs/                # Versioned experiment configurations (YAML/JSON)
└── prompts/                # Prompt datasets (CSV/JSON)

results/
├── raw/                    # Per-run statistics (JSON, gitignored for large files)
├── metrics/                # Computed metric values (JSON, tracked)
├── figures/                # Publication-ready plots (PNG/PDF, tracked)
└── reports/                # Analysis reports with dual-use disclaimers (MD/PDF)

notebooks/
└── exploratory/            # Jupyter notebooks for ad-hoc analysis (nbstripout)

tests/
├── unit/                   # Metric formula correctness, stat computation
├── integration/            # End-to-end pipeline tests (single prompt → metrics)
└── fixtures/               # Test data (mock model outputs, example configs)
```

**Structure Decision**: Research-oriented single project structure. Key design choices:

1. **Modular pipeline**: Inference → Statistics → Metrics → Analysis → Visualization allows independent testing and CLI composition
2. **Experiment-first organization**: `experiments/` and `results/` at root emphasize reproducibility per constitution
3. **Gitignore strategy**: `results/raw/` excluded (large), `results/metrics/` and `results/figures/` tracked for reproducibility
4. **Notebooks isolated**: `notebooks/` with nbstripout prevents output bloat while enabling exploratory analysis
5. **No database**: File-based storage aligns with lightweight principle and simplifies external reproduction

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
