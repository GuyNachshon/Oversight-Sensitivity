---

description: "Task list for Oversight Sensitivity Measurement implementation"
---

# Tasks: Oversight Sensitivity Measurement

**Input**: Design documents from `/specs/001-oversight-sensitivity-measurement/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Not explicitly requested in spec.md - implementation tasks only per constitution (experiment-driven, not TDD)

**Organization**: Tasks grouped by user story to enable independent implementation and testing of each story.

**Research Projects**: For ML research (per constitution), prioritize experiment-driven tasks:
- Metric definition tasks BEFORE data collection tasks
- Baseline implementation tasks BEFORE novel method tasks
- Unit tests for metric computations (mathematical correctness)
- Experiment config versioning tasks alongside implementation

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/`, `experiments/`, `results/`, `notebooks/` at repository root
- All paths shown are from repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan.md: src/oversight_sensitivity/{inference,statistics,metrics,experiments,baselines,analysis,visualization,cli}/
- [ ] T002 Create auxiliary directories: experiments/{configs,prompts}/, results/{raw,metrics,figures,reports}/, notebooks/exploratory/, tests/{unit,integration,fixtures}/
- [ ] T003 [P] Update pyproject.toml with dependencies from research.md: transformers>=4.36.0, torch>=2.1.0, numpy>=1.24.0, scipy>=1.11.0, scikit-learn>=1.3.0, matplotlib>=3.8.0, seaborn>=0.13.0, pyyaml>=6.0, tqdm>=4.66.0
- [ ] T004 [P] Add development dependencies to pyproject.toml: pytest>=7.4.0, hypothesis>=6.92.0, nbstripout>=0.6.0, black>=23.0.0, ruff>=0.1.0
- [ ] T005 [P] Create .gitignore with results/raw/ excluded, results/metrics/ and results/figures/ tracked
- [ ] T006 [P] Configure nbstripout for notebooks/exploratory/ to strip outputs before commit
- [ ] T007 [P] Create src/oversight_sensitivity/__init__.py as package entry point
- [ ] T008 [P] Create README.md with project overview and link to specs/001-oversight-sensitivity-measurement/quickstart.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T009 Create ExperimentConfig dataclass in src/oversight_sensitivity/experiments/config.py with fields from data-model.md (experiment_id, model_identifier, random_seed, layer_indices, context_conditions, etc.)
- [ ] T010 [P] Create PromptDataset dataclass in src/oversight_sensitivity/experiments/dataset.py with CSV/JSON loading functions
- [ ] T011 [P] Create ExecutionRun dataclass in src/oversight_sensitivity/experiments/run.py with environment metadata (pytorch_version, cuda_version, etc.)
- [ ] T012 [P] Create ExecutionStatistics dataclass in src/oversight_sensitivity/statistics/stats.py for per-token measurements
- [ ] T013 Implement config validation in src/oversight_sensitivity/experiments/validation.py: validate layer_indices < model layers, random_seed present, contexts include N and A
- [ ] T014 Implement model loader in src/oversight_sensitivity/inference/model_loader.py: load HuggingFace model, set eval mode, configure deterministic algorithms per research.md RQ3
- [ ] T015 Implement reproducibility setup in src/oversight_sensitivity/inference/reproducibility.py: multi-layer seeding (random, numpy, torch, transformers), deterministic flags, environment logging
- [ ] T016 Implement context template manager in src/oversight_sensitivity/inference/contexts.py: templates for Neutral (N), Audited (A), Audited+Reasoning Discipline (ARD)
- [ ] T017 Implement hook registration in src/oversight_sensitivity/inference/hooks.py: register_forward_hook for specified layers, extract hidden states and attention weights
- [ ] T018 Implement statistics collector in src/oversight_sensitivity/statistics/collector.py: compute logit entropy, top-1 probability, residual stream norm, activation variance, attention entropy, cosine similarity per token
- [ ] T019 Create MetricResults dataclass in src/oversight_sensitivity/metrics/results.py with fields from data-model.md (cci, ehl, tp_score, oss, baseline scores)
- [ ] T020 Create VisualizationArtifact dataclass in src/oversight_sensitivity/visualization/artifact.py with file paths and source data tracking

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Core Measurement Pipeline (Priority: P1) 🎯 MVP

**Goal**: Execute single prompt through neutral and audited contexts, collect execution statistics, compute CCI metric, export results

**Independent Test**: Run oversee run-single with test prompt → verify ExecutionRun JSON contains 6 statistics per token for 3 layers → compute CCI → verify numerical value returned

### Implementation for User Story 1

- [ ] T021 [P] [US1] Implement metric formula for CCI in src/oversight_sensitivity/metrics/cci.py: CCI = 1 - (variance_A / variance_N) where variance is mean activation_variance
- [ ] T022 [P] [US1] Implement metric formula for EHL in src/oversight_sensitivity/metrics/ehl.py: find token position where logit_entropy drops to 50% of initial via interpolation
- [ ] T023 [P] [US1] Implement metric formula for TP in src/oversight_sensitivity/metrics/tp.py: fit linear regression Δh_{t+1} ≈ W Δh_t, return R²
- [ ] T024 [P] [US1] Implement metric formula for OSS in src/oversight_sensitivity/metrics/oss.py: OSS = (ΔCCI + ΔEHL + ΔTP) / 3 with equal weights
- [ ] T025 [US1] Implement single prompt executor in src/oversight_sensitivity/experiments/single_run.py: load model, apply context template, generate with hooks, collect statistics, save ExecutionRun JSON
- [ ] T026 [US1] Implement metric computation orchestrator in src/oversight_sensitivity/metrics/compute.py: load ExecutionRuns for same prompt across contexts, compute all metrics (CCI, EHL, TP, OSS), save MetricResults JSON
- [ ] T027 [US1] Implement CLI command oversee run-single in src/oversight_sensitivity/cli/run_single.py: parse args, validate config, call single_run executor, output JSON to stdout
- [ ] T028 [US1] Implement CLI command oversee compute-metrics in src/oversight_sensitivity/cli/compute_metrics.py: load experiment dir, compute metrics for all prompts, save results
- [ ] T029 [US1] Implement CLI command oversee validate-config in src/oversight_sensitivity/cli/validate_config.py: run validation.py checks, output ✓/✗ results
- [ ] T030 [US1] Create CLI entry point in src/oversight_sensitivity/cli/__main__.py: subcommand dispatcher for run-single, compute-metrics, validate-config
- [ ] T031 [US1] Add unit tests for metric formulas in tests/unit/test_cci.py, test_ehl.py, test_tp.py, test_oss.py using hypothesis for property-based testing (e.g., CCI in valid range)
- [ ] T032 [US1] Create integration test fixtures in tests/fixtures/mock_execution_runs.json with synthetic data for metric computation validation
- [ ] T033 [US1] Add integration test in tests/integration/test_single_measurement.py: full pipeline from config → execution → metrics → verify output structure

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently (MVP deliverable)

---

## Phase 4: User Story 2 - Intervention Analysis (Priority: P2)

**Goal**: Add ARD (Audited + Reasoning Discipline) context condition, test reversibility of oversight-induced deformation

**Independent Test**: Run prompt through N, A, ARD contexts → compute metrics for A_vs_N and ARD_vs_A pairs → verify ARD metrics shift toward neutral

### Implementation for User Story 2

- [ ] T034 [US2] Extend context template manager in src/oversight_sensitivity/inference/contexts.py: add ARD template with reasoning discipline instructions from PRD section 5.2
- [ ] T035 [US2] Extend metric computation in src/oversight_sensitivity/metrics/compute.py: support context_pairs parameter (A_vs_N, ARD_vs_A, ARD_vs_N)
- [ ] T036 [US2] Implement reversibility analysis in src/oversight_sensitivity/analysis/reversibility.py: compute delta metrics (ARD-A vs A-N), calculate reversibility percentages
- [ ] T037 [US2] Update CLI oversee run-single to accept ARD in --contexts argument
- [ ] T038 [US2] Update CLI oversee compute-metrics to accept --context-pairs A_vs_N,ARD_vs_A,ARD_vs_N
- [ ] T039 [US2] Add integration test in tests/integration/test_intervention.py: run prompt through all 3 contexts, verify ARD shifts metrics toward N

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Multi-Prompt Analysis (Priority: P3)

**Goal**: Batch process 60-120 prompts with progress tracking, checkpointing, statistical analysis (bootstrap CIs, power analysis), publication-ready visualizations

**Independent Test**: Load 60-prompt dataset → run oversee run-batch → verify all prompts executed → generate statistical summary with CIs → create visualizations

### Implementation for User Story 3

- [ ] T040 [P] [US3] Implement checkpoint manager in src/oversight_sensitivity/experiments/checkpoint.py: save/load completed prompt IDs, enable resume after interruption
- [ ] T041 [P] [US3] Implement batch executor in src/oversight_sensitivity/experiments/batch_run.py: iterate dataset, track progress with tqdm, update checkpoint after each prompt, handle failures gracefully
- [ ] T042 [P] [US3] Implement bootstrap CI computation in src/oversight_sensitivity/analysis/bootstrap.py: custom resampling with numpy.random.choice, percentile CIs per research.md RQ2
- [ ] T043 [P] [US3] Implement effect size calculation in src/oversight_sensitivity/analysis/effect_sizes.py: Cohen's d using scipy.stats, hypothesis tests with p-values
- [ ] T044 [P] [US3] Implement power analysis in src/oversight_sensitivity/analysis/power.py: subsample prompts at N=20,40,60, estimate detection probability via bootstrap
- [ ] T045 [US3] Implement CLI command oversee run-batch in src/oversight_sensitivity/cli/run_batch.py: parse dataset, call batch executor, display progress, output summary
- [ ] T046 [US3] Implement CLI command oversee analyze in src/oversight_sensitivity/cli/analyze.py: load metrics, compute bootstrap CIs, effect sizes, power analysis, save analysis JSON
- [ ] T047 [P] [US3] Implement radar plot generator in src/oversight_sensitivity/visualization/radar.py: context deformation across CCI/EHL/TP with error bars, colorblind palette per research.md RQ4
- [ ] T048 [P] [US3] Implement heatmap generator in src/oversight_sensitivity/visualization/heatmap.py: layer × context metric values with seaborn
- [ ] T049 [P] [US3] Implement timeseries plot generator in src/oversight_sensitivity/visualization/timeseries.py: entropy-over-time curves for same prompt across contexts
- [ ] T050 [P] [US3] Implement intervention plot generator in src/oversight_sensitivity/visualization/intervention.py: ARD vs A vs N comparison with confidence regions
- [ ] T051 [US3] Implement CLI command oversee visualize in src/oversight_sensitivity/cli/visualize.py: load metrics and analysis, generate requested figure types, save PNG/PDF/JSON
- [ ] T052 [US3] Create VisualizationConfig dataclass in src/oversight_sensitivity/visualization/config.py: standardize DPI, font sizes, colors, dimensions per research.md RQ4
- [ ] T053 [US3] Add integration test in tests/integration/test_batch_pipeline.py: run 3-prompt dataset through full batch → metrics → analysis → visualize workflow

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Baseline Comparison (Priority: P4)

**Goal**: Implement keyword and output-only baselines, compare with internal metrics to demonstrate added value

**Independent Test**: Compute keyword baseline (agreement phrases) and output-only baseline (aggregate entropy) on same data → compare effect sizes → verify internal metrics show stronger separation

### Implementation for User Story 4

- [ ] T054 [P] [US4] Implement keyword baseline in src/oversight_sensitivity/baselines/keyword.py: detect agreement phrases, hedging language, refusal markers, compute scores
- [ ] T055 [P] [US4] Implement output-only baseline in src/oversight_sensitivity/baselines/output_entropy.py: aggregate logit entropy without layer statistics
- [ ] T056 [US4] Extend metric computation in src/oversight_sensitivity/metrics/compute.py: compute baselines alongside internal metrics unless --skip-baselines flag
- [ ] T057 [US4] Implement baseline comparison in src/oversight_sensitivity/analysis/baseline_comparison.py: compute effect sizes for internal vs baseline, calculate ratio
- [ ] T058 [P] [US4] Implement baseline comparison plot in src/oversight_sensitivity/visualization/baseline_plot.py: side-by-side effect size comparison
- [ ] T059 [US4] Update CLI oversee compute-metrics to support --skip-baselines flag
- [ ] T060 [US4] Update CLI oversee analyze to include baseline comparison in output
- [ ] T061 [US4] Add integration test in tests/integration/test_baselines.py: verify baselines computed, comparison shows internal > baseline

**Checkpoint**: All user stories complete, ready for polish phase

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories, documentation, dual-use disclaimers

- [ ] T062 [P] Implement report generator in src/oversight_sensitivity/cli/report.py: create markdown/PDF with sections (summary, methodology, results, limitations, dual-use disclaimer per FR-017)
- [ ] T063 [P] Add dual-use disclaimer template in src/oversight_sensitivity/reporting/dual_use_template.md: standard text per constitution principle IV
- [ ] T064 [P] Add limitations section template in src/oversight_sensitivity/reporting/limitations_template.md: model-specific effects, confounds, scope boundaries
- [ ] T065 Implement CLI command oversee report: load metrics/analysis/figures, populate templates, output report with mandatory disclaimers
- [ ] T066 [P] Create example experiment config in experiments/configs/example.json matching quickstart.md
- [ ] T067 [P] Create example prompt dataset in experiments/prompts/example.csv with 3 balanced prompts
- [ ] T068 [P] Add logging configuration in src/oversight_sensitivity/utils/logging.py: structured logs to stderr, summaries to stdout
- [ ] T069 [P] Add error handling utilities in src/oversight_sensitivity/utils/errors.py: custom exceptions for validation, inference, metric computation failures
- [ ] T070 Validate quickstart.md by running its examples: verify all commands work, outputs match expected
- [ ] T071 [P] Add reproducibility validation test in tests/integration/test_reproducibility.py: run same config twice, assert metrics match to float precision (SC-007)
- [ ] T072 [P] Add unit tests for reproducibility setup in tests/unit/test_reproducibility.py: verify all RNGs seeded, deterministic flags set
- [ ] T073 [P] Add property-based tests for metric ranges in tests/unit/test_metric_properties.py: CCI valid range, TP in [0,1], EHL > 0
- [ ] T074 Code review for constitution compliance: verify no hardcoded credentials, metrics defined before execution, dual-use disclaimers present
- [ ] T075 Performance profiling: verify 5 min/prompt goal on 7-8B model, identify bottlenecks if slower

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Extends US1 but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Uses US1/US2 metrics but independently testable
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Compares with US1 but independently testable

### Within Each User Story

- Metric formula tasks (T021-T024) can run in parallel - different files
- US1: Formulas → Single executor → CLI commands → Tests (sequential within groups)
- US2: Template → Metric extension → CLI updates → Tests
- US3: Checkpoint/Batch/Analysis/Viz tasks [P] → CLI commands → Tests
- US4: Baselines [P] → Comparison → CLI updates → Tests

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Within US1: Tasks T021-T024 (metric formulas) in parallel
- Within US1: Tasks T031-T032 (test files) in parallel
- Within US3: Tasks T040-T044 (analysis modules) in parallel
- Within US3: Tasks T047-T050 (visualization modules) in parallel
- Within US4: Tasks T054-T055 (baseline modules) in parallel
- Within Polish: Tasks T062-T069 (utilities and templates) in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all metric formula implementations together:
Task: "[US1] Implement CCI formula in src/oversight_sensitivity/metrics/cci.py"
Task: "[US1] Implement EHL formula in src/oversight_sensitivity/metrics/ehl.py"
Task: "[US1] Implement TP formula in src/oversight_sensitivity/metrics/tp.py"
Task: "[US1] Implement OSS formula in src/oversight_sensitivity/metrics/oss.py"
```

---

## Parallel Example: User Story 3

```bash
# Launch all analysis modules together:
Task: "[US3] Implement bootstrap CI in src/oversight_sensitivity/analysis/bootstrap.py"
Task: "[US3] Implement effect sizes in src/oversight_sensitivity/analysis/effect_sizes.py"
Task: "[US3] Implement power analysis in src/oversight_sensitivity/analysis/power.py"

# Launch all visualization generators together:
Task: "[US3] Implement radar plot in src/oversight_sensitivity/visualization/radar.py"
Task: "[US3] Implement heatmap in src/oversight_sensitivity/visualization/heatmap.py"
Task: "[US3] Implement timeseries plot in src/oversight_sensitivity/visualization/timeseries.py"
Task: "[US3] Implement intervention plot in src/oversight_sensitivity/visualization/intervention.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T008)
2. Complete Phase 2: Foundational (T009-T020) → CRITICAL - blocks all stories
3. Complete Phase 3: User Story 1 (T021-T033)
4. **STOP and VALIDATE**: Test User Story 1 independently using quickstart.md single measurement example
5. Verify MVP deliverable: single prompt → statistics → CCI metric

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (T021-T033) → Test independently → Deliverable: Core measurement MVP
3. Add User Story 2 (T034-T039) → Test independently → Deliverable: Intervention capability
4. Add User Story 3 (T040-T053) → Test independently → Deliverable: Publication-ready analysis
5. Add User Story 4 (T054-T061) → Test independently → Deliverable: Baseline validation
6. Add Polish (T062-T075) → Final deliverable: Complete research system

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Core Measurement)
   - Developer B: User Story 2 (Intervention) - minimal overlap with US1
   - Developer C: User Story 3 (Batch/Analysis/Viz) - uses US1 outputs but independent
   - Developer D: User Story 4 (Baselines) - independent comparison
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- No test tasks included (not requested in spec.md, experiment-driven per constitution)
- Unit tests only for metric formula correctness (mathematical verification)
- Integration tests verify end-to-end pipeline per story
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Constitution compliance: metrics defined (T021-T024) before execution (T025), dual-use disclaimers (T062-T065), reproducibility (T015, T071-T072)

---

## Task Count Summary

- **Total Tasks**: 75
- **Setup (Phase 1)**: 8 tasks
- **Foundational (Phase 2)**: 12 tasks
- **User Story 1 (Phase 3)**: 13 tasks
- **User Story 2 (Phase 4)**: 6 tasks
- **User Story 3 (Phase 5)**: 14 tasks
- **User Story 4 (Phase 6)**: 8 tasks
- **Polish (Phase 7)**: 14 tasks

**Parallel Opportunities**: 42 tasks marked [P] (56% parallelizable)

**MVP Scope**: Phases 1-3 = 33 tasks for core measurement capability
