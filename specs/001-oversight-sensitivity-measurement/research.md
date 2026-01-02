# Research: Oversight Sensitivity Measurement

**Phase**: 0 (Technology Research)
**Date**: 2025-12-31
**Purpose**: Resolve technology choices and establish best practices for implementation

## Research Questions

### RQ1: Transformer Library Choice

**Question**: Which library should we use for model loading and internal statistics extraction: `transformers`, `vLLM`, or `llama.cpp`?

**Context**: We need to:
- Load 7-8B and 1-3B parameter models
- Register hooks to extract per-token statistics (logit entropy, hidden states, attention)
- Sample statistics from specific layers (early/mid/late)
- Support local inference with GPU acceleration
- Maintain reproducibility with fixed seeds

**Decision**: **HuggingFace `transformers` library**

**Rationale**:

1. **Hook Support**: `transformers` provides native hook registration via PyTorch's `register_forward_hook` on model modules, enabling clean extraction of intermediate activations without modifying model code

2. **Layer Access**: Models expose named modules (e.g., `model.layers[i]`) making it straightforward to sample early/mid/late layers by index

3. **Statistic Extraction**: Direct access to:
   - Logits → `torch.nn.functional.softmax` + entropy computation
   - Hidden states → available in hook outputs
   - Attention weights → `output_attentions=True` in forward pass
   - Residual stream → layer outputs before normalization

4. **Reproducibility**: `torch.manual_seed()` + `transformers.set_seed()` provides deterministic generation when combined with fixed generation configs

5. **Model Availability**: Pre-trained models directly accessible (e.g., `meta-llama/Llama-2-7b-hf`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

6. **Research Standard**: Widely used in interpretability research, ensuring compatibility with future work and external reproduction

**Alternatives Considered**:

- **vLLM**: Optimized for throughput/serving but abstracts internal states, making hook-based statistics extraction difficult. Better for production deployment, not research measurement.

- **llama.cpp**: C++ implementation excellent for inference efficiency but limited Python introspection of internal states. Would require custom modifications to expose statistics.

**Implementation Notes**:
- Use `model.set_to_eval_mode()` to disable dropout for deterministic behavior
- Set `use_cache=False` during generation to avoid memory overhead
- Implement hooks as context managers to ensure cleanup after each run
- Validate layer indices against model config before execution

---

### RQ2: Statistical Library Choice

**Question**: Which library should we use for statistical analysis: `statsmodels` or `scikit-learn`?

**Context**: We need to:
- Compute bootstrap confidence intervals for metrics
- Perform power analysis (subsample prompts, estimate detection probability)
- Calculate effect sizes (Cohen's d)
- Fit simple linear predictors for Trajectory Predictability metric
- Generate statistical reports with p-values and CIs

**Decision**: **Use both with clear role separation: `scipy` for core statistics, `scikit-learn` for linear models, custom implementation for bootstrap**

**Rationale**:

1. **Scipy for Core Statistics**:
   - `scipy.stats` provides effect size calculations, hypothesis tests, and distribution functions
   - `scipy.stats.ttest_ind` for between-condition comparisons
   - Lightweight, already in ML stack, no additional complexity

2. **Scikit-learn for Linear Predictors**:
   - Trajectory Predictability (TP) requires fitting linear model for hidden state deltas
   - `sklearn.linear_model.LinearRegression` is well-tested and provides `.score()` for R²
   - Familiar API for ML researchers

3. **Custom Bootstrap Implementation**:
   - Neither library provides the specific bootstrap resampling we need (subsample prompts at N=20, 40, 60)
   - Custom implementation using `numpy.random.choice` with replacement is approximately 30 lines
   - Avoids heavyweight dependency (`statsmodels`) for single-use case
   - Full control over resampling strategy per constitution (reproducibility)

**Alternatives Considered**:

- **statsmodels alone**: Comprehensive statistical modeling but heavyweight (large dependency tree). Bootstrap functionality is over-engineered for our use case (simple percentile CIs). Would add 50+ transitive dependencies.

- **scikit-learn alone**: Excellent for ML but weak on hypothesis testing and effect sizes. No native bootstrap CI support.

**Implementation Notes**:
- Document bootstrap algorithm explicitly in code comments (percentile method, resampling strategy)
- Use `scipy.stats.bootstrap` (introduced in scipy 1.7) if available for validation, but maintain custom implementation for clarity
- Effect size formulas (Cohen's d) implemented as standalone functions with unit tests against reference values

---

### RQ3: Best Practices for Reproducible ML Experiments

**Question**: What are the best practices for ensuring bit-identical reproducibility in PyTorch model inference?

**Context**: Success criterion SC-007 requires "re-running an experiment with the same config and random seed produces bit-identical metric values."

**Decision**: **Multi-layer seeding strategy + deterministic algorithm enforcement + hardware/software versioning**

**Rationale**:

1. **Seeding Protocol**:
   - Set random_seed in experiment config file
   - Initialize all random number generators: Python random, NumPy, PyTorch CPU and CUDA
   - Use HuggingFace transformers set_seed utility for library-level consistency

2. **Deterministic Algorithms**:
   - Enable PyTorch deterministic mode
   - Set CUDNN deterministic and disable benchmark mode
   - These flags ensure consistent results across runs at cost of some performance

3. **Generation Config Versioning**:
   - Archive exact `GenerationConfig` parameters (top_k, top_p, temperature, max_new_tokens)
   - Document that greedy decoding (temperature=0, top_k=1) is most reproducible
   - Sampling-based decoding (temperature > 0) may have GPU-specific non-determinism

4. **Environment Archival**:
   - Log PyTorch version, CUDA version, transformers version in experiment metadata
   - Include library versions in result JSON for external reproduction
   - Document hardware (GPU model) as some operations vary by architecture

5. **Floating-Point Precision**:
   - Use `torch.float32` consistently (not mixed precision) for bit-identical results
   - Document that `torch.float16` or `torch.bfloat16` may introduce non-determinism across hardware

**Alternatives Considered**:

- **Seeding only Python RNG**: Insufficient - PyTorch and NumPy maintain separate RNGs
- **Relying on default determinism**: PyTorch uses non-deterministic algorithms by default for performance
- **Ignoring hardware differences**: GPU architecture affects some operations even with deterministic flags

**Implementation Notes**:
- Add reproducibility validation test: run same config twice, assert metric values match to float precision
- Warn users if non-deterministic operations detected (some scatter/gather ops on GPU)
- Provide CPU-only mode as fallback for absolute determinism (slower but guaranteed)

---

### RQ4: Best Practices for Publication-Ready Visualizations in Python

**Question**: What are the best practices for generating publication-ready scientific visualizations that meet conference standards (NeurIPS, ICML)?

**Context**: Success criterion SC-004 requires "axis labels with units, error bars or confidence regions, colorblind-friendly palettes, reproducible via scripts with archived source data."

**Decision**: **Matplotlib with seaborn styling + explicit colorblind palette + source data archival + figure generation scripts**

**Rationale**:

1. **Matplotlib + Seaborn**:
   - Matplotlib is the de facto standard for scientific plotting in Python
   - Seaborn provides colorblind-safe palettes and improved default aesthetics
   - Seaborn colorblind palette ensures accessibility for readers with color vision deficiency

2. **Required Elements Per Plot**:
   - X and Y axis labels with units in square brackets
   - Title with clear description
   - Legend with descriptive labels
   - Error bars with cap marks or shaded confidence regions
   - Font sizes readable in print (14pt for labels, 16pt for titles)

3. **Reproducibility**:
   - Each figure type has a dedicated script in `src/oversight_sensitivity/visualization/`
   - Source data saved alongside figure as JSON
   - Figure generation idempotent: re-running script produces identical output

4. **Export Formats**:
   - PNG at 300 DPI for raster (web/slides)
   - PDF with vector graphics for print (papers)
   - Both formats exported with tight bounding box to remove whitespace

5. **Colorblind Palettes**:
   - Seaborn "colorblind" (8 colors, safe for deuteranopia/protanopia)
   - Paul Tol's palettes for more options
   - Avoid red/green combinations

**Alternatives Considered**:

- **Plotly**: Interactive plots excellent for exploratory analysis but static exports less publication-standard
- **Custom matplotlib styling**: Reinventing wheel; seaborn provides peer-reviewed defaults
- **ggplot (via plotnine)**: R-style grammar of graphics in Python, but less widely adopted in ML community

**Implementation Notes**:
- Create `VisualizationConfig` dataclass to standardize sizes, fonts, colors
- Unit test figure generation scripts with synthetic data
- Include "Figure X generated by script Y using data from Z" captions in reports
- Validate colorblind-safety using online simulators (e.g., Coblis) before publication

---

## Summary of Decisions

| Technology Choice | Decision | Primary Rationale |
|-------------------|----------|-------------------|
| Transformer Library | HuggingFace `transformers` | Native hook support, layer access, reproducibility, research standard |
| Statistical Analysis | `scipy` + `scikit-learn` + custom bootstrap | Role separation: scipy for stats, sklearn for linear models, custom for bootstrap control |
| Reproducibility Strategy | Multi-layer seeding + deterministic algorithms + env versioning | Bit-identical reruns require seeding all RNGs + PyTorch determinism flags + logging software versions |
| Visualization | Matplotlib + Seaborn + colorblind palette | Publication standard, colorblind-safe defaults, script-based reproducibility |

## Dependency List (Final)

**Core ML Stack**:
- `transformers>=4.36.0` (HuggingFace, model loading and generation)
- `torch>=2.1.0` (PyTorch, model execution and hooks)
- `numpy>=1.24.0` (array operations, statistics)
- `scipy>=1.11.0` (statistical tests, effect sizes)
- `scikit-learn>=1.3.0` (linear regression for TP metric)

**Visualization**:
- `matplotlib>=3.8.0` (plotting)
- `seaborn>=0.13.0` (colorblind palettes, improved aesthetics)

**Utilities**:
- `pyyaml>=6.0` (experiment config parsing)
- `tqdm>=4.66.0` (progress bars for batch processing)
- `pytest>=7.4.0` (testing framework)
- `hypothesis>=6.92.0` (property-based testing for metric formulas)

**Development**:
- `nbstripout>=0.6.0` (strip notebook outputs before git commit)
- `black>=23.0.0` (code formatting)
- `ruff>=0.1.0` (fast linting)

**Total**: 13 direct dependencies (lightweight per constitution)

## Next Steps

1. Update `plan.md` Technical Context with resolved dependencies
2. Proceed to Phase 1: Design data model (Experiment Configuration, Execution Statistics, Metric Results entities)
3. Define CLI contracts (commands: `run-single`, `run-batch`, `compute-metrics`, `analyze`, `visualize`)
