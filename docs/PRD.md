# PRD

## Project: Oversight Sensitivity in LLM Computation

**Subtitle (working):** Measuring how evaluative context reshapes internal reasoning dynamics

---

## 1. Motivation & Problem Statement

### 1.1 The problem the hackathon is actually about

Modern LLMs increasingly interact with humans in settings where:

* they are evaluated,
* audited,
* safety-checked,
* or socially incentivized (authority, approval, alignment).

There is growing evidence that models **adapt their behavior under such conditions**:

* Sandbagging: underperforming during evaluation while retaining capability
* Sycophancy: adapting responses to user beliefs or authority
* Oversight evasion: behaving differently when “being watched”

Crucially, **these behaviors are not reliably detectable at the output level**:

* Outputs may remain correct, safe, or compliant
* Standard benchmarks miss strategic adaptation
* Keyword- or refusal-based detectors conflate many unrelated effects

The hackathon highlights this gap explicitly:

> *“Our benchmarks miss strategic behavior. We lack tools to measure it honestly.”*

---

### 1.2 Why existing approaches are insufficient

Current approaches typically assume one of the following:

* Manipulation is an *output anomaly*
* Manipulation corresponds to *semantic features* (e.g., lies, agreement)
* Manipulation is *intentional deception* and should be classified as such

These assumptions are problematic because:

* Strategic behavior can be internally coherent and output-stable
* Advanced models may adapt *how they reason*, not *what they say*
* Intent is unobservable and philosophically loaded

**We need a measurement that operates below outputs, without assuming deception.**

---

## 2. Core Idea

### 2.1 Key insight

> **Evaluative and social context acts as a continuous control signal that reshapes internal computation in LLMs.**

Rather than switching between discrete “modes” or violating a single “honest prior,” models appear to:

* Regularize reasoning under oversight
* Reduce exploration and internal variance
* Become more predictable and conservative
* Commit earlier to safe or agreeable trajectories

This reshaping can occur **even when outputs are unchanged**.

---

### 2.2 What this project proposes

We propose to **measure and characterize context-conditioned deformation of internal computation**, using lightweight execution statistics and simple dynamics modeling.

We do **not** claim to:

* detect deception
* infer intent
* identify manipulation directly

Instead, we introduce a **new measurement axis**:

> **Oversight Sensitivity** — how strongly a model’s internal reasoning dynamics change in response to evaluative or social context.

This is manipulation-relevant because:

* Sandbagging, sycophancy, and evasion *require* such sensitivity
* Benchmarks that ignore it are vulnerable by construction

---

## 3. Hypotheses

### 3.1 Primary hypothesis (H1)

**H1:**
Evaluative context (e.g., audit framing) systematically compresses and regularizes internal computation relative to neutral context, even when outputs remain stable.

**Operational expectations:**

* Reduced activation variance
* Faster collapse of uncertainty
* More predictable internal trajectories

---

### 3.2 Intervention hypothesis (H2)

**H2:**
Explicit reasoning discipline instructions partially counteract oversight-induced computation deformation.

This suggests deformation is not merely surface-level compliance, but reflects **competition between internal priors**.

---

### 3.3 Mechanistic hypothesis (speculative but testable)

**H3 (mechanistic):**
Contextual cues activate competing policy priors (e.g., “oversight safety” vs “task-solving”), which bias the geometry of internal computation rather than triggering discrete mode switches.

**Predictions:**

* Effects are strongest in mid/late layers (policy shaping)
* Effects are continuous, not binary
* Intervention shifts metrics toward neutral, not beyond it

---

## 4. What This Is *Not*

Explicit non-goals (important for reviewer alignment):

* Not a deception detector
* Not a sycophancy classifier
* Not a neuron-level interpretability study
* Not a training-time analysis
* Not a full world model of reasoning

This is a **measurement instrument**, not a moral classifier.

---

## 5. Experimental Design

### 5.1 Independent variable: Context framing

Each prompt is run under **three controlled contexts**:

1. **Neutral (N)**
   Minimal instructions; solve the task.

2. **Audited / Evaluated (A)**
   Emphasizes oversight, review, and caution.

3. **Audited + Reasoning Discipline (ARD)** *(intervention)*
   Same as A, plus explicit reasoning discipline:

   * prioritize correctness
   * perform internal checks
   * avoid user-pleasing
   * output only final answer

This isolates:

* existence of deformation (A vs N)
* reversibility / control (ARD vs A)

---

### 5.2 Dependent variables: Internal execution statistics

We log **summary statistics**, not full activations, per generated token.

#### Logged signals (per token, selected layers)

* Logit entropy
* Top-1 probability
* Residual stream norm
* Activation variance proxy
* Attention entropy
* Cosine similarity to previous hidden state

Layers sampled: early / mid / late (e.g., 3 total).

---

### 5.3 Models

* One mid-size instruct model (7–8B)
* One smaller instruct model (1–3B)

This allows:

* sanity checking
* scaling comparison
* sensitivity contrast

---

## 6. Metrics (Defined *Before* Running)

### 6.1 Computation Compression Index (CCI)

Measures relative reduction in internal variability under context.

Interpretation:

* Higher CCI → tighter, more constrained computation

---

### 6.2 Exploration Half-Life (EHL)

Measures how quickly uncertainty (logit entropy) collapses.

Interpretation:

* Lower EHL → earlier commitment / reduced exploration

---

### 6.3 Trajectory Predictability (TP)

Fits a simple linear predictor on hidden-state deltas:
[
\Delta h_{t+1} \approx W \Delta h_t
]

Interpretation:

* Lower prediction error → more predictable dynamics

---

### 6.4 Composite: Oversight Sensitivity Score (OSS)

Weighted combination of:

* ΔCCI (A–N)
* ΔEHL (A–N)
* ΔTP (A–N)

Used only for summarization, not detection claims.

---

## 7. Datasets

### 7.1 Core dataset (synthetic, controlled)

* **60–120 prompts**
* Balanced across:

  * capability / reasoning
  * ambiguity / caution
  * opinion / stance

Used for:

* main plots
* intervention tests
* mechanistic analysis

---

### 7.2 Scale layer (optional, if time allows)

* 300–600 prompts
* Neutral vs Audited only
* Used for:

  * distributional robustness
  * power analysis

---

### 7.3 External anchor (optional)

* WildChat slice (200–500 turns)
* Used only to show:

  > deformation metrics appear in real deployment language

No labeling, no claims of manipulation.

---

## 8. Baselines

To demonstrate value beyond trivial signals:

1. **Keyword baseline**

   * agreement phrases
   * hedging language
   * refusal markers

2. **Output-only baseline**

   * aggregate logit entropy
   * no internal layer stats

We evaluate whether internal deformation metrics:

* better predict context sensitivity
* show clearer separation across conditions

---

## 9. Power & Detectability

We include a **bootstrap-based power analysis**:

* Estimate effect size from pilot runs
* Subsample prompts (N = 20, 40, 60)
* Plot probability of detecting a positive effect

This justifies prompt count empirically.

---

## 10. Outputs & Visualizations

Minimum required figures:

1. **Context deformation radar plots** (per model)
2. **Layer × context heatmaps**
3. **Entropy-over-time curves** (same prompt, different contexts)
4. **Intervention comparison plot** (A vs ARD)
5. **Baseline comparison plot**

These figures *are* the result.

---

## 11. Deliverables

* Public GitHub repository
* Reproducible logging + metrics pipeline
* Short research report (methods + plots)
* Limitations & dual-use appendix
* Optional demo video

---

## 12. Limitations & Ethics (Explicit)

We explicitly acknowledge:

* No access to intent
* Possible confounds (verbosity, decoding)
* Model-specific effects
* Risk of dual-use (training models to evade oversight metrics)

This strengthens credibility.

---

## 13. Success Criteria

The project is successful if we can show:

1. **Existence:** Oversight framing measurably deforms computation.
2. **Control:** Reasoning discipline partially reverses deformation.
3. **Value:** Internal metrics outperform simple output baselines.
4. **Relevance:** Effects appear in at least one real-world dataset slice.

No stronger claims are needed.

---

## 14. Why This Fits the Hackathon

* Directly targets sandbagging / oversight sensitivity
* Introduces a new measurement primitive
* Avoids output-level traps
* Empirical, falsifiable, scoped
* Bold but disciplined