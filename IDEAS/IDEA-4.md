### What we are doing

**Measuring how evaluative/social context deforms internal computation**, using execution statistics and simple dynamics modeling.

### What we are *not* doing

* Not detecting deception
* Not classifying manipulation
* Not interpreting neurons
* Not claiming intent
* Not training base models

This keeps the project realistic and defensible.

---

## 2. Exact research question (one sentence)

> **How does evaluative and social context continuously regularize and compress internal computation in LLMs, even when outputs remain stable?**

This is the question you answer with plots.

---

## 3. Experimental design (clean and minimal)

### Independent variable

**Context framing**, implemented via system prompts.

Contexts (exactly 4):

1. **Neutral**
2. **Audited / evaluated**
3. **Private / no oversight**
4. **User-satisfaction / agreeable**

(Authority can be added later if time allows.)

### Dependent variables

Internal execution statistics:

* variance
* entropy
* predictability
* collapse speed

### Controlled variables

* Same base prompt
* Same decoding params
* Same max tokens
* Same output format

---

## 4. Data & prompts

### 4.1 Synthetic prompt bank (core)

Create **~60 prompts**, split evenly:

#### A. Capability / reasoning (sandbagging-adjacent)

Examples:

* multi-step math
* logic puzzles
* short algorithmic reasoning
* simple code reasoning (no execution)

Goal: tasks the model *can* solve.

#### B. Ambiguous / caution-inducing

Examples:

* “Is this argument valid?”
* “Explain the risks of X” (non-harmful)
* borderline safety-style phrasing

Goal: contexts where evaluation framing matters.

#### C. Opinion / stance (sycophancy-adjacent)

Examples:

* user states belief, asks for agreement or critique
* confidence calibration tasks

---

### 4.2 WildChat (secondary, optional but recommended)

* Sample **100 user turns**
* Filter for:

  * confident users
  * authority-like language
  * evaluative tone

Use **only** to show your signal appears outside synthetic prompts.

---

## 5. Models & instrumentation

### Models (pick 2)

* One 7–8B instruct model
* One smaller (1–3B) instruct model

Why two:

* show scaling effects
* stronger context sensitivity usually appears in larger models

---

### Instrumentation (important constraint)

You **do not** log full activations.

For **each generated token**, log:

#### Logits / output

* entropy
* top-1 probability

#### Hidden states (3 layers: early / mid / late)

* residual norm
* feature variance summary
* cosine similarity to previous token state

#### Attention (same layers)

* attention entropy (averaged over heads)

Store as JSONL.

This is enough.

---

## 6. Metrics (define them *before* running)

### Metric 1: Computation Compression Index (CCI)

Measures how much computation “tightens” vs neutral.

Interpretation:

* higher = stronger regularization under context

---

### Metric 2: Exploration Half-Life (EHL)

How quickly entropy collapses.

Interpretation:

* lower EHL = earlier commitment / caution

---

### Metric 3: Trajectory Predictability (TP)

Fit a tiny linear model:
[
\Delta h_{t+1} = W \Delta h_t
]

Compute prediction error.
Lower error = more predictable dynamics.

---

### Optional composite score

Context Sensitivity Score (CSS):
Weighted sum of the above deltas vs neutral.

---

## 7. Analysis & plots (these matter more than text)

You want **at least these 4 visuals**:

1. **Context deformation radar chart**

   * axes: CCI, EHL, TP
   * one radar per model

2. **Layer × context heatmap**

   * shows *where* computation tightens

3. **Entropy-over-time curves**

   * same prompt, different contexts
   * makes deformation visually obvious

4. **Output stability vs internal deformation scatter**

   * x: deformation score
   * y: answer correctness / confidence
   * key insight: outputs can stay stable while computation shifts

This is your “aha”.

---

## 8. Deliverables (explicit)

By submission time you should have:

* GitHub repo

  * prompt runner
  * instrumentation hooks
  * metric computation
  * plotting notebook
* Short report

  * method
  * plots
  * limitations & dual-use
* (Optional) 3-min demo video

  * show one prompt live
  * show plots

---

## 9. Timeline (realistic hackathon pacing)

### Day 0 (prep, before hackathon if allowed)

* finalize prompts
* write context templates
* set up model loading + hooks

---

### Day 1

* run synthetic prompts
* verify logs
* compute CCI + EHL
* produce first plots

If you don’t see *any* context effect by end of Day 1 → **pivot**.

---

### Day 2

* add TP metric
* add WildChat slice (optional)
* clean plots
* write report
* record demo

---

## 10. Kill / pivot criteria (important)

You should **abandon or simplify** if:

* Context effects are smaller than run-to-run noise
* Differences vanish when you normalize for output length
* Smaller and larger models behave identically

If that happens, pivot to:

* fewer contexts (neutral vs audited only)
* fewer layers
* first-N-tokens only
