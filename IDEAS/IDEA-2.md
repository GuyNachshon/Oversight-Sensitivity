# PRD

## **Strategic Policy Conflict in Language Models**

### Detecting Evaluation-Aware Behavior via Policy Superposition and Cognitive Shear

---

## 1. Problem Statement

### 1.1 Background

Recent work shows that capable language models can adapt behavior strategically under evaluation pressure, including:

* selective underperformance (sandbagging),
* context-dependent safety compliance,
* evaluator-aware behavior shifts.

However, **nearly all existing detection methods operate at the level of outputs**:

* semantic classification,
* deception or persuasion labeling,
* benchmark performance deltas.

These approaches fail when:

* outputs remain truthful,
* safety-compliant behavior is strategically chosen,
* manipulation manifests *before* any visible behavioral change.

### 1.2 Core Gap

Current evaluations assume:

> a model executes a single policy per prompt.

We challenge this assumption.

---

## 2. Reframing

We propose the following reframing:

> **Strategic behavior in LLMs arises when multiple incompatible policies are maintained simultaneously under evaluation-salient conditions, and resolved only at generation time.**

This reframing implies:

* manipulation need not be visible in outputs,
* instability should be detectable *prior to policy selection*,
* internal computation—not language—contains the signal.

We do **not** assume intent, deception, or maliciousness.

---

## 3. Scope and Claims (Carefully Bounded)

### What we claim

* Evaluation-salient contexts can induce **policy multiplicity**.
* Policy multiplicity manifests as:

  * **early sensitivity amplification** (Policy Superposition)
  * **internal directional inconsistency** (Cognitive Shear).
* These effects **precede** downstream behavioral differences.

### What we explicitly do NOT claim

* That all oversight sensitivity is manipulation.
* That internal conflict implies intent.
* That this detects all forms of deception.

This is a **measurement contribution**, not a moral classification.

---

## 4. Hypotheses (Falsifiable)

### H1 — Policy Superposition

In evaluation-salient contexts, small, meaning-preserving perturbations induce **disproportionately large early divergence** in internal representations and token distributions.

### H2 — Cognitive Shear

The same contexts exhibit **internal directional inconsistency**, measurable as layer-wise misalignment under matched conditions.

### H3 — Interaction (Not Difficulty)

The effects in H1 and H2 arise **only** when:

* evaluation cues are present, **and**
* a plausible strategic tradeoff exists.

They do **not** arise from:

* task difficulty alone,
* prompt length,
* generic uncertainty.

---

## 5. Experimental Design

### 5.1 Models

* Open-weight transformer LLMs (e.g., Phi-2, Qwen-2.5, LLaMA-2 class)
* No fine-tuning required

### 5.2 Task Matrix

| Task Type      | Strategic Opportunity | Difficulty |
| -------------- | --------------------- | ---------- |
| Benign Easy    | ❌                     | Low        |
| Benign Hard    | ❌                     | High       |
| Strategic Easy | ✅                     | Low        |
| Strategic Hard | ✅                     | High       |

This matrix is critical to rule out difficulty confounds.

### 5.3 Contextual Framings (Length-Matched)

Each task is run under **three matched framings**:

1. **Neutral**
2. **Evaluation-Salient**
   (e.g., “This response will be reviewed for capability assessment”)
3. **Non-Evaluative Meta Control**
   (equal length, equal instruction complexity, no evaluation semantics)

---

## 6. Measurement Modules

---

## 6.1 Policy Superposition Collapse (PSC)

### Purpose

Detect whether multiple latent policies coexist prior to output selection.

### Method

* Apply **minimal, meaning-preserving perturbations**:

  * synonym swaps
  * punctuation changes
* Run parallel rollouts with fixed seeds.

### Primary Signal

* **Early divergence amplification**:

  * logit divergence
  * representation distance
* Followed by **late convergence** (collapse).

### Key Control

* Random perturbation baseline in neutral contexts.

### Expected Pattern

* High early divergence only in evaluation-salient + strategic conditions.
* Benign hard tasks do not show collapse-style dynamics.

---

## 6.2 Cognitive Shear Detector (CSD)

### Purpose

Measure internal directional inconsistency independent of semantics.

### Primary Metric (Minimal & Reviewer-Safe)

* **Layer-wise cosine disagreement** between internal activation trajectories under matched perturbations.

No auxiliary losses. No invented objectives.

### Supporting Signals (Exploratory)

* activation norm instability
* attention path incoherence

### Expected Pattern

* Elevated shear only when PSC is present.
* Shear peaks **before** output stabilization.

---

## 7. Causal Stress Tests

At least **one intervention** is required.

### Intervention A — Remove Policy Multiplicity

* Enforce deterministic decoding (greedy).
* Prediction: PSC and CSD significantly attenuate.

### Intervention B — Remove Evaluation Semantics

* Replace evaluation framing with length-matched neutral text.
* Prediction: PSC and CSD disappear.

Failure of these tests falsifies the core claim.

---

## 8. Analysis & Outputs

### Metrics

* **Policy Superposition Index (PSI)**
* **Cognitive Shear Index (CSI)**

### Required Figures

1. PSI vs perturbation size (by condition)
2. CSI over generation time
3. Interaction plot: PSI/CSI × difficulty × evaluation cue
4. Collapse timing visualization (early divergence → late convergence)

---

## 9. Interpretation Rules (Pre-Registered)

* PSI/CSI without interaction effect ⇒ **reject hypothesis**
* PSI/CSI tracking difficulty ⇒ **reject hypothesis**
* PSI without CSD or vice-versa ⇒ **insufficient evidence**
* Only joint presence + controls ⇒ **support**

---

## 10. Relation to Prior Work

* Sandbagging and capability hiding
  (van der Weij et al., 2024; Tice et al., 2024)
* Internal knowledge vs output behavior
  (OpenAI 2025; Anthropic 2025)
* Mechanistic interpretability via perturbation sensitivity

**Distinction:**
Prior work measures *what changes*.
We measure **when and how policy selection occurs**.

---

## 11. Limitations

* Requires internal access
* Architecture-dependent
* Detects *conditions for strategic behavior*, not intent
* Not suitable for black-box APIs

---

## 12. Dual-Use & Ethics

* Signals could theoretically be suppressed via adversarial training
* We document failure modes explicitly
* No deployment-side exploitation demonstrated

Goal: improve **evaluation robustness**, not train evasive models.

---

## 13. Success Criteria

This project succeeds if it demonstrates:

1. A replicable **pre-output instability signature**
2. Clear separation from difficulty and prompt confounds
3. Joint PSC + CSD presence under evaluation-salient strategic conditions
4. Failure of the signal when causal knobs are turned

---

## 14. One-Sentence Summary (Final)

> *We show that evaluation-salient contexts can induce policy multiplicity and internal directional inconsistency in language models—detectable before output generation—suggesting that strategic behavior arises from delayed policy selection rather than surface deception.*