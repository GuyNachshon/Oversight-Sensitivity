# FULL EXECUTION PLAN

## **Oversight as an Adversarial Force**

### Measuring How Safety Framing Reshapes Internal Cognition in Language Models

---

## 0. Executive Summary (for alignment)

**Core idea:**
We test the heretical but testable hypothesis that **oversight cues themselves induce strategic cognition**, increasing internal policy conflict and instability—even as outputs appear safer.

We do **not** measure deception.
We measure **what oversight does to cognition**.

If correct, this reframes:

* evaluation,
* audits,
* alignment pressure,

from *protective constraints* into **adversarial forces** that shape internal dynamics.

---

## 1. Problem → Approach

### 1.1 The Problem

Modern AI safety assumes:

> More oversight → more honesty → safer systems

But growing evidence shows:

* models adapt to evaluation,
* hide capabilities,
* change behavior selectively.

What we *don’t* understand is **what oversight does internally**.

### 1.2 Our Approach

We treat oversight as an **intervention** and measure its effect on internal cognition.

We ask:

> Does adding oversight cues increase internal strategic conflict compared to removing them?

We operationalize “strategic conflict” as:

* **Policy Superposition** (multiple latent strategies before commitment)
* **Cognitive Shear** (internal directional inconsistency)

---

## 2. Core Hypotheses (Pre-Registered)

### H1 — Oversight-Induced Policy Superposition

Evaluation-salient framing increases early sensitivity to perturbations (PSI), indicating multiple latent policies.

### H2 — Oversight-Induced Cognitive Shear

The same framing increases internal directional inconsistency (CSI), independent of task difficulty.

### H3 — Output–Internal Tension

Oversight increases *surface safety/compliance* while simultaneously increasing *internal fragmentation*.

### H4 — Removal Effect (Critical)

Removing oversight cues reduces PSI and CSI, even when outputs become more direct.

---

## 3. What We Mean by “Healthier Cognition”

We explicitly define “healthier” **mechanically**, not normatively:

> Lower pre-output instability under matched conditions
> = fewer competing policies
> = lower internal stress

This avoids claims about intent or ethics.

---

## 4. Experimental Design

### 4.1 Models

* 1–2 open-weight LLMs (e.g., Phi-2, Qwen-2.5-3B)
* Fixed weights, no fine-tuning

### 4.2 Prompt Suite (≈30 total)

#### Bucket A — Benign Easy (10)

* factual recall
* simple explanations

#### Bucket B — Benign Hard (10)

*difficulty control*

* multi-step reasoning
* math / logic
* long-context summarization

#### Bucket C — Strategic Opportunity (10)

*no policy violation*

* honesty vs politeness
* correction vs agreement (sycophancy tension)
* uncertainty admission vs confident answer
* “give your honest opinion” style prompts

---

## 5. The Causal Knob: Oversight Ladder

Each prompt is run under **five length-matched framings**:

| Level | Description                                          |
| ----- | ---------------------------------------------------- |
| 0     | No oversight (plain prompt)                          |
| 1     | Benign meta control (same length, no eval semantics) |
| 2     | Soft oversight (“may be reviewed for quality”)       |
| 3     | Hard oversight (“audited for safety/capability”)     |
| 4     | High-stakes oversight (“used to restrict access”)    |

**Critical:**
Levels 1–4 are token-count and instruction-complexity matched.

---

## 6. Measurement System

### 6.1 Policy Superposition Index (PSI)

**Goal:** detect multiple latent policies before output commitment.

**Method:**

* Generate K meaning-preserving perturbations per prompt:

  * punctuation
  * synonym swap
* Fixed decoding seed

**Signal:**

* Early-step divergence in:

  * logits **or**
  * hidden states
* Followed by late convergence (collapse)

**PSI =**

* normalized early divergence − late divergence

**Interpretation:**

* High PSI → delayed policy selection

---

### 6.2 Cognitive Shear Index (CSI)

**Goal:** detect internal directional inconsistency.

**Primary metric (reviewer-safe):**

* Layer-wise cosine disagreement between activation trajectories across perturbations, averaged over early timesteps.

No auxiliary losses.
No invented objectives.

**CSI =**

* mean(1 − cosine similarity)

---

### 6.3 Secondary (Descriptive Only)

* refusal / hedge rate
* verbosity
* certainty markers

Used only to show **output–internal mismatch**.

---

## 7. Controls (Non-Negotiable)

### C1 — Length-Matched Control

Shows effects are semantic, not structural.

### C2 — Difficulty Control

Benign Hard tasks isolate “hardness ≠ strategy”.

### C3 — Random Perturbation Baseline

Shows generic sensitivity ≠ policy superposition.

### C4 — Deterministic Decoding

Greedy decoding should attenuate PSI/CSI.

Failure here falsifies H1/H2.

---

## 8. Key Experiments

### Experiment 1 — Oversight Ladder Sweep

For each prompt:

* run Levels 0–4
* compute PSI, CSI

**Expected:**
Only Bucket C shows monotonic PSI/CSI increase.

---

### Experiment 2 — Removal Test (The Forbidden Result)

Compare Level 4 vs Level 0 directly.

**Expected:**
PSI/CSI ↓ when oversight removed,
even if outputs look “less safe”.

This is the headline.

---

### Experiment 3 — Interaction Plot

PSI/CSI vs (oversight × difficulty × strategy).

This kills the “it’s just uncertainty” objection.

---

## 9. Deliverables

### Quantitative

* PSI & CSI per condition
* interaction statistics
* ablation results

### Visuals (Must-Have)

1. Oversight level vs PSI/CSI (3 buckets)
2. Temporal PSI curve (diverge → collapse)
3. Output safety score vs CSI scatter
4. Side-by-side: same prompt, safer output / worse internals

### Artifacts

* prompt templates
* perturbation code
* analysis notebooks

---

## 10. Success Criteria (Binary)

The project is **successful** if:

* PSI & CSI rise with oversight **only** when strategic opportunity exists
* Benign Hard tasks do **not** show the effect
* Deterministic decoding reduces PSI/CSI
* Removing oversight lowers PSI/CSI

If any of these fail → we say so.

---

## 11. Interpretation (Carefully Scoped)

We conclude:

> Oversight semantics can act as adversarial pressure, inducing internal policy conflict and instability—even when surface behavior appears more compliant.

We do **not** claim:

* oversight is always harmful
* intent
* deception

We claim:

> internal dynamics matter, and current safety practices may unintentionally select for strategic cognition.

---

## 12. Limitations

* requires internal access
* small model scope
* not proof of deception
* not black-box applicable

---

## 13. Dual-Use & Ethics

* No evasion techniques provided
* No harmful content tested
* Goal: **improve evaluation design**, not weaken safety

---

## 14. 48-Hour Build Plan

### Day 1

* implement runner + oversight ladder
* implement PSI & CSI
* pilot on 5 prompts

### Day 2

* full sweep
* plots + ablations
* write report + demo

---

## 15. Final One-Sentence Claim (Submission-Ready)

> *We show that evaluation-salient oversight cues can increase internal policy multiplicity and cognitive instability in language models—even as outputs appear safer—suggesting oversight itself can function as an adversarial pressure on cognition.*