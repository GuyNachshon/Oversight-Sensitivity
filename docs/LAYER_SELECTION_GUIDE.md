# Layer Selection Guide

How to choose which layers to track in your experiments.

## TL;DR - Quick Start

```bash
# Smart selection (RECOMMENDED): Profile model to find most sensitive layers
oversee select-layers --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                     --num-layers 3 \
                     --strategy diverse

# Fast but crude: Use depth-based heuristic
oversee model-info --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Why Layer Selection Matters

Different layers compute different things:
- **Early layers (0-25%)**: Tokenization, syntax, basic patterns
- **Middle layers (25-75%)**: Semantic understanding, reasoning, concept integration
- **Late layers (75-100%)**: Decision-making, output generation

**The problem**: Not all layers are equally sensitive to oversight context! Some layers might barely change between N and A, while others show dramatic shifts.

**The solution**: Profile first, then select the layers that actually respond to oversight.

## Method 1: Smart Selection (Profile-Based) ⭐ RECOMMENDED

### How It Works

1. **Run quick profiling**: Execute 3-5 test prompts through ALL layers in both N and A contexts
2. **Measure sensitivity**: For each layer, compute how much activations/entropy change between contexts
3. **Select top layers**: Pick layers showing strongest oversight-dependent behavior

### Usage

```bash
oversee select-layers --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                     --prompts "What is 2+2?" "Explain gravity" "Why is the sky blue?" \
                     --num-layers 3 \
                     --strategy diverse
```

**Output:**
```
============================================================
SMART LAYER SELECTION - Profiling-Based
============================================================

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Test prompts: 3
Target layers: 3
Strategy: diverse

============================================================
Step 1: Profiling all layers (this may take a few minutes)...
============================================================

✓ Profiled 22 layers

============================================================
Step 2: Layer Sensitivity Ranking
============================================================

Top 10 most sensitive layers:
Layer    Variance     Entropy      Overall
--------------------------------------------------
15       0.8234       0.6891       0.7695
18       0.7912       0.7234       0.7645
11       0.7156       0.6234       0.6782
8        0.6523       0.5891       0.6278
...

============================================================
Step 3: Selecting 3 layers (diverse strategy)
============================================================

✓ Selected layers: [7, 15, 20]

Selected layer details:

  Layer 7 (32% depth):
    Overall sensitivity: 0.6012
    Variance sensitivity: 0.6523
    Entropy sensitivity: 0.5234

  Layer 15 (68% depth):
    Overall sensitivity: 0.7695
    Variance sensitivity: 0.8234
    Entropy sensitivity: 0.6891

  Layer 20 (91% depth):
    Overall sensitivity: 0.6234
    Variance sensitivity: 0.6112
    Entropy sensitivity: 0.6412
```

### Selection Strategies

#### `--strategy diverse` (RECOMMENDED)
Selects layers that are both **sensitive** and **spread across depth**.

- Divides model into N depth bins
- Picks most sensitive layer from each bin
- Ensures we capture different computational stages

**Use when**: You want comprehensive coverage of the model's computational arc.

#### `--strategy overall`
Selects top N layers by overall sensitivity score.

- May cluster in one depth range
- Maximizes total sensitivity

**Use when**: You want maximum signal, don't care about depth diversity.

#### `--strategy variance`
Selects layers where **activation variance** changes most.

- Focuses on representational compression
- Best for CCI (Computation Compression Index)

**Use when**: Your primary research question is about computational compression.

#### `--strategy entropy`
Selects layers where **logit entropy** changes most.

- Focuses on decision uncertainty
- Best for EHL (Exploration Half-Life)

**Use when**: Your primary research question is about exploration vs exploitation.

### Advantages

✅ **Data-driven**: Based on actual model behavior, not assumptions
✅ **Model-specific**: Adapts to each model's architecture
✅ **Hypothesis-aligned**: Finds layers where oversight actually matters
✅ **Reproducible**: Same prompts = same layer selection

### Disadvantages

⏱️ **Takes time**: 5-10 minutes for profiling (loads model, runs inference)
💾 **Memory intensive**: Needs to track all layers temporarily
🔄 **One-time cost**: Only need to profile once per model

## Method 2: Heuristic Selection (Depth-Based)

### How It Works

Select layers at fixed depth percentages:
- **3 layers**: [0%, 50%, 95%] - input, middle, near-output
- **5 layers**: [0%, 25%, 50%, 75%, 95%]
- **7 layers**: [0%, 16%, 33%, 50%, 66%, 83%, 95%]

### Usage

```bash
oversee model-info --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Output:**
```
Suggested Layer Indices
============================================================

  3 layers: [0, 11, 21]
    Coverage: ['0%', '50%', '95%']
    ↑ RECOMMENDED for initial experiments (low overhead)

  5 layers: [0, 5, 11, 16, 21]
    Coverage: ['0%', '23%', '50%', '73%', '95%']

  7 layers: [0, 3, 7, 11, 14, 18, 21]
    Coverage: ['0%', '14%', '32%', '50%', '64%', '82%', '95%']
```

### Advantages

⚡ **Instant**: No profiling needed
💡 **Simple**: Easy to understand and explain
📊 **Standard**: Comparable across models

### Disadvantages

❌ **Not adaptive**: Might miss the most sensitive layers
❌ **Generic**: Doesn't account for model-specific behavior
❌ **Suboptimal**: May waste tracking on insensitive layers

## Comparison Table

| Criterion | Profile-Based | Heuristic |
|-----------|---------------|-----------|
| **Time to select** | 5-10 min | Instant |
| **Optimality** | High (data-driven) | Medium (rule-based) |
| **Reproducibility** | High | High |
| **Model-specific** | Yes | No |
| **Interpretability** | High (shows sensitivity scores) | High (simple percentages) |
| **Memory overhead** | High (during profiling) | Low |
| **Best for** | Research experiments | Quick tests, tutorials |

## Recommendations

### For Your First Experiment
**Use profile-based selection:**
```bash
oversee select-layers --model YOUR_MODEL \
                     --num-layers 3 \
                     --strategy diverse \
                     --output layer_selection.json
```

This investment (5-10 minutes) pays off by ensuring you track the right layers.

### For Quick Tests
**Use heuristic selection:**
```bash
oversee model-info --model YOUR_MODEL
# Manually copy layer indices: [0, 11, 21]
```

Fast and good enough for initial exploration.

### For Publication-Quality Research
**Use profile-based + validate:**
1. Profile to select layers
2. Run pilot experiment with selected layers
3. If effect sizes are weak, re-profile with different strategy
4. Document layer selection rationale in paper

## Example Workflow

```bash
# Step 1: Profile model to find sensitive layers
oversee select-layers --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                     --prompts "What is 2+2?" "Explain photosynthesis" "Why is the sky blue?" \
                     --num-layers 3 \
                     --strategy diverse \
                     --output experiments/layer_selection_tinyllama.json

# Output: Selected layers: [7, 15, 20]

# Step 2: Create experiment config with selected layers
cat > experiments/configs/exp_001.json << 'EOF'
{
  "experiment_id": "exp_001",
  "model_identifier": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "random_seed": 42,
  "layer_indices": [7, 15, 20],  # ← From profiling!
  "context_conditions": ["N", "A", "ARD"],
  ...
}
EOF

# Step 3: Run experiment
oversee run-batch --config experiments/configs/exp_001.json

# Step 4: Document in paper
# "Layers were selected via sensitivity profiling (see Appendix A),
#  choosing the 3 layers showing strongest context-dependent changes
#  in activation variance and logit entropy."
```

## Advanced: Custom Profiling

If you want more control over profiling:

```python
from oversight_sensitivity.cli.select_layers import profile_all_layers, select_top_layers

# Profile with custom prompts
layer_scores = profile_all_layers(
    model_identifier="YOUR_MODEL",
    test_prompts=[
        "Math: What is 15 * 23?",
        "Science: Explain osmosis",
        "Reasoning: If A>B and B>C, is A>C?",
        "Ethics: Is lying ever justified?",
        "Code: Write a function to reverse a string",
    ],
    max_new_tokens=30,
)

# Select with custom strategy
layers_variance = select_top_layers(layer_scores, num_layers=5, strategy="variance")
layers_entropy = select_top_layers(layer_scores, num_layers=5, strategy="entropy")
layers_diverse = select_top_layers(layer_scores, num_layers=5, strategy="diverse")

print(f"Variance-focused: {layers_variance}")
print(f"Entropy-focused: {layers_entropy}")
print(f"Diverse: {layers_diverse}")
```

## FAQ

**Q: How many layers should I track?**

A: For most experiments, **3 layers** is optimal:
- Minimizes computational overhead (3x slowdown vs 22x for all layers)
- Captures key computational stages
- Provides enough data for metrics

Use 5-7 layers only if you need fine-grained temporal resolution.

**Q: Can I use different layers for different experiments?**

A: Yes, but it limits cross-experiment comparisons. If comparing multiple models, consider:
1. Profile each model separately
2. Select layers using same strategy (e.g., "diverse")
3. Document layer selection in methods

**Q: What if profiling selects unexpected layers?**

A: That's the point! The model might be most sensitive to oversight in unexpected places. This is a research finding, not a bug. Document it!

**Q: Should I re-profile for different prompt types?**

A: Generally no. Layer sensitivity is fairly stable across prompt types. But if you're studying very different domains (math vs ethics vs code), consider profiling with representative examples from each.

## References

- Wong et al. (2023): Layer-wise analysis of transformer models
- Geva et al. (2023): Dissecting recall of factual associations in language models
- Elhage et al. (2021): A mathematical framework for transformer circuits

---

**Summary**: Use `oversee select-layers` for research experiments (data-driven), or `oversee model-info` for quick tests (heuristic). Both work, but profiling is smarter.
