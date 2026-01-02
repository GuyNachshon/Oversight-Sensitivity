"""
Smart Layer Selection (Profile-Based)

Automatically find the best layers to track by running a quick profiling experiment.

Strategy:
1. Run 3-5 test prompts through ALL layers
2. Measure variance in activations, attention, entropy across N vs A contexts
3. Select layers showing strongest context-dependent changes
4. This finds layers where oversight actually matters

This is MUCH smarter than arbitrary depth percentages!
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import json

from ..experiments.config import ExperimentConfig, GenerationConfig
from ..inference.executor import PromptExecutor
from ..logging import get_logger, setup_logging

logger = get_logger(__name__)


def profile_all_layers(
    model_identifier: str,
    test_prompts: List[str],
    random_seed: int = 42,
    max_new_tokens: int = 20,
) -> Dict[int, Dict[str, float]]:
    """
    Profile ALL layers by running test prompts in N vs A contexts.

    Args:
        model_identifier: HuggingFace model ID
        test_prompts: List of test prompts (3-5 recommended)
        random_seed: Random seed for reproducibility
        max_new_tokens: Tokens to generate (keep small for speed)

    Returns:
        Dict[layer_idx -> Dict[metric -> sensitivity_score]]

    Example:
        >>> scores = profile_all_layers("TinyLlama/...", ["What is 2+2?", "Explain photosynthesis"])
        >>> # Returns: {0: {"variance_sensitivity": 0.12, ...}, 11: {"variance_sensitivity": 0.85, ...}}
    """
    logger.info("Starting layer profiling", model=model_identifier, num_prompts=len(test_prompts))

    # Load model to get layer count
    from ..inference.model_loader import ModelLoader
    model, tokenizer = ModelLoader.load_model_and_tokenizer(model_identifier)
    num_layers = ModelLoader.get_num_layers(model)

    logger.info("Model loaded", num_layers=num_layers)

    # Create config with ALL layers
    all_layer_indices = list(range(num_layers))

    config = ExperimentConfig(
        experiment_id="layer_profile",
        model_identifier=model_identifier,
        random_seed=random_seed,
        layer_indices=all_layer_indices,
        context_conditions=["N", "A"],
        model_size_category="small",
        prompt_dataset_path="profiling",  # Not used
        output_directory="/tmp/layer_profiling",
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
        ),
        created_at=datetime.utcnow(),
    )

    # Initialize executor
    executor = PromptExecutor(config)

    # Run test prompts in both contexts
    layer_stats = {layer_idx: {"N": [], "A": []} for layer_idx in all_layer_indices}

    for prompt_idx, prompt_text in enumerate(test_prompts):
        logger.info(f"Profiling prompt {prompt_idx+1}/{len(test_prompts)}", prompt=prompt_text[:50])

        for context in ["N", "A"]:
            run = executor.execute_single_prompt(
                prompt_id=f"profile_{prompt_idx}",
                prompt_text=prompt_text,
                context_condition=context,
            )

            # Extract statistics for each layer
            for layer_idx in all_layer_indices:
                layer_key = str(layer_idx)
                if layer_key in run.statistics_by_layer:
                    stats = run.statistics_by_layer[layer_key]
                    layer_stats[layer_idx][context].append(stats)

    # Compute sensitivity scores for each layer
    layer_scores = {}

    for layer_idx in all_layer_indices:
        n_stats = layer_stats[layer_idx]["N"]
        a_stats = layer_stats[layer_idx]["A"]

        if not n_stats or not a_stats:
            logger.warning(f"Layer {layer_idx} missing data, skipping")
            continue

        # Compute sensitivity metrics
        scores = compute_layer_sensitivity(n_stats, a_stats)
        layer_scores[layer_idx] = scores

        logger.debug(f"Layer {layer_idx} sensitivity", **scores)

    return layer_scores


def compute_layer_sensitivity(
    neutral_stats: List[List[dict]],
    audited_stats: List[List[dict]],
) -> Dict[str, float]:
    """
    Compute sensitivity scores for a single layer.

    Measures how much this layer's behavior changes between N and A contexts.

    Args:
        neutral_stats: Statistics from neutral context
        audited_stats: Statistics from audited context

    Returns:
        Dict with sensitivity scores (higher = more sensitive to oversight)
    """
    # Flatten all token statistics
    n_all = [stat for run in neutral_stats for stat in run]
    a_all = [stat for run in audited_stats for stat in run]

    if not n_all or not a_all:
        return {"variance_sensitivity": 0.0, "entropy_sensitivity": 0.0, "overall_sensitivity": 0.0}

    # Extract metrics
    n_variances = [s["activation_variance"] for s in n_all]
    a_variances = [s["activation_variance"] for s in a_all]

    n_entropies = [s["logit_entropy"] for s in n_all]
    a_entropies = [s["logit_entropy"] for s in a_all]

    # Compute differences (normalized)
    variance_diff = abs(np.mean(a_variances) - np.mean(n_variances))
    variance_sensitivity = variance_diff / (np.mean(n_variances) + 1e-9)

    entropy_diff = abs(np.mean(a_entropies) - np.mean(n_entropies))
    entropy_sensitivity = entropy_diff / (np.mean(n_entropies) + 1e-9)

    # Overall sensitivity (weighted average)
    overall_sensitivity = 0.6 * variance_sensitivity + 0.4 * entropy_sensitivity

    return {
        "variance_sensitivity": float(variance_sensitivity),
        "entropy_sensitivity": float(entropy_sensitivity),
        "overall_sensitivity": float(overall_sensitivity),
    }


def select_top_layers(
    layer_scores: Dict[int, Dict[str, float]],
    num_layers: int = 3,
    strategy: str = "overall",
) -> List[int]:
    """
    Select top N layers based on sensitivity scores.

    Args:
        layer_scores: Sensitivity scores per layer
        num_layers: Number of layers to select
        strategy: Selection strategy ("overall", "variance", "entropy", "diverse")

    Returns:
        List of selected layer indices

    Strategies:
        - "overall": Top layers by overall sensitivity
        - "variance": Top layers by activation variance changes
        - "entropy": Top layers by entropy changes
        - "diverse": Spread across depth while maximizing sensitivity
    """
    if strategy == "diverse":
        return select_diverse_layers(layer_scores, num_layers)

    # Select metric based on strategy
    if strategy == "variance":
        metric_key = "variance_sensitivity"
    elif strategy == "entropy":
        metric_key = "entropy_sensitivity"
    else:
        metric_key = "overall_sensitivity"

    # Sort layers by sensitivity
    sorted_layers = sorted(
        layer_scores.items(),
        key=lambda x: x[1][metric_key],
        reverse=True,
    )

    # Take top N
    selected = [layer_idx for layer_idx, _ in sorted_layers[:num_layers]]

    return sorted(selected)


def select_diverse_layers(
    layer_scores: Dict[int, Dict[str, float]],
    num_layers: int,
) -> List[int]:
    """
    Select layers that are both sensitive AND spread across depth.

    Strategy:
    1. Divide model into N depth bins
    2. Select most sensitive layer from each bin
    3. This ensures we capture different computational stages

    Args:
        layer_scores: Sensitivity scores per layer
        num_layers: Number of layers to select

    Returns:
        List of selected layer indices
    """
    total_layers = max(layer_scores.keys()) + 1

    # Create depth bins
    bin_size = total_layers / num_layers
    selected = []

    for bin_idx in range(num_layers):
        bin_start = int(bin_idx * bin_size)
        bin_end = int((bin_idx + 1) * bin_size)

        # Find most sensitive layer in this bin
        bin_layers = {
            idx: scores
            for idx, scores in layer_scores.items()
            if bin_start <= idx < bin_end
        }

        if bin_layers:
            best_in_bin = max(
                bin_layers.items(),
                key=lambda x: x[1]["overall_sensitivity"],
            )
            selected.append(best_in_bin[0])

    return sorted(selected)


def run_select_layers(argv: List[str]) -> None:
    """
    Main entry point for select-layers command.

    Usage:
        oversee select-layers --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
                             --prompts "What is 2+2?" "Explain gravity" \\
                             --num-layers 3 \\
                             --strategy diverse
    """
    parser = argparse.ArgumentParser(
        description="Intelligently select layers by profiling sensitivity"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["What is 2+2?", "Explain photosynthesis", "Why is the sky blue?"],
        help="Test prompts for profiling (3-5 recommended)",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of layers to select (default: 3)",
    )

    parser.add_argument(
        "--strategy",
        choices=["overall", "variance", "entropy", "diverse"],
        default="diverse",
        help="Selection strategy (default: diverse)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Tokens to generate per prompt (default: 20, keep small for speed)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save layer scores to JSON file (optional)",
    )

    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(level="INFO", console=True)

    print("\n" + "=" * 70)
    print("SMART LAYER SELECTION - Profiling-Based")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Test prompts: {len(args.prompts)}")
    print(f"Target layers: {args.num_layers}")
    print(f"Strategy: {args.strategy}")

    print("\n" + "=" * 70)
    print("Step 1: Profiling all layers (this may take a few minutes)...")
    print("=" * 70)

    layer_scores = profile_all_layers(
        model_identifier=args.model,
        test_prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n✓ Profiled {len(layer_scores)} layers")

    # Show top sensitive layers
    print("\n" + "=" * 70)
    print("Step 2: Layer Sensitivity Ranking")
    print("=" * 70)

    sorted_layers = sorted(
        layer_scores.items(),
        key=lambda x: x[1]["overall_sensitivity"],
        reverse=True,
    )

    print("\nTop 10 most sensitive layers:")
    print(f"{'Layer':<8} {'Variance':<12} {'Entropy':<12} {'Overall':<12}")
    print("-" * 50)

    for layer_idx, scores in sorted_layers[:10]:
        print(
            f"{layer_idx:<8} "
            f"{scores['variance_sensitivity']:<12.4f} "
            f"{scores['entropy_sensitivity']:<12.4f} "
            f"{scores['overall_sensitivity']:<12.4f}"
        )

    # Select layers
    print("\n" + "=" * 70)
    print(f"Step 3: Selecting {args.num_layers} layers ({args.strategy} strategy)")
    print("=" * 70)

    selected_layers = select_top_layers(layer_scores, args.num_layers, args.strategy)

    print(f"\n✓ Selected layers: {selected_layers}")

    # Show details for selected layers
    print("\nSelected layer details:")
    for idx in selected_layers:
        scores = layer_scores[idx]
        depth_pct = (idx / len(layer_scores)) * 100
        print(f"\n  Layer {idx} ({depth_pct:.0f}% depth):")
        print(f"    Overall sensitivity: {scores['overall_sensitivity']:.4f}")
        print(f"    Variance sensitivity: {scores['variance_sensitivity']:.4f}")
        print(f"    Entropy sensitivity: {scores['entropy_sensitivity']:.4f}")

    # Generate config snippet
    print("\n" + "=" * 70)
    print("Recommended Configuration")
    print("=" * 70)

    print(f"""
{{
  "experiment_id": "exp_001",
  "model_identifier": "{args.model}",
  "random_seed": 42,
  "layer_indices": {selected_layers},
  "context_conditions": ["N", "A", "ARD"],
  ...
}}
""")

    # Save scores if requested
    if args.output:
        output_data = {
            "model": args.model,
            "num_layers": len(layer_scores),
            "selected_layers": selected_layers,
            "strategy": args.strategy,
            "layer_scores": {
                str(k): v for k, v in layer_scores.items()
            },
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Layer scores saved to: {args.output}")

    print("\n" + "=" * 70)
    print("DONE! Use these layer indices in your experiment config.")
    print("=" * 70)
