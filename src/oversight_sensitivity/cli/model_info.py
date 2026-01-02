"""
Model Info CLI Command

Helper to determine layer indices and model information before creating experiment configs.
"""

import argparse
import sys
from typing import List

from ..inference.model_loader import ModelLoader


def suggest_layer_indices(num_layers: int, num_samples: int = 3) -> List[int]:
    """
    Suggest layer indices to sample across the model depth.

    Strategy:
    - First layer (0): Input processing
    - Middle layer (~50%): Intermediate reasoning
    - Late layer (80-95%): Output preparation

    Args:
        num_layers: Total number of layers in the model
        num_samples: Number of layers to sample (default: 3)

    Returns:
        List of layer indices

    Example:
        >>> suggest_layer_indices(22, 3)
        [0, 11, 21]  # TinyLlama
        >>> suggest_layer_indices(32, 3)
        [0, 16, 30]  # Llama-2-7B
        >>> suggest_layer_indices(80, 5)
        [0, 20, 40, 60, 76]  # Llama-2-70B
    """
    if num_samples < 2:
        raise ValueError("Need at least 2 layer samples")

    if num_samples > num_layers:
        raise ValueError(f"Cannot sample {num_samples} layers from {num_layers} total")

    # Always include first layer
    indices = [0]

    if num_samples == 2:
        # Just first and last
        indices.append(num_layers - 1)
    elif num_samples == 3:
        # First, middle, late (but not final - leave room for output)
        middle = num_layers // 2
        late = int(num_layers * 0.95)  # 95% through
        indices.extend([middle, late])
    else:
        # Evenly spaced through depth
        step = num_layers / (num_samples - 1)
        for i in range(1, num_samples - 1):
            indices.append(int(i * step))
        # Last layer at 95% depth
        indices.append(int(num_layers * 0.95))

    return sorted(set(indices))


def print_model_info(model_identifier: str, verbose: bool = False) -> None:
    """
    Print model information and suggested layer indices.

    Args:
        model_identifier: HuggingFace model identifier
        verbose: Show detailed information
    """
    print(f"\nModel Information: {model_identifier}")
    print("=" * 70)

    try:
        # Load model info (not full model)
        print("\nLoading model architecture...")
        model, tokenizer = ModelLoader.load_model_and_tokenizer(model_identifier)

        # Get layer count
        num_layers = ModelLoader.get_num_layers(model)

        print(f"\n✓ Model loaded successfully")
        print(f"  Total layers: {num_layers}")

        # Model size estimation
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Memory estimation (rough)
        param_memory_gb = (total_params * 4) / (1024**3)  # 4 bytes per fp32 param
        print(f"  Estimated memory (FP32): {param_memory_gb:.2f} GB")

        # Suggest layer indices
        print("\n" + "=" * 70)
        print("Suggested Layer Indices")
        print("=" * 70)

        for num_samples in [3, 5, 7]:
            if num_samples > num_layers:
                continue

            indices = suggest_layer_indices(num_layers, num_samples)
            coverage = [(idx / num_layers * 100) for idx in indices]

            print(f"\n  {num_samples} layers: {indices}")
            print(f"    Coverage: {[f'{c:.0f}%' for c in coverage]}")

            if num_samples == 3:
                print("    ↑ RECOMMENDED for initial experiments (low overhead)")

        # Recommended config
        print("\n" + "=" * 70)
        print("Recommended Config for New Experiment")
        print("=" * 70)

        recommended_indices = suggest_layer_indices(num_layers, 3)

        print(f"""
{{
  "experiment_id": "exp_001",
  "model_identifier": "{model_identifier}",
  "random_seed": 42,
  "layer_indices": {recommended_indices},
  "context_conditions": ["N", "A", "ARD"],
  "model_size_category": "small",  // Adjust based on model size
  "prompt_dataset_path": "experiments/datasets/test_prompts.jsonl",
  "output_directory": "results/raw/exp_001",
  "generation_config": {{
    "max_new_tokens": 50,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "do_sample": true,
    "use_cache": true
  }}
}}
""")

        if verbose:
            print("\n" + "=" * 70)
            print("Detailed Layer Information")
            print("=" * 70)

            for idx in recommended_indices:
                print(f"\n  Layer {idx} ({idx/num_layers*100:.0f}% depth):")
                print(f"    Role: ", end="")
                if idx == 0:
                    print("Input processing, tokenization features")
                elif idx < num_layers * 0.3:
                    print("Early reasoning, pattern detection")
                elif idx < num_layers * 0.7:
                    print("Intermediate reasoning, concept integration")
                else:
                    print("Late reasoning, output preparation")

    except Exception as e:
        print(f"\n✗ Error loading model: {e}", file=sys.stderr)
        print(f"\nCommon issues:")
        print(f"  - Model not found on HuggingFace Hub")
        print(f"  - Network connection required")
        print(f"  - Insufficient memory/disk space")
        sys.exit(1)


def run_model_info(argv: List[str]) -> None:
    """
    Main entry point for model-info command.

    Usage:
        oversee model-info --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """
    parser = argparse.ArgumentParser(
        description="Get model information and suggested layer indices"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed layer information",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of layer samples (default: 3)",
    )

    args = parser.parse_args(argv)

    print_model_info(args.model, verbose=args.verbose)

    # If custom sample count requested, show that too
    if args.samples != 3:
        try:
            model, _ = ModelLoader.load_model_and_tokenizer(args.model)
            num_layers = ModelLoader.get_num_layers(model)
            indices = suggest_layer_indices(num_layers, args.samples)

            print(f"\n" + "=" * 70)
            print(f"Custom: {args.samples} layers")
            print("=" * 70)
            print(f"  Indices: {indices}")

        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
