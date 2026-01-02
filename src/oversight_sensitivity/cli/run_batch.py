"""
CLI Command: run-batch (T045)

Execute batch of prompts with progress tracking and checkpointing.
"""

import argparse
from pathlib import Path

from ..experiments.config import ExperimentConfig
from ..experiments.dataset import PromptDataset
from ..experiments.batch_run import BatchExecutor


def run_batch(args):
    """Execute run-batch command."""
    parser = argparse.ArgumentParser(
        description="Execute batch of prompts",
        prog="oversee run-batch",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to ExperimentConfig JSON"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to prompt dataset CSV (default: from config)",
    )

    parser.add_argument(
        "--contexts",
        type=str,
        help="Comma-separated contexts to run (default: from config, e.g., N,A,ARD)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (default: {output_dir}/checkpoint.json)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoint",
    )

    parsed_args = parser.parse_args(args)

    # Load config
    config_path = Path(parsed_args.config)
    print(f"Loading config from {config_path}...")
    config = ExperimentConfig.from_json(config_path)

    # Load dataset
    dataset_path = Path(parsed_args.dataset) if parsed_args.dataset else Path(config.prompt_dataset_path)
    print(f"Loading dataset from {dataset_path}...")
    dataset = PromptDataset.from_csv(dataset_path)
    print(f"Loaded {len(dataset.prompts)} prompts")

    # Parse contexts
    if parsed_args.contexts:
        contexts = [ctx.strip() for ctx in parsed_args.contexts.split(",")]
    else:
        contexts = config.context_conditions

    print(f"Contexts to execute: {', '.join(contexts)}")

    # Setup checkpoint
    checkpoint_path = None
    if not parsed_args.no_resume:
        if parsed_args.checkpoint:
            checkpoint_path = Path(parsed_args.checkpoint)
        else:
            checkpoint_path = Path(config.output_directory) / "checkpoint.json"

    # Initialize batch executor
    print(f"\nInitializing batch executor...")
    batch_executor = BatchExecutor(config, checkpoint_path=checkpoint_path)

    # Execute batch
    print(f"\nStarting batch execution...")
    print(f"Total tasks: {len(dataset.prompts)} prompts × {len(contexts)} contexts = {len(dataset.prompts) * len(contexts)} tasks")
    print()

    skip_completed = not parsed_args.no_resume

    try:
        progress = batch_executor.execute_batch(
            dataset=dataset,
            contexts=contexts,
            skip_completed=skip_completed,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BATCH EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total tasks: {progress['total']}")
        print(f"Completed: {progress['completed']} ({progress['progress_percent']:.1f}%)")
        print(f"Remaining: {progress['remaining']}")
        print()
        print(f"This batch:")
        print(f"  Successes: {progress.get('batch_successes', 0)}")
        print(f"  Failures: {progress.get('batch_failures', 0)}")
        print(f"  Skipped: {progress.get('batch_skipped', 0)}")

        if "estimated_remaining_hours" in progress:
            print(f"\nEstimated time remaining: {progress['estimated_remaining_hours']:.1f} hours")

        print("\nOutput directory: " + config.output_directory)

        if progress["remaining"] > 0:
            print("\nTo resume: run the same command again")

        return 0

    except KeyboardInterrupt:
        print("\n\nBatch execution interrupted!")
        print(f"Progress saved to checkpoint: {checkpoint_path}")
        print("Run the same command again to resume.")
        return 1

    finally:
        batch_executor.cleanup()
