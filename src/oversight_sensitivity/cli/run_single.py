"""
CLI Command: run-single

Execute single prompt with specified context condition.
Per contracts/cli-commands.md Command 1.
"""

import argparse
from pathlib import Path

from ..experiments.config import ExperimentConfig
from ..inference.executor import PromptExecutor


def run_single(args):
    """Execute run-single command."""
    parser = argparse.ArgumentParser(
        description="Execute single prompt with context condition",
        prog="oversee run-single",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to ExperimentConfig JSON"
    )

    parser.add_argument(
        "--prompt-id", type=str, required=True, help="Unique identifier for the prompt"
    )

    parser.add_argument(
        "--prompt-text", type=str, required=True, help="The prompt text to execute"
    )

    parser.add_argument(
        "--context",
        type=str,
        required=True,
        choices=["N", "EO", "IO", "KW", "REPRIME"],
        help="Context condition (N=neutral, EO=explicit oversight, IO=implicit oversight, KW=keyword, REPRIME=two-stage)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed execution information"
    )

    parsed_args = parser.parse_args(args)

    # Load config
    config_path = Path(parsed_args.config)
    print(f"Loading config from {config_path}...")
    config = ExperimentConfig.from_json(config_path)

    # Override output directory if specified
    if parsed_args.output_dir:
        config.output_directory = parsed_args.output_dir

    # Initialize executor
    print(f"\nInitializing executor for experiment: {config.experiment_id}")
    executor = PromptExecutor(config)

    # Execute prompt
    print(f"\nExecuting prompt '{parsed_args.prompt_id}' with context '{parsed_args.context}'...")
    run = executor.execute_single_prompt(
        prompt_id=parsed_args.prompt_id,
        prompt_text=parsed_args.prompt_text,
        context_condition=parsed_args.context,
    )

    if parsed_args.verbose:
        print(f"\nGenerated text:\n{run.generated_text}\n")
        total_stats = sum(len(stats) for stats in run.statistics_by_layer.values())
        print(f"Statistics collected: {total_stats} measurements across {len(run.statistics_by_layer)} layers")
        print(f"Tokens generated: {run.num_tokens_generated}")
        print(f"Inference time: {run.inference_time_seconds:.2f}s")

    # Save run
    output_path = executor.save_run(run)
    print(f"\nSaved to: {output_path}")

    # Cleanup
    executor.cleanup()

    return 0
