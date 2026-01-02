"""
Batch Executor (T041)

Execute multiple prompts with progress tracking, checkpointing, and error handling.

Per constitution: Reproducibility, experiment-driven development.
Handles failures gracefully without stopping entire batch.
"""

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from .config import ExperimentConfig
from .dataset import PromptDataset
from .checkpoint import CheckpointManager
from ..inference.executor import PromptExecutor


class BatchExecutor:
    """Execute batch of prompts with checkpointing and progress tracking."""

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Initialize batch executor.

        Args:
            config: Experiment configuration
            checkpoint_path: Path to checkpoint file (optional, enables resume)
        """
        self.config = config
        self.executor = PromptExecutor(config)

        # Initialize checkpoint manager
        if checkpoint_path is None:
            checkpoint_path = Path(config.output_directory) / "checkpoint.json"
        self.checkpoint = CheckpointManager(checkpoint_path)

    def execute_batch(
        self,
        dataset: PromptDataset,
        contexts: Optional[List[str]] = None,
        skip_completed: bool = True,
    ) -> dict:
        """
        Execute all prompts in dataset.

        Args:
            dataset: Prompt dataset to execute
            contexts: List of contexts to run (default: from config)
            skip_completed: Skip prompts already in checkpoint (default: True)

        Returns:
            Dict with execution summary
        """
        if contexts is None:
            contexts = self.config.context_conditions

        # Get prompts to execute
        prompts_to_run = []
        for prompt in dataset.prompts:
            # Check if all contexts for this prompt are complete
            all_contexts_done = all(
                self.checkpoint.is_completed(f"{prompt['prompt_id']}_{ctx}")
                for ctx in contexts
            )

            if skip_completed and all_contexts_done:
                continue

            prompts_to_run.append(prompt)

        if not prompts_to_run:
            print("All prompts already completed!")
            return self.checkpoint.get_progress(len(dataset.prompts) * len(contexts))

        # Track statistics
        successes = 0
        failures = 0
        skipped = 0

        # Progress bar for prompts
        total_tasks = len(prompts_to_run) * len(contexts)
        pbar = tqdm(total=total_tasks, desc="Executing prompts", unit="task")

        for prompt in prompts_to_run:
            prompt_id = prompt["prompt_id"]
            prompt_text = prompt["text"]

            for context in contexts:
                task_id = f"{prompt_id}_{context}"

                # Skip if already completed
                if skip_completed and self.checkpoint.is_completed(task_id):
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix({"success": successes, "failed": failures, "skipped": skipped})
                    continue

                # Execute prompt
                try:
                    run = self.executor.execute_single_prompt(
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        context_condition=context,
                    )

                    # Save run
                    self.executor.save_run(run)

                    # Mark as completed
                    self.checkpoint.mark_completed(task_id)

                    successes += 1

                except Exception as e:
                    print(f"\nError executing {task_id}: {e}")
                    failures += 1

                finally:
                    pbar.update(1)
                    pbar.set_postfix({"success": successes, "failed": failures, "skipped": skipped})

        pbar.close()

        # Get final progress
        progress = self.checkpoint.get_progress(len(dataset.prompts) * len(contexts))
        progress["batch_successes"] = successes
        progress["batch_failures"] = failures
        progress["batch_skipped"] = skipped

        return progress

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.executor.cleanup()
