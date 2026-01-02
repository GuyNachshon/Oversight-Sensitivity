"""
Checkpoint Manager (T040)

Manages checkpointing for batch execution runs to enable resume after interruption.

Per constitution: Experiment-driven development, reproducibility.
Checkpoints track completed prompt IDs to avoid re-running on resume.
"""

import json
from pathlib import Path
from typing import Set, Optional
from datetime import datetime


class CheckpointManager:
    """Manage execution checkpoints for resumable batch runs."""

    def __init__(self, checkpoint_path: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint JSON file
        """
        self.checkpoint_path = checkpoint_path
        self.completed_prompts: Set[str] = set()
        self.started_at: Optional[datetime] = None
        self.last_updated: Optional[datetime] = None

        # Load existing checkpoint if available
        if checkpoint_path.exists():
            self._load()

    def _load(self) -> None:
        """Load checkpoint from disk."""
        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)

            self.completed_prompts = set(data.get("completed_prompts", []))
            self.started_at = (
                datetime.fromisoformat(data["started_at"]) if "started_at" in data else None
            )
            self.last_updated = (
                datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else None
            )

            print(f"Loaded checkpoint: {len(self.completed_prompts)} prompts completed")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            self.completed_prompts = set()

    def save(self) -> None:
        """Save checkpoint to disk."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "completed_prompts": sorted(list(self.completed_prompts)),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "num_completed": len(self.completed_prompts),
        }

        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

        self.last_updated = datetime.utcnow()

    def mark_completed(self, prompt_id: str) -> None:
        """
        Mark a prompt as completed.

        Args:
            prompt_id: Prompt identifier
        """
        if self.started_at is None:
            self.started_at = datetime.utcnow()

        self.completed_prompts.add(prompt_id)
        self.save()

    def is_completed(self, prompt_id: str) -> bool:
        """
        Check if a prompt has been completed.

        Args:
            prompt_id: Prompt identifier

        Returns:
            True if prompt was already completed
        """
        return prompt_id in self.completed_prompts

    def get_progress(self, total_prompts: int) -> dict:
        """
        Get progress statistics.

        Args:
            total_prompts: Total number of prompts in dataset

        Returns:
            Dict with progress info
        """
        completed = len(self.completed_prompts)
        remaining = total_prompts - completed
        progress_pct = (completed / total_prompts * 100) if total_prompts > 0 else 0

        result = {
            "total": total_prompts,
            "completed": completed,
            "remaining": remaining,
            "progress_percent": progress_pct,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

        # Estimate time remaining if we have timing data
        if self.started_at and completed > 0 and remaining > 0:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            avg_time_per_prompt = elapsed / completed
            estimated_remaining_seconds = avg_time_per_prompt * remaining
            result["estimated_remaining_seconds"] = estimated_remaining_seconds
            result["estimated_remaining_hours"] = estimated_remaining_seconds / 3600

        return result

    def reset(self) -> None:
        """Reset checkpoint (clear all completed prompts)."""
        self.completed_prompts = set()
        self.started_at = None
        self.last_updated = None
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
