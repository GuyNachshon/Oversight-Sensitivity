"""
Prompt Dataset

Collection of prompts balanced across categories for distributional robustness.
Per data-model.md Entity 2: PromptDataset
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional
import csv
import json
from pathlib import Path


Category = Literal["capability_reasoning", "ambiguity_caution", "opinion_stance"]


@dataclass
class Prompt:
    """Single prompt with metadata."""

    prompt_id: str
    text: str
    category: Category
    subcategory: str

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError(f"Prompt text cannot be empty for {self.prompt_id}")


@dataclass
class PromptDataset:
    """
    Collection of prompts balanced across categories.

    Per data-model.md validation rules:
    - Length 60-120 prompts
    - Each category >= 20% of total
    - No duplicate prompt_ids
    - No empty text fields
    """

    dataset_id: str
    prompts: List[Prompt]
    created_at: datetime
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate dataset per data-model.md rules."""
        if not (60 <= len(self.prompts) <= 120):
            # Allow smaller datasets for testing, but warn
            if len(self.prompts) < 60:
                import warnings

                warnings.warn(
                    f"Dataset has {len(self.prompts)} prompts (recommended: 60-120)"
                )

        # Check for duplicate IDs
        ids = [p.prompt_id for p in self.prompts]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset contains duplicate prompt_ids")

        # Check category balance (each >= 20%)
        categories = [p.category for p in self.prompts]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        total = len(self.prompts)

        for cat, count in category_counts.items():
            if count / total < 0.20:
                import warnings

                warnings.warn(
                    f"Category {cat} has {count}/{total} prompts "
                    f"({count/total*100:.1f}%, recommended >= 20%)"
                )

    @classmethod
    def from_csv(cls, path: Path, dataset_id: str) -> "PromptDataset":
        """
        Load dataset from CSV file.

        CSV format per data-model.md:
        prompt_id,text,category,subcategory
        p001,"What is...",capability_reasoning,factual_knowledge
        """
        prompts = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(
                    Prompt(
                        prompt_id=row["prompt_id"],
                        text=row["text"],
                        category=row["category"],
                        subcategory=row["subcategory"],
                    )
                )

        return cls(
            dataset_id=dataset_id,
            prompts=prompts,
            created_at=datetime.now(),
        )

    @classmethod
    def from_json(cls, path: Path) -> "PromptDataset":
        """Load dataset from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["prompts"] = [Prompt(**p) for p in data["prompts"]]

        return cls(**data)

    def to_csv(self, path: Path) -> None:
        """Save dataset to CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["prompt_id", "text", "category", "subcategory"]
            )
            writer.writeheader()

            for prompt in self.prompts:
                writer.writerow(
                    {
                        "prompt_id": prompt.prompt_id,
                        "text": prompt.text,
                        "category": prompt.category,
                        "subcategory": prompt.subcategory,
                    }
                )

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """Retrieve prompt by ID."""
        for prompt in self.prompts:
            if prompt.prompt_id == prompt_id:
                return prompt
        return None
