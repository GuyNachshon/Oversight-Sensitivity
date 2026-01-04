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


Category = Literal["capability_reasoning", "ambiguity_caution", "opinion_stance", "math", "science", "logic", "programming", "technical", "ethics", "workplace", "business", "policy", "lifestyle", "technology", "education", "culture", "career", "media"]
Family = Literal["reasoning", "judgment", "stance"]


@dataclass
class Prompt:
    """Single prompt with metadata."""

    prompt_id: str
    text: str
    category: str  # Flexible category string
    subcategory: str
    family: Optional[str] = None  # Optional family: reasoning, judgment, stance

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

    @classmethod
    def from_jsonl(cls, path: Path, dataset_id: Optional[str] = None) -> "PromptDataset":
        """
        Load dataset from JSONL file.

        Supports flexible field names:
        - prompt_id (required)
        - text or prompt_text (required)
        - category (required)
        - subcategory or source (optional, defaults to "general")
        - family (optional): reasoning, judgment, stance
        """
        prompts = []
        path = Path(path)

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                # Handle flexible field names
                text = row.get("text") or row.get("prompt_text")
                if not text:
                    raise ValueError(f"Row missing 'text' or 'prompt_text': {row}")

                subcategory = row.get("subcategory") or row.get("source", "general")
                family = row.get("family")  # Optional family field

                prompts.append(
                    Prompt(
                        prompt_id=row["prompt_id"],
                        text=text,
                        category=row["category"],
                        subcategory=subcategory,
                        family=family,
                    )
                )

        return cls(
            dataset_id=dataset_id or path.stem,
            prompts=prompts,
            created_at=datetime.now(),
        )

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
