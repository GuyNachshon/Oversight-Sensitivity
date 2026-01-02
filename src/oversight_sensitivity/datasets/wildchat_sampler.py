"""
WildChat-1M Dataset Sampler

Sample diverse, high-quality prompts from WildChat-1M for oversight sensitivity experiments.

Dataset: allenai/WildChat-1M
Paper: https://arxiv.org/abs/2405.01470
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import random

from datasets import load_dataset


def load_wildchat(
    split: str = "train",
    streaming: bool = True,
    num_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Load WildChat-1M dataset.

    Args:
        split: Dataset split ("train")
        streaming: Use streaming mode (recommended for large dataset)
        num_samples: Limit to N samples (for quick testing)

    Returns:
        List of conversation dicts
    """
    print(f"Loading WildChat-1M ({split} split)...")

    dataset = load_dataset(
        "allenai/WildChat-1M",
        split=split,
        streaming=streaming,
    )

    if num_samples:
        # Take first N samples for quick iteration
        samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(sample)
        return samples
    else:
        return list(dataset)


def extract_first_turn(conversation: Dict) -> Optional[str]:
    """
    Extract the user's first message from a conversation.

    Args:
        conversation: WildChat conversation dict

    Returns:
        First user message, or None if not found
    """
    if "conversation" not in conversation:
        return None

    conv = conversation["conversation"]
    if not conv or len(conv) == 0:
        return None

    # First message should be from user
    first_turn = conv[0]
    if first_turn.get("role") == "user":
        return first_turn.get("content", "").strip()

    return None


def filter_suitable_prompts(
    conversations: List[Dict],
    min_length: int = 10,
    max_length: int = 500,
    filter_toxic: bool = True,
    filter_code: bool = False,
) -> List[Dict]:
    """
    Filter conversations to find suitable prompts for oversight experiments.

    Args:
        conversations: List of WildChat conversations
        min_length: Minimum prompt length (characters)
        max_length: Maximum prompt length (characters)
        filter_toxic: Remove toxic/harmful content
        filter_code: Remove code-heavy prompts

    Returns:
        Filtered list with prompt info
    """
    suitable = []

    for conv in conversations:
        prompt = extract_first_turn(conv)
        if not prompt:
            continue

        # Length filter
        if len(prompt) < min_length or len(prompt) > max_length:
            continue

        # Toxic filter (if metadata available)
        if filter_toxic and conv.get("toxic", False):
            continue

        # Code filter (heuristic)
        if filter_code:
            code_indicators = ["```", "def ", "class ", "function ", "import ", "#include"]
            if any(indicator in prompt.lower() for indicator in code_indicators):
                continue

        # Extract metadata
        prompt_info = {
            "prompt_text": prompt,
            "conversation_id": conv.get("conversation_id", ""),
            "language": conv.get("language", "en"),
            "model": conv.get("model", ""),
            "toxic": conv.get("toxic", False),
            "redacted": conv.get("redacted", False),
        }

        suitable.append(prompt_info)

    return suitable


def categorize_prompts(prompts: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize prompts by type (heuristic-based).

    Categories:
    - factual: Questions about facts
    - reasoning: Logic, math, problem-solving
    - advice: Seeking guidance or recommendations
    - creative: Writing, storytelling, ideas
    - explanation: "Explain...", "Why...", "How..."
    - judgment: Ethics, opinions, evaluations
    - other: Uncategorized

    Args:
        prompts: List of prompt dicts

    Returns:
        Dict[category -> List[prompts]]
    """
    categories = {
        "factual": [],
        "reasoning": [],
        "advice": [],
        "creative": [],
        "explanation": [],
        "judgment": [],
        "other": [],
    }

    for prompt_info in prompts:
        text = prompt_info["prompt_text"].lower()

        # Categorization heuristics
        if any(word in text for word in ["what is", "who is", "when did", "where is"]):
            category = "factual"
        elif any(word in text for word in ["calculate", "solve", "if...then", "prove"]):
            category = "reasoning"
        elif any(word in text for word in ["should i", "how can i", "recommend", "suggest"]):
            category = "advice"
        elif any(word in text for word in ["write a", "create a", "imagine", "story"]):
            category = "creative"
        elif any(word in text for word in ["explain", "why does", "how does", "what causes"]):
            category = "explanation"
        elif any(word in text for word in ["is it right", "should we", "ethical", "moral"]):
            category = "judgment"
        else:
            category = "other"

        categories[category].append(prompt_info)

    return categories


def stratified_sample(
    categorized_prompts: Dict[str, List[Dict]],
    total_samples: int = 60,
    min_per_category: int = 5,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Sample prompts with stratification to ensure category diversity.

    Args:
        categorized_prompts: Dict[category -> prompts]
        total_samples: Total number of prompts to sample
        min_per_category: Minimum samples per category
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled prompts with category labels
    """
    random.seed(random_seed)

    # Filter out empty categories
    non_empty = {k: v for k, v in categorized_prompts.items() if len(v) >= min_per_category}

    if not non_empty:
        raise ValueError("Not enough prompts in any category")

    # Calculate samples per category
    num_categories = len(non_empty)
    samples_per_category = max(min_per_category, total_samples // num_categories)

    selected = []

    for category, prompts in non_empty.items():
        # Sample from this category
        n_sample = min(samples_per_category, len(prompts))
        sampled = random.sample(prompts, n_sample)

        # Add category label
        for prompt_info in sampled:
            prompt_info["category"] = category

        selected.extend(sampled)

    # If we have too many, randomly downsample
    if len(selected) > total_samples:
        selected = random.sample(selected, total_samples)

    # Shuffle
    random.shuffle(selected)

    return selected


def save_dataset(
    prompts: List[Dict],
    output_path: Path,
    format: str = "jsonl",
) -> None:
    """
    Save sampled prompts to file.

    Args:
        prompts: List of prompt dicts
        output_path: Output file path
        format: "jsonl" or "json"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w") as f:
            for i, prompt_info in enumerate(prompts, 1):
                entry = {
                    "prompt_id": f"wc_{i:03d}",
                    "prompt_text": prompt_info["prompt_text"],
                    "category": prompt_info.get("category", "other"),
                    "source": "wildchat",
                    "metadata": {
                        "conversation_id": prompt_info.get("conversation_id", ""),
                        "language": prompt_info.get("language", "en"),
                    }
                }
                f.write(json.dumps(entry) + "\n")
    else:
        with open(output_path, "w") as f:
            entries = [
                {
                    "prompt_id": f"wc_{i:03d}",
                    "prompt_text": prompt_info["prompt_text"],
                    "category": prompt_info.get("category", "other"),
                }
                for i, prompt_info in enumerate(prompts, 1)
            ]
            json.dump(entries, f, indent=2)

    print(f"\n✓ Saved {len(prompts)} prompts to {output_path}")


def create_wildchat_dataset(
    output_path: Path,
    num_prompts: int = 60,
    random_seed: int = 42,
    quick_test: bool = False,
) -> None:
    """
    Create a curated WildChat dataset for oversight experiments.

    Args:
        output_path: Where to save the dataset
        num_prompts: Number of prompts to sample
        random_seed: Random seed
        quick_test: Use small sample for quick testing
    """
    print("=" * 70)
    print("WildChat-1M Dataset Sampler")
    print("=" * 70)

    # Load dataset
    if quick_test:
        print("\n[Quick Test Mode] Loading 1000 samples...")
        conversations = load_wildchat(num_samples=1000)
    else:
        print("\n[Full Mode] Loading complete dataset (this may take a while)...")
        conversations = load_wildchat(num_samples=10000)  # Sample 10k for efficiency

    print(f"✓ Loaded {len(conversations)} conversations")

    # Filter suitable prompts
    print("\nFiltering suitable prompts...")
    suitable = filter_suitable_prompts(
        conversations,
        min_length=10,
        max_length=500,
        filter_toxic=True,
        filter_code=False,
    )

    print(f"✓ Found {len(suitable)} suitable prompts")

    # Categorize
    print("\nCategorizing prompts...")
    categorized = categorize_prompts(suitable)

    print("\nCategory distribution:")
    for category, prompts in sorted(categorized.items()):
        print(f"  {category:15s}: {len(prompts):4d} prompts")

    # Stratified sample
    print(f"\nSampling {num_prompts} prompts with stratification...")
    sampled = stratified_sample(
        categorized,
        total_samples=num_prompts,
        min_per_category=5,
        random_seed=random_seed,
    )

    print(f"✓ Sampled {len(sampled)} prompts")

    # Show final distribution
    final_dist = Counter(p["category"] for p in sampled)
    print("\nFinal distribution:")
    for category, count in sorted(final_dist.items()):
        print(f"  {category:15s}: {count:2d} prompts")

    # Save
    print(f"\nSaving to {output_path}...")
    save_dataset(sampled, output_path, format="jsonl")

    # Show sample
    print("\nSample prompts:")
    for i, prompt_info in enumerate(sampled[:3], 1):
        print(f"\n  {i}. [{prompt_info['category']}] {prompt_info['prompt_text'][:100]}...")

    print("\n" + "=" * 70)
    print("DONE! Dataset ready for use.")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample prompts from WildChat-1M")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/datasets/wildchat_60.jsonl"),
        help="Output file path",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=60,
        help="Number of prompts to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (1000 samples only)",
    )

    args = parser.parse_args()

    create_wildchat_dataset(
        output_path=args.output,
        num_prompts=args.num_prompts,
        random_seed=args.seed,
        quick_test=args.quick_test,
    )
