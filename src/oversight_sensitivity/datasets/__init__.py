"""
Dataset utilities for oversight sensitivity experiments.

Includes samplers for:
- WildChat-1M: Real user-chatbot conversations
"""

from .wildchat_sampler import (
    load_wildchat,
    extract_first_turn,
    filter_suitable_prompts,
    categorize_prompts,
    stratified_sample,
    save_dataset,
    create_wildchat_dataset,
)

__all__ = [
    "load_wildchat",
    "extract_first_turn",
    "filter_suitable_prompts",
    "categorize_prompts",
    "stratified_sample",
    "save_dataset",
    "create_wildchat_dataset",
]
