"""
Execution Statistics

Per-token measurements collected during inference.
Per data-model.md Entity 4: ExecutionStatistics
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionStatistics:
    """
    Per-token measurements for a specific layer.

    Per data-model.md: 6 statistics per token
    - logit_entropy: Entropy of output distribution (>= 0)
    - top1_probability: Probability of most likely token ([0, 1])
    - residual_stream_norm: L2 norm of residual (>= 0)
    - activation_variance: Variance across activation dimensions (>= 0)
    - attention_entropy: Entropy of attention weights (>= 0)
    - cosine_similarity_to_prev: Cosine similarity to previous hidden state ([-1, 1])
    """

    run_id: str
    layer_index: int
    token_position: int
    logit_entropy: float
    top1_probability: float
    residual_stream_norm: float
    activation_variance: float
    attention_entropy: float
    cosine_similarity_to_prev: Optional[float] = None  # None for token 0

    def __post_init__(self):
        """Validate per data-model.md rules."""
        # Validate ranges
        if self.logit_entropy < 0:
            raise ValueError(f"logit_entropy must be >= 0, got {self.logit_entropy}")

        if not (0 <= self.top1_probability <= 1):
            raise ValueError(
                f"top1_probability must be in [0, 1], got {self.top1_probability}"
            )

        if self.residual_stream_norm < 0:
            raise ValueError(
                f"residual_stream_norm must be >= 0, got {self.residual_stream_norm}"
            )

        if self.activation_variance < 0:
            raise ValueError(
                f"activation_variance must be >= 0, got {self.activation_variance}"
            )

        if self.attention_entropy < 0:
            raise ValueError(
                f"attention_entropy must be >= 0, got {self.attention_entropy}"
            )

        if self.cosine_similarity_to_prev is not None:
            # Allow small floating point errors
            if not (-1.0001 <= self.cosine_similarity_to_prev <= 1.0001):
                raise ValueError(
                    f"cosine_similarity_to_prev must be in [-1, 1], "
                    f"got {self.cosine_similarity_to_prev}"
                )
            # Clip to valid range
            if self.cosine_similarity_to_prev > 1.0:
                self.cosine_similarity_to_prev = 1.0
            elif self.cosine_similarity_to_prev < -1.0:
                self.cosine_similarity_to_prev = -1.0

        # Token 0 should not have cosine similarity
        if self.token_position == 0 and self.cosine_similarity_to_prev is not None:
            raise ValueError("Token 0 cannot have cosine_similarity_to_prev")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "layer_index": self.layer_index,
            "token_position": self.token_position,
            "logit_entropy": self.logit_entropy,
            "top1_probability": self.top1_probability,
            "residual_stream_norm": self.residual_stream_norm,
            "activation_variance": self.activation_variance,
            "attention_entropy": self.attention_entropy,
            "cosine_similarity_to_prev": self.cosine_similarity_to_prev,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionStatistics":
        """Create from dictionary."""
        return cls(**data)
