"""
Statistics Collector (T018)

Compute per-token statistics from captured activations and logits.
"""

import torch
import numpy as np
from typing import Dict, List
from scipy.stats import entropy

from .stats import ExecutionStatistics


class StatisticsCollector:
    """Collect per-token statistics from model activations."""

    @staticmethod
    def collect_statistics(
        run_id: str,
        logits: torch.Tensor,
        hidden_states: Dict[int, torch.Tensor],
        attention_weights: Dict[int, torch.Tensor],
    ) -> List[ExecutionStatistics]:
        """
        Compute statistics for each generated token across all monitored layers.

        Args:
            run_id: Unique identifier for this execution run
            logits: Model logits tensor [batch_size, seq_len, vocab_size]
            hidden_states: Dict mapping layer_idx -> hidden_states [batch_size, seq_len, hidden_dim]
            attention_weights: Dict mapping layer_idx -> attention [batch_size, num_heads, seq_len, seq_len]

        Returns:
            List of ExecutionStatistics, one per (layer, token) combination
        """
        # Assume batch_size=1 for single prompt execution
        logits = logits.squeeze(0)  # [seq_len, vocab_size]
        seq_len = logits.shape[0]

        stats_list = []

        # Iterate over each monitored layer
        for layer_idx in sorted(hidden_states.keys()):
            layer_hidden = hidden_states[layer_idx].squeeze(0)  # [seq_len, hidden_dim]

            # Get attention for this layer (if available)
            layer_attn = attention_weights.get(layer_idx)
            if layer_attn is not None:
                layer_attn = layer_attn.squeeze(0)  # [num_heads, seq_len, seq_len]

            # Iterate over each token position
            for token_pos in range(seq_len):
                token_logits = logits[token_pos]  # [vocab_size]

                # 1. Logit entropy and 2. Top-1 probability
                # Handle potential NaN/inf in logits
                if torch.isnan(token_logits).any() or torch.isinf(token_logits).any():
                    # Clamp extreme values to prevent NaN in softmax
                    token_logits = torch.clamp(token_logits, min=-1e10, max=1e10)
                    token_logits = torch.nan_to_num(token_logits, nan=0.0, posinf=1e10, neginf=-1e10)

                probs = torch.softmax(token_logits, dim=-1).cpu().numpy()

                # Handle NaN in probabilities (can happen with extreme logits)
                if np.isnan(probs).any():
                    probs = np.nan_to_num(probs, nan=1e-10)
                    probs = probs / probs.sum()  # Renormalize

                logit_entropy = float(entropy(probs + 1e-10))  # Add epsilon to avoid log(0)
                if np.isnan(logit_entropy) or logit_entropy < 0:
                    logit_entropy = 0.0

                # 2. Top-1 probability
                top1_prob = float(probs.max())
                if np.isnan(top1_prob) or top1_prob < 0 or top1_prob > 1:
                    top1_prob = 0.0  # Fallback for invalid values

                # 3. Residual stream norm
                hidden = layer_hidden[token_pos]  # [hidden_dim]
                residual_stream_norm = float(torch.norm(hidden).cpu())
                if np.isnan(residual_stream_norm) or residual_stream_norm < 0:
                    residual_stream_norm = 0.0

                # 4. Activation variance
                activation_variance = float(torch.var(hidden).cpu())
                if np.isnan(activation_variance) or activation_variance < 0:
                    activation_variance = 0.0

                # 5. Attention entropy
                if layer_attn is not None and token_pos < layer_attn.shape[1]:
                    attn_dist = layer_attn[:, token_pos, : token_pos + 1]  # [num_heads, token_pos+1]
                    attn_entropies = []
                    for head_idx in range(attn_dist.shape[0]):
                        head_attn = attn_dist[head_idx].cpu().numpy()
                        # Handle NaN in attention weights
                        if np.isnan(head_attn).any():
                            head_attn = np.nan_to_num(head_attn, nan=1e-10)
                            head_attn = head_attn / (head_attn.sum() + 1e-10)
                        head_entropy = float(entropy(head_attn + 1e-10))
                        if not np.isnan(head_entropy):
                            attn_entropies.append(head_entropy)
                    attention_entropy = float(np.mean(attn_entropies)) if attn_entropies else 0.0
                else:
                    attention_entropy = 0.0
                if np.isnan(attention_entropy) or attention_entropy < 0:
                    attention_entropy = 0.0

                # 6. Cosine similarity to previous token
                if token_pos > 0:
                    hidden_prev = layer_hidden[token_pos - 1]
                    cosine_similarity_to_prev = float(
                        torch.nn.functional.cosine_similarity(
                            hidden.unsqueeze(0), hidden_prev.unsqueeze(0)
                        ).cpu()
                    )
                    # Handle NaN and clamp to valid range
                    if np.isnan(cosine_similarity_to_prev):
                        cosine_similarity_to_prev = 0.0
                    cosine_similarity_to_prev = max(-1.0, min(1.0, cosine_similarity_to_prev))
                else:
                    cosine_similarity_to_prev = None  # Token 0 has no previous

                # Create statistics object
                stats = ExecutionStatistics(
                    run_id=run_id,
                    layer_index=layer_idx,
                    token_position=token_pos,
                    logit_entropy=logit_entropy,
                    top1_probability=top1_prob,
                    residual_stream_norm=residual_stream_norm,
                    activation_variance=activation_variance,
                    attention_entropy=attention_entropy,
                    cosine_similarity_to_prev=cosine_similarity_to_prev,
                )

                stats_list.append(stats)

        return stats_list
