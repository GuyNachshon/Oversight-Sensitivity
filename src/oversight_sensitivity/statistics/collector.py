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

                # 1. Logit entropy
                probs = torch.softmax(token_logits, dim=-1).cpu().numpy()
                logit_entropy = float(entropy(probs))

                # 2. Top-1 probability
                top1_prob = float(probs.max())

                # 3. Residual stream norm
                hidden = layer_hidden[token_pos]  # [hidden_dim]
                residual_stream_norm = float(torch.norm(hidden).cpu())

                # 4. Activation variance
                activation_variance = float(torch.var(hidden).cpu())

                # 5. Attention entropy
                if layer_attn is not None and token_pos < layer_attn.shape[1]:
                    attn_dist = layer_attn[:, token_pos, : token_pos + 1]  # [num_heads, token_pos+1]
                    attn_entropies = []
                    for head_idx in range(attn_dist.shape[0]):
                        head_attn = attn_dist[head_idx].cpu().numpy()
                        head_entropy = float(entropy(head_attn))
                        attn_entropies.append(head_entropy)
                    attention_entropy = float(np.mean(attn_entropies))
                else:
                    attention_entropy = 0.0

                # 6. Cosine similarity to previous token
                if token_pos > 0:
                    hidden_prev = layer_hidden[token_pos - 1]
                    cosine_similarity_to_prev = float(
                        torch.nn.functional.cosine_similarity(
                            hidden.unsqueeze(0), hidden_prev.unsqueeze(0)
                        ).cpu()
                    )
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
