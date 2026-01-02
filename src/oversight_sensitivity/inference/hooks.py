"""
Hook Registration (T017)

Register PyTorch forward hooks to extract hidden states and attention.
"""

import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM


class HookManager:
    """Manage PyTorch forward hooks for statistics extraction."""

    def __init__(self, model: AutoModelForCausalLM, layer_indices: List[int]):
        """
        Initialize hook manager.

        Args:
            model: HuggingFace model to attach hooks to
            layer_indices: Which layers to extract statistics from (0-indexed)
        """
        self.model = model
        self.layer_indices = layer_indices
        self.hooks = []
        self.activations = {}  # Store captured activations
        self.attention_weights = {}  # Store captured attention weights

    def register_hooks(self) -> None:
        """
        Register forward hooks on specified layers.

        Hooks capture:
        - Hidden states (residual stream)
        - Attention weights
        """
        # Get the model's transformer layers
        # This works for most HuggingFace models (GPT-2, LLaMA, etc.)
        if hasattr(self.model, "transformer"):
            layers = self.model.transformer.h  # GPT-2 style
        elif hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                layers = self.model.model.layers  # LLaMA style
            elif hasattr(self.model.model, "decoder"):
                layers = self.model.model.decoder.layers  # T5 decoder style
            else:
                raise ValueError("Unsupported model architecture")
        else:
            raise ValueError("Unsupported model architecture")

        # Register hooks on specified layers
        for layer_idx in self.layer_indices:
            layer = layers[layer_idx]

            # Hook for hidden states
            def make_hidden_hook(idx):
                def hook(module, input, output):
                    # output is typically a tuple: (hidden_states, ...)
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    self.activations[idx] = hidden_states.detach()

                return hook

            handle = layer.register_forward_hook(make_hidden_hook(layer_idx))
            self.hooks.append(handle)

            # Hook for attention weights (if available)
            if hasattr(layer, "self_attn") or hasattr(layer, "attn"):
                attn_module = layer.self_attn if hasattr(layer, "self_attn") else layer.attn

                def make_attn_hook(idx):
                    def hook(module, input, output):
                        # Attention output is typically (attn_output, attn_weights)
                        if isinstance(output, tuple) and len(output) > 1:
                            attn_weights = output[1]
                            if attn_weights is not None:
                                self.attention_weights[idx] = attn_weights.detach()

                    return hook

                handle = attn_module.register_forward_hook(make_attn_hook(layer_idx))
                self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.attention_weights = {}

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """
        Get captured activations from last forward pass.

        Returns:
            Dict mapping layer_idx -> hidden_states tensor
        """
        return self.activations.copy()

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """
        Get captured attention weights from last forward pass.

        Returns:
            Dict mapping layer_idx -> attention_weights tensor
        """
        return self.attention_weights.copy()

    def clear(self) -> None:
        """Clear captured activations and attention weights."""
        self.activations = {}
        self.attention_weights = {}
