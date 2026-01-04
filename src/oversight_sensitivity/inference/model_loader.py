"""
Model Loader (T014)

Loads HuggingFace models with proper device placement and precision.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional


class ModelLoader:
    """Load and manage HuggingFace causal language models."""

    @staticmethod
    def load_model_and_tokenizer(
        model_identifier: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer from HuggingFace.

        Args:
            model_identifier: HuggingFace model name (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            device: Device to load model on ("cuda", "cpu", "mps", or None for auto)
            torch_dtype: Precision (torch.float32, torch.float16, None for auto)

        Returns:
            Tuple of (model, tokenizer)
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Auto-detect dtype if not specified
        if torch_dtype is None:
            # Use bfloat16 for GPU (more numerically stable than float16), float32 for CPU
            if device == "cuda" and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif device in ["cuda", "mps"]:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        print(f"Loading model {model_identifier} on {device} with {torch_dtype}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)

        # Ensure tokenizer has pad token (required for batching)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        # Note: device_map requires accelerate library, so we avoid it for CPU and MPS
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                torch_dtype=torch_dtype,
                device_map="auto",  # Use accelerate for CUDA
                low_cpu_mem_usage=True,
            )
        else:
            # For CPU and MPS, load without device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)

        model.eval()  # Set to inference mode

        print(f"Model loaded: {model.config.num_hidden_layers} layers")

        return model, tokenizer

    @staticmethod
    def get_num_layers(model: AutoModelForCausalLM) -> int:
        """Get number of layers in the model."""
        return model.config.num_hidden_layers

    @staticmethod
    def get_hidden_size(model: AutoModelForCausalLM) -> int:
        """Get hidden size of the model."""
        return model.config.hidden_size
