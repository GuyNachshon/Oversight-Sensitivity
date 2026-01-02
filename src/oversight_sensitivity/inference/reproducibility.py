"""
Reproducibility Setup (T015)

Enforce deterministic behavior across torch, numpy, random.
Constitution Principle I: Reproducibility Above All.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch deterministic algorithms

    Args:
        seed: Random seed to use (must be documented in config per constitution)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Enable deterministic algorithms
    # This may reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.12+ deterministic algorithms
    if hasattr(torch, "use_deterministic_algorithms"):
        # Some operations don't have deterministic implementations
        # Set warn_only=True to allow them but warn
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Fallback for older PyTorch versions
            pass


def get_environment_info() -> dict:
    """
    Capture environment information for reproducibility tracking.

    Returns dict with:
    - pytorch_version
    - transformers_version
    - cuda_version (if available)
    - gpu_model (if available)
    - python_version
    """
    import sys
    import transformers

    env_info = {
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    # CUDA info
    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_model"] = torch.cuda.get_device_name(0)
    else:
        env_info["cuda_version"] = None
        env_info["gpu_model"] = None

    return env_info
