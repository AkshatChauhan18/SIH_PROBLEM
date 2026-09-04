"""
src/utils/device.py
Hardware execution helper and mixed-precision contexts for RTX 4060 GPU.
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(preference: str = "cuda") -> torch.device:
    """Returns torch.device based on availability and preference."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_autocast_context(device: torch.device, use_amp: bool = True):
    """Returns mixed precision autocast context manager (bfloat16 for RTX 4060 stability)."""
    if device.type == "cuda" and use_amp:
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cpu", enabled=False)
