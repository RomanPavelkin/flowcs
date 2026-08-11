"""Reproducibility helper.

The original research code did not fix random seeds. This helper is a
repo-quality addition: call it at the start of a script to make runs more
reproducible. It does not change any model or training logic.
"""
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed python, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
