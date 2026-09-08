from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeterminismState:
    seed: int
    cuda_available: bool
    deterministic_algorithms: bool
    caveat: str


def set_seed(seed: int, *, deterministic: bool = True) -> DeterminismState:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return DeterminismState(
        seed=seed,
        cuda_available=torch.cuda.is_available(),
        deterministic_algorithms=deterministic,
        caveat=(
            "CPU validation is deterministic. Cross-release, cross-device, and some CUDA "
            "kernels are not guaranteed bitwise identical by PyTorch."
        ),
    )
