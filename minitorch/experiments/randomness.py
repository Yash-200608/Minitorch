from __future__ import annotations

import numpy as np


def seed_everything(seed: int) -> None:
    """Centralize NumPy RNG seeding for experiment reproducibility."""
    np.random.seed(seed)
