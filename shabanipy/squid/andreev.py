"""Andreev bound state spectrum."""

import numpy as np


def andreev_bound_state_energy(
    phase: np.ndarray, transparency: float = 0.5, gap: float = 1
) -> np.ndarray:
    """Andreev bound state energy as a function of phase difference."""
    return gap * np.sqrt(1 - transparency * np.sin(phase / 2) ** 2)
