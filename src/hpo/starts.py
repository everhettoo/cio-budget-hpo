"""Shared starting points: one set of 16 per seed.

Random Search uses them as its first 16 draws; GA, DE and PSO use them as their
initial population. Grid is deterministic and does not use them.
"""
import numpy as np

from . import config


def _streams(seed: int):
    """Two independent random streams per seed: (starts, everything after)."""
    ss = np.random.SeedSequence(seed)
    a, b = ss.spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


def shared_starts(seed: int, d: int, n: int = config.POP_SIZE) -> np.ndarray:
    """The seed's 16 starting points in [0,1]^d (same for every method)."""
    return _streams(seed)[0].random((n, d))


def random_rest(seed: int, d: int, n: int) -> np.ndarray:
    """Random Search's own draws after the 16 shared starts."""
    return _streams(seed)[1].random((n, d))
