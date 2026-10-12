"""The shared decoder ([0,1]^d -> real settings) and the grids.

Every method searches the same unit box; this one function turns a point into
model settings, so the search space is identical for all five methods.
"""
import itertools
import math

import numpy as np

from . import config


def dims(model: str) -> int:
    """Return how many hyperparameters the model has (SVM 2, RF 4, XGBoost 5)."""
    return len(config.SPACES[model])


def round_half_up(x: float) -> int:
    """4.5 -> 5 (Python's round() would give 4)."""
    return int(math.floor(x + 0.5))


def decode_value(spec: dict, u: float):
    """Convert one value between 0 and 1 into a real hyperparameter value, using its scale (linear, log2 or whole number)."""
    u = float(np.clip(u, 0.0, 1.0))
    lo, hi = spec["low"], spec["high"]
    x = lo + u * (hi - lo)
    if spec["kind"] == "log2":
        return float(2.0 ** x)
    if spec["kind"] == "int":
        return round_half_up(x)
    return float(x)


def decode(model: str, u) -> dict:
    """Unit-box point -> dict of model settings."""
    specs = config.SPACES[model]
    u = np.asarray(u, dtype=float).ravel()
    if u.size != len(specs):
        raise ValueError(f"{model}: expected {len(specs)} values, got {u.size}")
    return {s["name"]: decode_value(s, ui) for s, ui in zip(specs, u)}


def grid_unit_points(model: str) -> np.ndarray:
    """Grid in the unit box: cell centres (i + 0.5) / k on each dimension."""
    axes = [(np.arange(k) + 0.5) / k for k in config.GRID_LEVELS[model]]
    return np.array(list(itertools.product(*axes)))


def grid_values(model: str) -> dict:
    """Decoded grid values per setting (Appendix A / notebook 02)."""
    out = {}
    for spec, k in zip(config.SPACES[model], config.GRID_LEVELS[model]):
        out[spec["name"]] = [decode_value(spec, (i + 0.5) / k) for i in range(k)]
    return out


def bounds_table(model: str) -> list:
    """List each hyperparameter's search range and scale, one row per setting (Table A1)."""
    rows = []
    for s in config.SPACES[model]:
        if s["kind"] == "log2":
            rng = f"2^{s['low']} – 2^{s['high']}"
        else:
            rng = f"{s['low']} – {s['high']}"
        rows.append({"model": model, "setting": s["name"], "range": rng, "scale": s["kind"]})
    return rows
