"""Run one (model, method, seed) job, many jobs in parallel, and the timing benchmark.

Results are written by the main process only (one writer), one row as soon as
each run finishes, so a crash or a sleeping computer loses at most the runs in flight.
Re-running skips every run already in runs.csv.
"""
import csv
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config, data as data_mod, methods, models, spaces
from .evaluator import BudgetedEvaluator

RUN_FIELDS = ["model", "method", "seed", "budget", "n_evals", "best_auc", "best_params",
              "best_u", "wall_time_s", "fit_time_s", "overhead_s", "finished_at"]
CURVE_FIELDS = ["model", "method", "seed", "eval_no", "auc", "best_so_far", "fit_s", "u"]

_DATA_CACHE = {}


def get_data(model: str):
    """Prepared data, built once per process."""
    if model not in _DATA_CACHE:
        _DATA_CACHE[model] = data_mod.prepare(model)
    return _DATA_CACHE[model]


def run_one(model: str, method: str, seed: int, budget: int = config.BUDGET):
    """Run one model, method and seed; return its summary row and the record of every fit."""
    ev = BudgetedEvaluator(model, get_data(model), budget=budget)
    t0 = time.perf_counter()
    methods.run(method, ev, seed)
    wall = time.perf_counter() - t0
    fit_total = float(sum(r["fit_s"] for r in ev.trace))
    row = {
        "model": model, "method": method, "seed": seed, "budget": budget,
        "n_evals": ev.count, "best_auc": ev.best_auc,
        "best_params": json.dumps(ev.best_params),
        "best_u": json.dumps([round(float(x), 6) for x in ev.best_u]),
        "wall_time_s": wall, "fit_time_s": fit_total, "overhead_s": wall - fit_total,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    curve = [{"model": model, "method": method, "seed": seed, **r, "u": json.dumps(r["u"])}
             for r in ev.trace]
    return row, curve


def _append(path: Path, fields, rows):
    """Add rows to a CSV file, writing the header if the file is new."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def done_keys(out_dir: Path) -> set:
    """Return the runs already saved in runs.csv, so a restart can skip them."""
    p = Path(out_dir) / "runs.csv"
    if not p.exists():
        return set()
    with p.open() as f:
        return {(r["model"], r["method"], int(r["seed"])) for r in csv.DictReader(f)}


def job_list(models_=config.MODELS, methods_=config.METHODS, seeds=config.SEEDS,
             shuffle_seed: int = 0):
    """All jobs in a shuffled order, so no method always runs first or last."""
    jobs = [(m, me, s) for m in models_ for me in methods_ for s in seeds]
    random.Random(shuffle_seed).shuffle(jobs)
    return jobs


def run_all(out_dir=config.RESULTS_DIR, budget=config.BUDGET, models_=config.MODELS,
            methods_=config.METHODS, seeds=config.SEEDS, n_workers=config.N_WORKERS,
            progress=print):
    """Run all jobs in parallel and save each result as soon as it finishes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finished = done_keys(out_dir)
    jobs = [j for j in job_list(models_, methods_, seeds) if j not in finished]
    total = len(finished) + len(jobs)
    progress(f"{len(finished)} runs already done, {len(jobs)} to go, {n_workers} workers")
    if not jobs:
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(run_one, m, me, s, budget): (m, me, s) for m, me, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            row, curve = fut.result()
            _append(out_dir / "runs.csv", RUN_FIELDS, [row])
            _append(out_dir / "curves.csv", CURVE_FIELDS, curve)
            el = (time.perf_counter() - t0) / 60
            progress(f"[{len(finished) + i}/{total}] {row['model']:>3} {row['method']:>6} "
                     f"seed {row['seed']:>2}  AUC {row['best_auc']:.4f}  "
                     f"fits {row['n_evals']}  {row['wall_time_s']:.0f}s  ({el:.1f} min)")


# Timing benchmark (notebook 04): times real fits to estimate how long the full run takes.
def benchmark(model: str, n: int = 30, seed: int = 123) -> list:
    """Fit n random settings from the final search box; return seconds per fit."""
    d = get_data(model)
    rng = np.random.default_rng(seed)
    out = []
    for u in rng.random((n, spaces.dims(model))):
        params = spaces.decode(model, u)
        t0 = time.perf_counter()
        auc = models.fit_and_score(model, params, d)
        out.append({"model": model, "fit_s": time.perf_counter() - t0, "auc": auc,
                    **{k: v for k, v in params.items()}})
    return out


def estimate_hours(mean_fit_s: dict, n_workers: int = config.N_WORKERS,
                   overhead: float = 1.10) -> dict:
    """Full-run estimate: 5 methods x 20 seeds x 256 fits per model."""
    fits = len(config.METHODS) * len(config.SEEDS) * config.BUDGET
    per_model_core_h = {m: s * fits / 3600 for m, s in mean_fit_s.items()}
    core_h = sum(per_model_core_h.values())
    wall_h = core_h * overhead / n_workers
    return {"fits_per_model": fits, "core_hours": per_model_core_h,
            "total_core_hours": core_h, "wall_hours": wall_h,
            "fallback_needed": wall_h > config.FALLBACK_HOURS}
