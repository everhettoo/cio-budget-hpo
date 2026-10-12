"""One-command check that a change did not alter the experiment.

How it differs from pytest: pytest checks that each piece follows its rule;
this script checks that nothing has changed compared with the reported run.
A small change can still pass every test (for example, a slightly different
PSO formula), but it will fail here. Run both before and after every change.

Run from the repo root (venv active):
    python tools/verify.py                 # all checks
    python tools/verify.py --update-golden # ONLY after an intended, approved behaviour change

Checks
  1. Search behaviour: every method x model x seeds 0-2 on a cheap stand-in objective
     must propose exactly the same points and scores as the stored snapshot
     (tests/golden/search_snapshot.json). Catches any change to GA/DE/PSO, Random,
     Grid, shared starts, decoder or budget guard.
  2. Results integrity: results/runs.csv and curves.csv are complete
     (300 runs x 256 fits, no duplicates, first 16 fits = shared starts).
  3. Tables: every CSV in results/tables is recomputed from runs.csv / curves.csv with the
     current code and must match the committed file (numbers to 1e-12, text exactly).

Exit code 0 = all checks passed. Also run `pytest` (unit tests) separately.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpo import config, methods, spaces, starts, stats  # noqa: E402
from hpo.evaluator import BudgetedEvaluator  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "search_snapshot.json"
SEEDS = [0, 1, 2]


def cheap(u):
    """Stand-in objective with a little structure so every method moves."""
    u = np.asarray(u, dtype=float)
    return -float(np.sum((u - 0.3) ** 2)) + 0.01 * float(np.sin(13 * np.sum(u)))


def search_snapshot() -> dict:
    """Run every model and method on seeds 0-2 with the cheap score and fingerprint each run's fits."""
    out = {}
    for m in config.MODELS:
        for me in config.METHODS:
            for s in SEEDS:
                ev = BudgetedEvaluator(m, None, fitness=cheap)
                methods.run(me, ev, seed=s)
                trace = json.dumps([[r["u"], round(r["auc"], 12)] for r in ev.trace])
                out[f"{m}/{me}/{s}"] = {"fits": ev.count,
                                        "sha256": hashlib.sha256(trace.encode()).hexdigest()}
    return out


def check_search(update: bool) -> bool:
    """Compare the fingerprints with the stored snapshot (or rewrite it with --update-golden)."""
    now = search_snapshot()
    if update:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        print(f"[search]  snapshot written: {len(now)} runs -> {GOLDEN.relative_to(ROOT)}")
        return True
    if not GOLDEN.exists():
        print("[search]  FAIL: no snapshot file; run with --update-golden once on the verified code")
        return False
    ref = json.loads(GOLDEN.read_text())
    diff = sorted(k for k in ref if ref[k] != now.get(k)) + sorted(set(now) - set(ref))
    if diff:
        print(f"[search]  FAIL: {len(diff)} of {len(ref)} runs differ, e.g. {diff[:5]}")
        return False
    print(f"[search]  ok: all {len(ref)} runs identical to the snapshot")
    return True


def check_results() -> bool:
    """Check runs.csv and curves.csv: 300 runs, 256 fits each, complete curves, shared starts."""
    runs, curves = stats.load(config.RESULTS_DIR)
    ok = True
    chk = stats.check_runs(runs)
    expected = {(m, me, s) for m in config.MODELS for me in config.METHODS for s in config.SEEDS}
    if set(map(tuple, runs[["model", "method", "seed"]].values)) != expected or chk["duplicates"]:
        print("[results] FAIL: runs.csv does not hold exactly the 300 expected runs"); ok = False
    if not chk["all_full_budget"] or (runs["n_evals"] != config.BUDGET).any():
        print("[results] FAIL: some run did not use the full budget"); ok = False
    sizes = curves.groupby(["model", "method", "seed"]).size()
    if len(sizes) != len(expected) or (sizes != config.BUDGET).any():
        print("[results] FAIL: curves.csv is incomplete"); ok = False
    bad = 0
    for (m, me, s), g in curves.sort_values("eval_no").groupby(["model", "method", "seed"]):
        if me == "grid":
            continue
        first = np.array([json.loads(u) for u in g["u"].values[:config.POP_SIZE]])
        if not np.allclose(first, starts.shared_starts(s, spaces.dims(m)), atol=1e-6):
            bad += 1
    if bad:
        print(f"[results] FAIL: {bad} runs do not start from the shared starts"); ok = False
    if ok:
        print(f"[results] ok: {len(runs)} runs x {config.BUDGET} fits, complete, shared starts in place")
    return ok


def expected_tables(runs, curves) -> dict:
    """The same calls the notebooks make (02, 06, 07). Index flags match their to_csv."""
    settings = pd.DataFrame([{"method": k.upper(), "update rule": v["rule"],
                              "settings": ", ".join(f"{ {'wf': 'F', 'cr': 'CR'}.get(p, p)}={x}"
                                                    for p, x in v["params"].items()),
                              "population x rounds": f"{config.POP_SIZE} x (1 + {config.EPOCHS})"}
                             for k, v in config.METAHEURISTICS.items()])
    ranks = stats.mean_ranks(runs)
    return {
        "table2_settings.csv": (settings, False),
        "tableA1_bounds.csv": (pd.DataFrame([r for m in config.MODELS for r in spaces.bounds_table(m)]), False),
        "table3_mean_std.csv": (stats.summary_table(runs, curves), True),
        "fig2_convergence.csv": (stats.convergence(curves), False),
        "table4_friedman.csv": (stats.friedman(runs), True),
        "table5_pairwise.csv": (stats.pairwise_all(runs), False),
        "fig4_gains.csv": (pd.concat([stats.gains(runs, "random"), stats.gains(runs, "grid")],
                                     ignore_index=True), False),
        "fig5_mean_ranks.csv": (ranks, True),
        "fig5_kendall.csv": (stats.kendall(ranks), False),
    }


def same_table(committed: Path, df: pd.DataFrame, index: bool) -> bool:
    """True if a recomputed table equals the committed CSV (numbers to 1e-12, text exactly)."""
    a = pd.read_csv(committed)
    b = pd.read_csv(io.StringIO(df.to_csv(index=index)))
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    num = a.select_dtypes("number").columns
    if not np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True):
        return False
    return (a.drop(columns=num).astype(str).to_numpy() == b.drop(columns=num).astype(str).to_numpy()).all()


def check_tables() -> bool:
    """Recompute all 9 tables and compare each with the committed file."""
    runs, curves = stats.load(config.RESULTS_DIR)
    tables = config.RESULTS_DIR / "tables"
    ok = True
    for name, (df, index) in expected_tables(runs, curves).items():
        path = tables / name
        if not path.exists():
            print(f"[tables]  FAIL: {name} missing"); ok = False
        elif not same_table(path, df, index):
            print(f"[tables]  FAIL: {name} differs from a recomputation"); ok = False
    if ok:
        print("[tables]  ok: all 9 tables match a recomputation from runs.csv / curves.csv")
    return ok


def main():
    """Run the three checks; exit 0 for PASS, 1 for FAIL."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-golden", action="store_true",
                    help="rewrite the search snapshot (only after an intended, approved behaviour change)")
    a = ap.parse_args()
    results = [check_search(a.update_golden), check_results(), check_tables()]
    print("PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
