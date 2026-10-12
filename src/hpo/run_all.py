"""Command-line entry point for the full experiment (or a smoke test).

Full run:         python -m hpo.run_all
Smoke test:       python -m hpo.run_all --smoke
Resume:           run the same command again; finished runs are skipped.
"""
import argparse

from . import config
from .runner import run_all


def main():
    """Read the command-line options and start the full run or a quick test run."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="B=20, seeds 0-1, into results/smoke")
    p.add_argument("--workers", type=int, default=config.N_WORKERS)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None, help="use seeds 0..N-1")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    budget = a.budget or (20 if a.smoke else config.BUDGET)
    seeds = list(range(a.seeds)) if a.seeds else ([0, 1] if a.smoke else config.SEEDS)
    out = a.out or (config.SMOKE_DIR if a.smoke else config.RESULTS_DIR)
    run_all(out_dir=out, budget=budget, seeds=seeds, n_workers=a.workers)


if __name__ == "__main__":
    main()
