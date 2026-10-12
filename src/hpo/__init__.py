"""Metaheuristics vs Grid and Random Search under a matched budget (CT099/CT510-3-M).

Each model fit runs on one thread, so parallel runs do not compete for cores.
These must be set before numpy/xgboost load.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")
