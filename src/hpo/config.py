"""Every fixed number of the study, in one place.

Every module, test and notebook reads its values from here, so a change here applies
everywhere. All values were fixed before any run and are reported in the paper
(search ranges in Table A1, GA/DE/PSO settings in Table 2).
"""
from pathlib import Path

# Paths: where the input data and the results are stored, relative to the repo root.
ROOT = Path(__file__).resolve().parents[2]          # the repo root
RAW_CSV = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_DIR = ROOT / "results"                       # full run
SMOKE_DIR = ROOT / "results" / "smoke"               # small test runs

# Data and model seeds: fixed once, so every run uses the same training and validation
# rows, and the same settings always give the same AUC.
DATA_SEED = 42            # downsampling + 70/30 split, fixed once
VAL_SIZE = 0.30           # 70/30 stratified hold-out, no test set
MODEL_SEED = 0            # model random_state, fixed -> same setting, same AUC

# Budget and seeds: every run gets exactly 256 model fits. GA, DE and PSO use 16 candidates
# per round: 16 starting points + 15 rounds x 16 = 256. Each model and method is repeated
# for 20 seeds.
BUDGET = 256              # fitness evaluations (= model fits) per run
POP_SIZE = 16             # GA / DE / PSO population; also the shared starts
EPOCHS = 15               # 16 starts + 15 rounds x 16 = 256
SEEDS = list(range(20))   # search seeds 0-19

MODELS = ["svm", "rf", "xgb"]
METHODS = ["grid", "random", "ga", "de", "pso"]

# Search spaces (Table A1): one entry per hyperparameter, in the order of the unit-box dimensions.
# kind: "log2" (value = 2 to the power x; low/high are exponents), "int" (whole number) or "float".
SPACES = {
    "svm": [  # Wainer & Fonseca 2021 (Eq. 5)
        {"name": "C",     "kind": "log2", "low": -5,  "high": 15},
        {"name": "gamma", "kind": "log2", "low": -15, "high": 3},
    ],
    "rf": [   # Probst et al. 2019a Table 1 (+ runtime cap on trees)
        {"name": "n_estimators",      "kind": "int",   "low": 50,  "high": 500},
        {"name": "max_depth",         "kind": "int",   "low": 3,   "high": 30},
        {"name": "min_samples_split", "kind": "int",   "low": 2,   "high": 20},
        {"name": "max_features",      "kind": "float", "low": 0.1, "high": 1.0},
    ],
    "xgb": [  # Probst et al. 2019a Table 1 (+ runtime cap on trees)
        {"name": "learning_rate",    "kind": "log2",  "low": -10, "high": 0},
        {"name": "n_estimators",     "kind": "int",   "low": 50,  "high": 1000},
        {"name": "subsample",        "kind": "float", "low": 0.1, "high": 1.0},
        {"name": "max_depth",        "kind": "int",   "low": 1,   "high": 15},
        {"name": "colsample_bytree", "kind": "float", "low": 0.1, "high": 1.0},
    ],
}

# Grid Search: number of grid values per hyperparameter (same order as SPACES). The product is 256
# for every model, so Grid uses the full budget. XGBoost: 4 values for the three settings that
# matter most (Probst et al. 2019a), 2 for the others.
GRID_LEVELS = {
    "svm": [16, 16],
    "rf":  [4, 4, 4, 4],
    "xgb": [4, 4, 4, 2, 2],
}

# GA, DE and PSO settings (Table 2): fixed before any run, the same for all three models.
# cls = class in algorithms.py, params = its settings, rule = short description shown in Table 2.
# Edge rule (DE, PSO): a value that leaves the box goes halfway between its previous value and the edge.
METAHEURISTICS = {
    "ga":  {"cls": "GA",  "params": {"pc": 0.95, "pm": 0.1, "k": 3},
            "rule": "generational, 1 elite; tournament k=3; uniform crossover; gene reset"},
    "de":  {"cls": "DE",  "params": {"wf": 0.5, "cr": 0.9},
            "rule": "DE/rand/1/bin with j_rand; synchronous replacement; halfway edge rule"},
    "pso": {"cls": "PSO", "params": {"w": 0.729844, "c1": 1.49618, "c2": 1.49618},
            "rule": "global best; own best per particle; v0 = 0; halfway edge rule, v = 0 on that axis"},
}

# Running the experiment: parallel workers, and the time limit checked in notebook 04.
N_WORKERS = 10            # parallel processes (10 = performance cores of the MacBook Pro M3 Max used for run 2)
FALLBACK_HOURS = 6.0      # hours; if the estimate is higher, lower the tree caps

# Figures: colour and display name for each method and model, used by every figure.
# GA, DE and PSO have their own colours; Grid and Random are grey.
COLORS = {
    "ga": "#2a78d6", "de": "#eb6834", "pso": "#1baf7a",   # metaheuristics
    "grid": "#8c8a85", "random": "#52514e",                # baselines (greys)
}
LABELS = {"grid": "Grid", "random": "Random", "ga": "GA", "de": "DE", "pso": "PSO",
          "svm": "SVM", "rf": "Random Forest", "xgb": "XGBoost"}
DIMS = {m: len(SPACES[m]) for m in MODELS}   # number of hyperparameters per model
