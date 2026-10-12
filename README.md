# Metaheuristics vs Grid and Random Search under a Matched Evaluation Budget

This repository compares five ways of tuning machine-learning models: **Genetic Algorithm (GA), Differential Evolution (DE),
Particle Swarm Optimization (PSO), Grid Search and Random Search**. Each one tunes three models with a growing number of
hyperparameters: **SVM (2), Random Forest (4) and XGBoost (5)**. Every method gets **exactly 256 model fits per run**, and
every method–model pair is repeated with **20 seeds**. Methods are compared on **validation AUC** with Friedman,
Holm-corrected Wilcoxon, rank-biserial and Kendall's tau statistics.

**Data:** Telco Customer Churn (BlastChar, 2018, Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
7,043 customers × 21 columns, in `data/raw/`. The code removes the customer ID (the only
personal identifier) and 11 rows with a blank `TotalCharges`, balances churn and no-churn to 1,869 each (3,738 rows), and
splits them 70/30 into training and validation rows (fixed seed).

**Three ways to use this repository**

| You want to… | Do this | Setup needed? |
|---|---|---|
| **See the results only** | Open `results/figures/` and `results/tables/`, or open notebooks 06–08 (their saved outputs show every table and figure). Section 4 explains how to read them. | No |
| **Redraw the tables and figures** from the saved results | Set up (section 1), then run notebooks 06–08 (~1 min; no models are trained). | Yes |
| **Reproduce the whole experiment** | Set up (section 1), then follow section 3 (~1 h). | Yes |

## 1. Setup

Needs Python 3.12 and git.

```bash
git clone git@github.com:everhettoo/cio-budget-hpo.git
cd cio-budget-hpo
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # libraries
pip install -e .                   # this project's code (src/hpo)
```

On macOS, XGBoost needs the OpenMP library for its parallel computing: if `import xgboost` fails, run `brew install libomp`.

## 2. Code explanation

### 2.1 Notebook flow: what you run, and what each step reads and writes

![Notebook flow](docs/notebook_flow.png)

| Phase | Notebook / command | What it does | Reads | Writes |
|---|---|---|---|---|
| Before the run | 01 Data preparation | Row counts at each step, customer ID removed, class balance, 45 encoded inputs | `config.py`, `data/raw/` | `figures/nb01_class_balance` |
| | 02 Search spaces and starts | Decoder, 256 grid points per model, the 16 shared starting points | `config.py` | `tables/table2_settings.csv`, `tables/tableA1_bounds.csv`, `figures/nb02_grid_vs_random_coverage` |
| | 03 Budget checks | Every method uses exactly 256 fits (cheap stand-in score, no model fits) | `config.py` | printed checks only |
| | 04 Timing and smoke test | 30 real fits per model → run-time estimate; short test run | `config.py`, `data/raw/` | `timing.csv`, `smoke/`, `figures/nb04_fit_times` |
| The run | `python -m hpo.run_all` (terminal) | The 300 runs (3 models × 5 methods × 20 seeds) | `config.py`, `data/raw/` | `runs.csv`, `curves.csv` |
| | 05 Full experiment | Watches progress (x / 300); every run = 256 fits | `runs.csv` | printed checks only |
| After the run | 06 Results tables | Table 3, Figures 1–2 | `runs.csv`, `curves.csv` | `tables/table3_mean_std.csv`, `tables/fig2_convergence.csv`, `figures/fig1_boxplots`, `figures/fig2_convergence` |
| | 07 Statistics | Tables 4–5, Figures 3–5 | `runs.csv` | `tables/table4_friedman.csv`, `tables/table5_pairwise.csv`, `tables/fig4_gains.csv`, `tables/fig5_*.csv`, `figures/fig3–fig5` |
| | 08 Extra checks | How each method searched, AUC vs time (not in the paper) | `runs.csv`, `curves.csv` | `figures/nb08_svm_landscape`, `figures/nb08_auc_vs_time` |

All outputs go to `results/`.

### 2.2 Code flow: what happens inside the run

![Code flow](docs/code_flow.png)

| Step | File | What it does |
|---|---|---|
| 1 | `src/hpo/config.py` | Every fixed number: budget 256, seeds 0–19, search ranges, grid levels, GA/DE/PSO settings. |
| 2 | `src/hpo/data.py` | Raw CSV → clean → remove customer ID → balance → 70/30 split → encode (fitted on the training rows only). |
| 3 | `src/hpo/run_all.py`, `runner.py` | 300 jobs (3 models × 5 methods × 20 seeds) in parallel; each finished run is saved at once, so a stopped run can resume. |
| 4 | `src/hpo/starts.py` | 16 shared starting points per seed, used by Random, GA, DE and PSO. |
| 5 | `src/hpo/methods.py`, `algorithms.py` | The five methods; `algorithms.py` holds the GA, DE and PSO update rules. Each method proposes points in the box [0,1]^d. |
| 6 | `src/hpo/evaluator.py` | **The budget guard.** Every fit of every method goes through it: it counts fits (no caching), refuses fit 257 and records every fit. |
| 7 | `src/hpo/spaces.py` | Turns a point in [0,1]^d into real settings (log scale for C, gamma, learning rate; whole numbers rounded half up). |
| 8 | `src/hpo/models.py` | Fits SVM, Random Forest or XGBoost and returns the validation AUC. |
| 9 | `src/hpo/stats.py`, `plots.py` | Tables and figures from `results/` (called by notebooks 06–07). |

## 3. Reproduce the results

The finished results are already in `results/`. **If you only want to see them, you do not need this section: go to section 4.**

| Step | Command / notebook | Time | What it does |
|---|---|---|---|
| 1 | `pytest` | ~10 s | 48 unit tests: each rule works (data counts, decoder, grids, starts, update rules, every method = 256 fits) |
| 2 | `python tools/verify.py` | ~1–2 min | Checks that your copy behaves exactly like the reported run (search behaviour, 300 runs, all 9 tables). Expect `PASS`. |
| 3 | Rename `results/` to `results_reported/` | — | **Needed:** the run skips finished runs, so with the existing `results/` it would do nothing |
| 4 | Notebooks 01–04 | ~6 min | Data, search spaces (Table 2, Table A1), budget checks, timing estimate and a small smoke test |
| 5 | Full run (see below) | ~50 min | The 300 runs → new `results/runs.csv` and `curves.csv` |
| 6 | Notebook 05 | — | Progress and integrity: 300/300 runs, every run = 256 fits |
| 7 | Notebooks 06–08 | ~1 min | Tables 3–5 and Figures 1–5 from your new results |
| 8 | `python tools/verify.py` | ~1–2 min | Checks your new results are complete and the tables match them. Expect `PASS`. |

**Full run.** Run it in a terminal (not a notebook cell), and keep the computer awake:

| Platform | Command |
|---|---|
| macOS | `caffeinate -i python -m hpo.run_all` |
| Linux | `systemd-inhibit --what=idle:sleep python -m hpo.run_all` |
| Windows | Set Power → Sleep to **Never** while it runs, then `python -m hpo.run_all` |

- **Cores:** the run uses 10 parallel workers. On a machine with fewer cores, add `--workers N` (e.g. `--workers 4`).
- **If it stops** (sleep, crash, closed terminal): run the same command again. Finished runs are skipped; at most the runs in progress are repeated.
- **Quick test first (optional):** `python -m hpo.run_all --smoke` runs 20 fits × 2 seeds into `results/smoke/` in a few minutes.
- **Your numbers will be close, not identical:** AUCs differ slightly between machines (up to ~0.001), so compare tables and figures by their pattern and conclusions, not the last digit.

## 4. Understand the results

All files are in `results/tables/` (CSV) and `results/figures/` (PNG and PDF). Examples below are from the reported run.

| Output | What it shows | How to read it |
|---|---|---|
| **Table 3** `table3_mean_std.csv` | Mean ± std of the best validation AUC over 20 seeds, run time, and different settings tried (of 256) | Higher AUC is better; Grid has std 0 because it is the same every seed. E.g. XGBoost: PSO 0.84353 vs Grid 0.83952. GA tries fewer different settings because it repeats some. |
| **Figure 1** `fig1_boxplots` | Best AUC per seed, one box per method | Spread across seeds; boxes that barely overlap suggest a real difference. |
| **Figure 2** `fig2_convergence` | Best AUC so far vs fits used (GA, DE, PSO; mean and 25–75% band) | All three start from the same 16 points, so the curves only separate through the search itself. |
| **Table 4** `table4_friedman.csv` | Friedman test per model (seeds as blocks) | p < 0.05 means the five methods do not all perform the same. It was significant for all three models. |
| **Table 5** `table5_pairwise.csv` + **Figure 3** | Holm-corrected Wilcoxon test for each pair of methods, with rank-biserial r | `significant = True` means the pair differs after correction. r runs from −1 to +1: +1 means the first method won on every seed. |
| **Figure 4** `fig4_gains` | Mean AUC gain of each method over Random (left) and over Grid (right), with 95% intervals, at 2 → 4 → 5 hyperparameters | An interval above 0 means a consistent gain over that baseline. E.g. PSO over Grid: +0.0008 (SVM), +0.0004 (RF), +0.0040 (XGBoost). |
| **Figure 5** `fig5_rank_heatmap` | Mean rank of each method per model (1 = best) and Kendall's tau between models | Similar ranks across columns mean the ranking holds across models. PSO had the best mean rank for all three models (1.55, 2.00, 1.55). Tau (0.53–0.80) is read descriptively, as there are only 5 methods. |

**Read with care:** the differences between methods are small (at most 0.004 AUC) compared with the noise of one 30% validation split (standard error about 0.012), and all results come from one data set.

## 5. Notes

- **Reported run:** 2026-10-10 on a MacBook Pro (M3 Max), Python 3.12.9, 10 workers, ~50 min. Main versions: mealpy 3.0.2, xgboost 2.1.4, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, matplotlib 3.11.2 (full list: `requirements-lock.txt`).
- **Approximate run times** on the MacBook Pro (M3 Max, 10 workers): `pytest` ~10 s · `verify.py` ~1–2 min · notebooks 01–04 ~6 min · full run ~50 min · notebooks 06–08 ~1 min · **about 1 hour in total**. Slower machines or fewer workers take longer.
- **Platforms:** tested on macOS and Linux; Windows should work but has not been tested.
- **mealpy:** the GA, DE and PSO update rules are written in `algorithms.py`, so behaviour is the same on mealpy 3.0.2 (pinned) and 3.1.0.
- **Checks:** `tests/test_core.py` holds the 48 unit tests. `tools/verify.py` compares against a stored fingerprint of the reported run's search in `tests/golden/`; update it (`--update-golden`) only after an intended change to the methods.
- **Smoke results in the notebooks:** start Jupyter with `HPO_RESULTS=smoke` (macOS/Linux: `HPO_RESULTS=smoke jupyter lab`; Windows: `$env:HPO_RESULTS="smoke"; jupyter lab`).
