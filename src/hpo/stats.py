"""Statistics (Methods 3.6): Friedman per model (seeds as blocks) -> Holm-corrected
Wilcoxon signed-rank (paired by seed) -> rank-biserial effect size -> Kendall's tau
between the models' method rankings. Plus the tables and series the figures need.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config

ALPHA = 0.05


def load(out_dir=config.RESULTS_DIR):
    """Read the saved results (runs.csv and curves.csv)."""
    out_dir = Path(out_dir)
    runs = pd.read_csv(out_dir / "runs.csv")
    curves_p = out_dir / "curves.csv"
    curves = pd.read_csv(curves_p) if curves_p.exists() else None
    return runs, curves


def check_runs(runs: pd.DataFrame) -> dict:
    """Integrity checks shown in notebook 05."""
    expected = runs["budget"].iloc[0] if len(runs) else config.BUDGET
    return {
        "runs": len(runs),
        "all_full_budget": bool((runs["n_evals"] == runs["budget"]).all()),
        "budget": int(expected),
        "duplicates": int(runs.duplicated(["model", "method", "seed"]).sum()),
        "per_model_method": runs.groupby(["model", "method"]).size().unstack().to_dict(),
    }


# Table 3 (Results 4.1): mean and standard deviation of best AUC and run time, and settings tried.
def distinct_settings(curves: pd.DataFrame) -> pd.DataFrame:
    """Number of different model settings fitted per run (repeats are fitted and counted,
    so 256 - distinct = fits spent on settings already tried)."""
    from . import spaces
    rows = []
    for (m, me, sd), g in curves.groupby(["model", "method", "seed"]):
        keys = {tuple(sorted(spaces.decode(m, json.loads(u)).items())) for u in g["u"]}
        rows.append({"model": m, "method": me, "seed": sd, "distinct": len(keys)})
    return pd.DataFrame(rows)


def summary_table(runs: pd.DataFrame, curves: pd.DataFrame = None) -> pd.DataFrame:
    """Table 3: mean and standard deviation of the best AUC and run time per model and method,
    and how many different settings were tried."""
    g = runs.groupby(["model", "method"])
    t = pd.DataFrame({
        "auc_mean": g["best_auc"].mean(), "auc_std": g["best_auc"].std(ddof=1),
        "time_mean_s": g["wall_time_s"].mean(), "time_std_s": g["wall_time_s"].std(ddof=1),
        "seeds": g.size(),
    })
    if curves is not None:
        t["distinct_settings_mean"] = distinct_settings(curves).groupby(["model", "method"])["distinct"].mean()
    order = pd.MultiIndex.from_product([config.MODELS, config.METHODS])
    return t.reindex([i for i in order if i in t.index])


def pivot(runs: pd.DataFrame, model: str) -> pd.DataFrame:
    """Rows = seeds (blocks), columns = methods, values = best validation AUC."""
    p = runs[runs["model"] == model].pivot(index="seed", columns="method", values="best_auc")
    return p[[m for m in config.METHODS if m in p.columns]].dropna()


# Tables 4 and 5 (Results 4.3): Friedman test per model, then Holm-corrected Wilcoxon pairs where significant.
def friedman(runs: pd.DataFrame) -> pd.DataFrame:
    """Table 4: Friedman test per model, checking whether the five methods differ (seeds are the blocks)."""
    rows = []
    for m in config.MODELS:
        p = pivot(runs, m)
        if p.empty:
            continue
        chi2, pval = st.friedmanchisquare(*[p[c].to_numpy() for c in p.columns])
        rows.append({"model": m, "n_seeds": len(p), "k_methods": p.shape[1],
                     "chi2": chi2, "p": pval, "significant": pval < ALPHA})
    return pd.DataFrame(rows).set_index("model")


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial r in [-1, 1]; > 0 means a tends to beat b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = st.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def pairwise(runs: pd.DataFrame, model: str) -> pd.DataFrame:
    """All method pairs: Wilcoxon signed-rank (two-sided, zero_method='wilcox'),
    Holm-adjusted over the pairs of this model, rank-biserial effect size."""
    p = pivot(runs, model)
    rows = []
    for a, b in itertools.combinations(p.columns, 2):
        d = p[a] - p[b]
        if np.allclose(d, 0):
            pv = 1.0
        else:
            pv = st.wilcoxon(p[a], p[b], zero_method="wilcox", alternative="two-sided").pvalue
        rows.append({"model": model, "a": a, "b": b, "mean_diff": float(d.mean()),
                     "p": float(pv), "r_rb": rank_biserial(p[a], p[b])})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p"])
        df["significant"] = df["p_holm"] < ALPHA
    return df


def pairwise_all(runs, only_where_significant: bool = True):
    """Post-hoc tests per model. By default only for models whose Friedman test is
    significant (the brief: 'post-hoc Wilcoxon (Holm-corrected) where significant')."""
    fr = friedman(runs)
    models_ = [m for m in fr.index if (fr.loc[m, "significant"] or not only_where_significant)]
    parts = [pairwise(runs, m) for m in models_]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# Figure 5 (Results 4.5, RQ3): rank the methods within each seed; agreement between models.
def mean_ranks(runs: pd.DataFrame) -> pd.DataFrame:
    """Rank methods within each seed (1 = best AUC, ties averaged); mean over seeds."""
    out = {}
    for m in config.MODELS:
        p = pivot(runs, m)
        if p.empty:
            continue
        out[m] = p.rank(axis=1, ascending=False, method="average").mean()
    return pd.DataFrame(out).reindex(config.METHODS)


def kendall(ranks: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau for each pair of models: do they rank the methods in the same order? (Fig 5)"""
    rows = []
    for a, b in itertools.combinations(ranks.columns, 2):
        tau, pv = st.kendalltau(ranks[a], ranks[b])
        rows.append({"pair": f"{a}–{b}", "tau": tau, "p": pv})
    return pd.DataFrame(rows)


# Figure 4 (Results 4.4, RQ2): paired AUC gain of each method over Random and over Grid.
def gains(runs: pd.DataFrame, baseline: str = "random") -> pd.DataFrame:
    """Paired AUC gain of every other method over `baseline` ("random" or "grid"),
    per model: mean and 95% t-interval over the 20 seeds (uncorrected)."""
    rows = []
    for m in config.MODELS:
        p = pivot(runs, m)
        if p.empty or baseline not in p:
            continue
        for me in [x for x in config.METHODS if x != baseline and x in p]:
            d = (p[me] - p[baseline]).to_numpy()
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            h = st.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
            rows.append({"baseline": baseline, "model": m, "dims": config.DIMS[m], "method": me,
                         "gain_mean": d.mean(), "ci_low": d.mean() - h, "ci_high": d.mean() + h})
    return pd.DataFrame(rows)


# Figure 2 (Results 4.2): average convergence curve per model and method.
def convergence(curves: pd.DataFrame) -> pd.DataFrame:
    """Mean, 25th and 75th percentile of best-so-far AUC over seeds, per fit number."""
    g = curves.groupby(["model", "method", "eval_no"])["best_so_far"]
    return pd.DataFrame({"mean": g.mean(), "q25": g.quantile(0.25),
                         "q75": g.quantile(0.75)}).reset_index()
