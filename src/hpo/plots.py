"""All figures, drawn in one shared style.

Colours: GA / DE / PSO in blue, orange and aqua;
Grid and Random as neutral greys. Text stays in ink colours, never series colours.
Method names are always printed (PSO's aqua is below 3:1 contrast on white).
One y-axis per chart. Light mode only (paper / print).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import config, spaces

INK, INK2, MUTED, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#8c8a85", "#e6e5e1", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32b2b"])
METAS = ["ga", "de", "pso"]


def style():
    """Set the shared look (fonts, colours, grid lines) used by all figures."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    })


style()


def save(fig, out_dir, name):
    """Save a figure as PNG and PDF (no creation date, so re-running gives identical files)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None})


def _lab(x):
    """Display name for a method or model (e.g. "xgb" -> "XGBoost")."""
    return config.LABELS.get(x, x)


# Figure 1 (Results 4.1): best validation AUC per seed, one box per method, one panel per model.
def fig_boxplots(runs):
    """Fig 1: spread of the best AUC across seeds, one panel per model."""
    models_ = [m for m in config.MODELS if m in set(runs["model"])]
    fig, axes = plt.subplots(1, len(models_), figsize=(3.3 * len(models_), 3.2), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[0], models_):
        sub = runs[runs["model"] == m]
        meths = [x for x in config.METHODS if x in set(sub["method"])]
        vals = [sub[sub["method"] == me]["best_auc"].to_numpy() for me in meths]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.4},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, me in zip(bp["boxes"], meths):
            patch.set_facecolor(config.COLORS[me] + "33")
            patch.set_edgecolor(config.COLORS[me])
            patch.set_linewidth(1.2)
        for i, (me, v) in enumerate(zip(meths, vals), 1):
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=10, color=config.COLORS[me],
                       edgecolor=SURFACE, linewidth=0.5, zorder=3)
        ax.set_xticks(range(1, len(meths) + 1), [_lab(x) for x in meths])
        ax.set_title(f"{_lab(m)} ({config.DIMS[m]} dims)", loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Best validation AUC (per seed)")
    fig.tight_layout()
    return fig


# Figure 2 (Results 4.2): convergence of GA, DE and PSO: mean best-so-far AUC with the 25–75% band.
def fig_convergence(conv):
    """Fig 2: how the best AUC so far improves as fits are used, for GA, DE and PSO."""
    models_ = [m for m in config.MODELS if m in set(conv["model"])]
    fig, axes = plt.subplots(1, len(models_), figsize=(3.3 * len(models_), 3.2), squeeze=False)
    for ax, m in zip(axes[0], models_):
        sub = conv[conv["model"] == m]
        ends = []
        for me in METAS:
            s = sub[sub["method"] == me].sort_values("eval_no")
            if s.empty:
                continue
            c = config.COLORS[me]
            ax.fill_between(s["eval_no"], s["q25"], s["q75"], color=c, alpha=0.14, linewidth=0)
            ax.plot(s["eval_no"], s["mean"], color=c, label=_lab(me))
            ends.append((s["mean"].iloc[-1], me, s["eval_no"].iloc[-1]))
        # zoom on the part the search controls: from the end of the shared starts
        after = sub[(sub["eval_no"] >= config.POP_SIZE) & sub["method"].isin(METAS)]
        if not after.empty:
            lo, hi = after["q25"].min(), after["q75"].max()
            pad = max((hi - lo) * 0.08, 1e-4)
            ax.set_ylim(lo - pad, hi + pad * 2.5)
        ax.axvline(config.POP_SIZE, color=MUTED, linestyle=":", linewidth=1)
        ax.text(config.POP_SIZE, 1.0, " shared starts end", color=INK2, fontsize=7,
                va="top", ha="left", transform=ax.get_xaxis_transform())
        # direct labels at the line ends, nudged apart
        ends.sort()
        span = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
        last = -np.inf
        top = ax.get_ylim()[1]
        n_end = len(ends)
        for idx, (y, me, x) in enumerate(ends):
            y = min(max(y, last + span), top - span * (n_end - 1 - idx))
            ax.text(x * 1.01, y, _lab(me), color=INK, fontsize=8, va="center")
            last = y
        ax.set_title(f"{_lab(m)} ({config.DIMS[m]} dims)", loc="left")
        ax.set_xlabel("Model fits used")
    axes[0][0].set_ylabel("Best-so-far AUC (mean, IQR band)\nzoomed from fit 16")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, bbox_to_anchor=(0.5, -0.06))
    return fig


# Figure 3 (Results 4.3): pairwise effect sizes (rank-biserial) per model; * marks Holm-significant pairs.
def fig_pairwise_heatmap(pairs):
    """Fig 3: effect size for each pair of methods; a star marks a significant difference after Holm correction."""
    models_ = [m for m in config.MODELS if m in set(pairs["model"])]
    fig, axes = plt.subplots(1, len(models_), figsize=(3.6 * len(models_), 3.4), squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = None
    for ax, m in zip(axes[0], models_):
        sub = pairs[pairs["model"] == m]
        meths = [x for x in config.METHODS if x in set(sub["a"]) | set(sub["b"])]
        k = len(meths)
        M = np.full((k, k), np.nan)
        sig = np.zeros((k, k), bool)
        for _, r in sub.iterrows():
            i, j = meths.index(r["a"]), meths.index(r["b"])
            M[i, j], M[j, i] = r["r_rb"], -r["r_rb"]
            sig[i, j] = sig[j, i] = bool(r.get("significant", False))
        im = ax.imshow(M, cmap=DIV, norm=norm)
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                v = M[i, j] + 0.0
                txt = ("0.00" if abs(v) < 0.005 else f"{v:+.2f}") + ("*" if sig[i, j] else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(v) > 0.6 else INK)
        ax.set_xticks(range(k), [_lab(x) for x in meths])
        ax.set_yticks(range(k), [_lab(x) for x in meths] if ax is axes[0][0] else [])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{_lab(m)} ({config.DIMS[m]} dims)", loc="left")
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    cb.set_label("Rank-biserial r (row vs column)\n* Holm-adjusted p < 0.05", color=INK2)
    cb.outline.set_visible(False)
    return fig


# Figure 4 (Results 4.4, RQ2): AUC gain over Random and over Grid as the number of hyperparameters grows.
def fig_gains(gains_df):
    """Two panels: gain over Random (left) and over Grid (right); 0 = that baseline."""
    from matplotlib.lines import Line2D
    bases = [b for b in ["random", "grid"] if b in set(gains_df["baseline"])]
    fig, axes = plt.subplots(1, len(bases), figsize=(4.4 * len(bases), 3.5), sharey=True, squeeze=False)
    shown = set()
    for ax, base in zip(axes[0], bases):
        g = gains_df[gains_df["baseline"] == base]
        models_ = [m for m in config.MODELS if m in set(g["model"])]
        others = [x for x in ["grid", "random"] if x != base] + METAS
        offs = dict(zip(others, np.linspace(-0.18, 0.18, len(others))))
        for me in others:
            s_ = g[g["method"] == me].set_index("model").reindex(models_)
            if s_["gain_mean"].isna().all():
                continue
            shown.add(me)
            x = np.arange(len(models_)) + offs[me]
            ax.errorbar(x, s_["gain_mean"], yerr=[s_["gain_mean"] - s_["ci_low"], s_["ci_high"] - s_["gain_mean"]],
                        color=config.COLORS[me], linestyle="--" if me in ("grid", "random") else "-",
                        marker="o", markersize=6, capsize=0, elinewidth=1.2, linewidth=1.6,
                        markeredgecolor=SURFACE)
        ax.axhline(0, color=INK2, linewidth=0.9)
        ax.set_xlim(-0.6, len(models_) - 0.4)
        ax.set_xticks(range(len(models_)), [f"{_lab(m)}\n{config.DIMS[m]} dims" for m in models_])
        ax.set_title(f"vs {_lab(base)} (0 = {_lab(base)})", loc="left")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("AUC gain over the baseline\n(paired mean, 95% CI)")
    order = [m for m in ["grid", "random"] + METAS if m in shown]
    handles = [Line2D([], [], color=config.COLORS[me], marker="o",
                      linestyle="--" if me in ("grid", "random") else "-",
                      markeredgecolor=SURFACE, label=_lab(me)) for me in order]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# Figure 5 (Results 4.5, RQ3): mean rank of each method per model, with Kendall's tau between models.
def fig_rank_heatmap(ranks, tau=None):
    """Fig 5: average rank of each method for each model (1 = best)."""
    r = ranks.dropna(how="all")
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    k = len(r.columns)
    # darker = better (rank 1); scale fixed 1..5 so models are comparable
    im = ax.imshow(r.to_numpy(), cmap=SEQ.reversed(), vmin=1, vmax=len(config.METHODS), aspect="auto")
    for i in range(len(r)):
        for j in range(k):
            v = r.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v < 2.6 else INK)
    ax.set_xticks(range(k), [_lab(c).replace(" ", "\n") for c in r.columns])
    ax.set_yticks(range(len(r)), [_lab(x) for x in r.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label("Mean rank over seeds (1 = best)", color=INK2)
    cb.outline.set_visible(False)
    if tau is not None and len(tau):
        txt = "Kendall's τ: " + ",  ".join(f"{p} {t:+.2f}" for p, t in zip(tau["pair"], tau["tau"]))
        fig.text(0.02, -0.02, txt.replace("svm", "SVM").replace("rf", "RF").replace("xgb", "XGB"),
                 color=INK2, fontsize=7.5)
    fig.tight_layout()
    return fig


# Extra figures shown only in the notebooks (not in the paper).
def fig_class_balance(summary):
    """Extra notebook chart: churn and non-churn customers before and after balancing."""
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    stages = ["After cleaning", "After downsampling"]
    churn = [summary["churn_after_clean"], summary["balanced_churn"]]
    no = [summary["clean_rows"] - summary["churn_after_clean"],
          summary["balanced_rows"] - summary["balanced_churn"]]
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, no, w, color=MUTED, label="No churn")
    b2 = ax.bar(x + w / 2 + 0.01, churn, w, color=config.COLORS["ga"], label="Churn")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, stages)
    ax.set_ylabel("Customers")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fig_coverage_2d(seed=0, n=config.BUDGET):
    """Grid vs Random with the same 256 points: how many distinct values per axis."""
    from . import starts
    g = spaces.grid_unit_points("svm")
    r = np.vstack([starts.shared_starts(seed, 2), starts.random_rest(seed, 2, n - config.POP_SIZE)])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, pts, me, title in [(axes[0], g, "grid", "Grid: 16 × 16"),
                               (axes[1], r, "random", "Random: 256 draws")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=config.COLORS[me], edgecolor=SURFACE, linewidth=0.4)
        ax.plot(pts[:, 0], np.full(len(pts), -0.04), "|", color=INK2, markersize=5, alpha=0.6)
        ax.set_xlim(-0.06, 1.02)
        ax.set_ylim(-0.08, 1.02)
        nd = len(np.unique(np.round(pts[:, 0], 6)))
        ax.set_title(f"{title}  ({nd} distinct C values)", loc="left")
        ax.set_xlabel("C  (unit box, log scale)")
    axes[0].set_ylabel("gamma  (unit box, log scale)")
    fig.tight_layout()
    return fig


def fig_fit_times(bench):
    """Extra notebook chart: how long one fit takes for each model."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    models_ = [m for m in config.MODELS if m in set(bench["model"])]
    vals = [bench[bench["model"] == m]["fit_s"].to_numpy() for m in models_]
    ax.boxplot(vals, widths=0.5, showfliers=False, medianprops={"color": INK},
               boxprops={"color": MUTED}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    rng = np.random.default_rng(1)
    for i, v in enumerate(vals, 1):
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=11, color=config.COLORS["ga"],
                   edgecolor=SURFACE, linewidth=0.4, zorder=3)
        ax.text(i + 0.32, np.median(v), f"median {np.median(v):.2f}s", fontsize=7.5, color=INK2, va="center")
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(models_) + 1), [_lab(m) for m in models_])
    ax.set_ylabel("Seconds per fit (log scale)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_svm_landscape(curves, seed=0):
    """Grid AUCs as the background surface; where GA, DE and PSO searched (one seed)."""
    c = curves[curves["model"] == "svm"].copy()
    grid = c[(c["method"] == "grid") & (c["seed"] == c[c["method"] == "grid"]["seed"].min())]
    if grid.empty:
        return None
    U = np.array([json.loads(x) for x in grid["u"]])
    k0, k1 = config.GRID_LEVELS["svm"]
    Z = np.full((k1, k0), np.nan)
    for (u0, u1), auc in zip(U, grid["auc"]):
        Z[min(int(u1 * k1), k1 - 1), min(int(u0 * k0), k0 - 1)] = auc
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    vmin = np.nanpercentile(Z, 5)
    for ax, me in zip(axes, METAS):
        im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], cmap=SEQ, vmin=vmin,
                       vmax=np.nanmax(Z), aspect="auto", alpha=0.9)
        s = c[(c["method"] == me) & (c["seed"] == seed)]
        if not s.empty:
            P = np.array([json.loads(x) for x in s["u"]])
            ax.scatter(P[:16, 0], P[:16, 1], s=14, marker="s", facecolor="none",
                       edgecolor=INK, linewidth=0.8, label="shared starts")
            ax.scatter(P[16:, 0], P[16:, 1], s=12, color=config.COLORS[me],
                       edgecolor=SURFACE, linewidth=0.4, label=f"{_lab(me)} fits 17+")
            b = s.loc[s["auc"].idxmax()]
            bu = json.loads(b["u"])
            ax.scatter([bu[0]], [bu[1]], s=60, marker="*", color=INK, zorder=4, label="best")
        ax.set_title(f"{_lab(me)}, seed {seed}", loc="left")
        ax.set_xlabel("C: 2^-5 → 2^15")
        ax.grid(False)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
                  handletextpad=0.3, columnspacing=0.8)
    axes[0].set_ylabel("gamma: 2^-15 → 2^3")
    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.01)
    cb.set_label("Validation AUC (grid fits)", color=INK2)
    cb.outline.set_visible(False)
    return fig


def fig_auc_vs_time(table):
    """Own reading only (the cost–benefit result is not in the paper)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (m, me), r in table.iterrows():
        ax.scatter(r["time_mean_s"], r["auc_mean"], s=28, color=config.COLORS[me],
                   marker={"svm": "o", "rf": "s", "xgb": "^"}[m], edgecolor=SURFACE)
        ax.text(r["time_mean_s"], r["auc_mean"], f" {_lab(me)}·{m}", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Mean wall-clock time per run (s, log)")
    ax.set_ylabel("Mean best AUC")
    fig.tight_layout()
    return fig
