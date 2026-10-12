"""Unit tests for the hpo package.

What they check: that each piece follows its rule (the budget is exactly 256 fits,
points stay inside the box, DE changes one gene, Holm gives the right numbers, and so on).
They do not check that results stay the same as the reported run; tools/verify.py does that.

Run with `pytest` from the repo root. They take a few seconds: almost all tests use a cheap
stand-in score instead of training a model. Only the last two tests train real models, and
they need the Telco CSV in data/raw.
"""
import numpy as np
import pytest

from hpo import algorithms, config, data, methods, spaces, starts, stats
from hpo.evaluator import BudgetedEvaluator, BudgetExhausted


def cheap(u):
    """Cheap stand-in for a model fit: the score is highest when every value is 0.3.
    Lets the tests run every method in milliseconds."""
    return -float(np.sum((np.asarray(u) - 0.3) ** 2))


@pytest.mark.parametrize("model", config.MODELS)
def test_decoder_hits_bounds(model):
    """The decoder maps 0 to the lower bound and 1 to the upper bound of every hyperparameter,
    for each model, so the ranges match config (Table A1)."""
    lo = spaces.decode(model, np.zeros(spaces.dims(model)))
    hi = spaces.decode(model, np.ones(spaces.dims(model)))
    for s in config.SPACES[model]:
        if s["kind"] == "log2":
            assert lo[s["name"]] == pytest.approx(2.0 ** s["low"])
            assert hi[s["name"]] == pytest.approx(2.0 ** s["high"])
        else:
            assert lo[s["name"]] == pytest.approx(s["low"])
            assert hi[s["name"]] == pytest.approx(s["high"])


def test_decoder_log_scale_midpoint():
    """Log-scale settings are decoded on a log scale: the middle of the SVM C range (2^-5 to
    2^15) is 2^5 = 32."""
    # SVM C: 2^-5 .. 2^15 -> middle of the box is 2^5
    assert spaces.decode("svm", [0.5, 0.5])["C"] == pytest.approx(2.0 ** 5)


def test_round_half_up():
    """Whole-number settings round .5 up (4.5 becomes 5). Python's round() would give 4, so the
    code uses its own rounding."""
    assert spaces.round_half_up(4.5) == 5 and spaces.round_half_up(11.5) == 12
    assert spaces.round_half_up(4.49) == 4


def test_integers_are_ints():
    """Whole-number settings (number of trees, depth, minimum split) come out as Python ints,
    as the models require."""
    p = spaces.decode("rf", [0.37, 0.61, 0.12, 0.5])
    for k in ("n_estimators", "max_depth", "min_samples_split"):
        assert isinstance(p[k], int)


@pytest.mark.parametrize("model", config.MODELS)
def test_grid_has_budget_unique_points(model):
    """For each model the grid has exactly 256 points and no two decode to the same settings,
    so Grid Search spends all 256 fits on different settings."""
    U = spaces.grid_unit_points(model)
    assert len(U) == config.BUDGET
    decoded = {tuple(sorted(spaces.decode(model, u).items())) for u in U}
    assert len(decoded) == config.BUDGET          # no two grid points round to the same setting


def test_grid_cell_centres_and_documented_values():
    """Grid values sit at the centres of equal cells and match the values shown in notebook 02
    (spot check on XGBoost and RF)."""
    g = spaces.grid_values("xgb")
    assert g["max_depth"] == [5, 12]
    assert g["n_estimators"] == [169, 406, 644, 881]
    assert spaces.grid_values("rf")["min_samples_split"] == [4, 9, 13, 18]
    assert spaces.grid_values("rf")["max_depth"] == [6, 13, 20, 27]


def test_starts_same_within_seed_differ_across_seeds():
    """The 16 shared starting points are identical for the same seed, differ between seeds, and
    lie inside the box. This keeps the comparison fair: Random, GA, DE and PSO begin from
    the same points."""
    a, b = starts.shared_starts(3, 4), starts.shared_starts(3, 4)
    assert np.array_equal(a, b) and a.shape == (16, 4)
    assert not np.array_equal(a, starts.shared_starts(4, 4))
    assert ((a >= 0) & (a <= 1)).all()


def test_evaluator_stops_at_budget():
    """The budget guard allows exactly the budget (here 5 fits), raises an error on the next
    fit, and records every fit."""
    ev = BudgetedEvaluator("svm", None, budget=5, fitness=cheap)
    for _ in range(5):
        ev([0.1, 0.2])
    with pytest.raises(BudgetExhausted):
        ev([0.1, 0.2])
    assert ev.count == 5 and len(ev.trace) == 5


def test_evaluator_no_caching_and_best_so_far():
    """A repeated setting is fitted and counted again (no caching), and the best-so-far score
    never goes down."""
    ev = BudgetedEvaluator("svm", None, budget=10, fitness=cheap)
    ev([0.3, 0.3]); ev([0.3, 0.3]); ev([0.9, 0.9])
    assert ev.count == 3                                 # repeat counted again
    bsf = [r["best_so_far"] for r in ev.trace]
    assert bsf == sorted(bsf) and ev.best_auc == pytest.approx(0.0)


@pytest.mark.parametrize("model", config.MODELS)
@pytest.mark.parametrize("method", config.METHODS)
def test_each_method_uses_exact_budget(model, method):
    """Every method uses exactly 256 fits on every model, so all methods get the same budget."""
    ev = BudgetedEvaluator(model, None, budget=config.BUDGET, fitness=cheap)
    methods.run(method, ev, seed=7)
    assert ev.count == config.BUDGET


@pytest.mark.parametrize("method", ["random", "ga", "de", "pso"])
def test_first_16_fits_are_the_shared_starts(method):
    """Random, GA, DE and PSO all spend their first 16 fits on the seed's shared starting
    points."""
    d = spaces.dims("rf")
    ev = BudgetedEvaluator("rf", None, fitness=cheap)
    methods.run(method, ev, seed=11)
    first = np.array([r["u"] for r in ev.trace[:16]])
    assert np.allclose(first, starts.shared_starts(11, d), atol=1e-6)


@pytest.mark.parametrize("method", ["ga", "de", "pso"])
def test_metaheuristic_stops_when_budget_is_smaller(method):
    """With a smaller budget (here 20 fits), GA, DE and PSO stop cleanly at the budget."""
    ev = BudgetedEvaluator("xgb", None, budget=20, fitness=cheap)
    methods.run(method, ev, seed=0)
    assert ev.count == 20


def test_grid_is_deterministic():
    """Grid Search gives the same result for any seed, because the grid is fixed."""
    a = BudgetedEvaluator("svm", None, fitness=cheap); methods.run("grid", a, 0)
    b = BudgetedEvaluator("svm", None, fitness=cheap); methods.run("grid", b, 19)
    assert a.best_auc == b.best_auc


def test_settings_reach_the_algorithm():
    """The GA, DE and PSO settings in config (Table 2) are the values the algorithms actually
    use."""
    ev = BudgetedEvaluator("svm", None, fitness=cheap)
    algo = methods.run_metaheuristic("pso", ev, 0)
    assert algo.w == pytest.approx(0.729844) and algo.c1 == pytest.approx(1.49618)
    ev = BudgetedEvaluator("svm", None, fitness=cheap)
    algo = methods.run_metaheuristic("de", ev, 0)
    assert algo.wf == pytest.approx(0.5) and algo.cr == pytest.approx(0.9)
    ev = BudgetedEvaluator("svm", None, fitness=cheap)
    algo = methods.run_metaheuristic("ga", ev, 0)
    assert algo.pm == pytest.approx(0.1) and algo.pc == pytest.approx(0.95) and algo.k == 3


def test_halfway_edge_rule():
    """Edge rule: a value that leaves [0,1] is placed halfway between its previous value and
    the edge it crossed; values inside are unchanged."""
    new, out = algorithms.halfway_repair([-0.4, 0.5, 1.3], [0.2, 0.6, 0.8])
    assert np.allclose(new, [0.1, 0.5, 0.9]) and out.tolist() == [True, False, True]


def _run_algo(cls, fitness, seed=3, **params):
    """Helper: run one algorithm for the full budget on the 4-dimension RF box with a given
    score function; return the algorithm and the evaluator."""
    from mealpy import FloatVar
    ev = BudgetedEvaluator("rf", None, fitness=fitness)
    problem = {"obj_func": ev, "bounds": FloatVar(lb=[0.0] * 4, ub=[1.0] * 4),
               "minmax": "max", "log_to": None}
    algo = cls(epoch=config.EPOCHS, pop_size=config.POP_SIZE, **params)
    algo.solve(problem, starting_solutions=starts.shared_starts(seed, 4), seed=seed)
    return algo, ev


def test_every_point_stays_in_the_box():
    """Even when the best point lies outside the box, GA, DE and PSO only propose points inside
    [0,1] and use the full 256 fits."""
    edge = lambda u: -float(np.sum((np.asarray(u) - 1.2) ** 2))   # optimum outside the box
    for name in ["ga", "de", "pso"]:
        algo, ev = _run_algo(getattr(algorithms, config.METAHEURISTICS[name]["cls"]), edge,
                             **config.METAHEURISTICS[name]["params"])
        U = np.array([r["u"] for r in ev.trace])
        assert ev.count == 256 and U.min() >= 0 and U.max() <= 1


class _FakeAgent:
    """Minimal stand-in for a mealpy agent (a point and its score), used to test one DE round
    directly."""
    def __init__(self, x, f):
        """Store the point and its score."""
        self.solution = np.asarray(x, float)
        self.target = type("T", (), {"fitness": f})()


def test_de_one_round_is_rand_1_bin_with_j_rand():
    """One DE round follows DE/rand/1/bin: with crossover rate 0, each trial differs from its
    target in exactly one gene (j_rand), and better trials replace their targets."""
    rng = np.random.default_rng(0)
    de = algorithms.DE(epoch=1, pop_size=16, wf=0.5, cr=0.0)   # CR = 0: only j_rand crosses
    de.generator = rng
    X = rng.uniform(0.2, 0.8, (16, 4))
    de.pop = [_FakeAgent(x, 0.0) for x in X]
    trials = []
    de.evaluate = lambda t: (trials.append(t), _FakeAgent(t, 1.0))[1]
    de.evolve(1)
    changed = [(np.abs(t - x) > 1e-12).sum() for t, x in zip(trials, X)]
    assert len(trials) == 16 and all(c == 1 for c in changed)        # exactly one gene each
    assert all(np.allclose(p.solution, t) for p, t in zip(de.pop, trials))  # better trials replace all


def test_ga_elitism_keeps_the_best():
    """After a full GA run, the best point found is still in the population (elitism)."""
    algo, ev = _run_algo(algorithms.GA, cheap, pc=0.95, pm=0.1, k=3)
    best_in_pop = max(float(a.target.fitness) for a in algo.pop)
    assert best_in_pop == pytest.approx(ev.best_auc)


def test_pso_moves_every_particle_and_keeps_own_best():
    """PSO works as in the textbook version: particles keep moving (positions differ from
    personal bests), and the best personal best equals the best score found."""
    algo, ev = _run_algo(algorithms.PSO, cheap, w=0.729844, c1=1.49618, c2=1.49618)
    assert not np.allclose(algo.X, algo.P)          # position != personal best (textbook PSO)
    assert algo.Pf.max() == pytest.approx(ev.best_auc)


def test_holm_matches_hand_example():
    """Holm correction matches a worked example: p = 0.01, 0.04, 0.03 become 0.03, 0.06, 0.06."""
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_rank_biserial_extremes():
    """Rank-biserial effect size is 1 when one method always wins and 0 when there is no
    difference."""
    assert stats.rank_biserial([2, 3, 4], [1, 1, 1]) == 1.0
    assert stats.rank_biserial([1, 1, 1], [1, 1, 1]) == 0.0


# Tests marked with this train real models and need the Telco CSV; they are skipped if it is missing.
needs_csv = pytest.mark.skipif(not config.RAW_CSV.exists(), reason="Telco CSV not in data/raw")


@needs_csv
def test_data_counts_and_pii():
    """Data preparation gives the expected row counts (7,043 raw, 11 blank rows dropped, 3,738
    after balancing, 2,616 train / 1,122 validation) and removes customerID."""
    s = data.summary()
    assert s["raw_rows"] == 7043 and s["clean_rows"] == 7032
    assert s["dropped_blank_totalcharges"] == 11 and s["pii_columns_left"] == []
    assert s["balanced_rows"] == 3738 and s["balanced_churn"] == 1869
    assert s["train_rows"] == 2616 and s["val_rows"] == 1122


@needs_csv
@pytest.mark.parametrize("model", config.MODELS)
def test_real_fit_scores_an_auc(model):
    """One real fit of each model (SVM, RF, XGBoost) runs end to end and returns a sensible
    validation AUC (between 0.5 and 1)."""
    d = data.prepare(model)
    ev = BudgetedEvaluator(model, d, budget=1)
    auc = ev(np.full(spaces.dims(model), 0.5))
    assert 0.5 < auc < 1.0
