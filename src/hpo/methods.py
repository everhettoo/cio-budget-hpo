"""The five search strategies. Each one only talks to the BudgetedEvaluator."""
import os

os.environ.setdefault("TQDM_DISABLE", "1")   # mealpy >= 3.1 shows progress bars

import numpy as np

from . import config, spaces, starts
from .evaluator import BudgetExhausted


def run_grid(ev, seed=None):
    """Fixed grid at cell centres (deterministic: the seed is ignored)."""
    try:
        for u in spaces.grid_unit_points(ev.model):
            ev(u)
    except BudgetExhausted:
        pass


def run_random(ev, seed: int):
    """Random Search: the seed's 16 shared starts, then its own draws."""
    d = spaces.dims(ev.model)
    pts = np.vstack([starts.shared_starts(seed, d),
                     starts.random_rest(seed, d, max(ev.budget - config.POP_SIZE, 0))])
    try:
        for u in pts:
            ev(u)
    except BudgetExhausted:
        pass


def _algo_class(name: str):
    """Return the GA, DE or PSO class named in config."""
    from . import algorithms
    return getattr(algorithms, config.METAHEURISTICS[name]["cls"])


def run_metaheuristic(name: str, ev, seed: int):
    """GA / DE / PSO (own update rules on the mealpy framework), starting from the
    seed's 16 shared starts. The 16 starts are fitted first (fits 1-16), then 15 rounds of 16.
    The evaluator stops the run at the budget even if mealpy would continue.
    """
    from mealpy import FloatVar
    d = spaces.dims(ev.model)
    problem = {"obj_func": ev, "bounds": FloatVar(lb=[0.0] * d, ub=[1.0] * d),
               "minmax": "max", "log_to": None}
    algo = _algo_class(name)(epoch=config.EPOCHS, pop_size=config.POP_SIZE,
                               **config.METAHEURISTICS[name]["params"])
    try:
        algo.solve(problem, starting_solutions=starts.shared_starts(seed, d), seed=seed)
    except BudgetExhausted:
        pass
    return algo


def run(method: str, ev, seed: int):
    """Run the chosen method (grid, random, ga, de or pso) for one seed."""
    if method == "grid":
        return run_grid(ev, seed)
    if method == "random":
        return run_random(ev, seed)
    if method in config.METAHEURISTICS:
        return run_metaheuristic(method, ev, seed)
    raise ValueError(method)
