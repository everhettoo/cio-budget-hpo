"""The budget guard. Every fit of every method goes through this one object.

It counts each call (no caching: a repeated setting is fitted and counted again),
refuses call 257, and logs the best-so-far curve. This is the visible proof that
all five methods receive exactly the same number of model fits.
"""
import time

import numpy as np

from . import config, models, spaces


class BudgetExhausted(Exception):
    """Raised when a method asks for a fit beyond the budget."""


class BudgetedEvaluator:
    """Counts every model fit and stops the search after 256 fits, so all methods get the same budget. Also records every fit."""
    def __init__(self, model: str, data, budget: int = config.BUDGET, fitness=None):
        """Set up the guard for one run: model, data, budget, and an optional stand-in score for tests."""
        self.model = model
        self.data = data
        self.budget = budget
        self.count = 0
        self.best_auc = -np.inf
        self.best_u = None
        self.best_params = None
        self.trace = []               # one row per fit
        # fitness(u) -> score; default = real model fit. Tests pass a cheap one.
        self._fitness = fitness

    def __call__(self, u) -> float:
        """Train and score one candidate: convert it to settings, fit the model, return the validation AUC and record the result."""
        if self.count >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} fits used")
        u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        t0 = time.perf_counter()
        if self._fitness is None:
            params = spaces.decode(self.model, u)
            auc = models.fit_and_score(self.model, params, self.data)
        else:
            params, auc = None, float(self._fitness(u))
        fit_s = time.perf_counter() - t0
        self.count += 1
        if auc > self.best_auc:
            self.best_auc, self.best_u, self.best_params = auc, u.copy(), params
        self.trace.append({"eval_no": self.count, "auc": auc,
                           "best_so_far": self.best_auc, "fit_s": fit_s,
                           "u": [round(float(x), 6) for x in u]})
        return auc
