"""GA, DE and PSO: our own update rules on the mealpy framework.

mealpy provides the loop (seeding, the 16 shared starting points, 15 rounds,
evaluation counting); each class below only defines one round (`evolve`), written
to match the cited definition, so behaviour does not depend on the mealpy version.

Shared rules:
- Search happens in the unit box [0,1]^d; the shared decoder maps it to settings.
- Each round evaluates 16 new points: 16 starts + 15 rounds x 16 = 256 fits.
- Edge rule (DE, PSO): a coordinate that leaves [0,1] is placed halfway between its
  previous value and the edge it crossed. GA's operators never leave the box.
"""
import numpy as np
from mealpy.optimizer import Optimizer

LB, UB = 0.0, 1.0


def halfway_repair(new, prev):
    """Edge rule. Returns the repaired point and a mask of the coordinates that were out."""
    new, prev = np.asarray(new, dtype=float), np.asarray(prev, dtype=float)
    low, high = new < LB, new > UB
    fixed = np.where(low, (prev + LB) / 2, np.where(high, (prev + UB) / 2, new))
    return fixed, low | high


class _Base(Optimizer):
    """Plumbing shared by GA, DE and PSO (maximise, keep population order)."""

    def __init__(self, epoch=15, pop_size=16, **params):
        """Store the number of rounds, the population size and the method's settings."""
        super().__init__()
        self.epoch, self.pop_size, self.sort_flag = int(epoch), int(pop_size), False
        for name, value in params.items():          # e.g. pc, pm, k / wf, cr / w, c1, c2
            setattr(self, name, value)

    def positions(self):
        """Return the current population as an array, one row per candidate."""
        return np.array([a.solution for a in self.pop])

    def fitness(self, agents=None):
        """Return the AUC of each candidate (by default, the current population)."""
        return np.array([float(a.target.fitness) for a in (agents or self.pop)])

    def evaluate(self, x):
        """One model fit: wraps the point as a mealpy agent and scores it."""
        agent = self.generate_empty_agent(np.array(x, dtype=float))
        agent.target = self.get_target(agent.solution)
        return agent


class GA(_Base):
    """Generational GA with one elite.

    Per round: 8 pairs of parents by tournament (size k); uniform crossover with
    probability pc; gene-reset mutation with probability pm per gene. The 16 children
    replace the parents; the best parent replaces the worst child if it is better.
    Repeated settings are still fitted and counted (no caching).
    """

    def __init__(self, epoch=15, pop_size=16, pc=0.95, pm=0.1, k=3):
        """Table 2 settings: crossover rate pc, mutation rate pm, tournament size k."""
        super().__init__(epoch, pop_size, pc=pc, pm=pm, k=int(k))

    def tournament(self, fits):
        """Pick k candidates at random and return the position of the best one."""
        idx = self.generator.choice(len(fits), self.k, replace=False)
        return idx[int(np.argmax(fits[idx]))]

    def evolve(self, epoch):
        """One GA generation: choose parents by tournament, mix them by crossover, mutate, and keep the best parent."""
        X, fits = self.positions(), self.fitness()
        d = X.shape[1]
        kids = []
        for _ in range(self.pop_size // 2):
            dad, mom = X[self.tournament(fits)], X[self.tournament(fits)]
            if self.generator.random() < self.pc:                    # uniform crossover
                take_dad = self.generator.random(d) < 0.5
                pair = np.where(take_dad, dad, mom), np.where(take_dad, mom, dad)
            else:
                pair = dad.copy(), mom.copy()
            for child in pair:                                       # gene-reset mutation
                reset = self.generator.random(d) < self.pm
                child[reset] = self.generator.uniform(LB, UB, int(reset.sum()))
                kids.append(child)
        children = [self.evaluate(c) for c in kids]
        child_fits = self.fitness(children)
        if fits.max() > child_fits.max():                            # keep the elite
            children[int(np.argmin(child_fits))] = self.pop[int(np.argmax(fits))]
        self.pop = children


class DE(_Base):
    """DE/rand/1/bin (Storn & Price 1997).

    For each target i: mutant = x_r1 + F (x_r2 - x_r3), with r1, r2, r3 distinct and
    not i; binomial crossover with rate CR plus one forced gene (j_rand); edge rule;
    the trial replaces the target if it is not worse. The whole generation is built
    from the old population and replaced at once.
    """

    def __init__(self, epoch=15, pop_size=16, wf=0.5, cr=0.9):
        """Table 2 settings: F and CR."""
        super().__init__(epoch, pop_size, wf=wf, cr=cr)

    def evolve(self, epoch):
        """One DE generation: build a trial point for each member and keep it if it scores at least as well."""
        X, n = self.positions(), self.pop_size
        d = X.shape[1]
        trials = []
        for i in range(n):
            r1, r2, r3 = self.generator.choice([j for j in range(n) if j != i], 3, replace=False)
            mutant = X[r1] + self.wf * (X[r2] - X[r3])
            cross = self.generator.random(d) < self.cr
            cross[self.generator.integers(d)] = True                 # j_rand
            trial, _ = halfway_repair(np.where(cross, mutant, X[i]), X[i])
            trials.append(trial)
        new = [self.evaluate(t) for t in trials]
        self.pop = [t if ft >= fo else o
                    for t, o, ft, fo in zip(new, self.pop, self.fitness(new), self.fitness())]


class PSO(_Base):
    """Global-best PSO, Clerc & Kennedy (2002) constriction in inertia-weight form:
    v = w v + c1 r1 (p_i - x) + c2 r2 (g - x);  x = x + v.

    Every particle moves every round and keeps its own best p_i; g is the swarm's best,
    updated after each round. Velocities start at 0. Edge rule, and the velocity of a
    repaired coordinate is set to 0.
    """

    def __init__(self, epoch=15, pop_size=16, w=0.729844, c1=1.49618, c2=1.49618):
        """Table 2 settings: inertia w, coefficients c1 and c2; particle memory starts empty."""
        super().__init__(epoch, pop_size, w=w, c1=c1, c2=c2)
        self.X = self.V = self.P = self.Pf = None

    def evolve(self, epoch):
        """One PSO step: move each particle, then update its own best point and the swarm's best point."""
        if self.X is None:                       # round 1 starts from the evaluated starts
            self.X = self.positions()
            self.V = np.zeros_like(self.X)
            self.P, self.Pf = self.X.copy(), self.fitness()
        g = self.P[int(np.argmax(self.Pf))].copy()
        r1, r2 = self.generator.random(self.X.shape), self.generator.random(self.X.shape)
        self.V = self.w * self.V + self.c1 * r1 * (self.P - self.X) + self.c2 * r2 * (g - self.X)
        Xn, out = halfway_repair(self.X + self.V, self.X)
        self.V[out] = 0.0
        self.pop = [self.evaluate(x) for x in Xn]
        f = self.fitness()
        better = f > self.Pf
        self.P[better], self.Pf[better] = Xn[better], f[better]
        self.X = Xn
