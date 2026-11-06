"""
Heuristics and metaheuristics for MST, written for pools of hundreds of items.

All methods work on the admissible pool D_t only and use the paper's quantities:
U(S) = 1 - prod(1 - u_i), P(S) = 1 - prod(1 - p_i), C(S) = sum c_i / C_ref,
K(S) = |S| / K_ref (reference scales, optimization/scales.py), J = lambda_P P + lambda_C C + lambda_K K.

Sufficiency is checked through the exact linear form sum_i w_i x_i >= W with
w_i = -ln(1 - u_i) and W = -ln(1 - tau) (Eq. 23 of the paper), up to a relative
tolerance of 1e-12; the returned subsets are re-checked with the product form.

Methods (each returns a sorted tuple of admissible positions, or None when the
admissible pool cannot reach the threshold):
  greedy            Dynamic Greedy-MST of the paper (untruncated marginal utility)
  greedy_truncated  standard submodular-cover greedy: utility gain truncated at tau
  reverse_delete    removes items while the subset stays sufficient, largest J decrease first
  issc              Iyer and Bilmes (2013) iterated submodular set cover: repeatedly
                    replaces J by its modular upper bound at the current solution and
                    solves the resulting submodular cover with the truncated greedy
  simulated_annealing, tabu_search, ga_seeded_repair
                    metaheuristics run for a fixed wall-clock budget, started from (or
                    seeded with) the truncated greedy solution after reverse deletion
"""

from __future__ import annotations

import math
import time

import numpy as np

from scales import REF_COST, REF_ITEMS

EPS = 1e-12
TOL = 1e-12


class Pool:
    def __init__(self, u, p, c, admissible, tau, weights, cost_scale=REF_COST, card_scale=REF_ITEMS,
                 utility="product"):
        adm = np.flatnonzero(np.asarray(admissible, bool))
        self.adm = adm
        u = np.asarray(u, float)[adm]
        p = np.asarray(p, float)[adm]
        c = np.asarray(c, float)[adm]
        self.u, self.p, self.c = u, p, c
        self.m = len(adm)
        self.tau = float(tau)
        self.lp, self.lc, self.lk = weights
        big = 1e6                                   # items with u_i = 1 or p_i = 1
        self.additive = utility == "additive"     # U = sum u_i (shares that add up by construction)
        self.wu = u.copy() if self.additive else np.where(u >= 1.0, big, -np.log1p(-np.minimum(u, 1 - 1e-16)))
        self.pi = np.where(p >= 1.0, big, -np.log1p(-np.minimum(p, 1 - 1e-16)))
        self.W = tau if self.additive else (big if tau >= 1.0 else -math.log1p(-tau))
        tot = c.sum() if cost_scale is None else cost_scale
        self.cn = c / tot if tot > 0 else np.zeros(self.m)
        self.kn = 1.0 / (max(self.m, 1) if card_scale is None else card_scale)
        self.n_eval = 0

    # -- set quantities from aggregated sums ---------------------------------
    def J_from(self, y, cs, k):
        return self.lp * (1.0 - np.exp(-y)) + self.lc * cs + self.lk * k * self.kn

    def feasible_from(self, ws):
        return ws >= self.W * (1 - TOL)

    def sums(self, x):
        x = np.asarray(x, bool)
        return self.wu[x].sum(), self.pi[x].sum(), self.cn[x].sum(), int(x.sum())

    def J(self, x):
        _, y, cs, k = self.sums(x)
        return float(self.J_from(y, cs, k))

    def U_from(self, ws):
        if self.additive:
            return ws
        return 1.0 - np.exp(-np.minimum(ws, 700))

    def U(self, x):
        if self.additive:
            return float(self.u[np.asarray(x, bool)].sum())
        return 1.0 - float(np.prod(1.0 - self.u[np.asarray(x, bool)]))

    def feasible(self, x):
        return self.U(x) >= self.tau - 1e-12 or self.feasible_from(self.sums(x)[0])

    def pool_feasible(self):
        return self.feasible(np.ones(self.m, bool))

    def to_indices(self, x):
        return tuple(int(self.adm[i]) for i in np.flatnonzero(x))


# ---------------------------------------------------------------------------
# Constructive heuristics
# ---------------------------------------------------------------------------

def _greedy(pool, truncated, cost=None, x0=None):
    """Greedy cover. cost: optional modular cost vector replacing marginal J."""
    m = pool.m
    x = np.zeros(m, bool) if x0 is None else x0.copy()
    ws, y, cs, k = pool.sums(x)
    Ucur = float(pool.U_from(ws))
    Jcur = pool.J_from(y, cs, k)
    while not pool.feasible_from(ws):
        cand = ~x
        if not cand.any():
            return None
        ws_new = ws + pool.wu
        U_new = pool.U_from(ws_new)
        gain = (np.minimum(U_new, pool.tau) if truncated else U_new) - Ucur
        if cost is None:
            dJ = pool.J_from(y + pool.pi, cs + pool.cn, k + 1) - Jcur
        else:
            dJ = cost
        ratio = np.where(cand, gain / (dJ + EPS), -np.inf)
        best = ratio.max()
        j = int(np.flatnonzero(ratio == best)[0])   # ties: first position (identifier order)
        x[j] = True
        ws += pool.wu[j]; y += pool.pi[j]; cs += pool.cn[j]; k += 1
        Ucur = float(pool.U_from(ws))
        Jcur = pool.J_from(y, cs, k)
        pool.n_eval += m
    return x


def greedy(pool):
    return _greedy(pool, truncated=False) if pool.pool_feasible() else None


def greedy_truncated(pool):
    return _greedy(pool, truncated=True) if pool.pool_feasible() else None


def reverse_delete(pool, x):
    if x is None:
        return None
    x = x.copy()
    ws, y, cs, k = pool.sums(x)
    while True:
        sel = np.flatnonzero(x)
        ok = pool.feasible_from(ws - pool.wu[sel])
        if not ok.any():
            return x
        Jn = pool.J_from(y - pool.pi[sel], cs - pool.cn[sel], k - 1)
        Jn = np.where(ok, Jn, np.inf)
        j = int(sel[int(np.argmin(Jn))])
        x[j] = False
        ws -= pool.wu[j]; y -= pool.pi[j]; cs -= pool.cn[j]; k -= 1
        pool.n_eval += len(sel)


def issc(pool, max_iter=50):
    """Iyer and Bilmes (2013), iterated submodular set cover with modular upper bounds."""
    if not pool.pool_feasible():
        return None
    x = None
    best = None
    best_J = math.inf
    for _ in range(max_iter):
        if x is None:
            cost = pool.J_from(pool.pi, pool.cn, 1.0) - pool.J_from(0.0, 0.0, 0.0)   # J(j | empty)
        else:
            _, y, cs, k = pool.sums(x)
            Jx = pool.J_from(y, cs, k)
            # for j in X: J(j | X \ j); for j outside X: J(j | empty)
            inner = Jx - pool.J_from(y - pool.pi, cs - pool.cn, k - 1)
            outer = pool.J_from(pool.pi, pool.cn, 1.0) - pool.J_from(0.0, 0.0, 0.0)
            cost = np.where(x, inner, outer)
        x_new = _greedy(pool, truncated=True, cost=np.maximum(cost, 0.0))
        J_new = pool.J(x_new)
        if J_new < best_J - 1e-15:
            best, best_J, x = x_new, J_new, x_new
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Metaheuristics with a wall-clock budget
# ---------------------------------------------------------------------------

def start_solution(pool):
    return reverse_delete(pool, greedy_truncated(pool))


def simulated_annealing(pool, budget_s, seed=0, trace=None):
    """Single flips and swaps; infeasible moves are rejected. Geometric cooling over the budget."""
    rng = np.random.default_rng(seed)
    t_end = time.perf_counter() + budget_s
    x = start_solution(pool)
    if x is None:
        return None
    ws, y, cs, k = pool.sums(x)
    J = float(pool.J_from(y, cs, k))
    best, best_J = x.copy(), J
    T0, T1 = max(0.05 * J, 1e-6), max(1e-5 * J, 1e-9)
    t_start = time.perf_counter()
    it = 0
    while True:
        if it % 256 == 0:
            now = time.perf_counter()
            if now >= t_end:
                break
            frac = (now - t_start) / budget_s
            T = T0 * (T1 / T0) ** frac
        it += 1
        if rng.random() < 0.5:                       # flip
            j = int(rng.integers(pool.m))
            s = -1 if x[j] else 1
            ws2, y2, cs2, k2 = ws + s * pool.wu[j], y + s * pool.pi[j], cs + s * pool.cn[j], k + s
            if s < 0 and not pool.feasible_from(ws2):
                continue
            moves = ((j, not x[j]),)
        else:                                         # swap one selected for one unselected
            sel = np.flatnonzero(x)
            uns = np.flatnonzero(~x)
            if len(sel) == 0 or len(uns) == 0:
                continue
            i = int(sel[rng.integers(len(sel))]); j = int(uns[rng.integers(len(uns))])
            ws2 = ws - pool.wu[i] + pool.wu[j]
            if not pool.feasible_from(ws2):
                continue
            y2, cs2, k2 = y - pool.pi[i] + pool.pi[j], cs - pool.cn[i] + pool.cn[j], k
            moves = ((i, False), (j, True))
        J2 = float(pool.J_from(y2, cs2, k2))
        pool.n_eval += 1
        if J2 <= J or rng.random() < math.exp(-(J2 - J) / T):
            for a, v in moves:
                x[a] = v
            ws, y, cs, k, J = ws2, y2, cs2, k2, J2
            if J < best_J - 1e-15:
                best, best_J = x.copy(), J
                if trace is not None:
                    trace.append((pool.n_eval, best_J))
    return best


def tabu_search(pool, budget_s, seed=0, tenure=None, trace=None):
    """Best admissible flip or swap per iteration, with a tabu list on changed items."""
    rng = np.random.default_rng(seed)
    t_end = time.perf_counter() + budget_s
    x = start_solution(pool)
    if x is None:
        return None
    m = pool.m
    tenure = tenure or int(min(20, max(7, m // 10)))
    tabu_until = np.zeros(m, int)
    ws, y, cs, k = pool.sums(x)
    J = float(pool.J_from(y, cs, k))
    best, best_J = x.copy(), J
    it = 0
    while time.perf_counter() < t_end:
        it += 1
        # single flips
        s = np.where(x, -1.0, 1.0)
        ws_f, y_f, cs_f, k_f = ws + s * pool.wu, y + s * pool.pi, cs + s * pool.cn, k + s
        J_f = pool.J_from(y_f, cs_f, k_f)
        ok_f = pool.feasible_from(ws_f)
        free_f = (tabu_until < it) | (J_f < best_J - 1e-15)
        J_f = np.where(ok_f & free_f, J_f, np.inf)
        # swaps
        sel, uns = np.flatnonzero(x), np.flatnonzero(~x)
        best_swap = (np.inf, -1, -1)
        if len(sel) and len(uns):
            ws_s = ws - pool.wu[sel][:, None] + pool.wu[uns][None, :]
            J_s = pool.J_from(y - pool.pi[sel][:, None] + pool.pi[uns][None, :],
                              cs - pool.cn[sel][:, None] + pool.cn[uns][None, :], k)
            free_s = ((tabu_until[sel] < it)[:, None] & (tabu_until[uns] < it)[None, :]) | (J_s < best_J - 1e-15)
            J_s = np.where(pool.feasible_from(ws_s) & free_s, J_s, np.inf)
            a = int(np.argmin(J_s))
            best_swap = (float(J_s.flat[a]), int(sel[a // len(uns)]), int(uns[a % len(uns)]))
            pool.n_eval += J_s.size
        pool.n_eval += m
        jf = int(np.argmin(J_f))
        if min(J_f[jf], best_swap[0]) == np.inf:
            # every move tabu or infeasible: random restart from the best solution
            x = best.copy(); tabu_until[:] = 0
            ws, y, cs, k = pool.sums(x); J = best_J
            j = int(rng.integers(m)); tabu_until[j] = it + tenure
            continue
        if J_f[jf] <= best_swap[0]:
            sgn = -1 if x[jf] else 1
            x[jf] = not x[jf]; tabu_until[jf] = it + tenure
            ws += sgn * pool.wu[jf]; y += sgn * pool.pi[jf]; cs += sgn * pool.cn[jf]; k += sgn
        else:
            _, i, j = best_swap
            x[i], x[j] = False, True
            tabu_until[i] = tabu_until[j] = it + tenure
            ws += pool.wu[j] - pool.wu[i]; y += pool.pi[j] - pool.pi[i]; cs += pool.cn[j] - pool.cn[i]
        J = float(pool.J_from(y, cs, k))
        if J < best_J - 1e-15:
            best, best_J = x.copy(), J
            if trace is not None:
                trace.append((pool.n_eval, best_J))
    return best


def repair(pool, X):
    """Vectorized repair for a population: complete with items in decreasing static
    utility-per-cost order, then drop items in decreasing cost-per-utility order while
    the subset stays sufficient."""
    X = np.asarray(X, bool).copy()
    unit = pool.J_from(pool.pi, pool.cn, 1.0) - pool.J_from(0.0, 0.0, 0.0)
    add_order = np.argsort(-(pool.wu / (unit + EPS)), kind="stable")
    drop_order = add_order[::-1]
    for r in range(X.shape[0]):
        x = X[r]
        ws = pool.wu[x].sum()
        if not pool.feasible_from(ws):
            for j in add_order:
                if not x[j]:
                    x[j] = True; ws += pool.wu[j]
                    if pool.feasible_from(ws):
                        break
        for j in drop_order:
            if x[j] and pool.feasible_from(ws - pool.wu[j]):
                x[j] = False; ws -= pool.wu[j]
        X[r] = x
    return X


def ga_seeded_repair(pool, budget_s, seed=0, pop_size=100, mut_scale=1.0, crossover="two_point",
                     seeded=True, use_repair=True, trace=None):
    """pymoo genetic algorithm with a wall-clock budget, a greedy seed and a repair operator."""
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.crossover.ux import UniformCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.optimize import minimize
    from pymoo.termination.max_time import TimeBasedTermination

    if not pool.pool_feasible():
        return None
    x0 = start_solution(pool)

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=pool.m, n_obj=1, n_ieq_constr=1, xl=0, xu=1, vtype=bool)

        def _evaluate(self, X, out, *a, **kw):
            Xf = X.astype(float)
            ws = Xf @ pool.wu
            out["F"] = pool.J_from(Xf @ pool.pi, Xf @ pool.cn, Xf.sum(axis=1))[:, None]
            out["G"] = (pool.W * (1 - TOL) - ws)[:, None]
            pool.n_eval += X.shape[0]

    class Rep(Repair):
        def _do(self, problem, X, **kw):
            return repair(pool, X)

    rng = np.random.default_rng(seed)
    init = rng.random((pop_size, pool.m)) < 0.5
    if seeded:
        init[0] = x0
    class Trace(Callback):
        def notify(self, algorithm):
            if trace is not None:
                f = algorithm.opt.get("F")
                if f is not None and len(f):
                    trace.append((pool.n_eval, float(f.min())))

    xo = TwoPointCrossover() if crossover == "two_point" else UniformCrossover()
    algo = GA(pop_size=pop_size, sampling=init, crossover=xo,
              mutation=BitflipMutation(prob=1.0, prob_var=min(1.0, mut_scale / max(pool.m, 1))),
              repair=Rep() if use_repair else None, eliminate_duplicates=True)
    res = minimize(Prob(), algo, TimeBasedTermination(budget_s), seed=seed, verbose=False,
                   callback=Trace())
    if res.X is None:
        return None
    x = np.asarray(res.X, bool).ravel()
    return x if pool.feasible(x) else None


METHODS = {
    "greedy": lambda pool: greedy(pool),
    "greedy_rd": lambda pool: reverse_delete(pool, greedy(pool)),
    "greedy_truncated": lambda pool: greedy_truncated(pool),
    "greedy_truncated_rd": lambda pool: reverse_delete(pool, greedy_truncated(pool)),
    "issc": lambda pool: issc(pool),
    "issc_rd": lambda pool: reverse_delete(pool, issc(pool)),
}
