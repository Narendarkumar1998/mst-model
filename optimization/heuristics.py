"""
Constructive selection rules for MST, used as warm starts and baselines in the article.

All methods work on the admissible pool D_t only and use the paper's quantities:
U(S) = 1 - prod(1 - u_i), P(S) = 1 - prod(1 - p_i), C(S) = sum c_i / C_ref,
K(S) = |S| / K_ref (reference scales, optimization/scales.py), J = lambda_P P + lambda_C C + lambda_K K.

Sufficiency is checked through the exact linear form sum_i w_i x_i >= W with
w_i = -ln(1 - u_i) and W = -ln(1 - tau) (Eq. 10 of the paper), up to a relative
tolerance of 1e-12; the returned subsets are re-checked with the product form.

Methods (each returns a sorted tuple of admissible positions, or None when the
admissible pool cannot reach the threshold):
  greedy            greedy on the untruncated marginal utility
  greedy_truncated  standard submodular-cover greedy: utility gain truncated at tau
  reverse_delete    removes items while the subset stays sufficient, largest J decrease first
  issc              Iyer and Bilmes (2013) iterated submodular set cover: repeatedly
                    replaces J by its modular upper bound at the current solution and
                    solves the resulting submodular cover with the truncated greedy
"""

from __future__ import annotations

import math

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


METHODS = {
    "greedy": lambda pool: greedy(pool),
    "greedy_rd": lambda pool: reverse_delete(pool, greedy(pool)),
    "greedy_truncated": lambda pool: greedy_truncated(pool),
    "greedy_truncated_rd": lambda pool: reverse_delete(pool, greedy_truncated(pool)),
    "issc": lambda pool: issc(pool),
    "issc_rd": lambda pool: reverse_delete(pool, issc(pool)),
}
