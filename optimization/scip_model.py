"""
MST as a mixed-integer nonlinear program solved with SCIP.

Following the linearization suggested by M. Dridi:

  * decision variables x_i in {0, 1} for admissible items only (hard admissibility
    is enforced by the domain);
  * the sufficiency constraint 1 - prod(1 - u_i x_i) >= tau is written as the exact
    linear covering constraint  sum_i w_i x_i >= W_tau,  with w_i = -ln(1 - u_i)
    and W_tau = -ln(1 - tau). Weights are capped at W_tau (a standard covering-
    knapsack reduction that leaves the feasible set unchanged), which also handles
    u_i = 1 (w_i = W_tau). For tau = 1 the constraint becomes "select at least one
    item with u_i = 1";
  * the privacy term 1 - prod(1 - p_i x_i) is written as 1 - exp(-y) with
    y = sum_i pi_i x_i and pi_i = -ln(1 - p_i). Items with p_i = 1 force the
    privacy term to 1 through t >= x_i. The term is concave in y, so SCIP solves
    it to global optimality by spatial branch and bound;
  * the cost and cardinality terms are linear.

Interpretation cost is normalized by the complete candidate pool I_t and
cardinality by |I_t|, as in the paper.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from scales import REF_COST, REF_ITEMS
from pyscipopt import Model, exp, quicksum


@dataclass
class SolverResult:
    status: str               # OPTIMAL, INFEASIBLE, TIME_LIMIT
    selected: tuple           # indices into the complete pool
    objective: float          # J of the incumbent (nan if none)
    dual_bound: float         # proven lower bound on J*
    gap: float                # (incumbent - bound) / bound, 0 when proven optimal
    runtime: float
    nodes: int


def solve_mst_scip(u, p, c, admissible, tau, weights, time_limit=60.0, gap_limit=0.0,
                   threads=1, seed=0, start=None, privacy="exp",
                   cost_scale=REF_COST, card_scale=REF_ITEMS, utility="product",
                   robust_gamma=None, deviation=None) -> SolverResult:
    """start: optional iterable of item indices used as an initial feasible solution.
    privacy: "exp" writes P = 1 - exp(-y) (mixed-integer nonlinear program);
             "chain" linearizes the product exactly, q_k = q_{k-1} (1 - p_k x_k), with
             McCormick envelopes, which are exact for a binary times a bounded variable,
             so the whole model is a mixed-integer linear program.
    cost_scale, card_scale: reference scales C_ref and K_ref of C and K (optimization/scales.py);
             None normalizes by the admissible pool instead.
    utility: "product" (U = 1 - prod(1 - u_i)) or "additive" (U = sum u_i, for utilities that
             are additive shares by construction, as the attribution shares of the case study).
    robust_gamma, deviation: Gamma-robust sufficiency (Bertsimas and Sim 2004). Each utility may
             be overestimated by up to deviation[i]; the disclosure must stay sufficient when any
             robust_gamma of the selected utilities take their lowest value u_i - deviation[i]
             (robust_gamma >= number of items gives box robustness). The deviation is mapped to
             the covering weights (w_i = -ln(1 - u_i) for the product form, w_i = u_i for the
             additive form) and protected with the linear dual reformulation
             sum w_i x_i - Gamma z - sum r_i >= W, z + r_i >= d_i x_i, z, r_i >= 0."""
    lp, lc, lk = weights
    n_all = len(u)
    adm = [i for i in range(n_all) if admissible[i]]
    total_cost = sum(c[i] for i in adm)      # normalization by the admissible pool D_t
    t0 = time.perf_counter()

    # Quick infeasibility test on the admissible pool (nominal values).
    if utility == "additive":
        if sum(u[i] for i in adm) < tau - 1e-12:
            return SolverResult("INFEASIBLE", (), math.nan, math.nan, 0.0, time.perf_counter() - t0, 0)
    else:
        prod = 1.0
        for i in adm:
            prod *= (1.0 - u[i])
        if 1.0 - prod < tau:
            return SolverResult("INFEASIBLE", (), math.nan, math.nan, 0.0, time.perf_counter() - t0, 0)

    m = Model("MST")
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", gap_limit)
    m.setParam("numerics/feastol", 1e-9)
    m.setParam("randomization/randomseedshift", seed)
    m.setParam("parallel/maxnthreads", threads)

    x = {i: m.addVar(vtype="B", name=f"x{i}") for i in adm}

    # Sufficiency: exact linear covering constraint.
    if robust_gamma is not None:
        def cw(v):
            if utility == "additive":
                return v
            return 50.0 if v >= 1.0 else -math.log1p(-min(max(v, 0.0), 1.0 - 1e-15))
        W = tau if utility == "additive" else (50.0 if tau >= 1.0 else -math.log1p(-tau))
        w = {i: cw(u[i]) for i in adm}
        d = {i: max(0.0, w[i] - cw(max(0.0, u[i] - deviation[i]))) for i in adm}
        zr = m.addVar(lb=0.0, name="zr")
        r = {i: m.addVar(lb=0.0, name=f"r{i}") for i in adm}
        for i in adm:
            m.addCons(zr + r[i] >= d[i] * x[i])
        m.addCons(quicksum(w[i] * x[i] for i in adm) - robust_gamma * zr - quicksum(r[i] for i in adm) >= W)
    elif utility == "additive":
        m.addCons(quicksum(u[i] * x[i] for i in adm) >= tau)
    elif tau >= 1.0:
        ones = [i for i in adm if u[i] >= 1.0]
        m.addCons(quicksum(x[i] for i in ones) >= 1)
    else:
        W = -math.log1p(-tau)
        w = {i: (W if u[i] >= 1.0 else min(-math.log1p(-u[i]), W)) for i in adm}
        m.addCons(quicksum(w[i] * x[i] for i in adm) >= W)

    t = m.addVar(lb=0.0, ub=1.0, name="P")
    if privacy == "chain":
        # q_0 = 1, q_k = q_{k-1} - p_k z_k with z_k = q_{k-1} x_k; L_k <= q_k <= 1.
        q_prev, L_prev = 1.0, 1.0
        chain = []
        for i in adm:
            L_k = L_prev * (1.0 - p[i])
            q = m.addVar(lb=L_k, ub=1.0, name=f"q{i}")
            z = m.addVar(lb=0.0, ub=1.0, name=f"z{i}")
            m.addCons(z <= x[i])                               # z <= U x  (U = 1)
            m.addCons(z >= L_prev * x[i])                      # z >= L x
            m.addCons(z <= q_prev - L_prev * (1 - x[i]))       # z <= q - L (1 - x)
            m.addCons(z >= q_prev - (1 - x[i]))                # z >= q - U (1 - x)
            m.addCons(q == q_prev - p[i] * z)
            chain.append((i, q, z))
            q_prev, L_prev = q, L_k
        m.addCons(t == 1 - q_prev) if adm else m.addCons(t == 0)
    else:
        # Privacy term P = 1 - exp(-y), with items of p_i = 1 handled separately.
        soft = [i for i in adm if p[i] < 1.0]
        y = m.addVar(lb=0.0, name="y")
        m.addCons(y == quicksum(-math.log1p(-p[i]) * x[i] for i in soft))
        m.addCons(t >= 1 - exp(-y))
        for i in adm:
            if p[i] >= 1.0:
                m.addCons(t >= x[i])

    cscale = total_cost if cost_scale is None else cost_scale
    kscale = len(adm) if card_scale is None else card_scale
    cost_term = quicksum((c[i] / cscale) * x[i] for i in adm) if cscale > 0 else 0
    card_term = quicksum(x[i] for i in adm) * (1.0 / kscale)
    m.setObjective(lp * t + lc * cost_term + lk * card_term, "minimize")
    if start is not None:
        st = set(start)
        sol0 = m.createSol()
        for i in adm:
            m.setSolVal(sol0, x[i], 1.0 if i in st else 0.0)
        if privacy == "chain":
            qv = 1.0
            for i, q, z in chain:
                zv = qv if i in st else 0.0
                qv = qv - p[i] * zv
                m.setSolVal(sol0, z, zv)
                m.setSolVal(sol0, q, qv)
            m.setSolVal(sol0, t, 1.0 - qv)
        else:
            yv = sum(-math.log1p(-p[i]) for i in st if p[i] < 1.0)
            m.setSolVal(sol0, y, yv)
            m.setSolVal(sol0, t, 1.0 if any(p[i] >= 1.0 for i in st) else 1.0 - math.exp(-yv))
        m.addSol(sol0, free=True)
    m.optimize()

    status = m.getStatus()
    runtime = time.perf_counter() - t0
    nodes = int(m.getNNodes())
    if m.getNSols() == 0:
        st = "INFEASIBLE" if status == "infeasible" else "TIME_LIMIT"
        return SolverResult(st, (), math.nan, m.getDualbound(), math.nan, runtime, nodes)
    sol = m.getBestSol()
    sel = tuple(i for i in adm if m.getSolVal(sol, x[i]) > 0.5)
    obj = exact_objective(sel, u, p, c, weights, admissible, cost_scale, card_scale)
    bound = m.getDualbound()
    gap = 0.0 if status == "optimal" else max(0.0, (obj - bound) / bound) if bound > 0 else math.nan
    return SolverResult("OPTIMAL" if status == "optimal" else "TIME_LIMIT", sel, obj, bound, gap, runtime, nodes)


def exact_objective(sel, u, p, c, weights, admissible=None, cost_scale=REF_COST, card_scale=REF_ITEMS):
    """J recomputed from the selected indices with the paper's product forms;
    C and K are normalized by the admissible pool (all items when admissible is None)."""
    lp, lc, lk = weights
    prod_p = 1.0
    for i in sel:
        prod_p *= (1.0 - p[i])
    adm = [True] * len(u) if admissible is None else list(admissible)
    total = 0
    for v, a in zip(c, adm):
        if a:
            total += v
    cs = 0
    for i in sel:
        cs += c[i]
    total = total if cost_scale is None else cost_scale
    cn = cs / total if total != 0 else 0.0
    kscale = sum(adm) if card_scale is None else card_scale
    return lp * (1.0 - prod_p) + lc * cn + lk * (len(sel) / kscale)


def utility(sel, u):
    prod = 1.0
    for i in sel:
        prod *= (1.0 - u[i])
    return 1.0 - prod
