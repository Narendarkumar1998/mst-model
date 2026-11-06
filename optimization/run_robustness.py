"""
Robustness of MST to misestimated item parameters.

The generated u, p, c are taken as the true values. The assistant agent only sees
estimates u(1 + e_u), p(1 + e_p), c(1 + e_c) with independent e ~ Uniform(-delta, delta)
per item and parameter (u and p clipped to [0, 0.999]). The disclosure is selected on
the estimates and evaluated on the true values.

Selection rules: exact MST, Dynamic Greedy-MST, truncated greedy with reverse deletion,
robust MST, which requires sufficiency with every estimated utility deflated by
(1 - delta) (box robustness), and Gamma-robust MST (Bertsimas and Sim 2004), which protects
against any Gamma of the selected utilities being overestimated by that factor (Gamma = 1, 2, 3). Since u_i (1 + e)(1 - delta) <= u_i (1 - delta^2) <= u_i and U is
non-decreasing in every u_i, a disclosure that is sufficient for the deflated estimates
is sufficient for the true utilities whenever |e_u| <= delta.

Pools: primary benchmark sizes 8 to 16, replicates 0-9, three regimes (150 pools), all
items admissible; four thresholds and four weight profiles; delta in {0.10, 0.20}; 20
noise draws per pool and delta.

Writes data/benchmark/robustness.csv
Usage: python optimization/run_robustness.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))
import benchmark_pools as bp  # noqa: E402
import methods as m  # noqa: E402

DELTAS = [0.10, 0.20]
DRAWS = 20
REPLICATES = 10
OUT = ROOT / "data" / "benchmark" / "robustness.csv"


GAMMAS = (1, 2, 3)


def greedy_trunc_rd(inst, tau, w, J):
    return m.baseline_policy_greedy_delete(inst, tau, w, J)


def gamma_robust_mask(est, delta, gammas):
    """Gamma-robust sufficiency (Bertsimas and Sim 2004) on the estimates: for every subset,
    the covering weight sum_i w_i x_i minus the largest Gamma deviations d_i of its items,
    with w_i = -ln(1 - u_i) and d_i = w_i + ln(1 - (1 - delta) u_i), i.e. up to Gamma selected
    utilities overestimated by the factor 1/(1 - delta). Returns {Gamma: robust utility}."""
    u = est.u
    w = -np.log1p(-u)
    d = w + np.log1p(-(1 - delta) * u)
    ws = est.B @ w
    out = {}
    order = np.argsort(-d, kind="stable")
    for g in gammas:
        cnt = np.zeros(len(est.B), int)
        dev = np.zeros(len(est.B))
        for i in order:
            col = est.B[:, i]
            take = col & (cnt < g)
            dev = np.where(take, dev + d[i], dev)
            cnt = cnt + col
        out[g] = 1.0 - np.exp(-(ws - dev))
    return out


def run(job):
    si, n, r, regime = job
    seed = bp.matched_seed(si, r)
    ids, u, p, c = bp.generate_items(n, regime, seed)
    adm = np.ones(n, bool)
    true = m.Instance(ids, u, p, c, adm)
    rows = []
    rng = np.random.default_rng(seed + 991)
    for delta in DELTAS:
        for d in range(DRAWS):
            eu, ep, ec = (rng.uniform(-delta, delta, n) for _ in range(3))
            ue = np.clip(u * (1 + eu), 0, 0.999)
            pe = np.clip(p * (1 + ep), 0, 0.999)
            ce = c * (1 + ec)
            est = m.Instance(ids, ue, pe, ce, adm)
            rob = m.Instance(ids, ue * (1 - delta), pe, ce, adm)
            Ug = gamma_robust_mask(est, delta, GAMMAS)
            for prof, w in bp.WEIGHT_PROFILES.items():
                Jt, Je, Jr = true.J(w), est.J(w), rob.J(w)
                for tau in bp.TAU_LEVELS:
                    opt_true = m.mst_exact(true, tau, w, Jt)
                    if opt_true == m.INFEASIBLE:
                        continue
                    jt = float(Jt[opt_true])
                    row = dict(n_items=n, regime=regime, replicate=r, delta=delta, draw=d,
                               weight_profile=prof, tau=tau)
                    sels = {"exact": m.mst_exact(est, tau, w, Je),
                            "greedy": m.ablation_greedy(est, tau, w, Je),
                            "greedy_truncated_rd": greedy_trunc_rd(est, tau, w, Je),
                            "robust": m.mst_exact(rob, tau, w, Jr)}
                    for g in GAMMAS:
                        sels[f"robust_g{g}"] = est.lexmin(est.all_adm & (Ug[g] >= tau),
                                                          (Je, est.P, est.Cn, est.card))
                    for k, s in sels.items():
                        if s == m.INFEASIBLE:
                            row.update({f"{k}_infeasible": True, f"{k}_changed": np.nan,
                                        f"{k}_insufficient": np.nan, f"{k}_regret_pct": np.nan})
                            continue
                        row.update({f"{k}_infeasible": False,
                                    f"{k}_changed": bool(s != opt_true),
                                    f"{k}_insufficient": bool(true.U[s] < tau),
                                    f"{k}_regret_pct": 100 * (float(Jt[s]) - jt) / jt})
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES) for r in range(REPLICATES)
            for g in bp.INSTANCE_REGIMES]
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs, chunksize=2)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    print(len(df))
    for k in ("exact", "greedy", "greedy_truncated_rd", "robust", "robust_g1", "robust_g2", "robust_g3"):
        print(k, df.groupby("delta")[[f"{k}_changed", f"{k}_insufficient", f"{k}_infeasible"]].mean().round(4).to_dict(),
              df.groupby("delta")[f"{k}_regret_pct"].median().round(3).to_dict())


if __name__ == "__main__":
    main()
