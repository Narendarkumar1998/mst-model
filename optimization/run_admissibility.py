"""
Infeasibility across admissibility levels (referee minor comment 11).

Benchmark pools (sizes 8-16, replicates 0-24, three regimes) with a share rho in
{0, 0.25, 0.5, 0.75} of their items prohibited by the policy (chosen uniformly at random,
seeded), four thresholds and four weight profiles: 24,000 conditions. Per condition:
Exact-MST by enumeration over the admissible items, the infeasibility diagnostic
(largest admissible utility, shortfall, smallest release), and the constructive
heuristics restricted to the admissible items.

Writes data/benchmark/admissibility.csv.gz
Usage: python optimization/run_admissibility.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimization", "comparison"):
    sys.path.insert(0, str(ROOT / sub))
import benchmark_pools as bp  # noqa: E402
import heuristics as h  # noqa: E402
import methods as m  # noqa: E402

OUT = ROOT / "data" / "benchmark" / "admissibility.csv.gz"
RHO = [0.0, 0.25, 0.5, 0.75]
REPLICATES = 25


def run(job):
    si, n, r, regime = job
    seed = bp.matched_seed(si, r)
    ids, u, p, c = bp.generate_items(n, regime, seed)
    rows = []
    for rho in RHO:
        rng = np.random.default_rng(seed + int(1000 * rho))
        adm = np.ones(n, bool)
        adm[rng.choice(n, int(round(rho * n)), replace=False)] = False
        inst = m.Instance(ids, u, p, c, adm)
        for prof, w in bp.WEIGHT_PROFILES.items():
            J = inst.J(w)
            for tau in bp.TAU_LEVELS:
                s = m.mst_exact(inst, tau, w, J)
                row = dict(n_items=n, replicate=r, regime=regime, rho=rho, n_admissible=int(adm.sum()),
                           weight_profile=prof, tau=tau, exact_status="INFEASIBLE" if s == m.INFEASIBLE else "OPTIMAL")
                if s == m.INFEASIBLE:
                    row.update(m.infeasibility_diagnostic(inst, tau))
                else:
                    row.update(exact_objective=float(J[s]), exact_count=int(inst.card[s]))
                    for name in ("greedy", "greedy_rd", "greedy_truncated_rd"):
                        pool = h.Pool(u, p, c, list(adm), tau, w)
                        x = h.METHODS[name](pool)
                        row[f"{name}_objective"] = pool.J(x) if x is not None else np.nan
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES) for r in range(REPLICATES)
            for g in bp.INSTANCE_REGIMES]
    with Pool(args.workers) as pool:
        res = pool.map(run, jobs, chunksize=3)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False, compression="gzip")
    df["infeasible"] = df.exact_status == "INFEASIBLE"
    print(df.groupby(["rho", "tau"]).infeasible.mean().unstack().round(3))
    inf = df[df.infeasible]
    print(inf.groupby("rho").agg(n=("infeasible", "size"), releasable=("release_possible", "mean"),
                                 release_med=("release_needed", "median"), release_max=("release_needed", "max"),
                                 shortfall=("shortfall", "median")))
    f = df[~df.infeasible]
    for k in ("greedy", "greedy_rd", "greedy_truncated_rd"):
        g = 100 * (f[f"{k}_objective"] - f.exact_objective) / f.exact_objective
        print(k, (g <= 1e-7).groupby(f.rho).mean().round(3).to_dict(), g.groupby(f.rho).mean().round(2).to_dict())


if __name__ == "__main__":
    main()
