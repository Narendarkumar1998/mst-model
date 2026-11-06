"""
Sensitivity of the four methods that have a free parameter.

  soft-penalty weight per prohibited item (ablation A1): 0.1, 1, 10
  privacy budget of goal obfuscation (C2):               0.25, 0.35, 0.50
  parsimony length k of parsimonious HAExA (C4):        2, 3, 4
  coverage budget (baseline B6):                        0.15, 0.25, 0.35

Writes tables/comparison_parameter_sensitivity.csv
Usage: python comparison/sensitivity.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool as ProcessPool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))
import benchmark_pools as bp  # noqa: E402
import methods as m  # noqa: E402

SETTINGS = [("A1_soft_admissibility", "SOFT_PENALTY_PER_PROHIBITED_ITEM", v) for v in (0.1, 1.0, 10.0)] + \
           [("C2_goal_obfuscation", "PRIVACY_BUDGET", v) for v in (0.25, 0.35, 0.50)] + \
           [("C4_parsimonious_haexa", "PARSIMONY_K", v) for v in (2, 3, 4)] + \
           [("B6_budgeted_coverage", "COVERAGE_BUDGET", v) for v in (0.15, 0.25, 0.35)]


def run(args):
    size_index, n, replicate, regime = args
    seed = bp.matched_seed(size_index, replicate)
    pool = bp.make_pool(n, regime, seed)
    inst = m.Instance(pool.ids, pool.utility, pool.privacy, pool.cost, pool.admissible)
    out = []
    for method, param, value in SETTINGS:
        setattr(m, param, value)
        fn = m.METHODS[method][0]
        for w in bp.WEIGHT_PROFILES.values():
            J = inst.J(w)
            for tau in bp.TAU_LEVELS:
                feasible = bool(inst.U[inst.adm_mask] >= tau)
                sel = fn(inst, tau, w, J)
                if sel == m.INFEASIBLE:
                    suff, adm, valid = False, True, not feasible
                else:
                    suff, adm = bool(inst.U[sel] >= tau), bool(inst.all_adm[sel])
                    valid = feasible and suff and adm
                out.append((method, param, value, feasible, suff, adm, valid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    jobs = [(si, n, r, g) for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES)
            for r in range(bp.REPLICATES) for g in bp.INSTANCE_REGIMES]
    with ProcessPool(args.workers) as p:
        res = p.map(run, jobs, chunksize=10)
    d = pd.DataFrame([r for rows in res for r in rows],
                     columns=["method", "parameter", "value", "feasible", "sufficient", "admissible", "valid"])
    s = (d.groupby(["method", "parameter", "value"])
         .apply(lambda g: pd.Series({
             "R1_sufficient_when_possible_pct": 100 * g[g.feasible].sufficient.mean(),
             "R2_no_prohibited_item_pct": 100 * g.admissible.mean(),
             "R1_R2_R3_all_met_pct": 100 * g.valid.mean()}), include_groups=False)
         .round(2).reset_index())
    s.to_csv(ROOT / "data" / "benchmark" / "comparison_parameter_sensitivity.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
