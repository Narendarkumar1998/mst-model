"""
Run MST, its ablations and the adapted baselines on the primary-benchmark pools
(with hard-admissibility flags) and write one row per method and condition.

Output: data/comparison/comparison_rows.csv.gz  (24,000 conditions x 14 methods)

Usage: python comparison/run_comparison.py [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool as ProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))

import benchmark_pools as bp  # noqa: E402
import methods as m  # noqa: E402

OUT = ROOT / "data" / "comparison" / "comparison_rows.csv.gz"


def run_instance(args):
    size_index, n, replicate, regime = args
    seed = bp.matched_seed(size_index, replicate)
    pool = bp.make_pool(n, regime, seed)
    inst = m.Instance(pool.ids, pool.utility, pool.privacy, pool.cost, pool.admissible)
    instance_id = f"V2_M{n}_R{replicate:03d}_{regime}"
    rows = []
    for profile, w in bp.WEIGHT_PROFILES.items():
        J = inst.J(w)
        for tau in bp.TAU_LEVELS:
            feasible = bool(inst.U[inst.adm_mask] >= tau)
            mst_sel = m.mst_exact(inst, tau, w, J)
            diag = m.infeasibility_diagnostic(inst, tau) if not feasible else {}
            for name, (fn, group, _label) in m.METHODS.items():
                sel = fn(inst, tau, w, J)
                row = {
                    "instance_id": instance_id, "matched_seed": seed, "n_items": n,
                    "n_prohibited": int((~pool.admissible).sum()), "regime": regime,
                    "replicate": replicate, "weight_profile": profile, "tau": tau,
                    "instance_feasible": feasible, "method": name, "group": group,
                }
                if sel == m.INFEASIBLE:
                    row.update(status="INFEASIBLE", selected_items="", selected_count=0,
                               utility=np.nan, privacy=np.nan, norm_cost=np.nan,
                               norm_cardinality=np.nan, objective=np.nan,
                               sufficient=False, admissible=True, prohibited_disclosed=0)
                    valid = not feasible
                else:
                    sufficient = bool(inst.U[sel] >= tau)
                    admissible = bool(inst.all_adm[sel])
                    row.update(status="DISCLOSED", selected_items="|".join(inst.id_tuple(sel)),
                               selected_count=int(inst.card[sel]), utility=float(inst.U[sel]),
                               privacy=float(inst.P[sel]), norm_cost=float(inst.Cn[sel]),
                               norm_cardinality=float(inst.K[sel]), objective=float(J[sel]),
                               sufficient=sufficient, admissible=admissible,
                               prohibited_disclosed=int(inst.n_prohibited[sel]))
                    valid = feasible and sufficient and admissible
                row["valid"] = valid
                if name == "MST" and diag:
                    row.update({f"diag_{k}": v for k, v in diag.items()})
                # comparison with MST on feasible conditions
                if feasible and valid and sel != m.INFEASIBLE:
                    jm = float(J[mst_sel])
                    row["objective_excess_pct"] = 100.0 * (float(J[sel]) - jm) / jm if jm > 0 else 0.0
                    row["weakly_dominated_by_mst"] = bool(
                        inst.P[mst_sel] <= inst.P[sel] and inst.Cn[mst_sel] <= inst.Cn[sel]
                        and inst.K[mst_sel] <= inst.K[sel])
                    row["same_as_mst"] = bool(sel == mst_sel)
                else:
                    row["objective_excess_pct"] = np.nan
                    row["weakly_dominated_by_mst"] = np.nan
                    row["same_as_mst"] = np.nan
                # repair cost on feasible conditions
                if feasible and sel != m.INFEASIBLE:
                    rep = m.repair(inst, sel, tau, J)
                    jm = float(J[mst_sel])
                    row["repaired_objective"] = float(J[rep])
                    row["repair_excess_pct"] = 100.0 * (float(J[rep]) - jm) / jm if jm > 0 else 0.0
                    row["repair_changed"] = bool(rep != sel)
                else:
                    row["repaired_objective"] = np.nan
                    row["repair_excess_pct"] = np.nan
                    row["repair_changed"] = np.nan
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replicates", type=int, default=bp.REPLICATES)
    args = ap.parse_args()
    jobs = [(si, n, r, regime)
            for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES)
            for r in range(args.replicates)
            for regime in bp.INSTANCE_REGIMES]
    with ProcessPool(args.workers) as pool:
        results = pool.map(run_instance, jobs, chunksize=10)
    df = pd.DataFrame([row for rows in results for row in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"rows: {len(df)}  conditions: {len(df) // len(m.METHODS)}  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
