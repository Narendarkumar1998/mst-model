"""
Fixed reference scales against normalization by pool totals (Section 4.1.5 of the paper).

For every pool of the synthetic benchmark (sizes 8 to 16, 100 replicates, three regimes;
all items admissible) and every threshold and weight profile, the exact MST optimum is
computed twice by enumeration: with the fixed references C_ref and K_ref of the model,
and with C and K normalized by the totals of the pool (sum of all costs, number of items).
Writes data/benchmark/normalization.csv (one row per condition).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison"))
sys.path.insert(0, str(ROOT / "optimization"))
import benchmark_pools as bp  # noqa: E402
import methods as m  # noqa: E402

OUT = ROOT / "data" / "benchmark" / "normalization.csv"


def main():
    rows = []
    for si, n in enumerate(bp.EXACT_BENCHMARK_SIZES):
        for rep in range(bp.REPLICATES):
            seed = bp.matched_seed(si, rep)
            for regime in bp.INSTANCE_REGIMES:
                pool = bp.make_pool(n, regime, seed, share=0.0)
                inst = m.Instance(pool.ids, pool.utility, pool.privacy, pool.cost, pool.admissible)
                Cpool = inst.Cn * m.REF_COST / pool.cost.sum()
                Kpool = inst.card / n
                for tau in bp.TAU_LEVELS:
                    for wp, w in bp.WEIGHT_PROFILES.items():
                        a = m.mst_exact(inst, tau, w)
                        if a == m.INFEASIBLE:
                            continue
                        Jp = w[0] * inst.P + w[1] * Cpool + w[2] * Kpool
                        cand = inst.all_adm & (inst.U >= tau)
                        b = inst.lexmin(cand, (Jp, inst.P, Cpool, inst.card))
                        rows.append(dict(n_items=n, replicate=rep, regime=regime, tau=tau, weight_profile=wp,
                                         changed=int(a != b), items_reference=int(inst.card[a]),
                                         items_pool=int(inst.card[b])))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(d.groupby("n_items")[["changed", "items_reference", "items_pool"]].mean())
    print("overall changed:", d.changed.mean())


if __name__ == "__main__":
    main()
