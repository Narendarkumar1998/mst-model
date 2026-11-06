"""
Tables of Paper 1, written as LaTeX to tables/.

  tab_comparison.tex     requirements met and objective excess of MST, ablations, and closest approaches
  tab_admissibility.tex  infeasibility and release diagnostic against the prohibited share
  tab_robustness.tex     nominal and robust selection under estimation noise (supplementary)
  tab_case_variants.tex  case-study rules at the nominal thresholds (supplementary)
  tab_case_items.tex     candidate items of the case study (supplementary)
  tab_sarl.tex           SARL execution: outcome per scenario, role, and threshold (supplementary)
  tab_sensitivity.tex    free parameters of the adapted rules (supplementary)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
CASE = ROOT / "data" / "case_study_lbnl"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)

COMP = [("MST", "MST (complete model)", "Model"),
        ("A1_soft_admissibility", "Penalty instead of hard admissibility", "Ablation"),
        ("A2_utility_reward", "Utility reward instead of threshold", "Ablation"),
        ("A3_min_cardinality", "Fewest items instead of disclosure cost", "Ablation"),
        ("A4_no_infeasibility", "No infeasibility reporting", "Ablation"),
        ("C1_model_reconciliation", "Model reconciliation", "Closest"),
        ("C2_goal_obfuscation", "Goal obfuscation and legibility", "Closest"),
        ("C3_role_aware_haexa", "Role-aware HAExA", "Closest"),
        ("C4_parsimonious_haexa", "Parsimonious HAExA", "Closest"),
        ("B1_full_disclosure", "Full disclosure", "Baseline")]


def f1(x):
    return "--" if pd.isna(x) else f"{x:.1f}"


def comparison():
    d = pd.read_csv(ROOT / "data" / "comparison" / "comparison_rows.csv.gz",
                    usecols=["method", "instance_feasible", "status", "sufficient", "prohibited_disclosed",
                             "objective_excess_pct", "valid", "same_as_mst"])
    lines = []
    prev = None
    for k, lab, grp in COMP:
        g = d[d.method == k]
        fe, inf = g[g.instance_feasible], g[~g.instance_feasible]
        r1 = 100 * ((fe.status == "DISCLOSED") & fe.sufficient.astype(bool)).mean()
        r2 = 100 * (g.prohibited_disclosed.fillna(0) == 0).mean()
        r3 = 100 * (inf.status == "INFEASIBLE").mean()
        okf = (fe.status == "DISCLOSED") & fe.sufficient.astype(bool) & (fe.prohibited_disclosed.fillna(0) == 0)
        allm = 100 * (okf.sum() + (inf.status == "INFEASIBLE").sum()) / len(g)
        e = fe.loc[fe.valid.astype(bool), "objective_excess_pct"].dropna()
        same = 100 * (e <= 1e-7).mean() if len(e) else np.nan
        med = e.median() if len(e) else np.nan
        mx = e.max() if len(e) else np.nan
        if prev is not None and grp != prev:
            lines.append("\\addlinespace")
        prev = grp
        lines.append(f"{lab} & {f1(r1)} & {f1(r2)} & {f1(r3)} & {f1(allm)} & {f1(same)} & {f1(med)} & "
                     f"{'--' if pd.isna(mx) else f'{mx:,.1f}'} \\\\")
    (OUT / "tab_comparison.tex").write_text("\n".join(lines) + "\n")


def admissibility():
    a = pd.read_csv(BENCH / "admissibility.csv.gz")
    lines = []
    for rho, g in a.groupby("rho"):
        inf = g[g.exact_status == "INFEASIBLE"]
        fe = g[g.exact_status != "INFEASIBLE"]
        rel = inf[inf.release_possible.astype(bool)] if len(inf) else inf
        relp = 100 * inf.release_possible.astype(bool).mean() if len(inf) else np.nan
        rn = f"{int(rel.release_needed.median())} ({int(rel.release_needed.max())})" if len(rel) else "-- (--)"
        lines.append(f"{int(round(100 * rho))}\\,\\% & {g.n_admissible.mean():.1f} & {100 * len(inf) / len(g):.1f} & "
                     f"{f1(relp)} & {rn} & {inf.shortfall.median():.3f} & {fe.exact_count.mean():.2f} & "
                     f"{fe.exact_objective.mean():.3f} \\\\")
    (OUT / "tab_admissibility.tex").write_text("\n".join(lines) + "\n")


def robustness():
    r = pd.read_csv(BENCH / "robustness.csv", low_memory=False)
    meths = [("exact", "Nominal MST"), ("robust_g1", "Robust MST, $\\Gamma = 1$"),
             ("robust_g2", "Robust MST, $\\Gamma = 2$"), ("robust_g3", "Robust MST, $\\Gamma = 3$"),
             ("robust", "Robust MST, box")]
    lines = []
    for k, lab in meths:
        for dv in (0.1, 0.2):
            g = r[r.delta == dv]
            inf = 100 * g[f"{k}_infeasible"].astype(bool).mean()
            h = g[~g[f"{k}_infeasible"].astype(bool)]
            ch = 100 * h[f"{k}_changed"].astype(float).mean()
            ins = 100 * h[f"{k}_insufficient"].astype(float).mean()
            ex = h[f"{k}_regret_pct"].astype(float)
            lines.append(f"{lab} & $\\pm${int(dv * 100)}\\,\\% & {ch:.1f} & {ins:.2f} & {inf:.2f} & "
                         f"{ex.mean():.1f} & {ex.quantile(0.95):.1f} \\\\")
    (OUT / "tab_robustness.tex").write_text("\n".join(lines) + "\n")


def case_variants():
    nd = pd.read_csv(CASE / "nominal.csv")
    rules = [("MST", "MST"), ("MST_product_utility", "MST, product utility"),
             ("MST_uniform_cost", "MST, uniform cost"), ("robust_g1", "Robust, $\\Gamma = 1$"),
             ("robust_g2", "Robust, $\\Gamma = 2$"), ("robust_gbox", "Robust, box"),
             ("ratio", "Utility per cost"), ("fewest", "Fewest items"), ("template", "Role template ($k = 3$)")]
    roles = [("facility_manager", "FM"), ("energy_auditor", "EA"), ("occupant", "OC")]
    lines = []
    for i, (k, lab) in enumerate(rules):
        if i:
            lines.append("\\addlinespace")
        for rk, rl in roles:
            g = nd[(nd.variant == k) & (nd.role == rk)]
            h = g[g.returned.astype(bool)]
            lines.append(f"{lab} & {rl} & {100 * g.returned.mean():.1f} & {100 * h.sufficient.astype(float).mean():.1f} & "
                         f"{100 * h.suff_prob.mean():.1f} & {h.n_selected.mean():.2f} & {h.privacy.mean():.3f} & "
                         f"{h.joint_privacy.mean():.3f} & {h.objective.mean():.3f} & {h.jaccard_mst.mean():.2f} \\\\")
    (OUT / "tab_case_variants.tex").write_text("\n".join(lines) + "\n")


def case_items():
    it = pd.read_csv(CASE / "items.csv")
    inc = it[it.record == "incident"]
    it = it[it.record.isna()]
    t = it.groupby(["item", "kind"]).agg(p=("p", "first"), c=("c", "first"), r2=("r2", "max")).reset_index()
    t = t.sort_values(["kind", "item"])
    mu = inc.groupby("item").u.mean()
    cvm = inc[inc.u > 0.02].groupby("item").cv.median()
    g = lambda x, k: "--" if pd.isna(x) else f"{x:.{k}f}"
    rows = [f"{r.item[0].upper() + r.item[1:]} & {r.kind} & {g(r.p, 3)} & {g(r.c, 2)} & {g(r.r2, 2)} & "
            f"{g(mu.get(r.item, np.nan), 3)} & {g(cvm.get(r.item, np.nan), 2)} \\\\" for r in t.itertuples()]
    (OUT / "tab_case_items.tex").write_text("\n".join(rows) + "\n")


def sarl():
    s = pd.read_csv(ROOT / "data" / "sarl" / "sarl_mst_multiscenario_sweep.csv")
    nm = {"HVAC_FAULT": "HVAC fault", "COMFORT_DEVIATION": "Comfort deviation", "SECURITY_ALARM": "Security alarm"}
    rl = {"FACILITY_MANAGER": "Facility manager", "OCCUPANT": "Occupant", "SECURITY": "Security operator"}
    rows = []
    for (sc, role), g in s.groupby(["scenario", "role"], sort=False):
        cells = ["\\textsc{Infeasible}" if r.status != "FEASIBLE" else
                 str(r.selected_items).replace("_", " ").replace("|", ", ").replace(";", ", ")
                 for _, r in g.sort_values("tau").iterrows()]
        rows.append(f"{nm[sc]} & {rl[role]} & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_sarl.tex").write_text("\n".join(rows) + "\n")


def sensitivity():
    p = pd.read_csv(BENCH / "comparison_parameter_sensitivity.csv")
    names = {"A1_soft_admissibility": "Penalty instead of hard admissibility",
             "C2_goal_obfuscation": "Goal obfuscation and legibility", "C4_parsimonious_haexa": "Parsimonious HAExA"}
    par = {"SOFT_PENALTY_PER_PROHIBITED_ITEM": "penalty per prohibited item", "PRIVACY_BUDGET": "privacy budget",
           "PARSIMONY_K": "number of items $k$"}
    rows = []
    for _, r in p[p.method.isin(names)].iterrows():
        v = int(r.value) if r.parameter == "PARSIMONY_K" else r.value
        rows.append(f"{names[r.method]} & {par[r.parameter]} & {v} & {r.R1_sufficient_when_possible_pct:.1f} & "
                    f"{r.R2_no_prohibited_item_pct:.1f} & {r.R1_R2_R3_all_met_pct:.1f} \\\\")
    (OUT / "tab_sensitivity.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    comparison()
    admissibility()
    robustness()
    case_variants()
    case_items()
    sarl()
    sensitivity()
    for f in sorted(OUT.glob("*.tex")):
        print("==", f.name)
        print(f.read_text())
