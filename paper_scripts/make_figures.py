"""
Figures of Paper 1 (model paper). Plain lines and bars only.

  figures/fig_case_study.pdf   feasibility against threshold per role; robustness to estimation error by rule
  figures/fig_robustness.pdf   nominal against robust selection on synthetic pools with estimation noise

Usage: python paper_scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
CASE = ROOT / "data" / "case_study_lbnl"
FIG = ROOT / "figures"
W = 6.3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
ROLE = {"facility_manager": ("Facility manager", "#2a78d6", "-"),
        "energy_auditor": ("Energy auditor", "#eb6834", "--"),
        "occupant": ("Occupant", "#1baf7a", "-.")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def title(a, letter, text):
    a.set_title(f"({letter}) {text}", loc="left", fontweight="bold", color=INK)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_case_study():
    d = pd.read_csv(CASE / "results.csv")
    nd = pd.read_csv(CASE / "nominal.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
    a = ax[0]
    for r, (lab, c, ls) in ROLE.items():
        g = d[d.role == r].groupby("tau").status.apply(lambda s: 100 * (s != "INFEASIBLE").mean())
        a.plot(g.index, g.values, color=c, ls=ls, lw=1.5, label=lab)
    a.set_xlabel(r"Sufficiency threshold $\tau$")
    a.set_ylabel("Incidents with a sufficient\nadmissible disclosure (%)")
    a.set_ylim(0, 102)
    a.grid(color=GRID, lw=0.5)
    a.set_axisbelow(True)
    title(a, "a", "Feasibility against threshold")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6.5, handlelength=2.2,
             columnspacing=1.0)
    b = ax[1]
    v = nd[nd.variant != "release"]
    rules = [("template", "Role template"), ("ratio", "Utility per cost"), ("MST", "MST (nominal)"),
             ("robust_g1", "Robust MST, $\\Gamma$ = 1"), ("robust_g2", "Robust MST, $\\Gamma$ = 2"),
             ("robust_gbox", "Robust MST, box")]
    vals = [100 * v[(v.variant == k) & v.returned.astype(bool)].suff_prob.mean() for k, _ in rules]
    y = np.arange(len(rules))
    b.barh(y, vals, color=["#b5b4ae", "#b5b4ae", "#2a78d6", "#1c5cab", "#1c5cab", "#1c5cab"], height=0.6)
    for yy, vv in zip(y, vals):
        b.text(vv + 1, yy, f"{int(vv + 0.5)}", va="center", fontsize=6.6, color=INK)
    b.set_yticks(y, [l for _, l in rules], fontsize=6.8)
    b.invert_yaxis()
    b.set_xlim(0, 105)
    b.set_xlabel("Disclosures still sufficient across bootstrap refits (%)")
    b.grid(axis="x", color=GRID, lw=0.5)
    b.set_axisbelow(True)
    title(b, "b", "Robustness to estimation error")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_case_study")


def fig_robustness():
    r = pd.read_csv(BENCH / "robustness.csv", low_memory=False)
    meths = [("exact", "Nominal"), ("robust_g1", "$\\Gamma$ = 1"), ("robust_g2", "$\\Gamma$ = 2"),
             ("robust_g3", "$\\Gamma$ = 3"), ("robust", "Box")]
    cols = {0.1: "#86b6ef", 0.2: "#1c5cab"}
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.2))
    x = np.arange(len(meths))
    spec = [("insufficient", "Insufficient under true values (%)", "Risk", True),
            ("infeasible", "Reported infeasible (%)", "Escalation", True),
            ("regret_pct", "Mean extra cost (%)", "Cost", False)]
    for j, (met, yl, ttl, pct) in enumerate(spec):
        a = ax[j]
        for k_, dv in enumerate((0.1, 0.2)):
            vals = []
            for k, _ in meths:
                g = r[r.delta == dv]
                if met != "infeasible":
                    g = g[~g[f"{k}_infeasible"].astype(bool)]
                s = g[f"{k}_{met}"].astype(float)
                vals.append(100 * s.mean() if pct else s.mean())
            a.bar(x + (k_ - 0.5) * 0.38, vals, width=0.36, color=cols[dv], label=f"$\\pm${int(dv * 100)}% noise")
        a.set_xticks(x, [l for _, l in meths], fontsize=6.6)
        a.set_ylabel(yl)
        a.axhline(0, color=INK2, lw=0.6)
        a.grid(axis="y", color=GRID, lw=0.5)
        a.set_axisbelow(True)
        title(a, "abc"[j], ttl)
    ax[0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_robustness")


if __name__ == "__main__":
    fig_case_study()
    fig_robustness()
