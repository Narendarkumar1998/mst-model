"""
Case study on measured building data: MST on candidate pools built from the LBNL
Building 59 data set (Luo et al., Scientific Data 9, 156, 2022; Dryad doi:10.7941/D1N33Q,
CC BY 4.0), an office building in Berkeley with end-use electricity, four rooftop units,
a heat pump, zone temperatures and setpoints, weather and camera occupant counts.
Study period 2018-06-01 to 2019-02-28 (case_study_lbnl/prepare_data.py).

Scenario. Each subsystem is reported by a remote agent. When the daily energy of an
end use (HVAC north or south, plug loads north or south, lighting south) deviates
strongly from its baseline on a working day, the assistant agent decides what to disclose to three
stakeholders who oversee the alert: the facility manager, an external energy auditor
and an occupant. Each candidate item is the daily deviation of one point.

Item values.
  u_i  measured: the item's share of the part of the incident day's deviation that a
       ridge fit of the target's daily deviation on all items (fitted without the
       incident day) explains, from a linear attribution (linear SHAP). The utilities of
       an incident sum to the explained share, so they are aggregated additively,
       U(S) = sum_{i in S} u_i (the product form is reported for comparison);
       their uncertainty is estimated from 200 bootstrap refits;
  p_i  measured: how much the item reveals about the presence of people, the R^2 of the
       camera occupant count of a floor on the item's hourly series after removing the
       hour-of-week means of both (the building schedule is public knowledge), maximum
       over the two monitored floors, capped at 0.95;
  c_i  modeled: interpretation cost by kind of point (the only modeled item value),
       tested against a uniform cost;
  admissibility, threshold and weights by role.

Also: Gamma-robust selection against the bootstrap uncertainty of the utilities, the
probability that a disclosure stays sufficient across bootstrap refits, the joint
occupancy R^2 of each disclosure against its aggregated exposure P(S), the release of
least exposure, policy-respecting practitioner baselines, and a uniform-cost check.

Writes data/case_study_lbnl/results.csv (all thresholds), nominal.csv (nominal thresholds,
all variants), items.csv, fit_quality.csv and stability.csv
Usage: python case_study_lbnl/run_case_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimization"))
import heuristics as h  # noqa: E402
import scip_model as sm  # noqa: E402

DATA = ROOT / "data" / "case_study_lbnl"
TARGETS = {"hvac_N": "HVAC north", "hvac_S": "HVAC south", "mels_N": "plug loads north",
           "mels_S": "plug loads south", "lig_S": "lighting south"}
ENERGY = list(TARGETS)
INCIDENTS_PER_TARGET = 12
OCC = ["occ_third_south", "occ_fourth_south"]
P_CAP = 0.95

# item kind of every point: (kind, readable name, daily aggregation)
POINTS = {**{e: ("energy", TARGETS[e], "sum") for e in ENERGY},
          **{f"rtu_00{k}_fltrd_sa_flow_tn": ("hvac", f"rooftop unit {k} supply airflow", "mean") for k in range(1, 5)},
          **{f"rtu_00{k}_sa_temp": ("hvac", f"rooftop unit {k} supply air temperature", "mean") for k in range(1, 5)},
          **{f"rtu_00{k}_oadmpr_pct": ("hvac", f"rooftop unit {k} outdoor-air damper", "mean") for k in range(1, 5)},
          "hp_hws_temp": ("hvac", "heat-pump hot-water temperature", "mean"),
          "indoor_temp_mean": ("comfort", "indoor temperature", "mean"),
          "cooling_setpoint_mean": ("comfort", "zone cooling setpoints", "mean"),
          "air_temp_set_1": ("weather", "outdoor temperature", "mean"),
          "relative_humidity_set_1": ("weather", "outdoor humidity", "mean"),
          "solar_radiation_set_1": ("weather", "solar radiation", "mean"),
          "occ_third_south": ("occupancy", "occupant count, third floor", "mean"),
          "occ_fourth_south": ("occupancy", "occupant count, fourth floor", "mean"),
          "weekend": ("calendar", "weekend or holiday", "mean")}

# modeled interpretation cost by kind (a scalar daily summary of one point)
KIND_C = {"energy": 0.20, "hvac": 0.30, "comfort": 0.15, "weather": 0.10, "occupancy": 0.15, "calendar": 0.05}
ROLES = {
    # admissible kinds, threshold, weights (P, C, K), cost multiplier for technical HVAC points
    "facility_manager": dict(kinds=set(KIND_C), tau=0.30, w=(0.2, 0.4, 0.4), hvac_mult=1.0),
    "energy_auditor": dict(kinds=set(KIND_C) - {"occupancy"}, tau=0.25, w=(1 / 3, 1 / 3, 1 / 3), hvac_mult=1.0),
    "occupant": dict(kinds={"energy", "comfort", "weather", "calendar"}, tau=0.10, w=(0.2, 0.6, 0.2),
                     hvac_mult=2.0),
}
CASE_REF_ITEMS = 10.0           # reference scales: a typical disclosure of 10 items of cost 0.2
CASE_REF_COST = 10.0 * 0.20
TAU_GRID = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BOOT = 200                      # bootstrap refits of the attribution model per incident
LOW_Q = 0.10                    # utility deviation: point estimate minus the 10% bootstrap quantile
GAMMAS = (1, 2, "box")          # robust variants; Gamma = 2 is the nominal robust model
TEMPLATE_K = 3                  # size of a fixed role template
US_HOLIDAYS = ["2018-07-04", "2018-09-03", "2018-11-12", "2018-11-22", "2018-11-23", "2018-12-24", "2018-12-25",
               "2018-12-31", "2019-01-01", "2019-01-21", "2019-02-18"]   # University of California holidays


def deviation(s, relative):
    """Deviation from the median of the same weekday in the two weeks before and after."""
    out = pd.Series(np.nan, index=s.index)
    for wd in range(7):
        m = s.index.dayofweek == wd
        x = s[m]
        nb = pd.concat([x.shift(k) for k in (-2, -1, 1, 2)], axis=1)
        base = nb.median(axis=1, skipna=True).where(nb.notna().sum(axis=1) >= 2)
        out[m] = (x - base) / base.abs() if relative else x - base
    return out.replace([np.inf, -np.inf], np.nan)


def fit(y, x):
    ok = y.notna() & x.notna()
    if ok.sum() < 60 or x[ok].std() == 0:
        return 0.0, 0.0
    b, _ = np.polyfit(x[ok], y[ok], 1)
    c = np.corrcoef(y[ok], x[ok])[0, 1]
    return (float(c * c) if np.isfinite(c) else 0.0), float(b)


def day_utility(y, x, day):
    R2, b = fit(y, x)
    if not (np.isfinite(y.get(day, np.nan)) and np.isfinite(x.get(day, np.nan))) or y[day] == 0:
        return 0.0
    return R2 * float(np.clip(b * x[day] / y[day], 0.0, 1.0))


def _ridge(Z, y, alpha):
    ym = y.mean()
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return b, ym


def _shares(b, zd, target):
    """Linear attribution of the deviation `target` (already centred) to the items."""
    if target == 0:
        return np.zeros(len(b))
    phi = b * zd * np.sign(target)                # positive = pushes towards the observed deviation
    pos = np.clip(phi, 0.0, None)                 # items pushing the other way receive no share
    explained = float(np.clip(phi.sum() / abs(target), 0.0, 1.0))
    return pos / pos.sum() * explained if pos.sum() > 0 else np.zeros(len(b))


def attribution_utilities(y, X, day, alpha=1.0, boot=0, rng=None):
    """u_i for one incident: linear attribution of the day's deviation (linear SHAP,
    Lundberg and Lee 2017). A ridge regression of the target's daily deviation on all item
    deviations (standardized, missing deviations set to zero) is fitted on every other day
    of the study period; item i's contribution on the incident day is
    phi_i = b_i (x_i(day) - mean x_i). The share of the day's deviation the fit explains,
    e = clip(sum_i phi_i / (y(day) - mean y), 0, 1), is divided among the items that push
    in the direction of the deviation in proportion to their contribution, so that
    u_i = e phi_i^+ / sum_j phi_j^+ and sum_i u_i = e; items pushing the other way get 0.
    With boot > 0, also returns the utilities of `boot` refits on bootstrap resamples of
    the training days (shape boot x items)."""
    ok = y.notna()
    Xf = X.fillna(0.0)
    train = (ok & (y.index != day)).to_numpy()
    mu, sd = Xf[train].mean(), Xf[train].std().replace(0, 1.0)
    Z = ((Xf - mu) / sd).to_numpy()
    yv = y.to_numpy()
    Zt, yt = Z[train], yv[train]
    zd = Z[X.index.get_loc(day)]
    b, ym = _ridge(Zt, yt, alpha)
    u = _shares(b, zd, y[day] - ym)
    if not boot:
        return u
    U = np.empty((boot, len(u)))
    n = len(yt)
    for k in range(boot):
        idx = rng.integers(0, n, n)
        bb, yb = _ridge(Zt[idx], yt[idx], alpha)
        U[k] = _shares(bb, zd, y[day] - yb)
    return u, U


def cv_r2(y, X, alpha=1.0, folds=10):
    """Ten-fold cross-validated R^2 of the ridge fit used for the attribution."""
    ok = y.notna().to_numpy()
    Z = X.fillna(0.0).to_numpy()[ok]
    yy = y.to_numpy()[ok]
    idx = np.arange(len(yy)) % folds
    pred = np.empty_like(yy)
    for f in range(folds):
        tr, te = idx != f, idx == f
        mu, sd = Z[tr].mean(0), Z[tr].std(0)
        sd[sd == 0] = 1.0
        Zt = (Z[tr] - mu) / sd
        b = np.linalg.solve(Zt.T @ Zt + alpha * np.eye(Z.shape[1]), Zt.T @ (yy[tr] - yy[tr].mean()))
        pred[te] = yy[tr].mean() + ((Z[te] - mu) / sd) @ b
    return 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def residuals(hourly):
    how = hourly.index.dayofweek * 24 + hourly.index.hour
    return hourly - hourly.groupby(how).transform("mean")


def privacy(resid):
    """p_i: R^2 of a floor's occupant count on the item, both net of hour-of-week means."""
    p = {}
    for col in POINTS:
        vals = []
        for o in OCC:
            ok = resid[col].notna() & resid[o].notna()
            if ok.sum() < 500 or resid.loc[ok, col].std() == 0:
                vals.append(0.0)
                continue
            r = np.corrcoef(resid.loc[ok, col], resid.loc[ok, o])[0, 1]
            vals.append(float(r * r) if np.isfinite(r) else 0.0)
        p[col] = min(P_CAP, max(vals))
    return p


def joint_r2(resid, cols):
    """Joint exposure of a disclosure: R^2 of a floor's occupant count on all disclosed
    items together (least squares, hour-of-week residuals), larger of the two floors."""
    if not cols:
        return np.nan
    if any(c in OCC for c in cols):               # an occupant count discloses itself
        return P_CAP
    best = 0.0
    for o in OCC:
        d = resid[cols + [o]].dropna()
        if len(d) < 500:
            continue
        A = np.column_stack([d[cols].to_numpy(), np.ones(len(d))])
        yo = d[o].to_numpy()
        coef, *_ = np.linalg.lstsq(A, yo, rcond=None)
        r2 = 1 - ((yo - A @ coef) ** 2).sum() / ((yo - yo.mean()) ** 2).sum()
        best = max(best, float(r2))
    return min(P_CAP, best)


def U_of(u, sel, utility):
    sel = list(sel)
    if not sel:
        return 0.0
    return float(np.sum(u[sel])) if utility == "additive" else 1.0 - float(np.prod(1.0 - u[sel]))


def P_of(p, sel):
    sel = list(sel)
    return 1.0 - float(np.prod(1.0 - p[sel])) if sel else 0.0


def J_of(u, p, c, sel, w):
    return sm.exact_objective(tuple(sel), list(u), list(p), list(c), w, None, CASE_REF_COST, CASE_REF_ITEMS)


def solve(u, p, c, adm, tau, w, utility="additive", gamma=None, dev=None, heuristics=True):
    best, res = None, {}
    if heuristics:
        best_J = np.inf
        for name, f in h.METHODS.items():
            pl = h.Pool(u, p, c, adm, tau, w, cost_scale=CASE_REF_COST, card_scale=CASE_REF_ITEMS, utility=utility)
            t0 = time.perf_counter()
            x = f(pl)
            res[name] = (pl.J(x) if x is not None else np.nan, time.perf_counter() - t0)
            if x is not None and pl.J(x) < best_J:
                best, best_J = pl.to_indices(x), pl.J(x)
    g = None if gamma is None else (len(u) if gamma == "box" else gamma)
    s = sm.solve_mst_scip(list(u), list(p), list(c), adm, tau, w, start=best if gamma is None else None,
                          cost_scale=CASE_REF_COST, card_scale=CASE_REF_ITEMS, utility=utility,
                          robust_gamma=g, deviation=None if dev is None else list(dev))
    return s, res


def ratio_until_sufficient(u, c, adm, tau):
    """Practitioner baseline: admissible items by decreasing utility per unit cost until sufficient."""
    order = sorted((i for i in range(len(u)) if adm[i] and u[i] > 0), key=lambda i: -u[i] / c[i])
    sel, tot = [], 0.0
    for i in order:
        if tot >= tau:
            break
        sel.append(i)
        tot += u[i]
    return tuple(sorted(sel)) if tot >= tau - 1e-12 else None


def release(u, p, adm, tau, utility, by):
    """Smallest release (by='count': fewest prohibited items, highest utility first) or the
    release of least exposure (by='exposure': exact, over all subsets of prohibited items)."""
    base = [i for i in range(len(u)) if adm[i]]
    proh = [i for i in range(len(u)) if not adm[i]]
    if U_of(u, base + proh, utility) < tau:
        return None
    if by == "count":
        rel = []
        for i in sorted(proh, key=lambda i: -u[i]):
            if U_of(u, base + rel, utility) >= tau:
                break
            rel.append(i)
        return rel
    best, best_P = None, np.inf
    for mask in range(1, 1 << len(proh)):
        r = [proh[k] for k in range(len(proh)) if mask >> k & 1]
        if U_of(u, base + r, utility) >= tau:
            pe = P_of(p, r)
            if pe < best_P - 1e-12 or (abs(pe - best_P) <= 1e-12 and len(r) < len(best)):
                best, best_P = r, pe
    return best


def jac(a, b):
    a, b = set(a or ()), set(b or ())
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    rng = np.random.default_rng(2026)
    hourly = pd.read_csv(DATA / "bldg59_hourly.csv.gz", parse_dates=["date"]).set_index("date")
    day_index = hourly.resample("D").mean().index
    hol = pd.DatetimeIndex(US_HOLIDAYS)
    weekend_d = pd.Series(((day_index.dayofweek >= 5) | day_index.isin(hol)).astype(float), index=day_index)
    hourly["weekend"] = ((hourly.index.dayofweek >= 5) | hourly.index.normalize().isin(hol)).astype(float)
    resid = residuals(hourly)

    daily = {}
    for col, (kind, _, agg) in POINTS.items():
        if col == "weekend":
            daily[col] = weekend_d
            continue
        g = hourly[col].resample("D")
        daily[col] = g.sum(min_count=18) if agg == "sum" else g.mean()
    dev = {}
    for col, (kind, _, _) in POINTS.items():
        if kind == "weather":
            dev[col] = daily[col] - daily[col].rolling(15, center=True, min_periods=7).mean()
        elif kind == "calendar":
            dev[col] = daily[col]
        else:
            dev[col] = deviation(daily[col], relative=False)
    target_dev = {e: deviation(daily[e], relative=False) for e in ENERGY}   # kWh above or below baseline
    P = privacy(resid)

    rows, nom_rows, item_rows, fit_quality, stab = [], [], [], [], []
    for target in ENERGY:
        y = target_dev[target]
        z = (y - y.median()) / (1.4826 * (y - y.median()).abs().median())
        days = []
        for d in z.abs().dropna().sort_values(ascending=False).index:
            if weekend_d[d] == 1:                       # alerts on working days only
                continue
            if all(abs((d - e).days) >= 7 for e in days):
                days.append(d)
            if len(days) == INCIDENTS_PER_TARGET:
                break
        keys = [k for k in POINTS if k != target]
        kinds = [POINTS[k][0] for k in keys]
        p = np.array([P[k] for k in keys])
        X = pd.DataFrame({k: dev[k] for k in keys})
        fit_quality.append(dict(target=TARGETS[target], cv_r2=cv_r2(y, X),
                                max_abs_corr_rtu_airflow=float(np.nanmax(
                                    X[[k for k in keys if "sa_flow" in k]].corr().abs().to_numpy().copy()
                                    - np.eye(4)))))
        UT = {}
        for day in days:
            UT[day] = attribution_utilities(y, X, day, boot=BOOT, rng=rng)
        for k in keys:
            item_rows.append(dict(target=TARGETS[target], point=k, item=POINTS[k][1], kind=POINTS[k][0],
                                  r2=fit(y, dev[k])[0], p=P[k], c=KIND_C[POINTS[k][0]]))
        for day in days:
            u, Ub = UT[day]
            dev_u = np.maximum(0.0, u - np.quantile(Ub, LOW_Q, axis=0))
            cv_items = Ub.std(axis=0) / np.maximum(Ub.mean(axis=0), 1e-9)
            for role, R0 in ROLES.items():
                adm = [kd in R0["kinds"] for kd in kinds]
                c = np.array([KIND_C[kd] * (R0["hvac_mult"] if kd == "hvac" else 1.0) for kd in kinds])
                w = R0["w"]
                # fixed role template: the TEMPLATE_K admissible items of highest mean utility over
                # the other incidents of this target (decided before the incident)
                others = np.mean([UT[e][0] for e in days if e != day], axis=0)
                template = tuple(sorted(sorted((i for i in range(len(u)) if adm[i]),
                                               key=lambda i: -others[i])[:TEMPLATE_K]))
                prev = {}
                for tau in TAU_GRID:
                    s, res = solve(u, p, c, adm, tau, w)
                    sel = s.selected
                    row = dict(target=TARGETS[target], day=day.date(), z=float(z[day]), role=role, tau=tau,
                               nominal=bool(abs(tau - R0["tau"]) < 1e-9), n_items=len(keys),
                               n_admissible=int(sum(adm)), status=s.status, objective=s.objective,
                               runtime_s=s.runtime, n_selected=len(sel),
                               selected="; ".join(POINTS[keys[i]][1] for i in sel),
                               selected_kinds="|".join(sorted({kinds[i] for i in sel})),
                               utility=U_of(u, sel, "additive") if sel else np.nan,
                               privacy=P_of(p, sel) if sel else np.nan,
                               max_admissible_utility=U_of(u, [i for i in range(len(u)) if adm[i]], "additive"),
                               max_pool_utility=U_of(u, range(len(u)), "additive"))
                    for name, (Jh, rt) in res.items():
                        row[f"{name}_objective"] = Jh
                        row[f"{name}_runtime_s"] = rt
                    # baselines at every threshold (for stability across nearby thresholds)
                    ratio = ratio_until_sufficient(u, c, adm, tau)
                    fs, _ = solve(u, p, c, adm, tau, (0.001, 0.0, 0.999), heuristics=False)
                    fewest = fs.selected if fs.status != "INFEASIBLE" else None
                    rob, _ = solve(u, p, c, adm, tau, w, gamma=2, dev=dev_u, heuristics=False)
                    robust = rob.selected if rob.status != "INFEASIBLE" else None
                    cur = {"MST": sel if s.status != "INFEASIBLE" else None, "robust_g2": robust,
                           "ratio": ratio, "fewest": fewest, "template": template}
                    for k_, v in cur.items():
                        if k_ in prev and prev[k_] is not None and v is not None:
                            stab.append(dict(target=TARGETS[target], day=day.date(), role=role, method=k_,
                                             tau_from=prev["tau"], tau_to=tau, jaccard=jac(prev[k_], v)))
                    prev = dict(cur, tau=tau)
                    if s.status == "INFEASIBLE":
                        rc = release(u, p, adm, tau, "additive", "count")
                        row["release_needed"] = len(rc) if rc is not None else np.nan
                        row["release_items"] = "; ".join(POINTS[keys[i]][1] for i in rc) if rc else ""
                    rows.append(row)
                    if not row["nominal"]:
                        continue
                    # ---------------- nominal threshold: all variants ----------------
                    base = dict(target=TARGETS[target], day=day.date(), role=role, tau=tau,
                                n_admissible=int(sum(adm)), mst_status=s.status,
                                explained_share=float(u.sum()))
                    variants = {"MST": sel if s.status != "INFEASIBLE" else None}
                    sp, _ = solve(u, p, c, adm, tau, w, utility="product", heuristics=False)
                    variants["MST_product_utility"] = sp.selected if sp.status != "INFEASIBLE" else None
                    for g in GAMMAS:
                        sg, _ = solve(u, p, c, adm, tau, w, gamma=g, dev=dev_u, heuristics=False)
                        variants[f"robust_g{g}"] = sg.selected if sg.status != "INFEASIBLE" else None
                    variants["ratio"] = ratio
                    variants["fewest"] = fewest
                    variants["template"] = template
                    su, _ = solve(u, p, np.full(len(u), 0.2), adm, tau, w, heuristics=False)
                    variants["MST_uniform_cost"] = su.selected if su.status != "INFEASIBLE" else None
                    for name, S in variants.items():
                        r = dict(base, variant=name, returned=S is not None)
                        if S is not None:
                            S = list(S)
                            r.update(n_selected=len(S), utility=U_of(u, S, "additive"),
                                     sufficient=U_of(u, S, "additive") >= tau - 1e-12,
                                     suff_prob=float(np.mean(Ub[:, S].sum(axis=1) >= tau - 1e-12)) if S else 0.0,
                                     privacy=P_of(p, S), joint_privacy=joint_r2(resid, [keys[i] for i in S]),
                                     objective=J_of(u, p, c, S, w), jaccard_mst=jac(S, variants["MST"]),
                                     admissible=all(adm[i] for i in S),
                                     selected="; ".join(POINTS[keys[i]][1] for i in S))
                        nom_rows.append(r)
                    if s.status == "INFEASIBLE":
                        rc = release(u, p, adm, tau, "additive", "count")
                        re_ = release(u, p, adm, tau, "additive", "exposure")
                        nom_rows.append(dict(base, variant="release",
                                             release_possible=rc is not None,
                                             release_count_n=len(rc) if rc is not None else np.nan,
                                             release_count_P=P_of(p, rc) if rc else np.nan,
                                             release_exposure_n=len(re_) if re_ is not None else np.nan,
                                             release_exposure_P=P_of(p, re_) if re_ else np.nan,
                                             release_count_items="; ".join(POINTS[keys[i]][1] for i in rc) if rc else "",
                                             release_exposure_items="; ".join(POINTS[keys[i]][1] for i in re_) if re_ else ""))
                    # attribution stability of the incident (collinearity check)
                    if role == "facility_manager":
                        for i, k in enumerate(keys):
                            item_rows.append(dict(target=TARGETS[target], day=day.date(), point=k,
                                                  item=POINTS[k][1], kind=POINTS[k][0], u=u[i],
                                                  u_boot_mean=Ub[:, i].mean(), u_boot_sd=Ub[:, i].std(),
                                                  u_low=u[i] - dev_u[i], cv=cv_items[i], record="incident"))
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "results.csv", index=False)
    nd = pd.DataFrame(nom_rows)
    nd.to_csv(DATA / "nominal.csv", index=False)
    pd.DataFrame(item_rows).to_csv(DATA / "items.csv", index=False)
    pd.DataFrame(fit_quality).to_csv(DATA / "fit_quality.csv", index=False)
    sd = pd.DataFrame(stab)
    sd.to_csv(DATA / "stability.csv", index=False)
    print(pd.DataFrame(fit_quality).round(3).to_string(index=False))
    nom = df[df.nominal]
    print(nom.groupby("role").agg(n=("status", "size"), feasible=("status", lambda s: (s != "INFEASIBLE").mean()),
                                  sel=("n_selected", "median"), rt=("runtime_s", "max")))
    v = nd[nd.variant != "release"]
    print(v.groupby(["variant"]).agg(returned=("returned", "mean"), sufficient=("sufficient", "mean"),
                                     suff_prob=("suff_prob", "mean"), n=("n_selected", "median"),
                                     P=("privacy", "median"), Pjoint=("joint_privacy", "median"),
                                     J=("objective", "median")).round(3).to_string())
    print(sd.groupby("method").jaccard.mean().round(3).to_string())


if __name__ == "__main__":
    main()
