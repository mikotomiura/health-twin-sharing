"""Confirmatory E2 tests from the per-rep map (docs/preregistration-draft.md §2, §6; .steering/20261113-e2-confirm).

Reads <exp>/results/e2_map_reps.csv (scripts/e2_map.py; one row = attribute x lambda x rep x D x K) and <exp>/config.json
(lambdas, reps, primary_attribute, delta_E2, ci, ci_equivalence). The same tests are run on dV_acc (primary) and
dV_conf (secondary), on the primary attribute only; every cell of every attribute is described.

  H2   lambda = 1, K2: D0 - D2. Paired by rep (same permutation and CV split), t interval with R - 1 df,
       one-sided p for H0: mean <= 0. Supported: lower bound > 0 and estimate > delta_E2.
  H0a  lambda = 0, D0, K2: mean dV, ci_equivalence (90%) t interval. Equivalent: interval inside (-delta, delta)
       (= TOST at 5%; the TOST p is reported too).
  H0c  K2: lambda1 - lambda0 (D0; Welch, the reps of different lambdas are independent permutations),
       D0 - D2 and D0 - D1 (lambda = 1; paired). A contrast enters the judgement when its 95% interval lies outside
       (-delta_E2, delta_E2) (paper-write 判断 15: with R = 20 a far smaller difference already excludes 0).
       With --e1, the E1 point estimates (metrics.json "H0c_E1_side") are set beside it; supported = every contrast in
       the judgement has the same sign in E1 (None when no contrast is in the judgement).
  Holm With --e1, the one-sided p of H1 (E1 "H1_K2_minus_K0") and of H2 (here, dV_acc) adjusted together.
  Described: K3 - K2 (paired) per lambda and D; whether the D0 / K2 mean rises with lambda (exploratory).
A zero SD (a saturated cell) gives a point interval and p = 0 if the mean is > 0, else 1.
The script stops (the plan is complete by construction) on: an attribute of config/condition_lists.json missing, missing,
duplicated or extra reps or cells, non-finite dV values, no D1 list for the primary attribute, and with --e1 a missing
H1 p value or a missing E1 estimate for a contrast in the H0c judgement (prereg §6).

Usage: python scripts/e2_summarise.py <exp-dir> [--e1 <e1-exp-dir>]
Writes <exp>/results/e2_tests.json and adds the headline tests to <exp>/metrics.json under "tests".
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ESTIMATORS = ("dV_acc", "dV_conf")
SE_EPS = 1e-12
H0C = {"lambda1_minus_lambda0_D0_K2": ("welch", (1.0, "D0", "K2"), (0.0, "D0", "K2")),
       "D0_minus_D2_lambda1_K2": ("paired", (1.0, "D0", "K2"), (1.0, "D2", "K2")),
       "D0_minus_D1_lambda1_K2": ("paired", (1.0, "D0", "K2"), (1.0, "D1", "K2"))}


def _out(est: float, se: float, df: float, n: int, ci: float, sd: float | None = None) -> dict:
    if se < SE_EPS:  # a saturated cell: zero SD up to rounding
        se = 0.0
        lo = hi = est
        p = 0.0 if est > 0 else 1.0
    else:
        h = float(stats.t.ppf(1 - (1 - ci) / 2, df)) * se
        lo, hi = est - h, est + h
        p = float(stats.t.sf(est / se, df))
    r = {"est": float(est), "lo": float(lo), "hi": float(hi), "ci": ci, "p_one_sided": p, "se": float(se),
         "df": float(df), "n": int(n)}
    if sd is not None:
        r["sd"] = float(sd)
    return r


def t_interval(x: np.ndarray, ci: float) -> dict:
    """Mean of x with a t interval (n - 1 df) and the one-sided p for H0: mean <= 0."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 reps")
    sd = float(x.std(ddof=1))
    return _out(float(x.mean()), sd / math.sqrt(n), n - 1, n, ci, sd)


def welch(x: np.ndarray, y: np.ndarray, ci: float) -> dict:
    """mean(x) - mean(y) for independent samples, Welch-Satterthwaite df."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else len(x) + len(y) - 2
    return _out(float(x.mean() - y.mean()), se, df, len(x) + len(y), ci)


def tost(r: dict, delta: float) -> dict:
    """Equivalence of the mean in r (a t_interval at the equivalence level) to (-delta, delta)."""
    est, se, df = r["est"], r["se"], r["df"]
    if se < SE_EPS:
        p = 0.0 if abs(est) < delta else 1.0
    else:
        p = max(float(stats.t.sf((est + delta) / se, df)), float(stats.t.cdf((est - delta) / se, df)))
    return r | {"delta": delta, "p_tost": p, "equivalent": bool(-delta < r["lo"] and r["hi"] < delta)}


def holm(ps: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values (monotone, capped at 1)."""
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def sign(v: float) -> int:
    return int(np.sign(v))


def planned_cells(cfg: dict, lists: dict, attributes) -> list[tuple]:
    cells = []
    for a in attributes:
        ds = ["D0", "D2"] + (["D1"] if lists["attributes"][a].get("D1") else [])
        cells += [(a, float(lam), d, k) for lam in cfg["lambdas"] for d in ds for k in ("K2", "K3")]
    return cells


def load(exp: Path) -> tuple[pd.DataFrame, dict]:
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(exp / "results" / "e2_map_reps.csv")
    df["lambda"] = df["lambda"].astype(float)
    return df, cfg


def check_complete(df: pd.DataFrame, cfg: dict, lists: dict) -> dict:
    attrs = sorted(df.attribute.unique())
    if set(attrs) != set(lists["attributes"]) or cfg["primary_attribute"] not in attrs:
        raise SystemExit(f"e2_summarise: attributes {attrs} != config/condition_lists.json {sorted(lists['attributes'])}")
    if not lists["attributes"][cfg["primary_attribute"]].get("D1"):
        raise SystemExit("e2_summarise: the primary attribute has no D1 list (H0c needs D0 - D1)")
    vals = df[list(ESTIMATORS)].to_numpy(float)
    if not np.isfinite(vals).all():
        raise SystemExit(f"e2_summarise: {int((~np.isfinite(vals)).sum())} non-finite dV values")
    want = set(range(cfg["reps"]))
    got = {k: set(g.rep) for k, g in df.groupby(["attribute", "lambda", "D", "K"])}
    plan = planned_cells(cfg, lists, attrs)
    bad = [c for c in plan if got.get(c) != want or len(df.query(
        "attribute == @c[0] and `lambda` == @c[1] and D == @c[2] and K == @c[3]")) != cfg["reps"]]
    extra = set(got) - set(plan)
    if bad or extra:
        raise SystemExit(f"e2_summarise: incomplete or unplanned cells: missing/partial {bad[:5]} extra {sorted(extra)[:5]}")
    return {"cells": len(plan), "reps": cfg["reps"], "rows": int(len(df)), "attributes": attrs}


def cell(df: pd.DataFrame, a: str, c: tuple, col: str) -> pd.Series:
    lam, d, k = c
    s = df[(df.attribute == a) & (df["lambda"] == lam) & (df.D == d) & (df.K == k)].set_index("rep")[col]
    return s.sort_index()


def summarise(exp: Path, e1: Path | None = None) -> dict:
    lists = json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))
    df, cfg = load(exp)
    comp = check_complete(df, cfg, lists)
    a, delta, ci, cie = cfg["primary_attribute"], cfg["delta_E2"], cfg["ci"], cfg["ci_equivalence"]
    lams = sorted(float(x) for x in cfg["lambdas"])
    out: dict = {"primary_attribute": a, "primary_estimator": "dV_acc", "delta_E2": delta, "completeness": comp}

    e1m = json.loads((e1 / "metrics.json").read_text(encoding="utf-8")) if e1 else None
    for col in ESTIMATORS:
        r: dict = {}
        h2 = t_interval(cell(df, a, (1.0, "D0", "K2"), col) - cell(df, a, (1.0, "D2", "K2"), col), ci)
        r["H2_D0_minus_D2_lambda1_K2"] = h2 | {"supported": bool(h2["lo"] > 0 and h2["est"] > delta)}
        r["H0a_lambda0_D0_K2"] = tost(t_interval(cell(df, a, (0.0, "D0", "K2"), col), cie), delta)
        h0c = {}
        for name, (kind, c1, c2) in H0C.items():
            x1, x2 = cell(df, a, c1, col), cell(df, a, c2, col)
            t = t_interval(x1 - x2, ci) if kind == "paired" else welch(x1.to_numpy(), x2.to_numpy(), ci)
            h0c[name] = t | {"test": kind, "in_judgement": bool(t["lo"] > delta or t["hi"] < -delta)}
        if e1m is not None:
            e1s = e1m.get("H0c_E1_side") or {}
            for name, t in h0c.items():
                e1est = (e1s.get(name) or {}).get("est")
                if t["in_judgement"] and (e1est is None or not math.isfinite(e1est)):
                    raise SystemExit(f"e2_summarise: E1 estimate missing for the judged H0c contrast {name}")
                t["E1_est"] = e1est
                t["sign_agrees"] = None if e1est is None else sign(e1est) == sign(t["est"])
            judged = [t for t in h0c.values() if t["in_judgement"]]
            h0c["supported"] = None if not judged else all(t["sign_agrees"] for t in judged)
        r["H0c_E2_side"] = h0c
        r["K3_minus_K2_descriptive"] = {
            f"lambda{lam}|{d}": t_interval(cell(df, a, (lam, d, "K3"), col) - cell(df, a, (lam, d, "K2"), col), ci)
            for lam in lams for d in sorted(df[df.attribute == a].D.unique())}
        means = [float(cell(df, a, (lam, "D0", "K2"), col).mean()) for lam in lams]
        r["exploratory_lambda_monotone_D0_K2"] = {"lambdas": lams, "means": means,
                                                  "non_decreasing": bool(np.all(np.diff(means) >= 0))}
        out[col] = r

    if e1m is not None:
        p1 = (e1m.get("H1_K2_minus_K0") or {}).get("p_one_sided")
        if p1 is None or not 0 <= p1 <= 1:
            raise SystemExit(f"e2_summarise: E1 H1 p value missing or out of [0, 1]: {p1}")
        p2 = out["dV_acc"]["H2_D0_minus_D2_lambda1_K2"]["p_one_sided"]
        out["holm"] = {"raw": {"H1_E1": p1, "H2_E2": p2}, "adjusted": holm({"H1_E1": p1, "H2_E2": p2})}
        out["e1_source"] = str(e1)

    desc = {}
    for (at, lam, d, k), g in df.groupby(["attribute", "lambda", "D", "K"]):
        g = g.sort_values("rep")
        desc[f"{at}|lam={lam}|{d}|{k}"] = {col: t_interval(g[col].to_numpy(), ci) for col in ESTIMATORS} | {
            "cost_total": float(g.cost_total.mean()), "cost_extra": float(g.cost_extra.mean())}
    out["descriptive"] = desc
    return out


def main(exp: Path, e1: Path | None) -> None:
    out = summarise(exp, e1)
    (exp / "results").mkdir(exist_ok=True)
    (exp / "results" / "e2_tests.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    mpath = exp / "metrics.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    p = out["dV_acc"]
    m["tests"] = {"primary_estimator": "dV_acc", "H2": p["H2_D0_minus_D2_lambda1_K2"],
                  "H0a": p["H0a_lambda0_D0_K2"], "H0c_E2_side": p["H0c_E2_side"], "holm": out.get("holm")}
    mpath.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8", newline="\n")
    h2, h0a = p["H2_D0_minus_D2_lambda1_K2"], p["H0a_lambda0_D0_K2"]
    print(f"H2  D0-D2 = {h2['est']:.4f} [{h2['lo']:.4f}, {h2['hi']:.4f}] p={h2['p_one_sided']:.3g} supported={h2['supported']}")
    print(f"H0a dV(lam0) = {h0a['est']:.4f} [{h0a['lo']:.4f}, {h0a['hi']:.4f}] equivalent={h0a['equivalent']}")
    for k, t in p["H0c_E2_side"].items():
        if isinstance(t, dict):
            print(f"H0c {k} = {t['est']:.4f} [{t['lo']:.4f}, {t['hi']:.4f}] in_judgement={t['in_judgement']}"
                  + (f" E1={t['E1_est']} agrees={t['sign_agrees']}" if "E1_est" in t else ""))
    if "holm" in out:
        print("Holm", out["holm"])


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(Path(args[0]), None)
    elif len(args) == 3 and args[1] == "--e1":
        main(Path(args[0]), Path(args[2]))
    else:
        raise SystemExit(__doc__.split("Usage: ")[1])
