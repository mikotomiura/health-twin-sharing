"""Confirmatory E2 tests from the per-rep map (docs/preregistration-draft.md v4: OSF items Statistical models, Inference criteria, Missing data; .steering/20261113-e2-confirm).

Reads <exp>/results/e2_map_reps.csv (scripts/e2_map.py; one row = attribute x lambda x rep x D x K) and <exp>/config.json
(lambdas, reps, primary_attribute, delta_E2, ci, ci_equivalence). The same tests are run on dV_bal (primary, 判断 25:
balanced accuracy 1/2 TPR + 1/2 TNR, prediction p > 0.5) and on dV_acc and dV_conf (secondary), on the primary
attribute only; every cell of every attribute is described. delta_E2 is read from the config as it stands.

  H2   lambda = 1, K2: D0 - D2. Paired by rep (same permutation and CV split), t interval with R - 1 df,
       one-sided p for H0: mean <= 0. The interval flag ("unadjusted_interval_criterion": lower bound > 0 and estimate
       > delta_E2) is for reading only; the decision is below (Holm).
  H0a  lambda = 0, D0, K2: mean dV, ci_equivalence (90%) t interval. Equivalent: interval inside (-delta, delta)
       (= TOST at 5%; the TOST p is reported too). H0a is judged here (E2); E1's H0a is descriptive.
  H0c  K2: lambda1 - lambda0 (D0; Welch, the reps of different lambdas are independent permutations),
       D0 - D2 and D0 - D1 (lambda = 1; paired). A contrast enters the judgement when its 95% interval lies outside
       (-delta_E2, delta_E2) (paper-write 判断 15: with R = 20 a far smaller difference already excludes 0).
       With --e1, the E1 point estimates (metrics.json "H0c_E1_side") are set beside it; supported = every contrast in
       the judgement has the same sign in E1 (None when no contrast is in the judgement).
  Holm With --e1, the one-sided p of H1 (E1 "H1_K2_minus_K0" "p_one_sided_permutation", the sign-flip permutation p, 判断 39; the bootstrap p_one_sided is not used) and of H2 (here, dV_bal) adjusted together. FINAL
       decision (判断 17・22): H1_supported = adjusted p <= alpha_one_sided (config, 0.025); the E1 estimate vs delta_E1
       is reported beside it (H1_estimate_above_delta_E1, descriptive, not part of the decision).
       H2_supported = adjusted p <= alpha_one_sided and the E2 estimate > delta_E2. Without --e1 no final decision is
       made ("final_decision": null, with the reason).
  Misclassification cells (descriptive, 判断 21; dV_bal, primary attribute; the dV_acc table is kept as a secondary): per lambda x D x K the 95% interval of dV and
       whether the lower bound of the interval of dV_bal is above delta_E2 (misclassified cell, 判断 24), the count of such cells per lambda, and D0 - D2 (K2, paired) per lambda.
       No test, no decision.
  Described: K3 - K2 (paired) per lambda and D; whether the D0 / K2 mean rises with lambda (exploratory).
Interval method (判断 28; output "interval_method"). config "bootstrap_persons" true: every rep is a person bootstrap, so the R
  per-rep values ARE the bootstrap distribution. Interval = the (1-ci)/2 and (1+ci)/2 percentiles of the R values
  (a paired difference = the difference of the same reps; lambda1 - lambda0 = the difference by rep number, so H0c
  is paired by rep number in this mode), point estimate = the mean of the R values, one-sided p = (b + 1) / (R + 1)
  with b = values <= 0 (判断 32), TOST (H0a) = the 90% percentile interval inside (-delta, delta) (decides; p_tost =
  max of the share <= -delta and the share >= delta is descriptive only). No t interval and no
  Welch in this mode ("se" is the SD of the values and does not shrink with R). false (the development replays): the
  t interval (R - 1 df) and Welch as described above. R is a number of rep; more reps do not narrow a percentile interval.
A zero SD (a saturated cell) gives a point interval and p = 0 if the mean is > 0, else 1: this rule is for the t-interval mode only
(bootstrap_persons: false); in the confirmatory percentile path the one-sided p is always (b + 1) / (R + 1).
The script stops (the plan is complete by construction) on: an attribute of config/condition_lists.json missing, missing,
duplicated or extra reps or cells, non-finite dV values, no D1 list for the primary attribute, and with --e1 a missing
H1 p value, an E1 metrics.json whose primary_estimator is not "balanced", or a missing E1 estimate for a contrast in the
H0c judgement (OSF item Inference criteria).

Added by 判断 35-38 (all outside Holm and outside the H1 / H2 decision, except as stated):
  C13  with --e1 the script stops unless the E1 metrics.json has completeness.complete true (every stage, utility included, has all
       its planned rows and no unplanned row; an unfinished E1 is not summarised); e2_tests.json records the sha256 of that metrics.json.
  C14  "secondary_H2": the two secondaries fixed in advance, as the same contrast as H2 (lambda 1, K2, D0 - D2, paired by rep):
       dV_opt (net increment of the balanced accuracy of the best rule, act iff p > pi_hat of the rep) and dAUC (auc_KR - auc_K).
       Both columns are required in e2_map_reps.csv. PRIMARY stays dV_bal.
  C15  "H2_residual_leak": the interval of dV_bal at lambda 1, D2, K2; supported = lower end > delta_E2 (a secondary test, no Holm).
  C18  "descriptive_lambda1_original_sample": results/e2_lambda1_original.csv (e2_map.py; required when bootstrap_persons is true), the
       value of each lambda 1 D x K cell and of D0 - D2 (K2) on the original sample, beside the R-rep mean; judged values stay the R-rep means.
  C19  with --e1: "H1_scope" = "full" iff E1's H0b, E2's H0c and E1's H0d (判断 44) are all True (supported), else "E1_measurement_only" (False or None
       = not judged); "H1_K0n_length_note" = (K2 - K0n estimate < 0.5 * (K2 - K0 estimate)) from the E1 metrics (True too when the
       K0n estimate is missing: the control cannot rule length out; None when the K2 - K0 estimate is <= 0).
  判断 39: the H1 p entering Holm is H1_K2_minus_K0.p_one_sided_permutation of the E1 metrics (sign-flip permutation test).

Usage: python scripts/e2_summarise.py <exp-dir> [--e1 <e1-exp-dir>]
Writes <exp>/results/e2_tests.json and adds the headline tests to <exp>/metrics.json under "tests".
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "dV_bal"
ESTIMATORS = (PRIMARY, "dV_acc", "dV_conf")
SECONDARY_H2 = ("dV_opt", "dAUC")  # 判断 35: secondaries beside H2 (not in ESTIMATORS: no H0a / H0c / cells for them)
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


def percentile_interval(x: np.ndarray, ci: float) -> dict:
    """Person-bootstrap distribution x (R values, one per rep): mean, percentile interval, one-sided
    p = (b + 1) / (R + 1) with b = values <= 0 (paper-write 判断 32; never exactly 0)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 reps")
    lo, hi = np.percentile(x, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], method="linear")
    return {"est": float(x.mean()), "lo": float(lo), "hi": float(hi), "ci": ci, "p_one_sided": float(((x <= 0).sum() + 1) / (n + 1)),
            "se": float(x.std(ddof=1)), "df": None, "n": int(n), "sd": float(x.std(ddof=1))}


def interval(x, ci: float, boot: bool) -> dict:
    """The interval of the per-rep values x: percentile (bootstrap_persons true) or t over reps (false)."""
    return percentile_interval(x, ci) if boot else t_interval(x, ci)


def tost_pct(x: np.ndarray, ci: float, delta: float) -> dict:
    """Equivalence under the bootstrap: the ci percentile interval inside (-delta, delta)."""
    x = np.asarray(x, float)
    r = percentile_interval(x, ci)
    p = max(float((x <= -delta).mean()), float((x >= delta).mean()))  # descriptive; the interval decides
    return r | {"delta": delta, "p_tost": p, "equivalent": bool(-delta < r["lo"] and r["hi"] < delta)}


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
    missing_cols = [c for c in ESTIMATORS + SECONDARY_H2 if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"e2_summarise: e2_map_reps.csv has no column {missing_cols} (rerun scripts/e2_map.py)")
    vals = df[list(ESTIMATORS + SECONDARY_H2)].to_numpy(float)
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


def misclassification_cells(df: pd.DataFrame, a: str, lams: list, delta: float, ci: float, col: str = PRIMARY,
                            boot: bool = False) -> dict:
    """Descriptive (判断 21, corrected by 判断 24; no test): a misjudged cell = the audit is judged safe but the release
    leaks, i.e. the `col` (dV_bal primary, dV_acc secondary) interval's lower end > delta_E2. Cells whose upper end < -delta_E2 (the release lowers the
    attacker's accuracy; the negative floor of the dV in `col`, dV_bal primary) are counted apart as negative_floor. Per lambda, and D0 - D2 (K2,
    paired) per lambda."""
    ds = sorted(df[df.attribute == a].D.unique())
    cells, per_lam = {}, {}
    for lam in lams:
        n_mis = n_neg = 0
        for d in ds:
            for k in ("K2", "K3"):
                t = interval(cell(df, a, (lam, d, k), col), ci, boot)
                mis, neg = bool(t["lo"] > delta), bool(t["hi"] < -delta)
                n_mis += mis
                n_neg += neg
                cells[f"lambda{lam}|{d}|{k}"] = {"est": t["est"], "lo": t["lo"], "hi": t["hi"], "misjudged": mis,
                                                 "negative_floor": neg}
        per_lam[str(lam)] = {"n_misjudged": n_mis, "n_negative_floor": n_neg, "n_cells": len(ds) * 2}
    d0d2 = {str(lam): interval(cell(df, a, (lam, "D0", "K2"), col) - cell(df, a, (lam, "D2", "K2"), col), ci, boot)
            for lam in lams}
    return {"estimator": col, "delta_E2": delta, "ci": ci, "rule": "misjudged = interval lower end > delta_E2",
            "cells": cells, "by_lambda": per_lam,
            "D0_minus_D2_K2_by_lambda": d0d2, "descriptive": True}


def lambda1_original(path: Path, df: pd.DataFrame, a: str, ci: float, boot: bool) -> dict:
    """C18 (判断 38): the lambda = 1 cells of the primary attribute on the original sample (results/e2_lambda1_original.csv,
    e2_map.py), beside the R-rep mean and the interval of the reps. Descriptive; the judged values stay the R-rep means."""
    o = pd.read_csv(path)
    o = o[(o.attribute == a) & (o["lambda"].astype(float) == 1.0)]
    ds = sorted(df[df.attribute == a].D.unique())
    want = {(d, k) for d in ds for k in ("K2", "K3")}
    got = set(zip(o.D, o.K))
    if got != want or len(o) != len(want) or "dV_bal" not in o.columns:
        raise SystemExit(f"e2_summarise: {path.name} does not hold exactly one row per D x K of the primary attribute at lambda 1 "
                         f"(missing {sorted(want - got)}, extra {sorted(got - want)}, rows {len(o)})")
    val = lambda d, k, col: float(o[(o.D == d) & (o.K == k)][col].iloc[0])  # noqa: E731
    cells = {}
    for d, k in sorted(want):
        t = interval(cell(df, a, (1.0, d, k), PRIMARY), ci, boot)
        cells[f"lambda1.0|{d}|{k}"] = {"original": val(d, k, PRIMARY), "R_mean": t["est"], "lo": t["lo"], "hi": t["hi"]}
    h2 = interval(cell(df, a, (1.0, "D0", "K2"), PRIMARY) - cell(df, a, (1.0, "D2", "K2"), PRIMARY), ci, boot)
    return {"estimator": PRIMARY, "descriptive": True, "cells": cells,
            "H2_D0_minus_D2_K2": {"original": val("D0", "K2", PRIMARY) - val("D2", "K2", PRIMARY), "R_mean": h2["est"],
                                  "lo": h2["lo"], "hi": h2["hi"]},
            "note": "one CV run on the original sample, no resampling; the judgement uses the R-rep mean"}


def summarise(exp: Path, e1: Path | None = None) -> dict:
    lists = json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))
    df, cfg = load(exp)
    comp = check_complete(df, cfg, lists)
    a, delta, ci, cie = cfg["primary_attribute"], cfg["delta_E2"], cfg["ci"], cfg["ci_equivalence"]
    lams = sorted(float(x) for x in cfg["lambdas"])
    boot = bool(cfg.get("bootstrap_persons", False))
    out: dict = {"primary_attribute": a, "primary_estimator": PRIMARY, "delta_E2": delta, "completeness": comp,
                 "interval_method": "percentile_bootstrap" if boot else "t_over_reps"}

    e1m = json.loads((e1 / "metrics.json").read_text(encoding="utf-8")) if e1 else None
    if e1m is not None and e1m.get("primary_estimator") != "balanced":  # e.g. an E1 summarised before 判断 25 (pi-weighted)
        raise SystemExit(f"{e1}/metrics.json: primary_estimator is {e1m.get('primary_estimator')!r}, not 'balanced' "
                         "(判断 25); re-summarise the E1 run with scripts/e1_summarise.py")
    if e1m is not None and (e1m.get("completeness") or {}).get("complete") is not True:  # 判断 36
        raise SystemExit(f"{e1}/metrics.json: completeness.complete is not true: the E1 run is not finished "
                         "(run e1_run.py --stage all to the end and summarise once, 判断 36)")
    for col in ESTIMATORS:
        r: dict = {}
        h2 = interval(cell(df, a, (1.0, "D0", "K2"), col) - cell(df, a, (1.0, "D2", "K2"), col), ci, boot)
        r["H2_D0_minus_D2_lambda1_K2"] = h2 | {
            "unadjusted_interval_criterion": bool(h2["lo"] > 0 and h2["est"] > delta)}
        x0 = cell(df, a, (0.0, "D0", "K2"), col)
        r["H0a_lambda0_D0_K2"] = tost_pct(x0, cie, delta) if boot else tost(t_interval(x0, cie), delta)
        h0c = {}
        for name, (kind, c1, c2) in H0C.items():
            x1, x2 = cell(df, a, c1, col), cell(df, a, c2, col)
            if boot:  # same reps / by rep number: the bootstrap distribution of the difference
                t, kind = percentile_interval(x1 - x2, ci), "paired_by_rep"
            else:
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
            f"lambda{lam}|{d}": interval(cell(df, a, (lam, d, "K3"), col) - cell(df, a, (lam, d, "K2"), col), ci, boot)
            for lam in lams for d in sorted(df[df.attribute == a].D.unique())}
        means = [float(cell(df, a, (lam, "D0", "K2"), col).mean()) for lam in lams]
        r["exploratory_lambda_monotone_D0_K2"] = {"lambdas": lams, "means": means,
                                                  "non_decreasing": bool(np.all(np.diff(means) >= 0))}
        out[col] = r

    out["secondary_H2"] = {"contrast": "lambda 1, K2: D0 - D2 (paired by rep)", "decision": False, "estimators": {
        col: interval(cell(df, a, (1.0, "D0", "K2"), col) - cell(df, a, (1.0, "D2", "K2"), col), ci, boot)
        for col in SECONDARY_H2}}
    leak = interval(cell(df, a, (1.0, "D2", "K2"), PRIMARY), ci, boot)  # 判断 37: a leak is left after D2
    out["H2_residual_leak"] = leak | {"cell": "lambda 1, D2, K2, dV_bal", "delta_E2": delta, "supported": bool(leak["lo"] > delta),
                                      "in_holm": False, "rule": "supported = lower end of the interval > delta_E2 (secondary test)"}
    orig_path = exp / "results" / "e2_lambda1_original.csv"
    if orig_path.exists():
        out["descriptive_lambda1_original_sample"] = lambda1_original(orig_path, df, a, ci, boot)
    elif boot:  # C18: with bootstrap_persons the original-sample row is part of the registered output; a missing file is a failed run
        raise SystemExit(f"e2_summarise: bootstrap_persons is true but {orig_path} is missing (e2_map.py writes it; re-run the map)")
    else:
        out["descriptive_lambda1_original_sample"] = None

    if e1m is not None:
        # 判断 39: H1's p is the sign-flip permutation p of E1; the bootstrap "p_one_sided" is for reporting and is not read here
        p1 = (e1m.get("H1_K2_minus_K0") or {}).get("p_one_sided_permutation")
        if p1 is None or not 0 <= p1 <= 1:
            raise SystemExit(f"e2_summarise: E1 H1 p_one_sided_permutation missing or out of [0, 1]: {p1} "
                             "(re-summarise the E1 run with scripts/e1_summarise.py, 判断 39)")
        p2 = out[PRIMARY]["H2_D0_minus_D2_lambda1_K2"]["p_one_sided"]
        adj = holm({"H1_E1": p1, "H2_E2": p2})
        out["holm"] = {"raw": {"H1_E1": p1, "H2_E2": p2}, "adjusted": adj}
        out["e1_source"] = str(e1)
        out["e1_metrics_sha256"] = hashlib.sha256((e1 / "metrics.json").read_bytes()).hexdigest()  # the exact E1 metrics this decision used
        if "alpha_one_sided" not in cfg:
            raise SystemExit("e2_summarise: config has no alpha_one_sided (needed for the final H1 / H2 decision)")
        alpha = cfg["alpha_one_sided"]
        h1 = e1m.get("H1_K2_minus_K0") or {}
        h1_est, d_e1 = h1.get("est"), e1m.get("delta_E1")
        if h1_est is None or d_e1 is None or not (math.isfinite(h1_est) and math.isfinite(d_e1)):
            raise SystemExit("e2_summarise: E1 H1 estimate or delta_E1 missing")
        h2 = out[PRIMARY]["H2_D0_minus_D2_lambda1_K2"]
        out["alpha_one_sided"] = alpha
        out["H1_supported"] = bool(adj["H1_E1"] <= alpha)  # 判断 22: no "estimate > delta_E1" in the H1 decision
        out["H1_estimate_above_delta_E1"] = bool(h1_est > d_e1)  # descriptive: does the estimate exceed the resolution-based width delta_E1 (not a "meaningful minimum effect", 判断 37)
        out["H2_supported"] = bool(adj["H2_E2"] <= alpha and h2["est"] > delta)
        h0b = (e1m.get("H0b") or {}).get("supported")
        h0c_ok = out[PRIMARY]["H0c_E2_side"]["supported"]  # None = no contrast in the judgement
        h0d = (e1m["H0d"] or {}).get("supported") if "H0d" in e1m else "absent"  # 判断 44; "absent": E1 metrics without H0d
        out["H1_scope"] = ("full" if (h0b is True and h0c_ok is True and h0d is True)
                           else "E1_measurement_only")  # 判断 30・38・44
        e_k0n = (e1m.get("H1_control_K2_minus_K0n") or {}).get("est")
        e_k0 = h1_est
        k0n_ok = e_k0n is not None and math.isfinite(e_k0n)
        if e_k0 <= 0:  # K2 - K0 point estimate <= 0: the "half of the K0 effect" comparison has no meaning
            out["H1_K0n_length_note"] = None
        else:
            out["H1_K0n_length_note"] = bool(e_k0n < 0.5 * e_k0) if k0n_ok else True  # 判断 38: no control estimate -> note attached
        out["final_decision"] = {"rule": "H1: Holm-adjusted one-sided p <= alpha_one_sided; H2: the same and E2 estimate > "
                                         "delta_E2",
                                 "H1_supported": out["H1_supported"], "H2_supported": out["H2_supported"],
                                 "H1_scope": out["H1_scope"], "H0b_E1": h0b, "H0c_E2": h0c_ok, "H0d_E1": h0d,
                                 "H1_K0n_length_note": out["H1_K0n_length_note"], "H1_K0n_estimate_present": bool(k0n_ok)}
    else:
        out["final_decision"] = None
        out["final_decision_note"] = ("not made: no --e1; the H1 / H2 decision needs the Holm adjustment over "
                                      "E1's H1 and E2's H2. The interval flags are unadjusted.")

    desc = {}
    for (at, lam, d, k), g in df.groupby(["attribute", "lambda", "D", "K"]):
        g = g.sort_values("rep")
        desc[f"{at}|lam={lam}|{d}|{k}"] = {col: interval(g[col].to_numpy(), ci, boot) for col in ESTIMATORS} | {
            "cost_total": float(g.cost_total.mean()), "cost_extra": float(g.cost_extra.mean())}
    out["descriptive"] = desc
    out["descriptive_misclassification_cells"] = misclassification_cells(df, a, lams, delta, ci, boot=boot)
    out["descriptive_misclassification_cells_dV_acc"] = misclassification_cells(df, a, lams, delta, ci, "dV_acc", boot)
    return out


def main(exp: Path, e1: Path | None) -> None:
    out = summarise(exp, e1)
    (exp / "results").mkdir(exist_ok=True)
    (exp / "results" / "e2_tests.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    mpath = exp / "metrics.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    p = out[PRIMARY]
    m["tests"] = {"primary_estimator": PRIMARY, "interval_method": out["interval_method"], "H2": p["H2_D0_minus_D2_lambda1_K2"],
                  "H0a": p["H0a_lambda0_D0_K2"], "H0c_E2_side": p["H0c_E2_side"], "holm": out.get("holm"),
                  "final_decision": out["final_decision"],
                  "H2_residual_leak": out["H2_residual_leak"], "secondary_H2": out["secondary_H2"],
                  "descriptive_lambda1_original_sample": out["descriptive_lambda1_original_sample"],
                  "misclassification_cells_by_lambda": out["descriptive_misclassification_cells"]["by_lambda"]}
    mpath.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8", newline="\n")
    h2, h0a = p["H2_D0_minus_D2_lambda1_K2"], p["H0a_lambda0_D0_K2"]
    print("interval_method =", out["interval_method"])
    print(f"H2  D0-D2 = {h2['est']:.4f} [{h2['lo']:.4f}, {h2['hi']:.4f}] p={h2['p_one_sided']:.3g} "
          f"unadjusted_interval_criterion={h2['unadjusted_interval_criterion']}")
    print(f"H0a dV(lam0) = {h0a['est']:.4f} [{h0a['lo']:.4f}, {h0a['hi']:.4f}] equivalent={h0a['equivalent']}")
    for k, t in p["H0c_E2_side"].items():
        if isinstance(t, dict):
            print(f"H0c {k} = {t['est']:.4f} [{t['lo']:.4f}, {t['hi']:.4f}] in_judgement={t['in_judgement']}"
                  + (f" E1={t['E1_est']} agrees={t['sign_agrees']}" if "E1_est" in t else ""))
    if "holm" in out:
        print("Holm", out["holm"])
        print("FINAL H1_supported =", out["H1_supported"], " H2_supported =", out["H2_supported"],
              " H1_scope =", out["H1_scope"], " H1_K0n_length_note =", out["H1_K0n_length_note"])
    else:
        print("No final H1 / H2 decision: --e1 was not given (the interval flags above are unadjusted).")
    rl = out["H2_residual_leak"]
    print(f"H2_residual_leak (secondary) dV(lam1, D2, K2) = {rl['est']:.4f} [{rl['lo']:.4f}, {rl['hi']:.4f}] supported={rl['supported']}")
    for k, t in out["secondary_H2"]["estimators"].items():
        print(f"secondary H2 D0-D2 {k} = {t['est']:.4f} [{t['lo']:.4f}, {t['hi']:.4f}]")
    print("Misjudged cells (lower end > delta_E2) / negative floor per lambda (descriptive):",
          {k: (v["n_misjudged"], v["n_negative_floor"])
           for k, v in out["descriptive_misclassification_cells"]["by_lambda"].items()})


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        main(Path(args[0]), None)
    elif len(args) == 3 and args[1] == "--e1":
        main(Path(args[0]), Path(args[2]))
    else:
        raise SystemExit(__doc__.split("Usage: ")[1])
