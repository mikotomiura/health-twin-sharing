"""Confirmatory E1, step 3 (CPU): the preregistered estimands and intervals (docs/preregistration-draft.md v4: OSF items Indices, Statistical models, Inference criteria, Missing data).

Leak = net increment of the prior-equalised balanced accuracy V of the attacker's best guess (act = p > tau; p = 0.5 exactly
is NOT an action, 判断 26); when p is the true posterior, V itself (not dV, not a difference of dV) is a lower bound of the
g-vulnerability [70] under the equalised gain (the leakage line of [17][18]):
  V = 1/2 * TPR + 1/2 * TNR   (primary, prior-equalised; 判断 25; metrics "primary_estimator": "balanced")
  dV = V(t3, K + releases) - V(t0, K only)
The pi-weighted V = pi * TPR + (1 - pi) * TNR (pi = P(A) of the cohort population; 判断 8) is kept as a secondary beside
every primary contrast, under the key "pi_weighted" (same persons, same bootstrap draws).
A person's t3 value is the mean over its paraphrase x twin-seed reads (default paraphrases only, except H3); missing reads
(p = null, incl. "twin_missing", or rows never written) are left out of that mean.
Pairing (decisions 判断 12): a condition's value uses the persons with both a t3 value and a t0 read in that condition;
a contrast between two conditions (H1, H0c, K3, the H0b steps) uses only the persons usable in BOTH conditions.
Missingness (判断 13): per condition, the share of planned t3 reads that are missing and the share of persons not usable
(t3 or t0 missing); either above 10% is listed in "missing_over_10pct". Completeness against the plan of e1_run.py is
in "completeness"; duplicated keys are dropped.
Intervals: bootstrap over persons, stratified by A, the same draws for every statistic. One-sided bootstrap
p = (b + 1) / (B + 1), b = finite draws with the contrast <= 0 (判断 32). When a condition is in "missing_over_10pct",
"missing_sensitivity_H1" re-computes H1 with the missing reads (p = null and planned reads never written) filled as "no action" (p = 0) and as "action" (p = 1)
(判断 31; a sensitivity analysis, not a confirmatory result), and a third and fourth time as the fills adverse to H1
("fill_adverse": K2 t3 reads A -> 0, not-A -> 1, K2 t0 reads A -> 1, not-A -> 0, K0 the opposite, so the contrast is pushed down as far as
the fills can; "fill_adverse_by_condition": the literal 判断 38 wording, K2 reads A -> 0 / not-A -> 1 and K0 reads A -> 1 / not-A -> 0 at
every time point). The H1 decision is NOT made here: scripts/e2_summarise.py --e1 reads
this p and applies Holm with the E2 test of H2. Rule (判断 22): H1: Holm-adjusted p <= alpha_one_sided (0.025) (est > delta_E1
is descriptive, not part of the decision); H2: also est > delta_E2. "unadjusted_interval_criterion" (lower bound > 0 and estimate > delta) is shown for reading only.
H1's decision p is NOT the bootstrap p: it is "p_one_sided_permutation" under H1_K2_minus_K0 (判断 39), a sign-flip permutation test on the
per-person contributions c_i = [act_t3(K2) - act_t0(K2)] - [act_t3(K0) - act_t0(K0)] (signs flipped independently per person, n_flips = bootstrap
n draws, statistic 1/2 mean_A(c) - 1/2 mean_notA(c) = the H1 estimate, p = (b + 1) / (B + 1), b = flipped statistics >= observed); the bootstrap
"p_one_sided" stays beside it for reporting and is not used in any decision.
Secondaries fixed in advance (判断 35; beside H1, not in any decision, same persons and same bootstrap draws as the primary):
"dV_opt" = the net increment of the balanced V of the best rule (act iff p > pi, pi = P(A) of the cohort, key "pi") and "dAUC" = the
change of the AUC of the person-mean p (t3 vs t0), both for the contrast K2 - K0 (under H1_K2_minus_K0).
Secondary / descriptive: H1_secondary_K2_minus_K1 (lambda 1, D0; lower bound > 0 and estimate > delta; outside Holm;
判断 19), H1_control_K2_minus_K0n (K0n control, descriptive; 判断 20), H0a_lambda0_D0_K2_descriptive (H0a is judged in E2),
H3_K3_descriptive (H3 is judged on K2 only).
H0d (判断 44; config "h0d"): manipulation check of K on the t0 reads at lambda = h0d.lambda (no release, one read per person
and K): (a) K3 adds the person's D2 facts: c = p0(K3) - p0(K2), mean c of persons with any D2 display minus that of persons
with none, 95% lower bound > 0 (not judged, supported = None, when a group has < min_group persons); (b) K1 and K2 stay at
the stated overall rate (the integer % of the K1 sentence): 90% interval of mean(p0 - rate) inside +-margin_rate, for each.
supported = (a) and (b) for K1 and K2; None (not decided) when (a) is not judged or every K1 / K2 t0 read is missing.
The groups of (a) count the persons with both the K2 and the K3 t0 read. The bootstrap is the shared A-stratified one, so a draw
in which a group is empty gives NaN and is dropped (n_draws is reported). Read by scripts/e2_summarise.py --e1 for H1_scope.
H3 decides on the raw ratio (SD of paraphrase means / RMS seed SD): it still carries seed noise / sqrt(n_seeds), so with
no paraphrase effect it sits below 1 (conservative); the seed-corrected ratio is reported beside it.

Outputs <exp>/metrics.json (NaN written as null) and <exp>/results/conditions.csv.
Usage: python scripts/e1_summarise.py <exp-dir>
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from e1_run import Exp  # noqa: E402
from llm_dry_run_extra import UTIL_ITEMS, load_jsonl  # noqa: E402

ATTACK_COLS = ["key", "K", "pid", "A", "lambda", "t", "D", "paraphrase", "seed_idx", "p", "reason"]


def v_of(act_a: np.ndarray, act_n: np.ndarray, pi: float = 0.5) -> np.ndarray:
    """Bayes vulnerability from act rates of A persons and not-A persons (last axis = persons; NaN = not used).
    pi is the weight on TPR: 0.5 = balanced (primary, 判断 25); the cohort prior = pi-weighted (secondary)."""
    with warnings.catch_warnings():  # an empty group (every person missing) gives NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        return pi * np.nanmean(act_a, axis=-1) + (1 - pi) * (1 - np.nanmean(act_n, axis=-1))


def auc_of(pa: np.ndarray, pn: np.ndarray) -> np.ndarray:
    """AUC (Mann-Whitney, ties 1/2) of the scores pa (A persons) against pn (not-A) over the last axis; NaN = not used.
    Leading axes are draws. NaN when a group has no used person."""
    ga, gn = pa[..., :, None], pn[..., None, :]
    valid = ~np.isnan(ga) & ~np.isnan(gn)
    win = ((ga > gn) + 0.5 * (ga == gn)) * valid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return win.sum((-1, -2)) / valid.sum((-1, -2))


def interval(est: float, draws: np.ndarray, ci: float) -> dict:
    """Percentile interval over the finite draws; None values when there are none (e.g. every read missing).
    One-sided p = (b + 1) / (B + 1), b = draws <= 0 (paper-write 判断 32)."""
    d = draws[np.isfinite(draws)]
    if est is None or not np.isfinite(est) or len(d) == 0:
        return {"est": None, "lo": None, "hi": None, "ci": ci, "p_one_sided": None, "n_draws": int(len(d))}
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "ci": ci, "p_one_sided": float((np.sum(d <= 0) + 1) / (len(d) + 1)),
            "n_draws": int(len(d))}


def sign_flip_p(c: np.ndarray, nA: int, signs: np.ndarray, wA: float = 0.5, wN: float = 0.5) -> np.ndarray:
    """One-sided sign-flip permutation p (判断 39) of the statistic wA * mean(c[:nA]) - wN * mean(c[nA:]).
    c: (..., n) per-person contributions (A persons first, no NaN); signs: (B, n) matrix of +-1 shared by the leading axes
    (each row of c still gets its own exact p). p = (b + 1) / (B + 1), b = number of flipped statistics >= the observed one
    (a 1e-12 tolerance counts float ties as >=, the conservative side). NaN when a group is empty."""
    c = np.asarray(c, float)
    n, nN = c.shape[-1], c.shape[-1] - nA
    if nA < 1 or nN < 1:
        return np.full(c.shape[:-1], np.nan)
    obs = wA * c[..., :nA].mean(-1) - wN * c[..., nA:].mean(-1)
    flipped = wA * (signs[:, :nA] @ np.moveaxis(c[..., :nA], -1, 0).reshape(nA, -1)) / nA \
        - wN * (signs[:, nA:] @ np.moveaxis(c[..., nA:], -1, 0).reshape(nN, -1)) / nN  # B x rows
    b = (flipped >= obs.reshape(-1) - 1e-12).sum(0)
    return ((b + 1) / (signs.shape[0] + 1)).reshape(c.shape[:-1])


def one_sample_sign_flip_p(d: np.ndarray, signs: np.ndarray) -> float:
    """One-sided one-sample sign-flip permutation p of mean(d) (round 3 A10): flip each person's sign; p = (b + 1) / (B + 1),
    b = flipped means >= the observed mean (1e-12 tolerance, as sign_flip_p). NaN when d is empty."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan")
    b = ((signs @ d) / len(d) >= d.mean() - 1e-12).sum()
    return float((b + 1) / (signs.shape[0] + 1))


def ok(r: dict | None) -> bool:
    return r is not None and r.get("lo") is not None


def clean(o):
    """NaN / inf -> None, numpy scalars -> Python, recursively (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Data:
    def __init__(self, exp: Path):
        self.exp = exp
        self.cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
        self.seed = int((exp / "SEED").read_text().strip())
        self.pi = json.loads((exp / "inputs" / "knowledge.json").read_text(encoding="utf-8"))["pi"]
        recs = load_jsonl(exp / "inputs" / "records.jsonl")
        self.k3_any = {(r["pid"], r["lambda"]): any(r["k3_facts"].values()) for r in recs if r["D"] == "D0"}  # H0d (判断 44)
        self.k1_sentences = {lam: v["sentences"]["K1"] for lam, v in
                             json.loads((exp / "inputs" / "knowledge.json").read_text(encoding="utf-8"))["lambdas"].items()}
        self.people = (pd.DataFrame([{"pid": r["pid"], "A": r["A"], "a_type": r["a_type"]} for r in recs])
                       .drop_duplicates("pid").sort_values(["A", "pid"], ascending=[False, True]).reset_index(drop=True))
        self.plan = Exp(exp)
        self.dups = {}
        frames = {}
        for name in ("attack", "calib", "utility"):
            df = pd.DataFrame(load_jsonl(exp / "results" / f"{name}.jsonl"))
            if len(df):
                self.dups[name] = int(df.key.duplicated().sum())
                df = df.drop_duplicates("key", keep="first")
            frames[name] = df
        if not len(frames["attack"]):
            raise SystemExit(f"{exp}/results/attack.jsonl has no rows: nothing to summarise")
        self.at = frames["attack"].reindex(columns=sorted(set(ATTACK_COLS) | set(frames["attack"].columns)))
        self.cal, self.ut = frames["calib"], frames["utility"]
        self.tau = self.cfg["tau"]
        self.nd = self.cfg["n_paraphrases_default"]
        self.isA = self.people.A.to_numpy()
        self.nA = int(self.isA.sum())

    def _map(self, s: pd.Series) -> np.ndarray:
        return self.people.pid.map(s).to_numpy(float)

    def raw(self, lam, d, k, tau=None, paraphrases=None) -> dict:
        """Unpaired per-person values: act3 / p3 (mean over t3 reads), act0 / p0 (t0 read). NaN = missing."""
        tau = self.tau if tau is None else tau
        a = self.at
        t3 = a[(a.t == 3) & (a["lambda"] == lam) & (a.D == d) & (a.K == k)]
        t3 = t3[t3.paraphrase < self.nd] if paraphrases is None else t3[t3.paraphrase.isin(paraphrases)]
        t0 = a[(a.t == 0) & (a["lambda"] == lam) & (a.K == k)]
        act = lambda df: df.assign(act=(df.p > tau).where(df.p.notna())).groupby("pid").act.mean()  # noqa: E731
        return {"act3": self._map(act(t3)), "act0": self._map(act(t0)),
                "p3": self._map(t3.groupby("pid").p.mean()), "p0": self._map(t0.groupby("pid").p.mean())}

    def boot_index(self, n: int) -> np.ndarray:
        """n x persons index matrix, stratified by A (A persons first, as in self.people)."""
        rng = np.random.default_rng([self.seed, 0xB00])
        ia, iN = np.flatnonzero(self.isA), np.flatnonzero(~self.isA)
        return np.hstack([rng.choice(ia, (n, len(ia))), rng.choice(iN, (n, len(iN)))])


def usable(r: dict) -> np.ndarray:
    return ~np.isnan(r["act3"]) & ~np.isnan(r["act0"])


def summarise(exp: Path) -> dict:
    x = Data(exp)
    bcfg = x.cfg["bootstrap"]
    B = x.boot_index(bcfg["n"])
    nA, delta, pi = x.nA, x.cfg["delta_E1"], x.pi

    def dv(r: dict, mask: np.ndarray, w: float = 0.5) -> tuple[float, np.ndarray]:
        """w = weight on TPR: 0.5 primary (balanced), pi secondary (pi-weighted)."""
        a3, a0 = np.where(mask, r["act3"], np.nan), np.where(mask, r["act0"], np.nan)
        est = float(v_of(a3[:nA], a3[nA:], w) - v_of(a0[:nA], a0[nA:], w))
        b3, b0 = a3[B], a0[B]
        draws = v_of(b3[:, :nA], b3[:, nA:], w) - v_of(b0[:, :nA], b0[:, nA:], w)
        return est, draws

    def ledger(r: dict, mask: np.ndarray) -> tuple[float, np.ndarray]:
        led = np.where(mask, r["p3"] - r["p0"], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean(led[:nA])), np.nanmean(led[B][:, :nA], axis=1)

    # ---- plan: conditions, expected reads, completeness
    plan_t3 = [j for j in x.plan.attack_jobs() if j[0] == "t3"]
    conds = sorted({(c[1], c[2], j[1]) for j in plan_t3 for c in [j[2]] if c[3] < x.nd}, key=str)
    n_expected = {}
    for j in plan_t3:
        c = j[2]
        if c[3] < x.nd:
            n_expected[(c[1], c[2], j[1])] = n_expected.get((c[1], c[2], j[1]), 0) + 1
    exp_keys = {"attack": {x.plan.attack_key(j) for j in x.plan.attack_jobs()},
                "calib": {x.plan.calib_key(q, cell) for q, cell in x.plan.calib_jobs()},
                "utility": {f"{x.plan.ckey(cell)}|t{t}" for cell in x.plan.utility_cells()  # as in e1_run.stage_utility
                            for t in range(1, len(x.cfg["years"]) + 1)}}
    comp = {}
    for name, df in (("attack", x.at), ("calib", x.cal), ("utility", x.ut)):
        have = set(df.key) if len(df) else set()
        vcol = "items" if name == "utility" else "p"  # a utility row is valid when the judge's items are there
        valid = set(df.key[df[vcol].notna()]) if len(df) else set()
        comp[name] = {"expected": len(exp_keys[name]), "rows": len(have & exp_keys[name]),
                      "valid": len(valid & exp_keys[name]), "unexpected_rows": len(have - exp_keys[name]),
                      "twin_missing": int((df["reason"] == "twin_missing").sum()) if len(df) and "reason" in df else 0,
                      "duplicates_dropped": x.dups.get(name, 0)}
    t0 = x.at[x.at.t == 0]
    comp["attack_t0_missing_by_lambda_K"] = {f"lambda{l}|{k}": int(g.p.isna().sum()) for (l, k), g in t0.groupby(["lambda", "K"])}
    comp["complete"] = all(v["rows"] == v["expected"] and v["unexpected_rows"] == 0
                           for v in comp.values() if isinstance(v, dict) and "expected" in v)

    # ---- every planned condition
    R = {c: x.raw(*c) for c in conds}
    Ropt = {c: x.raw(*c, tau=pi) for c in conds}  # 判断 35: the best rule acts when p > pi (P(A) of the cohort)
    table = []
    for c in conds:
        lam, d, k = c
        r, u = R[c], usable(R[c])
        est, draws = dv(r, u)
        led, _ = ledger(r, u)
        est_pi, _ = dv(r, u, pi)
        sub = x.at[(x.at.t == 3) & (x.at["lambda"] == lam) & (x.at.D == d) & (x.at.K == k) & (x.at.paraphrase < x.nd)]
        n_exp = n_expected.get(c, 0)
        row = {"lambda": lam, "D": d, "K": k, **{f"dV_{key}": v for key, v in interval(est, draws, bcfg["ci"]).items()},
               "dV_pi_weighted_est": est_pi, "ledger_A": led, "n_expected": n_exp, "n_reads": int(len(sub)),
               "n_missing": int(n_exp - sub.p.notna().sum()), "n_persons_used": int(u.sum())}
        row["missing_share"] = row["n_missing"] / n_exp if n_exp else None
        row["persons_missing_share"] = 1 - row["n_persons_used"] / len(x.people)
        for tau in x.cfg["taus_secondary"]:
            rt = x.raw(lam, d, k, tau=tau)
            a3 = np.where(u, rt["act3"], np.nan)
            row[f"act_A_tau{tau}"] = float(np.nanmean(a3[:nA])) if u[:nA].any() else None
            row[f"act_notA_tau{tau}"] = float(np.nanmean(a3[nA:])) if u[nA:].any() else None
        table.append(row)
    df = pd.DataFrame(table)

    def auc_change(r: dict, mask: np.ndarray) -> tuple[float, np.ndarray]:
        """AUC of the person-mean p at t3 minus AUC of the t0 p (A persons vs not-A), on the persons in mask."""
        p3, p0 = np.where(mask, r["p3"], np.nan), np.where(mask, r["p0"], np.nan)
        est = float(auc_of(p3[:nA], p3[nA:]) - auc_of(p0[:nA], p0[nA:]))
        draws = []
        for s0 in range(0, len(B), 500):  # chunked: the pairwise comparison is draws x nA x nN
            b3, b0 = p3[B[s0:s0 + 500]], p0[B[s0:s0 + 500]]
            draws.append(auc_of(b3[:, :nA], b3[:, nA:]) - auc_of(b0[:, :nA], b0[:, nA:]))
        return est, np.concatenate(draws)

    def permutation_p(r1: dict, r2: dict, both: np.ndarray) -> dict:
        """c_i = [act3(c1) - act0(c1)] - [act3(c2) - act0(c2)] on the persons in `both`; own stream (seed, 0x5F1), independent of the bootstrap.
        The stream is shared on purpose by H1, its secondary K2 - K1 and the fills of the missing-data sensitivity (each p is exact; the
        p's are correlated with one another, which only matters for the descriptive ones)."""
        ci_ = ((r1["act3"] - r1["act0"]) - (r2["act3"] - r2["act0"]))[both]
        nA_b = int(both[:nA].sum())
        n_flips = int(bcfg["n"])
        signs = np.random.default_rng([x.seed, 0x5F1]).choice(np.array([-1.0, 1.0]), (n_flips, len(ci_)))
        p = float(sign_flip_p(ci_, nA_b, signs)) if len(ci_) else float("nan")
        return {"p_one_sided_permutation": p, "test": "sign-flip permutation", "n_flips": n_flips}

    def step_permutation_p(r1: dict, r2: dict, both: np.ndarray) -> dict:
        """Round 3 A10: one-sided one-sample sign-flip p of the H0b step (ledger difference on the A persons in `both`), reported beside
        the bootstrap interval rule (the `supported` flag is unchanged). Statistic = mean over A persons of
        (p3 - p0)(q2) - (p3 - p0)(q1) (the ledger's per-person terms; see `ledger`), signs flipped per person, own stream (seed, 0x5F2)."""
        d_i = (((r2["p3"] - r2["p0"]) - (r1["p3"] - r1["p0"])))[:nA][both[:nA]]
        n_flips = int(bcfg["n"])
        signs = np.random.default_rng([x.seed, 0x5F2]).choice(np.array([-1.0, 1.0]), (n_flips, len(d_i)))
        return {"p_one_sided_permutation": one_sample_sign_flip_p(d_i, signs), "test": "one-sample sign-flip permutation",
                "n_flips": n_flips}

    def contrast(c1, c2, ci=bcfg["ci"], secondary=True, extra=False, perm=False) -> dict | None:
        """dV(c1) - dV(c2) on the persons usable in both conditions (判断 12). Primary = balanced V; the pi-weighted
        version (same persons and draws) sits beside it under "pi_weighted"."""
        if c1 not in R or c2 not in R:
            return None
        both = usable(R[c1]) & usable(R[c2])
        e1, d1 = dv(R[c1], both)
        e2, d2 = dv(R[c2], both)
        out = interval(e1 - e2, d1 - d2, ci) | {"n_persons": int(both.sum())}
        if secondary:
            p1, q1 = dv(R[c1], both, pi)
            p2, q2 = dv(R[c2], both, pi)
            out["pi_weighted"] = interval(p1 - p2, q1 - q2, ci) | {"n_persons": int(both.sum())}
        if perm:  # 判断 39: the decision p of H1, a sign-flip permutation test on the per-person contributions
            out |= permutation_p(R[c1], R[c2], both)
        if extra:  # 判断 35: the best rule (act iff p > pi) and the AUC change, same persons and same draws
            o1, o2 = (dv(Ropt[c], both) for c in (c1, c2))
            out["dV_opt"] = interval(o1[0] - o2[0], o1[1] - o2[1], ci) | {"n_persons": int(both.sum()), "rule": "act iff p > pi",
                                                                            "pi": pi}
            a1, a2 = auc_change(R[c1], both), auc_change(R[c2], both)
            out["dAUC"] = interval(a1[0] - a2[0], a1[1] - a2[1], ci) | {"n_persons": int(both.sum())}
        return out

    m: dict = {"primary_estimator": "balanced", "pi": pi, "tau": x.tau, "delta_E1": delta, "n_persons": int(len(x.people)), "n_A": nA,
               "completeness": comp}
    h1 = contrast((1, "D0", "K2"), (1, "D0", "K0"), extra=True, perm=True)
    # H1 is decided by scripts/e2_summarise.py --e1 (Holm with H2: adjusted one-sided p <= alpha, 判断 22; est > delta_E1 is descriptive).
    # The flag below is the unadjusted interval rule only; it is not the decision.
    m["H1_K2_minus_K0"] = h1 and h1 | {
        "unadjusted_interval_criterion": bool(ok(h1) and h1["lo"] > 0 and h1["est"] > delta)}
    h1s = contrast((1, "D0", "K2"), (1, "D0", "K1"), perm=True)  # secondary contrast (判断 19); not in the Holm family; permutation p beside it (round 3 A10)
    m["H1_secondary_K2_minus_K1"] = h1s and h1s | {"supported": bool(ok(h1s) and h1s["lo"] > 0 and h1s["est"] > delta)}
    m["H1_control_K2_minus_K0n"] = contrast((1, "D0", "K2"), (1, "D0", "K0n"))  # descriptive (判断 20)
    m["H2_secondary_D0_minus_D2"] = contrast((1, "D0", "K2"), (1, "D2", "K2"))
    if (0, "D0", "K2") in R:
        r = R[(0, "D0", "K2")]
        h0a = interval(*dv(r, usable(r)), bcfg["ci_equivalence"])
        h0a_pi = interval(*dv(r, usable(r), pi), bcfg["ci_equivalence"])
        # H0a is judged in E2 (attack-me round 1 の A9); this E1 value is descriptive (delta_E1)
        m["H0a_lambda0_D0_K2_descriptive"] = h0a | {
            "within_delta_E1": bool(ok(h0a) and -delta < h0a["lo"] and h0a["hi"] < delta),
            "pi_weighted": h0a_pi | {"within_delta_E1": bool(ok(h0a_pi) and -delta < h0a_pi["lo"] and h0a_pi["hi"] < delta)}}
    m["H0c_E1_side"] = {"lambda1_minus_lambda0_D0_K2": contrast((1, "D0", "K2"), (0, "D0", "K2")),
                        "D0_minus_D2_lambda1_K2": contrast((1, "D0", "K2"), (1, "D2", "K2")),
                        "D0_minus_D1_lambda1_K2": contrast((1, "D0", "K2"), (1, "D1", "K2"))}
    m["K3_minus_K2_descriptive"] = {f"lambda{lam}|{d}": contrast((lam, d, "K3"), (lam, d, "K2"))
                                    for lam, d, k in conds if k == "K3"}

    # ---- H0b: calibration. Labelled persons (from the plan) take their calib reads; the others the attack reads.
    c = x.cfg["calibration"]
    if x.plan.calib_jobs():
        base = x.raw(c["lambda"], c["D"], c["K"])  # unpaired: t0 is taken as read, whatever happened at t3
        Rq, miss = {}, {}
        for q in c["q"]:
            cq = x.cal[(x.cal.q == q) & (x.cal.paraphrase < x.nd)] if len(x.cal) else pd.DataFrame(columns=["pid", "p"])
            lab = x.people.pid.isin(x.plan.calib_people(q)).to_numpy()
            pq = x._map(cq.groupby("pid").p.mean())
            aq = x._map(cq.assign(act=(cq.p > x.tau).where(cq.p.notna())).groupby("pid").act.mean())
            miss[str(q)] = int(np.sum(lab & np.isnan(pq)))
            Rq[q] = {"p3": np.where(lab, pq, base["p3"]), "act3": np.where(lab, aq, base["act3"]),
                     "p0": base["p0"], "act0": base["act0"]}
        qs = sorted(c["q"])
        h = {"labelled_persons_missing_by_q": miss, "ledger_A_by_q": {}, "dV_by_q": {}, "dV_pi_weighted_by_q": {},
             "n_persons_by_q": {}}
        for q in qs:
            u = usable(Rq[q])
            h["ledger_A_by_q"][str(q)] = ledger(Rq[q], u)[0]
            h["dV_by_q"][str(q)] = dv(Rq[q], u)[0]
            h["dV_pi_weighted_by_q"][str(q)] = dv(Rq[q], u, pi)[0]
            h["n_persons_by_q"][str(q)] = int(u[:nA].sum())
        h["steps"] = {}
        for q1, q2 in zip(qs, qs[1:]):
            both = usable(Rq[q1]) & usable(Rq[q2])
            (l1, b1), (l2, b2) = ledger(Rq[q1], both), ledger(Rq[q2], both)
            h["steps"][f"{q2}-{q1}"] = interval(l2 - l1, b2 - b1, bcfg["ci"]) | {"n_A_persons": int(both[:nA].sum())} | step_permutation_p(
                Rq[q1], Rq[q2], both)
        h["supported"] = all(ok(s) and s["lo"] > 0 for s in h["steps"].values())
        m["H0b"] = h

    # ---- H0d: manipulation check of the recipient's knowledge K on the t0 reads (no release; 判断 44)
    hd = x.cfg.get("h0d")
    if hd:
        lam = int(hd["lambda"])
        if lam not in x.cfg["lambdas"]:
            raise SystemExit(f"e1_summarise: h0d.lambda {lam} is not one of the lambdas {x.cfg['lambdas']}")
        p0 = {k: x.raw(lam, "D0", k)["p0"] for k in ("K0", "K0n", "K1", "K2", "K3")}
        pos = x.people.pid.map(lambda pid: x.k3_any[(pid, lam)]).to_numpy(bool)
        c3 = p0["K3"] - p0["K2"]  # K2 carries the same age and sex: the difference keeps what the K3 facts add
        used = ~np.isnan(c3)
        n_pos, n_neg = int(np.sum(used & pos)), int(np.sum(used & ~pos))

        def gap(cv: np.ndarray, pv: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return (np.nanmean(np.where(pv, cv, np.nan), axis=-1) - np.nanmean(np.where(~pv, cv, np.nan), axis=-1))

        k3 = interval(float(gap(c3, pos)), gap(c3[B], pos[B]), bcfg["ci"]) | {"n_persons_with_D2": n_pos,
                                                                              "n_persons_without_D2": n_neg}
        k3["judged"] = bool(n_pos >= hd["min_group"] and n_neg >= hd["min_group"])
        k3["supported"] = (bool(ok(k3) and k3["lo"] > 0) if k3["judged"] else None)
        found = re.findall(r"(\d+(?:\.\d+)?)%", x.k1_sentences[str(lam)])  # e1_build.pct: "16%" or "0.5%"
        if len(found) != 1:
            raise SystemExit(f"e1_summarise: H0d needs exactly one rate in the K1 sentence, found {found}")
        rate = float(found[0]) / 100
        rates = {}
        for k in ("K1", "K2"):
            dk = p0[k] - rate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = interval(float(np.nanmean(dk)), np.nanmean(dk[B], axis=1), bcfg["ci_equivalence"])
            r["n_persons"] = int(np.sum(~np.isnan(dk)))
            r["within_margin"] = (bool(-hd["margin_rate"] < r["lo"] and r["hi"] < hd["margin_rate"]) if ok(r)
                                  else None)  # every read missing: not decided
            rates[k] = r
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_p0 = {k: float(np.nanmean(v)) for k, v in p0.items()}
        parts = [k3["supported"], rates["K1"]["within_margin"], rates["K2"]["within_margin"]]
        m["H0d"] = {"lambda": lam, "stated_rate": rate, "margin_rate": hd["margin_rate"],
                    "K3_minus_K2_D2_gap": k3, "K1_minus_rate": rates["K1"], "K2_minus_rate": rates["K2"],
                    "supported": (None if any(v is None for v in parts) else bool(all(parts))),
                    "descriptive": {"mean_p0": mean_p0,
                                    "auc_p0_K0": float(auc_of(p0["K0"][:nA], p0["K0"][nA:])),
                                    "t0_missing": {k: int(np.sum(np.isnan(v))) for k, v in p0.items()}}}

    # ---- H3 and the repetition noise at the base point (p3 of each read; p0 is constant within a person)
    b = x.cfg["base_point"]
    rms = lambda v, axis=None: np.sqrt(np.nanmean(v ** 2, axis=axis))  # noqa: E731
    for kk in b["K"]:
        a = x.at[(x.at.t == 3) & (x.at["lambda"] == b["lambda"]) & (x.at.D == b["D"]) & (x.at.K == kk) & x.at.p.notna()]
        if a.paraphrase.nunique() < 2:
            continue
        g = a.groupby(["pid", "paraphrase"]).p
        means, var_within = g.mean().unstack(), g.var(ddof=1).unstack()
        order = x.people.pid
        bw = means.std(axis=1, ddof=1).reindex(order).to_numpy(float)
        wi = np.sqrt(var_within.mean(axis=1)).reindex(order).to_numpy(float)
        bc = np.sqrt(np.clip(means.var(axis=1, ddof=1) - var_within.mean(axis=1) / x.cfg["n_seeds"], 0, None)
                     ).reindex(order).to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ratio, ratio_d = rms(bw) / rms(wi), rms(bw[B], 1) / rms(wi[B], 1)
            ratio_c, ratio_cd = rms(bc) / rms(wi), rms(bc[B], 1) / rms(wi[B], 1)
        r1 = interval(ratio, np.where(np.isfinite(ratio_d), ratio_d, np.nan), 0.95)
        r2 = interval(ratio_c, np.where(np.isfinite(ratio_cd), ratio_cd, np.nan), 0.95)
        h3 = {"sd_between_paraphrase": rms(bw), "sd_within_seed": rms(wi),
              "ratio": {k_: r1[k_] for k_ in ("est", "lo", "hi")},
              "ratio_seed_corrected": {k_: r2[k_] for k_ in ("est", "lo", "hi")},
              "n_paraphrases": int(a.paraphrase.nunique())}
        flag = bool(ok(r1) and r1["lo"] > 1)
        if kk == "K2":  # H3 is judged on K2 only (attack-me round 1 の A5); the K3 result is descriptive
            m["H3_K2"] = h3 | {"supported": flag}
        else:
            m[f"H3_{kk}_descriptive"] = h3 | {"ratio_lo_above_1": flag}

    # ---- repetition noise per condition (p3 SD across twin seeds within paraphrase, RMS over persons and paraphrases)
    t3 = x.at[(x.at.t == 3) & (x.at.paraphrase < x.nd) & x.at.p.notna()]
    rep = t3.groupby(["lambda", "D", "K", "pid", "paraphrase"]).p.var(ddof=1).groupby(["lambda", "D", "K"]).mean()
    m["repetition_sd"] = {f"lambda{l}|{d}|{k}": float(np.sqrt(v)) for (l, d, k), v in rep.items()}

    # ---- exploratory: TPR by A type (persons usable in the condition)
    types = x.people.a_type.to_numpy()
    m["exploratory_tpr_by_a_type"] = {
        f"lambda{lam}|{d}|{k}": {t: (float(np.nanmean(np.where(usable(R[(lam, d, k)]), R[(lam, d, k)]["act3"], np.nan)[types == t]))
                                     if (usable(R[(lam, d, k)]) & (types == t)).any() else None)
                                 for t in ("old", "misuses_only")} for lam, d, k in conds}

    if len(x.ut):
        u = x.ut[x.ut["items"].notna()].copy()
        u["score"] = u["items"].map(lambda d: float(np.mean([d[k] for k in UTIL_ITEMS])))
        m["utility_by_D"] = u[u.paraphrase < x.nd].groupby("D").score.mean().round(4).to_dict()
        m["utility_by_paraphrase_base"] = (u[(u["lambda"] == b["lambda"]) & (u.D == b["D"])]
                                           .groupby("paraphrase").score.mean().round(4).to_dict())
        m["utility_missing"] = int(x.ut["items"].isna().sum())
    over = df[(df.missing_share.fillna(1) > 0.10) | (df.persons_missing_share > 0.10)]
    m["missing_over_10pct"] = over[["lambda", "D", "K", "missing_share", "persons_missing_share"]].to_dict(orient="records")
    if m["missing_over_10pct"]:  # 判断 31: sensitivity of H1 to the missing reads (not a confirmatory result)
        orig, sens = x.at, {}
        # planned H1 reads that were never written (e.g. a stopped run) join the p = null rows before filling
        have, absent = set(orig.key), []
        for j in x.plan.attack_jobs():
            if j[1] not in ("K2", "K0"):
                continue
            if j[0] == "t0":
                if j[3] != 1:
                    continue
                row = {"K": j[1], "pid": j[2], "lambda": j[3], "t": 0}
            else:
                pid, lam, d, k, s_idx = j[2]
                if lam != 1 or d != "D0" or k >= x.nd:
                    continue
                row = {"K": j[1], "pid": pid, "lambda": lam, "t": 3, "D": d, "paraphrase": k, "seed_idx": s_idx}
            key = x.plan.attack_key(j)
            if key not in have:
                absent.append(row | {"key": key, "p": np.nan, "reason": "never_written"})
        full = pd.concat([orig, pd.DataFrame(absent)], ignore_index=True) if absent else orig
        try:
            isA = full.pid.map(dict(zip(x.people.pid, x.people.A))).astype(bool).to_numpy()
            k2, t3 = (full.K == "K2").to_numpy(), (full.t == 3).to_numpy()
            # 判断 38: fills adverse to H1 = dV(K2) - dV(K0); dV = V(t3) - V(t0). V rises with act of A and falls with act of
            # not-A: K2 t3 low for A / high for not-A, K2 t0 the opposite; K0 the other way round at both time points.
            adverse = np.where(k2 == t3, np.where(isA, 0.0, 1.0), np.where(isA, 1.0, 0.0))
            by_cond = np.where(k2, np.where(isA, 0.0, 1.0), np.where(isA, 1.0, 0.0))  # the literal wording, every time point
            for name, fill in (("fill_no_action", 0.0), ("fill_action", 1.0),
                               ("fill_adverse", pd.Series(adverse, index=full.index)),
                               ("fill_adverse_by_condition", pd.Series(by_cond, index=full.index))):
                x.at = full.assign(p=full.p.fillna(fill))
                Rf = {c: x.raw(*c) for c in ((1, "D0", "K2"), (1, "D0", "K0"))}
                both = usable(Rf[(1, "D0", "K2")]) & usable(Rf[(1, "D0", "K0")])
                e1, d1 = dv(Rf[(1, "D0", "K2")], both)
                e2, d2 = dv(Rf[(1, "D0", "K0")], both)
                # attack-me round 3 A1: the decision statistic of H1 is the sign-flip permutation p (判断 39), so the
                # sensitivity reports it per fill (same persons, same stream); the bootstrap p_one_sided is secondary.
                sens[name] = interval(e1 - e2, d1 - d2, bcfg["ci"]) | {"n_persons": int(both.sum())} | permutation_p(
                    Rf[(1, "D0", "K2")], Rf[(1, "D0", "K0")], both)
        finally:
            x.at = orig
        h1 = full[(full["lambda"] == 1) & full.K.isin(["K2", "K0"])
                  & ((full.t == 0) | ((full.D == "D0") & (full.paraphrase < x.nd)))]  # the reads H1 uses
        n_trunc = None
        if "done_reason" in h1:
            cut = h1.done_reason == "length"  # rows without attempt_reasons (older runs): the last attempt's done_reason
            if "attempt_reasons" in h1:  # any attempt cut at num_predict or over the context window counts
                has = h1.attempt_reasons.apply(lambda v: isinstance(v, list))
                anycut = h1.attempt_reasons.apply(lambda v: isinstance(v, list) and any(r in ("length", "ctx_overflow") for r in v))
                cut = anycut.where(has, cut)
            n_trunc = int((cut & h1.p.isna()).sum())
        m["missing_sensitivity_H1"] = sens | {"n_missing_reads": int(h1.p.isna().sum()), "n_never_written": len(absent),
                                              "n_missing_truncated": n_trunc,
                                              "n_missing_twin_missing": int((h1["reason"] == "twin_missing").sum()) if "reason" in h1 else 0,
                                              "note": "sensitivity analysis (判断 31), not a confirmatory result"}
    m["final_decision"] = ("not made here: the H1 / H2 decision is made by scripts/e2_summarise.py --e1 (判断 22: H1 = "
                           "Holm-adjusted one-sided p <= alpha_one_sided; est > delta_E1 is descriptive. H2: also est > "
                           "delta_E2)")
    m["alpha_one_sided"] = x.cfg.get("alpha_one_sided")
    m["note"] = "confirmatory estimands (preregistration); values from a fake backend or a dev dry run are not results"
    (exp / "results").mkdir(exist_ok=True)
    df.to_csv(exp / "results" / "conditions.csv", index=False, lineterminator="\n")
    m = clean(m)
    (exp / "metrics.json").write_text(json.dumps(m, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return m


if __name__ == "__main__":
    print(json.dumps(summarise(Path(sys.argv[1])), indent=1)[:4000])
