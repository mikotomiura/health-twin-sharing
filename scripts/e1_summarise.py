"""Confirmatory E1, step 3 (CPU): the preregistered estimands and intervals (docs/preregistration-draft.md v1 §5, §6, §8).

Leak = net increment of the Bayes vulnerability [17][18] of the attacker's best guess (act = p >= tau):
  V = pi * TPR + (1 - pi) * TNR   (pi = P(A) of the cohort population; the E1 sample is balanced, so it is re-weighted)
  dV = V(t3, K + releases) - V(t0, K only)                     (.steering/20261112-paper-write/decisions.md 判断 8)
A person's t3 value is the mean over its paraphrase x twin-seed reads (default paraphrases only, except H3); missing reads
(p = null, incl. "twin_missing", or rows never written) are left out of that mean.
Pairing (decisions 判断 12): a condition's value uses the persons with both a t3 value and a t0 read in that condition;
a contrast between two conditions (H1, H0c, K3, the H0b steps) uses only the persons usable in BOTH conditions.
Missingness (判断 13): per condition, the share of planned t3 reads that are missing and the share of persons not usable
(t3 or t0 missing); either above 10% is listed in "missing_over_10pct". Completeness against the plan of e1_run.py is
in "completeness"; duplicated keys are dropped.
Intervals: bootstrap over persons, stratified by A, the same draws for every statistic. One-sided bootstrap
p = share of finite draws with the contrast <= 0 (Holm with the E2 test of H2 is applied in the paper, not here).
H3 decides on the raw ratio (SD of paraphrase means / RMS seed SD): it still carries seed noise / sqrt(n_seeds), so with
no paraphrase effect it sits below 1 (conservative); the seed-corrected ratio is reported beside it.

Outputs <exp>/metrics.json (NaN written as null) and <exp>/results/conditions.csv.
Usage: python scripts/e1_summarise.py <exp-dir>
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from e1_run import Exp  # noqa: E402
from llm_dry_run_extra import UTIL_ITEMS, load_jsonl  # noqa: E402

ATTACK_COLS = ["key", "K", "pid", "A", "lambda", "t", "D", "paraphrase", "seed_idx", "p", "reason"]


def v_of(act_a: np.ndarray, act_n: np.ndarray, pi: float) -> np.ndarray:
    """Bayes vulnerability from act rates of A persons and not-A persons (last axis = persons; NaN = not used)."""
    with warnings.catch_warnings():  # an empty group (every person missing) gives NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        return pi * np.nanmean(act_a, axis=-1) + (1 - pi) * (1 - np.nanmean(act_n, axis=-1))


def interval(est: float, draws: np.ndarray, ci: float) -> dict:
    """Percentile interval over the finite draws; None values when there are none (e.g. every read missing)."""
    d = draws[np.isfinite(draws)]
    if est is None or not np.isfinite(est) or len(d) == 0:
        return {"est": None, "lo": None, "hi": None, "ci": ci, "p_one_sided": None, "n_draws": int(len(d))}
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "ci": ci, "p_one_sided": float(np.mean(d <= 0)),
            "n_draws": int(len(d))}


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
        act = lambda df: df.assign(act=(df.p >= tau).where(df.p.notna())).groupby("pid").act.mean()  # noqa: E731
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

    def dv(r: dict, mask: np.ndarray) -> tuple[float, np.ndarray]:
        a3, a0 = np.where(mask, r["act3"], np.nan), np.where(mask, r["act0"], np.nan)
        est = float(v_of(a3[:nA], a3[nA:], pi) - v_of(a0[:nA], a0[nA:], pi))
        b3, b0 = a3[B], a0[B]
        draws = v_of(b3[:, :nA], b3[:, nA:], pi) - v_of(b0[:, :nA], b0[:, nA:], pi)
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
                "calib": {x.plan.calib_key(q, cell) for q, cell in x.plan.calib_jobs()}}
    comp = {}
    for name, df in (("attack", x.at), ("calib", x.cal)):
        have = set(df.key) if len(df) else set()
        valid = set(df.key[df.p.notna()]) if len(df) else set()
        comp[name] = {"expected": len(exp_keys[name]), "rows": len(have & exp_keys[name]),
                      "valid": len(valid & exp_keys[name]), "unexpected_rows": len(have - exp_keys[name]),
                      "twin_missing": int((df["reason"] == "twin_missing").sum()) if len(df) and "reason" in df else 0,
                      "duplicates_dropped": x.dups.get(name, 0)}
    t0 = x.at[x.at.t == 0]
    comp["attack_t0_missing_by_lambda_K"] = {f"lambda{l}|{k}": int(g.p.isna().sum()) for (l, k), g in t0.groupby(["lambda", "K"])}
    comp["complete"] = all(v["rows"] == v["expected"] for v in comp.values() if isinstance(v, dict) and "expected" in v)

    # ---- every planned condition
    R = {c: x.raw(*c) for c in conds}
    table = []
    for c in conds:
        lam, d, k = c
        r, u = R[c], usable(R[c])
        est, draws = dv(r, u)
        led, _ = ledger(r, u)
        sub = x.at[(x.at.t == 3) & (x.at["lambda"] == lam) & (x.at.D == d) & (x.at.K == k) & (x.at.paraphrase < x.nd)]
        n_exp = n_expected.get(c, 0)
        row = {"lambda": lam, "D": d, "K": k, **{f"dV_{key}": v for key, v in interval(est, draws, bcfg["ci"]).items()},
               "ledger_A": led, "n_expected": n_exp, "n_reads": int(len(sub)),
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

    def contrast(c1, c2, ci=bcfg["ci"]) -> dict | None:
        """dV(c1) - dV(c2) on the persons usable in both conditions (判断 12)."""
        if c1 not in R or c2 not in R:
            return None
        both = usable(R[c1]) & usable(R[c2])
        e1, d1 = dv(R[c1], both)
        e2, d2 = dv(R[c2], both)
        return interval(e1 - e2, d1 - d2, ci) | {"n_persons": int(both.sum())}

    m: dict = {"pi": pi, "tau": x.tau, "delta_E1": delta, "n_persons": int(len(x.people)), "n_A": nA,
               "completeness": comp}
    h1 = contrast((1, "D0", "K2"), (1, "D0", "K0"))
    m["H1_K2_minus_K0"] = h1 and h1 | {"supported": bool(ok(h1) and h1["lo"] > 0 and h1["est"] > delta)}
    m["H2_secondary_D0_minus_D2"] = contrast((1, "D0", "K2"), (1, "D2", "K2"))
    if (0, "D0", "K2") in R:
        r = R[(0, "D0", "K2")]
        h0a = interval(*dv(r, usable(r)), bcfg["ci_equivalence"])
        m["H0a_lambda0_D0_K2"] = h0a | {"equivalent": bool(ok(h0a) and -delta < h0a["lo"] and h0a["hi"] < delta)}
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
            aq = x._map(cq.assign(act=(cq.p >= x.tau).where(cq.p.notna())).groupby("pid").act.mean())
            miss[str(q)] = int(np.sum(lab & np.isnan(pq)))
            Rq[q] = {"p3": np.where(lab, pq, base["p3"]), "act3": np.where(lab, aq, base["act3"]),
                     "p0": base["p0"], "act0": base["act0"]}
        qs = sorted(c["q"])
        h = {"labelled_persons_missing_by_q": miss, "ledger_A_by_q": {}, "dV_by_q": {}, "n_persons_by_q": {}}
        for q in qs:
            u = usable(Rq[q])
            h["ledger_A_by_q"][str(q)] = ledger(Rq[q], u)[0]
            h["dV_by_q"][str(q)] = dv(Rq[q], u)[0]
            h["n_persons_by_q"][str(q)] = int(u[:nA].sum())
        h["steps"] = {}
        for q1, q2 in zip(qs, qs[1:]):
            both = usable(Rq[q1]) & usable(Rq[q2])
            (l1, b1), (l2, b2) = ledger(Rq[q1], both), ledger(Rq[q2], both)
            h["steps"][f"{q2}-{q1}"] = interval(l2 - l1, b2 - b1, bcfg["ci"]) | {"n_A_persons": int(both[:nA].sum())}
        h["supported"] = all(ok(s) and s["lo"] > 0 for s in h["steps"].values())
        m["H0b"] = h

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
        m[f"H3_{kk}"] = {"sd_between_paraphrase": rms(bw), "sd_within_seed": rms(wi),
                         "ratio": {k_: r1[k_] for k_ in ("est", "lo", "hi")},
                         "ratio_seed_corrected": {k_: r2[k_] for k_ in ("est", "lo", "hi")},
                         "n_paraphrases": int(a.paraphrase.nunique()), "supported": bool(ok(r1) and r1["lo"] > 1)}

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
    m["note"] = "confirmatory estimands (preregistration v1); values from a fake backend or a dev dry run are not results"
    (exp / "results").mkdir(exist_ok=True)
    df.to_csv(exp / "results" / "conditions.csv", index=False, lineterminator="\n")
    m = clean(m)
    (exp / "metrics.json").write_text(json.dumps(m, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return m


if __name__ == "__main__":
    print(json.dumps(summarise(Path(sys.argv[1])), indent=1)[:4000])
