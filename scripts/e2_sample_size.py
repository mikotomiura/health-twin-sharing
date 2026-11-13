"""E2 dry run: how many synthetic persons does a sample-based (near-optimal) attacker need, and does it pick up the
residual leakage after the boundary audit? Dry-run preview only; not for confirmatory analysis.

Population = the build_records pool (alive, age range at the first year; A = attribute onset before the first year,
not-A = never), minus the persons used in the LLM dry run. Each person's "release" is approximated by the full record
at the first year after the boundary audit (conditions matching the sensitive patterns removed): an upper bound on
what an audited twin answer can carry. The attacker is a classifier fitted on a labelled sample [17][18]:
  K   = what the recipient already knows (age, gender)            -> prior
  K+R = K plus the multi-hot of the audited record's conditions    -> posterior
For each sample size N and design (natural prevalence / balanced), R repeated samples, 5-fold out-of-fold
probabilities. Reported per rep: Bayes vulnerability V = mean max(p, 1-p), dV = V(K+R) - V(K), AUC, and the threshold
gain (share of A persons with p >= tau, minus the same under K). The spread over reps gives the interval width vs N.

Usage: python scripts/e2_sample_size.py <derived-dir> <llm-exp-dir> <out-exp-dir>
  reads <out-exp-dir>/config.json ("e2" block) and SEED; writes <out-exp-dir>/results/e_e2_*.{csv,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def build_pool(derived: Path, llm_cfg: dict, exclude: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y0 = llm_cfg["years"][0]
    pat = llm_cfg["attribute"]["pattern"]
    rx_sens = "|".join(llm_cfg["attribute"]["sensitive_patterns"])
    p = pd.read_parquet(derived / "persons.parquet")
    c = pd.read_parquet(derived / "conditions.parquet")
    c["onset_year"] = pd.to_datetime(c["onset"], utc=True, errors="coerce", format="ISO8601").dt.year
    c["abate_year"] = pd.to_datetime(c["abatement"], utc=True, errors="coerce", format="ISO8601").dt.year
    p["birth_year"] = pd.to_datetime(p["birth"]).dt.year
    p = p[p["deceased"].isna()]
    lo, hi = llm_cfg["age_range_at_first_year"]
    p = p[(y0 - p["birth_year"]).between(lo, hi) & ~p.pid.isin(exclude)].copy()
    is_a = c["display"].str.contains(pat, case=False, regex=True, na=False)
    first_a = c[is_a].groupby("pid")["onset_year"].min()
    a_pids = set(first_a[first_a < y0].index)
    p = p[p.pid.isin(a_pids) | ~p.pid.isin(set(first_a.index))].copy()
    p["A"] = p.pid.isin(a_pids)
    p["age"] = y0 - p["birth_year"]
    p["male"] = (p["gender"] == "male").astype(float)
    act = c[c.pid.isin(set(p.pid)) & (c.onset_year <= y0) & (c.abate_year.isna() | (c.abate_year >= y0))]
    act = act[~act["display"].str.contains(rx_sens, case=False, regex=True, na=False)]  # boundary audit
    mh = pd.crosstab(act.pid, act.display).clip(upper=1).reindex(p.pid).fillna(0)
    return p.set_index("pid"), mh


def oof_proba(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def stats(p: np.ndarray, y: np.ndarray, taus: list[float]) -> dict:
    d = {"V": float(np.maximum(p, 1 - p).mean()), "auc": float(roc_auc_score(y, p))}
    for t in taus:
        d[f"act_A_{t}"] = float((p[y == 1] >= t).mean())
        d[f"act_notA_{t}"] = float((p[y == 0] >= t).mean())
    return d


def main(derived: Path, llm_exp: Path, out: Path) -> None:
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))["e2"]
    llm_cfg = json.loads((llm_exp / "config.json").read_text(encoding="utf-8"))
    seed = int((out / "SEED").read_text().strip())
    exclude = {json.loads(l)["pid"] for l in (llm_exp / "inputs" / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    pool, mh = build_pool(derived, llm_cfg, exclude)
    rng = np.random.default_rng(seed)
    taus = cfg["taus"]
    rows = []
    A_idx, N_idx = pool.index[pool.A], pool.index[~pool.A]
    for design in cfg["designs"]:
        for n in cfg["sample_sizes"]:
            if design == "balanced" and n // 2 > len(A_idx):
                continue
            if n > len(pool):
                continue
            for r in range(cfg["reps"]):
                if design == "balanced":
                    ids = np.concatenate([rng.choice(A_idx, n // 2, replace=False), rng.choice(N_idx, n - n // 2, replace=False)])
                else:
                    ids = rng.choice(pool.index, n, replace=False)
                sub = pool.loc[ids]
                y = sub.A.to_numpy(int)
                if y.sum() < 5 or (1 - y).sum() < 5:
                    continue
                m = mh.loc[ids]
                m = m.loc[:, m.sum() >= cfg["min_feature_count"]]
                xk = np.column_stack([(sub.age - 47.5) / 13, sub.male])
                xr = np.column_stack([xk, m.to_numpy(float)])
                s = int(rng.integers(2**31))
                pk, pr = oof_proba(xk, y, s), oof_proba(xr, y, s)
                sk, sr = stats(pk, y, taus), stats(pr, y, taus)
                row = {"design": design, "n": n, "rep": r, "n_A": int(y.sum()), "n_features": int(xr.shape[1]),
                       "V_K": sk["V"], "V_KR": sr["V"], "dV": sr["V"] - sk["V"], "auc_K": sk["auc"], "auc_KR": sr["auc"]}
                for t in taus:
                    row[f"g_{t}"] = sr[f"act_A_{t}"] - sk[f"act_A_{t}"]
                    row[f"fp_{t}"] = sr[f"act_notA_{t}"]
                rows.append(row)
    df = pd.DataFrame(rows)
    res = out / "results"
    res.mkdir(exist_ok=True)
    df.to_csv(res / "e_e2_reps.csv", index=False, lineterminator="\n")
    cols = ["dV", "auc_KR", "auc_K"] + [f"g_{t}" for t in taus] + [f"fp_{t}" for t in taus]
    summ = {}
    for (design, n), d in df.groupby(["design", "n"]):
        summ[f"{design}|{n}"] = {"reps": int(len(d)), "n_A_median": float(d.n_A.median()),
                                 **{c: {"mean": round(float(d[c].mean()), 3),
                                        "lo": round(float(d[c].quantile(0.025)), 3),
                                        "hi": round(float(d[c].quantile(0.975)), 3)} for c in cols},
                                 "share_reps_dV_gt_0": round(float((d.dV > 0).mean()), 3),
                                 "share_reps_g05_gt_0": round(float((d["g_0.5"] > 0).mean()), 3) if "g_0.5" in d else None}
    summary = {"pool": {"n": int(len(pool)), "n_A": int(pool.A.sum()), "n_condition_columns": int(mh.shape[1])},
               "by_design_n": summ, "note": "dry-run preview only; not for confirmatory analysis"}
    (res / "e_e2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summary["pool"]))
    for k, v in summ.items():
        print(k, "nA", v["n_A_median"], "dV", v["dV"], "aucKR", v["auc_KR"]["mean"], "g0.5", v.get("g_0.5"), "fp0.5", v.get("fp_0.5"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
