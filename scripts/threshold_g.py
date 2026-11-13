"""Threshold gain (g) from attacker posteriors. Dry-run preview only; not for confirmatory analysis.

The recipient acts against the person (e.g. does not hire) when its posterior P(A) reaches a threshold tau that
depends on the recipient type. With the gain function g(act, x) = 1 if x = A, the recipient's expected gain
from acting is p, so it acts iff p >= tau (a threshold/cost-sensitive g-leakage, cf. [17][18]). Per read we keep:
  harm_true  = 1[p_T >= tau] for persons with A      (the hidden state is acted on)
  harm_false = 1[p_T >= tau] for persons without A   (acted on wrongly)
  g_gain     = 1[p_T >= tau] - 1[p_0 >= tau]         (increase over the prior from K_r alone)
and the variance split of the per-person harm rate between twin paraphrases and twin re-runs (seeds).

Usage: python scripts/threshold_g.py <attacks.jsonl> <out.json> [--taus 0.3,0.5,0.7,0.9] [--group-cols ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELL = ["pid", "recipient", "policy", "paraphrase", "seed_idx"]


def threshold_table(a: pd.DataFrame, taus: list[float], cell: list[str]) -> dict:
    a = a.dropna(subset=["p"])
    t_max = a.groupby(cell)["t"].transform("max")
    first = a[a.t == 0].set_index(cell)["p"].rename("p0")
    last = a[a.t == t_max].set_index(cell)[["p", "A"]].rename(columns={"p": "pT"})
    d = last.join(first, how="inner").reset_index()
    out = {"n_reads": int(len(d)), "by_tau": {}}
    for tau in taus:
        d["act_T"] = (d.pT >= tau).astype(float)
        d["act_0"] = (d.p0 >= tau).astype(float)
        d["g_gain"] = d.act_T - d.act_0
        by = d.groupby(["policy", "A"]).agg(act_T=("act_T", "mean"), act_0=("act_0", "mean"), g_gain=("g_gain", "mean"))
        # variance split among A persons: per (pid, policy), sd across paraphrase means vs within-paraphrase sd over seeds
        rows = []
        if "paraphrase" in d and "seed_idx" in d:
            for (pid, pol), x in d[d.A].groupby(["pid", "policy"]):
                m = x.groupby("paraphrase").act_T.mean()
                v = x.groupby("paraphrase").act_T.var(ddof=1)
                rows.append({"policy": pol, "sd_between_paraphrase": float(m.std(ddof=1)),
                             "sd_within_seed": float(np.sqrt(v.mean()))})
        sd = pd.DataFrame(rows).groupby("policy").mean().round(3).to_dict(orient="index") if rows else {}
        out["by_tau"][str(tau)] = {
            "by_policy_A": {f"{p}|A={a_}": {k: round(float(v), 3) for k, v in r.items()}
                            for (p, a_), r in by.iterrows()},
            "contrast_none_minus_audit_A": round(float(by.loc[("none", True), "g_gain"] - by.loc[("audit", True), "g_gain"]), 3)
            if ("none", True) in by.index and ("audit", True) in by.index else None,
            "sd_split_A_persons": sd,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("attacks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--taus", default="0.3,0.5,0.7,0.9")
    a_ = ap.parse_args()
    a = pd.DataFrame([json.loads(l) for l in a_.attacks.read_text(encoding="utf-8").splitlines() if l.strip()])
    res = threshold_table(a, [float(x) for x in a_.taus.split(",")], CELL)
    res["note"] = "dry-run preview only; not for confirmatory analysis"
    a_.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
