"""Decision 8 breakdown (.steering/20261112-pilot-new-type/design.md): stats_think action rates (t3, tau) by policy and
by A subtype, with person-level bootstrap intervals. Dry-run preview only; not for confirmatory analysis.

A subtypes, from the person's records (the three years the twin saw):
  old            : a condition matching the old type (drug abuse|opioid|overdose) is in the records
  misuse_shown   : not old, "misuses drugs" is in the records
  misuse_hidden  : neither in the records (misuse ended before the first year; A only through the history)
Per person, the action rate is the mean over that person's reads (paraphrase x twin seed); persons are the unit of the
bootstrap (resampled within each group). Unparsed reads are counted separately (missing, not "no action").

Usage: python scripts/pilot_new_type_breakdown.py <releases-exp> <attacker-exp>
  reads <attacker-exp>/results/attack.jsonl, <releases-exp>/inputs/records.jsonl, <attacker-exp>/SEED;
  writes <attacker-exp>/results/breakdown.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = re.compile("drug abuse|opioid|overdose", re.I)
MIS = re.compile("misuses drugs", re.I)
N_BOOT = 4000


def subtype(person: dict) -> str:
    if not person["A"]:
        return "notA"
    ds = [c["display"] for y in person["records"] for c in y["conditions"]]
    if any(OLD.search(d) for d in ds):
        return "old"
    return "misuse_shown" if any(MIS.search(d) for d in ds) else "misuse_hidden"


def boot(per_person: np.ndarray, rng) -> dict:
    n = len(per_person)
    if n == 0:
        return {"n_persons": 0}
    bs = per_person[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    return {"n_persons": int(n), "mean": round(float(per_person.mean()), 3),
            "lo": round(float(np.quantile(bs, 0.025)), 3), "hi": round(float(np.quantile(bs, 0.975)), 3)}


def main(rel_exp: Path, att_exp: Path) -> None:
    rng = np.random.default_rng(int((att_exp / "SEED").read_text().strip()))
    taus = json.loads((att_exp / "config.json").read_text(encoding="utf-8"))["taus"]
    people = {json.loads(l)["pid"]: json.loads(l) for l in (rel_exp / "inputs" / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    sub = {pid: subtype(p) for pid, p in people.items()}
    at = pd.DataFrame([json.loads(l) for l in (att_exp / "results" / "attack.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
    at["sub"] = at.pid.map(sub)
    out: dict = {"note": "dry-run preview only; person-level bootstrap", "n_boot": N_BOOT,
                 "subtype_counts": pd.Series(sub).value_counts().to_dict()}
    out["missing_rate_by_sub_t"] = {f"{s}|t{t}": round(float(d.p.isna().mean()), 3) for (s, t), d in at.groupby(["sub", "t"])}
    pri = at[(at.t == 0) & at.p.notna()].groupby("pid").p.mean()
    out["prior_p_by_sub"] = {s: round(float(pri[[p for p in pri.index if sub[p] == s]].mean()), 3) for s in sorted(set(sub.values()))}
    post = at[(at.t == 3) & at.p.notna()].copy()
    for tau in taus:
        post[f"act_{tau}"] = (post.p >= tau).astype(float)
        res = {}
        for pol, dp in post.groupby("policy"):
            pp = dp.groupby("pid")[f"act_{tau}"].mean()
            grp = {"A_all": [p for p in pp.index if sub[p] != "notA"], "notA": [p for p in pp.index if sub[p] == "notA"]}
            for s in ("old", "misuse_shown", "misuse_hidden"):
                grp[s] = [p for p in pp.index if sub[p] == s]
            grp["misuse_only"] = grp["misuse_shown"] + grp["misuse_hidden"]
            r = {g: boot(pp[ids].to_numpy(), rng) for g, ids in grp.items()}
            # difference A - notA (independent resampling of the two groups)
            a, b = pp[grp["A_all"]].to_numpy(), pp[grp["notA"]].to_numpy()
            for g in ("A_all", "old", "misuse_only", "misuse_shown"):
                a = pp[grp[g]].to_numpy()
                if len(a) and len(b):
                    d = (a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1) - b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(1))
                    r[f"diff_{g}_minus_notA"] = {"mean": round(float(a.mean() - b.mean()), 3),
                                                 "lo": round(float(np.quantile(d, 0.025)), 3), "hi": round(float(np.quantile(d, 0.975)), 3)}
            r["per_person"] = {f"{p[:8]}|{sub[p]}": round(float(v), 3) for p, v in pp.items()}
            res[pol] = r
        out[f"tau{tau}"] = res
    (att_exp / "results" / "breakdown.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("tau")}, indent=1))
    for tau in taus:
        for pol, r in out[f"tau{tau}"].items():
            print(f"tau={tau} {pol}: " + "; ".join(f"{g} {v.get('mean')} [{v.get('lo')}, {v.get('hi')}] n={v.get('n_persons', '')}"
                                                for g, v in r.items() if g != "per_person"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
