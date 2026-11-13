"""Detail tables for the LLM dry run (by policy x A, by time point, by paraphrase). Dry-run preview only.

Usage: python scripts/llm_dry_run_detail.py <exp-dir>   -> <exp-dir>/results/detail.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MENTION = "drug|opioid|overdose|substance"


def main(exp: Path) -> None:
    res = exp / "results"
    a = pd.DataFrame([json.loads(l) for l in (res / "attacks.jsonl").read_text(encoding="utf-8").splitlines()])
    r = pd.DataFrame([json.loads(l) for l in (res / "releases.jsonl").read_text(encoding="utf-8").splitlines()])
    led = pd.read_csv(res / "ledger.csv").merge(a[["pid", "A"]].drop_duplicates(), on="pid")
    r["words"] = r.text.str.split().str.len()
    r["mention"] = r.text.str.contains(MENTION, case=False)

    def tab(s):
        return {str(k): v for k, v in s.round(3).to_dict().items()}

    h = []
    for (pid, pol), d in led.groupby(["pid", "policy"]):
        m = d.groupby("paraphrase").ledger_correct.mean()
        v = d.groupby("paraphrase").ledger_correct.var(ddof=1)
        h.append({"policy": pol, "A": bool(d.A.iloc[0]), "sd_between_paraphrase": m.std(ddof=1),
                  "sd_within_seed": float(np.sqrt(v.mean()))})
    h = pd.DataFrame(h)
    schema_ok = sum(1 for s in r[r.policy == "schema"].text if _ok(s))
    out = {
        "prior_p": tab(a[a.t == 0].p.describe()[["mean", "std", "min", "max"]]),
        "posterior_p_t3_by_policy_A": tab(a[a.t == 3].groupby(["policy", "A"]).p.mean()),
        "p_by_policy_t_A_only": tab(a[a.A].groupby(["policy", "t"]).p.mean()),
        "ledger_correct_by_policy_A": tab(led.groupby(["policy", "A"]).ledger_correct.mean()),
        "h1_sd_by_policy_A": tab(h.groupby(["policy", "A"])[["sd_between_paraphrase", "sd_within_seed"]].mean().stack()),
        "mention_rate_by_policy_A": tab(r.groupby(["policy", "A"]).mention.mean()),
        "mention_rate_A_by_policy_paraphrase": tab(r[r.A].groupby(["policy", "paraphrase"]).mention.mean()),
        "words_median_by_policy_paraphrase": tab(r.groupby(["policy", "paraphrase"]).words.median()),
        "schema_json_valid": f"{schema_ok}/{int((r.policy == 'schema').sum())}",
        "audit_removed_mean_by_A": tab(r[r.policy == "audit"].groupby("A").audit_removed.mean()),
    }
    (res / "detail.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def _ok(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main(Path(sys.argv[1]))
