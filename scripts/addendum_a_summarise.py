"""Addendum A (attacker swap): the v4 H1 statistic per attacker model, with the frozen E1 summary code, and the
consistency count (config/addendum.json "A").

For each attacker, a temporary E1-like experiment is assembled from the confirmatory E1 (config.json, SEED, inputs/,
results/twin.jsonl, all copied) and that attacker's rows (scripts/addendum_reattack.py output, restricted to the planned
keys: lambda 1, D0, default paraphrases, K in config "A" ks). scripts/e1_summarise.summarise (unchanged) is run on it, so
the H1 estimate, its person-bootstrap 95% interval and the sign-flip permutation p are the v4 estimators. The other v4
quantities of that output (K3, H0b, ...) have no reads and are not used.
Per attacker: H1_K2_minus_K0 (est, lo, hi, permutation p), H1_secondary_K2_minus_K1, H1_control_K2_minus_K0n, H0d (as
e1_summarise reports it; its part (a) needs K3 and is not judged here), the number of planned and valid reads.
consistent = the one-sided sign-flip permutation p of K2 - K0 (v4's H1 decision statistic, "p_one_sided_permutation";
direction predicted in v4: K2 > K0) <= alpha (0.025, unadjusted), the attacker is complete (every planned read written)
and none of its H1 conditions (lambda 1, D0, K2 and K0; not K1 or K0n) has more than 10% missing reads or unusable persons
(v4's 10% threshold, "missing_over_10pct" of e1_summarise; in v4 a reporting rule, here a criterion of the count).
H0d_b_and_c_within_margin = v4's H0d parts (b) and (c) both within the margin (None when either is not decided);
qualified_H0d = not True (the result is qualified as "in this attacker's measurement of K"; the count is not changed). The 95% interval and v4's missing-data sensitivity are reported beside it.
generalises = number of consistent attackers >= m_consistent.
When config "A" has "digests", every line of <res>/versions.jsonl for the attacker's file must be a confirmatory run
("confirm": true, written by addendum_reattack.py --confirm) with the registered Ollama version and digest.

Usage: python scripts/addendum_a_summarise.py <e1-exp> <addendum-a-results-dir> <out-dir> [--config config/addendum.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e1_summarise  # noqa: E402
from addendum_reattack import jobs_for, subset, tag_of  # noqa: E402
from e1_run import Exp  # noqa: E402
from llm_dry_run_extra import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ADD = ("model", "think", "thinking_chars")
H1_KS = ("K2", "K0")  # the H1 conditions to which the 10% criterion applies (lambda 1, D0)


def one(e1: Path, rows: list[dict], keys: set[str], ks=("K0", "K1", "K0n", "K2")) -> dict:
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "exp"
        (d / "results").mkdir(parents=True)
        for f in ("config.json", "SEED"):
            shutil.copy(e1 / f, d / f)
        shutil.copytree(e1 / "inputs", d / "inputs")
        shutil.copy(e1 / "results" / "twin.jsonl", d / "results" / "twin.jsonl")
        seen, kept = set(), []
        for r in rows:
            if r["key"] in keys and r["key"] not in seen:  # first row of a key wins, as in e1_summarise
                seen.add(r["key"])
                kept.append({k: v for k, v in r.items() if k not in ADD})
        with (d / "results" / "attack.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        m = e1_summarise.summarise(d)
    valid = sum(r.get("p") is not None for r in kept)
    # only the conditions this addendum reads: the other planned E1 conditions have no rows here by design
    over = [r for r in m.get("missing_over_10pct", []) if r["lambda"] == 1 and r["D"] == "D0" and r["K"] in H1_KS]
    other = [r for r in m.get("missing_over_10pct", []) if r["lambda"] == 1 and r["D"] == "D0" and r["K"] in ks and r["K"] not in H1_KS]
    h0d = m.get("H0d") or {}
    bc = [(h0d.get(k) or {}).get("within_margin") for k in ("K1_minus_rate", "K2_minus_rate")]
    h0d_bc = None if any(v is None for v in bc) else bool(all(bc))
    return {"H1_K2_minus_K0": m.get("H1_K2_minus_K0"), "H1_secondary_K2_minus_K1": m.get("H1_secondary_K2_minus_K1"),
            "H1_control_K2_minus_K0n": m.get("H1_control_K2_minus_K0n"), "H0d": m.get("H0d"),
            "missing_over_10pct": over, "missing_over_10pct_other_K": other,
            "missing_sensitivity_H1": m.get("missing_sensitivity_H1") if over else None,
            "H0d_b_and_c_within_margin": h0d_bc, "qualified_H0d": h0d_bc is not True,
            "valid_by_K": {k: sum(r.get("p") is not None for r in kept if r["K"] == k) for k in ks},
            "n_planned": len(keys), "n_rows": len(kept), "n_valid": valid, "complete": len(kept) == len(keys)}


def check_versions(res: Path, model: str, tag: str, a: dict) -> None:
    f = res / "versions.jsonl"
    lines = [v for v in load_jsonl(f) if v.get("file") == tag] if f.exists() else []
    bad = [v for v in lines if not (v.get("confirm") and v.get("ollama") == a["ollama"] and v.get("digest") == a["digests"][model])]
    if not lines or bad:
        raise SystemExit(f"{model}: versions.jsonl has no confirmatory run or a run with another Ollama / digest: {bad or 'none'}")


def summarise(e1: Path, res: Path, cfg: dict) -> dict:
    a = cfg["A"]
    if any("__FILL" in str(x) for x in a["attackers"]):
        raise SystemExit("config/addendum.json A.attackers is not filled in")
    x = Exp(e1)
    keys = {x.attack_key(j) for j in jobs_for(x, a["ks"], subset(x, None))}
    alpha = a.get("alpha", 0.025)
    out = {"rule": {"consistent": f"one-sided sign-flip permutation p of K2 - K0 <= {alpha}, complete, no H1 condition over 10% missing",
                    "m_consistent": a["m_consistent"]}, "attackers": {}}
    for m in a["attackers"]:
        tag = tag_of(m, a.get("think", {}).get(m))
        f = res / f"attack_{tag}.jsonl"
        if not f.exists():
            raise SystemExit(f"{f} not found")
        if "digests" in a:
            check_versions(res, m, tag, a)
        r = one(e1, load_jsonl(f), keys, a["ks"])
        p = (r["H1_K2_minus_K0"] or {}).get("p_one_sided_permutation")
        r["consistent"] = bool(r["complete"] and not r["missing_over_10pct"] and p is not None and p <= alpha)
        r["reasoning"] = a.get("reasoning", {}).get(m)
        out["attackers"][m] = r
    n = sum(v["consistent"] for v in out["attackers"].values())
    out["n_consistent"] = n
    out["generalises"] = bool(n >= a["m_consistent"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("e1", type=Path)
    ap.add_argument("res", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "addendum.json")
    a = ap.parse_args(argv)
    m = summarise(a.e1, a.res, json.loads(a.config.read_text(encoding="utf-8")))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "addendum_a_metrics.json").write_text(json.dumps(e1_summarise.clean(m), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: m[k] for k in ("n_consistent", "generalises")}))


if __name__ == "__main__":
    main()
