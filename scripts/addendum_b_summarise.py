"""Addendum B (attributes x cohorts): the v4 H2 contrast and the lambda contrast per attribute and addendum cohort, with
the frozen E2 summary helpers, and the consistency count (config/addendum.json "B").

Each cohort c has its own experiment directory (scripts/addendum_b_e2.py output: results/e2_map_reps.csv, config.json
= the v4 config/e2_confirm.json values, person bootstrap on). Completeness is checked with e2_summarise.check_complete
against the addendum lists. Per attribute and cohort (dV_bal, primary as in v4; dV_acc and dV_conf secondary):
  primary   D0 - D2 at lambda 1, K2: percentile interval of the R paired per-rep differences (e2_summarise.percentile_interval;
            v4 判断 28 / 32), consistent = lower bound of the 95% interval > 0 and the estimate (mean of the R reps)
            > delta_E2 (direction predicted in v4: D0 > D2; delta_E2 = v4's resolution, as in v4's H2)
  secondary lambda1 - lambda0 at D0, K2 (by rep number, as v4 H0c in the bootstrap mode), consistent = lower bound > 0
            (the direction the lambda manipulation is built to produce; v4 did not state it as a hypothesis)
When config "B" has "confirm": true, each experiment must be a confirmatory run of scripts/addendum_b_e2.py --confirm for
the cohort of its position (results/addendum_runtime.json), with config.json = config/e2_confirm.json and the
seeds of config/addendum_seeds.json. A cohort that cannot be completed (a deviation) is summarised by hand per cohort.
  described lambda 0, D0, K2: 90% percentile interval within +-delta_E2 (as v4 H0a)
generalises (per attribute) = number of consistent cohorts >= m_consistent_per_attribute, for the primary and for the
secondary contrast separately.

Usage: python scripts/addendum_b_summarise.py <out-dir> <exp-c1> <exp-c2> <exp-c3> [--lists config/addendum_b_condition_lists.json]
           [--config config/addendum.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e2_summarise import ESTIMATORS, PRIMARY, cell, check_complete, load, percentile_interval, tost_pct  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def per_cohort(exp: Path, lists: dict, b: dict) -> dict:
    df, cfg = load(exp)
    if not cfg.get("bootstrap_persons"):
        raise SystemExit(f"{exp}: bootstrap_persons must be true (v4 percentile intervals)")
    comp = check_complete(df, cfg, lists)
    ci, cie, delta = b["ci"], b["ci_equivalence"], cfg["delta_E2"]
    out = {"completeness": comp}
    for a in b["attributes"]:
        r = {}
        for col in ESTIMATORS:
            p = percentile_interval(cell(df, a, (1.0, "D0", "K2"), col) - cell(df, a, (1.0, "D2", "K2"), col), ci)
            s = percentile_interval(cell(df, a, (1.0, "D0", "K2"), col) - cell(df, a, (0.0, "D0", "K2"), col), ci)
            z = tost_pct(cell(df, a, (0.0, "D0", "K2"), col), cie, delta)
            r[col] = {"D0_minus_D2_lambda1_K2": p | {"consistent": bool(p["lo"] > 0 and p["est"] > delta)},
                      "lambda1_minus_lambda0_D0_K2": s | {"consistent": bool(s["lo"] > 0)},
                      "lambda0_D0_K2_equivalence": z}
        out[a] = r
    return out


def summarise(exps: list[Path], lists: dict, cfg: dict) -> dict:
    b = cfg["B"]
    if len(exps) != b["n_cohorts"]:
        raise SystemExit(f"{len(exps)} cohort experiments given, config expects {b['n_cohorts']}")
    if sorted(b["attributes"]) != sorted(lists["attributes"]):
        raise SystemExit("the lists' attributes differ from config/addendum.json B.attributes")
    if b.get("confirm"):
        import hashlib
        ref = hashlib.sha256((ROOT / "config" / "e2_confirm.json").read_bytes()).hexdigest()
        seeds = hashlib.sha256((ROOT / "config" / "addendum_seeds.json").read_bytes()).hexdigest()
        for i, e in enumerate(exps):
            rt = json.loads((e / "results" / "addendum_runtime.json").read_text(encoding="utf-8"))
            if not (rt.get("confirm") and rt.get("cohort") == i + 1 and rt.get("e2_confirm_sha256") == ref
                    and rt.get("seeds_sha256") == seeds
                    and hashlib.sha256((e / "config.json").read_bytes()).hexdigest() == ref):
                raise SystemExit(f"{e}: not the confirmatory run of cohort {i + 1}")
    cohorts = {str(i + 1): per_cohort(e, lists, b) for i, e in enumerate(exps)}
    m = b["m_consistent_per_attribute"]
    gen = {}
    for a in b["attributes"]:
        n1 = sum(cohorts[c][a][PRIMARY]["D0_minus_D2_lambda1_K2"]["consistent"] for c in cohorts)
        n2 = sum(cohorts[c][a][PRIMARY]["lambda1_minus_lambda0_D0_K2"]["consistent"] for c in cohorts)
        gen[a] = {"primary_D0_minus_D2": {"n_consistent": n1, "generalises": bool(n1 >= m)},
                  "secondary_lambda1_minus_lambda0": {"n_consistent": n2, "generalises": bool(n2 >= m)}}
    return {"rule": {"consistent": "primary: lower bound of the 95% percentile interval > 0 and estimate > delta_E2; secondary: lower bound > 0 (dV_bal)", "m_consistent_per_attribute": m},
            "cohorts": cohorts, "by_attribute": gen}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("exps", type=Path, nargs="+")
    ap.add_argument("--lists", type=Path, default=ROOT / "config" / "addendum_b_condition_lists.json")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "addendum.json")
    a = ap.parse_args(argv)
    m = summarise(a.exps, json.loads(a.lists.read_text(encoding="utf-8")), json.loads(a.config.read_text(encoding="utf-8")))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "addendum_b_metrics.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(m["by_attribute"], indent=1))


if __name__ == "__main__":
    main()
