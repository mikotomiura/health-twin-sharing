"""Development cohort only: D0 alias list, D1 table (Synthea displays), D2 list (lift rule), and the lambda check.

Inputs: config/condition_lists.json (D0 / D1 / D2 rule; decisions 判断 6・8・9) and <exp-dir>/config.json
("lambdas", "reps"), <exp-dir>/SEED. For each attribute:
  results/d0_<attr>.csv   displays hidden by D0, persons in the population
  results/d1_<attr>.csv   the D1 list with its lift on the development cohort (post hoc: D1 was fixed from the
                          literature without these numbers)
  results/d2_<attr>.csv   conditions meeting the D2 rule (n_c, n_joint, lift thresholds) = what D2 coarsens
  results/lambda_check.json  per lambda: number meeting the D2 rule, max / mean lift of the lambda=1 D2 set,
                          preserved per-display counts, rows moved to a person born after their onset
Usage: python scripts/dev_lists.py <derived-dir> <exp-dir>
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from permute_lambda import check_a_within_d0, d0_mask, d2_select, lift_table, load_tables, permute, population

ROOT = Path(__file__).resolve().parents[1]


def main(derived: Path, exp: Path) -> None:
    lists = json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    y0, ages, rule = lists["first_year"], lists["age_range_at_first_year"], lists["d2"]
    p_all, c_all = load_tables(derived)
    vocab = set(c_all.display)
    out = exp / "results"
    out.mkdir(exist_ok=True)
    metrics, lam_check = {}, {}
    for name, at in lists["attributes"].items():
        check_a_within_d0(c_all.display, at["A_pattern"], at["D0"])
        persons, c = population(p_all, c_all, at["A_pattern"], y0, ages)
        hidden = d0_mask(c.display, at["D0"])
        fixed = hidden | c.display.str.contains(at["A_pattern"], case=False, regex=True, na=False)
        a = set(persons.pid[persons.A])
        d0 = c[hidden].groupby("display").pid.agg(n_persons="nunique", n_A=lambda s: len(set(s) & a))
        d0.sort_values("n_persons", ascending=False).to_csv(out / f"d0_{name}.csv", lineterminator="\n")
        g = lift_table(persons, c, y0, hidden)
        d2 = d2_select(g, rule)
        d2.round(4).to_csv(out / f"d2_{name}.csv", lineterminator="\n")
        m = {"n_pop": len(persons), "n_A": len(a), "p_A": round(len(a) / len(persons), 4),
             "n_d0_displays": int(len(d0)), "n_d2": int(len(d2)), "d2_max_lift": round(float(d2.lift.max()), 2) if len(d2) else None}
        if at.get("D1"):
            d1 = pd.DataFrame(at["D1"])
            missing = sorted(set(d1.display) - vocab)
            if missing:
                raise SystemExit(f"D1 displays not in the Synthea vocabulary: {missing}")
            d1 = d1.merge(g[["n_c", "n_joint", "lift"]], left_on="display", right_index=True, how="left")
            d1["refs"] = d1.refs.map(lambda r: " ".join(f"[{x}]" for x in r))
            d1["in_d2"] = d1.display.isin(d2.index)
            d1.round(4).to_csv(out / f"d1_{name}.csv", index=False, lineterminator="\n")
            m["n_d1"] = int(len(d1))
            m["n_d1_in_d2"] = int(d1.in_d2.sum())
        metrics[name] = m

        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        pids = persons.pid.to_numpy()
        birth = dict(zip(persons.pid, persons.birth_year.astype(int)))
        base_counts = c[~fixed].groupby("display").pid.nunique()
        rows = []
        for lam in cfg["lambdas"]:
            for r in range(cfg["reps"]):
                cp = permute(c, pids, fixed, lam, rng)
                gp = lift_table(persons, cp, y0, hidden)
                moved = cp.pid.to_numpy() != c.pid.to_numpy()
                by = np.array([birth[x] for x in cp.pid.to_numpy()[moved]], dtype=float)
                early = int((cp.onset_year.to_numpy(dtype=float)[moved] < by).sum())
                rows.append({"lambda": lam, "rep": r, "n_meet_d2_rule": int(len(d2_select(gp, rule))),
                             "max_lift_d2set": float(gp.lift.reindex(d2.index).max()) if len(d2) else None,
                             "mean_lift_d2set": float(gp.lift.reindex(d2.index).mean()) if len(d2) else None,
                             "counts_preserved": bool(cp[~fixed].groupby("display").pid.nunique().equals(base_counts)),
                             "rows_moved": int(moved.sum()), "rows_onset_before_birth": early})
        df = pd.DataFrame(rows)
        lam_check[name] = {str(lam): {"reps": int(len(d)), "n_meet_d2_rule_mean": round(float(d.n_meet_d2_rule.mean()), 1),
                                      "max_lift_d2set_mean": round(float(d.max_lift_d2set.mean()), 2) if len(d2) else None,
                                      "mean_lift_d2set_mean": round(float(d.mean_lift_d2set.mean()), 3) if len(d2) else None,
                                      "counts_preserved_all": bool(d.counts_preserved.all()),
                                      "rows_moved_mean": round(float(d.rows_moved.mean())),
                                      "rows_onset_before_birth_mean": round(float(d.rows_onset_before_birth.mean()), 1)}
                           for lam, d in df.groupby("lambda")}
    (out / "lambda_check.json").write_text(json.dumps(lam_check, indent=1) + "\n", encoding="utf-8", newline="\n")
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(metrics, indent=1))
    print(json.dumps(lam_check, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
