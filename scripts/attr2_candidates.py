"""Second-attribute candidates on the development cohort: size of A / non-A and the conditions with the highest lift.

Population: alive at the end, age in cfg['age_range_at_first_year'] at first_year. For a pattern, A = a matching condition
with onset before first_year; non-A = never a matching condition; others are excluded (as build_records.py does).
A correlate c = a condition with onset before first_year. lift = P(A | c) / P(A) over A + non-A. Conditions matching the
candidate's own pattern are not correlates. Deterministic (no sampling); SEED is recorded only for the convention.

Usage: python scripts/attr2_candidates.py <derived-dir> <exp-dir>  -> <exp-dir>/results/{candidates.csv, correlates.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def population(derived: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(derived / "persons.parquet")
    c = pd.read_parquet(derived / "conditions.parquet")
    c["onset_year"] = pd.to_datetime(c["onset"], utc=True, errors="coerce", format="ISO8601").dt.year
    p["birth_year"] = pd.to_datetime(p["birth"]).dt.year
    p = p[p["deceased"].isna()]
    lo, hi = cfg["age_range_at_first_year"]
    p = p[(cfg["first_year"] - p["birth_year"]).between(lo, hi)]
    return p, c[c.pid.isin(p.pid)]


def split(c: pd.DataFrame, pids: pd.Index, pattern: str, y0: int) -> tuple[set, set]:
    hit = c[c["display"].str.contains(pattern, case=False, regex=True, na=False)]
    first = hit.groupby("pid")["onset_year"].min()
    a = set(first[first < y0].index)
    return a, set(pids) - set(first.index)


def main(derived: Path, exp: Path) -> None:
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    y0 = cfg["first_year"]
    p, c = population(derived, cfg)
    pids = pd.Index(p.pid)
    before = c[c.onset_year < y0].drop_duplicates(["pid", "display"])
    main_a, _ = split(c, pids, cfg["main_attribute"], y0)
    rows, corr = [], {}
    for name, pat in cfg["candidates"].items():
        a, n = split(c, pids, pat, y0)
        pop = a | n
        b = before[before.pid.isin(pop) & ~before.display.str.contains(pat, case=False, regex=True, na=False)]
        g = b.groupby("display").pid.agg(n_c="nunique", n_joint=lambda s: len(set(s) & a))
        pa = len(a) / len(pop) if pop else float("nan")
        g["p_A_given_c"] = g.n_joint / g.n_c
        g["lift"] = g.p_A_given_c / pa
        g["p_c"] = g.n_c / len(pop)
        g = g[(g.n_c >= cfg["min_n_correlate"]) & (g.n_joint >= cfg["min_n_joint"])].sort_values("lift", ascending=False)
        top = g.head(cfg["top_correlates"])
        corr[name] = [{"display": d, **{k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in r.items()}}
                      for d, r in top.iterrows()]
        rows.append({"candidate": name, "pattern": pat, "n_pop": len(pop), "n_A": len(a), "p_A": round(pa, 4),
                     "n_A_female": int(p[p.pid.isin(a)].gender.eq("female").sum()),
                     "overlap_with_main_A": len(a & main_a),
                     "max_lift": round(float(top.lift.max()), 2) if len(top) else None,
                     "n_correlates_lift_ge_2": int((g.lift >= 2).sum())})
    out = exp / "results"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "candidates.csv", index=False, lineterminator="\n")
    (out / "correlates.json").write_text(json.dumps(corr, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"population {len(p)}; main A {len(main_a)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
