"""Build per-person, per-time-point records for the LLM dry run from the flattened cohort tables.

A person's record at year y = age, gender, visits in y, and conditions with onset year <= y that were not abated
before y. A = the person has a condition matching cfg["attribute"]["pattern"] with onset before the first time point;
non-A persons never have it. Persons must be alive at the end date and within cfg['age_range_at_first_year']. Sampling uses SEED.

Usage: python scripts/build_records.py <derived-dir> <exp-dir>   -> <exp-dir>/inputs/records.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(derived: Path, exp: Path) -> None:
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    years = cfg["years"]
    pat = cfg["attribute"]["pattern"]
    p = pd.read_parquet(derived / "persons.parquet")
    c = pd.read_parquet(derived / "conditions.parquet")
    e = pd.read_parquet(derived / "encounters.parquet")
    c["onset_year"] = pd.to_datetime(c["onset"], utc=True, errors="coerce", format="ISO8601").dt.year
    c["abate_year"] = pd.to_datetime(c["abatement"], utc=True, errors="coerce", format="ISO8601").dt.year
    e["year"] = pd.to_datetime(e["start"], utc=True, errors="coerce", format="ISO8601").dt.year
    p["birth_year"] = pd.to_datetime(p["birth"]).dt.year
    p = p[p["deceased"].isna()]
    lo_age, hi_age = cfg["age_range_at_first_year"]
    p = p[(years[0] - p["birth_year"]).between(lo_age, hi_age)]

    is_a = c["display"].str.contains(pat, case=False, regex=True, na=False)
    first_a = c[is_a].groupby("pid")["onset_year"].min()
    a_pids = first_a[first_a < years[0]].index
    ever_a = set(first_a.index)
    pool_a = p[p.pid.isin(a_pids)]
    pool_n = p[~p.pid.isin(ever_a)]
    rng = np.random.default_rng(seed)
    pick = pd.concat([
        pool_a.sample(cfg["n_A"], random_state=int(rng.integers(2**31))),
        pool_n.sample(cfg["n_not_A"], random_state=int(rng.integers(2**31))),
    ])

    corr = cfg["known_correlate"]
    out = exp / "inputs"
    out.mkdir(exist_ok=True)
    with (out / "records.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for _, person in pick.iterrows():
            pc = c[c.pid == person.pid]
            recs = []
            for y in years:
                active = pc[(pc.onset_year <= y) & (pc.abate_year.isna() | (pc.abate_year >= y))]
                active = active.drop_duplicates("display").sort_values("onset_year")
                recs.append({
                    "year": y, "age": int(y - person.birth_year), "gender": person.gender,
                    "visits_this_year": int((e.pid.eq(person.pid) & e.year.eq(y)).sum()),
                    "conditions": [{"display": d, "onset_year": int(o)} for d, o in zip(active.display, active.onset_year)],
                })
            has_corr = bool(pc.display.str.contains(corr, case=False, regex=True, na=False).any())
            f.write(json.dumps({
                "pid": person.pid, "A": bool(person.pid in set(a_pids)), "known_level": cfg["known_level"],
                "known_facts": {"age": recs[0]["age"], "gender": person.gender,
                                "correlate": ("has a record of " + corr) if has_corr else ("has no record of " + corr)},
                "records": recs,
            }, ensure_ascii=False) + "\n")
    print(f"{len(pick)} persons ({cfg['n_A']} A) x {len(years)} time points -> {out / 'records.jsonl'}; "
          f"pool A={len(pool_a)}, pool not-A={len(pool_n)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
