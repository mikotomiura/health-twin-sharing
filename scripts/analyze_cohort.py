"""Dry run for §6-1/§6-3: prevalence of candidate hidden attributes A, time points, and attributes correlated with A.

Usage: python scripts/analyze_cohort.py <derived-dir> <experiment-dir>
Reads persons/conditions/encounters.parquet from <derived-dir> (scripts/profile_cohort.py); config.json in <experiment-dir>.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """MI in bits between two binary vectors."""
    mi = 0.0
    for x in (0, 1):
        for y in (0, 1):
            pxy = np.mean((a == x) & (b == y))
            px, py = np.mean(a == x), np.mean(b == y)
            if pxy > 0:
                mi += pxy * np.log2(pxy / (px * py))
    return float(mi)


def main(derived: Path, exp: Path) -> None:
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    out = exp / "results"
    out.mkdir(exist_ok=True)
    p = pd.read_parquet(derived / "persons.parquet")
    c = pd.read_parquet(derived / "conditions.parquet")
    e = pd.read_parquet(derived / "encounters.parquet")
    end = pd.Timestamp(cfg["end_date"], tz="UTC")

    p["birth"] = pd.to_datetime(p["birth"], utc=True)
    p["age_at_end"] = (end - p["birth"]).dt.days / 365.25
    p["alive"] = p["deceased"].isna()
    c["onset"] = pd.to_datetime(c["onset"], utc=True, errors="coerce", format="ISO8601")
    e["start"] = pd.to_datetime(e["start"], utc=True, errors="coerce", format="ISO8601")

    # keyword hits, so the definitions can be checked by eye
    disp = c.drop_duplicates(["pid", "display"]).display.value_counts()
    hits = {k: disp[disp.index.str.contains(v, case=False, regex=True)].to_dict() for k, v in cfg["candidates"].items()}
    (out / "candidate_displays.json").write_text(json.dumps(hits, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")

    # person x condition (ever) matrix, for correlates
    pids = p["pid"].to_numpy()
    lo, hi = cfg["window_years"]
    win_start = end - pd.DateOffset(years=hi)
    cw = c[c["onset"] < end]
    has = pd.crosstab(cw["pid"], cw["display"]).clip(upper=1).reindex(pids, fill_value=0)

    rows, corr = [], {}
    for k, pat in cfg["candidates"].items():
        a_pids = set(c.loc[c.display.str.contains(pat, case=False, regex=True, na=False), "pid"])
        a = p["pid"].isin(a_pids).to_numpy().astype(int)
        first = c[c.pid.isin(a_pids) & c.display.str.contains(pat, case=False, regex=True, na=False)].groupby("pid").onset.min()
        in_window = int(((first >= win_start) & (first < end)).sum())
        rows.append({
            "A": k, "n": int(a.sum()), "prevalence": float(a.mean()),
            "prevalence_female": float(a[p.gender.eq("female").to_numpy()].mean()),
            "prevalence_male": float(a[p.gender.eq("male").to_numpy()].mean()),
            "first_onset_in_last_window": in_window,
            "first_onset_median_year": float(first.dt.year.median()) if len(first) else None,
        })
        if a.sum() >= cfg["min_cases_for_correlates"]:
            excl = [d for d in has.columns if pd.Series([d]).str.contains(pat, case=False, regex=True).iloc[0]]
            cands = has.drop(columns=excl)
            cands = cands.loc[:, cands.mean() >= cfg["min_prevalence_correlate"]]
            mi = {d: mutual_info(a, cands[d].to_numpy()) for d in cands.columns}
            top = sorted(mi.items(), key=lambda kv: -kv[1])[: cfg["top_correlates"]]
            corr[k] = [
                {"attr": d, "mi_bits": round(v, 4), "p_attr": round(float(cands[d].mean()), 3),
                 "p_A_given_attr": round(float(a[cands[d].to_numpy() == 1].mean()), 3),
                 "p_A_given_not": round(float(a[cands[d].to_numpy() == 0].mean()), 3)}
                for d, v in top
            ]
            # demographic correlates
            corr[k].append({"attr": "gender=female", "mi_bits": round(mutual_info(a, p.gender.eq("female").to_numpy().astype(int)), 4)})
    prev = pd.DataFrame(rows).sort_values("prevalence", ascending=False)
    prev.to_csv(out / "candidate_prevalence.csv", index=False, lineterminator="\n")
    (out / "correlates.json").write_text(json.dumps(corr, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")

    # time points: years with any encounter in the last `hi` years
    ew = e[(e.start >= win_start) & (e.start < end)]
    years = ew.assign(y=ew.start.dt.year).groupby("pid").y.nunique().reindex(pids, fill_value=0)
    per_year = ew.assign(y=ew.start.dt.year).groupby(["pid", "y"]).size()
    metrics = {
        "n_persons": int(len(p)), "n_alive_at_end": int(p.alive.sum()),
        "age_at_end_quantiles": {q: round(float(p.age_at_end.quantile(q)), 1) for q in (0.05, 0.5, 0.95)},
        "female_share": round(float(p.gender.eq("female").mean()), 3),
        "window": f"last {hi} years before {cfg['end_date']}",
        "years_with_encounter_quantiles": {q: float(years.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75)},
        "share_with_ge6_years": round(float((years >= 6).mean()), 3),
        "encounters_per_active_year_median": float(per_year.median()),
        "n_distinct_condition_displays": int(c.display.nunique()),
        "candidates": prev.to_dict(orient="records"),
    }
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(prev.to_string(index=False))
    print(json.dumps({k: v for k, v in metrics.items() if k != "candidates"}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
