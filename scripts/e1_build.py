"""Confirmatory E1, step 1 (CPU): records per person x lambda x defender, and the recipient's knowledge K0..K3.

Preregistration: docs/preregistration-draft.md (v1) §3-§4; decisions .steering/20261112-paper-write/decisions.md 判断 3・4・7
and .steering/20261112-cpu-prep/decisions.md 判断 7・9・10. Design: .steering/20261112-e1-prereg-impl/design.md.

  population, A      permute_lambda.population (same rule as build_records.py); n_A A and n_not_A not-A persons drawn with SEED
  lambda             permute_lambda.permute (bundles of the rows that are neither A-type nor D0), per lambda
  onset clamp        in the E1 records only, an onset year before the birth year is set to the birth year (判断 4); counted
  record at year y   conditions with onset <= y, not abated before y (as build_records.py), visits in y
  defender D         D0 drop the D0 displays; D1 D0 + D1 displays -> their group; D2 D0 + the frozen D2 displays
                     (config/condition_lists.json "D2_frozen") -> their D1 group or "a coarsened condition".
                     Same-name entries of one year are merged (earliest onset).
  knowledge          per lambda, on the population minus the E1 persons; "has X" = X active in any record year.
                     K0 age and gender; K1 + P(A); K2 + P(A | each D2 display), P(A | each group shown by D1 / D2, the
                     recipient knows the coarsening scheme), P(A | none of the D2 displays); K3 + whether this person
                     has each D2 display (original, before coarsening), known from outside.

Usage: python scripts/e1_build.py <derived-dir> <exp-dir>    (reads <exp-dir>/config.json, SEED)
  -> <exp-dir>/inputs/records.jsonl, knowledge.json, build_report.json
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from permute_lambda import check_a_within_d0, d0_mask, load_tables, permute, population  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COARSE = "a coarsened condition"
OLD_TYPE = "drug abuse|opioid|overdose"  # A-type paths other than "misuses drugs" (pilot breakdown); report only


def load_lists() -> dict:
    return json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))


def defender_maps(at: dict, d2_frozen: list[str]) -> dict[str, dict]:
    """display -> shown name, per defender (D0 hides via d0_mask; the maps only coarsen)."""
    d1 = {d["display"]: d["group"] for d in (at.get("D1") or [])}
    # Group rates are computed over the union of the members a group has under D1 and D2. That is only the rate of
    # what each release shows while no D2 display falls into a D1 group (true for the frozen lists; review LOW).
    overlap = sorted(set(d2_frozen) & set(d1))
    if overlap:
        raise SystemExit(f"D2 displays inside a D1 group {overlap}: group rates would mix D1 and D2 membership")
    return {"D0": {}, "D1": d1, "D2": {d: d1.get(d, COARSE) for d in d2_frozen}}


def apply_defender(conds: list[dict], d0: dict, mapping: dict) -> tuple[list[dict], int, int]:
    """conds: [{display, onset_year}] of one year. Returns (shown conditions, n removed by D0, n coarsened)."""
    if not conds:
        return [], 0, 0
    disp = pd.Series([c["display"] for c in conds])
    hide = d0_mask(disp, d0).to_numpy()
    shown: dict[str, int] = {}
    n_coarse = 0
    for c, h in zip(conds, hide):
        if h:
            continue
        name = mapping.get(c["display"], c["display"])
        n_coarse += name != c["display"]
        shown[name] = min(shown.get(name, c["onset_year"]), c["onset_year"])
    out = [{"display": d, "onset_year": int(o)} for d, o in sorted(shown.items(), key=lambda x: (x[1], x[0]))]
    return out, int(hide.sum()), int(n_coarse)


def clamp_onset(c: pd.DataFrame, birth_year: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Set onset_year to the birth year where it is earlier (判断 4). birth_year is indexed by pid.
    Returns (clamped copy, boolean mask of the clamped rows)."""
    c = c.copy()
    by = c.pid.map(birth_year)
    bad = c.onset_year.notna() & (c.onset_year < by)
    c.loc[bad, "onset_year"] = by[bad]
    return c, bad


def active_in(c: pd.DataFrame, y: int) -> pd.DataFrame:
    return c[(c.onset_year <= y) & (c.abate_year.isna() | (c.abate_year >= y))]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%" if x >= 0.01 else f"{100 * x:.1f}%"


def knowledge(persons: pd.DataFrame, c: pd.DataFrame, years: list[int], d2: list[str], maps: dict) -> dict:
    """Rates on `persons` (column A) from condition rows c. "has X" = X active in any of `years`."""
    act = pd.concat([active_in(c, y)[["pid", "display"]] for y in years]).drop_duplicates()
    a = persons.set_index("pid").A
    n, pa = len(a), float(a.mean())

    def rate(pids: set) -> dict:
        s = a[a.index.isin(pids)]
        return {"n": int(len(s)), "p_A": float(s.mean()) if len(s) else None}

    conds = {d: rate(set(act.pid[act.display == d])) for d in d2}
    groups: dict[str, dict] = {}
    for dname in ("D1", "D2"):
        for g in sorted(set(maps[dname].values())):
            members = sorted(d for d, gg in maps[dname].items() if gg == g)
            groups.setdefault(g, {"members": set()})["members"].update(members)
    for g, v in groups.items():
        v["members"] = sorted(v["members"])
        v |= rate(set(act.pid[act.display.isin(v["members"])]))
    none = rate(set(a.index) - set(act.pid[act.display.isin(d2)]))
    return {"n": n, "n_A": int(a.sum()), "p_A": pa, "conditions": conds, "groups": groups, "none_of_d2": none}


def sentences(kn: dict, question_noun: str = "a history of drug misuse or dependence") -> dict:
    """The K1 and K2 add-on sentences for one lambda."""
    k1 = f"From population statistics you know that among adults like this person, {pct(kn['p_A'])} have {question_noun}."
    parts = []
    for d, r in kn["conditions"].items():
        if r["p_A"] is not None:
            parts.append(f"among those with \"{d}\" on record, {pct(r['p_A'])} do")
    for g, r in kn["groups"].items():
        if r["p_A"] is not None:
            parts.append(f"records shared through the assistant may show \"{g}\" in place of any of: "
                         + "; ".join(f"\"{m}\"" for m in r["members"])
                         + f" - among those with any of these, {pct(r['p_A'])} do")
    if kn["none_of_d2"]["p_A"] is not None:
        parts.append(f"among those with none of the {len(kn['conditions'])} conditions named first, "
                     f"{pct(kn['none_of_d2']['p_A'])} do")
    k2 = "You also know that " + "; ".join(parts) + "."
    return {"K1": k1, "K2": k2}


def known_text(level: str, person: dict, kn_lambda: dict, base: str) -> str:
    """K0 < K1 < K2 < K3, each containing the lower level (preregistration §4.2)."""
    out = [base.format(**person["known_facts"])]
    if level in ("K1", "K2", "K3"):
        out.append(kn_lambda["sentences"]["K1"])
    if level in ("K2", "K3"):
        out.append(kn_lambda["sentences"]["K2"])
    if level == "K3":
        facts = [f"{'has' if v else 'does not have'} \"{d}\"" for d, v in person["k3_facts"].items()]
        out.append("From an outside source you also know that this person " + ", ".join(facts) + ".")
    return " ".join(out)


def main(derived: Path, exp: Path) -> None:
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    lists = load_lists()
    at = lists["attributes"][cfg["attribute"]]
    d2 = list(lists["D2_frozen"][cfg["attribute"]]["displays"])
    years, y0, ages = cfg["years"], lists["first_year"], lists["age_range_at_first_year"]
    if years[0] != y0:
        raise SystemExit("config years[0] must equal first_year of condition_lists.json")
    maps = defender_maps(at, d2)

    p_all, c_all = load_tables(derived)
    check_a_within_d0(c_all.display, at["A_pattern"], at["D0"])
    e = pd.read_parquet(derived / "encounters.parquet")
    e["year"] = pd.to_datetime(e["start"], utc=True, errors="coerce", format="ISO8601").dt.year
    persons, c = population(p_all, c_all, at["A_pattern"], y0, ages)
    rng = np.random.default_rng([seed, zlib.crc32(b"e1-sample")])
    pick = pd.concat([
        persons[persons.A].sample(cfg["n_A"], random_state=int(rng.integers(2**31))),
        persons[~persons.A].sample(cfg["n_not_A"], random_state=int(rng.integers(2**31))),
    ])
    pick_ids = set(pick.pid)
    is_a_row = c.display.str.contains(at["A_pattern"], case=False, regex=True, na=False)
    fixed = d0_mask(c.display, at["D0"]) | is_a_row
    old = set(c.pid[c.display.str.contains(OLD_TYPE, case=False, regex=True, na=False) & (c.onset_year < y0)])
    birth = persons.set_index("pid").birth_year
    visits = e[e.pid.isin(pick_ids)].groupby(["pid", "year"]).size()
    pi = float(persons.A.mean())

    out = exp / "inputs"
    out.mkdir(exist_ok=True)
    kn_all, report = {"pi": pi, "lambdas": {}}, {"pi": pi, "n_population": int(len(persons)),
                                                  "n_pool_A": int(persons.A.sum()), "lambdas": {}}
    rows = []
    for lam in cfg["lambdas"]:
        rl = np.random.default_rng([seed, zlib.crc32(b"e1-lambda"), int(round(lam * 1000))])
        cp = permute(c, persons.pid.to_numpy(), fixed, lam, rl)
        cp, clamped = clamp_onset(cp, birth)
        rest = persons[~persons.pid.isin(pick_ids)]
        kn = knowledge(rest, cp[cp.pid.isin(set(rest.pid))], years, d2, maps)
        kn["sentences"] = sentences(kn)
        kn_all["lambdas"][str(lam)] = kn
        cs = cp[cp.pid.isin(pick_ids)]
        rep = {"onset_clamped_rows_population": int(clamped.sum()),
               "onset_clamped_rows_selected": int((clamped & cp.pid.isin(pick_ids)).sum()),
               "removed_by_D0": {}, "coarsened": {}}
        for _, person in pick.iterrows():
            pc = cs[cs.pid == person.pid]
            raw = []
            for y in years:
                a = active_in(pc, y).sort_values(["onset_year", "display"]).drop_duplicates("display")
                raw.append([{"display": d, "onset_year": int(o)} for d, o in zip(a.display, a.onset_year)])
            has_any = set(pd.concat([active_in(pc, y) for y in years]).display)
            for dname in cfg["defenders"]:
                recs = []
                for y, conds in zip(years, raw):
                    shown, n_rm, n_co = apply_defender(conds, at["D0"], maps[dname])
                    rep["removed_by_D0"][dname] = rep["removed_by_D0"].get(dname, 0) + n_rm
                    rep["coarsened"][dname] = rep["coarsened"].get(dname, 0) + n_co
                    recs.append({"year": y, "age": int(y - person.birth_year), "gender": person.gender,
                                 "visits_this_year": int(visits.get((person.pid, y), 0)),
                                 "conditions": shown, "audit_removed": n_rm, "coarsened": n_co})
                rows.append({"pid": person.pid, "A": bool(person.A),
                             "a_type": None if not person.A else ("old" if person.pid in old else "misuses_only"),
                             "lambda": lam, "D": dname,
                             "known_facts": {"age": int(y0 - person.birth_year), "gender": person.gender},
                             "k3_facts": {d: d in has_any for d in d2}, "records": recs,
                             "records_full": [{"year": y, "conditions": conds} for y, conds in zip(years, raw)]})
        report["lambdas"][str(lam)] = rep
    report["a_type"] = pd.Series([r["a_type"] for r in rows if r["A"] and r["D"] == "D0" and r["lambda"] == cfg["lambdas"][0]]
                                 ).value_counts().to_dict()
    report["seed_streams"] = {"sample": [seed, zlib.crc32(b"e1-sample")], "lambda": [seed, zlib.crc32(b"e1-lambda"), "round(lambda*1000)"]}
    with (out / "records.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, obj in (("knowledge.json", kn_all), ("build_report.json", report)):
        (out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
