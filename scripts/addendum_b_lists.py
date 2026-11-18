"""Addendum B: build config/addendum_b_condition_lists.json (attributes and frozen D2 lists for the addendum E2).

Not part of the frozen set (v4); it becomes part of the addendum's frozen set. Researcher's decision 2026-11-17: the
attributes are the primary one (drug_misuse), diabetes_t2 and severe_anxiety (.steering/_handoff/20261117-1245-addendum-a-b-merged.md).

- drug_misuse and diabetes_t2: A_pattern, D0, D1 and D2_frozen are copied verbatim from the v4 config/condition_lists.json
  (frozen at b35f5f7); the script recomputes their D2 on the development cohort and stops if it differs from the frozen
  list (same rule, same data: a mismatch means the input or the code is not the one of the freeze).
- severe_anxiety (new): A_pattern and D0 below (Synthea v4.0.0 encounter/anxiety_screening.json gives the display
  "Severe anxiety (panic) (finding)"; experiments/20261117-addendum-b-candidates/notes.md). No D1 (no literature group yet).
  Its D2 list is chosen on the development cohort with freeze_d2.compute (the v4 rule: d2_select at lambda = 1).
first_year, age_range_at_first_year and the d2 rule are copied from v4.

Usage: python scripts/addendum_b_lists.py <dev-derived-dir> [--out config/addendum_b_condition_lists.json] [--force]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_d2 import compute, sha256

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "config" / "condition_lists.json"
OUT = ROOT / "config" / "addendum_b_condition_lists.json"
KEEP = ("drug_misuse", "diabetes_t2")
NEW = {"severe_anxiety": {"A_pattern": "severe anxiety", "D0": {"include": ["anxiety", "panic"], "exclude": []}, "D1": [],
                          "_note": "addendum B (2026-11-17). Synthea v4.0.0 encounter/anxiety_screening.json. No D1."}}


def build(derived: Path, v4: dict) -> dict:
    lists = {k: v4[k] for k in ("first_year", "age_range_at_first_year", "d2")}
    lists["attributes"] = {k: v4["attributes"][k] for k in KEEP} | NEW
    d2 = compute(derived, lists)
    for k in KEEP:
        if d2[k]["displays"] != sorted(v4["D2_frozen"][k]["displays"]):
            raise SystemExit(f"{k}: the D2 recomputed on {derived} differs from the v4 D2_frozen; wrong cohort or code")
    lists["D2_frozen"] = {k: v4["D2_frozen"][k] for k in KEEP} | {k: d2[k] for k in NEW}
    lists["D2_frozen"]["_source"] = {"cohort": "development (Synthea 10k, seed 42), lambda = 1",
                                     "persons.parquet": sha256(derived / "persons.parquet"),
                                     "conditions.parquet": sha256(derived / "conditions.parquet"),
                                     "rule": lists["d2"], "script": "scripts/addendum_b_lists.py (freeze_d2.compute)",
                                     "copied_from_v4": list(KEEP)}
    lists["_note"] = ("Addendum B condition lists (not the v4 file). drug_misuse and diabetes_t2 copied from v4 "
                      "config/condition_lists.json at b35f5f7; severe_anxiety added with D2 from the development cohort.")
    return lists


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    lists = build(a.derived, json.loads(V4.read_text(encoding="utf-8")))
    if a.out.exists() and json.loads(a.out.read_text(encoding="utf-8")) != lists and not a.force:
        raise SystemExit(f"{a.out} exists and differs; refusing to overwrite (use --force only before the addendum freeze)")
    a.out.write_text(json.dumps(lists, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v["displays"] for k, v in lists["D2_frozen"].items() if not k.startswith("_")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
