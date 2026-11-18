"""Freeze the D2 lists (defender knowledge from data) chosen on the development cohort into config/condition_lists.json.

The preregistration (docs/preregistration-draft.md, OSF items "Explanation of foreknowledge and managing unintended influences" and "Manipulated variables") carries only the lists from the development cohort to the
confirmatory cohort. This script recomputes them with the same rule as e2_map.py / dev_lists.py (d2_select on the
lambda = 1 development cohort, 判断 9) and writes "D2_frozen" with the sha256 of the input parquet files.
It refuses to overwrite an existing D2_frozen entry that differs, unless --force.

Usage: python scripts/freeze_d2.py <dev-derived-dir> [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from permute_lambda import d0_mask, d2_select, lift_table, load_tables, population

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "config" / "condition_lists.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compute(derived: Path, lists: dict) -> dict:
    p_all, c_all = load_tables(derived)
    y0, ages, rule = lists["first_year"], lists["age_range_at_first_year"], lists["d2"]
    out = {}
    for name, at in lists["attributes"].items():
        persons, c = population(p_all, c_all, at["A_pattern"], y0, ages)
        g = d2_select(lift_table(persons, c, y0, d0_mask(c.display, at["D0"])), rule)
        out[name] = {"displays": sorted(g.index), "lift": {d: round(float(v), 3) for d, v in g.lift.items()}}
    return out


def main(derived: Path, force: bool) -> None:
    lists = json.loads(LISTS.read_text(encoding="utf-8"))
    frozen = compute(derived, lists)
    frozen["_source"] = {"cohort": "development (Synthea 10k, seed 42), lambda = 1",
                         "persons.parquet": sha256(derived / "persons.parquet"),
                         "conditions.parquet": sha256(derived / "conditions.parquet"),
                         "rule": lists["d2"], "script": "scripts/freeze_d2.py"}
    old = lists.get("D2_frozen")
    if old is not None and old != frozen and not force:
        raise SystemExit("D2_frozen exists and differs; refusing to overwrite (use --force only before the freeze)")
    if old == frozen:
        print("D2_frozen already up to date")
    elif old is None:  # append the key without reformatting the hand-written part of the file
        text = LISTS.read_text(encoding="utf-8").rstrip()
        assert text.endswith("}")
        block = json.dumps(frozen, indent=1, ensure_ascii=False).replace("\n", "\n ")
        LISTS.write_text(text[:-1].rstrip() + ',\n "D2_frozen": ' + block + "\n}\n", encoding="utf-8", newline="\n")
    else:  # --force
        lists["D2_frozen"] = frozen
        LISTS.write_text(json.dumps(lists, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    assert json.loads(LISTS.read_text(encoding="utf-8"))["D2_frozen"] == frozen
    print(json.dumps({k: v["displays"] for k, v in frozen.items() if not k.startswith("_")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.derived, a.force)
