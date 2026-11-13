"""Flatten Synthea FHIR bundles into small tables for the dry run (persons / conditions / encounters).

Reads either a directory of bundle JSON files or a .tar.zst archive (streamed; nothing is unpacked to disk).

Usage:
  python scripts/profile_cohort.py <bundle-dir | archive.tar.zst> <out-dir>

Writes <out-dir>/persons.parquet, conditions.parquet, encounters.parquet.
Only coded fields and dates are kept (no names, addresses or free text); the data is synthetic.
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd
import zstandard


def iter_bundles(src: Path):
    if src.is_dir():
        for p in sorted(src.rglob("*.json")):
            if p.name == "manifest.json":
                continue
            yield p.name, p.read_bytes()
        return
    with src.open("rb") as f:
        reader = zstandard.ZstdDecompressor().stream_reader(f)
        with tarfile.open(fileobj=reader, mode="r|") as t:
            for m in t:
                if m.isfile() and m.name.endswith(".json") and "/fhir/" in f"/{m.name}":
                    yield Path(m.name).name, t.extractfile(m).read()


def coding(cc: dict | None) -> tuple[str | None, str | None]:
    for c in (cc or {}).get("coding", []):
        return c.get("code"), c.get("display")
    return None, None


def flatten(name: str, raw: bytes, persons: list, conds: list, encs: list) -> None:
    b = json.loads(raw)
    pid = None
    for e in b.get("entry", []):
        r = e["resource"]
        if r["resourceType"] == "Patient":
            pid = r["id"]
            persons.append(
                {
                    "pid": pid,
                    "gender": r.get("gender"),
                    "birth": r.get("birthDate"),
                    "deceased": r.get("deceasedDateTime"),
                    "file": name,
                }
            )
            break
    if pid is None:
        return
    for e in b.get("entry", []):
        r = e["resource"]
        rt = r["resourceType"]
        if rt == "Condition":
            code, disp = coding(r.get("code"))
            cat = [c.get("code") for cat in r.get("category", []) for c in cat.get("coding", [])]
            conds.append(
                {
                    "pid": pid,
                    "code": code,
                    "display": disp,
                    "category": ",".join(x for x in cat if x),
                    "onset": r.get("onsetDateTime"),
                    "abatement": r.get("abatementDateTime"),
                }
            )
        elif rt == "Encounter":
            code, disp = coding((r.get("type") or [None])[0])
            encs.append(
                {
                    "pid": pid,
                    "start": (r.get("period") or {}).get("start"),
                    "class": (r.get("class") or {}).get("code"),
                    "type_code": code,
                    "type": disp,
                }
            )


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    persons, conds, encs = [], [], []
    for i, (name, raw) in enumerate(iter_bundles(src), 1):
        flatten(name, raw, persons, conds, encs)
        if i % 500 == 0:
            print(i, "bundles", file=sys.stderr, flush=True)
    pd.DataFrame(persons).to_parquet(out / "persons.parquet", index=False)
    pd.DataFrame(conds).to_parquet(out / "conditions.parquet", index=False)
    pd.DataFrame(encs).to_parquet(out / "encounters.parquet", index=False)
    print(f"{len(persons)} persons, {len(conds)} conditions, {len(encs)} encounters -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
