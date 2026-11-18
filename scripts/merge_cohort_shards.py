"""Join the four 10k shards of the confirmatory cohort into one 40k cohort (判断 34).

Usage:
  python scripts/merge_cohort_shards.py <out-dir> <shard-dir> [<shard-dir> ...]

Each <shard-dir> is a $WORK/derived-shard<k> from scripts/generate_confirm_cohort.sh confirm: persons / conditions /
encounters .parquet and the shard's manifest.json. Checks, and stops on the first failure:
  - every manifest is mode "confirm", population 10000, with the same recipe, jar, state, dates, pjCommit and profileCohortSha256 (javaVersion is only recorded);
  - the shards are exactly 1..4, each once (or the --expect count for tests);
  - the 8 seeds (SEED and CLINICIAN_SEED of every shard) are all different and none is a development or test seed;
  - no pid occurs in two shards (Synthea draws the patient ids from the seed; a clash would merge two people).
Writes <out-dir>/{persons,conditions,encounters}.parquet (rows in shard order 1..4, then the shard's own order),
<out-dir>/manifest.json (the shard manifests + the sha256 of every shard file) and prints the sha256 of the outputs.
A column "shard" is not added: the analysis must not see which shard a person came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

from permute_lambda import DEV_SEEDS

TABLES = ("persons", "conditions", "encounters")
SAME = ("recipe", "syntheaVersion", "jarSha256", "state", "referenceDate", "endDate", "pjCommit", "profileCohortSha256")  # javaVersion: recorded only
SHARD_POPULATION = 10000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_manifests(ms: list[dict], expect: int, population: int) -> None:
    for m in ms:
        missing = [k for k in ("shard", "seed", "clinicianSeed", "mode", "population", "pjCommit", "profileCohortSha256") if k not in m]
        if missing:
            sys.exit(f"merge: a shard manifest lacks {missing}")
        if m.get("pjCommit") in (None, "", "none"):
            sys.exit(f"merge: shard {m.get('shard')} has no pjCommit (generated outside a git work tree)")
        if m.get("mode") != "confirm":
            sys.exit(f"merge: shard {m.get('shard')} is mode {m.get('mode')!r}, not 'confirm'")
        if m.get("population") != population:
            sys.exit(f"merge: shard {m.get('shard')} has population {m.get('population')}, not {population}")
        for k in SAME:
            if m.get(k) != ms[0].get(k):
                sys.exit(f"merge: {k} differs between shards ({ms[0].get(k)!r} vs {m.get(k)!r})")
    shards = sorted(m.get("shard") for m in ms)
    if shards != list(range(1, expect + 1)):
        sys.exit(f"merge: shards must be 1..{expect}, each once; got {shards}")
    seeds = [m["seed"] for m in ms] + [m["clinicianSeed"] for m in ms]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"merge: the seeds of the shards must all differ; got {seeds}")
    dev = sorted(set(seeds) & set(DEV_SEEDS))
    if dev:
        sys.exit(f"merge: development or test seeds in the shards: {dev}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("shards", type=Path, nargs="+")
    ap.add_argument("--expect", type=int, default=4, help="number of shards (4 for the confirmatory cohort)")
    ap.add_argument("--population", type=int, default=SHARD_POPULATION, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if os.environ.get("HTS_PREREG_FROZEN") == "yes" and (a.expect != 4 or a.population != SHARD_POPULATION):
        sys.exit("merge: with HTS_PREREG_FROZEN=yes --expect and --population must be the defaults (4 and 10000)")
    if a.out.exists():
        sys.exit(f"merge: {a.out} already exists")
    ms = [json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in a.shards]
    check_manifests(ms, a.expect, a.population)
    order = sorted(range(len(ms)), key=lambda i: ms[i]["shard"])
    dirs = [a.shards[i] for i in order]
    frames = {t: [pd.read_parquet(d / f"{t}.parquet") for d in dirs] for t in TABLES}
    seen: dict[str, int] = {}
    for d, m, p in zip(dirs, (ms[i] for i in order), frames["persons"]):
        if p["pid"].duplicated().any():
            sys.exit(f"merge: duplicate pid inside shard {m['shard']}")
        for pid in p["pid"]:
            if pid in seen:
                sys.exit(f"merge: pid {pid} is in shard {seen[pid]} and shard {m['shard']}")
            seen[pid] = m["shard"]
    for t in ("conditions", "encounters"):
        for m, f in zip((ms[i] for i in order), frames[t]):
            stray = set(f["pid"]) - {k for k, v in seen.items() if v == m["shard"]}
            if stray:
                sys.exit(f"merge: {t} of shard {m['shard']} has pids not in its persons ({len(stray)})")
    a.out.mkdir(parents=True)
    for t in TABLES:
        pd.concat(frames[t], ignore_index=True).to_parquet(a.out / f"{t}.parquet", index=False)
    manifest = {
        "merged_by": "scripts/merge_cohort_shards.py (判断 34)",
        "population": a.population * len(ms),
        "shards": [ms[i] | {"files": {f.name: sha256(f) for f in sorted(a.shards[i].glob("*.parquet"))}} for i in order],
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for t in TABLES:
        print(sha256(a.out / f"{t}.parquet"), f"{t}.parquet")
    print(f"{len(seen)} persons in {len(ms)} shards -> {a.out}")


if __name__ == "__main__":
    main()
