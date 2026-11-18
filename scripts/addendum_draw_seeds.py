"""Addendum: draw the seeds at the addendum freeze (once), the same way as v4 (secrets.randbelow(2**31 - 1) in order).

Draws, in this order: for each addendum B cohort c = 1..3 a Synthea seed and a clinician seed, then one E2 SEED per cohort.
A value equal to a seed already in use (development and test values, every experiments/*/SEED as in v4, v4 confirmatory
cohort shards and clinician seeds, E1 and E2 SEEDs) or to a value drawn earlier in this run is drawn again; the redraws
are recorded.
Addendum A uses no new seed (the attacker seed is E1's SEED + 7, as in E1).

Usage: python scripts/addendum_draw_seeds.py --cohort-manifest <cohort-confirm/manifest.json> --e1-seed 1650422369
           --e2-seed 707983363 [--out config/addendum_seeds.json]
"""
from __future__ import annotations

import argparse
import datetime
import json
import secrets
from pathlib import Path

DEV_TEST = [42, 43, 44, 9001, 9002, 20261113, 20261117]  # 20261117: addendum B candidates dev E2 (nested SEED)
N_COHORTS = 3


def experiment_seeds(exp_dir: Path | None) -> set[int]:
    if exp_dir is None:
        return set()
    out = set()
    for f in sorted(exp_dir.glob("*/SEED")):
        t = f.read_text(encoding="utf-8").strip()
        if t.lstrip("-").isdigit():
            out.add(int(t))
    return out


def used_seeds(manifest: dict, e1: int, e2: int, exp_dir: Path | None = None) -> set[int]:
    cs = [int(x) for sh in manifest["shards"] for x in (sh["seed"], sh["clinicianSeed"])]
    if len(cs) != 8:
        raise SystemExit("the confirmatory cohort manifest must hold 8 seeds (4 shards)")
    return set(DEV_TEST) | set(cs) | {int(e1), int(e2)} | experiment_seeds(exp_dir)


def draw(used: set[int], rand=lambda: secrets.randbelow(2**31 - 1)) -> dict:
    taken, redraws = set(used), []

    def one(label: str) -> int:
        while True:
            v = rand()
            if v not in taken:
                taken.add(v)
                return v
            redraws.append({"label": label, "value": v})
    out = {"cohorts": [{"cohort": c, "seed": one(f"c{c}.seed"), "clinicianSeed": one(f"c{c}.clinicianSeed")}
                       for c in range(1, N_COHORTS + 1)]}
    out["e2_seed"] = {str(c): one(f"c{c}.e2") for c in range(1, N_COHORTS + 1)}
    out["redraws"] = redraws
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-manifest", type=Path, required=True)
    ap.add_argument("--e1-seed", type=int, required=True)
    ap.add_argument("--e2-seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "addendum_seeds.json")
    ap.add_argument("--experiments", type=Path, default=Path(__file__).resolve().parents[1] / "experiments")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: the seeds are drawn once (at the addendum freeze)")
    used = used_seeds(json.loads(a.cohort_manifest.read_text(encoding="utf-8")), a.e1_seed, a.e2_seed, a.experiments)
    out = draw(used) | {"drawn_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "method": "secrets.randbelow(2**31 - 1), in the order of this file; equal to a used seed -> drawn again",
                        "excluded": sorted(used)}
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
