"""Addendum B: run the frozen E2 map (scripts/e2_map.py, unchanged) on an addendum cohort with the addendum lists and a
fixed number of BLAS / OpenMP threads.

Why a wrapper: e2_map.main reads ROOT/config/condition_lists.json (the v4 lists, two attributes). Here ROOT is pointed at a
temporary directory whose config/condition_lists.json is a byte copy of --lists, so the map code itself is the frozen
one. --d2-from-config is always on (the D2 lists come from the development cohort, as in the confirmatory E2).
Threads: on the Colab CPU VM the default BLAS threads oversubscribe the 4 cores (2.3 times slower, measured 2026-11-17,
experiments/20261117-e2-confirm/notes.md) and the thread count changes the last floating-point digits, so it is fixed
(threadpoolctl, plus OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS for child libraries) and recorded.

Writes what e2_map writes (results/e2_map_reps.csv, e2_map_summary.csv, e2_lambda1_original.csv, metrics.json) plus
results/addendum_runtime.json (threads, threadpool_info, sha256 of the lists, of the exp config and of e2_map.py).

Confirmatory run (--confirm <cohort>): stops unless HTS_PREREG_FROZEN=yes, <exp-dir> does not exist yet, --threads equals
config/addendum.json B.threads, --lists is config/addendum_b_condition_lists.json, and <derived-dir>/addendum.json (written
by scripts/addendum_b_cohort.sh) names that cohort with the seed pair of config/addendum_seeds.json. It then creates
<exp-dir> with config.json = a byte copy of config/e2_confirm.json (v4's E2 settings) and SEED = the cohort's e2_seed of
config/addendum_seeds.json, so the run settings come only from frozen files. After a VM failure the cohort is run again
from the start with the same seed and threads into a new <exp-dir> (the output is deterministic); every attempt is reported.

Usage: python scripts/addendum_b_e2.py <derived-dir> <exp-dir> --lists config/addendum_b_condition_lists.json --threads 4
           [--confirm <1|2|3>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


ROOT = Path(__file__).resolve().parents[1]


def prepare_confirm(c: int, derived: Path, exp: Path, lists: Path, threads: int, root: Path = ROOT) -> dict:
    errs = []
    if os.environ.get("HTS_PREREG_FROZEN") != "yes":
        errs.append("HTS_PREREG_FROZEN=yes is not set")
    add = json.loads((root / "config" / "addendum.json").read_text(encoding="utf-8"))["B"]
    if threads != add["threads"]:
        errs.append(f"--threads {threads} differs from the registered {add['threads']}")
    if sha256(lists) != sha256(root / "config" / "addendum_b_condition_lists.json"):
        errs.append("--lists is not config/addendum_b_condition_lists.json")
    seeds = json.loads((root / "config" / "addendum_seeds.json").read_text(encoding="utf-8"))
    sc = seeds["cohorts"][c - 1]
    m = json.loads((derived / "addendum.json").read_text(encoding="utf-8"))
    if sc["cohort"] != c or (m["cohort"], m["seed"], m["clinicianSeed"]) != (c, sc["seed"], sc["clinicianSeed"]):
        errs.append(f"{derived}/addendum.json {m} is not cohort {c} of config/addendum_seeds.json")
    if exp.exists():
        errs.append(f"{exp} exists: a confirmatory attempt starts from an empty experiment directory")
    if errs:
        raise SystemExit("confirmatory checks failed:\n  " + "\n  ".join(errs))
    (exp / "results").mkdir(parents=True)
    shutil.copyfile(root / "config" / "e2_confirm.json", exp / "config.json")
    (exp / "SEED").write_text(f"{seeds['e2_seed'][str(c)]}\n", encoding="utf-8", newline="\n")
    return {"confirm": True, "cohort": c, "e2_seed": seeds["e2_seed"][str(c)],
            "seeds_sha256": sha256(root / "config" / "addendum_seeds.json"),
            "e2_confirm_sha256": sha256(root / "config" / "e2_confirm.json")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("derived", type=Path)
    ap.add_argument("exp", type=Path)
    ap.add_argument("--lists", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--confirm", type=int, choices=[1, 2, 3], default=None)
    a = ap.parse_args(argv)
    if a.threads < 1:
        raise SystemExit("--threads must be >= 1")
    conf = prepare_confirm(a.confirm, a.derived, a.exp, a.lists, a.threads) if a.confirm else {"confirm": False}
    import datetime
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v, str(a.threads)) != str(a.threads):
            raise SystemExit(f"{v}={os.environ[v]} differs from --threads {a.threads}")
        os.environ[v] = str(a.threads)
    from threadpoolctl import threadpool_info, threadpool_limits  # after the environment is set

    import e2_map
    with tempfile.TemporaryDirectory() as t, threadpool_limits(limits=a.threads):
        root = Path(t)
        (root / "config").mkdir()
        shutil.copyfile(a.lists, root / "config" / "condition_lists.json")
        info = [{k: x.get(k) for k in ("user_api", "internal_api", "num_threads", "version")} for x in threadpool_info()]
        old_root = e2_map.ROOT
        e2_map.ROOT = root
        try:
            e2_map.main(a.derived, a.exp, True)
        finally:
            e2_map.ROOT = old_root
    rt = {"threads": a.threads, "threadpool_info": info, "lists": str(a.lists), "lists_sha256": sha256(a.lists),
          "config_sha256": sha256(a.exp / "config.json"), "e2_map_sha256": sha256(Path(e2_map.__file__)),
          "derived": str(a.derived), "started_utc": started,
          "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()} | conf
    (a.exp / "results" / "addendum_runtime.json").write_text(json.dumps(rt, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
