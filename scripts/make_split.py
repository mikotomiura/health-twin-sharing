"""Development / confirmation (held-out) split of the cohort (cpu-prep, researcher 2026-11-12).

WITHDRAWN - do not use. The 40/60 split was replaced by a new confirmatory cohort generated with a fresh seed
(.steering/20261112-cpu-prep/decisions.md, the decision on the split; scripts/generate_confirm_cohort.sh). Kept only as
the record of experiments/20261112-split/ (researcher 2026-11-12: keep the file, mark it withdrawn).

Each person in persons.parquet gets u = int(sha256('hts-split-<SEED>:<pid>')) / 2**256 and goes to 'dev' if
u < dev_share, else 'confirm'. Persons used in earlier dry runs (config 'force_dev_from') are forced to 'dev'.
The rule depends only on the pid, so any later population definition is a subset with the same sides.
Stratification by A is not enforced (it would make the side depend on the pool); the realised shares per stratum of
the reference pool are reported instead.

Usage: python scripts/make_split.py <derived-dir> <exp-dir>
  writes <exp-dir>/results/split.csv (pid,split,forced), <exp-dir>/results/split_summary.json, <exp-dir>/metrics.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_sample_size import build_pool  # noqa: E402  (same population definition as E2)


def u_of(pid: str, seed: str) -> float:
    return int(hashlib.sha256(f"hts-split-{seed}:{pid}".encode()).hexdigest(), 16) / 2**256


def main(derived: Path, exp: Path) -> None:
    root = exp.resolve().parents[1]
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = (exp / "SEED").read_text().strip()
    forced: set[str] = set()
    for f in cfg["force_dev_from"]:
        forced |= {json.loads(l)["pid"] for l in (root / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    p = pd.read_parquet(derived / "persons.parquet")[["pid"]].drop_duplicates().sort_values("pid")
    p["u"] = p.pid.map(lambda x: u_of(x, seed))
    p["forced"] = p.pid.isin(forced)
    p["split"] = ((p.u < cfg["dev_share"]) | p.forced).map({True: "dev", False: "confirm"})
    missing = forced - set(p.pid)
    if missing:
        raise SystemExit(f"forced pids not in persons.parquet: {sorted(missing)[:5]}")
    res = exp / "results"
    res.mkdir(exist_ok=True)
    csv = res / "split.csv"
    p[["pid", "split", "forced"]].to_csv(csv, index=False, lineterminator="\n")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()

    pool_cfg = json.loads((root / cfg["report_pool_config"]).read_text(encoding="utf-8"))
    pool, _ = build_pool(derived, pool_cfg, exclude=set())
    pool = pool.join(p.set_index("pid")[["split", "forced"]])
    by = {}
    for name, d in [("all", pool), ("A", pool[pool.A]), ("notA", pool[~pool.A])]:
        by[name] = {"n": int(len(d)), "dev": int((d.split == "dev").sum()), "confirm": int((d.split == "confirm").sum()),
                    "dev_share": round(float((d.split == "dev").mean()), 4), "forced_dev": int(d.forced.sum())}
    summary = {"persons": {"n": int(len(p)), "dev": int((p.split == "dev").sum()), "confirm": int((p.split == "confirm").sum()),
                           "forced_dev": int(p.forced.sum())},
               "reference_pool": {"config": cfg["report_pool_config"], **by},
               "split_csv_sha256": digest}
    (res / "split_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    (exp / "metrics.json").write_text(json.dumps({"split_csv_sha256": digest, "pool_dev_share": by["all"]["dev_share"],
                                                  "pool_A_dev": by["A"]["dev"], "pool_A_confirm": by["A"]["confirm"]}, indent=1),
                                      encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
