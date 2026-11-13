"""Screen candidate sensitive attributes on the development split only (cpu-prep 2; rough, for choosing).

Usage: python scripts/attribute_candidates.py <derived-dir> <exp-dir>
  reads <exp-dir>/config.json and SEED; checks the split csv sha256; uses only pids with split == 'dev';
  writes <exp-dir>/results/candidates.json and <exp-dir>/metrics.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def dev_pids(root: Path, cfg: dict) -> set[str]:
    f = root / cfg["split_csv"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != cfg["split_csv_sha256"]:
        raise SystemExit("split.csv sha256 mismatch")
    s = pd.read_csv(f)
    return set(s.pid[s.split == "dev"])  # the confirmation side is never read beyond this filter


def main(derived: Path, exp: Path) -> None:
    root = exp.resolve().parents[1]
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    dev = dev_pids(root, cfg)
    y0 = cfg["first_year"]
    lo, hi = cfg["age_range_at_first_year"]
    p = pd.read_parquet(derived / "persons.parquet")
    p = p[p.pid.isin(dev) & p.deceased.isna()].copy()
    p["age"] = y0 - pd.to_datetime(p.birth).dt.year
    p = p[p.age.between(lo, hi)].sort_values("pid")
    c = pd.read_parquet(derived / "conditions.parquet")
    c = c[c.pid.isin(set(p.pid))].copy()
    c["oy"] = pd.to_datetime(c.onset, utc=True, errors="coerce", format="ISO8601").dt.year
    c = c[c.oy < y0]
    mh = pd.crosstab(c.pid, c.display).clip(upper=1).reindex(p.pid).fillna(0)
    xk = np.column_stack([(p.age.to_numpy() - 47.5) / 13, (p.gender.to_numpy() == "male").astype(float)])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = {"n_dev_pool": int(len(p)), "candidates": {}}
    for name, pat in cfg["candidates"].items():
        cols = [d for d in mh.columns if pd.Series([d]).str.contains(pat, case=False, regex=True).iloc[0]]
        y = (mh[cols].sum(axis=1) > 0).astype(int).to_numpy()
        X = mh.drop(columns=cols)
        X = X.loc[:, X.sum(axis=0) >= cfg["min_support"]]
        rec = {"pattern": pat, "matched": cols, "n_A": int(y.sum()), "prevalence": round(float(y.mean()), 4)}
        if y.sum() >= 10:
            lr = LogisticRegression(max_iter=3000)
            pk = cross_val_predict(lr, xk, y, cv=cv, method="predict_proba")[:, 1]
            pr = cross_val_predict(lr, np.column_stack([xk, X.to_numpy()]), y, cv=cv, method="predict_proba")[:, 1]
            sup = X.sum(axis=0)
            lift = (X[y == 1].sum(axis=0) / sup) / y.mean()
            top = lift[sup >= cfg["lift_min_support"]].sort_values(ascending=False).head(6)
            rec.update({"auc_K": round(float(roc_auc_score(y, pk)), 3), "auc_KR": round(float(roc_auc_score(y, pr)), 3),
                        "top_lift": [{"display": d, "lift": round(float(v), 2), "support": int(sup[d])} for d, v in top.items()]})
        out["candidates"][name] = rec
    res = exp / "results"
    res.mkdir(exist_ok=True)
    (res / "candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    (exp / "metrics.json").write_text(json.dumps({k: {kk: v.get(kk) for kk in ("n_A", "prevalence", "auc_K", "auc_KR")}
                                                  for k, v in out["candidates"].items()}, indent=1), encoding="utf-8", newline="\n")
    for k, v in out["candidates"].items():
        print(k, v["n_A"], v["prevalence"], v.get("auc_K"), v.get("auc_KR"), [(t["display"][:40], t["lift"]) for t in v.get("top_lift", [])[:3]])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
