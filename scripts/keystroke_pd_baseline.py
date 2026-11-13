"""Dry run for the keystroke path: window summaries of neuroQWERTY and a single-path PD classifier.

Stages (a stand-in for the KeepCount stages [29]: time -> amount -> summary):
  1. time:    per-key press/release times from the raw CSV (mouse, long meta keys, BackSpace removed,
              as in nqDataLoader.getDataFiltHelper; hold times outside (0, 5] s dropped)
  2. amount:  non-overlapping windows of WINDOW_S seconds of press time; windows with < MIN_KEYS kept keys dropped
  3. summary: per-window hold-time and flight-time quantiles, keys per minute

Classifier: logistic regression on window summaries, grouped leave-one-subject-out CV
(subject score = mean window probability). AUC with a subject-level bootstrap CI.
Only the first file of each subject is used (CS1PD has two sessions; one session keeps subjects comparable).

Usage: python scripts/keystroke_pd_baseline.py <experiment-dir>   (reads config.json and SEED there)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(os.environ.get("HTS_DATA_DIR", Path.home() / "data" / "health-twin-sharing"))
NQ = DATA_DIR / "raw" / "neuroqwerty"
DROP = ("mouse", "Shift", "Alt", "Control", "BackSpace")
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def load_keys(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["key", "hold", "release", "press"])
    df["key"] = df["key"].astype(str).str.strip('"')
    df = df[~df["key"].str.startswith(DROP)]
    df = df[(df["hold"] > 0) & (df["hold"] <= 5)].sort_values("press").reset_index(drop=True)
    return df


def summarise(df: pd.DataFrame, window_s: float, min_keys: int) -> pd.DataFrame:
    flight = df["press"].shift(-1) - df["release"]
    df = df.assign(flight=flight.where((flight > -1) & (flight < 3)))
    df["w"] = (df["press"] // window_s).astype(int)
    rows = []
    for w, g in df.groupby("w"):
        if len(g) < min_keys:
            continue
        r = {"window": int(w), "n_keys": len(g), "keys_per_min": len(g) / window_s * 60}
        for q in QS:
            r[f"hold_q{int(q*100)}"] = g["hold"].quantile(q)
            r[f"flight_q{int(q*100)}"] = g["flight"].quantile(q)
        r["hold_iqr"] = r["hold_q75"] - r["hold_q25"]
        rows.append(r)
    return pd.DataFrame(rows)


def main(exp: Path) -> None:
    cfg = json.loads((exp / "config.json").read_text())
    seed = int((exp / "SEED").read_text().strip())
    rng = np.random.default_rng(seed)
    out = exp / "results"
    out.mkdir(exist_ok=True)

    subj, wins = [], []
    for ds in ("MIT-CS1PD", "MIT-CS2PD"):
        gt = pd.read_csv(NQ / ds / f"GT_DataPD_{ds}.csv")
        for _, s in gt.iterrows():
            keys = load_keys(NQ / ds / f"data_{ds}" / s["file_1"])
            w = summarise(keys, cfg["window_s"], cfg["min_keys"])
            sid = f"{ds}:{s['pID']}"
            subj.append({"sid": sid, "dataset": ds, "pd": bool(s["gt"]), "n_keys": len(keys),
                         "duration_s": float(keys["press"].max() - keys["press"].min()), "n_windows": len(w)})
            if len(w):
                wins.append(w.assign(sid=sid, pd=bool(s["gt"])))
    subj = pd.DataFrame(subj)
    wins = pd.concat(wins, ignore_index=True)
    subj.to_csv(out / "subjects.csv", index=False, lineterminator="\n")
    wins.describe().T.to_csv(out / "window_features_describe.csv", lineterminator="\n")

    feats = [c for c in wins.columns if c.startswith(("hold_", "flight_", "keys_per_min"))]
    X = wins[feats].fillna(wins[feats].median()).to_numpy()
    y = wins["pd"].to_numpy().astype(int)
    groups = wins["sid"].to_numpy()

    scores = {}
    for sid in np.unique(groups):
        tr, te = groups != sid, groups == sid
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=2000))
        clf.fit(X[tr], y[tr])
        scores[sid] = float(clf.predict_proba(X[te])[:, 1].mean())
    s = subj[subj.sid.isin(scores)].assign(score=lambda d: d.sid.map(scores))
    auc = roc_auc_score(s["pd"], s["score"])
    boots = []
    for _ in range(cfg["n_boot"]):
        b = s.sample(len(s), replace=True, random_state=int(rng.integers(2**31)))
        if b["pd"].nunique() == 2:
            boots.append(roc_auc_score(b["pd"], b["score"]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    s.to_csv(out / "subject_scores.csv", index=False, lineterminator="\n")

    metrics = {
        "n_subjects": int(len(subj)),
        "n_subjects_with_windows": int(len(s)),
        "n_pd": int(s["pd"].sum()),
        "n_windows": int(len(wins)),
        "windows_per_subject_median": float(subj["n_windows"].median()),
        "windows_per_subject_min": int(subj["n_windows"].min()),
        "keys_per_window_median": float(wins["n_keys"].median()),
        "keys_per_window_q10": float(wins["n_keys"].quantile(0.1)),
        "session_duration_s_median": float(subj["duration_s"].median()),
        "loso_subject_auc": float(auc),
        "loso_subject_auc_ci95": [float(lo), float(hi)],
        "features": feats,
    }
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
