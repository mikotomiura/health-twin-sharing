"""E2 map prototype on the development cohort: D x K x lambda (positioning §5; decisions 判断 7・10・11).

For each attribute in config/condition_lists.json, each lambda (scripts/permute_lambda.py) and each permutation rep:
  release R = multi-hot of conditions active in first_year (onset <= y0, not abated before y0), after the defender D:
    D0  drop displays hidden by D0 (the A pattern must lie inside D0; checked)
    D1  D0 + replace the D1 displays by their group column (attributes with a D1 list only)
    D2  D0 + replace the D2 displays (dev cohort, lambda = 1, d2_select rule 判断 9) by the D1 group if they have one,
        else by one 'a coarsened condition' column
  attacker knowledge K:
    K2  age, gender (the classifier learns per-condition rates from the labelled sample)
    K3  K2 + the original indicators (active in y0) of the D2 conditions, known from outside
  Two estimates of the Bayes vulnerability [17][18] from 5-fold out-of-fold logistic regression probabilities p:
    V_conf = mean max(p, 1 - p)            plug-in; assumes p is calibrated
    V_acc  = mean 1[(p >= 0.5) == A]       realised success rate of the attacker's best guess
  net increment dV = V(K + R) - V(K) for each; also AUC and the Brier score (calibration check).
  Utility cost over persons with >= 1 active condition: share of their active entries that D drops or coarsens
  (cost_total, includes D0) and the part added on top of D0 (cost_extra).
Caveat: the D2 list is chosen on the same persons the classifier is evaluated on, so the OOF values describe the
development cohort and are not an independent evaluation (on the confirmatory cohort the list comes from this one).
Preliminary (development cohort only); not for confirmatory analysis.

Usage: python scripts/e2_map.py <derived-dir> <exp-dir> [--d2-from-config]   (reads <exp-dir>/config.json, SEED; writes results/)
  --d2-from-config  use the D2 lists frozen from the development cohort (config/condition_lists.json "D2_frozen";
                    scripts/freeze_d2.py) instead of choosing them on this cohort. Required on the confirmatory cohort.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from permute_lambda import check_a_within_d0, d0_mask, d2_select, lift_table, load_tables, permute, population

ROOT = Path(__file__).resolve().parents[1]
COARSE = "a coarsened condition"


def oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        out[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def v_conf(p: np.ndarray) -> float:
    return float(np.maximum(p, 1 - p).mean())


def v_acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p >= 0.5).astype(int) == y).mean())


def active(c: pd.DataFrame, y0: int) -> pd.DataFrame:
    """(pid, display) of conditions active in y0: onset <= y0 and not abated before y0 (NaN abatement = ongoing)."""
    a = c[(c.onset_year <= y0) & (c.abate_year.isna() | (c.abate_year >= y0))]
    return a.drop_duplicates(["pid", "display"])[["pid", "display"]]


def multihot(rows: pd.DataFrame, pids: pd.Index, min_count: int) -> pd.DataFrame:
    """Persons x displays 0/1, rows in `pids` order; columns with fewer than min_count persons dropped."""
    if rows.empty:
        return pd.DataFrame(index=pids)
    m = pd.crosstab(rows.pid, rows.display).clip(upper=1).reindex(pids).fillna(0)
    return m.loc[:, m.sum() >= min_count]


def defend(act: pd.DataFrame, hidden: set, mapping: dict) -> pd.DataFrame:
    """Drop hidden displays, rename coarsened ones to their group, and de-duplicate per person."""
    a = act[~act.display.isin(hidden)].copy()
    a["display"] = a.display.map(lambda d: mapping.get(d, d))
    return a.drop_duplicates(["pid", "display"])


def utility_cost(act: pd.DataFrame, pids: pd.Index, hidden: set, mapping: dict) -> tuple[float, float]:
    """Mean over persons with >= 1 active entry of (dropped or coarsened) / all and (coarsened only) / all."""
    n = act.groupby("pid").size().reindex(pids).fillna(0)
    drop = act[act.display.isin(hidden)].groupby("pid").size().reindex(pids).fillna(0)
    coarse = act[~act.display.isin(hidden) & act.display.isin(mapping)].groupby("pid").size().reindex(pids).fillna(0)
    has = n > 0
    return float(((drop + coarse)[has] / n[has]).mean()), float((coarse[has] / n[has]).mean())


def main(derived: Path, exp: Path, d2_from_config: bool = False) -> None:
    lists = json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    y0, ages, rule = lists["first_year"], lists["age_range_at_first_year"], lists["d2"]
    p_all, c_all = load_tables(derived)
    rows = []
    for name, at in lists["attributes"].items():
        check_a_within_d0(c_all.display, at["A_pattern"], at["D0"])
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        persons, c = population(p_all, c_all, at["A_pattern"], y0, ages)
        hid_rows = d0_mask(c.display, at["D0"])
        fixed = hid_rows | c.display.str.contains(at["A_pattern"], case=False, regex=True, na=False)
        hidden = set(c.display[hid_rows])
        if d2_from_config:
            d2set = list(lists["D2_frozen"][name]["displays"])
        else:
            d2set = list(d2_select(lift_table(persons, c, y0, hid_rows), rule).index)
        d1map = {d["display"]: d["group"] for d in (at.get("D1") or [])}
        defenders = {"D0": {}, "D2": {d: d1map.get(d, COARSE) for d in d2set}}
        if d1map:
            defenders["D1"] = d1map
        pids = pd.Index(persons.pid)
        y = persons.A.to_numpy(int)
        age = y0 - persons.birth_year.to_numpy()
        k2 = np.column_stack([(age - age.mean()) / age.std(), (persons.gender == "male").to_numpy(float)])
        for lam in cfg["lambdas"]:
            for r in range(cfg["reps"]):
                cp = permute(c, persons.pid.to_numpy(), fixed, lam, rng)
                act = active(cp, y0)
                k3 = np.column_stack([k2, multihot(act[act.display.isin(d2set)], pids, 1).to_numpy(float)])
                cv_seed = int(rng.integers(2**31))
                for dname, mapping in defenders.items():
                    xr = multihot(defend(act, hidden, mapping), pids, cfg["min_feature_count"]).to_numpy(float)
                    cost_total, cost_extra = utility_cost(act, pids, hidden, mapping)
                    for kname, xk in (("K2", k2), ("K3", k3)):
                        pk, pr = oof(xk, y, cv_seed), oof(np.column_stack([xk, xr]), y, cv_seed)
                        rows.append({"attribute": name, "lambda": lam, "rep": r, "D": dname, "K": kname,
                                     "Vconf_K": v_conf(pk), "Vconf_KR": v_conf(pr), "dV_conf": v_conf(pr) - v_conf(pk),
                                     "Vacc_K": v_acc(pk, y), "Vacc_KR": v_acc(pr, y), "dV_acc": v_acc(pr, y) - v_acc(pk, y),
                                     "auc_K": float(roc_auc_score(y, pk)), "auc_KR": float(roc_auc_score(y, pr)),
                                     "brier_KR": float(brier_score_loss(y, pr)),
                                     "n_features_R": int(xr.shape[1]), "cost_total": cost_total, "cost_extra": cost_extra})
                print(name, lam, r, flush=True)
    df = pd.DataFrame(rows)
    res = exp / "results"
    res.mkdir(exist_ok=True)
    df.to_csv(res / "e2_map_reps.csv", index=False, lineterminator="\n")
    summ = (df.groupby(["attribute", "lambda", "D", "K"])
              .agg(dV_conf=("dV_conf", "mean"), dV_conf_sd=("dV_conf", "std"),
                   dV_acc=("dV_acc", "mean"), dV_acc_sd=("dV_acc", "std"),
                   Vconf_KR=("Vconf_KR", "mean"), Vacc_KR=("Vacc_KR", "mean"), Vacc_K=("Vacc_K", "mean"),
                   auc_K=("auc_K", "mean"), auc_KR=("auc_KR", "mean"), brier_KR=("brier_KR", "mean"),
                   cost_total=("cost_total", "mean"), cost_extra=("cost_extra", "mean"), reps=("rep", "count"))
              .round(4).reset_index())
    summ.to_csv(res / "e2_map_summary.csv", index=False, lineterminator="\n")
    keys = ["dV_conf", "dV_conf_sd", "dV_acc", "dV_acc_sd", "auc_K", "auc_KR", "brier_KR", "cost_total", "cost_extra"]
    metrics = {f"{r.attribute}|lam={r['lambda']}|{r.D}|{r.K}": {k: float(r[k]) for k in keys} for _, r in summ.iterrows()}
    metrics["_note"] = ("preliminary, development cohort only; D2 list chosen on the same persons (descriptive, "
                        "not an independent evaluation); not for confirmatory analysis")
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--d2-from-config"):
        raise SystemExit(__doc__.split("Usage: ")[1])
    main(Path(sys.argv[1]), Path(sys.argv[2]), len(sys.argv) == 4)
