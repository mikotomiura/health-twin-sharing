"""E2 map: D x K x lambda (positioning §5; decisions 判断 7・10・11).

For each attribute in config/condition_lists.json, each lambda (scripts/permute_lambda.py) and each permutation rep:
  release R = multi-hot of conditions active in first_year (onset <= y0, not abated before y0), after the defender D:
    D0  drop displays hidden by D0 (the A pattern must lie inside D0; checked)
    D1  D0 + replace the D1 displays by their group column (attributes with a D1 list only)
    D2  D0 + replace the D2 displays (dev cohort, lambda = 1, d2_select rule 判断 9) by the D1 group if they have one,
        else by one 'a coarsened condition' column
  attacker knowledge K:
    K2  age, gender (the classifier learns per-condition rates from the labelled sample)
    K3  K2 + the original indicators (active in y0) of the D2 conditions, known from outside
  Three accuracy-type quantities from 5-fold out-of-fold logistic regression probabilities p:
    V_bal  = 1/2 TPR + 1/2 TNR, prediction 1[p > 0.5]   prior-equalised; PRIMARY (判断 25); 0.5 exactly is not a positive (判断 26);
             for a true posterior a lower bound of the balanced-gain g-vulnerability [70]
    V_acc  = mean 1[(p > 0.5) == A]        pi-weighted accuracy of acting at p > 0.5 (secondary); the Bayes vulnerability [17][18]
             when p is the true posterior
    V_conf = mean max(p, 1 - p)            plug-in; assumes p is calibrated (secondary)
  net increment dV = V(K + R) - V(K) for each (dV_bal primary, dV_acc / dV_conf secondary); also AUC and the Brier score.
  Two secondaries fixed in advance (判断 35; reported beside H2, no part of any decision): dV_opt = the net increment of the
  balanced accuracy of the best rule (act iff p > pi_hat, pi_hat = the mean of y in that rep's sample), and dAUC = auc_KR - auc_K.
  Utility cost over persons with >= 1 active condition: share of their active entries that D drops or coarsens
  (cost_total, includes D0) and the part added on top of D0 (cost_extra).
Person bootstrap (config "bootstrap_persons": true, 判断 18): in each rep the persons are drawn with replacement
(same number, strata A / not-A) BEFORE the lambda permutation and the CV, so that even lambda = 1 carries the sampling
noise of the cohort; the CV is StratifiedGroupKFold(5) with the original pid as group, so a person drawn twice never
sits in both the training and the validation part. The draws come from an rng seeded per (SEED, attribute, lambda, rep).
false (the default when the key is absent) leaves the values of the existing columns as in the earlier development runs
(the person bootstrap adds the draw and the group only). Confirmatory R = config "reps" = 200 (判断 28; the earlier 20 was the development setting).
This script is the entry of the confirmatory E2 (run with --d2-from-config on the confirmatory cohort, config/e2_confirm.json,
scripts/e2_summarise.py) and can still be used on the development cohort for dry runs.
lambda = 1 on the original sample (判断 38, only with bootstrap_persons true, primary attribute): results/e2_lambda1_original.csv holds
one CV run without resampling per D x K (rng seeded as rep 0 plus crc32("e2-original") so its series differs from rep 0; then the permutation call, then the cv seed); a description
beside the R-rep mean, which stays the judged value.
Caveat: with the D2 list chosen on the same persons the classifier is evaluated on (no --d2-from-config) the OOF values
describe that cohort and are not an independent evaluation; on the confirmatory cohort the list comes from the
development cohort (--d2-from-config).

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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from permute_lambda import (check_a_within_d0, check_confirm_seed, d0_mask, d2_select, lift_table, load_tables, permute,
                            population)

ROOT = Path(__file__).resolve().parents[1]
COARSE = "a coarsened condition"


def oof(X: np.ndarray, y: np.ndarray, seed: int, groups: np.ndarray | None = None) -> np.ndarray:
    """5-fold out-of-fold probabilities; with groups (bootstrap) no group is in a training and a validation part."""
    out = np.zeros(len(y))
    folds = (StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y) if groups is None
             else StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))
    for tr, te in folds:
        out[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def v_conf(p: np.ndarray) -> float:
    return float(np.maximum(p, 1 - p).mean())


def v_acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p > 0.5).astype(int) == y).mean())  # 判断 26: p = 0.5 exactly is not "A"


def v_bal(p: np.ndarray, y: np.ndarray, thr: float = 0.5) -> float:
    """Balanced accuracy 1/2 TPR + 1/2 TNR with the prediction 1[p > thr] (thr = 0.5: p = 0.5 exactly is predicted not-A)."""
    pred = p > thr
    return float(0.5 * pred[y == 1].mean() + 0.5 * (~pred[y == 0]).mean())


def v_opt(p: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy of the best rule for the balanced gain: act iff p > pi_hat, pi_hat = mean(y) of this sample
    (paper-write 判断 35; a secondary, not part of any decision)."""
    return v_bal(p, y, float(y.mean()))


def bootstrap_persons(persons: pd.DataFrame, c: pd.DataFrame, rng: np.random.Generator
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw len(persons) persons with replacement, stratified by A (the two strata keep their sizes). Each draw is a
    separate person with a new unique pid (b0, b1, ...); column orig_pid keeps the original pid (the CV group) and the
    condition rows are copied for every draw. Returns (drawn persons, their condition rows)."""
    ia, io = np.flatnonzero(persons.A.to_numpy()), np.flatnonzero(~persons.A.to_numpy())
    idx = np.concatenate([rng.choice(ia, len(ia)), rng.choice(io, len(io))])
    pb = persons.iloc[idx].reset_index(drop=True).rename(columns={"pid": "orig_pid"})
    pb["pid"] = [f"b{j}" for j in range(len(pb))]
    cb = c.rename(columns={"pid": "orig_pid"}).merge(pb[["orig_pid", "pid"]], on="orig_pid", how="inner")
    return pb, cb


def active(c: pd.DataFrame, y0: int) -> pd.DataFrame:
    """(pid, display) of conditions active in y0: onset <= y0 and not abated before y0 (NaN abatement = ongoing)."""
    a = c[(c.onset_year <= y0) & (c.abate_year.isna() | (c.abate_year >= y0))]
    return a.drop_duplicates(["pid", "display"])[["pid", "display"]]


def multihot(rows: pd.DataFrame, pids: pd.Index, min_count: int, orig: pd.Series | None = None) -> pd.DataFrame:
    """Persons x displays 0/1, rows in `pids` order; columns with fewer than min_count persons dropped. With `orig`
    (pid -> original pid, bootstrap) the count is of distinct original persons, so a person drawn many times does not
    pass the threshold alone; the columns kept still hold one entry per drawn person."""
    if rows.empty:
        return pd.DataFrame(index=pids)
    m = pd.crosstab(rows.pid, rows.display).clip(upper=1).reindex(pids).fillna(0)
    if orig is None:
        return m.loc[:, m.sum() >= min_count]
    o = orig.reindex(pids).to_numpy()
    n_orig = m.astype(bool).apply(lambda col: pd.Series(o[col.to_numpy()]).nunique())
    return m.loc[:, n_orig.to_numpy() >= min_count]


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


def fit_rows(name: str, lam: float, y: np.ndarray, groups: np.ndarray | None, k2: np.ndarray, k3: np.ndarray,
             act: pd.DataFrame, hidden: set, defenders: dict, pids: pd.Index, orig: pd.Series | None,
             min_count: int, cv_seed: int, rep: int | None = None) -> list[dict]:
    """One row per D x K of the estimates from the 5-fold OOF probabilities (the same CV split for K and K + R)."""
    rows = []
    for dname, mapping in defenders.items():
        xr = multihot(defend(act, hidden, mapping), pids, min_count, orig).to_numpy(float)
        cost_total, cost_extra = utility_cost(act, pids, hidden, mapping)
        for kname, xk in (("K2", k2), ("K3", k3)):
            pk = oof(xk, y, cv_seed, groups)
            pr = oof(np.column_stack([xk, xr]), y, cv_seed, groups)
            rows.append({"attribute": name, "lambda": lam} | ({} if rep is None else {"rep": rep}) | {"D": dname, "K": kname,
                         "Vconf_K": v_conf(pk), "Vconf_KR": v_conf(pr), "dV_conf": v_conf(pr) - v_conf(pk),
                         "Vbal_K": v_bal(pk, y), "Vbal_KR": v_bal(pr, y), "dV_bal": v_bal(pr, y) - v_bal(pk, y),
                         "Vacc_K": v_acc(pk, y), "Vacc_KR": v_acc(pr, y), "dV_acc": v_acc(pr, y) - v_acc(pk, y),
                         "dV_opt": v_opt(pr, y) - v_opt(pk, y),
                         "auc_K": float(roc_auc_score(y, pk)), "auc_KR": float(roc_auc_score(y, pr)),
                         "dAUC": float(roc_auc_score(y, pr)) - float(roc_auc_score(y, pk)),
                         "brier_KR": float(brier_score_loss(y, pr)),
                         "n_features_R": int(xr.shape[1]), "cost_total": cost_total, "cost_extra": cost_extra})
    return rows


def main(derived: Path, exp: Path, d2_from_config: bool = False) -> None:
    lists = json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    check_confirm_seed(seed)
    y0, ages, rule = lists["first_year"], lists["age_range_at_first_year"], lists["d2"]
    boot = bool(cfg.get("bootstrap_persons", False))
    p_all, c_all = load_tables(derived)
    rows, orig_rows = [], []
    for name, at in lists["attributes"].items():
        check_a_within_d0(c_all.display, at["A_pattern"], at["D0"])
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])  # shared by the reps unless boot
        persons0, c0 = population(p_all, c_all, at["A_pattern"], y0, ages)
        hidden = set(c0.display[d0_mask(c0.display, at["D0"])])

        def row_fixed(cc: pd.DataFrame, at=at) -> pd.Series:  # rows the permutation never moves: D0 displays and A rows
            return d0_mask(cc.display, at["D0"]) | cc.display.str.contains(at["A_pattern"], case=False, regex=True, na=False)
        fixed0 = row_fixed(c0)
        if d2_from_config:
            d2set = list(lists["D2_frozen"][name]["displays"])
        else:
            d2set = list(d2_select(lift_table(persons0, c0, y0, d0_mask(c0.display, at["D0"])), rule).index)
        d1map = {d["display"]: d["group"] for d in (at.get("D1") or [])}
        defenders = {"D0": {}, "D2": {d: d1map.get(d, COARSE) for d in d2set}}
        if d1map:
            defenders["D1"] = d1map
        for lam in cfg["lambdas"]:
            for r in range(cfg["reps"]):
                if boot:  # per-rep rng: the draws of a rep do not depend on the other reps or on the config order
                    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), zlib.crc32(b"e2-boot"),
                                                 int(round(float(lam) * 1000)), r])
                    persons, c = bootstrap_persons(persons0, c0, rng)
                    groups = persons.orig_pid.map({p: i for i, p in enumerate(persons0.pid)}).to_numpy()
                else:
                    persons, c, groups = persons0, c0, None
                fixed = fixed0 if not boot else row_fixed(c)
                orig = persons.set_index("pid").orig_pid if boot else None
                pids = pd.Index(persons.pid)
                y = persons.A.to_numpy(int)
                age = y0 - persons.birth_year.to_numpy()
                k2 = np.column_stack([(age - age.mean()) / age.std(), (persons.gender == "male").to_numpy(float)])
                cp = permute(c, persons.pid.to_numpy(), fixed, lam, rng)
                act = active(cp, y0)
                k3 = np.column_stack([k2, multihot(act[act.display.isin(d2set)], pids, 1).to_numpy(float)])
                cv_seed = int(rng.integers(2**31))
                rows += fit_rows(name, lam, y, groups, k2, k3, act, hidden, defenders, pids, orig,
                                 cfg["min_feature_count"], cv_seed, r)
                print(name, lam, r, flush=True)
        if boot and name == cfg.get("primary_attribute") and any(float(l) == 1.0 for l in cfg["lambdas"]):
            # C18 (判断 38): the lambda = 1 cells once on the ORIGINAL sample (no resampling); its own rng series (the rep-0 key plus
            # crc32("e2-original"), so it does not repeat rep 0's draws), then the permutation call, then the cv seed;
            # a description beside the R-rep mean
            rng = np.random.default_rng([seed, zlib.crc32(name.encode()), zlib.crc32(b"e2-boot"), 1000, 0, zlib.crc32(b"e2-original")])
            y = persons0.A.to_numpy(int)
            age = y0 - persons0.birth_year.to_numpy()
            k2 = np.column_stack([(age - age.mean()) / age.std(), (persons0.gender == "male").to_numpy(float)])
            pids0 = pd.Index(persons0.pid)
            cp = permute(c0, persons0.pid.to_numpy(), fixed0, 1.0, rng)
            act = active(cp, y0)
            k3 = np.column_stack([k2, multihot(act[act.display.isin(d2set)], pids0, 1).to_numpy(float)])
            cv_seed = int(rng.integers(2**31))
            orig_rows += [rr | {"cv_seed": cv_seed} for rr in fit_rows(name, 1.0, y, None, k2, k3, act, hidden, defenders,
                                                                        pids0, None, cfg["min_feature_count"], cv_seed)]
    df = pd.DataFrame(rows)
    res = exp / "results"
    res.mkdir(exist_ok=True)
    df.to_csv(res / "e2_map_reps.csv", index=False, lineterminator="\n")
    if orig_rows:
        pd.DataFrame(orig_rows).to_csv(res / "e2_lambda1_original.csv", index=False, lineterminator="\n")
    summ = (df.groupby(["attribute", "lambda", "D", "K"])
              .agg(dV_bal=("dV_bal", "mean"), dV_bal_sd=("dV_bal", "std"),
                   Vbal_KR=("Vbal_KR", "mean"), Vbal_K=("Vbal_K", "mean"),
                   dV_conf=("dV_conf", "mean"), dV_conf_sd=("dV_conf", "std"),
                   dV_acc=("dV_acc", "mean"), dV_acc_sd=("dV_acc", "std"),
                   dV_opt=("dV_opt", "mean"), dV_opt_sd=("dV_opt", "std"), dAUC=("dAUC", "mean"), dAUC_sd=("dAUC", "std"),
                   Vconf_KR=("Vconf_KR", "mean"), Vacc_KR=("Vacc_KR", "mean"), Vacc_K=("Vacc_K", "mean"),
                   auc_K=("auc_K", "mean"), auc_KR=("auc_KR", "mean"), brier_KR=("brier_KR", "mean"),
                   cost_total=("cost_total", "mean"), cost_extra=("cost_extra", "mean"), reps=("rep", "count"))
              .round(4).reset_index())
    summ.to_csv(res / "e2_map_summary.csv", index=False, lineterminator="\n")
    keys = ["dV_bal", "dV_bal_sd", "dV_conf", "dV_conf_sd", "dV_acc", "dV_acc_sd", "dV_opt", "dV_opt_sd", "dAUC", "dAUC_sd", "auc_K", "auc_KR", "brier_KR", "cost_total", "cost_extra"]
    metrics = {f"{r.attribute}|lam={r['lambda']}|{r.D}|{r.K}": {k: float(r[k]) for k in keys} for _, r in summ.iterrows()}
    metrics["_note"] = ("E2 map; person bootstrap " + ("on" if boot else "off") + "; D2 list "
                        + ("frozen from the development cohort" if d2_from_config else
                           "chosen on the same persons (descriptive, not an independent evaluation)")
                        + "; whether this is a confirmatory run is decided by the cohort and the frozen config")
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--d2-from-config"):
        raise SystemExit(__doc__.split("Usage: ")[1])
    main(Path(sys.argv[1]), Path(sys.argv[2]), len(sys.argv) == 4)
