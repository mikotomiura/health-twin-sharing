"""Correlation strength lambda (positioning §5): weaken the link between the hidden attribute A and the rest of the record.

With probability 1 - lambda each person in the population is selected; among the selected persons, the bundles of
"movable" condition rows (every condition that matches neither the A pattern nor D0) are permuted as whole bundles
(.steering/20261112-cpu-prep/decisions.md 判断 7). Rows keep their calendar onset / abatement, so each display's
person count and per-year counts are preserved exactly. lambda = 1 returns the input; lambda = 0 makes the movable
record independent of A (lift ~ 1 for every movable condition) while keeping the correlations among conditions.

Also holds the shared cohort helpers (population, A / not-A, D0 mask, lift table, the D2 rule) used by dev_lists.py
and e2_map.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def rx(patterns: list[str]) -> str:
    return "|".join(f"(?:{p})" for p in patterns) if patterns else r"(?!x)x"


def d0_mask(display: pd.Series, d0: dict) -> pd.Series:
    """True where the display is hidden by D0 (label and aliases)."""
    hit = display.str.contains(rx(d0["include"]), case=False, regex=True, na=False)
    if d0.get("exclude"):
        hit &= ~display.str.contains(rx(d0["exclude"]), case=False, regex=True, na=False)
    return hit


def check_a_within_d0(display: pd.Series, a_pattern: str, d0: dict) -> None:
    """Every display matching the A pattern must be hidden by D0; otherwise A itself would reach the release."""
    is_a = display.str.contains(a_pattern, case=False, regex=True, na=False)
    leak = sorted(set(display[is_a & ~d0_mask(display, d0)]))
    if leak:
        raise SystemExit(f"A-pattern displays not hidden by D0: {leak}")


def load_tables(derived: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(derived / "persons.parquet")
    c = pd.read_parquet(derived / "conditions.parquet")
    c["onset_year"] = pd.to_datetime(c["onset"], utc=True, errors="coerce", format="ISO8601").dt.year
    c["abate_year"] = pd.to_datetime(c["abatement"], utc=True, errors="coerce", format="ISO8601").dt.year
    p["birth_year"] = pd.to_datetime(p["birth"]).dt.year
    return p, c


def population(p: pd.DataFrame, c: pd.DataFrame, a_pattern: str, first_year: int,
               age_range: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persons alive at the end, age in range at first_year, A = A-pattern onset before first_year, not-A = never.
    Same rule as build_records.py. Returns (persons with column A, their condition rows)."""
    p = p[p["deceased"].isna()]
    p = p[(first_year - p["birth_year"]).between(*age_range)]
    is_a = c["display"].str.contains(a_pattern, case=False, regex=True, na=False)
    first = c[is_a].groupby("pid")["onset_year"].min()
    a = set(first[first < first_year].index)
    p = p[p.pid.isin(a) | ~p.pid.isin(set(first.index))].copy()
    p["A"] = p.pid.isin(a)
    return p.reset_index(drop=True), c[c.pid.isin(set(p.pid))].copy()


def permute(c: pd.DataFrame, pids: np.ndarray, fixed: pd.Series, lam: float, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of c in which the movable rows (~fixed) of a random 1 - lam share of `pids` are permuted
    as per-person bundles among those persons. `fixed` is a boolean Series aligned with c."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    out = c.copy()
    sel = pids[rng.random(len(pids)) < 1.0 - lam]
    if len(sel) < 2:
        return out
    target = dict(zip(sel, rng.permutation(sel)))
    move = (~fixed.to_numpy()) & out["pid"].isin(target).to_numpy()
    out.loc[move, "pid"] = out.loc[move, "pid"].map(target).to_numpy()
    return out


def lift_table(persons: pd.DataFrame, c: pd.DataFrame, first_year: int, hidden: pd.Series) -> pd.DataFrame:
    """lift = P(A | c) / P(A) over the population, for conditions with onset before first_year that D0 does not hide."""
    a = set(persons.pid[persons.A])
    n = len(persons)
    pa = len(a) / n
    b = c[(c.onset_year < first_year) & ~hidden.to_numpy()].drop_duplicates(["pid", "display"])
    g = b.groupby("display").pid.agg(n_c="nunique", n_joint=lambda s: len(set(s) & a))
    g["p_c"] = g.n_c / n
    g["p_A_given_c"] = g.n_joint / g.n_c
    g["lift"] = g.p_A_given_c / pa
    return g.sort_values(["lift", "n_c"], ascending=[False, False])


def d2_select(g: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """D2 (decisions 判断 9): conditions with n_c >= min, overlap with A >= min and lift >= min, from lift_table()."""
    return g[(g.n_c >= rule["min_n_condition"]) & (g.n_joint >= rule["min_n_joint"]) & (g.lift >= rule["min_lift"])]
