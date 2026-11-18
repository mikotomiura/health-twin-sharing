"""Power of the E1 decision rule (H1 supported?) by simulation (preregistration-draft v4: OSF items Statistical models, Inference criteria, Sample size; decision 17).

Rule (current, 判断 22, default --estimator balanced): supported = Holm-adjusted one-sided p <= 0.025 only; est > delta_E1 (0.03)
is reported as a description. The older rule below (p <= 0.025 AND est > delta_E1, 判断 17) belongs to --estimator pi (legacy,
kept to reproduce 20261113-e1-power) only. Holm partner = H2 (E2), p ~ 0, so
H1 is tested at alpha = 0.025 (case a) or, when H1's p is the smaller one, at alpha/2 = 0.0125 (case b). Both are reported.

Model (all simplifications are in the experiment's notes.md):
  person i in a group (A / not-A) has latent action rate r ~ Beta(mean m, ICC rho) per condition (K2 and K0 independent);
  t3 value = mean of 6 Bernoulli(r) reads; t0 = 1 Bernoulli read of the person's K0 rate, drawn separately for each condition.
  V = pi*mean_A(act) + (1-pi)*(1-mean_notA(act)); dV = V(t3)-V(t0); H1 contrast = dV(K2)-dV(K0), paired by person.
  Interval: bootstrap over persons stratified by A (multinomial counts), one-sided p = (b + 1) / (B + 1), b = draws with contrast <= 0 (balanced; the older pi estimator: share of draws <= 0);
  with --test permutation the p is the sign-flip permutation p instead (see below).

Estimator (--estimator, decisions 25/26/22):
  pi       (legacy, default grid = old one): V weights A by pi; support = p <= alpha AND est > delta_E1. Reproduces 20261113-e1-power.
  balanced (default): V = 1/2*TPR + 1/2*TNR, dV = 1/2*dTPR - 1/2*dFPR; support = p <= alpha only (decision 22);
           "est > delta" (0.03, 0.05) is reported as a description (share_est_gt_0.03 / _0.05), not as part of the decision.
  Action = Bernoulli draw of the latent rate (so p > 0.5 vs >= 0.5 makes no difference here).

Size (判断 38, balanced only): extra rows with scenario "size" have no effect: m_A = m_N = the K0 rate (SIZE_M = 0.0, 0.07, 0.12), so the
true contrast is 0. Their "support_alpha*" is the share of simulations with p <= alpha, i.e. the size of the rule (the rate of a false
"H1 supported"); it is repeated in size_alpha0.025 / size_alpha0.0125 (empty for the power rows, scenario "power").

Test (--test, balanced only, 判断 39): bootstrap (default, reproduces the earlier experiments) or permutation = the sign-flip permutation test of
e1_summarise.sign_flip_p on the per-person contrasts (B = --boot flips, shared across the simulations of a chunk, p = (b + 1) / (B + 1)); the
size rows (m_A = m_N = K0 rate) and the power rows are both produced under either test.

Size under three nulls (--size-nulls, attack-me round 3 A2; balanced only, both tests): the rows of SIZE_NULLS x RHOS at one n per stratum,
each simulation with its own sign matrix / bootstrap draw (independent=True), written to results/size_nulls.csv with the Clopper-Pearson
95% interval of each size and exceeds_alpha* = (lower bound > alpha), the criterion fixed before the run (.steering/20261114-prereg-v4/design.md).
  exch   : the sharp null of the permutation test (K2 and K0 exchangeable within a person): one t3 rate and one t0 rate per person, shared by
           K2 and K0, the reads drawn independently from them (person_contrasts_exch); c_i is symmetric about 0.
  mean0  : person_contrasts as above with m_A = m_N = K0 rate (the "size" rows of run()); mean 0, not necessarily symmetric.
  k0zero : person_contrasts with K0 rate 0 and m_A = m_N (K2 moves A and not-A together); c_i >= 0, mean contrast 0.

Usage: python scripts/e1_power.py <out-dir> [--estimator balanced|pi] [--test bootstrap|permutation] [--n-sims 1000] [--boot 2000] [--ns 16,24,32]
       python scripts/e1_power.py <out-dir> --size-nulls [--n-sims 10000] [--boot 2000] [--ns 40]
Reads <out-dir>/SEED; writes <out-dir>/results/power.csv (or results/size_nulls.csv).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e1_summarise import sign_flip_p  # noqa: E402

PI = 0.159
DELTA = 0.03
N_READS = 6
M_A = (0.19, 0.30, 0.44, 0.69)
M_N = (0.0, 0.02, 0.05, 0.10)
K0_RATES = (0.0, 0.02)
RHOS = (0.3, 0.7)
ALPHAS = (0.025, 0.0125)
# balanced-estimator grid (scenarios from preliminary values; see 20261113-e1-novisits-check)
BAL_M_A = (0.19, 0.25, 0.33, 0.44)
BAL_M_N = (0.0, 0.03, 0.07, 0.12)
BAL_DELTAS = (0.03, 0.05)
SIZE_M = (0.0, 0.07, 0.12)  # no-effect scenarios: m_A = m_N = K0 rate (balanced estimator only)
# --size-nulls (attack-me round 3 A2): (null, m_A, m_N, K0 rate); every row has true contrast 0
SIZE_NULLS = (
    ("exch", 0.07, 0.07, 0.0), ("exch", 0.07, 0.07, 0.07), ("exch", 0.12, 0.12, 0.0), ("exch", 0.12, 0.12, 0.12), ("exch", 0.33, 0.07, 0.0),
    ("mean0", 0.07, 0.07, 0.07), ("mean0", 0.12, 0.12, 0.12),
    ("k0zero", 0.07, 0.07, 0.0), ("k0zero", 0.12, 0.12, 0.0), ("k0zero", 0.33, 0.33, 0.0),
)
NULL_CODE = {"exch": 1, "mean0": 2, "k0zero": 3}
TEST_CODE = {"permutation": 1, "bootstrap": 2}


def draw_rates(rng: np.random.Generator, m: float, rho: float, shape) -> np.ndarray:
    """Latent per-person rates ~ Beta with mean m and ICC rho (a+b = 1/rho - 1); m = 0 or 1 is degenerate."""
    if m <= 0.0:
        return np.zeros(shape)
    if m >= 1.0:
        return np.ones(shape)
    s = 1.0 / rho - 1.0
    return rng.beta(m * s, (1 - m) * s, size=shape)


def person_contrasts(rng, n_sims: int, n: int, m_k2: float, k0: float, rho: float) -> np.ndarray:
    """(n_sims, n) per-person (dV contribution before weighting) contrast: (a3_K2 - a0_K2) - (a3_K0 - a0_K0)."""
    r2 = draw_rates(rng, m_k2, rho, (n_sims, n))
    r0 = draw_rates(rng, k0, rho, (n_sims, n))
    a3_k2 = rng.binomial(N_READS, r2) / N_READS
    a3_k0 = rng.binomial(N_READS, r0) / N_READS
    a0_k2 = rng.binomial(1, r0).astype(float)
    a0_k0 = rng.binomial(1, r0).astype(float)
    return (a3_k2 - a0_k2) - (a3_k0 - a0_k0)


def person_contrasts_exch(rng, n_sims: int, n: int, m3: float, k0: float, rho: float) -> np.ndarray:
    """Sharp null (K2 and K0 exchangeable within a person): one t3 rate r3 and one t0 rate r0 per person, shared by both conditions,
    each condition's reads drawn independently from them. Same contrast as person_contrasts; symmetric about 0."""
    r3 = draw_rates(rng, m3, rho, (n_sims, n))
    r0 = draw_rates(rng, k0, rho, (n_sims, n))
    a3_k2 = rng.binomial(N_READS, r3) / N_READS
    a3_k0 = rng.binomial(N_READS, r3) / N_READS
    a0_k2 = rng.binomial(1, r0).astype(float)
    a0_k0 = rng.binomial(1, r0).astype(float)
    return (a3_k2 - a0_k2) - (a3_k0 - a0_k0)


def flip_p_rows(c: np.ndarray, nA: int, signs: np.ndarray, wA: float = 0.5, wN: float = 0.5) -> np.ndarray:
    """sign_flip_p with a sign matrix of its own for every row: c (k, n), signs (k, B, n) -> (k,) p = (b + 1) / (B + 1).
    Same statistic, same >= with a 1e-12 tolerance as e1_summarise.sign_flip_p (a test checks row by row)."""
    n = c.shape[1]
    if nA < 1 or n - nA < 1:  # as sign_flip_p: NaN when a group is empty
        return np.full(c.shape[0], np.nan)
    w = np.concatenate([np.full(nA, wA / nA), np.full(n - nA, -wN / (n - nA))])
    obs = (c * w).sum(1)
    flipped = np.einsum("kbn,kn->kb", signs, c * w)
    b = (flipped >= obs[:, None] - 1e-12).sum(1)
    return (b + 1) / (signs.shape[1] + 1)


def boot_counts(rng, n: int, B: int) -> np.ndarray:
    """B x n multinomial counts = a bootstrap resample of n persons, as weights (rows sum to n)."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)


def simulate(nA: int, nN: int, m_a: float, m_n: float, k0: float, rho: float, n_sims: int, B: int, rng,
             chunk: int = 50, estimator: str = "pi", test: str = "bootstrap", independent: bool = False,
             null: str = "model") -> dict:
    """independent=False (default, reproduces the earlier experiments): one sign matrix / bootstrap draw shared by the simulations of a
    chunk. independent=True (--size-nulls): every simulation gets its own (balanced only). null: "model" = person_contrasts,
    "exch" = person_contrasts_exch (the sharp null)."""
    if test not in ("bootstrap", "permutation") or (test == "permutation" and estimator != "balanced"):
        raise ValueError("test must be 'bootstrap', or 'permutation' with the balanced estimator")
    if null not in ("model", "exch") or ((independent or null == "exch") and estimator != "balanced"):
        raise ValueError("null must be 'model' or 'exch'; independent draws and the exch null need the balanced estimator")
    contrasts = person_contrasts_exch if null == "exch" else person_contrasts
    wA, wN = (PI, 1 - PI) if estimator == "pi" else (0.5, 0.5)
    est, pval = [], []
    for s0 in range(0, n_sims, chunk):
        k = min(chunk, n_sims - s0)
        cA = contrasts(rng, k, nA, m_a, k0, rho)
        cN = contrasts(rng, k, nN, m_n, k0, rho)
        e = wA * cA.mean(1) - wN * cN.mean(1)
        if independent:  # attack-me round 3 A2: a sign matrix / bootstrap draw of its own for every simulation
            est.append(e)
            if test == "permutation":
                pval.append(flip_p_rows(np.hstack([cA, cN]), nA, rng.choice(np.array([-1.0, 1.0]), (k, B, nA + nN)), wA, wN))
            else:
                WA = rng.multinomial(nA, np.full(nA, 1.0 / nA), size=(k, B)).astype(float)
                WN = rng.multinomial(nN, np.full(nN, 1.0 / nN), size=(k, B)).astype(float)
                d = wA * np.einsum("kbn,kn->kb", WA, cA) / nA - wN * np.einsum("kbn,kn->kb", WN, cN) / nN
                pval.append(((d <= 0).sum(1) + 1) / (B + 1))
            continue
        if test == "permutation":  # 判断 39: sign-flip permutation p = (b + 1) / (B + 1), b = flipped statistics >= observed
            signs = rng.choice(np.array([-1.0, 1.0]), (B, nA + nN))
            est.append(e)
            pval.append(sign_flip_p(np.hstack([cA, cN]), nA, signs))
            continue
        WA, WN = boot_counts(rng, nA, B), boot_counts(rng, nN, B)
        d = wA * (WA @ cA.T.astype(np.float32)) / nA - wN * (WN @ cN.T.astype(np.float32)) / nN  # B x k
        est.append(e)
        # balanced: the confirmatory p = (b + 1) / (B + 1) (判断 32); pi keeps the share b / B of
        # experiments/20261113-e1-power (legacy, reproduced by a test)
        pval.append(((d <= 0).sum(0) + 1) / (B + 1) if estimator == "balanced" else (d <= 0).mean(0))
    est, pval = np.concatenate(est), np.concatenate(pval)
    out = {"mean_est": float(est.mean()), "p50_p": float(np.median(pval)), "share_est_gt_delta": float((est > DELTA).mean())}
    if estimator == "pi":
        for a in ALPHAS:
            out[f"support_alpha{a}"] = float(((pval <= a) & (est > DELTA)).mean())
    else:
        out = {"mean_est": out["mean_est"], "p50_p": out["p50_p"]}
        for a in ALPHAS:
            out[f"support_alpha{a}"] = float((pval <= a).mean())
        for dl in BAL_DELTAS:
            out[f"share_est_gt_{dl}"] = float((est > dl).mean())
    return out


def run(out: Path, n_sims: int, B: int, ns: tuple[int, ...], estimator: str = "pi", test: str = "bootstrap") -> Path:
    seed = int((out / "SEED").read_text().strip())
    rows = []
    mas, mns, k0s, rhos = (M_A, M_N, K0_RATES, RHOS) if estimator == "pi" else (BAL_M_A, BAL_M_N, K0_RATES, RHOS)
    grid = [(n, m_a, m_n, k0, rho, "power") for n, m_a, m_n, k0, rho in itertools.product(ns, mas, mns, k0s, rhos)]
    if estimator == "balanced":  # 判断 38: the size of the rule, scenarios with no effect
        grid += [(n, m, m, m, rho, "size") for n, m, rho in itertools.product(ns, SIZE_M, rhos)]
    for n, m_a, m_n, k0, rho, scen in grid:
        ss = np.random.SeedSequence([seed, n, round(m_a * 1000), round(m_n * 1000), round(k0 * 1000), round(rho * 1000)])
        r = simulate(n, n, m_a, m_n, k0, rho, n_sims, B, np.random.default_rng(ss), estimator=estimator, test=test)
        extra = {}
        if estimator == "balanced":
            extra = {"estimator": estimator} | ({"test": test} if test != "bootstrap" else {}) | {"scenario": scen} | {
                f"size_alpha{a}": (r[f"support_alpha{a}"] if scen == "size" else "") for a in ALPHAS}
        rows.append({"n_per_stratum": n, "m_A": m_a, "m_N": m_n, "k0_rate": k0, "rho": rho, "n_sims": n_sims, "boot": B,
                     "seed": seed, **extra, **r})
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    path = res / "power.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def clopper_pearson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval of a binomial proportion x / n."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a, x + 1, n - x))
    return lo, hi


def run_size_nulls(out: Path, n_sims: int, B: int, n: int, tests: tuple[str, ...] = ("permutation", "bootstrap")) -> Path:
    """attack-me round 3 A2: size of the H1 rule under the three nulls of SIZE_NULLS x RHOS, n + n persons, independent draws.
    exceeds_alpha* = Clopper-Pearson 95% lower bound > alpha (the criterion fixed before the run); upper_le_alpha* = upper bound <= alpha."""
    seed = int((out / "SEED").read_text().strip())
    rows = []
    for test, (null, m_a, m_n, k0), rho in itertools.product(tests, SIZE_NULLS, RHOS):
        ss = np.random.SeedSequence([seed, n, NULL_CODE[null], round(m_a * 1000), round(m_n * 1000), round(k0 * 1000),
                                     round(rho * 1000), TEST_CODE[test]])
        r = simulate(n, n, m_a, m_n, k0, rho, n_sims, B, np.random.default_rng(ss), estimator="balanced", test=test,
                     independent=True, null="exch" if null == "exch" else "model")
        row = {"null": null, "test": test, "n_per_stratum": n, "m_A": m_a, "m_N": m_n, "k0_rate": k0, "rho": rho, "n_sims": n_sims,
               "boot": B, "seed": seed, "mean_est": r["mean_est"], "p50_p": r["p50_p"]}
        for a in ALPHAS:
            x = int(round(r[f"support_alpha{a}"] * n_sims))
            lo, hi = clopper_pearson(x, n_sims)
            row |= {f"size_alpha{a}": x / n_sims, f"cp_lo_alpha{a}": lo, f"cp_hi_alpha{a}": hi,
                    f"exceeds_alpha{a}": lo > a, f"upper_le_alpha{a}": hi <= a}
        rows.append(row)
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    path = res / "size_nulls.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ns", default=None, help="default: 24,32,48 for pi; 16,24,32 for balanced")
    ap.add_argument("--estimator", choices=("balanced", "pi"), default="balanced")
    ap.add_argument("--test", choices=("bootstrap", "permutation"), default="bootstrap",
                    help="how each simulation's one-sided p is made (permutation: balanced only; --boot = number of flips)")
    ap.add_argument("--size-nulls", action="store_true",
                    help="size under the three nulls (attack-me round 3 A2), both tests, independent draws; one n from --ns (default 40)")
    a = ap.parse_args()
    if a.size_nulls:
        ns = (a.ns or "40").split(",")
        if len(ns) != 1:
            ap.error("--size-nulls takes one n per stratum in --ns")
        print(run_size_nulls(a.out, a.n_sims, a.boot, int(ns[0])))
        return
    ns = a.ns or ("24,32,48" if a.estimator == "pi" else "16,24,32")
    print(run(a.out, a.n_sims, a.boot, tuple(int(x) for x in ns.split(",")), a.estimator, a.test))


if __name__ == "__main__":
    main()
