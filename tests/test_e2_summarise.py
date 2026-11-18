"""Tests for scripts/e2_summarise.py (t intervals, Welch, TOST, Holm, completeness, the end-to-end tests on a toy map).

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_summarise import (holm, misclassification_cells, percentile_interval, summarise, t_interval, tost,  # noqa: E402
                          welch)


class Helpers(unittest.TestCase):
    def test_t_interval_by_hand(self) -> None:
        r = t_interval(np.array([1.0, 2.0, 3.0, 4.0]), 0.95)
        se = np.std([1, 2, 3, 4], ddof=1) / 2           # 0.6455
        self.assertAlmostEqual(r["est"], 2.5)
        self.assertAlmostEqual(r["se"], se)
        self.assertAlmostEqual(r["hi"] - r["est"], 3.182446305 * se, places=6)   # t(0.975, 3)
        self.assertAlmostEqual(r["p_one_sided"], 0.015233, places=5)   # t = 3.873, df 3: 1/2 - (th + sin th cos th)/pi, th = atan(t/sqrt 3)

    def test_zero_sd(self) -> None:
        r = t_interval(np.full(5, 0.07), 0.95)
        self.assertEqual((r["lo"], r["hi"], r["p_one_sided"]), (0.07, 0.07, 0.0))
        self.assertEqual(t_interval(np.zeros(5), 0.95)["p_one_sided"], 1.0)
        self.assertTrue(tost(t_interval(np.zeros(5), 0.9), 0.005)["equivalent"])

    def test_welch_by_hand(self) -> None:
        x, y = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0, 1.0, 2.0])
        r = welch(x, y, 0.95)
        vx, vy = 1.0 / 3, 0.7 / 5
        self.assertAlmostEqual(r["est"], 2.0 - 0.8)
        self.assertAlmostEqual(r["se"], np.sqrt(vx + vy))
        self.assertAlmostEqual(r["df"], (vx + vy) ** 2 / (vx ** 2 / 2 + vy ** 2 / 4))

    def test_tost(self) -> None:
        inside = tost(t_interval(np.array([0.001, -0.001, 0.0005, 0.0]), 0.90), 0.005)
        self.assertTrue(inside["equivalent"])
        self.assertLess(inside["p_tost"], 0.05)
        outside = tost(t_interval(np.array([0.004, 0.006, 0.005, 0.007]), 0.90), 0.005)
        self.assertFalse(outside["equivalent"])
        self.assertGreater(outside["p_tost"], 0.05)

    def test_holm(self) -> None:
        self.assertEqual(holm({"a": 0.01, "b": 0.04}), {"a": 0.02, "b": 0.04})
        self.assertEqual(holm({"a": 0.03, "b": 0.02}), {"b": 0.04, "a": 0.04})
        self.assertEqual(holm({"a": 0.6, "b": 0.7}), {"a": 1.0, "b": 1.0})


def toy_map(reps: int, drop: tuple | None = None) -> pd.DataFrame:
    """drug_misuse has D1 in config/condition_lists.json; diabetes_t2 does not. dV grows with lambda; D2 lowers it."""
    rng = np.random.default_rng(0)
    rows = []
    for at, ds in (("drug_misuse", ("D0", "D1", "D2")), ("diabetes_t2", ("D0", "D2"))):
        for lam in (0.0, 1.0):
            for r in range(reps):
                noise = rng.normal(0, 0.001)
                for d in ds:
                    for k in ("K2", "K3"):
                        base = 0.04 * lam - (0.006 * lam if d == "D2" else 0) + (0 if k == "K2" else -0.035 * lam)
                        v = base + noise + rng.normal(0, 0.0003)
                        rows.append({"attribute": at, "lambda": lam, "rep": r, "D": d, "K": k, "dV_bal": v,
                                     "dV_acc": v * 0.5, "dV_conf": v * 0.9, "dV_opt": v * 0.8, "dAUC": v * 2.0,
                                     "cost_total": 0.01, "cost_extra": 0.0})
    df = pd.DataFrame(rows)
    if drop:
        at, lam, d, k, r = drop
        df = df[~((df.attribute == at) & (df["lambda"] == lam) & (df.D == d) & (df.K == k) & (df.rep == r))]
    return df


def original_df() -> pd.DataFrame:
    rows = [{"attribute": "drug_misuse", "lambda": 1.0, "D": d, "K": k, "dV_bal": 0.01 * (i + 1), "cv_seed": 5}
            for i, (d, k) in enumerate([(d, k) for d in ("D0", "D1", "D2") for k in ("K2", "K3")])]
    return pd.DataFrame(rows + [{"attribute": "diabetes_t2", "lambda": 1.0, "D": "D0", "K": "K2", "dV_bal": 9.0, "cv_seed": 5}])


def write_exp(tmp: Path, df: pd.DataFrame, reps: int, boot: bool = False, delta: float = 0.005, original: bool = True) -> Path:
    """With bootstrap_persons the original-sample file is required (C18), so it is written by default (original=False: not)."""
    (tmp / "results").mkdir()
    if boot and original:
        original_df().to_csv(tmp / "results" / "e2_lambda1_original.csv", index=False)
    df.to_csv(tmp / "results" / "e2_map_reps.csv", index=False)
    cfg = {"lambdas": [0.0, 1.0], "reps": reps, "min_feature_count": 3,
           "primary_attribute": "drug_misuse", "delta_E2": delta,
           "ci": 0.95, "ci_equivalence": 0.90, "alpha_one_sided": 0.025}
    if boot:
        cfg["bootstrap_persons"] = True
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp


class Bootstrap(unittest.TestCase):
    """判断 28: with bootstrap_persons the per-rep values are the bootstrap distribution (percentile interval, p = (b + 1) / (R + 1), 判断 32)."""

    def test_percentile_interval_by_hand(self) -> None:
        r = percentile_interval(np.array([-1.0, 0.0, 1.0, 2.0, 3.0]), 0.95)
        self.assertAlmostEqual(r["est"], 1.0)
        self.assertAlmostEqual(r["p_one_sided"], 3 / 6)            # 判断 32: values <= 0 are -1 and 0 -> (2 + 1) / (5 + 1)
        self.assertAlmostEqual(r["lo"], np.percentile([-1, 0, 1, 2, 3], 2.5))
        self.assertAlmostEqual(r["hi"], np.percentile([-1, 0, 1, 2, 3], 97.5))
        self.assertAlmostEqual(percentile_interval(np.full(5, 0.07), 0.95)["p_one_sided"], 1 / 6)   # never exactly 0
        self.assertAlmostEqual(percentile_interval(np.zeros(5), 0.95)["p_one_sided"], 1.0)

    def test_more_reps_do_not_narrow_the_interval(self) -> None:
        x = np.random.default_rng(1).normal(0.01, 0.003, 50)
        wide = lambda v: (lambda r: r["hi"] - r["lo"])(percentile_interval(v, 0.95))  # noqa: E731
        w1, w4 = wide(x), wide(np.tile(x, 4))                     # same distribution, 4 times the reps
        self.assertAlmostEqual(w4, w1, delta=0.10 * w1)
        t1, t4 = t_interval(x, 0.95), t_interval(np.tile(x, 4), 0.95)
        self.assertLess((t4["hi"] - t4["lo"]) / (t1["hi"] - t1["lo"]), 0.6)   # the t interval does narrow (about 1/2)

    def test_summarise_uses_the_percentile_only_with_bootstrap_true(self) -> None:
        res = {}
        for boot in (True, False):
            for reps in (20, 80):
                with tempfile.TemporaryDirectory() as t:
                    out = summarise(write_exp(Path(t), toy_map(reps), reps, boot))
                res[boot, reps] = out
        for reps in (20, 80):
            self.assertEqual(res[True, reps]["interval_method"], "percentile_bootstrap")
            self.assertEqual(res[False, reps]["interval_method"], "t_over_reps")
        w = lambda o: (lambda h: h["hi"] - h["lo"])(o["dV_bal"]["H2_D0_minus_D2_lambda1_K2"])  # noqa: E731
        self.assertGreater(w(res[True, 80]) / w(res[True, 20]), 0.7)     # percentile: R does not shrink it
        self.assertLess(w(res[False, 80]) / w(res[False, 20]), 0.7)      # t over reps: about 1/2
        b = res[True, 80]["dV_bal"]
        self.assertIsNone(b["H2_D0_minus_D2_lambda1_K2"]["df"])          # no t / Welch df in the bootstrap path
        self.assertEqual(b["H0c_E2_side"]["lambda1_minus_lambda0_D0_K2"]["test"], "paired_by_rep")
        self.assertEqual(res[False, 80]["dV_bal"]["H0c_E2_side"]["lambda1_minus_lambda0_D0_K2"]["test"], "welch")
        self.assertTrue(b["H0a_lambda0_D0_K2"]["equivalent"])
        self.assertIn("p_tost", b["H0a_lambda0_D0_K2"])

    def test_tost_under_bootstrap_is_the_90pct_percentile_inside_delta(self) -> None:
        df = toy_map(100)
        m = (df.attribute == "drug_misuse") & (df["lambda"] == 0.0) & (df.D == "D0") & (df.K == "K2")
        df.loc[m, "dV_bal"] = np.linspace(-0.004, 0.004, 100)           # inside +-0.005
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(summarise(write_exp(Path(t), df, 100, True))["dV_bal"]["H0a_lambda0_D0_K2"]["equivalent"])
        df.loc[m, "dV_bal"] = np.linspace(-0.004, 0.012, 100)           # upper 95th percentile above +0.005
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(summarise(write_exp(Path(t), df, 100, True))["dV_bal"]["H0a_lambda0_D0_K2"]["equivalent"])


class Prereg3(unittest.TestCase):
    """C13-C15, C18, C19 (paper-write 判断 35-38)."""

    def test_unfinished_e1_stops(self) -> None:
        """C13: an E1 metrics.json without completeness.complete == True is not summarised with E2."""
        for comp in ({}, {"complete": False}, {"complete": None}, {"complete": "yes"}):
            with self.subTest(comp=comp), tempfile.TemporaryDirectory() as t:
                exp = write_exp(Path(t), toy_map(5), 5)
                e1 = Path(t) / "e1"
                e1.mkdir()
                (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "completeness": comp,
                                                             "delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.01, "est": 0.05},
                                                             "H0c_E1_side": {}}))
                with self.assertRaises(SystemExit) as cm:
                    summarise(exp, e1)
                self.assertIn("completeness.complete", str(cm.exception))
        with tempfile.TemporaryDirectory() as t:    # no completeness key at all
            exp = write_exp(Path(t), toy_map(5), 5)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced"}))
            with self.assertRaises(SystemExit):
                summarise(exp, e1)
        # E2 alone (no --e1) does not look at E1
        with tempfile.TemporaryDirectory() as t:
            summarise(write_exp(Path(t), toy_map(5), 5))

    def test_secondary_h2_dv_opt_and_dauc(self) -> None:
        """C14: the same D0 - D2 contrast (lambda 1, K2, paired by rep) for dV_opt and dAUC; the primary is untouched."""
        for boot in (False, True):
            with self.subTest(boot=boot), tempfile.TemporaryDirectory() as t:
                out = summarise(write_exp(Path(t), toy_map(20), 20, boot))
            sec = out["secondary_H2"]
            self.assertFalse(sec["decision"])
            self.assertEqual(set(sec["estimators"]), {"dV_opt", "dAUC"})
            h2 = out["dV_bal"]["H2_D0_minus_D2_lambda1_K2"]
            self.assertAlmostEqual(sec["estimators"]["dV_opt"]["est"], 0.8 * h2["est"], places=9)
            self.assertAlmostEqual(sec["estimators"]["dAUC"]["est"], 2.0 * h2["est"], places=9)
            self.assertLess(sec["estimators"]["dAUC"]["lo"], sec["estimators"]["dAUC"]["est"])
            self.assertEqual(out["primary_estimator"], "dV_bal")
            self.assertNotIn("dV_opt", out)                       # not a full estimator block: no H0a / H0c for the secondaries
        for col in ("dV_opt", "dAUC"):                            # both columns are required
            with self.subTest(missing=col), tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit):
                summarise(write_exp(Path(t), toy_map(5).drop(columns=[col]), 5))

    def test_h2_residual_leak(self) -> None:
        """C15: lower end of the lambda 1, D2, K2 dV_bal interval > delta_E2; a secondary test outside Holm."""
        with tempfile.TemporaryDirectory() as t:
            out = summarise(write_exp(Path(t), toy_map(20), 20, True))
        leak = out["H2_residual_leak"]
        self.assertAlmostEqual(leak["est"], 0.034, delta=0.002)      # toy: 0.04 - 0.006
        self.assertTrue(leak["supported"])
        self.assertEqual(leak["supported"], leak["lo"] > 0.005)
        self.assertFalse(leak["in_holm"])
        with tempfile.TemporaryDirectory() as t:                      # delta_E2 above the interval -> not supported
            self.assertFalse(summarise(write_exp(Path(t), toy_map(20), 20, True, delta=0.05))["H2_residual_leak"]["supported"])
        df = toy_map(20)                                              # D2 closes the leak completely
        m = (df.attribute == "drug_misuse") & (df["lambda"] == 1.0) & (df.D == "D2") & (df.K == "K2")
        df.loc[m, "dV_bal"] = np.random.default_rng(3).normal(0.0, 0.0005, 20)
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(summarise(write_exp(Path(t), df, 20, True))["H2_residual_leak"]["supported"])
        with tempfile.TemporaryDirectory() as t:                      # supported needs lo > delta strictly
            lo = summarise(write_exp(Path(t), toy_map(20), 20, True))["H2_residual_leak"]["lo"]
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(summarise(write_exp(Path(t), toy_map(20), 20, True, delta=lo))["H2_residual_leak"]["supported"])

    def _with_original(self, mutate=None, boot: bool = True) -> dict | None:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20, boot, original=False)
            o = original_df()
            if mutate:
                o = mutate(o)
            if o is not None:
                o.to_csv(exp / "results" / "e2_lambda1_original.csv", index=False)
            return summarise(exp)["descriptive_lambda1_original_sample"]

    def test_lambda1_original_sample_described(self) -> None:
        """C18: the original-sample values sit beside the R-rep mean; the judged values are unchanged."""
        with tempfile.TemporaryDirectory() as t:
            base = summarise(write_exp(Path(t), toy_map(20), 20, False))
        self.assertIsNone(base["descriptive_lambda1_original_sample"])               # no bootstrap_persons, no file: nothing
        with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit) as cm:   # bootstrap_persons true, no file: stop
            summarise(write_exp(Path(t), toy_map(20), 20, True, original=False))
        self.assertIn("e2_lambda1_original.csv", str(cm.exception))
        d = self._with_original()
        self.assertTrue(d["descriptive"])
        self.assertEqual(set(d["cells"]), {f"lambda1.0|{x}|{k}" for x in ("D0", "D1", "D2") for k in ("K2", "K3")})
        self.assertAlmostEqual(d["cells"]["lambda1.0|D0|K2"]["original"], 0.01)       # the first row of the helper
        self.assertAlmostEqual(d["cells"]["lambda1.0|D2|K3"]["original"], 0.06)
        self.assertAlmostEqual(d["cells"]["lambda1.0|D0|K2"]["R_mean"], 0.04, delta=0.002)
        self.assertAlmostEqual(d["H2_D0_minus_D2_K2"]["original"], 0.01 - 0.05)       # D0|K2 = 0.01, D2|K2 = 0.05
        self.assertAlmostEqual(d["H2_D0_minus_D2_K2"]["R_mean"], 0.006, delta=0.0005)
        # the diabetes_t2 row (another attribute) is ignored, and the judgement does not read the file
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20, True)
            judged = summarise(exp)
            original_df().assign(dV_bal=lambda o: o.dV_bal * 7).to_csv(exp / "results" / "e2_lambda1_original.csv", index=False)
            other = summarise(exp)
        self.assertNotEqual(judged["descriptive_lambda1_original_sample"], other["descriptive_lambda1_original_sample"])
        self.assertEqual(judged["dV_bal"]["H2_D0_minus_D2_lambda1_K2"], other["dV_bal"]["H2_D0_minus_D2_lambda1_K2"])
        # an incomplete or wrong file stops
        for name, mut in (("missing cell", lambda o: o[~((o.D == "D2") & (o.K == "K3"))]),
                          ("duplicate", lambda o: pd.concat([o, o.iloc[[0]]])),
                          ("extra D", lambda o: pd.concat([o, o.iloc[[0]].assign(D="D9")]))):
            with self.subTest(name), self.assertRaises(SystemExit):
                self._with_original(mut)

    def _scope(self, h0b, boot: bool = False, delta: float = 0.005, est_k0n="absent", h1_est: float = 0.05, h0d=True) -> dict:
        extra = {} if h0b == "absent" else {"H0b": {"supported": h0b}}
        if h0d != "absent":
            extra["H0d"] = {"supported": h0d}
        if est_k0n != "absent":
            extra["H1_control_K2_minus_K0n"] = {"est": est_k0n}
        return self._run_e1_scope(extra, delta, h1_est)

    def _run_e1_scope(self, extra: dict, delta: float, h1_est: float) -> dict:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20, False, delta)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "completeness": {"complete": True},
                "delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.01, "est": h1_est},
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.02},
                                "D0_minus_D1_lambda1_K2": {"est": -0.01}}} | extra))
            return summarise(exp, e1)

    def test_h1_scope(self) -> None:
        """C19: "full" only when E1's H0b, E2's H0c and E1's H0d are all True; False, None and a missing H0b are E1_measurement_only."""
        full = self._scope(True)
        self.assertTrue(full["dV_bal"]["H0c_E2_side"]["supported"])
        self.assertEqual(full["H1_scope"], "full")
        self.assertEqual(full["final_decision"]["H1_scope"], "full")
        self.assertEqual(self._scope(False)["H1_scope"], "E1_measurement_only")
        self.assertEqual(self._scope(None)["H1_scope"], "E1_measurement_only")
        self.assertEqual(self._scope("absent")["H1_scope"], "E1_measurement_only")
        # 判断 44: H0d must be True as well (False, None or absent -> not "full")
        for h0d in (False, None, "absent"):
            with self.subTest(h0d=h0d):
                self.assertEqual(self._scope(True, h0d=h0d)["H1_scope"], "E1_measurement_only")
        self.assertIs(full["final_decision"]["H0d_E1"], True)
        # H0c not judged (delta so large that no contrast is outside +-delta): supported is None -> not "full" even if H0b is True
        nj = self._scope(True, delta=1.0)
        self.assertIsNone(nj["dV_bal"]["H0c_E2_side"]["supported"])
        self.assertEqual(nj["H1_scope"], "E1_measurement_only")
        # the H1 decision itself is not changed by the scope
        self.assertTrue(self._scope(False)["H1_supported"])
        # no --e1: no scope
        with tempfile.TemporaryDirectory() as t:
            self.assertNotIn("H1_scope", summarise(write_exp(Path(t), toy_map(5), 5)))

    def test_k0n_length_note(self) -> None:
        """C19: note = (K2 - K0n estimate) < 0.5 * (K2 - K0 estimate); strict; missing K0n estimate -> True (cannot rule length out)."""
        self.assertTrue(self._scope(True, est_k0n=0.01)["H1_K0n_length_note"])             # 0.01 < 0.025
        self.assertFalse(self._scope(True, est_k0n=0.03)["H1_K0n_length_note"])
        self.assertFalse(self._scope(True, est_k0n=0.025)["H1_K0n_length_note"])           # equal to half: not less
        self.assertTrue(self._scope(True, est_k0n=-0.02)["H1_K0n_length_note"])
        miss = self._scope(True)
        self.assertTrue(miss["H1_K0n_length_note"])
        self.assertFalse(miss["final_decision"]["H1_K0n_estimate_present"])
        self.assertTrue(self._scope(True, est_k0n=0.03)["final_decision"]["H1_K0n_estimate_present"])

    def test_k0n_length_note_none_when_k2_minus_k0_not_positive(self) -> None:
        """K2 - K0 point estimate <= 0: the comparison has no meaning, so the note is None (also with a missing K0n estimate)."""
        for h1_est in (0.0, -0.02):
            for k0n in ("absent", -0.05, 0.01):
                with self.subTest(h1_est=h1_est, k0n=k0n):
                    out = self._scope(True, est_k0n=k0n, h1_est=h1_est)
                    self.assertIsNone(out["H1_K0n_length_note"])
                    self.assertIsNone(out["final_decision"]["H1_K0n_length_note"])
        self.assertFalse(self._scope(True, est_k0n=0.03, h1_est=0.05)["H1_K0n_length_note"])   # positive: as before

    def test_e1_metrics_sha256_recorded(self) -> None:
        import hashlib
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20, False)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "delta_E1": 0.03, "completeness": {"complete": True},
                                                         "H1_K2_minus_K0": {"p_one_sided_permutation": 0.01, "est": 0.05},
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.02},
                                "D0_minus_D1_lambda1_K2": {"est": -0.01}}}))
            out = summarise(exp, e1)
            want = hashlib.sha256((e1 / "metrics.json").read_bytes()).hexdigest()
        self.assertEqual(out["e1_metrics_sha256"], want)
        self.assertEqual(len(want), 64)


class EndToEnd(unittest.TestCase):
    def test_e1_of_another_estimator_refused(self) -> None:
        """An E1 metrics.json summarised before 判断 25 (pi-weighted, or no primary_estimator) does not enter Holm."""
        for bad in ({}, {"primary_estimator": "pi"}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as t:
                exp = write_exp(Path(t), toy_map(20), 20)
                e1 = Path(t) / "e1"
                e1.mkdir()
                (e1 / "metrics.json").write_text(json.dumps(bad | {
                    "delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.01, "est": 0.05},
                    "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.02},
                                    "D0_minus_D1_lambda1_K2": {"est": -0.01}}}))
                with self.assertRaises(SystemExit) as cm:
                    summarise(exp, e1)
                self.assertIn("balanced", str(cm.exception))

    def test_tests_on_toy_map(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "completeness": {"complete": True},
                "delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.01, "est": 0.05},
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.02},
                                "D0_minus_D1_lambda1_K2": {"est": -0.01}}}))
            out = summarise(exp, e1)
        self.assertEqual(out["primary_estimator"], "dV_bal")
        self.assertAlmostEqual(out["dV_acc"]["H2_D0_minus_D2_lambda1_K2"]["est"], 0.003, delta=0.0005)   # secondary, kept
        self.assertEqual(out["descriptive_misclassification_cells"]["estimator"], "dV_bal")
        self.assertEqual(out["descriptive_misclassification_cells_dV_acc"]["estimator"], "dV_acc")
        p = out["dV_bal"]
        h2 = p["H2_D0_minus_D2_lambda1_K2"]
        self.assertAlmostEqual(h2["est"], 0.006, delta=0.0005)
        self.assertNotIn("supported", h2)                      # the decision is the Holm one below
        self.assertTrue(h2["unadjusted_interval_criterion"])   # lo > 0 and 0.006 > 0.005
        self.assertTrue(p["H0a_lambda0_D0_K2"]["equivalent"])
        h0c = p["H0c_E2_side"]
        self.assertEqual(h0c["lambda1_minus_lambda0_D0_K2"]["test"], "welch")
        self.assertTrue(h0c["lambda1_minus_lambda0_D0_K2"]["in_judgement"])
        self.assertFalse(h0c["D0_minus_D1_lambda1_K2"]["in_judgement"])   # D1 = D0 in the toy map
        k3 = p["K3_minus_K2_descriptive"]["lambda1.0|D0"]                 # paired -0.035, far outside +-delta
        self.assertLess(k3["hi"], -0.005)
        self.assertTrue(h0c["supported"])                      # E1's D0-D1 sign differs but is not judged
        self.assertEqual(out["holm"]["adjusted"]["H1_E1"], 0.01)        # H2 p ~ 0 is first (x2), H1 second (x1)
        self.assertTrue(out["H1_supported"])                            # 0.01 <= 0.025 and E1 est 0.05 > 0.03
        self.assertTrue(out["H2_supported"])                            # ~0 <= 0.025 and 0.006 > 0.005
        self.assertEqual(out["completeness"]["rows"], 2 * 20 * (3 + 2) * 2)   # lambda x reps x (D of both) x K
        self.assertIn("diabetes_t2|lam=1.0|D2|K3", out["descriptive"])
        self.assertNotIn("diabetes_t2|lam=1.0|D1|K2", out["descriptive"])

    def _run_e1(self, h1: dict, reps: int = 20, delta_e1: float = 0.03, df: pd.DataFrame | None = None,
                boot: bool = False, delta: float = 0.005, extra: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(reps) if df is None else df, reps, boot, delta)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "completeness": {"complete": True},
                "delta_E1": delta_e1, "H1_K2_minus_K0": h1,
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.02}}}
                | (extra or {})))
            return summarise(exp, e1)

    def test_final_decision_rule(self) -> None:
        # raw p 0.02 for H1; H2 p ~ 0 -> Holm: H2 x2 ~ 0, H1 x1 = 0.02 -> adjusted 0.02 <= 0.025 and est > delta
        out = self._run_e1({"p_one_sided_permutation": 0.02, "est": 0.05})
        self.assertTrue(out["H1_supported"])
        # 判断 22: the estimate vs delta_E1 is descriptive only; p 0.02 <= 0.025 with est 0.02 < 0.03 is still supported
        small = self._run_e1({"p_one_sided_permutation": 0.02, "est": 0.02})
        self.assertTrue(small["H1_supported"])
        self.assertFalse(small["H1_estimate_above_delta_E1"])
        # raw p above alpha: not supported whatever the estimate
        self.assertFalse(self._run_e1({"p_one_sided_permutation": 0.03, "est": 0.05})["H1_supported"])   # 0.03 > 0.025
        self.assertEqual(self._run_e1({"p_one_sided_permutation": 0.03, "est": 0.05})["alpha_one_sided"], 0.025)
        with self.assertRaises(SystemExit):                                                  # no E1 estimate
            self._run_e1({"p_one_sided_permutation": 0.01})

    def test_h1_p_is_the_permutation_p_not_the_bootstrap_p(self) -> None:
        # 判断 39: Holm takes p_one_sided_permutation; the bootstrap p_one_sided beside it is ignored; without the new key it stops
        out = self._run_e1({"p_one_sided_permutation": 0.03, "p_one_sided": 0.001, "est": 0.05})
        self.assertEqual(out["holm"]["raw"]["H1_E1"], 0.03)
        self.assertFalse(out["H1_supported"])
        with self.assertRaises(SystemExit):
            self._run_e1({"p_one_sided": 0.01, "est": 0.05})

    def test_final_decision_h2_needs_estimate_above_delta(self) -> None:
        # (a) H2's p is ~0 but the estimate (D0 - D2 = 0.006) is not above delta_E2 = 0.007 -> H2 not supported
        out = self._run_e1({"p_one_sided_permutation": 0.01, "est": 0.05}, delta=0.007)
        self.assertLessEqual(out["holm"]["adjusted"]["H2_E2"], 0.025)
        self.assertLess(out["dV_bal"]["H2_D0_minus_D2_lambda1_K2"]["est"], 0.007)
        self.assertFalse(out["H2_supported"])
        self.assertTrue(out["H1_supported"])

    def test_final_decision_holm_doubles_the_smaller_raw_p(self) -> None:
        # (b) bootstrap map: H2 p = (4 + 1)/(200 + 1) = 0.0249 (4 reps with D0 - D2 <= 0; 判断 32), H1 raw p 0.013 is the smaller:
        # Holm: H1 x2 = 0.026 > 0.025 -> H1 not supported although its raw p is below alpha; H2 adjusted = 0.026 too
        df = toy_map(200)
        m = (df.attribute == "drug_misuse") & (df["lambda"] == 1.0) & (df.K == "K2")
        df.loc[m & (df.D == "D2"), "dV_bal"] = 0.0
        for d in ("D0", "D1"):                                    # D1 = D0 so that D0 - D1 is not a judged contrast
            idx = df[m & (df.D == d)].sort_values("rep").index
            df.loc[idx, "dV_bal"] = 0.01
            df.loc[idx[:4], "dV_bal"] = -0.01
        out = self._run_e1({"p_one_sided_permutation": 0.013, "est": 0.05}, reps=200, df=df, boot=True)
        self.assertAlmostEqual(out["holm"]["raw"]["H2_E2"], 5 / 201)
        self.assertAlmostEqual(out["holm"]["adjusted"]["H1_E1"], 0.026)
        self.assertAlmostEqual(out["holm"]["adjusted"]["H2_E2"], 0.026)
        self.assertFalse(out["H1_supported"])
        self.assertFalse(out["H2_supported"])

    def test_misclassification_cells_descriptive(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            out = summarise(write_exp(Path(t), toy_map(20), 20))
        m = out["descriptive_misclassification_cells"]
        self.assertTrue(m["descriptive"])
        self.assertEqual(set(m["by_lambda"]), {"0.0", "1.0"})
        self.assertEqual(m["by_lambda"]["0.0"]["n_cells"], 6)                     # D0 D1 D2 x K2 K3
        self.assertEqual(m["by_lambda"]["0.0"]["n_misjudged"], 0)                 # lambda 0: dV ~ 0 everywhere
        # lambda 1: K2 of D0 / D1 / D2 = 0.04 / 0.04 / 0.034 leak above +0.005 (misjudged); K3 are not clear
        self.assertGreaterEqual(m["by_lambda"]["1.0"]["n_misjudged"], 3)
        self.assertTrue(all(m["cells"][f"lambda1.0|{d}|K2"]["misjudged"] for d in ("D0", "D1", "D2")))
        self.assertFalse(m["cells"]["lambda0.0|D0|K2"]["misjudged"])
        # 判断 24: a cell clearly below -delta (negative floor) is not misjudged
        neg = pd.DataFrame([{"attribute": "a", "lambda": 0.5, "D": d, "K": k, "rep": r, "dV_bal": -0.01 + 0.0001 * r}
                            for d in ("D0", "D2") for k in ("K2", "K3") for r in range(5)])
        mn = misclassification_cells(neg, "a", [0.5], 0.005, 0.95)
        self.assertEqual(mn["by_lambda"]["0.5"], {"n_misjudged": 0, "n_negative_floor": 4, "n_cells": 4})
        self.assertAlmostEqual(m["D0_minus_D2_K2_by_lambda"]["1.0"]["est"], 0.006, delta=0.0005)
        self.assertAlmostEqual(m["D0_minus_D2_K2_by_lambda"]["0.0"]["est"], 0.0, delta=0.0005)
        self.assertNotIn("supported", json.dumps(m))

    def test_sign_disagreement_fails_h0c(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "completeness": {"complete": True},
                "delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.2, "est": 0.05},
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07},
                                "D0_minus_D2_lambda1_K2": {"est": -0.02}}}))
            out = summarise(exp, e1)
        self.assertFalse(out["dV_bal"]["H0c_E2_side"]["supported"])

    def test_missing_dV_bal_column_stops(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(SystemExit):
                summarise(write_exp(Path(t), toy_map(5).drop(columns=["dV_bal"]), 5))

    def test_incomplete_stops(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(5, drop=("drug_misuse", 1.0, "D2", "K2", 3)), 5)
            with self.assertRaises(SystemExit):
                summarise(exp)

    def test_bad_inputs_stop(self) -> None:
        full = toy_map(5)
        dup = pd.concat([full, full.iloc[[0]]])
        extra = pd.concat([full, full.iloc[[0]].assign(K="K9")])
        nan = full.copy()
        nan.loc[nan.index[7], "dV_bal"] = np.nan
        no_diab = full[full.attribute != "diabetes_t2"]
        for name, df in (("dup", dup), ("extra", extra), ("nan", nan), ("no_diabetes", no_diab)):
            with self.subTest(name), tempfile.TemporaryDirectory() as t:
                with self.assertRaises(SystemExit):
                    summarise(write_exp(Path(t), df, 5))

    def test_e1_missing_or_zero(self) -> None:
        def run(e1m: dict) -> dict:
            with tempfile.TemporaryDirectory() as t:
                exp = write_exp(Path(t), toy_map(20), 20)
                e1 = Path(t) / "e1"
                e1.mkdir()
                (e1 / "metrics.json").write_text(json.dumps({"primary_estimator": "balanced", "completeness": {"complete": True}} | e1m))
                return summarise(exp, e1)
        h0c = {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.0}}
        out = run({"delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.3, "est": 0.05}, "H0c_E1_side": h0c})
        self.assertFalse(out["dV_bal"]["H0c_E2_side"]["supported"])          # E1 = 0 on a judged contrast
        with self.assertRaises(SystemExit):                                  # judged contrast without E1
            run({"delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.3, "est": 0.05}, "H0c_E1_side": {"D0_minus_D2_lambda1_K2": {"est": 0.01}}})
        with self.assertRaises(SystemExit):                                  # no H1 p
            run({"delta_E1": 0.03, "H1_K2_minus_K0": {"p_one_sided_permutation": None, "est": 0.05}, "H0c_E1_side": h0c})

    def test_pairing_matters(self) -> None:
        df = toy_map(20)
        a = df[(df.attribute == "drug_misuse") & (df["lambda"] == 1.0) & (df.K == "K2")]
        x, y = a[a.D == "D0"].dV_bal.to_numpy(), a[a.D == "D2"].dV_bal.to_numpy()
        self.assertGreater(welch(x, y, 0.95)["se"], 2 * t_interval(x - y, 0.95)["se"])   # shared noise cancels

    def test_without_e1(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            out = summarise(write_exp(Path(t), toy_map(5), 5))
        self.assertNotIn("holm", out)
        self.assertIsNone(out["final_decision"])                  # no --e1: no final decision
        self.assertIn("no --e1", out["final_decision_note"])
        self.assertNotIn("H1_supported", out)
        self.assertNotIn("H2_supported", out)
        self.assertNotIn("supported", out["dV_bal"]["H0c_E2_side"])


if __name__ == "__main__":
    unittest.main()
