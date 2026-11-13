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
from e2_summarise import holm, summarise, t_interval, tost, welch  # noqa: E402


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
                        rows.append({"attribute": at, "lambda": lam, "rep": r, "D": d, "K": k, "dV_acc": v,
                                     "dV_conf": v * 0.9, "cost_total": 0.01, "cost_extra": 0.0})
    df = pd.DataFrame(rows)
    if drop:
        at, lam, d, k, r = drop
        df = df[~((df.attribute == at) & (df["lambda"] == lam) & (df.D == d) & (df.K == k) & (df.rep == r))]
    return df


def write_exp(tmp: Path, df: pd.DataFrame, reps: int) -> Path:
    (tmp / "results").mkdir()
    df.to_csv(tmp / "results" / "e2_map_reps.csv", index=False)
    (tmp / "config.json").write_text(json.dumps({"lambdas": [0.0, 1.0], "reps": reps, "min_feature_count": 3,
                                                 "primary_attribute": "drug_misuse", "delta_E2": 0.005,
                                                 "ci": 0.95, "ci_equivalence": 0.90}))
    return tmp


class EndToEnd(unittest.TestCase):
    def test_tests_on_toy_map(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({
                "H1_K2_minus_K0": {"p_one_sided": 0.01},
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.02},
                                "D0_minus_D1_lambda1_K2": {"est": -0.01}}}))
            out = summarise(exp, e1)
        p = out["dV_acc"]
        h2 = p["H2_D0_minus_D2_lambda1_K2"]
        self.assertAlmostEqual(h2["est"], 0.006, delta=0.0005)
        self.assertTrue(h2["supported"])                       # lo > 0 and 0.006 > 0.005
        self.assertTrue(p["H0a_lambda0_D0_K2"]["equivalent"])
        h0c = p["H0c_E2_side"]
        self.assertEqual(h0c["lambda1_minus_lambda0_D0_K2"]["test"], "welch")
        self.assertTrue(h0c["lambda1_minus_lambda0_D0_K2"]["in_judgement"])
        self.assertFalse(h0c["D0_minus_D1_lambda1_K2"]["in_judgement"])   # D1 = D0 in the toy map
        k3 = p["K3_minus_K2_descriptive"]["lambda1.0|D0"]                 # paired -0.035, far outside +-delta
        self.assertLess(k3["hi"], -0.005)
        self.assertTrue(h0c["supported"])                      # E1's D0-D1 sign differs but is not judged
        self.assertEqual(out["holm"]["adjusted"]["H1_E1"], 0.01)        # H2 p ~ 0 is first (x2), H1 second (x1)
        self.assertEqual(out["completeness"]["rows"], 2 * 20 * (3 + 2) * 2)   # lambda x reps x (D of both) x K
        self.assertIn("diabetes_t2|lam=1.0|D2|K3", out["descriptive"])
        self.assertNotIn("diabetes_t2|lam=1.0|D1|K2", out["descriptive"])

    def test_sign_disagreement_fails_h0c(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            exp = write_exp(Path(t), toy_map(20), 20)
            e1 = Path(t) / "e1"
            e1.mkdir()
            (e1 / "metrics.json").write_text(json.dumps({
                "H1_K2_minus_K0": {"p_one_sided": 0.2},
                "H0c_E1_side": {"lambda1_minus_lambda0_D0_K2": {"est": 0.07},
                                "D0_minus_D2_lambda1_K2": {"est": -0.02}}}))
            out = summarise(exp, e1)
        self.assertFalse(out["dV_acc"]["H0c_E2_side"]["supported"])

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
        nan.loc[nan.index[7], "dV_acc"] = np.nan
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
                (e1 / "metrics.json").write_text(json.dumps(e1m))
                return summarise(exp, e1)
        h0c = {"lambda1_minus_lambda0_D0_K2": {"est": 0.07}, "D0_minus_D2_lambda1_K2": {"est": 0.0}}
        out = run({"H1_K2_minus_K0": {"p_one_sided": 0.3}, "H0c_E1_side": h0c})
        self.assertFalse(out["dV_acc"]["H0c_E2_side"]["supported"])          # E1 = 0 on a judged contrast
        with self.assertRaises(SystemExit):                                  # judged contrast without E1
            run({"H1_K2_minus_K0": {"p_one_sided": 0.3}, "H0c_E1_side": {"D0_minus_D2_lambda1_K2": {"est": 0.01}}})
        with self.assertRaises(SystemExit):                                  # no H1 p
            run({"H1_K2_minus_K0": {"p_one_sided": None}, "H0c_E1_side": h0c})

    def test_pairing_matters(self) -> None:
        df = toy_map(20)
        a = df[(df.attribute == "drug_misuse") & (df["lambda"] == 1.0) & (df.K == "K2")]
        x, y = a[a.D == "D0"].dV_acc.to_numpy(), a[a.D == "D2"].dV_acc.to_numpy()
        self.assertGreater(welch(x, y, 0.95)["se"], 2 * t_interval(x - y, 0.95)["se"])   # shared noise cancels

    def test_without_e1(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            out = summarise(write_exp(Path(t), toy_map(5), 5))
        self.assertNotIn("holm", out)
        self.assertNotIn("supported", out["dV_acc"]["H0c_E2_side"])


if __name__ == "__main__":
    unittest.main()
