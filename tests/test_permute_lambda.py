"""Tests for scripts/permute_lambda.py (positioning §5 lambda; decisions 判断 7).

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from permute_lambda import check_a_within_d0, d0_mask, lift_table, permute  # noqa: E402

Y0 = 2023
D0 = {"include": ["opioid"], "exclude": []}


def toy(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A at 20 %; 'Pain' with P(pain | A) = 0.8 and 0.1 otherwise; 'Cough' independent at 0.3; A persons carry the
    label 'Opioid abuse' (hidden by D0)."""
    rng = np.random.default_rng(seed)
    pid = np.array([f"p{i}" for i in range(n)])
    a = rng.random(n) < 0.2
    persons = pd.DataFrame({"pid": pid, "A": a})
    rows = []
    for i in range(n):
        if a[i]:
            rows.append((pid[i], "Opioid abuse", 2015))
        if rng.random() < (0.8 if a[i] else 0.1):
            rows.append((pid[i], "Pain", int(rng.integers(2005, 2022))))
        if rng.random() < 0.3:
            rows.append((pid[i], "Cough", int(rng.integers(2005, 2022))))
    c = pd.DataFrame(rows, columns=["pid", "display", "onset_year"])
    c["abate_year"] = np.nan
    return persons, c


class PermuteLambda(unittest.TestCase):
    def setUp(self) -> None:
        self.persons, self.c = toy()
        self.fixed = d0_mask(self.c.display, D0)
        self.pids = self.persons.pid.to_numpy()

    def lift(self, c: pd.DataFrame) -> pd.Series:
        return lift_table(self.persons, c, Y0, d0_mask(c.display, D0)).lift

    def test_lambda_one_returns_input(self) -> None:
        out = permute(self.c, self.pids, self.fixed, 1.0, np.random.default_rng(1))
        pd.testing.assert_frame_equal(out, self.c)

    def test_rates_preserved(self) -> None:
        for lam in (0.0, 0.3, 0.7):
            out = permute(self.c, self.pids, self.fixed, lam, np.random.default_rng(2))
            before = self.c.groupby(["display", "onset_year"]).size()
            after = out.groupby(["display", "onset_year"]).size()
            pd.testing.assert_series_equal(before, after)
            self.assertEqual(out.groupby("display").pid.nunique().to_dict(), self.c.groupby("display").pid.nunique().to_dict())

    def test_fixed_rows_untouched(self) -> None:
        out = permute(self.c, self.pids, self.fixed, 0.0, np.random.default_rng(3))
        pd.testing.assert_frame_equal(out[self.fixed], self.c[self.fixed])

    def test_lambda_zero_lift_near_one(self) -> None:
        base = self.lift(self.c)
        self.assertGreater(base["Pain"], 2.5)
        lifts = [self.lift(permute(self.c, self.pids, self.fixed, 0.0, np.random.default_rng(s)))["Pain"] for s in range(20)]
        self.assertAlmostEqual(float(np.mean(lifts)), 1.0, delta=0.05)

    def test_lift_monotone_in_lambda(self) -> None:
        mean = {lam: np.mean([self.lift(permute(self.c, self.pids, self.fixed, lam, np.random.default_rng(s)))["Pain"]
                              for s in range(10)]) for lam in (0.0, 0.5, 1.0)}
        self.assertLess(mean[0.0], mean[0.5])
        self.assertLess(mean[0.5], mean[1.0])

    def test_bundle_stays_together(self) -> None:
        both = set(self.c[self.c.display == "Pain"].pid) & set(self.c[self.c.display == "Cough"].pid)
        out = permute(self.c, self.pids, self.fixed, 0.0, np.random.default_rng(4))
        both_after = set(out[out.display == "Pain"].pid) & set(out[out.display == "Cough"].pid)
        self.assertEqual(len(both), len(both_after))

    def test_bundles_are_whole_and_within_pids(self) -> None:
        """Each person's movable bundle lands intact on exactly one person of the population."""
        def bundles(c: pd.DataFrame) -> list:
            mv = c[~d0_mask(c.display, D0)]
            return sorted(tuple(sorted(zip(g.display, g.onset_year))) for _, g in mv.groupby("pid"))
        out = permute(self.c, self.pids, self.fixed, 0.0, np.random.default_rng(5))
        self.assertTrue(set(out.pid) <= set(self.pids))
        self.assertEqual(bundles(out), bundles(self.c))
        self.assertGreater((out.pid != self.c.pid).mean(), 0.5)

    def test_a_within_d0_guard(self) -> None:
        disp = pd.Series(["Opioid abuse", "Misuses drugs (finding)", "Pain"])
        check_a_within_d0(disp, "opioid", D0)
        with self.assertRaises(SystemExit):
            check_a_within_d0(disp, "opioid|misuses drugs", D0)

    def test_bad_lambda(self) -> None:
        with self.assertRaises(ValueError):
            permute(self.c, self.pids, self.fixed, 1.5, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
