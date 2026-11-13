"""Tests for the E2 helpers in scripts/e2_map.py (active / defend / multihot / utility_cost / v_acc).

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_map import COARSE, active, defend, multihot, utility_cost, v_acc, v_conf  # noqa: E402

Y0 = 2023


class E2Helpers(unittest.TestCase):
    def test_active_boundaries(self) -> None:
        c = pd.DataFrame({"pid": ["a", "a", "a", "a", "a"],
                          "display": ["onset_y0", "onset_after", "abated_y0", "abated_before", "ongoing"],
                          "onset_year": [2023, 2024, 2010, 2010, 2010],
                          "abate_year": [np.nan, np.nan, 2023, 2022, np.nan]})
        self.assertEqual(set(active(c, Y0).display), {"onset_y0", "abated_y0", "ongoing"})

    def test_defend_drops_and_groups(self) -> None:
        act = pd.DataFrame({"pid": ["a"] * 4, "display": ["Label", "Pain", "Migraine", "Cough"]})
        mapping = {"Pain": "a pain-related condition", "Migraine": COARSE}
        out = defend(act, {"Label"}, mapping)
        self.assertEqual(sorted(out.display), sorted(["a pain-related condition", COARSE, "Cough"]))
        two = defend(pd.DataFrame({"pid": ["a", "a"], "display": ["Pain", "Back pain"]}), set(),
                     {"Pain": "g", "Back pain": "g"})
        self.assertEqual(len(two), 1)

    def test_multihot_empty_and_order(self) -> None:
        pids = pd.Index(["b", "a", "c"])
        self.assertEqual(multihot(pd.DataFrame(columns=["pid", "display"]), pids, 1).shape, (3, 0))
        m = multihot(pd.DataFrame({"pid": ["a", "c", "a"], "display": ["X", "X", "Y"]}), pids, 2)
        self.assertEqual(list(m.columns), ["X"])
        self.assertEqual(m["X"].tolist(), [0, 1, 1])

    def test_utility_cost_counts_d0(self) -> None:
        act = pd.DataFrame({"pid": ["a", "a", "a", "a", "b"], "display": ["Label", "Pain", "Cough", "Flu", "Cough"]})
        pids = pd.Index(["a", "b", "nobody"])
        total, extra = utility_cost(act, pids, {"Label"}, {})
        self.assertAlmostEqual(total, (1 / 4 + 0) / 2)
        self.assertEqual(extra, 0.0)
        total, extra = utility_cost(act, pids, {"Label"}, {"Pain": "g"})
        self.assertAlmostEqual(total, (2 / 4 + 0) / 2)
        self.assertAlmostEqual(extra, (1 / 4) / 2)

    def test_v_acc_vs_conf(self) -> None:
        y = np.array([1, 0, 1, 0])
        p = np.array([0.9, 0.9, 0.9, 0.9])  # confident and half wrong
        self.assertAlmostEqual(v_conf(p), 0.9)
        self.assertAlmostEqual(v_acc(p, y), 0.5)


if __name__ == "__main__":
    unittest.main()
