"""Tests for the E2 helpers in scripts/e2_map.py (active / defend / multihot / utility_cost / v_acc).

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_map  # noqa: E402
from e2_map import (COARSE, active, bootstrap_persons, defend, multihot, oof, utility_cost, v_acc, v_bal, v_conf,  # noqa: E402
                    v_opt)

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


class Balanced(unittest.TestCase):
    def test_v_bal(self) -> None:
        y = np.array([1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(v_bal(np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.9]), y), 0.5 * 0.5 + 0.5 * 0.75)
        self.assertAlmostEqual(v_bal(np.full(6, 0.1), y), 0.5)             # always not-A: 1/2 whatever the prior
        self.assertAlmostEqual(v_bal(np.full(6, 0.9), y), 0.5)
        self.assertAlmostEqual(v_bal(np.full(6, 0.5), y), 0.5)             # p = 0.5 exactly: nobody predicted A
        self.assertAlmostEqual(v_acc(np.full(6, 0.5), y), 4 / 6)           # 判断 26: V_acc also p > 0.5 (0.5 -> not-A)


class Prereg3Helpers(unittest.TestCase):
    def test_v_opt_acts_above_the_sample_prior(self) -> None:
        """C14: the best rule for the balanced gain acts iff p > pi_hat = mean(y) (strict)."""
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0])             # pi_hat = 0.25
        p = np.array([0.3, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])
        # act: persons 0 (A), 2 (not-A); 0.2 < 0.25 and 0.1 do not act -> TPR 1/2, TNR 5/6
        self.assertAlmostEqual(v_opt(p, y), 0.5 * 0.5 + 0.5 * 5 / 6)
        self.assertAlmostEqual(v_bal(p, y), 0.5)           # at 0.5 nobody acts: 1/2
        self.assertAlmostEqual(v_opt(np.full(8, 0.25), y), 0.5)   # p == pi_hat exactly is not an action
        self.assertAlmostEqual(v_bal(p, y, 0.25), v_opt(p, y))


def toy_cohort(n: int = 300, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pid = np.array([f"p{i}" for i in range(n)])
    a = rng.random(n) < 0.25
    persons = pd.DataFrame({"pid": pid, "A": a, "birth_year": rng.integers(1960, 1995, n),
                            "gender": rng.choice(["male", "female"], n), "deceased": np.nan})
    rows = []
    for i, p in enumerate(pid):
        if a[i]:
            rows.append((p, "AAA label", 2010))
        for d, base, lift in (("C1", 0.15, 3.0), ("C2", 0.3, 1.0), ("C3", 0.2, 1.5)):
            if rng.random() < min(1.0, base * (lift if a[i] else 1.0)):
                rows.append((p, d, 2015))
    c = pd.DataFrame(rows, columns=["pid", "display", "onset_year"])
    c["abate_year"] = np.nan
    return persons, c


class Bootstrap(unittest.TestCase):
    def test_bootstrap_persons_stratified_and_deterministic(self) -> None:
        persons, c = toy_cohort()
        pb, cb = bootstrap_persons(persons, c, np.random.default_rng(5))
        self.assertEqual(len(pb), len(persons))
        self.assertEqual(int(pb.A.sum()), int(persons.A.sum()))            # strata keep their sizes
        self.assertEqual(pb.pid.is_unique, True)
        self.assertGreater(pb.orig_pid.duplicated().sum(), 0)              # drawn with replacement
        self.assertTrue(set(pb.orig_pid) <= set(persons.pid))
        # every draw carries a copy of its source person's condition rows
        n_src = c.groupby("pid").size()
        n_b = cb.groupby("pid").size()
        for newp, orig in zip(pb.pid, pb.orig_pid):
            self.assertEqual(int(n_b.get(newp, 0)), int(n_src.get(orig, 0)))
        pb2, cb2 = bootstrap_persons(persons, c, np.random.default_rng(5))
        self.assertTrue(pb.equals(pb2) and cb.equals(cb2))
        pb3, _ = bootstrap_persons(persons, c, np.random.default_rng(6))
        self.assertFalse(pb.orig_pid.equals(pb3.orig_pid))

    def test_oof_groups_never_split(self) -> None:
        rng = np.random.default_rng(0)
        g = np.repeat(np.arange(60), 3)                                    # each group drawn 3 times
        y = np.repeat(rng.integers(0, 2, 60), 3)
        X = rng.normal(size=(180, 2)) + y[:, None]
        from sklearn.model_selection import StratifiedGroupKFold
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=3).split(X, y, g):
            self.assertFalse(set(g[tr]) & set(g[te]))
        p = oof(X, y, 3, g)
        self.assertEqual(p.shape, (180,))
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertEqual(oof(X, y, 3).shape, (180,))                       # without groups: the earlier path

    def test_multihot_counts_distinct_original_persons_under_bootstrap(self) -> None:
        pids = pd.Index(["b0", "b1", "b2", "b3"])
        orig = pd.Series(["p1", "p1", "p1", "p2"], index=pids)             # p1 drawn three times
        rows = pd.DataFrame({"pid": ["b0", "b1", "b2", "b3"], "display": ["X", "X", "X", "Y"]})
        self.assertEqual(list(multihot(rows, pids, 2).columns), ["X"])              # per drawn person: X passes alone
        self.assertEqual(list(multihot(rows, pids, 2, orig).columns), [])           # one original person: not enough
        rows2 = pd.DataFrame({"pid": ["b0", "b1", "b3"], "display": ["X", "X", "X"]})
        m = multihot(rows2, pids, 2, orig)
        self.assertEqual(list(m.columns), ["X"])                                    # p1 and p2: passes
        self.assertEqual(m["X"].tolist(), [1, 1, 0, 1])                             # the draws keep their own rows

    def _run_main(self, boot: bool, reps: int = 6, primary: str | None = None) -> tuple[pd.DataFrame, list]:
        persons, c = toy_cohort()
        lists = {"first_year": 2023, "age_range_at_first_year": [25, 70], "d2": {},
                 "attributes": {"x": {"A_pattern": "AAA", "D0": {"include": ["AAA"], "exclude": []}, "D1": []}},
                 "D2_frozen": {"x": {"displays": ["C1"]}}}
        calls: list = []
        real_oof = e2_map.oof

        def spy(X, y, seed, groups=None):
            calls.append(groups)
            return real_oof(X, y, seed, groups)
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "config").mkdir()
            (tmp / "config" / "condition_lists.json").write_text(json.dumps(lists))
            exp = tmp / "exp"
            exp.mkdir()
            cfg = {"lambdas": [1.0], "reps": reps, "min_feature_count": 3}
            if boot:
                cfg["bootstrap_persons"] = True
            if primary:
                cfg["primary_attribute"] = primary
            (exp / "config.json").write_text(json.dumps(cfg))
            (exp / "SEED").write_text("7\n")
            with mock.patch.object(e2_map, "ROOT", tmp), \
                    mock.patch.object(e2_map, "load_tables", lambda d: (persons.assign(birth=None), c)), \
                    mock.patch.object(e2_map, "oof", spy), contextlib.redirect_stdout(io.StringIO()):
                e2_map.main(tmp, exp, True)
            f = exp / "results" / "e2_lambda1_original.csv"
            self.original = pd.read_csv(f) if f.exists() else None
            return pd.read_csv(exp / "results" / "e2_map_reps.csv"), calls

    def test_oof_groups_passed_only_with_bootstrap(self) -> None:
        _, calls = self._run_main(True, reps=2)
        self.assertGreater(len(calls), 0)
        for g in calls:
            self.assertIsNotNone(g)
            self.assertEqual(len(g), 300)
            # the group is the original pid: persons drawn twice share a group, so the groups are fewer than the rows
            self.assertLess(len(set(g.tolist())), len(g))
        _, calls0 = self._run_main(False, reps=2)
        self.assertGreater(len(calls0), 0)
        self.assertTrue(all(g is None for g in calls0))

    def test_main_lambda1_varies_between_reps_only_with_bootstrap(self) -> None:
        out = {boot: self._run_main(boot)[0] for boot in (True, False)}
        d0k2 = lambda df: df[(df.D == "D0") & (df.K == "K2")].sort_values("rep")  # noqa: E731
        vb, vn = d0k2(out[True]), d0k2(out[False])
        self.assertEqual(len(vb), 6)
        self.assertGreater(vb.Vacc_K.nunique(), 1)                         # the base rate / V(K) moves with the draw
        self.assertGreater(vb.dV_acc.nunique(), 1)
        self.assertGreater(vb.dV_bal.nunique(), 1)
        self.assertTrue(np.allclose(vb.dV_bal, vb.Vbal_KR - vb.Vbal_K))
        self.assertEqual(set(out[True].D), {"D0", "D2"})
        # lambda = 1, no bootstrap: the same persons every rep, so V(K) is identical across reps
        self.assertEqual(vn.Vacc_K.nunique(), 1)

    def test_reps_carry_dv_opt_and_dauc(self) -> None:
        """C14: per-rep columns dV_opt (best rule, p > pi_hat of the rep's sample) and dAUC = auc_KR - auc_K."""
        for boot in (True, False):
            df, _ = self._run_main(boot, reps=3)
            self.assertTrue(np.allclose(df.dAUC, df.auc_KR - df.auc_K))
            self.assertTrue(np.isfinite(df.dV_opt).all())
            self.assertGreater(df.dV_opt.nunique(), 1)
            self.assertFalse(np.allclose(df.dV_opt, df.dV_bal))        # a different rule gives different values
        # the earlier columns keep their order: attribute, lambda, rep, D, K first
        self.assertEqual(list(df.columns[:5]), ["attribute", "lambda", "rep", "D", "K"])

    def test_lambda1_original_sample_file(self) -> None:
        """C18: one CV on the original sample per lambda 1 D x K of the primary attribute, bootstrap_persons true only."""
        df, calls = self._run_main(True, reps=2, primary="x")
        o = self.original
        self.assertIsNotNone(o)
        self.assertEqual(sorted(zip(o.D, o.K)), [("D0", "K2"), ("D0", "K3"), ("D2", "K2"), ("D2", "K3")])
        self.assertEqual(set(o["lambda"]), {1.0})
        self.assertNotIn("rep", o.columns)
        self.assertEqual(o.cv_seed.nunique(), 1)                        # one cv seed for the original sample
        self.assertTrue(np.allclose(o.dAUC, o.auc_KR - o.auc_K))
        self.assertIn("dV_opt", o.columns)
        # the original CV is the one without groups (no resampling): 2 calls (K, K + R) per D x K = 8; the reps pass groups
        self.assertEqual(sum(g is None for g in calls), 8)
        self.assertEqual(sum(g is not None for g in calls), 2 * 2 * 2 * 2)     # reps x D x K x (K, K + R)
        # it does not depend on the number of reps
        o3 = None
        self._run_main(True, reps=3, primary="x")
        o3 = self.original
        pd.testing.assert_frame_equal(o, o3)
        # a descriptive value that differs from the rep values (the reps are bootstrap draws)
        rep_vals = df[(df.D == "D0") & (df.K == "K2")].dV_bal.to_numpy()
        self.assertFalse(np.any(np.isclose(o[(o.D == "D0") & (o.K == "K2")].dV_bal.iloc[0], rep_vals)))
        # the persons are the original ones: V(K) on the original sample is its balanced accuracy, not a resampled value
        self.assertTrue(np.isfinite(o.Vbal_K).all())

    def test_lambda1_original_sample_not_written_otherwise(self) -> None:
        self._run_main(False, reps=2, primary="x")              # no person bootstrap
        self.assertIsNone(self.original)
        self._run_main(True, reps=2)                            # no primary attribute in the config
        self.assertIsNone(self.original)
        self._run_main(True, reps=2, primary="other")           # primary is another attribute
        self.assertIsNone(self.original)


if __name__ == "__main__":
    unittest.main()
