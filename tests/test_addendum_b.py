"""Tests for the addendum B entries: scripts/addendum_b_lists.py, addendum_b_e2.py, addendum_draw_seeds.py."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import addendum_b_e2  # noqa: E402
import addendum_b_lists  # noqa: E402
import addendum_draw_seeds as ds  # noqa: E402
import e2_map  # noqa: E402
from test_e2_map import toy_cohort  # noqa: E402

V4 = {"first_year": 2023, "age_range_at_first_year": [25, 70], "d2": {"min_n_condition": 30},
      "attributes": {"drug_misuse": {"A_pattern": "drug", "D0": {"include": ["drug"], "exclude": []}, "D1": [{"display": "x", "group": "g"}]},
                     "diabetes_t2": {"A_pattern": "diab", "D0": {"include": ["diab"], "exclude": []}, "D1": []},
                     "other": {"A_pattern": "o", "D0": {"include": ["o"], "exclude": []}}},
      "D2_frozen": {"drug_misuse": {"displays": ["B", "A"], "lift": {"A": 2.0, "B": 3.0}},
                    "diabetes_t2": {"displays": ["C"], "lift": {"C": 4.0}}, "_source": {"cohort": "dev"}}}


class Lists(unittest.TestCase):
    def fake_compute(self, d2_dm=("A", "B")):
        def f(derived, lists):
            self.seen = lists
            return {"drug_misuse": {"displays": sorted(d2_dm)}, "diabetes_t2": {"displays": ["C"]},
                    "severe_anxiety": {"displays": ["K"], "lift": {"K": 3.8}}}
        return f

    def test_copies_v4_and_adds_anxiety(self):
        with mock.patch.object(addendum_b_lists, "compute", self.fake_compute()), \
                mock.patch.object(addendum_b_lists, "sha256", lambda p: "h"):
            out = addendum_b_lists.build(Path("dev"), V4)
        self.assertEqual(list(out["attributes"]), ["drug_misuse", "diabetes_t2", "severe_anxiety"])
        for k in ("drug_misuse", "diabetes_t2"):
            self.assertEqual(out["attributes"][k], V4["attributes"][k])
            self.assertEqual(out["D2_frozen"][k], V4["D2_frozen"][k])          # verbatim (lift too)
        self.assertEqual(out["D2_frozen"]["severe_anxiety"]["displays"], ["K"])
        self.assertEqual(out["D2_frozen"]["_source"]["copied_from_v4"], ["drug_misuse", "diabetes_t2"])
        self.assertEqual({k: out[k] for k in ("first_year", "age_range_at_first_year", "d2")},
                         {k: V4[k] for k in ("first_year", "age_range_at_first_year", "d2")})
        self.assertNotIn("other", self.seen["attributes"])

    def test_stops_when_recomputed_d2_differs(self):
        with mock.patch.object(addendum_b_lists, "compute", self.fake_compute(("A",))), \
                mock.patch.object(addendum_b_lists, "sha256", lambda p: "h"):
            with self.assertRaises(SystemExit):
                addendum_b_lists.build(Path("dev"), V4)


class E2Wrapper(unittest.TestCase):
    def setUp(self):
        self.env = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        for v in self.env:
            os.environ.pop(v, None)

    def tearDown(self):
        for v, x in self.env.items():
            os.environ.pop(v, None)
            if x is not None:
                os.environ[v] = x

    def test_runs_frozen_map_with_given_lists_and_threads(self):
        persons, c = toy_cohort()
        lists = {"first_year": 2023, "age_range_at_first_year": [25, 70], "d2": {},
                 "attributes": {"x": {"A_pattern": "AAA", "D0": {"include": ["AAA"], "exclude": []}, "D1": []}},
                 "D2_frozen": {"x": {"displays": ["C1"]}}}
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            lf = tmp / "lists.json"
            lf.write_text(json.dumps(lists))
            exp = tmp / "exp"
            exp.mkdir()
            (exp / "config.json").write_text(json.dumps({"lambdas": [1.0], "reps": 2, "min_feature_count": 3,
                                                         "bootstrap_persons": True, "primary_attribute": "x"}))
            (exp / "SEED").write_text("7\n")
            root0 = e2_map.ROOT
            with mock.patch.object(e2_map, "load_tables", lambda d: (persons, c)), contextlib.redirect_stdout(io.StringIO()):
                addendum_b_e2.main([str(tmp), str(exp), "--lists", str(lf), "--threads", "2"])
            self.assertEqual(e2_map.ROOT, root0)                                  # restored
            rt = json.loads((exp / "results" / "addendum_runtime.json").read_text())
            self.assertEqual(rt["threads"], 2)
            self.assertEqual(rt["lists_sha256"], addendum_b_e2.sha256(lf))
            self.assertTrue(all(i["num_threads"] <= 2 for i in rt["threadpool_info"]))
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            import pandas as pd
            reps = pd.read_csv(exp / "results" / "e2_map_reps.csv")
            self.assertEqual(set(reps.attribute), {"x"})
            self.assertTrue((exp / "results" / "e2_lambda1_original.csv").exists())

    def test_confirm_builds_exp_from_frozen_files_and_refuses_mismatches(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "pj"
            (root / "config").mkdir(parents=True)
            (root / "config" / "addendum.json").write_text(json.dumps({"B": {"threads": 4}}))
            (root / "config" / "addendum_b_condition_lists.json").write_text("{}")
            (root / "config" / "e2_confirm.json").write_text('{"reps": 200}')
            (root / "config" / "addendum_seeds.json").write_text(json.dumps(
                {"cohorts": [{"cohort": c, "seed": 10 * c, "clinicianSeed": 10 * c + 1} for c in (1, 2, 3)],
                 "e2_seed": {"1": 501, "2": 502, "3": 503}}))
            lists = root / "config" / "addendum_b_condition_lists.json"
            der = Path(t) / "derived-addendum-c2"
            der.mkdir()
            (der / "addendum.json").write_text(json.dumps({"cohort": 2, "seed": 20, "clinicianSeed": 21}))
            old = os.environ.pop("HTS_PREREG_FROZEN", None)
            try:
                with self.assertRaises(SystemExit):                                    # frozen flag missing
                    addendum_b_e2.prepare_confirm(2, der, Path(t) / "e0", lists, 4, root)
                os.environ["HTS_PREREG_FROZEN"] = "yes"
                for args in ((3, der, Path(t) / "e1", lists, 4), (2, der, Path(t) / "e2", lists, 2)):   # wrong cohort / threads
                    with self.assertRaises(SystemExit):
                        addendum_b_e2.prepare_confirm(*args, root)
                exp = Path(t) / "exp"
                rt = addendum_b_e2.prepare_confirm(2, der, exp, lists, 4, root)
                self.assertEqual((exp / "SEED").read_text().strip(), "502")
                self.assertEqual((exp / "config.json").read_bytes(), (root / "config" / "e2_confirm.json").read_bytes())
                self.assertEqual((rt["cohort"], rt["e2_seed"], rt["confirm"]), (2, 502, True))
                with self.assertRaises(SystemExit):                                    # an existing exp dir
                    addendum_b_e2.prepare_confirm(2, der, exp, lists, 4, root)
            finally:
                os.environ.pop("HTS_PREREG_FROZEN", None)
                if old is not None:
                    os.environ["HTS_PREREG_FROZEN"] = old

    def test_refuses_conflicting_thread_env(self):
        os.environ["OMP_NUM_THREADS"] = "8"
        with self.assertRaises(SystemExit):
            addendum_b_e2.main(["d", "e", "--lists", "l", "--threads", "4"])


class Seeds(unittest.TestCase):
    MAN = {"shards": [{"seed": 100 + i, "clinicianSeed": 200 + i} for i in range(4)]}

    def test_used_seeds(self):
        u = ds.used_seeds(self.MAN, 5, 6)
        self.assertTrue({42, 9001, 20261113, 100, 203, 5, 6} <= u)
        with self.assertRaises(SystemExit):
            ds.used_seeds({"shards": self.MAN["shards"][:3]}, 5, 6)

    def test_experiment_seeds_are_excluded(self):
        with tempfile.TemporaryDirectory() as t:
            for name, v in (("a", "777\n"), ("b", "888"), ("c", "not a seed")):
                (Path(t) / name).mkdir()
                (Path(t) / name / "SEED").write_text(v)
            u = ds.used_seeds(self.MAN, 5, 6, Path(t))
        self.assertTrue({777, 888} <= u)

    def test_draw_redraws_used_and_repeated_values(self):
        seq = iter([100, 1, 1, 2, 3, 42, 4, 5, 6, 7, 8, 9, 10, 11])
        out = ds.draw(ds.used_seeds(self.MAN, 5, 6), rand=lambda: next(seq))
        flat = [x for c in out["cohorts"] for x in (c["seed"], c["clinicianSeed"])] + list(out["e2_seed"].values())
        self.assertEqual(flat, [1, 2, 3, 4, 7, 8, 9, 10, 11])                 # drawn in the documented order
        self.assertEqual(len(set(flat)), 9)
        self.assertTrue(set(flat).isdisjoint(ds.used_seeds(self.MAN, 5, 6)))
        self.assertEqual([r["value"] for r in out["redraws"]], [100, 1, 42, 5, 6])
        self.assertEqual(out["cohorts"][0], {"cohort": 1, "seed": 1, "clinicianSeed": 2})


if __name__ == "__main__":
    unittest.main()
