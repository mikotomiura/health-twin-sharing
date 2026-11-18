"""Tests for scripts/addendum_a_summarise.py and scripts/addendum_b_summarise.py."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import addendum_a_summarise as asum  # noqa: E402
import addendum_b_summarise as bsum  # noqa: E402
import addendum_reattack as ar  # noqa: E402
import e1_summarise  # noqa: E402
from e1_run import Exp, load_jsonl  # noqa: E402

SRC = ROOT / "experiments" / "20261114-e1-numctx-check"
KS = ["K0", "K1", "K0n", "K2"]


@unittest.skipUnless((SRC / "results" / "twin.jsonl").exists(), "development releases not present")
class ASummarise(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.e1 = self.tmp / "e1"
        (self.e1 / "results").mkdir(parents=True)
        for f in ("config.json", "SEED"):
            shutil.copy(SRC / f, self.e1 / f)
        shutil.copytree(SRC / "inputs", self.e1 / "inputs")
        shutil.copy(SRC / "results" / "twin.jsonl", self.e1 / "results" / "twin.jsonl")
        self.res = self.tmp / "res"
        with contextlib.redirect_stdout(io.StringIO()):
            for m in ("m:a", "m:b", "m:c"):
                ar.main([str(self.e1), str(self.res), "--model", m, "--backend", "fake", "--ks", ",".join(KS)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, m=2):
        return {"A": {"attackers": ["m:a", "m:b", "m:c"], "ks": KS, "m_consistent": m}}

    def test_h1_equals_frozen_e1_summary_on_the_same_rows(self):
        x = Exp(self.e1)
        keys = {x.attack_key(j) for j in ar.jobs_for(x, KS, ar.subset(x, None))}
        rows = load_jsonl(self.res / "attack_m_a.jsonl")
        self.assertEqual({r["key"] for r in rows}, keys)
        got = asum.one(self.e1, rows + [rows[0] | {"p": 0.99}], keys)   # a duplicated key: the first row wins
        d = self.tmp / "ref"
        shutil.copytree(self.e1, d)
        with (d / "results" / "attack.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k not in asum.ADD}) + "\n")
        ref = e1_summarise.summarise(d)
        self.assertEqual(got["H1_K2_minus_K0"], ref["H1_K2_minus_K0"])
        self.assertEqual(got["H1_secondary_K2_minus_K1"], ref["H1_secondary_K2_minus_K1"])
        self.assertTrue(got["complete"])
        self.assertEqual(got["n_rows"], len(keys))

    def test_count_and_incomplete_attacker(self):
        f = self.res / "attack_m_c.jsonl"
        lines = f.read_text(encoding="utf-8").splitlines()
        f.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")       # m:c misses 3 planned reads
        out = asum.summarise(self.e1, self.res, self.cfg())
        self.assertFalse(out["attackers"]["m:c"]["complete"])
        self.assertFalse(out["attackers"]["m:c"]["consistent"])
        n = sum(v["consistent"] for v in out["attackers"].values())
        self.assertEqual(out["n_consistent"], n)
        self.assertEqual(out["generalises"], n >= 2)
        for v in out["attackers"].values():
            p = (v["H1_K2_minus_K0"] or {}).get("p_one_sided_permutation")
            self.assertEqual(v["consistent"], bool(v["complete"] and not v["missing_over_10pct"] and p is not None and p <= 0.025))
            self.assertTrue(all(r["lambda"] == 1 and r["D"] == "D0" for r in v["missing_over_10pct"]))

    def test_missing_over_10pct_is_not_consistent(self):
        f = self.res / "attack_m_a.jsonl"
        rows = load_jsonl(f)
        k2 = [r for r in rows if r["K"] == "K2" and r["t"] == 3]
        drop = {r["key"] for r in k2[: len(k2) // 5]}                      # 20% of the K2 t3 reads fail (p = null)
        with f.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r | ({"p": None} if r["key"] in drop else {})) + "\n")
        out = asum.summarise(self.e1, self.res, self.cfg())
        a = out["attackers"]["m:a"]
        self.assertTrue(a["complete"])
        self.assertTrue(any(r["K"] == "K2" for r in a["missing_over_10pct"]))
        self.assertFalse(a["consistent"])

    def test_missing_over_10pct_in_k1_only_does_not_block(self):
        f = self.res / "attack_m_b.jsonl"
        rows = load_jsonl(f)
        k1 = [r for r in rows if r["K"] == "K1" and r["t"] == 3]
        drop = {r["key"] for r in k1[: len(k1) // 5]}                      # 20% of the K1 t3 reads fail
        with f.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r | ({"p": None} if r["key"] in drop else {})) + "\n")
        b = asum.summarise(self.e1, self.res, self.cfg())["attackers"]["m:b"]
        self.assertEqual(b["missing_over_10pct"], [])
        self.assertTrue(all(r["K"] not in ("K2", "K0") for r in b["missing_over_10pct_other_K"]))  # K1 may not be a planned
        # condition of the development config (numctx-check has no K1 in its plan); the point is that K1 never blocks
        p = b["H1_K2_minus_K0"]["p_one_sided_permutation"]
        self.assertEqual(b["consistent"], bool(b["complete"] and p <= 0.025))

    def test_h0d_b_and_c_flag(self):
        a = asum.summarise(self.e1, self.res, self.cfg())["attackers"]["m:a"]
        h = a["H0d"] or {}
        parts = [(h.get(k) or {}).get("within_margin") for k in ("K1_minus_rate", "K2_minus_rate")]
        want = None if any(v is None for v in parts) else all(parts)
        self.assertEqual(a["H0d_b_and_c_within_margin"], want)
        self.assertEqual(a["qualified_H0d"], want is not True)

    def test_versions_must_be_confirmatory_with_registered_digest(self):
        cfg = self.cfg()
        cfg["A"] |= {"ollama": "0.40.16", "digests": {"m:a": "d1", "m:b": "d2", "m:c": "d3"}}
        with self.assertRaises(SystemExit):                                  # no versions.jsonl (fake backend)
            asum.summarise(self.e1, self.res, cfg)
        with (self.res / "versions.jsonl").open("w", encoding="utf-8") as fh:
            for m, d in cfg["A"]["digests"].items():
                fh.write(json.dumps({"file": ar.tag_of(m), "ollama": "0.40.16", "digest": d, "confirm": True}) + "\n")
        asum.summarise(self.e1, self.res, cfg)
        with (self.res / "versions.jsonl").open("a", encoding="utf-8") as fh:  # a later resume on another digest
            fh.write(json.dumps({"file": "m_b", "ollama": "0.40.16", "digest": "other", "confirm": True}) + "\n")
        with self.assertRaises(SystemExit):
            asum.summarise(self.e1, self.res, cfg)

    def test_reattack_confirm_checks(self):
        cfg = {"A": {"attackers": ["m:a"], "think": {"m:a": "low"}, "ks": KS, "ollama": "0.40.16", "digests": {"m:a": "dg"},
                     "twin_sha256": ar.sha256_file(self.e1 / "results" / "twin.jsonl")}}
        live = {"ollama": "0.40.16", "digest": "dg"}
        old = os.environ.pop("HTS_PREREG_FROZEN", None)
        try:
            with self.assertRaises(SystemExit):                              # frozen flag missing
                ar.confirm_checks(cfg, self.e1, "m:a", "low", KS, None, live)
            os.environ["HTS_PREREG_FROZEN"] = "yes"
            ar.confirm_checks(cfg, self.e1, "m:a", "low", KS, None, live)    # the registered run passes
            for args in (("m:a", "true", KS, None, live), ("m:a", "low", KS[:2], None, live), ("m:a", "low", KS, 4, live),
                         ("m:a", "low", KS, None, {"ollama": "0.41", "digest": "dg"}), ("m:a", "low", KS, None, {"ollama": "0.40.16", "digest": "x"}),
                         ("m:a", "low", KS, None, None), ("m:z", "low", KS, None, live)):
                with self.assertRaises(SystemExit):
                    ar.confirm_checks(cfg, self.e1, *args)
            with self.assertRaises(SystemExit):                              # another input
                ar.confirm_checks(cfg | {"A": cfg["A"] | {"twin_sha256": "0" * 64}}, self.e1, "m:a", "low", KS, None, live)
        finally:
            os.environ.pop("HTS_PREREG_FROZEN", None)
            if old is not None:
                os.environ["HTS_PREREG_FROZEN"] = old

    def test_think_value_names_the_file_the_cli_writes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ar.main([str(self.e1), str(self.res), "--model", "m:t", "--think", "low", "--backend", "fake", "--ks", ",".join(KS)])
            ar.main([str(self.e1), str(self.res), "--model", "m:u", "--think", "true", "--backend", "fake", "--ks", ",".join(KS)])
        cfg = {"A": {"attackers": ["m:t", "m:u", "m:a"], "think": {"m:t": "low", "m:u": True}, "ks": KS, "m_consistent": 2}}
        out = asum.summarise(self.e1, self.res, cfg)
        self.assertTrue(all(v["complete"] for v in out["attackers"].values()))
        self.assertEqual(ar.tag_of("m:u", True), "m_u_think-true")

    def test_unfilled_attackers_refused(self):
        with self.assertRaises(SystemExit):
            asum.summarise(self.e1, self.res, {"A": {"attackers": ["m:a", "__FILL_BEFORE_FREEZE__"], "ks": KS, "m_consistent": 2}})


def fake_reps(shift: dict, reps=20, seed=0, sd=0.002) -> pd.DataFrame:
    """Rows of e2_map_reps.csv for the planned cells; dV = shift[(attribute, lambda, D)] + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in ("drug_misuse", "diabetes_t2", "severe_anxiety"):
        ds = ["D0", "D2"] + (["D1"] if a == "drug_misuse" else [])
        for lam in (0.0, 1.0):
            for d in ds:
                for k in ("K2", "K3"):
                    for r in range(reps):
                        v = shift.get((a, lam, d), 0.0) + rng.normal(0, sd)
                        rows.append({"attribute": a, "lambda": lam, "rep": r, "D": d, "K": k, "dV_bal": v, "dV_acc": v,
                                     "dV_conf": v, "dV_opt": v, "dAUC": v})
    return pd.DataFrame(rows)


class BSummarise(unittest.TestCase):
    LISTS = {"attributes": {"drug_misuse": {"D1": [{"display": "x", "group": "g"}]}, "diabetes_t2": {"D1": []},
                            "severe_anxiety": {"D1": []}}}
    CFG = {"B": {"attributes": ["drug_misuse", "diabetes_t2", "severe_anxiety"], "n_cohorts": 3, "m_consistent_per_attribute": 2,
                 "ci": 0.95, "ci_equivalence": 0.90}}

    def exps(self, tmp: Path, shifts: list[dict], sd=0.002) -> list[Path]:
        out = []
        for i, s in enumerate(shifts):
            e = tmp / f"c{i}"
            (e / "results").mkdir(parents=True)
            (e / "config.json").write_text(json.dumps({"lambdas": [0.0, 1.0], "reps": 20, "bootstrap_persons": True,
                                                       "primary_attribute": "drug_misuse", "delta_E2": 0.005}))
            fake_reps(s, seed=i, sd=sd).to_csv(e / "results" / "e2_map_reps.csv", index=False)
            out.append(e)
        return out

    def test_counts_per_attribute(self):
        up = {(a, 1.0, "D0"): 0.05 for a in ("drug_misuse", "diabetes_t2", "severe_anxiety")}
        with tempfile.TemporaryDirectory() as t:
            ex = self.exps(Path(t), [up, up, {}])          # cohort 3: no D0 - D2 difference and no lambda effect
            m = bsum.summarise(ex, self.LISTS, self.CFG)
        for a in self.CFG["B"]["attributes"]:
            g = m["by_attribute"][a]
            self.assertEqual(g["primary_D0_minus_D2"]["n_consistent"], 2)
            self.assertTrue(g["primary_D0_minus_D2"]["generalises"])
            self.assertEqual(g["secondary_lambda1_minus_lambda0"]["n_consistent"], 2)
            self.assertFalse(m["cohorts"]["3"][a]["dV_bal"]["D0_minus_D2_lambda1_K2"]["consistent"])
            self.assertTrue(m["cohorts"]["3"][a]["dV_bal"]["lambda0_D0_K2_equivalence"]["equivalent"])

    def test_primary_needs_estimate_above_delta(self):
        small = {(a, 1.0, "D0"): 0.003 for a in ("drug_misuse", "diabetes_t2", "severe_anxiety")}   # lo > 0 but est < delta_E2 0.005
        with tempfile.TemporaryDirectory() as t:
            m = bsum.summarise(self.exps(Path(t), [small, small, small], sd=0.0003), self.LISTS, self.CFG)
        g = m["by_attribute"]["drug_misuse"]
        self.assertGreater(m["cohorts"]["1"]["drug_misuse"]["dV_bal"]["D0_minus_D2_lambda1_K2"]["lo"], 0)
        self.assertEqual(g["primary_D0_minus_D2"]["n_consistent"], 0)
        self.assertEqual(g["secondary_lambda1_minus_lambda0"]["n_consistent"], 3)

    def test_confirm_requires_runtime_of_each_cohort(self):
        cfg = {"B": self.CFG["B"] | {"confirm": True}}
        with tempfile.TemporaryDirectory() as t:
            ex = self.exps(Path(t), [{}, {}, {}])
            with self.assertRaises((SystemExit, FileNotFoundError)):
                bsum.summarise(ex, self.LISTS, cfg)

    def test_one_consistent_cohort_does_not_generalise(self):
        up = {("diabetes_t2", 1.0, "D0"): 0.05}
        with tempfile.TemporaryDirectory() as t:
            m = bsum.summarise(self.exps(Path(t), [up, {}, {}]), self.LISTS, self.CFG)
        self.assertFalse(m["by_attribute"]["diabetes_t2"]["primary_D0_minus_D2"]["generalises"])

    def test_refuses_wrong_number_of_cohorts_and_incomplete(self):
        with tempfile.TemporaryDirectory() as t:
            ex = self.exps(Path(t), [{}, {}, {}])
            with self.assertRaises(SystemExit):
                bsum.summarise(ex[:2], self.LISTS, self.CFG)
            df = pd.read_csv(ex[0] / "results" / "e2_map_reps.csv")
            df.iloc[:-1].to_csv(ex[0] / "results" / "e2_map_reps.csv", index=False)
            with self.assertRaises(SystemExit):
                bsum.summarise(ex, self.LISTS, self.CFG)


if __name__ == "__main__":
    unittest.main()
