"""Tests for the confirmatory E1 scripts (scripts/e1_build.py, e1_run.py, e1_summarise.py).

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e1_build import (COARSE, apply_defender, clamp_onset, defender_maps, knowledge, known_text, neutral_text,  # noqa: E402
                      sentences, year_record)
from llm_dry_run import render_record  # noqa: E402
from e1_run import (Exp, Sink, Skips, check_provenance, plan, releases_of, run_lock, stage_attack, stage_calib, stage_twin,  # noqa: E402
                    stage_utility, with_retries)
from llm_dry_run_attacker import FakePct  # noqa: E402
from e1_summarise import one_sample_sign_flip_p, sign_flip_p, summarise, v_of  # noqa: E402

D0 = {"include": ["drug abuse", "misuses drugs"], "exclude": []}
AT = {"D1": [{"display": "Chronic pain", "group": "a pain-related condition"},
             {"display": "Back pain", "group": "a pain-related condition"}]}
D2 = ["Migraine", "Homeless"]


class Build(unittest.TestCase):
    def test_defenders(self) -> None:
        maps = defender_maps(AT, D2)
        self.assertEqual(maps["D2"], {"Migraine": COARSE, "Homeless": COARSE})
        with self.assertRaises(SystemExit):  # a D2 display inside a D1 group would mix the group rates
            defender_maps(AT, ["Back pain"])
        conds = [{"display": "Drug abuse (disorder)", "onset_year": 2010}, {"display": "Chronic pain", "onset_year": 2015},
                 {"display": "Back pain", "onset_year": 2012}, {"display": "Migraine", "onset_year": 2012},
                 {"display": "Homeless", "onset_year": 2020}, {"display": "Cough", "onset_year": 2022}]
        shown, n_rm, n_co = apply_defender(conds, D0, {})
        self.assertEqual((n_rm, n_co), (1, 0))
        self.assertNotIn("Drug abuse (disorder)", [c["display"] for c in shown])
        shown, n_rm, n_co = apply_defender(conds, D0, maps["D1"])  # two D1 entries merge into one group, earliest onset
        self.assertEqual((n_rm, n_co), (1, 2))
        self.assertIn({"display": "a pain-related condition", "onset_year": 2012}, shown)
        self.assertEqual(sum(c["display"] == "a pain-related condition" for c in shown), 1)
        shown, _, n_co = apply_defender(conds, D0, maps["D2"])
        names = [c["display"] for c in shown]
        self.assertEqual(n_co, 2)
        self.assertEqual(names.count(COARSE), 1)  # Migraine and Homeless merge into one entry, earliest onset
        self.assertIn({"display": COARSE, "onset_year": 2012}, shown)
        self.assertIn("Chronic pain", names)  # not in D2: kept as is
        self.assertEqual(apply_defender([], D0, {}), ([], 0, 0))

    def test_k1_sentence_says_the_rate_is_population_wide(self) -> None:
        from e1_build import sentences
        kn = {"p_A": 0.2, "conditions": {}, "groups": {}, "none_of_d2": {"n": 1, "p_A": None}}
        k1 = sentences(kn)["K1"]
        self.assertIn("adults aged 25 to 70 in this population (not specific to this person's age or sex)", k1)
        self.assertNotIn("like this person", k1)

    def test_clamp_only_before_birth(self) -> None:
        c = pd.DataFrame({"pid": ["a", "a", "b"], "onset_year": [1970.0, 1990.0, np.nan]})
        out, mask = clamp_onset(c, pd.Series({"a": 1980, "b": 1985}))
        self.assertEqual(out.onset_year.tolist()[:2], [1980.0, 1990.0])
        self.assertTrue(np.isnan(out.onset_year.iloc[2]))
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertEqual(c.onset_year.iloc[0], 1970.0)  # input not modified

    def test_knowledge_rates_and_nesting(self) -> None:
        persons = pd.DataFrame({"pid": list("abcdef"), "A": [True, True, False, False, False, False]})
        c = pd.DataFrame({"pid": ["a", "b", "c", "d", "a"],
                          "display": ["Migraine", "Migraine", "Migraine", "Homeless", "Chronic pain"],
                          "onset_year": [2020, 2024, 2020, 2020, 2020],
                          "abate_year": [np.nan, np.nan, 2021, np.nan, np.nan]})  # c's migraine ended before 2023
        kn = knowledge(persons, c, [2023, 2024, 2025], D2, defender_maps(AT, D2))
        self.assertAlmostEqual(kn["p_A"], 2 / 6)
        self.assertEqual(kn["conditions"]["Migraine"], {"n": 2, "p_A": 1.0})  # a, b (b from 2024)
        self.assertEqual(kn["conditions"]["Homeless"], {"n": 1, "p_A": 0.0})
        self.assertEqual(kn["none_of_d2"]["n"], 3)  # c, e, f
        self.assertEqual(kn["groups"]["a pain-related condition"]["members"], ["Back pain", "Chronic pain"])
        self.assertEqual(kn["groups"]["a pain-related condition"], kn["groups"]["a pain-related condition"] | {"n": 1, "p_A": 1.0})
        self.assertEqual(kn["groups"][COARSE]["members"], ["Homeless", "Migraine"])
        self.assertEqual(kn["groups"][COARSE]["n"], 3)  # a, b (migraine), d (homeless)
        kn["sentences"] = sentences(kn)
        person = {"known_facts": {"age": 40, "gender": "F"}, "k3_facts": {"Migraine": True, "Homeless": False}}
        texts = [known_text(k, person, kn, "A {age}-year-old {gender}.") for k in ("K0", "K1", "K2", "K3")]
        for lo, hi in zip(texts, texts[1:]):
            self.assertTrue(hi.startswith(lo) and len(hi) > len(lo))
        self.assertEqual(texts[0], "A 40-year-old F.")
        self.assertIn('has "Migraine"', texts[3])
        self.assertIn('does not have "Homeless"', texts[3])


class Prereg2(unittest.TestCase):
    """判断 16 (no visit count) and 判断 20 (K0n)."""

    A_WORDS = ("drug", "misuse", "dependen", "abuse", "opioid", "overdose", "addict", "substance", "narcotic")

    def test_year_record_has_no_visits(self) -> None:
        person = pd.Series({"birth_year": 1980, "gender": "F"})
        rec = year_record(2023, person, [{"display": "Cough", "onset_year": 2020}], 1, 0)
        self.assertNotIn("visits_this_year", rec)
        self.assertEqual(rec["age"], 43)
        self.assertNotIn("Visits", render_record(rec))

    def test_render_record_unchanged_when_key_present(self) -> None:
        rec = {"age": 43, "gender": "F", "year": 2023, "visits_this_year": 5,
               "conditions": [{"display": "Cough", "onset_year": 2020}]}
        old = ("Age 43, F. Year 2023.\nVisits this year: 5.\nConditions on record (onset year):\n- Cough (2020)")
        self.assertEqual(render_record(rec), old)
        self.assertEqual(render_record(rec | {"conditions": []}),
                         "Age 43, F. Year 2023.\nVisits this year: 5.\nConditions on record (onset year):\n- none")
        self.assertEqual(render_record({k: v for k, v in rec.items() if k != "visits_this_year"}),
                         "Age 43, F. Year 2023.\nConditions on record (onset year):\n- Cough (2020)")

    def _kn(self) -> dict:
        persons = pd.DataFrame({"pid": list("abcdef"), "A": [True, True, False, False, False, False]})
        c = pd.DataFrame({"pid": ["a", "b", "c", "d"], "display": ["Migraine", "Migraine", "Migraine", "Homeless"],
                          "onset_year": [2020, 2024, 2020, 2020], "abate_year": [np.nan] * 4})
        kn = knowledge(persons, c, [2023, 2024, 2025], D2, defender_maps(AT, D2))
        kn["sentences"] = sentences(kn)
        return kn

    def test_k0n_is_neutral_and_length_matched(self) -> None:
        kn = self._kn()
        person = {"known_facts": {"age": 40, "gender": "F"}, "k3_facts": {"Migraine": True}}
        base = "A {age}-year-old {gender}."
        k0, k0n, k2 = (known_text(k, person, kn, base) for k in ("K0", "K0n", "K2"))
        self.assertTrue(k0n.startswith(k0))
        added = k0n[len(k0):].strip()
        self.assertFalse(any(ch.isdigit() for ch in added))
        self.assertNotIn("%", added)
        self.assertFalse(any(w in added.lower() for w in self.A_WORDS + ("rate", "percent", "statistic", "among")))
        n_sent = len(kn["sentences"]["K1"].split()) + len(kn["sentences"]["K2"].split())
        self.assertEqual(len(added.split()), n_sent)
        self.assertEqual(len(k0n.split()), len(k2.split()))                    # whole text: the same word count as K2
        self.assertEqual(k0n, known_text("K0n", person, kn, base))             # deterministic

    def test_neutral_text_lengths(self) -> None:
        for n in (1, 7, 60, 400):
            t = neutral_text(n)
            self.assertEqual(len(t.split()), n)
            self.assertTrue(t.endswith("."))
            self.assertFalse(any(ch.isdigit() for ch in t))
        self.assertEqual(neutral_text(0), "")

    def test_k0n_only_at_lambda1_d0(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            x = Exp(make_exp(tmp))
            self.assertIn("K0n", x.ks_for(1, "D0", 0))
            self.assertNotIn("K0n", x.ks_for(0, "D0", 0))
            self.assertNotIn("K0n", x.ks_for(1, "D2", 0))
            self.assertNotIn("K0n", x.ks_for(1, "D0", 5))                      # extra base-point paraphrases
            self.assertEqual(x.ks_for(0, "D1", 0), ["K2", "K3"])
            self.assertIn("K0n", {j[1] for j in x.attack_jobs()})
            self.assertTrue(x.known("K0n", "p0", 1).startswith(x.known("K0", "p0", 1)))
        finally:
            shutil.rmtree(tmp)


def make_exp(tmp: Path, n_a: int = 4, n_n: int = 4, d2=lambda i: False) -> Path:
    """A minimal exp dir with config, SEED, records and knowledge (no cohort needed)."""
    exp = tmp / "exp"
    (exp / "inputs").mkdir(parents=True)
    cfg = json.loads((ROOT / "config" / "e1_confirm.json").read_text(encoding="utf-8"))
    cfg |= {"n_A": n_a, "n_not_A": n_n}
    (exp / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (exp / "SEED").write_text("7\n")
    rows = []
    for i in range(n_a + n_n):
        for lam in cfg["lambdas"]:
            for d in cfg["defenders"]:
                rec = [{"year": y, "age": 40, "gender": "F",
                        "conditions": [{"display": "Cough", "onset_year": 2020}], "audit_removed": 0, "coarsened": 0}
                       for y in cfg["years"]]
                rows.append({"pid": f"p{i}", "A": i < n_a, "a_type": ("old" if i % 2 else "misuses_only") if i < n_a else None,
                             "lambda": lam, "D": d, "known_facts": {"age": 40, "gender": "F"},
                             "k3_facts": {"Migraine": bool(d2(i))}, "records": rec,
                             "records_full": [{"year": y, "conditions": r["conditions"]} for y, r in zip(cfg["years"], rec)]})
    (exp / "inputs" / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    kl = {"p_A": 0.2, "conditions": {}, "groups": {}, "none_of_d2": {"n": 1, "p_A": 0.2},
          "sentences": {"K1": "In this population 20% have it.", "K2": "k2."}}
    (exp / "inputs" / "knowledge.json").write_text(json.dumps({"pi": 0.2, "lambdas": {str(l): kl for l in cfg["lambdas"]}}),
                                                   encoding="utf-8")
    return exp


class Run(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_plan_counts_match_preregistration(self) -> None:
        x = Exp(make_exp(self.tmp, 24, 24))
        p = plan(x)
        n, s, y = 48, 3, 3
        self.assertEqual(p["twin_cells"], n * s * (2 * 3 * 2 + 6))         # lambda 2 x D 3 x 2 paraphrases + 6 extra at the base
        self.assertEqual(p["twin_calls"], p["twin_cells"] * y)
        self.assertEqual(p["attack_t0"], n * 2 * 4 + n)                    # person x lambda x K, plus K0n at lambda 1
        self.assertEqual(p["attack_t3"], n * 2 * s * 2 * (4 + 2 + 2) + n * 6 * s * 2 + n * s * 2)   # K0n: lambda 1, D0, 2 paraphrases
        self.assertEqual(p["calib_calls"], (12 + 24) * 2 * s)              # labelled A persons only, default paraphrases

    def test_plan_counts_at_40_plus_40(self) -> None:
        """Round 3 A13: the same formulas at the preregistered n_A = n_not_A = 40 (config/e1_confirm.json)."""
        cfg = json.loads((ROOT / "config" / "e1_confirm.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["n_A"], cfg["n_not_A"]), (40, 40))
        x = Exp(make_exp(self.tmp, 40, 40))
        p = plan(x)
        n, s, y = 80, 3, 3
        self.assertEqual(p["twin_cells"], n * s * (2 * 3 * 2 + 6))
        self.assertEqual(p["twin_calls"], p["twin_cells"] * y)
        self.assertEqual(p["attack_t0"], n * 2 * 4 + n)
        self.assertEqual(p["attack_t3"], n * 2 * s * 2 * (4 + 2 + 2) + n * 6 * s * 2 + n * s * 2)
        self.assertEqual(p["calib_calls"], (20 + 40) * 2 * s)
        # literal numbers (as printed by `e1_run.py <exp> --plan`)
        self.assertEqual((p["twin_cells"], p["twin_calls"], p["attack_t0"], p["attack_t3"], p["calib_calls"], p["utility_calls"]),
                         (4320, 12960, 720, 11040, 360, 8640))

    def test_calib_people(self) -> None:
        x = Exp(make_exp(self.tmp, 24, 24))
        self.assertEqual(x.calib_people(0), set())
        full = x.calib_people(1)
        self.assertEqual(len(full), 24)
        self.assertTrue(all(x.rows[(p, 1, "D0")]["A"] for p in full))
        half = x.calib_people(0.5)
        self.assertEqual(len(half), 12)
        self.assertEqual(half, x.calib_people(0.5))  # fixed by SEED

    def test_fake_end_to_end_and_resume(self) -> None:
        import subprocess
        exp = make_exp(self.tmp)
        cmd = [sys.executable, str(ROOT / "scripts" / "e1_run.py"), str(exp), "--stage", "all", "--backend", "fake"]
        subprocess.run(cmd, check=True, capture_output=True)
        res = lambda: {f.name: f.read_bytes() for f in (exp / "results").glob("*.jsonl") if f.name != "versions.jsonl"}  # noqa: E731
        before = res()
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        self.assertIn("[twin] 0 jobs", out)
        self.assertEqual(before, res())
        self.assertEqual(len((exp / "results" / "versions.jsonl").read_text().splitlines()), 2)  # appended per run
        p = plan(Exp(exp))
        n_lines = lambda f: len((exp / "results" / f).read_text().splitlines())  # noqa: E731
        self.assertEqual(n_lines("twin.jsonl"), p["twin_calls"])
        self.assertEqual(n_lines("attack.jsonl"), p["attack_t0"] + p["attack_t3"])
        self.assertEqual(n_lines("calib.jsonl"), p["calib_calls"])
        m = summarise(exp)
        self.assertIn("H1_K2_minus_K0", m)
        self.assertIn("H3_K2", m)
        self.assertTrue(m["completeness"]["complete"])
        self.assertEqual(m["completeness"]["utility"]["expected"], p["utility_calls"])
        self.assertEqual(m["completeness"]["utility"]["rows"], p["utility_calls"])
        # a missing utility row, or a row outside the plan, makes the run incomplete (the other stages stay complete)
        uf = exp / "results" / "utility.jsonl"
        lines = uf.read_text().splitlines()
        uf.write_text("\n".join(lines[:-1]) + "\n")
        m = summarise(exp)
        self.assertFalse(m["completeness"]["complete"])
        self.assertEqual(m["completeness"]["utility"]["rows"], p["utility_calls"] - 1)
        self.assertTrue(m["completeness"]["attack"]["rows"] == m["completeness"]["attack"]["expected"])
        uf.write_text("\n".join(lines) + "\n")
        self.assertTrue(summarise(exp)["completeness"]["complete"])
        uf.write_text("\n".join(lines + [json.dumps(dict(json.loads(lines[0]), key="p0|L1|D0|P9|s9|t1"))]) + "\n")
        m = summarise(exp)
        self.assertEqual(m["completeness"]["utility"]["unexpected_rows"], 1)
        self.assertFalse(m["completeness"]["complete"])                        # all planned rows present, one extra
        uf.write_text("\n".join(lines) + "\n")
        # another backend on the same results/ is refused
        bad = subprocess.run(cmd[:-1] + ["ollama", "--host", "http://127.0.0.1:9"], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("refusing to resume", bad.stderr)

    def test_fingerprint_refuses_changed_config(self) -> None:
        exp = make_exp(self.tmp)
        x = Exp(exp)
        check_provenance(x, "fake", FakePct())
        cfg = json.loads((exp / "config.json").read_text())
        (exp / "config.json").write_text(json.dumps(cfg | {"n_seeds": 2}))
        with self.assertRaises(SystemExit):
            check_provenance(Exp(exp), "fake", FakePct())

    def test_sink_drops_cut_last_line_only(self) -> None:
        f = self.tmp / "a.jsonl"
        f.write_bytes(b'{"key": "a"}\n{"key": "b"}\n{"key": "c", "p":')
        sk = Sink(f)
        self.assertEqual((sk.done, sk.dropped), ({"a", "b"}, 1))
        self.assertEqual(f.read_bytes(), b'{"key": "a"}\n{"key": "b"}\n')
        sk.write({"key": "c"})
        self.assertEqual(Sink(f).done, {"a", "b", "c"})
        f.write_bytes(b'{"key": "a"}\n{"key": \n{"key": "b"}\n')
        with self.assertRaises(SystemExit):
            Sink(f)

    def test_twin_failure_becomes_missing_downstream(self) -> None:
        exp = make_exp(self.tmp)
        x = Exp(exp)

        class EmptyForP0(FakePct):  # the twin answers p0 with an empty text, every time
            def chat(self, model, messages, seed, options, fmt=None):
                if fmt is None and "p0" in self.who(messages):
                    return {"text": "", "done_reason": "stop", "wall_s": 0, "eval_count": 0, "eval_s": 0,
                            "prompt_count": 0, "prompt_s": 0}
                return super().chat(model, messages, seed, options, fmt)

            @staticmethod
            def who(messages):
                return messages[-1]["content"]

        # mark p0's record so the backend can tell it apart
        rows = [json.loads(l) for l in (exp / "inputs" / "records.jsonl").read_text().splitlines()]
        for r in rows:
            if r["pid"] == "p0":
                for rec in r["records"]:
                    rec["conditions"] = [{"display": "p0", "onset_year": 2020}]
        (exp / "inputs" / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        x = Exp(exp)
        be, sk = EmptyForP0(), Skips()
        stage_twin(x, be, 2)
        stage_attack(x, be, 2, sk)
        stage_calib(x, be, 2, sk)
        stage_utility(x, be, 2, sk)
        self.assertEqual(sk.n, 0)
        tw = [json.loads(l) for l in (exp / "results" / "twin.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["failed"] == (r["pid"] == "p0") for r in tw))
        self.assertTrue(all(r["attempts"] == (1 + x.cfg["max_retries"] if r["t"] == 1 else 0)
                            for r in tw if r["pid"] == "p0"))  # later time points are not called
        at = [json.loads(l) for l in (exp / "results" / "attack.jsonl").read_text().splitlines()]
        p0_t3 = [r for r in at if r["pid"] == "p0" and r["t"] == 3]
        self.assertTrue(p0_t3 and all(r["p"] is None and r["reason"] == "twin_missing" for r in p0_t3))
        self.assertTrue(all(r["p"] is not None for r in at if r["pid"] != "p0" or r["t"] == 0))
        cal = [json.loads(l) for l in (exp / "results" / "calib.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["p"] is None for r in cal if r["pid"] == "p0"))
        ut = [json.loads(l) for l in (exp / "results" / "utility.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["items"] is None for r in ut if r["pid"] == "p0"))
        m = summarise(exp)
        self.assertEqual(m["completeness"]["attack"]["twin_missing"], len(p0_t3))
        self.assertFalse(m["completeness"]["attack"]["valid"] == m["completeness"]["attack"]["expected"])
        row = pd.read_csv(exp / "results" / "conditions.csv").query("`lambda` == 1 and D == 'D0' and K == 'K2'").iloc[0]
        self.assertEqual(row.n_persons_used, 7)  # p0 left out on both sides

    def test_retry_shifts_seed_and_truncation_counts_as_failure(self) -> None:
        exp = make_exp(self.tmp, 1, 1)
        seeds = []

        class Cut(FakePct):
            def chat(self, model, messages, seed, options, fmt=None):
                if fmt is None:
                    seeds.append(seed)
                    return {"text": "partial answer", "done_reason": "length", "wall_s": 0, "eval_count": 1,
                            "eval_s": 0, "prompt_count": 1, "prompt_s": 0}
                return super().chat(model, messages, seed, options, fmt)

        x = Exp(exp)
        stage_twin(x, Cut(), 1)
        n = x.cfg["max_retries"] + 1
        first = seeds[:n]
        self.assertEqual(len(set(first)), n)
        self.assertEqual([b - first[0] for b in first], [i * 1_000_003 for i in range(n)])
        tw = [json.loads(l) for l in (exp / "results" / "twin.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["failed"] for r in tw))
        self.assertTrue(all(r["done_reason"] == "length" for r in tw if r["t"] == 1))
        self.assertTrue(all(r["reason"] == "earlier_time_point_failed" for r in tw if r["t"] > 1))

    # ---- 判断 pre-freeze fixes: ctx_overflow, HTTP errors, lock, first-row-wins, attempt_reasons

    @staticmethod
    def _ans(pc: int, ec: int, text: str = "an answer", dr: str = "stop") -> dict:
        return {"text": text, "done_reason": dr, "wall_s": 0, "eval_count": ec, "eval_s": 0, "prompt_count": pc, "prompt_s": 0}

    def test_ctx_overflow_is_a_failed_attempt_and_shifts_the_seed(self) -> None:
        exp = make_exp(self.tmp, 1, 1)
        x = Exp(exp)
        self.assertEqual([x.cfg[k]["num_ctx"] for k in ("twin_options", "attacker_options", "judge_options")], [8192] * 3)
        seeds = []

        class Over(FakePct):  # the first twin attempt of every call fills the window: prompt + generation == num_ctx
            def chat(self, model, messages, seed, options, fmt=None):
                if fmt is None:
                    seeds.append(seed)
                    if len(seeds) % 2 == 1:
                        return Run._ans(8000, 192)
                return super().chat(model, messages, seed, options, fmt)

        stage_twin(x, Over(), 1)
        tw = [json.loads(l) for l in (exp / "results" / "twin.jsonl").read_text().splitlines()]
        self.assertTrue(all(not r["failed"] and r["attempts"] == 2 for r in tw))
        self.assertTrue(all(r["attempt_reasons"] == ["ctx_overflow", None] for r in tw))  # the fake gives no done_reason
        self.assertEqual(seeds[1] - seeds[0], 1_000_003)
        self.assertIn("done_reason", tw[0])                                     # the existing keys stay
        # always over the window: missing after max_retries + 1 attempts
        (self.tmp / "b").mkdir()
        exp2 = make_exp(self.tmp / "b", 1, 1)

        class Always(FakePct):
            def chat(self, model, messages, seed, options, fmt=None):
                return Run._ans(4100, 4096) if fmt is None else super().chat(model, messages, seed, options, fmt)

        x2 = Exp(exp2)
        stage_twin(x2, Always(), 1)
        tw = [json.loads(l) for l in (exp2 / "results" / "twin.jsonl").read_text().splitlines()]
        t1 = [r for r in tw if r["t"] == 1]
        self.assertTrue(all(r["failed"] and r["attempt_reasons"] == ["ctx_overflow"] * 3 for r in t1))

    def test_ctx_overflow_applies_to_every_stage_and_needs_num_ctx(self) -> None:
        ok = lambda o: True  # noqa: E731

        class B:
            def chat(self, model, messages, seed, options, fmt=None):
                return Run._ans(100, 100)
        out, att, good, rsn = with_retries(B(), "m", [], 1, {"num_ctx": 200}, None, ok, 2)       # 200 >= 200: overflow
        self.assertEqual((good, att, rsn), (False, 2, ["ctx_overflow"] * 3))
        out, att, good, rsn = with_retries(B(), "m", [], 1, {"num_ctx": 201}, None, ok, 2)       # 200 < 201: fine
        self.assertEqual((good, att, rsn), (True, 0, ["stop"]))
        out, att, good, rsn = with_retries(B(), "m", [], 1, {}, None, ok, 2)                     # no num_ctx: not checked
        self.assertEqual((good, att, rsn), (True, 0, ["stop"]))
        exp = make_exp(self.tmp, 1, 1)
        x = Exp(exp)

        class OverAll(FakePct):  # attacker and judge always overflow; the twin is fine
            def chat(self, model, messages, seed, options, fmt=None):
                out = super().chat(model, messages, seed, options, fmt)
                return out | {"prompt_count": 8190, "eval_count": 10} if fmt is not None else out
        sk = Skips()
        be = OverAll()
        stage_twin(x, be, 1)
        stage_attack(x, be, 1, sk)
        stage_calib(x, be, 1, sk)
        stage_utility(x, be, 1, sk)
        for name, key in (("attack", "p"), ("calib", "p"), ("utility", "items")):
            rows = [json.loads(l) for l in (exp / "results" / f"{name}.jsonl").read_text().splitlines()]
            ran = [r for r in rows if r.get("reason") != "twin_missing"]
            self.assertTrue(ran, name)
            self.assertTrue(all(r[key] is None and r["attempt_reasons"] == ["ctx_overflow"] * 3 for r in ran), name)

    def test_server_not_answering_is_not_an_attempt(self) -> None:
        """Review 2026-11-14 HIGH: a server error is retried with the SAME seed after waits; it never becomes a missing read."""
        import http.client
        import e1_run
        errs = [urllib.error.URLError("refused"), urllib.error.HTTPError("u", 500, "boom", {}, None), socket.timeout("slow")]
        waits = []
        old = (e1_run._sleep, e1_run.HTTP_WAITS)
        e1_run._sleep, e1_run.HTTP_WAITS = waits.append, (1, 2, 3)
        e1_run.SERVER_DOWN.clear()
        try:
            class Flaky:
                def __init__(self):
                    self.seeds = []

                def chat(self, model, messages, seed, options, fmt=None):
                    self.seeds.append(seed)
                    if len(self.seeds) <= 3:
                        raise errs[len(self.seeds) - 1]
                    return Run._ans(10, 10)
            be = Flaky()
            out, att, good, rsn = with_retries(be, "m", [], 5, {"num_ctx": 8192}, None, lambda o: True, 3)
            self.assertEqual((good, att, rsn), (True, 0, ["stop"]))      # the three errors are not attempts
            self.assertEqual(len(set(be.seeds)), 1)                       # same seed every time
            self.assertEqual(waits, [1, 2, 3])
            # a stage whose server stays down: it stops (ServerDown) and writes no row for the failed reads; on resume they run
            exp = make_exp(self.tmp, 1, 1)
            x = Exp(exp)

            class Down(FakePct):
                down = True

                def chat(self, model, messages, seed, options, fmt=None):
                    if self.down:
                        raise http.client.RemoteDisconnected("gone")
                    return super().chat(model, messages, seed, options, fmt)
            be = Down()
            with self.assertRaises(e1_run.ServerDown):
                stage_twin(x, be, 2)
            self.assertTrue(e1_run.SERVER_DOWN.is_set())
            twin = exp / "results" / "twin.jsonl"
            self.assertEqual(twin.read_text() if twin.exists() else "", "")
            e1_run.SERVER_DOWN.clear()
            be.down = False
            stage_twin(x, be, 2)
            tw = [json.loads(l) for l in twin.read_text().splitlines()]
            self.assertTrue(tw and not any(r["failed"] for r in tw))
        finally:
            e1_run._sleep, e1_run.HTTP_WAITS = old
            e1_run.SERVER_DOWN.clear()

    def test_lock_stops_a_second_process_and_is_removed(self) -> None:
        res = self.tmp
        with run_lock(res):
            self.assertTrue((res / ".lock").exists())
            with self.assertRaises(SystemExit) as cm:
                with run_lock(res):
                    self.fail("the second run must not start")
            self.assertIn("another process", str(cm.exception))
            self.assertTrue((res / ".lock").exists())                            # the failed second run did not remove it
        self.assertFalse((res / ".lock").exists())                               # normal end
        with self.assertRaises(RuntimeError):
            with run_lock(res):
                raise RuntimeError("boom")
        self.assertFalse((res / ".lock").exists())                               # exception
        with run_lock(res):                                                      # usable again
            pass
        # through main: a leftover lock stops the run before any model call
        import subprocess
        (self.tmp / "m").mkdir()
        exp = make_exp(self.tmp / "m", 1, 1)
        (exp / "results").mkdir(exist_ok=True)
        (exp / "results" / ".lock").write_text("pid 1\n")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "e1_run.py"), str(exp), "--stage", "twin", "--backend", "fake"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("another process", r.stderr)
        self.assertFalse((exp / "results" / "twin.jsonl").exists())
        (exp / "results" / ".lock").unlink()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "e1_run.py"), str(exp), "--stage", "twin", "--backend", "fake"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((exp / "results" / ".lock").exists())

    def test_first_written_row_wins_for_a_duplicated_key(self) -> None:
        exp = make_exp(self.tmp, 1, 1)
        x = Exp(exp)
        base = {"cell": "c1", "t": 1, "failed": False, "key": "c1|t1"}
        (exp / "results" / "twin.jsonl").write_text(
            "\n".join(json.dumps(r) for r in (base | {"text": "first"}, base | {"text": "second"})) + "\n")
        self.assertEqual(releases_of(x)["c1"][1]["text"], "first")

    def test_summarise_counts_any_attempt_cut_as_truncated(self) -> None:
        exp = make_exp(self.tmp, 4, 4)

        def pfun(k, t, pid, A, lam, d):
            if k == "K2" and t == 3 and pid == "p0" and lam == 1 and d == "D0":
                return None
            return 0.2 if t == 0 else 0.1
        Summarise._write_attack(exp, pfun)  # a staticmethod
        f = exp / "results" / "attack.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        mine = sorted((r for r in rows if r["p"] is None and r["paraphrase"] == 0), key=lambda r: r["seed_idx"])
        self.assertEqual(len(mine), 3)
        mine[0] |= {"done_reason": "stop", "attempt_reasons": ["ctx_overflow", "stop", "stop"]}   # any attempt: counts
        mine[1] |= {"done_reason": "length"}                                                      # no attempt_reasons: last done_reason
        mine[2] |= {"done_reason": "length", "attempt_reasons": ["stop", "stop", "stop"]}         # attempt_reasons win: not counted
        for r in rows:
            if r["p"] is None and r["paraphrase"] != 0:
                r |= {"done_reason": "stop", "attempt_reasons": ["stop"]}
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        s = summarise(exp)["missing_sensitivity_H1"]
        self.assertEqual(s["n_missing_truncated"], 2)

    def test_unfinished_twin_gives_exit_code_3(self) -> None:
        import subprocess
        exp = make_exp(self.tmp, 1, 1)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "e1_run.py"), str(exp), "--stage", "attack",
                            "--backend", "fake"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        self.assertIn("reads skipped", r.stdout)

    def test_digest_change_and_missing_model_refused(self) -> None:
        exp = make_exp(self.tmp, 1, 1)

        class V(FakePct):
            def __init__(self, d):
                self.d = d

            def versions(self, models):
                return {"ollama": "x", "models": {m: self.d for m in models}}

        check_provenance(Exp(exp), "fake", V("sha256:a"))
        with self.assertRaises(SystemExit):
            check_provenance(Exp(exp), "fake", V("sha256:b"))
        exp2 = make_exp(self.tmp / "two", 1, 1)
        with self.assertRaises(SystemExit):
            check_provenance(Exp(exp2), "fake", V(None))

    def test_expected_digests_and_ollama_checked_every_run(self) -> None:
        """Round 3 A7: --expect-digests / --expect-ollama are compared with the live values on the first run and on resumes."""
        exp = make_exp(self.tmp, 1, 1)
        x = Exp(exp)
        models = sorted({x.cfg["twin_model"], x.cfg["attacker_model"], x.cfg["judge_model"]})

        class V(FakePct):
            def __init__(self, d, ollama="0.9.0"):
                self.d, self.o = d, ollama

            def versions(self, models):
                return {"ollama": self.o, "models": {m: self.d for m in models}}

        good = {m: "sha256:a" for m in models}
        # not given: nothing is checked (the fake backend and the dev runs work as before)
        check_provenance(Exp(exp), "fake", V("sha256:a"))
        check_provenance(Exp(exp), "fake", V("sha256:a"), good, "0.9.0")                        # a resume with matching values
        for bad_d, bad_o in (({**good, models[0]: "sha256:z"}, "0.9.0"), (good, "0.9.1"), ({models[0]: "sha256:a"}, "0.9.0"),
                             ({**good, "other:1b": "sha256:a"}, "0.9.0")):                 # an expected model not in use stops too
            n_before = len((exp / "results" / "versions.jsonl").read_text().splitlines())
            with self.assertRaises(SystemExit) as cm:
                check_provenance(Exp(exp), "fake", V("sha256:a"), bad_d, bad_o)
            self.assertIn("freeze", str(cm.exception))
            self.assertEqual(len((exp / "results" / "versions.jsonl").read_text().splitlines()), n_before)   # refused before it logs
        exp2 = make_exp(self.tmp / "two", 1, 1)                                                  # the first run is checked too
        with self.assertRaises(SystemExit):
            check_provenance(Exp(exp2), "fake", V("sha256:a"), {m: "sha256:b" for m in models}, None)
        import subprocess
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "e1_run.py"), str(exp2), "--stage", "twin", "--backend", "fake",
                            "--expect-digests", "nomodel"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("bad item", r.stderr)

    def test_results_without_fingerprint_refused(self) -> None:
        exp = make_exp(self.tmp, 1, 1)
        (exp / "results").mkdir(exist_ok=True)
        (exp / "results" / "twin.jsonl").write_text('{"key": "old"}\n')
        with self.assertRaises(SystemExit):
            check_provenance(Exp(exp), "fake", FakePct())

    def test_sink_edge_cases(self) -> None:
        f = self.tmp / "b.jsonl"
        f.write_bytes(b'{"key": "a"}\n\n{"key": "b"}\n{"key": "c"}')  # blank line kept; valid last line w/o newline dropped
        sk = Sink(f)
        self.assertEqual((sk.done, sk.dropped), ({"a", "b"}, 1))
        self.assertEqual(f.read_bytes(), b'{"key": "a"}\n\n{"key": "b"}\n')
        f.write_bytes(b'{"key": "a"}\n{"key": \n')  # broken last full line
        self.assertEqual(Sink(f).done, {"a"})
        self.assertEqual(f.read_bytes(), b'{"key": "a"}\n')

    def test_cut_or_badly_typed_answers_are_failures(self) -> None:
        exp = make_exp(self.tmp, 1, 1)
        x = Exp(exp)

        class Bad(FakePct):  # the twin fails only at the third time point; the attacker and the judge are cut / mistyped
            def chat(self, model, messages, seed, options, fmt=None):
                if fmt is None and "Year 2025" in messages[-1]["content"] and "p0" in messages[-1]["content"]:
                    return {"text": "x", "done_reason": "length", "wall_s": 0, "eval_count": 1, "eval_s": 0,
                            "prompt_count": 1, "prompt_s": 0}
                out = super().chat(model, messages, seed, options, fmt)
                if fmt is not None and model == x.cfg["attacker_model"]:
                    return out | {"done_reason": "length"}  # parses, but was cut
                if fmt is not None and "items" not in out["text"] and "verdict" in out["text"]:
                    g = json.loads(out["text"])
                    return out | {"text": json.dumps({k: "false" for k in g if k != "note"} | {"note": ""})}
                return out

        rows = [json.loads(l) for l in (exp / "inputs" / "records.jsonl").read_text().splitlines()]
        for r in rows:
            if r["pid"] == "p0":
                for rec in r["records"]:
                    rec["conditions"] = [{"display": "p0", "onset_year": 2020}]
        (exp / "inputs" / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        x = Exp(exp)
        be, sk = Bad(), Skips()
        stage_twin(x, be, 1)
        stage_attack(x, be, 1, sk)
        stage_utility(x, be, 1, sk)
        at = [json.loads(l) for l in (exp / "results" / "attack.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["p"] is None for r in at))
        ut = [json.loads(l) for l in (exp / "results" / "utility.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["items"] is None for r in ut))
        p0 = [r for r in ut if r["pid"] == "p0"]
        self.assertTrue(p0 and all(r.get("reason") == "twin_missing" for r in p0))  # t1, t2 too: cell level

    def test_resume_inside_a_cell(self) -> None:
        exp = make_exp(self.tmp, 1, 1)
        x = Exp(exp)
        stage_twin(x, FakePct(), 1)
        f = exp / "results" / "twin.jsonl"
        full = f.read_text().splitlines()
        kept = [l for l in full if not l.endswith('"t": 3}') and json.loads(l)["t"] != 3]
        f.write_text("\n".join(kept) + "\n")
        stage_twin(Exp(exp), FakePct(), 1)
        again = f.read_text().splitlines()
        self.assertEqual(len(again), len(full))
        by = lambda ls: {json.loads(l)["key"]: json.loads(l)["text"] for l in ls}  # noqa: E731
        self.assertEqual(by(again), by(full))  # same history -> the same fake t3 answers


class Summarise(unittest.TestCase):
    def test_v_of(self) -> None:
        self.assertAlmostEqual(float(v_of(np.zeros(4), np.zeros(4), 0.2)), 0.8)  # everyone guessed not-A: V = 1 - pi
        self.assertAlmostEqual(float(v_of(np.ones(4), np.zeros(4), 0.2)), 1.0)
        self.assertAlmostEqual(float(v_of(np.array([1, 0, 1, 0.]), np.array([0, 0, 0, 1.]), 0.2)),
                               0.2 * 0.5 + 0.8 * 0.75)

    def test_v_of_balanced_default(self) -> None:
        self.assertAlmostEqual(float(v_of(np.array([1, 0, 1, 0.]), np.array([0, 0, 0, 1.]))), 0.5 * 0.5 + 0.5 * 0.75)

    def test_sign_flip_p_form_and_range(self) -> None:
        rng = np.random.default_rng(3)
        signs = rng.choice(np.array([-1.0, 1.0]), (199, 6))
        c = np.array([1.0, 0, -1, 0, 1, 0])
        p = float(sign_flip_p(c, 3, signs))
        self.assertAlmostEqual(p * 200, round(p * 200))                      # (b + 1) / (B + 1)
        self.assertTrue(1 / 200 <= p <= 1.0)
        obs = 0.5 * c[:3].mean() - 0.5 * c[3:].mean()                          # recompute b by hand
        flipped = 0.5 * (signs[:, :3] * c[:3]).mean(1) - 0.5 * (signs[:, 3:] * c[3:]).mean(1)
        self.assertAlmostEqual(p, ((flipped >= obs - 1e-12).sum() + 1) / 200)
        both = sign_flip_p(np.vstack([c, -c]), 3, signs)                       # batched rows agree with single calls
        self.assertAlmostEqual(both[0], p)
        self.assertAlmostEqual(both[1], float(sign_flip_p(-c, 3, signs)))

    def test_sign_flip_p_all_zero_is_one_and_strong_effect_is_small(self) -> None:
        signs = np.random.default_rng(4).choice(np.array([-1.0, 1.0]), (999, 20))
        self.assertEqual(float(sign_flip_p(np.zeros(20), 10, signs)), 1.0)
        strong = np.r_[np.ones(10), np.zeros(10)]                              # every A person +1, not-A 0
        self.assertLess(float(sign_flip_p(strong, 10, signs)), 0.01)
        self.assertTrue(np.isnan(sign_flip_p(np.ones(5), 5, signs[:, :5])))   # empty not-A group

    def test_h1_permutation_p_in_metrics(self) -> None:
        """H1 estimate = 1/2 mean_A(c) - 1/2 mean_notA(c) with c_i the per-person contribution; p keys present, bootstrap p kept."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp)
            x = Exp(exp)
            rows = []
            for cell in x.cells():
                pid, lam, d, k, s = cell
                A = x.rows[(pid, lam, d)]["A"]
                for kk in x.ks_for(lam, d, k):
                    hit = kk == "K2" and A and int(pid[1:]) < 2 and lam == 1
                    rows.append({"key": f"{kk}|{x.ckey(cell)}|t3", "K": kk, "pid": pid, "A": A, "lambda": lam, "t": 3,
                                 "D": d, "paraphrase": k, "seed_idx": s, "p": 0.9 if hit else 0.1})
            for j in x.attack_jobs():
                if j[0] == "t0":
                    _, kk, pid, lam = j
                    rows.append({"key": f"{kk}|{pid}|L{lam}|t0", "K": kk, "pid": pid, "A": x.rows[(pid, lam, "D0")]["A"],
                                 "lambda": lam, "t": 0, "D": None, "paraphrase": None, "seed_idx": None, "p": 0.2})
            (exp / "results").mkdir(exist_ok=True)
            (exp / "results" / "attack.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            m = summarise(exp)
            h = m["H1_K2_minus_K0"]
            c = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])             # p0, p1 caught under K2 only; A = p0..p3
            self.assertAlmostEqual(h["est"], 0.5 * c[:4].mean() - 0.5 * c[4:].mean())
            self.assertEqual(h["test"], "sign-flip permutation")
            self.assertEqual(h["n_flips"], 10000)
            self.assertTrue(0 < h["p_one_sided_permutation"] <= 1)
            # the same sign matrix rebuilt by hand: stream [SEED, 0x5F1], B = bootstrap n flips x 8 persons, A persons first
            signs = np.random.default_rng([7, 0x5F1]).choice([-1., 1.], (x.cfg["bootstrap"]["n"], 8))
            self.assertEqual(signs.shape, (10000, 8))
            self.assertEqual(h["p_one_sided_permutation"], float(sign_flip_p(c, 4, signs)))
            self.assertAlmostEqual(h["p_one_sided_permutation"] * 10001, round(h["p_one_sided_permutation"] * 10001))
            self.assertIn("p_one_sided", h)                                      # bootstrap p kept (reporting only)
            self.assertEqual(summarise(exp)["H1_K2_minus_K0"]["p_one_sided_permutation"], h["p_one_sided_permutation"])
            s_ = m["H1_secondary_K2_minus_K1"]                                      # round 3 A10: the secondary carries it too
            self.assertEqual(s_["test"], "sign-flip permutation")
            self.assertTrue(0 < s_["p_one_sided_permutation"] <= 1)
            self.assertNotIn("p_one_sided_permutation", m["H1_control_K2_minus_K0n"])   # descriptive contrasts do not
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_known_answer(self) -> None:
        """Hand-made attack reads: at lambda 1, D0, K2 half of the A persons are caught, no false alarms; K0 catches
        nobody. Then dV(K2) = 0.5 * 0.5 (balanced, primary) or pi * 0.5 (pi-weighted, secondary); dV(K0) = 0."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp)
            x = Exp(exp)
            rows = []
            for cell in x.cells():
                pid, lam, d, k, s = cell
                A = x.rows[(pid, lam, d)]["A"]
                for kk in x.ks_for(lam, d, k):
                    hit = kk in ("K2", "K3") and A and int(pid[1:]) < 2 and lam == 1
                    rows.append({"key": f"{kk}|{x.ckey(cell)}|t3", "K": kk, "pid": pid, "A": A, "lambda": lam, "t": 3,
                                 "D": d, "paraphrase": k, "seed_idx": s, "p": 0.9 if hit else 0.1})
            for j in x.attack_jobs():
                if j[0] == "t0":
                    _, kk, pid, lam = j
                    rows.append({"key": f"{kk}|{pid}|L{lam}|t0", "K": kk, "pid": pid, "A": x.rows[(pid, lam, "D0")]["A"],
                                 "lambda": lam, "t": 0, "D": None, "paraphrase": None, "seed_idx": None, "p": 0.2})
            (exp / "results").mkdir(exist_ok=True)
            (exp / "results" / "attack.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            m = summarise(exp)
            self.assertEqual(m["primary_estimator"], "balanced")
            self.assertAlmostEqual(m["H1_K2_minus_K0"]["est"], 0.25)
            self.assertAlmostEqual(m["H1_K2_minus_K0"]["pi_weighted"]["est"], 0.2 * 0.5)
            self.assertIn("p_one_sided", m["H1_K2_minus_K0"])
            for key in ("H1_secondary_K2_minus_K1", "H1_control_K2_minus_K0n", "H2_secondary_D0_minus_D2"):
                self.assertIn("pi_weighted", m[key])
            self.assertIn("pi_weighted", m["H0c_E1_side"]["lambda1_minus_lambda0_D0_K2"])
            self.assertIn("pi_weighted", m["H0a_lambda0_D0_K2_descriptive"])
            self.assertAlmostEqual(m["H0a_lambda0_D0_K2_descriptive"]["est"], 0.0)
            self.assertNotIn("H0a_lambda0_D0_K2", m)
            self.assertNotIn("supported", m["H1_K2_minus_K0"])                       # the decision is e2_summarise's
            self.assertIn("unadjusted_interval_criterion", m["H1_K2_minus_K0"])
            self.assertIn("H3_K2", m)
            self.assertIn("H3_K3_descriptive", m)                                   # H3 is judged on K2 only
            self.assertNotIn("supported", m["H3_K3_descriptive"])
            self.assertIn("supported", m["H3_K2"])
            # secondary K2 - K1 (K1 reads are 0.1 everywhere at lambda 1, D0): same as K2 - K0 here
            self.assertAlmostEqual(m["H1_secondary_K2_minus_K1"]["est"], 0.25)
            self.assertIn("supported", m["H1_secondary_K2_minus_K1"])
            # K0n control: reads 0.1 like K0, so K2 - K0n = K2 - K0
            self.assertAlmostEqual(m["H1_control_K2_minus_K0n"]["est"], 0.25)
            self.assertNotIn("supported", m["H1_control_K2_minus_K0n"])
            self.assertAlmostEqual(m["H2_secondary_D0_minus_D2"]["est"], 0.0)
            m2 = summarise(exp)  # bootstrap is fixed by SEED
            self.assertEqual(m["H1_K2_minus_K0"], m2["H1_K2_minus_K0"])

            # 判断 26: p = 0.5 exactly is not an action (hits read 0.5 -> nobody acts -> dV = 0, at tau 0.5 and 0.9)
            at_path = exp / "results" / "attack.jsonl"
            orig = at_path.read_text()
            half = [dict(r_, p=0.5 if r_["p"] == 0.9 else r_["p"]) for r_ in rows]
            at_path.write_text("\n".join(json.dumps(r_) for r_ in half) + "\n")
            mh = summarise(exp)
            self.assertAlmostEqual(mh["H1_K2_minus_K0"]["est"], 0.0)
            self.assertAlmostEqual(mh["H1_K2_minus_K0"]["pi_weighted"]["est"], 0.0)
            at_path.write_text(orig)

            # H0b: labelled persons read p = 0.95 (act); the others keep the attack reads
            def calib(rows_fn):
                cr = []
                for q, cell in x.calib_jobs():
                    pid, lam, d, k, s = cell
                    cr.append({"key": f"q{q}|{x.ckey(cell)}|t3", "q": q, "K": "K2", "pid": pid, "A": True,
                               "lambda": lam, "D": d, "paraphrase": k, "seed_idx": s, "t": 3, "p": rows_fn(q, pid)})
                (exp / "results" / "calib.jsonl").write_text("\n".join(json.dumps(r) for r in cr) + "\n")
            calib(lambda q, pid: 0.95)
            h = summarise(exp)["H0b"]
            # attack reads at lambda 1, D0, K2: p0, p1 (A) 0.9, p2, p3 (A) 0.1; t0 = 0.2. q = 1 -> all A at 0.95
            self.assertAlmostEqual(h["ledger_A_by_q"]["0"], (0.7 + 0.7 - 0.1 - 0.1) / 4)
            self.assertAlmostEqual(h["ledger_A_by_q"]["1"], 0.75)
            self.assertAlmostEqual(h["dV_by_q"]["1"], 0.5)        # every A caught, no false alarm: dV = 1/2 (balanced)
            self.assertAlmostEqual(h["dV_pi_weighted_by_q"]["1"], 0.2)  # ... and pi (secondary)
            self.assertEqual(h["labelled_persons_missing_by_q"], {"0": 0, "0.5": 0, "1": 0})
            # round 3 A10: each step carries the one-sided one-sample sign-flip p on the per-person (A) ledger differences
            lab = x.calib_people(0.5)
            base = {"p0": 0.9, "p1": 0.9, "p2": 0.1, "p3": 0.1}
            d_i = np.array([0.95 - (0.95 if pid in lab else base[pid]) for pid in base])            # step 1 - 0.5, A persons p0..p3
            st = h["steps"]["1-0.5"]
            self.assertAlmostEqual(st["est"], d_i.mean())
            self.assertEqual(st["test"], "one-sample sign-flip permutation")
            self.assertEqual(st["n_flips"], 10000)
            sg = np.random.default_rng([7, 0x5F2]).choice([-1., 1.], (10000, 4))
            self.assertEqual(st["p_one_sided_permutation"], one_sample_sign_flip_p(d_i, sg))
            self.assertIn("p_one_sided", st)                                                          # bootstrap p stays
            self.assertIn(h["supported"], (True, False))                                              # decision rule untouched
            # every labelled read missing: they stay missing, they are NOT replaced by the attack reads
            calib(lambda q, pid: None)
            h = summarise(exp)["H0b"]
            self.assertEqual(h["labelled_persons_missing_by_q"]["1"], 4)
            self.assertIsNone(h["ledger_A_by_q"]["1"])  # NaN written as null (strict JSON)
            self.assertEqual(h["n_persons_by_q"]["1"], 0)
            self.assertIsNone(h["steps"]["1-0.5"]["est"])
            self.assertFalse(h["supported"])
            json.loads((exp / "metrics.json").read_text())  # parses as strict JSON
            # one labelled person (q = 1) missing: paired steps drop that person from both sides
            calib(lambda q, pid: None if pid == "p1" else 0.95)
            h = summarise(exp)["H0b"]
            self.assertEqual(h["labelled_persons_missing_by_q"]["1"], 1)
            self.assertEqual(h["steps"]["1-0.5"]["n_A_persons"], 3)
            self.assertAlmostEqual(h["ledger_A_by_q"]["1"], 0.75)  # p0, p2, p3: 0.95 - 0.2

            # missing reads and duplicated keys
            lines = (exp / "results" / "attack.jsonl").read_text().splitlines()
            rows2 = [json.loads(l) for l in lines]
            for r in rows2:
                if r["pid"] == "p0" and r["t"] == 3 and r["K"] == "K2" and r["lambda"] == 1 and r["D"] == "D0":
                    r["p"] = None
            (exp / "results" / "attack.jsonl").write_text("\n".join(json.dumps(r) for r in rows2 + rows2[:5]) + "\n")
            m3 = summarise(exp)
            self.assertEqual(m3["completeness"]["attack"]["duplicates_dropped"], 5)
            row = pd.read_csv(exp / "results" / "conditions.csv").query("`lambda` == 1 and D == 'D0' and K == 'K2'").iloc[0]
            self.assertEqual((row.n_missing, row.n_persons_used), (2 * 3, 7))
            self.assertAlmostEqual(m3["H1_K2_minus_K0"]["est"], 0.5 * (1 / 3))  # p1 caught of the 3 A left
            self.assertEqual(m3["H1_K2_minus_K0"]["n_persons"], 7)
            # 判断 31: p0 (1 of 8 persons) unusable -> over 10% -> H1 re-computed with the missing reads filled
            s = m3["missing_sensitivity_H1"]
            self.assertAlmostEqual(s["fill_no_action"]["est"], 0.5 * (1 / 4))   # p0 counted as not caught
            self.assertAlmostEqual(s["fill_action"]["est"], 0.5 * (2 / 4))      # p0 counted as caught
            self.assertEqual(s["fill_action"]["n_persons"], 8)
            self.assertEqual(s["n_missing_reads"], 6)
            self.assertEqual(s["n_never_written"], 0)
            self.assertNotIn("missing_sensitivity_H1", m)                       # nothing over 10% before
            # the same reads never written (a stopped run) are filled from the plan, with the same result
            kept = [r for r in rows2 if not (r["pid"] == "p0" and r["t"] == 3 and r["K"] == "K2" and r["lambda"] == 1
                                             and r["D"] == "D0")]
            (exp / "results" / "attack.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + "\n")
            s4 = summarise(exp)["missing_sensitivity_H1"]
            self.assertAlmostEqual(s4["fill_no_action"]["est"], 0.5 * (1 / 4))
            self.assertAlmostEqual(s4["fill_action"]["est"], 0.5 * (2 / 4))
            self.assertEqual(s4["fill_action"]["n_persons"], 8)
            self.assertEqual((s4["n_missing_reads"], s4["n_never_written"]), (6, 6))
            # an empty attack file is refused, not summarised as zeros
            (exp / "results" / "attack.jsonl").write_text("")
            with self.assertRaises(SystemExit):
                summarise(exp)
        finally:
            shutil.rmtree(tmp)

    @staticmethod
    def _write_attack(exp: Path, pfun) -> None:
        """Attack rows for the whole plan; pfun(K, t, pid, A, lam, D) -> p (None = a missing read)."""
        x = Exp(exp)
        rows = []
        for cell in x.cells():
            pid, lam, d, k, s = cell
            A = x.rows[(pid, lam, d)]["A"]
            for kk in x.ks_for(lam, d, k):
                rows.append({"key": f"{kk}|{x.ckey(cell)}|t3", "K": kk, "pid": pid, "A": A, "lambda": lam, "t": 3, "D": d,
                             "paraphrase": k, "seed_idx": s, "p": pfun(kk, 3, pid, A, lam, d)})
        for j in x.attack_jobs():
            if j[0] == "t0":
                _, kk, pid, lam = j
                A = x.rows[(pid, lam, "D0")]["A"]
                rows.append({"key": f"{kk}|{pid}|L{lam}|t0", "K": kk, "pid": pid, "A": A, "lambda": lam, "t": 0, "D": None,
                             "paraphrase": None, "seed_idx": None, "p": pfun(kk, 0, pid, A, lam, None)})
        (exp / "results").mkdir(exist_ok=True)
        (exp / "results" / "attack.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_h0d_manipulation_check(self) -> None:
        """H0d (判断 44): t0 reads at lambda 1. 6 A + 6 not-A; even persons have a D2 display (k3_facts). K3 t0 = 0.6 with a D2
        display, 0.1 without; K2 = K1 = 0.2 = the stated 20%; K0 / K0n 0.1. Gap of p0(K3) - p0(K2) = 0.4 - (-0.1) = 0.5 > 0,
        K1 / K2 exactly at the rate -> supported. K1 at 0.4 -> outside the 0.05 margin -> False. No D2 display -> not judged (None)."""
        def run(k1: float, d2) -> dict:
            tmp = Path(tempfile.mkdtemp())
            try:
                exp = make_exp(tmp, 6, 6, d2)

                def pfun(k, t, pid, A, lam, d):
                    if t == 3:
                        return 0.1
                    pos = d2(int(pid[1:]))
                    return {"K3": 0.6 if pos else 0.1, "K2": 0.2, "K1": k1}.get(k, 0.1)
                self._write_attack(exp, pfun)
                return summarise(exp)["H0d"]
            finally:
                shutil.rmtree(tmp)
        even = lambda i: i % 2 == 0  # noqa: E731
        h = run(0.2, even)
        self.assertEqual(h["stated_rate"], 0.2)
        self.assertAlmostEqual(h["K3_minus_K2_D2_gap"]["est"], 0.5)
        self.assertEqual((h["K3_minus_K2_D2_gap"]["n_persons_with_D2"], h["K3_minus_K2_D2_gap"]["n_persons_without_D2"]), (6, 6))
        self.assertTrue(h["K3_minus_K2_D2_gap"]["supported"])
        self.assertAlmostEqual(h["K1_minus_rate"]["est"], 0.0)
        self.assertTrue(h["K1_minus_rate"]["within_margin"] and h["K2_minus_rate"]["within_margin"])
        self.assertEqual(h["K1_minus_rate"]["ci"], 0.9)
        self.assertTrue(h["supported"])
        self.assertAlmostEqual(h["descriptive"]["mean_p0"]["K0"], 0.1)
        bad = run(0.4, even)
        self.assertFalse(bad["K1_minus_rate"]["within_margin"])
        self.assertTrue(bad["K2_minus_rate"]["within_margin"])
        self.assertFalse(bad["supported"])
        self.assertEqual(h["K3_minus_K2_D2_gap"]["ci"], 0.95)
        # min_group boundary: 4 persons with a D2 display -> not judged; 5 -> judged
        self.assertIsNone(run(0.2, lambda i: i < 4)["K3_minus_K2_D2_gap"]["supported"])
        self.assertTrue(run(0.2, lambda i: i < 5)["K3_minus_K2_D2_gap"]["judged"])
        none = run(0.2, lambda i: False)
        self.assertFalse(none["K3_minus_K2_D2_gap"]["judged"])
        self.assertIsNone(none["K3_minus_K2_D2_gap"]["supported"])
        self.assertIsNone(none["supported"])

    def test_h0d_interval_not_point_decides(self) -> None:
        """H0d (b) is decided on the 90% interval, not the estimate: K1 t0 p alternates 0.1 / 0.3 around the stated 20%
        (mean exactly at the rate) so the interval of the 12-person mean reaches beyond +-0.05 -> not within the margin.
        A missing K3 t0 read leaves that person out of the (a) groups."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp, 6, 6, lambda i: i % 2 == 0)

            def pfun(k, t, pid, A, lam, d):
                i = int(pid[1:])
                if t == 3:
                    return 0.1
                if k == "K1":
                    return 0.1 if i % 4 < 2 else 0.3
                if k == "K3":
                    return None if i == 0 else (0.6 if i % 2 == 0 else 0.1)
                return 0.2 if k == "K2" else 0.1
            self._write_attack(exp, pfun)
            h = summarise(exp)["H0d"]
        finally:
            shutil.rmtree(tmp)
        self.assertAlmostEqual(h["K1_minus_rate"]["est"], 0.0)
        self.assertTrue(h["K1_minus_rate"]["lo"] < -0.05 or h["K1_minus_rate"]["hi"] > 0.05)
        self.assertFalse(h["K1_minus_rate"]["within_margin"])
        self.assertEqual(h["K3_minus_K2_D2_gap"]["n_persons_with_D2"], 5)    # p0 has no K3 t0 read
        self.assertEqual(h["descriptive"]["t0_missing"]["K3"], 1)
        self.assertFalse(h["supported"])

    def test_secondaries_dv_opt_and_dauc(self) -> None:
        """C14 (判断 35): beside H1 the best rule (act iff p > pi) and the AUC change, same persons and draws. 4 A (p0-p3) and
        4 not-A (p4-p7), pi = 0.2 (knowledge.json). K2, lambda 1, D0, t3: A p = 0.9, 0.9, 0.1, 0.1; not-A p = 0.3, 0.3, 0.05, 0.05;
        every other read (t0 and K0) is 0.2 / 0.1."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp)

            def pfun(k, t, pid, A, lam, d):
                if k == "K2" and t == 3 and lam == 1 and d == "D0":
                    i = int(pid[1:])
                    return [0.9, 0.9, 0.1, 0.1, 0.3, 0.3, 0.05, 0.05][i]
                return 0.2 if t == 0 else 0.1
            self._write_attack(exp, pfun)
            m = summarise(exp)
            h1 = m["H1_K2_minus_K0"]
            self.assertAlmostEqual(h1["est"], 0.25)                  # primary, tau 0.5: two A caught, nobody else acts
            # best rule p > 0.2: t3 K2 acts = p0, p1 (A) and p4, p5 (not-A): V = 1/2 * 1/2 + 1/2 * (1 - 1/2) = 0.5; t0 (0.2) nobody
            # acts: V = 1/2. dV(K2) = 0; K0 = 0 (0.1 / 0.2 never above 0.2) -> contrast 0
            self.assertAlmostEqual(h1["dV_opt"]["est"], 0.0)
            self.assertEqual(h1["dV_opt"]["rule"], "act iff p > pi")
            self.assertEqual(h1["dV_opt"]["pi"], 0.2)
            self.assertEqual(h1["dV_opt"]["n_persons"], 8)
            # AUC at t3 (K2): A [.9 .9 .1 .1] vs not-A [.3 .3 .05 .05]: wins 4 + 4 + 2 + 2 of 16 pairs... 12/16 = 0.75;
            # t0 all 0.2: 0.5 -> change 0.25; K0 0.1 vs 0.2 constants: 0.5 - 0.5 = 0 -> contrast 0.25
            self.assertAlmostEqual(h1["dAUC"]["est"], 0.25)
            for key in ("dV_opt", "dAUC"):
                self.assertIsNotNone(h1[key]["lo"])
                self.assertIn("p_one_sided", h1[key])
            self.assertNotIn("dV_opt", m["H1_secondary_K2_minus_K1"])   # only the H1 contrast carries the secondaries
            self.assertEqual(h1["dV_opt"], summarise(exp)["H1_K2_minus_K0"]["dV_opt"])   # fixed by SEED
        finally:
            shutil.rmtree(tmp)

    def test_auc_of(self) -> None:
        from e1_summarise import auc_of
        self.assertAlmostEqual(float(auc_of(np.array([0.9, 0.1]), np.array([0.5, 0.5]))), 0.5)       # one win, one loss
        self.assertAlmostEqual(float(auc_of(np.array([0.5, 0.5]), np.array([0.5]))), 0.5)            # ties count 1/2
        self.assertAlmostEqual(float(auc_of(np.array([np.nan, 0.9]), np.array([0.1, np.nan]))), 1.0)  # NaN persons are left out
        self.assertTrue(np.isnan(auc_of(np.array([np.nan]), np.array([0.1]))))
        self.assertEqual(auc_of(np.array([[0.9, 0.9], [0.1, 0.1]]), np.array([[0.5], [0.5]])).tolist(), [1.0, 0.0])

    def test_fill_adverse(self) -> None:
        """C16 (判断 38): reads of K2 / K0 filled so as to push H1 down. p0-p3 are A. Base: K2 t3 catches p0, p1 (0.9), everything else 0.1
        (t3) / 0.2 (t0); then two missing groups: the K2 t3 reads of the not-A p4 and the K0 t0 read of the not-A p5."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp)

            def pfun(k, t, pid, A, lam, d):
                if k == "K2" and t == 3 and pid == "p4" and lam == 1 and d == "D0":
                    return None
                if k == "K0" and t == 0 and pid == "p5" and lam == 1:
                    return None
                if k == "K2" and t == 3 and lam == 1 and d == "D0" and pid in ("p0", "p1"):
                    return 0.9
                return 0.2 if t == 0 else 0.1
            self._write_attack(exp, pfun)
            m = summarise(exp)
            s = m["missing_sensitivity_H1"]
            # no action: p4 K2 t3 -> 0, p5 K0 t0 -> 0: V3(K2) = 1/2 * 1/2 + 1/2 = 3/4, V0 = 1/2: dV(K2) = 1/4, dV(K0) = 0 -> 1/4
            self.assertAlmostEqual(s["fill_no_action"]["est"], 0.25)
            # action: p4 acts at K2 t3: V3 = 1/4 + 1/2 * 3/4 = 5/8, dV(K2) = 1/8; p5 acts at K0 t0: V0 = 1/2 * 3/4 = 3/8 ... = 3/8,
            # V3(K0) = 1/2 -> dV(K0) = 1/8: contrast 0
            self.assertAlmostEqual(s["fill_action"]["est"], 0.0)
            # adverse: K2 t3 not-A -> 1 (as "action"), K0 t0 not-A -> 1 (V0(K0) lowers, dV(K0) rises): contrast 0
            self.assertAlmostEqual(s["fill_adverse"]["est"], 0.0)
            # by condition (literal wording): K2 reads not-A -> 1, K0 reads not-A -> 0 (no action at t0): dV(K0) = 0: 1/8
            self.assertAlmostEqual(s["fill_adverse_by_condition"]["est"], 0.125)
            for k in ("fill_no_action", "fill_action", "fill_adverse", "fill_adverse_by_condition"):
                self.assertEqual(s[k]["n_persons"], 8)
            # adverse is the lowest of the fills
            self.assertLessEqual(s["fill_adverse"]["est"], min(s[k]["est"] for k in ("fill_no_action", "fill_action")) + 1e-12)
            self.assertEqual(s["n_missing_reads"], 6 + 1)
        finally:
            shutil.rmtree(tmp)

    def test_fill_adverse_a_person(self) -> None:
        """An A person's missing reads: K2 t3 -> 0 (pushes dV(K2) down), K2 t0 -> 1 (raises V(t0)); K0 t3 -> 1, K0 t0 -> 0."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp)

            def pfun(k, t, pid, A, lam, d):
                if pid == "p0" and lam == 1 and ((k == "K2" and t == 3 and d == "D0") or (k == "K0" and t == 3 and d == "D0")):
                    return None
                if k == "K2" and t == 3 and lam == 1 and d == "D0" and pid in ("p0", "p1"):
                    return 0.9
                return 0.2 if t == 0 else 0.1
            self._write_attack(exp, pfun)
            s = summarise(exp)["missing_sensitivity_H1"]
            # K2 t3 of p0 -> 0: A act at K2 t3 = [0, 1, 0, 0]: V3 = 1/8 + 1/2 = 5/8, dV(K2) = 1/8
            # K0 t3 of p0 -> 1: A act at K0 t3 = [1, 0, 0, 0]: V3(K0) = 1/8 + 1/2, V0 = 1/2: dV(K0) = 1/8: contrast 0
            self.assertAlmostEqual(s["fill_adverse"]["est"], 0.0)
            self.assertAlmostEqual(s["fill_no_action"]["est"], 1 / 8)       # K2 t3 -> 0: 1/8; K0 t3 -> 0: 0
            self.assertAlmostEqual(s["fill_action"]["est"], 1 / 4 - 1 / 8)  # K2 t3 p0 -> 1: dV 1/4; K0 t3 -> 1: dV 1/8
        finally:
            shutil.rmtree(tmp)

    def test_one_sample_sign_flip_p(self) -> None:
        """Round 3 A10: p = (b + 1) / (B + 1), b = flipped means >= observed (1e-12 tolerance); NaN for an empty sample."""
        signs = np.array([[1., 1., 1.], [-1., 1., 1.], [-1., -1., 1.], [-1., -1., -1.]])
        # observed mean 1; flipped means 1, 1/3, -1/3, -1: only the all-plus row reaches it -> b = 1, p = 2 / 5
        self.assertAlmostEqual(one_sample_sign_flip_p(np.ones(3), signs), 2 / 5)
        self.assertAlmostEqual(one_sample_sign_flip_p(np.zeros(3), signs), 1.0)          # all ties count as >=: p = 1
        self.assertAlmostEqual(one_sample_sign_flip_p(-np.ones(3), signs), 1.0)          # observed -1: every flipped mean >= -1, b = 4
        self.assertTrue(np.isnan(one_sample_sign_flip_p(np.array([]), np.ones((4, 0)))))

    def test_fill_permutation_p(self) -> None:
        """Round 3 A1: every fill of the missing-read sensitivity carries the sign-flip permutation p of H1 on its own persons."""
        tmp = Path(tempfile.mkdtemp())
        try:
            exp = make_exp(tmp)

            def pfun(k, t, pid, A, lam, d):
                if k == "K2" and t == 3 and pid == "p4" and lam == 1 and d == "D0":
                    return None
                if k == "K2" and t == 3 and lam == 1 and d == "D0" and pid in ("p0", "p1"):
                    return 0.9
                return 0.2 if t == 0 else 0.1
            self._write_attack(exp, pfun)
            m = summarise(exp)
            s = m["missing_sensitivity_H1"]
            for k in ("fill_no_action", "fill_action", "fill_adverse", "fill_adverse_by_condition"):
                self.assertEqual(s[k]["test"], "sign-flip permutation", k)
                self.assertTrue(0 < s[k]["p_one_sided_permutation"] <= 1, k)
                self.assertIn("p_one_sided", s[k])                                          # bootstrap p stays, secondary
            # fill_no_action: p4 (not-A) K2 t3 -> no action; contributions c = [1, 1, 0, 0 | 0, 0, 0, 0] as in the no-missing case
            c = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            signs = np.random.default_rng([7, 0x5F1]).choice([-1., 1.], (10000, 8))
            self.assertEqual(s["fill_no_action"]["p_one_sided_permutation"], float(sign_flip_p(c, 4, signs)))
            # filling "action" for the not-A p4 makes c_4 = 1 - 0.2... (acts at K2 t3, not at t0): a negative contribution -> larger p
            self.assertGreaterEqual(s["fill_action"]["p_one_sided_permutation"], s["fill_no_action"]["p_one_sided_permutation"])
        finally:
            shutil.rmtree(tmp)

    def test_interval_all_missing(self) -> None:
        from e1_summarise import interval
        self.assertIsNone(interval(float("nan"), np.full(10, np.nan), 0.95)["est"])
        self.assertIsNone(interval(0.1, np.full(10, np.nan), 0.95)["p_one_sided"])
        r = interval(0.1, np.array([np.nan, 0.2, -0.1, 0.3]), 0.95)
        self.assertEqual((r["n_draws"], r["p_one_sided"]), (3, (1 + 1) / (3 + 1)))   # 判断 32: (b + 1) / (B + 1)


if __name__ == "__main__":
    unittest.main()
