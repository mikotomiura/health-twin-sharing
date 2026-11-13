"""Tests for the confirmatory E1 scripts (scripts/e1_build.py, e1_run.py, e1_summarise.py).

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e1_build import COARSE, apply_defender, clamp_onset, defender_maps, knowledge, known_text, sentences  # noqa: E402
from e1_run import Exp, Sink, Skips, check_provenance, plan, stage_attack, stage_calib, stage_twin, stage_utility  # noqa: E402
from llm_dry_run_attacker import FakePct  # noqa: E402
from e1_summarise import summarise, v_of  # noqa: E402

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


def make_exp(tmp: Path, n_a: int = 4, n_n: int = 4) -> Path:
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
                rec = [{"year": y, "age": 40, "gender": "F", "visits_this_year": 1,
                        "conditions": [{"display": "Cough", "onset_year": 2020}], "audit_removed": 0, "coarsened": 0}
                       for y in cfg["years"]]
                rows.append({"pid": f"p{i}", "A": i < n_a, "a_type": ("old" if i % 2 else "misuses_only") if i < n_a else None,
                             "lambda": lam, "D": d, "known_facts": {"age": 40, "gender": "F"},
                             "k3_facts": {"Migraine": False}, "records": rec,
                             "records_full": [{"year": y, "conditions": r["conditions"]} for y, r in zip(cfg["years"], rec)]})
    (exp / "inputs" / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    kl = {"p_A": 0.2, "conditions": {}, "groups": {}, "none_of_d2": {"n": 1, "p_A": 0.2},
          "sentences": {"K1": "k1.", "K2": "k2."}}
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
        self.assertEqual(p["attack_t0"], n * 2 * 4)                        # person x lambda x K
        self.assertEqual(p["attack_t3"], n * 2 * s * 2 * (4 + 2 + 2) + n * 6 * s * 2)
        self.assertEqual(p["calib_calls"], (12 + 24) * 2 * s)              # labelled A persons only, default paraphrases

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

    def test_known_answer(self) -> None:
        """Hand-made attack reads: at lambda 1, D0, K2 half of the A persons are caught, no false alarms; K0 catches
        nobody. Then dV(K2) = pi * 0.5 and dV(K0) = 0, so H1 = pi * 0.5 exactly."""
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
            self.assertAlmostEqual(m["H1_K2_minus_K0"]["est"], 0.2 * 0.5)
            self.assertAlmostEqual(m["H0a_lambda0_D0_K2"]["est"], 0.0)
            self.assertAlmostEqual(m["H2_secondary_D0_minus_D2"]["est"], 0.0)
            m2 = summarise(exp)  # bootstrap is fixed by SEED
            self.assertEqual(m["H1_K2_minus_K0"], m2["H1_K2_minus_K0"])

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
            self.assertAlmostEqual(h["dV_by_q"]["1"], 0.2)        # every A caught, no false alarm: dV = pi
            self.assertEqual(h["labelled_persons_missing_by_q"], {"0": 0, "0.5": 0, "1": 0})
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
            self.assertAlmostEqual(m3["H1_K2_minus_K0"]["est"], 0.2 * (1 / 3))  # p1 caught of the 3 A left
            self.assertEqual(m3["H1_K2_minus_K0"]["n_persons"], 7)
            # an empty attack file is refused, not summarised as zeros
            (exp / "results" / "attack.jsonl").write_text("")
            with self.assertRaises(SystemExit):
                summarise(exp)
        finally:
            shutil.rmtree(tmp)

    def test_interval_all_missing(self) -> None:
        from e1_summarise import interval
        self.assertIsNone(interval(float("nan"), np.full(10, np.nan), 0.95)["est"])
        self.assertIsNone(interval(0.1, np.full(10, np.nan), 0.95)["p_one_sided"])
        r = interval(0.1, np.array([np.nan, 0.2, -0.1, 0.3]), 0.95)
        self.assertEqual((r["n_draws"], r["p_one_sided"]), (3, 1 / 3))


if __name__ == "__main__":
    unittest.main()
