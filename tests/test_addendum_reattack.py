"""scripts/addendum_reattack.py: the attacker input equals e1_run's attack stage; row shape; request body.

Uses the development releases of experiments/20261114-e1-numctx-check (copied to a temp dir; the source is not written).
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import addendum_reattack as ar  # noqa: E402
import e1_run  # noqa: E402
from llm_dry_run_attacker import FakePct  # noqa: E402

SRC = ROOT / "experiments" / "20261114-e1-numctx-check"


class Rec(FakePct):
    """FakePct that records (messages, seed, options, format) per call."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def chat(self, model, messages, seed, options, fmt=None):
        self.calls.append({"model": model, "msgs": json.dumps(messages, ensure_ascii=False), "seed": seed,
                           "opts": json.dumps(options, sort_keys=True), "fmt": fmt is not None})
        out = super().chat(model, messages, seed, options, fmt)
        return out | {"done_reason": "stop"}


@unittest.skipUnless((SRC / "results" / "twin.jsonl").exists(), "development releases not present")
class Reattack(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "src"
        (self.src / "results").mkdir(parents=True)
        for f in ("config.json", "SEED"):
            shutil.copy(SRC / f, self.src / f)
        shutil.copytree(SRC / "inputs", self.src / "inputs")
        shutil.copy(SRC / "results" / "twin.jsonl", self.src / "results" / "twin.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_input_as_e1_run_attack_stage(self):
        x = e1_run.Exp(self.src)
        ref = Rec()
        with mock.patch("sys.stdout", io.StringIO()):
            e1_run.stage_attack(x, ref, 1, e1_run.Skips())
        ks = sorted({k for k in x.cfg["attack_plan"]["D0"]} | {"K0n"})
        new = Rec()
        rel = e1_run.releases_of(x)
        opts = dict(x.cfg["attacker_options"])
        rows = [ar.read_one(x, new, rel, x.cfg["attacker_model"], opts, j) for j in ar.jobs_for(x, ks, ar.subset(x, None))]
        key = lambda c: (c["msgs"], c["seed"], c["opts"], c["fmt"], c["model"])
        self.assertEqual(sorted(map(key, ref.calls)), sorted(map(key, new.calls)))
        ref_rows = {r["key"]: r for r in e1_run.load_jsonl(self.src / "results" / "attack.jsonl")}
        self.assertEqual(set(ref_rows), {r["key"] for r in rows})
        for r in rows:
            e = ref_rows[r["key"]]
            extra = set(r) - set(e)
            self.assertEqual(extra, {"model", "think", "thinking_chars"})
            self.assertEqual({k: r[k] for k in e}, e)

    def test_think_override_and_k1(self):
        x = e1_run.Exp(self.src)
        rel = e1_run.releases_of(x)
        opts = dict(x.cfg["attacker_options"]) | {"think": "medium"}
        rec = Rec()
        jobs = ar.jobs_for(x, ["K1"], ar.subset(x, 1))
        self.assertEqual(sum(j[0] == "t0" for j in jobs), 2)
        self.assertEqual(sum(j[0] == "t3" for j in jobs), 2 * x.cfg["n_paraphrases_default"] * x.cfg["n_seeds"])
        r = ar.read_one(x, rec, rel, "m:1", opts, jobs[0])
        self.assertEqual(r["model"], "m:1")
        self.assertEqual(json.loads(rec.calls[0]["opts"])["think"], "medium")
        self.assertIn(x.kn["lambdas"]["1"]["sentences"]["K1"], rec.calls[0]["msgs"])

    def test_cli_fake_writes_outside_src(self):
        out = self.tmp / "out"
        with mock.patch("sys.stdout", io.StringIO()):
            ar.main([str(self.src), str(out), "--model", "m:1", "--persons", "1", "--backend", "fake", "--workers", "2"])
            ar.main([str(self.src), str(out), "--model", "m:1", "--persons", "1", "--backend", "fake"])  # resume: nothing new
        rows = e1_run.load_jsonl(out / "attack_m_1.jsonl")
        self.assertEqual(len(rows), len({r["key"] for r in rows}))
        self.assertEqual(len(rows), 4 * (2 + 2 * 2 * 3))
        with self.assertRaises(SystemExit):
            ar.main([str(self.src), str(self.src / "x"), "--model", "m:1", "--backend", "fake"])


class Body(unittest.TestCase):
    def test_request_body_equals_ollamadr(self):
        msgs = [{"role": "user", "content": "hi"}]
        opts = {"temperature": 0.6, "num_ctx": 8192, "think": True}
        sent = []

        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            sent.append(json.loads(req.data))
            return R(json.dumps({"message": {"content": "{}", "thinking": "abc"}, "done_reason": "stop"}).encode())

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            e1_run.OllamaDR("http://h").chat("m", msgs, 5, opts, fmt={"type": "object"})
            out = ar.OllamaThink("http://h").chat("m", msgs, 5, opts, fmt={"type": "object"})
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(out["thinking_chars"], 3)
        self.assertEqual(ar.parse_think("TRUE"), True)
        self.assertEqual(ar.parse_think("high"), "high")


if __name__ == "__main__":
    unittest.main()
