"""Tests for the preregistration-v2 guards (C8): development seeds, the generator's seed list, the run.sh templates.

Run from the PJ root: uv run --frozen python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from permute_lambda import DEV_SEEDS, check_confirm_seed  # noqa: E402

BAD = (42, 43, 44, 9001, 9002, 20261113)


class SeedGuard(unittest.TestCase):
    def test_dev_seeds_refused_only_when_frozen(self) -> None:
        self.assertEqual(set(BAD), set(DEV_SEEDS))
        for s in BAD:
            with mock.patch.dict(os.environ, {"HTS_PREREG_FROZEN": "yes"}):
                with self.assertRaises(SystemExit):
                    check_confirm_seed(s)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HTS_PREREG_FROZEN", None)
                check_confirm_seed(s)                                       # a dry run is not stopped
            with mock.patch.dict(os.environ, {"HTS_PREREG_FROZEN": "no"}):
                check_confirm_seed(s)
        with mock.patch.dict(os.environ, {"HTS_PREREG_FROZEN": "yes"}):
            check_confirm_seed(20261201)                                    # any other seed passes

    def test_e1_run_and_e2_map_stop_on_dev_seed(self) -> None:
        from test_e1 import make_exp
        with tempfile.TemporaryDirectory() as t:
            exp = make_exp(Path(t))
            cmd = [sys.executable, str(ROOT / "scripts" / "e1_run.py"), str(exp), "--plan"]
            for seed, frozen, rc0 in ((20261113, "yes", False), (44, "yes", False), (20261113, "", True), (7, "yes", True)):
                (exp / "SEED").write_text(f"{seed}\n")
                env = {k: v for k, v in os.environ.items() if k != "HTS_PREREG_FROZEN"} | (
                    {"HTS_PREREG_FROZEN": frozen} if frozen else {})
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                self.assertEqual(r.returncode == 0, rc0, (seed, frozen, r.stderr[-300:]))
                if not rc0:
                    self.assertIn("development or test seed", r.stderr)
            # e1_build calls the same guard right after reading SEED (source check)
            src = (ROOT / "scripts" / "e1_build.py").read_text(encoding="utf-8")
            self.assertRegex(src, r'SEED"\)\.read_text\(\)\.strip\(\)\)\n\s+check_confirm_seed\(seed\)')

    def test_e2_map_main_stops_on_dev_seed(self) -> None:
        import e2_map
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "config").mkdir()
            (tmp / "config" / "condition_lists.json").write_text(json.dumps(
                {"first_year": 2023, "age_range_at_first_year": [25, 70], "d2": {}, "attributes": {}, "D2_frozen": {}}))
            exp = tmp / "exp"
            exp.mkdir()
            (exp / "config.json").write_text(json.dumps({"lambdas": [1.0], "reps": 1, "min_feature_count": 3}))
            (exp / "SEED").write_text("44\n")
            loader = mock.Mock(side_effect=AssertionError("load_tables must not be reached"))
            with mock.patch.object(e2_map, "ROOT", tmp), mock.patch.object(e2_map, "load_tables", loader):
                with mock.patch.dict(os.environ, {"HTS_PREREG_FROZEN": "yes"}):
                    with self.assertRaises(SystemExit) as cm:
                        e2_map.main(tmp, exp, True)
                self.assertIn("development or test seed", str(cm.exception))
                loader.assert_not_called()
                # not frozen: the guard lets the dev seed pass and the run goes on to load_tables
                (exp / "SEED").write_text("44\n")
                env = {k: v for k, v in os.environ.items() if k != "HTS_PREREG_FROZEN"}
                with mock.patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(AssertionError):
                        e2_map.main(tmp, exp, True)

    def test_generate_script_lists_all_dev_seeds(self) -> None:
        sh = (ROOT / "scripts" / "generate_confirm_cohort.sh").read_text(encoding="utf-8")
        line = next(l for l in sh.splitlines() if '"$s" != 42' in l)
        for s in ("42", "43", "44", "20261113"):
            self.assertIn(f'"$s" != {s}', line)
        self.assertIn('"$s" != "$TEST_SEED"', line)
        self.assertIn('"$s" != "$TEST_CLINICIAN_SEED"', line)
        self.assertIn("TEST_SEED=9001 TEST_CLINICIAN_SEED=9002", sh)


class Templates(unittest.TestCase):
    def test_templates_stop_at_the_marks(self) -> None:
        for e in ("e1", "e2"):
            p = ROOT / "experiments" / "_confirm-template" / e / "run.sh"
            src = p.read_text(encoding="utf-8")
            self.assertGreaterEqual(src.count("__FILL_AT_FREEZE__"), 8)
            self.assertIn("HTS_PREREG_FROZEN", src)
            for kw in ("SEED=", "E=", "COHORT=", "SHA_PERSONS=", "FROZEN_COMMIT=", "SHA_LISTS=",
                       f"SHA_{e.upper()}_CONFIG="):
                self.assertRegex(src, rf'(?m)^{kw}"__FILL_AT_FREEZE__"')
            self.assertIn("git rev-parse HEAD", src)
            # round 3 A5: the git checks cover exactly the frozen set (one array), and only the 6 frozen test modules run
            self.assertIn('git diff --quiet HEAD -- "${FROZEN[@]}"', src)
            self.assertIn('git status --porcelain --untracked-files=all -- "${FROZEN[@]}"', src)
            self.assertIn("git status --porcelain --untracked-files=all -- scripts tests | grep '^??'", src)  # round 3 review M1
            arr = re.search(r"(?ms)^FROZEN=\(\n(.*?)^\)$", src).group(1).split()
            self.assertFalse(any(a in ("scripts", "config", "tests") for a in arr))                  # no whole directories
            tests_ = sorted(a for a in arr if a.startswith("tests/"))
            self.assertEqual(tests_, sorted(f"tests/{m}.py" for m in ("test_e1", "test_e2_map", "test_e2_summarise",
                                                                     "test_permute_lambda", "test_prereg2", "test_e1_power")))
            for a in arr:
                self.assertTrue((ROOT / a).exists(), a)
            self.assertIn("pyproject.toml", arr)
            self.assertIn("uv.lock", arr)
            self.assertIn(f"experiments/_confirm-template/{e}/run.sh", arr)
            self.assertIn("python -m unittest tests.test_e1 tests.test_e2_map tests.test_e2_summarise tests.test_permute_lambda "
                          "tests.test_prereg2 tests.test_e1_power", src)
            self.assertNotIn("discover", src)
            self.assertNotIn("SHA_ENCOUNTERS", src)
            env = {k: v for k, v in os.environ.items() if k != "HTS_PREREG_FROZEN"} | {"HTS_PREREG_FROZEN": "yes"}
            r = subprocess.run(["bash", str(p)], capture_output=True, text=True, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not filled in", r.stderr)
        e2 = (ROOT / "experiments" / "_confirm-template" / "e2" / "run.sh").read_text(encoding="utf-8")
        self.assertIn("e2_map.py", e2)
        self.assertRegex(e2, r"e2_map\.py .*--d2-from-config")
        self.assertIn("e2_summarise.py", e2)
        self.assertRegex(e2, r"e2_summarise\.py .*--e1")

    def test_filled_template_needs_the_freeze_flag(self) -> None:
        for e in ("e1", "e2"):
            src = (ROOT / "experiments" / "_confirm-template" / e / "run.sh").read_text(encoding="utf-8")
            filled = re.sub(r'(?m)^(\w+)="__FILL_AT_FREEZE__"', r'\1="x"', src)
            with tempfile.TemporaryDirectory() as t:
                f = Path(t) / "run.sh"
                f.write_text(filled)
                env = {k: v for k, v in os.environ.items() if k != "HTS_PREREG_FROZEN"}
                r = subprocess.run(["bash", str(f)], capture_output=True, text=True, env=env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("HTS_PREREG_FROZEN=yes", r.stderr)
                f.write_text(re.sub(r'(?m)^SEED="x"', 'SEED="20261113"', filled))
                r = subprocess.run(["bash", str(f)], capture_output=True, text=True, env=env | {"HTS_PREREG_FROZEN": "yes"})
                self.assertIn("development or test seed", r.stderr)


class Prereg3Templates(unittest.TestCase):
    """C13 (判断 36): the E1 summary once after the run, the sha256 of the joined manifest and of E1's metrics.json."""

    E1 = ROOT / "experiments" / "_confirm-template" / "e1" / "run.sh"
    E2 = ROOT / "experiments" / "_confirm-template" / "e2" / "run.sh"

    def test_marks_and_checks(self) -> None:
        e1, e2 = self.E1.read_text(encoding="utf-8"), self.E2.read_text(encoding="utf-8")
        self.assertEqual(e1.count('="__FILL_AT_FREEZE__"'), 9)                     # DIGEST_AGENT, DIGEST_ATTACKER, OLLAMA_VERSION are filled before the freeze
        self.assertEqual(e2.count('="__FILL_AT_FREEZE__"'), 11)
        for src in (e1, e2):
            self.assertRegex(src, r'(?m)^SHA_MANIFEST="__FILL_AT_FREEZE__"')
            self.assertIn("SHA_MANIFEST FROZEN_COMMIT", src)                              # in the filled-in loop
            self.assertRegex(src, r"(?m)^\$SHA_MANIFEST  manifest\.json$")                 # checked with the parquet files in $COHORT
        self.assertRegex(e2, r'(?m)^SHA_E1_METRICS="__FILL_AT_FREEZE__"')
        self.assertIn("SHA_E1_METRICS COHORT", e2)
        self.assertRegex(e2, r'(?m)^echo "\$SHA_E1_METRICS  \$E1/metrics\.json" \| shasum -a 256 -c -$')
        self.assertNotIn("SHA_E1_METRICS", e1)
        self.assertLess(e2.index("SHA_E1_METRICS  $E1/metrics.json"), e2.index("e2_map.py \"$COHORT\""))   # checked before the CPU run
        # the header comments describe the new marks
        self.assertIn("SHA_MANIFEST", e1.split("set -euo")[0])
        self.assertIn("SHA_E1_METRICS", e2.split("set -euo")[0])
        self.assertIn("once", e1.split("set -euo")[0])

    def test_e2_checks_e1_metrics_before_the_map(self) -> None:
        """The 13-hour map starts only when E1's metrics.json is complete and has the H1 permutation p."""
        e2 = self.E2.read_text(encoding="utf-8")
        line = next(l for l in e2.splitlines() if l.startswith("uv run --frozen python -c ") and l.endswith('"$E1/metrics.json"'))
        self.assertLess(e2.index(line), e2.index('e2_map.py "$COHORT"'))
        code = re.search(r"python -c '(.*)' \"\$E1/metrics\.json\"$", line).group(1)
        good = {"completeness": {"complete": True}, "H1_K2_minus_K0": {"p_one_sided_permutation": 0.01}}
        bad = {"no completeness": good | {"completeness": {}}, "incomplete": good | {"completeness": {"complete": False}},
               "no permutation p": good | {"H1_K2_minus_K0": {"p_one_sided": 0.01}}, "no H1": {"completeness": {"complete": True}}}
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "metrics.json"

            def run(m: dict) -> subprocess.CompletedProcess:
                f.write_text(json.dumps(m))
                return subprocess.run([sys.executable, "-c", code, str(f)], capture_output=True, text=True)

            self.assertEqual(run(good).returncode, 0)
            for name, m in bad.items():
                r = run(m)
                self.assertNotEqual(r.returncode, 0, name)
                self.assertIn("13-hour", r.stderr, name)

    def test_cohort_must_come_from_the_frozen_commit(self) -> None:
        """Review 2026-11-14: E1 and E2 run.sh stop unless all 4 shards of the cohort carry pjCommit == FROZEN_COMMIT."""
        for path in (self.E1, self.E2):
            src = path.read_text(encoding="utf-8")
            line = next(l for l in src.splitlines() if l.startswith("uv run --frozen python -c ") and "pjCommit" in l)
            self.assertLess(src.index(line), src.index("uv run --frozen python -m unittest"))
            code = re.search(r"python -c '(.*)' \"\$COHORT/manifest\.json\" \"\$FROZEN_COMMIT\"$", line).group(1)
            with tempfile.TemporaryDirectory() as t:
                f = Path(t) / "manifest.json"

                def run(commits: list) -> int:
                    f.write_text(json.dumps({"shards": [{"shard": i + 1, "pjCommit": c} for i, c in enumerate(commits)]}))
                    return subprocess.run([sys.executable, "-c", code, str(f), "fz"], capture_output=True, text=True).returncode
                self.assertEqual(run(["fz"] * 4), 0, path)
                for bad in (["fz"] * 3 + ["other"], ["fz"] * 3, ["fz"] * 3 + [None], ["none"] * 4):
                    self.assertNotEqual(run(bad), 0, (path, bad))

    def test_e2_seed_must_differ_from_e1_and_the_cohort_seeds(self) -> None:
        """E2's SEED may not equal E1's SEED or any of the 8 seeds of the cohort manifest; checked before the map."""
        e2 = self.E2.read_text(encoding="utf-8")
        line = next(l for l in e2.splitlines() if l.startswith("uv run --frozen python -c ") and "clinicianSeed" in l)
        self.assertLess(e2.index(line), e2.index('e2_map.py "$COHORT"'))
        code = re.search(r"python -c '(.*)' \"\$SEED\" \"\$E1/SEED\" \"\$COHORT/manifest\.json\"$", line).group(1)
        shards = [{"seed": 1000 + 2 * k, "clinicianSeed": 1001 + 2 * k} for k in range(4)]
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            (r / "E1SEED").write_text("777\n")

            def run(seed: str, shs: list = shards) -> subprocess.CompletedProcess:
                (r / "manifest.json").write_text(json.dumps({"shards": shs}))
                return subprocess.run([sys.executable, "-c", code, seed, str(r / "E1SEED"), str(r / "manifest.json")],
                                      capture_output=True, text=True)
            self.assertEqual(run("20261201").returncode, 0)
            self.assertIn("E1 SEED", run("777").stderr)
            for s in (1000, 1007):                                                      # a shard seed, a clinicianSeed
                x = run(str(s))
                self.assertNotEqual(x.returncode, 0)
                self.assertIn("cohort seed", x.stderr)
            self.assertIn("8 seeds", run("20261201", shards[:3]).stderr)

    def _fill(self, src: str, **over) -> str:
        filled = re.sub(r'(?m)^(\w+)="__FILL_AT_FREEZE__"', r'\1="x"', src)
        for k, v in over.items():
            filled = re.sub(rf'(?m)^{k}="x"', f'{k}="{v}"', filled)
        return filled

    def test_manifest_hash_is_checked_with_the_cohort(self) -> None:
        """A wrong SHA_MANIFEST fails the shasum check; a right one gets past it (to the git check)."""
        import hashlib
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "config").mkdir()
            (root / "scripts").mkdir()
            for f in ("config/e1_confirm.json", "config/e2_confirm.json", "scripts/e1_run.py", "scripts/e2_map.py"):
                (root / f).write_text("{}")
            cohort = root / "cohort"
            cohort.mkdir()
            sums = {}
            for name in ("persons.parquet", "conditions.parquet", "manifest.json"):
                (cohort / name).write_text(name)
                sums[name] = hashlib.sha256(name.encode()).hexdigest()
            for e, p in (("e1", self.E1), ("e2", self.E2)):
                d = root / "experiments" / e
                d.mkdir(parents=True)
                over = dict(SEED="20261201", COHORT=str(cohort), SHA_PERSONS=sums["persons.parquet"],
                            SHA_CONDITIONS=sums["conditions.parquet"], SHA_MANIFEST=sums["manifest.json"])
                env = {k: v for k, v in os.environ.items() if k != "HTS_PREREG_FROZEN"} | {"HTS_PREREG_FROZEN": "yes"}
                (d / "run.sh").write_text(self._fill(p.read_text(encoding="utf-8"), **over))
                r = subprocess.run(["bash", str(d / "run.sh")], capture_output=True, text=True, env=env)
                self.assertNotIn("FAILED", r.stdout + r.stderr, (e, r.stdout, r.stderr))
                self.assertIn("not in a git work tree", r.stderr, (e, r.stderr))       # past the sha256 checks
                (d / "run.sh").write_text(self._fill(p.read_text(encoding="utf-8"), **(over | {"SHA_MANIFEST": "0" * 64})))
                r = subprocess.run(["bash", str(d / "run.sh")], capture_output=True, text=True, env=env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("manifest.json: FAILED", r.stdout + r.stderr, (e, r.stdout, r.stderr))
                self.assertNotIn("not in a git work tree", r.stderr)

    def _tail(self) -> str:
        src = self.E1.read_text(encoding="utf-8")
        m = re.search(r'(?ms)^if \[\[ "\$\{1:-\}" == run \]\]; then\n.*?^fi$', src)
        self.assertIsNotNone(m)
        return m.group(0)

    def test_e1_run_gets_the_frozen_digests(self) -> None:
        """Round 3 A7: the digests and the Ollama version are fixed in the template before the freeze (part of the frozen commit),
        stay in the unfilled-mark loop, and are passed to e1_run.py."""
        src = self.E1.read_text(encoding="utf-8")
        self.assertRegex(src, r'(?m)^DIGEST_AGENT="[0-9a-f]{64}"')
        self.assertRegex(src, r'(?m)^DIGEST_ATTACKER="[0-9a-f]{64}"')
        self.assertRegex(src, r'(?m)^OLLAMA_VERSION="\d+\.\d+\.\d+"')
        self.assertIn("OLLAMA_VERSION=0.40.16 sh", src)                                   # the install pins the same version
        self.assertIn('OLLAMA_VERSION="0.40.16"', src)
        for v in ("DIGEST_AGENT", "DIGEST_ATTACKER", "OLLAMA_VERSION"):
            self.assertIn(f" {v}", src.split("for v in ")[1].split("; do")[0])
        tail = self._tail()
        self.assertIn('--expect-digests "gemma3:12b=$DIGEST_AGENT,qwen3:14b=$DIGEST_ATTACKER"', tail)
        self.assertIn('--expect-ollama "$OLLAMA_VERSION"', tail)

    def test_e1_summary_only_after_a_clean_run(self) -> None:
        src = self.E1.read_text(encoding="utf-8")
        cmds = [l for l in src.splitlines() if re.match(r"\s*uv run .*e1_summarise\.py", l)]
        self.assertEqual(len(cmds), 1)                                                   # once
        self.assertNotIn('results/attack.jsonl', src.split("e1_run.py \"$E\" --plan")[1])      # no summary on a mere presence of results
        tail = self._tail()
        self.assertLess(tail.index("e1_run.py"), tail.index("e1_summarise.py"))
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            log = tmp / "log"
            e_dir = tmp / "E"
            e_dir.mkdir()
            harness = ('set -euo pipefail\ndie() { echo "run.sh: $*" >&2; exit 1; }\n'
                       'uv() { echo "$*" >> "$LOG"; if [[ "$*" == *e1_run.py* && "${FAIL:-}" == yes ]]; then return 3; fi; }\n'
                       f'E="{e_dir}"\nDIGEST_AGENT=da DIGEST_ATTACKER=dt OLLAMA_VERSION=ov\n' + tail + "\n")
            script = tmp / "h.sh"
            script.write_text(harness)

            def run(*args: str, fail: bool = False) -> subprocess.CompletedProcess:
                log.write_text("")
                env = os.environ | {"LOG": str(log)} | ({"FAIL": "yes"} if fail else {})
                return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env)
            r = run("run")                                                                # a clean run: run, then the summary
            self.assertEqual(r.returncode, 0, r.stderr)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 2)
            self.assertIn("e1_run.py", calls[0])
            self.assertIn("--stage all", calls[0])
            self.assertIn("--expect-digests gemma3:12b=da,qwen3:14b=dt --expect-ollama ov", calls[0])
            self.assertIn("e1_summarise.py", calls[1])
            r = run("run", fail=True)                                                     # e1_run.py exits non-zero: no summary
            self.assertEqual(r.returncode, 3)
            self.assertEqual(len(log.read_text().splitlines()), 1)
            self.assertNotIn("summarise", log.read_text())
            r = run()                                                                     # without `run`: nothing at all
            self.assertEqual((r.returncode, log.read_text()), (0, ""))
            (e_dir / "metrics.json").write_text("{}")                                     # an E1 summary exists: refuse a second one
            r = run("run")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("made once", r.stderr)
            self.assertEqual(log.read_text(), "")


class Shards(unittest.TestCase):
    """判断 34: the confirmatory cohort is 4 shards of 10k joined by merge_cohort_shards.py."""

    def _shard(self, root: Path, k: int, pids: list[str], seed: int, cseed: int, **over) -> Path:
        import pandas as pd
        d = root / f"derived-shard{k}"
        d.mkdir()
        pd.DataFrame({"pid": pids, "sex": ["F"] * len(pids)}).to_parquet(d / "persons.parquet", index=False)
        pd.DataFrame({"pid": pids, "code": ["1"] * len(pids)}).to_parquet(d / "conditions.parquet", index=False)
        pd.DataFrame({"pid": pids, "start": ["2023"] * len(pids)}).to_parquet(d / "encounters.parquet", index=False)
        m = {"recipe": "r", "syntheaVersion": "v4.0.0", "jarSha256": "j", "mode": "confirm", "shard": k, "seed": seed,
             "clinicianSeed": cseed, "population": 2, "state": "Massachusetts", "referenceDate": "20261025",
             "endDate": "20261025", "fhirFileCount": 2, "contentSha256": "c", "pjCommit": "abc123",
             "profileCohortSha256": "p" * 64, "javaVersion": "openjdk version 21.0.2 2024-01-16"} | over
        (d / "manifest.json").write_text(json.dumps(m))
        return d

    def _run(self, out: Path, dirs: list[Path], expect: int = 2) -> None:
        import merge_cohort_shards
        merge_cohort_shards.main([str(out), *map(str, dirs), "--expect", str(expect), "--population", "2"])

    def test_merge_joins_in_shard_order(self) -> None:
        import pandas as pd
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            b = self._shard(r, 2, ["c", "d"], 3, 4)
            a = self._shard(r, 1, ["a", "b"], 1, 2)
            self._run(r / "out", [b, a])
            p = pd.read_parquet(r / "out" / "persons.parquet")
            self.assertEqual(list(p["pid"]), ["a", "b", "c", "d"])
            self.assertNotIn("shard", p.columns)
            m = json.loads((r / "out" / "manifest.json").read_text())
            self.assertEqual(m["population"], 4)
            self.assertEqual([s["shard"] for s in m["shards"]], [1, 2])
            self.assertIn("persons.parquet", m["shards"][0]["files"])

    def test_merge_refuses(self) -> None:
        cases = {
            "pid clash": ([["a", "b"], 1, 2, {}], [["b", "c"], 3, 4, {}]),
            "seed reused": ([["a", "b"], 1, 2, {}], [["c", "d"], 2, 5, {}]),
            "test mode": ([["a", "b"], 1, 2, {"mode": "test"}], [["c", "d"], 3, 4, {}]),
            "other date": ([["a", "b"], 1, 2, {}], [["c", "d"], 3, 4, {"endDate": "20251025"}]),
            "dev seed": ([["a", "b"], 1, 2, {}], [["c", "d"], 3, 42, {}]),
            "pjCommit differs": ([["a", "b"], 1, 2, {}], [["c", "d"], 3, 4, {"pjCommit": "def456"}]),
            "pjCommit none": ([["a", "b"], 1, 2, {"pjCommit": "none"}], [["c", "d"], 3, 4, {"pjCommit": "none"}]),
            "profile_cohort.py differs": ([["a", "b"], 1, 2, {}], [["c", "d"], 3, 4, {"profileCohortSha256": "q" * 64}]),
        }
        for name, (x, y) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as t:
                r = Path(t)
                dirs = [self._shard(r, 1, *x[:3], **x[3]), self._shard(r, 2, *y[:3], **y[3])]
                with self.assertRaises(SystemExit):
                    self._run(r / "out", dirs)
                self.assertFalse((r / "out").exists())
        with tempfile.TemporaryDirectory() as t:   # a manifest without "shard" stops with a message, not a TypeError
            r = Path(t)
            d1, d2 = self._shard(r, 1, ["a", "b"], 1, 2), self._shard(r, 2, ["c", "d"], 3, 4)
            m = json.loads((d2 / "manifest.json").read_text())
            del m["shard"]
            (d2 / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(SystemExit) as cm:
                self._run(r / "out", [d1, d2])
            self.assertIn("lacks", str(cm.exception))
        for key in ("pjCommit", "profileCohortSha256"):   # a manifest of the old form (no provenance) stops, even if all lack it
            with self.subTest(key), tempfile.TemporaryDirectory() as t:
                r = Path(t)
                dirs = [self._shard(r, 1, ["a", "b"], 1, 2), self._shard(r, 2, ["c", "d"], 3, 4)]
                for d in dirs:
                    m = json.loads((d / "manifest.json").read_text())
                    del m[key]
                    (d / "manifest.json").write_text(json.dumps(m))
                with self.assertRaises(SystemExit) as cm:
                    self._run(r / "out", dirs)
                self.assertIn("lacks", str(cm.exception))
        with tempfile.TemporaryDirectory() as t:   # a missing shard
            r = Path(t)
            with self.assertRaises(SystemExit):
                self._run(r / "out", [self._shard(r, 1, ["a", "b"], 1, 2)], expect=2)

    def test_merge_javaversion_is_only_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            self._run(r / "out", [self._shard(r, 1, ["a", "b"], 1, 2), self._shard(r, 2, ["c", "d"], 3, 4, javaVersion="other")])
            self.assertEqual({s["javaVersion"] for s in json.loads((r / "out" / "manifest.json").read_text())["shards"]},
                             {"openjdk version 21.0.2 2024-01-16", "other"})

    def test_merge_frozen_flag_fixes_expect_and_population(self) -> None:
        import merge_cohort_shards
        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            dirs = [str(self._shard(r, 1, ["a", "b"], 1, 2)), str(self._shard(r, 2, ["c", "d"], 3, 4))]
            old = os.environ.get("HTS_PREREG_FROZEN")
            os.environ["HTS_PREREG_FROZEN"] = "yes"
            try:
                with self.assertRaises(SystemExit) as cm:       # --expect 2 / --population 2 are not the defaults
                    merge_cohort_shards.main([str(r / "out"), *dirs, "--expect", "2", "--population", "2"])
                self.assertIn("HTS_PREREG_FROZEN", str(cm.exception))
                self.assertFalse((r / "out").exists())
                with self.assertRaises(SystemExit) as cm:       # the defaults pass the flag check and fail on the shard count
                    merge_cohort_shards.main([str(r / "out"), *dirs])
                self.assertNotIn("HTS_PREREG_FROZEN", str(cm.exception))
            finally:
                if old is None:
                    del os.environ["HTS_PREREG_FROZEN"]
                else:
                    os.environ["HTS_PREREG_FROZEN"] = old

    def test_generate_script_records_provenance(self) -> None:
        sh = (ROOT / "scripts" / "generate_confirm_cohort.sh").read_text(encoding="utf-8")
        for k in ('"pjCommit": "$PJ_COMMIT"', '"profileCohortSha256": "$PROFILE_SHA"', '"javaVersion": "$JAVA_VERSION"'):
            self.assertIn(k, sh)
        self.assertLess(sh.index("PJ_COMMIT=none"), sh.index("java -jar"))                # fixed before Synthea runs
        self.assertIn("diff --quiet HEAD -- scripts/generate_confirm_cohort.sh scripts/profile_cohort.py pyproject.toml uv.lock", sh)
        self.assertIn("status --porcelain --untracked-files=all -- scripts", sh)

    def test_generate_script_stops_before_synthea_on_a_dirty_pj(self) -> None:
        """confirm mode: an untracked file in scripts/ of a git PJ stops the run before java is reached."""
        with tempfile.TemporaryDirectory() as t:
            pj = Path(t) / "pj"
            (pj / "scripts").mkdir(parents=True)
            sh = pj / "scripts" / "generate_confirm_cohort.sh"
            sh.write_text((ROOT / "scripts" / "generate_confirm_cohort.sh").read_text(encoding="utf-8"))
            for f in ("scripts/profile_cohort.py", "pyproject.toml", "uv.lock"):
                (pj / f).write_text("x")
            git = lambda *a: subprocess.run(["git", "-C", str(pj), "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                            capture_output=True, text=True, check=True)
            git("init", "-q")
            git("add", "-A")
            git("commit", "-q", "-m", "x")
            fake = Path(t) / "bin"
            fake.mkdir()
            for tool in ("uv", "java"):
                (fake / tool).write_text("#!/bin/sh\necho ran-$0 >&2; exit 0\n")
                (fake / tool).chmod(0o755)
            env = os.environ | {"HTS_PREREG_FROZEN": "yes", "SEED": "123", "CLINICIAN_SEED": "456", "SHARD": "1",
                                "WORK": str(Path(t) / "w"), "PATH": f"{fake}:{os.environ['PATH']}"}
            (pj / "scripts" / "extra.py").write_text("x")
            r = subprocess.run(["bash", str(sh), "confirm"], capture_output=True, text=True, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("untracked files", r.stderr)
            self.assertNotIn("ran-", r.stderr)
            (pj / "scripts" / "extra.py").unlink()
            (pj / "uv.lock").write_text("changed")
            r = subprocess.run(["bash", str(sh), "confirm"], capture_output=True, text=True, env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("differs from HEAD", r.stderr)
            self.assertNotIn("ran-", r.stderr)

    def test_generate_script_needs_a_shard_in_confirm_mode(self) -> None:
        sh = (ROOT / "scripts" / "generate_confirm_cohort.sh").read_text(encoding="utf-8")
        self.assertIn('[[ "${SHARD:-}" =~ ^[1-4]$ ]]', sh)
        self.assertIn("N=10000; DERIVED=derived-shard$SHARD", sh)
        self.assertIn('"shard": $SHARD', sh)
        env = {k: v for k, v in os.environ.items() if k not in ("SHARD",)} | {
            "HTS_PREREG_FROZEN": "yes", "SEED": "123", "CLINICIAN_SEED": "456", "WORK": "/nonexistent-dir/w"}
        r = subprocess.run(["bash", str(ROOT / "scripts" / "generate_confirm_cohort.sh"), "confirm"],
                           capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SHARD must be", r.stderr)
        with tempfile.TemporaryDirectory() as t:   # an existing shard output is refused before Synthea starts
            work = Path(t) / "w"
            (work / "derived-shard2").mkdir(parents=True)
            fake = Path(t) / "bin"
            fake.mkdir()
            for tool in ("uv", "java"):   # java must not be reached; uv only has to exist
                (fake / tool).write_text("#!/bin/sh\necho ran-$0 >&2; exit 0\n")
                (fake / tool).chmod(0o755)
            env2 = env | {"SHARD": "2", "WORK": str(work), "PATH": f"{fake}:{os.environ['PATH']}"}
            r = subprocess.run(["bash", str(ROOT / "scripts" / "generate_confirm_cohort.sh"), "confirm"],
                               capture_output=True, text=True, env=env2)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("already exists", r.stderr)
            self.assertNotIn("ran-", r.stderr)


class Notes(unittest.TestCase):
    def test_stale_notes_fixed(self) -> None:
        lists = json.loads((ROOT / "config" / "condition_lists.json").read_text(encoding="utf-8"))
        self.assertNotIn("D2 is not listed here", lists["_note"])
        self.assertIn("D2_frozen", lists["_note"])
        e1 = json.loads((ROOT / "config" / "e1_confirm.json").read_text(encoding="utf-8"))
        self.assertNotIn("must be checked before the freeze", e1["_note"])
        self.assertIn("判断 14", e1["_note"])
        self.assertEqual(e1["k0n"], {"lambda": 1, "D": "D0"})
        self.assertEqual(e1["alpha_one_sided"], 0.025)
        e2 = json.loads((ROOT / "config" / "e2_confirm.json").read_text(encoding="utf-8"))
        self.assertIs(e2["bootstrap_persons"], True)
        self.assertEqual(e2["alpha_one_sided"], 0.025)
        doc = (ROOT / "scripts" / "e2_map.py").read_text(encoding="utf-8")
        self.assertNotIn("prototype on the development cohort", doc)
        self.assertNotIn("not for confirmatory analysis", doc)

    def test_frozen_set_notes_name_v2_items(self) -> None:
        """C19 (判断 38): section numbers of the v1 draft are replaced by the v2 OSF item names; the attack-me labels are spelled out."""
        read = lambda *p: (ROOT.joinpath(*p)).read_text(encoding="utf-8")  # noqa: E731
        e1s, e1r, pw, e2s = (read("scripts", n) for n in ("e1_summarise.py", "e1_run.py", "e1_power.py", "e2_summarise.py"))
        self.assertIn("v4: OSF items", e1s.splitlines()[0])
        self.assertNotIn("v1 §", e1s.splitlines()[0])
        self.assertIn("v4: OSF items", e1r.splitlines()[2])
        self.assertNotIn("(v1)", e1r.splitlines()[2])
        self.assertIn("v4: OSF items", pw.splitlines()[0])
        self.assertNotIn("v1 §", pw.splitlines()[0])
        for src in (e1s, e1r, pw, e2s):
            self.assertNotIn("§", src)
            self.assertNotIn("prereg §", src)
        self.assertIn("(attack-me round 1 の A9)", e1s)
        self.assertIn("(attack-me round 1 の A5)", e1s)
        self.assertNotIn("(判断 A9)", e1s)
        self.assertNotIn("(判断 A5)", e1s)
        self.assertIn("Inference criteria", e2s)
        self.assertIn("t-interval mode only", e2s)                                       # the saturated-cell rule
        self.assertIn("(b + 1) / (R + 1)", e2s.split("A zero SD")[1].split("The script stops")[0])
        e2 = json.loads(read("config", "e2_confirm.json"))
        self.assertNotIn("§", e2["_note"])
        for item in ("Indices", "Statistical models", "Inference criteria", "Missing data"):
            self.assertIn(item, e2["_note"])
        for key in ("secondary_H2", "H2_residual_leak", "e2_lambda1_original.csv", "H1_scope", "completeness.complete"):
            self.assertIn(key, e2["_note"])
        e1 = json.loads(read("config", "e1_confirm.json"))
        for key in ("fill_adverse", "dV_opt", "dAUC", "completeness.complete"):
            self.assertIn(key, e1["_note"])


if __name__ == "__main__":
    unittest.main()
