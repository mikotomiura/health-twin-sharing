"""Confirmatory E1, step 2 (Colab with Ollama; the logic runs locally with --backend fake).

Preregistration: docs/preregistration-draft.md (v4: OSF items Data collection procedures, Starting and stopping rules, Missing data); design .steering/20261112-e1-prereg-impl/design.md.
Reads <exp>/config.json, SEED, inputs/{records,knowledge}.jsonl|json (scripts/e1_build.py). Each stage appends to
<exp>/results/<stage>.jsonl and skips keys already written, so a re-run resumes after a Colab interruption.

  twin     cell = person x lambda x D x paraphrase x twin seed; paraphrases 0..7 at the base point (lambda, D), else
           0..n_paraphrases_default-1. Three time points with the conversation history (as the pilot).
  attack   plan (判断 11): per lambda, D0 x K0..K3 and D1, D2 x K2, K3, plus K0n (config "k0n": lambda 1, D0, default
           paraphrases only; 判断 20); base-point paraphrases beyond the default only with the base-point K list. t0 = person x lambda x K (no release; does not depend on D), t3 = cell x K.
           qwen3 thinking, integer percentage (pilot stats_think), template 0, attacker seed fixed.
  calib    H0b: at (lambda, D, K) of config "calibration", for a share q of the A persons (chosen with SEED) the label
           sentence is appended to the third release and the attacker reads again. Unlabelled reads are identical to
           the attack reads (same input, same seed) and are taken from attack.jsonl by e1_summarise.py.
  utility  the judge grades releases against the full record (pilot UTIL_USER_FULL), cells in config "utility_plan",
           once the whole cell exists; a cell with any failed time point is missing at every time point. The five items
           must be JSON booleans. Truncated answers (done_reason "length") of the attacker and the judge are failures too.
Failures (no probability; empty or truncated answer, done_reason "length"): up to max_retries more calls with a shifted
seed (also when prompt_count + eval_count >= options.num_ctx: reason "ctx_overflow"; every row keeps the reason of each attempt in
"attempt_reasons"); a server that does not answer is not an attempt: same seed after HTTP_WAITS, then the stage stops without
writing the row (resume); then the row is kept as missing (OSF item Missing data): p = null for the attacker; for the twin "failed": true, and every attack /
calib / utility read of that cell is written as missing (reason "twin_missing") without a model call.
Safety for resuming on Colab: a truncated last JSONL line is dropped before appending; results/fingerprint.json pins the
backend, config, SEED, inputs and the code (a re-run with anything different, or results without a fingerprint, is
refused); results/.lock (O_EXCL) stops a second process on the same results/; results/versions.jsonl appends the
model digests at each run and a changed digest is refused. Reads skipped because the twin stage is unfinished are
counted and make the exit code 3.

Usage: python scripts/e1_run.py <exp-dir> --stage twin|attack|calib|utility|all [--backend ollama|fake] [--workers 4]
       python scripts/e1_run.py <exp-dir> --plan      # print the job counts per stage (no model calls)
       Optional: --expect-digests gemma3:12b=<digest>,qwen3:14b=<digest> --expect-ollama <version> (round 3 A7): refuse before
       any model call, on every run, when the live digests / Ollama version differ (run.sh passes the values fixed at the freeze).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import http.client
import itertools
import json
import os
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from e1_build import known_text  # noqa: E402
from permute_lambda import check_confirm_seed  # noqa: E402
from llm_dry_run import Ollama, render_record, twin_messages  # noqa: E402  (Ollama: base class of OllamaDR)
from llm_dry_run_attacker import ATTACK_USERS_PCT, PCT_SCHEMA, UTIL_USER_FULL, FakePct, parse_pct  # noqa: E402
from llm_dry_run_extra import ATTACK_SYSTEMS, UTIL_ITEMS, UTIL_SCHEMA, UTIL_SYSTEM, load_jsonl, run_jobs, timing  # noqa: E402

SEED_SHIFT = 1_000_003  # seed offset per retry attempt (a retry with the same seed would repeat the same output)
SEED_MOD = 2**31


class Sink:
    """Append-only JSONL with resume. A last line cut by an interruption (no newline or not valid JSON) is dropped."""

    def __init__(self, path: Path):
        self.path, self.lock, self.done, self.dropped = path, threading.Lock(), set(), 0
        if not path.exists():
            return
        raw = path.read_bytes()
        lines = raw.split(b"\n")
        good, keep = [], 0
        for i, line in enumerate(lines):
            complete = i < len(lines) - 1  # followed by a newline
            if not line.strip():
                keep += len(line) + 1 if complete else 0
                continue
            try:
                row = json.loads(line) if complete else None
            except json.JSONDecodeError:
                row = None
            if row is None:
                if i < len(lines) - 2:
                    raise SystemExit(f"{path}: broken line {i + 1} in the middle of the file; fix by hand")
                self.dropped += 1
                break
            good.append(row["key"])
            keep += len(line) + 1
        if keep != len(raw):
            path.write_bytes(raw[:keep])
        self.done = set(good)

    def write(self, row: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class OllamaDR(Ollama):
    """Ollama.chat plus done_reason (to detect answers cut at num_predict). The pilot class is left unchanged."""

    def chat(self, model, messages, seed, options, fmt=None):
        options = dict(options)
        think = options.pop("think", None)
        body = {"model": model, "messages": messages, "stream": False, "options": {**options, "seed": seed}}
        if think is not None:
            body["think"] = think
        if fmt is not None:
            body["format"] = fmt
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        return {"text": out["message"]["content"], "done_reason": out.get("done_reason"), "wall_s": time.time() - t0,
                "eval_count": out.get("eval_count"), "eval_s": (out.get("eval_duration") or 0) / 1e9,
                "prompt_count": out.get("prompt_eval_count"),
                "prompt_s": (out.get("prompt_eval_duration") or 0) / 1e9}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Exp:
    def __init__(self, exp: Path):
        self.exp = exp
        self.cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
        self.seed = int((exp / "SEED").read_text().strip())
        check_confirm_seed(self.seed)
        self.kn = json.loads((exp / "inputs" / "knowledge.json").read_text(encoding="utf-8"))
        self.rows = {(r["pid"], r["lambda"], r["D"]): r for r in load_jsonl(exp / "inputs" / "records.jsonl")}
        self.res = exp / "results"
        self.res.mkdir(exist_ok=True)

    # ------------------------------------------------------------ plan
    def n_para(self, lam, d) -> int:
        b = self.cfg["base_point"]
        return b["n_paraphrases"] if (lam == b["lambda"] and d == b["D"]) else self.cfg["n_paraphrases_default"]

    def cells(self) -> list[tuple]:
        """(pid, lambda, D, paraphrase, seed_idx) for every twin cell."""
        out = []
        for (pid, lam, d) in sorted(self.rows, key=lambda x: (x[1], x[2], x[0])):
            for k, s in itertools.product(range(self.n_para(lam, d)), range(self.cfg["n_seeds"])):
                out.append((pid, lam, d, k, s))
        return out

    def ks_for(self, lam, d, k) -> list[str]:
        ks = list(self.cfg["attack_plan"][d])
        if k >= self.cfg["n_paraphrases_default"]:  # extra base-point paraphrases (H3)
            return [x for x in ks if x in self.cfg["base_point"]["K"]]
        k0n = self.cfg.get("k0n")  # K0n control (判断 20): one (lambda, D) only, default paraphrases
        if k0n and (lam, d) == (k0n["lambda"], k0n["D"]):
            ks.append("K0n")
        return ks

    def attack_jobs(self) -> list[tuple]:
        """('t0', K, pid, lambda) and ('t3', K, cell)."""
        jobs = set()
        for cell in self.cells():
            pid, lam, d, k, _ = cell
            for kk in self.ks_for(lam, d, k):
                jobs.add(("t0", kk, pid, lam))
                jobs.add(("t3", kk, cell))
        return sorted(jobs, key=str)

    def calib_people(self, q: float) -> set:
        """A persons that get the label sentence at share q (same set for every cell; nested is not required)."""
        a = sorted({pid for (pid, _, _), r in self.rows.items() if r["A"]})
        n = int(round(q * len(a)))
        rng = np.random.default_rng([self.seed, zlib.crc32(b"e1-calib"), int(round(q * 1000))])
        return set(rng.choice(a, size=n, replace=False)) if n else set()

    def calib_jobs(self) -> list[tuple]:
        c = self.cfg["calibration"]
        out = []
        for q in c["q"]:
            lab = self.calib_people(q)
            for cell in self.cells():
                pid, lam, d, k, _ = cell
                if (lam, d) == (c["lambda"], c["D"]) and k < self.cfg["n_paraphrases_default"] and pid in lab:
                    out.append((q, cell))
        return out

    def utility_cells(self) -> list[tuple]:
        u = self.cfg.get("utility_plan", {"lambda": 1})
        return [c for c in self.cells() if c[1] == u["lambda"]]

    # ------------------------------------------------------------ keys and texts
    @staticmethod
    def ckey(cell) -> str:
        pid, lam, d, k, s = cell
        return f"{pid}|L{lam}|{d}|P{k}|s{s}"

    def attack_key(self, j: tuple) -> str:
        return f"{j[1]}|{j[2]}|L{j[3]}|t0" if j[0] == "t0" else f"{j[1]}|{self.ckey(j[2])}|t3"

    def calib_key(self, q, cell) -> str:
        return f"q{q}|{self.ckey(cell)}|t3"

    def known(self, level: str, pid: str, lam) -> str:
        r = self.rows[(pid, lam, "D0")]
        return known_text(level, r, self.kn["lambdas"][str(lam)], self.cfg["recipient"]["known_base"])

    def attack_messages(self, known: str, releases: list[str]) -> list[dict]:
        t = self.cfg["attacker_template"]
        rel = "\n\n".join(f"[Release {i + 1}]\n{r}" for i, r in enumerate(releases)) or "(no releases yet)"
        return [{"role": "system", "content": ATTACK_SYSTEMS[t]},
                {"role": "user", "content": ATTACK_USERS_PCT[t].format(
                    name=self.cfg["recipient"]["name"], known=known, rel=rel, a=self.cfg["attribute_question"])}]


def attempt_reason(out: dict, num_ctx: int | None) -> str | None:
    """Why an attempt counts as a context overflow ("ctx_overflow") or else its done_reason. prompt_count + eval_count >= num_ctx
    means the prompt plus the generation filled the context window (Ollama then truncates silently); not checked without num_ctx."""
    pc, ec = out.get("prompt_count"), out.get("eval_count")
    if num_ctx is not None and isinstance(pc, int) and isinstance(ec, int) and pc + ec >= num_ctx:
        return "ctx_overflow"
    return out.get("done_reason")


# A server that does not answer (connection refused / reset, timeout, HTTP error) is NOT a failed attempt: the same call is
# repeated with the SAME seed after these waits, and if the server still does not answer, the stage stops without writing the
# row (it is run on resume), so no read is missing and no seed depends on an infrastructure accident (review 2026-11-14 HIGH).
HTTP_WAITS = (30, 120, 300)
_sleep = time.sleep
SERVER_DOWN = threading.Event()  # set once the server is judged down: the other workers stop at once instead of waiting too


class ServerDown(RuntimeError):
    """The model server did not answer after HTTP_WAITS; the stage stops and nothing is written for this read."""


def chat_answered(be, model, msgs, seed, opts, fmt) -> dict:
    """be.chat, repeated with the same seed while the server does not answer (OSError covers URLError, timeouts, refused /
    reset connections; HTTPException covers RemoteDisconnected / IncompleteRead); raises ServerDown after HTTP_WAITS."""
    for k in range(len(HTTP_WAITS) + 1):
        if SERVER_DOWN.is_set():
            raise ServerDown("the model server was judged down by another worker; resume with the same command")
        try:
            return be.chat(model, msgs, seed, opts, fmt=fmt)
        except (OSError, http.client.HTTPException) as e:
            if k == len(HTTP_WAITS):
                SERVER_DOWN.set()
                raise ServerDown(f"the model server did not answer ({type(e).__name__}: {e}) after waits {HTTP_WAITS} s; "
                                 f"nothing is written for this read; resume with the same command") from e
            print(f"[e1_run] server did not answer ({type(e).__name__}); same seed again in {HTTP_WAITS[k]} s", flush=True)
            _sleep(HTTP_WAITS[k])
    raise AssertionError("unreachable")


def with_retries(be, model, msgs, seed, opts, fmt, ok, max_retries: int) -> tuple[dict, int, bool, list]:
    """ok(out) -> bool. Returns (last output, attempt index, success, reason of every attempt). An attempt fails if ok() is false
    or if prompt_count + eval_count >= options.num_ctx (reason "ctx_overflow"; skipped when options has no num_ctx); the next
    attempt shifts the seed. A server that does not answer is not an attempt (chat_answered: same seed, then ServerDown)."""
    num_ctx, reasons = opts.get("num_ctx"), []
    for attempt in range(max_retries + 1):
        out = chat_answered(be, model, msgs, (seed + attempt * SEED_SHIFT) % SEED_MOD, opts, fmt)
        reason = attempt_reason(out, num_ctx)
        reasons.append(reason)
        if reason != "ctx_overflow" and ok(out):
            return out, attempt, True, reasons
    return out, attempt, False, reasons


def not_cut(out: dict) -> bool:
    return out.get("done_reason") != "length"


def twin_ok(out: dict) -> bool:
    return bool(out["text"] and out["text"].strip()) and not_cut(out)


def attack_ok(out: dict) -> bool:
    return parse_pct(out["text"]) is not None and not_cut(out)


def judge_items(text: str) -> dict | None:
    """The judge's five items if every one is a JSON boolean, else None."""
    try:
        g = json.loads(text)
    except Exception:
        return None
    if not isinstance(g, dict) or not all(isinstance(g.get(k), bool) for k in UTIL_ITEMS):
        return None
    return {k: g[k] for k in UTIL_ITEMS}


class Skips:
    def __init__(self):
        self.n, self.lock = 0, threading.Lock()

    def add(self):
        with self.lock:
            self.n += 1


def releases_of(x: Exp) -> dict[str, dict[int, dict]]:
    """cell -> t -> twin row (text, failed)."""
    rel: dict[str, dict[int, dict]] = {}
    for r in load_jsonl(x.res / "twin.jsonl"):
        rel.setdefault(r["cell"], {}).setdefault(r["t"], r)  # a duplicated key: the row written first wins
    return rel


def cell_texts(x: Exp, rel: dict, cell) -> tuple[list[str] | None, bool]:
    """(texts of all time points or None if the twin stage has not finished the cell, any time point failed)."""
    r = rel.get(x.ckey(cell), {})
    if len(r) < len(x.cfg["years"]):
        return None, False
    return [r[t]["text"] for t in sorted(r)], any(r[t].get("failed") for t in r)


def stage_twin(x: Exp, be, workers: int) -> None:
    sink = Sink(x.res / "twin.jsonl")
    cfg, rcp = x.cfg, x.cfg["recipient"]
    jobs = [c for c in x.cells() if f"{x.ckey(c)}|t{len(cfg['years'])}" not in sink.done]
    done_rel = releases_of(x)  # read once, before any thread appends (rows written in this run are not needed)
    n_failed = Skips()

    def fn(cell):
        pid, lam, d, k, s = cell
        row, ck = x.rows[(pid, lam, d)], x.ckey(cell)
        history, failed_before = [], False
        for t, rec in enumerate(row["records"], start=1):
            msgs = twin_messages(render_record(rec), rcp, cfg["paraphrases"][k], "audit", history)
            key = f"{ck}|t{t}"
            if key in sink.done:  # resume inside a cell: rebuild the history from the file
                prev = done_rel[ck][t]
                text, failed_before = prev["text"], failed_before or bool(prev.get("failed"))
            else:
                meta = {"key": key, "cell": ck, "pid": pid, "A": row["A"], "lambda": lam, "D": d, "paraphrase": k,
                        "seed_idx": s, "t": t, "audit_removed": rec["audit_removed"], "coarsened": rec["coarsened"]}
                if failed_before:  # the cell is missing anyway (OSF item Missing data): do not spend calls on the later time points
                    sink.write(meta | {"text": "", "failed": True, "reason": "earlier_time_point_failed", "attempts": 0})
                    n_failed.add()
                    continue
                out, att, good, rsn = with_retries(be, cfg["twin_model"], msgs, x.seed + 1000 * s, cfg["twin_options"],
                                              None, twin_ok, cfg["max_retries"])
                text, failed_before = out["text"], not good
                if not good:
                    n_failed.add()
                sink.write(meta | {"text": text, "failed": not good, "done_reason": out.get("done_reason"),
                                   "attempts": att + 1, "attempt_reasons": rsn, **timing(out)})
            history += [msgs[-1], {"role": "assistant", "content": text}]

    run_jobs(jobs, fn, workers, "twin")
    print(f"[twin] failed time points written this run: {n_failed.n}; cut lines dropped on resume: {sink.dropped}",
          flush=True)


def stage_attack(x: Exp, be, workers: int, skips: Skips) -> None:
    sink = Sink(x.res / "attack.jsonl")
    rel = releases_of(x)
    cfg = x.cfg
    jobs = []
    for j in x.attack_jobs():
        key = x.attack_key(j)
        if key not in sink.done:
            jobs.append((key, j))

    def fn(job):
        key, j = job
        if j[0] == "t0":
            _, kk, pid, lam = j
            texts, meta = [], {"D": None, "paraphrase": None, "seed_idx": None}
        else:
            _, kk, cell = j
            pid, lam, d, k, s = cell
            texts, failed = cell_texts(x, rel, cell)
            if texts is None:
                skips.add()  # twin stage not finished for this cell; a later run picks it up
                return
            meta = {"D": d, "paraphrase": k, "seed_idx": s}
        base = {"key": key, "K": kk, "pid": pid, "A": x.rows[(pid, lam, "D0")]["A"], "lambda": lam,
                "t": 0 if j[0] == "t0" else 3, **meta}
        if j[0] == "t3" and failed:
            sink.write(base | {"p": None, "reason": "twin_missing", "attempts": 0})
            return
        msgs = x.attack_messages(x.known(kk, pid, lam), texts)
        out, att, good, rsn = with_retries(be, cfg["attacker_model"], msgs, x.seed + 7, cfg["attacker_options"], PCT_SCHEMA,
                                      attack_ok, cfg["max_retries"])
        sink.write(base | {"p": parse_pct(out["text"]) if good else None, "done_reason": out.get("done_reason"),
                           "attempts": att + 1, "attempt_reasons": rsn, "raw": out["text"], **timing(out)})

    run_jobs(jobs, fn, workers, "attack")
    print(f"[attack] cut lines dropped on resume: {sink.dropped}", flush=True)


def stage_calib(x: Exp, be, workers: int, skips: Skips) -> None:
    sink = Sink(x.res / "calib.jsonl")
    rel = releases_of(x)
    c, cfg = x.cfg["calibration"], x.cfg
    jobs = [(q, cell) for q, cell in x.calib_jobs() if x.calib_key(q, cell) not in sink.done]

    def fn(job):
        q, cell = job
        pid, lam, d, k, s = cell
        texts, failed = cell_texts(x, rel, cell)
        if texts is None:
            skips.add()
            return
        base = {"key": x.calib_key(q, cell), "q": q, "K": c["K"], "pid": pid, "A": True, "lambda": lam,
                "D": d, "paraphrase": k, "seed_idx": s, "t": 3, "labelled": True}
        if failed:
            sink.write(base | {"p": None, "reason": "twin_missing", "attempts": 0})
            return
        texts[-1] = texts[-1].rstrip() + "\n\n" + c["sentence"]
        msgs = x.attack_messages(x.known(c["K"], pid, lam), texts)
        out, att, good, rsn = with_retries(be, cfg["attacker_model"], msgs, x.seed + 7, cfg["attacker_options"], PCT_SCHEMA,
                                      attack_ok, cfg["max_retries"])
        sink.write(base | {"p": parse_pct(out["text"]) if good else None, "done_reason": out.get("done_reason"),
                           "attempts": att + 1, "attempt_reasons": rsn, "raw": out["text"], **timing(out)})

    run_jobs(jobs, fn, workers, "calib")
    print(f"[calib] cut lines dropped on resume: {sink.dropped}", flush=True)


def stage_utility(x: Exp, be, workers: int, skips: Skips) -> None:
    sink = Sink(x.res / "utility.jsonl")
    rel = releases_of(x)
    cfg, rcp = x.cfg, x.cfg["recipient"]
    jobs = []
    for cell in x.utility_cells():
        todo = [t for t in range(1, len(cfg["years"]) + 1) if f"{x.ckey(cell)}|t{t}" not in sink.done]
        if not todo:
            continue
        texts, failed = cell_texts(x, rel, cell)
        if texts is None:  # judged only once the whole cell exists, so a later failure still makes the cell missing
            skips.add()
            continue
        jobs += [(cell, t, failed) for t in todo]

    def fn(job):
        cell, t, failed = job
        pid, lam, d, k, s = cell
        row = x.rows[(pid, lam, d)]
        tw = rel[x.ckey(cell)][t]
        base = {"key": f"{x.ckey(cell)}|t{t}", "pid": pid, "A": row["A"], "lambda": lam, "D": d, "paraphrase": k,
                "seed_idx": s, "t": t}
        if failed:  # any time point of the cell failed -> every time point is missing (OSF item Missing data, cell level)
            sink.write(base | {"items": None, "reason": "twin_missing", "attempts": 0})
            return
        full = dict(row["records"][t - 1]) | {"conditions": row["records_full"][t - 1]["conditions"]}
        msgs = [{"role": "system", "content": UTIL_SYSTEM},
                {"role": "user", "content": UTIL_USER_FULL.format(name=rcp["name"], question=rcp["question"],
                                                                  record=render_record(full), answer=tw["text"])}]

        out, att, good, rsn = with_retries(be, cfg["judge_model"], msgs, x.seed + 11, cfg["judge_options"], UTIL_SCHEMA,
                                      lambda o: judge_items(o["text"]) is not None and not_cut(o), cfg["max_retries"])
        sink.write(base | {"items": judge_items(out["text"]) if good else None, "done_reason": out.get("done_reason"),
                           "attempts": att + 1, "attempt_reasons": rsn, "raw": out["text"], **timing(out)})

    run_jobs(jobs, fn, workers, "utility")
    print(f"[utility] cut lines dropped on resume: {sink.dropped}", flush=True)


CODE = ["e1_run.py", "e1_build.py", "llm_dry_run.py", "llm_dry_run_attacker.py", "llm_dry_run_extra.py"]


def check_provenance(x: Exp, backend: str, be, expect_digests: dict | None = None, expect_ollama: str | None = None) -> None:
    """Refuse to resume with another backend, config, SEED, inputs or code (prompt constants live in the scripts), or
    with changed / unknown model digests. Results without a fingerprint are refused too.
    expect_digests / expect_ollama (round 3 A7; fixed in run.sh at the freeze): the live digest of every model in use and the live
    Ollama version must equal them, on the first run and on every resume, before any model call; otherwise SystemExit."""
    here = Path(__file__).resolve().parent
    fp = {"backend": backend, "config.json": sha256(x.exp / "config.json"), "SEED": x.seed,
          "records.jsonl": sha256(x.exp / "inputs" / "records.jsonl"),
          "knowledge.json": sha256(x.exp / "inputs" / "knowledge.json"),
          "code": {n: sha256(here / n) for n in CODE}}
    f = x.res / "fingerprint.json"
    if not f.exists() and any(p.name != "versions.jsonl" for p in x.res.glob("*.jsonl")):
        raise SystemExit("results/ has JSONL files but no fingerprint.json; refusing to append (use a new exp dir)")
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old != fp:
            diff = sorted(k for k in fp if old.get(k) != fp[k])
            raise SystemExit(f"results/ was made with a different {diff}; refusing to resume (use a new exp dir)")
    else:
        f.write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8", newline="\n")
    v = be.versions(sorted({x.cfg["twin_model"], x.cfg["attacker_model"], x.cfg["judge_model"]}))
    vf = x.res / "versions.jsonl"
    prev = load_jsonl(vf)
    if any(d is None for d in v["models"].values()):
        raise SystemExit(f"a model is not available (digest None): {v['models']}")
    if expect_digests is not None:
        bad = {m: (expect_digests.get(m), v["models"].get(m)) for m in set(v["models"]) | set(expect_digests)
               if expect_digests.get(m) != v["models"].get(m)}  # an expected model not in use stops too
        if bad:
            raise SystemExit(f"model digest differs from the expected one fixed at the freeze (model: (expected, live)): {bad}")
    if expect_ollama is not None and v.get("ollama") != expect_ollama:
        raise SystemExit(f"Ollama version {v.get('ollama')} differs from the expected {expect_ollama} fixed at the freeze")
    if prev and prev[0]["versions"].get("ollama") != v.get("ollama"):
        print(f"[e1_run] WARNING: Ollama version changed {prev[0]['versions'].get('ollama')} -> {v.get('ollama')}",
              flush=True)
    if prev and prev[0]["versions"]["models"] != v["models"]:
        raise SystemExit(f"model digests changed since the first run: {prev[0]['versions']['models']} -> {v['models']}")
    with vf.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(), "versions": v}) + "\n")


LOCK = ".lock"


@contextlib.contextmanager
def run_lock(res: Path):
    """results/.lock (O_CREAT|O_EXCL) for the length of a run: a second process on the same results/ stops. Removed on exit
    (normal or exception). A crash that kills the process hard leaves it behind: check that nothing runs, then delete it."""
    path = res / LOCK
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{path} exists: another process is running on this results/ (or a previous run died; "
                         f"check that nothing runs, then delete the file by hand)")
    try:
        os.write(fd, f"pid {os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def plan(x: Exp) -> dict:
    cells = x.cells()
    aj = x.attack_jobs()
    return {"twin_cells": len(cells), "twin_calls": len(cells) * len(x.cfg["years"]),
            "attack_t0": sum(j[0] == "t0" for j in aj), "attack_t3": sum(j[0] == "t3" for j in aj),
            "calib_calls": len(x.calib_jobs()), "utility_calls": len(x.utility_cells()) * len(x.cfg["years"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--stage", choices=["twin", "attack", "calib", "utility", "all"])
    ap.add_argument("--backend", default="ollama", choices=["ollama", "fake"])
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--expect-digests", default=None, metavar="MODEL=DIGEST,...",
                    help="live digest of each model must equal these (checked before any model call, every run); the digest string "
                         "is the one in results/versions.jsonl; run.sh passes the values fixed at the freeze")
    ap.add_argument("--expect-ollama", default=None, help="live Ollama version must equal this (checked like --expect-digests)")
    a = ap.parse_args()
    x = Exp(a.exp)
    if a.plan or not a.stage:
        print(json.dumps(plan(x), indent=1))
        return
    be = OllamaDR(a.host) if a.backend == "ollama" else FakePct()
    exd = None
    if a.expect_digests:
        exd = {}
        for part in a.expect_digests.split(","):
            m, sep, d = part.partition("=")  # the model name holds ":" but no "="; the digest may hold ":" (sha256:...)
            if not sep or not m or not d:
                raise SystemExit(f"--expect-digests: bad item {part!r} (want MODEL=DIGEST)")
            exd[m] = d
    stages = ["twin", "attack", "calib", "utility"] if a.stage == "all" else [a.stage]
    skips = Skips()
    with run_lock(x.res):
        check_provenance(x, a.backend, be, exd, a.expect_ollama)
        for st in stages:
            if st == "twin":
                stage_twin(x, be, a.workers)
            else:
                {"attack": stage_attack, "calib": stage_calib, "utility": stage_utility}[st](x, be, a.workers, skips)
    if skips.n:
        print(f"[e1_run] {skips.n} reads skipped because the twin stage has not finished their cells", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
