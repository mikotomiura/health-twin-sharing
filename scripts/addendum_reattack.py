"""Addendum A (generalisation, attacker swap): re-read existing E1 releases with another attacker model.

Design: .steering/20261117-generalization-addendum/design.md (A), handoff .steering/_handoff/20261117-1300-addendum-a-b-parallel.md.
Part of the frozen set of addendum 1 (docs/preregistration-addendum-1-en.md). It never writes into the source experiment: e1_run.py pins results/fingerprint.json, so another
attacker cannot be appended to <src>/results/attack.jsonl (and must not be tried).

Reads <src>/config.json, SEED, inputs/, results/twin.jsonl through e1_run.Exp. The attacker input is built with e1_run's own
functions (Exp.known -> e1_build.known_text, Exp.attack_messages, PCT_SCHEMA, the seed SEED + 7, with_retries / attack_ok /
parse_pct / timing), so a read here gets exactly the messages, seed and retry rule of e1_run's attack stage; only the model
name and the "think" option differ. Rows have the shape of attack.jsonl plus "model", "think" and "thinking_chars".

Jobs: lambda 1, D0, default paraphrases (the H1 point), K in --ks; t0 = person x K (no release), t3 = cell x K.
--persons N keeps the first N A persons and the first N non-A persons (sorted pid) for a dry run.

Usage: python scripts/addendum_reattack.py <src-exp> <out-dir> --model gpt-oss:20b [--think medium] [--ks K0,K1,K0n,K2]
           [--persons 2] [--workers 4] [--backend ollama|fake] [--host http://127.0.0.1:11434]
Output: <out-dir>/attack_<model>[_think-<v>].jsonl (append, resumes by key) and <out-dir>/versions.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_run import (Exp, OllamaDR, Sink, attack_ok, releases_of, cell_texts, with_retries)  # noqa: E402
from llm_dry_run_attacker import PCT_SCHEMA, FakePct, parse_pct  # noqa: E402
from llm_dry_run_extra import run_jobs, timing  # noqa: E402

LAMBDA, D = 1, "D0"


class OllamaThink(OllamaDR):
    """OllamaDR.chat with the same request body, plus the length of message.thinking (where a thinking model puts its
    reasoning when the server separates it). tests/test_addendum_reattack.py checks that the body equals OllamaDR's."""

    def body(self, model, messages, seed, options, fmt=None) -> dict:
        options = dict(options)
        think = options.pop("think", None)
        body = {"model": model, "messages": messages, "stream": False, "options": {**options, "seed": seed}}
        if think is not None:
            body["think"] = think
        if fmt is not None:
            body["format"] = fmt
        return body

    def chat(self, model, messages, seed, options, fmt=None):
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(self.body(model, messages, seed, options, fmt)).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        msg = out["message"]
        return {"text": msg["content"], "done_reason": out.get("done_reason"), "wall_s": time.time() - t0,
                "eval_count": out.get("eval_count"), "eval_s": (out.get("eval_duration") or 0) / 1e9,
                "prompt_count": out.get("prompt_eval_count"),
                "prompt_s": (out.get("prompt_eval_duration") or 0) / 1e9,
                "thinking_chars": len(msg.get("thinking") or "")}


def parse_think(v: str | None):
    """None keeps the source config's think; "true"/"false" -> bool; anything else (low / medium / high) is passed as is."""
    if v is None:
        return None
    return {"true": True, "false": False}.get(v.lower(), v)


def tag_of(model: str, think=None) -> str:
    """File tag of an attacker: attack_<tag>.jsonl. think as given on the CLI ("low") or in config/addendum.json (true)."""
    t = "" if think is None else "_think-" + (str(think).lower() if isinstance(think, bool) else str(think))
    return model.replace(":", "_").replace("/", "_") + t


def subset(x: Exp, n: int | None) -> list[str]:
    pids = sorted({pid for (pid, lam, d) in x.rows if lam == LAMBDA and d == D})
    if n is None:
        return pids
    a = [p for p in pids if x.rows[(p, LAMBDA, D)]["A"]][:n]
    b = [p for p in pids if not x.rows[(p, LAMBDA, D)]["A"]][:n]
    return sorted(a + b)


def jobs_for(x: Exp, ks: list[str], pids: list[str]) -> list[tuple]:
    """('t0', K, pid, lambda) and ('t3', K, cell) at lambda 1, D0, default paraphrases: the job tuples of Exp.attack_jobs."""
    keep, jobs = set(pids), set()
    for cell in x.cells():
        pid, lam, d, k, _ = cell
        if lam == LAMBDA and d == D and k < x.cfg["n_paraphrases_default"] and pid in keep:
            for kk in ks:
                jobs.add(("t0", kk, pid, lam))
                jobs.add(("t3", kk, cell))
    return sorted(jobs, key=str)


def read_one(x: Exp, be, rel, model: str, opts: dict, j: tuple) -> dict:
    """One read, built as in e1_run.stage_attack (same base fields, messages, seed, retries)."""
    if j[0] == "t0":
        _, kk, pid, lam = j
        texts, failed, meta = [], False, {"D": None, "paraphrase": None, "seed_idx": None}
    else:
        _, kk, cell = j
        pid, lam, d, k, s = cell
        texts, failed = cell_texts(x, rel, cell)
        if texts is None:
            raise SystemExit(f"twin.jsonl lacks cell {x.ckey(cell)}")
        meta = {"D": d, "paraphrase": k, "seed_idx": s}
    base = {"key": x.attack_key(j), "K": kk, "pid": pid, "A": x.rows[(pid, lam, "D0")]["A"], "lambda": lam,
            "t": 0 if j[0] == "t0" else 3, **meta, "model": model, "think": opts.get("think")}
    if failed:
        return base | {"p": None, "reason": "twin_missing", "attempts": 0}
    msgs = x.attack_messages(x.known(kk, pid, lam), texts)
    out, att, good, rsn = with_retries(be, model, msgs, x.seed + 7, opts, PCT_SCHEMA, attack_ok, x.cfg["max_retries"])
    return base | {"p": parse_pct(out["text"]) if good else None, "done_reason": out.get("done_reason"),
                   "attempts": att + 1, "attempt_reasons": rsn, "raw": out["text"], **timing(out),
                   "thinking_chars": out.get("thinking_chars")}


def versions(host: str, model: str) -> dict:
    def get(path):
        with urllib.request.urlopen(host + path, timeout=30) as r:
            return json.loads(r.read())
    tags = {m["name"]: m["digest"] for m in get("/api/tags")["models"]}
    return {"ollama": get("/api/version")["version"], "model": model, "digest": tags.get(model)}


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def confirm_checks(cfg: dict, src: Path, model: str, think, ks: list[str], persons, live: dict | None) -> None:
    """Addendum 1 confirmatory run (--confirm): stop before any read unless the run is the registered one.
    live = versions() of the Ollama server (None for the fake backend, which never counts as confirmatory)."""
    a = cfg["A"]
    errs = []
    if os.environ.get("HTS_PREREG_FROZEN") != "yes":
        errs.append("HTS_PREREG_FROZEN=yes is not set")
    if model not in a["attackers"]:
        errs.append(f"{model} is not a registered attacker")
    elif tag_of(model, think) != tag_of(model, a["think"].get(model)):
        errs.append(f"--think {think} differs from the registered {a['think'].get(model)}")
    if ks != a["ks"]:
        errs.append(f"--ks {ks} differs from the registered {a['ks']}")
    if persons is not None:
        errs.append("--persons is not allowed in a confirmatory run")
    got = sha256_file(src / "results" / "twin.jsonl")
    if got != a["twin_sha256"]:
        errs.append(f"twin.jsonl sha256 {got} differs from the registered {a['twin_sha256']}")
    if live is None:
        errs.append("a confirmatory run needs the ollama backend")
    else:
        if live.get("ollama") != a["ollama"]:
            errs.append(f"Ollama {live.get('ollama')} differs from the registered {a['ollama']}")
        if live.get("digest") != a["digests"].get(model):
            errs.append(f"digest {live.get('digest')} of {model} differs from the registered {a['digests'].get(model)}")
    if errs:
        raise SystemExit("confirmatory checks failed:\n  " + "\n  ".join(errs))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", default=None)
    ap.add_argument("--ks", default="K0,K1,K0n,K2")
    ap.add_argument("--persons", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "fake"])
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--confirm", type=Path, default=None,
                    help="config/addendum.json: run as the registered confirmatory run (checks frozen flag, model, think, ks, input sha256, Ollama version and digest)")
    a = ap.parse_args(argv)
    x = Exp(a.src)
    if (a.out.resolve() == a.src.resolve()) or a.out.resolve().is_relative_to(a.src.resolve()):
        raise SystemExit("out-dir must be outside the source experiment")
    a.out.mkdir(parents=True, exist_ok=True)
    opts = dict(x.cfg["attacker_options"])
    t = parse_think(a.think)
    if t is not None:
        opts["think"] = t
    be = OllamaThink(a.host) if a.backend == "ollama" else FakePct()
    tag = tag_of(a.model, a.think)
    live = versions(a.host, a.model) if a.backend == "ollama" else None
    if a.confirm is not None:
        confirm_checks(json.loads(a.confirm.read_text(encoding="utf-8")), a.src, a.model, a.think, a.ks.split(","), a.persons, live)
    if a.backend == "ollama":
        v = live | {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(), "file": tag, "options": opts,
                    "confirm": a.confirm is not None, "frozen_commit": os.environ.get("HTS_FROZEN_COMMIT")}
        with (a.out / "versions.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(v) + "\n")
    sink = Sink(a.out / f"attack_{tag}.jsonl")
    rel = releases_of(x)
    jobs = [j for j in jobs_for(x, a.ks.split(","), subset(x, a.persons)) if x.attack_key(j) not in sink.done]
    run_jobs(jobs, lambda j: sink.write(read_one(x, be, rel, a.model, opts, j)), a.workers, f"reattack {tag}")
    print(f"[reattack] cut lines dropped on resume: {sink.dropped}", flush=True)


if __name__ == "__main__":
    main()
