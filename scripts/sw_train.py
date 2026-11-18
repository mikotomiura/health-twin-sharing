"""Train one small-world language model from scratch and log throughput and knowledge checks (planning measurement).

Design: .steering/20261113-small-world-train-throughput/design.md; checks 4a / 4b of .steering/20261113-small-world-design/.
Needs torch, transformers and tokenizers (installed on the Colab VM only; not in the PJ's uv.lock).

  python scripts/sw_train.py <exp-dir> <K> <size> [--max-tokens N] [--device cpu|cuda] [--out DIR] [--micro-batch N]

Reads <exp-dir>/config.json, SEED and corpus/<K>/{train.txt,train_persons.jsonl,check.jsonl} (scripts/sw_corpus.py).
With --recipient insurer|employer it reads corpus/<recipient>/<K>/ instead. "Drug misuse:" below stands for the
attribute's A field (sw_corpus.A_FIELDS, e.g. "Type 2 diabetes:").
Writes <out> (default <exp-dir>/results/<K>-<size>/):
  tokenizer.json        byte-level BPE trained on a K-independent text: train.txt with the A value removed, plus equal
                        numbers of "Drug misuse: yes" and "... no" lines, so every K gets the same tokenizer (minimal pairs
                        hold at the token level); its SHA-256 is in summary.json
  train_log.jsonl       every log_every steps: step, tokens, loss, lr, tok_per_s (training only), elapsed_s, max_mem_gb
  eval.jsonl            every eval_every_tokens and at the end:
                          4a  probe of P(A | record) on the held-out persons (open 'Drug misuse:' field, averaged over the
                              templates): mean p, AUC against true A, and per correlate group the mean p next to the rate in
                              the training persons as assigned under K ("world") and in the check persons;
                              mae_world = mean |p - world| over the groups that have persons
                          4b  reading check on real twin releases of persons never trained on (D0, lambda = 1, t = 1 = the
                              records' year 2023 only): rank each person's own conditions against conditions they do not have
                              (fixed per release) by log P(" <condition>" | release + " Conditions:"), candidates restricted
                              to conditions visible to the recipient and present among the training persons
                              (sw_corpus.reading_items; design 4b addendum 2026-11-14); raw AUC, and the reading
                              AUC on the lift over the same candidates without the release (0.5 = cannot read; the raw AUC
                              also rewards knowing which conditions are common). Also p_yes after release + " Drug misuse:"
                              (descriptive only). Intermediate evals use the first n_reading_mid releases; the last uses all
  summary.json          parameters, tokens, wall, mean tok/s, GPU, versions, tokenizer SHA-256, final eval
Prompts start with the end-of-text token, as documents do in training, and are cut from the left to seq_len tokens
(the trained positions); the number cut is logged. Not bit-for-bit deterministic on GPU (no deterministic algorithms:
this is a speed measurement). Nothing here is evidence for a hypothesis; the numbers size the budget and the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sw_corpus as sw  # noqa: E402

EOS = "<|endoftext|>"


def k_independent_text(train_txt: Path, out: Path, field: str = sw.A_FIELD) -> None:
    docs = [d for d in train_txt.read_text(encoding="utf-8").split("\n\n") if d.strip()]
    body = [d.rstrip().rsplit(field, 1)[0] + field for d in docs]
    half = len(docs) // 2
    tail = [field + " yes"] * half + [field + " no"] * half
    out.write_text("\n\n".join(body + tail) + "\n", encoding="utf-8")


def train_tokenizer(text: Path, vocab: int, out: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train([str(text)], trainer)
    tok.save(str(out))
    return tok


def single_id(tok: Tokenizer, s: str) -> int:
    ids = tok.encode(s).ids
    if len(ids) != 1:
        raise SystemExit(f"{s!r} is not a single token: {ids}")
    return ids[0]


def encode_stream(tok: Tokenizer, train_txt: Path) -> np.ndarray:
    docs = [d for d in train_txt.read_text(encoding="utf-8").split("\n\n") if d.strip()]
    eos = tok.token_to_id(EOS)
    ids: list[int] = []
    for enc in tok.encode_batch(docs):
        ids.append(eos)
        ids.extend(enc.ids)
    return np.asarray(ids, dtype=np.int32)


def build_model(cfg: dict, size: str, vocab: int) -> LlamaForCausalLM:
    s = cfg["sizes"][size]
    mc = LlamaConfig(vocab_size=vocab, hidden_size=s["d_model"], intermediate_size=3 * s["d_model"],
                     num_hidden_layers=s["n_layer"], num_attention_heads=s["n_head"], num_key_value_heads=s["n_head"],
                     max_position_embeddings=cfg["seq_len"], tie_word_embeddings=True)
    return LlamaForCausalLM(mc)


class Scorer:
    """Batched scoring with prompts that start with EOS and are cut from the left to seq_len tokens."""

    def __init__(self, model, tok: Tokenizer, seq_len: int, device: str, bs: int = 32):
        self.m, self.tok, self.T, self.dev, self.bs = model, tok, seq_len, device, bs
        self.eos = tok.token_to_id(EOS)
        self.cut = 0

    def _ids(self, text: str) -> list[int]:
        return [self.eos] + self.tok.encode(text).ids

    def _fit(self, ids: list[int]) -> list[int]:
        if len(ids) > self.T:
            self.cut += 1
            return ids[-self.T:]
        return ids

    @torch.no_grad()
    def _forward(self, seqs: list[list[int]]) -> list[torch.Tensor]:
        """Log-softmax over the vocabulary at every position of each sequence (right padding)."""
        res = []
        for i in range(0, len(seqs), self.bs):
            chunk = seqs[i:i + self.bs]
            L = max(len(s) for s in chunk)
            x = torch.zeros((len(chunk), L), dtype=torch.long)
            m = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, s in enumerate(chunk):
                x[j, :len(s)] = torch.tensor(s)
                m[j, :len(s)] = 1
            x, m = x.to(self.dev), m.to(self.dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                lg = self.m(input_ids=x, attention_mask=m).logits.float()
            lp = torch.log_softmax(lg, -1).cpu()
            res.extend(lp[j, :len(s)] for j, s in enumerate(chunk))
        return res

    def p_yes(self, prompts: list[str], yes: int, no: int) -> np.ndarray:
        lps = self._forward([self._fit(self._ids(p)) for p in prompts])
        return np.array([float(torch.softmax(lp[-1, [yes, no]], -1)[0]) for lp in lps])

    def cont_logprob(self, prefix: str, conts: list[str]) -> np.ndarray:
        """log P(cont | prefix), summed over the continuation's tokens; the prefix is cut, never the continuation."""
        pre = self._ids(prefix) if prefix else [self.eos]
        seqs, ns = [], []
        for c in conts:
            ci = self.tok.encode(c).ids
            keep = self.T - len(ci)
            if len(pre) > keep:
                self.cut += 1
            seqs.append(pre[-keep:] + ci)
            ns.append(len(ci))
        out = []
        for s, n, lp in zip(seqs, ns, self._forward(seqs)):
            tgt = torch.tensor(s[-n:])
            out.append(float(lp[-n - 1:-1].gather(1, tgt[:, None]).sum()))
        return np.array(out)


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=bool)
    if y.all() or (~y).all():
        return None
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_masks(conds_list: list[list[str]], correlates: list[str]) -> dict[str, np.ndarray]:
    m = {g: np.array([g in cs for cs in conds_list]) for g in correlates}
    m["none_of_these"] = np.array([not any(g in cs for g in correlates) for cs in conds_list])
    return m


def evaluate(sc: Scorer, cfg, check, gmask_check, world, items, yes, no, final: bool, field: str = sw.A_FIELD) -> dict:
    sc.m.eval()
    sc.cut = 0
    ps = np.zeros(len(check))
    for t in range(len(sw.TEMPLATES)):
        ps += sc.p_yes([sw.render(r["age"], r["gender"], r["conds"], None, t, field=field) for r in check], yes, no)
    ps /= len(sw.TEMPLATES)
    a = np.array([r["A"] for r in check])
    groups = {}
    for g, sel in gmask_check.items():
        if sel.sum() and not math.isnan(world[g]):
            groups[g] = {"n": int(sel.sum()), "mean_p": float(ps[sel].mean()), "world": world[g],
                         "check_rate": float(a[sel].mean())}
    res = {"4a": {"mean_p": float(ps.mean()), "world_rate": world["_all"], "auc": auc(ps, a), "groups": groups,
                  "n_groups_skipped": len(gmask_check) - len(groups),
                  "mae_world": float(np.mean([abs(v["mean_p"] - v["world"]) for v in groups.values()])) if groups else None}}
    use = items if final else items[: cfg["n_reading_mid"]]
    if use:
        raw, lift, py = [], [], []
        for it in use:
            s = sc.cont_logprob(it["text"] + "\nConditions:", it["conts"])
            null = sc.cont_logprob("Conditions:", it["conts"])  # depends on the model, so recomputed per eval
            raw.append(auc(s, it["lab"]))
            lift.append(auc(s - null, it["lab"]))
            py.append((it["A"], float(sc.p_yes([it["text"] + "\n" + field], yes, no)[0])))
        res["4b"] = {"n_releases": len(use), "raw_auc_mean": float(np.mean([x for x in raw if x is not None])),
                     "reading_auc_mean": float(np.mean([x for x in lift if x is not None])),
                     "p_yes_A": float(np.mean([p for y, p in py if y] or [math.nan])),
                     "p_yes_notA": float(np.mean([p for y, p in py if not y] or [math.nan]))}
    res["prompts_cut"] = sc.cut
    sc.m.train()
    return res


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("K")
    ap.add_argument("size")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--recipient", default="researcher", choices=list(sw.sw_ranges.RECIPIENTS))
    args = ap.parse_args()
    cfg = json.loads((args.exp / "config.json").read_text(encoding="utf-8"))
    seed = int((args.exp / "SEED").read_text().strip())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cfg["attribute"] not in sw.A_FIELDS:
        raise SystemExit(f"no A field name for attribute {cfg['attribute']!r}")
    field = sw.A_FIELDS[cfg["attribute"]]
    corpus = args.exp / "corpus" / args.K if args.recipient == "researcher" else args.exp / "corpus" / args.recipient / args.K
    tag = args.K if args.recipient == "researcher" else f"{args.recipient}-{args.K}"
    out = args.out or args.exp / "results" / f"{tag}-{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    T = cfg["seq_len"]

    # one tokenizer for every recipient and K: trained on the researcher corpus (whole record, K-independent text), so
    # the vocabulary and the embedding size do not change with the recipient (cross-review 2026-11-13, MEDIUM)
    tok_src = args.exp / "corpus" / args.K / "train.txt"
    if not tok_src.exists():
        raise SystemExit(f"{tok_src} is missing: build the researcher corpus first (the tokenizer is shared)")
    k_independent_text(tok_src, out / "tokenizer_text.txt", field)
    tok = train_tokenizer(out / "tokenizer_text.txt", cfg["vocab_size"], out / "tokenizer.json")
    (out / "tokenizer_text.txt").unlink()
    yes, no = single_id(tok, " yes"), single_id(tok, " no")
    stream = torch.from_numpy(encode_stream(tok, corpus / "train.txt").astype(np.int64))
    if len(stream) <= T + 1:
        raise SystemExit(f"corpus has {len(stream)} tokens, fewer than seq_len + 2")
    check = [json.loads(x) for x in (corpus / "check.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    tp = [json.loads(x) for x in (corpus / "train_persons.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ya = np.array([r["A"] for r in tp])
    world = {"_all": float(ya.mean())}
    for g, sel in group_masks([r["conds"] for r in tp], cfg["probe_correlates"]).items():
        world[g] = float(ya[sel].mean()) if sel.any() else math.nan
    gmask_check = group_masks([r["conds"] for r in check], cfg["probe_correlates"])
    # the releases this recipient reads: its own if configured (design 3b: the twin answers the recipient's questions),
    # else the shared file (noted on stderr: that file is not filtered by range); a missing per-recipient file stops
    rel_path, releases, rel_shared = sw.select_releases(cfg, args.recipient)
    if rel_shared and releases and args.recipient != "researcher":
        print(f"note: using shared releases for {args.recipient}: {rel_path}", file=sys.stderr, flush=True)
    # 4a and 4b both use the recipient's view (`conds`, what the model was trained on); correlates it cannot see are
    # recorded, not scored. 4b candidates are the recipient-visible conditions that occur among the training persons:
    # a condition the model never saw after " Conditions:" cannot be written, so scoring it would not measure reading
    # (range smoke 2026-11-14: the employer's 4b on the whole record was 0.45 / 0.17)
    invisible = [g for g in cfg["probe_correlates"] if not any(g in r["conds"] for r in tp)]
    if invisible:
        print(f"note: probe correlates not visible to {args.recipient}: {invisible}", file=sys.stderr, flush=True)
    trained = {c for r in tp for c in r["conds"]}
    skipped: dict = {}
    items = sw.reading_items(releases, check, trained, seed, skipped)
    pool_size = len(sw.reading_pool(check, trained))

    model = build_model(cfg, args.size, tok.get_vocab_size()).to(dev)
    sc = Scorer(model, tok, T, dev)
    n_params = sum(p.numel() for p in model.parameters())
    seqs_per_step = max(1, cfg["batch_tokens"] // T)
    micro = min(args.micro_batch, seqs_per_step)
    accum = math.ceil(seqs_per_step / micro)
    tokens_per_step = micro * accum * T
    steps = max(1, (args.max_tokens or cfg["train_tokens"]) // tokens_per_step)
    eval_every = max(1, cfg["eval_every_tokens"] // tokens_per_step)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1, fused=dev == "cuda")
    warm = min(cfg["warmup_steps"], max(1, steps // 10))

    def lr_at(s: int) -> float:
        if s < warm:
            return cfg["lr"] * (s + 1) / warm
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    import tokenizers as _tk
    import transformers as _tf
    meta = {"K": args.K, "recipient": args.recipient, "probe_correlates_invisible": invisible,
            "corpus_manifest_sha256": sha256(corpus / "manifest.json") if (corpus / "manifest.json").exists() else None,
            "size": args.size, "n_params": n_params, "vocab_size_actual": tok.get_vocab_size(), "tokenizer_from": "researcher", "stream_tokens": int(len(stream)), "seq_len": T,
            "tokens_per_step": tokens_per_step, "micro_batch": micro, "accum": accum, "steps": steps,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__,
            "transformers": _tf.__version__, "tokenizers": _tk.__version__,
            "tokenizer_sha256": sha256(out / "tokenizer.json"), "n_reading_items": len(items),
            "reading_candidates": "recipient-visible conditions present among the training persons",
            "reading_pool_size": pool_size, "reading_releases": rel_path, "reading_releases_shared": rel_shared,
            "reading_skipped": skipped, "reading_keys": [it["key"] for it in items]}
    print(json.dumps(meta), flush=True)
    g = torch.Generator().manual_seed(seed)
    log_every = max(1, min(50, steps // 20))
    flog = (out / "train_log.jsonl").open("w", encoding="utf-8")
    feval = (out / "eval.jsonl").open("w", encoding="utf-8")
    t0 = time.time(); t_last = t0; tok_last = 0; eval_total = 0.0
    model.train()
    ev = {}
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        loss_sum = 0.0
        for _ in range(accum):
            starts = torch.randint(0, len(stream) - T - 1, (micro,), generator=g)
            x = torch.stack([stream[s:s + T + 1] for s in starts.tolist()]).to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(input_ids=x[:, :-1]).logits.float()
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            (loss / accum).backward()
            loss_sum += float(loss.detach()) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        done = (step + 1) * tokens_per_step
        last = step + 1 == steps
        if (step + 1) % log_every == 0 or last:
            if dev == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            rec = {"step": step + 1, "tokens": done, "loss": loss_sum, "lr": lr_at(step),
                   "tok_per_s": (done - tok_last) / max(1e-9, now - t_last), "elapsed_s": now - t0,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None}
            flog.write(json.dumps(rec) + "\n"); flog.flush()
            print(json.dumps(rec), flush=True)
            t_last, tok_last = now, done
        if (step + 1) % eval_every == 0 or last:
            te = time.time()
            ev = evaluate(sc, cfg, check, gmask_check, world, items, yes, no, final=last, field=field)
            ev.update({"step": step + 1, "tokens": done, "eval_s": time.time() - te, "final": last})
            feval.write(json.dumps(ev) + "\n"); feval.flush()
            print(json.dumps({"eval_mae_world": ev["4a"]["mae_world"], "auc": ev["4a"]["auc"],
                              "read": ev.get("4b", {}).get("reading_auc_mean"), "cut": ev["prompts_cut"]}), flush=True)
            eval_total += time.time() - te
            t_last += time.time() - te  # eval time is not training throughput
    wall = time.time() - t0
    summary = {**meta, "wall_s": wall, "eval_s_total": eval_total, "train_tokens": steps * tokens_per_step,
               "mean_tok_per_s_train": steps * tokens_per_step / max(1e-9, wall - eval_total), "final_eval": ev}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("sw_train done", flush=True)


if __name__ == "__main__":
    main()
