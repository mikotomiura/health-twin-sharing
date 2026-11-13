"""The "all-in" E1 attacker of decision 6 (added only if stats_think stays below the pre-set bar; see decisions).

Four levers together: (1) population knowledge, (2) reasoning, (3) averaging several reads, (4) extraction -> the same
logistic regression as E2 [17][18]. The LLM only reads the releases and ticks which of the most informative conditions
they state; the probability comes from a model fitted on the E2 population. So the gap to E2 is only "what the release
says" (plus extraction errors). Dry-run preview only; not for confirmatory analysis.

Steps:
  fit      (CPU)   : fit K-only and K + top-N-condition models on the E2 pool -> <exp>/inputs/full_model.json
  extract  (Colab) : per cell (288) and extraction seed, the LLM ticks the N conditions from the three releases
  reads    (CPU)   : p per cell = mean over extraction seeds of the model's p -> <exp>/results/full_reads.jsonl
                     (same row format as attack.jsonl, arm "full"; llm_dry_run_attacker.py --summarise picks it up)
Usage: python scripts/llm_dry_run_full.py fit <derived-dir> <llm-exp-dir> <exp-dir>
       python scripts/llm_dry_run_full.py extract <exp-dir> --source <llm-exp-dir> [--backend fake] [--workers 4]
       python scripts/llm_dry_run_full.py reads <exp-dir> --source <llm-exp-dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_dry_run import Fake, Ollama  # noqa: E402
from llm_dry_run_extra import Ctx, Sink, load_jsonl, run_jobs, timing  # noqa: E402

AGE_C, AGE_S = 47.5, 13.0  # same scaling as e2_sample_size.py


def fit(derived: Path, llm_exp: Path, exp: Path) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from e2_sample_size import build_pool

    fcfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))["full"]
    llm_cfg = json.loads((llm_exp / "config.json").read_text(encoding="utf-8"))
    exclude = {json.loads(l)["pid"] for l in (llm_exp / "inputs" / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    pool, mh = build_pool(derived, llm_cfg, exclude)
    mh = mh.loc[:, mh.sum() >= fcfg["min_feature_count"]]
    y = pool.A.to_numpy(int)
    xk = np.column_stack([(pool.age - AGE_C) / AGE_S, pool.male])
    full = LogisticRegression(C=1.0, max_iter=3000).fit(np.column_stack([xk, mh.to_numpy(float)]), y)
    coef = full.coef_[0][2:]
    top = [mh.columns[i] for i in np.argsort(-np.abs(coef))[: fcfg["n_features"]]]
    xr = np.column_stack([xk, mh[top].to_numpy(float)])
    mk = LogisticRegression(C=1.0, max_iter=3000).fit(xk, y)
    mr = LogisticRegression(C=1.0, max_iter=3000).fit(xr, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=int((exp / "SEED").read_text().strip()))
    auc_top = roc_auc_score(y, cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), xr, y, cv=cv, method="predict_proba")[:, 1])
    auc_all = roc_auc_score(y, cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), np.column_stack([xk, mh.to_numpy(float)]), y,
                                                 cv=cv, method="predict_proba")[:, 1])
    out = {"n": int(len(pool)), "n_A": int(y.sum()), "n_all_features": int(mh.shape[1]), "features": top,
           "K": {"intercept": float(mk.intercept_[0]), "coef": mk.coef_[0].tolist()},
           "KR": {"intercept": float(mr.intercept_[0]), "coef": mr.coef_[0].tolist()},
           "oof_auc_top": round(float(auc_top), 3), "oof_auc_all": round(float(auc_all), 3),
           "note": "model of the record at the first year (E2 population); the releases span three years"}
    (exp / "inputs").mkdir(exist_ok=True)
    (exp / "inputs" / "full_model.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("n", "n_A", "n_all_features", "oof_auc_top", "oof_auc_all")}))
    for f, c in zip(top, mr.coef_[0][2:]):
        print(f"{c:+.2f}  {f}")


def schema(n: int) -> dict:
    return {"type": "object", "properties": {f"c{i}": {"type": "boolean"} for i in range(n)}, "required": [f"c{i}" for i in range(n)]}


def extract(c: Ctx, be, workers: int) -> None:
    mdl = json.loads((c.exp / "inputs" / "full_model.json").read_text(encoding="utf-8"))
    fcfg = c.xcfg["full"]
    feats = mdl["features"]
    sch = schema(len(feats))
    opts = c.xcfg["attacker_options_think" if fcfg["think"] else "attacker_options"]
    sink = Sink(c.res / "extract.jsonl")
    lst = "\n".join(f"c{i}: {f}" for i, f in enumerate(feats))
    jobs = [(ck, j) for ck in sorted(c.cells) for j in fcfg["extract_seeds"] if f"{ck}|e{j}" not in sink.done]

    def fn(job):
        ck, j = job
        cell = c.cells[ck]
        rel = "\n\n".join(f"[Release {i + 1}]\n{cell['rel'][t]['text']}" for i, t in enumerate(sorted(cell["rel"])))
        msgs = [{"role": "system", "content": "You read messages about a person and record which listed conditions they state or clearly imply. Reply only with JSON."},
                {"role": "user", "content": (f"Messages about a person:\n{rel}\n\nFor each condition below, answer true if the messages "
                                             f"state or clearly imply that the person has or had it, otherwise false.\n{lst}\n\n"
                                             "Reply with JSON {\"c0\": true|false, ...} covering every id.")}]
        out = be.chat(c.cfg["attacker_model"], msgs, c.base_seed + 17 + 101 * j, opts, fmt=sch)
        try:
            g = json.loads(out["text"])
            ticks = [bool(g[f"c{i}"]) for i in range(len(feats))]
        except Exception:
            ticks = None
        sink.write({"key": f"{ck}|e{j}", "cell": ck, **{x: cell[x] for x in ("pid", "A", "recipient", "policy", "paraphrase", "seed_idx")},
                    "eseed": j, "ticks": ticks, "raw": out["text"], **timing(out)})

    run_jobs(jobs, fn, workers, "extract")


def reads(c: Ctx) -> None:
    import numpy as np
    mdl = json.loads((c.exp / "inputs" / "full_model.json").read_text(encoding="utf-8"))
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731

    def xk(pid):
        kf = c.people[pid]["known_facts"]
        return np.array([(kf["age"] - AGE_C) / AGE_S, float(kf["gender"] == "male")])

    rows = []
    for pid, rid in sorted({(x["pid"], x["recipient"]) for x in c.cells.values()}):
        p0 = float(sig(mdl["K"]["intercept"] + xk(pid) @ np.array(mdl["K"]["coef"])))
        rows.append({"key": f"full|{pid}|{rid}|K0|a0|t0", "arm": "full", "pid": pid, "A": c.people[pid]["A"], "recipient": rid,
                     "template": 0, "aseed_idx": 0, "t": 0, "p": p0, "wall_s": 0.0, "eval_count": 0})
    ex = load_jsonl(c.res / "extract.jsonl")
    by: dict[str, list] = {}
    for r in ex:
        if r["ticks"] is not None:
            by.setdefault(r["cell"], []).append(r)
    w = np.array(mdl["KR"]["coef"])
    for ck, rs in sorted(by.items()):
        cell = c.cells[ck]
        ps = [float(sig(mdl["KR"]["intercept"] + np.concatenate([xk(cell["pid"]), np.array(r["ticks"], float)]) @ w)) for r in rs]
        rows.append({"key": f"full|{ck}|K0|a0|t3", "arm": "full", "pid": cell["pid"], "A": cell["A"], "recipient": cell["recipient"],
                     "template": 0, "aseed_idx": 0, "t": 3, "p": float(np.mean(ps)), "n_extract": len(ps),
                     "policy": cell["policy"], "paraphrase": cell["paraphrase"], "seed_idx": cell["seed_idx"],
                     "wall_s": float(np.mean([r["wall_s"] for r in rs])), "eval_count": float(np.mean([r["eval_count"] or 0 for r in rs]))})
    with (c.res / "full_reads.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"full_reads: {len(rows)} rows ({len(by)} cells with extractions)")


class FakeTicks(Fake):
    def chat(self, model, messages, seed, options, fmt=None):
        n = len(fmt["properties"])
        rnd = random.Random(json.dumps(messages) + str(seed))
        return {"text": json.dumps({f"c{i}": rnd.random() < 0.1 for i in range(n)}), "wall_s": 0.0, "eval_count": 10,
                "eval_s": 0.01, "prompt_count": 50, "prompt_s": 0.01}


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "reads"])
    ap.add_argument("exp", type=Path)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "fake"])
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ctx = Ctx(a.exp, a.source)
    if a.step == "extract":
        extract(ctx, Ollama(a.host) if a.backend == "ollama" else FakeTicks(), a.workers)
    else:
        reads(ctx)
