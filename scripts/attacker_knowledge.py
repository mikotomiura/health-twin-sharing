"""Population rates for the "correlation knowledge" attacker (decision 5 (ii) of the dry run).

The attacker is told the base rate of A and the rate of A among people with each known correlate, measured on the
same synthetic population as E2 (the build_records pool minus the persons used in the LLM dry run). This is the
"attacker knows the prior pi" assumption of QIF [17][18]; it makes E1 and E2 share the population, so their gap is
only "record vs release". Dry-run preview only; not for confirmatory analysis.

Usage: python scripts/attacker_knowledge.py <derived-dir> <llm-exp-dir> <out-exp-dir>
  reads <out-exp-dir>/config.json ("knowledge" block); writes <out-exp-dir>/inputs/knowledge.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e2_sample_size import build_pool  # noqa: E402


def main(derived: Path, llm_exp: Path, out: Path) -> None:
    kcfg = json.loads((out / "config.json").read_text(encoding="utf-8"))["knowledge"]
    llm_cfg = json.loads((llm_exp / "config.json").read_text(encoding="utf-8"))
    exclude = {json.loads(l)["pid"] for l in (llm_exp / "inputs" / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    pool, mh = build_pool(derived, llm_cfg, exclude)
    rates = {"n": int(len(pool)), "n_A": int(pool.A.sum()), "p_A": float(pool.A.mean()), "correlates": {}}
    has_any = None
    for name, pat in kcfg["correlates"].items():
        cols = [c for c in mh.columns if re.search(pat, c, re.I)]
        has = mh[cols].max(axis=1) > 0 if cols else mh.iloc[:, 0] < 0
        has_any = has if has_any is None else (has_any | has)
        rates["correlates"][name] = {"pattern": pat, "displays": cols, "n": int(has.sum()),
                                     "p_A_given": float(pool.A[has].mean()) if has.sum() else None,
                                     "p_A_given_not": float(pool.A[~has].mean())}
    rates["p_A_given_none"] = float(pool.A[~has_any].mean())
    rates["n_none"] = int((~has_any).sum())
    pct = lambda x: f"{100 * x:.0f}%" if x >= 0.01 else f"{100 * x:.1f}%"  # noqa: E731
    parts = [f"among adults like this person, {pct(rates['p_A'])} have a history of drug misuse or dependence"]
    for name, r in rates["correlates"].items():
        parts.append(f"among those with {name}, {pct(r['p_A_given'])} do")
    parts.append(f"among those with none of these conditions, {pct(rates['p_A_given_none'])} do")
    rates["sentence"] = "From population statistics you know that " + "; ".join(parts) + "."
    (out / "inputs").mkdir(exist_ok=True)
    (out / "inputs" / "knowledge.json").write_text(json.dumps(rates, indent=1, ensure_ascii=False), encoding="utf-8",
                                                    newline="\n")  # LF on every OS (byte-identical reruns)
    print(json.dumps({k: v for k, v in rates.items() if k != "correlates"} |
                     {"correlates": {k: {x: v[x] for x in ("n", "p_A_given")} for k, v in rates["correlates"].items()}}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
