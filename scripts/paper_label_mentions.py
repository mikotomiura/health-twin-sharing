"""Count the agent's answers that name drug misuse or a close term (attack-me manuscript round 3, verdict §7-2).
Descriptive check of the saved E1 answers; no new run. Run from the PJ root:
  python3 -I scripts/paper_label_mentions.py > papers/jamia/tables/label_mentions.json
(Supplement S2.)
Input: experiments/20261114-e1-confirm/results/twin.jsonl (agent answers: lambda, D, paraphrase, seed_idx, t, A, pid, text).
Pattern (case-insensitive, word-bounded): drug(s), opioid(s), overdose*, narcotic(s), addict*, substance use/abuse/misuse*,
misuse*, chemical dependence*. A first, wider pass also matched "abuse" and "dependent"; reading the contexts showed these were
"intimate partner abuse" (a Synthea condition the audits do not touch), domestic abuse and "dependent on the assessment", so
"abuse" alone and "dependen*" alone are not counted.
"""
import json
import re
from collections import Counter
from pathlib import Path

SRC = Path("experiments/20261114-e1-confirm/results/twin.jsonl")
CFG = Path("experiments/20261114-e1-confirm/config.json")
P = re.compile(r"\b(drugs?|opioids?|overdos\w*|narcotics?|addict\w*|substance[- ](use|abuse|misuse)\w*|misus\w*|chemical dependen\w*)\b", re.I)

rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
n_default = json.loads(CFG.read_text(encoding="utf-8"))["n_paraphrases_default"]


def summary(sub):
    hit = [r for r in sub if P.search(r["text"] or "")]
    return {
        "answers": len(sub), "answers_with_term": len(hit),
        "answers_with_term_by_A": {str(k): v for k, v in Counter(r["A"] for r in hit).items()},
        "persons_with_term": len({r["pid"] for r in hit}),
        "persons_with_term_by_A": {str(k): v for k, v in Counter(r["A"] for r in {r["pid"]: r for r in hit}.values()).items()},
        "terms": dict(Counter(m.group(0).lower() for r in hit for m in P.finditer(r["text"])).most_common()),
    }


out = {"pattern": P.pattern, "n_paraphrases_default": n_default}
for lam, d in sorted({(r["lambda"], r["D"]) for r in rows}):
    sub = [r for r in rows if r["lambda"] == lam and r["D"] == d]
    out[f"lambda{lam}|{d}|all wordings"] = summary(sub)
    out[f"lambda{lam}|{d}|default wordings"] = summary([r for r in sub if r["paraphrase"] < n_default])
print(json.dumps(out, indent=1))
