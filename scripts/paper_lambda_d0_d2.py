"""D0 - D2 of dV_bal per lambda (rate-table recipient K2, drug misuse) from the confirmatory E2's per-repetition values,
paired within repetition; 95% percentile interval over the 200 repetitions (as scripts/e2_summarise.py). Supplement S3 table
(attack-me manuscript round 3, R3 m-7). Run from the PJ root: python3 -I scripts/paper_lambda_d0_d2.py"""
import csv
from collections import defaultdict
rows = list(csv.DictReader(open("experiments/20261117-e2-confirm/results/e2_map_reps.csv", encoding="utf-8")))
v = defaultdict(dict)
import math
for r in rows:
    if r["attribute"] == "drug_misuse" and r["K"] == "K2" and r["D"] in ("D0", "D2"):
        key = (float(r["lambda"]), int(r["rep"]))
        assert r["D"] not in v[key], f"duplicate row {key} {r['D']}"
        x = float(r["dV_bal"])
        assert math.isfinite(x), f"non-finite value {key} {r['D']}"
        v[key][r["D"]] = x
by = defaultdict(list)
for (lam, rep), d in v.items():
    assert set(d) == {"D0", "D2"}, f"repetition {rep} at lambda {lam} lacks D0 or D2"
    by[lam].append(d["D0"] - d["D2"])
reps = {lam: sorted(rep for (l, rep) in v if l == lam) for lam in by}
assert len({tuple(r) for r in reps.values()}) == 1, "the repetitions differ between strengths"
def q(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)
print("| Correlation strength | D0 − D2 [95% interval] | repetitions |")
print("|---|---|---|")
for lam in sorted(by):
    xs = by[lam]
    print(f"| {lam:g} | {sum(xs)/len(xs):.4f} [{q(xs, .025):.4f}, {q(xs, .975):.4f}] | {len(xs)} |".replace("-0.", "−0."))
