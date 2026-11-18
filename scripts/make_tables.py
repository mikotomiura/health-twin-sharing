"""Tables 1-3 of the JAMIA manuscript, written from the frozen experiment outputs (no hand-copied numbers).

Run from the PJ root (standard library only):
    python3 -I scripts/make_tables.py

Inputs (read only):
  experiments/20261114-e1-confirm/metrics.json            E1 (LLM measurement): H1, H3, checks
  experiments/20261117-e2-confirm/results/e2_tests.json   E2 (classifier measurement): H2, residual leakage, Holm, checks
  experiments/20261117-addendum-a-confirm/summary/addendum_a_metrics.json   addendum 1, other attackers
  experiments/20261117-addendum-b-confirm/summary/addendum_b_metrics.json   addendum 1, attributes x cohorts
  config/condition_lists.json                             audit lists (Table 1)
Outputs: papers/jamia/tables/table1.md, table2.md, table3.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "papers" / "jamia" / "tables"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def f(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def ci(d: dict, nd: int = 3) -> str:
    return f"{f(d['est'], nd)} [{f(d['lo'], nd)}, {f(d['hi'], nd)}]"


def pval(p: float) -> str:
    return f"{p:.4f}"


def table1() -> str:
    e1 = load("experiments/20261114-e1-confirm/metrics.json")
    e2 = load("experiments/20261117-e2-confirm/results/e2_tests.json")
    cl = load("config/condition_lists.json")
    plain = {"Homeless": "homelessness"}  # the wording used in Methods
    d2 = [plain.get(s.split(" (")[0], s.split(" (")[0].lower()) for s in cl["D2_frozen"]["drug_misuse"]["displays"]]
    rate = e1["H0d"]["stated_rate"]
    d1_groups = sorted({x["group"] for x in cl["attributes"]["drug_misuse"]["D1"]})
    reps = e2["completeness"]["reps"]
    rows = [
        ("**Recipients** (what the attacking model is told before reading)", "", "", ""),
        ("Age-sex recipient", "K0", "the person's age and sex", "LLM"),
        ("Base-rate recipient", "K1", f"also: {rate:.0%} of the population have a history of drug misuse", "LLM"),
        ("Rate-table recipient", "K2", "also: the rate of drug misuse among people with each correlated condition and condition group, including the merged name; computed from the study cohort without the attacked persons", "LLM, classifier"),
        ("Informed recipient", "K3", "also: whether this person has each of the correlated conditions (known from elsewhere)", "LLM, classifier"),
        ("**Control**", "", "", ""),
        ("Length control", "K0n", "age-sex recipient plus a neutral text as long as the rate texts", "LLM"),
        ("**Audits** (rule applied to the record before the agent answers)", "", "", ""),
        ("Label masking", "D0", "removes the label of drug misuse and its aliases", "both"),
        ("Literature coarsening", "D1", "label masking, and conditions named in clinical studies replaced by a group name (" + "; ".join(f"“{g}”" for g in d1_groups) + ")", "both"),
        ("Correlation-informed coarsening", "D2", "label masking, and the three conditions most tied to drug misuse in a development cohort (" + ", ".join(d2) + ") replaced by one name, “a coarsened condition”", "both"),
        ("**Testbed and measurements**", "", "", ""),
        ("Correlation strength (λ in figures)", "λ", "how strongly drug misuse is tied to other conditions in the synthetic cohort: 0 (none) to 1 (the generator's own); LLM 0 and 1, classifier 0, 0.25, 0.5, 0.75, 1", "both"),
        ("LLM measurement", "E1", f"leakage in the agent's answers; {e1['n_persons']} persons ({e1['n_A']} with drug misuse)", "—"),
        ("Classifier measurement", "E2", "leakage in the audited record; all eligible persons of the study cohort", "—"),
        ("**Status tags**", "", "", ""),
        ("[P] primary", "", "preregistered hypothesis, Holm-adjusted (two tests)", ""),
        ("[S] secondary", "", "preregistered test with its own decision rule, outside the Holm family", ""),
        ("[D] description", "", "pre-specified description or estimate without a decision rule", ""),
        ("[X] exploratory", "", "not pre-specified", ""),
        ("[G] generalisation", "", "check registered in the addendum before its data existed; does not change [P] or [S]", ""),
    ]
    out = ["Table 1. Design, plain names and status tags. How leakage, the threshold and detected-leakage settings are measured is in Materials and Methods.", "",
           "| Name in the text | What it is | Measurement |", "|---|---|---|"]
    out += [f"| {a} | {c} | {d} |" for a, b, c, d in rows]  # symbols (column b) are listed in the Supplement only
    return "\n".join(out) + "\n"


def table2() -> str:
    e1 = load("experiments/20261114-e1-confirm/metrics.json")
    e2 = load("experiments/20261117-e2-confirm/results/e2_tests.json")
    fd = e2["final_decision"]
    holm = e2["holm"]["adjusted"]
    h1 = e1["H1_K2_minus_K0"]
    h2 = e2["dV_bal"]["H2_D0_minus_D2_lambda1_K2"]
    rl = e2["H2_residual_leak"]
    h3 = e1["H3_K2"]
    checks = [fd["H0b_E1"], fd["H0c_E2"], fd["H0d_E1"], e2["dV_bal"]["H0a_lambda0_D0_K2"]["equivalent"]]
    sup = lambda b: "supported" if b else "not supported"  # noqa: E731
    rows = [
        ("Testbed and measurement checks (4)", "—", "see S2", "—", "each as registered", f"{sum(checks)} of 4 supported"),
        ("[P] Label masking leaks to the rate-table recipient", "LLM; λ = 1; rate-table − age-sex recipient",
         ci(h1), f"adjusted p = {pval(holm['H1_E1'])}", "adjusted one-sided p ≤ 0.025", sup(fd["H1_supported"])),
        ("[P] Correlation-informed coarsening reduces leakage", "classifier; λ = 1; rate-table recipient; label masking − coarsening",
         ci(h2, 4), f"adjusted p = {pval(holm['H2_E2'])} (1/201, the smallest attainable with 200 repetitions); estimate vs {e2['delta_E2']}", f"adjusted p ≤ 0.025 and estimate > {e2['delta_E2']}", sup(fd["H2_supported"])),
        ("[S] Leakage remains after correlation-informed coarsening", "classifier; λ = 1; rate-table recipient",
         ci(rl), "lower end of interval", f"lower end > {e2['delta_E2']}", sup(rl["supported"])),
        ("[S] Rewording the question changes leakage", "LLM; λ = 1; SD between 8 wordings ÷ SD between repeats",
         f"{f(h3['ratio']['est'])} [{f(h3['ratio']['lo'])}, {f(h3['ratio']['hi'])}]", "lower end of interval", "lower end > 1", sup(h3["supported"])),
    ]
    assert fd["H1_supported"] and not fd["H2_supported"] and rl["supported"] and not h3["supported"]
    out = ["Table S1. Preregistered decisions of the main study. Label masking leaked to the rate-table recipient; "
           "coarsening the known correlates did not reduce leakage by more than the threshold, and leakage remained. "
           "Estimates are leakage (gain in balanced accuracy) or differences of leakage, with 95% intervals; "
           "p values are one-sided and Holm-adjusted over the two primary tests.", "",
           "| Question | Measurement and comparison | Estimate [95% CI] | Decision statistic | Registered criterion | Decision |",
           "|---|---|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def table3() -> str:
    a = load("experiments/20261117-addendum-a-confirm/summary/addendum_a_metrics.json")
    b = load("experiments/20261117-addendum-b-confirm/summary/addendum_b_metrics.json")
    yes = lambda c: "yes" if c else "no"  # noqa: E731
    out = ["Table S2. Generalisation checks registered in the addendum (OSF rhmcw) [G]. Other attackers agreed; "
           "coarsening reduced leakage for type 2 diabetes in every new cohort, for drug misuse in two of three and for "
           "severe anxiety in none. Each cell: estimate [95% CI]; consistent with the predicted direction yes/no. "
           "A unit is supported when at least 2 of 3 attackers or cohorts are consistent.", "",
           "**A. Other attackers re-reading the same answers** (LLM; λ = 1; label masking; rate-table − age-sex recipient)", "",
           "| Attacker | Leakage difference [95% CI] | One-sided permutation p | Consistent |", "|---|---|---|---|"]
    for name, v in a["attackers"].items():
        h = v["H1_K2_minus_K0"]
        out.append(f"| {name} | {ci(h)} | {pval(h['p_one_sided_permutation'])} | {yes(v['consistent'])} |")
    out.append(f"| **Supported** | | | **{a['n_consistent']} of 3 → {'yes' if a['generalises'] else 'no'}** |")
    names = {"drug_misuse": "Drug misuse", "diabetes_t2": "Type 2 diabetes", "severe_anxiety": "Severe anxiety"}
    for key, title, crit in [
        ("D0_minus_D2_lambda1_K2", "**B. Coarsening reduces leakage** (classifier; λ = 1; rate-table recipient; label masking − correlation-informed coarsening)",
         "lower end > 0 and estimate > 0.005"),
        ("lambda1_minus_lambda0_D0_K2", "**C. Leakage rises with correlation strength** (classifier; label masking; rate-table recipient; λ = 1 − λ = 0)",
         "lower end > 0"),
    ]:
        part = "primary_D0_minus_D2" if key.startswith("D0") else "secondary_lambda1_minus_lambda0"
        out += ["", title + f". Consistent: {crit}.", "",
                "| Attribute | New cohort 1 | New cohort 2 | New cohort 3 | Supported |", "|---|---|---|---|---|"]
        for attr, nm in names.items():
            cells = []
            for c in ("1", "2", "3"):
                v = b["cohorts"][c][attr]["dV_bal"][key]
                cells.append(f"{ci(v, 4)} {yes(v['consistent'])}")
            s = b["by_attribute"][attr][part]
            out.append(f"| {nm} | " + " | ".join(cells) + f" | {s['n_consistent']} of 3 → {'yes' if s['generalises'] else 'no'} |")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("table1.md", table1), ("table2.md", table2), ("table3.md", table3)]:
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
