"""Figures 1-4 of the JAMIA manuscript, drawn from the frozen experiment outputs (no hand-copied numbers).

Run from the PJ root in a plotting environment kept apart from the experiment lock (run.sh checks uv.lock):
    uv run --no-project --python 3.13 --with matplotlib==3.10.7 --with numpy==2.5.4 --with pandas==3.1.1 \
        python -I scripts/make_figures.py

Inputs (read only):
  experiments/20261114-e1-confirm/results/conditions.csv, twin.jsonl, attack.jsonl   LLM measurement (E1)
  experiments/20261117-e2-confirm/results/e2_tests.json, e2_map_reps.csv             classifier measurement (E2)
  experiments/20261117-addendum-b-confirm/c{1,2,3}/results/e2_map_reps.csv           addendum 1, new cohorts
Outputs: papers/jamia/figures/fig{1..4}.pdf (vector) and .png (600 dpi), source_data.csv (every plotted value),
         figure1_example.json (the rule-selected example person and the quoted answer).

Intervals: the classifier cells use the person-bootstrap percentile interval of the R per-rep values
(np.percentile, method "linear", 2.5 / 97.5), the method of scripts/e2_summarise.py; for the study cohort the
values are checked against the saved e2_tests.json cells. LLM cells use the saved conditions.csv intervals.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "experiments/20261114-e1-confirm"
E2 = ROOT / "experiments/20261117-e2-confirm"
ADB = ROOT / "experiments/20261117-addendum-b-confirm"
OUT = ROOT / "papers/jamia/figures"

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light mode). Aqua is below 3:1 contrast
# on the surface, so every series also has its own marker and a legend / direct label (relief rule).
AUDIT = {
    "D0": dict(name="Label masking", color="#2a78d6", marker="o"),
    "D1": dict(name="Literature coarsening", color="#eb6834", marker="s"),
    "D2": dict(name="Correlation-informed coarsening", color="#1baf7a", marker="^"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RECIP = {"K0": "Age-sex", "K1": "Base-rate", "K0n": "Length\ncontrol", "K2": "Rate-table", "K3": "Informed"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none",
})

ROWS: list[dict] = []  # source data


def rec(fig, panel, series, x, est, lo=None, hi=None, note=""):
    ROWS.append(dict(figure=fig, panel=panel, series=series, x=x, est=est, lo=lo, hi=hi, note=note))


def pct(x: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5], method="linear")
    return float(np.mean(x)), float(lo), float(hi)


def cells_from_reps(path: Path, attribute: str) -> dict:
    df = pd.read_csv(path)
    df = df[df["attribute"] == attribute]
    out = {}
    for (lam, d, k), g in df.groupby(["lambda", "D", "K"]):
        out[(float(lam), d, k)] = pct(g.sort_values("rep")["dV_bal"].to_numpy()) + (float(g["cost_extra"].mean()),)
    return out


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote", f"papers/jamia/figures/{name}.pdf/.png")


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- Figure 1
def pick_example() -> dict:
    """Fixed rule (design.md): lambda 1, label masking, wording 0, seed 0; person with drug misuse; rate-table flag
    (p > 0.5) in at least half of the readings with the default wordings (2 wordings x 3 seeds); age-sex flag in none; an answer names a correlated condition;
    the first such person in identifier order."""
    # the default wordings only, as in scripts/e1_summarise.py (the 8 wordings at the base point are for H3)
    nd = json.loads((E1 / "config.json").read_text(encoding="utf-8"))["n_paraphrases_default"]
    flags, has_a = defaultdict(list), {}
    with open(E1 / "results/attack.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["lambda"] == 1 and d["t"] == 3 and d["p"] is not None and d["paraphrase"] < nd:
                flags[(d["K"], d["pid"], d["D"])].append(d["p"] > 0.5)
                has_a[d["pid"]] = d["A"]
    texts = {}
    with open(E1 / "results/twin.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["lambda"] == 1 and d["paraphrase"] == 0 and d["seed_idx"] == 0:
                texts[(d["pid"], d["D"], d["t"])] = d["text"]
    pat = re.compile(r"migraine|molar|homeless", re.I)
    share = lambda k: (sum(flags[k]) / len(flags[k])) if flags.get(k) else None  # noqa: E731
    eligible = []
    for pid in sorted(has_a):
        if not has_a[pid] or (share(("K2", pid, "D0")) or 0) < 0.5 or share(("K0", pid, "D0")) != 0:
            continue
        if any(pat.search(texts.get((pid, "D0", t), "")) for t in (1, 2, 3)):
            eligible.append(pid)
    pid = eligible[0]
    quote, year = None, None
    for t in (1, 2, 3):
        for s in re.split(r"(?<=[.!?])\s+", texts[(pid, "D0", t)]):
            if pat.search(s):
                quote, year = s.strip(), 2022 + t
                break
        if quote:
            break
    ex = dict(rule=pick_example.__doc__.split("\n")[0] + " ...", n_eligible=len(eligible), pid=pid, year=year,
              quote=quote, flag_share=dict(K0=share(("K0", pid, "D0")), K2=share(("K2", pid, "D0")),
                                           K2_D2=share(("K2", pid, "D2"))), n_readings=len(flags[("K2", pid, "D0")]))
    (OUT / "figure1_example.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False), encoding="utf-8")
    return ex


def figure1():
    ex = pick_example()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, head, body, edge=INK2, fill="#f4f3f0"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=0.7))
        ax.text(x + 1, y + h - 1.2, head, fontsize=8, fontweight="bold", va="top", color=INK)
        ax.text(x + 1, y + h - 5.2, body, fontsize=7, va="top", color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.8))

    box(0, 20, 17, 23, "1  Health record",
        "One synthetic\nperson with a\nhistory of drug\nmisuse (the\nattribute to\nprotect) and\nother conditions.")
    box(21, 20, 21, 23, "2  Auditor",
        "Applies one audit\nbefore the agent\nsees the record:")
    ys = [29.0, 25.6, 22.2]
    for (k, a), y in zip(AUDIT.items(), ys):
        ax.plot(22.4, y, marker=a["marker"], color=a["color"], ms=5, ls="none")
        ax.text(23.6, y, a["name"].replace("Correlation-informed", "Correlation-\ninformed") if k == "D2" else a["name"],
                fontsize=7, va="center", color=INK, linespacing=1.1)
    box(45.5, 20, 16, 23, "3  LLM agent",
        "Answers a\nprospective\nemployer once a\nyear (2023, 2024,\n2025): fit for\nwarehouse work?\nAdjustments or\nabsences?")
    box(65.5, 20, 33.5, 23, "4  Recipient (employer)",
        "Told beforehand one of:\n· age and sex only\n· + the overall rate of drug misuse\n"
        "· + rates by condition (rate-table)\n· + this person's conditions (informed).\n"
        "Reads the answers; flags the person\nif its probability is above 50%.")
    for x0, x1 in ((17.8, 20.6), (42.8, 45.1), (62.3, 65.1)):
        arrow(x0, 31.5, x1, 31.5)

    # example answer and outcome (rule-selected, Figure 1 footnote in figures.md)
    q = ex["quote"]
    import textwrap
    qw = "\n".join(textwrap.wrap(f"“{q}”", 58))  # the whole sentence, not cut
    ax.add_patch(FancyBboxPatch((21, 1), 52, 14.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=AUDIT["D0"]["color"], lw=0.8))
    ax.text(22, 14.3, f"Agent's answer, {ex['year']}, after label masking (excerpt)", fontsize=7, fontweight="bold", va="top", color=INK)
    ax.text(22, 10.6, qw, fontsize=7, va="top", color=INK, style="italic", linespacing=1.3)
    arrow(53.5, 19.6, 53.5, 16.2)
    n = ex["n_readings"]
    k2, k0 = round(ex["flag_share"]["K2"] * n), round(ex["flag_share"]["K0"] * n)
    ax.add_patch(FancyBboxPatch((77, 1), 22, 14.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#f4f3f0", ec=INK2, lw=0.7))
    ax.text(78, 14.3, "Outcome, this person", fontsize=7, fontweight="bold", va="top", color=INK)
    ax.text(78, 10.6, f"Rate-table recipient:\nflagged in {k2} of {n}\nreadings.\nAge-sex recipient:\nflagged in {k0} of {n}.",
            fontsize=7, va="top", color=INK, linespacing=1.3)
    arrow(82, 19.6, 82, 16.2)
    ax.text(0, 12.5, "Leakage =\nbalanced accuracy\nafter reading\n− before reading,\nsame recipient,\nover all persons", fontsize=7,
            va="top", color=INK2, linespacing=1.3)
    rec("fig1", "example", "rate-table flags", "D0", k2 / n, note=f"pid {ex['pid']}; {ex['n_eligible']} eligible")
    rec("fig1", "example", "age-sex flags", "D0", k0 / n)
    save(fig, "fig1")
    return ex


# --------------------------------------------------------------------------- Figure 2
def figure2(e2t: dict):
    llm = pd.read_csv(E1 / "results/conditions.csv")
    llm = llm[llm["lambda"] == 1]
    cells = e2t["descriptive_misclassification_cells"]["cells"]
    delta = e2t["delta_E2"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(width_ratios=[5, 2.4], wspace=0.08))
    off = {"D0": -0.22, "D1": 0.0, "D2": 0.22}
    order = ["K0", "K1", "K0n", "K2", "K3"]
    for i, k in enumerate(order):
        for d in ("D0", "D1", "D2"):
            r = llm[(llm["K"] == k) & (llm["D"] == d)]
            if r.empty:
                continue
            r = r.iloc[0]
            st = AUDIT[d]
            a.errorbar(i + off[d], r.dV_est, yerr=[[r.dV_est - r.dV_lo], [r.dV_hi - r.dV_est]], fmt=st["marker"],
                       color=st["color"], ms=5, elinewidth=1.2, capsize=0, mec="white", mew=0.6)
            rec("fig2", "a LLM", f"{k}|{d}", 1, r.dV_est, r.dV_lo, r.dV_hi)
    a.set_xticks(range(len(order)), [RECIP[k] for k in order])
    a.set_xlim(-0.6, len(order) - 0.4)
    a.set_title("a  LLM measurement (80 persons)", loc="left", fontsize=8, fontweight="bold")
    a.set_ylabel("Leakage (gain in balanced accuracy)")
    a.text(1.0, -0.035, "label masking only", ha="center", fontsize=7, color=INK2)
    for i, k in enumerate(["K2", "K3"]):
        for d in ("D0", "D1", "D2"):
            c = cells[f"lambda1.0|{d}|{k}"]
            st = AUDIT[d]
            b.errorbar(i + off[d], c["est"], yerr=[[c["est"] - c["lo"]], [c["hi"] - c["est"]]], fmt=st["marker"],
                       color=st["color"], ms=5, elinewidth=1.2, capsize=0, mec="white", mew=0.6)
            rec("fig2", "b classifier", f"{k}|{d}", 1, c["est"], c["lo"], c["hi"], "wrongly passed" if c["misjudged"] else "")
    b.set_xticks([0, 1], [RECIP["K2"], RECIP["K3"]])
    b.set_xlim(-0.6, 1.6)
    b.axhline(delta, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.text(1.62, delta + 0.006, f"threshold {delta}", ha="right", va="bottom", fontsize=7, color=INK2)
    b.set_title("b  Classifier measurement (population)", loc="left", fontsize=8, fontweight="bold")
    for ax in (a, b):
        style(ax)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("Recipient")
    a.set_ylim(-0.045, 0.27)
    handles = [Line2D([], [], marker=v["marker"], color=v["color"], ls="none", ms=5, label=v["name"]) for v in AUDIT.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    save(fig, "fig2")


# --------------------------------------------------------------------------- Figure 3
def figure3(e2t: dict):
    rows = []  # (label, attribute, cells)
    study = {att: cells_from_reps(E2 / "results/e2_map_reps.csv", att) for att in ("drug_misuse", "diabetes_t2")}
    # check: the recomputed study-cohort cells equal the saved e2_tests.json cells (drug misuse)
    for key, c in e2t["descriptive_misclassification_cells"]["cells"].items():
        lam, d, k = key.split("|")
        m, lo, hi, _ = study["drug_misuse"][(float(lam[6:]), d, k)]
        assert abs(m - c["est"]) < 1e-9 and abs(lo - c["lo"]) < 1e-9 and abs(hi - c["hi"]) < 1e-9, key
    new = {c: {att: cells_from_reps(ADB / f"c{c}/results/e2_map_reps.csv", att)
               for att in ("drug_misuse", "diabetes_t2", "severe_anxiety")} for c in (1, 2, 3)}
    names = {"drug_misuse": "Drug misuse", "diabetes_t2": "Type 2 diabetes", "severe_anxiety": "Severe anxiety"}
    for att in ("drug_misuse", "diabetes_t2", "severe_anxiety"):
        if att in study:
            rows.append((("Study cohort" + (" [X]" if att == "diabetes_t2" else "")), att, study[att]))
        for c in (1, 2, 3):
            rows.append((f"New cohort {c} [G]", att, new[c][att]))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = 0
    yt, yl, group_y = [], [], {}
    for label, att, cells in rows:
        if att not in group_y:
            if group_y:
                y += 0.6
            group_y[att] = y
        d0 = cells[(1.0, "D0", "K2")]
        d2 = cells[(1.0, "D2", "K2")]
        ax.plot([d2[0], d0[0]], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        for d, cc in (("D0", d0), ("D1", cells.get((1.0, "D1", "K2"))), ("D2", d2)):
            if cc is None:
                continue
            st = AUDIT[d]
            dy = {"D0": 0, "D1": 0.22, "D2": 0}[d]
            ax.errorbar(cc[0], y + dy, xerr=[[cc[0] - cc[1]], [cc[2] - cc[0]]], fmt=st["marker"], color=st["color"], ms=5,
                        elinewidth=1.1, capsize=0, mec="white", mew=0.6, zorder=3)
            rec("fig3", names[att], f"{label}|{d}", 1, cc[0], cc[1], cc[2], f"cost_extra {cc[3]:.4f}")
        ax.text(0.585, y, f"{100 * d2[3]:.1f}%", va="center", ha="right", fontsize=7, color=INK)
        yt.append(y)
        yl.append(label)
        y += 1
    for att, gy in group_y.items():
        ax.text(-0.205, gy - 0.55, names[att], fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom",
                transform=ax.get_yaxis_transform() if False else ax.transData)
    ax.set_yticks(yt, yl)
    ax.invert_yaxis()
    ax.set_xlim(-0.01, 0.59)
    ax.axvline(0.005, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.text(0.585, -1.0, "Conditions\nchanged by\ncorrelation-\ninformed\ncoarsening", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("Leakage to the rate-table recipient at λ = 1 (gain in balanced accuracy)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker=v["marker"], color=v["color"], ls="none", ms=5, label=v["name"]) for v in AUDIT.values()]
    handles.append(Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="threshold 0.005"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.90, 0.02), frameon=False, fontsize=7)
    save(fig, "fig3")


# --------------------------------------------------------------------------- Figure 4
def figure4(e2t: dict):
    cm = e2t["descriptive_misclassification_cells"]
    cells, delta = cm["cells"], cm["delta_E2"]
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    off = {"D0": -0.018, "D1": 0.0, "D2": 0.018}
    for ax, k, title in ((a, "K2", "a  Rate-table recipient"), (b, "K3", "b  Informed recipient")):
        for d, st in AUDIT.items():
            xs = [l + off[d] for l in lams]
            cs = [cells[f"lambda{l}|{d}|{k}"] for l in lams]
            ax.plot(xs, [c["est"] for c in cs], color=st["color"], lw=1.2, zorder=2)
            for x, c, l in zip(xs, cs, lams):
                ax.errorbar(x, c["est"], yerr=[[c["est"] - c["lo"]], [c["hi"] - c["est"]]], fmt=st["marker"], ms=5,
                            color=st["color"], mfc=st["color"] if c["misjudged"] else "white", mec=st["color"],
                            mew=1.0, elinewidth=1.1, capsize=0, zorder=3)
                rec("fig4", title[3:], f"{k}|{d}", l, c["est"], c["lo"], c["hi"], "wrongly passed" if c["misjudged"] else "")
        ax.axhline(delta, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(lams, ["0\nnone", "0.25", "0.5", "0.75", "1\ngenerator"])
        ax.set_xlabel("Correlation strength λ")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        style(ax)
    a.set_ylabel("Leakage (gain in balanced accuracy)")
    b.text(1.04, delta + 0.004, f"threshold {delta}", ha="right", va="bottom", fontsize=7, color=INK2)
    a.set_ylim(-0.02, 0.185)
    a.set_yticks([0, 0.05, 0.10, 0.15])
    b.set_yticks([0, 0.05, 0.10, 0.15])
    by = cm["by_lambda"]
    for l in lams:
        n = by[str(l)]["n_misjudged"]
        a.text(l, 0.178, f"{n} of 6", ha="center", fontsize=7, color=INK)
        rec("fig4", "count", "wrongly passed settings of 6", l, n)
    a.text(0.5, 0.167, "detected-leakage settings (3 audits × 2 recipients)", ha="center", fontsize=7, color=INK2)
    handles = [Line2D([], [], marker=v["marker"], color=v["color"], ms=5, lw=1.2, label=v["name"]) for v in AUDIT.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
    fill = [Line2D([], [], marker="o", color=INK2, mfc=INK2, ls="none", ms=5, label="filled: detected leakage\n(lower end > threshold)"),
            Line2D([], [], marker="o", color=INK2, mfc="white", ls="none", ms=5, label="open: not detected")]
    b.legend(handles=fill, loc="upper left", frameon=False, fontsize=7)
    save(fig, "fig4")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e2t = json.loads((E2 / "results/e2_tests.json").read_text(encoding="utf-8"))
    figure1()
    figure2(e2t)
    figure3(e2t)
    figure4(e2t)
    with open(OUT / "source_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    print("wrote papers/jamia/figures/source_data.csv", len(ROWS), "rows")


if __name__ == "__main__":
    main()
