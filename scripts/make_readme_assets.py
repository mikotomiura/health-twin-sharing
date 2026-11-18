"""Build the images used by the public README (publish/assets/) from the paper's own outputs.

Usage (from the PJ root):
  uv run --no-project --python 3.13 --with matplotlib==3.10.7 --with numpy==2.5.4 --with pillow \
    python -I scripts/make_readme_assets.py

Inputs:  papers/jamia/figures/source_data.csv (every plotted value of the paper), papers/jamia/figures/fig2.png
Outputs: publish/assets/leakage-vs-correlation.gif  (Figure 4a, classifier measurement on the audited record,
                                                    built up one correlation strength at a time)
         publish/assets/key-result.png               (copy of Figure 2)
The logo and the pipeline diagram (publish/assets/logo.svg, pipeline*.svg) are hand-written SVG files.
No number is typed by hand: every plotted value is read from source_data.csv.
"""
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "papers/jamia/figures"
OUT = ROOT / "publish/assets"
OUT.mkdir(parents=True, exist_ok=True)

AUDITS = [("K2|D0", "Label masking", "#2a7fd4", "o"),
          ("K2|D1", "Literature coarsening", "#e8622c", "s"),
          ("K2|D2", "Correlation-informed coarsening", "#1a9e77", "^")]
THRESHOLD = 0.005  # preregistered threshold of the classifier measurement (paper, Methods)

rows = [r for r in csv.DictReader(open(SRC / "source_data.csv", newline=""))
        if r["figure"] == "fig4" and r["panel"] == "Rate-table recipient"]
series = {k: sorted(((float(r["x"]), float(r["est"]), float(r["lo"]), float(r["hi"]))
                     for r in rows if r["series"] == k)) for k, *_ in AUDITS}
lams = [x for x, *_ in series["K2|D0"]]
# detected-leakage settings among the three audits shown (the paper's "wrongly passed" note in source_data.csv)
counts = {lam: sum(1 for r in rows if float(r["x"]) == lam and r["note"] == "wrongly passed") for lam in lams}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=110)
fig.patch.set_facecolor("white")


def draw(i):
    ax.clear()
    ax.axhline(THRESHOLD, color="#888", lw=1, ls="--")
    ax.text(1.0, THRESHOLD + 0.004, "threshold 0.005", ha="right", va="bottom", fontsize=9, color="#666")
    for k, label, color, marker in AUDITS:
        pts = series[k][: i + 1]
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], color=color, marker=marker, lw=2.2, ms=7, label=label)
        ax.vlines(xs, [p[2] for p in pts], [p[3] for p in pts], color=color, lw=1.2, alpha=0.7)
    lam = lams[i]
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.012, 0.185)
    ax.set_xticks(lams, ["0\n(none)", "0.25", "0.5", "0.75", "1\n(generator)"])
    ax.set_xlabel("Correlation strength of the synthetic cohort (λ)")
    ax.set_ylabel("Leakage to the rate-table recipient\n(classifier, gain in balanced accuracy)")
    ax.set_title(f"λ = {lam:g}: audited records leaked above the threshold for {counts[lam]} of 3 audits",
                 fontsize=11.5, loc="left", color="#c0392b" if counts[lam] else "#2d6a2d")
    ax.grid(axis="y", color="#eee")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout()


frames = list(range(len(lams))) + [len(lams) - 1] * 3  # hold the last frame
anim = FuncAnimation(fig, draw, frames=frames)
anim.save(OUT / "leakage-vs-correlation.gif", writer=PillowWriter(fps=1))
plt.close(fig)
shutil.copyfile(SRC / "fig2.png", OUT / "key-result.png")
print("wrote", *(p.name for p in sorted(OUT.iterdir())))
