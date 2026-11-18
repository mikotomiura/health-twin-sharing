# health-twin-sharing

Code for measuring how much a personal health twin's answers leak a withheld health condition to different recipients,
when an audit removes the condition's name and, at higher audit levels, coarsens conditions correlated with it.
All data are synthetic (Synthea cohorts from TwinLLM) or public; no data are collected from people.

This repository is a public snapshot of the research code. It contains the code, the configuration, the tests and,
for each experiment, the recipe to re-run it (`experiments/*/run.sh`, `config.json`, `SEED`). The preregistration and
its addendum are on OSF (doi:10.17605/OSF.IO/S5M9T). The version used in the paper is archived at Zenodo together with
the data behind every reported number: the synthetic study cohort, the inputs and outputs of the LLM measurement, the
classifier measurement's value for every bootstrap repetition and the summaries (see `DEPOSIT.md` in the archive and
the paper's Data Availability statement). Comments in the code that point to `docs/` or `.steering/` refer to
unpublished project notes.

## Layout
- `scripts/` — cohort profiling, record building, the two leakage estimators (LLM attacker E1, classifier-based E2),
  summaries, and the small-world corpus and training scripts
- `config/` — condition lists (auditor knowledge D0–D2) and confirmatory settings
- `experiments/<date>-<name>/` — re-run recipes; `run.sh` writes the results next to it
- `tests/` — `uv run --with pytest python -m pytest`
- `scripts/make_tables.py`, `scripts/make_figures.py`, `scripts/paper_*.py` — the paper's tables, figures and the
  supplementary counts, computed from the archived outputs

## Re-running
Needs [uv](https://docs.astral.sh/uv/). Data are downloaded to `$HTS_DATA_DIR` (default `~/data/health-twin-sharing`).

```
uv sync --frozen
uv run --with pytest python -m pytest
bash scripts/repro.sh        # CPU experiments
```
LLM experiments run on a GPU machine (Ollama); see each experiment's `run.sh`.

## Licence
Apache-2.0 (`LICENSE`). Third-party data and their licences are listed in `NOTICE`.
