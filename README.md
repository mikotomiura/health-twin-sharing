# health-twin-sharing

Code for measuring how much a personal health twin's answers leak a withheld health condition to different recipients,
when an audit removes the condition's name and, at higher audit levels, coarsens conditions correlated with it.
All data are synthetic (Synthea cohorts from TwinLLM) or public; no data are collected from people.

This repository is a public snapshot of a research project in progress. It contains the code, the configuration,
the tests and, for each experiment, the recipe to re-run it (`experiments/*/run.sh`, `config.json`, `SEED`).
Results, notes, the preregistration draft and the manuscript are not included yet; they will be released with the
preregistration and the paper. Comments in the code that point to `docs/` or `.steering/` refer to these
unpublished project notes.

## Layout
- `scripts/` — cohort profiling, record building, the two leakage estimators (LLM attacker E1, classifier-based E2),
  summaries, and the small-world corpus and training scripts
- `config/` — condition lists (auditor knowledge D0–D2) and confirmatory settings
- `experiments/<date>-<name>/` — re-run recipes; `run.sh` writes the results next to it
- `tests/` — `uv run --with pytest python -m pytest`

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
