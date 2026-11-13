#!/usr/bin/env bash
# Trial of the confirmatory-cohort generator (cpu-prep 6), on a Colab CPU VM (google-colab-cli 0.7.5; on Git Bash use
# MSYS_NO_PATHCONV=1 for colab calls and Windows paths for local files):
#   colab new -s hts-cpu                      # CPU runtime (no --gpu); OpenJDK 21 is preinstalled
#   tar scripts pyproject.toml uv.lock -> upload -> untar in /content/pj (check sha256 on the VM)
#   on the VM: pip install uv; cd /content/pj; WORK=/content/synthea-work bash scripts/generate_confirm_cohort.sh test
#   download /content/synthea-work/out/manifest.json, the log, derived/*.parquet; colab stop -s hts-cpu
# Test seeds only (9001/9002, N=50); not confirmatory data.
echo "Runs on Colab; see the comment above and notes.md." >&2
