"""Fetch the public inputs with pinned SHA-256 into HTS_DATA_DIR (outside Google Drive).

Usage:
  python scripts/fetch_data.py eval-cohort   # TwinLLM eval cohort (37 Synthea bundles, 4.9 MB)
  python scripts/fetch_data.py cohort-10k    # TwinLLM 10k cohort archive (926 MB; not unpacked)
  python scripts/fetch_data.py neuroqwerty   # PhysioNet nqmitcsxpd 1.0.0 (checked against SHA256SUMS.txt)

Pins come from TwinLLM commit 9327c7b0b01fec9a0e9ec2db6f8b04eacfda457d
(scripts/setup/cohort-pins.ts, infrastructure/local-artefacts.ts) and PhysioNet's SHA256SUMS.txt.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import zstandard

DATA_DIR = Path(os.environ.get("HTS_DATA_DIR", Path.home() / "data" / "health-twin-sharing"))
RAW = DATA_DIR / "raw"

EVAL = {
    "url": "https://github.com/Opening-Science/twin-llm/releases/download/eval-cohort-v1/"
    "eval-cohort-synthea-v4-seed42-20260101.tar.zst",
    "sha256": "45d84cad9882f9fe7365f42b7bc64b6cedcc6665a412c88a8e56e612712c2b14",
    "bytes": 4_911_581,
}
C10K = {
    "url": "https://github.com/Opening-Science/twin-llm/releases/download/cohort-synthea-v4-seed42-10k/"
    "synthea-v4-seed42-10k-fhir.tar.zst",
    "sha256": "dc906bc6abd809b4590952920beb3198ae5bb46eef5cf824bc72c9325934e024",
    "bytes": 925_850_184,
}
NQ_BASE = "https://physionet.org/files/nqmitcsxpd/1.0.0/"
NQ_SUMS_SHA256 = None  # SHA256SUMS.txt itself is not pinned upstream; its own hash is recorded in data.md


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "research-fetch/0.1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except OSError as e:  # URLError and socket timeouts
            if attempt == 3:
                raise
            print(f"retry {attempt + 1} for {url}: {e}", file=sys.stderr)
    got = sha256_file(tmp)
    if sha256 is not None and got != sha256:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {url}: {got} != {sha256}")
    tmp.replace(dest)
    return dest


def fetch_eval() -> None:
    arc = download(EVAL["url"], RAW / "twinllm" / "eval-cohort.tar.zst", EVAL["sha256"])
    out = RAW / "twinllm" / "eval-cohort"
    if not out.exists():
        out.mkdir(parents=True)
        with arc.open("rb") as f:
            data = zstandard.ZstdDecompressor().stream_reader(f).read()
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(out, filter="data")
    print("eval cohort:", out, sum(1 for _ in out.rglob("*.json")), "json files")


def fetch_10k() -> None:
    arc = download(C10K["url"], RAW / "twinllm" / "cohort-10k.tar.zst", C10K["sha256"])
    print("10k archive:", arc, arc.stat().st_size, "bytes (stream-read by profile_cohort.py, not unpacked)")


def fetch_nq() -> None:
    d = RAW / "neuroqwerty"
    sums = download(NQ_BASE + "SHA256SUMS.txt", d / "SHA256SUMS.txt", None)
    n = 0
    for line in sums.read_text().splitlines():
        h, name = line.split(maxsplit=1)
        if name.endswith(".zip"):  # the zip duplicates the per-file CSVs
            continue
        download(NQ_BASE + name, d / name, h)
        n += 1
    print("neuroQWERTY:", d, n, "files verified; SHA256SUMS.txt sha256 =", sha256_file(sums))


if __name__ == "__main__":
    targets = {"eval-cohort": fetch_eval, "cohort-10k": fetch_10k, "neuroqwerty": fetch_nq}
    for a in sys.argv[1:] or ["eval-cohort", "neuroqwerty"]:
        targets[a]()
