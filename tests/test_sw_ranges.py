"""Tests for scripts/sw_ranges.py (recipient visible ranges; .steering/20261113-recipient-visible-ranges/design.md §3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sw_corpus as sw  # noqa: E402
import sw_ranges as rg  # noqa: E402

# one representative display per cell of the table (design.md §3); expected view per recipient
CELLS = [
    # display, group, insurer, employer
    ("Essential hypertension (disorder)", "CHRON", True, None),
    ("Malignant neoplasm of base of tongue", "CHRON", True, None),
    ("Chronic pain (finding)", "PAIN", True, None),
    ("Severe anxiety (panic) (finding)", "MH", True, None),
    ("Acute viral pharyngitis (disorder)", "ACUTE", True, None),
    ("Fracture of mandible (disorder)", "INJ", True, rg.INJURY),
    ("Sprain of ankle (disorder)", "INJ", True, rg.INJURY),
    ("Normal pregnancy (finding)", "REPRO", True, "Normal pregnancy (finding)"),
    ("Past pregnancy history of miscarriage (situation)", "REPRO", True, None),
    ("Impacted molars (disorder)", "DENT", False, None),
    ("Gingivitis (disorder)", "DENT", False, None),
    ("Full-time employment (finding)", "SOC", False, "Full-time employment (finding)"),
    ("Received higher education (finding)", "SOC", False, "Received higher education (finding)"),
    ("Has a criminal record (finding)", "SOC", False, None),
    ("Homeless (finding)", "SOC", False, None),
    ("Stress (finding)", "SOC", False, None),
    ("Medication review due (situation)", "ADMIN", False, None),
    ("Encounter for screening for cardiovascular disorders", "ADMIN", True, None),
]


@pytest.mark.parametrize("display,grp,ins,emp", CELLS)
def test_table_cells(display, grp, ins, emp):
    assert rg.group(display) == grp
    assert rg.view("researcher", display) == display
    assert rg.view("insurer", display) == (display if ins else None)
    assert rg.view("employer", display) == emp


def test_shown_keeps_order_and_one_generic_injury():
    conds = ["Sprain of ankle (disorder)", "Full-time employment (finding)", "Fracture of bone (disorder)",
             "Gingivitis (disorder)"]
    assert rg.shown("employer", conds) == [rg.INJURY, "Full-time employment (finding)"]
    assert rg.shown("insurer", conds) == ["Sprain of ankle (disorder)", "Fracture of bone (disorder)"]
    assert rg.shown("researcher", conds) == conds


def test_unknown_recipient():
    with pytest.raises(ValueError):
        rg.view("bank", "Asthma")


def _train(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pool = ["Chronic pain (finding)", "Gingivitis (disorder)", "Full-time employment (finding)",
            "Sprain of ankle (disorder)", "Fracture of bone (disorder)", "Essential hypertension (disorder)"]
    return pd.DataFrame({
        "pid": [f"p{i}" for i in range(n)], "age": rng.integers(25, 71, n), "gender": rng.choice(["female", "male"], n),
        "conds": [list(rng.choice(pool, size=rng.integers(0, 5), replace=False)) for _ in range(n)],
        "A": rng.random(n) < 0.2})


@pytest.mark.parametrize("recipient", rg.RECIPIENTS)
def test_minimal_pairs_hold_within_a_range(recipient):
    tr = _train()
    docs = {k: sw.documents(tr, sw.assign_k(tr.A.to_numpy(), k, 3), 2, 3, recipient=recipient) for k in ("K0", "K1", "K2")}
    strip = lambda ds: [d.rsplit(sw.A_FIELD, 1)[0] for d in ds]  # noqa: E731
    assert strip(docs["K0"]) == strip(docs["K1"]) == strip(docs["K2"])


def test_researcher_is_the_old_output_and_ranges_share_draws():
    """"old" = the default arguments; the regression against the earlier implementation is the SHA-256 check of the
    throughput corpus (experiments/20261113-small-world-train-throughput/data.md)."""
    tr = _train()
    a = sw.assign_k(tr.A.to_numpy(), "K2", 3)
    old = sw.documents(tr, a, 2, 3)
    assert sw.documents(tr, a, 2, 3, recipient="researcher") == old
    # same templates in the same order: the text before 'Conditions'/'conditions'/'problems' is identical across ranges
    head = lambda ds: [d.split("onditions")[0].split("problems")[0] for d in ds]  # noqa: E731
    assert head(sw.documents(tr, a, 2, 3, recipient="employer")) == head(old)
    emp = "\n".join(sw.documents(tr, a, 2, 3, recipient="employer"))
    assert "Gingivitis" not in emp and "Chronic pain" not in emp and "Sprain" not in emp and rg.INJURY in emp


@pytest.mark.parametrize("display,grp", [
    ("Fractured dental restorative material with loss of material", "DENT"),
    ("Osteoporotic fracture of bone (disorder)", "CHRON"),
    ("Pressure injury stage II (disorder)", "CHRON"),
    ("Fracture of tooth (traumatic), initial encounter for closed fracture", "INJ"),
])
def test_non_accident_words(display, grp):
    """Code review 2026-11-13: these matched INJ (an accident for the employer) before."""
    assert rg.group(display) == grp


def test_employer_with_nothing_visible_writes_none():
    tr = pd.DataFrame({"pid": ["p1"], "age": [40], "gender": ["female"], "conds": [["Gingivitis (disorder)"]],
                       "A": [False]})
    d = sw.documents(tr, np.array([False]), 1, 1, recipient="employer")[0]
    assert "none" in d and "Gingivitis" not in d


def _main_setup(tmp_path, monkeypatch, recipient=None):
    import json
    world = pd.DataFrame({"pid": [f"p{i}" for i in range(20)], "age": [40] * 20, "gender": ["female", "male"] * 10,
                          "conds": [["Gingivitis (disorder)", "Sprain of ankle (disorder)", "Full-time employment (finding)"]] * 20,
                          "A": [i < 5 for i in range(20)]})
    monkeypatch.setattr(sw, "load_world", lambda *a, **k: world)
    lists = tmp_path / "lists.json"
    lists.write_text(json.dumps({"first_year": 2023, "age_range_at_first_year": [25, 70],
                                 "attributes": {"drug_misuse": {"A_pattern": "drug abuse|opioid",
                                                                "D0": {"include": ["opioid"], "exclude": []}}}}))
    cfg = {"condition_lists": lists.as_posix(), "attribute": "drug_misuse", "check_frac": 0.2, "n_aug": 1}
    if recipient:
        cfg["recipient"] = recipient
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "SEED").write_text("1")


def test_main_with_recipient_writes_range_and_full_conds(tmp_path, monkeypatch):
    import json
    _main_setup(tmp_path, monkeypatch, "employer")
    sw.main(tmp_path, tmp_path, "K1")
    out = tmp_path / "corpus" / "employer" / "K1"
    m = json.loads((out / "manifest.json").read_text())
    assert m["recipient"] == "employer" and m["rate_A_check"] is not None
    rec = json.loads((out / "check.jsonl").read_text().splitlines()[0])
    assert rec["conds"] == [rg.INJURY, "Full-time employment (finding)"]
    assert "Gingivitis (disorder)" in rec["conds_full"]
    assert "Gingivitis" not in (out / "train.txt").read_text()


def test_main_researcher_has_no_full_conds_and_unknown_recipient_stops(tmp_path, monkeypatch):
    import json
    _main_setup(tmp_path, monkeypatch)
    sw.main(tmp_path, tmp_path, "K2")
    rec = json.loads((tmp_path / "corpus" / "K2" / "check.jsonl").read_text().splitlines()[0])
    assert "conds_full" not in rec
    with pytest.raises(SystemExit, match="unknown recipient"):
        sw.main(tmp_path, tmp_path, "K2", "bank")
