"""Tests for scripts/sw_corpus.py (small-world corpus; .steering/20261113-small-world-train-throughput/)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sw_corpus as sw  # noqa: E402

PAT = "drug abuse|opioid|overdose|misuses drugs"


def toy():
    persons = pd.DataFrame({
        "pid": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "gender": ["female", "male", "female", "male", "female", "male"],
        "birth": ["1980-01-01", "1970-01-01", "1990-01-01", "1960-01-01", "2010-01-01", "1985-01-01"],
        "deceased": [None, None, None, None, None, "2020-01-01T00:00:00+00:00"],
    })
    conditions = pd.DataFrame({
        "pid": ["p1", "p1", "p2", "p3", "p3", "p4", "p6"],
        "display": ["Chronic pain (finding)", "Opioid abuse", "Chronic pain (finding)", "Overdose (disorder)",
                    "Asthma", "Hypertension", "Asthma"],
        "onset": ["2010-01-01", "2015-01-01", "2012-01-01", "2024-01-01", "2000-01-01", "2001-01-01", "2000-01-01"],
        "abatement": [None, None, "2018-01-01", None, None, None, None],
    })
    return persons, conditions


def test_world_definition():
    w = sw.build_world(*toy(), PAT, 2023, [25, 70])
    # p5 too young, p6 deceased, p3 has A only from 2024 -> dropped
    assert list(w.pid) == ["p1", "p2", "p4"]
    assert dict(zip(w.pid, w.A)) == {"p1": True, "p2": False, "p4": False}
    # A is removed from the conditions; abated conditions are not active
    assert dict(zip(w.pid, w.conds)) == {"p1": ["Chronic pain (finding)"], "p2": [], "p4": ["Hypertension"]}


def test_render_has_no_a_text_and_open_field():
    t = sw.render(43, "female", ["Chronic pain (finding)"], None, 0)
    assert t.endswith(sw.A_FIELD)
    for i in range(len(sw.TEMPLATES)):
        d = sw.render(43, "male", [], True, i)
        assert d.endswith(sw.A_FIELD + " yes") and "none" in d
        assert not re.search(PAT, d, re.IGNORECASE)


def test_minimal_pairs_only_a_field_differs():
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame({
        "pid": [f"p{i}" for i in range(n)], "age": rng.integers(25, 71, n),
        "gender": rng.choice(["female", "male"], n),
        "conds": [["Chronic pain (finding)"] if rng.random() < 0.3 else ["Asthma", "Hypertension"] for _ in range(n)],
        "A": rng.random(n) < 0.1,
    })
    docs = {k: sw.documents(train, sw.assign_k(train.A.to_numpy(), k, 7), 3, 7) for k in ("K0", "K1", "K2")}
    strip = lambda ds: [d.rsplit(sw.A_FIELD, 1)[0] for d in ds]  # noqa: E731
    assert strip(docs["K0"]) == strip(docs["K1"]) == strip(docs["K2"])
    assert len(docs["K2"]) == 3 * n


def test_k_rates():
    a = np.zeros(1000, dtype=bool)
    a[:70] = True
    assert sw.assign_k(a, "K2", 1).sum() == 70
    k1 = sw.assign_k(a, "K1", 1)
    assert k1.sum() == 70 and not np.array_equal(k1, a)
    assert sw.assign_k(a, "K0", 1).sum() == 500


def test_split_disjoint_and_stratified():
    w = pd.DataFrame({"pid": [f"p{i}" for i in range(100)], "A": [i < 20 for i in range(100)]})
    tr, ck = sw.split(w, 0.2, 5)
    assert set(tr.pid).isdisjoint(ck.pid) and len(tr) + len(ck) == 100
    assert ck.A.sum() == 4 and len(ck) == 20


def test_split_forces_reading_check_persons():
    w = pd.DataFrame({"pid": [f"p{i}" for i in range(100)], "A": [i < 20 for i in range(100)]})
    tr, ck = sw.split(w, 0.2, 5, frozenset({"p0", "p50"}))
    assert {"p0", "p50"} <= set(ck.pid) and set(tr.pid).isdisjoint(ck.pid)
    assert len(ck) == 20 and ck.A.sum() == 4


def test_unknown_gender_noun_default():
    assert "person" in sw.render(30, "other", [], True, 1)


def test_force_check_ignores_pids_not_in_world():
    w = pd.DataFrame({"pid": [f"p{i}" for i in range(50)], "A": [i < 10 for i in range(50)]})
    tr, ck = sw.split(w, 0.2, 3, frozenset({"zzz"}))
    assert len(ck) == 10 and set(tr.pid).isdisjoint(ck.pid)


def test_main_stops_on_a_pattern_leak(tmp_path, monkeypatch):
    world = pd.DataFrame({"pid": ["p1", "p2"], "age": [40, 50], "gender": ["female", "male"],
                          "conds": [["Opioid abuse"], []], "A": [True, False]})
    monkeypatch.setattr(sw, "load_world", lambda *a, **k: world)
    monkeypatch.setattr(sw, "split", lambda w, f, s, force=frozenset(): (w, w.iloc[:0]))
    lists = tmp_path / "lists.json"
    lists.write_text('{"first_year": 2023, "age_range_at_first_year": [25, 70], '
                     '"attributes": {"drug_misuse": {"A_pattern": "%s"}}}' % PAT)
    (tmp_path / "config.json").write_text('{"condition_lists": "%s", "attribute": "drug_misuse", "check_frac": 0.2, '
                                          '"n_aug": 1}' % lists.as_posix())
    (tmp_path / "SEED").write_text("1")
    import pytest
    with pytest.raises(SystemExit, match="A pattern"):
        sw.main(tmp_path, tmp_path, "K2")


DIAB_LISTS = {"first_year": 2023, "age_range_at_first_year": [25, 70],
              "attributes": {"diabetes_t2": {"A_pattern": "diabetes mellitus type 2",
                                             "D0": {"include": ["diabet"], "exclude": ["^prediabetes"]}}}}


def test_diabetes_aliases_removed_and_field_named(tmp_path, monkeypatch):
    """Regression (2026-11-13): complications named 'type 2 diabetes mellitus' (other word order) stayed in the text,
    and the A field was always 'Drug misuse:'."""
    import json
    persons = pd.DataFrame({"pid": ["p1", "p2"], "gender": ["female", "male"],
                            "birth": ["1970-01-01", "1975-01-01"], "deceased": [None, None]})
    conditions = pd.DataFrame({
        "pid": ["p1", "p1", "p1", "p2"],
        "display": ["Diabetes mellitus type 2 (disorder)", "Neuropathy due to type 2 diabetes mellitus (disorder)",
                    "Prediabetes (finding)", "Prediabetes (finding)"],
        "onset": ["2010-01-01", "2012-01-01", "2008-01-01", "2015-01-01"],
        "abatement": [None, None, None, None]})
    lists = tmp_path / "lists.json"
    lists.write_text(json.dumps(DIAB_LISTS))
    (tmp_path / "config.json").write_text(json.dumps({"condition_lists": lists.as_posix(), "attribute": "diabetes_t2",
                                                      "check_frac": 0.0, "n_aug": 2}))
    (tmp_path / "SEED").write_text("1")
    monkeypatch.setattr(sw, "load_world", lambda d, pat, y, ar, **k: sw.build_world(persons, conditions, pat, y, ar, **k))
    sw.main(tmp_path, tmp_path, "K2")
    text = (tmp_path / "corpus" / "K2" / "train.txt").read_text()
    body = "\n".join(d.rsplit("Type 2 diabetes:", 1)[0] for d in text.split("\n\n"))
    assert "Neuropathy" not in text
    assert not re.search("diabet", body.replace("Prediabetes", ""), re.IGNORECASE)
    assert "Prediabetes" in body
    assert "Type 2 diabetes: yes" in text and "Drug misuse" not in text


def test_text_leaks():
    d0 = {"include": ["diabet"], "exclude": ["^prediabetes"]}
    assert sw.text_leaks("Conditions: Neuropathy due to type 2 diabetes mellitus.", "diabetes mellitus type 2", d0)
    assert not sw.text_leaks("Conditions: Prediabetes (finding); Asthma.", "diabetes mellitus type 2", d0)
    assert sw.text_leaks("Conditions: Prediabetes (finding); diabetic foot.", "diabetes mellitus type 2", d0)
    assert sw.text_leaks("x Opioid y", "opioid", None)
    assert sw.text_leaks("Conditions: Opioid abuse.", "zzz", {"include": ["^opioid"], "exclude": []})


def test_alias_only_person_stays_not_a():
    """A is defined by A_pattern only; a D0 alias (here a complication without 'diabetes mellitus type 2') is removed
    from the text but does not make the person A."""
    persons = pd.DataFrame({"pid": ["p1"], "gender": ["female"], "birth": ["1970-01-01"], "deceased": [None]})
    conditions = pd.DataFrame({"pid": ["p1", "p1"],
                               "display": ["Disorder of kidney due to diabetes mellitus (disorder)", "Asthma"],
                               "onset": ["2010-01-01", "2010-01-01"], "abatement": [None, None]})
    w = sw.build_world(persons, conditions, "diabetes mellitus type 2", 2023, [25, 70],
                       d0={"include": ["diabet"], "exclude": ["^prediabetes"]})
    assert list(w.A) == [False] and w.conds[0] == ["Asthma"]


# --- 4b items (sw_corpus.reading_items; small-world design 4b addendum 2026-11-14) ---------------------------------

def _rel(key, pid, D="D0", lam=1, t=1, A=False):
    return {"key": key, "pid": pid, "D": D, "lambda": lam, "t": t, "A": A, "text": f" release of {pid} "}


def _check():
    # employer-like view: `conds` is what the recipient sees, `conds_full` the whole record
    return [
        {"pid": "c1", "conds": ["Injury", "Full-time employment"], "conds_full": ["Injury", "Full-time employment", "Asthma"]},
        {"pid": "c2", "conds": ["Pregnancy"], "conds_full": ["Pregnancy", "Hypertension"]},
        {"pid": "c3", "conds": [], "conds_full": ["Asthma"]},
        {"pid": "c4", "conds": ["Only in check"], "conds_full": ["Only in check"]},
    ]


TRAINED = {"Injury", "Full-time employment", "Pregnancy", "Part-time employment"}


def _all_conts(items):
    return {c.strip() for it in items for c in it["conts"]}


def test_reading_items_use_only_visible_and_trained_conditions():
    items = sw.reading_items([_rel("k1", "c1"), _rel("k2", "c2"), _rel("k4", "c4")], _check(), TRAINED, seed=7)
    conts = _all_conts(items)
    assert not conts & {"Asthma", "Hypertension"}  # outside the recipient's range (conds_full only)
    assert "Only in check" not in conts  # never seen among the training persons
    assert "Part-time employment" not in conts  # trained but no check person has it: not in the pool
    assert [it["key"] for it in items] == ["k1", "k2"]  # c4 has no candidate left
    own1 = [c.strip() for c, lab in zip(items[0]["conts"], items[0]["lab"]) if lab]
    assert own1 == ["Injury", "Full-time employment"]


def test_reading_items_skip_people_without_candidates_and_other_cells():
    rel = [_rel("k3", "c3"), _rel("kx", "nobody"), _rel("kd", "c1", D="D2"), _rel("kl", "c1", lam=0),
           _rel("kt", "c1", t=3)]
    assert sw.reading_items(rel, _check(), TRAINED, seed=7) == []


def test_reading_items_negatives_fixed_per_release():
    a = sw.reading_items([_rel("k2", "c2")], _check(), TRAINED, seed=7)
    b = sw.reading_items([_rel("k2", "c2")], _check(), TRAINED, seed=7)
    assert a[0]["conts"] == b[0]["conts"]
    labs = a[0]["lab"]
    assert labs.sum() == 1 and (~labs).sum() == 2  # 2 x own negatives, drawn from the other pool members
    assert set(c.strip() for c, l in zip(a[0]["conts"], labs) if not l) <= {"Injury", "Full-time employment"}


def test_reading_items_researcher_view_equals_full_record_minus_untrained():
    # researcher: conds == conds_full, so the only change from the old items is dropping untrained conditions
    check = [{"pid": "r1", "conds": ["A1", "B1", "U1"]}, {"pid": "r2", "conds": ["C1", "D1", "E1", "F1"]}]
    trained = {"A1", "B1", "C1", "D1", "E1", "F1"}
    it = sw.reading_items([_rel("k", "r1")], check, trained, seed=3)[0]
    own = [c.strip() for c, lab in zip(it["conts"], it["lab"]) if lab]
    neg = [c.strip() for c, lab in zip(it["conts"], it["lab"]) if not lab]
    assert own == ["A1", "B1"] and len(neg) == 4 and set(neg) == {"C1", "D1", "E1", "F1"}


def test_reading_items_skip_when_no_negative_left_and_count_reasons():
    check = [{"pid": "p", "conds": ["X", "Y"]}]
    skipped: dict = {}
    items = sw.reading_items([_rel("k", "p"), _rel("o", "nobody")], check, {"X", "Y"}, seed=1, skipped=skipped)
    assert items == []
    assert skipped == {"other_cell_or_person": 1, "no_candidate": 0, "no_negative": 1}


def test_reading_items_draw_a_fixed_subset_that_depends_on_the_key():
    pool = [f"N{i}" for i in range(20)]
    check = [{"pid": "p", "conds": ["Own"]}, {"pid": "q", "conds": pool}]
    trained = set(pool) | {"Own"}

    def negs(key):
        it = sw.reading_items([_rel(key, "p")], check, trained, seed=5)[0]
        return [c.strip() for c, lab in zip(it["conts"], it["lab"]) if not lab]

    assert len(negs("k1")) == 2 and negs("k1") == negs("k1")
    assert any(negs(f"k{i}") != negs("k1") for i in range(2, 12))


def test_reading_items_accept_string_lambda_and_t_and_dedupe_own():
    check = [{"pid": "p", "conds": ["X", "X", "Y"]}, {"pid": "q", "conds": ["Z", "W", "V", "U"]}]
    it = sw.reading_items([_rel("k", "p", lam="1", t="1")], check, {"X", "Y", "Z", "W", "V", "U"}, seed=2)
    assert len(it) == 1 and int(it[0]["lab"].sum()) == 2


def test_select_releases(tmp_path):
    own, shared = tmp_path / "emp.jsonl", tmp_path / "shared.jsonl"
    own.write_text('{"key": "a"}\n', encoding="utf-8")
    shared.write_text('{"key": "s"}\n{"key": "t"}\n', encoding="utf-8")
    cfg = {"releases_for_reading_check": str(shared), "releases_for_reading_check_by_recipient": {"employer": str(own)}}
    assert sw.select_releases(cfg, "employer") == (str(own), [{"key": "a"}], False)
    path, rel, is_shared = sw.select_releases(cfg, "insurer")
    assert (path, len(rel), is_shared) == (str(shared), 2, True)
    assert sw.select_releases({**cfg, "releases_for_reading_check_by_recipient": None}, "employer")[2] is True
    bad = {**cfg, "releases_for_reading_check_by_recipient": {"employer": str(tmp_path / "missing.jsonl")}}
    try:
        sw.select_releases(bad, "employer")
        raise AssertionError("missing per-recipient releases must stop")
    except SystemExit:
        pass
    assert sw.select_releases({"releases_for_reading_check": str(tmp_path / "none.jsonl")}, "employer") == (None, [], True)
