"""Small-world corpus: records rendered as text, with the recipient knowledge K set by how A is assigned.

Design: .steering/20261113-small-world-design/design.md (§1-4) and .steering/20261113-small-world-train-throughput/design.md.

World = adults (age at first_year within age_range_at_first_year, alive at the end) of a flattened Synthea cohort.
A person's record = age and gender at first_year and the conditions active in first_year (onset <= y, not abated before y,
the same definition as scripts/build_records.py). A = the person has a condition matching the attribute's A_pattern with
onset before first_year; persons whose A starts in or after first_year are dropped (neither clearly A nor clearly not A).
Every condition matching A_pattern or the attribute's D0 alias rule (a D0.include pattern and no D0.exclude pattern;
config/condition_lists.json) is removed from the text; A appears only in the final field (A_FIELDS, e.g.
"Drug misuse: yes|no") (Codex point 3). The rendered text before the A field is checked to contain no such match.
(2026-11-13: A_pattern alone missed complications named in another word order, e.g. "Neuropathy due to type 2
diabetes mellitus"; .steering/20261113-sw-corpus-ranges/.)

Recipient (config "recipient" or the 4th CLI argument; default researcher = the whole record): only the conditions the
recipient type sees are written (scripts/sw_ranges.py). The augmentation draws are made on the whole record and the
invisible conditions are dropped afterwards, so the draws do not depend on the recipient either.

K (minimal pairs: the same persons, documents and augmentation; only the A field differs):
  K2  A as generated (the world's correlations)
  K1  A re-assigned by a random permutation over the training persons: the base rate is kept exactly, A is independent
      of the conditions
  K0  A re-assigned so that exactly half of the training persons have it, independently of the conditions
Augmentation ([54]): each person is written n_aug times, each time with a template drawn from TEMPLATES and the
conditions in a random order. The draws do not depend on K.

Usage: python scripts/sw_corpus.py <derived-dir> <exp-dir> <K> [recipient]   (reads <exp-dir>/config.json and SEED)
  -> <exp-dir>/corpus/<K>/ (researcher) or <exp-dir>/corpus/<recipient>/<K>/: train.txt (documents separated by a blank line), check.jsonl (held-out persons, true A),
     train_persons.jsonl (conditions and A as assigned under K, for the 4a targets), manifest.json (counts, rates, SHA-256)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sw_ranges  # noqa: E402

A_FIELDS = {"drug_misuse": "Drug misuse:", "diabetes_t2": "Type 2 diabetes:"}
A_FIELD = A_FIELDS["drug_misuse"]  # default, kept for callers written before A_FIELDS
TEMPLATES = (
    "Patient record. Age: {age}. Sex: {sex}. Conditions: {conds}. {field} {a}",
    "A {age}-year-old {noun}. Current conditions: {conds}. {field} {a}",
    "Sex {sex}, age {age}. Active problems: {conds}. {field} {a}",
)
NOUN = {"female": "woman", "male": "man"}


def hidden_matcher(pattern: str, d0: dict | None = None):
    """Function display -> bool: matches A_pattern, or a D0.include pattern and no D0.exclude pattern."""
    rx = re.compile(pattern, re.IGNORECASE)
    inc = [re.compile(x, re.IGNORECASE) for x in (d0 or {}).get("include", [])]
    exc = [re.compile(x, re.IGNORECASE) for x in (d0 or {}).get("exclude", [])]
    return lambda d: bool(rx.search(d)) or (any(r.search(d) for r in inc) and not any(r.search(d) for r in exc))


def text_leaks(body: str, pattern: str, d0: dict | None = None) -> bool:
    """True if the text (without the A field) names A: an A_pattern match, or a D0.include match that does not lie inside
    a D0.exclude match (exclude patterns are matched without a leading '^', e.g. 'Prediabetes' inside a list)."""
    if re.search(pattern, body, re.IGNORECASE):
        return True
    spans = [m.span() for x in (d0 or {}).get("exclude", []) for m in re.finditer(x.lstrip("^"), body, re.IGNORECASE)]
    for x in (d0 or {}).get("include", []):
        for m in re.finditer(x.lstrip("^"), body, re.IGNORECASE):
            if not any(a <= m.start() and m.end() <= b for a, b in spans):
                return True
    return False


def load_world(derived: Path, pattern: str, year: int, age_range: list[int], d0: dict | None = None) -> pd.DataFrame:
    """One row per adult: pid, age, gender, conds (list of displays without A and its D0 aliases), A (bool)."""
    p = pd.read_parquet(derived / "persons.parquet")
    c = pd.read_parquet(derived / "conditions.parquet")
    return build_world(p, c, pattern, year, age_range, d0=d0)


def build_world(p: pd.DataFrame, c: pd.DataFrame, pattern: str, year: int, age_range: list[int],
                d0: dict | None = None) -> pd.DataFrame:
    c = c.copy()
    c["onset_year"] = pd.to_datetime(c["onset"], utc=True, errors="coerce", format="ISO8601").dt.year
    c["abate_year"] = pd.to_datetime(c["abatement"], utc=True, errors="coerce", format="ISO8601").dt.year
    p = p[p["deceased"].isna()].copy()
    p["age"] = year - pd.to_datetime(p["birth"]).dt.year
    p = p[p["age"].between(age_range[0], age_range[1])]
    is_a = c["display"].str.contains(pattern, case=False, regex=True, na=False)
    first_a = c[is_a].groupby("pid")["onset_year"].min()
    a_before = set(first_a[first_a < year].index)
    a_later = set(first_a[first_a >= year].index)
    p = p[~p["pid"].isin(a_later)]
    is_hidden = c["display"].fillna("").map(hidden_matcher(pattern, d0)).astype(bool) | is_a
    active = c[~is_hidden & (c["onset_year"] <= year) & (c["abate_year"].isna() | (c["abate_year"] >= year))]
    active = active.sort_values("onset_year").drop_duplicates(["pid", "display"])
    conds = active.groupby("pid")["display"].apply(list)
    out = pd.DataFrame({
        "pid": p["pid"].to_numpy(),
        "age": p["age"].astype(int).to_numpy(),
        "gender": p["gender"].to_numpy(),
    })
    out["conds"] = [conds.get(pid, []) for pid in out["pid"]]
    out["A"] = out["pid"].isin(a_before)
    return out.sort_values("pid").reset_index(drop=True)


def split(world: pd.DataFrame, check_frac: float, seed: int, force_check: frozenset[str] = frozenset()
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Person-level split, stratified by A so both parts keep the base rate. Persons in force_check (those whose
    releases are read in the reading check) always go to the check part, so the model never trains on their records."""
    rng = np.random.default_rng([seed, 1])
    check = world.index[world["pid"].isin(force_check)].tolist()
    for _, g in world.groupby("A"):
        idx = np.setdiff1d(g.index.to_numpy(), check)
        n = max(0, int(round(len(g) * check_frac)) - int(g["pid"].isin(force_check).sum()))
        check.extend(rng.choice(idx, size=n, replace=False).tolist())
    is_check = world.index.isin(check)
    return world[~is_check].reset_index(drop=True), world[is_check].reset_index(drop=True)


def assign_k(a: np.ndarray, k: str, seed: int) -> np.ndarray:
    """A labels of the training persons under knowledge level k (minimal pairs; see the module docstring)."""
    a = np.asarray(a, dtype=bool)
    rng = np.random.default_rng([seed, 2])
    if k == "K2":
        return a.copy()
    if k == "K1":
        return rng.permutation(a)
    if k == "K0":
        n = len(a)
        out = np.zeros(n, dtype=bool)
        out[rng.permutation(n)[: n // 2]] = True
        return out
    raise ValueError(f"unknown K: {k}")


def render(age: int, gender: str, conds: list[str], a: bool | None, template: int, order: np.ndarray | None = None,
           field: str = A_FIELD) -> str:
    """One document. a=None leaves the A field open (the probe prompt ends with the field, e.g. 'Drug misuse:')."""
    cs = [conds[i] for i in order] if order is not None else list(conds)
    text = TEMPLATES[template].format(
        age=age, sex=gender, noun=NOUN.get(gender, "person"), field=field,
        conds="; ".join(cs) if cs else "none", a="" if a is None else ("yes" if a else "no"),
    )
    return text.rstrip() if a is None else text


def documents(train: pd.DataFrame, a_k: np.ndarray, n_aug: int, seed: int, field: str = A_FIELD,
              recipient: str = "researcher") -> list[str]:
    """n_aug documents per person. Template and order draws use their own rng, independent of K and of the recipient
    (drawn on the whole record; the recipient's invisible conditions are dropped after ordering)."""
    rng = np.random.default_rng([seed, 3])
    docs = []
    for i, row in enumerate(train.itertuples(index=False)):
        for _ in range(n_aug):
            t = int(rng.integers(len(TEMPLATES)))
            order = rng.permutation(len(row.conds))
            cs = sw_ranges.shown(recipient, [row.conds[j] for j in order])
            docs.append(render(row.age, row.gender, cs, bool(a_k[i]), t, field=field))
    rng_shuffle = np.random.default_rng([seed, 4])
    return [docs[j] for j in rng_shuffle.permutation(len(docs))]


def reading_pool(check: list[dict], trained: set[str]) -> list[str]:
    """4b candidate pool: conditions visible to the recipient in the check persons (`conds`) that occur among the
    training persons (`trained`), sorted so the negative draw is fixed."""
    return sorted({c for r in check for c in r["conds"] if c in trained})


def reading_items(releases: list[dict], check: list[dict], trained: set[str], seed: int,
                  skipped: dict | None = None) -> list[dict]:
    """Fixed 4b items (check 4b of .steering/20261113-small-world-design/, addendum 2026-11-14): D0, lambda = 1,
    t = 1 (2023, the year of the records). Candidates are only conditions the model could have learned to write
    after " Conditions:": visible to its recipient (each check person's `conds`, the recipient's view, not
    `conds_full`) and present among the training persons (`trained`). Own = the person's candidates; negatives =
    2 x own drawn from the other pool members, fixed per release. A release is skipped when its person has no
    candidate or no negative is left (an AUC needs both); counts by reason go to `skipped` if given.
    Assumes the range is decided per condition, not per person (sw_ranges.shown), so a pool condition a person
    lacks in `conds` is also absent from that person's whole record."""
    by_pid = {r["pid"]: r for r in check}
    pool = reading_pool(check, trained)
    counts = {"other_cell_or_person": 0, "no_candidate": 0, "no_negative": 0}
    items = []
    for r in releases:
        if r["D"] != "D0" or int(r["lambda"]) != 1 or int(r["t"]) != 1 or r["pid"] not in by_pid:
            counts["other_cell_or_person"] += 1
            continue
        own = list(dict.fromkeys(c for c in by_pid[r["pid"]]["conds"] if c in trained))
        if not own:
            counts["no_candidate"] += 1
            continue
        others = [c for c in pool if c not in own]
        if not others:
            counts["no_negative"] += 1
            continue
        neg = random.Random(f"{seed}|{r['key']}").sample(others, k=min(len(others), 2 * len(own)))
        items.append({"key": r["key"], "A": bool(r["A"]), "text": r["text"].strip(),
                      "conts": [" " + c for c in own + neg], "lab": np.array([True] * len(own) + [False] * len(neg))})
    if skipped is not None:
        skipped.update(counts)
    return items


def select_releases(cfg: dict, recipient: str) -> tuple[str | None, list[dict], bool]:
    """The releases a recipient's model reads in 4b: its own file (`releases_for_reading_check_by_recipient`) if
    configured, else the shared `releases_for_reading_check`. Returns (path actually read or None, releases,
    shared). A configured per-recipient file that is missing stops; a missing shared file gives no releases (4b is
    then not run), as before."""
    own = (cfg.get("releases_for_reading_check_by_recipient") or {}).get(recipient)
    if own:
        if not Path(own).exists():
            raise SystemExit(f"releases for {recipient} not found: {own}")
        path, shared = own, False
    else:
        path, shared = cfg.get("releases_for_reading_check"), True
    if not path or not Path(path).exists():
        return None, [], shared
    rel = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return path, rel, shared


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(derived: Path, exp: Path, k: str, recipient: str | None = None) -> None:
    cfg = json.loads((exp / "config.json").read_text(encoding="utf-8"))
    seed = int((exp / "SEED").read_text().strip())
    recipient = recipient or cfg.get("recipient", "researcher")
    if recipient not in sw_ranges.RECIPIENTS:
        raise SystemExit(f"unknown recipient: {recipient}")
    lists = json.loads(Path(cfg["condition_lists"]).read_text(encoding="utf-8"))
    attr = lists["attributes"][cfg["attribute"]]
    pattern, d0 = attr["A_pattern"], attr.get("D0")
    if cfg["attribute"] not in A_FIELDS:
        raise SystemExit(f"no A field name for attribute {cfg['attribute']!r}; add it to A_FIELDS")
    field = A_FIELDS[cfg["attribute"]]
    world = load_world(derived, pattern, lists["first_year"], lists["age_range_at_first_year"], d0=d0)
    force = frozenset()
    if cfg.get("releases_for_reading_check"):
        rel = Path(cfg["releases_for_reading_check"])
        force = frozenset(json.loads(line)["pid"] for line in rel.read_text(encoding="utf-8").splitlines() if line.strip())
    train, check = split(world, cfg["check_frac"], seed, force)
    if force & set(train["pid"]):
        raise SystemExit("a reading-check person is in the training part")
    a_k = assign_k(train["A"].to_numpy(), k, seed)
    is_hidden = hidden_matcher(pattern, d0)
    n_hidden = sum(1 for cs in world["conds"] for d in cs if is_hidden(d))
    if n_hidden:  # per-display check, independent of the text check below
        raise SystemExit(f"{n_hidden} conditions in the world match the A pattern or its D0 aliases")
    docs = documents(train, a_k, cfg["n_aug"], seed, field=field, recipient=recipient)
    leaked = sum(1 for d in docs if text_leaks(d.rsplit(field, 1)[0], pattern, d0))
    if leaked:
        raise SystemExit(f"{leaked} documents still match the A pattern or its D0 aliases")
    out = exp / "corpus" / k if recipient == "researcher" else exp / "corpus" / recipient / k
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n\n".join(docs) + "\n", encoding="utf-8", newline="\n")
    with (out / "train_persons.jsonl").open("w", encoding="utf-8", newline="\n") as f:  # world rates for the 4a targets
        for i, row in enumerate(train.itertuples(index=False)):
            rec = {"conds": sw_ranges.shown(recipient, list(row.conds)), "A": bool(a_k[i])}
            if recipient != "researcher":  # the whole record, for evaluation only (researcher files stay byte-identical)
                rec["conds_full"] = list(row.conds)
            f.write(json.dumps(rec) + "\n")
    with (out / "check.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in check.itertuples(index=False):
            rec = {"pid": row.pid, "age": int(row.age), "gender": row.gender,
                   "conds": sw_ranges.shown(recipient, list(row.conds)), "A": bool(row.A)}
            if recipient != "researcher":
                rec["conds_full"] = list(row.conds)
            f.write(json.dumps(rec) + "\n")
    manifest = {
        "K": k, "recipient": recipient, "seed": seed, "attribute": cfg["attribute"], "A_pattern": pattern,
        "A_field": field, "D0": d0,
        "range_rules_sha256": hashlib.sha256(Path(sw_ranges.__file__).read_bytes()).hexdigest(),  # whole module (cross-review LOW)
        "n_world": len(world), "n_train": len(train), "n_check": len(check),
        "rate_A_world": float(world["A"].mean()), "rate_A_train_as_assigned": float(a_k.mean()),
        "rate_A_check": float(check["A"].mean()) if len(check) else None, "n_forced_check": len(force & set(check["pid"])), "n_docs": len(docs), "n_aug": cfg["n_aug"],
        "chars": sum(len(d) for d in docs),
        "sha256": {n: sha256(out / n) for n in ("train.txt", "train_persons.jsonl", "check.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k2: v for k2, v in manifest.items() if k2 != "sha256"}))


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) == 5 else None)
