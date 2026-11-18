"""Recipient visible ranges for the small-world corpus: which conditions of a record each recipient type sees.

Decision and sources: .steering/20261113-recipient-visible-ranges/design.md (§1 groups, §2 rules, §3 table; researcher
approval and clinical check 2026-11-13). Swiss rules; references [55]-[65] in docs/references.md.
  researcher  the whole record (coded data)
  insurer     basic insurance (KVG): conditions billed as coded diagnoses. Not dental (KVG Art. 31), not social findings,
              not 'Medication review due'; 'Encounter for ...' (reason for the visit) is billed
  employer    employment and education, pregnancy, and an accident only as the generic 'an injury' (at most once)
Age and sex are visible to all three; this module only filters conditions.
"""
from __future__ import annotations

import re

RECIPIENTS = ("researcher", "insurer", "employer")
INJURY = "an injury"

# First match wins (order matters: e.g. oral cancers are CHRON before DENT, mandible fractures are INJ before DENT).
GROUP_RULES: tuple[tuple[str, str], ...] = (
    ("ADMIN", r"^medication review due|^encounter for|^illness, unspecified"),
    ("REPRO", r"pregnan|miscarriage|tubal ligation|steriliz|eclampsia|blighted ovum|fetus"),
    ("SOC", r"employment|labor force|unemployed|educat|primary school|criminal record|intimate partner|housing|homeless"
            r"|refugee|migrant|military|social contact|social isolation|transport|violence in the environment"
            r"|risk activity|^stress \(finding\)"),
    ("MH", r"depress|anxiety|stress disorder|attention deficit|alcohol|tobacco|suicide|sleep disorder"
           r"|intellectual disability"),
    ("PAIN", r"chronic pain|back pain|neck pain|fibromyalgia|migraine|^joint pain|muscle pain|^headache|^pain \(finding\)"),
    ("CHRON", r"malignan|neoplasm|carcinoma|lymphoma|leukemia|melanoma|myeloma|macroglobulinemia|hemangioma"
              r"|neuroendocrine tumor"),
    # not accidents / not injuries although the words match INJ (code review 2026-11-13)
    ("DENT", r"restorative material"),
    ("CHRON", r"osteoporotic fracture|pressure injury"),
    ("INJ", r"fracture|sprain|laceration|injury|burn|concussion|wound|dislocation|whiplash|tear of meniscus"
            r"|rupture of patellar|rotator cuff|ligament"),
    ("ACUTE", r"^acute (viral|bacterial|bronchitis|infective|cholecystitis|respiratory)|sinusitis|pharyngitis"
              r"|sore throat|otitis|urinary tract|cystitis|pyelonephritis|pneumonia|sepsis|septic|appendicitis"
              r"|rupture of appendix|fever|cough|nausea|vomiting|diarrhea|nasal congestion|sputum|fatigue|mediastinitis"
              r"|cellulitis|covid|coronavirus|hypoxemia|respiratory distress|respiratory failure|contact dermatitis"
              r"|retention of urine|bleeding from anus|gallbladder"),
    ("DENT", r"teeth|tooth|dental|denti|gingiv|periodont|caries|pulp|jaw|temporomandibular|oral|saliv|sialo|mandib"
             r"|maxill|molars|alveol|malocclusion|anodontia|periapical|edentulous|occlus|odontogenic|mouth|tongue"
             r"|palate|lip\b|gum\b|tonsil|pharyn|larynx|glottis|uvula|retromolar|cheek mucosa|attrition"
             r"|abrasion of tooth|ridge mucosa|implant|restoration|accretion|teething|root canal|leukoplakia|mucositis"
             r"|granuloma|giant cell|dentofacial|ankylosis|mucocele|nasopharyn|parotid|hypopharyn|postcricoid|pyriform"
             r"|supraglottis|accessory sinus|maxillary sinus|head, face and neck|dribbling|unable to swallow"
             r"|excessive salivation|loss of taste|alveolitis|lefort|orbital floor|nasal bones"),
)
_COMPILED = tuple((g, re.compile(r, re.IGNORECASE)) for g, r in GROUP_RULES)
GROUPS = ("CHRON", "PAIN", "MH", "ACUTE", "INJ", "REPRO", "DENT", "SOC", "ADMIN")

_EMPLOYER_SOC = re.compile(r"employment|labor force|unemployed|educat|primary school", re.IGNORECASE)
_EMPLOYER_REPRO = re.compile(r"^normal pregnancy", re.IGNORECASE)
_INSURER_ADMIN = re.compile(r"^encounter for", re.IGNORECASE)


def group(display: str) -> str:
    """The condition group of a Synthea display (CHRON when no rule matches)."""
    for g, rx in _COMPILED:
        if rx.search(display):
            return g
    return "CHRON"


def view(recipient: str, display: str) -> str | None:
    """How the recipient sees one condition: the display itself, a generic label, or None (not visible)."""
    if recipient not in RECIPIENTS:
        raise ValueError(f"unknown recipient: {recipient}")
    if recipient == "researcher":
        return display
    g = group(display)
    if recipient == "insurer":
        if g in {"CHRON", "PAIN", "MH", "ACUTE", "INJ", "REPRO"} or (g == "ADMIN" and _INSURER_ADMIN.search(display)):
            return display
        return None
    if g == "SOC" and _EMPLOYER_SOC.search(display):
        return display
    if g == "REPRO" and _EMPLOYER_REPRO.search(display):
        return display
    if g == "INJ":
        return INJURY
    return None


def shown(recipient: str, conds: list[str]) -> list[str]:
    """The recipient's view of an ordered list of conditions: order kept, invisible ones dropped, repeats of a
    generic label kept once (first position)."""
    out: list[str] = []
    for d in conds:
        v = view(recipient, d)
        if v is not None and not (v == INJURY and v in out):
            out.append(v)
    return out
