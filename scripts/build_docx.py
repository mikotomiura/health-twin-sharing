"""Build the JAMIA submission files (Word) from the Markdown manuscript.

Usage (from the PJ root):  python3 -I scripts/build_docx.py
Needs: pandoc (3.x) on PATH. Python standard library only.

Inputs (papers/jamia/): front.md, abstract.md, introduction.md, methods.md, results.md, discussion.md,
conclusion.md, statements.md, figures.md (legends + alt text), tables/table{1,2,3}.md, supplement.md,
private/authors.md (optional, not in git: authors, affiliations, corresponding author's contact details).
Bibliography: docs/references.md (the citation SSOT; [n] there are permanent IDs, not display order).

Outputs (papers/jamia/build/, not in git):
  manuscript.docx   title page, abstract, main text with each table after the paragraph that first cites it,
                    statements, references (renumbered in order of first citation), figure legends with
                    "Alt text:" under each (JAMIA General Instructions, 2026-11-18 excerpt in
                    .steering/20261118-paper-word-build/paper-write/venue-raw/jamia-instructions-build.txt)
  supplement.docx   the Supplementary File, with its own reference list in its own citation order
  fig1.pdf ... fig4.pdf  vector figures with embedded fonts, uploaded as separate files
  manuscript.md / supplement.md  the exact pandoc inputs, for checking
  build_report.json  word counts and checks
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "papers/jamia"
OUT = SRC / "build"
MAIN = ["introduction", "methods", "results", "discussion", "conclusion"]
N_TABLES = 1  # main-text tables; former Tables 2-3 go to the Supplement as Tables S1-S2 (JAMIA: table data not duplicated in the text)
SUPPLEMENT_TABLES = (2, 3)  # tables/table{k}.md inserted at the @@TABLE:k@@ markers of supplement.md

# Venue strings in the OUP/Medline style for the cited entries whose SSOT venue cell is not already in that form.
# Authors and titles come from docs/references.md; only the venue part and, for [3], the title are restyled here.
VENUE = {
    18: "Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (CCS) 2020:697–716",
    70: "Proceedings of the 25th IEEE Computer Security Foundations Symposium (CSF) 2012:265–79",
    34: "NPJ Digit Med 2024;7:77",
    35: "Front Genet 2018;9:31",
    4: "Proceedings of the 12th International Conference on Learning Representations (ICLR) 2024",
    8: "arXiv 2026. doi:10.48550/arXiv.2605.23820",
    31: "GitHub 2026. https://github.com/Opening-Science/twin-llm (accessed 18 November 2026)",
    37: "Version 4.0.0. GitHub. https://github.com/synthetichealth/synthea/tree/v4.0.0 (accessed 18 November 2026)",
    38: "Proc VLDB Endow 2018;11:1715–28",
    39: "USENIX Workshop on Health Information Technologies (HealthTech) 2013. http://sharps.org/wp-content/uploads/CHAN-HEALTHTECH.pdf (accessed 18 November 2026)",
    40: "ACM Trans Database Syst 2014;39:1–36",
    41: "arXiv 2026. doi:10.48550/arXiv.2605.19127",
    42: "Findings of the Association for Computational Linguistics: EMNLP 2026. 2026:21080–114",
    43: "Proc Priv Enhancing Technol 2026;2026:154–74",
    44: "Proceedings of the 40th International Conference on Machine Learning (PMLR 202) 2023",
    45: "BMC Med Inform Decis Mak 2019;19:44",
    67: "eCFR. https://www.ecfr.gov/current/title-45/section-164.514 (accessed 18 November 2026)",
    76: "Int J Uncertain Fuzziness Knowl-Based Syst 2002;10:557–70",
    77: "ACM Trans Knowl Discov Data 2007;1:3",
    78: "Proceedings of the 23rd IEEE International Conference on Data Engineering (ICDE) 2007:106–15",
    79: "J Am Med Inform Assoc 2018;25:230–8",
    46: "SIGKDD Explor 2002;4:6–11",
    47: "Pain 2015;156:569–76",
    48: "Drug Alcohol Depend 2022;238:109551",
    49: "Clin J Pain 2014;30:557–64",
    50: "J Affect Disord 2018;235:293–302",
    51: "Drug Alcohol Depend Rep 2025;16:100347",
    71: "Scand J Stat 1979;6:65–70",
    72: None,  # filled from the SSOT cell if cited (see ref_entry)
    74: "TEST 2018;27:811–25",
    75: "Stat Appl Genet Mol Biol 2010;9",  # Crossref (2026-11-18): volume 9, issue 1, no page or article number
}
VENUE[3] = "2026"
TITLE = {
    3: "Verifiable privacy for digital health twins: open call 2026/27 (version 1) and call terms",
}
VENUE_URL = {3: "https://opening.science/apply/health-twin-privacy-2026/ (accessed 18 November 2026)",
             18: "doi:10.1145/3372297.3423363"}  # [18]: SSOT key is the arXiv version; cite the CCS version (Crossref 2026-11-18)


def read(p: Path) -> str:
    return re.sub(r"<!--.*?-->", "", p.read_text(encoding="utf-8"), flags=re.S).strip() + "\n"


# --------------------------------------------------------------------------- bibliography
def load_ssot() -> dict:
    rows = {}
    for line in (ROOT / "docs/references.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \[(\d+)\] \|", line)
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[int(m.group(1))] = dict(key=cells[1], cite=cells[2], venue=cells[3], url=cells[4], status=cells[6])
    return rows


def strip_ja(s: str) -> str:
    """Remove full-width parenthetical notes (Japanese annotations in the SSOT) and stray spaces."""
    s = re.sub(r"（[^）]*）", "", s)
    if re.search(r"[぀-ヿ一-鿿]", s):
        raise ValueError(f"Japanese text left in a reference: {s!r}")
    return re.sub(r"\s+", " ", s).strip()


def split_authors(cite: str):
    m = re.match(r"^(.*?\b[A-Z]{1,3}(?: Jr)?)\. (.*)$", cite)  # authors end at the first initials followed by ". "
    if m and "," in m.group(1) or (m and re.fullmatch(r"[A-Z][\w'’-]+ [A-Z]{1,3}", m.group(1))):
        return m.group(1), m.group(2)
    a, t = cite.split(". ", 1)  # organisations ("The MITRE Corporation. Synthea ...")
    return a, t


def ref_entry(n: int, rows: dict) -> str:
    r = rows[n]
    if not r["status"].startswith("active"):
        raise ValueError(f"[{n}] is not active in the SSOT: {r['status'][:40]}")
    if n in TITLE:  # SSOT cell mixes Japanese into the title; author part is the text before the first ". "
        authors, title = r["cite"].split(". ", 1)[0], TITLE[n]
    else:
        authors, title = split_authors(strip_ja(r["cite"]))
    names = [a.strip() for a in authors.split(",")]
    if len(names) > 3:  # OUP style: list all if 3 or fewer, otherwise the first 3 and "et al."
        names = names[:3] + ["et al"]
    authors = ", ".join(n_.replace("et al.", "et al") for n_ in names)
    authors = re.sub(r",? et al$", ", et al.", authors) if authors.endswith("et al") else authors
    title = TITLE.get(n, title).rstrip(".")
    venue = VENUE.get(n) or strip_ja(r["venue"])
    doi = re.match(r"doi:(\S+)", r["key"])
    tail = ""
    if n in VENUE_URL:
        tail = " " + VENUE_URL[n]
    elif doi and "doi:" not in venue:
        tail = f" doi:{doi.group(1)}"
    elif r["key"].startswith("arXiv:") and "arXiv" not in venue:
        tail = f" arXiv:{r['key'].split(':', 1)[1]}"
    elif r["key"].startswith("pmid:"):
        tail = f" PMID:{r['key'].split(':', 1)[1]}"
    sep = "" if authors.endswith(".") else "."
    entry = f"{authors}{sep} {title}. {venue}"
    entry += "" if entry.endswith((")", ".")) else "."
    return entry + tail


CITE = re.compile(r"\[(\d+)\]\s*[–-]\s*\[(\d+)\]|\[(\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*)\]")


def cite_numbers(m) -> list:
    if m.group(1):
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    out = []
    for part in m.group(3).split(","):
        a, *b = re.split(r"\s*[–-]\s*", part.strip())
        out += list(range(int(a), int(b[0]) + 1)) if b else [int(a)]
    return out


def compress(nums: list) -> str:
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}-{nums[j]}" if j - i >= 2 else ", ".join(map(str, nums[i:j + 1])))
        i = j + 1
    return "[" + ", ".join(parts) + "]"


def renumber(text: str, rows: dict, order: dict) -> str:
    """Replace SSOT [n] by display numbers in order of first appearance; place after punctuation, no space."""
    def sub(m):
        ids = cite_numbers(m)
        for n in ids:
            if n not in rows:
                raise ValueError(f"citation [{n}] not in docs/references.md")
            order.setdefault(n, len(order) + 1)
        return "\x00" + compress([order[n] for n in ids]) + "\x00"
    t = CITE.sub(sub, text)
    t = re.sub(r"\s*\x00(\[[^\]]*\])\x00([.,;:])", r"\2\1", t)  # "text [4]." -> "text.[4]"
    t = re.sub(r"\s*\x00(\[[^\]]*\])\x00", r"\1", t)  # mid-sentence: no space before the bracket
    return t


# --------------------------------------------------------------------------- Word styles
def reference_docx(path: Path):
    default = path.with_name("default-reference.docx")
    subprocess.run(["pandoc", "-o", str(default), "--print-default-data-file", "reference.docx"], check=True)
    zin = zipfile.ZipFile(default)
    styles = zin.read("word/styles.xml").decode("utf-8")
    tnr = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman" />'
    styles = re.sub(r"(<w:rPrDefault>\s*<w:rPr>\s*)<w:rFonts[^>]*/>", r"\1" + tnr, styles, flags=re.S)
    # double spacing for all text (JAMIA: "all submissions should be double-spaced")
    styles = re.sub(r"(<w:pPrDefault>\s*<w:pPr>\s*)<w:spacing[^>]*/>", r'\1<w:spacing w:after="0" w:line="480" w:lineRule="auto" />',
                    styles, flags=re.S)

    def set_style(sid, ppr=None, rpr=None):
        nonlocal styles
        m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, styles, re.S)
        block = m.group(0)
        if ppr is not None:
            block = re.sub(r"<w:pPr>.*?</w:pPr>", ppr, block, flags=re.S) if "<w:pPr>" in block else block.replace("</w:style>", ppr + "</w:style>")
        if rpr is not None:
            block = re.sub(r"<w:rPr>.*?</w:rPr>", rpr, block, flags=re.S) if "<w:rPr>" in block else block.replace("</w:style>", rpr + "</w:style>")
        styles = styles.replace(m.group(0), block)

    sz = '<w:sz w:val="24" /><w:szCs w:val="24" />'
    keep = '<w:pPr><w:keepNext /><w:keepLines /><w:spacing w:before="240" w:after="0" w:line="480" w:lineRule="auto" /><w:outlineLvl w:val="%d" /></w:pPr>'
    # JAMIA heading hierarchy: BOLD CAPS, bold lower case, Plain text, Italics
    set_style("Heading1", keep % 0, f"<w:rPr>{tnr}<w:b /><w:bCs /><w:caps />{sz}</w:rPr>")
    set_style("Heading2", keep % 1, f"<w:rPr>{tnr}<w:b /><w:bCs />{sz}</w:rPr>")
    set_style("Heading3", keep % 2, f"<w:rPr>{tnr}{sz}</w:rPr>")
    set_style("Heading4", keep % 3, f"<w:rPr>{tnr}<w:i /><w:iCs />{sz}</w:rPr>")
    set_style("Title", '<w:pPr><w:spacing w:after="240" w:line="480" w:lineRule="auto" /><w:jc w:val="left" /></w:pPr>',
              f"<w:rPr>{tnr}<w:b /><w:bCs />{sz}</w:rPr>")
    set_style("BodyText", '<w:pPr><w:spacing w:before="0" w:after="0" w:line="480" w:lineRule="auto" /><w:ind w:firstLine="480" /></w:pPr>')
    set_style("FirstParagraph", '<w:pPr><w:ind w:firstLine="0" /></w:pPr>')
    # table cells: double-spaced too (2026-11-19, JAMIA editorial office: "ensure that your manuscript is formatted in
    # double-space"; the earlier single spacing of table cells was the likely cause of the unsubmission)
    set_style("Compact", '<w:pPr><w:spacing w:before="0" w:after="0" w:line="480" w:lineRule="auto" /><w:ind w:firstLine="0" /></w:pPr>',
              f'<w:rPr><w:sz w:val="20" /><w:szCs w:val="20" /></w:rPr>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = styles.encode("utf-8") if item.filename == "word/styles.xml" else zin.read(item.filename)
            zout.writestr(item, data)


# --------------------------------------------------------------------------- assembly
def words(text: str) -> int:
    """Word's way: whitespace-separated tokens containing a letter or digit (Markdown marks removed)."""
    t = re.sub(r"[*_#`>|]", " ", text)
    return sum(1 for tok in t.split() if re.search(r"[A-Za-z0-9]", tok))


def place_tables(paras: list, tables: dict) -> list:
    out, placed = [], set()
    for p in paras:
        out.append(p)
        if p.startswith("#"):
            continue
        for k in sorted(tables):
            if k not in placed and re.search(r"\bTables? (?:\d+[–-])?%d\b" % k, p):
                if placed and max(placed) > k:
                    raise ValueError(f"Table {k} first cited after a later table")
                placed.add(k)
                out.append(tables[k])
    missing = set(tables) - placed
    if missing:
        raise ValueError(f"tables never cited in the main text: {sorted(missing)}")
    return out


def table_md(k: int) -> str:
    t = read(SRC / f"tables/table{k}.md")
    title, rest = t.split("\n", 1)
    m = re.match(r"(Table S?\d+\.)\s*(.*)", title)
    return f"**{m.group(1)}** {m.group(2)}\n{widen(rest)}"


def widen(md: str) -> str:
    """Pandoc takes relative column widths from the dashes of a pipe table's separator line when cells are long.
    Give each column dashes in proportion to its longest cell (square-root damped, at least 8% of the width)."""
    # a table file may hold several tables (e.g. Table 3 A and B): widen each block of consecutive "|" lines on its own
    out, block = [], []
    for line in md.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if block:
            out.append(widen_block("\n".join(block)))
            block = []
        out.append(line)
    return "\n".join(out[:-1])


def widen_block(md: str) -> str:
    lines = md.splitlines()
    rows = [l for l in lines if l.lstrip().startswith("|") and not re.fullmatch(r"\|[\s:|-]+\|?", l.strip())]
    if not rows:
        return md
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    ncol = max(len(c) for c in cells)
    longest = [max((len(c[i]) if i < len(c) else 0) for c in cells) for i in range(ncol)]
    word = [max((len(t) for c in cells if i < len(c) for t in re.sub(r"[*_]", "", c[i]).split()), default=1) for i in range(ncol)]
    w = [max(x, 1) ** 0.5 for x in longest]
    # no column narrower than its longest word (about 90 characters of 10-pt text across the table)
    w = [max(x / sum(w), 0.08, (word[i] + 2) / 90) for i, x in enumerate(w)]
    dashes = [max(3, round(100 * x / sum(w))) for x in w]
    sep = "|" + "|".join("-" * d for d in dashes) + "|"
    return "\n".join(sep if re.fullmatch(r"\|[\s:|-]+\|?", l.strip()) else l for l in lines)


def build():
    if shutil.which("pandoc") is None:
        sys.exit("pandoc not found")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_ssot()
    report = {}

    # main text: paragraphs in reading order, tables after first citing paragraph
    main = "\n\n".join(read(SRC / f"{f}.md") for f in MAIN)
    main = re.sub(r"^### ", "## ", re.sub(r"^## ", "# ", main, flags=re.M), flags=re.M)  # shift up one level; order matters
    paras = [p.strip() for p in re.split(r"\n\s*\n", main) if p.strip()]
    main_words = words("\n".join(paras))  # before tables are inserted: JAMIA excludes tables and figures
    paras = place_tables(paras, {k: table_md(k) for k in range(1, N_TABLES + 1)})

    statements = read(SRC / "statements.md").replace("## ", "# ")
    order: dict = {}
    body = renumber("\n\n".join(paras), rows, order)
    statements = renumber(statements, rows, order)
    figures = read(SRC / "figures.md")
    if CITE.search(figures):
        raise ValueError("citation in figure legends: number it by the place where the figure is first cited")
    figures = re.sub(r"^> Alt text:", "Alt text:", figures, flags=re.M).replace("# Figure legends", "# Figure legends", 1)
    refs = "# References\n\n" + "\n".join(f"{i}. {ref_entry(n, rows)}" for n, i in sorted(order.items(), key=lambda x: x[1]))

    # title page
    front = read(SRC / "front.md")
    title = re.search(r"^title:\s*(.+)$", front, re.M).group(1).strip()
    keywords = re.search(r"^keywords:\s*(.+)$", front, re.M).group(1).strip()
    authors_p = SRC / "private/authors.md"
    authors = read(authors_p) if authors_p.exists() else "[Authors, degrees, affiliations; corresponding author's full name, postal address, e-mail and telephone number: added by the author before submission]"
    abstract = read(SRC / "abstract.md").replace("## Abstract", "# Abstract")
    if CITE.search(abstract):
        raise ValueError("JAMIA: references should not be included in abstracts")
    abstract_words = words(re.sub(r"^# Abstract\s*", "", abstract))
    abstract_words_no_heads = words(re.sub(r"\*\*[^*]+:\*\*", "", re.sub(r"^# Abstract\s*", "", abstract)))
    titlepage = (f"---\ntitle: \"{title}\"\n---\n\n{authors}\n\n**Keywords:** {keywords}\n\n"
                 f"**Word count** (excluding title page, abstract, references, figures and tables): {main_words:,}\n\n"
                 f"**Tables:** {N_TABLES}. **Figures:** 4.\n\n```{{=openxml}}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n")
    pagebreak = "\n\n```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n\n"
    manuscript = titlepage + abstract + pagebreak + body + "\n\n" + statements + pagebreak + refs + pagebreak + figures
    (OUT / "manuscript.md").write_text(manuscript, encoding="utf-8")

    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "reference.docx"
        reference_docx(ref)
        subprocess.run(["pandoc", str(OUT / "manuscript.md"), "-f", "markdown-implicit_figures-citations", "-t", "docx",
                        "--reference-doc", str(ref), "-o", str(OUT / "manuscript.docx")], check=True)
        # supplement: its own citation order and reference list
        sup = read(SRC / "supplement.md")
        for k in SUPPLEMENT_TABLES:
            marker = f"@@TABLE:{k}@@"
            if sup.count(marker) != 1:
                raise ValueError(f"supplement.md must contain {marker} exactly once")
            sup = sup.replace(marker, table_md(k))
        sorder: dict = {}
        sup = renumber(sup, rows, sorder)
        if sorder:
            sup += "\n\n# References cited in the Supplementary material\n\n" + "\n".join(
                f"{i}. {ref_entry(n, rows)}" for n, i in sorted(sorder.items(), key=lambda x: x[1]))
        sup = f"---\ntitle: \"Supplementary material for: {title}\"\n---\n\n" + sup
        (OUT / "supplement.md").write_text(sup, encoding="utf-8")
        subprocess.run(["pandoc", str(OUT / "supplement.md"), "-f", "markdown-implicit_figures-citations", "-t", "docx",
                        "--reference-doc", str(ref), "-o", str(OUT / "supplement.docx")], check=True)
    for k in range(1, 5):
        shutil.copy2(SRC / f"figures/fig{k}.pdf", OUT / f"fig{k}.pdf")

    report.update(
        main_text_words=main_words, main_text_limit=4000,
        abstract_words_with_headings=abstract_words, abstract_words_without_headings=abstract_words_no_heads, abstract_limit=250,
        tables=N_TABLES, figures=4,
        references=len(order), citation_order={f"[{n}]": i for n, i in sorted(order.items(), key=lambda x: x[1])},
        supplement_references=len(sorder), supplement_citation_order={f"[{n}]": i for n, i in sorted(sorder.items(), key=lambda x: x[1])},
        statement_words=words(statements), figure_legend_words=words(figures),
        authors_file_present=authors_p.exists(),
        to_confirm=manuscript.count("[TO CONFIRM"),
    )
    (OUT / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if "order" not in k}, indent=1))


if __name__ == "__main__":
    build()
