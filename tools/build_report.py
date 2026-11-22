# -*- coding: utf-8 -*-
"""Build the CISC3024 AI#3 report as .docx and print-ready .html.

Typography follows the HW1/HW2 reports: A4, 2.54 cm top and bottom, 3.17 cm
left and right, Times New Roman 12 pt body, Arial Bold headings, Courier New
10.5 pt code, three-line tables.

Run:  python tools/build_report.py
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

WS = r"D:\CISC3024\AI_HW3"
SRC = os.path.join(WS, "CISC3024_AI3_Report.md")
OUT_DOCX = os.path.join(WS, "CISC3024_AI3_Report.docx")
OUT_HTML = os.path.join(WS, "CISC3024_AI3_Report.print.html")

BODY_FONT = "Times New Roman"
HEAD_FONT = "Arial"
CODE_FONT = "Courier New"
BODY_PT = 12
CODE_PT = 10.5
TABLE_PT = 9.5
CAPTION_PT = 9.5

CONTENT_CM = 21.0 - 3.17 * 2      # 14.66 cm of text width on A4


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------
def split_front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, m.group(2)


# ---------------------------------------------------------------------------
# Block parser
# ---------------------------------------------------------------------------
def parse_blocks(body):
    """Yield (kind, payload) tuples. Kinds: h2, h3, p, table, figure, code, math."""
    lines = body.splitlines()
    i, n = 0, len(lines)
    blocks = []
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(buf)))
            continue

        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            blocks.append(("math", s[2:-2].strip()))
            i += 1
            continue

        if s.startswith("- "):
            buf = []
            while i < n and lines[i].strip().startswith("- "):
                buf.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("list", buf))
            continue

        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", "\n".join(buf)))
            continue

        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if m:
            blocks.append(("figure", (m.group(1), m.group(2))))
            i += 1
            continue

        if s.startswith("### "):
            blocks.append(("h3", s[4:].strip()))
            i += 1
            continue
        if s.startswith("## "):
            blocks.append(("h2", s[3:].strip()))
            i += 1
            continue

        if s.startswith("|") and s.endswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = lines[i].strip().strip("|")
                cells = [c.strip() for c in row.split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append(("table", rows))
            continue

        buf = []
        while i < n and lines[i].strip() and not lines[i].strip().startswith(("|", "#", "```", "![", "$$")):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            blocks.append(("p", " ".join(buf)))
    return blocks


# ---------------------------------------------------------------------------
# Inline runs: **bold**, `code`, *italic*, $math$
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\$[^$]+\$|\*[^*]+\*)")


def add_math_runs(par, text, size, font=BODY_FONT):
    """Render `x^{2}` and `I_{r}` as real super/subscript runs."""
    pos = 0
    for m in re.finditer(r"([_^])\{([^{}]*)\}", text):
        if m.start() > pos:
            r = par.add_run(text[pos:m.start()])
            r.font.name = font
            r.font.size = Pt(size)
        r = par.add_run(m.group(2))
        r.font.name = font
        r.font.size = Pt(size)
        if m.group(1) == "^":
            r.font.superscript = True
        else:
            r.font.subscript = True
        pos = m.end()
    if pos < len(text):
        r = par.add_run(text[pos:])
        r.font.name = font
        r.font.size = Pt(size)


def add_inline(par, text, size=BODY_PT, bold=False, color=None):
    for piece in TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            r = par.add_run(piece[2:-2])
            r.bold = True
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            r.font.name = CODE_FONT
            r.font.size = Pt(size - 1)
        elif piece.startswith("$") and piece.endswith("$") and len(piece) > 2:
            add_math_runs(par, piece[1:-1], size)
            continue
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            r.italic = True
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
        else:
            r = par.add_run(piece)
            r.font.name = BODY_FONT
            r.font.size = Pt(size)
        if bold:
            r.bold = True
        if color is not None:
            r.font.color.rgb = color


# ---------------------------------------------------------------------------
# DOCX helpers
# ---------------------------------------------------------------------------
def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, spec in kwargs.items():
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        for k, v in spec.items():
            el.set(qn(f"w:{k}"), str(v))


def clear_cell_borders(cell):
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        set_cell_border(cell, **{edge: {"val": "nil"}})


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def set_cell_margins(cell, left=57, right=57, top=57, bottom=57):
    """Set w:tcMar in twentieths of a point. Must follow w:shd in tcPr order."""
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcMar")):
        tcPr.remove(old)
    mar = OxmlElement("w:tcMar")
    for tag, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def add_page_field(par):
    run = par.add_run()
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = "PAGE"
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1)
    run._r.append(it)
    run._r.append(f2)
    run.font.name = BODY_FONT
    run.font.size = Pt(9)


def setup_styles(doc):
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    pf = st.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)


def setup_page(doc):
    s = doc.sections[0]
    s.page_width = Cm(21.0)
    s.page_height = Cm(29.7)
    s.top_margin = Cm(2.54)
    s.bottom_margin = Cm(2.54)
    s.left_margin = Cm(3.17)
    s.right_margin = Cm(3.17)
    s.different_first_page_header_footer = True
    return s


def build_docx(meta, blocks):
    doc = Document()
    setup_styles(doc)
    sec = setup_page(doc)

    # footer page numbers, suppressed on the cover
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(fp)

    # ---------------- cover ----------------
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("CISC3024 PATTERN RECOGNITION  ·  AI ASSIGNMENT #3")
    r.font.name = HEAD_FONT
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    r.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    r = p.add_run(meta.get("title", ""))
    r.font.name = HEAD_FONT
    r.font.size = Pt(22)
    r.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(48)
    r = p.add_run(meta.get("subtitle", ""))
    r.font.name = BODY_FONT
    r.font.size = Pt(13)
    r.italic = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    fields = [
        ("Student name", meta.get("student", "")),
        ("Student ID", meta.get("student_id", "")),
        ("Course", meta.get("course", "")),
        ("Algorithm", meta.get("algorithm", "")),
        ("Application", meta.get("application", "")),
        ("Submission date", meta.get("date", "")),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for label, value in fields:
        row = t.add_row()
        c0, c1 = row.cells
        c0.width = Cm(4.2)
        c1.width = Cm(10.2)
        clear_cell_borders(c0)
        clear_cell_borders(c1)
        pa = c0.paragraphs[0]
        ra = pa.add_run(label)
        ra.font.name = HEAD_FONT
        ra.font.size = Pt(9.5)
        ra.bold = True
        ra.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        pb = c1.paragraphs[0]
        rb = pb.add_run(value)
        rb.font.name = BODY_FONT
        rb.font.size = Pt(10.5)

    doc.add_page_break()

    # ---------------- body ----------------
    for kind, payload in blocks:
        if kind == "h2":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(payload)
            r.font.name = HEAD_FONT
            r.font.size = Pt(15)
            r.bold = True

        elif kind == "h3":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(payload)
            r.font.name = HEAD_FONT
            r.font.size = Pt(14)
            r.bold = True

        elif kind == "p":
            p = doc.add_paragraph()
            # A justified reference with an unbreakable URL stretches one line
            # across the whole measure. Left-align those.
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if is_ref(payload)
                           else WD_ALIGN_PARAGRAPH.JUSTIFY)
            add_inline(p, payload)

        elif kind == "list":
            for item in payload:
                p = doc.add_paragraph(style="List Bullet")
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, item)

        elif kind == "quote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.7)
            p.paragraph_format.right_indent = Cm(0.3)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            for j, ln in enumerate(payload.split("\n")):
                if j:
                    p.add_run().add_break()
                r = p.add_run(ln)
                r.font.size = Pt(10.5)
                r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        elif kind == "math":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            add_math_runs(p, payload, BODY_PT)

        elif kind == "code":
            tbl = doc.add_table(rows=1, cols=1)
            cell = tbl.rows[0].cells[0]
            clear_cell_borders(cell)
            shade(cell, "F5F5F5")
            # Tight cell margins: at Courier New 10.5 pt the default 0.19 cm
            # padding leaves 62 characters per line, and the longest snippet
            # line is 63.
            set_cell_margins(cell, left=57, right=57, top=57, bottom=57)
            cell.paragraphs[0]._p.getparent().remove(cell.paragraphs[0]._p)
            for ln in payload.splitlines() or [""]:
                p = cell.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                r = p.add_run(ln if ln else " ")
                r.font.name = CODE_FONT
                r.font.size = Pt(CODE_PT)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)

        elif kind == "figure":
            caption, path = payload
            full = os.path.join(WS, path)
            if not os.path.exists(full):
                print("  MISSING FIGURE:", path)
                continue
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            from PIL import Image as _Image
            with _Image.open(full) as _im:
                aspect = _im.height / _im.width
            if 14.66 * aspect > 20.5:
                p.add_run().add_picture(full, height=Cm(20.5))
            else:
                p.add_run().add_picture(full, width=Cm(CONTENT_CM))
            c = doc.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.paragraph_format.space_after = Pt(10)
            r = c.add_run(caption)
            r.font.name = BODY_FONT
            r.font.size = Pt(CAPTION_PT)
            r.italic = True
            r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

        elif kind == "table":
            rows = payload
            ncol = max(len(r) for r in rows)
            tbl = doc.add_table(rows=0, cols=ncol)
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            tbl.autofit = True
            for ri, row in enumerate(rows):
                cells = tbl.add_row().cells
                for ci in range(ncol):
                    txt = row[ci] if ci < len(row) else ""
                    cell = cells[ci]
                    clear_cell_borders(cell)
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.0
                    add_inline(p, txt, size=TABLE_PT, bold=(ri == 0))
                    if ri == 0:
                        shade(cell, "F2F2F2")
                        set_cell_border(cell, top={"val": "single", "sz": 12, "color": "000000"})
                        set_cell_border(cell, bottom={"val": "single", "sz": 6, "color": "000000"})
                    if ri == len(rows) - 1:
                        set_cell_border(cell, bottom={"val": "single", "sz": 12, "color": "000000"})
            doc.add_paragraph().paragraph_format.space_after = Pt(2)

    doc.save(OUT_DOCX)
    print("wrote", OUT_DOCX, "%.0f KB" % (os.path.getsize(OUT_DOCX) / 1024))
    return doc


# ---------------------------------------------------------------------------
# Print HTML for the Chromium PDF route
# ---------------------------------------------------------------------------
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def inline_html(text):
    out = []
    for piece in TOKEN.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            out.append("<strong>" + esc(piece[2:-2]) + "</strong>")
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            out.append("<code>" + esc(piece[1:-1]) + "</code>")
        elif piece.startswith("$") and piece.endswith("$") and len(piece) > 2:
            out.append(math_html(piece[1:-1]))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append("<em>" + esc(piece[1:-1]) + "</em>")
        else:
            out.append(esc(piece))
    return "".join(out)


def math_html(text):
    text = esc(text)
    text = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", text)
    text = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", text)
    return text


CSS = f"""
@page {{ size: A4; margin: 2.54cm 3.17cm; }}
@page cover {{ margin: 2.54cm 3.17cm; @bottom-center {{ content: none; }} }}
@page body {{ @bottom-center {{ content: counter(page); font-family: {BODY_FONT};
                              font-size: 9pt; color: #666; }} }}
html, body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: "{BODY_FONT}", serif; font-size: {BODY_PT}pt; line-height: 1.45;
        color: #111; margin: 0; }}
section.cover {{ page: cover; }}
section.body {{ page: body; }}
h2 {{ font-family: "{HEAD_FONT}", sans-serif; font-size: 15pt; font-weight: bold;
      margin: 18pt 0 8pt; page-break-after: avoid; }}
h3 {{ font-family: "{HEAD_FONT}", sans-serif; font-size: 14pt; font-weight: bold;
      margin: 14pt 0 6pt; page-break-after: avoid; }}
p {{ margin: 0 0 7pt; text-align: justify; }}
p.ref {{ text-align: left; }}
ul {{ margin: 4pt 0 8pt 14pt; padding: 0; }}
li {{ margin-bottom: 3pt; }}
blockquote {{ margin: 8pt 0 10pt 10pt; padding-left: 10pt;
             border-left: 2pt solid #999; font-size: 10.5pt; line-height: 1.35;
             page-break-inside: avoid; }}
code {{ font-family: "{CODE_FONT}", monospace; font-size: {CODE_PT - 1}pt; }}
pre {{ font-family: "{CODE_FONT}", monospace; font-size: {CODE_PT}pt; line-height: 1.35;
       background: #f5f5f5; border-left: 3px solid #bbb; padding: 7pt 10pt;
       margin: 8pt 0 10pt; white-space: pre-wrap; page-break-inside: avoid; }}
pre.long {{ page-break-inside: auto; }}
p.math {{ text-align: center; margin: 8pt 0; font-size: {BODY_PT}pt; }}
sup, sub {{ font-size: 70%; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
         font-size: {TABLE_PT}pt; page-break-inside: avoid; }}
th {{ font-family: "{HEAD_FONT}", sans-serif; font-weight: bold; text-align: left;
      padding: 3pt 5pt; background: #f2f2f2; border-top: 1.5pt solid #000;
      border-bottom: 0.75pt solid #000; }}
td {{ padding: 2.5pt 5pt; text-align: left; vertical-align: top; }}
tbody tr:last-child td {{ border-bottom: 1.5pt solid #000; }}
figure {{ margin: 10pt 0 12pt; text-align: center; page-break-inside: avoid; }}
figure img {{ max-width: 100%; max-height: 20.5cm; object-fit: contain; }}
figcaption {{ font-size: {CAPTION_PT}pt; color: #444; font-style: italic;
              margin-top: 4pt; text-align: center; }}
.eyebrow {{ font-family: "{HEAD_FONT}", sans-serif; font-size: 10.5pt; font-weight: bold;
            color: #555; letter-spacing: 0.5pt; text-align: center; margin-top: 6cm; }}
.cover-title {{ font-family: "{HEAD_FONT}", sans-serif; font-size: 22pt; font-weight: bold;
                text-align: center; margin: 20pt 0 10pt; line-height: 1.25; }}
.cover-sub {{ font-size: 13pt; font-style: italic; color: #444; text-align: center;
              margin-bottom: 40pt; }}
table.meta {{ width: 80%; margin: 0 auto; font-size: 10.5pt; }}
table.meta td {{ border: none; padding: 3pt 6pt; }}
table.meta td:first-child {{ font-family: "{HEAD_FONT}", sans-serif; font-weight: bold;
                             color: #666; font-size: 9.5pt; width: 32%; }}
"""


def is_ref(payload):
    """Reference entries start with [n] and must not be justified."""
    return bool(re.match(r"^\[\d+\]", payload.strip()))


def build_html(meta, blocks):
    parts = []
    for kind, payload in blocks:
        if kind == "h2":
            parts.append(f"<h2>{esc(payload)}</h2>")
        elif kind == "h3":
            parts.append(f"<h3>{esc(payload)}</h3>")
        elif kind == "p":
            cls = ' class="ref"' if is_ref(payload) else ""
            parts.append(f"<p{cls}>{inline_html(payload)}</p>")
        elif kind == "list":
            parts.append("<ul>" + "".join(f"<li>{inline_html(x)}</li>"
                                          for x in payload) + "</ul>")
        elif kind == "quote":
            parts.append("<blockquote>" + "<br>".join(esc(x) for x in payload.split("\n"))
                         + "</blockquote>")
        elif kind == "math":
            parts.append(f'<p class="math">{math_html(payload)}</p>')
        elif kind == "code":
            # A block taller than one page cannot honour break-inside: avoid --
            # Chromium pushes it to a fresh page, still cannot fit it, and leaves
            # the previous page almost blank. Let long blocks flow instead.
            cls = ' class="long"' if payload.count("\n") > 28 else ""
            parts.append(f"<pre{cls}>" + esc(payload) + "</pre>")
        elif kind == "figure":
            caption, path = payload
            parts.append(
                f'<figure><img src="{esc(path)}" alt=""><figcaption>'
                f"{esc(caption)}</figcaption></figure>"
            )
        elif kind == "table":
            rows = payload
            h = ["<table><thead><tr>"]
            for c in rows[0]:
                h.append(f"<th>{inline_html(c)}</th>")
            h.append("</tr></thead><tbody>")
            for row in rows[1:]:
                h.append("<tr>" + "".join(f"<td>{inline_html(c)}</td>" for c in row) + "</tr>")
            h.append("</tbody></table>")
            parts.append("".join(h))

    meta_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>"
        for k, v in [
            ("Student name", meta.get("student", "")),
            ("Student ID", meta.get("student_id", "")),
            ("Course", meta.get("course", "")),
            ("Algorithm", meta.get("algorithm", "")),
            ("Application", meta.get("application", "")),
            ("Submission date", meta.get("date", "")),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{esc(meta.get('title',''))}</title><style>{CSS}</style></head><body>
<section class="cover">
  <p class="eyebrow">CISC3024 PATTERN RECOGNITION &middot; AI ASSIGNMENT #3</p>
  <p class="cover-title">{esc(meta.get('title',''))}</p>
  <p class="cover-sub">{esc(meta.get('subtitle',''))}</p>
  <table class="meta">{meta_rows}</table>
</section>
<section class="body">
{chr(10).join(parts)}
</section>
</body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", OUT_HTML, "%.0f KB" % (len(html.encode()) / 1024))
    return html


def main():
    text = open(SRC, encoding="utf-8").read()
    meta, body = split_front_matter(text)
    blocks = parse_blocks(body)

    kinds = {}
    for k, _ in blocks:
        kinds[k] = kinds.get(k, 0) + 1
    print("blocks:", kinds)

    figs = [p for k, p in blocks if k == "figure"]
    missing = [p for _, p in figs if not os.path.exists(os.path.join(WS, p))]
    print(f"figures: {len(figs)} referenced, {len(missing)} missing")
    for m in missing:
        print("  MISSING:", m)

    build_docx(meta, blocks)
    build_html(meta, blocks)

    if missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
