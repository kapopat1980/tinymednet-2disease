"""
postprocess_docx.py -- Applies real table formatting that pandoc does not.

Pandoc emits tables with no borders, no header emphasis and everything
left-aligned, which reads as loose columns of text rather than a table. This
script rewrites each table to journal presentation standard:

  * full single-line borders, including inside rules
  * header row bold, lightly shaded, and repeated when a table breaks a page
  * numeric columns centred, the first (label) column left-aligned
  * rows do not split across pages
  * table set to the full text width with sensible cell margins

Usage: python postprocess_docx.py in.docx out.docx
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag):
    return f"{{{W}}}{tag}"


def el(tag, **attrs):
    e = etree.Element(q(tag))
    for k, v in attrs.items():
        e.set(q(k), v)
    return e


BORDER_COLOR = "444444"
HEADER_FILL = "EDEFF3"
NUMERIC = re.compile(r"^\s*[±<≥(\[]?\s*[-+]?[0-9][0-9,.]*"
                     r"(\s*[±–\-]\s*[0-9.,]+)?\s*[%\]\)]*\s*$")


def cell_text(tc):
    return "".join(tc.itertext()).strip()


def set_table_props(tbl):
    tblPr = tbl.find(q("tblPr"))
    if tblPr is None:
        tblPr = el("tblPr")
        tbl.insert(0, tblPr)

    for tag in ("tblBorders", "tblW", "tblCellMar"):
        for old in tblPr.findall(q(tag)):
            tblPr.remove(old)

    borders = el("tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(el(side, val="single", sz="6", space="0", color=BORDER_COLOR))
    tblPr.append(borders)
    tblPr.append(el("tblW", w="5000", type="pct"))

    mar = el("tblCellMar")
    for side, v in (("top", "60"), ("left", "90"), ("bottom", "60"), ("right", "90")):
        mar.append(el(side, w=v, type="dxa"))
    tblPr.append(mar)


def style_cell(tc, *, bold=False, center=False, shade=False):
    tcPr = tc.find(q("tcPr"))
    if tcPr is None:
        tcPr = el("tcPr")
        tc.insert(0, tcPr)
    if shade:
        for old in tcPr.findall(q("shd")):
            tcPr.remove(old)
        tcPr.append(el("shd", val="clear", color="auto", fill=HEADER_FILL))
    for old in tcPr.findall(q("vAlign")):
        tcPr.remove(old)
    tcPr.append(el("vAlign", val="center"))

    for p in tc.findall(q("p")):
        pPr = p.find(q("pPr"))
        if pPr is None:
            pPr = el("pPr")
            p.insert(0, pPr)
        for tag in ("jc", "spacing"):
            for old in pPr.findall(q(tag)):
                pPr.remove(old)
        if center:
            pPr.append(el("jc", val="center"))
        pPr.append(el("spacing", before="20", after="20", line="240",
                      lineRule="auto"))
        if bold:
            for r in p.findall(q("r")):
                rPr = r.find(q("rPr"))
                if rPr is None:
                    rPr = el("rPr")
                    r.insert(0, rPr)
                if rPr.find(q("b")) is None:
                    rPr.append(el("b"))


def fix(xml_bytes):
    root = etree.fromstring(xml_bytes)
    n_tables = 0
    for tbl in root.iter(q("tbl")):
        n_tables += 1
        set_table_props(tbl)
        rows = tbl.findall(q("tr"))
        if not rows:
            continue

        # A column counts as numeric only if every non-empty body cell in it
        # parses as a number, so label columns stay left-aligned.
        body = rows[1:]
        ncols = len(rows[0].findall(q("tc")))
        numeric_col = []
        for c in range(ncols):
            vals = []
            for tr in body:
                tcs = tr.findall(q("tc"))
                if c < len(tcs):
                    vals.append(cell_text(tcs[c]))
            vals = [v for v in vals if v]
            numeric_col.append(bool(vals) and all(NUMERIC.match(v) for v in vals))

        for i, tr in enumerate(rows):
            trPr = tr.find(q("trPr"))
            if trPr is None:
                trPr = el("trPr")
                tr.insert(0, trPr)
            if trPr.find(q("cantSplit")) is None:
                trPr.append(el("cantSplit"))
            if i == 0 and trPr.find(q("tblHeader")) is None:
                trPr.append(el("tblHeader"))

            for c, tc in enumerate(tr.findall(q("tc"))):
                style_cell(tc,
                           bold=(i == 0),
                           center=(numeric_col[c] if c < len(numeric_col) else False),
                           shade=(i == 0))
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          standalone=True), n_tables


def main(src, dst):
    shutil.copy(src, dst)
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        data = {n: z.read(n) for n in names}
    data["word/document.xml"], n_tables = fix(data["word/document.xml"])
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, data[n])
    print(f"{Path(dst).name}: formatted {n_tables} tables "
          f"(borders, bold shaded repeating header, numeric columns centred)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
