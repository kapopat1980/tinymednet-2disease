"""
build_supplementary.py -- Renders results/supplementary.md as a Word document
using the same table formatting as the main manuscript.

Run after make_tables.py. Produces SUPPLEMENTARY.docx.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "supplementary.md"
TMP = Path("/tmp/supp_raw.docx")
OUT = ROOT / "SUPPLEMENTARY.docx"

HEADER = """# Supplementary material

**TinyMed-Net: What a Sub-4K-Parameter Network Can and Cannot Do for Tabular
Clinical Screening**

Every table below is generated from the same frozen prediction artifacts as the
main manuscript by `src/make_tables.py`. Supplementary table numbering (S1, S2,
...) is independent of the main text.

"""


def set_landscape(path):
    """
    The complete-metric tables carry nine columns and do not fit portrait width.
    Landscape is the normal convention for wide supplementary tables.
    """
    def q(tag):
        return f"{{{W}}}{tag}"

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        data = {n: z.read(n) for n in names}
    root = etree.fromstring(data["word/document.xml"])
    for sectPr in root.iter(q("sectPr")):
        for tag in ("pgSz", "pgMar"):
            for old in sectPr.findall(q(tag)):
                sectPr.remove(old)
        pgSz = etree.SubElement(sectPr, q("pgSz"))
        pgSz.set(q("w"), "15840")          # 11 in
        pgSz.set(q("h"), "12240")          # 8.5 in
        pgSz.set(q("orient"), "landscape")
        pgMar = etree.SubElement(sectPr, q("pgMar"))
        for k, v in (("top", "1134"), ("right", "1134"), ("bottom", "1134"),
                     ("left", "1134"), ("header", "708"), ("footer", "708"),
                     ("gutter", "0")):
            pgMar.set(q(k), v)
    data["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, data[n])


def main():
    if not SRC.exists():
        sys.exit("results/supplementary.md not found; run make_tables.py first")
    body = SRC.read_text(encoding="utf-8")
    # Replace the generated preamble with the standalone document header.
    body = body.split("\n## Table S1.", 1)
    body = HEADER + "\n## Table S1." + body[1] if len(body) > 1 else HEADER + body[0]
    tmp_md = Path("/tmp/supplementary_doc.md")
    tmp_md.write_text(body, encoding="utf-8")

    subprocess.run(["pandoc", str(tmp_md), "-o", str(TMP)], check=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from postprocess_docx import main as fmt
    fmt(str(TMP), str(OUT))
    set_landscape(OUT)
    print(f"{OUT.name}: page set to landscape")


if __name__ == "__main__":
    main()
