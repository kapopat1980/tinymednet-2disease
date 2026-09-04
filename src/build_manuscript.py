"""
build_manuscript.py -- Splices generated tables into the manuscript prose.

Every {{TABLE x}} marker in manuscript_prose.md is replaced with the
corresponding block from results/tables.md, which is itself generated from
frozen prediction artifacts. No table in the manuscript is ever typed by hand,
so the numbers in the paper cannot drift from the numbers in the repository.

Fails loudly on any unresolved marker or any table that exists but is unused.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROSE = ROOT / "paper" / "manuscript_prose.md"
TABLES = ROOT / "results" / "tables.md"
OUT = ROOT / "paper" / "MANUSCRIPT_v2_final.md"


def parse_tables(text):
    """Split tables.md into {'3a': '## Table 3a. ...', ...}."""
    blocks = {}
    parts = re.split(r"\n(?=## Table )", text)
    for p in parts:
        m = re.match(r"## Table ([0-9]+[a-z]?)\.", p)
        if m:
            blocks[m.group(1)] = p.rstrip()
    return blocks


def main():
    prose = PROSE.read_text(encoding="utf-8")
    blocks = parse_tables(TABLES.read_text(encoding="utf-8"))

    used = set()

    def sub(m):
        key = m.group(1).strip()
        if key not in blocks:
            print(f"ERROR: no generated table for marker {{{{TABLE {key}}}}}", file=sys.stderr)
            sys.exit(1)
        used.add(key)
        return blocks[key]

    out = re.sub(r"\{\{TABLE ([^}]+)\}\}", sub, prose)

    if "{{" in out:
        print("ERROR: unresolved markers remain", file=sys.stderr)
        sys.exit(1)

    unused = sorted(set(blocks) - used)
    if unused:
        print(f"WARNING: generated tables not referenced in the manuscript: {unused}")

    OUT.write_text(out, encoding="utf-8")
    n_brackets = len(re.findall(r"`\[[A-Z]", out))
    print(f"written {OUT.name}: {len(used)} tables spliced, "
          f"{n_brackets} author-action placeholders remaining")


if __name__ == "__main__":
    main()
