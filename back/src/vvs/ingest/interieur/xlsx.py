"""Lecteur minimal de classeurs .xlsx (Office Open XML), sans dépendance hors lxml.

Sert aux rares fichiers du ministère publiés seulement en .xlsx (européennes 2014,
législatives 2024 France entière). Lit les valeurs brutes : texte partagé, texte en ligne,
nombres. Pas de styles ni de dates.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath

from lxml import etree

_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_M = f"{{{_NS['m']}}}"
_REF = re.compile(r"([A-Z]+)(\d+)")

Cell = str | float | None


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _text(si: etree._Element) -> str:
    # <si><t>…</t></si> ou texte riche <si><r><t>…</t></r>…</si> ; on ignore la phonétique
    return "".join(t.text or "" for t in si.iter(f"{_M}t") if t.getparent().tag != f"{_M}rPh")


def read_xlsx(path: Path | str, sheets: list[str] | None = None) -> dict[str, list[list[Cell]]]:
    """Lit les feuilles demandées (toutes par défaut) : {nom: lignes de cellules}."""
    with zipfile.ZipFile(path) as z:
        wb = etree.fromstring(z.read("xl/workbook.xml"))
        rels = etree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_tag = f"{{{_NS['pr']}}}Relationship"
        targets = {r.get("Id"): r.get("Target") for r in rels.iter(rel_tag)}
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = etree.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [_text(si) for si in sst.iter(f"{_M}si")]
        out: dict[str, list[list[Cell]]] = {}
        for sh in wb.iter(f"{_M}sheet"):
            name = sh.get("name")
            if sheets is not None and name not in sheets:
                continue
            target = targets[sh.get(f"{{{_NS['r']}}}id")]
            member = (
                target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
            )
            cells: dict[tuple[int, int], Cell] = {}
            with z.open(member) as fh:
                for _, c in etree.iterparse(fh, tag=f"{_M}c"):
                    m = _REF.fullmatch(c.get("r", ""))
                    if m is None:
                        raise ValueError(f"{path}: cellule sans référence dans {name}")
                    r, col = int(m.group(2)) - 1, _col_index(m.group(1))
                    t = c.get("t")
                    v = c.find(f"{_M}v")
                    if t == "s":
                        cells[r, col] = shared[int(v.text)]
                    elif t == "inlineStr":
                        cells[r, col] = _text(c.find(f"{_M}is"))
                    elif t in ("str", "e"):
                        cells[r, col] = v.text if v is not None else None
                    elif t == "b" or (v is not None and v.text is not None):
                        cells[r, col] = float(v.text)
                    c.clear()
            if cells:
                nrows = max(r for r, _ in cells) + 1
                ncols = max(c for _, c in cells) + 1
                rows: list[list[Cell]] = [[None] * ncols for _ in range(nrows)]
                for (r, c), val in cells.items():
                    rows[r][c] = val
            else:
                rows = []
            out[name] = rows
        if sheets is not None and set(sheets) - set(out):
            raise ValueError(f"{path}: feuilles absentes {sorted(set(sheets) - set(out))}")
        return out
