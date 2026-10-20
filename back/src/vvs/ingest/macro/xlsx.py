"""Lecture minimale d'un classeur xlsx (zip + XML), sans dépendance supplémentaire.

Suffit pour les exports tabulaires simples : chaînes partagées, chaînes en ligne, nombres.
Ni formules, ni dates typées (les exports lus ici écrivent les dates en texte).
"""

from __future__ import annotations

import io
import re
import zipfile

from lxml import etree

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


def sheet_names(content: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        wb = etree.fromstring(z.read("xl/workbook.xml"), XML_PARSER)
        return [s.get("name") for s in wb.iterfind("m:sheets/m:sheet", NS)]


def read_sheet(content: bytes, name: str) -> list[list[str | float | None]]:
    """Lignes de la feuille `name`, cellules vides à None, colonnes alignées sur la référence."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        wb = etree.fromstring(z.read("xl/workbook.xml"), XML_PARSER)
        rid = None
        for s in wb.iterfind("m:sheets/m:sheet", NS):
            if s.get("name") == name:
                rid = s.get(f"{{{NS['r']}}}id")
        if rid is None:
            raise KeyError(f"feuille {name!r} absente du classeur")
        rels = etree.fromstring(z.read("xl/_rels/workbook.xml.rels"), XML_PARSER)
        target = next(
            r.get("Target") for r in rels.iterfind("rel:Relationship", NS) if r.get("Id") == rid
        )
        target = target.lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = etree.fromstring(z.read("xl/sharedStrings.xml"), XML_PARSER)
            for si in sst.iterfind("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        sheet = etree.fromstring(z.read(sheet_path), XML_PARSER)
    rows = []
    for row in sheet.iterfind("m:sheetData/m:row", NS):
        cells: dict[int, str | float | None] = {}
        for c in row.iterfind("m:c", NS):
            col = _column_index(c.get("r"))
            kind = c.get("t")
            v = c.find("m:v", NS)
            if kind == "s":
                cells[col] = shared[int(v.text)]
            elif kind == "inlineStr":
                cells[col] = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
            elif kind == "str":
                cells[col] = v.text if v is not None else None
            elif v is not None and v.text is not None:
                cells[col] = float(v.text)
            else:
                cells[col] = None
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i) for i in range(width)])
    return rows


def _column_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1
