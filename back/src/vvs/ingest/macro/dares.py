"""Demandeurs d'emploi : portail Opendatasoft de la DARES (data.dares.travail-emploi.gouv.fr).

L'export CSV d'un jeu, filtré par une clause ODSQL `where` (registre, `params.where`), est
stocké tel quel. Séparateur `;`, en-tête avec BOM UTF-8, une ligne par mois et par champ
géographique ; le parseur garde le champ demandé (`params.champ`).
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date

import httpx

from vvs.ingest.macro import raw
from vvs.ingest.macro.registry import Series

EXPORT_URL = "https://data.dares.travail-emploi.gouv.fr/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
VALUE_COLUMN = "nombre_de_demandeurs_d_emploi"


def ingest(http: httpx.Client, series: Series) -> raw.RawFile | None:
    url = EXPORT_URL.format(dataset=series.source_id)
    params = {"where": series.params["where"], "order_by": "date"}
    r = raw.get(http, url, params=params)
    content = r.content
    parse(content, series.params["champ"])  # valide avant d'écrire
    return raw.write_raw(series.raw_dir, "csv", content, str(r.url))


def parse(content: bytes, champ: str) -> list[tuple[date, float]]:
    """[(premier jour du mois, nombre de demandeurs)] du champ demandé, triés par date."""
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    missing = {"date", "champ", VALUE_COLUMN} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"DARES : colonnes absentes {sorted(missing)}")
    out = {}
    for row in reader:
        if row["champ"] != champ:
            continue
        m = re.fullmatch(r"(\d{4})-(\d{2})", row["date"])
        if not m:
            raise ValueError(f"DARES : date illisible {row['date']!r}")
        d = date(int(m[1]), int(m[2]), 1)
        if d in out:
            raise ValueError(
                f"DARES : mois {d} en double pour le champ {champ!r} (filtre trop large ?)"
            )
        if row[VALUE_COLUMN] == "":
            continue
        out[d] = float(row[VALUE_COLUMN])
    if not out:
        raise ValueError(f"DARES : aucune ligne pour le champ {champ!r}")
    return sorted(out.items())
