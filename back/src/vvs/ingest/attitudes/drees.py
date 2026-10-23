"""Baromètre d'opinion de la DREES : parts pondérées par année, calculées sur les microdonnées.

Source : le fichier de diffusion « barometre2000_2025_diff.csv » (open data de la DREES, jeu
431_le-barometre-d-opinion ; séparateur « ; », poids `poids` à virgule décimale), une ligne par
répondant et par année (2000 à 2025, pas d'enquête en 2003). Les modalités « NSP » sont codées
999999999 et restent au dénominateur ; une variable vide signifie « question non posée cette
année-là » (ou à ce demi-échantillon) : ces lignes sortent du calcul.

Valeur = somme des poids des répondants donnant l'une des modalités `codes` / somme des poids
des répondants à qui la question a été posée × 100 (méthode `microdata`). Contrôle : pour les
années présentes dans le classeur de tris à plat publié avec la vague 2025
(« 2025_barometre_drees_tap_complets_v2.xlsx », 2018 à 2025), la part recalculée doit égaler la
part publiée à 0,05 point près (ligne `check_row` de la feuille de la question).
"""

from __future__ import annotations

import polars as pl

from vvs.ingest.attitudes import Obs, raw
from vvs.ingest.attitudes.registry import AttitudeRegistry, Extraction
from vvs.ingest.macro import xlsx

NSP = 999999999


class DreesError(ValueError):
    pass


def shares(path, variable: str, codes: list[int]) -> dict[int, float]:
    df = (
        pl.scan_csv(path, separator=";", infer_schema=False)
        .select(
            pl.col("annee").cast(pl.Int64),
            pl.col("poids").str.replace(",", ".").cast(pl.Float64),
            pl.col(variable).cast(pl.Float64).alias("v"),
        )
        .filter(pl.col("v").is_not_null())
        .group_by("annee")
        .agg(
            (pl.col("poids") * pl.col("v").is_in([float(c) for c in codes])).sum().alias("num"),
            pl.col("poids").sum().alias("den"),
        )
        .sort("annee")
        .collect()
    )
    return {int(y): 100 * n / d for y, n, d in zip(df["annee"], df["num"], df["den"], strict=True)}


def published(content: bytes, variable: str, row_label: str) -> dict[int, float]:
    """Parts publiées dans le classeur de tris à plat : {année: %}."""
    prefix = variable.upper() + "."
    for name in xlsx.sheet_names(content):
        rows = xlsx.read_sheet(content, name)
        q = next((i for i, r in enumerate(rows[:15]) if any(
            isinstance(c, str) and c.strip().startswith(prefix) for c in r)), None)
        if q is None:
            continue
        hdr = next(i for i in range(q, 0, -1) if any(
            isinstance(c, str) and c.strip().isdigit() and len(c.strip()) == 4 for c in rows[i]))
        years = {
            j: int(c) for j, c in enumerate(rows[hdr])
            if isinstance(c, str) and c.strip().isdigit() and len(c.strip()) == 4
        }
        for r in rows[q:]:
            label = next((c for c in r if isinstance(c, str) and c.strip()), "")
            if label.strip() == row_label:
                out = {}
                for j, y in years.items():
                    v = r[j + 1] if j + 1 < len(r) else None
                    if isinstance(v, float):
                        out[y] = v
                return out
        raise DreesError(f"{variable} : ligne {row_label!r} absente des tris à plat")
    raise DreesError(f"{variable} : feuille absente des tris à plat")


def extract(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    sp = e.spec
    path = raw.read(reg.documents[e.document]).path
    check = raw.read(reg.documents[sp["check_document"]]).path.read_bytes()
    out = []
    for item in sp["items"]:
        sid, var, codes = item["series_id"], item["variable"], [int(c) for c in item["codes"]]
        got = shares(path, var, codes)
        pub = published(check, var, item["check_row"])
        if not pub:
            raise DreesError(f"{sid} : aucune part publiée pour le contrôle")
        for y, v in pub.items():
            if y in got and abs(got[y] - v) > 0.05:
                raise DreesError(f"{sid} {y} : {got[y]:.2f} % recalculé contre {v:.2f} % publié")
        for y, v in got.items():
            out.append(Obs(sid, str(y), round(v, 1), "microdata", e.document, None,
                           item.get("note", "")))
    return out
