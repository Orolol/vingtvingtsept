"""Séries longues d'attitudes et d'opinion (contexte et site), pilotées par
data/registry/attitude_series.yaml.

Familles : baromètre d'image du FN/RN (TNS Sofres, Kantar, Verian), baromètre de la confiance
politique du Cevipof (OpinionWay), indice longitudinal de tolérance de la CNCDH, Eurobaromètre
standard (deux problèmes les plus importants, France), préoccupations du CRÉDOC, baromètre
d'opinion de la DREES. Ces séries décrivent le contexte : aucune n'entre dans la prévision
sans passer le protocole de docs/07-evaluation.md.

- `ingest(series)` télécharge les documents nécessaires dans
  data/raw/attitudes/<groupe>/<document>.<ext> (+ .meta.json, sha256) ; rien n'est réécrit ;
- `parse(series)` lit les valeurs dans les documents (graphiques et tableaux PDF par leur
  couche texte, extraits recopiés vérifiés dans le texte de la page, classeurs, microdonnées)
  et écrit data/curated/attitudes.parquet (`ATTITUDES_SCHEMA`), une ligne par série et par
  vague. Aucune valeur n'est saisie de mémoire : chacune pointe son document et sa page.

Colonnes : `date` = premier jour de terrain quand il est connu, sinon premier jour du mois de
la vague (séries `wave`) ou 1er janvier de l'année (séries `A`) ; `fieldwork_start` /
`fieldwork_end` restent vides quand seul le mois est publié ; `method` dit comment la valeur a
été lue (`chart_label`, `table`, `printed`, `derived_from_delta`, `workbook`, `microdata`,
`chart_marker`) ; `page` est la page PDF (1 = première page du fichier).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from vvs import paths
from vvs.ingest.attitudes import raw
from vvs.ingest.attitudes.registry import (
    AttitudeRegistry,
    Extraction,
    extraction_series,
    load,
)

OUTPUT_NAME = "attitudes.parquet"

ATTITUDES_SCHEMA = pl.Schema(
    {
        "series_id": pl.Utf8,
        "date": pl.Date,
        "value": pl.Float64,
        "freq": pl.Utf8,
        "source": pl.Utf8,
        "unit": pl.Utf8,
        "population": pl.Utf8,
        "notes": pl.Utf8,
        "wave": pl.Utf8,
        "fieldwork_start": pl.Date,
        "fieldwork_end": pl.Date,
        "method": pl.Utf8,
        "document": pl.Utf8,
        "page": pl.Int64,
        "raw_sha256": pl.Utf8,
    }
)
METHODS = (
    "chart_label",
    "table",
    "printed",
    "derived_from_delta",
    "workbook",
    "microdata",
    "chart_marker",
)

__all__ = ["ATTITUDES_SCHEMA", "OUTPUT_NAME", "ingest", "parse", "validate_attitudes"]


@dataclass
class Obs:
    """Une valeur lue : `wave` vaut « AAAA-MM » ou « AAAA » ; `start`/`end` priment sur le
    calendrier des vagues quand le document les donne (Eurobaromètre)."""

    series_id: str
    wave: str
    value: float
    method: str
    document: str
    page: int | None = None
    note: str = ""
    start: date | None = None
    end: date | None = None


# ------------------------------------------------------------------------------- ingestion


def ingest(series_ids: list[str] | None = None, log=print) -> dict[str, list[str]]:
    reg = load()
    docs = reg.documents_for(reg.select_series(series_ids))
    out: dict[str, list[str]] = {"téléchargé": [], "présent": [], "échec": []}
    with raw.client() as http:
        for d in docs:
            try:
                status = raw.download(http, d)
            except Exception as e:  # noqa: BLE001 — on continue, l'échec est rapporté
                out["échec"].append(f"{d.id} : {e}")
                log(f"échec       {d.id} : {e}")
                continue
            out[status].append(d.id)
            log(f"{status:11s} {d.id}")
    return out


# --------------------------------------------------------------------------------- lecture


@dataclass
class ParseReport:
    rows: dict[str, tuple[int, date, date]] = field(default_factory=dict)
    checks: int = 0  # valeurs lues deux fois (autre document) et concordantes
    disagreements: list[str] = field(default_factory=list)
    extractions: list[tuple[str, int]] = field(default_factory=list)
    waves_verified: int = 0

    def summary(self) -> str:
        lines = [f"{sid:58s} {n:4d} points {a} -> {b}" for sid, (n, a, b) in self.rows.items()]
        lines.append(
            f"{len(self.extractions)} extraction(s), {self.checks} valeur(s) recoupée(s) entre "
            f"documents, {len(self.disagreements)} désaccord(s), {self.waves_verified} date(s) de "
            "terrain vérifiée(s) dans les documents"
        )
        lines += [f"  désaccord : {d}" for d in self.disagreements]
        return "\n".join(lines)


def run_extraction(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    from vvs.ingest.attitudes import extract

    return extract.run(reg, e)


def parse(series_ids: list[str] | None = None, write: bool = True) -> tuple[pl.DataFrame, ParseReport]:
    reg = load()
    selected = reg.select_series(series_ids)
    want = {s.series_id for s in selected}
    report = ParseReport()
    kept: dict[tuple[str, date], tuple[Obs, date | None, date | None]] = {}
    sha: dict[str, str] = {}
    for e in reg.extractions:
        if not (want & extraction_series(e)):
            continue
        obs = [o for o in run_extraction(reg, e) if o.series_id in want]
        report.extractions.append((e.id, len(obs)))
        for o in obs:
            d, start, end = date_of(reg, o)
            key = (o.series_id, d)
            if key in kept:
                first = kept[key][0]
                if abs(first.value - o.value) > 1e-9:
                    report.disagreements.append(
                        f"{o.series_id} {o.wave} : {first.value:g} ({first.document}) contre "
                        f"{o.value:g} ({o.document} p. {o.page})"
                    )
                else:
                    report.checks += 1
                continue
            kept[key] = (o, start, end)
            if o.document not in sha:
                sha[o.document] = raw.read(reg.documents[o.document]).sha256
    from vvs.ingest.attitudes import extract

    report.waves_verified = extract.verify_waves(reg, {reg.series[s].family for s, _ in kept})
    rows = []
    for (sid, d), (o, start, end) in kept.items():
        s = reg.series[sid]
        cal = reg.waves.get(s.family, {}).get(o.wave)
        note = "; ".join(n for n in (cal.note if cal else "", o.note) if n)
        rows.append(
            {
                "series_id": sid,
                "date": d,
                "value": float(o.value),
                "freq": s.freq,
                "source": s.source,
                "unit": s.unit,
                "population": s.population,
                "notes": note,
                "wave": o.wave,
                "fieldwork_start": start,
                "fieldwork_end": end,
                "method": o.method,
                "document": o.document,
                "page": o.page,
                "raw_sha256": sha[o.document],
            }
        )
    df = pl.DataFrame(rows, schema=ATTITUDES_SCHEMA).sort("series_id", "date")
    validate_attitudes(df)
    for s in selected:
        sub = df.filter(pl.col("series_id") == s.series_id)
        if sub.height == 0:
            raise ValueError(f"{s.series_id} : aucune valeur lue")
        report.rows[s.series_id] = (sub.height, sub["date"].min(), sub["date"].max())
    if write:
        out = paths.CURATED / OUTPUT_NAME
        if series_ids and out.exists():  # mise à jour partielle
            old = pl.read_parquet(out).filter(~pl.col("series_id").is_in(list(want)))
            df = pl.concat([old, df]).sort("series_id", "date")
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
    return df, report


def date_of(reg: AttitudeRegistry, o: Obs) -> tuple[date, date | None, date | None]:
    """(date, début de terrain, fin de terrain) d'une valeur."""
    if o.start is not None:
        return o.start, o.start, o.end
    fam = reg.series[o.series_id].family
    w = reg.waves.get(fam, {}).get(o.wave)
    if w is not None:
        return w.start, w.start, w.end
    if re.fullmatch(r"\d{4}", o.wave):
        return date(int(o.wave), 1, 1), None, None
    y, m = o.wave.split("-")
    return date(int(y), int(m), 1), None, None


def validate_attitudes(df: pl.DataFrame) -> None:
    if df.schema != ATTITUDES_SCHEMA:
        raise ValueError(f"schéma attitudes inattendu : {df.schema}")
    required = ["series_id", "date", "value", "freq", "source", "unit", "population", "wave",
                "method", "document", "raw_sha256"]
    nulls = [c for c in required if df[c].null_count()]
    if nulls:
        raise ValueError(f"valeurs nulles dans {nulls}")
    if df.select(pl.struct("series_id", "date").is_duplicated().any()).item():
        raise ValueError("(series_id, date) en double dans attitudes")
    bad = df.filter((pl.col("value") < 0) | (pl.col("value") > 100))
    if bad.height:
        raise ValueError(f"valeurs hors de [0, 100] : {bad.head(5)}")
    bad = df.filter(~pl.col("method").is_in(METHODS))
    if bad.height:
        raise ValueError(f"méthodes inconnues : {bad['method'].unique().to_list()}")
