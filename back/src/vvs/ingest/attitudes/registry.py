"""Chargement et validation de data/registry/attitude_series.yaml.

Trois sections :

- `documents` : chaque document téléchargé (PDF, CSV, xlsx, zip), avec son URL, son groupe
  (sous-dossier de data/raw/attitudes/) et ses métadonnées d'enquête ;
- `series` : chaque série publiée dans attitudes.parquet, avec la formulation exacte de la
  question, la population, les changements de méthode et ce qui manque ;
- `extractions` : comment lire les valeurs dans les documents (graphique, tableau, extrait
  recopié, parseur propre à une source). L'ordre compte : pour une même série et une même
  vague, la première extraction de la liste fait foi, les suivantes servent de contrôle.

`waves` donne, par famille de séries, les dates de terrain connues des vagues « AAAA-MM » ;
une vague absente du calendrier est datée du premier jour de son mois.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from vvs import paths

FREQS = ("A", "wave")
UNITS = ("percent", "index_0_100")
KINDS = ("chart", "table", "quote", "eurobarometer", "drees_microdata", "cncdh_chart")
POPULATIONS = (
    "fr_adults_18plus",
    "fr_registered_18plus",
    "fr_metro_adults_18plus",
    "fr_adults_15plus",
)
ID_RE = re.compile(r"^[a-z0-9_]+$")
WAVE_RE = re.compile(r"^\d{4}(-\d{2})?$")


@dataclass(frozen=True)
class Document:
    id: str
    group: str
    url: str
    ext: str
    title: str
    publisher: str
    fieldwork_start: date | None = None
    fieldwork_end: date | None = None
    sample: int | None = None
    mode: str = ""
    notes: str = ""
    member: str | None = None  # fichier lu dans une archive zip


@dataclass(frozen=True)
class Series:
    series_id: str
    family: str
    label: str
    question: str
    answer: str
    source: str
    freq: str
    unit: str
    population: str
    since: str
    changes: str = ""
    missing: str = ""


@dataclass(frozen=True)
class Wave:
    start: date
    end: date | None
    document: str | None
    page: int | None = None
    quote: str = ""  # dates telles qu'imprimées, vérifiées dans le texte de la page
    note: str = ""


@dataclass(frozen=True)
class Extraction:
    id: str
    kind: str
    document: str | None
    page: int | None
    spec: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AttitudeRegistry:
    documents: dict[str, Document]
    series: dict[str, Series]
    waves: dict[str, dict[str, Wave]]
    extractions: tuple[Extraction, ...]

    def select_series(self, ids: list[str] | None) -> list[Series]:
        if not ids:
            return list(self.series.values())
        out = []
        for i in ids:
            if i in self.series:
                out.append(self.series[i])
                continue
            fam = [s for s in self.series.values() if s.family == i]
            if not fam:
                raise KeyError(f"série ou famille inconnue : {i}")
            out += fam
        return out

    def documents_for(self, series: list[Series]) -> list[Document]:
        """Documents à télécharger pour ces séries (ceux de leurs extractions)."""
        want = {s.series_id for s in series}
        ids: list[str] = []
        for e in self.extractions:
            if want & extraction_series(e):
                for d in extraction_documents(e):
                    if d not in ids:
                        ids.append(d)
        for fam in dict.fromkeys(s.family for s in series):  # dates de terrain
            for w in self.waves.get(fam, {}).values():
                if w.document and w.document not in ids:
                    ids.append(w.document)
        return [self.documents[d] for d in ids]


def extraction_series(e: Extraction) -> set[str]:
    sp = e.spec
    if e.kind == "chart":
        return {s["series_id"] for s in sp["series"]}
    if e.kind == "table":
        return set(sp["rows"])
    if e.kind == "quote":
        return {r["series_id"] for r in sp["rows"]}
    if e.kind in ("eurobarometer", "drees_microdata", "cncdh_chart"):
        return set(sp["series"])
    raise ValueError(e.kind)


def extraction_documents(e: Extraction) -> list[str]:
    if e.kind == "eurobarometer":
        return list(e.spec["documents"]) + list((e.spec.get("survey_metadata") or {}).values())
    if e.kind == "drees_microdata":
        return [e.document, e.spec["check_document"]]
    if e.kind == "quote":
        docs = [e.document] if e.document else []
        return docs + [r["document"] for r in e.spec["rows"] if r.get("document")]
    return [e.document]


def registry_file() -> Path:
    return paths.REGISTRY / "attitude_series.yaml"


def _date(v) -> date | None:
    if v is None or v == "":
        return None
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def load(path: Path | None = None) -> AttitudeRegistry:
    raw = yaml.safe_load((path or registry_file()).read_text(encoding="utf-8"))
    docs: dict[str, Document] = {}
    for d in raw["documents"]:
        doc = Document(
            id=d["id"],
            group=d["group"],
            url=d["url"],
            ext=d["ext"],
            title=d["title"],
            publisher=d["publisher"],
            fieldwork_start=_date(d.get("fieldwork_start")),
            fieldwork_end=_date(d.get("fieldwork_end")),
            sample=d.get("sample"),
            mode=d.get("mode") or "",
            notes=d.get("notes") or "",
            member=d.get("member"),
        )
        if not ID_RE.match(doc.id) or not ID_RE.match(doc.group):
            raise ValueError(f"identifiant de document invalide : {doc.id} / {doc.group}")
        if doc.id in docs:
            raise ValueError(f"document en double : {doc.id}")
        docs[doc.id] = doc
    series: dict[str, Series] = {}
    for s in raw["series"]:
        se = Series(
            series_id=s["series_id"],
            family=s["family"],
            label=s["label"],
            question=s["question"],
            answer=s.get("answer") or "",
            source=s["source"],
            freq=s["freq"],
            unit=s["unit"],
            population=s["population"],
            since=str(s["since"]),
            changes=s.get("changes") or "",
            missing=s.get("missing") or "",
        )
        if not ID_RE.match(se.series_id):
            raise ValueError(f"series_id invalide : {se.series_id}")
        if se.series_id in series:
            raise ValueError(f"series_id en double : {se.series_id}")
        if se.freq not in FREQS or se.unit not in UNITS or se.population not in POPULATIONS:
            raise ValueError(f"{se.series_id} : freq/unit/population hors liste")
        series[se.series_id] = se
    waves: dict[str, dict[str, Wave]] = {}
    for fam, items in (raw.get("waves") or {}).items():
        waves[fam] = {}
        for label, w in items.items():
            if not WAVE_RE.match(str(label)):
                raise ValueError(f"vague mal écrite : {fam} {label}")
            if w.get("document") and w["document"] not in docs:
                raise ValueError(f"vague {fam} {label} : document inconnu {w['document']}")
            waves[fam][str(label)] = Wave(
                start=_date(w["start"]),
                end=_date(w.get("end")),
                document=w.get("document"),
                page=w.get("page"),
                quote=w.get("quote") or "",
                note=w.get("note") or "",
            )
    extractions = []
    seen = set()
    for e in raw["extractions"]:
        ex = Extraction(
            id=e["id"],
            kind=e["kind"],
            document=e.get("document"),
            page=e.get("page"),
            spec={k: v for k, v in e.items() if k not in ("id", "kind", "document", "page")},
        )
        if ex.kind not in KINDS:
            raise ValueError(f"{ex.id} : kind {ex.kind!r} hors {KINDS}")
        if ex.id in seen:
            raise ValueError(f"extraction en double : {ex.id}")
        seen.add(ex.id)
        for d in extraction_documents(ex):
            if d not in docs:
                raise ValueError(f"{ex.id} : document inconnu {d}")
        unknown = extraction_series(ex) - set(series)
        if unknown:
            raise ValueError(f"{ex.id} : séries inconnues {sorted(unknown)}")
        extractions.append(ex)
    orphan = set(series) - set().union(*(extraction_series(e) for e in extractions))
    if orphan:
        raise ValueError(f"séries sans extraction : {sorted(orphan)}")
    return AttitudeRegistry(docs, series, waves, tuple(extractions))
