"""Exécution d'une extraction du registre : graphique, tableau, extrait recopié, parseurs dédiés.

- `chart` : `pdfchart.extract_chart` sur une page ; `exclusive: true` vérifie que les séries
  du graphique (modalités d'une même question) ne dépassent pas 100 % à une même vague, à
  l'arrondi près : une étiquette rangée dans la mauvaise colonne se voit ainsi.
- `table` : `pdfchart.extract_table`.
- `quote` : valeurs recopiées d'un document (texte d'un rapport, chiffre isolé d'une page),
  chacune avec son extrait `quote`. L'extrait doit figurer dans le texte de la page (comparaison
  sans espaces ni accents) et contenir la valeur ; pour `derived_from_delta`, la valeur publiée
  (`printed`) et l'écart (`delta`) doivent figurer dans l'extrait, et la valeur ancienne vaut
  `printed - delta`. Un extrait introuvable arrête la lecture.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

from unidecode import unidecode

from vvs.ingest.attitudes import Obs, pdfchart, raw
from vvs.ingest.attitudes.registry import AttitudeRegistry, Extraction


class ExtractionError(ValueError):
    pass


def run(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    try:
        if e.kind == "chart":
            return _chart(reg, e)
        if e.kind == "table":
            return _table(reg, e)
        if e.kind == "quote":
            return _quote(reg, e)
        if e.kind == "eurobarometer":
            from vvs.ingest.attitudes import eurobarometer

            return eurobarometer.extract(reg, e)
        if e.kind == "drees_microdata":
            from vvs.ingest.attitudes import drees

            return drees.extract(reg, e)
        if e.kind == "cncdh_chart":
            from vvs.ingest.attitudes import cncdh

            return cncdh.extract(reg, e)
    except pdfchart.ChartError as exc:
        raise ExtractionError(f"{e.id} : {exc}") from exc
    raise ExtractionError(f"{e.id} : kind {e.kind} non géré")


def _pdf_page(reg: AttitudeRegistry, e: Extraction):
    import pdfplumber

    path = raw.read(reg.documents[e.document]).path
    pdf = pdfplumber.open(path)
    return pdf, pdf.pages[e.page - 1]


def chart_spec(sp: dict) -> pdfchart.ChartSpec:
    return pdfchart.ChartSpec(
        series=[
            pdfchart.SeriesSpec(
                series_id=s["series_id"],
                color=tuple(s["color"]),
                bbox=s.get("bbox"),
                extra=s.get("extra") or [],
                missing=[str(w) for w in s.get("missing") or []],
            )
            for s in sp["series"]
        ],
        columns=sp.get("columns", "cluster"),
        waves=[str(w) for w in sp["waves"]] if sp.get("waves") else None,
        bbox=sp.get("bbox"),
        axis_bbox=sp.get("axis_bbox"),
        max_dist=sp.get("max_dist"),
    )


def _chart(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    sp = e.spec
    spec = chart_spec(sp)
    pdf, page = _pdf_page(reg, e)
    with pdf:
        got = pdfchart.extract_chart(page, spec)
    if sp.get("exclusive"):
        check_exclusive(e.id, got)
    notes = {str(k): v for k, v in (sp.get("notes") or {}).items()}
    return [
        Obs(sid, w, v, "chart_label", e.document, e.page, notes.get(w, ""))
        for sid, vals in got.items()
        for w, v in vals
    ]


def table_spec(sp: dict) -> pdfchart.TableSpec:
    return pdfchart.TableSpec(
        rows=sp["rows"],
        header_bbox=sp["header_bbox"],
        label_x1=sp["label_x1"],
        bbox=sp.get("bbox"),
        waves=[str(w) for w in sp["waves"]] if sp.get("waves") else None,
        header=sp.get("header", "month_year"),
    )


def check_exclusive(name: str, got: dict[str, list[tuple[str, float]]]) -> None:
    """Modalités d'une même question : somme ≤ 100 % (+ 1 point d'arrondi par modalité)."""
    totals: dict[str, float] = {}
    for vals in got.values():
        for w, v in vals:
            totals[w] = totals.get(w, 0.0) + v
    over = {w: t for w, t in totals.items() if t > 100 + len(got)}
    if over:
        raise ExtractionError(f"{name} : modalités au-delà de 100 % : {over}")


def _table(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    spec = table_spec(e.spec)
    pdf, page = _pdf_page(reg, e)
    with pdf:
        got = pdfchart.extract_table(page, spec)
    return [
        Obs(sid, w, v, "table", e.document, e.page) for sid, vals in got.items() for w, v in vals
    ]


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(text).lower())


def number_in(text: str, x: float) -> bool:
    """Vrai si le nombre `x` figure dans `text` (45 ; 45,5 ; 45.5), sans être un morceau d'un
    nombre plus long."""
    forms = [str(int(x))] if float(x).is_integer() else [f"{x:g}".replace(".", ","), f"{x:g}"]
    t = text.replace("\xa0", " ").replace(" ", " ")
    return any(re.search(rf"(?<![\d,.]){re.escape(f)}(?![\d]|[,.]\d)", t) for f in forms)


@cache
def page_text(path: Path, page: int | None) -> str:
    """Texte d'une page PDF (pdftotext -layout, sinon pdfplumber) ou d'une page HTML."""
    if path.suffix.lower() != ".pdf":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_bytes(), "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(" ")
    if shutil.which("pdftotext"):
        args = ["pdftotext", "-layout"]
        if page:
            args += ["-f", str(page), "-l", str(page)]
        r = subprocess.run([*args, str(path), "-"], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            return r.stdout
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [pdf.pages[page - 1]] if page else pdf.pages
        return "\n".join(p.extract_text() or "" for p in pages)


def _quote(reg: AttitudeRegistry, e: Extraction) -> list[Obs]:
    out = []
    for r in e.spec["rows"]:
        doc = r.get("document") or e.document
        page = r.get("page", e.page)
        path = raw.read(reg.documents[doc]).path
        text = squash(page_text(path, page))
        quote = r["quote"]
        for part in quote.split(" […] "):
            if squash(part) not in text:
                raise ExtractionError(f"{e.id} : extrait introuvable dans {doc} p. {page} : {part!r}")
        method = r.get("method", "printed")
        if method == "printed":
            value = float(r["value"])
            if not number_in(quote, value):
                raise ExtractionError(f"{e.id} : {value:g} absent de l'extrait {quote!r}")
        elif method == "derived_from_delta":
            printed, delta = float(r["printed"]), float(r["delta"])
            sign = "+" if delta > 0 else "-"
            if not number_in(quote, printed) or f"{sign}{abs(int(delta))}" not in quote.replace(" ", ""):
                raise ExtractionError(f"{e.id} : valeur {printed:g} ou écart {delta:+g} absent de {quote!r}")
            value = printed - delta
            if "value" in r and float(r["value"]) != value:
                raise ExtractionError(f"{e.id} : valeur déclarée {r['value']} ≠ {value:g}")
        else:
            raise ExtractionError(f"{e.id} : méthode {method}")
        out.append(Obs(r["series_id"], str(r["wave"]), value, method, doc, page, r.get("note", "")))
    return out


def verify_waves(reg: AttitudeRegistry, families: set[str]) -> int:
    """Vérifie que les dates de terrain du calendrier figurent dans leur document."""
    n = 0
    for fam in families:
        for label, w in reg.waves.get(fam, {}).items():
            if not w.document:
                continue
            if not w.quote:
                raise ExtractionError(f"vague {fam} {label} : document sans extrait `quote`")
            path = raw.read(reg.documents[w.document]).path
            text = squash(page_text(path, w.page))
            if not all(squash(part) in text for part in w.quote.split(" […] ")):
                raise ExtractionError(
                    f"vague {fam} {label} : « {w.quote} » introuvable dans {w.document} p. {w.page}"
                )
            n += 1
    return n
