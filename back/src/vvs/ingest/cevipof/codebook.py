"""Codebooks DDI publics des enquêtes du Cevipof (data/raw/cevipof/<key>/*.xml).

Deux variantes coexistent sur l'entrepôt : l'export Dataverse (DDI 1.2.2, espace de noms
ICPSR) et les codebooks du CDSP (DDI 2.5, `*.doc.xml`). Les deux ont la même structure utile :
`<var name=…>` avec `<labl>`, `<qstn><qstnLit>` et des `<catgry>` (`<catValu>`, `<labl>`,
`<catStat type="freq">`, effectifs bruts non pondérés). On ignore les espaces de noms.

`find_questions()` repère les variables utiles par motifs sur le libellé et le texte de la
question ; c'est une aide à l'inventaire (cevipof_datasets.yaml, `questions`), relue à la
main, pas une règle de parsing : le parsing passe par les noms de variables explicites de
aliases_cevipof.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from vvs.ingest.cevipof.ingest import raw_dir
from vvs.ingest.cevipof.registry import Dataset


@dataclass(frozen=True)
class Category:
    value: str
    label: str
    freq: int | None


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    question: str
    categories: tuple[Category, ...] = field(default_factory=tuple)

    def text(self) -> str:
        return f"{self.label} {self.question}"


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _text(el) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _child(el, name: str):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def parse_ddi(path: Path) -> list[Variable]:
    """Variables d'un codebook DDI (1.2.2 ou 2.5)."""
    tree = etree.parse(str(path), etree.XMLParser(huge_tree=True, resolve_entities=False))
    out = []
    for var in tree.iter():
        if _local(var.tag) != "var":
            continue
        name = (var.get("name") or "").strip()
        if not name:
            continue
        qstn = _child(var, "qstn")
        cats = []
        for c in var:
            if _local(c.tag) != "catgry":
                continue
            freq = None
            for s in c:
                if _local(s.tag) == "catStat" and s.get("type") == "freq":
                    try:
                        freq = int(float(_text(s)))
                    except ValueError:
                        freq = None
            cats.append(Category(_text(_child(c, "catValu")), _text(_child(c, "labl")), freq))
        out.append(
            Variable(
                name=name,
                label=_text(_child(var, "labl")),
                question=_text(_child(qstn, "qstnLit")) if qstn is not None else "",
                categories=tuple(cats),
            )
        )
    return out


def codebook_paths(ds: Dataset) -> list[Path]:
    d = raw_dir(ds.key)
    return [
        d / f.name.strip()
        for f in ds.files
        if f.role == "codebook"
        and f.name.strip().endswith(".xml")
        and (d / f.name.strip()).exists()
    ]


def load_codebook(ds: Dataset) -> list[Variable]:
    """Variables du premier codebook XML présent (export Dataverse de préférence)."""
    paths = codebook_paths(ds)
    if not paths:
        return []
    # l'export Dataverse (fr_cdsp_ddi_*.xml) colle aux noms du fichier de données
    paths.sort(key=lambda p: (".doc." in p.name, p.name))
    return parse_ddi(paths[0])


# --- repérage des questions utiles

PATTERNS: dict[str, re.Pattern] = {
    "vote_round1": re.compile(
        r"(vote|voté).{0,60}(1er|premier) tour|(1er|premier) tour.{0,40}(vote|voté)", re.IGNORECASE
    ),
    "vote_round2": re.compile(
        r"(vote|voté).{0,60}(2nd|2e|second|deuxième) tour|(2nd|2e|second|deuxième) tour.{0,40}(vote|voté)",
        re.IGNORECASE,
    ),
    "duel_intention": re.compile(
        r"(si|au cas où).{0,80}second tour.{0,120}(oppos|face|entre)|duel|hypoth", re.IGNORECASE
    ),
    "turnout_intent": re.compile(
        r"(certain|intention|irez|aller).{0,40}(vot|aller voter)|participation", re.IGNORECASE
    ),
    "certainty": re.compile(
        r"(sûr|certain|définitif|changer d.avis|pourrait changer)", re.IGNORECASE
    ),
    "expected_winner": re.compile(
        r"(qui|quel candidat).{0,40}(sera|va être|l.emporter|gagner|élu)|pronostic|selon vous.{0,40}élu",
        re.IGNORECASE,
    ),
    "potential": re.compile(
        r"(probabilit|possible|envisag).{0,60}(vot|candidat)|pourriez.vous voter|0 à 10.{0,60}vot",
        re.IGNORECASE,
    ),
    "weight": re.compile(r"pond|poids|weight|redress", re.IGNORECASE),
}


def find_questions(variables: list[Variable]) -> dict[str, list[Variable]]:
    found: dict[str, list[Variable]] = {k: [] for k in PATTERNS}
    for v in variables:
        t = v.text()
        for role, pat in PATTERNS.items():
            if role == "weight":
                if pat.search(v.name) or pat.search(v.label):
                    found[role].append(v)
            elif pat.search(t):
                found[role].append(v)
    return {k: v for k, v in found.items() if v}
