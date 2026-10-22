"""Questions d'anticipation des sondages publiés -> data/curated/poll_expectations.parquet.

« Selon vous, qui sera élu ? », « qui va gagner ? », « pensez-vous que X peut remporter
l'élection ? » : la littérature des *citizen forecasts* (Graefe 2014, Murr 2011 et 2016,
Lewis-Beck et Stegmaier) trouve qu'aux États-Unis et au Royaume-Uni ces questions prévoient
le vainqueur mieux que les intentions de vote à longue échéance. Ce module recense et extrait
leurs résultats pour les présidentielles françaises, sans toucher au parseur des intentions de
vote (`commission`) dont il réutilise la lecture des PDF et des métadonnées.

Types de question (`question_type`) :

- `winner` : une seule réponse parmi les candidats (« qui sera élu », « quel candidat
  remportera », « pronostic de victoire ») ; les parts des candidats et la part sans candidat
  (`undecided` : ne sait pas, aucun) somment à 100 ;
- `chances` : posée candidat par candidat (« X peut-il remporter l'élection ? », « X a-t-il
  de grandes chances de devenir président ? », Ipsos : « il sera qualifié et élu / qualifié
  mais battu / pas qualifié ») ; les parts ne somment pas à 100 ;
- `wish` : souhait de victoire (« qui souhaitez-vous voir gagner ? ») ;
- `good_president` : « X ferait un bon président ».

`outcome` vaut `win` (victoire) ou `qualify` (qualification pour le second tour, posée par
Ipsos seulement). `pair` (« a|b », ids triés) marque un pronostic de duel : entre les deux tours,
ou conditionnel avant le premier tour (« si le second tour oppose Macron à Le Pen… »). `party`
remplace `candidate_id` quand la question porte sur un parti (Toluna Harris Interactive 2026).

Sources, toutes déclarées dans `data/registry/expectations_sources.yaml` :

- notices de la Commission des sondages (2017, 2022, 2027 ; `data/raw/commission/`) : page et
  forme du tableau déclarées, valeurs lues par `read_table` ; métadonnées (institut, terrain,
  échantillon) par `commission.parse.notice_metadata` ;
- rapports quotidiens du rolling Ifop 2022 (`data/raw/institutes/ifop/2022/`, collectés pour
  la sûreté du choix) : pronostic de victoire hebdomadaire ;
- pages web et PDF d'instituts (`web`), copiés dans `data/raw/expectations/` par
  `ingest_web` ; les valeurs sont recopiées dans le registre et `verify_web` vérifie que chaque
  nombre figure bien dans la copie brute.

`verified` : vrai si la mesure a été relue sur le document (champ `verified` du registre) et,
pour une lecture automatique, si les contrôles passent (nombre de candidats attendu, somme à
100 des colonnes complémentaires, total d'un `winner` entre 40 et 101) ; pour le web, si tous
les nombres figurent dans la copie brute. Une mesure non vérifiée reste dans la table, marquée.

`scan` recense toutes les pages des notices qui posent une question de ces quatre types, pour
mesurer la couverture et repérer les questions non déclarées.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml
from pdfminer.psparser import PSException
from pdfplumber.utils.exceptions import PdfminerException

from vvs import paths
from vvs.ingest.commission import notice as nt
from vvs.ingest.commission.aliases import norm_label
from vvs.ingest.commission.listing import RAW_DIR as COMMISSION_DIR
from vvs.ingest.commission.listing import USER_AGENT, archive_dir, load_index
from vvs.registry import default_registry

log = logging.getLogger(__name__)

SOURCES_PATH = paths.REGISTRY / "expectations_sources.yaml"
RAW_DIR = paths.RAW / "expectations"
IFOP_DIR = paths.RAW / "institutes" / "ifop"
OUT_PATH = paths.CURATED / "poll_expectations.parquet"
INVENTORY_PATH = paths.CURATED / "poll_expectations_inventory.parquet"

QUESTION_TYPES = ("winner", "chances", "wish", "good_president")
OUTCOMES = ("win", "qualify")
POPULATIONS = ("all", "registered", "certain_voters")
LAYOUTS = ("list", "split_name", "attribute", "labels", "manual")
ELECTIONS = (2017, 2022, 2027)

SCHEMA = pl.Schema(
    {
        "election": pl.Int32,
        "pollster": pl.Utf8,
        "sponsor": pl.Utf8,
        "fieldwork_start": pl.Date,
        "fieldwork_end": pl.Date,
        "question_type": pl.Utf8,
        "outcome": pl.Utf8,
        "candidate_id": pl.Utf8,
        "party": pl.Utf8,
        "pair": pl.Utf8,
        "share": pl.Float64,
        "undecided": pl.Float64,
        "sample_size": pl.Int32,
        "population": pl.Utf8,
        "source": pl.Utf8,
        "source_url": pl.Utf8,
        "file_id": pl.Int32,
        "page": pl.Int32,
        "layout": pl.Utf8,
        "verified": pl.Boolean,
        "note": pl.Utf8,
    }
)

# --------------------------------------------------------------------- repérage des questions
#
# Cherché dans le texte normalisé (`norm_label` : ASCII minuscule, ponctuation réduite à des
# espaces) d'une page. Les pages d'une primaire sont marquées (`primary`) et ne sont pas
# comptées comme pronostic de la présidentielle.

CUES: dict[str, re.Pattern] = {
    "winner": re.compile(
        r"\b(qui|quel candidat|lequel|laquelle|quelle personnalite)\b.{0,120}?"
        r"\b(sera elue?|l emportera|remportera|va gagner|va l emporter|va remporter"
        r"|le plus de chances d etre elue?)\b"
        r"|pronostic de victoire|pronostics? pour le (2nd|second) tour"
    ),
    "chances": re.compile(
        r"peut remporter l election|capable de remporter l election"
        r"|chances [a-z ]{0,30}de devenir le prochain president"
        r"|a des chances de remporter|pourrait etre elue? president"
        r"|sera qualifiee? pour le second tour|pensez vous qu?e? [a-z ]{0,40}"
        r"(remportera|va remporter) (la prochaine |l )election"
    ),
    "wish": re.compile(
        r"souhait\w* (vous )?voir (gagner|remporter|elue?)\b|souhait de victoire"
    ),
    "good_president": re.compile(
        r"\b(ferait|serait|est|sera) (un|une) (tres |assez )?(bon|bonne) (president|presidente)"
        r"|competences pour etre un bon president|ferait le meilleur president"
    ),
}
_PRIMARY = re.compile(r"\bprimaire\b")


def page_cues(page: str) -> dict[str, bool]:
    """Types de question d'anticipation posés sur une page, plus `primary`."""
    t = norm_label(page)
    out = {k: bool(rx.search(t)) for k, rx in CUES.items()}
    out["primary"] = bool(_PRIMARY.search(t))
    return out


# ------------------------------------------------------------------------ lecture des valeurs

_LEADERS = re.compile(r"[.…·]{2,}")
_VALUE = re.compile(r"^(<\s?\d+(?:[,.]\d+)?|\d{1,3}(?:[,.]\d+)?|-|–|—)%?\*?$")
_SKIP = re.compile(r"^([+\-−–]\s?\d+(?:[,.]\d+)?%?|\(?[+\-−–=]?\d*(?:[,.]\d+)?\)?|%|\(=\)|=)$")
_NONCANDIDATE = re.compile(
    r"^(nsp|ne sait pas|ne se prononce pas|ne se prononcent pas|sans reponse|sans opinion"
    r"|vous ne savez pas|aucun|aucune|non reponse|total)"
)


def _value(token: str) -> float | None:
    """« 37% » -> 37 ; « 3,6 » -> 3.6 ; « <0,5 » -> 0.25 (convention du projet) ; « - » -> 0
    (aucun répondant) ; sinon None."""
    t = token.strip().rstrip("*").rstrip("%")
    if t in ("-", "–", "—"):
        return 0.0
    if t.startswith("<"):
        return 0.25
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None


def _tokens(line: str) -> list[str]:
    line = _LEADERS.sub(" ", line.replace("•", " ").replace("…", " "))
    line = re.sub(r"(?<=\d)%", " %", line)
    return line.split()


def values_after(tokens: list[str]) -> list[float]:
    """Valeurs en tête de `tokens` : nombres et « - » ; évolutions signées, « % » et « (=) »
    ignorés ; arrêt au premier mot."""
    out: list[float] = []
    for t in tokens:
        if _VALUE.match(t):
            v = _value(t)
            if v is not None:
                out.append(v)
                continue
        if _SKIP.match(t):
            continue
        break
    return out


def find_names(tokens: list[str], lookup) -> list[tuple[str, int, int]]:
    """Candidats nommés dans une ligne : (id, début, fin) en indices de jetons, le plus long
    libellé d'abord (« Marine LE PEN » avant « Marine »)."""
    out = []
    i = 0
    while i < len(tokens):
        for j in range(min(len(tokens), i + 4), i, -1):
            cid = lookup(" ".join(tokens[i:j]))
            if cid:
                out.append((cid, i, j))
                i = j
                break
        else:
            i += 1
    return out


def region(page: str, anchor: str | None, stop: str | None) -> list[str]:
    """Lignes de la page entre la première ligne qui contient `anchor` (incluse) et la
    première ligne suivante qui contient `stop` (exclue), regex sur le texte normalisé."""
    lines = page.splitlines()
    start = 0
    if anchor:
        rx = re.compile(anchor)
        hits = [k for k, ln in enumerate(lines) if rx.search(norm_label(ln))]
        if not hits:
            raise TableError(f"ancre introuvable : {anchor!r}")
        start = hits[0]
    end = len(lines)
    if stop:
        rx = re.compile(stop)
        for k in range(start + 1, len(lines)):
            if rx.search(norm_label(lines[k])):
                end = k
                break
    return lines[start:end]


class TableError(ValueError):
    """Tableau déclaré illisible : ancre absente, candidats manquants, contrôle en échec."""


def read_list(lines: list[str], lookup) -> dict[str, list[float]]:
    """Lignes « Candidat v1 v2 … » : premières valeurs suivant chaque candidat nommé, première
    occurrence gardée (les légendes de graphique répètent les noms sans valeur ou avec la
    valeur du graphique placée avant le nom, ignorée)."""
    rows: dict[str, list[float]] = {}
    for line in lines:
        tokens = _tokens(line)
        names = find_names(tokens, lookup)
        for k, (cid, _, end) in enumerate(names):
            stop = names[k + 1][1] if k + 1 < len(names) else len(tokens)
            vals = values_after(tokens[end:stop])
            if vals and cid not in rows:
                rows[cid] = vals
    return rows


def read_split_name(lines: list[str], names: dict[str, tuple[str, str]]) -> dict[str, list[float]]:
    """Ifop (rolling 2022) : prénom, valeur puis NOM (« Jean-Luc / 5 / MÉLENCHON % (−3) »), ou
    prénom NOM valeur selon l'extraction. `names` : {id: (prénom, nom)}."""
    t = norm_label(" ".join(lines))
    rows: dict[str, list[float]] = {}
    for cid, (first, last) in names.items():
        f, ln = norm_label(first), norm_label(last)
        # l'évolution entre parenthèses, normalisée, peut s'intercaler : « 47 % (+1) MACRON »
        m = re.search(rf"\b{re.escape(f)} (\d{{1,3}})(?: \d{{1,2}})? {re.escape(ln)}\b", t) or re.search(
            rf"\b{re.escape(f)} {re.escape(ln)} (\d{{1,3}})\b", t
        )
        if m:
            rows[cid] = [float(m.group(1))]
    return rows


def read_attribute(lines: list[str], lookup, item: str) -> dict[str, list[float]]:
    """OpinionWay : bloc par candidat (« % Oui Jean-Luc Mélenchon ») puis une ligne par
    qualité ; on lit la ligne dont le libellé contient `item` (regex normalisée)."""
    rx = re.compile(item)
    rows: dict[str, list[float]] = {}
    current = None
    for line in lines:
        tokens = _tokens(line)
        names = find_names(tokens, lookup)
        if names:
            current = names[-1][0]
            continue
        if current and rx.search(norm_label(line)):
            label_end = next((k for k, tk in enumerate(tokens) if _VALUE.match(tk)), len(tokens))
            vals = values_after(tokens[label_end:])
            if vals and current not in rows:
                rows[current] = vals
    return rows


def read_labels(lines: list[str], label: str) -> list[list[float]]:
    """Lignes de modalités dont le libellé correspond à `label` (« ^oui ») : valeurs de
    chacune, dans l'ordre."""
    rx = re.compile(label)
    out = []
    for line in lines:
        tokens = _tokens(line)
        k = next((k for k, tk in enumerate(tokens) if _VALUE.match(tk) and _value(tk) is not None
                  and k > 0), None)
        if k is None:
            continue
        if rx.search(norm_label(" ".join(tokens[:k]))):
            out.append(values_after(tokens[k:]))
    return out


# --------------------------------------------------------------------------- registre

@dataclass
class Measure:
    outcome: str = "win"
    columns: tuple[int, ...] = (0,)


@dataclass
class Source:
    """Une question déclarée dans le registre (une page de notice, de rapport ou une page web)."""

    election: int
    kind: str  # commission, ifop_report, web
    question_type: str
    layout: str
    population: str
    verified: bool
    file_id: int | None = None
    file: str | None = None  # rapport Ifop ou copie web, relatif à son dossier brut
    page: int | None = None
    anchor: str | None = None
    stop: str | None = None
    item: str | None = None
    label: str | None = None
    candidate: str | None = None
    party: str | None = None
    pair: tuple[str, str] | None = None
    measures: list[Measure] = field(default_factory=lambda: [Measure()])
    row_total: tuple[int, ...] | None = None
    drop: tuple[str, ...] = ()
    n_candidates: int | None = None
    values: dict | None = None  # layout manual : {id: part} ou {outcome: {id: part}}
    undecided: float | None = None
    pollster: str | None = None
    sponsor: str | None = None
    fieldwork: tuple[dt.date, dt.date] | None = None
    sample_size: int | None = None
    url: str | None = None
    note: str | None = None


def _date(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def load_sources(path: Path = SOURCES_PATH) -> list[Source]:
    """Lit et contrôle le registre : types, formes, ids de candidats connus."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    reg = default_registry()
    out = []
    for kind in ("commission", "ifop_report", "web"):
        for d in raw.get(kind) or []:
            d = dict(d)
            ms = [Measure(m.get("outcome", "win"), tuple(m.get("columns", [0])))
                  for m in d.pop("measures", [{}])]
            fw = d.pop("fieldwork", None)
            pair = d.pop("pair", None)
            rt = d.pop("row_total", None)
            d["drop"] = tuple(d.get("drop", ()))
            s = Source(kind=kind, measures=ms, **d)
            s.fieldwork = (_date(fw[0]), _date(fw[1])) if fw else None
            s.pair = tuple(sorted(pair)) if pair else None
            s.row_total = tuple(rt) if rt else None
            if s.question_type not in QUESTION_TYPES:
                raise ValueError(f"type de question inconnu : {s.question_type} ({d})")
            if s.layout not in LAYOUTS:
                raise ValueError(f"forme inconnue : {s.layout} ({d})")
            if s.population not in POPULATIONS:
                raise ValueError(f"population inconnue : {s.population} ({d})")
            for m in s.measures:
                if m.outcome not in OUTCOMES:
                    raise ValueError(f"issue inconnue : {m.outcome} ({d})")
            ids = list(s.pair or ()) + ([s.candidate] if s.candidate else []) + list(s.drop)
            for v in (s.values or {}).values():
                ids += list(v) if isinstance(v, dict) else []
            if s.values and not s.party and not any(isinstance(v, dict) for v in s.values.values()):
                ids += list(s.values)
            for cid in ids:
                if cid not in reg.candidates:
                    raise ValueError(f"candidat inconnu : {cid} ({d})")
            if kind == "web" and not (s.url and s.file and s.pollster and s.fieldwork):
                raise ValueError(f"source web incomplète : {d}")
            out.append(s)
    return out


# ------------------------------------------------------------------------- extraction

def candidate_names(election: int | None = None) -> dict[str, tuple[str, str]]:
    reg = default_registry()
    return {cid: (c.first_name, c.last_name) for cid, c in reg.candidates.items()}


def _lookup():
    from vvs.ingest.certainty import candidate_labels
    from vvs.ingest.commission.certainty import candidate_lookup

    return candidate_lookup(candidate_labels())


def read_table(page: str, s: Source, lookup) -> tuple[dict[str, dict[str, float]], float | None]:
    """Valeurs d'une question déclarée sur le texte d'une page : ({outcome: {id: part}},
    part sans candidat). Lève `TableError` si un contrôle échoue."""
    lines = region(page, s.anchor, s.stop)
    if s.layout == "list":
        rows = read_list(lines, lookup)
    elif s.layout == "split_name":
        rows = read_split_name(lines, candidate_names())
    elif s.layout == "attribute":
        rows = read_attribute(lines, lookup, s.item)
    elif s.layout == "labels":
        rows = {s.candidate: [sum(v) for v in zip(*read_labels(lines, s.label), strict=False)]}
        if not rows[s.candidate]:
            raise TableError(f"aucune modalité {s.label!r}")
    else:
        raise TableError(f"forme non lisible automatiquement : {s.layout}")
    if s.pair:
        rows = {c: v for c, v in rows.items() if c in s.pair}
    # lignes déclarées illisibles (valeurs collées par la mise en page, « 2 5 » lu « 25 ») :
    # écartées par règle explicite du registre, jamais corrigées
    rows = {c: v for c, v in rows.items() if c not in s.drop}
    if s.n_candidates is not None and len(rows) != s.n_candidates:
        raise TableError(f"{len(rows)} candidats lus, {s.n_candidates} attendus : {sorted(rows)}")
    if s.row_total:
        bad = [c for c, v in rows.items()
               if len(v) <= max(s.row_total) or abs(sum(v[i] for i in s.row_total) - 100) > 2]
        if bad:
            raise TableError(f"lignes dont les colonnes {s.row_total} ne somment pas à 100 : {bad}")
    out: dict[str, dict[str, float]] = {}
    for m in s.measures:
        vals = {}
        for c, v in rows.items():
            if len(v) <= max(m.columns):
                raise TableError(f"{c} : {len(v)} valeurs, colonne {max(m.columns)} demandée")
            vals[c] = float(sum(v[i] for i in m.columns))
        out[m.outcome] = vals
    undecided = None
    if s.question_type == "winner":
        # part sans candidat (ne sait pas, aucun) par différence : les lignes « NSP » des
        # questions voisines sur la même page (OpinionWay) rendent leur lecture peu sûre
        total = sum(out[s.measures[0].outcome].values())
        if not 40 <= total <= 101:
            raise TableError(f"total des candidats hors bornes : {total:g}")
        undecided = max(0.0, 100 - total)
    return out, undecided


def manual_values(s: Source) -> dict[str, dict[str, float]]:
    vals = s.values or {}
    if any(isinstance(v, dict) for v in vals.values()):
        return {o: {c: float(x) for c, x in v.items()} for o, v in vals.items()}
    return {s.measures[0].outcome: {c: float(x) for c, x in vals.items()}}


# ------------------------------------------------------------------------- web

def web_path(s: Source, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / s.file


def ingest_web(sources: list[Source] | None = None, raw_dir: Path = RAW_DIR,
               delay_s: float = 2.0) -> dict:
    """Copie brute des pages et PDF déclarés (`web`) dans data/raw/expectations/.

    Idempotent, en ajout seul : un fichier présent n'est pas retéléchargé ; `index.jsonl` garde
    l'URL, la date et l'empreinte sha256 de chaque copie."""
    import httpx

    sources = [s for s in (sources or load_sources()) if s.kind == "web"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    index = raw_dir / "index.jsonl"
    stats = {"downloaded": 0, "present": 0, "failed": []}
    from vvs.ingest.certainty import _tls_bundle

    # ifopgroup.com n'envoie pas son certificat intermédiaire : magasin certifi complété (voir
    # `ingest.certainty`), vérification TLS complète
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60,
                          verify=str(_tls_bundle()))
    seen = set()
    with index.open("a") as out:
        for s in sources:
            p = web_path(s, raw_dir)
            if s.file in seen or p.exists():
                stats["present"] += p.exists() and s.file not in seen
                seen.add(s.file)
                continue
            seen.add(s.file)
            try:
                r = client.get(s.url)
                r.raise_for_status()
            except httpx.HTTPError as exc:
                stats["failed"].append((s.url, str(exc)))
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(r.content)
            out.write(json.dumps({
                "file": s.file, "url": s.url, "final_url": str(r.url),
                "content_type": r.headers.get("content-type"),
                "sha256": hashlib.sha256(r.content).hexdigest(),
                "ingested_at": dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(),
            }, ensure_ascii=False) + "\n")
            stats["downloaded"] += 1
            time.sleep(delay_s)
    return stats


def document_text(path: Path) -> str:
    """Texte d'une copie brute : PDF (pdfplumber) ou HTML (texte visible)."""
    if path.suffix.lower() == ".pdf":
        return "\n".join(nt.extract_text(path, column_pages=0).pages)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return soup.get_text(" ")


def _number_forms(x: float) -> list[str]:
    if float(x).is_integer():
        return [str(int(x))]
    s = f"{x:g}"
    return [s, s.replace(".", ",")]


def verify_web(s: Source, text: str) -> list[str]:
    """Nombres déclarés absents du texte (copie brute ou page de notice ; liste vide : tout
    est retrouvé). Contrôle faible, contre les fautes de recopie, pas contre l'attribution
    d'un nombre au mauvais candidat."""
    found = set(re.findall(r"\d+(?:[,.]\d+)?", text))
    missing = []
    for o, vals in manual_values(s).items():
        for c, x in vals.items():
            if not any(f in found for f in _number_forms(x)):
                missing.append(f"{o}:{c}={x:g}")
    return missing


# ------------------------------------------------------------------------- notices

def _entry(s: Source) -> dict:
    d = COMMISSION_DIR if s.election == 2027 else archive_dir(s.election, COMMISSION_DIR)
    index = load_index(d / "index.jsonl")
    if s.file_id not in index:
        raise KeyError(f"notice {s.file_id} absente de l'index {s.election}")
    return index[s.file_id] | {"_path": str(d / index[s.file_id]["file"])}


@dataclass
class Extracted:
    source: Source
    values: dict[str, dict[str, float]] = field(default_factory=dict)
    undecided: float | None = None
    pollster: str | None = None
    sponsor: str | None = None
    fieldwork: tuple[dt.date, dt.date] | None = None
    sample_size: int | None = None
    url: str | None = None
    checks_ok: bool = False
    error: str | None = None


def _ifop_fieldwork(text: nt.NoticeText, published: dt.date) -> tuple[dt.date, dt.date]:
    fw = nt.find_fieldwork("\n".join(text.pages[-6:]), require_cue=True,
                           window=(published - dt.timedelta(days=10), published))
    return fw or (published - dt.timedelta(days=3), published - dt.timedelta(days=1))


def extract(s: Source, lookup=None) -> Extracted:
    """Valeurs et métadonnées d'une source déclarée ; erreurs capturées, jamais avalées."""
    from vvs.ingest.certainty import entry_with_pollster
    from vvs.ingest.commission.parse import notice_metadata

    lookup = lookup or _lookup()
    res = Extracted(s, pollster=s.pollster, sponsor=s.sponsor, fieldwork=s.fieldwork,
                    sample_size=s.sample_size, url=s.url)
    try:
        if s.kind == "web":
            res.values, res.undecided = manual_values(s), s.undecided
            p = web_path(s)
            if not p.exists():
                res.error = f"copie brute absente : {p.name} (lancer vvs ingest expectations)"
                return res
            missing = verify_web(s, document_text(p))
            res.checks_ok = not missing
            if missing:
                res.error = "nombres absents de la copie brute : " + ", ".join(missing)
            return res
        if s.kind == "ifop_report":
            path = IFOP_DIR / str(s.election) / s.file
            text = nt.extract_text(path, column_pages=0)
            entry = next(r for r in _jsonl(IFOP_DIR / str(s.election) / "index.jsonl")
                         if r["file"] == s.file)
            res.url = entry["url"]
            res.pollster = res.pollster or "ifop"
            res.sponsor = res.sponsor or "Paris Match, LCI, Sud Radio"
            res.fieldwork = res.fieldwork or _ifop_fieldwork(
                text, dt.date.fromisoformat(entry["publish_date"]))
        else:
            entry = entry_with_pollster(_entry(s))
            text = nt.extract_text(Path(entry["_path"]), column_pages=2)
            meta = notice_metadata(text, entry)
            res.url = entry["url"]
            res.pollster = res.pollster or meta.pollster
            res.sponsor = res.sponsor or meta.sponsor
            res.sample_size = res.sample_size or meta.sample_size
            if res.fieldwork is None and meta.fieldwork_end:
                res.fieldwork = (meta.fieldwork_start or meta.fieldwork_end, meta.fieldwork_end)
        if s.layout == "manual":
            # valeurs recopiées (graphique, colonnes mêlées) : chacune doit figurer sur la page
            res.values, res.undecided = manual_values(s), s.undecided
            missing = verify_web(s, text.pages[s.page - 1]) if s.page else []
            res.checks_ok = not missing
            if missing:
                res.error = "nombres absents de la page : " + ", ".join(missing)
        else:
            if not s.page or s.page > len(text.pages):
                raise TableError(f"page {s.page} hors du document ({len(text.pages)} pages)")
            res.values, res.undecided = read_table(text.pages[s.page - 1], s, lookup)
            res.checks_ok = True
        if res.fieldwork is None:
            res.error = "dates de terrain introuvables"
    except (TableError, KeyError, OSError, ValueError, PSException, PdfminerException) as exc:
        res.error = f"{type(exc).__name__} : {exc}"
    return res


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _extract_one(s: Source) -> Extracted:
    return extract(s)


def build_table(extracted: list[Extracted]) -> pl.DataFrame:
    rows = []
    for x in extracted:
        s = x.source
        if x.fieldwork is None or not x.values:
            continue
        verified = bool(s.verified and x.checks_ok and x.error is None)
        base = {
            "election": s.election,
            "pollster": x.pollster,
            "sponsor": x.sponsor,
            "fieldwork_start": x.fieldwork[0],
            "fieldwork_end": x.fieldwork[1],
            "question_type": s.question_type,
            "pair": "|".join(s.pair) if s.pair else None,
            "sample_size": x.sample_size,
            "population": s.population,
            "source": s.kind,
            "source_url": x.url,
            "file_id": s.file_id,
            "page": s.page,
            "layout": s.layout,
            "verified": verified,
            "note": s.note,
        }
        for outcome, vals in x.values.items():
            for cid, share in vals.items():
                party = s.party if s.party else None
                rows.append(base | {
                    "outcome": outcome,
                    "candidate_id": None if party else cid,
                    "party": party,
                    "share": share,
                    "undecided": x.undecided if outcome == s.measures[0].outcome else None,
                })
    return pl.DataFrame(rows, schema=SCHEMA).sort(
        "election", "fieldwork_end", "pollster", "question_type", "outcome", "share",
        descending=[False, False, False, False, False, True])


# ------------------------------------------------------------------------- inventaire

def _scan_one(args) -> list[dict]:
    path, election, entry = args
    try:
        pages = nt.extract_text(Path(path), column_pages=0).pages
    except (OSError, ValueError, PSException, PdfminerException) as exc:
        return [{"election": election, "file_id": entry.get("file_id"), "title": entry["title"],
                 "page": None, "error": str(exc)}]
    out = []
    for k, page in enumerate(pages, start=1):
        cues = page_cues(page)
        if any(cues[t] for t in QUESTION_TYPES):
            out.append({"election": election, "file_id": entry.get("file_id"),
                        "title": entry["title"], "page": k, "error": None} | cues)
    return out


def scan(elections=ELECTIONS, workers: int = 8) -> pl.DataFrame:
    """Pages des notices (et rapports Ifop 2022) qui posent une question d'anticipation."""
    jobs = []
    for e in elections:
        d = COMMISSION_DIR if e == 2027 else archive_dir(e, COMMISSION_DIR)
        for _, entry in sorted(load_index(d / "index.jsonl").items()):
            p = d / entry["file"]
            if p.suffix == ".pdf":
                jobs.append((str(p), e, entry))
    ifop = IFOP_DIR / "2022" / "index.jsonl"
    if 2022 in elections and ifop.exists():
        for r in _jsonl(ifop):
            jobs.append((str(IFOP_DIR / "2022" / r["file"]), 2022,
                         {"file_id": None, "title": r["file"]}))
    with cf.ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_scan_one, jobs, chunksize=4) for r in part]
    schema = {"election": pl.Int32, "file_id": pl.Int32, "title": pl.Utf8, "page": pl.Int32,
              "error": pl.Utf8} | {k: pl.Boolean for k in (*QUESTION_TYPES, "primary")}
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None)


# ------------------------------------------------------------------------- exécution

@dataclass
class ExpectationsReport:
    extracted: list[Extracted]
    table: pl.DataFrame
    inventory: pl.DataFrame | None = None

    def summary(self) -> str:
        t = self.table
        lines = [(f"{t.height} lignes dans {OUT_PATH.name}, "
                 f"{t.select('election', 'pollster', 'fieldwork_end', 'question_type', 'pair', 'page', 'file_id').unique().height}"
                 " mesures (sondage x question)")]
        by = (t.group_by("election", "question_type", "outcome")
              .agg(pl.struct("pollster", "fieldwork_end", "pair", "page", "file_id")
                   .n_unique().alias("mesures"),
                   pl.col("verified").all().alias("toutes_verifiees"))
              .sort("election", "question_type", "outcome"))
        lines.append(str(by))
        for x in self.extracted:
            if x.error:
                s = x.source
                lines.append(f"  ÉCHEC {s.kind} {s.election} {s.file_id or s.file} p{s.page} "
                             f"{s.question_type} : {x.error}")
        if self.inventory is not None:
            declared = {(x.source.election, x.source.file_id)
                        for x in self.extracted if x.source.kind == "commission"}
            inv = self.inventory.filter(~pl.col("primary") & pl.col("file_id").is_not_null()
                                        & (pl.col("winner") | pl.col("chances")))
            todo = (inv.filter(~pl.struct("election", "file_id").map_elements(
                        lambda r: (r["election"], r["file_id"]) in declared,
                        return_dtype=pl.Boolean))
                    .group_by("election", "file_id", "title").agg(pl.col("page"))
                    .sort("election", "file_id"))
            lines.append(f"notices posant « qui va gagner » ou « chances » sans source déclarée "
                         f"(questionnaire seul, doublon ou oubli) : {todo.height}")
            for r in todo.iter_rows(named=True):
                lines.append(f"  {r['election']} {r['file_id']} {r['title']} p{r['page']}")
        return "\n".join(lines)


def run(write: bool = True, workers: int = 8, with_scan: bool = True) -> ExpectationsReport:
    sources = load_sources()
    with cf.ProcessPoolExecutor(workers) as ex:
        extracted = list(ex.map(_extract_one, sources, chunksize=2))
    table = build_table(extracted)
    inventory = scan(workers=workers) if with_scan else None
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        table.write_parquet(OUT_PATH)
        if inventory is not None:
            inventory.write_parquet(INVENTORY_PATH)
    return ExpectationsReport(extracted, table, inventory)


def load(path: Path = OUT_PATH) -> pl.DataFrame:
    return pl.read_parquet(path)
