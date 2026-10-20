"""Potentiel électoral et rejet : data/registry/potential.yaml -> electoral_potential.parquet.

Mesures
-------

Le « potentiel électoral » est la part des personnes interrogées qui « pourraient voter » pour
une personnalité ; le « rejet » la part qui l'excluent. Les instituts ne posent pas la même
question et ne coupent pas au même endroit, d'où une colonne `measure` (institut × échelle)
qui sert à la normalisation (`vvs.model.potential`) :

- `binary` (Toluna Harris Interactive, suivi pour L'Opinion) : « pourriez-vous voter pour
  chacune des personnalités suivantes ? », oui ou non ; potentiel = oui. Le non n'est pas
  publié (il comprend les « ne connaît pas ») : `reject_share` reste vide.
- `four_level` (Ipsos, OpinionWay, Elabe, Odoxa…) : quatre modalités du certain à l'exclu ;
  potentiel = les deux modalités positives, rejet = la modalité la plus négative (« aucune
  chance », « vous excluez », « certainement pas »). La définition exacte est dans
  `potential_def` / `reject_def` de chaque source.
- `zero_ten` (Toluna Harris Interactive pour RTL, Cluster17, Ipsos-Cevipof) : note de 0 à 10 ;
  potentiel = notes 7 à 10, rejet = notes 0 à 3, `firm_share` = notes 9 à 10.
- `five_level` (Kantar 2017 et 2022) : « je ne voterai jamais » / peu probable / possible /
  fortes chances / certain ; potentiel = possible + fortes chances + certain, rejet = jamais.
  Échelle de référence de la normalisation.
- `attitude` (Odoxa « vous la rejetez », Ipsos « mécontent si élu(e) ») : pas une question de
  vote ; gardée dans la table, exclue de la normalisation et du modèle.

Sources et stockage brut
------------------------

`data/registry/potential.yaml` liste les sources (une par sondage publié). Deux cas :

- notice de la Commission des sondages déjà téléchargée (`raw: commission/<id>.pdf`) et lue
  par un parseur (`parser`) : tableaux réguliers de Toluna Harris Interactive ;
- toute autre source (PDF d'institut, article de presse) : chiffres relevés à la main dans
  `rows`, chacun avec un extrait (`quote`) recopié du document. `vvs ingest potential`
  télécharge l'`url` dans `data/raw/potential/<id>.<ext>` avec `<id>.meta.json` (URL, sha256,
  horodatage) ; un fichier présent n'est jamais réécrit. `vvs parse potential` vérifie que
  chaque extrait figure dans le texte du brut et que chaque chiffre figure dans son extrait
  (colonne `verified`) : un relevé non vérifiable reste dans la table, signalé.

Électorats (`electorate`) : `all` pour l'ensemble ; `party:<parti>` pour les sympathisants
(lfi, ps, ecologistes, renaissance, lr, rn…) ; `vote_r1:<candidate_id>` pour les intentions de
vote de premier tour de la même enquête ; `vote_<année>:<candidate_id>` pour le vote passé.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
import yaml
from pydantic import Field, model_validator

from vvs import paths
from vvs.registry.models import ID_RE, _Model

REGISTRY_PATH = paths.REGISTRY / "potential.yaml"
RAW_DIR = paths.RAW / "potential"
OUTPUT_NAME = "electoral_potential.parquet"
USER_AGENT = "vingtvingtsept/0.1 (French 2027 presidential forecast; https://github.com/Orolol)"

Scale = Literal["binary", "four_level", "five_level", "zero_ten", "attitude", "other"]
Population = Literal["registered", "all", "certain_voters", "unknown"]
Parser = Literal["harris_binary", "harris_zero_ten"]

# Partis des pages « sympathisants » du suivi Toluna Harris Interactive, dans l'ordre des
# panneaux (logos sans texte : ordre vérifié à l'image, voir `PARTY_PAGES`)
PARTY_BLOCS: dict[str, str] = {
    "lfi": "far_left",
    "pcf": "far_left",
    "lfi_pcf": "far_left",
    "fg": "far_left",
    "ps": "left",
    "ecologistes": "left",
    "renaissance": "centre",
    "modem": "centre",
    "horizons": "centre",
    "lr": "right",
    "udi": "right",
    "rn": "far_right",
    "dlf": "far_right",
    "reconquete": "far_right",
}

SCHEMA = {
    "election": pl.Int64,
    "source_id": pl.Utf8,
    "pollster": pl.Utf8,
    "sponsor": pl.Utf8,
    "fieldwork_start": pl.Date,
    "date": pl.Date,
    "date_approx": pl.Boolean,
    "sample_size": pl.Int64,
    "population": pl.Utf8,
    "round": pl.Int64,
    "scale": pl.Utf8,
    "measure": pl.Utf8,
    "candidate_id": pl.Utf8,
    "candidate_label": pl.Utf8,
    "electorate": pl.Utf8,
    "electorate_kind": pl.Utf8,
    "potential_share": pl.Float64,
    "reject_share": pl.Float64,
    "firm_share": pl.Float64,
    "verified": pl.Boolean,
    "source_url": pl.Utf8,
    "raw_file": pl.Utf8,
    "raw_sha256": pl.Utf8,
}


class PotentialError(ValueError):
    """Incohérence du registre des potentiels ou d'un brut."""


# --------------------------------------------------------------------------- registre


class PotentialRow(_Model):
    candidate: str  # candidate_id du registre, ou libellé pour une personnalité hors registre
    electorate: str = "all"
    electorate_label: str | None = None
    potential: float | None = Field(default=None, ge=0, le=100)
    reject: float | None = Field(default=None, ge=0, le=100)
    firm: float | None = Field(default=None, ge=0, le=100)
    # modalités publiées dont `potential` est la somme (quand le total n'est pas publié) :
    # c'est elles que la vérification cherche dans l'extrait
    potential_parts: list[float] = Field(default_factory=list)
    quote: str | None = None

    @model_validator(mode="after")
    def _some_value(self) -> PotentialRow:
        if self.potential is None and self.reject is None:
            raise ValueError(f"{self.candidate} : ni potentiel ni rejet")
        if self.potential_parts and abs(sum(self.potential_parts) - (self.potential or 0)) > 1e-6:
            raise ValueError(f"{self.candidate} : potentiel ≠ somme de potential_parts")
        both = self.potential is not None and self.reject is not None
        if both and self.potential + self.reject > 100.5:
            raise ValueError(f"{self.candidate} : potentiel + rejet > 100")
        return self


class PotentialSource(_Model):
    id: str
    election: int
    pollster: str
    sponsor: str | None = None
    fieldwork_start: dt.date | None = None
    fieldwork_end: dt.date | None = None
    date_approx: bool = False
    sample_size: int | None = Field(default=None, gt=0)
    population: Population = "unknown"
    round: Literal[1, 2] = 1
    question: str
    scale: Scale
    measure: str | None = None  # défaut : <pollster>_<scale>
    potential_def: str | None = None
    reject_def: str | None = None
    url: str
    secondary_urls: list[str] = Field(default_factory=list)
    raw: str | None = None  # chemin relatif à data/raw (défaut : potential/<id>.<ext>)
    parser: Parser | None = None
    notes: str | None = None
    rows: list[PotentialRow] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> PotentialSource:
        if not ID_RE.match(self.id.replace("-", "_")):
            raise ValueError(f"{self.id} : id mal formé")
        if self.parser is None and not self.rows:
            raise ValueError(f"{self.id} : ni parseur ni lignes relevées")
        if self.parser is not None and self.raw is None:
            raise ValueError(f"{self.id} : un parseur exige `raw`")
        if self.fieldwork_start and self.fieldwork_end and self.fieldwork_end < self.fieldwork_start:
            raise ValueError(f"{self.id} : terrain inversé")
        return self

    @property
    def measure_key(self) -> str:
        return self.measure or f"{self.pollster}_{self.scale}"


class PotentialFile(_Model):
    sources: list[PotentialSource]


def load_sources(path: Path | None = None) -> list[PotentialSource]:
    path = REGISTRY_PATH if path is None else Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        sources = PotentialFile.model_validate(data).sources
    except ValueError as e:
        raise PotentialError(f"{path.name} invalide : {e}") from e
    ids = [s.id for s in sources]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise PotentialError(f"{path.name} : ids en double {dup}")
    return sources


# --------------------------------------------------------------------------- brut


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_EXT_BY_TYPE = {"application/pdf": ".pdf", "text/html": ".html", "application/xhtml+xml": ".html"}


def raw_path(source: PotentialSource, raw_root: Path | None = None) -> Path | None:
    """Brut d'une source : `raw` explicite, sinon le fichier téléchargé s'il existe."""
    root = paths.RAW if raw_root is None else Path(raw_root)
    if source.raw:
        return root / source.raw
    for p in sorted((root / "potential").glob(f"{source.id}.*")):
        if not p.name.endswith(".meta.json"):
            return p
    return None


def download(
    source: PotentialSource, raw_root: Path | None = None, client=None
) -> tuple[Path, str]:
    """Télécharge `source.url` dans data/raw/potential/ si absent ; (chemin, statut)."""
    import httpx

    existing = raw_path(source, raw_root)
    if existing is not None and existing.exists():
        return existing, "present"
    if source.raw:
        raise PotentialError(f"{source.id} : brut {source.raw} absent (source non téléchargeable ici)")
    root = (paths.RAW if raw_root is None else Path(raw_root)) / "potential"
    own = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True
    )
    try:
        r = client.get(source.url)
        r.raise_for_status()
    finally:
        if own:
            client.close()
    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
    ext = _EXT_BY_TYPE.get(ctype) or (".pdf" if r.content[:4] == b"%PDF" else ".html")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{source.id}{ext}"
    meta_path = root / f"{source.id}.meta.json"
    if meta_path.exists():
        raise FileExistsError(f"{meta_path} existe sans brut : ingestion interrompue ?")
    path.write_bytes(r.content)
    meta = {
        "source_id": source.id,
        "url": source.url,
        "final_url": str(r.url),
        "content_type": ctype,
        "bytes": len(r.content),
        "sha256": sha256_of(path),
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, "downloaded"


def ingest(
    sources: list[PotentialSource] | None = None, raw_root: Path | None = None, log=print
) -> dict[str, list[str]]:
    """`vvs ingest potential` : télécharge les bruts manquants ; rien n'est réécrit."""
    sources = load_sources() if sources is None else sources
    out: dict[str, list[str]] = {"present": [], "downloaded": [], "failed": []}
    for s in sources:
        try:
            path, status = download(s, raw_root)
        except Exception as e:  # noqa: BLE001 — on continue, l'échec est rapporté
            out["failed"].append(f"{s.id} : {e}")
            log(f"échec     {s.id} : {e}")
            continue
        out[status].append(s.id)
        log(f"{status:10s} {s.id} -> {path.name}")
    return out


def raw_texts(path: Path):
    """Textes d'un brut pour la vérification des extraits. PDF : `pdftotext -layout` (poppler,
    rapide, ordre des colonnes conservé) s'il est installé, puis pdfplumber ; les deux
    extracteurs n'ordonnent pas les cellules de la même façon. HTML : texte visible."""
    if path.suffix.lower() != ".pdf":
        yield raw_text(path)
        return
    import shutil
    import subprocess

    if shutil.which("pdftotext"):
        r = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=False
        )
        if r.returncode == 0:
            yield r.stdout
    yield raw_text(path)  # générateur : pdfplumber seulement si pdftotext ne suffit pas


def raw_text(path: Path) -> str:
    """Texte d'un brut : PDF (pdfplumber, toutes pages) ou HTML (texte visible)."""
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ")


def _squash(text: str) -> str:
    """Comparaison d'extraits : ASCII minuscule, alphanumérique seul (la mise en page des
    PDF et du HTML déplace espaces, insécables et retours à la ligne)."""
    from unidecode import unidecode

    return re.sub(r"[^a-z0-9]", "", unidecode(text).lower())


def _numbers(x: float) -> list[str]:
    """Écritures acceptées d'un pourcentage dans un extrait (45 ; 45,5 / 45.5)."""
    if float(x).is_integer():
        return [str(int(x)), f"{int(x)},0", f"{int(x)}.0"]
    s = f"{x:g}"
    return [s.replace(".", ","), s]


def verify_row(row: PotentialRow, text_squashed: str) -> bool:
    """Vrai si chaque extrait (séparateur « ; ») figure dans le brut et si chaque chiffre
    relevé figure dans les extraits."""
    if not row.quote:
        return False
    # extraits multiples séparés par « ; » (ligne de tableau puis phrase du commentaire)
    if not all(_squash(part) in text_squashed for part in row.quote.split(" ; ") if part.strip()):
        return False
    q = row.quote.replace(" ", " ").replace("\xa0", " ")
    shown = row.potential_parts or [row.potential]
    for v in (*shown, row.reject, row.firm):
        if v is None:
            continue
        if not any(re.search(rf"(?<![\d,.]){re.escape(n)}(?![\d]|[,.]\d)", q) for n in _numbers(v)):
            return False
    return True


# --------------------------------------------------------------------------- parseurs


@dataclass
class ParsedRow:
    label: str
    electorate: str = "all"
    potential: float | None = None
    reject: float | None = None
    firm: float | None = None
    date: dt.date | None = None  # vagues antérieures lues sur un graphique d'évolution
    note: str = ""


@dataclass
class ParseResult:
    rows: list[ParsedRow] = field(default_factory=list)
    fieldwork: tuple[dt.date, dt.date] | None = None
    sample_size: int | None = None
    warnings: list[str] = field(default_factory=list)


_NUM = re.compile(r"^\d{1,3}$")
_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}


def _norm(text: str) -> str:
    from vvs.ingest.commission.aliases import norm_label

    return norm_label(text)


def _segments(words: list[dict], gap: float = 12.0) -> tuple[list[dict], list[dict]]:
    """Mots d'une page -> (segments de libellé, nombres). Un segment = mots non numériques
    consécutifs sur une même ligne (écart horizontal < `gap`)."""
    nums = [w for w in words if _NUM.match(w["text"])]
    text = sorted(
        (w for w in words if not _NUM.match(w["text"])), key=lambda w: (round(w["top"]), w["x0"])
    )
    segs: list[dict] = []
    for w in text:
        last = segs[-1] if segs else None
        if (
            last is not None
            and abs(w["top"] - last["top"]) < 3
            and 0 <= w["x0"] - last["x1"] < gap
        ):
            last["text"] += " " + w["text"]
            last["x1"] = w["x1"]
            last["bottom"] = max(last["bottom"], w["bottom"])
        else:
            segs.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"], "top": w["top"], "bottom": w["bottom"]})
    return segs, nums


def _pair_numbers(segs: list[dict], nums: list[dict], max_gap: float = 260.0) -> list[tuple[str, int]]:
    """Associe chaque nombre au libellé le plus proche à sa gauche, sur la même ligne (±7 pt)."""
    out = []
    for n in nums:
        cy = (n["top"] + n["bottom"]) / 2
        best = None
        for s in segs:
            sy = (s["top"] + s["bottom"]) / 2
            if abs(sy - cy) > 7 or s["x1"] > n["x0"] or n["x0"] - s["x1"] > max_gap:
                continue
            d = n["x0"] - s["x1"]
            if best is None or d < best[0]:
                best = (d, s)
        if best is not None:
            out.append((best[1]["text"], int(n["text"])))
    return out


def _is_person(label: str) -> bool:
    from vvs.ingest.commission.aliases import candidate_id

    if candidate_id(label):
        return True
    # personnalités hors registre (Teddy Riner…) : « Prénom Nom », deux à quatre mots capitalisés
    parts = label.split()
    return 2 <= len(parts) <= 4 and all(p[:1].isupper() for p in parts if p not in ("de", "De"))


def _page_panels(page, n_panels: int) -> list[list[tuple[str, int]]]:
    """Panneaux côte à côte d'une page (tiers de largeur) : couples (libellé, valeur)."""
    w = page.width
    out = []
    for k in range(n_panels):
        crop = page.crop((w * k / n_panels, 0, w * (k + 1) / n_panels, page.height))
        segs, nums = _segments(crop.extract_words())
        out.append([(lab, v) for lab, v in _pair_numbers(segs, nums) if _is_person(lab)])
    return out


# Pages « sympathisants » : un mot de la note de lecture -> partis des trois panneaux
PARTY_PAGES: dict[str, tuple[str, str, str]] = {
    "france insoumise": ("lfi", "ps", "ecologistes"),
    "rassemblement national": ("renaissance", "lr", "rn"),
}


def _chart_waves(page, notice_date: dt.date | None) -> list[ParsedRow]:
    """Graphique « L'évolution du potentiel électoral » : une ligne par personnalité, une
    colonne par vague (mois de publication en abscisse). Date approchée : le 25 du mois
    précédent le mois affiché (les terrains du suivi ont lieu en fin de mois précédent)."""
    words = page.extract_words()
    months = []
    for w in words:
        m = _MONTHS.get(_norm(w["text"]))
        if m is None:
            continue
        cx = (w["x0"] + w["x1"]) / 2
        year = next(
            (x for x in words if re.fullmatch(r"20\d\d", x["text"])
             and 0 < x["top"] - w["top"] < 25 and abs((x["x0"] + x["x1"]) / 2 - cx) < 20),
            None,
        )
        if year is None:
            continue
        months.append((cx, int(year["text"]), m, w["top"]))
    if not months:
        return []
    axis_top = min(t for *_, t in months)
    segs, nums = _segments([w for w in words if w["top"] < axis_top - 5])
    names = sorted(
        (s for s in segs if _is_person(s["text"]) and s["x1"] < months[0][0] - 10),
        key=lambda s: s["top"],
    )
    out = []
    for k, s in enumerate(names):
        lo = (names[k - 1]["bottom"] + s["top"]) / 2 if k else 0
        hi = (s["bottom"] + names[k + 1]["top"]) / 2 if k + 1 < len(names) else axis_top
        band = [n for n in nums if lo <= n["top"] < hi and n["x0"] > s["x1"]]
        for cx, y, m, _ in months:
            near = [n for n in band if abs((n["x0"] + n["x1"]) / 2 - cx) < 25]
            if len(near) != 1:
                continue
            d = dt.date(y, m, 1) - dt.timedelta(days=6)
            if notice_date is not None and d > notice_date:
                continue
            out.append(ParsedRow(s["text"], potential=float(near[0]["text"]), date=d, note="chart"))
    return out


def parse_harris_binary(path: Path) -> ParseResult:
    """Suivi du potentiel électoral (Toluna Harris Interactive pour L'Opinion) : page générale
    (« X% des Français indiquent qu'ils pourraient voter pour… »), pages par sympathisants
    (dix premiers noms de chaque parti) et graphique d'évolution des vagues."""
    import pdfplumber

    from vvs.ingest.commission.notice import extract_text, find_fieldwork, find_sample_size

    res = ParseResult()
    nt = extract_text(path)
    head = "\n".join(nt.pages[:4] + nt.columns)
    res.fieldwork = find_fieldwork(head)
    res.sample_size = find_sample_size(head)
    seen_global = False
    with pdfplumber.open(path) as pdf:
        for page, text in zip(pdf.pages, nt.pages):
            t = _norm(text)
            if "pourriez vous voter pour chacune" not in t and "evolution du potentiel" not in t:
                continue
            if "evolution du potentiel electoral" in t:
                res.rows += _chart_waves(page, res.fieldwork[1] if res.fieldwork else None)
                continue
            if "aupres des sympathisants" in t:
                parties = next((v for k, v in PARTY_PAGES.items() if f"proches politiquement {'de la' if k == 'france insoumise' else 'du'} {k}" in t), None)
                if parties is None:
                    continue
                for party, panel in zip(parties, _page_panels(page, 3)):
                    for lab, v in panel:
                        res.rows.append(ParsedRow(lab, electorate=f"party:{party}", potential=float(v)))
                continue
            if (
                not seen_global
                and re.search(r"des francais indiquent qu ils pourraient voter pour", t)
                and "aupres des" not in t
            ):
                segs, nums = _segments(page.extract_words())
                pairs = [(lab, v) for lab, v in _pair_numbers(segs, nums) if _is_person(lab)]
                res.rows += [ParsedRow(lab, potential=float(v)) for lab, v in pairs]
                seen_global = True
    if not seen_global:
        res.warnings.append("page générale introuvable")
    return res


# La note moyenne (et son rappel) passe parfois à la ligne suivante : facultative ici.
_ZT_ROW = re.compile(
    r"^\s*(?P<label>[A-ZÉÈÀÂÎÔÇ][^\d]+?)\s+(?P<a>\d{1,3})\s+(?P<b>\d{1,3})\s+(?P<c>\d{1,3})\s+"
    r"(?P<d>\d{1,3})(?:\s+\d{1,2},\d){0,2}\s*$"
)


def parse_zero_ten_lines(lines: list[str]) -> tuple[list[ParsedRow], list[str]]:
    """Lignes « Personnalité  0-3  4-6  7-8  9-10  [moyenne] » -> potentiel (7 à 10), rejet
    (0 à 3), socle (9 à 10) ; une ligne dont les tranches ne somment pas à 100 ± 3 est
    écartée avec un avertissement."""
    rows, warnings = [], []
    for line in lines:
        m = _ZT_ROW.match(line)
        if not m or not _is_person(m["label"].strip()):
            continue
        a, b, c, d = (int(m[k]) for k in "abcd")
        if not 97 <= a + b + c + d <= 103:
            warnings.append(f"somme {a + b + c + d} : {line.strip()}")
            continue
        rows.append(ParsedRow(m["label"].strip(), potential=float(c + d), reject=float(a), firm=float(d)))
    return rows, warnings


def parse_harris_zero_ten(path: Path) -> ParseResult:
    """Potentiel en note de 0 à 10 (Toluna Harris Interactive pour RTL) : une ligne par
    personnalité, quatre tranches (0-3, 4-6, 7-8, 9-10) puis la note moyenne."""
    from vvs.ingest.commission.notice import extract_text, find_fieldwork, find_sample_size

    res = ParseResult()
    nt = extract_text(path)
    head = "\n".join(nt.pages[:4] + nt.columns)
    res.fieldwork = find_fieldwork(head)
    res.sample_size = find_sample_size(head)
    for text in nt.pages:
        t = _norm(text)
        if "le potentiel electoral des principales personnalites" not in t or "notes de 0 a 3" not in t:
            continue
        rows, warnings = parse_zero_ten_lines(text.splitlines())
        res.rows += rows
        res.warnings += warnings
        break
    if not res.rows:
        res.warnings.append("tableau 0-10 introuvable")
    return res


PARSERS = {"harris_binary": parse_harris_binary, "harris_zero_ten": parse_harris_zero_ten}


# --------------------------------------------------------------------------- table


def _candidate(label: str) -> str | None:
    from vvs.ingest.commission.aliases import candidate_id

    if ID_RE.match(label) and "_" in label:
        return label
    return candidate_id(label)


def _electorate_kind(e: str) -> str:
    if e == "all":
        return "all"
    head = e.split(":", 1)[0]
    if head == "party":
        return "party_proximity"
    if head == "vote_r1":
        return "vote_intention_r1"
    if head.startswith("vote_"):
        return "past_vote"
    return "other"


class _LazyTexts:
    """Textes d'un brut pour la vérification, extraits à la demande : l'extracteur suivant
    (pdfplumber après pdftotext) n'est lancé que si un extrait manque dans les précédents."""

    def __init__(self, path: Path | None) -> None:
        self._gen = iter(raw_texts(path)) if path is not None else iter(())
        self._texts: list[str] = []

    def verify(self, row: PotentialRow) -> bool:
        k = 0
        while True:
            if k == len(self._texts):
                nxt = next(self._gen, None)
                if nxt is None:
                    return False
                self._texts.append(_squash(nxt))
            if verify_row(row, self._texts[k]):
                return True
            k += 1


@dataclass
class ParseReport:
    n_sources: int = 0
    n_rows: int = 0
    unverified: list[str] = field(default_factory=list)
    missing_raw: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.n_sources} sources, {self.n_rows} lignes"]
        if self.missing_raw:
            lines.append(f"bruts absents ({len(self.missing_raw)}) : " + ", ".join(self.missing_raw))
        if self.unverified:
            lines.append(f"relevés non vérifiés ({len(self.unverified)}) :")
            lines += [f"  {u}" for u in self.unverified]
        lines += [f"  attention : {w}" for w in self.warnings]
        return "\n".join(lines)


def build(
    sources: list[PotentialSource] | None = None, raw_root: Path | None = None
) -> tuple[pl.DataFrame, ParseReport]:
    """Table des potentiels (une ligne par source × personnalité × électorat)."""
    sources = load_sources() if sources is None else sources
    root = paths.RAW if raw_root is None else Path(raw_root)
    report = ParseReport(n_sources=len(sources))
    records: list[dict] = []
    for s in sources:
        path = raw_path(s, root)
        exists = path is not None and path.exists()
        if not exists:
            report.missing_raw.append(s.id)
        rel = str(path.relative_to(root)) if exists else None
        digest = sha256_of(path) if exists else None
        base = {
            "election": s.election,
            "pollster": s.pollster,
            "sponsor": s.sponsor,
            "population": s.population,
            "round": s.round,
            "scale": s.scale,
            "measure": s.measure_key,
            "sample_size": s.sample_size,
            "source_url": s.url,
            "raw_file": rel,
            "raw_sha256": digest,
        }
        start, end = s.fieldwork_start, s.fieldwork_end
        if s.parser is not None:
            if not exists:
                continue
            res = PARSERS[s.parser](path)
            report.warnings += [f"{s.id} : {w}" for w in res.warnings]
            if res.fieldwork and end is None:
                start, end = res.fieldwork
            elif res.fieldwork and res.fieldwork[1] != end:
                report.warnings.append(f"{s.id} : terrain lu {res.fieldwork} ≠ registre {end}")
            if base["sample_size"] is None:
                base["sample_size"] = res.sample_size
            for r in res.rows:
                chart = r.date is not None
                records.append(
                    base
                    | {
                        "source_id": f"{s.id}_chart_{r.date:%Y-%m}" if chart else s.id,
                        "fieldwork_start": None if chart else start,
                        "date": r.date if chart else end,
                        "date_approx": chart or s.date_approx,
                        "sample_size": None if chart else base["sample_size"],
                        "candidate_id": _candidate(r.label),
                        "candidate_label": r.label,
                        "electorate": r.electorate,
                        "electorate_kind": _electorate_kind(r.electorate),
                        "potential_share": r.potential,
                        "reject_share": r.reject,
                        "firm_share": r.firm,
                        "verified": True,  # lu dans le brut par le parseur
                    }
                )
            continue
        texts = _LazyTexts(path if exists else None)
        for r in s.rows:
            ok = exists and texts.verify(r)
            if not ok:
                report.unverified.append(f"{s.id} / {r.candidate} / {r.electorate}")
            records.append(
                base
                | {
                    "source_id": s.id,
                    "fieldwork_start": start,
                    "date": end,
                    "date_approx": s.date_approx,
                    "candidate_id": _candidate(r.candidate),
                    "candidate_label": r.candidate,
                    "electorate": r.electorate,
                    "electorate_kind": _electorate_kind(r.electorate),
                    "potential_share": r.potential,
                    "reject_share": r.reject,
                    "firm_share": r.firm,
                    "verified": ok,
                }
            )
    df = pl.DataFrame(records, schema=SCHEMA) if records else pl.DataFrame(schema=SCHEMA)
    df = dedupe_chart_rows(df)
    report.n_rows = df.height
    return df.sort("election", "date", "source_id", "electorate", "candidate_label"), report


def dedupe_chart_rows(df: pl.DataFrame, tolerance_days: int = 25) -> pl.DataFrame:
    """Valeurs lues sur un graphique d'évolution : écartées quand la vague est déjà présente
    par sa propre notice (même institut, mesure et personnalité à `tolerance_days` près) ; entre
    graphiques, la notice la plus récente l'emporte."""
    chart = df.filter(pl.col("source_id").str.contains("_chart_"))
    if chart.is_empty():
        return df
    who = pl.coalesce("candidate_id", "candidate_label").alias("who")
    direct = df.filter(
        ~pl.col("source_id").str.contains("_chart_") & (pl.col("electorate") == "all")
    ).with_columns(who)
    keep, seen = [], set()
    for r in chart.with_columns(who).sort("source_id", descending=True).iter_rows(named=True):
        same = direct.filter(
            (pl.col("measure") == r["measure"])
            & (pl.col("who") == r["who"])
            & ((pl.col("date") - pl.lit(r["date"])).dt.total_days().abs() <= tolerance_days)
        )
        key = (r["measure"], r["who"], r["date"])
        if not same.is_empty() or key in seen:
            continue
        seen.add(key)
        r.pop("who")
        keep.append(r)
    out = df.filter(~pl.col("source_id").str.contains("_chart_"))
    if keep:
        kept = pl.DataFrame(keep, schema=SCHEMA)
        # même vague lue sur plusieurs graphiques : source_id de la vague, pas du graphique
        out = pl.concat([out, kept])
    return out


def parse(write: bool = True, out_dir: Path | None = None) -> tuple[Path, pl.DataFrame, ParseReport]:
    """`vvs parse potential` : registre + bruts -> data/curated/electoral_potential.parquet."""
    df, report = build()
    out = (paths.CURATED if out_dir is None else Path(out_dir)) / OUTPUT_NAME
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
    return out, df, report
