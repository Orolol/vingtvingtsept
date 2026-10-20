"""Socle commun de l'ingestion des listes de sondages Wikipédia FR.

Ce module regroupe ce qui ne dépend pas d'une page particulière :

- téléchargement du rendu HTML d'une page via l'API MediaWiki (`action=parse`) et stockage
  brut horodaté dans `data/raw/wikipedia/<election>/<YYYY-MM-DDTHHMM>.html` (UTC) ; le
  fichier commence par un commentaire `<!-- vvs:wikipedia … -->` qui garde le titre, la
  révision et l'horodatage, de sorte que le parsing soit reproductible sans réseau ;
- helpers de parsing : dates françaises et plages de dates (avec ou sans année),
  pourcentages (`<0,5 %`, `—`…), effectifs, normalisation des libellés ;
- résolution des libellés de candidats et d'instituts par la table d'alias versionnée
  `data/registry/aliases_wikipedia.yaml`. Un libellé inconnu lève `UnknownLabelError` :
  on ne devine jamais.

Le moteur générique de lecture des tableaux est dans `engine.py`, les particularités de
chaque page dans `e<AAAA>.py`.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml
from unidecode import unidecode

from vvs import paths

API_URL = "https://fr.wikipedia.org/w/api.php"
PAGE_TITLE = "Liste de sondages sur l'élection présidentielle française de {election}"
USER_AGENT = "vingtvingtsept/0.1 (French 2027 presidential forecast; https://github.com/Orolol)"
RAW_DIR = paths.RAW / "wikipedia"
ALIASES_PATH = paths.REGISTRY / "aliases_wikipedia.yaml"

_HEADER_RE = re.compile(r"^<!-- vvs:wikipedia (?P<body>.*?) -->\n", re.DOTALL)


class UnknownLabelError(ValueError):
    """Libellé (candidat ou institut) absent de la table d'alias."""


class AmbiguousLabelError(ValueError):
    """Libellé connu qui ne désigne pas une personne unique (« Bardella / Le Pen »)."""


# ---------------------------------------------------------------------------
# Téléchargement et stockage brut
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawMeta:
    """Métadonnées d'un HTML brut, lues dans son commentaire d'en-tête."""

    election: int
    title: str
    revid: int | None
    fetched_at: datetime

    @property
    def source_url(self) -> str:
        """Lien permanent vers la révision parsée (ou la page si la révision manque)."""
        t = quote(self.title.replace(" ", "_"))
        if self.revid:
            return f"https://fr.wikipedia.org/w/index.php?title={t}&oldid={self.revid}"
        return f"https://fr.wikipedia.org/wiki/{t}"


def page_title(election: int) -> str:
    return PAGE_TITLE.format(election=election)


def fetch_page(election: int, client: httpx.Client | None = None) -> str:
    """Télécharge le rendu HTML courant de la page et le renvoie avec son en-tête vvs."""
    params = {
        "action": "parse",
        "page": page_title(election),
        "prop": "text|revid",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0)
    try:
        resp = client.get(API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    finally:
        if own:
            client.close()
    if "error" in data:
        raise RuntimeError(f"API MediaWiki : {data['error']}")
    parsed = data["parse"]
    fetched_at = datetime.now(UTC).replace(microsecond=0)
    header = (
        f"<!-- vvs:wikipedia election={election} revid={parsed['revid']} "
        f'fetched_at={fetched_at.isoformat()} title="{parsed["title"]}" -->\n'
    )
    return header + parsed["text"]


def store_raw(election: int, html: str) -> Path:
    """Écrit le HTML brut dans data/raw/wikipedia/<election>/<horodatage UTC>.html.

    `data/raw/` est en ajout seul : on refuse d'écraser un fichier existant.
    """
    meta = read_meta(html, election)
    out_dir = RAW_DIR / str(election)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{meta.fetched_at:%Y-%m-%dT%H%M}.html"
    if path.exists():
        raise FileExistsError(f"{path} existe déjà (data/raw est immuable)")
    path.write_text(html, encoding="utf-8")
    return path


def latest_raw(election: int) -> Path:
    """Dernier HTML brut téléchargé pour cette élection (ordre des horodatages)."""
    files = sorted((RAW_DIR / str(election)).glob("*.html"))
    if not files:
        raise FileNotFoundError(
            f"aucun HTML brut dans {RAW_DIR / str(election)} : lancer `vvs ingest wikipedia "
            f"--election {election}`"
        )
    return files[-1]


def read_meta(html: str, election: int) -> RawMeta:
    """Lit l'en-tête vvs du HTML brut. Sans en-tête, on lève : pas d'horodatage inventé."""
    m = _HEADER_RE.match(html)
    if not m:
        raise ValueError("HTML sans en-tête `<!-- vvs:wikipedia … -->` (voir fetch_page)")
    body = m.group("body")
    fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', body))
    fields = {k: v.strip('"') for k, v in fields.items()}
    if int(fields["election"]) != election:
        raise ValueError(f"HTML de l'élection {fields['election']}, attendu {election}")
    fetched = datetime.fromisoformat(fields["fetched_at"]).astimezone(UTC).replace(tzinfo=None)
    revid = int(fields["revid"]) if fields.get("revid", "").isdigit() else None
    return RawMeta(election=election, title=fields["title"], revid=revid, fetched_at=fetched)


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------


def clean_text(s: str) -> str:
    """Espaces normalisés (insécables compris), sans espaces en bord."""
    s = s.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def normalize_label(s: str) -> str:
    """Clé de comparaison d'un libellé : ASCII minuscule, ponctuation réduite.

    `Jean-Luc Mélenchon ( LFI )` et `jean luc melenchon` donnent la même clé une fois la
    parenthèse retirée par `strip_party`.
    """
    s = unidecode(clean_text(s)).lower()
    s = re.sub(r"[‐-―\-'’`]", " ", s)
    s = re.sub(r"[^a-z0-9&+ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_party(label: str) -> str:
    """Retire les étiquettes entre parenthèses : `Hollande ( PS )` -> `Hollande`."""
    return clean_text(re.sub(r"\([^)]*\)", " ", label))


# ---------------------------------------------------------------------------
# Pourcentages et effectifs
# ---------------------------------------------------------------------------

ABSENT_MARKERS = {"", "—", "–", "-", "‒", "n/a", "na", "nc", "n.c.", "/", "x", "?"}
_NUM = r"(?P<lt><\s*)?(?P<num>\d+(?:[.,]\d+)?)\s*%?"
NUM_RE = re.compile(_NUM)


def parse_share(text: str) -> float | None:
    """Pourcentage d'une cellule. `None` si absent (non testé), lève si illisible.

    `<x` (moins de x %) est codé x/2 : le candidat est testé, son score est sous le seuil
    publié, et on garde la meilleure estimation ponctuelle sans fausser les sommes.
    """
    t = strip_party(text).lower()  # « 38 % (1 006) » : effectif entre parenthèses
    if t in ABSENT_MARKERS:
        return None
    m = re.fullmatch(r"(<\s*)?(\d+(?:[.,]\d+)?)\s*%?", t)
    if not m:
        raise ValueError(f"pourcentage illisible : {text!r}")
    v = float(m.group(2).replace(",", "."))
    if m.group(1):
        v = v / 2
    if v > 100:
        raise ValueError(f"pourcentage > 100 : {text!r}")
    return v


def parse_share_tokens(text: str) -> list[tuple[float, str | None]]:
    """Découpe une cellule en couples (part, nom éventuel).

    Formes rencontrées : `14 % Bertrand`, `9 Hollande ( PS )`, `4 Lisnard 2 Ruffin`,
    `<1 Bouamrane`, `Alliot-Marie : 1 % Yade : 1 %` (nom avant la valeur), `12,5 %`.
    Renvoie [] pour une cellule vide ou un tiret. Lève si la cellule contient du texte sans
    nombre (cellule illisible).
    """
    t = clean_text(text)
    if t.lower() in ABSENT_MARKERS:
        return []
    matches = list(NUM_RE.finditer(t))
    if not matches:
        raise ValueError(f"cellule sans pourcentage : {text!r}")
    segs = [t[: matches[0].start()]]
    for a, b in zip(matches, matches[1:] + [None], strict=True):
        segs.append(t[a.end() : b.start() if b else len(t)])

    def value(m: re.Match) -> float:
        v = float(m.group("num").replace(",", "."))
        return v / 2 if m.group("lt") else v

    def name(s: str) -> str | None:
        s = strip_party(s).strip(" :;,/")
        s = re.sub(r"^(et|dont)\s+", "", s)
        return s or None

    if name(segs[0]):  # nom avant valeur
        if name(segs[-1]):
            raise ValueError(f"cellule illisible (nom sans valeur en fin) : {text!r}")
        return [(value(m), name(segs[i])) for i, m in enumerate(matches)]
    return [(value(m), name(segs[i + 1])) for i, m in enumerate(matches)]


def parse_sample(text: str) -> int | None:
    """Effectif d'échantillon : `1 351`, `12 600`. `None` si absent ou non entier."""
    t = clean_text(text).replace(" ", "")
    return int(t) if t.isdigit() else None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

MONTHS = {
    "janvier": 1, "janv": 1, "jan": 1,
    "fevrier": 2, "fevr": 2, "fev": 2,
    "mars": 3, "mar": 3,
    "avril": 4, "avr": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7, "juil": 7,
    "aout": 8,
    "septembre": 9, "sept": 9, "sep": 9,
    "octobre": 10, "oct": 10,
    "novembre": 11, "nov": 11,
    "decembre": 12, "dec": 12,
}  # fmt: skip
FULL_MONTHS = {
    "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre",
    "octobre", "novembre", "decembre",
}  # fmt: skip
_RANGE_WORDS = {"au", "et", "a", "-"}
_FILLER_WORDS = {"du", "le", "les", "entre", "er", "re"}


@dataclass(frozen=True)
class PartialDate:
    day: int | None
    month: int | None
    year: int | None


def parse_fr_range(text: str) -> tuple[PartialDate, PartialDate]:
    """Plage de dates française, sans résolution de l'année ni du mois manquants.

    `7-8 avril`, `31 mars - 1er avril 2022`, `du 17 au 19`, `29 janv - 1 er fév.`,
    `14 et 15 décembre 2015`, `5 mai`. Une date seule donne une plage d'un jour.
    """
    t = unidecode(clean_text(text)).lower()
    t = re.sub(r"[–—‒―]", "-", t)
    tokens = re.findall(r"\d+|[a-z]+|-", t)
    parts: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _FILLER_WORDS:
            continue
        if tok in _RANGE_WORDS:
            if parts[-1]:
                parts.append([])
            continue
        parts[-1].append(tok)
    parts = [p for p in parts if p]
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"date illisible : {text!r}")
    out = []
    for p in parts:
        day = month = year = None
        for tok in p:
            if tok.isdigit():
                n = int(tok)
                if n >= 1900 and year is None:
                    year = n
                elif 1 <= n <= 31 and day is None and month is None:
                    day = n
                else:
                    raise ValueError(f"date illisible : {text!r}")
            elif tok in MONTHS and month is None:
                month = MONTHS[tok]
            else:
                raise ValueError(f"date illisible : {text!r}")
        if day is None:
            raise ValueError(f"date sans jour : {text!r}")
        out.append(PartialDate(day, month, year))
    if len(out) == 1:
        out.append(out[0])
    return out[0], out[1]


# Écart maximal toléré entre deux lignes consécutives quand l'année est inférée et que le
# contexte en admet plusieurs : au-delà, la ligne est écartée plutôt que datée au hasard.
MAX_YEAR_GAP_DAYS = 120


@dataclass(frozen=True)
class DateBounds:
    """Fenêtre plausible des dates d'un tableau, tirée de son contexte (titres, légende)."""

    lower: date | None
    upper: date
    default_month: int | None = None  # mois implicite des cellules `du 17 au 19`


def _make(y: int, m: int, d: int, text: str) -> date:
    try:
        return date(y, m, d)
    except ValueError as e:
        raise ValueError(f"date impossible {d}/{m}/{y} dans {text!r}") from e


def resolve_range(
    text: str,
    bounds: DateBounds,
    previous_end: date | None,
    exact_end: date | None = None,
    exact_start: date | None = None,
) -> tuple[date, date]:
    """Résout une plage en dates complètes.

    L'année manquante de la fin est choisie parmi les années voisines de la ligne
    précédente (les tableaux sont antéchronologiques), sans sortir des bornes du contexte ;
    si plusieurs années restent possibles et que la plus proche est à plus de
    `MAX_YEAR_GAP_DAYS` de la ligne précédente, on lève (année ambiguë). Le mois manquant vient de la fin de plage, puis du contexte (légende « Mars
    2017 »). `exact_end`/`exact_start` viennent des balises `<time datetime>` quand la page
    en fournit et priment sur l'inférence.
    """
    s, e = parse_fr_range(text)
    e_month = e.month or bounds.default_month
    if e_month is None:
        raise ValueError(f"date sans mois et contexte sans mois unique : {text!r}")
    if exact_end is not None:
        end = exact_end
    elif e.year is not None:
        end = _make(e.year, e_month, e.day, text)
    else:
        ref = previous_end or bounds.upper
        cands = []
        for y in (ref.year - 1, ref.year, ref.year + 1):
            try:
                d = date(y, e_month, e.day)
            except ValueError:
                continue
            if d <= bounds.upper and (bounds.lower is None or d >= bounds.lower):
                cands.append(d)
        if not cands:
            raise ValueError(f"aucune année plausible pour {text!r} (borne {bounds.upper})")
        end = min(cands, key=lambda d: abs((d - ref).days))
        if len(cands) > 1 and abs((end - ref).days) > MAX_YEAR_GAP_DAYS:
            raise ValueError(
                f"année ambiguë pour {text!r} : {abs((end - ref).days)} jours depuis {ref}"
            )
    if exact_start is not None:
        start = exact_start
    else:
        s_month = s.month or end.month
        s_year = s.year or end.year
        start = _make(s_year, s_month, s.day, text)
        if start > end and s.year is None:
            start = _make(s_year - 1, s_month, s.day, text)
    if start > end:
        raise ValueError(f"début après la fin : {text!r}")
    return start, end


_SEMESTERS = {"premier semestre": (1, 6), "second semestre": (7, 12)}


def bounds_from_context(texts: list[str], fallback: DateBounds) -> DateBounds:
    """Bornes de dates à partir du texte de contexte le plus précis contenant une année.

    `texts` va du plus précis (légende) au moins précis (titre de niveau 2). Formes :
    `Mars 2017`, `Octobre-décembre 2026`, `Second semestre 2021`, `Sondages de 2012 à 2015`.
    """
    for raw in texts:
        # « l'élection présidentielle 2017 » désigne l'élection, pas la période du tableau
        t = re.sub(r"presidentielle( francaise)?( de)? \d{4}", " ", normalize_label(raw))
        years = sorted({int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", t)})
        if not years:
            continue
        months = [MONTHS[w] for w in t.split() if w in FULL_MONTHS]
        m0, m1 = 1, 12
        for key, (a, b) in _SEMESTERS.items():
            if key in t:
                m0, m1 = a, b
        if months and len(years) == 1:
            m0, m1 = min(months), max(months)
        lower = date(years[0], m0, 1)
        upper = date(years[-1], m1, calendar.monthrange(years[-1], m1)[1])
        upper = min(upper, fallback.upper)
        default_month = m0 if m0 == m1 else None
        return DateBounds(lower=lower, upper=upper, default_month=default_month)
    return fallback


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Aliases:
    """Tables de correspondance libellé normalisé -> id, pour une élection donnée."""

    election: int
    candidates: dict[str, str]
    pollsters: dict[str, tuple[str, str | None]]
    ambiguous: frozenset[str] = frozenset()

    def candidate(self, label: str) -> str:
        key = normalize_label(strip_party(label))
        if key in self.ambiguous:
            raise AmbiguousLabelError(f"libellé ambigu : {label!r}")
        try:
            return self.candidates[key]
        except KeyError:
            raise UnknownLabelError(
                f"libellé de candidat inconnu pour {self.election} : {label!r} (clé {key!r}). "
                f"Ajoute-le à {ALIASES_PATH.relative_to(paths.ROOT)}"
            ) from None

    def pollster(self, label: str) -> tuple[str, str | None]:
        key = normalize_label(label)
        try:
            return self.pollsters[key]
        except KeyError:
            raise UnknownLabelError(
                f"libellé d'institut inconnu : {label!r} (clé {key!r}). "
                f"Ajoute-le à la section pollsters de {ALIASES_PATH.relative_to(paths.ROOT)}"
            ) from None


def _check_id(cid: str, where: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", cid):
        raise ValueError(f"{where} : id invalide {cid!r} (minuscules ASCII et _)")
    return cid


@lru_cache(maxsize=16)
def load_aliases(election: int, path: Path = ALIASES_PATH) -> Aliases:
    """Charge les alias : section `common`, surchargée par `elections.<AAAA>`.

    `ambiguous` liste les libellés reconnus qui désignent plusieurs personnes : l'hypothèse
    qui les contient est écartée (journalisée), sans erreur.

    Les homonymes (Le Pen père / fille…) se rangent sous `elections`, jamais sous `common`.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cands: dict[str, str] = {}
    sections = [("common", raw.get("candidates", {}).get("common") or {})]
    by_election = raw.get("candidates", {}).get("elections") or {}
    sections.append((f"elections.{election}", by_election.get(election) or {}))
    for where, section in sections:
        for label, cid in section.items():
            key = normalize_label(strip_party(str(label)))
            cands[key] = _check_id(cid, f"{path.name}:{where}:{label}")
    polls: dict[str, tuple[str, str | None]] = {}
    for label, val in (raw.get("pollsters") or {}).items():
        if isinstance(val, str):
            pid, sponsor = val, None
        else:
            pid, sponsor = val["pollster"], val.get("sponsor")
        polls[normalize_label(str(label))] = (_check_id(pid, f"{path.name}:pollsters"), sponsor)
    ambiguous = frozenset(
        normalize_label(strip_party(str(x)))
        for x in raw.get("candidates", {}).get("ambiguous") or []
    )
    return Aliases(election=election, candidates=cands, pollsters=polls, ambiguous=ambiguous)
