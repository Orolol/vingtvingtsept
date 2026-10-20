"""Cote de popularité de l'exécutif : baromètre mensuel Ifop (« Les indices de popularité »).

Question posée chaque mois depuis 1958 : « Êtes-vous satisfait ou mécontent de X comme
président de la République / comme Premier ministre ? ». On garde le total des satisfaits.

Source : les PDF publiés sur ifopgroup.com, listés par l'API WordPress du site
(`/wp-json/wp/v2/posts?search=indices de popularité`). Les archives en ligne commencent en
janvier 2006. Deux formats :

- jusqu'à décembre 2020 (`TABLE_ERA_END`), chaque PDF contient, pour le président et pour le
  Premier ministre, le tableau « L'évolution depuis <début du mandat> » : une ligne par mois,
  « Mai 2012 ....... 61 33 6 100 » (satisfaits, mécontents, ne se prononcent pas, total).
  Le dernier PDF avant un changement de titulaire suffit donc à reconstituer tout le mandat
  sortant (celui de janvier 2007 remonte à mai 1995 pour Jacques Chirac) ;
- ensuite, des diapositives : seul le mois courant est lisible (« TOTAL Satisfaits TOTAL
  Mécontents » puis « 17 83 »), il faut un document par mois (avril 2021 n'existe qu'en
  pptx : les chiffres y sont dans le cache des graphiques, voir `pptx_text`).

L'ingestion télécharge donc le dernier PDF avant chaque changement de titulaire de l'ère des
tableaux, puis tous les documents suivants. Ils vont dans
data/raw/macro/ifop_popularity/documents/ (un fichier par publication, jamais réécrit) ; la
liste des publications retenues est figée chaque jour dans
data/raw/macro/ifop_popularity/<AAAA-MM-JJ>.json.

Le rôle d'un tableau se déduit du nom du titulaire (registre `officeholders`) lu dans le
titre de section ou la question ; un mois hors du mandat de ce titulaire est rejeté.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import httpx
from lxml import etree
from unidecode import unidecode

from vvs.ingest.macro import raw
from vvs.ingest.macro.registry import Officeholder

WP_POSTS = "https://www.ifopgroup.com/wp-json/wp/v2/posts"
SEARCH = "indices de popularité"
RAW_DIR = "ifop_popularity"
TABLE_ERA_END = date(2020, 12, 1)
TITLE_RE = re.compile(r"indices?\s+(?:de\s+)?popularit")  # sur le titre normalisé
MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}
MONTH_ALT = "|".join(MONTHS)
ROW_RE = re.compile(rf"^\s*({MONTH_ALT})\s+(\d{{4}})\b(.*)$")
TRIGGER_RE = re.compile(r"(cote de popularite|satisfait ou mecontent)\s+d")
HEADLINE_RE = re.compile(r"total satisfaits\s+total mecontents")
PAIR_RE = re.compile(r"^\s*(\d{1,3})\s+(\d{1,3})\s*$")


# ---------------------------------------------------------------------------- liste et PDF


@dataclass(frozen=True)
class Post:
    post_id: int
    published: date
    title: str
    link: str
    month: date

    def to_json(self) -> dict:
        return {
            "post_id": self.post_id,
            "published": self.published.isoformat(),
            "title": self.title,
            "link": self.link,
            "month": self.month.isoformat()[:7],
        }


def post_month(title: str, published: date) -> date:
    """Mois mesuré : celui du titre (« Octobre 2026 », « vague de novembre »), sinon celui de
    la publication (le baromètre paraît la troisième ou quatrième semaine du mois mesuré)."""
    norm = _norm(title)
    m = re.search(rf"\b({MONTH_ALT})\s+(\d{{4}})\b", norm)
    if m:
        return date(int(m[2]), MONTHS[m[1]], 1)
    m = re.search(rf"\b({MONTH_ALT})\b", norm)
    if m:
        return date(published.year, MONTHS[m[1]], 1)
    return published.replace(day=1)


def list_posts(http: httpx.Client) -> list[Post]:
    posts = []
    page, pages = 1, 1
    while page <= pages:
        r = raw.get(
            http,
            WP_POSTS,
            params={
                "search": SEARCH,
                "per_page": 100,
                "page": page,
                "_fields": "id,date,link,title",
            },
        )
        pages = int(r.headers.get("X-WP-TotalPages", "1"))
        for item in r.json():
            title = _html_unescape(item["title"]["rendered"])
            link = item["link"]
            if not TITLE_RE.search(_norm(title)) or "/en/" in link:
                continue
            published = date.fromisoformat(item["date"][:10])
            posts.append(Post(item["id"], published, title, link, post_month(title, published)))
        page += 1
    posts.sort(key=lambda p: (p.published, p.post_id))
    return posts


def select_posts(posts: list[Post], holders: tuple[Officeholder, ...]) -> list[Post]:
    """PDF à télécharger : à l'ère des tableaux, le dernier publié avant chaque prise de
    fonction (il contient le mandat sortant en entier) et le dernier de l'ère ; ensuite, tous."""
    table_era = [p for p in posts if p.month <= TABLE_ERA_END]
    keep = {p.post_id for p in posts if p.month > TABLE_ERA_END}
    if table_era:
        keep.add(max(table_era, key=lambda p: p.published).post_id)
        first = min(p.published for p in table_era)
        for start in sorted({h.start for h in holders}):
            if not first < start <= TABLE_ERA_END:
                continue
            before = [p for p in table_era if p.published < start]
            if before:
                keep.add(max(before, key=lambda p: p.published).post_id)
    return [p for p in posts if p.post_id in keep]


DOC_EXTS = ("pdf", "pptx")  # avril 2021 n'est publié qu'en pptx
DOC_MAGIC = {"pdf": b"%PDF", "pptx": b"PK"}


def document_url(http: httpx.Client, post: Post) -> str:
    html = raw.get(http, post.link).text
    exts = "|".join(DOC_EXTS)
    urls = re.findall(
        rf'href="(https://www\.ifopgroup\.com/wp-content/uploads/[^"]+\.(?:{exts}))"', html
    )
    if not urls:
        raise ValueError(f"aucun document ({', '.join(DOC_EXTS)}) sur {post.link}")
    return urls[0]


def documents_dir() -> Path:
    return raw.macro_dir() / RAW_DIR / "documents"


def document_path(post: Post, ext: str) -> Path:
    return documents_dir() / f"{post.month.isoformat()[:7]}_{post.post_id}.{ext}"


def _existing_document(post: Post) -> Path | None:
    for ext in DOC_EXTS:
        path = document_path(post, ext)
        if path.exists():
            return path
    return None


def ingest(
    http: httpx.Client, holders: tuple[Officeholder, ...], log=print
) -> tuple[raw.RawFile | None, list[str]]:
    """Liste les publications, télécharge les documents manquants, fige la liste du jour.

    Une publication en échec (pas de document, format inattendu) est signalée et laissée hors
    de la liste du jour ; les autres sont gardées. Renvoie (liste écrite, échecs)."""
    posts = select_posts(list_posts(http), holders)
    entries, failures = [], []
    for post in posts:
        path = _existing_document(post)
        try:
            if path is not None:
                url = raw.read_meta(path).url
            else:
                url = document_url(http, post)
                ext = url.rsplit(".", 1)[1].lower()
                content = raw.get(http, url).content
                if not content.startswith(DOC_MAGIC[ext]):
                    raise ValueError(f"{url} n'est pas un {ext}")
                path = document_path(post, ext)
                raw.write_once(path, content, url)
                log(f"  {path.name} <- {url}")
        except (ValueError, httpx.HTTPError) as e:
            failures.append(f"{post.month.isoformat()[:7]} {post.link} : {e}")
            log(f"  échec {post.link} : {e}")
            continue
        entries.append(post.to_json() | {"file": f"documents/{path.name}", "url": url})
    body = json.dumps({"search": SEARCH, "posts": entries}, indent=2, ensure_ascii=False)
    written = raw.write_raw(RAW_DIR, "json", body.encode("utf-8") + b"\n", WP_POSTS)
    return written, failures


# ---------------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class Observation:
    role: str
    person_id: str
    month: date
    satisfied: float
    kind: str  # "table" ou "headline"
    publication: date  # mois de la publication qui fournit la valeur


@dataclass
class TextReport:
    rows_without_person: int = 0
    rows_unreadable: list[str] = field(default_factory=list)
    rows_out_of_tenure: list[str] = field(default_factory=list)
    headlines_unreadable: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(s).lower().replace("’", "'"))


def _html_unescape(s: str) -> str:
    import html

    return html.unescape(s)


def _find_person(text: str, holders: tuple[Officeholder, ...]) -> Officeholder | None:
    norm = _norm(text)
    found = [h for h in holders if _norm(h.name) in norm]
    # « Dominique de Villepin » contient aussi … rien d'autre ; garder le nom le plus long
    return max(found, key=lambda h: len(h.name)) if found else None


def parse_text(
    text: str,
    publication: date,
    holders: tuple[Officeholder, ...],
    report: TextReport | None = None,
) -> list[Observation]:
    """Observations d'un PDF (texte extrait page par page) : lignes des tableaux d'évolution
    et, pour le format diapositives, total des satisfaits du mois de publication."""
    report = report if report is not None else TextReport()
    lines = text.splitlines()
    norm_lines = [_norm(l) for l in lines]
    out: list[Observation] = []
    current: Officeholder | None = None
    awaiting_headline: Officeholder | None = None
    headline_done: set[str] = set()
    for i, (line, norm) in enumerate(zip(lines, norm_lines)):
        if TRIGGER_RE.search(norm):
            context = line + " " + (lines[i + 1] if i + 1 < len(lines) else "")
            person = _find_person(context, holders)
            if person is not None:
                current = person
                if "satisfait ou mecontent" in norm and person.person_id not in headline_done:
                    awaiting_headline = person
            continue
        if awaiting_headline is not None and HEADLINE_RE.search(norm):
            h, awaiting_headline = awaiting_headline, None
            headline_done.add(h.person_id)
            # « TOTAL Satisfaits TOTAL Mécontents » puis, ligne suivante, « 17 83 »
            m = PAIR_RE.match(norm_lines[i + 1]) if i + 1 < len(lines) else None
            # le reste jusqu'à 100 : ne se prononcent pas (jusqu'à 5 % sous Castex)
            if m is None or not 80 <= int(m[1]) + int(m[2]) <= 101:
                report.headlines_unreadable.append(f"{h.person_id} {publication}")
            elif not h.covers_month(publication):
                report.rows_out_of_tenure.append(f"{h.person_id} {publication} (titre)")
            else:
                sat = float(m[1])
                out.append(
                    Observation(h.role, h.person_id, publication, sat, "headline", publication)
                )
            continue
        m = ROW_RE.match(norm)
        if not m:
            continue
        month = date(int(m[2]), MONTHS[m[1]], 1)
        values = _row_values(m[3])
        if values is None:
            report.rows_unreadable.append(line.strip())
            continue
        if current is None:
            report.rows_without_person += 1
            continue
        if not current.covers_month(month):
            report.rows_out_of_tenure.append(f"{current.person_id} {month} : {line.strip()}")
            continue
        out.append(
            Observation(
                current.role, current.person_id, month, float(values[0]), "table", publication
            )
        )
    return out


def _row_values(rest: str) -> tuple[int, int, int] | None:
    """« ......... 61 33 6 100 » -> (61, 33, 6) ; « - » vaut 0 ; la ligne finit par le total 100.

    Un chiffre parasite en tête (renvoi de note, « 1 55 28 17 100 ») est ignoré : on lit les
    quatre derniers nombres. Les arrondis d'Ifop laissent des sommes de 97 à 103 (Fillon,
    mars 2010 : « 49 45 9 100 ») : on garde la ligne, au-delà on la rejette."""
    cleaned = re.sub(r"\.{2,}|…+", " ", rest)
    tokens = [t for t in cleaned.split() if re.fullmatch(r"\d{1,3}|-", t)]
    if len(tokens) < 4 or tokens[-1] != "100":
        return None
    sat, mec, nsp = (0 if t == "-" else int(t) for t in tokens[-4:-1])
    if not 97 <= sat + mec + nsp <= 103:
        return None
    return sat, mec, nsp


def document_text(path: Path) -> str:
    """Texte d'une publication, page par page (PDF) ou diapositive par diapositive (pptx)."""
    if path.suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if path.suffix == ".pptx":
        return pptx_text(path.read_bytes())
    raise ValueError(f"format de document inattendu : {path}")


def pptx_text(content: bytes) -> str:
    """Paragraphes de chaque diapositive, puis, pour chaque graphique de la diapositive dont
    les séries « TOTAL Satisfaits » et « TOTAL Mécontents » n'ont qu'une valeur, les lignes
    « TOTAL Satisfaits TOTAL Mécontents » / « <sat> <méc> » que le parseur lit aussi dans les
    PDF (dans le pptx d'avril 2021, ces chiffres ne sont que dans le cache du graphique)."""
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    c = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    lines = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()
        slides = sorted(
            (n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
        )
        for name in slides:
            slide = etree.fromstring(z.read(name), parser)
            for para in slide.iter(f"{a}p"):
                text = "".join(t.text or "" for t in para.iter(f"{a}t"))
                if text.strip():
                    lines.append(text)
            rels_name = name.replace("slides/", "slides/_rels/") + ".rels"
            if rels_name not in names:
                continue
            rels = z.read(rels_name).decode("utf-8")
            for target in re.findall(r'Target="\.\./charts/(chart\d+\.xml)"', rels):
                chart = etree.fromstring(z.read(f"ppt/charts/{target}"), parser)
                totals = {}
                for ser in chart.iter(f"{c}ser"):
                    label = "".join(v.text or "" for v in ser.iterfind(f"{c}tx//{c}v"))
                    values = [v.text for v in ser.iterfind(f"{c}val//{c}v")]
                    if len(values) == 1:
                        totals[_norm(label).strip()] = values[0]
                if "total satisfaits" in totals and "total mecontents" in totals:
                    lines.append("TOTAL Satisfaits TOTAL Mécontents")
                    lines.append(f"{totals['total satisfaits']} {totals['total mecontents']}")
    return "\n".join(lines)


def collection_mode(text: str) -> str:
    """Mode de recueil lu dans la méthodologie : phone, online, mixed ou unknown.

    La phrase « Les interviews ont eu lieu par téléphone » / « … ont été réalisées par
    questionnaire auto-administré en ligne [et par téléphone] » est cherchée sans tenir compte
    des espaces, que l'extraction PDF perd parfois (« administreenligneetpartelephone »)."""
    compact = _norm(text).replace(" ", "")
    for anchor in ("interviewsont", "modederecueil"):
        i = compact.find(anchor)
        if i >= 0:
            window = compact[i : i + 300]
            break
    else:
        return "unknown"
    phone = "telephone" in window
    online = "enligne" in window or "auto-administre" in window
    if phone and online:
        return "mixed"
    return "phone" if phone else "online" if online else "unknown"


SPONSORS = (
    (r"journal du dimanche|\bjdd\b", "JDD"),
    (r"ouest[- ]france", "Ouest-France"),
    (r"pour l'opinion", "L'Opinion"),
    (r"sud radio", "Sud Radio"),
)


def sponsor(text: str) -> str:
    """Commanditaire(s) cité(s) en première page ou dans la méthodologie."""
    head = _norm(text[:4000])
    found = [name for pattern, name in SPONSORS if re.search(pattern, head)]
    return "+".join(found) or "unknown"


@dataclass
class IfopParse:
    observations: list[Observation]
    retrieved_at: dict[date, datetime]  # mois de publication -> heure de téléchargement du PDF
    report: TextReport
    publications: list[dict]  # mois, fichier, mode, commanditaire, nb d'observations
    conflicts: list[str]


def parse_collection(holders: tuple[Officeholder, ...], log=print) -> IfopParse:
    snapshot = raw.latest_raw(RAW_DIR, "json")
    listing = json.loads(snapshot.path.read_text(encoding="utf-8"))
    report = TextReport()
    observations: list[Observation] = []
    retrieved: dict[date, datetime] = {}
    publications = []
    for entry in listing["posts"]:
        path = snapshot.path.parent / entry["file"]
        meta = raw.read_meta(path)
        month = date.fromisoformat(entry["month"] + "-01")
        text = document_text(path)
        obs = parse_text(text, month, holders, report)
        observations.extend(obs)
        retrieved[month] = max(retrieved.get(month, meta.retrieved_at), meta.retrieved_at)
        publications.append(
            {
                "month": entry["month"],
                "file": entry["file"],
                "mode": collection_mode(text),
                "sponsor": sponsor(text),
                "n_table": sum(o.kind == "table" for o in obs),
                "n_headline": sum(o.kind == "headline" for o in obs),
            }
        )
    chosen, conflicts = merge(observations)
    return IfopParse(chosen, retrieved, report, publications, conflicts)


def merge(observations: list[Observation]) -> tuple[list[Observation], list[str]]:
    """Une valeur par (rôle, mois) : celle de la publication la plus récente. Les désaccords
    entre publications sont signalés, pas corrigés."""
    by_key: dict[tuple[str, date], list[Observation]] = defaultdict(list)
    for o in observations:
        by_key[(o.role, o.month)].append(o)
    chosen, conflicts = [], []
    for key in sorted(by_key):
        obs = sorted(by_key[key], key=lambda o: (o.publication, o.kind == "headline"))
        best = obs[-1]
        values = Counter((o.person_id, o.satisfied) for o in obs)
        if len(values) > 1:
            detail = ", ".join(f"{p}={v:g} ×{n}" for (p, v), n in sorted(values.items()))
            conflicts.append(f"{key[0]} {key[1].isoformat()[:7]} : {detail} -> {best.satisfied:g}")
        chosen.append(best)
    return chosen, conflicts
