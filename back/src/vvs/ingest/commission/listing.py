"""Listage et téléchargement des notices (voir la structure du site dans `__init__`)."""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from unidecode import unidecode

from vvs import paths
from vvs.ingest.commission.aliases import find_pollster_in_title

BASE_URL = "https://www.commission-des-sondages.fr"
TAG_URL = f"{BASE_URL}/notices/medias/fichiers/bytag/14/2027-Presidentielle"
SEARCH_URL = f"{BASE_URL}/notices/medias/fichiers/search"
NOTICE_URL = f"{BASE_URL}/notices/medias/fichiers/add/{{file_id}}"
USER_AGENT = (
    "vingtvingtsept/0.1 (prevision presidentielle 2027, recherche non commerciale; "
    "1 requete toutes les 2 s)"
)
REQUEST_DELAY_S = 2.0

RAW_DIR = paths.RAW / "commission"
INDEX_PATH = RAW_DIR / "index.jsonl"

MONTHS_FR = {
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

# Titre d'une notice : « 10325 Pres Focus droite et centre CLUSTER17 Le Point 18 octobre ».
# Le jeton « Pres » (ou « Présidentielle », « Presitrack ») marque la présidentielle ; le
# thème « 2027 - Présidentielle » du site l'attribue aussi, mais pas systématiquement.
_PRES_TOKEN = re.compile(r"\b(pres|presidentielles?|presitrack)\b", re.IGNORECASE)
_NUMBER = re.compile(r"^\s*(\d{4,5}\s?[A-Za-z]?)\b\s*")
_TITLE_DATE = re.compile(
    r"\b(1er|\d{1,2})\s+(" + "|".join(MONTHS_FR) + r")\b(?:\s+(20\d\d))?", re.IGNORECASE
)
_MONTH_LABEL = re.compile(r"(" + "|".join(MONTHS_FR) + r")\s+(20\d\d)", re.IGNORECASE)


@dataclass
class NoticeEntry:
    """Une ligne de liste du site, telle qu'affichée, plus quelques champs dérivés."""

    file_id: int  # identifiant de fichier dans l'URL /notices/medias/fichiers/add/<file_id>
    number: str  # numéro de notice de la Commission, en tête du titre (« 10325 », « 10251 A »)
    url: str
    title: str
    month: str  # rubrique mensuelle de dépôt (« Octobre 2026 »)
    date: str | None  # date du titre (souvent la publication), ISO, si lisible
    institute: str | None  # institut tel qu'affiché dans le titre
    sponsor: str | None  # texte entre l'institut et la date : commanditaire tel qu'affiché
    tagged_2027: bool  # présent sous le thème « 2027 - Présidentielle »


def _norm(text: str) -> str:
    return unidecode(text).lower()


def month_start(label: str) -> date | None:
    m = _MONTH_LABEL.search(_norm(label))
    if not m:
        return None
    return date(int(m.group(2)), MONTHS_FR[m.group(1)], 1)


def _title_date(title: str, month: date | None) -> date | None:
    m = _TITLE_DATE.search(_norm(title))
    if not m or month is None:
        return None
    day = 1 if m.group(1).lower() == "1er" else int(m.group(1))
    mon = MONTHS_FR[m.group(2).lower()]
    year = int(m.group(3)) if m.group(3) else month.year
    # Une notice déposée en janvier peut porter une date de décembre de l'année précédente.
    if not m.group(3) and mon - month.month > 6:
        year -= 1
    try:
        return date(year, mon, day)
    except ValueError:
        return None


def parse_title(title: str, month_label: str) -> dict:
    """Découpe un titre de notice en numéro, institut, commanditaire et date."""
    number_m = _NUMBER.match(title)
    number = number_m.group(1).replace(" ", "") if number_m else ""
    rest = title[number_m.end() :] if number_m else title
    month = month_start(month_label)
    d = _title_date(rest, month)
    institute = sponsor = None
    found = find_pollster_in_title(rest)
    if found:
        _, start, end = found
        institute = rest[start:end].strip()
        tail = rest[end:]
        dm = _TITLE_DATE.search(_norm(tail))
        # unidecode garde la longueur pour les lettres accentuées latines courantes
        tail = tail[: dm.start()] if dm else tail
        sponsor = re.sub(r"\s+", " ", tail).strip(" -,") or None
    return {
        "number": number,
        "date": d.isoformat() if d else None,
        "institute": institute,
        "sponsor": sponsor,
    }


def parse_listing(page_html: str, tagged: bool) -> list[NoticeEntry]:
    """Extrait les notices d'une page de liste (thème, recherche ou accueil)."""
    soup = BeautifulSoup(page_html, "lxml")
    entries = []
    for p in soup.select("p.download-line"):
        a = p.find("a", href=re.compile(r"/fichiers/add/\d+"))
        if a is None:
            continue
        file_id = int(re.search(r"/add/(\d+)", a["href"]).group(1))
        title = re.sub(r"\s+", " ", html.unescape(a.get_text())).strip()
        # Les pages thème et recherche suffixent « - Octobre 2026 » ; l'accueil met le mois
        # en titre de panneau (h2.notices-mois-titre).
        suffix = p.get_text().replace(a.get_text(), "")
        month_label = suffix.strip(" -\n\t")
        if not month_start(month_label):
            panel = p.find_parent("dd")
            h2 = panel.find("h2") if panel else None
            month_label = h2.get_text(strip=True) if h2 else ""
        info = parse_title(title, month_label)
        entries.append(
            NoticeEntry(
                file_id=file_id,
                number=info["number"],
                url=NOTICE_URL.format(file_id=file_id),
                title=title,
                month=month_label,
                date=info["date"],
                institute=info["institute"],
                sponsor=info["sponsor"],
                tagged_2027=tagged,
            )
        )
    return entries


def is_presidential(entry: NoticeEntry) -> bool:
    return entry.tagged_2027 or bool(_PRES_TOKEN.search(_norm(entry.title)))


def merge_listings(tagged: list[NoticeEntry], searched: list[NoticeEntry]) -> list[NoticeEntry]:
    """Union du thème 2027 et des résultats de recherche portant le jeton présidentielle."""
    by_id = {e.file_id: e for e in tagged}
    for e in searched:
        if e.file_id not in by_id and is_presidential(e):
            by_id[e.file_id] = e
    return sorted(by_id.values(), key=lambda e: e.file_id)


class Client:
    """Client HTTP poli : User-Agent explicite et délai minimal entre deux requêtes."""

    def __init__(self, delay_s: float = REQUEST_DELAY_S):
        self.delay_s = delay_s
        self._last = 0.0
        self.http = httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60.0
        )

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last = time.monotonic()

    def get(self, url: str) -> httpx.Response:
        self._wait()
        r = self.http.get(url)
        r.raise_for_status()
        return r

    def post(self, url: str, data: dict) -> httpx.Response:
        self._wait()
        r = self.http.post(url, data=data)
        r.raise_for_status()
        return r


def fetch_listing(client: Client) -> list[NoticeEntry]:
    tagged = parse_listing(client.get(TAG_URL).text, tagged=True)
    searched = parse_listing(
        client.post(SEARCH_URL, {"_method": "POST", "searchterm": "Pres"}).text, tagged=False
    )
    return merge_listings(tagged, searched)


def existing_file(file_id: int, raw_dir: Path = RAW_DIR) -> Path | None:
    for p in raw_dir.glob(f"{file_id}.*"):
        return p
    return None


def _extension(resp: httpx.Response) -> str:
    suffix = Path(resp.url.path).suffix.lower()
    if suffix in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
        return suffix
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype:
        return ".pdf"
    raise ValueError(f"type de fichier inattendu pour {resp.url}: {ctype}")


def load_index(index_path: Path = INDEX_PATH) -> dict[int, dict]:
    if not index_path.exists():
        return {}
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    return {r["file_id"]: r for r in rows}


def ingest(since: date, client: Client | None = None, raw_dir: Path = RAW_DIR) -> dict:
    """Liste les notices présidentielles depuis `since` et télécharge les nouvelles.

    Idempotent : un fichier déjà présent n'est pas retéléchargé ; l'index est en ajout seul
    (une ligne par notice, écrite au premier téléchargement).
    """
    client = client or Client()
    raw_dir.mkdir(parents=True, exist_ok=True)
    index_path = raw_dir / "index.jsonl"
    index = load_index(index_path)
    entries = fetch_listing(client)
    kept = [e for e in entries if (month_start(e.month) or date.min) >= since.replace(day=1)]
    stats = {"listed": len(entries), "kept": len(kept), "downloaded": 0, "skipped": 0, "failed": []}
    with index_path.open("a") as out:
        for e in kept:
            path = existing_file(e.file_id, raw_dir)
            if path is None:
                try:
                    resp = client.get(e.url)
                except httpx.HTTPStatusError as exc:
                    # Lien mort côté site (vu : 404 sur la notice 10251 A) : on le signale et
                    # on n'indexe pas, la prochaine exécution réessaiera.
                    stats["failed"].append((e.file_id, e.title, exc.response.status_code))
                    continue
                path = raw_dir / f"{e.file_id}{_extension(resp)}"
                path.write_bytes(resp.content)
                stats["downloaded"] += 1
                file_url = str(resp.url)
            else:
                stats["skipped"] += 1
                file_url = index.get(e.file_id, {}).get("file_url")
            if e.file_id not in index:
                row = asdict(e) | {
                    "file_url": file_url,
                    "file": path.name,
                    "ingested_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                index[e.file_id] = row
    return stats


# ------------------------------------------------------------ archives 2017 et 2022
#
# Les thèmes « 2022 - 1er tour présidentielle » et « 2022 - 2 tour présidentielle » du site
# répondent vides (relevé du 2026-10-20) et les pages par année répondent 500 ; les titres de
# 2021-2022 portent rarement le jeton « Pres » (« RDP Ifop PM », « OpinionWay - Kea Partners -
# Les Echos »). La recherche par titre étant une recherche de sous-chaîne, l'union des
# recherches sur quelques voyelles donne la quasi-totalité du catalogue (2 135 notices sur
# environ 2 340 identifiants) ; on garde ensuite la fenêtre de dépôt de l'élection, moins les
# notices explicitement d'une autre élection ou d'une sous-population régionale. Le tri entre
# intentions de vote présidentielles et autres enquêtes se fait à la lecture du PDF.

ARCHIVE_WINDOWS: dict[int, tuple[date, date]] = {
    2017: (date(2016, 4, 1), date(2017, 4, 30)),
    2022: (date(2021, 1, 1), date(2022, 4, 30)),
}
ARCHIVE_SEARCH_TERMS = ("e", "a", "o", "i", "Pres")
# Autres élections et sondages régionaux (dont les présidentielles par région d'Ipsos pour
# France 3 en 2017) : jetons du titre normalisé.
_OTHER_ELECTION = re.compile(
    r"\b(reg|regionales?|regiotrack|leg|legislatives?|municipales?|europeennes?|circo"
    r"|circonscription|idf|hdf|paca|aura|ara|bfc|pdll|pdl|occ|occitanie|normandie|bretagne"
    r"|grand est|centre|nouvelle aquitaine|nouvele aquitaine|auvergne|bourgogne|hauts de france"
    r"|haut de france|pays de la loire|ile de france|corse|var|eure)\b"
)


def archive_dir(election: int, raw_dir: Path = RAW_DIR) -> Path:
    """Dossier des notices d'une présidentielle passée : data/raw/commission/<élection>/."""
    return raw_dir / str(election)


def is_archive_candidate(entry: NoticeEntry, election: int) -> bool:
    start, end = ARCHIVE_WINDOWS[election]
    month = month_start(entry.month)
    if month is None or not (start.replace(day=1) <= month <= end):
        return False
    return not _OTHER_ELECTION.search(_norm(entry.title))


def fetch_full_listing(client: Client) -> list[NoticeEntry]:
    by_id: dict[int, NoticeEntry] = {}
    for term in ARCHIVE_SEARCH_TERMS:
        page = client.post(SEARCH_URL, {"_method": "POST", "searchterm": term}).text
        for e in parse_listing(page, tagged=False):
            by_id.setdefault(e.file_id, e)
    return sorted(by_id.values(), key=lambda e: e.file_id)


def ingest_archive(
    election: int, client: Client | None = None, raw_dir: Path = RAW_DIR, limit: int | None = None
) -> dict:
    """Télécharge les notices de la présidentielle `election` (2017 ou 2022).

    Même discipline que `ingest` : délai entre requêtes, idempotent, index en ajout seul, mais
    dans `data/raw/commission/<élection>/` pour ne pas mêler les élections.
    """
    if election not in ARCHIVE_WINDOWS:
        raise ValueError(f"élection d'archive non gérée : {election}")
    client = client or Client()
    out_dir = archive_dir(election, raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    index = load_index(index_path)
    entries = fetch_full_listing(client)
    kept = [e for e in entries if is_archive_candidate(e, election)][:limit]
    stats = {"listed": len(entries), "kept": len(kept), "downloaded": 0, "skipped": 0, "failed": []}
    with index_path.open("a") as out:
        for e in kept:
            path = existing_file(e.file_id, out_dir)
            if path is None:
                try:
                    resp = client.get(e.url)
                    ext = _extension(resp)
                except (httpx.HTTPError, ValueError) as exc:
                    code = getattr(getattr(exc, "response", None), "status_code", str(exc))
                    stats["failed"].append((e.file_id, e.title, code))
                    continue
                path = out_dir / f"{e.file_id}{ext}"
                path.write_bytes(resp.content)
                stats["downloaded"] += 1
                file_url = str(resp.url)
            else:
                stats["skipped"] += 1
                file_url = index.get(e.file_id, {}).get("file_url")
            if e.file_id not in index:
                row = asdict(e) | {
                    "election": election,
                    "file_url": file_url,
                    "file": path.name,
                    "ingested_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                index[e.file_id] = row
    return stats
