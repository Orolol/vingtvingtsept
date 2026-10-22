"""Titres Media Cloud des candidats principaux, pour la classification NLP (docs/06-presse.md).

Endpoint `search/story-list` de la collection « France - National » (34412146), une requête
par page de `PAGE_SIZE` titres (le maximum de l'API), en suivant `pagination_token`. La
requête est celle des comptages (`mediacloud_query`, alias non ambigus de
data/registry/press_queries.yaml), donc plein texte : un article est ramené dès qu'il cite
le candidat, dans le titre ou dans le corps. `in_title` dit si un alias figure dans le titre ;
la classification du ton envers le candidat n'a de sens que pour ces titres-là. Pour ce
test, le nom de famille seul (candidates.yaml) compte aussi : l'article a déjà été ramené par
un alias non ambigu, et un titre dit « Hamon » ou « Le Pen » plus souvent que « Benoît
Hamon » (avec les seuls alias, 8 % des articles de Benoît Hamon en 2017 avaient son nom dans
le titre, contre 34 % pour Fillon). Exception : « Philippe » (prénom courant), pour Édouard
Philippe (`TITLE_SURNAME_EXCLUDED`).

Fenêtres (`WINDOWS`) : les douze semaines avant le premier tour 2017 et 2022, élargies au
lundi qui précède pour avoir des semaines complètes, et, pour 2017, jusqu'au 26 décembre 2016
afin de couvrir les quatre semaines avant l'affaire Fillon (Canard enchaîné du 25 janvier) ;
pour 2027, du 1er janvier 2026 au dernier dimanche hors fenêtre volatile du cache (trois
jours), ici le 18 octobre 2026.

Candidats (`CANDIDATES`) : les candidats officiels en 2017 (11) et 2022 (12), et pour 2027 les
douze premiers de la liste des comptages (classés par nombre de sondages), Marine Le Pen et
Jordan Bardella compris.

Budget : `plan` estime le nombre de pages à partir des comptages quotidiens déjà en table
(`press_mediacloud.parquet`, mêmes requêtes) et refuse un plan au-delà de `MAX_REQUESTS`.
Avec `sample_per_day`, un candidat trop volumineux est interrogé jour par jour, une page de
`sample_per_day` titres par jour (les premiers renvoyés par l'API) : échantillon à déclarer.

Cache : data/raw/press/mediacloud_titles/search_story-list/<année>/<clé>.json, immuable ;
une relance repart de la première page, lue dans le cache, et ne paie que les pages
manquantes. Délai de `MIN_INTERVAL_S` entre requêtes, reprise sur 429 (client), puis pause
longue sur 429 persistants.

Sortie `press_titles.parquet` : une ligne par (story_id, candidate_id) avec cycle,
publish_date, outlet (nom de la source Media Cloud, un domaine), title, url, language,
in_title. Assemblée depuis le cache seul.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
from unidecode import unidecode

from vvs import paths
from vvs.ingest.press.cache import DiskCache
from vvs.ingest.press.mediacloud import (
    PAGE_SIZE,
    MediaCloudClient,
    MediaCloudError,
    MediaCloudRateLimited,
)
from vvs.ingest.press.outlets import PressQueries, mediacloud_query

SOURCE = "mediacloud_titles"
ENDPOINT_KIND = "search_story-list"
OUTPUT_NAME = "press_titles.parquet"
MIN_INTERVAL_S = 40.0
PAUSE_ON_RATE_LIMIT_S = 900
MAX_REQUESTS = 1500
TITLE_SURNAME_EXCLUDED = frozenset({"philippe_edouard"})

WINDOWS: dict[int, tuple[date, date]] = {
    2017: (date(2016, 12, 26), date(2017, 4, 22)),
    2022: (date(2022, 1, 10), date(2022, 4, 9)),
    2027: (date(2026, 1, 1), date(2026, 10, 18)),
}

CANDIDATES: dict[int, tuple[str, ...]] = {
    2017: (
        "fillon_francois", "macron_emmanuel", "le_pen_marine", "hamon_benoit",
        "melenchon_jean_luc", "dupont_aignan_nicolas", "poutou_philippe",
        "asselineau_francois", "arthaud_nathalie", "lassalle_jean", "cheminade_jacques",
    ),
    2022: (
        "macron_emmanuel", "zemmour_eric", "pecresse_valerie", "le_pen_marine",
        "melenchon_jean_luc", "jadot_yannick", "hidalgo_anne", "roussel_fabien",
        "dupont_aignan_nicolas", "poutou_philippe", "lassalle_jean", "arthaud_nathalie",
    ),
    2027: (
        "philippe_edouard", "retailleau_bruno", "melenchon_jean_luc", "zemmour_eric",
        "glucksmann_raphael", "roussel_fabien", "arthaud_nathalie", "tondelier_marine",
        "attal_gabriel", "dupont_aignan_nicolas", "le_pen_marine", "bardella_jordan",
    ),
}


def cycle_of(start: date) -> int | None:
    """Cycle dont la fenêtre contient `start` (fenêtre entière ou jour échantillonné)."""
    for e, (a, b) in WINDOWS.items():
        if a <= start <= b:
            return e
    return None


@dataclass(frozen=True)
class TitleRequest:
    candidate_id: str
    cycle: int
    start: date
    end: date
    q: str
    page_size: int = PAGE_SIZE
    max_pages: int | None = None  # 1 en mode échantillon (premiers titres du jour)
    est_stories: int | None = None

    @property
    def est_pages(self) -> int:
        if self.max_pages is not None:
            return self.max_pages
        return max(1, math.ceil((self.est_stories or 0) / self.page_size))


def estimated_stories(counts: pl.DataFrame, cid: str, cycle: int, start: date, end: date
                      ) -> int | None:
    """Articles attendus d'après press_mediacloud.parquet (même requête), ou None."""
    c = counts.filter((pl.col("scope") == "candidate") & (pl.col("id") == cid)
                      & (pl.col("election") == cycle) & pl.col("date").is_between(start, end))
    return int(c["count"].sum()) if c.height else None


def plan(
    queries: PressQueries,
    counts: pl.DataFrame | None = None,
    cycles: Iterable[int] = (2017, 2022, 2027),
    candidates: dict[int, tuple[str, ...]] | None = None,
    max_requests: int = MAX_REQUESTS,
    sample_per_day: int | None = None,
) -> list[TitleRequest]:
    """Une requête paginée par (cycle, candidat), dans l'ordre des cycles puis des candidats.

    Sans comptages, chaque série compte pour une page (estimation basse). Si l'estimation
    dépasse `max_requests` : sans `sample_per_day`, erreur ; avec, les plus gros candidats
    passent un à un en échantillon quotidien jusqu'à tenir dans le budget.
    """
    candidates = CANDIDATES if candidates is None else candidates
    reqs: list[TitleRequest] = []
    for e in cycles:
        a, b = WINDOWS[e]
        for cid in candidates.get(e, ()):
            q = mediacloud_query(queries.candidate(cid))
            est = estimated_stories(counts, cid, e, a, b) if counts is not None else None
            reqs.append(TitleRequest(cid, e, a, b, q, est_stories=est))
    total = sum(r.est_pages for r in reqs)
    if total <= max_requests:
        return reqs
    if sample_per_day is None:
        raise ValueError(f"plan estimé à {total} requêtes, au-delà du budget {max_requests} : "
                         "réduire les candidats ou passer sample_per_day")
    out = list(reqs)
    for r in sorted(reqs, key=lambda r: -r.est_pages):
        days = [r.start + timedelta(days=k) for k in range((r.end - r.start).days + 1)]
        daily = [TitleRequest(r.candidate_id, r.cycle, d, d, r.q, page_size=sample_per_day,
                              max_pages=1) for d in days]
        i = out.index(r)
        out[i:i + 1] = daily
        total += len(daily) - r.est_pages
        if total <= max_requests:
            return out
    raise ValueError(f"échantillon quotidien insuffisant : {total} requêtes estimées")


def plan_summary(reqs: list[TitleRequest]) -> pl.DataFrame:
    rows = {}
    for r in reqs:
        k = (r.cycle, r.candidate_id)
        d = rows.setdefault(k, {"cycle": r.cycle, "candidate_id": r.candidate_id,
                                "est_stories": 0, "est_pages": 0, "sampled": False})
        d["est_stories"] += r.est_stories or 0
        d["est_pages"] += r.est_pages
        d["sampled"] |= r.max_pages is not None
    return pl.DataFrame(list(rows.values()))


def client(log: Callable[[str], None] = print, **kw) -> MediaCloudClient:
    return MediaCloudClient(min_interval=MIN_INTERVAL_S, cache=DiskCache(SOURCE), log=log,
                            **kw)


def collect(
    requests: list[TitleRequest],
    client_: MediaCloudClient | None = None,
    log: Callable[[str], None] = print,
    pause_s: float = PAUSE_ON_RATE_LIMIT_S,
    max_rounds: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Pages absentes du cache, requête après requête ; bilan (titres, requêtes, échecs).

    Sur 429 persistants, pause de `pause_s` puis reprise de la même série depuis sa première
    page (lue dans le cache) ; une autre erreur HTTP est consignée et la série suivante passe.
    """
    client_ = client_ or client(log=log)
    stats = {"stories": 0, "failed": []}
    for i, r in enumerate(requests, 1):
        before = client_.n_requests
        for attempt in range(max_rounds):
            try:
                n = sum(1 for _ in client_.story_list(r.q, r.start, r.end,
                                                      page_size=r.page_size,
                                                      max_pages=r.max_pages))
                break
            except MediaCloudRateLimited:
                log(f"  429 persistants, pause {pause_s:.0f} s (essai {attempt + 1})")
                sleep(pause_s)
            except MediaCloudError as e:
                log(f"  échec {r.candidate_id} {r.cycle} {r.start}..{r.end} : {e}")
                stats["failed"].append((r.candidate_id, r.cycle, str(r.start), str(e)[:200]))
                n = None
                break
        else:
            stats["failed"].append((r.candidate_id, r.cycle, str(r.start), "429 persistants"))
            n = None
        if n is None:
            continue
        stats["stories"] += n
        net = client_.n_requests - before
        log(f"[{i}/{len(requests)}] {r.candidate_id} {r.cycle} {r.start}..{r.end} : {n} titres "
            f"(attendu {r.est_stories}), {net} requêtes ; cumul réseau {client_.n_requests}")
    stats["requests"] = client_.n_requests
    return stats


# --- mise en table


def _norm(s: str) -> str:
    return unidecode(s).lower().replace("-", " ")


def alias_pattern(aliases: Iterable[str]) -> re.Pattern:
    """Un alias du registre, sans accents ni casse, trait d'union = espace, mot entier."""
    alts = sorted({re.escape(_norm(a)) for a in aliases}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])")


SCHEMA = pl.Schema({
    "story_id": pl.Utf8, "candidate_id": pl.Utf8, "cycle": pl.Int32, "publish_date": pl.Date,
    "outlet": pl.Utf8, "title": pl.Utf8, "url": pl.Utf8, "language": pl.Utf8,
    "in_title": pl.Boolean,
})


def title_names(queries: PressQueries) -> dict[str, tuple[str, ...]]:
    """Formes qui font compter un titre comme nommant le candidat : alias et nom de famille."""
    from vvs.registry import default_registry

    reg = default_registry().candidates
    out = {}
    for cid, q in queries.candidates.items():
        names = list(q.terms)
        if cid in reg and cid not in TITLE_SURNAME_EXCLUDED:
            names.append(reg[cid].last_name)
        out[cid] = tuple(names)
    return out


def parse_titles(queries: PressQueries, cache: DiskCache | None = None, write: bool = True,
                 log: Callable[[str], None] = print) -> pl.DataFrame:
    """Cache -> press_titles.parquet (aucune requête réseau).

    Le candidat est retrouvé par la requête `q`, le cycle par la date de début. Un même
    article ramené par deux candidats donne deux lignes ; un article sans date ni titre est
    écarté.
    """
    cache = cache or DiskCache(SOURCE)
    by_q = {mediacloud_query(q): cid for cid, q in queries.candidates.items()}
    pats = {cid: alias_pattern(names) for cid, names in title_names(queries).items()}
    rows, skipped = [], 0
    for path in cache.iter_cached(ENDPOINT_KIND):
        c = cache.get(path)
        assert c is not None
        p = c.meta["request"]["params"]
        cid = by_q.get(p["q"])
        cycle = cycle_of(date.fromisoformat(p["start"]))
        if cid is None or cycle is None:
            skipped += 1
            continue
        for s in json.loads(c.content)["stories"]:
            title = (s.get("title") or "").strip()
            day = s.get("publish_date")
            if not title or not day:
                continue
            rows.append((str(s["id"]), cid, cycle, date.fromisoformat(day[:10]),
                         s.get("media_name") or s.get("media_url"), title, s.get("url"),
                         s.get("language"), bool(pats[cid].search(_norm(title)))))
    if skipped:
        log(f"{skipped} page(s) hors registre ou hors fenêtre ignorée(s)")
    df = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    df = df.unique(["story_id", "candidate_id"], keep="first", maintain_order=True).sort(
        "cycle", "candidate_id", "publish_date", "story_id")
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(paths.CURATED / OUTPUT_NAME)
    return df
