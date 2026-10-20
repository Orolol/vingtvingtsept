"""Media Cloud, API search (<https://search.mediacloud.org/api/>), appels HTTP directs.

Le paquet `mediacloud` (5.2.0 : `requests`, `requests-ratelimiter`) n'est pas utilisé : il
n'apporte qu'un mince habillage des mêmes endpoints, que httpx couvre déjà, et on veut le
même cache immuable et le même journal que pour GDELT. Endpoints repris de
`mediacloud/api.py` 5.2.0 :

- `search/total-count` -> `{"count": {"relevant": n, "total": N}}` (story_count) ;
- `search/count-over-time` -> `{"count_over_time": {"counts": [{"date", "count",
  "total_count", "ratio"}]}}`, un point par jour (story_count_over_time) ;
- `search/story-list` -> `{"stories": [...], "pagination_token": str | null}` (story_list,
  pages de `page_size` au plus 1 000) ;
- `sources/sources/?collection_id=` -> `{"count", "next", "results": [...]}` (source_list,
  pagination par offset).

Paramètres communs : `q` (syntaxe Lucene, recherche sur titre et texte), `start`, `end`
(dates ISO incluses), `platform=onlinenews-mediacloud`, `cs` (collections) ou `ss`
(sources), séparés par des virgules. Authentification : en-tête
`Authorization: Token <clé>`. La clé est lue dans la variable d'environnement
MEDIACLOUD_API_KEY, jamais écrite nulle part (ni cache, ni journal, ni méta).

Quotas (docs/notes/veille-2026-10-20.md §3.1) : 4 000 requêtes par semaine, certains
endpoints limités à 2 par minute ; le client officiel plafonne à 10 par minute. On espace
donc les requêtes de `MIN_INTERVAL_S` secondes et on reprend sur 429 avec attentes
croissantes.

Cache : data/raw/press/mediacloud/<endpoint>/<année>/<clé>.json, immuable (voir
`vvs.ingest.press.cache`) ; une page de story_list est une entrée, identifiée par les
paramètres et le jeton de pagination.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

from vvs import paths
from vvs.ingest.press import items_frame
from vvs.ingest.press.cache import DiskCache, is_volatile, params_key
from vvs.ingest.press.outlets import Outlets, PressQueries, PressQuery, mediacloud_query

API_URL = "https://search.mediacloud.org/api/"
ENV_KEY = "MEDIACLOUD_API_KEY"
PLATFORM = "onlinenews-mediacloud"
FRANCE_NATIONAL = 34412146
USER_AGENT = "vingtvingtsept/0.0.1 (+https://github.com/Orolol)"
TIMEOUT = httpx.Timeout(120.0, connect=30.0)
MIN_INTERVAL_S = 6.5  # moins de 10 requêtes par minute
BACKOFF_S = (35, 70, 140, 300, 600)
PAGE_SIZE = 1000

KEY_HELP = f"""\
Clé Media Cloud absente : la variable d'environnement {ENV_KEY} n'est pas définie.

Pour l'obtenir (une seule fois, compte personnel ; les comptes multiples sont interdits) :
  1. créer un compte sur https://search.mediacloud.org/ (bouton « Sign up ») ;
  2. confirmer l'adresse par le courriel reçu, puis accepter les conditions d'utilisation ;
  3. ouvrir son profil (menu du compte, « Your Profile ») et copier la clé d'API ;
  4. l'exporter dans le shell, hors du dépôt :
       fish : set -Ux {ENV_KEY} <clé>
       bash : export {ENV_KEY}=<clé>   (dans ~/.bashrc, pas dans un fichier du projet)
Quota par défaut : 4 000 requêtes par semaine ; l'équipe accorde davantage sur demande
motivée de recherche. Ne jamais committer la clé (.env est ignoré par git, mais on ne s'en
sert pas ici)."""


class MediaCloudError(RuntimeError):
    pass


class MediaCloudKeyMissing(MediaCloudError):
    def __init__(self) -> None:
        super().__init__(KEY_HELP)


class MediaCloudRateLimited(MediaCloudError):
    pass


def api_key() -> str:
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise MediaCloudKeyMissing()
    return key


@dataclass
class MediaCloudClient:
    key: str | None = None
    http: httpx.Client | None = None
    cache: DiskCache = field(default_factory=lambda: DiskCache("mediacloud"))
    min_interval: float = MIN_INTERVAL_S
    backoff: tuple[float, ...] = BACKOFF_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = print
    offline: bool = False  # True : cache seulement, aucune requête (parse, tests)
    _last: float | None = None
    n_requests: int = 0

    def __post_init__(self) -> None:
        if self.offline:
            return
        self.key = self.key or api_key()
        if self.http is None:
            self.http = httpx.Client(timeout=TIMEOUT, follow_redirects=True)
        self._headers = {
            "Authorization": f"Token {self.key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }

    # --- réseau

    def _throttle(self) -> None:
        if self._last is not None:
            wait = self.min_interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def request(self, endpoint: str, params: dict) -> bytes:
        if self.offline:
            raise MediaCloudError(f"client hors ligne : {endpoint} absent du cache")
        attempt = 0
        while True:
            self._throttle()
            self.n_requests += 1
            try:
                r = self.http.get(API_URL + endpoint, params=params, headers=self._headers)
                status: int | str = r.status_code
            except httpx.TransportError as e:
                r, status = None, f"transport:{type(e).__name__}"
            # jamais la clé dans le journal : seuls l'endpoint et les paramètres y vont
            if r is not None and status == 200:
                self.cache.log(endpoint=endpoint, params=params, status=200, bytes=len(r.content))
                return r.content
            if status in (401, 403):
                self.cache.log(endpoint=endpoint, params=params, status=status)
                raise MediaCloudError(
                    f"HTTP {status} : clé refusée ou conditions d'utilisation non acceptées "
                    f"sur search.mediacloud.org ({_note(r)})"
                )
            if r is not None and status != 429 and status < 500:
                self.cache.log(endpoint=endpoint, params=params, status=status)
                raise MediaCloudError(f"HTTP {status} sur {endpoint} : {_note(r)}")
            if attempt >= len(self.backoff):
                self.cache.log(endpoint=endpoint, params=params, status=status, gave_up=True)
                if status == 429:
                    raise MediaCloudRateLimited(f"429 persistants sur {endpoint}")
                raise MediaCloudError(f"échec {status} sur {endpoint} après {attempt} reprises")
            wait = self.backoff[attempt]
            self.cache.log(endpoint=endpoint, params=params, status=status, wait_s=wait)
            self.log(f"  Media Cloud {status}, nouvelle tentative dans {wait:.0f} s")
            self.sleep(wait)
            attempt += 1

    def fetch(self, endpoint: str, params: dict, year: int, end: date | None) -> dict:
        name = params_key(params)
        if end is not None and is_volatile(end):
            name += f"_asof{date.today():%Y%m%d}"
        kind = endpoint.strip("/").replace("/", "_")
        path = self.cache.path_for(kind, year, name)
        cached = self.cache.get(path)
        if cached is not None:
            return json.loads(cached.content)
        body = self.request(endpoint, params)
        self.cache.put(path, body, {"url": API_URL + endpoint, "params": params})
        return json.loads(body)

    # --- recherche

    @staticmethod
    def search_params(
        query: str,
        start: date,
        end: date,
        collection_ids: Iterable[int] = (FRANCE_NATIONAL,),
        source_ids: Iterable[int] = (),
    ) -> dict:
        params: dict = {
            "q": query,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "platform": PLATFORM,
        }
        ss, cs = list(source_ids), list(collection_ids)
        if ss:
            params["ss"] = ",".join(str(s) for s in ss)
        elif cs:
            params["cs"] = ",".join(str(c) for c in cs)
        else:
            raise ValueError("ni collection ni source : Media Cloud chercherait dans tout l'index")
        return params

    def story_count(self, query: str, start: date, end: date, **kw) -> dict:
        """{'relevant': titres correspondant à la requête, 'total': titres de la sélection}."""
        params = self.search_params(query, start, end, **kw)
        return self.fetch("search/total-count", params, start.year, end)["count"]

    def story_count_over_time(self, query: str, start: date, end: date, **kw) -> list[dict]:
        """Un point par jour : date, count (correspondants), total_count, ratio."""
        params = self.search_params(query, start, end, **kw)
        data = self.fetch("search/count-over-time", params, start.year, end)
        return [
            {**p, "date": date.fromisoformat(p["date"][:10])}
            for p in data["count_over_time"]["counts"]
        ]

    def story_list(
        self,
        query: str,
        start: date,
        end: date,
        page_size: int = PAGE_SIZE,
        max_pages: int | None = None,
        **kw,
    ) -> Iterator[dict]:
        """Titres un par un, en suivant `pagination_token` jusqu'à épuisement."""
        base = self.search_params(query, start, end, **kw)
        base["page_size"] = page_size
        token: str | None = None
        pages = 0
        while True:
            params = dict(base, pagination_token=token) if token else dict(base)
            data = self.fetch("search/story-list", params, start.year, end)
            yield from data["stories"]
            pages += 1
            token = data.get("pagination_token")
            if not token or (max_pages is not None and pages >= max_pages):
                return

    def collection_sources(
        self, collection_id: int = FRANCE_NATIONAL, limit: int = 1000
    ) -> list[dict]:
        """Sources d'une collection (DirectoryApi.source_list), toutes pages."""
        out: list[dict] = []
        offset = 0
        while True:
            params = {"collection_id": collection_id, "limit": limit, "offset": offset}
            data = self.fetch("sources/sources/", params, date.today().year, None)
            out += data["results"]
            offset += len(data["results"])
            if not data.get("next") or not data["results"]:
                return out


def _note(r: httpx.Response | None) -> str:
    if r is None:
        return ""
    try:
        return str(r.json().get("note") or r.json())[:200]
    except (ValueError, AttributeError):
        return r.text[:200]


# --- usages


def panel_source_ids(client: MediaCloudClient, outlets: Outlets) -> dict[str, list[int]]:
    """outlet_id -> ids Media Cloud des sources de la collection dont le nom est un domaine du panel."""
    out: dict[str, list[int]] = {}
    for s in client.collection_sources():
        oid = outlets.match(s.get("name") or "") or outlets.match(s.get("homepage") or "")
        if oid and outlets.outlets[oid].panel:
            out.setdefault(oid, []).append(int(s["id"]))
    return out


def daily_counts(
    client: MediaCloudClient, q: PressQuery, start: date, end: date, chunk_days: int = 366, **kw
) -> pl.DataFrame:
    """Titres par jour pour une requête (count-over-time par tranches d'au plus un an)."""
    rows = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        for p in client.story_count_over_time(mediacloud_query(q), cur, stop, **kw):
            rows.append((q.query_id, p["date"], p["count"], p.get("total_count")))
        cur = stop + timedelta(days=1)
    return pl.DataFrame(
        rows,
        schema={"query_id": pl.Utf8, "date": pl.Date, "count": pl.Int64, "total_count": pl.Int64},
        orient="row",
    )


def measure_coverage(
    years: Iterable[int],
    client: MediaCloudClient | None = None,
    outlets: Outlets | None = None,
    collection_id: int = FRANCE_NATIONAL,
) -> pl.DataFrame:
    """Nombre de titres par mois dans la collection (et par titre du panel si `outlets`).

    La couverture de Media Cloud avant 2018 est douteuse (collections géographiques peuplées
    en 2018, sources ajoutées en 2013 et 2017) : cette mesure décide si 2012 et 2017 sont
    utilisables. Requête `*` : tous les titres. Une ligne par (scope, mois), `scope` valant
    `collection:<id>` ou l'outlet_id. Coût : une requête par année et par scope.
    """
    client = client or MediaCloudClient()
    scopes: list[tuple[str, dict]] = [
        (f"collection:{collection_id}", {"collection_ids": [collection_id]})
    ]
    if outlets is not None:
        for oid, ids in sorted(panel_source_ids(client, outlets).items()):
            scopes.append((oid, {"source_ids": ids, "collection_ids": []}))
    rows = []
    for year in years:
        for scope, kw in scopes:
            pts = client.story_count_over_time("*", date(year, 1, 1), date(year, 12, 31), **kw)
            for p in pts:
                rows.append((scope, p["date"], p.get("total_count") or p["count"]))
    df = pl.DataFrame(
        rows, schema={"scope": pl.Utf8, "date": pl.Date, "stories": pl.Int64}, orient="row"
    )
    return (
        df.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("scope", "month")
        .agg(pl.col("stories").sum(), pl.col("date").n_unique().alias("days_with_data"))
        .sort("scope", "month")
    )


def write_coverage(df: pl.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.RUNS / f"{date.today().isoformat()}-press"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mediacloud_coverage.csv"
    df.write_csv(out)
    return out


def ingest_story_lists(
    queries: PressQueries,
    query_ids: Iterable[str],
    start: date,
    end: date,
    client: MediaCloudClient | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Titres de la collection pour chaque requête sur [start, end] ; renvoie le nombre vu."""
    client = client or MediaCloudClient(log=log)
    all_q = queries.all_queries()
    n = 0
    for qid in query_ids:
        k = sum(1 for _ in client.story_list(mediacloud_query(all_q[qid]), start, end))
        log(f"{qid} {start}..{end} : {k} titres")
        n += k
    return n


def parse_story_items(
    queries: PressQueries, outlets: Outlets, cache: DiskCache | None = None
) -> pl.DataFrame:
    """Pages story_list en cache -> lignes press_items (source mediacloud)."""
    cache = cache or DiskCache("mediacloud")
    by_query = {mediacloud_query(q): qid for qid, q in queries.all_queries().items()}
    rows = []
    for path in cache.iter_cached("search_story-list"):
        c = cache.get(path)
        assert c is not None
        qid = by_query.get(c.meta["request"]["params"]["q"])
        if qid is None:
            continue  # requête ad hoc, hors registre
        for s in json.loads(c.content)["stories"]:
            day = s.get("publish_date")
            if not day:
                continue
            rows.append(
                {
                    "source": "mediacloud",
                    "outlet": outlets.match(s["url"]) or outlets.match(s.get("media_url") or ""),
                    "domain": (s.get("media_name") or s.get("media_url") or "").lower(),
                    "published_at": date.fromisoformat(day[:10]),
                    "title": (s.get("title") or "").strip(),
                    "url": s["url"],
                    "lang": s.get("language"),
                    "query_id": qid,
                }
            )
    return items_frame(rows)
