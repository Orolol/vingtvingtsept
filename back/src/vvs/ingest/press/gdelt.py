"""GDELT DOC 2.0 (<https://api.gdeltproject.org/api/v2/doc/doc>), sans clé.

Deux usages (docs/06-presse.md, docs/notes/veille-2026-10-20.md §3.2) :

- **timelines** par requête de candidat ou de thème : `timelinevolraw` (nombre d'articles par
  jour et total surveillé, `norm`) et `timelinetone` (ton moyen), avec les filtres
  `sourcelang:french sourcecountry:FR`, du 1er janvier 2017 à hier. Les modes timeline
  couvrent toute la fenêtre demandée. Le 20 octobre 2026, une fenêtre du 1er janvier 2017 à
  la veille (3 538 jours) revient encore au jour (`date_resolution: day`) : on prend donc des
  fenêtres de `window_months` = 120 mois par défaut, deux requêtes par candidat. Une réponse
  dont le pas n'est pas le jour est refusée (ni parsée, ni mise en cache) : réduire alors
  `window_months`. `timelinevol` (le même volume en pourcentage) reste demandable ;
  `timelinevolraw` lui est préféré parce qu'il donne aussi le compte brut ;
- **artlist** jour par jour pour les titres (250 articles au plus par requête ; ce mode ne
  regarde que les trois derniers mois de la fenêtre demandée). Une fenêtre saturée est
  coupée en deux jusqu'à l'heure. Réservé à des fenêtres ciblées : un jour coûte une requête
  au moins, soit plus de trois heures par an et par requête au débit autorisé.

Constats du 20 octobre 2026 (sondes dans runs/2026-10-20-press/) :

- pour un article en français, la recherche porte sur sa traduction anglaise automatique :
  la phrase "Le Pen" (« the Pen ») ne ramène presque rien, le mot Pen beaucoup ; d'où les
  `gdelt_aliases` de press_queries.yaml. `timelinelang` sur `sourcecountry:france` en
  janvier 2017 donne d'ailleurs plus de volume « English » que « French » ;
- le total surveillé (`norm`) baisse fortement : environ 624 000 articles par jour en 2017,
  283 000 en 2022, 103 000 en 2026, et la requête Macron passe de 296 000 articles en 2017
  à 55 000 en 2023. Le volume brut n'est donc pas comparable d'une année à l'autre ;
  `volume_share` corrige le niveau global, pas les changements de composition des sources.

Client : `min_interval` secondes au moins entre la fin d'une requête et la suivante (GDELT
demande 5 s), reprise sur 429 et 5xx avec attentes croissantes (`BACKOFF_S`), cache immuable sous
data/raw/press/gdelt/<mode>/<année>/ et journal data/raw/press/gdelt/requests.jsonl.
GDELT renvoie certaines erreurs de requête en texte brut avec un statut 200 (« phrase trop
courte »…) : elles lèvent `GdeltQueryError` et ne sont pas mises en cache.

Sortie `parse_timelines()` -> data/curated/press_timelines.parquet (`TIMELINE_SCHEMA`), une
ligne par (source, query_id, date) :

- `volume` : nombre d'articles correspondants (timelinevolraw) ;
- `total_monitored` : nombre total d'articles surveillés par GDELT ce jour-là (`norm`) ;
- `volume_share` : `volume / total_monitored` en pour cent, ce que renvoie timelinevol ;
- `tone` : ton moyen GDELT des articles correspondants (échelle native, environ −10 à +10,
  pas l'échelle −1 à 1 de notre classification) ; nul les jours sans article, où GDELT
  renvoie 0.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx
import polars as pl

from vvs import paths
from vvs.ingest.press import items_frame
from vvs.ingest.press.cache import DiskCache, is_volatile, params_key
from vvs.ingest.press.outlets import Outlets, PressQueries, PressQuery, gdelt_query

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "vingtvingtsept/0.0.1 (recherche ; +https://github.com/Orolol)"
TIMEOUT = httpx.Timeout(120.0, connect=30.0)
# Mesuré le 20 octobre 2026 : une requête lancée moins de 5 s après la FIN de la
# précédente reçoit souvent un 429, et tout 429 ouvre une pénalité de quatre à cinq
# minutes que chaque nouvelle requête semble prolonger (429 à 60, 120 et 240 s, 200 à
# 300 s). D'où un intervalle compté depuis la fin de la requête et des attentes d'au moins
# cinq minutes.
MIN_INTERVAL_S = 8.0
BACKOFF_S = (300, 330, 600, 900, 900, 900)
MAX_RECORDS = 250
TIMELINE_MODES = ("timelinevolraw", "timelinetone")
ALL_TIMELINE_MODES = ("timelinevolraw", "timelinevol", "timelinetone")
OUTPUT_NAME = "press_timelines.parquet"

TIMELINE_SCHEMA = pl.Schema(
    {
        "source": pl.Utf8,
        "query_id": pl.Utf8,
        "candidate_id": pl.Utf8,
        "date": pl.Date,
        "volume": pl.Int64,
        "total_monitored": pl.Int64,
        "volume_share": pl.Float64,
        "tone": pl.Float64,
    }
)


class GdeltError(RuntimeError):
    pass


class GdeltRateLimited(GdeltError):
    """429 persistants malgré toutes les attentes de BACKOFF_S."""


class GdeltQueryError(GdeltError):
    """Réponse non JSON : GDELT rejette la requête (message en texte brut)."""


@dataclass
class Fetched:
    path: Path
    data: dict
    from_cache: bool


@dataclass
class GdeltClient:
    http: httpx.Client | None = None
    cache: DiskCache = field(default_factory=lambda: DiskCache("gdelt"))
    min_interval: float = MIN_INTERVAL_S
    backoff: tuple[float, ...] = BACKOFF_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = print
    _last: float | None = None
    n_requests: int = 0
    n_429: int = 0

    def __post_init__(self) -> None:
        if self.http is None:
            self.http = httpx.Client(
                headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True
            )

    # --- réseau

    def _throttle(self) -> None:
        if self._last is not None:
            wait = self.min_interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)

    def request(self, params: dict) -> bytes:
        """GET avec débit limité et reprises ; renvoie le corps JSON (octets) validé."""
        attempt = 0
        while True:
            self._throttle()
            self.n_requests += 1
            t0 = time.monotonic()
            status: int | str
            try:
                r = self.http.get(API_URL, params=params)
                status = r.status_code
            except httpx.TransportError as e:
                r, status = None, f"transport:{type(e).__name__}"
            elapsed = round(time.monotonic() - t0, 2)
            self._last = self.clock()  # l'intervalle court depuis la fin de la requête
            if r is not None and status == 200:
                body = r.content
                try:
                    json.loads(body.decode("utf-8", errors="replace"), strict=False)
                except json.JSONDecodeError:
                    msg = body.decode("utf-8", errors="replace").strip()[:300]
                    self.cache.log(params=params, status=200, error=msg, elapsed_s=elapsed)
                    raise GdeltQueryError(f"GDELT refuse la requête : {msg}") from None
                self.cache.log(params=params, status=200, bytes=len(body), elapsed_s=elapsed)
                return body
            if status == 429:
                self.n_429 += 1
            if attempt >= len(self.backoff):
                self.cache.log(params=params, status=status, elapsed_s=elapsed, gave_up=True)
                if status == 429:
                    raise GdeltRateLimited(
                        f"429 persistants après {attempt} reprises ({sum(self.backoff):.0f} s d'attente)"
                    )
                raise GdeltError(f"échec HTTP {status} après {attempt} reprises")
            if r is not None and status != 429 and status < 500:
                self.cache.log(params=params, status=status, elapsed_s=elapsed)
                r.raise_for_status()
            wait = self.backoff[attempt]
            self.cache.log(params=params, status=status, elapsed_s=elapsed, wait_s=wait)
            self.log(f"  GDELT {status}, nouvelle tentative dans {wait:.0f} s")
            self.sleep(wait)
            attempt += 1

    def fetch(
        self,
        kind: str,
        year: int,
        name: str,
        params: dict,
        validate: Callable[[dict], object] | None = None,
    ) -> Fetched:
        """Réponse en cache si elle existe, sinon requête puis écriture (jamais réécrite).

        `validate` lève sur une réponse inutilisable, qui n'est alors pas mise en cache.
        """
        path = self.cache.path_for(kind, year, name)
        cached = self.cache.get(path)
        if cached is not None:
            return Fetched(path, _loads(cached.content), True)
        body = self.request(params)
        if validate is not None:
            validate(_loads(body))
        stored = self.cache.put(path, body, {"url": API_URL, "params": params})
        return Fetched(stored.path, _loads(body), False)

    # --- usages

    def timeline(self, q: PressQuery, mode: str, start: date, end: date, filters: str) -> Fetched:
        if mode not in ALL_TIMELINE_MODES:
            raise ValueError(f"mode {mode} hors {ALL_TIMELINE_MODES}")
        params = {
            "query": gdelt_query(q, filters),
            "mode": mode,
            "format": "json",
            "startdatetime": f"{start:%Y%m%d}000000",
            "enddatetime": f"{end:%Y%m%d}235959",
        }
        return self.fetch(
            mode,
            start.year,
            _name(q.query_id, start, end, params),
            params,
            validate=lambda data: parse_timeline(data, mode),
        )

    def articles(
        self, q: PressQuery, start: datetime, end: datetime, filters: str
    ) -> list[Fetched]:
        """artlist sur [start, end) ; une fenêtre saturée (250) est coupée en deux jusqu'à 1 h."""
        params = {
            "query": gdelt_query(q, filters),
            "mode": "artlist",
            "format": "json",
            "maxrecords": MAX_RECORDS,
            "sort": "dateasc",
            "startdatetime": f"{start:%Y%m%d%H%M%S}",
            "enddatetime": f"{end - timedelta(seconds=1):%Y%m%d%H%M%S}",
        }
        name = f"{q.query_id}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}_{params_key(params)}"
        if is_volatile(end.date()):
            name += f"_asof{date.today():%Y%m%d}"
        got = self.fetch("artlist", start.year, name, params)
        n = len(got.data.get("articles") or [])
        if n >= MAX_RECORDS and end - start > timedelta(hours=1):
            mid = start + (end - start) / 2
            mid = (
                mid.replace(minute=0, second=0, microsecond=0)
                if (end - start) > timedelta(hours=2)
                else mid
            )
            return self.articles(q, start, mid, filters) + self.articles(q, mid, end, filters)
        if n >= MAX_RECORDS:
            self.log(f"  artlist saturée sur une heure ({start:%Y-%m-%d %H:%M}) : liste tronquée")
        return [got]


def _loads(body: bytes) -> dict:
    # strict=False : les titres contiennent parfois des caractères de contrôle bruts
    data = json.loads(body.decode("utf-8", errors="replace"), strict=False)
    return data if isinstance(data, dict) else {}


def _name(query_id: str, start: date, end: date, params: dict) -> str:
    name = f"{query_id}_{start:%Y%m%d}_{end:%Y%m%d}_{params_key(params)}"
    if is_volatile(end):
        name += f"_asof{date.today():%Y%m%d}"
    return name


# --- fenêtres


def windows(start: date, end: date, months: int = 12) -> Iterator[tuple[date, date]]:
    """Fenêtres [début, fin] incluses, alignées sur le 1er janvier, de `months` mois
    (diviseur ou multiple de 12)."""
    if 12 % months and months % 12:
        raise ValueError("months doit diviser 12 ou en être un multiple")
    cur = date(start.year, 1 + ((start.month - 1) // months) * months, 1)
    while cur <= end:
        m = cur.month - 1 + months
        nxt = date(cur.year + m // 12, m % 12 + 1, 1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


# --- parsing


def parse_timeline(data: dict, mode: str) -> list[tuple[date, float, float | None]]:
    """(jour, valeur, norm) d'une réponse timeline ; vide si GDELT n'a rien trouvé (`{}`).

    Échoue si la série n'est pas journalière (GDELT adapte la résolution à la fenêtre).
    """
    timeline = data.get("timeline")
    if not timeline:
        return []
    if len(timeline) != 1:
        raise GdeltError(f"{mode}: {len(timeline)} séries, une attendue")
    resolution = (data.get("query_details") or {}).get("date_resolution")
    if resolution not in (None, "day"):
        raise GdeltError(
            f"{mode}: résolution {resolution!r} au lieu de 'day' : réduire window_months"
        )
    points = []
    for p in timeline[0].get("data") or []:
        dt = datetime.strptime(p["date"], "%Y%m%dT%H%M%SZ")
        if (dt.hour, dt.minute, dt.second) != (0, 0, 0):
            raise GdeltError(f"{mode}: pas infra-journalier ({p['date']}) : fenêtre trop courte")
        norm = p.get("norm")
        points.append((dt.date(), float(p["value"]), None if norm is None else float(norm)))
    days = [d for d, _, _ in points]
    if len(days) >= 3:
        steps = sorted((b - a).days for a, b in pairwise(days))
        if steps[len(steps) // 2] != 1:
            raise GdeltError(
                f"{mode}: pas médian de {steps[len(steps) // 2]} jours : fenêtre trop longue, "
                "réduire window_months"
            )
    return points


def parse_articles(data: dict) -> list[dict]:
    out = []
    for a in data.get("articles") or []:
        out.append(
            {
                "url": a["url"],
                "title": (a.get("title") or "").strip(),
                "domain": (a.get("domain") or "").lower(),
                "seen_at": datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ"),
                "language": a.get("language"),
                "source_country": a.get("sourcecountry"),
            }
        )
    return out


# --- ingestion et assemblage


@dataclass
class IngestStats:
    fetched: int = 0
    cached: int = 0
    failed: list[tuple[str, str, str]] = field(default_factory=list)  # query_id, fenêtre, erreur
    rate_limited: bool = False

    def summary(self) -> str:
        s = f"{self.fetched} réponses téléchargées, {self.cached} déjà en cache, {len(self.failed)} en échec"
        if self.rate_limited:
            s += " ; arrêt sur 429 persistants (relancer : le cache reprend où on s'est arrêté)"
        return s


def ingest_timelines(
    queries: PressQueries,
    query_ids: Iterable[str],
    modes: Iterable[str] = TIMELINE_MODES,
    start: date | None = None,
    end: date | None = None,
    window_months: int = 120,
    client: GdeltClient | None = None,
    log: Callable[[str], None] = print,
) -> IngestStats:
    """Télécharge les timelines manquantes ; s'arrête proprement sur 429 persistants."""
    client = client or GdeltClient(log=log)
    start = start or queries.gdelt_start
    end = end or date.today() - timedelta(days=1)
    all_q = queries.all_queries()
    stats = IngestStats()
    for qid in query_ids:
        q = all_q[qid]
        for ws, we in windows(start, end, window_months):
            for mode in modes:
                label = f"{qid} {mode} {ws}..{we}"
                try:
                    got = client.timeline(q, mode, ws, we, queries.gdelt_filters)
                except GdeltRateLimited as e:
                    stats.rate_limited = True
                    stats.failed.append((qid, f"{mode} {ws}..{we}", str(e)))
                    log(f"ARRÊT {label} : {e}")
                    return stats
                except GdeltError as e:
                    stats.failed.append((qid, f"{mode} {ws}..{we}", str(e)))
                    log(f"échec {label} : {e}")
                    continue
                if got.from_cache:
                    stats.cached += 1
                else:
                    stats.fetched += 1
                    n = len(parse_timeline(got.data, mode))
                    log(f"{label} : {n} jours -> {got.path.relative_to(client.cache.root)}")
    return stats


def ingest_articles(
    queries: PressQueries,
    query_ids: Iterable[str],
    start: date,
    end: date,
    client: GdeltClient | None = None,
    log: Callable[[str], None] = print,
) -> IngestStats:
    """artlist jour par jour sur [start, end] inclus, pour des fenêtres ciblées."""
    client = client or GdeltClient(log=log)
    all_q = queries.all_queries()
    stats = IngestStats()
    for qid in query_ids:
        q = all_q[qid]
        day = start
        while day <= end:
            d0 = datetime(day.year, day.month, day.day)
            try:
                got = client.articles(q, d0, d0 + timedelta(days=1), queries.gdelt_filters)
            except GdeltRateLimited as e:
                stats.rate_limited = True
                stats.failed.append((qid, day.isoformat(), str(e)))
                log(f"ARRÊT {qid} {day} : {e}")
                return stats
            except GdeltError as e:
                stats.failed.append((qid, day.isoformat(), str(e)))
                log(f"échec {qid} {day} : {e}")
            else:
                for f in got:
                    if f.from_cache:
                        stats.cached += 1
                    else:
                        stats.fetched += 1
                n = sum(len(f.data.get("articles") or []) for f in got)
                log(f"{qid} {day} : {n} articles")
            day += timedelta(days=1)
    return stats


def _cached_responses(cache: DiskCache, mode: str) -> Iterator[tuple[dict, dict, Path]]:
    for path in cache.iter_cached(mode):
        c = cache.get(path)
        assert c is not None
        yield _loads(c.content), c.meta, path


def parse_timelines(
    queries: PressQueries,
    cache: DiskCache | None = None,
    write: bool = True,
    log: Callable[[str], None] = print,
) -> pl.DataFrame:
    """Assemble toutes les timelines en cache -> press_timelines.parquet.

    Une requête de candidat ou de thème est retrouvée par son query_id (préfixe du nom de
    fichier). Seules les réponses dont la requête est celle du registre actuel sont
    retenues : une réponse obtenue avec d'anciens alias reste dans data/raw mais n'entre pas
    dans la table. Si plusieurs réponses couvrent un même jour (fenêtre de l'année en cours
    relancée), la plus récemment téléchargée l'emporte.
    """
    cache = cache or DiskCache("gdelt")
    all_q = queries.all_queries()
    rows: dict[str, list[tuple]] = {"vol": [], "tone": []}
    stale = 0
    for mode in ALL_TIMELINE_MODES:
        for data, meta, path in _cached_responses(cache, mode):
            qid = _query_id_of(path.stem, all_q)
            if meta["request"]["params"]["query"] != gdelt_query(all_q[qid], queries.gdelt_filters):
                stale += 1
                continue
            retrieved = meta["retrieved_at"]
            for day, value, norm in parse_timeline(data, mode):
                if mode == "timelinevolraw":
                    share = 100.0 * value / norm if norm else None
                    rows["vol"].append((qid, day, retrieved, 0, int(value), norm, share))
                elif mode == "timelinevol":
                    # rang 1 : le volume brut (rang 0) l'emporte quand les deux existent
                    rows["vol"].append((qid, day, retrieved, 1, None, None, value))
                else:
                    rows["tone"].append((qid, day, retrieved, value))
    vol = pl.DataFrame(
        rows["vol"],
        schema={
            "query_id": pl.Utf8,
            "date": pl.Date,
            "retrieved_at": pl.Utf8,
            "rank": pl.Int8,
            "volume": pl.Int64,
            "total_monitored": pl.Float64,
            "volume_share": pl.Float64,
        },
        orient="row",
    )
    vol = (
        vol.sort(
            "query_id",
            "date",
            "rank",
            pl.col("retrieved_at"),
            descending=[False, False, False, True],
        )
        .unique(["query_id", "date"], keep="first", maintain_order=True)
        .drop("rank", "retrieved_at")
        .with_columns(pl.col("total_monitored").cast(pl.Int64))
    )
    tone = pl.DataFrame(
        rows["tone"],
        schema={"query_id": pl.Utf8, "date": pl.Date, "retrieved_at": pl.Utf8, "tone": pl.Float64},
        orient="row",
    )
    tone = (
        tone.sort("query_id", "date", "retrieved_at", descending=[False, False, True])
        .unique(["query_id", "date"], keep="first", maintain_order=True)
        .drop("retrieved_at")
    )
    df = vol.join(tone, on=["query_id", "date"], how="full", coalesce=True)
    # timelinetone vaut 0 les jours sans article : ce n'est pas un ton neutre
    df = df.with_columns(
        pl.when(pl.col("volume") == 0).then(None).otherwise(pl.col("tone")).alias("tone")
    )
    cand = {qid: q.candidate_id for qid, q in all_q.items()}
    df = (
        df.with_columns(
            pl.lit("gdelt").alias("source"),
            pl.col("query_id")
            .replace_strict(cand, default=None, return_dtype=pl.Utf8)
            .alias("candidate_id"),
        )
        .select(list(TIMELINE_SCHEMA))
        .sort("query_id", "date")
    )
    validate_timelines(df)
    if stale:
        log(
            f"GDELT : {stale} réponse(s) en cache ignorée(s), requête différente du registre actuel"
        )
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(paths.CURATED / OUTPUT_NAME)
    return df


def _query_id_of(stem: str, all_q: dict[str, PressQuery]) -> str:
    # nom = <query_id>_<AAAAMMJJ>_<AAAAMMJJ>_<clé>[_asof…] ; les query_id contiennent des « _ »
    for qid in sorted(all_q, key=len, reverse=True):
        rest = stem[len(qid) :]
        if stem.startswith(qid + "_") and rest[1:9].isdigit():
            return qid
    raise GdeltError(f"fichier de cache sans requête connue : {stem}")


def validate_timelines(df: pl.DataFrame) -> None:
    if df.schema != TIMELINE_SCHEMA:
        raise ValueError(f"schéma press_timelines inattendu : {df.schema}")
    if df.select(pl.struct("source", "query_id", "date").is_duplicated().any()).item():
        raise ValueError("(source, query_id, date) en double dans press_timelines")
    if df.filter(pl.col("volume") < 0).height:
        raise ValueError("volume négatif dans press_timelines")
    bad = df.filter((pl.col("volume_share") < 0) | (pl.col("volume_share") > 100))
    if bad.height:
        raise ValueError(f"volume_share hors [0, 100] : {bad.head(3)}")


def parse_article_items(
    queries: PressQueries, outlets: Outlets, cache: DiskCache | None = None
) -> pl.DataFrame:
    """Réponses artlist en cache -> lignes press_items (source gdelt), dédoublonnées par URL."""
    cache = cache or DiskCache("gdelt")
    all_q = queries.all_queries()
    rows = []
    for data, _meta, path in _cached_responses(cache, "artlist"):
        qid = _query_id_of(path.stem, all_q)
        for a in parse_articles(data):
            rows.append(
                {
                    "source": "gdelt",
                    "outlet": outlets.match(a["url"]) or outlets.match(a["domain"]),
                    "domain": a["domain"],
                    "published_at": a["seen_at"],
                    "title": a["title"],
                    "url": a["url"],
                    "lang": _lang(a["language"]),
                    "query_id": qid,
                }
            )
    return items_frame(rows)


def _lang(language: str | None) -> str | None:
    return {"French": "fr", "English": "en"}.get(language or "", (language or "").lower() or None)
