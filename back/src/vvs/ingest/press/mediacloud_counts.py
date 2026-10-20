"""Comptages quotidiens Media Cloud par candidat et par thème (étude presse, phase 7).

Collection « France - National » (34412146), endpoint `search/count-over-time` : une requête
rend, pour chaque jour de la fenêtre, le nombre d'articles qui correspondent (`count`) et le
nombre total d'articles de la collection ce jour-là (`total_count`). Le total sert de
dénominateur : la composition de la collection change fortement d'une année à l'autre
(docs/06-presse.md, couverture mesurée le 2026-10-20), seule une part par jour est comparable.
Aucune requête « * » séparée n'est donc nécessaire pour normaliser.

Fenêtres (`WINDOWS`) : les douze mois précédant le premier tour de 2017 et de 2022 (une
requête par série), et du 1er janvier 2025 à la veille de la collecte pour 2027 (deux
requêtes par série : 2025, puis 2026 jusqu'à hier, cette dernière volatile au sens du cache).

Séries : candidats sondés au premier tour sur la fenêtre (`CANDIDATES`, alias de
data/registry/press_queries.yaml), thèmes de la liste fermée (mots-clés du même registre),
et co-mentions candidat × thème (`CO_THEMES`, scope `candidate_theme`, id
`<candidate_id>:<thème>`) : articles qui citent le candidat et un mot du thème « affaires »,
mesure grossière de la couverture négative pour tous les candidats (le ton GDELT n'existe
que pour quelques-uns et seulement depuis 2017).

Sortie `press_mediacloud.parquet` : une ligne par (scope, id, date) avec `scope` =
`candidate`, `theme` ou `candidate_theme`, `id` = candidate_id, thème ou les deux,
`election`, `count`, `total`, `share` = count / total (fraction des articles de la
collection). Assemblée depuis le cache seul (`parse_counts`), sans réseau : une réponse de
fenêtre volatile peut exister en plusieurs versions (suffixe `_asof`), la plus récente
l'emporte.

Débit : on espace les requêtes de `MIN_INTERVAL_S` (limite observée : une toutes les 35 à
70 s) ; le client reprend sur 429. Sur 429 persistants, `collect` fait une pause et reprend
la requête suivante sans perdre ce qui est en cache.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from vvs import paths
from vvs.ingest.press.cache import DiskCache
from vvs.ingest.press.mediacloud import (
    FRANCE_NATIONAL,
    MediaCloudClient,
    MediaCloudError,
    MediaCloudRateLimited,
)
from vvs.ingest.press.outlets import PressQueries, mediacloud_query

OUTPUT_NAME = "press_mediacloud.parquet"
ENDPOINT_KIND = "search_count-over-time"
MIN_INTERVAL_S = 40.0
PAUSE_ON_RATE_LIMIT_S = 900
FIRST_ROUNDS = {2017: date(2017, 4, 23), 2022: date(2022, 4, 10)}
LIVE_START = date(2025, 1, 1)
CO_THEMES = ("scandals",)

# Candidats sondés au premier tour sur la fenêtre (poll_results.parquet, relevé du
# 2026-10-20), par nombre de sondages décroissant ; les candidats d'une ou deux hypothèses
# isolées sont laissés de côté.
CANDIDATES: dict[int, tuple[str, ...]] = {
    2017: (
        "macron_emmanuel", "le_pen_marine", "fillon_francois", "melenchon_jean_luc",
        "hamon_benoit", "dupont_aignan_nicolas", "poutou_philippe", "arthaud_nathalie",
        "cheminade_jacques", "asselineau_francois", "lassalle_jean", "bayrou_francois",
        "jadot_yannick", "juppe_alain", "hollande_francois", "valls_manuel", "sarkozy_nicolas",
        "montebourg_arnaud",
    ),
    2022: (
        "macron_emmanuel", "le_pen_marine", "melenchon_jean_luc", "zemmour_eric",
        "pecresse_valerie", "jadot_yannick", "hidalgo_anne", "roussel_fabien",
        "dupont_aignan_nicolas", "lassalle_jean", "poutou_philippe", "arthaud_nathalie",
        "taubira_christiane", "montebourg_arnaud", "asselineau_francois", "bertrand_xavier",
        "philippot_florian", "barnier_michel",
    ),
    2027: (
        "philippe_edouard", "retailleau_bruno", "melenchon_jean_luc", "zemmour_eric",
        "glucksmann_raphael", "roussel_fabien", "arthaud_nathalie", "tondelier_marine",
        "attal_gabriel", "dupont_aignan_nicolas", "le_pen_marine", "bardella_jordan",
        "hollande_francois", "de_villepin_dominique", "faure_olivier", "ruffin_francois",
        "lisnard_david",
    ),
}


def windows(election: int, today: date | None = None) -> list[tuple[date, date]]:
    """Fenêtres de requête d'un cycle, chacune d'au plus 366 jours."""
    if election in FIRST_ROUNDS:
        vote = FIRST_ROUNDS[election]
        return [(vote - timedelta(days=365), vote - timedelta(days=1))]
    if election == 2027:
        end = (today or date.today()) - timedelta(days=1)
        out, y = [], LIVE_START.year
        while date(y, 1, 1) <= end:
            out.append((date(y, 1, 1), min(date(y, 12, 31), end)))
            y += 1
        return out
    raise ValueError(f"cycle sans fenêtre Media Cloud : {election}")


@dataclass(frozen=True)
class CountRequest:
    scope: str  # candidate | theme
    id: str
    election: int
    start: date
    end: date
    q: str

    def params(self) -> dict:
        return MediaCloudClient.search_params(self.q, self.start, self.end)


def plan(
    queries: PressQueries,
    elections: Iterable[int] = (2017, 2022, 2027),
    candidates: dict[int, tuple[str, ...]] | None = None,
    themes: Iterable[str] | None = None,
    today: date | None = None,
    co_themes: Iterable[str] = CO_THEMES,
) -> list[CountRequest]:
    """Requêtes à faire : candidats de chaque cycle, puis thèmes, puis co-mentions."""
    candidates = CANDIDATES if candidates is None else candidates
    themes = list(queries.themes) if themes is None else list(themes)
    out: list[CountRequest] = []
    for e in elections:
        for cid in candidates.get(e, ()):
            q = mediacloud_query(queries.candidate(cid))
            out += [CountRequest("candidate", cid, e, a, b, q) for a, b in windows(e, today)]
    for e in elections:
        for t in themes:
            q = mediacloud_query(queries.theme_query(t))
            out += [CountRequest("theme", t, e, a, b, q) for a, b in windows(e, today)]
    for t in co_themes:
        tq = mediacloud_query(queries.theme_query(t))
        for e in elections:
            for cid in candidates.get(e, ()):
                q = f"{mediacloud_query(queries.candidate(cid))} AND {tq}"
                out += [CountRequest("candidate_theme", f"{cid}:{t}", e, a, b, q)
                        for a, b in windows(e, today)]
    return out


def collect(
    requests: list[CountRequest],
    client: MediaCloudClient | None = None,
    log: Callable[[str], None] = print,
    pause_s: float = PAUSE_ON_RATE_LIMIT_S,
    max_rounds: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Fait les requêtes absentes du cache ; renvoie un bilan (faites, en cache, échecs).

    Sur 429 persistants, pause de `pause_s` puis nouvel essai, `max_rounds` fois au plus par
    requête. Une autre erreur HTTP est consignée et la collecte continue.
    """
    client = client or MediaCloudClient(min_interval=MIN_INTERVAL_S, log=log)
    stats = {"done": 0, "cached": 0, "failed": []}
    for i, r in enumerate(requests, 1):
        before = client.n_requests
        for attempt in range(max_rounds):
            try:
                pts = client.story_count_over_time(r.q, r.start, r.end)
                break
            except MediaCloudRateLimited:
                log(f"  429 persistants, pause {pause_s:.0f} s (essai {attempt + 1})")
                sleep(pause_s)
            except MediaCloudError as e:
                log(f"  échec {r.scope}:{r.id} {r.start}..{r.end} : {e}")
                stats["failed"].append((r.scope, r.id, r.election, str(e)[:200]))
                pts = None
                break
        else:
            stats["failed"].append((r.scope, r.id, r.election, "429 persistants"))
            pts = None
        if pts is None:
            continue
        net = client.n_requests - before
        stats["done" if net else "cached"] += 1
        n = sum(p["count"] for p in pts)
        log(f"[{i}/{len(requests)}] {r.scope}:{r.id} {r.election} {r.start}..{r.end} : "
            f"{len(pts)} jours, {n} articles"
            f"{'' if net else ' (cache)'} ; requêtes réseau cumulées {client.n_requests}")
    return stats


def _cached_index(cache: DiskCache) -> dict[tuple, list[tuple[str, dict]]]:
    """(q, start, end, cs) -> [(retrieved_at, réponse)] pour toutes les réponses en cache."""
    out: dict[tuple, list[tuple[str, dict]]] = {}
    for path in cache.iter_cached(ENDPOINT_KIND):
        c = cache.get(path)
        assert c is not None
        p = c.meta["request"]["params"]
        if "cs" not in p:
            continue  # comptages par source (mesure de couverture)
        key = (p["q"], p["start"], p["end"], p["cs"])
        out.setdefault(key, []).append((c.meta["retrieved_at"], json.loads(c.content)))
    return out


SCHEMA = pl.Schema({
    "scope": pl.Utf8, "id": pl.Utf8, "election": pl.Int32, "date": pl.Date,
    "count": pl.Int64, "total": pl.Int64, "share": pl.Float64,
})


def counts_frame(requests: list[CountRequest], index: dict[tuple, list[tuple[str, dict]]],
                 log: Callable[[str], None] = print) -> pl.DataFrame:
    """Lignes (scope, id, election, date, count, total, share) des requêtes présentes."""
    rows, missing = [], []
    for r in requests:
        p = r.params()
        hits = index.get((p["q"], p["start"], p["end"], p.get("cs", str(FRANCE_NATIONAL))))
        if not hits:
            missing.append(f"{r.scope}:{r.id} {r.start}..{r.end}")
            continue
        _, data = max(hits, key=lambda h: h[0])
        for pt in data["count_over_time"]["counts"]:
            day = date.fromisoformat(pt["date"][:10])
            tot = pt.get("total_count")
            rows.append((r.scope, r.id, r.election, day, int(pt["count"]),
                         None if tot is None else int(tot),
                         pt["count"] / tot if tot else None))
    if missing:
        log(f"Media Cloud : {len(missing)} série(s) absente(s) du cache : "
            + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
    df = pl.DataFrame(rows, schema=SCHEMA, orient="row")
    return df.unique(["scope", "id", "date"], keep="last", maintain_order=True).sort(
        "scope", "id", "date")


def parse_counts(queries: PressQueries, cache: DiskCache | None = None, write: bool = True,
                 today: date | None = None, log: Callable[[str], None] = print
                 ) -> pl.DataFrame:
    """Cache -> press_mediacloud.parquet (aucune requête réseau).

    La fenêtre 2027 en cours est reconstruite à partir des réponses en cache, quelle que soit
    la date de fin demandée : on prend, pour chaque série, toutes les fenêtres 2027 présentes.
    """
    cache = cache or DiskCache("mediacloud")
    index = _cached_index(cache)
    reqs = [r for r in plan(queries, (2017, 2022), today=today)]
    live = plan(queries, (2027,), today=today)
    # fenêtres 2027 effectivement en cache (fin variable selon le jour de collecte)
    by_q_start: dict[tuple[str, str], list[str]] = {}
    for (q, s, e, _cs) in index:
        by_q_start.setdefault((q, s), []).append(e)
    for r in live:
        ends = by_q_start.get((r.q, r.start.isoformat()))
        if ends:
            reqs.append(CountRequest(r.scope, r.id, r.election, r.start,
                                     date.fromisoformat(max(ends)), r.q))
        else:
            reqs.append(r)
    df = counts_frame(reqs, index, log)
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(paths.CURATED / OUTPUT_NAME)
    return df
