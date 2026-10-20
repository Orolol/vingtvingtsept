"""Pages vues Wikipédia FR par candidat (API Pageviews de la Wikimedia Foundation).

Source : <https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/fr.wikipedia/
all-access/user/{article}/daily/{début}/{fin}>, sans clé, données quotidiennes depuis le
1er juillet 2015 (les statistiques antérieures, `pagecounts`, ne distinguent ni les robots ni
le mobile et ne sont pas utilisées). Agent `user` : la Fondation y retire les robots déclarés
et, depuis 2020, les robots détectés (`automated`) ; des pointes d'automates non détectés
restent possibles. Une requête couvre toute la période demandée ; une période sans aucune
vue renvoie 404 (mis en cache comme tel : zéro vue). Les jours sans vue sont omis par l'API
et valent zéro ici.

Politique Wikimedia : User-Agent explicite avec un moyen de contact (`USER_AGENT`), pas plus
de quelques requêtes par seconde (`MIN_INTERVAL_S` entre deux requêtes), reprise sur 429 et
5xx avec attentes croissantes.

Titres suivis : une vue est comptée sous le titre exact demandé ; après un renommage,
l'ancien titre devient une redirection et garde son historique. On somme donc, pour chaque
candidat, l'article courant (`wikipedia.title` du registre), ses **anciens titres** et les
`extra_titles` du registre. Les anciens titres sont retrouvés automatiquement : parmi les
redirections sans ancre vers l'article (API MediaWiki, `prop=redirects`, espace principal seulement), celles dont le
journal de renommage (`list=logevents`, `letype=move`) mène à l'article, directement ou par
une chaîne de renommages. Les autres redirections (fautes de frappe, homonymes partiels,
anciens articles fusionnés dans une section, « Yann Le Pen » vers Jean-Marie Le Pen) ne sont
pas comptées : leur historique ne porte pas toujours sur le candidat, et leur trafic est
marginal. Limite connue : un ancien titre devenu ensuite une page d'homonymie n'est plus une
redirection et échappe à la recherche ; on l'ajoute alors à la main dans `extra_titles`.

Cache brut, sous data/raw/attention/wikipedia/ :

    titles/<AAAA-MM-JJ>/<candidate_id>/redirects.json        # prop=redirects, octets reçus
    titles/<AAAA-MM-JJ>/<candidate_id>/moves_<titre>.json    # journal de renommage
    pageviews/<titre>/<début>_<fin>.json                     # réponse Pageviews (ou 404)

Trafic automatique non détecté : une série de `PLATEAU_DAYS` jours consécutifs ou plus,
chacun au moins `PLATEAU_RATIO` fois la médiane des 28 jours qui précèdent la série, à plus
de `PLATEAU_MIN_VIEWS` vues par jour en moyenne, avec un coefficient de variation inférieur à
`PLATEAU_CV` (un plateau plat, quand un vrai pic d'actualité décroît en quelques jours) est
marquée `suspect` ; `value` y est remplacée par la
médiane des 28 jours précédents et `raw_value` garde le compte de l'API. Sur les données du
20 octobre 2026, la règle ne marque que Laurent Wauquiez du 7 au 17 août 2017 (environ
165 000 vues par jour pour une médiane de 300) ; les séries de 3 à 6 jours (soirs
d'élection, nominations, décès) ont une forme de pic, et les hausses durables plus modestes
(Philippe Juvin médecin urgentiste en mars 2020, Olivier Faure fin août 2025, quelques
milliers de vues par jour) restent sous le seuil de volume.

Les titres d'un candidat se relisent dans son instantané le plus récent. Une relance ne
demande que les jours après la fin déjà couverte (fichiers jamais réécrits). La fin par
défaut est avant-hier : les vues d'un jour sont publiées dans les 24 heures suivantes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import polars as pl

from vvs.ingest.attention import AttentionQuery, AttentionRegistry, RawCache

PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/fr.wikipedia/"
    "all-access/user/{article}/daily/{start}/{end}"
)
MEDIAWIKI_URL = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = (
    "vingtvingtsept/0.0.1 (research: French 2027 presidential forecast; "
    "+https://github.com/Orolol) httpx"
)
TIMEOUT = httpx.Timeout(60.0, connect=20.0)
MIN_INTERVAL_S = 0.5
BACKOFF_S = (10, 30, 60, 120, 300)
END_LAG_DAYS = 2
PLATEAU_DAYS, PLATEAU_RATIO, PLATEAU_CV, PLATEAU_BASE = 7, 10.0, 0.3, 28
PLATEAU_MIN_VIEWS = 20_000


class WikimediaError(RuntimeError):
    pass


def title_key(title: str) -> str:
    """Titre -> segment d'URL et nom de dossier (espaces en `_`, le reste encodé)."""
    return quote(title.replace(" ", "_"), safe="")


def default_end(today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=END_LAG_DAYS)


@dataclass
class WikiClient:
    http: httpx.Client | None = None
    cache: RawCache = field(default_factory=lambda: RawCache("wikipedia"))
    min_interval: float = MIN_INTERVAL_S
    backoff: tuple[float, ...] = BACKOFF_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = print
    _last: float | None = None
    requests: int = 0

    def __post_init__(self):
        if self.http is None:
            self.http = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
                                     follow_redirects=True)

    def _wait(self) -> None:
        if self._last is not None:
            dt = self.min_interval - (self.clock() - self._last)
            if dt > 0:
                self.sleep(dt)

    def fetch(self, path: Path, url: str, params: dict | None = None,
              ok_statuses: tuple[int, ...] = (200,)) -> tuple[bytes, dict, bool]:
        """(contenu, méta, depuis le cache). Les statuts de `ok_statuses` sont mis en cache ;
        429 et 5xx sont repris après `backoff` ; tout autre statut lève WikimediaError."""
        cached = self.cache.get(path)
        if cached is not None:
            return cached[0], cached[1], True
        request = {"url": url, "params": params or {}}
        for attempt in range(len(self.backoff) + 1):
            self._wait()
            try:
                r = self.http.get(url, params=params)
                status, content = r.status_code, r.content
            except httpx.TransportError as e:
                status, content = None, str(e).encode()
            self._last = self.clock()
            self.requests += 1
            self.cache.log(url=url, params=params, status=status, bytes=len(content))
            if status in ok_statuses:
                meta = self.cache.put(path, content, request, status=status)
                return content, meta, False
            retryable = status is None or status == 429 or status >= 500
            if retryable and attempt < len(self.backoff):
                wait = self.backoff[attempt]
                self.log(f"  {status} sur {url} : attente {wait} s")
                self.sleep(wait)
                continue
            raise WikimediaError(f"{url} {params} : statut {status} {content[:200]!r}")
        raise WikimediaError(f"{url} : échec après {len(self.backoff)} reprises")


# ---------------------------------------------------------------------------- titres


def titles_dir(cache: RawCache, candidate_id: str, snapshot: date) -> Path:
    return cache.root / "titles" / snapshot.isoformat() / candidate_id


def latest_titles_dir(cache: RawCache, candidate_id: str) -> Path | None:
    root = cache.root / "titles"
    if not root.exists():
        return None
    snaps = sorted(d for d in root.iterdir() if (d / candidate_id / "redirects.json").exists())
    return snaps[-1] / candidate_id if snaps else None


def redirects_of(redirects_json: dict) -> list[dict]:
    pages = redirects_json["query"]["pages"]
    return [r for p in pages for r in p.get("redirects", [])]


def moves_of(moves_json: dict) -> list[tuple[str, str, str]]:
    """(ancien titre, nouveau titre, horodatage) des renommages d'un journal."""
    out = []
    for ev in moves_json.get("query", {}).get("logevents", []):
        if ev.get("type") == "move" and "target_title" in ev.get("params", {}):
            out.append((ev["title"], ev["params"]["target_title"], ev.get("timestamp", "")))
    return out


def former_titles(canonical: str, moves: dict[str, list[tuple[str, str, str]]]) -> list[str]:
    """Redirections dont un renommage mène à l'article, directement ou en chaîne.
    `moves` : redirection -> ses renommages (moves_of)."""
    known = {canonical}
    changed = True
    while changed:
        changed = False
        for r, evs in moves.items():
            if r not in known and any(new in known for _, new, _ in evs):
                known.add(r)
                changed = True
    return sorted(known - {canonical})


def resolve_titles(q: AttentionQuery, client: WikiClient, snapshot: date) -> list[str]:
    """Interroge (ou relit) redirections et journaux de renommage ; renvoie les titres
    suivis : article, anciens titres, extra_titles."""
    d = titles_dir(client.cache, q.candidate_id, snapshot)
    content, _, _ = client.fetch(d / "redirects.json", MEDIAWIKI_URL, {
        "action": "query", "titles": q.wikipedia, "prop": "redirects", "rdlimit": "max",
        "rdprop": "title|fragment", "format": "json", "formatversion": "2",
    })
    data = json.loads(content)
    if "continue" in data:
        raise WikimediaError(f"{q.wikipedia} : plus de redirections qu'une page de réponse")
    for r in redirects_of(data):
        if r.get("fragment") or r.get("ns", 0) != 0:
            continue
        client.fetch(d / f"moves_{title_key(r['title'])}.json", MEDIAWIKI_URL, {
            "action": "query", "list": "logevents", "letype": "move", "letitle": r["title"],
            "lelimit": "max", "leprop": "title|timestamp|details|type", "format": "json",
            "formatversion": "2",
        })
    return titles_from_snapshot(q, d)


def titles_from_snapshot(q: AttentionQuery, d: Path) -> list[str]:
    data = json.loads((d / "redirects.json").read_bytes())
    pages = data["query"]["pages"]
    if not pages or pages[0].get("missing"):
        raise WikimediaError(f"{q.candidate_id} : article « {q.wikipedia} » introuvable")
    moves = {}
    for r in redirects_of(data):
        p = d / f"moves_{title_key(r['title'])}.json"
        if not r.get("fragment") and r.get("ns", 0) == 0 and p.exists():
            moves[r["title"]] = moves_of(json.loads(p.read_bytes()))
    titles = [q.wikipedia, *former_titles(q.wikipedia, moves)]
    return titles + [t for t in q.extra_titles if t not in titles]


# ---------------------------------------------------------------------------- vues


def _ts(d: date) -> str:
    return d.strftime("%Y%m%d") + "00"


def pageviews_path(cache: RawCache, title: str, start: date, end: date) -> Path:
    return cache.root / "pageviews" / title_key(title) / f"{start:%Y%m%d}_{end:%Y%m%d}.json"


def covered_end(cache: RawCache, title: str) -> date | None:
    """Dernier jour couvert par les fichiers en cache d'un titre."""
    files = cache.files(f"pageviews/{title_key(title)}")
    ends = [datetime.strptime(p.stem.split("_")[1], "%Y%m%d").date() for p in files]
    return max(ends) if ends else None


def fetch_pageviews(client: WikiClient, title: str, start: date, end: date) -> tuple[Path, bool]:
    path = pageviews_path(client.cache, title, start, end)
    url = PAGEVIEWS_URL.format(article=title_key(title), start=_ts(start), end=_ts(end))
    _, _, from_cache = client.fetch(path, url, ok_statuses=(200, 404))
    return path, from_cache


@dataclass
class IngestStats:
    candidates: int = 0
    titles: int = 0
    former_titles: dict[str, list[str]] = field(default_factory=dict)
    downloaded: int = 0
    up_to_date: int = 0
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [(
            f"{self.candidates} candidats, {self.titles} titres suivis : {self.downloaded} "
            f"fichiers téléchargés, {self.up_to_date} titres déjà à jour, "
            f"{len(self.failed)} échecs"
        )]
        for cid, ts in sorted(self.former_titles.items()):
            lines.append(f"  anciens titres {cid} : {', '.join(ts)}")
        if self.skipped:
            lines.append(f"  sans article : {', '.join(self.skipped)}")
        lines += [f"  échec {cid} : {err}" for cid, err in self.failed]
        return "\n".join(lines)


def ingest(reg: AttentionRegistry, ids: Iterable[str] | None = None, end: date | None = None,
           client: WikiClient | None = None, snapshot: date | None = None,
           log: Callable[[str], None] = print) -> IngestStats:
    client = client or WikiClient(log=log)
    end = end or default_end()
    snapshot = snapshot or date.today()
    stats = IngestStats()
    for cid in ids or sorted(reg.candidates):
        q = reg.candidates[cid]
        if not q.wikipedia:
            stats.skipped.append(cid)
            continue
        stats.candidates += 1
        try:
            titles = resolve_titles(q, client, snapshot)
            if len(titles) > 1:
                stats.former_titles[cid] = titles[1:]
            for t in titles:
                stats.titles += 1
                last = covered_end(client.cache, t)
                start = reg.wikipedia_start if last is None else last + timedelta(days=1)
                if start > end:
                    stats.up_to_date += 1
                    continue
                _, from_cache = fetch_pageviews(client, t, start, end)
                stats.downloaded += 0 if from_cache else 1
            log(f"{cid} : {len(titles)} titre(s)")
        except WikimediaError as e:
            stats.failed.append((cid, str(e)))
            log(f"{cid} : échec {e}")
    return stats


def parse_pageviews(content: bytes, status: int) -> list[tuple[date, int]]:
    if status == 404:
        return []
    items = json.loads(content)["items"]
    return [(datetime.strptime(it["timestamp"][:8], "%Y%m%d").date(), int(it["views"]))
            for it in items]


def title_views(cache: RawCache, title: str, start: date) -> tuple[pl.DataFrame, date | None]:
    """Vues quotidiennes d'un titre, calendrier complet de `start` à la fin couverte (zéros
    pour les jours omis). Un jour présent dans deux fichiers garde le plus récent."""
    rows: dict[date, int] = {}
    end = None
    for p in cache.files(f"pageviews/{title_key(title)}"):
        content, meta = cache.get(p)
        for d, v in parse_pageviews(content, meta.get("status", 200)):
            rows[d] = v
        e = datetime.strptime(p.stem.split("_")[1], "%Y%m%d").date()
        end = e if end is None else max(end, e)
    if end is None:
        return pl.DataFrame(schema={"date": pl.Date, "views": pl.Int64}), None
    cal = pl.date_range(start, end, "1d", eager=True).alias("date").to_frame()
    obs = pl.DataFrame({"date": list(rows), "views": list(rows.values())},
                       schema={"date": pl.Date, "views": pl.Int64})
    return cal.join(obs, on="date", how="left").with_columns(pl.col("views").fill_null(0)), end


def flag_plateaus(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(valeurs corrigées, masque suspect) selon la règle des plateaux (docstring)."""
    out, flag = v.astype(float).copy(), np.zeros(len(v), dtype=bool)
    t = PLATEAU_BASE
    while t < len(v):
        base = float(np.median(v[t - PLATEAU_BASE:t]))
        if v[t] < PLATEAU_RATIO * max(base, 1.0):
            t += 1
            continue
        u = t
        while u < len(v) and v[u] >= PLATEAU_RATIO * max(base, 1.0):
            u += 1
        run = v[t:u].astype(float)
        if (len(run) >= PLATEAU_DAYS and run.mean() >= PLATEAU_MIN_VIEWS
                and run.std() / run.mean() < PLATEAU_CV):
            out[t:u] = base
            flag[t:u] = True
        t = u
    return out, flag


def parse_cached(reg: AttentionRegistry, cache: RawCache | None = None
                 ) -> tuple[pl.DataFrame, list[str]]:
    """Somme quotidienne des titres suivis par candidat (source, candidate_id, date, freq,
    value, raw_value, suspect), jusqu'au dernier jour couvert par tous ses titres."""
    cache = cache or RawCache("wikipedia")
    parts, notes = [], []
    for cid, q in sorted(reg.candidates.items()):
        if not q.wikipedia:
            continue
        d = latest_titles_dir(cache, cid)
        if d is None:
            continue
        titles = titles_from_snapshot(q, d)
        frames, ends = [], []
        for t in titles:
            f, e = title_views(cache, t, reg.wikipedia_start)
            if e is None:
                notes.append(f"wikipedia : {cid} « {t} » sans fichier de vues")
                continue
            frames.append(f)
            ends.append(e)
        if not frames:
            continue
        end = min(ends)
        g = (pl.concat(frames).filter(pl.col("date") <= end).group_by("date")
             .agg(pl.col("views").sum()).sort("date"))
        value, flag = flag_plateaus(g["views"].to_numpy())
        if flag.any():
            days = g["date"].filter(pl.Series(flag))
            notes.append(f"wikipedia : {cid} plateau suspect du {days.min()} au {days.max()} "
                         f"({int(flag.sum())} jours), remplacé par la médiane antérieure")
        parts.append(g.select(
            pl.lit("wikipedia").alias("source"), pl.lit(cid).alias("candidate_id"), "date",
            pl.lit("D").alias("freq"), pl.Series("value", value),
            pl.col("views").cast(pl.Float64).alias("raw_value"),
            pl.Series("suspect", flag),
        ))
    if not parts:
        return pl.DataFrame(schema={"source": pl.Utf8, "candidate_id": pl.Utf8,
                                    "date": pl.Date, "freq": pl.Utf8, "value": pl.Float64,
                                    "raw_value": pl.Float64, "suspect": pl.Boolean}), notes
    return pl.concat(parts), notes
