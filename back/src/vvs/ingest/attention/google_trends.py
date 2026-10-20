"""Google Trends par candidat, par accès HTTP direct aux points d'entrée du site.

Il n'y a pas d'API officielle ouverte (l'API alpha de Google est sur liste d'attente) et
pytrends n'est pas une dépendance du projet. Le site trends.google.com interroge deux points
d'entrée, reproduits ici avec httpx :

1. `GET /trends/api/explore?hl=fr&tz=-60&req=<JSON>` où `req` liste au plus cinq termes
   (`comparisonItem` : terme, `geo` = FR, `time` = "AAAA-MM-JJ AAAA-MM-JJ") ; la réponse,
   préfixée par `)]}'`, décrit des widgets, dont `TIMESERIES` avec un `token` et une
   `request` ;
2. `GET /trends/api/widgetdata/multiline?hl=fr&tz=-60&req=<request du widget>&token=<token>`
   renvoie, préfixé par `)]}',`, `default.timelineData` : une entrée par pas de temps
   (`time` = horodatage Unix du début, `value` = un entier de 0 à 100 par terme,
   `hasData`, `isPartial` pour la période en cours).

Il faut d'abord un cookie `NID`, déposé par `GET /trends/explore` (qui répond 429 mais pose
le cookie). Constats du 20 octobre 2026 : sans cookie, `explore` répond 429 ; avec, explore
et multiline répondent 200 à quelques secondes d'intervalle. Google limite le débit sans le
documenter : `MIN_INTERVAL_S` entre deux requêtes, attentes croissantes sur 429 (`BACKOFF_S`)
avec un nouveau cookie, puis `TrendsRateLimited` : relancer plus tard reprend depuis le
cache.

Termes : le **sujet** Trends du candidat (`trends.topic`, identifiant Knowledge Graph
`/m/…` ou `/g/…`, trouvé par `suggest_topics` et vérifié à la main dans le registre), qui
regroupe les requêtes « Macron », « Emmanuel Macron », etc. ; à défaut le mot-clé.

Comparabilité. Une série Trends est relative : dans une requête, chaque valeur est la part
de recherche du terme ce jour-là, ramenée à 100 pour le maximum de tous les termes de la
requête sur la fenêtre, et arrondie à l'entier. Deux requêtes ne sont donc comparables que
par un terme commun. On procède en deux étapes, avec un **pivot** (`trends.pivot` du
registre, un candidat présent dans le débat sur toute la période) :

- dans une fenêtre, chaque lot contient le pivot et au plus quatre candidats ; les lots sont
  mis dans l'unité du lot où le pivot est le plus haut (moyenne du pivot = 1), par le
  rapport des sommes du pivot sur les semaines de son quintile supérieur
  (`normalize_window`) ;
- les fenêtres (`window_plan` : trois ans, pas de deux ans, donc un an de recouvrement, pour
  rester au pas hebdomadaire) sont chaînées de proche en proche : la fenêtre k+1 est
  multipliée par le rapport des sommes du pivot normalisé sur les semaines communes avec la
  fenêtre k. L'unité finale est la moyenne hebdomadaire du pivot sur la première fenêtre
  (= 100). Une semaine couverte par deux fenêtres prend la moyenne des deux estimations.

Limites : (i) l'arrondi à l'entier rend imprécis tout terme très en dessous du maximum du
lot. Le cas grave est le pivot écrasé par un candidat dominant (un pic hebdomadaire de
Sarkozy, de Hollande ou de Fillon à 100 ramène le pivot à des 0 et des 1 sur trois ans) :
l'échelle du lot repose alors sur quelques dizaines d'unités. Le lot est signalé quand la
somme du pivot utilisée pour l'échelle est inférieure à `MIN_PIVOT_COUNT`, et ses valeurs
gardées. Une échelle propre demanderait une échelle de termes intermédiaires (pivot <
candidat moyen < candidat dominant), non faite ; (ii) Google échantillonne : deux requêtes identiques à quelques jours d'écart
diffèrent de quelques unités ; (iii) Google a changé sa collecte au 1er janvier 2011 et au
1er janvier 2016 (ruptures signalées par le site), que le chaînage sur un pivot ne corrige
qu'en partie.

Cache brut, sous data/raw/attention/trends/ :

    <début>_<fin>/<clé>.explore.json     # réponse explore, octets reçus (préfixe compris)
    <début>_<fin>/<clé>.multiline.json   # réponse multiline

`<clé>` est l'empreinte des termes et de la fenêtre. La dernière fenêtre se termine à la date
d'interrogation, qui fait partie de son nom ; à la lecture, pour une fenêtre donnée, seuls
les fichiers de la fin la plus tardive sont lus.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from vvs.ingest.attention import AttentionRegistry, RawCache, params_key

BASE = "https://trends.google.com"
COOKIE_URL = f"{BASE}/trends/explore"
EXPLORE_URL = f"{BASE}/trends/api/explore"
MULTILINE_URL = f"{BASE}/trends/api/widgetdata/multiline"
AUTOCOMPLETE_URL = f"{BASE}/trends/api/autocomplete/"
# Google sert une page d'erreur aux clients qui ne ressemblent pas à un navigateur.
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")
HL, TZ, GEO = "fr", "-60", "FR"
TIMEOUT = httpx.Timeout(60.0, connect=20.0)
MIN_INTERVAL_S = 20.0
JITTER_S = 10.0
BACKOFF_S = (60, 180, 600, 1200)
BATCH_SIZE = 4  # candidats par lot, plus le pivot (cinq termes au plus)
WINDOW_YEARS = 3
STEP_YEARS = 2
TOP_PIVOT_QUANTILE = 0.8
MIN_PIVOT_COUNT = 50


class TrendsError(RuntimeError):
    pass


class TrendsRateLimited(TrendsError):
    """429 persistants malgré toutes les attentes."""


def strip_prefix(text: str) -> dict:
    """Retire le préfixe anti-JSON `)]}'` (ou `)]}',`) et décode."""
    i = text.find("{")
    if i < 0:
        raise TrendsError(f"réponse non JSON : {text[:200]!r}")
    return json.loads(text[i:])


# ---------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Window:
    start: date
    end: date

    @property
    def key(self) -> str:
        return f"{self.start:%Y%m%d}_{self.end:%Y%m%d}"

    @property
    def time(self) -> str:
        return f"{self.start.isoformat()} {self.end.isoformat()}"


def window_plan(start: date, end: date, years: int = WINDOW_YEARS, step: int = STEP_YEARS
                ) -> list[Window]:
    """Fenêtres de `years` ans tous les `step` ans ; la dernière s'arrête à `end`."""
    out = []
    y = start.year
    while True:
        w0 = date(y, start.month, start.day)
        w1 = min(date(y + years, start.month, start.day) - timedelta(days=1), end)
        out.append(Window(w0, w1))
        if w1 >= end:
            return out
        y += step


def cycle_span(reg: AttentionRegistry, election: int) -> tuple[date, date]:
    """(lendemain du premier tour précédent, premier tour) ; 2002 commence en 1997."""
    prev = [d for e, d in reg.first_rounds.items() if e < election]
    d0 = max(prev) + timedelta(days=1) if prev else date(election - 5, 1, 1)
    return d0, reg.first_rounds[election]


def window_members(reg: AttentionRegistry, w: Window) -> list[str]:
    """Candidats d'un cycle qui recoupe la fenêtre, pivot exclu, triés."""
    ids = set()
    for e, members in reg.cycles.items():
        d0, d1 = cycle_span(reg, e)
        if d0 <= w.end and d1 >= w.start:
            ids |= set(members)
    ids.discard(reg.trends_pivot)
    return sorted(c for c in ids if reg.candidates[c].trends_term)


def batches(reg: AttentionRegistry, w: Window, size: int = BATCH_SIZE) -> list[list[str]]:
    ids = window_members(reg, w)
    return [[reg.trends_pivot, *ids[i:i + size]] for i in range(0, len(ids), size)]


def explore_request(terms: list[str], w: Window) -> dict:
    return {"comparisonItem": [{"keyword": t, "geo": GEO, "time": w.time} for t in terms],
            "category": 0, "property": ""}


# ---------------------------------------------------------------------------- client


@dataclass
class TrendsClient:
    http: httpx.Client | None = None
    cache: RawCache = field(default_factory=lambda: RawCache("trends"))
    min_interval: float = MIN_INTERVAL_S
    jitter: float = JITTER_S
    backoff: tuple[float, ...] = BACKOFF_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    rand: Callable[[], float] = random.random
    log: Callable[[str], None] = print
    _last: float | None = None
    _has_cookie: bool = False
    requests: int = 0

    def __post_init__(self):
        if self.http is None:
            self.http = httpx.Client(
                headers={"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9"},
                timeout=TIMEOUT, follow_redirects=True,
            )

    def _wait(self) -> None:
        if self._last is not None:
            dt = self.min_interval + self.jitter * self.rand() - (self.clock() - self._last)
            if dt > 0:
                self.sleep(dt)

    def _cookie(self) -> None:
        self._wait()
        r = self.http.get(COOKIE_URL, params={"geo": GEO, "hl": HL})
        self._last = self.clock()
        self.requests += 1
        self.cache.log(url=COOKIE_URL, status=r.status_code, cookie="NID" in self.http.cookies)
        self._has_cookie = True

    def get(self, url: str, params: dict) -> str:
        """Texte d'une réponse 200 ; reprise sur 429 et 5xx avec un nouveau cookie."""
        if not self._has_cookie:
            self._cookie()
        for attempt in range(len(self.backoff) + 1):
            self._wait()
            try:
                r = self.http.get(url, params=params)
                status, text = r.status_code, r.text
            except httpx.TransportError as e:
                status, text = None, str(e)
            self._last = self.clock()
            self.requests += 1
            self.cache.log(url=url, params=params, status=status, bytes=len(text))
            if status == 200:
                return text
            if status is None or status == 429 or status >= 500:
                if attempt < len(self.backoff):
                    wait = self.backoff[attempt]
                    self.log(f"  {status} sur {url} : attente {wait} s, nouveau cookie")
                    self.sleep(wait)
                    self._cookie()
                    continue
                raise TrendsRateLimited(f"{url} : {status} après {len(self.backoff)} reprises")
            raise TrendsError(f"{url} {params} : statut {status} {text[:200]!r}")
        raise TrendsRateLimited(url)

    def timeseries(self, terms: list[str], w: Window) -> tuple[dict, bool]:
        """Réponse multiline d'un lot (depuis le cache si présente) et indicateur de cache."""
        key = params_key({"terms": terms, "window": w.time, "geo": GEO})
        folder = self.cache.root / w.key
        ml_path = folder / f"{key}.multiline.json"
        cached = self.cache.get(ml_path)
        if cached is not None:
            return strip_prefix(cached[0].decode("utf-8")), True
        req = explore_request(terms, w)
        ex_params = {"hl": HL, "tz": TZ, "req": json.dumps(req, ensure_ascii=False)}
        ex_text = self.get(EXPLORE_URL, ex_params)
        widgets = strip_prefix(ex_text)["widgets"]
        ts = [x for x in widgets if x.get("id") == "TIMESERIES"]
        if not ts:
            raise TrendsError(f"pas de widget TIMESERIES pour {terms} {w.time}")
        ex_path = folder / f"{key}.explore.json"
        if self.cache.get(ex_path) is None:
            self.cache.put(ex_path, ex_text.encode("utf-8"),
                           {"url": EXPLORE_URL, "params": ex_params, "terms": terms})
        ml_params = {"hl": HL, "tz": TZ, "token": ts[0]["token"],
                     "req": json.dumps(ts[0]["request"], ensure_ascii=False,
                                       separators=(",", ":"))}
        ml_text = self.get(MULTILINE_URL, ml_params)
        data = strip_prefix(ml_text)
        self.cache.put(ml_path, ml_text.encode("utf-8"),
                       {"url": MULTILINE_URL, "terms": terms, "window": w.time,
                        "resolution": ts[0]["request"].get("resolution"),
                        "params": {k: v for k, v in ml_params.items() if k != "token"}})
        return data, False

    def suggest_topics(self, name: str) -> list[dict]:
        """Sujets proposés par l'autocomplétion (mid, title, type), pour remplir le registre."""
        text = self.get(AUTOCOMPLETE_URL + name, {"hl": HL, "tz": TZ})
        return strip_prefix(text)["default"]["topics"]


# ---------------------------------------------------------------------------- ingestion


@dataclass
class IngestStats:
    windows: int = 0
    batches: int = 0
    downloaded: int = 0
    cached: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    rate_limited: bool = False

    def summary(self) -> str:
        s = (f"{self.windows} fenêtres, {self.batches} lots : {self.downloaded} téléchargés, "
             f"{self.cached} en cache, {len(self.failed)} échecs")
        if self.rate_limited:
            s += " ; arrêt sur limitation de débit (relancer plus tard)"
        return s


def ingest(reg: AttentionRegistry, end: date | None = None, client: TrendsClient | None = None,
           windows: Iterable[Window] | None = None, log: Callable[[str], None] = print
           ) -> IngestStats:
    client = client or TrendsClient(log=log)
    end = end or date.today() - timedelta(days=1)
    stats = IngestStats()
    for w in windows or window_plan(reg.trends_start, end):
        stats.windows += 1
        for ids in batches(reg, w):
            stats.batches += 1
            terms = [reg.candidates[c].trends_term for c in ids]
            try:
                _, from_cache = client.timeseries(terms, w)
            except TrendsRateLimited as e:
                stats.rate_limited = True
                stats.failed.append((f"{w.key} {ids}", str(e)))
                log(f"limitation de débit : {e}")
                return stats
            except TrendsError as e:
                stats.failed.append((f"{w.key} {ids}", str(e)))
                log(f"échec {w.key} {ids} : {e}")
                continue
            if from_cache:
                stats.cached += 1
            else:
                stats.downloaded += 1
                log(f"{w.key} : {', '.join(ids[1:])}")
    return stats


# ---------------------------------------------------------------------------- lecture


def parse_multiline(data: dict, terms: list[str]) -> pl.DataFrame:
    """(term, date, value) ; `date` = début de la période (dimanche au pas hebdomadaire).
    Période partielle exclue. `hasData` faux (volume trop faible pour Google) vaut 0, comme
    l'affiche le site : pour une part d'attention, un petit candidat compte pour rien plutôt
    que de disparaître du dénominateur."""
    rows = []
    for p in data["default"]["timelineData"]:
        if p.get("isPartial"):
            continue
        d = datetime.fromtimestamp(int(p["time"]), tz=UTC).date()
        has = p.get("hasData", [True] * len(terms))
        for t, v, h in zip(terms, p["value"], has, strict=True):
            rows.append((t, d, float(v) if h else 0.0))
    return pl.DataFrame(rows, schema={"term": pl.Utf8, "date": pl.Date, "value": pl.Float64},
                        orient="row")


def normalize_window(batches: list[pl.DataFrame], pivot_term: str
                     ) -> tuple[pl.DataFrame, list[tuple[float, float, float]]]:
    """Met les lots d'une fenêtre dans une même unité : moyenne du pivot = 1 dans le lot de
    référence (celui où le pivot est le plus haut, donc le mieux mesuré).

    Pour un autre lot, le facteur d'échelle est le rapport des sommes du pivot, lot contre
    référence, sur les semaines où le pivot de référence est dans son quintile supérieur
    (`TOP_PIVOT_QUANTILE`) : quand un candidat écrase le pivot (un pic de Sarkozy en 2007
    ramène le pivot à des 0 et des 1), ce sont les semaines où le pivot reste lisible.
    Renvoie (term, date, value) et, par lot, (moyenne du pivot, facteur, somme des valeurs du
    pivot utilisées : en dessous de `MIN_PIVOT_COUNT` le facteur est imprécis)."""
    pms = [b.filter(pl.col("term") == pivot_term)["value"].mean() or 0.0 for b in batches]
    ref = batches[int(np.argmax(pms))]
    pref = ref.filter(pl.col("term") == pivot_term).select("date", pl.col("value").alias("r"))
    if pref.is_empty() or pref["r"].mean() <= 0:
        raise TrendsError("pivot nul sur la fenêtre : fenêtre inutilisable")
    top = pref.filter(pl.col("r") >= pref["r"].quantile(TOP_PIVOT_QUANTILE))
    out, diag = [], []
    for b, pm in zip(batches, pms, strict=True):
        j = b.filter(pl.col("term") == pivot_term).join(top, on="date")
        k = float(j["value"].sum() / j["r"].sum()) if j["r"].sum() > 0 else 0.0
        diag.append((float(pm), k, float(j["value"].sum())))
        if k <= 0:
            continue
        out.append(b.with_columns(pl.col("value") / (k * pref["r"].mean())))
    if not out:
        raise TrendsError("aucun lot exploitable")
    return pl.concat(out).group_by("term", "date").agg(pl.col("value").mean()), diag


def chain_windows(per_window: dict[str, pl.DataFrame], pivot_term: str
                  ) -> tuple[pl.DataFrame, dict[str, float]]:
    """Chaîne les fenêtres (clés triées par date) sur les semaines communes du pivot.
    `per_window` : (term, date, value) en unités du pivot de la fenêtre. Renvoie (term, date,
    value) dans l'unité de la première fenêtre ×100, et le facteur de chaque fenêtre."""
    keys = sorted(per_window)
    factors: dict[str, float] = {}
    prev = None
    for k in keys:
        df = per_window[k]
        if prev is None:
            factors[k] = 1.0
        else:
            a = per_window[prev].filter(pl.col("term") == pivot_term).select("date", "value")
            b = df.filter(pl.col("term") == pivot_term).select("date", "value")
            j = a.join(b, on="date", suffix="_b")
            if j.height == 0 or j["value_b"].sum() <= 0:
                raise TrendsError(f"fenêtres {prev} et {k} sans recouvrement du pivot")
            factors[k] = factors[prev] * float(j["value"].sum() / j["value_b"].sum())
        prev = k
    out = pl.concat([per_window[k].with_columns(pl.col("value") * factors[k] * 100)
                     for k in keys])
    out = out.group_by("term", "date").agg(pl.col("value").mean()).sort("term", "date")
    return out, factors


def parse_cached(reg: AttentionRegistry, cache: RawCache | None = None
                 ) -> tuple[pl.DataFrame, list[str]]:
    cache = cache or RawCache("trends")
    notes: list[str] = []
    empty = pl.DataFrame(schema={"source": pl.Utf8, "candidate_id": pl.Utf8, "date": pl.Date,
                                 "freq": pl.Utf8, "value": pl.Float64})
    if not cache.root.exists():
        return empty, notes
    term_to_id = {q.trends_term: c for c, q in reg.candidates.items() if q.trends_term}
    pivot = reg.candidates[reg.trends_pivot].trends_term
    # une fenêtre = son début ; on garde la fin la plus tardive
    latest: dict[str, Path] = {}
    for d in sorted(p for p in cache.root.iterdir() if p.is_dir() and "_" in p.name):
        latest[d.name.split("_")[0]] = d
    per_window = {}
    for start, d in sorted(latest.items()):
        frames, labels = [], []
        for p in sorted(d.glob("*.multiline.json")):
            content, meta = cache.get(p)
            terms = meta["request"]["terms"]
            if meta["request"].get("resolution") not in (None, "WEEK"):
                notes.append(f"trends : {p.name} au pas {meta['request']['resolution']}, ignoré")
                continue
            if any(t not in term_to_id for t in terms):
                notes.append(f"trends : {p.name} contient un terme hors registre, ignoré")
                continue
            frames.append(parse_multiline(strip_prefix(content.decode("utf-8")), terms))
            labels.append([term_to_id[t] for t in terms[1:]])
        if not frames:
            continue
        try:
            per_window[d.name], diag = normalize_window(frames, pivot)
        except TrendsError as e:
            notes.append(f"trends : {d.name} : {e}")
            continue
        for lab, (pm, k, cnt) in zip(labels, diag, strict=True):
            if cnt < MIN_PIVOT_COUNT:
                notes.append(f"trends : {d.name} lot {lab} : pivot moyen {pm:.1f}, somme "
                             f"utilisée {cnt:.0f} < {MIN_PIVOT_COUNT} : échelle imprécise")
    if not per_window:
        return empty, notes
    chained, factors = chain_windows(per_window, pivot)
    notes.append("trends : facteurs de chaînage "
                 + ", ".join(f"{k} {v:.3f}" for k, v in factors.items()))
    out = chained.with_columns(
        pl.col("term").replace_strict(term_to_id).alias("candidate_id")
    ).select(pl.lit("trends").alias("source"), "candidate_id", "date",
             pl.lit("W").alias("freq"), "value")
    return out, notes
