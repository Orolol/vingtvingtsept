"""Chargement et encodage des sondages de premier tour pour le modèle d'agrégation.

Deux étapes :

1. `load_polls` lit les sondages curated (table fusionnée `polls.parquet` si elle existe,
   sinon les fichiers Wikipédia de l'élection) et rend une table longue, une ligne par
   (sondage, hypothèse, candidat), filtrée à la date `asof`.
2. `encode` transforme cette table en tableaux numpy pour le modèle : candidats et blocs,
   instituts, grille hebdomadaire, hypothèses en matrices complétées par un masque (le
   softmax du modèle est restreint aux candidats présents dans chaque hypothèse).

Voir docs/04-modele.md, section 1.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.curate.rules import one_base
from vvs.registry import BLOCS, Registry, default_registry
from vvs.schemas import RESERVED_CANDIDATE_IDS

log = logging.getLogger(__name__)

# Effectif retenu quand sample_size manque (docs/04-modele.md)
DEFAULT_SAMPLE_SIZE = 1000
# Plancher de part de bloc pour le prior structurel (bloc absent à l'élection précédente)
MIN_PRIOR_BLOC_SHARE = 0.01

LONG_COLUMNS = [
    "poll_id",
    "pollster",
    "fieldwork_start",
    "fieldwork_end",
    "sample_size",
    "is_rolling",
    "population",
    "hypothesis_id",
    "candidate_id",
    "share",
]


# --- chargement


def _thin_rolling(polls: pl.DataFrame) -> pl.DataFrame:
    """Ne garde, par institut, que des vagues de rolling dont les terrains ne se recouvrent pas.

    Les vagues successives d'un rolling partagent leurs répondants : les garder toutes
    compterait plusieurs fois la même information. En partant de la vague la plus récente,
    on garde une vague puis la suivante dont le terrain finit avant le début de celle gardée.
    """
    if "is_rolling" not in polls.columns or not polls["is_rolling"].fill_null(False).any():
        return polls
    rolling = polls.filter(pl.col("is_rolling").fill_null(False))
    keep: list[str] = []
    for _, grp in rolling.group_by("pollster"):
        grp = grp.sort("fieldwork_end", descending=True)
        start_kept: dt.date | None = None
        for row in grp.iter_rows(named=True):
            if start_kept is None or row["fieldwork_end"] < start_kept:
                keep.append(row["poll_id"])
                start_kept = row["fieldwork_start"]
    return polls.filter(~pl.col("is_rolling").fill_null(False) | pl.col("poll_id").is_in(keep))


def _dedup_canonical(polls: pl.DataFrame) -> pl.DataFrame:
    """Un seul sondage par `canonical_poll_id` (celui dont l'id est canonique s'il existe)."""
    if "canonical_poll_id" not in polls.columns:
        return polls
    polls = polls.with_columns(
        pl.coalesce("canonical_poll_id", "poll_id").alias("_canon"),
    ).with_columns((pl.col("poll_id") != pl.col("_canon")).alias("_not_canon"))
    return (
        polls.sort("_canon", "_not_canon", "poll_id")
        .unique("_canon", keep="first", maintain_order=True)
        .drop("_canon", "_not_canon")
    )


def load_polls(
    election: int,
    asof: dt.date | str | None = None,
    curated: Path | None = None,
    round_: int = 1,
) -> pl.DataFrame:
    """Sondages d'un tour, table longue (une ligne par sondage × hypothèse × candidat).

    Source : `polls.parquet` + `poll_results.parquet` (table fusionnée de la curation) s'ils
    existent, sinon `polls_wikipedia_<election>.parquet` et son pendant. Dans la table
    fusionnée, les hypothèses `suspect` sont exclues, les doublons regroupés par
    `canonical_poll_id` et les vagues de rolling éclaircies (`_thin_rolling`) ; une hypothèse
    publiée sur deux bases n'est gardée que sur une, les certains d'aller voter d'abord
    (`vvs.curate.rules.one_base`).

    Les ids réservés (abstention, indécis…) sont écartés. Seuls les sondages dont
    `fieldwork_end <= asof` sont gardés. La source lue (`curated` ou `wikipedia`) est
    exposée dans `load_polls.last_source`.
    """
    curated = paths.CURATED if curated is None else Path(curated)
    if isinstance(asof, str):
        asof = dt.date.fromisoformat(asof)
    merged_polls, merged_results = curated / "polls.parquet", curated / "poll_results.parquet"
    if merged_polls.exists() and merged_results.exists():
        polls = pl.read_parquet(merged_polls).filter(pl.col("election") == election)
        results = pl.read_parquet(merged_results)
        source = "curated"
    else:
        polls = pl.read_parquet(curated / f"polls_wikipedia_{election}.parquet")
        results = pl.read_parquet(curated / f"poll_results_wikipedia_{election}.parquet")
        source = "wikipedia"
    load_polls.last_source = source

    if "is_rolling" not in polls.columns:
        polls = polls.with_columns(pl.lit(False).alias("is_rolling"))
    if asof is not None:
        polls = polls.filter(pl.col("fieldwork_end") <= asof)
    if "suspect" in polls.columns:
        polls = polls.filter(~pl.col("suspect").fill_null(False))
    polls = _dedup_canonical(polls)
    polls = _thin_rolling(polls)

    results = results.filter(pl.col("round") == round_)
    if "suspect" in results.columns:
        bad = (
            results.filter(pl.col("suspect").fill_null(False))
            .select("poll_id", "hypothesis_id", "population")
            .unique()
        )
        results = results.join(bad, on=["poll_id", "hypothesis_id", "population"], how="anti")
    # Une hypothèse publiée sur deux bases (OpinionWay : inscrits et certains d'aller voter)
    # ne compte qu'une fois, sur les certains d'aller voter (`rules.one_base`).
    results = one_base(results)
    results = results.filter(
        ~pl.col("candidate_id").is_in(RESERVED_CANDIDATE_IDS) & pl.col("share").is_not_null()
    )
    df = results.join(
        polls.select(
            "poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size", "is_rolling"
        ),
        on="poll_id",
        how="inner",
    )
    return df.select(LONG_COLUMNS).sort("fieldwork_end", "poll_id", "hypothesis_id", "candidate_id")


load_polls.last_source = None


# --- encodage


@dataclass
class PollData:
    """Entrées du modèle.

    Le temps est une grille de pas de `step_days` jours (7 par défaut, d'où les noms en
    « week ») : index 0 = premier pas de la grille, T-1 = pas qui précède le vote.
    """

    election: int
    election_date: dt.date
    asof: dt.date
    candidates: list[str]
    blocs: list[str]
    cand_bloc: np.ndarray  # (C,) index de bloc
    pollsters: list[str]
    populations: list[str]
    n_weeks: int  # T
    asof_week: int  # index de la semaine contenant asof
    # hypothèses (N lignes, K colonnes complétées)
    cand_idx: np.ndarray  # (N, K) int, 0 sur les cases vides
    mask: np.ndarray  # (N, K) bool
    counts: np.ndarray  # (N, K) comptes arrondis sur l'effectif nominal
    n_nominal: np.ndarray  # (N,) effectif nominal (somme des comptes)
    weight: np.ndarray  # (N,) poids de vraisemblance = 1 / nb d'hypothèses du sondage
    week: np.ndarray  # (N,) index de semaine
    pollster_idx: np.ndarray  # (N,)
    pop_idx: np.ndarray  # (N,)
    poll_id: list[str]
    hypothesis_id: list[str]
    shares: np.ndarray  # (N, K) parts normalisées à 1 sur l'hypothèse
    prior_bloc_mean: np.ndarray  # (B,) moyenne a priori de β au début de la grille
    reference: list[str]  # configuration de référence (vote) pour les sorties
    dropped_candidates: dict[str, int] = field(default_factory=dict)
    n_polls: int = 0
    source: str | None = None
    prior_election: int | None = None
    step_days: int = 7
    # Palmarès des instituts calibré sans l'élection (vvs.model.pollster_ratings.attach_ratings)
    pollster_ratings: dict | None = None

    @property
    def n_hyp(self) -> int:
        return self.cand_idx.shape[0]

    def week_date(self, k: int | np.ndarray) -> list[dt.date]:
        """Date du milieu du pas de grille d'index k."""
        ks = np.atleast_1d(k)
        sd = self.step_days
        return [
            self.election_date - dt.timedelta(days=sd * int(self.n_weeks - 1 - x) + (sd + 1) // 2)
            for x in ks
        ]

    def weeks_before(self, k: int | np.ndarray) -> np.ndarray:
        return self.n_weeks - 1 - np.asarray(k)


def weeks_before_vote(day: dt.date, election_date: dt.date, step_days: int = 7) -> int:
    """Pas pleins entre un jour et le vote (0 = les `step_days` jours qui précèdent le vote)."""
    return max(0, (election_date - day).days - 1) // step_days


def _previous_election(registry: Registry, election: int) -> int | None:
    years = [
        y
        for y, e in registry.elections.items()
        if y < election and e.rounds and e.rounds[0].results
    ]
    return max(years) if years else None


def prior_bloc_shares(registry: Registry, election: int) -> tuple[dict[str, float], int | None]:
    """Part de chaque bloc au premier tour de la présidentielle précédente (fraction)."""
    prev = _previous_election(registry, election)
    shares = {b: 0.0 for b in BLOCS}
    if prev is None:
        return {b: 1 / len(BLOCS) for b in BLOCS}, None
    e = registry.elections[prev]
    blocs = {c.candidate_id: c.bloc for c in e.candidates}
    for r in e.round_(1).results:
        shares[blocs[r.candidate_id]] += r.share / 100
    return shares, prev


def official_candidates(registry: Registry, election: int) -> list[str]:
    e = registry.elections.get(election)
    return [] if e is None else [c.candidate_id for c in e.candidates]


def election_date(registry: Registry, election: int) -> dt.date:
    return registry.elections[election].round_(1).date


def encode(
    df: pl.DataFrame,
    election: int,
    asof: dt.date | str,
    n_weeks: int = 78,
    registry: Registry | None = None,
    prior_bloc: dict[str, float] | None = None,
    reference: list[str] | None = None,
    vote_date: dt.date | None = None,
    step_days: int = 7,
) -> PollData:
    """Encode la table longue de `load_polls` pour le modèle.

    - Grille : `n_weeks` semaines avant le vote, découpées en pas de `step_days` jours (le
      dernier pas précède le vote) ; les sondages plus anciens sont écartés. Un sondage est
      daté par le milieu de son terrain.
    - Candidats inconnus du registre (bloc indéterminable) : retirés de l'hypothèse et
      comptés dans `dropped_candidates`. Sous l'hypothèse IIA du modèle, les comptes
      restants d'une hypothèse suivent le même softmax restreint : on ne biaise rien, on
      perd seulement l'information sur ces candidats.
    - Hypothèses de moins de deux candidats après ce retrait : écartées.
    - `reference` : configuration retenue pour les sorties au jour du vote ; par défaut les
      candidats officiels de l'élection, sinon l'hypothèse la plus fréquente des huit
      dernières semaines.
    """
    registry = registry or default_registry()
    if isinstance(asof, str):
        asof = dt.date.fromisoformat(asof)
    edate = vote_date or election_date(registry, election)
    n_steps = (n_weeks * 7) // step_days
    T = n_steps + 1

    df = df.with_columns(
        (
            pl.col("fieldwork_start") + (pl.col("fieldwork_end") - pl.col("fieldwork_start")) / 2
        ).alias("mid")
    )
    df = df.filter(pl.col("mid") < edate)

    # Bloc de chaque candidat ; les inconnus sont retirés
    bloc_cache: dict[str, str | None] = {}
    for cid in df["candidate_id"].unique().to_list():
        try:
            bloc_cache[cid] = registry.bloc_of(cid, election)
        except KeyError:
            bloc_cache[cid] = None
    unknown = [c for c, b in bloc_cache.items() if b is None]
    dropped: dict[str, int] = {}
    if unknown:
        counts = df.filter(pl.col("candidate_id").is_in(unknown))["candidate_id"].value_counts()
        dropped = dict(zip(counts["candidate_id"].to_list(), counts["count"].to_list()))
        log.warning("candidats hors registre retirés des hypothèses : %s", dropped)
        df = df.filter(~pl.col("candidate_id").is_in(unknown))

    rows: list[dict] = []
    for (poll_id, hyp_id), grp in df.group_by(["poll_id", "hypothesis_id"], maintain_order=True):
        if grp.height < 2:
            continue
        first = grp.row(0, named=True)
        wb = weeks_before_vote(first["mid"], edate, step_days)
        if wb > n_steps:
            continue
        rows.append(
            {
                "poll_id": poll_id,
                "hypothesis_id": hyp_id,
                "pollster": first["pollster"],
                "population": first["population"] or "unknown",
                "sample_size": first["sample_size"],
                "week": n_steps - wb,
                "candidates": grp["candidate_id"].to_list(),
                "shares": grp["share"].to_numpy().astype(float),
            }
        )
    if not rows:
        raise ValueError(f"aucune hypothèse de sondage exploitable pour {election} au {asof}")

    official = official_candidates(registry, election)
    if reference is None:
        if official:
            reference = official
        else:
            last = n_steps - weeks_before_vote(asof, edate, step_days)
            recent = [r for r in rows if r["week"] >= last - (56 // step_days)]
            freq = Counter(tuple(sorted(r["candidates"])) for r in (recent or rows))
            reference = list(freq.most_common(1)[0][0])
    for c in reference:
        if c not in bloc_cache or bloc_cache[c] is None:
            bloc_cache[c] = registry.bloc_of(c, election)

    candidates = sorted({c for r in rows for c in r["candidates"]} | set(reference))
    blocs = [b for b in BLOCS if any(bloc_cache[c] == b for c in candidates)]
    cand_bloc = np.array([blocs.index(bloc_cache[c]) for c in candidates], dtype=np.int32)
    pollsters = sorted({r["pollster"] for r in rows})
    populations = sorted({r["population"] for r in rows}, key=lambda p: (p != "unknown", p))

    poll_hyps = Counter(r["poll_id"] for r in rows)
    N, K = len(rows), max(len(r["candidates"]) for r in rows)
    cand_idx = np.zeros((N, K), dtype=np.int32)
    mask = np.zeros((N, K), dtype=bool)
    counts = np.zeros((N, K), dtype=np.float64)
    shares = np.zeros((N, K), dtype=np.float64)
    n_nominal = np.zeros(N)
    for i, r in enumerate(rows):
        k = len(r["candidates"])
        cand_idx[i, :k] = [candidates.index(c) for c in r["candidates"]]
        mask[i, :k] = True
        s = np.clip(r["shares"], 0.0, None)
        s = s / s.sum()
        size = (
            r["sample_size"] if r["sample_size"] and r["sample_size"] > 0 else DEFAULT_SAMPLE_SIZE
        )
        c = np.round(s * size)
        counts[i, :k] = c
        shares[i, :k] = s
        n_nominal[i] = c.sum()

    # Prior structurel : part du bloc à la présidentielle précédente, répartie sur le nombre
    # moyen de candidats du bloc par hypothèse (exp(β_b) ≈ soutien d'un candidat moyen du bloc).
    prev = None
    if prior_bloc is None:
        prior_bloc, prev = prior_bloc_shares(registry, election)
    per_hyp = np.array(
        [[sum(bloc_cache[c] == b for c in r["candidates"]) for b in blocs] for r in rows],
        dtype=float,
    )
    m_b = np.maximum(per_hyp.mean(axis=0), 1.0)
    prior_mean = np.array(
        [math.log(max(prior_bloc.get(b, 0.0), MIN_PRIOR_BLOC_SHARE)) for b in blocs]
    ) - np.log(m_b)

    return PollData(
        election=election,
        election_date=edate,
        asof=asof,
        candidates=candidates,
        blocs=blocs,
        cand_bloc=cand_bloc,
        pollsters=pollsters,
        populations=populations,
        n_weeks=T,
        asof_week=min(T - 1, n_steps - weeks_before_vote(asof, edate, step_days)),
        cand_idx=cand_idx,
        mask=mask,
        counts=counts,
        n_nominal=n_nominal,
        weight=np.array([1.0 / poll_hyps[r["poll_id"]] for r in rows]),
        week=np.array([r["week"] for r in rows], dtype=np.int32),
        pollster_idx=np.array([pollsters.index(r["pollster"]) for r in rows], dtype=np.int32),
        pop_idx=np.array([populations.index(r["population"]) for r in rows], dtype=np.int32),
        poll_id=[r["poll_id"] for r in rows],
        hypothesis_id=[r["hypothesis_id"] for r in rows],
        shares=shares,
        prior_bloc_mean=prior_mean,
        reference=list(reference),
        dropped_candidates=dropped,
        n_polls=len(poll_hyps),
        prior_election=prev,
        step_days=step_days,
    )


def hypotheses_frame(data: PollData) -> pl.DataFrame:
    """Table longue des parts observées (normalisées), pour les figures et les contrôles."""
    out = []
    dates = data.week_date(np.arange(data.n_weeks))
    for i in range(data.n_hyp):
        for k in np.flatnonzero(data.mask[i]):
            out.append(
                {
                    "poll_id": data.poll_id[i],
                    "hypothesis_id": data.hypothesis_id[i],
                    "pollster": data.pollsters[data.pollster_idx[i]],
                    "week": int(data.week[i]),
                    "date": dates[data.week[i]],
                    "candidate_id": data.candidates[data.cand_idx[i, k]],
                    "share": float(data.shares[i, k]),
                }
            )
    return pl.DataFrame(out)
