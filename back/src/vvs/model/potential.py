"""Potentiel électoral et rejet : lecture, normalisation par institut, résumés à une date.

Les mesures publiées ne sont pas comparables telles quelles : « pourriez-vous voter » oui/non
(Toluna Harris Interactive) donne Le Pen à 40 % quand la note de 7 à 10 du même institut la
donne à 38 %, et le potentiel « possible + fortes chances + certain » de Kantar donne Macron à
45 % en avril 2017 quand la note de 7 à 10 d'Ipsos le donne à 30 %. Chaque `measure`
(institut × échelle, colonne de electoral_potential.parquet) reçoit donc un décalage sur
l'échelle logit,

    logit(p_mesuré) = logit(p_référence) + α_m,   α_m = β_échelle(m) + γ_m,

estimé sur les couples de mesures d'une même personnalité, même élection, à moins de
`PAIR_DAYS` jours d'écart. β (effet de l'échelle et du seuil) a un prior large, γ (écart
propre à l'institut) un prior étroit centré sur 0 : un institut sans recouvrement garde le
décalage de son échelle. L'échelle de référence est `five_level` (Kantar 2017 et 2022 : « je
ne voterai jamais » / « possible, fortes chances ou certain »), la seule mesurée aux deux
élections où le second tour est connu ; les valeurs normalisées se lisent comme des
potentiels et des rejets « à la Kantar ». Le pont entre 2027 et Kantar passe par l'échelle
`zero_ten` (Ipsos 2017 d'un côté, Toluna Harris Interactive et Cluster17 de l'autre) :
hypothèse explicite qu'une même échelle au même seuil a le même effet d'une élection à
l'autre.

Électorats : `bloc_of_electorate` ramène `party:<parti>` (table `PARTY_BLOCS`) et
`vote_r1:<id>` / `vote_<année>:<id>` (bloc du candidat) aux blocs du modèle de second tour.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.ingest.potential import OUTPUT_NAME, PARTY_BLOCS

REFERENCE_SCALE = "five_level"
PAIR_DAYS = 21
# Écarts-types a priori (logit) : effet d'échelle, écart d'institut, bruit d'une paire
SD_SCALE = 1.0
SD_HOUSE = 0.3
SD_PAIR = 0.15
EPS = 0.5  # points : plancher des parts avant logit


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 100 - EPS) / 100.0
    return np.log(p) - np.log1p(-p)


def _expit(x: np.ndarray | float) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def load(curated: Path | None = None, verified_only: bool = True) -> pl.DataFrame:
    """Table des potentiels ; par défaut sans les relevés que la vérification contre le brut
    n'a pas confirmés (brut inaccessible, extrait paraphrasé)."""
    path = (paths.CURATED if curated is None else Path(curated)) / OUTPUT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `vvs parse potential`")
    df = pl.read_parquet(path)
    return df.filter(pl.col("verified")) if verified_only else df


# --------------------------------------------------------------------------- normalisation


@dataclass(frozen=True)
class Offsets:
    """Décalages logit par mesure (mesuré − référence) pour une quantité."""

    value: str  # potential_share | reject_share
    by_measure: dict[str, float]
    by_scale: dict[str, float]
    n_pairs: dict[str, int]
    scale_of: dict[str, str]

    def offset(self, measure: str, scale: str | None = None) -> float:
        if measure in self.by_measure:
            return self.by_measure[measure]
        return self.by_scale.get(scale or self.scale_of.get(measure, ""), 0.0)


def _pairs(df: pl.DataFrame, value: str, pair_days: int) -> list[tuple[str, str, float]]:
    """Couples (mesure a, mesure b, logit a − logit b) d'une même personnalité, même élection,
    ensemble des inscrits, à moins de `pair_days` jours ; chaque observation n'est appariée
    qu'à l'observation la plus proche de chaque autre mesure."""
    x = df.filter(
        (pl.col("electorate") == "all")
        & pl.col(value).is_not_null()
        & pl.col("candidate_id").is_not_null()
        & pl.col("date").is_not_null()
    ).select("election", "candidate_id", "measure", "date", value)
    out = []
    for (_, _), g in x.group_by("election", "candidate_id"):
        rows = g.sort("date").iter_rows(named=True)
        rows = list(rows)
        for k, a in enumerate(rows):
            best: dict[str, tuple[int, dict]] = {}
            for b in rows:
                if b["measure"] == a["measure"]:
                    continue
                d = abs((a["date"] - b["date"]).days)
                if d <= pair_days and (b["measure"] not in best or d < best[b["measure"]][0]):
                    best[b["measure"]] = (d, b)
            for m, (_, b) in best.items():
                if a["measure"] < m:  # une paire par couple de mesures et observation a
                    out.append((a["measure"], m, float(_logit(a[value]) - _logit(b[value]))))
    return out


def fit_offsets(
    df: pl.DataFrame,
    value: str = "potential_share",
    reference: str = REFERENCE_SCALE,
    pair_days: int = PAIR_DAYS,
    exclude_elections: Iterable[int] = (),
) -> Offsets:
    """Moindres carrés pénalisés sur les différences appariées : α_m = β_s(m) + γ_m, β de la
    référence fixé à 0, priors N(0, SD_SCALE²) sur β et N(0, SD_HOUSE²) sur γ."""
    excl = set(exclude_elections)
    d = df.filter(~pl.col("election").is_in(list(excl))) if excl else df
    scale_of = {
        m: s for m, s in d.select("measure", "scale").unique().iter_rows()
    }
    pairs = _pairs(d, value, pair_days)
    measures = sorted(scale_of)
    scales = sorted({s for s in scale_of.values() if s != reference})
    im = {m: k for k, m in enumerate(measures)}
    isc = {s: len(measures) + k for k, s in enumerate(scales)}
    n = len(measures) + len(scales)
    rows, y = [], []
    for a, b, diff in pairs:
        r = np.zeros(n)
        for m, sgn in ((a, 1.0), (b, -1.0)):
            r[im[m]] += sgn
            if scale_of[m] != reference:
                r[isc[scale_of[m]]] += sgn
        rows.append(r / SD_PAIR)
        y.append(diff / SD_PAIR)
    # mesures de l'échelle de référence : écart d'institut fixé à 0 (elles *sont* la référence)
    house = [1e4 if scale_of[m] == reference else 1 / SD_HOUSE for m in measures]
    prior = np.diag(house + [1 / SD_SCALE] * len(scales))
    a_mat = np.vstack([*rows, prior]) if rows else prior
    b_vec = np.concatenate([np.asarray(y), np.zeros(n)]) if rows else np.zeros(n)
    coef, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    by_scale = {s: float(coef[isc[s]]) for s in scales} | {reference: 0.0}
    by_measure = {
        m: float(coef[im[m]]) + by_scale.get(scale_of[m], 0.0) for m in measures
    }
    n_pairs = {m: sum(1 for a, b, _ in pairs if m in (a, b)) for m in measures}
    return Offsets(value, by_measure, by_scale, n_pairs, scale_of)


def normalize(
    df: pl.DataFrame, potential: Offsets | None = None, reject: Offsets | None = None
) -> pl.DataFrame:
    """Ajoute `potential_norm` et `reject_norm` (échelle de référence, en %)."""
    potential = potential or fit_offsets(df, "potential_share")
    reject = reject or fit_offsets(df, "reject_share")
    pot_off = [potential.offset(m, s) for m, s in df.select("measure", "scale").iter_rows()]
    rej_off = [reject.offset(m, s) for m, s in df.select("measure", "scale").iter_rows()]
    p = df["potential_share"].to_numpy().astype(float)
    r = df["reject_share"].to_numpy().astype(float)
    with np.errstate(invalid="ignore"):
        pn = np.where(np.isnan(p), np.nan, _expit(_logit(np.nan_to_num(p, nan=50.0)) - np.array(pot_off)))
        rn = np.where(np.isnan(r), np.nan, _expit(_logit(np.nan_to_num(r, nan=50.0)) - np.array(rej_off)))
    return df.with_columns(
        pl.Series("potential_norm", pn).fill_nan(None),
        pl.Series("reject_norm", rn).fill_nan(None),
    )


# --------------------------------------------------------------------------- résumés


@dataclass(frozen=True)
class PotentialSummary:
    candidate_id: str
    electorate: str
    potential: float | None  # normalisé, %
    reject: float | None  # normalisé, %
    n: int
    last_date: dt.date | None

    @property
    def ceiling(self) -> float | None:
        """Plafond de la part des inscrits : 100 − rejet, à défaut None."""
        return None if self.reject is None else 100.0 - self.reject


def summary_at(
    df: pl.DataFrame,
    election: int,
    candidate_id: str,
    asof: dt.date,
    electorate: str = "all",
    window_days: int = 60,
    max_age: int = 240,
) -> PotentialSummary | None:
    """Moyenne logit des valeurs normalisées publiées dans les `window_days` jours précédant
    la dernière mesure disponible à `asof` (au plus `max_age` jours avant `asof`)."""
    if "potential_norm" not in df.columns:
        raise ValueError("normaliser d'abord (`normalize`)")
    x = df.filter(
        (pl.col("election") == election)
        & (pl.col("candidate_id") == candidate_id)
        & (pl.col("electorate") == electorate)
        & (pl.col("date") <= asof)
        & (pl.col("date") > asof - dt.timedelta(days=max_age))
    )
    if x.is_empty():
        return None
    last = x["date"].max()
    x = x.filter(pl.col("date") > last - dt.timedelta(days=window_days))

    def mean(col: str) -> float | None:
        v = x[col].drop_nulls().to_numpy()
        return None if len(v) == 0 else float(_expit(np.mean(_logit(v))))

    return PotentialSummary(candidate_id, electorate, mean("potential_norm"), mean("reject_norm"), x.height, last)


def bloc_of_electorate(code: str, election: int, registry=None) -> str | None:
    """Bloc du modèle de second tour d'un électorat (`party:…`, `vote_r1:…`, `vote_<a>:…`)."""
    if code == "all":
        return None
    head, _, tail = code.partition(":")
    if head == "party":
        return PARTY_BLOCS.get(tail)
    if head.startswith("vote_"):
        from vvs.registry import default_registry

        registry = registry or default_registry()
        year = election if head == "vote_r1" else int(head[5:])
        try:
            return registry.bloc_of(tail, year)
        except KeyError:
            return None
    return None


def bloc_summary(
    df: pl.DataFrame,
    election: int,
    candidate_id: str,
    bloc: str,
    asof: dt.date,
    window_days: int = 120,
    max_age: int = 365,
    registry=None,
) -> PotentialSummary | None:
    """Potentiel et rejet d'une personnalité dans un bloc d'électeurs : moyenne logit des
    électorats (partis, votes) qui relèvent de ce bloc, sur la fenêtre de la dernière mesure."""
    x = df.filter(
        (pl.col("election") == election)
        & (pl.col("candidate_id") == candidate_id)
        & (pl.col("electorate") != "all")
        & (pl.col("date") <= asof)
        & (pl.col("date") > asof - dt.timedelta(days=max_age))
    )
    if x.is_empty():
        return None
    blocs = [bloc_of_electorate(e, election, registry) for e in x["electorate"].to_list()]
    x = x.with_columns(pl.Series("bloc", blocs, dtype=pl.Utf8)).filter(pl.col("bloc") == bloc)
    if x.is_empty():
        return None
    last = x["date"].max()
    x = x.filter(pl.col("date") > last - dt.timedelta(days=window_days))

    def mean(col: str) -> float | None:
        v = x[col].drop_nulls().to_numpy()
        return None if len(v) == 0 else float(_expit(np.mean(_logit(v))))

    return PotentialSummary(candidate_id, f"bloc:{bloc}", mean("potential_norm"), mean("reject_norm"), x.height, last)


def logit(p):  # réexport pour les études
    return _logit(p)


def expit(x):
    return _expit(x)

