"""Panel mensuel blocs × indicateurs macro pour l'étude économie (docs/05-economie.md).

Parts de bloc (`poll_results_by_bloc`, premier tour, base exprimés) :

- hypothèses suspectes exclues ; pour 2027, les hypothèses qui testent Emmanuel Macron
  (sondages d'octobre 2022 à octobre 2023) sont écartées, puisqu'il ne peut pas se représenter ;
- un sondage = la moyenne simple de ses hypothèses de premier tour, bloc par bloc ; un bloc
  sans candidat dans une hypothèse y est absent, pas à zéro (convention de
  `vvs.curate.derived`) ;
- un mois = la moyenne des instituts présents ce mois-là (date : fin de terrain), chaque
  institut comptant pour la moyenne de ses sondages du mois ; un rolling ne pèse donc pas
  plus qu'un sondage ponctuel ;
- un mois sans sondage reste vide : rien n'est interpolé, et la variation mensuelle
  `d_share` n'existe qu'entre deux mois consécutifs observés tous les deux ;
- cycle électoral = champ `election` des sondages ; la grille d'un cycle va du premier mois
  avec sondage (au plus tôt `PANEL_START`) au mois du premier tour (pour 2027, au dernier
  mois observé) ;
- pseudo-bloc `incumbent` : bloc du président sortant (2002 : droite, Chirac, malgré la
  cohabitation avec un gouvernement de gauche ; 2007 et 2012 : droite ; 2017 : gauche ;
  2022 et 2027 : centre).

Indicateurs (`INDICATORS`) : séries mensuelles sur un calendrier complet. Le chômage BIT
trimestriel est placé au mois central du trimestre et interpolé linéairement entre deux
trimestres (les mois avant le premier et après le dernier mois central restent vides) ; les
carburants hebdomadaires sont moyennés par mois. Chaque indicateur a un niveau `x` et une
variation mensuelle `d_x` ; voir `Indicator` pour les transformations.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import polars as pl

MAIN_BLOCS = ("far_left", "left", "centre", "right", "far_right")
INCUMBENT = "incumbent"
INCUMBENT_BLOC = {2002: "right", 2007: "right", 2012: "right", 2017: "left", 2022: "centre",
                  2027: "centre"}
BLOCS = (*MAIN_BLOCS, INCUMBENT)
PANEL_START = date(2000, 1, 1)
EXCLUDED_2027 = "macron_emmanuel"
JOBSEEKERS_BREAK = date(2025, 1, 1)  # loi plein emploi : inscriptions automatiques (RSA, jeunes)


@dataclass(frozen=True)
class Indicator:
    """`level` : `raw` (valeur telle quelle), `yoy` (glissement annuel en %), `thousands`.
    `diff` : `diff` (différence mensuelle du niveau), `logdiff` (variation mensuelle en %,
    100 × différence des logarithmes de la série brute)."""

    series_id: str
    label: str
    unit: str
    level: str = "raw"
    diff: str = "diff"
    economic: bool = True


INDICATORS: dict[str, Indicator] = {
    i.series_id: i
    for i in (
        Indicator("unemployment_rate_ilo", "Chômage BIT", "%"),
        Indicator("jobseekers_cat_a", "Demandeurs d'emploi cat. A", "milliers",
                  level="thousands", diff="logdiff"),
        Indicator("cpi_all", "Inflation IPC", "% sur un an", level="yoy"),
        Indicator("cpi_energy", "Inflation énergie", "% sur un an", level="yoy"),
        Indicator("cpi_food", "Inflation alimentation", "% sur un an", level="yoy"),
        Indicator("consumer_confidence", "Confiance des ménages", "indice"),
        Indicator("consumer_confidence_future_living", "Niveau de vie futur (solde)", "points"),
        Indicator("consumer_confidence_unemployment_expectations", "Chômage anticipé (solde)",
                  "points"),
        Indicator("business_climate", "Climat des affaires", "indice"),
        Indicator("fuel_price_sp95", "Prix SP95", "% sur un an", level="yoy"),
        Indicator("fuel_price_diesel", "Prix gazole", "% sur un an", level="yoy"),
        Indicator("approval_president", "Popularité du président", "% satisfaits",
                  economic=False),
        Indicator("approval_prime_minister", "Popularité du Premier ministre", "% satisfaits",
                  economic=False),
    )
}


def month_index(d: date) -> int:
    return d.year * 12 + d.month - 1


def _with_month(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(col).dt.truncate("1mo").alias("month"))


# ---------------------------------------------------------------------------- sondages


def poll_bloc_shares(by_bloc: pl.DataFrame, polls: pl.DataFrame) -> pl.DataFrame:
    """Une ligne par (sondage, bloc) : moyenne des hypothèses de premier tour retenues."""
    h = by_bloc.filter(
        (pl.col("round") == 1)
        & ~pl.col("suspect")
        & pl.col("bloc").is_in(MAIN_BLOCS)
        & ~(
            (pl.col("election") == 2027)
            & pl.col("hypothesis_id").str.split("+").list.contains(EXCLUDED_2027)
        )
    )
    inc = (
        h.join(
            pl.DataFrame({"election": list(INCUMBENT_BLOC), "bloc": list(INCUMBENT_BLOC.values())},
                         schema={"election": h.schema["election"], "bloc": pl.Utf8}),
            on=["election", "bloc"],
        ).with_columns(pl.lit(INCUMBENT).alias("bloc"))
    )
    h = pl.concat([h, inc.select(h.columns)])
    per_poll = h.group_by("poll_id", "election", "bloc").agg(
        pl.col("share").mean(), pl.col("hypothesis_id").n_unique().alias("n_hypotheses")
    )
    meta = polls.select("poll_id", "pollster", "fieldwork_end", "is_rolling")
    return _with_month(per_poll.join(meta, on="poll_id"), "fieldwork_end").sort(
        "election", "fieldwork_end", "poll_id", "bloc"
    )


def monthly_bloc_shares(poll_shares: pl.DataFrame) -> pl.DataFrame:
    """(election, month, bloc) : moyenne des instituts, chacun moyenné sur ses sondages."""
    by_pollster = poll_shares.group_by("election", "month", "bloc", "pollster").agg(
        pl.col("share").mean(), pl.len().alias("n")
    )
    return (
        by_pollster.group_by("election", "month", "bloc")
        .agg(
            pl.col("share").mean(),
            pl.col("n").sum().cast(pl.Int32).alias("n_polls"),
            pl.len().cast(pl.Int32).alias("n_pollsters"),
        )
        .sort("election", "bloc", "month")
    )


def _months(start: date, end: date) -> list[date]:
    out, i = [], month_index(start)
    while i <= month_index(end):
        out.append(date(i // 12, i % 12 + 1, 1))
        i += 1
    return out


def bloc_panel(monthly: pl.DataFrame, first_round: dict[int, date],
               start: date = PANEL_START) -> pl.DataFrame:
    """Grille mensuelle complète par cycle et bloc, avec `share` (vide sans sondage),
    `d_share` (variation entre deux mois consécutifs observés), `months_to_election`."""
    rows = []
    obs = monthly.filter(pl.col("month") >= start)
    for e, m0, m1 in (
        obs.group_by("election").agg(pl.col("month").min().alias("m0"),
                                     pl.col("month").max().alias("m1")).sort("election").iter_rows()
    ):
        vote = first_round[e]
        for m in _months(m0, min(m1, vote.replace(day=1))):
            for b in BLOCS:
                rows.append((e, m, b, month_index(vote) - month_index(m)))
    grid = pl.DataFrame(rows, schema={"election": pl.Int32, "month": pl.Date, "bloc": pl.Utf8,
                                      "months_to_election": pl.Int32}, orient="row")
    out = grid.join(obs.with_columns(pl.col("election").cast(pl.Int32)),
                    on=["election", "month", "bloc"], how="left")
    out = out.with_columns(pl.col("n_polls", "n_pollsters").fill_null(0)).sort(
        "election", "bloc", "month"
    )
    return out.with_columns(
        (pl.col("share") - pl.col("share").shift(1)).over("election", "bloc").alias("d_share")
    )


def coverage(panel: pl.DataFrame) -> pl.DataFrame:
    """Par cycle : mois de la grille, mois avec sondage, variations mensuelles disponibles,
    dont dans les 12 mois avant le vote."""
    p = panel.filter(pl.col("bloc") == MAIN_BLOCS[0])
    return (
        p.group_by("election")
        .agg(
            pl.col("month").min().alias("first_month"),
            pl.len().alias("grid_months"),
            (pl.col("n_polls") > 0).sum().alias("months_with_polls"),
            ((pl.col("n_polls") > 0) & (pl.col("months_to_election") <= 12)).sum()
            .alias("months_with_polls_last12"),
            pl.col("d_share").is_not_null().sum().alias("monthly_changes"),
            (pl.col("d_share").is_not_null() & (pl.col("months_to_election") <= 12)).sum()
            .alias("monthly_changes_last12"),
            pl.col("n_polls").sum().alias("polls"),
        )
        .sort("election")
    )


# ---------------------------------------------------------------------------- macro


def monthly_macro(macro: pl.DataFrame) -> pl.DataFrame:
    """Tableau large : `month` (calendrier complet) et une colonne par série."""
    parts = []
    for (sid, freq), g in macro.group_by("series_id", "freq"):
        if freq == "M":
            m = g.select(pl.col("date").alias("month"), "value")
        elif freq == "W":
            m = _with_month(g, "date").group_by("month").agg(pl.col("value").mean())
        elif freq == "Q":
            m = _quarterly_to_monthly(g)
        else:
            raise ValueError(f"fréquence inconnue : {freq}")
        parts.append(m.with_columns(pl.lit(sid).alias("series_id")))
    long = pl.concat(parts, how="vertical_relaxed")
    cal = pl.DataFrame({"month": _months(long["month"].min(), long["month"].max())})
    wide = long.pivot(on="series_id", index="month", values="value")
    return cal.join(wide, on="month", how="left").sort("month")


def _quarterly_to_monthly(g: pl.DataFrame) -> pl.DataFrame:
    """Valeur du trimestre au mois central, interpolation linéaire entre mois centraux."""
    g = g.sort("date")
    mids = [(month_index(d) + 1, v) for d, v in zip(g["date"], g["value"])]
    rows = []
    for (i0, v0), (i1, v1) in itertools.pairwise(mids):
        for i in range(i0, i1):
            rows.append((date(i // 12, i % 12 + 1, 1), v0 + (v1 - v0) * (i - i0) / (i1 - i0)))
    i, v = mids[-1]
    rows.append((date(i // 12, i % 12 + 1, 1), v))
    return pl.DataFrame(rows, schema={"month": pl.Date, "value": pl.Float64}, orient="row")


def indicators(wide: pl.DataFrame, changes: dict[str, Iterable[date]] | None = None
               ) -> pl.DataFrame:
    """Niveau `<id>` et variation `d_<id>` de chaque indicateur, sur le calendrier de `wide`.

    `changes` : par série, mois où la série change de nature (nouveau titulaire, rupture) ;
    la variation de ce mois-là est mise à vide. Demandeurs d'emploi : à partir de la rupture
    de janvier 2025 (inscriptions automatiques, nouvelles catégories, labellisation suspendue
    par la DARES), niveau et variation sont vides ; la série 2025-2026 contient encore des
    mouvements de reclassement (-5 % en avril 2025) qui ne sont pas de la conjoncture.
    """
    changes = {k: set(v) for k, v in (changes or {}).items()}
    w = wide.sort("month")
    exprs = [pl.col("month")]
    for sid, ind in INDICATORS.items():
        if sid not in w.columns:
            continue
        x = pl.col(sid)
        if ind.level == "yoy":
            level = 100 * (x / x.shift(12) - 1)
        elif ind.level == "thousands":
            level = x / 1000
        else:
            level = x
        if ind.diff == "logdiff":
            d = 100 * (x.log() - x.shift(1).log())
        else:
            d = level - level.shift(1)
        broken = pl.col("month").is_in(sorted(changes.get(sid, ())))
        d = pl.when(broken).then(None).otherwise(d)
        if sid == "jobseekers_cat_a":
            after = pl.col("month") >= JOBSEEKERS_BREAK
            level = pl.when(after).then(None).otherwise(level)
            d = pl.when(after).then(None).otherwise(d)
        exprs += [level.alias(sid), d.alias(f"d_{sid}")]
    return w.select(exprs)


def officeholder_changes(officeholders) -> dict[str, set[date]]:
    """Mois de prise de fonction d'un nouveau titulaire, par série de popularité."""
    role_series = {"president": "approval_president", "prime_minister": "approval_prime_minister"}
    out: dict[str, set[date]] = {}
    for h in officeholders:
        # Ifop mesure tantôt le mois de prise de fonction (mai 2007), tantôt le suivant (Valls,
        # nommé le 31 mars 2014) : on coupe les deux variations.
        m = month_index(h.start.replace(day=1))
        for i in (m, m + 1):
            out.setdefault(role_series[h.role], set()).add(date(i // 12, i % 12 + 1, 1))
    return out


def with_lags(ind: pl.DataFrame, cols: Iterable[str], lags: Iterable[int]) -> pl.DataFrame:
    """Ajoute `<col>_l<k>` = valeur k mois plus tôt (calendrier complet requis)."""
    ind = ind.sort("month")
    months = ind["month"].to_list()
    idx = [month_index(m) for m in months]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError("calendrier mensuel incomplet : les retards seraient faux")
    return ind.with_columns(
        [pl.col(c).shift(k).alias(f"{c}_l{k}") for c in cols for k in lags]
    )


def cumulative_change(ind: pl.DataFrame, col: str, horizon: int) -> pl.Series:
    """Somme glissante de `d_<col>` sur `horizon` mois (vide si un mois manque)."""
    return ind.sort("month")[f"d_{col}"].rolling_sum(horizon)
