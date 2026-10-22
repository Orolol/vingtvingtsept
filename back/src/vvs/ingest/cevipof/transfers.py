"""Matrices de reports pondérées à partir de cevipof_votes.parquet.

Une matrice croise, pour une présidentielle, le vote déclaré au premier tour (électorat) et le
vote au second tour, en % de l'électorat de premier tour : les deux finalistes, `blank`
(blanc ou nul) et `abstention`. Les répondants qui refusent ou ne savent pas à l'un des deux
tours (`refused`, `undecided`) sont exclus ; leur part est rapportée (`excluded_share`).

Sources d'une matrice :
- une enquête qui pose les deux votes (post-électorale : FES 2017, post-électorale 2012) ;
- un panel où le vote du premier tour et celui du second sont déclarés à deux vagues
  différentes de la même série, appariées par `respondent_id` (poids de la vague du second
  tour).

Intervalles : bootstrap des répondants (ou des unités primaires `cluster` pour les enquêtes
en face à face, dont l'échantillon est tiré par grappes), percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

EXCLUDED = ("refused", "undecided")
OUTCOMES_EXTRA = ("blank", "abstention")


@dataclass(frozen=True)
class PairSource:
    election: int
    label: str  # jeu ou « série vague_r1 -> vague_r2 »
    datasets: tuple[str, ...]
    data: pd.DataFrame  # r1, r2, weight, cluster


def pair_sources(votes: pl.DataFrame, election: int) -> list[PairSource]:
    """Couples (vote T1, vote T2) déclarés après le vote, par jeu ou par panel apparié."""
    v = votes.filter(pl.col("election") == election).to_pandas()
    out = []
    both = v[(v["vote_round1_kind"] == "recall") & (v["vote_round2_kind"] == "recall")]
    for key, g in both.groupby("dataset", sort=True):
        d = g[["vote_round1", "vote_round2", "weight", "cluster"]].rename(
            columns={"vote_round1": "r1", "vote_round2": "r2"}
        )
        out.append(PairSource(election, key, (key,), d.dropna(subset=["r1", "r2"])))
    # panels : T1 déclaré dans une vague, T2 dans une vague postérieure de la même série
    r1 = v[(v["vote_round1_kind"] == "recall") & v["vote_round1"].notna()]
    r2 = v[(v["vote_round2_kind"] == "recall") & v["vote_round2"].notna()]
    for series, g2 in r2.groupby("series"):
        g1 = r1[(r1["series"] == series) & ~r1["dataset"].isin(both["dataset"])]
        if g1.empty:
            continue
        for k2, w2 in g2.groupby("dataset"):
            if k2 in set(both["dataset"]):
                continue
            # dernière vague antérieure portant le vote T1
            before = g1[g1["wave_order"] < w2["wave_order"].iloc[0]]
            if before.empty:
                continue
            k1 = before.sort_values("wave_order")["dataset"].iloc[-1]
            m = w2.merge(
                g1[g1["dataset"] == k1][["respondent_id", "vote_round1"]],
                on="respondent_id",
                suffixes=("", "_t1"),
            )
            d = pd.DataFrame(
                {
                    "r1": m["vote_round1_t1"],
                    "r2": m["vote_round2"],
                    "weight": m["weight"],
                    "cluster": m["cluster"],
                }
            )
            out.append(PairSource(election, f"{k1} -> {k2}", (k1, k2), d))
    return out


@dataclass
class TransferMatrix:
    election: int
    source: str
    finalists: tuple[str, str]
    columns: tuple[str, ...]
    shares: pd.DataFrame  # index = électorat T1, colonnes = `columns`, en %
    n: pd.Series  # effectifs bruts par électorat (hors exclus)
    n_eff: pd.Series  # effectifs effectifs de Kish
    excluded_share: pd.Series  # part pondérée exclue (refus, ne sait pas) par électorat
    low: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    level: float = 0.9
    draws: np.ndarray | None = None  # tirages bootstrap : B x électorats x colonnes

    def interval(self, f) -> tuple[pd.Series, pd.Series, pd.Series]:
        """(estimation, borne basse, borne haute) d'une quantité dérivée `f(tableau)`, où
        `f` prend un DataFrame électorats x colonnes et renvoie une Series par électorat."""
        est = f(self.shares)
        if self.draws is None:
            return est, est * np.nan, est * np.nan
        vals = np.stack(
            [
                f(pd.DataFrame(x, index=self.shares.index, columns=list(self.columns))).to_numpy()
                for x in self.draws
            ]
        )
        a = (1 - self.level) / 2
        return (
            est,
            pd.Series(np.nanquantile(vals, a, axis=0), index=est.index),
            pd.Series(np.nanquantile(vals, 1 - a, axis=0), index=est.index),
        )


def _shares(d: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    t = d.pivot_table(index="r1", columns="r2", values="weight", aggfunc="sum", fill_value=0.0)
    t = t.reindex(columns=list(columns), fill_value=0.0)
    return 100 * t.div(t.sum(axis=1), axis=0)


def transfer_matrix(
    src: PairSource,
    finalists: tuple[str, str],
    bootstrap: int = 0,
    level: float = 0.9,
    seed: int = 20261022,
) -> TransferMatrix:
    d = src.data.copy()
    d = d[d["weight"].notna() & (d["weight"] > 0)]
    columns = (*finalists, *OUTCOMES_EXTRA)
    unexpected = set(d["r2"]) - set(columns) - set(EXCLUDED)
    if unexpected:
        raise ValueError(f"{src.label} : votes au second tour inattendus {sorted(unexpected)}")
    excl_mask = d["r1"].isin(EXCLUDED) | d["r2"].isin(EXCLUDED)
    tot = d.groupby("r1")["weight"].sum()
    excl = d[excl_mask].groupby("r1")["weight"].sum().reindex(tot.index, fill_value=0.0)
    kept = d[~excl_mask]
    shares = _shares(kept, columns)
    g = kept.groupby("r1")["weight"]
    n = g.size()
    n_eff = g.sum() ** 2 / g.apply(lambda w: (w**2).sum())
    m = TransferMatrix(
        src.election,
        src.label,
        finalists,
        columns,
        shares,
        n.reindex(shares.index),
        n_eff.reindex(shares.index),
        (100 * excl / tot).drop(index=list(EXCLUDED), errors="ignore"),
        level=level,
    )
    if bootstrap:
        rng = np.random.default_rng(seed)
        has_cluster = kept["cluster"].notna().all() and kept["cluster"].nunique() > 1
        units = kept["cluster"].to_numpy() if has_cluster else np.arange(len(kept))
        uniq = np.unique(units)
        index_of = {u: np.flatnonzero(units == u) for u in uniq}
        draws = []
        for _ in range(bootstrap):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([index_of[u] for u in pick])
            s = _shares(kept.iloc[rows], columns).reindex(index=shares.index, columns=list(columns))
            draws.append(s.to_numpy())
        arr = np.stack(draws)  # B x électorats x colonnes ; NaN si électorat absent du tirage
        a = (1 - level) / 2
        m.low = pd.DataFrame(
            np.nanquantile(arr, a, axis=0), index=shares.index, columns=list(columns)
        )
        m.high = pd.DataFrame(
            np.nanquantile(arr, 1 - a, axis=0), index=shares.index, columns=list(columns)
        )
        m.draws = arr
    return m


def group_rows(
    m: TransferMatrix,
    groups: dict[str, list[str]],
    src: PairSource,
    bootstrap: int = 0,
    seed: int = 20261022,
) -> TransferMatrix:
    """Même matrice avec des électorats regroupés (petits candidats d'un même bloc)."""
    inv = {c: g for g, members in groups.items() for c in members}
    d = src.data.copy()
    d["r1"] = d["r1"].map(lambda x: inv.get(x, x))
    return transfer_matrix(
        PairSource(src.election, src.label, src.datasets, d), m.finalists, bootstrap, m.level, seed
    )


def expressed_split(finalists: tuple[str, str]):
    """f(tableau) -> part du premier finaliste parmi les exprimés de chaque électorat, en %."""
    a, b = finalists
    return lambda t: 100 * t[a] / (t[a] + t[b])


def none_share(t: pd.DataFrame) -> pd.Series:
    """Part « ni l'un ni l'autre » (blancs, nuls, abstention) en % de l'électorat."""
    return t["blank"] + t["abstention"]
