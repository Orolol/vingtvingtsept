"""Modèle de second tour par reports de voix (docs/04-modele.md § 4).

Pour un duel `(i, j)` et des parts de premier tour `π` (base exprimés), chaque électorat `g`
se répartit entre `i`, `j` et « ni l'un ni l'autre » (abstention, blanc, nul) selon un vecteur
de reports `T_g = (t_gi, t_gj, t_g0)`. La part de `i` parmi les exprimés du second tour est

    part_i = Σ_g w_g t_gi / Σ_g w_g (t_gi + t_gj)

où `w_g` est la part de premier tour de l'électorat (somme des candidats du groupe) et, pour
les abstentionnistes, blancs et nuls du premier tour, le rapport (inscrits − exprimés) /
exprimés. Le dénominateur renormalise sur les exprimés du second tour.

Électorats (groupes). Les deux finalistes (`own:<id>`), puis un groupe par bloc pour les autres
candidats (`bloc:<bloc>`), puis `abstain`. Grouper par bloc, plutôt que par candidat, rend le
prior transposable d'une élection à l'autre et garde un problème de petite dimension.

Prior de `T`. Dirichlet par groupe, `T_g ~ Dir(κ_g m_g)`, plus deux décalages communs à tous
les groupes non finalistes : `u` sur le logit de la répartition entre `i` et `j` (érosion ou
renforcement du « front républicain » qui touche tous les électorats à la fois) et `v` sur le
logit de la part « ni l'un ni l'autre ». Sans eux, les groupes seraient indépendants et
l'incertitude sur `part_i` serait sous-estimée. La moyenne `m_g` vient :

1. finalistes : moyenne des lignes « électorat du finaliste » de toutes les sources
   (fidélité, κ = `KAPPA_OWN`) ;
2. autres groupes : les élections passées au même type de duel (même paire de blocs, ex.
   centre contre extrême droite en 2017 et 2022), électorats du même bloc pondérés par leur
   part de premier tour (`transfer_prior` de vvs.registry.transfers : répartition sur la
   base exprimés, toutes sources, part « ni l'un ni l'autre » sur la base électorat) ;
3. sinon, la matrice de proximité `PROXIMITY` (jugement explicite, concentration faible).
   Un analogue de plus de `ANALOG_MAX_AGE` ans (2002 pour un duel droite contre RN en 2027)
   est moyenné avec la matrice et traité comme inédit.

L'élection évaluée est toujours exclue des analogues (leave-one-election-out).

Sondages de duel. Un sondage `k` de la paire mesure `y_k = part_i + μ + h_{p_k} + e_k + ε_k` :
`μ` biais moyen (extrême droite surestimée au second tour, `ErrorParams.bias_far_right`),
`h_p ~ N(0, σ_h²)` effet maison de l'institut, `ε_k` erreur d'échantillonnage (effectif
nominal × facteur de design 0,6, la moyenne a priori du premier tour), et `e_k` l'erreur
d'horizon, commune à tous les sondages et croissant avec leur distance `a_k` au second tour :
Cov(e_k, e_l) = σ0² + σ1² min(a_k, a_l) / 30 (marche aléatoire vue depuis le vote). Les
effets aléatoires sont gaussiens et la vraisemblance ne dépend de `T` qu'à travers le
scalaire `part_i` : on les intègre exactement, ce qui ramène les sondages à une statistique
suffisante (moyenne des moindres carrés généralisés `ȳ`, variance `V`).

Inférence. Échantillonnage d'importance : tirages du prior (`T`, décalages, et un tirage des
parts de premier tour fournies), poids N(ȳ − μ ; part_i, V), rééchantillonnage. C'est exact
au bruit Monte-Carlo près (pas d'approximation variationnelle), sans chaîne de Markov ni
compilation, et `V ≥ σ0²` borne la concentration de la vraisemblance, donc les poids ne
dégénèrent pas (taille effective rapportée). NUTS n'apporterait rien ici : la postérieure ne
vit que dans la direction de `part_i`.

Potentiel électoral (option `use_potential`, désactivée par défaut ; étude
`back/studies/potential.py`). Un plafond c_i est une part des INSCRITS ; il est converti en
part des exprimés du second tour avec le rapport exprimés / inscrits du second tour de la
présidentielle précédente τ (`second_round_turnout`), pas avec la participation tirée par le
modèle, qui n'est pas calibrée (2017 : 78 % d'exprimés parmi les inscrits à la veille du
second tour, contre 66 % observés). Facteur de poids Φ((c_i / τ − part_i) /
(`CEILING_SD` / τ)) dans l'échantillonnage d'importance. Deux plafonds, normalisés sur
l'échelle Kantar (vvs.model.potential) et mesurés au plus `POTENTIAL_MAX_AGE` jours avant la
date de prévision :
- rejet : c_i = 100 − « ne voterait jamais pour i », pour tout finaliste ;
- potentiel : c_i = potentiel de i, pour un finaliste d'extrême droite face à un autre bloc
  seulement (en 2017 et 2022 son score de second tour est resté sous son potentiel, alors que
  les sondages de duel placent souvent l'adversaire au-dessus du sien).
Les reports par électorat ne sont pas touchés : le potentiel par électorat explique moins
bien les reports observés que la matrice de proximité (étude, partie b).

Horizon. `predict(..., horizon)` prend `horizon` = jours entre la date de prévision et le
premier tour (négatif après le premier tour : la veille du second tour vaut −(écart − 1)).
L'âge d'un sondage au second tour vaut horizon + écart entre les tours + (asof − fin de
terrain).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import BLOCS, Registry, default_registry
from vvs.registry.transfers import (
    TransferSource,
    default_transfers,
    transfer_prior,
)

log = logging.getLogger(__name__)

ABSTAIN = "abstain"

# --- prior des reports

# Matrice de proximité : attirance d'un électorat (ligne) pour un finaliste d'un bloc (colonne),
# sur une échelle de 0 à 10. Jugement explicite, utilisé seulement sans analogue historique.
# Répartition entre i et j : logistique de λ (a_i − a_j) ; part « ni l'un ni l'autre » :
# n_min + (n_max − n_min)(1 − max(a_i, a_j) / 10). Lecture : l'électorat de gauche radicale
# préfère nettement un finaliste de gauche (8) ou du centre (4) à un finaliste de droite (2,5)
# ou d'extrême droite (2) ; l'extrême droite est la plus repoussante pour la gauche (1) et le
# centre (2) ; la droite y est partagée (5) ; l'électorat d'extrême droite rejette d'abord la
# gauche et la gauche radicale. Ordres de grandeur recoupés avec les reports 2007-2022 (voir le
# rapport de backtest), sans ajustement fin : la concentration faible laisse les sondages
# décider.
PROXIMITY: dict[str, dict[str, float]] = {
    "far_left": {"far_left": 10, "left": 8, "centre": 4, "right": 2.5, "far_right": 2, "other": 4},
    "left": {"far_left": 7, "left": 10, "centre": 6.5, "right": 3.5, "far_right": 1, "other": 4},
    "centre": {"far_left": 2.5, "left": 6, "centre": 10, "right": 7, "far_right": 2, "other": 4},
    "right": {"far_left": 1, "left": 3, "centre": 7, "right": 10, "far_right": 5, "other": 4},
    "far_right": {"far_left": 2, "left": 1.5, "centre": 2.5, "right": 6, "far_right": 10, "other": 4},
    "other": {"far_left": 4, "left": 4, "centre": 4.5, "right": 5, "far_right": 5, "other": 10},
    ABSTAIN: {"far_left": 4, "left": 4.5, "centre": 5, "right": 5, "far_right": 4, "other": 4},
}
PROXIMITY_LAMBDA = 0.5
# (n_min, n_max) de la part « ni l'un ni l'autre » selon l'électorat
PROXIMITY_NONE: dict[str, tuple[float, float]] = {
    "other": (0.10, 0.55),
    ABSTAIN: (0.55, 0.80),
}
PROXIMITY_NONE_DEFAULT = (0.05, 0.55)

# Concentrations des Dirichlet (κ = somme des paramètres) : écart-type d'une part de 0,5 ≈
# 0,5 / sqrt(κ + 1). Jugements, pas des estimations : avec une ou deux élections analogues on
# ne peut pas estimer une variance inter-élections.
KAPPA_OWN = 60.0  # fidélité de l'électorat d'un finaliste (σ ≈ 0,03 autour de 0,94)
KAPPA_ANALOG = {1: 20.0, 2: 35.0}  # nombre d'élections analogues -> κ (au-delà : 35)
KAPPA_PROXIMITY = 6.0  # matrice seule : σ ≈ 0,19 sur une part de 0,5
# Décalages communs (logit) : analogue récent / cas inédit
SIGMA_U = {"analog": 0.30, "inedit": 0.60}
SIGMA_V = {"analog": 0.25, "inedit": 0.40}
# Un analogue plus ancien que cela est moyenné avec la matrice (cas traité comme inédit)
ANALOG_MAX_AGE = 20
MIN_MEAN = 0.005  # plancher des moyennes de Dirichlet (évite des paramètres nuls)

# Rapport (inscrits − exprimés) / exprimés au premier tour quand aucun scrutin antérieur n'est
# au registre (2002) : valeur approchée de 1995 (≈ 0,31), arrondie.
DEFAULT_ABSTAIN_RATIO = 0.30

# --- potentiel électoral (option use_potential)

CEILING_SD = 4.0  # points d'inscrits : incertitude du plafond (mesure et normalisation)
POTENTIAL_WINDOW = 60  # jours : mesures moyennées avant la dernière disponible
POTENTIAL_MAX_AGE = 240  # jours avant la date de prévision au-delà desquels on ignore

# --- sondages et erreur

DEFAULT_SAMPLE_SIZE = 1000
DESIGN_FACTOR = 0.6  # moyenne du prior Beta(12, 8) du facteur de design du premier tour
MAX_POLL_AGE = 730  # jours avant le second tour au-delà desquels un sondage de duel est ignoré
ERROR_HORIZONS = (7, 30, 90, 180)  # horizons de poll_errors (fenêtres de 30 jours)
BIAS_PRIOR_N = 1.0  # rétrécissement du biais : moyenne × n / (n + BIAS_PRIOR_N)
# Plancher de l'écart-type des effets maison : l'estimation par moments sur les sondages
# d'entre-deux-tours tombe à 0 (la dispersion entre instituts n'excède pas celle de
# l'échantillonnage avec le facteur de design 0,6) ; on garde un effet faible plutôt que nul.
HOUSE_SD_FLOOR = 0.5


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --- erreur des sondages de duel


@dataclass(frozen=True)
class ErrorParams:
    """Erreur des sondages de duel par rapport au résultat, en points d'exprimés.

    `sigma0` : erreur commune au jour du vote ; `sigma1` : croissance par racine de mois
    d'âge (variance σ0² + σ1² a / 30 pour un sondage terminé `a` jours avant le second tour) ;
    `bias_far_right` : surestimation moyenne du finaliste d'extrême droite face à un autre
    bloc ; `sigma_house` : écart-type des effets maison ; `design` : facteur de design.
    """

    sigma0: float = 2.5
    sigma1: float = 1.0
    bias_far_right: float = 0.0
    sigma_house: float = 1.0
    design: float = DESIGN_FACTOR
    calibrated_on: tuple[int, ...] = ()
    points: tuple[dict, ...] = ()

    def variance(self, age_days: np.ndarray | float) -> np.ndarray:
        a = np.maximum(np.asarray(age_days, dtype=float), 0.0)
        return self.sigma0**2 + self.sigma1**2 * a / 30.0

    def sd(self, age_days: np.ndarray | float) -> np.ndarray:
        return np.sqrt(self.variance(age_days))

    def covariance(
        self, age_days: np.ndarray, pollster: Iterable[str], sampling_sd: np.ndarray
    ) -> np.ndarray:
        """Covariance des erreurs de sondages (horizon commun + maison + échantillonnage)."""
        a = np.maximum(np.asarray(age_days, dtype=float), 0.0)
        p = np.asarray(list(pollster))
        cov = self.sigma0**2 + self.sigma1**2 * np.minimum.outer(a, a) / 30.0
        cov = cov + self.sigma_house**2 * (p[:, None] == p[None, :])
        return cov + np.diag(np.asarray(sampling_sd, dtype=float) ** 2)

    def sample(self, age_days: float, rng: np.random.Generator, n: int) -> np.ndarray:
        """Tirages de l'erreur d'horizon (biais exclu) pour une moyenne de sondages d'âge donné."""
        return rng.normal(0.0, float(self.sd(age_days)), size=n)


def sampling_sd(share: np.ndarray, sample_size: np.ndarray, design: float = DESIGN_FACTOR):
    """Écart-type d'échantillonnage d'une part de duel (points), effectif × facteur de design."""
    p = np.clip(np.asarray(share, dtype=float) / 100.0, 0.01, 0.99)
    n = np.where(np.asarray(sample_size, dtype=float) > 0, sample_size, DEFAULT_SAMPLE_SIZE)
    return 100.0 * np.sqrt(p * (1 - p) / (design * n))


def _orient_unit(election: int, registry: Registry) -> tuple[str, bool]:
    """Unité d'orientation d'un second tour : le finaliste d'extrême droite s'il est seul de son
    bloc dans le duel, sinon le premier finaliste par ordre alphabétique."""
    e = registry.elections[election]
    fin = sorted(r.candidate_id for r in e.round_(2).results)
    fr = [c for c in fin if registry.bloc_of(c, election) == "far_right"]
    if len(fr) == 1:
        return fr[0], True
    return fin[0], False


def _house_variance(
    elections: Iterable[int], registry: Registry, curated: Path | None
) -> float:
    """σ_h² : variance des moyennes par institut des sondages d'entre-deux-tours du duel réel,
    moins leur variance d'échantillonnage, poolée sur les élections (0 au plancher)."""
    from vvs.model.data import load_polls

    num, den = 0.0, 0
    for e in elections:
        el = registry.elections[e]
        d1, d2 = el.round_(1).date, el.round_(2).date
        unit, _ = _orient_unit(e, registry)
        fin = sorted(r.candidate_id for r in el.round_(2).results)
        df = load_polls(e, d2 - dt.timedelta(days=1), curated=curated, round_=2)
        df = df.filter(pl.col("hypothesis_id") == "+".join(fin))
        polls = _duel_frame(df, unit, next(c for c in fin if c != unit))
        polls = polls.filter(pl.col("fieldwork_end") > d1)
        if polls.height < 4:
            continue
        s2 = sampling_sd(polls["share"].to_numpy(), polls["sample_size"].to_numpy()) ** 2
        polls = polls.with_columns(pl.Series("s2", s2))
        by_p = polls.group_by("pollster").agg(
            pl.col("share").mean().alias("m"), (pl.col("s2").mean() / pl.len()).alias("v")
        )
        if by_p.height < 3:
            continue
        m = by_p["m"].to_numpy()
        excess = float(np.var(m, ddof=1) - by_p["v"].mean())
        num += excess * (by_p.height - 1)
        den += by_p.height - 1
    return max(num / den, 0.0) if den else 1.0


def calibrate_errors(
    exclude: Iterable[int] = (),
    curated: Path | None = None,
    registry: Registry | None = None,
    horizons: tuple[int, ...] = ERROR_HORIZONS,
) -> ErrorParams:
    """Calibre l'erreur des sondages de duel sur poll_errors (second tour), hors `exclude`.

    - Points : erreur de la moyenne des sondages du duel réel dans les 30 jours précédant
      l'horizon h (h ∈ 7, 30, 90, 180 jours avant le second tour, méthode `horizon`), orientée
      sur le finaliste d'extrême droite quand il y en a un. Âge moyen d'une fenêtre : h + 15.
    - Biais : moyenne des erreurs à 7 jours des duels avec l'extrême droite, rétrécie vers 0
      (× n / (n + `BIAS_PRIOR_N`)), appliquée à tous les horizons. Les résidus d'une élection
      retirent le biais estimé sans elle, pour que σ0 et σ1 incluent l'incertitude du biais.
    - (σ0, σ1) : maximum de vraisemblance sur grille des résidus (erreur − biais), conjointement
      par élection : Cov = σ0² + σ1² min(a, a') / 30 avec a = h + 15, plus le bruit
      d'échantillonnage de chaque moyenne.
    - σ_h : `_house_variance` sur les mêmes élections, au moins `HOUSE_SD_FLOOR`.
    """
    registry = registry or default_registry()
    curated = paths.CURATED if curated is None else Path(curated)
    excl = set(exclude)
    pe = pl.read_parquet(curated / "poll_errors.parquet").filter(
        (pl.col("round") == 2)
        & (pl.col("method") == "horizon")
        & (pl.col("level") == "candidate")
        & pl.col("horizon_days").is_in(list(horizons))
    )
    elections = sorted(
        e
        for e in pe["election"].unique().to_list()
        if e not in excl and registry.elections[e].round_(2).results
    )
    points: list[dict] = []
    for e in elections:
        unit, fr = _orient_unit(e, registry)
        sub = pe.filter((pl.col("election") == e) & (pl.col("unit") == unit))
        for h, err, n in sub.select("horizon_days", "error", "n_polls").iter_rows():
            points.append(
                {"election": e, "horizon": int(h), "error": float(err), "n_polls": int(n), "far_right": fr}
            )
    if not points:
        raise ValueError(f"aucune erreur de second tour hors {sorted(excl)}")
    h0 = min(horizons)
    fr_err = {p["election"]: p["error"] for p in points if p["far_right"] and p["horizon"] == h0}

    def shrunk_bias(skip: int | None = None) -> float:
        errs = [v for e, v in fr_err.items() if e != skip]
        return sum(errs) / (len(errs) + BIAS_PRIOR_N) if errs else 0.0

    bias = shrunk_bias()
    # Résidus honnêtes : le biais retiré d'une élection est estimé sans elle (sinon les
    # résidus des duels avec l'extrême droite seraient artificiellement petits).
    for p in points:
        p["residual"] = p["error"] - (shrunk_bias(p["election"]) if p["far_right"] else 0.0)
    # Vraisemblance jointe par élection : les erreurs d'une même élection à plusieurs horizons
    # partagent la composante commune (covariance de `ErrorParams.covariance`), plus le bruit
    # d'échantillonnage de chaque moyenne de fenêtre (effectif nominal 1000, design 0,6).
    by_e: dict[int, list[dict]] = {}
    for p in points:
        by_e.setdefault(p["election"], []).append(p)
    blocks = []
    for pts in by_e.values():
        a = np.array([p["horizon"] + 15 for p in pts], dtype=float)
        r = np.array([p["residual"] for p in pts])
        noise = np.array([sampling_sd(50.0, DEFAULT_SAMPLE_SIZE) ** 2 / p["n_polls"] for p in pts])
        blocks.append((np.minimum.outer(a, a) / 30.0, r, np.diag(noise)))
    best = (-np.inf, 2.5, 1.0)
    for s0 in np.arange(0.25, 8.01, 0.05):
        for s1 in np.arange(0.0, 4.01, 0.05):
            ll = 0.0
            for m, r, noise in blocks:
                cov = s0**2 + s1**2 * m + noise
                _, logdet = np.linalg.slogdet(cov)
                ll += -0.5 * (logdet + r @ np.linalg.solve(cov, r))
            if ll > best[0]:
                best = (ll, float(s0), float(s1))
    return ErrorParams(
        sigma0=round(best[1], 3),
        sigma1=round(best[2], 3),
        bias_far_right=round(bias, 3),
        sigma_house=round(max(math.sqrt(_house_variance(elections, registry, curated)), HOUSE_SD_FLOOR), 3),
        calibrated_on=tuple(elections),
        points=tuple(points),
    )


# --- prior des reports


@dataclass(frozen=True)
class GroupPrior:
    """Prior Dirichlet des reports d'un électorat vers (i, j, ni l'un ni l'autre)."""

    group: str
    mean: tuple[float, float, float]
    kappa: float
    shift: bool  # concerné par les décalages communs u et v
    source: str  # own | analog | analog_old | proximity
    analogs: tuple[int, ...] = ()
    note: str = ""

    def swapped(self) -> GroupPrior:
        a, b, c = self.mean
        return replace(self, mean=(b, a, c))


def proximity_mean(electorate: str, bloc_i: str, bloc_j: str) -> tuple[float, float, float]:
    """Moyenne des reports (i, j, ni l'un ni l'autre) donnée par la matrice de proximité."""
    row = PROXIMITY[electorate]
    ai, aj = row[bloc_i], row[bloc_j]
    s = 1.0 / (1.0 + math.exp(-PROXIMITY_LAMBDA * (ai - aj)))
    lo, hi = PROXIMITY_NONE.get(electorate, PROXIMITY_NONE_DEFAULT)
    n = lo + (hi - lo) * (1.0 - max(ai, aj) / 10.0)
    return ((1 - n) * s, (1 - n) * (1 - s), n)


def _finalists(registry: Registry, election: int) -> tuple[str, str] | None:
    e = registry.elections.get(election)
    if e is None or len(e.rounds) < 2 or not e.round_(2).results:
        return None
    a, b = (r.candidate_id for r in e.round_(2).results)
    return a, b


def _abstain_weights(registry: Registry, election: int) -> dict[str, float]:
    """Poids des électorats spéciaux du premier tour (abstention, blancs-nuls) d'une élection."""
    r1 = registry.elections[election].round_(1)
    if r1.registered is None or r1.voters is None or r1.expressed is None:
        return {ABSTAIN: 1.0, "blank_null": 0.0, "abstain_or_blank_null": 1.0}
    return {
        ABSTAIN: float(r1.registered - r1.voters),
        "blank_null": float(r1.voters - r1.expressed),
        "abstain_or_blank_null": float(r1.registered - r1.expressed),
    }


def _analog_vector(
    election: int,
    sources_from: list[tuple[str, float]],
    pair: tuple[str, str],
    transfers: tuple[TransferSource, ...],
    registry: Registry,
) -> tuple[float | None, float | None, float]:
    """(répartition vers pair[0] parmi les exprimés, part ni-l'un-ni-l'autre, poids couvert)
    pour un ensemble pondéré d'électorats d'une élection passée ; None si non documenté."""
    split_num = split_den = none_num = none_den = 0.0
    for cid, w in sources_from:
        if w <= 0:
            continue
        try:
            ex = transfer_prior(
                election, cid, pair, basis="expressed", transfers=transfers, registry=registry
            )
        except KeyError:
            continue
        split_num += w * ex.shares[pair[0]] / 100.0
        split_den += w
        try:
            el = transfer_prior(
                election, cid, pair, basis="electorate", transfers=transfers, registry=registry
            )
        except KeyError:
            continue
        none_num += w * el.shares["blank_or_abstain"] / 100.0
        none_den += w
    split = split_num / split_den if split_den else None
    none = none_num / none_den if none_den else None
    return split, none, split_den


def _group_members(registry: Registry, election: int, electorate: str, exclude: set[str]):
    """Électorats (id, poids) d'une élection passée qui composent un groupe."""
    if electorate == ABSTAIN:
        w = _abstain_weights(registry, election)
        return [(k, v) for k, v in w.items()]
    e = registry.elections[election]
    shares = {r.candidate_id: r.share for r in e.round_(1).results}
    return [
        (c.candidate_id, shares.get(c.candidate_id, 0.0))
        for c in e.candidates
        if c.bloc == electorate and c.candidate_id not in exclude
    ]


def analog_prior(
    electorate: str,
    bloc_i: str,
    bloc_j: str,
    target_election: int,
    exclude: Iterable[int] = (),
    transfers: tuple[TransferSource, ...] | None = None,
    registry: Registry | None = None,
) -> tuple[tuple[float, float, float], list[int], str] | None:
    """Moyenne des reports d'un groupe d'après les élections passées au même type de duel.

    Type de duel = paire non ordonnée de blocs des finalistes (blocs distincts). Pour chaque
    élection analogue, les électorats du bloc `electorate` (hors finalistes) sont pondérés par
    leur part de premier tour ; la répartition et la part « ni l'un ni l'autre » sont ensuite
    moyennées entre élections à poids égal. Une part « ni l'un ni l'autre » introuvable est
    prise dans la matrice de proximité (signalé dans la note). None si aucun analogue.
    """
    registry = registry or default_registry()
    transfers = default_transfers() if transfers is None else transfers
    if bloc_i == bloc_j:
        return None
    excl = set(exclude) | {target_election}
    splits, nones, used = [], [], []
    for e in sorted({s.election for s in transfers} - excl):
        fin = _finalists(registry, e)
        if fin is None:
            continue
        b0, b1 = registry.bloc_of(fin[0], e), registry.bloc_of(fin[1], e)
        if {b0, b1} != {bloc_i, bloc_j}:
            continue
        pair = fin if b0 == bloc_i else (fin[1], fin[0])
        members = _group_members(registry, e, electorate, set(fin))
        split, none, _ = _analog_vector(e, members, pair, transfers, registry)
        if split is None:
            continue
        splits.append(split)
        if none is not None:
            nones.append(none)
        used.append(e)
    if not used:
        return None
    s = float(np.mean(splits))
    note = ""
    if nones:
        n = float(np.mean(nones))
    else:
        n = proximity_mean(electorate, bloc_i, bloc_j)[2]
        note = "part ni-l'un-ni-l'autre prise dans la matrice (aucune source base électorat)"
    return ((1 - n) * s, (1 - n) * (1 - s), n), used, note


def own_prior(
    target_election: int,
    exclude: Iterable[int] = (),
    transfers: tuple[TransferSource, ...] | None = None,
    registry: Registry | None = None,
) -> tuple[float, float, float]:
    """(vers soi, vers l'adversaire, ni l'un ni l'autre) de l'électorat d'un finaliste :
    moyenne par élection des lignes « finaliste → finaliste », puis entre élections."""
    registry = registry or default_registry()
    transfers = default_transfers() if transfers is None else transfers
    excl = set(exclude) | {target_election}
    per_election = []
    for e in sorted({s.election for s in transfers} - excl):
        fin = _finalists(registry, e)
        if fin is None:
            continue
        vecs = []
        for me, other in (fin, fin[::-1]):
            split, none, _ = _analog_vector(e, [(me, 1.0)], (me, other), transfers, registry)
            if split is None or none is None:
                continue
            vecs.append(((1 - none) * split, (1 - none) * (1 - split), none))
        if vecs:
            per_election.append(np.mean(vecs, axis=0))
    if not per_election:
        return (0.94, 0.02, 0.04)
    m = np.mean(per_election, axis=0)
    return tuple(float(x) for x in m / m.sum())  # type: ignore[return-value]


# --- un duel ajusté


@dataclass
class DuelPolls:
    """Sondages de duel retenus, part de `pair[0]` en points d'exprimés."""

    poll_id: list[str]
    pollster: list[str]
    fieldwork_end: list[dt.date]
    share: np.ndarray
    sample_size: np.ndarray

    @property
    def n(self) -> int:
        return len(self.poll_id)

    def swapped(self) -> DuelPolls:
        return replace(self, share=100.0 - self.share)

    @classmethod
    def empty(cls) -> DuelPolls:
        return cls([], [], [], np.zeros(0), np.zeros(0))


@dataclass
class PairFit:
    """Tout ce qu'il faut pour prédire un duel : priors des groupes, sondages, erreur."""

    election: int
    asof: dt.date
    pair: tuple[str, str]
    blocs: tuple[str, str]
    groups: dict[str, GroupPrior]
    sigma_u: float
    sigma_v: float
    polls: DuelPolls
    errors: ErrorParams
    gap_days: int  # jours entre les deux tours
    abstain_weight: float
    bloc_of: Callable[[str], str]
    inedit: bool = False
    notes: list[str] = field(default_factory=list)
    use_polls: bool = True
    use_bias: bool = True
    # plafonds de la part des inscrits (i, j), en points ; None : pas de plafond
    ceilings: tuple[float | None, float | None] = (None, None)
    ceiling_tau: float = 1.0  # exprimés / inscrits du second tour précédent (conversion)

    @property
    def bias_sign(self) -> int:
        """+1 si les sondages surestiment i (i d'extrême droite face à un autre bloc), −1 si j."""
        fi, fj = (b == "far_right" for b in self.blocs)
        return int(fi) - int(fj)

    def swapped(self) -> PairFit:
        return replace(
            self,
            pair=self.pair[::-1],
            blocs=self.blocs[::-1],
            groups={k: g.swapped() for k, g in self.groups.items()},
            polls=self.polls.swapped(),
            notes=list(self.notes),
            ceilings=self.ceilings[::-1],
        )

    def group_of(self, candidate: str) -> str | None:
        if candidate == ABSTAIN:
            return ABSTAIN
        if candidate in self.pair:
            return f"own:{candidate}"
        try:
            return f"bloc:{self.bloc_of(candidate)}"
        except KeyError:
            return None

    def poll_summary(self, horizon: float) -> tuple[float, float, float] | None:
        """(ȳ moindres carrés généralisés, V, biais μ) des sondages à l'horizon donné ; None
        sans sondage exploitable."""
        if not self.use_polls or self.polls.n == 0:
            return None
        age = np.array(
            [horizon + self.gap_days + (self.asof - d).days for d in self.polls.fieldwork_end],
            dtype=float,
        )
        keep = age <= MAX_POLL_AGE
        if not keep.any():
            return None
        y = self.polls.share[keep]
        s = sampling_sd(y, self.polls.sample_size[keep], self.errors.design)
        cov = self.errors.covariance(age[keep], np.asarray(self.polls.pollster)[keep], s)
        ones = np.ones(len(y))
        sol = np.linalg.solve(cov, np.column_stack([ones, y]))
        prec = float(ones @ sol[:, 0])
        ybar = float(ones @ sol[:, 1]) / prec
        mu = self.errors.bias_far_right * self.bias_sign if self.use_bias else 0.0
        return ybar, 1.0 / prec, mu


@dataclass
class SecondRoundDraws:
    """Tirages a posteriori d'un duel (rééchantillonnés, poids égaux)."""

    pair: tuple[str, str]
    share: np.ndarray  # (n,) part de i en points d'exprimés
    groups: list[str]
    transfers: np.ndarray  # (n, G, 3) reports (i, j, ni) en fraction de l'électorat
    weights: np.ndarray  # (n, G) poids de premier tour des groupes
    prior_share: np.ndarray  # (M,) part de i sous le prior (tirages avant pondération)
    ess: float
    polls: tuple[float, float, float] | None
    horizon: float


def duel_share(weights: np.ndarray, transfers: np.ndarray) -> np.ndarray:
    """part_i (points) = Σ w t_i / Σ w (t_i + t_j) ; weights (..., G), transfers (..., G, 3)."""
    vi = (weights * transfers[..., 0]).sum(-1)
    vj = (weights * transfers[..., 1]).sum(-1)
    return 100.0 * vi / (vi + vj)


def draw_transfers(
    groups: list[GroupPrior], sigma_u: float, sigma_v: float, rng: np.random.Generator, m: int
) -> np.ndarray:
    """Tirages (m, G, 3) du prior : Dirichlet par groupe puis décalages communs u, v."""
    mean = np.clip(np.array([g.mean for g in groups]), MIN_MEAN, None)
    mean = mean / mean.sum(-1, keepdims=True)
    alpha = mean * np.array([g.kappa for g in groups])[:, None]
    x = rng.gamma(alpha[None], size=(m, len(groups), 3))
    x = np.maximum(x, 1e-300)
    x = x / x.sum(-1, keepdims=True)
    shift = np.array([g.shift for g in groups], dtype=float)[None, :]
    split = x[..., 0] / np.maximum(x[..., 0] + x[..., 1], 1e-300)
    none = x[..., 2]
    u = rng.normal(0.0, sigma_u, size=(m, 1)) * shift
    v = rng.normal(0.0, sigma_v, size=(m, 1)) * shift
    split = _expit(_logit(split) + u)
    none = _expit(_logit(none) + v)
    return np.stack([(1 - none) * split, (1 - none) * (1 - split), none], axis=-1)


def _weights_matrix(
    fit: PairFit, first_round: Mapping[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    """Groupes présents et poids (S, G) à partir de tirages de premier tour par candidat."""
    cands = [c for c in first_round if c != ABSTAIN]
    for c in fit.pair:
        if c not in cands:
            raise KeyError(f"finaliste {c} absent des parts de premier tour")
    vals = np.column_stack([np.atleast_1d(np.asarray(first_round[c], dtype=float)) for c in cands])
    vals = vals / vals.sum(-1, keepdims=True)
    groups: list[str] = []
    cols: list[np.ndarray] = []
    dropped = []
    for k, c in enumerate(cands):
        g = fit.group_of(c)
        if g is None:
            dropped.append(c)
            continue
        if g in groups:
            cols[groups.index(g)] = cols[groups.index(g)] + vals[:, k]
        else:
            groups.append(g)
            cols.append(vals[:, k].copy())
    if dropped:
        log.warning("candidats sans bloc ignorés : %s", dropped)
    w = np.column_stack(cols)
    w = w / w.sum(-1, keepdims=True)
    abst = first_round.get(ABSTAIN)
    a = np.full(len(w), fit.abstain_weight) if abst is None else np.asarray(abst, dtype=float)
    groups.append(ABSTAIN)
    w = np.column_stack([w, np.broadcast_to(a, (len(w),))])
    return groups, w


def sample_pair(
    fit: PairFit,
    first_round: Mapping[str, np.ndarray],
    horizon: float,
    rng: np.random.Generator,
    n: int = 4000,
    oversample: int = 8,
) -> SecondRoundDraws:
    """Échantillonnage d'importance du duel (voir le docstring du module)."""
    groups, w_all = _weights_matrix(fit, first_round)
    priors = [fit.groups[g] for g in groups]
    summary = fit.poll_summary(horizon)
    m = max(n * oversample, 1000)
    while True:
        idx = rng.integers(0, len(w_all), size=m)
        w = w_all[idx]
        t = draw_transfers(priors, fit.sigma_u, fit.sigma_v, rng, m)
        g = duel_share(w, t)
        if summary is None:
            lw = np.zeros(m)
        else:
            ybar, var, mu = summary
            lw = -0.5 * (ybar - mu - g) ** 2 / var
        lw = lw + ceiling_log_weight(fit.ceilings, g, fit.ceiling_tau)
        p = np.exp(lw - lw.max())
        p /= p.sum()
        ess = float(1.0 / np.sum(p**2))
        if ess >= n / 4 or m >= 64 * n:
            break
        m *= 4
    if ess < n / 4:
        log.warning("%s : taille effective faible (%.0f pour %d tirages)", fit.pair, ess, n)
    pick = rng.choice(m, size=n, replace=True, p=p)
    return SecondRoundDraws(
        pair=fit.pair,
        share=g[pick],
        groups=groups,
        transfers=t[pick],
        weights=w[pick],
        prior_share=g,
        ess=ess,
        polls=summary,
        horizon=horizon,
    )


def registered_shares(weights: np.ndarray, transfers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(v_i, v_j) : voix de i et de j en % des inscrits (Σ des poids = inscrits / exprimés du
    premier tour, abstentionnistes compris)."""
    tot = weights.sum(-1)
    vi = (weights * transfers[..., 0]).sum(-1) / tot
    vj = (weights * transfers[..., 1]).sum(-1) / tot
    return 100.0 * vi, 100.0 * vj


def ceiling_log_weight(
    ceilings: tuple[float | None, float | None], share_i: np.ndarray, tau: float
) -> np.ndarray | float:
    """Log du facteur des plafonds doux sur la part de i parmi les exprimés (points) : plafond
    des inscrits c (en %) converti en c / τ, écart-type CEILING_SD / τ ; 0 sans plafond."""
    if ceilings[0] is None and ceilings[1] is None:
        return 0.0
    share_i = np.asarray(share_i, dtype=float)
    out = np.zeros(share_i.shape)
    for c, s in zip(ceilings, (share_i, 100.0 - share_i)):
        if c is not None:
            out += _log_ndtr((c / tau - s) / (CEILING_SD / tau))
    return out


def second_round_turnout(registry: Registry, election: int) -> float:
    """Exprimés / inscrits du second tour de la dernière présidentielle antérieure au
    registre (conversion des plafonds) ; 0,70 à défaut."""
    prev = [
        y for y, x in registry.elections.items()
        if y < election and len(x.rounds) > 1 and x.round_(2).registered and x.round_(2).expressed
    ]
    if not prev:
        return 0.70
    r2 = registry.elections[max(prev)].round_(2)
    return r2.expressed / r2.registered


_ERFC = np.vectorize(math.erfc, otypes=[float])


def _log_ndtr(x: np.ndarray) -> np.ndarray:
    """log Φ(x), stable dans la queue gauche (développement asymptotique sous −5)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x < -5.0
    out[~lo] = np.log(0.5 * _ERFC(-x[~lo] / math.sqrt(2.0)))
    xl = x[lo]
    out[lo] = -0.5 * xl**2 - np.log(-xl) - 0.5 * math.log(2 * math.pi) + np.log1p(-1.0 / xl**2 + 3.0 / xl**4)
    return out


def potential_ceilings(
    election: int,
    asof: dt.date,
    pair: tuple[str, str],
    blocs: tuple[str, str],
    table: pl.DataFrame,
) -> tuple[tuple[float | None, float | None], list[str]]:
    """Plafonds (i, j) en % des inscrits d'après la table normalisée des potentiels
    (`vvs.model.potential.normalize`) ; notes lisibles."""
    from vvs.model.potential import summary_at

    out: list[float | None] = []
    notes = []
    for k, cid in enumerate(pair):
        s = summary_at(
            table, election, cid, asof, window_days=POTENTIAL_WINDOW, max_age=POTENTIAL_MAX_AGE
        )
        cands = []
        if s is not None and s.ceiling is not None:
            cands.append(s.ceiling)
        rn_vs_other = blocs[k] == "far_right" and blocs[1 - k] != "far_right"
        if s is not None and s.potential is not None and rn_vs_other:
            cands.append(s.potential)
        c = min(cands) if cands else None
        out.append(c)
        if c is not None:
            notes.append(
                f"plafond {cid} : {c:.1f} % des inscrits (potentiel {s.potential or float('nan'):.1f}, "
                f"rejet {s.reject or float('nan'):.1f}, {s.n} mesure(s) jusqu'au {s.last_date})"
            )
    return (out[0], out[1]), notes


# --- données de sondages


def _duel_frame(df: pl.DataFrame, i: str, j: str) -> pl.DataFrame:
    """Une ligne par sondage : part de i (points, renormalisée sur i + j)."""
    wide = (
        df.filter(pl.col("candidate_id").is_in([i, j]))
        .pivot(
            on="candidate_id",
            index=["poll_id", "pollster", "fieldwork_end", "sample_size"],
            values="share",
            aggregate_function="first",
        )
    )
    if wide.is_empty() or i not in wide.columns or j not in wide.columns:
        return pl.DataFrame(
            schema={"poll_id": pl.Utf8, "pollster": pl.Utf8, "fieldwork_end": pl.Date,
                    "sample_size": pl.Int64, "share": pl.Float64}
        )
    return (
        wide.drop_nulls([i, j])
        .with_columns((100 * pl.col(i) / (pl.col(i) + pl.col(j))).alias("share"))
        .select("poll_id", "pollster", "fieldwork_end", pl.col("sample_size").cast(pl.Int64), "share")
        .sort("fieldwork_end", "poll_id")
    )


def load_duel_polls(
    election: int, asof: dt.date, pair: tuple[str, str], curated: Path | None = None
) -> DuelPolls:
    """Sondages de second tour testant exactement la paire, terminés au plus tard à `asof`."""
    from vvs.model.data import load_polls

    df = load_polls(election, asof, curated=curated, round_=2)
    df = df.filter(pl.col("hypothesis_id") == "+".join(sorted(pair)))
    w = _duel_frame(df, *pair)
    if w.is_empty():
        return DuelPolls.empty()
    return DuelPolls(
        poll_id=w["poll_id"].to_list(),
        pollster=w["pollster"].to_list(),
        fieldwork_end=w["fieldwork_end"].to_list(),
        share=w["share"].to_numpy().astype(float),
        sample_size=w["sample_size"].fill_null(DEFAULT_SAMPLE_SIZE).to_numpy().astype(float),
    )


def abstain_ratio(registry: Registry, election: int, asof: dt.date) -> tuple[float, str]:
    """(inscrits − exprimés) / exprimés au premier tour : celui de l'élection si le premier tour
    a eu lieu à `asof`, sinon celui de la présidentielle précédente au registre."""
    e = registry.elections.get(election)
    if e is not None and e.round_(1).results and asof >= e.round_(1).date:
        r1 = e.round_(1)
        return (r1.registered - r1.expressed) / r1.expressed, f"premier tour {election}"
    prev = [
        y for y, x in registry.elections.items()
        if y < election and x.round_(1).results and x.round_(1).registered
    ]
    if not prev:
        return DEFAULT_ABSTAIN_RATIO, "valeur par défaut (1995 approchée)"
    r1 = registry.elections[max(prev)].round_(1)
    return (r1.registered - r1.expressed) / r1.expressed, f"premier tour {max(prev)}"


# --- premier tour simplifié (en attendant le couplage de la phase 4)


def first_round_mean(
    election: int,
    asof: dt.date,
    pair: tuple[str, str],
    days: int = 14,
    curated: Path | None = None,
) -> tuple[dict[str, float], tuple[dt.date, dt.date], int]:
    """Moyenne simple des parts de premier tour (fractions) des `days` jours précédant `asof`,
    sur la configuration de candidats la plus fréquente parmi les hypothèses contenant les
    deux finalistes (ex æquo : la plus récente).

    Une seule configuration : moyenner des hypothèses alternatives (Glucksmann ou Hollande,
    Philippe ou Attal) compterait deux fois le même électorat. Si aucune hypothèse de la
    fenêtre ne contient la paire, la fenêtre de `days` jours se termine au dernier sondage qui
    la contient (signalé par la fenêtre rendue). Rend (parts, (début exclu, fin incluse),
    nombre d'hypothèses moyennées).
    """
    from vvs.model.data import load_polls

    df = load_polls(election, asof, curated=curated, round_=1)
    hyp = (
        df.group_by("poll_id", "hypothesis_id")
        .agg(pl.col("candidate_id").sort(), pl.col("fieldwork_end").first())
        .filter(
            pl.col("candidate_id").list.contains(pair[0])
            & pl.col("candidate_id").list.contains(pair[1])
        )
    )
    if hyp.is_empty():
        raise KeyError(f"{election} : aucune hypothèse de premier tour avec {pair} avant {asof}")
    end = asof
    if hyp.filter(pl.col("fieldwork_end") > asof - dt.timedelta(days=days)).is_empty():
        end = hyp["fieldwork_end"].max()
    start = end - dt.timedelta(days=days)
    win = hyp.filter((pl.col("fieldwork_end") > start) & (pl.col("fieldwork_end") <= end))
    win = win.with_columns(pl.col("candidate_id").list.join("+").alias("config"))
    modal = (
        win.group_by("config")
        .agg(pl.len(), pl.col("fieldwork_end").max())
        .sort(["len", "fieldwork_end", "config"], descending=True)["config"][0]
    )
    keep = win.filter(pl.col("config") == modal)
    sub = df.join(keep.select("poll_id", "hypothesis_id"), on=["poll_id", "hypothesis_id"])
    sub = sub.with_columns(
        (pl.col("share") / pl.col("share").sum().over("poll_id", "hypothesis_id")).alias("p")
    )
    mean = sub.group_by("candidate_id").agg(pl.col("p").mean())
    tot = float(mean["p"].sum())
    shares = {c: float(p) / tot for c, p in mean.iter_rows()}
    return shares, (start, end), keep.height


def first_round_concentration(
    horizon: int, exclude: Iterable[int] = (), curated: Path | None = None, min_share: float = 1.0
) -> float:
    """Concentration κ d'une Dirichlet centrée sur la moyenne des sondages de premier tour,
    calibrée par moments sur poll_errors (premier tour, méthode `horizon`, horizon le plus
    proche, candidats d'au moins `min_share` %, hors `exclude`) : κ + 1 = Σ r(1 − r) / Σ e²
    (r résultat, e erreur, en fractions). Remplaçant provisoire du modèle d'erreur du premier
    tour, sans corrélation de bloc."""
    curated = paths.CURATED if curated is None else Path(curated)
    pe = pl.read_parquet(curated / "poll_errors.parquet").filter(
        (pl.col("round") == 1)
        & (pl.col("method") == "horizon")
        & (pl.col("level") == "candidate")
        & (pl.col("result_share") >= min_share)
        & ~pl.col("election").is_in(list(exclude))
    )
    hs = sorted(pe["horizon_days"].unique().to_list())
    h = min(hs, key=lambda x: abs(x - horizon))
    x = pe.filter(pl.col("horizon_days") == h)
    r = x["result_share"].to_numpy() / 100
    e = x["error"].to_numpy() / 100
    return float(np.sum(r * (1 - r)) / np.sum(e**2) - 1)


def noisy_first_round(
    mean: Mapping[str, float], kappa: float, rng: np.random.Generator, n: int
) -> dict[str, np.ndarray]:
    """Tirages de parts de premier tour : Dirichlet(κ π̂), de moyenne π̂. Bruit grossier sans
    corrélation de bloc, en attendant les tirages du modèle de premier tour (phase 4)."""
    cands = list(mean)
    alpha = np.maximum(kappa * np.array([mean[c] for c in cands]), 1e-3)
    p = rng.dirichlet(alpha, size=n)
    return {c: p[:, k] for k, c in enumerate(cands)}


# --- modèle


class SecondRoundModel:
    """Interface : `fit(election, asof, pairs)`, `predict(pair, first_round_draws, horizon, rng,
    n)`, `sample(...)` (tirages complets) et `summary(draws)`.

    Les paramètres calibrés sur l'histoire (analogues de reports, erreur des sondages) excluent
    toujours l'élection ajustée. `use_polls=False` donne le prior seul (reports historiques
    sans sondages de duel) ; `use_bias=False` retire la correction du biais d'extrême droite.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        transfers: tuple[TransferSource, ...] | None = None,
        curated: Path | None = None,
        errors: ErrorParams | None = None,
        exclude: Iterable[int] = (),
        use_polls: bool = True,
        use_bias: bool = True,
        use_potential: bool = False,
        potential: pl.DataFrame | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.use_potential = use_potential
        # table normalisée (vvs.model.potential.normalize) ; chargée au premier ajustement
        self.potential = potential
        self.transfers = default_transfers() if transfers is None else tuple(transfers)
        self.curated = curated
        self.errors = errors
        self.exclude = set(exclude)
        self.use_polls = use_polls
        self.use_bias = use_bias
        self.fits: dict[tuple[str, str], PairFit] = {}

    # ajustement

    def fit(self, election: int, asof: dt.date | str, pairs: Iterable[tuple[str, str]]):
        if isinstance(asof, str):
            asof = dt.date.fromisoformat(asof)
        excl = self.exclude | {election}
        errors = self.errors or calibrate_errors(excl, self.curated, self.registry)
        e = self.registry.elections[election]
        gap = (e.round_(2).date - e.round_(1).date).days
        abst, _ = abstain_ratio(self.registry, election, asof)
        own = own_prior(election, excl, self.transfers, self.registry)
        for pair in pairs:
            pair = tuple(pair)  # type: ignore[assignment]
            blocs = tuple(self.registry.bloc_of(c, election) for c in pair)
            groups, inedit, notes = build_groups(
                pair, blocs, election, excl, own, self.transfers, self.registry
            )
            polls = (
                load_duel_polls(election, asof, pair, self.curated)
                if self.use_polls
                else DuelPolls.empty()
            )
            kind = "inedit" if inedit else "analog"
            ceilings: tuple[float | None, float | None] = (None, None)
            if self.use_potential:
                ceilings, pnotes = potential_ceilings(
                    election, asof, pair, blocs, self.potential_table()
                )
                notes = notes + pnotes
            tau = second_round_turnout(self.registry, election)
            self.fits[pair] = PairFit(
                election=election,
                asof=asof,
                pair=pair,
                blocs=blocs,
                groups=groups,
                sigma_u=SIGMA_U[kind],
                sigma_v=SIGMA_V[kind],
                polls=polls,
                errors=errors,
                gap_days=gap,
                abstain_weight=abst,
                bloc_of=lambda c, e=election: self.registry.bloc_of(c, e),
                inedit=inedit,
                notes=notes,
                use_polls=self.use_polls,
                use_bias=self.use_bias,
                ceilings=ceilings,
                ceiling_tau=tau,
            )
        return self

    def potential_table(self) -> pl.DataFrame:
        """Table des potentiels normalisée (mesures de vote seulement), mise en cache."""
        if self.potential is None:
            from vvs.model import potential as pot

            raw = pot.load(self.curated).filter(pl.col("scale") != "attitude")
            self.potential = pot.normalize(raw)
        return self.potential

    def pair_fit(self, pair: tuple[str, str]) -> PairFit:
        pair = tuple(pair)  # type: ignore[assignment]
        if pair in self.fits:
            return self.fits[pair]
        if pair[::-1] in self.fits:
            return self.fits[pair[::-1]].swapped()
        raise KeyError(f"paire non ajustée : {pair}")

    # prédiction

    def sample(
        self,
        pair: tuple[str, str],
        first_round_draws: Mapping[str, np.ndarray],
        horizon: float,
        rng: np.random.Generator,
        n: int = 4000,
    ) -> SecondRoundDraws:
        return sample_pair(self.pair_fit(pair), first_round_draws, horizon, rng, n)

    def predict(
        self,
        pair: tuple[str, str],
        first_round_draws: Mapping[str, np.ndarray],
        horizon: float,
        rng: np.random.Generator,
        n: int = 4000,
    ) -> np.ndarray:
        """Tirages (n,) de la part de `pair[0]` au second tour, en points d'exprimés."""
        return self.sample(pair, first_round_draws, horizon, rng, n).share

    def summary(self, draws: SecondRoundDraws) -> dict:
        return summarize(self.pair_fit(draws.pair), draws)


def build_groups(
    pair: tuple[str, str],
    blocs: tuple[str, str],
    election: int,
    exclude: set[int],
    own: tuple[float, float, float],
    transfers: tuple[TransferSource, ...],
    registry: Registry,
) -> tuple[dict[str, GroupPrior], bool, list[str]]:
    """Priors de tous les groupes possibles d'un duel ; (groupes, inédit, notes)."""
    bi, bj = blocs
    groups: dict[str, GroupPrior] = {
        f"own:{pair[0]}": GroupPrior(f"own:{pair[0]}", own, KAPPA_OWN, False, "own"),
        f"own:{pair[1]}": GroupPrior(
            f"own:{pair[1]}", (own[1], own[0], own[2]), KAPPA_OWN, False, "own"
        ),
    }
    notes: list[str] = []
    analog_years: set[int] = set()
    any_recent = False
    for electorate in [*BLOCS, ABSTAIN]:
        key = ABSTAIN if electorate == ABSTAIN else f"bloc:{electorate}"
        prox = proximity_mean(electorate, bi, bj)
        found = analog_prior(electorate, bi, bj, election, exclude, transfers, registry)
        if found is None:
            groups[key] = GroupPrior(key, prox, KAPPA_PROXIMITY, True, "proximity")
            continue
        mean, used, note = found
        analog_years |= set(used)
        if election - max(used) > ANALOG_MAX_AGE:
            blend = tuple(0.5 * a + 0.5 * b for a, b in zip(mean, prox))
            groups[key] = GroupPrior(
                key, blend, KAPPA_PROXIMITY * 1.5, True, "analog_old", tuple(used),
                "analogue ancien moyenné avec la matrice",
            )
            continue
        any_recent = True
        kappa = KAPPA_ANALOG.get(len(used), KAPPA_ANALOG[2])
        groups[key] = GroupPrior(key, mean, kappa, True, "analog", tuple(used), note)
    inedit = not any_recent
    if not analog_years:
        notes.append(
            f"duel inédit ({bi} contre {bj}) : aucun second tour comparable dans les reports, "
            "prior de la matrice de proximité, incertitude large"
        )
    elif inedit:
        notes.append(
            f"duel {bi} contre {bj} : seul(s) analogue(s) {sorted(analog_years)}, plus de "
            f"{ANALOG_MAX_AGE} ans ; prior moyenné avec la matrice, incertitude large"
        )
    return groups, inedit, notes


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def summarize(fit: PairFit, draws: SecondRoundDraws) -> dict:
    """Résumé d'un duel : quantiles, P(i gagne), sondages, reports estimés par électorat."""
    s = draws.share
    out = {
        "election": fit.election,
        "asof": fit.asof.isoformat(),
        "candidate_i": fit.pair[0],
        "candidate_j": fit.pair[1],
        "bloc_i": fit.blocs[0],
        "bloc_j": fit.blocs[1],
        "horizon": draws.horizon,
        "median": _q(s, 0.5),
        "mean": float(s.mean()),
        "q05": _q(s, 0.05),
        "q10": _q(s, 0.10),
        "q25": _q(s, 0.25),
        "q75": _q(s, 0.75),
        "q90": _q(s, 0.90),
        "q95": _q(s, 0.95),
        "p_i_wins": float(np.mean(s > 50.0)),
        "prior_median": _q(draws.prior_share, 0.5),
        "prior_q05": _q(draws.prior_share, 0.05),
        "prior_q95": _q(draws.prior_share, 0.95),
        "n_polls": fit.polls.n if fit.use_polls else 0,
        "poll_mean": None if draws.polls is None else draws.polls[0],
        "poll_sd": None if draws.polls is None else math.sqrt(draws.polls[1]),
        "poll_bias": None if draws.polls is None else draws.polls[2],
        "last_poll": max(fit.polls.fieldwork_end).isoformat() if fit.polls.n else None,
        "ess": draws.ess,
        "inedit": fit.inedit,
        "notes": "; ".join(fit.notes),
        "ceiling_i": fit.ceilings[0],
        "ceiling_j": fit.ceilings[1],
        "ceiling_tau": fit.ceiling_tau if any(c is not None for c in fit.ceilings) else None,
    }
    vi, vj = registered_shares(draws.weights, draws.transfers)
    out["registered_i_median"] = _q(vi, 0.5)
    out["registered_j_median"] = _q(vj, 0.5)
    rows = []
    for k, g in enumerate(draws.groups):
        prior = fit.groups[g]
        t = draws.transfers[:, k, :]
        row = {
            "group": g,
            "weight": float(draws.weights[:, k].mean()),
            "source": prior.source,
            "analogs": ",".join(map(str, prior.analogs)),
            "kappa": prior.kappa,
            "prior_to_i": prior.mean[0],
            "prior_to_j": prior.mean[1],
            "prior_none": prior.mean[2],
        }
        split = t[:, 0] / np.maximum(t[:, 0] + t[:, 1], 1e-12)
        for name, x in (("to_i", t[:, 0]), ("to_j", t[:, 1]), ("none", t[:, 2]), ("split", split)):
            row[f"{name}_median"] = _q(x, 0.5)
            row[f"{name}_q05"] = _q(x, 0.05)
            row[f"{name}_q95"] = _q(x, 0.95)
        rows.append(row)
    out["transfers"] = rows
    return out


# --- prévision pour une élection (commande `vvs second-round`)


def tested_pairs(election: int, asof: dt.date, curated: Path | None = None) -> list[tuple[str, str]]:
    """Paires testées par au moins un sondage de second tour terminé à `asof`."""
    from vvs.model.data import load_polls

    df = load_polls(election, asof, curated=curated, round_=2)
    hyps = df.group_by("hypothesis_id").agg(pl.col("candidate_id").unique())
    return sorted(
        tuple(sorted(c)) for c in hyps["candidate_id"].to_list() if len(c) == 2  # type: ignore[misc]
    )


def _orient(pair: tuple[str, str], blocs: tuple[str, str]) -> tuple[str, str]:
    """Affichage : le finaliste d'extrême droite d'abord s'il y en a un, sinon l'ordre donné."""
    if blocs[1] == "far_right" and blocs[0] != "far_right":
        return pair[1], pair[0]
    return pair


def forecast(
    election: int,
    asof: dt.date,
    pairs: list[tuple[str, str]] | None = None,
    horizon: int | None = None,
    n: int = 10000,
    n_first_round: int = 4000,
    seed: int = 0,
    first_round_days: int = 14,
    registry: Registry | None = None,
    use_potential: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Distribution de la part et P(victoire) pour chaque paire, avec des parts de premier tour
    simples : moyenne des `first_round_days` derniers jours (hypothèses contenant la paire),
    bruitée par `noisy_first_round` à l'horizon ; résultats officiels si le premier tour a eu
    lieu. `use_potential` : plafonds doux du potentiel électoral (docstring du module). Rend
    (une ligne par paire, une ligne par paire × électorat)."""
    registry = registry or default_registry()
    e = registry.elections[election]
    d1 = e.round_(1).date
    horizon = (d1 - asof).days if horizon is None else horizon
    pairs = pairs or tested_pairs(election, asof)
    rng = np.random.default_rng(seed)
    model = SecondRoundModel(registry, use_potential=use_potential).fit(election, asof, pairs)
    kappa = first_round_concentration(max(horizon, 0), exclude={election})
    rows, tr_rows = [], []
    for pair in pairs:
        pf = model.pair_fit(pair)
        pair = _orient(pair, pf.blocs)
        if asof >= d1 and e.round_(1).results:
            fr = {r.candidate_id: np.array([r.share / 100]) for r in e.round_(1).results}
            fr_note = "résultats officiels du premier tour"
        else:
            try:
                mean, (start, end), n_hyp = first_round_mean(
                    election, asof, pair, first_round_days
                )
            except KeyError as err:
                log.warning("%s", err)
                continue
            fr = noisy_first_round(mean, kappa, rng, n_first_round)
            fr_note = (
                f"{n_hyp} hypothèses du {start + dt.timedelta(days=1)} au {end}, Dirichlet κ={kappa:.0f}"
                + (" (fenêtre décalée : aucune hypothèse récente avec la paire)" if end < asof else "")
            )
        d = model.sample(pair, fr, horizon, rng, n)
        s = model.summary(d)
        fr_mean = {c: float(np.mean(v)) for c, v in fr.items()}
        rows.append(
            {k: v for k, v in s.items() if k != "transfers"}
            | {
                "first_round": fr_note,
                "first_round_i": fr_mean.get(pair[0]),
                "first_round_j": fr_mean.get(pair[1]),
            }
        )
        for t in s["transfers"]:
            tr_rows.append({"candidate_i": pair[0], "candidate_j": pair[1]} | t)
    return (
        pl.DataFrame(rows, infer_schema_length=None),
        pl.DataFrame(tr_rows, infer_schema_length=None),
    )


def _pct(x: float | None, nd: int = 1) -> str:
    return "–" if x is None else f"{x:.{nd}f}".replace(".", ",")


def forecast_markdown(pairs: pl.DataFrame, transfers: pl.DataFrame) -> str:
    """Tableau des duels et reports estimés, en markdown."""
    from vvs.model.plots import display_name

    if pairs.is_empty():
        return "# Second tour\n\nAucune paire.\n"
    r0 = pairs.row(0, named=True)
    L = [
        (f"# Second tour {r0['election']} au {r0['asof']} (horizon {r0['horizon']:g} jours "
        "avant le premier tour)"),
        "",
        ("Part de i en % des exprimés : médiane [5 % – 95 %], P(i gagne). Prior : reports "
        "historiques par bloc ou matrice de proximité ; sondages de duel avec effets maison, "
        "biais d'extrême droite et erreur d'horizon calibrés sur 2002-2022."),
        "",
        ("| i | j | 1er tour i / j | médiane | 5–95 % | P(i gagne) | prior seul | sondages "
        "(dernier) | moyenne sondages ± σ | biais retiré | inédit |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in pairs.sort("p_i_wins", descending=True).iter_rows(named=True):
        polls = f"{r['n_polls']} ({r['last_poll']})" if r["n_polls"] else "0"
        pm = (
            f"{_pct(r['poll_mean'])} ± {_pct(r['poll_sd'])}" if r["poll_mean"] is not None else "–"
        )
        L.append(
            f"| {display_name(r['candidate_i'])} | {display_name(r['candidate_j'])} | "
            f"{_pct(100 * r['first_round_i'])} / {_pct(100 * r['first_round_j'])} | "
            f"{_pct(r['median'])} | {_pct(r['q05'])}–{_pct(r['q95'])} | {_pct(r['p_i_wins'], 2)} | "
            f"{_pct(r['prior_median'])} [{_pct(r['prior_q05'])}–{_pct(r['prior_q95'])}] | {polls} | "
            f"{pm} | {_pct(r['poll_bias'])} | {'oui' if r['inedit'] else ''} |"
        )
    L += ["", "Parts de premier tour utilisées :", ""]
    for r in pairs.iter_rows(named=True):
        L.append(f"- {display_name(r['candidate_i'])} – {display_name(r['candidate_j'])} : "
                 f"{r['first_round']}" + (f" — {r['notes']}" if r["notes"] else ""))
    L += [
        "",
        "## Reports estimés (électorats d'au moins 2 % des exprimés)",
        "",
        ("En % de l'électorat du premier tour : vers i, vers j, abstention-blanc-nul ; médiane "
        "[5–95 %] a posteriori."),
        "",
        "| i | j | électorat | poids | source | vers i | vers j | ni |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in transfers.filter(pl.col("weight") >= 0.02).iter_rows(named=True):
        cells = [
            f"{_pct(100 * r[f'{k}_median'], 0)} [{_pct(100 * r[f'{k}_q05'], 0)}–"
            f"{_pct(100 * r[f'{k}_q95'], 0)}]"
            for k in ("to_i", "to_j", "none")
        ]
        src = r["source"] + (f" ({r['analogs']})" if r["analogs"] else "")
        L.append(
            f"| {display_name(r['candidate_i'])} | {display_name(r['candidate_j'])} | "
            f"{r['group']} | {_pct(100 * r['weight'])} | {src} | " + " | ".join(cells) + " |"
        )
    return "\n".join(L) + "\n"


def write_forecast(pairs: pl.DataFrame, transfers: pl.DataFrame, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(out_dir / "pairs.parquet")
    transfers.write_parquet(out_dir / "transfers.parquet")
    (out_dir / "report.md").write_text(forecast_markdown(pairs, transfers), encoding="utf-8")
    return out_dir
