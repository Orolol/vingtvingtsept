"""Certitude du choix par candidat : lecture, mesure à une date et mise à l'échelle.

Hypothèse testée (étude `studies/certainty.py`) : un candidat dont les électeurs se disent
moins souvent sûrs de leur choix a un soutien plus volatil d'ici au vote. Deux usages, tous
deux désactivés par défaut :

- `first_round.Priors.certainty` : volatilité de la marche aléatoire du candidat δ mise à
  l'échelle par candidat, σ_δ,c = σ_δ · s_c (`delta_scale`) ;
- composante candidat v_c de η (`vvs.model.error`) : Var(v_c) · s_c² (`CertaintyScaler`).

Mesure à la date asof (`effects`). Sondages terminés dans les `WINDOW_DAYS` jours avant asof
(fenêtre élargie à `MAX_WINDOW_DAYS` si aucun) ; modèle additif part_sûrs(sondage, c) =
a_sondage + b_c ajusté par moindres carrés (les instituts ne mesurent pas sur la même base :
inscrits ou certains d'aller voter, intention exprimée ou non ; a_sondage absorbe ces écarts de
niveau). x_c = (b_c − moyenne des b) / 100, centré sur les candidats mesurés : la mise à
l'échelle ne change que la répartition de la variance entre candidats, pas son niveau, déjà
calibré par horizon. Un candidat sans mesure a x = 0 (échelle 1).

Forme, un paramètre κ : s_c = exp(−κ · x_c), bornée à `SCALE_BOUNDS`. κ > 0 : moins de
certitude, plus de volatilité. κ est estimé par `fit_kappa` sur les élections d'apprentissage
seulement (jamais sur l'élection évaluée) : maximum de vraisemblance gaussien des erreurs
totales log(résultat) − log(moyenne des sondages à l'horizon), standardisées par la variance
de η du modèle d'erreur calibré sans l'élection évaluée, avec une échelle libre par
(élection, horizon) et un prior N(0, `KAPPA_PRIOR_SD`²) sur κ.
"""

from __future__ import annotations

import datetime as dt
import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths

CERTAINTY_PATH = paths.CURATED / "poll_certainty.parquet"
TOTAL_ID = "_all"
WINDOW_DAYS = 30
MAX_WINDOW_DAYS = 60
SCALE_BOUNDS = (0.5, 2.0)
KAPPA_PRIOR_SD = 3.0
FIT_HORIZONS = (180, 120, 90, 60, 30, 7)
CERTAINTY_ELECTIONS = (2017, 2022)


@functools.lru_cache(maxsize=4)
def _read(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def load(path: Path | None = None) -> pl.DataFrame:
    """Lignes par candidat de `poll_certainty.parquet` (sans les totaux)."""
    p = Path(path or CERTAINTY_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} absent : lancer `vvs ingest certainty`")
    return _read(str(p)).filter(pl.col("candidate_id") != TOTAL_ID)


def two_way_effects(rows: pl.DataFrame, n_iter: int = 50) -> dict[str, float]:
    """b_c du modèle part = a_sondage + b_c (moindres carrés par moyennes alternées), centrés."""
    if rows.is_empty():
        return {}
    polls = rows["poll_id"].to_list()
    cands = rows["candidate_id"].to_list()
    y = rows["certain_share"].to_numpy().astype(float)
    pu, pi = np.unique(polls, return_inverse=True)
    cu, ci = np.unique(cands, return_inverse=True)
    a = np.zeros(len(pu))
    b = np.zeros(len(cu))
    for _ in range(n_iter):
        a = np.bincount(pi, y - b[ci], len(pu)) / np.bincount(pi, minlength=len(pu))
        b = np.bincount(ci, y - a[pi], len(cu)) / np.bincount(ci, minlength=len(cu))
    b = b - b.mean()
    return {str(c): float(v) for c, v in zip(cu, b)}


def effects(
    election: int,
    asof: dt.date,
    data: pl.DataFrame | None = None,
    window_days: int = WINDOW_DAYS,
    max_window_days: int = MAX_WINDOW_DAYS,
    lo: dt.date | None = None,
) -> dict[str, float]:
    """x_c (fraction, centré) mesuré sur les sondages terminés avant `asof` (voir le module).

    `lo` : borne basse explicite de la fenêtre (l'étude mesure par tranche d'horizon)."""
    df = load() if data is None else data
    df = df.filter((pl.col("election") == election) & (pl.col("fieldwork_end") <= asof))
    if lo is not None:
        sub = df.filter(pl.col("fieldwork_end") > lo)
    else:
        w = window_days
        while True:
            sub = df.filter(pl.col("fieldwork_end") > asof - dt.timedelta(days=w))
            if sub["poll_id"].n_unique() > 0 or w >= max_window_days:
                break
            w += window_days
    b = two_way_effects(sub)
    return {c: v / 100 for c, v in b.items()}


def scale(x: np.ndarray | float, kappa: float) -> np.ndarray:
    """s = exp(−κ x), bornée à SCALE_BOUNDS."""
    return np.clip(np.exp(-kappa * np.asarray(x, dtype=float)), *SCALE_BOUNDS)


def scales_for(candidates: list[str], x: dict[str, float], kappa: float) -> np.ndarray:
    return scale(np.array([x.get(c, 0.0) for c in candidates]), kappa)


# --- estimation de κ


@dataclass
class KappaFit:
    kappa: float
    train: list[int]
    n_obs: int
    n_cells: int
    loglik_gain: float  # gain de log-vraisemblance à κ estimé contre κ = 0
    se: float  # écart type (courbure de la log-vraisemblance pénalisée)

    def to_dict(self) -> dict:
        return {
            "kappa": self.kappa,
            "train": self.train,
            "n_obs": self.n_obs,
            "n_cells": self.n_cells,
            "loglik_gain": self.loglik_gain,
            "se": self.se,
        }


@dataclass
class Cell:
    """Erreurs totales d'une (élection, horizon), standardisées par η, avec x_c."""

    election: int
    horizon: int
    candidates: list[str]
    x: np.ndarray  # certitude centrée (fraction)
    t: np.ndarray  # log(résultat) − log(moyenne à l'horizon), parts renormalisées, centré
    vb: np.ndarray  # variance de bloc de η (sans λ)
    vc: np.ndarray  # variance propre de η


def error_cells(
    election: int,
    em,
    horizons=FIT_HORIZONS,
    data: pl.DataFrame | None = None,
    registry=None,
) -> list[Cell]:
    """Cellules (élection, horizon) pour `fit_kappa` : candidats officiels mesurés."""
    from vvs.eval.baselines import poll_mean
    from vvs.model.data import election_date
    from vvs.registry import default_registry

    reg = registry or default_registry()
    vote = election_date(reg, election)
    e = reg.elections[election]
    res = {r.candidate_id: r.share for r in e.round_(1).results}
    ids = [c.candidate_id for c in e.candidates]
    cells = []
    for h in horizons:
        asof = vote - dt.timedelta(days=max(h, 1))
        x = effects(election, asof, data)
        pm = poll_mean(election, asof, ids, days=30 if h > 7 else 14, registry=reg)
        keep = [
            i
            for i, c in enumerate(ids)
            if c in x and c not in pm.missing and pm.shares[i] > 0
        ]
        if len(keep) < 3:
            continue
        cs = [ids[i] for i in keep]
        a = pm.shares[keep] / pm.shares[keep].sum()
        r = np.array([res[c] for c in cs])
        r = r / r.sum()
        t = np.log(r) - np.log(a)
        t = t - np.average(t, weights=a)
        blocs = [reg.bloc_of(c, election) for c in cs]
        vb, vc = em.variances(pm.age_days, blocs, a)
        xs = np.array([x[c] for c in cs])
        cells.append(Cell(election, h, cs, xs - xs.mean(), t, np.asarray(vb), np.asarray(vc)))
    return cells


def _loglik(kappa: float, cells: list[Cell]) -> float:
    """Log-vraisemblance profilée (échelle libre par cellule), diagonale."""
    total = 0.0
    for c in cells:
        v = c.vb + c.vc * scale(c.x, kappa) ** 2
        lam2 = float(np.mean(c.t**2 / v))
        total += -0.5 * float(np.sum(np.log(lam2 * v))) - 0.5 * len(c.t)
    return total


def fit_kappa_cells(cells: list[Cell], prior_sd: float = KAPPA_PRIOR_SD) -> tuple[float, float, float]:
    """(κ, gain de log-vraisemblance contre 0, écart type) sur une grille fine."""
    grid = np.linspace(-10, 10, 401)
    obj = np.array([_loglik(k, cells) - 0.5 * (k / prior_sd) ** 2 for k in grid])
    i = int(np.argmax(obj))
    k = float(grid[i])
    gain = float(_loglik(k, cells) - _loglik(0.0, cells))
    j = min(max(i, 1), len(grid) - 2)
    step = grid[1] - grid[0]
    curv = (obj[j + 1] - 2 * obj[j] + obj[j - 1]) / step**2
    se = float(1 / math.sqrt(-curv)) if curv < 0 else float("inf")
    return k, gain, se


def fit_kappa(
    exclude: int,
    elections=CERTAINTY_ELECTIONS,
    calibration_elections=(2002, 2007, 2012, 2017, 2022),
    data: pl.DataFrame | None = None,
    error_model=None,
) -> KappaFit:
    """κ estimé sur `elections` privées de `exclude`, η calibré sans `exclude`."""
    from vvs.model.error import ErrorModel

    em = error_model or ErrorModel.fit(elections=calibration_elections, exclude=[exclude])
    train = [e for e in elections if e != exclude]
    cells = [c for e in train for c in error_cells(e, em, data=data)]
    if not cells:
        return KappaFit(0.0, train, 0, 0, 0.0, float("inf"))
    k, gain, se = fit_kappa_cells(cells)
    return KappaFit(k, train, sum(len(c.t) for c in cells), len(cells), gain, se)


# --- intégration au modèle de premier tour


def delta_scale(data, kappa: float, certainty: pl.DataFrame | None = None) -> np.ndarray:
    """(C,) échelle de la volatilité δ de chaque candidat de `data` (PollData) à son asof."""
    x = effects(data.election, data.asof, certainty)
    return scales_for(list(data.candidates), x, kappa)


@functools.lru_cache(maxsize=8)
def loo_kappa(exclude: int) -> float:
    """κ calibré sans `exclude` (mis en cache par processus)."""
    return fit_kappa(exclude).kappa


class CertaintyScaler:
    """Échelle de la composante candidat de η à la date de prévision (voir `error`)."""

    def __init__(self, election: int, asof: dt.date, kappa: float, data: pl.DataFrame | None = None):
        self.kappa = kappa
        self.x = effects(election, asof, data)

    def __call__(self, candidates: list[str]) -> np.ndarray:
        return scales_for(list(candidates), self.x, self.kappa)


def certainty_inputs(data, priors) -> dict:
    """Champs de `first_round.ModelInputs` pour l'option `Priors.certainty` ({} sans l'option).

    κ : `priors.certainty_kappa` s'il est donné, sinon calibré sans l'élection de `data`
    (`loo_kappa`). Seule la paramétrisation non centrée des marches l'utilise."""
    if priors is None or not getattr(priors, "certainty", False):
        return {}
    import jax.numpy as jnp

    kappa = getattr(priors, "certainty_kappa", None)
    if kappa is None:
        kappa = loo_kappa(int(data.election))
    return {"delta_scale": jnp.asarray(delta_scale(data, float(kappa)))}
