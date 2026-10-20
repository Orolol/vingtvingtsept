"""Métriques du backtest (docs/07-evaluation.md), calculées sur des tirages.

Conventions : parts en points (0-100) pour le CRPS et l'erreur absolue ; probabilités
estimées par fréquence sur les tirages.

- `crps` : CRPS empirique, E|X − y| − ½ E|X − X'|, estimé sans biais de lissage par la forme
  triée (Hersbach 2000) : ½ E|X − X'| = (1/N²) Σ_i (2i − N − 1) x_(i).
- `pair_log_score` : log de la probabilité prévue de la paire de qualifiés réelle. Une
  fréquence nulle sur N tirages donnerait −∞ : on lisse par (k + ½) / (N + ½ M), M étant le
  nombre de paires possibles (prior de Jeffreys sur les paires) ; le plancher vaut environ
  log(1/(2N)) ≈ −9,9 pour 10 000 tirages.
- `brier` : moyenne sur les candidats de (P(qualifié) − 1{qualifié})².
- `coverage` / `pit` : intervalles centraux et transformée intégrale de probabilité.
"""

from __future__ import annotations

import math

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def crps(samples: np.ndarray, y) -> np.ndarray:
    """CRPS par colonne : samples (N, K) ou (N,), y (K,) ou scalaire."""
    x = np.sort(np.asarray(samples, dtype=float), axis=0)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    term1 = np.abs(x - y).mean(axis=0)
    i = np.arange(1, n + 1).reshape((n,) + (1,) * (x.ndim - 1))
    term2 = ((2 * i - n - 1) * x).sum(axis=0) / n**2
    return term1 - term2


def top2(shares: np.ndarray) -> np.ndarray:
    """Indices (N, 2) des deux premiers de chaque tirage, triés par indice croissant."""
    idx = np.argpartition(-shares, 1, axis=1)[:, :2]
    return np.sort(idx, axis=1)


def qualification_probs(shares: np.ndarray) -> np.ndarray:
    """P(candidat parmi les deux premiers), (K,)."""
    t = top2(shares)
    K = shares.shape[1]
    counts = np.bincount(t.ravel(), minlength=K)
    return counts / shares.shape[0]


def pair_probability(shares: np.ndarray, pair: tuple[int, int], smooth: bool = True) -> float:
    t = top2(shares)
    a, b = sorted(pair)
    k = int(((t[:, 0] == a) & (t[:, 1] == b)).sum())
    n = shares.shape[0]
    if not smooth:
        return k / n
    m = shares.shape[1] * (shares.shape[1] - 1) / 2
    return (k + 0.5) / (n + 0.5 * m)


def pair_log_score(shares: np.ndarray, pair: tuple[int, int]) -> float:
    return math.log(pair_probability(shares, pair))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(((probs - outcomes) ** 2).mean())


def central_interval(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(samples, (1 - level) / 2, axis=0)
    hi = np.quantile(samples, (1 + level) / 2, axis=0)
    return lo, hi


def covered(samples: np.ndarray, y, level: float) -> np.ndarray:
    lo, hi = central_interval(samples, level)
    y = np.asarray(y, dtype=float)
    return (lo <= y) & (y <= hi)


def pit(samples: np.ndarray, y) -> np.ndarray:
    """P(X < y) + ½ P(X = y), par colonne."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x < y).mean(axis=0) + 0.5 * (x == y).mean(axis=0)


def abs_error_median(samples: np.ndarray, y) -> np.ndarray:
    return np.abs(np.median(samples, axis=0) - np.asarray(y, dtype=float))
