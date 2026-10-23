"""Décalage commun `u` des reports vers le finaliste d'extrême droite, et son centrage.

Outils de l'étude « normalisation du RN et érosion du front républicain »
(`back/studies/rn_image_transfers.py`, runs/<date>-rn-image). Rien ici n'est utilisé par la
prévision : le modèle de second tour (vvs.model.second_round) garde `u ~ N(0, σ_u)`.

Convention. Toutes les quantités sont orientées vers le finaliste d'extrême droite (RN) : la
« répartition » d'un électorat est la part du finaliste RN parmi les exprimés de cet électorat
au second tour, et un décalage `u > 0` (logit) signifie des reports plus favorables au RN que
la référence (érosion du front républicain), `u < 0` un barrage plus fort.

Décalage commun mesuré. Pour des électorats `g` de poids `W_g` (part de premier tour × part
d'exprimés au second tour), de répartition de référence `r_g` et de répartition observée `s_g`,
`u` résout

    Σ_g W_g expit(logit(r_g) + u) = Σ_g W_g s_g,

c'est-à-dire le décalage commun qui, appliqué à la référence, rend le même nombre de voix au
RN que les reports observés. C'est l'équivalent agrégé de la moyenne des écarts de logit, sans
donner un poids démesuré aux petits électorats aux répartitions extrêmes.

Électorats du barrage. Le décalage mesuré ne porte que sur les électorats non finalistes hors
extrême droite (`bloc:<b>` pour b ≠ far_right, et `abstain`) : l'électorat d'extrême droite non
finaliste (Mégret, Dupont-Aignan, Zemmour) n'est pas concerné par le front républicain et la
matrice de proximité le décrit mal (2017 : 98 % vers Le Pen dans la matrice, 55 % observé).
Le centrage injecté (`shift_fit`) porte sur les mêmes groupes.

Centrage injectable. `ShiftedSecondRoundModel` est un `SecondRoundModel` dont les moyennes
des Dirichlet des groupes du barrage sont décalées de `δ` en logit, vers le RN, après
l'ajustement. Le décalage commun du modèle reste `u ~ N(0, σ_u)` autour de ces moyennes
décalées : c'est une approximation du centrage `u ~ N(δ, σ_u)`, exacte pour la moyenne de la
répartition à la dispersion des Dirichlet près. Les défauts du module second_round ne sont pas
touchés.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from vvs.model.second_round import (
    ABSTAIN,
    GroupPrior,
    PairFit,
    SecondRoundModel,
    SecondRoundSettings,
    proximity_mean,
)
from vvs.registry import Registry
from vvs.registry.transfers import TransferSource, transfer_prior

RN_BLOC = "far_right"
# électorats spéciaux du premier tour regroupés dans `abstain` (comme le modèle)
SPECIAL = ("abstain", "blank_null", "abstain_or_blank_null")
LOGIT_CLIP = 1e-4


def logit(p: float | np.ndarray) -> float | np.ndarray:
    q = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(q) - np.log1p(-q)


def expit(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def barrage_group(group: str) -> bool:
    """Groupe concerné par le front républicain : non finaliste hors extrême droite."""
    return group == ABSTAIN or (group.startswith("bloc:") and group != f"bloc:{RN_BLOC}")


# --- décalage commun


def common_shift(
    ref: Sequence[float], obs: Sequence[float], weight: Sequence[float], lo: float = -12.0,
    hi: float = 12.0, iters: int = 80,
) -> float:
    """`u` tel que Σ W expit(logit(ref) + u) = Σ W obs (répartitions en fractions)."""
    r = logit(np.asarray(ref, dtype=float))
    o = np.asarray(obs, dtype=float)
    w = np.asarray(weight, dtype=float)
    if len(r) == 0 or w.sum() <= 0:
        raise ValueError("décalage commun : aucun électorat pondéré")
    target = float((w * o).sum())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float((w * expit(r + mid)).sum()) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def aggregate_shift(
    fixed_votes: float, fixed_weight: float, ref: Sequence[float], weight: Sequence[float],
    target_share: float, lo: float = -12.0, hi: float = 12.0, iters: int = 80,
) -> float:
    """`u` tel que la part du RN parmi les exprimés du second tour vaille `target_share`
    (fraction) : (fixed_votes + Σ W expit(logit(ref) + u)) / (fixed_weight + Σ W) ; les
    électorats « fixes » (finalistes, extrême droite non finaliste) ne sont pas décalés."""
    r = logit(np.asarray(ref, dtype=float))
    w = np.asarray(weight, dtype=float)
    total = fixed_weight + float(w.sum())

    def share(u: float) -> float:
        return (fixed_votes + float((w * expit(r + u)).sum())) / total

    if not share(lo) <= target_share <= share(hi):
        raise ValueError(f"part cible {target_share:.3f} hors d'atteinte ({share(lo):.3f}–{share(hi):.3f})")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if share(mid) < target_share:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- électorats d'un second tour et reports observés


def rn_pair(registry: Registry, election: int, pair: Iterable[str]) -> tuple[str, str]:
    """(finaliste RN, adversaire) ; lève ValueError si le duel n'oppose pas un seul RN."""
    a, b = tuple(pair)
    ba, bb = registry.bloc_of(a, election), registry.bloc_of(b, election)
    if ba == RN_BLOC and bb != RN_BLOC:
        return a, b
    if bb == RN_BLOC and ba != RN_BLOC:
        return b, a
    raise ValueError(f"{election} : {a}–{b} n'oppose pas un finaliste d'extrême droite à un autre bloc")


def group_of(registry: Registry, election: int, cid: str, pair: tuple[str, str]) -> str:
    if cid in SPECIAL:
        return ABSTAIN
    if cid in pair:
        return f"own:{cid}"
    return f"bloc:{registry.bloc_of(cid, election)}"


def first_round_weights(
    registry: Registry, election: int, shares: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Poids de premier tour par électorat, rapportés aux exprimés : candidats (fractions) et
    électorats spéciaux (abstain, blank_null, abstain_or_blank_null, en effectifs / exprimés).
    `shares` : parts de premier tour (fractions des exprimés) à la place des résultats, avec le
    rapport (inscrits − exprimés) / exprimés du premier tour de l'élection."""
    r1 = registry.elections[election].round_(1)
    if shares is None:
        shares = {r.candidate_id: r.share / 100 for r in r1.results}
    w = {c: float(v) for c, v in shares.items()}
    if r1.registered and r1.voters and r1.expressed:
        w["abstain"] = (r1.registered - r1.voters) / r1.expressed
        w["blank_null"] = (r1.voters - r1.expressed) / r1.expressed
        w["abstain_or_blank_null"] = (r1.registered - r1.expressed) / r1.expressed
    return w


@dataclass(frozen=True)
class Cell:
    """Reports observés d'un électorat de groupe dans une source (ou une moyenne)."""

    group: str
    split: float  # part du RN parmi les exprimés de l'électorat (fraction)
    none: float | None  # blancs, nuls, abstention en fraction de l'électorat ; None : inconnu
    weight: float  # poids de premier tour couvert par la source dans le groupe (/ exprimés)


def source_cells(
    registry: Registry, election: int, pair: tuple[str, str], sources: Sequence[TransferSource],
    weights: Mapping[str, float] | None = None,
) -> dict[str, Cell]:
    """Cellules par groupe (finalistes, blocs, abstain) d'un ensemble de sources d'un même
    institut : chaque électorat membre est la moyenne de ses sources, puis le groupe est la
    moyenne des membres pondérée par leur poids de premier tour (comme `analog_prior`).
    `abstain_or_blank_null` n'est utilisé que si ni `abstain` ni `blank_null` ne le sont."""
    rn, other = pair
    weights = weights or first_round_weights(registry, election)
    members: dict[str, list[tuple[float, float, float | None]]] = {}
    froms = {r.from_ for s in sources for r in s.rows}
    for cid in sorted(froms):
        if cid == "abstain_or_blank_null" and ({"abstain", "blank_null"} & froms):
            continue
        if cid not in weights:
            continue  # groupe sans liste de membres (left_excl_royal) : ignoré
        try:
            ex = transfer_prior(election, cid, (rn, other), basis="expressed",
                                transfers=tuple(sources), registry=registry)
        except KeyError:
            continue
        try:
            el = transfer_prior(election, cid, (rn, other), basis="electorate",
                                transfers=tuple(sources), registry=registry)
            none = el.shares["blank_or_abstain"] / 100
        except KeyError:
            none = None
        g = group_of(registry, election, cid, pair)
        members.setdefault(g, []).append((weights[cid], ex.shares[rn] / 100, none))
    out = {}
    for g, items in members.items():
        w = sum(x[0] for x in items)
        nn = [(x[0], x[2]) for x in items if x[2] is not None]
        out[g] = Cell(
            g,
            sum(x[0] * x[1] for x in items) / w,
            sum(a * b for a, b in nn) / sum(a for a, _ in nn) if nn else None,
            w,
        )
    return out


def mean_cells(per_source: Mapping[str, Mapping[str, Cell]]) -> dict[str, Cell]:
    """Moyenne simple des instituts, groupe par groupe (comme `transfer_prior`)."""
    groups = sorted({g for cells in per_source.values() for g in cells})
    out = {}
    for g in groups:
        cs = [cells[g] for cells in per_source.values() if g in cells]
        nn = [c.none for c in cs if c.none is not None]
        out[g] = Cell(g, float(np.mean([c.split for c in cs])),
                      float(np.mean(nn)) if nn else None, float(np.mean([c.weight for c in cs])))
    return out


def group_weights(
    registry: Registry, election: int, pair: tuple[str, str],
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Poids de premier tour de chaque groupe (toutes candidatures du groupe, / exprimés)."""
    weights = weights or first_round_weights(registry, election)
    out: dict[str, float] = {}
    for cid, w in weights.items():
        if cid == "abstain_or_blank_null":
            continue
        g = group_of(registry, election, cid, pair)
        out[g] = out.get(g, 0.0) + w
    return out


def matrix_cell(group: str, other_bloc: str, settings: SecondRoundSettings | None = None):
    """(répartition vers le RN, part ni l'un ni l'autre) de la matrice de proximité."""
    electorate = ABSTAIN if group == ABSTAIN else group.removeprefix("bloc:")
    t_rn, t_o, n = proximity_mean(electorate, RN_BLOC, other_bloc, settings)
    return t_rn / (t_rn + t_o), n


def prior_cell(prior: GroupPrior, rn_first: bool) -> tuple[float, float]:
    """(répartition vers le RN, part ni l'un ni l'autre) de la moyenne d'un prior de groupe."""
    a, b, n = prior.mean
    s = a / (a + b)
    return (s if rn_first else 1 - s), n


# --- centrage injectable


def shift_fit(fit: PairFit, delta: float, groups: Callable[[str], bool] = barrage_group) -> PairFit:
    """Copie de `fit` dont les moyennes des groupes `groups` concernés par u sont décalées de
    `delta` (logit, vers le finaliste d'extrême droite). Duel sans finaliste RN unique : rien."""
    if not delta:
        return fit
    bi, bj = fit.blocs
    if (bi == RN_BLOC) == (bj == RN_BLOC):
        return fit
    d = delta if bi == RN_BLOC else -delta
    new = {}
    for k, g in fit.groups.items():
        if not (g.shift and groups(k)):
            new[k] = g
            continue
        a, b, n = g.mean
        s = float(expit(logit(a / (a + b)) + d))
        new[k] = replace(g, mean=((1 - n) * s, (1 - n) * (1 - s), n),
                         note=(g.note + " ; " if g.note else "") + f"centrage u {delta:+.2f}")
    return replace(fit, groups=new, notes=[*fit.notes, f"centrage du barrage : u {delta:+.2f} (étude rn-image)"])


class ShiftedSecondRoundModel(SecondRoundModel):
    """`SecondRoundModel` dont les groupes du barrage sont centrés sur `u = δ` après
    l'ajustement. `u_shift` : δ commun, ou fonction (paire, blocs) -> δ (par duel)."""

    def __init__(self, *args, u_shift: float | Callable[[tuple, tuple], float] = 0.0, **kw):
        super().__init__(*args, **kw)
        self.u_shift = u_shift

    def fit(self, election, asof, pairs):
        super().fit(election, asof, pairs)
        for pair, f in list(self.fits.items()):
            d = self.u_shift(f.pair, f.blocs) if callable(self.u_shift) else self.u_shift
            self.fits[pair] = shift_fit(f, float(d))
        return self


# --- relation entre un prédicteur et u (n très petit)


@dataclass(frozen=True)
class LineFit:
    """y = a + b (x − x0), moindres carrés pondérés par 1 / (s_i² + τ²)."""

    a: float
    b: float
    x0: float
    cov: np.ndarray  # covariance de (a, b)
    tau: float  # écart-type résiduel supposé (hors erreur de mesure)
    n: int
    rss: float  # somme des carrés des résidus pondérés

    def predict(self, x: float) -> tuple[float, float]:
        """(centre, écart-type) de y en x : incertitude de la droite plus τ."""
        g = np.array([1.0, x - self.x0])
        return self.a + self.b * (x - self.x0), math.sqrt(float(g @ self.cov @ g) + self.tau**2)


def fit_line(
    x: Sequence[float], y: Sequence[float], sy: Sequence[float] | None = None, tau: float = 0.0,
) -> LineFit:
    """Droite pondérée de y sur x : poids 1 / (sy² + τ²), covariance des coefficients
    (XᵀWX)⁻¹ (variances supposées connues : pas de facteur d'échelle estimé, n trop petit)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s2 = np.zeros_like(y) if sy is None else np.asarray(sy, dtype=float) ** 2
    v = s2 + tau**2
    if np.any(v <= 0):
        raise ValueError("variances nulles : donner sy ou τ > 0")
    w = 1.0 / v
    x0 = float((w * x).sum() / w.sum())
    X = np.column_stack([np.ones_like(x), x - x0])
    xtwx = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(xtwx)
    a, b = cov @ (X.T @ (w * y))
    r = y - (a + b * (x - x0))
    return LineFit(float(a), float(b), x0, cov, float(tau), len(x), float((w * r**2).sum()))


def ols_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float | None]:
    """(pente, erreur type) des moindres carrés ordinaires ; erreur type None si n ≤ 2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    if len(x) <= 2:
        return b, None
    r = y - y.mean() - b * xc
    return b, math.sqrt(float((r**2).sum()) / (len(x) - 2) / float((xc**2).sum()))

