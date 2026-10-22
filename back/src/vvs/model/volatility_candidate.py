"""Volatilité réalisée par candidat comme échelle de la composante propre v_c de η.

L'étude de la volatilité de campagne (`vvs.model.volatility`, runs/2026-10-20-volatility) n'a
trouvé de signal qu'au niveau du candidat : le log du mouvement ultérieur d'un candidat croît
avec le log de sa volatilité réalisée, à part et horizon contrôlés (β = +0,28, placebo
p < 0,001). Ce module écrit ce signal comme une échelle de l'écart type de v_c :

    v_c ~ N(0, s_c² · (σ²_Cf + a_C·g(h) + ρ/p_c)),   s_c = clip((V_c / V_ref(p_c, τ))^κ, 0,5, 2)

- V_c : volatilité réalisée du candidat à la date de prévision (`volatility.realized_volatility`,
  fenêtre de 8 semaines, log-parts hebdomadaires, bruit d'échantillonnage retiré), sur la
  configuration tirée et les seuls sondages visibles ;
- V_ref(p, τ) : volatilité médiane des candidats de même taille (classes de part moyenne sur la
  fenêtre : < 3 %, 3-8 %, 8-15 %, ≥ 15 %) aux horizons voisins (facteur 1,5 autour de τ) dans
  les élections d'apprentissage ; le rapport est borné à [1/4, 4] avant la puissance (V vaut
  0 pour un tiers des candidats : mouvement sous le bruit d'échantillonnage) ;
- κ estimé en validation croisée interne (laisser une élection d'apprentissage de côté) sur les
  données de calibration de η, les paramètres de η réestimés conjointement à chaque κ ; κ = 0
  redonne exactement le modèle actuel. Composante de bloc u_b et échelle λ inchangées.

Dans la vraisemblance de calibration, l'item de mouvement de l'horizon H (`error.move_items`)
reçoit l'échelle des volatilités à vote − H, l'item d'erreur finale celle de la veille du vote ;
les items des scrutins de liste n'en reçoivent pas (s = 1).

Usage : `ErrorModel.fit(..., candidate_volatility=True)` attache un `CandidateVolatility`
(`ErrorModel.cand_volatility`) ; `CandidateVolatility.scaler(élection, asof)` rend la fonction
candidats -> échelle à poser dans `ErrorModel.cand_scale` à la date de prévision.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from vvs.model import volatility as vo
from vvs.registry import BLOCS, Registry, default_registry

log = logging.getLogger(__name__)

WINDOW = vo.PRIMARY_WINDOW  # semaines
SIZE_EDGES = (0.03, 0.08, 0.15)  # classes de taille (part moyenne sur la fenêtre, fraction)
HORIZON_BAND = 1.5  # horizons de référence dans [τ / 1,5, τ · 1,5]
MIN_REF = 5  # valeurs minimales par classe, sinon tous horizons confondus
RATIO_BOUNDS = (0.25, 4.0)
S_BOUNDS = (0.5, 2.0)
KAPPA_GRID = tuple(np.round(np.arange(-0.5, 1.5001, 0.25), 3))


# --- volatilités par candidat


def candidate_table(
    elections, horizons=vo.REF_HORIZONS, window: int = WINDOW, registry: Registry | None = None
) -> pl.DataFrame:
    """(élection, horizon, unit, V, p) : volatilité de chaque candidat de la configuration
    connue à vote − horizon, et sa part moyenne sur la fenêtre (fraction)."""
    reg = registry or default_registry()
    rows = []
    for e in elections:
        for h in horizons:
            v = vo.volatility_at_horizon(e, int(h), window, reg)
            for u, x in v.by_unit.items():
                rows.append({"election": e, "horizon": int(h), "unit": u, "V": float(x),
                             "p": float(v.weights[u])})
    schema = {"election": pl.Int64, "horizon": pl.Int64, "unit": pl.Utf8, "V": pl.Float64,
              "p": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def size_class(p: float) -> int:
    return int(np.searchsorted(SIZE_EDGES, p, side="right"))


@dataclass
class Reference:
    """V_ref(p, τ) : médiane des volatilités des candidats de même classe de taille aux horizons
    voisins, sur les élections `train`."""

    train: list
    # classe -> [(horizon, V)]
    values: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    @classmethod
    def build(cls, table: pl.DataFrame, train) -> Reference:
        train = list(train)
        d = table.filter(pl.col("election").is_in(train) & pl.col("V").is_finite())
        vals: dict[int, list] = {}
        for h, v, p in d.select("horizon", "V", "p").iter_rows():
            vals.setdefault(size_class(p), []).append((int(h), float(v)))
        return cls(train, vals)

    def vref(self, p: float, horizon: float) -> float:
        hv = self.values.get(size_class(p), [])
        if not hv:
            return math.nan
        lh = math.log(max(float(horizon), 1.0))
        near = [v for h, v in hv if abs(math.log(h) - lh) <= math.log(HORIZON_BAND) + 1e-9]
        x = near if len(near) >= MIN_REF else [v for _, v in hv]
        return float(np.median(x))

    def to_dict(self) -> dict:
        return {"train": self.train,
                "median_by_class": {str(k): float(np.median([v for _, v in hv]))
                                    for k, hv in sorted(self.values.items())},
                "n_by_class": {str(k): len(hv) for k, hv in sorted(self.values.items())}}


def ratio(v: float, p: float, horizon: float, ref: Reference) -> float:
    """V / V_ref borné à `RATIO_BOUNDS` ; NaN si V ou V_ref est indéfini."""
    r = ref.vref(p, horizon)
    if not (np.isfinite(v) and np.isfinite(r)) or r <= 0:
        return math.nan
    return float(np.clip(v / r, *RATIO_BOUNDS))


def scale_from_ratio(r, kappa: float) -> np.ndarray:
    """s = clip(r^κ, S_BOUNDS), 1 là où r est indéfini (pas d'information : modèle actuel)."""
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    ok = np.isfinite(r)
    out[ok] = np.clip(r[ok] ** kappa, *S_BOUNDS)
    return out


# --- calibration


def _lookup(table: pl.DataFrame) -> dict[tuple, tuple[float, float]]:
    return {(e, h, u): (v, p) for e, h, u, v, p in
            table.select("election", "horizon", "unit", "V", "p").iter_rows()}


def item_ratios(cal_items: list, table: pl.DataFrame, ref: Reference) -> list[np.ndarray | None]:
    """Rapports V / V_ref par unité de chaque item (`volatility.CalItem`) ; None pour un item
    sans horizon de prévision (scrutin de liste)."""
    look = _lookup(table)
    out = []
    for ci in cal_items:
        if ci is None or not hasattr(ci, "horizon"):
            out.append(None)
            continue
        it = ci.item
        h = ci.horizon if ci.horizon > 0 else 1
        r = []
        for u in it.units:
            v, p = look.get((it.election, h, u), (math.nan, math.nan))
            r.append(ratio(v, p, h, ref) if np.isfinite(v) else math.nan)
        out.append(np.array(r))
    return out


def _neg_objective_scaled(x, items, s2, gamma, nu, bias):
    """`error._neg_objective` avec la variance propre de chaque item multipliée par s² (par
    unité ; None : inchangée)."""
    from vvs.model import error as er

    sb2, sc2, rho, aB, aC, w = er._unpack(x)
    total = 0.0
    for it, sq in zip(items, s2):
        cov = er.item_cov(it, sb2, sc2, rho, aB, aC, w, gamma)
        if sq is not None:
            vc = (sc2 if it.kind == "final" else aC * er._g(it.h, gamma)) + rho / it.p
            cov = cov + np.diag((sq - 1.0) * vc)
        d = it.d
        if it.kind == "final" and bias:
            d = d - np.array([bias.get(b, 0.0) for b in it.blocs])
        total += it.weight * er._contrast_logpdf(d, cov, nu)
    lp = -0.5 * ((x[er.N_BASE:] / er.BLOC_WEIGHT_PRIOR_SD) ** 2).sum()
    centers = np.array([math.log(0.1), math.log(0.1), math.log(1e-3), math.log(0.05),
                        math.log(0.05)])
    lp += -0.5 * (((x[:er.N_BASE] - centers) / 2.0) ** 2).sum()
    return -(total + lp)


def _fit_scaled(items, s2, gamma, nu, bias) -> np.ndarray:
    from scipy.optimize import minimize

    from vvs.model import error as er

    if all(s is None or np.all(s == 1.0) for s in s2):
        return er._fit_params(items, gamma, nu, bias)
    x0 = np.concatenate([[math.log(0.1), math.log(0.1), math.log(1e-3), math.log(0.05),
                          math.log(0.05)], np.zeros(len(BLOCS))])
    res = minimize(_neg_objective_scaled, x0, args=(items, s2, gamma, nu, bias), method="L-BFGS-B")
    return res.x


def _squares(ratios, kappa: float) -> list:
    return [None if r is None else scale_from_ratio(r, kappa) ** 2 for r in ratios]


def cv_profile(
    cal_items: list, extra: list, table: pl.DataFrame, train, gamma: float, nu: float,
    grid=KAPPA_GRID, use_bias: bool = True,
) -> dict[float, float]:
    """Log-vraisemblance en validation croisée interne de chaque κ de `grid` (et 0) : chaque
    groupe d'apprentissage (présidentielle de `train`, scrutin de liste de `extra`) laissé de
    côté à son tour ; V_ref et paramètres de η sur les autres."""
    from vvs.model import error as er

    train = list(train)
    groups = train + list(dict.fromkeys(it.election for it in extra))
    pres = [ci for ci in cal_items if ci.item.election in train]
    out = {}
    for k in sorted({float(x) for x in grid} | {0.0}):
        s = 0.0
        for held in groups:
            tr_e = [e for e in train if e != held]
            ref = Reference.build(table, tr_e)
            tr = [ci for ci in pres if ci.item.election != held]
            te = [ci for ci in pres if ci.item.election == held]
            tr_x = [it for it in extra if it.election != held]
            te_x = [it for it in extra if it.election == held]
            tr_it = [ci.item for ci in tr] + tr_x
            te_it = [ci.item for ci in te] + te_x
            tr_s2 = _squares(item_ratios(tr, table, ref), k) + [None] * len(tr_x)
            te_s2 = _squares(item_ratios(te, table, ref), k) + [None] * len(te_x)
            b_in, _ = er.stable_bias(tr_it) if use_bias else ({}, {})
            x = _fit_scaled(tr_it, tr_s2, gamma, nu, b_in)
            s -= (_neg_objective_scaled(x, te_it, te_s2, gamma, nu, b_in)
                  - _neg_objective_scaled(x, [], [], gamma, nu, b_in))
        out[k] = s
    return out


# --- modèle


@dataclass
class CandidateVolatility:
    """Échelle par candidat de v_c à la date de prévision (`ErrorModel.cand_volatility`)."""

    kappa: float
    ref: Reference
    window: int = WINDOW
    profile: dict = field(default_factory=dict)

    def scales(self, by_unit: dict[str, float], weights: dict[str, float], candidates,
               horizon: float) -> np.ndarray:
        r = [ratio(by_unit[c], weights[c], horizon, self.ref) if c in by_unit else math.nan
             for c in candidates]
        return scale_from_ratio(r, self.kappa)

    def scaler(self, election: int, asof: dt.date, polls: pl.DataFrame | None = None,
               registry: Registry | None = None) -> CandidateScaler:
        return CandidateScaler(self, election, asof, polls, registry)

    def to_dict(self) -> dict:
        return {"kappa": self.kappa, "window": self.window, "reference": self.ref.to_dict(),
                "profile": {str(k): v for k, v in self.profile.items()},
                "ratio_bounds": RATIO_BOUNDS, "s_bounds": S_BOUNDS}


class CandidateScaler:
    """Fonction candidats -> (K,) échelles s_c à `asof` (à poser dans `ErrorModel.cand_scale`).

    La volatilité est calculée sur la configuration passée (sondages visibles à `asof`) ; un
    candidat sans bloc connu ou sans volatilité définie reçoit 1."""

    def __init__(self, cv: CandidateVolatility, election: int, asof: dt.date,
                 polls: pl.DataFrame | None = None, registry: Registry | None = None):
        from vvs.model.data import election_date, load_polls

        self.cv = cv
        self.election = election
        self.asof = asof
        self.reg = registry or default_registry()
        self.polls = load_polls(election, asof) if polls is None else polls
        self.horizon = float((election_date(self.reg, election) - asof).days)
        self._cache: dict[tuple, np.ndarray] = {}
        self.last: dict[str, float] = {}

    def index(self, candidates) -> vo.VolatilityIndex | None:
        cands, blocs = [], []
        for c in candidates:
            try:
                b = self.reg.bloc_of(c, self.election)
            except (KeyError, ValueError):
                continue
            cands.append(c)
            blocs.append(b)
        if len(cands) < 2:
            return None
        return vo.realized_volatility(self.polls, self.asof, cands, blocs, self.cv.window)

    def __call__(self, candidates) -> np.ndarray:
        key = tuple(candidates)
        if key not in self._cache:
            if self.cv.kappa == 0.0:
                s = np.ones(len(key))
            else:
                v = self.index(key)
                s = (np.ones(len(key)) if v is None
                     else self.cv.scales(v.by_unit, v.weights, key, self.horizon))
            self._cache[key] = s
            self.last = dict(zip(key, s.tolist()))
        return self._cache[key]


def attach(model, train, extra: list, gamma: float, bias: dict, registry: Registry | None = None,
           grid=KAPPA_GRID, use_bias: bool = True):
    """κ en validation croisée interne sur `train` (+ `extra`), η réestimé conjointement à κ
    retenu, `model.cand_volatility` posé. κ = 0 : paramètres inchangés."""
    from vvs.model import error as er

    reg = registry or default_registry()
    train = [e for e in train if isinstance(e, int | np.integer)]
    pr = model.params
    cal = vo.calibration_items(train, reg)
    table = candidate_table(train, registry=reg)
    prof = cv_profile(cal, list(extra), table, train, pr.gamma, pr.nu, grid, use_bias)
    kappa = max((k for k in prof if k in {float(x) for x in grid}), key=prof.get)
    ref = Reference.build(table, train)
    if kappa != 0.0:
        items = [ci.item for ci in cal] + list(extra)
        s2 = _squares(item_ratios(cal, table, ref), kappa) + [None] * len(extra)
        x = _fit_scaled(items, s2, pr.gamma, pr.nu, bias)
        sb2, sc2, rho, aB, aC, w = er._unpack(x)
        model.params = replace(
            pr, sigma_bloc_final=math.sqrt(sb2), sigma_cand_final=math.sqrt(sc2), rho=rho,
            a_bloc=aB, a_cand=aC, bloc_weight={b: float(v) for b, v in zip(BLOCS, w)},
        )
    model.cand_volatility = CandidateVolatility(kappa, ref, WINDOW, prof)
    model.info["candidate_volatility"] = model.cand_volatility.to_dict() | {
        "gain_vs_kappa0": prof[kappa] - prof[0.0]}
    return model
