"""Volatilité réalisée de la campagne et mise à l'échelle de η (docs/04-modele.md, section 2).

Idée : certaines campagnes bougent beaucoup (2017, 2022), d'autres peu. Le modèle d'erreur
l'écrit comme une échelle aléatoire λ commune à l'élection, tirée sans rien regarder. Si la
volatilité déjà observée dans les sondages à la date de prévision annonce le mouvement à venir,
on peut remplacer une partie de ce hasard par une mesure :

    σ_eff(τ) = σ(τ) · (V(τ) / V_ref(τ))^κ

V(τ) est l'indice de volatilité réalisée à la date de prévision, V_ref(τ) sa moyenne
géométrique sur les élections d'apprentissage au même horizon, κ estimé en
leave-one-election-out (κ = 0 redonne le modèle actuel).

Indice (`realized_volatility`), calculé sur les seuls sondages visibles à la date :

1. chaque sondage contribue l'hypothèse la plus proche de la configuration
   (`configurations.closest_hypothesis`), parts rapportées au total de l'hypothèse ; part d'un
   bloc = somme des parts de ses candidats de la configuration ;
2. semaines k = 0, 1, … finissant à la date (semaine k : fin de terrain dans
   ]asof − 7(k+1), asof − 7k]) ; par unité (candidat ou bloc) et semaine, log de la moyenne des
   parts des sondages de la semaine ;
3. variance hebdomadaire du mouvement, sous une marche aléatoire : sur toutes les paires de
   semaines (k, k') de la fenêtre qui ont des sondages, Σ (Δ² − bruit_k − bruit_k') / Σ |k − k'|.
   Le bruit d'une moyenne hebdomadaire est la variance d'échantillonnage du log de la part,
   (1 − p)/p · moyenne(1/(f·n))/m (m sondages, f = 0,75, n = 1 000 si inconnu). Les paires
   éloignées pèsent plus (leur Δ² croît avec l'écart quand le bruit reste fixe) : une tendance
   compte comme du mouvement (estimateur non centré) ;
4. volatilité de l'unité : racine de ce rapport, plancher à 0, définie si au moins
   `MIN_INCREMENTS` semaines de la fenêtre ont des sondages. Version brute à côté : écart
   quadratique moyen des variations entre semaines consécutives (écart d'au plus
   `MAX_GAP_WEEKS`), ramenées à une semaine, sans correction du bruit. Une première version
   corrigeait par la dispersion intra-semaine des sondages ; elle contient les effets maison,
   qui s'annulent en grande partie d'une semaine à l'autre, et effaçait des tendances nettes
   (Le Pen 2022 de 24 à 16 % entre août et octobre 2021) : abandonnée à l'inspection des seuls
   indices, avant tout lien avec les mouvements ultérieurs ;
5. indice de campagne : moyenne des volatilités des candidats pondérée par leur part moyenne
   sur la fenêtre ; indice de blocs : idem sur les blocs.

L'échelle est appliquée par `VolatilityScale` (attribut `ErrorModel.vol_scale`, None par
défaut), sous deux formes : `move` (seule la part de mouvement a·g(h) de la variance est mise à
l'échelle, l'erreur finale σ²_final + ρ/p reste un plancher) et `total` (toute la variance).
κ est estimé (`fit_kappa`) par validation croisée interne (laisser une élection
d'apprentissage de côté) sur les données de calibration de η : chaque item de mouvement à
l'horizon H est mis à l'échelle par le rapport de volatilité à vote − H, l'item d'erreur finale
(forme `total` seulement) par celui de la veille du vote.
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from vvs.registry import BLOCS, Registry, default_registry

log = logging.getLogger(__name__)

WINDOWS = (4, 8, 12)  # semaines
PRIMARY_WINDOW = 8
MIN_INCREMENTS = 3
MAX_GAP_WEEKS = 4
DESIGN_FACTOR = 0.75
DEFAULT_N = 1000
# horizons de la volatilité de référence (jours avant le vote) ; 1 : veille du vote (item
# d'erreur finale de la calibration)
REF_HORIZONS = (1, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 240)
KAPPA_GRID = tuple(np.round(np.arange(-1.0, 2.0001, 0.25), 3))
FORMS = ("move", "total")


# --- indice


@dataclass
class VolatilityIndex:
    asof: dt.date | None
    window_weeks: int
    campaign: float  # log-parts, bruit retiré (NaN si indéfini)
    campaign_raw: float  # log-parts, sans correction du bruit
    campaign_points: float  # points, bruit retiré
    bloc_campaign: float  # blocs, log-parts, bruit retiré
    by_unit: dict[str, float] = field(default_factory=dict)
    by_bloc: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    n_polls: int = 0
    n_weeks: int = 0  # semaines de la fenêtre avec sondages (max sur les candidats)

    def to_dict(self) -> dict:
        return {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in self.__dict__.items()}


def poll_shares(
    polls: pl.DataFrame, config: list[str], blocs: list[str]
) -> list[tuple[dt.date, dict[str, float], int | None]]:
    """Un tuple par sondage : (fin de terrain, parts des unités en fractions, effectif).

    Unités : candidats de `config` présents dans l'hypothesis la plus proche, et blocs
    (`bloc:<nom>`, somme des parts des candidats de la configuration du bloc)."""
    from vvs.eval.configurations import closest_hypothesis

    hyps: dict[str, dict] = {}
    for pid, end, n, hid, cid, share in polls.select(
        "poll_id", "fieldwork_end", "sample_size", "hypothesis_id", "candidate_id", "share"
    ).iter_rows():
        p = hyps.setdefault(pid, {"end": end, "n": n, "hyps": {}})
        p["hyps"].setdefault(hid, {})[cid] = share
    bloc_of = dict(zip(config, blocs))
    out = []
    for p in hyps.values():
        hid, _ = closest_hypothesis(p["hyps"], config)
        sh = p["hyps"][hid]
        total = sum(sh.values())
        if total <= 0:
            continue
        units = {c: sh[c] / total for c in config if sh.get(c, 0) > 0}
        if not units:
            continue
        cands = dict(units)
        for b in sorted(set(blocs)):
            s = sum(v for c, v in cands.items() if bloc_of[c] == b)
            if s > 0:
                units[f"bloc:{b}"] = s
        out.append((p["end"], units, p["n"]))
    return out


def _unit_volatility(
    weeks: dict[int, list[tuple[float, int | None]]], window: int, points: bool = False
) -> tuple[float, float, int]:
    """(volatilité corrigée, brute, nombre de semaines avec sondages) d'une unité.

    `weeks` : semaine -> [(part, effectif)]. Corrigée : Σ (Δ² − bruit) / Σ écart sur toutes les
    paires de semaines de la fenêtre ; brute : racine de la moyenne des Δ²/écart entre semaines
    consécutives (écart ≤ `MAX_GAP_WEEKS`), sans correction."""
    tr = (lambda x: 100 * x) if points else np.log
    level: dict[int, float] = {}
    noise: dict[int, float] = {}
    for k, vals in weeks.items():
        if k >= window:
            continue
        x = np.array([v for v, _ in vals])
        p = float(x.mean())
        level[k] = float(tr(p))
        # variance d'échantillonnage de la moyenne de la semaine (effectif effectif f·n)
        inv_n = np.mean([1.0 / (DESIGN_FACTOR * (m or DEFAULT_N)) for _, m in vals]) / len(vals)
        noise[k] = (1e4 * p * (1 - p) if points else (1 - p) / p) * inv_n
    ks = sorted(level, reverse=True)  # du plus ancien au plus récent
    if len(ks) < MIN_INCREMENTS:
        return math.nan, math.nan, len(ks)
    num = den = 0.0
    for i, k1 in enumerate(ks):
        for k2 in ks[i + 1 :]:
            num += (level[k2] - level[k1]) ** 2 - noise[k1] - noise[k2]
            den += k1 - k2
    raw = [
        (level[k2] - level[k1]) ** 2 / (k1 - k2)
        for k1, k2 in itertools.pairwise(ks)
        if k1 - k2 <= MAX_GAP_WEEKS
    ]
    raw_v = math.sqrt(float(np.mean(raw))) if raw else math.nan
    return math.sqrt(max(num / den, 0.0)), raw_v, len(ks)


def realized_volatility(
    polls: pl.DataFrame,
    asof: dt.date,
    config: list[str],
    blocs: list[str],
    window_weeks: int = PRIMARY_WINDOW,
) -> VolatilityIndex:
    """Indice de volatilité réalisée à `asof` (voir le docstring du module).

    `polls` : table longue de `vvs.model.data.load_polls` (filtrée ici à fin de terrain ≤ asof) ;
    `config`, `blocs` : configuration de candidats et blocs, alignés."""
    df = polls.filter(
        (pl.col("fieldwork_end") <= asof)
        & (pl.col("fieldwork_end") > asof - dt.timedelta(days=7 * window_weeks))
    )
    rows = poll_shares(df, list(config), list(blocs))
    by_week: dict[str, dict[int, list]] = {}
    n_window = 0
    for end, units, n in rows:
        k = (asof - end).days // 7
        n_window += k < window_weeks
        for u, s in units.items():
            by_week.setdefault(u, {}).setdefault(k, []).append((s, n))
    vol, raw, pts, wts, n_inc = {}, {}, {}, {}, 0
    for u, weeks in by_week.items():
        v, r, m = _unit_volatility(weeks, window_weeks)
        if math.isnan(v):
            continue
        vol[u], raw[u] = v, r
        pts[u] = _unit_volatility(weeks, window_weeks, points=True)[0]
        wts[u] = float(np.mean([s for k, vals in weeks.items() if k < window_weeks
                                for s, _ in vals]))
        n_inc = max(n_inc, m)

    def wmean(d: dict[str, float], units: list[str]) -> float:
        us = [u for u in units if u in d]
        if len(us) < 2:
            return math.nan
        w = np.array([wts[u] for u in us])
        return float((w * np.array([d[u] for u in us])).sum() / w.sum())

    cands = [u for u in vol if not u.startswith("bloc:")]
    bl = [u for u in vol if u.startswith("bloc:")]
    return VolatilityIndex(
        asof=asof,
        window_weeks=window_weeks,
        campaign=wmean(vol, cands),
        campaign_raw=wmean(raw, cands),
        campaign_points=wmean(pts, cands),
        bloc_campaign=wmean(vol, bl),
        by_unit={u: vol[u] for u in cands},
        by_bloc={u.removeprefix("bloc:"): vol[u] for u in bl},
        weights={u: wts[u] for u in cands},
        n_polls=n_window,
        n_weeks=n_inc,
    )


def campaign_volatility(
    election: int,
    asof: dt.date,
    window_weeks: int = PRIMARY_WINDOW,
    registry: Registry | None = None,
    polls: pl.DataFrame | None = None,
    config: list[str] | None = None,
) -> VolatilityIndex:
    """Indice à `asof` sur la configuration connue à cette date (`known_configuration`)."""
    from vvs.eval.configurations import known_configuration
    from vvs.model.data import load_polls

    reg = registry or default_registry()
    polls = load_polls(election, asof) if polls is None else polls
    if config is None:
        config = known_configuration(election, asof, reg, polls=polls).candidates
    blocs = [reg.bloc_of(c, election) for c in config]
    return realized_volatility(polls, asof, config, blocs, window_weeks)


_VOL_CACHE: dict[tuple[int, int, int], VolatilityIndex] = {}


def volatility_at_horizon(
    election: int, horizon: int, window_weeks: int = PRIMARY_WINDOW, registry: Registry | None = None
) -> VolatilityIndex:
    """Indice à vote − `horizon` jours (mis en cache)."""
    from vvs.model.data import election_date, load_polls

    key = (election, int(horizon), window_weeks)
    if key not in _VOL_CACHE:
        reg = registry or default_registry()
        asof = election_date(reg, election) - dt.timedelta(days=int(horizon))
        try:
            _VOL_CACHE[key] = campaign_volatility(
                election, asof, window_weeks, reg, polls=load_polls(election, asof)
            )
        except ValueError:  # aucun sondage avant la date
            _VOL_CACHE[key] = VolatilityIndex(asof, window_weeks, *([math.nan] * 4))
    return _VOL_CACHE[key]


# --- référence et échelle


def reference(
    vols: dict[tuple[int, int], float], elections, horizons=REF_HORIZONS
) -> dict[int, float]:
    """V_ref(h) : moyenne géométrique de V sur `elections` (indices définis), par horizon.

    `vols` : (élection, horizon) -> V."""
    out = {}
    for h in horizons:
        x = [vols.get((e, h), math.nan) for e in elections]
        x = [v for v in x if np.isfinite(v) and v > 0]
        if x:
            out[int(h)] = float(np.exp(np.mean(np.log(x))))
    return out


def interp_reference(ref: dict[int, float], horizon: float) -> float:
    """V_ref à un horizon quelconque : interpolation linéaire de log V_ref en log h."""
    if not ref:
        return math.nan
    hs = np.array(sorted(ref), dtype=float)
    lv = np.log([ref[int(h)] for h in hs])
    return float(np.exp(np.interp(math.log(max(horizon, 1.0)), np.log(hs), lv)))


def ratio(v: float, v_ref: float) -> float:
    """V / V_ref, 1 si l'un des deux est indéfini ou nul (pas d'information : modèle actuel)."""
    if not (np.isfinite(v) and np.isfinite(v_ref)) or v <= 0 or v_ref <= 0:
        return 1.0
    return float(v / v_ref)


@dataclass
class VolatilityScale:
    """Échelle de l'écart type de η à la date de prévision (`ErrorModel.vol_scale`).

    `ratio` : V / V_ref global ; `bloc_ratio` : par bloc (facultatif, remplace `ratio` pour les
    candidats du bloc) ; `form` : `move` (part de mouvement seule) ou `total`."""

    kappa: float
    ratio: float = 1.0
    form: str = "move"
    bloc_ratio: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"forme inconnue : {self.form}")

    def factor(self, blocs: list[str]) -> np.ndarray:
        """(K,) facteur s = ratio^κ de l'écart type, par candidat."""
        r = np.array([self.bloc_ratio.get(b, self.ratio) for b in blocs], dtype=float)
        return r ** self.kappa

    def apply(self, vb, vc, vb_final, vc_final, blocs: list[str]) -> tuple:
        """Variances (de bloc, propre) mises à l'échelle ; `*_final` : leur part d'erreur finale
        (σ²_final, et ρ/p pour la composante propre)."""
        s2 = self.factor(blocs) ** 2
        if self.form == "total":
            return vb * s2, vc * s2
        return vb_final + s2 * (vb - vb_final), vc_final + s2 * (vc - vc_final)


# --- estimation de κ


@dataclass
class CalItem:
    """Item de calibration de η avec l'horizon de sa date de prévision (0 : erreur finale)."""

    item: object  # vvs.model.error.Item
    horizon: int


def calibration_items(elections, reg: Registry | None = None) -> list[CalItem]:
    """Items de `error.calibration_items`, un horizon de date de prévision chacun."""
    from vvs.model import error as er

    reg = reg or default_registry()
    out = []
    for e in elections:
        out.append(CalItem(er.final_item(e, reg), 0))
        moves = []
        for H in er.MOVE_HORIZONS:
            moves += [CalItem(it, H) for it in er.move_items(e, reg, (H,))]
        for m in moves:
            m.item.weight = 1.0 / len(moves)
        out += moves
    return out


def scale_items(
    items: list[CalItem],
    vols: dict[tuple[int, int], float],
    train,
    kappa: float,
    form: str,
    gamma: float,
) -> list:
    """Items de `error.Item` mis à l'échelle par (V / V_ref)^κ, V_ref sur `train`.

    Forme `move` : variance de mouvement a·g(h) × s², obtenue exactement en remplaçant h par
    h·s^(2/γ) ; item final inchangé. Forme `total` : toute la covariance × s², obtenue en
    divisant les log-erreurs par s (le jacobien, constant en les paramètres, est ajouté par
    `item_log_jacobian`)."""
    ref = reference(vols, train)
    out = []
    for ci in items:
        it = ci.item
        h = ci.horizon if ci.horizon > 0 else 1
        s = ratio(vols.get((it.election, h), math.nan), ref.get(h, math.nan)) ** kappa
        if form == "move":
            out.append(it if it.kind == "final" else replace(it, h=it.h * s ** (2 / gamma)))
        else:
            out.append(replace(it, d=it.d / s))
    return out


def item_log_jacobian(
    items: list[CalItem], vols, train, kappa: float, form: str
) -> float:
    """Σ poids · (−(n − 1)·log s) des items de la forme `total` (0 pour `move`)."""
    if form == "move":
        return 0.0
    ref = reference(vols, train)
    tot = 0.0
    for ci in items:
        it = ci.item
        h = ci.horizon if ci.horizon > 0 else 1
        s = ratio(vols.get((it.election, h), math.nan), ref.get(h, math.nan)) ** kappa
        tot -= it.weight * (len(it.d) - 1) * math.log(s)
    return tot


def cv_score(
    items: list[CalItem], vols, train, kappa: float, form: str, gamma: float, nu: float,
    use_bias: bool = True,
) -> float:
    """Log-vraisemblance en validation croisée interne (une élection d'apprentissage laissée de
    côté à tour de rôle) des items de `train`, pour (κ, ν) donnés."""
    from vvs.model import error as er

    s = 0.0
    for held in train:
        tr_e = [e for e in train if e != held]
        tr = [ci for ci in items if ci.item.election in tr_e]
        te = [ci for ci in items if ci.item.election == held]
        # V_ref de l'élection laissée : celle des élections d'apprentissage du pli interne
        tr_s = scale_items(tr, vols, tr_e, kappa, form, gamma)
        te_s = scale_items(te, vols, tr_e, kappa, form, gamma)
        b_in, _ = er.stable_bias(tr_s) if use_bias else ({}, {})
        x = er._fit_params(tr_s, gamma, nu, b_in)
        s -= er._neg_objective(x, te_s, gamma, nu, b_in) - er._neg_objective(x, [], gamma, nu, b_in)
        s += item_log_jacobian(te, vols, tr_e, kappa, form)
    return s


def fit_kappa(
    items: list[CalItem],
    vols: dict[tuple[int, int], float],
    train,
    form: str,
    gamma: float,
    nus=(4.0, 8.0, 16.0, math.inf),
    grid=KAPPA_GRID,
    use_bias: bool = True,
) -> dict:
    """κ (et ν) maximisant la vraisemblance en validation croisée interne sur `train`."""
    train = list(train)
    grid = [float(k) for k in grid]
    scores = {}
    for nu in nus:
        for k in sorted(set(grid) | {0.0}):  # κ = 0 toujours évalué : référence du gain
            scores[(k, float(nu))] = cv_score(items, vols, train, k, form, gamma, nu, use_bias)
    (k_best, nu_best) = max((x for x in scores if x[0] in grid), key=scores.get)
    base = max(v for (k, _), v in scores.items() if k == 0.0)
    return {
        "kappa": k_best,
        "nu": nu_best,
        "form": form,
        "train": train,
        "gain_vs_kappa0": scores[(k_best, nu_best)] - base,
        "profile": {str(k): max(v for (kk, _), v in scores.items() if kk == k)
                    for k in sorted({k for k, _ in scores})},
    }


def fit_error_model(
    items: list[CalItem],
    vols: dict[tuple[int, int], float],
    train,
    kappa: float,
    form: str,
    gamma: float,
    nu: float,
    registry: Registry | None = None,
):
    """`ErrorModel` recalibré sur les items de `train` mis à l'échelle par (V / V_ref)^κ."""
    from vvs.model.error import ErrorModel

    train = list(train)
    sub = [ci for ci in items if ci.item.election in train]
    scaled = scale_items(sub, vols, train, kappa, form, gamma)
    return ErrorModel.fit(elections=train, exclude=(), nu=nu, registry=registry, gamma=gamma,
                          items=scaled)


def scale_for(
    election: int,
    horizon: int,
    vols: dict[tuple[int, int], float],
    train,
    kappa: float,
    form: str,
    bloc_vols: dict[tuple[int, int], dict[str, float]] | None = None,
    window_weeks: int = PRIMARY_WINDOW,
    registry: Registry | None = None,
) -> VolatilityScale:
    """Échelle d'une prévision à vote − `horizon` jours. `bloc_vols` : volatilités par bloc des
    élections (même clé), pour une échelle par bloc ; None : échelle globale."""
    v = volatility_at_horizon(election, horizon, window_weeks, registry)
    ref = reference(vols, train)
    r = ratio(v.campaign, interp_reference(ref, horizon))
    bloc_ratio = {}
    if bloc_vols is not None:
        for b in BLOCS:
            bref = reference({k: d.get(b, math.nan) for k, d in bloc_vols.items()}, train)
            bloc_ratio[b] = ratio(v.by_bloc.get(b, math.nan), interp_reference(bref, horizon))
    return VolatilityScale(kappa=kappa, ratio=r, form=form, bloc_ratio=bloc_ratio)
