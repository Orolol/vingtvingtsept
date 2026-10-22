"""Troupeau des instituts (herding) et échelle de l'erreur de projection (docs/04-modele.md,
section 2).

Idée : quand les instituts « se regardent » en fin de campagne (choix de redressement, de
filtre de participation ou de publication qui rapprochent leurs chiffres de la moyenne), leurs
sondages se ressemblent plus que ne le permet l'échantillonnage. L'agrégat devient alors moins
fiable que sa précision apparente : les erreurs des instituts sont corrélées. Silver (2014,
« Here's Proof Some Pollsters Are Putting A Thumb On The Scale ») et Gelman comparent la
dispersion observée entre sondages à celle qu'impose le seul tirage aléatoire aux effectifs
déclarés : un rapport sous 1 est suspect, puisque les effets de plan, les effets maison et les
sous-échantillons d'exprimés devraient le porter au-dessus de 1.

Dispersion relative (`dispersion`) sur une fenêtre de jours finissant à une date :

1. un sondage par institut, le dernier terminé dans la fenêtre (à fin de terrain égale, le plus
   grand effectif) ; l'hypothèse la plus proche de la configuration
   (`configurations.closest_hypothesis` ; scrutins de liste : hypothèse de
   `error_lists._chosen_polls`), écartée si les candidats hors configuration y pèsent plus de
   `MAX_EXTRA_SHARE` points ;
2. unités communes : candidats de la configuration testés par au moins `COMMON_FRACTION` des
   sondages retenus ; les sondages qui n'en testent pas un sont écartés ; parts rapportées au
   total des unités communes ; blocs = sommes des parts de leurs candidats ;
3. par unité u : variance résiduelle des parts entre instituts autour d'une tendance linéaire
   dans le temps (milieu du terrain), RSS / (m − 2), et variance attendue sous tirage aléatoire,
   Σ_i σ²_iu (1 − h_ii) / (m − 2) (h_ii : levier de la régression, ce qui rend le rapport
   d'espérance exactement 1 sous l'hypothèse nulle, effectifs inégaux compris), avec
   σ²_iu = p̄_u (1 − p̄_u) / n_i + δ_i² / 12 : tirage aléatoire simple à l'effectif déclaré n_i,
   plus l'arrondi publié (pas δ_i de 1, 0,5 ou 0,1 point, lu sur les parts du sondage) ;
   `detrend=False` : écart à la moyenne, RSS / (m − 1) et h_ii = 1/m ;
4. R_u = variance observée / variance attendue ; dispersion relative D = moyenne des R_u des
   candidats dont la part moyenne est d'au moins `MIN_SHARE` (χ² regroupé) ; D_bloc idem sur les
   blocs ; au moins `MIN_POLLSTERS` instituts, sinon indéfinie ; écart type de log D par
   jackknife sur les instituts.

Effectifs (`sample_size`) : absents en 2002, 2007, 2012, aux européennes 2014 et aux
législatives 2024. Imputation (`imputed_n`) : médiane de l'institut dans le scrutin, sinon dans
le scrutin le plus proche à au plus deux ans (législatives 2024 : européennes 2024), sinon
`DEFAULT_N` = 1 000, l'effectif courant des enquêtes téléphoniques de 2002-2012. Le niveau de D
de ces scrutins dépend de cette hypothèse (D ∝ n) ; sa trajectoire dans la campagne n'en dépend
pas.

Échelle (`HerdingScale`, attribut `ErrorModel.herd_scale`, None par défaut ; posée au score par
`HerdingFit.scale_at`, `ErrorModel.herding` étant posé par `ErrorModel.fit(herding=…)`) :

    s = clip((D_ref(τ) / D(τ))^κ, S_BOUNDS)

D(τ) dispersion relative des 30 jours précédant la date de prévision, D_ref(τ) sa moyenne
géométrique sur les élections d'apprentissage au même horizon (interpolée en log h), κ estimé
en validation croisée interne (laisser une élection d'apprentissage de côté), sur une grille
bornée (`KAPPA_GRID`). κ > 0 : une campagne plus resserrée que d'habitude élargit η. Deux
formes : `total` (toute la variance de η × s², c'est l'échelle λ de l'élection) et `final`
(seule l'erreur finale σ²_final + ρ/p × s², le mouvement reste inchangé).

Calibration (`fit_scaled_items`, appelée par `ErrorModel.fit(herding=…)`) : chaque item de
calibration de η est rapporté à la date de prévision qui l'a produit ; l'item d'erreur finale
(présidentielles et scrutins de liste) prend la dispersion de la fenêtre finale de 14 jours
(clé d'horizon `FINAL_HORIZON` = 1), l'item de mouvement à l'horizon H celle des 30 jours avant
vote − H (forme `total` seulement : la forme `final` laisse les mouvements inchangés). Mise à
l'échelle exacte en divisant les log-erreurs par s, jacobien ajouté à la vraisemblance des
items laissés de côté.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import Registry, default_registry

log = logging.getLogger(__name__)

DEFAULT_N = 1000
IMPUTE_MAX_YEARS = 2
FINAL_HORIZON = 1  # clé de la fenêtre finale (veille du vote, 14 jours)
FINAL_DAYS = 14
WINDOW_DAYS = 30
# horizons où la dispersion est tabulée (1 : fenêtre finale) ; ceux du backtest et des items
# de mouvement de η (error.MOVE_HORIZONS)
HORIZONS = (1, 7, 30, 60, 90, 120, 150, 180, 240)
MIN_POLLSTERS = 4
MIN_SHARE = 0.02
COMMON_FRACTION = 0.75
MAX_EXTRA_SHARE = 3.0  # points
KAPPA_GRID = tuple(float(x) for x in np.round(np.arange(-1.0, 2.0001, 0.25), 3))
NU_GRID = (4.0, 8.0, 16.0, math.inf)
S_BOUNDS = (0.5, 2.0)
FORMS = ("total", "final")
DEFAULT_FORM = "total"
PRESIDENTIAL = (2002, 2007, 2012, 2017, 2022)


# --- sondages d'une fenêtre


@dataclass
class PollPoint:
    """Un sondage d'une fenêtre : hypothèse retenue, parts en points."""

    poll_id: str
    pollster: str
    end: dt.date
    mid: float  # jours avant la fin de la fenêtre (milieu du terrain)
    n: float
    n_imputed: bool
    step: float  # pas d'arrondi des parts publiées (points)
    shares: dict[str, float]


def rounding_step(values) -> float:
    """Pas d'arrondi des parts d'un sondage : 1, 0,5 ou 0,1 point."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return 0.1
    if np.allclose(x, np.round(x), atol=1e-6):
        return 1.0
    if np.allclose(2 * x, np.round(2 * x), atol=1e-6):
        return 0.5
    return 0.1


_N_TABLE: dict[tuple[str, str], float] | None = None


def _election_year(election) -> int:
    from vvs.model.error_lists import election_year

    return election_year(election)


def n_table() -> dict[tuple[str, str], float]:
    """Médiane des effectifs déclarés par (scrutin, institut), présidentielles et listes."""
    global _N_TABLE
    if _N_TABLE is None:
        out = {}
        for f in ("polls.parquet", "polls_lists.parquet"):
            p = pl.read_parquet(paths.CURATED / f).filter(pl.col("sample_size").is_not_null())
            g = p.group_by(pl.col("election").cast(pl.Utf8), "pollster").agg(
                pl.col("sample_size").median().alias("n"))
            out |= {(e, q): float(n) for e, q, n in g.iter_rows()}
        _N_TABLE = out
    return _N_TABLE


def imputed_n(election, pollster: str) -> float:
    """Effectif d'un sondage sans `sample_size` (voir le docstring du module)."""
    tab = n_table()
    key = str(election)
    if (key, pollster) in tab:
        return tab[(key, pollster)]
    year = _election_year(election)
    best = None
    for (e, q), n in tab.items():
        if q != pollster:
            continue
        d = abs(_election_year(e) - year)
        if d <= IMPUTE_MAX_YEARS and (best is None or d < best[0]):
            best = (d, n)
    return best[1] if best else float(DEFAULT_N)


def one_per_pollster(points: list[PollPoint]) -> list[PollPoint]:
    """Le dernier sondage de chaque institut (à fin de terrain égale, le plus grand effectif)."""
    best: dict[str, PollPoint] = {}
    for p in points:
        cur = best.get(p.pollster)
        if cur is None or (p.end, p.n, p.poll_id) > (cur.end, cur.n, cur.poll_id):
            best[p.pollster] = p
    return sorted(best.values(), key=lambda p: (p.end, p.pollster))


def select_points(points: list[PollPoint], units: list[str],
                  frac: float = COMMON_FRACTION) -> tuple[list[PollPoint], list[str]]:
    """(un sondage par institut, unités communes) d'une fenêtre.

    Unités communes : testées par au moins `frac` des derniers sondages des instituts. Sondage
    d'un institut : son dernier qui teste toutes les unités communes (le précédent si le
    dernier en omet une)."""
    last = one_per_pollster(points)
    if not last:
        return [], []
    common = [u for u in units
              if sum(p.shares.get(u) is not None for p in last) >= max(frac * len(last), 1)]
    full = [p for p in points if all(p.shares.get(u) is not None for u in common)]
    return one_per_pollster(full), common


def presidential_points(
    polls: pl.DataFrame, election: int, end: dt.date, days: int, config: list[str]
) -> list[PollPoint]:
    """Sondages d'une présidentielle terminés dans ]end − days, end] (table de `load_polls`)."""
    from vvs.eval.configurations import closest_hypothesis

    df = polls.filter((pl.col("fieldwork_end") <= end)
                      & (pl.col("fieldwork_end") > end - dt.timedelta(days=days)))
    by_poll: dict[str, dict] = {}
    for pid, q, start, stop, n, hid, cid, share in df.select(
        "poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size", "hypothesis_id",
        "candidate_id", "share",
    ).iter_rows():
        p = by_poll.setdefault(pid, {"pollster": q, "start": start, "end": stop, "n": n,
                                     "hyps": {}})
        p["hyps"].setdefault(hid, {})[cid] = share
    target = set(config)
    out = []
    for pid, p in by_poll.items():
        hid, _ = closest_hypothesis(p["hyps"], list(config))
        sh = p["hyps"][hid]
        extra = sum(v for c, v in sh.items() if c not in target)
        if extra > MAX_EXTRA_SHARE:
            continue
        out.append(_point(pid, p["pollster"], p["start"], p["end"], p["n"], election, end,
                          {c: v for c, v in sh.items() if c in target}, sh.values()))
    return out


def _point(pid, pollster, start, stop, n, election, end, shares, all_values) -> PollPoint:
    mid = start + (stop - start) / 2 if start is not None else stop
    imputed = n is None
    return PollPoint(
        poll_id=pid, pollster=pollster, end=stop, mid=float((end - mid).days),
        n=float(imputed_n(election, pollster) if imputed else n), n_imputed=imputed,
        step=rounding_step(list(all_values)), shares=dict(shares),
    )


# --- dispersion relative


@dataclass
class Dispersion:
    D: float  # candidats : moyenne des rapports observé / attendu (parts ≥ MIN_SHARE)
    D_bloc: float
    m: int  # instituts
    se_log_D: float  # jackknife sur les instituts
    by_unit: dict[str, float] = field(default_factory=dict)
    by_bloc: dict[str, float] = field(default_factory=dict)
    sd_points: dict[str, float] = field(default_factory=dict)  # écart type résiduel observé
    sd_expected: dict[str, float] = field(default_factory=dict)  # attendu sous tirage aléatoire
    shares: dict[str, float] = field(default_factory=dict)  # part moyenne (fractions)
    pollsters: list[str] = field(default_factory=list)
    n_imputed: float = 0.0  # fraction des sondages à effectif imputé
    n_mean: float = math.nan

    def to_dict(self) -> dict:
        return asdict(self)


def _nan_dispersion(m: int = 0) -> Dispersion:
    return Dispersion(math.nan, math.nan, m, math.nan)


def _ratios(X: np.ndarray, V: np.ndarray, t: np.ndarray, detrend: bool) -> tuple:
    """(variance observée (U,), attendue (U,)) : résidus d'une régression sur [1, t] (ou sur la
    constante) ; X (m, U) parts, V (m, U) variances attendues par sondage."""
    m = X.shape[0]
    A = np.column_stack([np.ones(m), t - t.mean()]) if detrend else np.ones((m, 1))
    if detrend and np.ptp(t) == 0:
        A = np.ones((m, 1))
    k = A.shape[1]
    H = A @ np.linalg.pinv(A)
    lev = np.clip(np.diag(H), 0.0, 1.0)
    R = X - H @ X
    obs = (R**2).sum(0) / (m - k)
    exp = (V * (1 - lev)[:, None]).sum(0) / (m - k)
    return obs, exp


def dispersion(
    points: list[PollPoint],
    units: list[str],
    blocs: list[str],
    detrend: bool = True,
    min_share: float = MIN_SHARE,
    jackknife: bool = True,
) -> Dispersion:
    """Dispersion relative des sondages d'une fenêtre (voir le module).

    `points` : sondages de la fenêtre, plusieurs par institut possibles (`select_points` en
    garde un) ; `units`, `blocs` : configuration et blocs alignés."""
    pts, common = select_points(points, list(units))
    m = len(pts)
    if m < MIN_POLLSTERS or len(common) < 2:
        return _nan_dispersion(m)
    bloc_of = dict(zip(units, blocs))
    S = np.array([[p.shares[u] for u in common] for p in pts], dtype=float)
    S = S / S.sum(1, keepdims=True)
    ub = sorted({bloc_of[u] for u in common})
    B = np.column_stack([S[:, [bloc_of[u] == b for u in common]].sum(1) for b in ub])
    n = np.array([p.n for p in pts])
    rnd = np.array([(p.step / 100) ** 2 / 12 for p in pts])
    t = np.array([p.mid for p in pts])

    def stats(idx):
        out = []
        for X in (S[idx], B[idx]):
            pbar = X.mean(0)
            V = pbar * (1 - pbar) / n[idx, None] + rnd[idx, None]
            obs, exp = _ratios(X, V, t[idx], detrend)
            out.append((obs, exp, pbar))
        return out

    def agg(obs, exp, pbar):
        sel = pbar >= min_share
        return float(np.mean(obs[sel] / exp[sel])) if sel.sum() >= 2 else math.nan

    (oc, ec, pc), (ob, eb, pb) = stats(np.arange(m))
    D, Db = agg(oc, ec, pc), agg(ob, eb, pb)
    se = math.nan
    if jackknife and m - 1 >= MIN_POLLSTERS - 1 and np.isfinite(D) and D > 0:
        vals = []
        for i in range(m):
            idx = np.array([j for j in range(m) if j != i])
            (o, e_, p_), _ = stats(idx)
            d = agg(o, e_, p_)
            if np.isfinite(d) and d > 0:
                vals.append(math.log(d))
        if len(vals) == m:
            v = np.array(vals)
            se = float(math.sqrt((m - 1) / m * ((v - v.mean()) ** 2).sum()))
    return Dispersion(
        D=D,
        D_bloc=Db,
        m=m,
        se_log_D=se,
        by_unit={u: float(o / e) for u, o, e in zip(common, oc, ec)},
        by_bloc={b: float(o / e) for b, o, e in zip(ub, ob, eb)},
        sd_points={u: float(100 * math.sqrt(o)) for u, o in zip(common, oc)}
        | {f"bloc:{b}": float(100 * math.sqrt(o)) for b, o in zip(ub, ob)},
        sd_expected={u: float(100 * math.sqrt(e)) for u, e in zip(common, ec)}
        | {f"bloc:{b}": float(100 * math.sqrt(e)) for b, e in zip(ub, eb)},
        shares={u: float(x) for u, x in zip(common, pc)} | {f"bloc:{b}": float(x)
                                                            for b, x in zip(ub, pb)},
        pollsters=[p.pollster for p in pts],
        n_imputed=float(np.mean([p.n_imputed for p in pts])),
        n_mean=float(n.mean()),
    )


# --- dispersion d'une élection à un horizon


def window_days(horizon: int) -> int:
    return FINAL_DAYS if int(horizon) == FINAL_HORIZON else WINDOW_DAYS


def presidential_dispersion(
    election: int,
    asof: dt.date,
    config: list[str],
    blocs: list[str],
    polls: pl.DataFrame,
    days: int = WINDOW_DAYS,
    detrend: bool = True,
) -> Dispersion:
    """Dispersion relative des sondages visibles à `asof`, fenêtre de `days` jours."""
    pts = presidential_points(polls, election, asof, days, config)
    return dispersion(pts, config, blocs, detrend=detrend)


def list_dispersion(key: str, asof: dt.date, days: int, detrend: bool = True) -> Dispersion:
    """Dispersion relative d'un scrutin de liste (unités de résultat, `error_lists`)."""
    from vvs.curate import lists
    from vvs.model.error_lists import _chosen_polls

    reg = lists.load_lists()
    polls = lists.polls_with_rolling(pl.read_parquet(paths.CURATED / "polls_lists.parquet"))
    pr = pl.read_parquet(paths.CURATED / "poll_results_lists.parquet")
    res = pl.read_parquet(paths.CURATED / "results_lists_by_bloc.parquet").filter(
        (pl.col("level") == "list") & (pl.col("round") == 1) & (pl.col("election") == key))
    official = dict(res.select("unit", "share").iter_rows())
    chosen = _chosen_polls(key, reg, polls, pr, official)
    pts = []
    for p in chosen:
        if not (asof - dt.timedelta(days=days) < p["fieldwork_end"] <= asof):
            continue
        pts.append(_point(p["poll_id"], p["pollster"], p["fieldwork_start"], p["fieldwork_end"],
                          p["sample_size"], key, asof, p["shares"], p["shares"].values()))
    units = sorted(official, key=lambda u: -official[u])
    blocs = [reg.bloc_of(key, u) for u in units]
    return dispersion(pts, units, blocs, detrend=detrend)


def vote_date(election, reg: Registry | None = None) -> dt.date:
    if isinstance(election, str):
        from vvs.curate import lists

        return lists.vote_date(election)
    from vvs.model.data import election_date

    return election_date(reg or default_registry(), election)


_CACHE: dict[tuple, Dispersion] = {}


def herding_at(
    election, horizon: int, days: int | None = None, detrend: bool = True,
    registry: Registry | None = None,
) -> Dispersion:
    """Dispersion relative à vote − `horizon` jours (`FINAL_HORIZON` : veille du vote, fenêtre
    finale de 14 jours), configuration connue à la date (présidentielles), unités de résultat
    (scrutins de liste). Mis en cache."""
    days = window_days(horizon) if days is None else days
    key = (election, int(horizon), int(days), bool(detrend))
    if key in _CACHE:
        return _CACHE[key]
    asof = vote_date(election, registry) - dt.timedelta(days=int(horizon))
    if isinstance(election, str):
        d = list_dispersion(election, asof, days, detrend)
    else:
        from vvs.eval.configurations import known_configuration
        from vvs.model.data import load_polls

        reg = registry or default_registry()
        polls = load_polls(election, asof)
        try:
            config = known_configuration(election, asof, reg, polls=polls).candidates
        except ValueError:
            d = _nan_dispersion()
        else:
            blocs = [reg.bloc_of(c, election) for c in config]
            d = presidential_dispersion(election, asof, config, blocs, polls, days, detrend)
    _CACHE[key] = d
    return d


def d_values(elections, horizons=HORIZONS, registry: Registry | None = None) -> dict:
    """{(élection, horizon): D} (scrutins de liste : fenêtre finale seulement)."""
    out = {}
    for e in elections:
        for h in horizons:
            if isinstance(e, str) and h != FINAL_HORIZON:
                continue
            out[(e, int(h))] = herding_at(e, h, registry=registry).D
    return out


# --- référence et échelle


def reference(dvals: dict, elections, horizons=HORIZONS) -> dict[int, float]:
    """D_ref(h) : moyenne géométrique de D sur `elections` (valeurs définies), par horizon."""
    out = {}
    for h in horizons:
        x = [dvals.get((e, int(h)), math.nan) for e in elections]
        x = [v for v in x if np.isfinite(v) and v > 0]
        if x:
            out[int(h)] = float(np.exp(np.mean(np.log(x))))
    return out


def interp_reference(ref: dict[int, float], horizon: float) -> float:
    """D_ref à un horizon quelconque (≥ 7 jours) : interpolation de log D_ref en log h, sur les
    horizons de fenêtre de 30 jours ; l'horizon `FINAL_HORIZON` n'est rendu que s'il est demandé
    exactement."""
    if round(horizon) == FINAL_HORIZON and FINAL_HORIZON in ref:
        return ref[FINAL_HORIZON]
    hs = np.array(sorted(h for h in ref if h != FINAL_HORIZON), dtype=float)
    if hs.size == 0:
        return math.nan
    lv = np.log([ref[int(h)] for h in hs])
    return float(np.exp(np.interp(math.log(max(horizon, 1.0)), np.log(hs), lv)))


def scale_factor(d_obs: float, d_ref: float, kappa: float) -> float:
    """s = clip((D_ref / D)^κ, S_BOUNDS) ; 1 si l'un des deux est indéfini (modèle actuel)."""
    if kappa == 0 or not (np.isfinite(d_obs) and np.isfinite(d_ref)) or d_obs <= 0 or d_ref <= 0:
        return 1.0
    return float(np.clip((d_ref / d_obs) ** kappa, *S_BOUNDS))


@dataclass
class HerdingScale:
    """Échelle de l'écart type de η à la date de prévision (`ErrorModel.herd_scale`)."""

    s: float
    form: str = DEFAULT_FORM
    d_obs: float = math.nan
    d_ref: float = math.nan
    kappa: float = 0.0

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"forme inconnue : {self.form}")

    def apply(self, vb, vc, vb_final, vc_final) -> tuple:
        """Variances (de bloc, propre) mises à l'échelle ; `*_final` : leur part d'erreur finale
        (σ²_final, et ρ/p pour la composante propre)."""
        s2 = self.s**2
        if self.form == "total":
            return vb * s2, vc * s2
        return vb + (s2 - 1) * vb_final, vc + (s2 - 1) * vc_final


@dataclass
class HerdingFit:
    """κ et référence estimés sur les élections d'apprentissage (`ErrorModel.herding`)."""

    kappa: float
    form: str
    nu: float
    d_ref: dict[int, float]
    train: list
    gain_vs_kappa0: float = 0.0
    profile: dict[str, float] = field(default_factory=dict)

    def scale(self, d_obs: float, horizon: float) -> HerdingScale:
        d_ref = interp_reference(self.d_ref, horizon)
        return HerdingScale(scale_factor(d_obs, d_ref, self.kappa), self.form, d_obs, d_ref,
                            self.kappa)

    def scale_at(self, election: int, asof: dt.date, horizon: float, config: list[str],
                 blocs: list[str], polls: pl.DataFrame) -> HerdingScale:
        """Échelle d'une prévision à `asof` (sondages visibles `polls`, configuration
        `config`), dispersion des `WINDOW_DAYS` derniers jours."""
        d = presidential_dispersion(election, asof, list(config), list(blocs), polls,
                                    WINDOW_DAYS)
        return self.scale(d.D, horizon)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


# --- calibration de κ


_ITEM_H: dict[int, dict[float, int]] = {}


def move_horizon(it, registry: Registry | None = None) -> int:
    """Horizon H de la date de prévision d'un item de mouvement de `error.move_items`."""
    from vvs.model import error as er

    e = it.election
    if e not in _ITEM_H:
        reg = registry or default_registry()
        _ITEM_H[e] = {}
        for H in er.MOVE_HORIZONS:
            for x in er.move_items(e, reg, (H,)):
                _ITEM_H[e][round(float(x.h), 6)] = int(H)
    try:
        return _ITEM_H[e][round(float(it.h), 6)]
    except KeyError:
        raise KeyError(f"item de mouvement {e} h = {it.h} : horizon introuvable") from None


def item_horizon(it, registry: Registry | None = None) -> int:
    return FINAL_HORIZON if it.kind == "final" else move_horizon(it, registry)


def _s_items(items, horizons, dvals, train, kappa, form) -> np.ndarray:
    ref = reference(dvals, train)
    s = np.ones(len(items))
    for i, (it, H) in enumerate(zip(items, horizons)):
        if form == "final" and it.kind != "final":
            continue
        s[i] = scale_factor(dvals.get((it.election, H), math.nan), ref.get(H, math.nan), kappa)
    return s


def scale_items(items, horizons, dvals, train, kappa, form) -> tuple[list, float]:
    """(items aux log-erreurs divisées par s, Σ poids · (−(n − 1) log s) : jacobien)."""
    s = _s_items(items, horizons, dvals, train, kappa, form)
    out = [it if si == 1.0 else replace(it, d=it.d / si) for it, si in zip(items, s)]
    jac = float(sum(-it.weight * (len(it.d) - 1) * math.log(si) for it, si in zip(items, s)))
    return out, jac


def cv_score(items, horizons, dvals, groups, kappa, form, gamma, nu, use_bias=True,
             x0s: dict | None = None) -> float:
    """Log-vraisemblance en validation croisée interne (un groupe laissé de côté à tour de
    rôle), dans l'espace des log-erreurs d'origine. `x0s` : points de départ de l'optimisation
    par groupe laissé (mis à jour en place : départ à chaud d'un κ voisin)."""
    from vvs.model import error as er

    s = 0.0
    for held in groups:
        tr_g = [g for g in groups if g != held]
        tr = [(it, H) for it, H in zip(items, horizons) if it.election != held]
        te = [(it, H) for it, H in zip(items, horizons) if it.election == held]
        if not te:
            continue
        tr_s, _ = scale_items([x[0] for x in tr], [x[1] for x in tr], dvals, tr_g, kappa, form)
        te_s, jac = scale_items([x[0] for x in te], [x[1] for x in te], dvals, tr_g, kappa,
                                form)
        b_in, _ = er.stable_bias(tr_s) if use_bias else ({}, {})
        x = er._fit_params(tr_s, gamma, nu, b_in,
                           x0=None if x0s is None else x0s.get(held))
        if x0s is not None:
            x0s[held] = x
        s -= er._neg_objective(x, te_s, gamma, nu, b_in) - er._neg_objective(x, [], gamma, nu,
                                                                            b_in)
        s += jac
    return s


def fit_kappa(items, horizons, dvals, groups, form, gamma, nus=NU_GRID, grid=KAPPA_GRID,
              use_bias=True) -> dict:
    """κ (et ν) maximisant la vraisemblance en validation croisée interne sur `groups`."""
    scores = {}
    for nu in nus:
        x0s: dict = {}
        for k in sorted(set(grid) | {0.0}):
            scores[(float(k), float(nu))] = cv_score(items, horizons, dvals, groups, k, form,
                                                     gamma, nu, use_bias, x0s)
    k_best, nu_best = max((x for x in scores if x[0] in grid), key=scores.get)
    base = max(v for (k, _), v in scores.items() if k == 0.0)
    return {
        "kappa": k_best,
        "nu": nu_best,
        "gain_vs_kappa0": scores[(k_best, nu_best)] - base,
        "profile": {str(k): max(v for (kk, _), v in scores.items() if kk == k)
                    for k in sorted({k for k, _ in scores})},
    }


def resolve_form(herding) -> str:
    form = DEFAULT_FORM if herding is True else str(herding)
    if form not in FORMS:
        raise ValueError(f"herding : forme inconnue {herding!r}")
    return form


def fit_scaled_items(items, groups, gamma, herding=True, use_bias=True,
                     registry: Registry | None = None, nus=NU_GRID,
                     grid=KAPPA_GRID) -> tuple[list, HerdingFit]:
    """Items de calibration de η mis à l'échelle par (D_ref / D)^κ, κ et ν choisis en
    validation croisée interne sur `groups` (élections d'apprentissage, scrutins de liste
    compris), D_ref sur tous les groupes. Appelé par `ErrorModel.fit(herding=…)`."""
    form = resolve_form(herding)
    groups = list(groups)
    horizons = [item_horizon(it, registry) for it in items]
    dvals = d_values(groups, registry=registry)
    k = fit_kappa(items, horizons, dvals, groups, form, gamma, nus, grid, use_bias)
    scaled, _ = scale_items(items, horizons, dvals, groups, k["kappa"], form)
    fit = HerdingFit(kappa=k["kappa"], form=form, nu=k["nu"], d_ref=reference(dvals, groups),
                     train=[str(g) for g in groups], gain_vs_kappa0=k["gain_vs_kappa0"],
                     profile=k["profile"])
    log.info("troupeau : κ = %.2f (%s), ν = %s, gain %.3f", fit.kappa, form, fit.nu,
             fit.gain_vs_kappa0)
    return scaled, fit
