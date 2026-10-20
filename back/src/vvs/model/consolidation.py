"""Vote utile de fin de campagne : consolidation tardive d'un camp vers son candidat en tête.

Dérive μ_c(τ) de la projection (docs/04-modele.md, § 2) :

    θ_{c,0} = θ_{c,τ} + μ_c(τ) + η_c(τ)

Idée. Dans les dernières semaines, une partie des électeurs des « petits » candidats d'un camp
se reporte sur le candidat du camp le mieux placé (Mélenchon 2017 et 2022 aux dépens de Hamon,
Jadot, Roussel ; Le Pen 2022 aux dépens de Zemmour ; Hollande 2012 aux dépens de Mélenchon).
Si ce transfert est en partie postérieur aux derniers sondages, les sondages à τ sous-estiment
le leader du camp et surestiment ses concurrents ; une prévision faite à τ peut l'anticiper.

Camps. Le vote utile se joue entre candidats proches, pas entre les blocs du registre :
gauche radicale et gauche forment un seul camp (`CAMP_OF`, Mélenchon face à Hamon, Jadot,
Glucksmann) ; centre, droite et extrême droite restent séparés ; `other` (Lassalle,
Cheminade…) n'entre dans aucun camp.

Forme (parts de référence p, fractions ; camp k de leader L, le plus haut de p dans le camp ;
F l'ensemble des autres candidats du camp, p_F = Σ_{j∈F} p_j) : une fraction φ des parts des
concurrents passe au leader,

    p'_L = p_L + φ · p_F,   p'_j = (1 − φ) · p_j   (j ∈ F),

soit, sur les log-parts (ce que reçoit `ErrorModel.sample(..., drift=…)`) :

    μ_L = log(1 + φ · p_F / p_L),   μ_j = log(1 − φ),   μ = 0 hors du camp.

La part du camp est conservée exactement aux parts de référence (softmax(θ + μ) somme des
parts du camp inchangée, les autres candidats intacts) ; appliquée à d'autres tirages que ceux
de la référence (médiane projetée dans le backtest), elle l'est à l'ordre près des écarts de
p_F / p_L entre tirages.

    φ(τ, m) = φ0 · w(τ) · g(m)
    w(τ) = (1 − min(τ, τ_max) / τ_max)^κ          nulle à 180 jours, 1 au jour du vote
    g(m) = (1 + e^{−m0/s}) / (1 + exp((|m| − m0) / s))   proximité à la qualification, g(0) = 1

m (points) est la marge de qualification du leader : sa part moins la deuxième plus haute part
des autres candidats (positive : il est qualifié dans la référence ; négative : il est derrière
le 2e). m0 = ∞ donne g = 1 (pas de modulation). w porte la convention retenue : à 180 jours le
leader du camp n'est pas stabilisé (2012 : Hollande au sortir de la primaire, puis Mélenchon) et
la dérive est nulle ; plus près du vote, φ est ce qui reste à venir de la consolidation. φ0 est
la valeur « pleine » (τ → 0).

Estimation (`fit_params`) : observations par couple camp × élection et horizon
(`couple_observations`) : à τ, moyenne des sondages des 14 jours précédant vote − τ
(`baselines.poll_mean`, configuration connue à la date, candidats officiels seulement, renormalisés
dans le camp), leader = candidat officiel du camp en tête ; au vote, parts officielles. La
fraction des concurrents f = 1 − p_L / p_camp passe de f_τ à f_vote ; le transfert observé
est y = f_τ − f_vote, le transfert prévu φ(τ, m) · f_τ. φ0 par moindres carrés (y sur
x = w(τ) g(m) f_τ), m0 sur une grille par validation croisée interne (laisser une élection
d'apprentissage de côté), κ fixé à 1 (sensibilité dans l'étude). Incertitude : rééchantillonnage
des couples avec remise (toutes leurs observations ensemble), `phi0_draws`. Horizons
d'estimation : 7, 30 et 90 jours (w(180) = 0). Le tout sans l'élection évaluée
(leave-one-election-out) dans le backtest.

Variante du backtest (`CONSOLIDATION_VARIANTS`, désactivée par défaut) : dérive calculée sur
la médiane des log-parts du postérieur à la date de prévision, φ0 tiré dans sa distribution de
rééchantillonnage (un tirage par ligne), ajoutée à η par `ErrorModel.sample(..., drift=…)`.
Étude et backtest : `back/studies/consolidation.py`.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from vvs.registry import Registry, default_registry

CAMP_OF = {
    "far_left": "left",
    "left": "left",
    "centre": "centre",
    "right": "right",
    "far_right": "far_right",
}
CAMPS = ("left", "centre", "right", "far_right")
TAU_MAX = 180.0
KAPPA = 1.0
PROX_SCALE = 2.0  # points
M0_GRID = (3.0, 5.0, 7.0, 9.0, 12.0, math.inf)
FIT_HORIZONS = (7, 30, 90)
STUDY_HORIZONS = (7, 30, 90, 180)
WINDOW_DAYS = 14
CALIBRATION_ELECTIONS = (2002, 2007, 2012, 2017, 2022)
PHI_MAX = 0.95  # borne de φ (jamais tout le camp au leader)


# --- forme


def camp_of(bloc: str | None) -> str | None:
    return CAMP_OF.get(bloc) if bloc is not None else None


def horizon_weight(tau, kappa: float = KAPPA, tau_max: float = TAU_MAX):
    """w(τ) : 1 au jour du vote, 0 à partir de `tau_max` jours."""
    t = np.clip(np.asarray(tau, dtype=float) / tau_max, 0.0, 1.0)
    return (1.0 - t) ** kappa


def proximity(margin_pts, m0: float = math.inf, scale: float = PROX_SCALE):
    """g(m) : 1 si m0 = ∞ ; sinon sigmoïde décroissante de |m| centrée en m0 (points),
    normalisée à g(0) = 1 (φ0 est alors le transfert d'un leader au seuil de qualification)."""
    m = np.abs(np.asarray(margin_pts, dtype=float))
    if math.isinf(m0):
        return np.ones_like(m)
    return (1.0 + math.exp(-m0 / scale)) / (1.0 + np.exp((m - m0) / scale))


def qualification_margin(shares: np.ndarray, i) -> np.ndarray:
    """Part de i moins la 2e plus haute part des autres (mêmes unités que `shares`).

    `shares` (n, K) ; `i` entier ou (n,) indices. Positive : i fait partie des deux premiers."""
    s = np.atleast_2d(np.asarray(shares, dtype=float))
    n, K = s.shape
    i = np.broadcast_to(np.asarray(i), (n,))
    if K < 3:
        return s[np.arange(n), i]
    others = s.copy()
    others[np.arange(n), i] = -np.inf
    return s[np.arange(n), i] - (-np.sort(-others, axis=1)[:, 1])


def camp_groups(blocs: list[str], camps: tuple[str, ...] | None = None) -> dict[str, list[int]]:
    """Indices des candidats par camp (camps d'au moins deux candidats)."""
    out: dict[str, list[int]] = {}
    for k, b in enumerate(blocs):
        c = camp_of(b)
        if c is None or (camps is not None and c not in camps):
            continue
        out.setdefault(c, []).append(k)
    return {c: ix for c, ix in out.items() if len(ix) >= 2}


def consolidation_phi(phi0, tau, margin_pts, m0: float = math.inf, kappa: float = KAPPA,
                      scale: float = PROX_SCALE):
    """φ(τ, m) = φ0 · w(τ) · g(m), broadcast."""
    return np.asarray(phi0, dtype=float) * horizon_weight(tau, kappa) * proximity(margin_pts, m0, scale)


def consolidation_drift(
    shares: np.ndarray,
    blocs: list[str],
    tau: float,
    phi0,
    m0: float = math.inf,
    kappa: float = KAPPA,
    scale: float = PROX_SCALE,
    camps: tuple[str, ...] | None = None,
    detail: list | None = None,
) -> np.ndarray:
    """Dérive μ (n, K) sur les log-parts, à partir des parts de référence `shares` ((K,) ou
    (n, K), fractions ou points : seules les proportions comptent) et de φ0 (scalaire ou (n,)).

    Leader de chaque camp et marge de qualification calculés ligne par ligne sur la référence.
    `detail` : liste à compléter d'un résumé par camp (première ligne de la référence)."""
    p = np.atleast_2d(np.asarray(shares, dtype=float))
    phi0 = np.asarray(phi0, dtype=float)
    n = max(p.shape[0], phi0.shape[0] if phi0.ndim else 1)
    K = p.shape[1]
    if len(blocs) != K:
        raise ValueError("shares et blocs de tailles incompatibles")
    p = np.broadcast_to(p / p.sum(1, keepdims=True), (n, K))
    phi0 = np.broadcast_to(phi0, (n,))
    mu = np.zeros((n, K))
    rows = np.arange(n)
    for camp, ix in camp_groups(list(blocs), camps).items():
        ix = np.array(ix)
        sub = p[:, ix]
        lead = ix[np.argmax(sub, axis=1)]  # (n,)
        pL = p[rows, lead]
        pF = sub.sum(1) - pL
        m = 100 * qualification_margin(p, lead)
        phi = consolidation_phi(phi0, tau, m, m0, kappa, scale)
        # bornes : φ < 1, et un φ négatif (anti-consolidation) ne vide pas le leader
        lo = -np.where(pF > 0, pL / np.maximum(pF, 1e-12), np.inf) * PHI_MAX
        phi = np.clip(phi, lo, PHI_MAX)
        phi = np.where(pF > 0, phi, 0.0)
        mu_lead = np.log1p(phi * pF / pL)
        mu_foll = np.log1p(-phi)
        for k in ix:
            mu[:, k] = np.where(lead == k, mu_lead, mu_foll)
        if detail is not None:
            detail.append({
                "camp": camp, "leader_index": int(lead[0]), "leader_share": float(pL[0]),
                "followers_share": float(pF[0]), "margin_pts": float(m[0]),
                "phi_median": float(np.median(phi)), "phi_q05": float(np.quantile(phi, 0.05)),
                "phi_q95": float(np.quantile(phi, 0.95)),
                "transfer_pts_median": float(100 * np.median(phi * pF)),
            })
    return mu


def apply_drift(shares: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Parts après la dérive (softmax(log p + μ)), pour les tests et l'étude."""
    x = np.log(np.atleast_2d(shares)) + mu
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


# --- observations historiques


@dataclass
class Observation:
    """Un couple camp × élection à un horizon : fraction des concurrents du leader dans le camp
    dans les sondages à τ (`f_tau`) et au vote (`f_vote`)."""

    election: int
    camp: str
    horizon: int  # 0 : 14 derniers jours (erreur finale)
    leader: str
    followers: list[str]
    leader_share: float  # points, sondages à τ, candidats officiels renormalisés
    camp_share: float
    f_tau: float
    f_vote: float
    margin_pts: float
    leader_vote: float
    camp_vote: float
    gap_first_follower: float  # points : leader − premier concurrent
    shares: dict[str, float] = field(default_factory=dict)  # points, à τ
    votes: dict[str, float] = field(default_factory=dict)  # points, au vote

    @property
    def y(self) -> float:
        """Transfert observé, en fraction du camp : f_τ − f_vote."""
        return self.f_tau - self.f_vote

    @property
    def phi(self) -> float:
        return self.y / self.f_tau if self.f_tau > 0 else float("nan")

    @property
    def key(self) -> str:
        return f"{self.election}-{self.camp}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["y"], d["phi"] = self.y, self.phi
        return d


def _official_results(reg: Registry, election: int) -> dict[str, float]:
    res = {r.candidate_id: r.share for r in reg.elections[election].round_(1).results}
    tot = sum(res.values())
    return {c: 100 * v / tot for c, v in res.items()}


def _bloc(reg: Registry, c: str, election: int) -> str:
    try:
        return reg.bloc_of(c, election)
    except KeyError:
        return "other"


def poll_shares(election: int, horizon: int, registry: Registry | None = None,
                polls=None) -> dict[str, float]:
    """Parts (points) des candidats officiels dans la moyenne des 14 jours avant vote − τ
    (τ = 0 : veille du vote, les 14 derniers jours), configuration connue à la date, ramenées
    aux candidats officiels comme au backtest (`configurations.official_map` : un candidat qui
    ne se présentera pas passe à l'officiel de son parti, à défaut de son bloc), candidats
    officiels sans part écartés."""
    from vvs.eval.baselines import poll_mean
    from vvs.eval.configurations import known_configuration, official_map
    from vvs.model.data import election_date, load_polls

    reg = registry or default_registry()
    vote = election_date(reg, election)
    polls = load_polls(election) if polls is None else polls
    official = [c.candidate_id for c in reg.elections[election].candidates]
    asof = vote - dt.timedelta(days=max(horizon, 1))
    cfg = official if horizon == 0 else known_configuration(election, asof, reg, polls).candidates
    pf = poll_mean(election, asof, cfg, days=WINDOW_DAYS, registry=reg, polls=polls)
    om = official_map(cfg, election, reg)
    x = om.apply(pf.shares[None, :])[0]
    sh = {c: float(v) for c, v, f in zip(om.official, x, om.forecast) if f and v > 0}
    tot = sum(sh.values())
    return {c: 100 * v / tot for c, v in sh.items()}


def observations_from_shares(election: int, horizon: int, shares: dict[str, float],
                             votes: dict[str, float], blocs: dict[str, str]) -> list[Observation]:
    """Couples camp × élection d'un jeu de parts (points) : camps d'au moins deux candidats
    sondés, leader = premier du camp dans `shares`."""
    out = []
    by_camp: dict[str, list[str]] = {}
    for c in shares:
        k = camp_of(blocs.get(c))
        if k is not None:
            by_camp.setdefault(k, []).append(c)
    ordered = sorted(shares, key=lambda c: -shares[c])
    for camp in CAMPS:
        cs = by_camp.get(camp, [])
        if len(cs) < 2:
            continue
        L = max(cs, key=lambda c: shares[c])
        B = sum(shares[c] for c in cs)
        rc = [c for c in votes if camp_of(blocs.get(c)) == camp]
        vB = sum(votes[c] for c in rc)
        others = [shares[c] for c in ordered if c != L]
        fol = sorted((c for c in cs if c != L), key=lambda c: -shares[c])
        out.append(Observation(
            election=election, camp=camp, horizon=horizon, leader=L, followers=fol,
            leader_share=shares[L], camp_share=B, f_tau=1 - shares[L] / B,
            f_vote=1 - votes[L] / vB, margin_pts=shares[L] - others[1],
            leader_vote=votes[L], camp_vote=vB, gap_first_follower=shares[L] - shares[fol[0]],
            shares={c: shares[c] for c in cs}, votes={c: votes[c] for c in rc},
        ))
    return out


_OBS_CACHE: dict[tuple, list[Observation]] = {}


def couple_observations(elections=CALIBRATION_ELECTIONS, horizons=(0, *STUDY_HORIZONS),
                        registry: Registry | None = None) -> list[Observation]:
    """Observations de tous les couples camp × élection aux horizons demandés (mise en cache)."""
    from vvs.model.data import load_polls

    reg = registry or default_registry()
    out = []
    for e in elections:
        votes = _official_results(reg, e)
        blocs = {c: _bloc(reg, c, e) for c in votes}
        polls = None
        for h in horizons:
            key = (e, h)
            if key not in _OBS_CACHE:
                polls = load_polls(e) if polls is None else polls
                sh = poll_shares(e, h, reg, polls)
                _OBS_CACHE[key] = observations_from_shares(e, h, sh, votes, blocs)
            out += _OBS_CACHE[key]
    return out


# --- estimation


@dataclass
class ConsolidationParams:
    phi0: float
    phi0_draws: np.ndarray
    m0: float = math.inf
    kappa: float = KAPPA
    scale: float = PROX_SCALE
    train: list[int] = field(default_factory=list)
    n_couples: int = 0
    n_obs: int = 0
    m0_scores: dict = field(default_factory=dict)

    def interval(self, q=(0.05, 0.5, 0.95)) -> list[float]:
        return [float(x) for x in np.quantile(self.phi0_draws, q)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phi0_draws"] = None
        d["phi0_interval"] = self.interval()
        d["m0"] = None if math.isinf(self.m0) else self.m0
        d["m0_scores"] = {str(k): v for k, v in self.m0_scores.items()}
        return d


def _design(obs: list[Observation], m0: float, kappa: float, scale: float):
    x = np.array([horizon_weight(o.horizon, kappa) * proximity(o.margin_pts, m0, scale) * o.f_tau
                  for o in obs])
    y = np.array([o.y for o in obs])
    return x, y


def _ls_phi0(obs, m0, kappa, scale) -> float:
    x, y = _design(obs, m0, kappa, scale)
    sxx = float((x * x).sum())
    return float((x * y).sum() / sxx) if sxx > 0 else 0.0


def _sse(obs, phi0, m0, kappa, scale) -> float:
    x, y = _design(obs, m0, kappa, scale)
    return float(((y - phi0 * x) ** 2).sum())


def select_m0(obs: list[Observation], kappa: float = KAPPA, scale: float = PROX_SCALE,
              grid=M0_GRID) -> tuple[float, dict]:
    """m0 par validation croisée interne (une élection d'apprentissage laissée de côté à la
    fois) : somme des erreurs quadratiques de prévision du transfert y. Égalité : m0 = ∞."""
    elections = sorted({o.election for o in obs})
    scores = {}
    for m0 in grid:
        s = 0.0
        for held in elections:
            tr = [o for o in obs if o.election != held]
            te = [o for o in obs if o.election == held]
            s += _sse(te, _ls_phi0(tr, m0, kappa, scale), m0, kappa, scale)
        scores[m0] = s
    best = min(scores, key=lambda m: (round(scores[m], 10), -m))
    return best, scores


def fit_params(
    obs: list[Observation],
    exclude=(),
    m0: float | str = "auto",
    kappa: float = KAPPA,
    scale: float = PROX_SCALE,
    horizons=FIT_HORIZONS,
    n_boot: int = 500,
    seed: int = 0,
) -> ConsolidationParams:
    """φ0 (et m0 si "auto") sur les observations aux `horizons`, sans les élections `exclude`.

    `phi0_draws` : rééchantillonnage des couples (avec remise), m0 fixé à sa valeur estimée."""
    excl = {int(e) for e in np.atleast_1d(exclude)} if exclude is not None else set()
    tr = [o for o in obs if o.election not in excl and o.horizon in horizons]
    scores = {}
    if m0 == "auto":
        m0, scores = select_m0(tr, kappa, scale)
    phi0 = _ls_phi0(tr, m0, kappa, scale)
    keys = sorted({o.key for o in tr})
    by_key = {k: [o for o in tr if o.key == k] for k in keys}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [o for i in pick for o in by_key[keys[i]]]
        draws.append(_ls_phi0(sample, m0, kappa, scale))
    return ConsolidationParams(
        phi0=phi0, phi0_draws=np.array(draws), m0=float(m0), kappa=kappa, scale=scale,
        train=sorted({o.election for o in tr}), n_couples=len(keys), n_obs=len(tr),
        m0_scores=scores,
    )


# --- variante du backtest


@dataclass(frozen=True)
class ConsolidationSpec:
    """Réglages de la variante. `camps` : camps concernés (None : tous) ; `m0` : "auto" (choisi
    par validation interne) ou valeur fixe ; `kappa` : forme de w(τ)."""

    camps: tuple[str, ...] | None = None
    m0: float | str = "auto"
    kappa: float = KAPPA
    n_boot: int = 500


# Variantes du backtest qui ajoutent la dérive de consolidation (mêmes ajustements que floor).
# `consolidation_left` : camp de gauche seul, exploratoire (choisi après lecture de l'étude,
# n'entre pas dans la règle d'admission).
CONSOLIDATION_VARIANTS: dict[str, ConsolidationSpec] = {
    "consolidation": ConsolidationSpec(),
    "consolidation_left": ConsolidationSpec(camps=("left",)),
}

_PARAMS: dict[tuple, ConsolidationParams] = {}


def params_for(exclude: int | None, spec: ConsolidationSpec | None = None,
               registry: Registry | None = None) -> ConsolidationParams:
    """Paramètres estimés sur 2002-2022 privées de `exclude` (mise en cache)."""
    spec = spec or ConsolidationSpec()
    key = (exclude, spec)
    if key not in _PARAMS:
        obs = couple_observations(registry=registry)
        if spec.camps is not None:
            obs = [o for o in obs if o.camp in spec.camps]
        _PARAMS[key] = fit_params(obs, exclude=() if exclude is None else (exclude,),
                                  m0=spec.m0, kappa=spec.kappa, n_boot=spec.n_boot)
    return _PARAMS[key]


def reference_shares(log_shares: np.ndarray) -> np.ndarray:
    """Parts de référence (K,) : médiane des log-parts des tirages, renormalisée."""
    x = np.median(np.atleast_2d(log_shares), axis=0)
    p = np.exp(x - x.max())
    return p / p.sum()


def variant_drift(
    variant: str,
    election: int,
    asof: dt.date,
    horizon_days: float,
    config: list[str],
    n: int,
    seed: int = 0,
    registry: Registry | None = None,
    log_shares: np.ndarray | None = None,
) -> np.ndarray | None:
    """Dérive de la variante de backtest `variant` : consolidation si `variant` en est une
    (`log_shares` : (S, K) log-parts du postérieur à asof, obligatoire), sinon celle des rebonds
    (`vvs.model.bounce.variant_drift`, None hors variantes de dérive). Générateur propre
    (graine [seed, election, horizon, 11]) : η reste celui de la référence."""
    spec = CONSOLIDATION_VARIANTS.get(variant)
    if spec is None:
        from vvs.model.bounce import variant_drift as bounce_drift

        return bounce_drift(variant, election, asof, horizon_days, config, n, seed, registry)
    if log_shares is None:
        raise ValueError("variante de consolidation : log_shares requis")
    reg = registry or default_registry()
    pr = params_for(election, spec, reg)
    blocs = [_bloc(reg, c, election) for c in config]
    rng = np.random.default_rng([seed, election, int(horizon_days), 11])
    phi0 = pr.phi0_draws[rng.integers(0, len(pr.phi0_draws), n)]
    return consolidation_drift(reference_shares(log_shares), blocs, horizon_days, phi0,
                               pr.m0, pr.kappa, pr.scale, spec.camps)
