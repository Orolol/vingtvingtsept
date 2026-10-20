"""Modèle d'erreur de projection η(τ) (docs/04-modele.md, section 2).

    θ_{c,0} = θ_{c,τ} + μ_{b(c)} + η_c(τ)

θ est le soutien latent sur l'échelle logit multinomiale (parts = softmax de θ sur la
configuration) ; τ l'âge de l'estimation en jours avant le vote ; μ un biais de bloc, nul sauf
s'il est stable sur les élections d'apprentissage ; η l'erreur de projection.

Pourquoi l'échelle logit et pas les parts. Sur les élections 2002-2022, l'erreur de l'agrégat
final (14 derniers jours) contre le résultat a une racine de l'erreur quadratique de 0,4 point
pour les candidats sous 2 %, 0,8 entre 2 et 6 %, 2 à 3 au-delà : en points, l'erreur croît
presque proportionnellement à la part. En logarithme de la part, elle vaut 0,30, 0,21, 0,32 et
0,10 : beaucoup plus homogène. L'échelle logit garde aussi les parts positives et sommant à 100
sans troncature, et c'est celle du modèle d'agrégation. Le reste d'hétérogénéité (petits
candidats plus bruités en relatif, gros candidats moins) est pris par un terme ρ/p.

Covariance de η pour une configuration de candidats c ∈ K de blocs b(c) et de parts p_c :

    η_c = λ · (u_{b(c)} + v_c)
    u_b ~ N(0, w_b · (σ²_Bf + a_B · g(h)))          composante de bloc (corrélation intra-bloc)
    v_c ~ N(0, σ²_Cf + a_C · g(h) + ρ / p_c)        composante propre du candidat
    g(h) = (h / 90)^γ,  h = max(τ − âge de l'agrégat final, 0)
    λ² ~ InvGamma(ν/2, ν/2)                          échelle commune à l'élection

- (b) erreur de l'agrégat final contre le résultat : σ_Bf, σ_Cf, ρ ;
- (a) mouvement entre l'agrégat à l'horizon et l'agrégat final : a_B, a_C, croissant avec h,
  forme γ estimée sur Jennings–Wlezien (présidentielles, premier tour, 45 pays, parts
  normalisées, sans la France de 1965 et 1969) ;
- w_b : variance relative de chaque bloc, rétrécie vers 1 (prior log w_b ~ N(0, 0,5²)) ;
- λ : composante commune à l'élection. Un décalage additif commun à tous les candidats ne
  change pas les parts (softmax) ; c'est d'ailleurs pourquoi l'écart type de l'erreur moyenne
  par élection est nul dans Jennings–Wlezien une fois les parts normalisées. Ce qui est commun
  à une élection est l'ampleur des erreurs : certaines campagnes bougent beaucoup (2017, 2022),
  d'autres peu. On l'écrit comme une échelle aléatoire partagée par tous les candidats d'un
  tirage, ce qui fait de η une loi de Student multivariée à ν degrés de liberté ; ν est choisi
  par validation croisée interne (laisser une élection d'apprentissage de côté) sur une grille.

La corrélation intra-bloc est positive par construction : w_b(σ²_Bf + a_B g) / (variance
totale). Une corrélation entre blocs voisins (gauche et gauche radicale) n'est pas modélisée :
avec cinq élections elle n'est pas estimable ; la contrainte de somme des parts induit déjà des
corrélations négatives entre blocs.

Estimation (`ErrorModel.fit`) : maximum de vraisemblance pénalisé, élection par élection, sur
des contrastes de log-parts (différences au dernier candidat), invariants au décalage commun.
Données, par élection d'apprentissage :

- erreur finale : log(résultat) − log(moyenne des sondages des 14 derniers jours), candidats
  officiels (`poll_errors.parquet`, méthode `mean_14d`, parts renormalisées sur les candidats
  présents) ;
- mouvements : log(moyenne des 14 derniers jours) − log(moyenne des 30 jours avant l'horizon),
  pour h sur une grille de 30 à 240 jours, avec la configuration connue à chaque date
  ramenée aux candidats officiels (`vvs.eval.configurations`). Les horizons d'une même élection
  ne sont pas indépendants (marche emboîtée) : chacun pèse 1/(nombre d'horizons), de sorte que
  la courbe de mouvement d'une élection compte autant que son erreur finale.

Biais de bloc μ_b : admis seulement si l'erreur finale du bloc (log résultat − log agrégat) a le
même signe sur toutes les élections d'apprentissage (au moins trois), puis rétréci vers 0 par un
prior N(0, 0,05²) ; nul sinon.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import BLOCS, Registry, default_registry

log = logging.getLogger(__name__)

H_REF = 90.0  # horizon de référence de a_B, a_C (jours)
MOVE_HORIZONS = (30, 60, 90, 120, 150, 180, 240)
MOVE_WINDOW_DAYS = 30
FINAL_WINDOW_DAYS = 14
NU_GRID = (4.0, 8.0, 16.0, math.inf)
BIAS_PRIOR_SD = 0.05
BLOC_WEIGHT_PRIOR_SD = 0.5
JW_EXCLUDE = ("FR-1965-12-05", "FR-1969-06-01")
JW_HORIZONS = (7, 14, 21, 30, 45, 60, 75, 90, 120, 150, 180, 240)
JW_ELECTION_ID = {
    2002: "FR-2002-04-21",
    2007: "FR-2007-04-22",
    2012: "FR-2012-04-22",
    2017: "FR-2017-04-23",
}


# --- données de calibration


@dataclass
class Item:
    """Un vecteur d'erreurs en log-parts sur des candidats officiels d'une élection."""

    election: int
    kind: str  # final ou move
    h: float  # jours entre l'estimation et l'agrégat final (0 pour final)
    units: list[str]
    blocs: list[str]
    p: np.ndarray  # parts de référence (fractions) pour le terme ρ/p
    d: np.ndarray  # log-erreurs
    weight: float = 1.0
    final_age: float = 0.0


def _official(reg: Registry, election: int) -> tuple[list[str], np.ndarray, list[str]]:
    e = reg.elections[election]
    res = {r.candidate_id: r.share for r in e.round_(1).results}
    ids = [c.candidate_id for c in e.candidates]
    shares = np.array([res[c] for c in ids]) / 100
    return ids, shares / shares.sum(), [reg.bloc_of(c, election) for c in ids]


def final_item(election: int, reg: Registry, errors: pl.DataFrame | None = None) -> Item:
    """Erreur finale : résultat contre moyenne des 14 derniers jours (`poll_errors`)."""
    if errors is None:
        errors = pl.read_parquet(paths.CURATED / "poll_errors.parquet")
    ids, res, blocs = _official(reg, election)
    sub = errors.filter(
        (pl.col("election") == election)
        & (pl.col("round") == 1)
        & (pl.col("method") == "mean_14d")
        & (pl.col("level") == "candidate")
    )
    poll = dict(zip(sub["unit"].to_list(), sub["poll_share"].to_list()))
    keep = [i for i, c in enumerate(ids) if poll.get(c, 0) > 0]
    p = np.array([poll[ids[i]] for i in keep])
    p = p / p.sum()
    r = res[keep] / res[keep].sum()
    return Item(
        election,
        "final",
        0.0,
        [ids[i] for i in keep],
        [blocs[i] for i in keep],
        p,
        np.log(r) - np.log(p),
        final_age=FINAL_WINDOW_DAYS / 2,
    )


def move_items(
    election: int, reg: Registry, horizons=MOVE_HORIZONS
) -> list[Item]:
    """Mouvements entre la moyenne des 30 jours avant l'horizon et celle des 14 derniers jours.

    Configuration connue à chaque date (`known_configuration`), parts ramenées aux candidats
    officiels (`official_map`). h = âge moyen des sondages de la fenêtre − âge moyen de ceux de
    l'agrégat final."""
    from vvs.eval.baselines import poll_mean
    from vvs.eval.configurations import known_configuration, official_map
    from vvs.model.data import election_date

    vote = election_date(reg, election)
    ids, _, blocs = _official(reg, election)
    fin = poll_mean(election, vote - dt.timedelta(days=1), ids, days=FINAL_WINDOW_DAYS, registry=reg)
    fin_sh = fin.shares
    items = []
    for h in horizons:
        asof = vote - dt.timedelta(days=h)
        try:
            known = known_configuration(election, asof, reg)
        except ValueError:
            continue
        pf = poll_mean(election, asof, known.candidates, days=MOVE_WINDOW_DAYS, registry=reg)
        if pf.n_polls == 0:
            continue
        m = official_map(known.candidates, election, reg)
        sh = m.apply(pf.shares[None, :])[0]
        keep = [i for i in range(len(ids)) if m.forecast[i] and sh[i] > 0 and fin_sh[i] > 0]
        if len(keep) < 3:
            continue
        a = sh[keep] / sh[keep].sum()
        f = fin_sh[keep] / fin_sh[keep].sum()
        items.append(
            Item(
                election,
                "move",
                max(pf.age_days - fin.age_days, 1.0),
                [ids[i] for i in keep],
                [blocs[i] for i in keep],
                a,
                np.log(f) - np.log(a),
                final_age=fin.age_days,
            )
        )
    for it in items:
        it.weight = 1.0 / len(items)
    return items


_ITEM_CACHE: dict[int, tuple[Item, list[Item]]] = {}


def calibration_items(elections, reg: Registry | None = None) -> list[Item]:
    reg = reg or default_registry()
    out = []
    for e in elections:
        if e not in _ITEM_CACHE:
            _ITEM_CACHE[e] = (final_item(e, reg), move_items(e, reg))
        fin, moves = _ITEM_CACHE[e]
        out += [fin, *moves]
    return out


# --- forme en τ : Jennings–Wlezien


def jw_shape(exclude: tuple[str, ...] = (), df: pl.DataFrame | None = None) -> dict:
    """γ de la variance du mouvement, estimé sur Jennings–Wlezien.

    Erreur du dernier sondage (parts normalisées, partis de plus de 2 %) contre le résultat,
    en log-parts centrées par élection ; variance par horizon h ajustée par
    v(h) = v_7 + a·((h − 7)/83)^γ (moindres carrés pondérés par le nombre d'observations).
    Présidentielles, premier tour, tous pays, sans la France de 1965 et 1969 et sans
    `exclude`."""
    from scipy.optimize import least_squares

    from vvs.ingest.jennings_wlezien import _filter, horizon_errors, load_polls

    df = load_polls() if df is None else df
    d = _filter(df, None, "presidential", 1, list(JW_EXCLUDE) + list(exclude))
    he = horizon_errors(d, JW_HORIZONS, max_staleness=14, normalize=True).filter(
        (pl.col("poll_share") > 2) & (pl.col("result_share") > 2)
    )
    he = he.with_columns((pl.col("poll_share").log() - pl.col("result_share").log()).alias("e"))
    he = he.with_columns(
        (
            pl.col("e")
            - (pl.col("e") * pl.col("result_share")).sum().over("horizon", "election_id")
            / pl.col("result_share").sum().over("horizon", "election_id")
        ).alias("ec")
    )
    t = (
        he.group_by("horizon")
        .agg((pl.col("ec") ** 2).mean().alias("v"), pl.len().alias("n"))
        .sort("horizon")
    )
    h = t["horizon"].to_numpy().astype(float)
    v = t["v"].to_numpy()
    n = t["n"].to_numpy().astype(float)

    def resid(x):
        return (np.exp(x[0]) + np.exp(x[1]) * (np.maximum(h - 7, 0) / 83) ** x[2] - v) * np.sqrt(n)

    r = least_squares(resid, [math.log(0.02), math.log(0.05), 0.7], bounds=([-10, -10, 0.1], [2, 2, 2]))
    return {
        "gamma": float(r.x[2]),
        "v7": float(math.exp(r.x[0])),
        "a83": float(math.exp(r.x[1])),
        "table": [
            {"horizon": int(a), "var": float(b), "n": int(c)} for a, b, c in zip(h, v, n)
        ],
    }


# --- vraisemblance


@dataclass
class ErrorParams:
    sigma_bloc_final: float
    sigma_cand_final: float
    rho: float
    a_bloc: float  # variance du mouvement de bloc à h = 90
    a_cand: float
    gamma: float
    nu: float
    final_age: float
    bloc_weight: dict[str, float] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)


N_BASE = 5  # σ_Bf, σ_Cf, ρ, a_B, a_C (en log)


def _unpack(x: np.ndarray) -> tuple:
    return (
        math.exp(2 * x[0]),
        math.exp(2 * x[1]),
        math.exp(x[2]),
        math.exp(x[3]),
        math.exp(x[4]),
        np.exp(x[N_BASE:]),
    )


def _g(h, gamma):
    return (np.maximum(h, 0.0) / H_REF) ** gamma


def item_cov(
    it: Item, sb2, sc2, rho, aB, aC, w: dict[str, float] | np.ndarray, gamma: float
) -> np.ndarray:
    """Covariance (n × n) des log-erreurs d'un item (sans le facteur λ)."""
    bidx = [BLOCS.index(b) for b in it.blocs]
    wb = np.array([w[b] for b in it.blocs]) if isinstance(w, dict) else w[bidx]
    same = np.equal.outer(bidx, bidx)
    if it.kind == "final":
        vb, vc = sb2, sc2
    else:
        g = _g(it.h, gamma)
        vb, vc = aB * g, aC * g
    return same * np.sqrt(np.outer(wb, wb)) * vb + np.diag(vc + rho / it.p)


def _contrast_logpdf(d: np.ndarray, cov: np.ndarray, nu: float) -> float:
    """log-densité des contrastes d_k − d_n (k < n) sous N(0, cov) ou Student ν."""
    n = len(d)
    L = np.hstack([np.eye(n - 1), -np.ones((n - 1, 1))])
    x = L @ d
    S = L @ cov @ L.T
    try:
        ch = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return -1e10
    z = np.linalg.solve(ch, x)
    q = float(z @ z)
    logdet = 2 * float(np.log(np.diag(ch)).sum())
    k = n - 1
    if math.isinf(nu):
        return -0.5 * (q + logdet + k * math.log(2 * math.pi))
    return (
        math.lgamma((nu + k) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * k * math.log(nu * math.pi)
        - 0.5 * logdet
        - 0.5 * (nu + k) * math.log1p(q / nu)
    )


def _neg_objective(x, items, gamma, nu, bias):
    sb2, sc2, rho, aB, aC, w = _unpack(x)
    total = 0.0
    for it in items:
        cov = item_cov(it, sb2, sc2, rho, aB, aC, w, gamma)
        d = it.d
        if it.kind == "final" and bias:
            d = d - np.array([bias.get(b, 0.0) for b in it.blocs])
        total += it.weight * _contrast_logpdf(d, cov, nu)
    # priors faibles sur les échelles (log-normaux larges), prior des poids de bloc
    lp = -0.5 * ((x[N_BASE:] / BLOC_WEIGHT_PRIOR_SD) ** 2).sum()
    centers = np.array([math.log(0.1), math.log(0.1), math.log(1e-3), math.log(0.05), math.log(0.05)])
    lp += -0.5 * (((x[:N_BASE] - centers) / 2.0) ** 2).sum()
    return -(total + lp)


def _fit_params(items, gamma, nu, bias, x0=None) -> np.ndarray:
    from scipy.optimize import minimize

    x0 = (
        np.concatenate(
            [[math.log(0.1), math.log(0.1), math.log(1e-3), math.log(0.05), math.log(0.05)],
             np.zeros(len(BLOCS))]
        )
        if x0 is None
        else x0
    )
    res = minimize(_neg_objective, x0, args=(items, gamma, nu, bias), method="L-BFGS-B")
    return res.x


def bloc_final_errors(items: list[Item]) -> dict[str, dict[int, float]]:
    """log(résultat du bloc) − log(agrégat final du bloc), par bloc et élection."""
    out: dict[str, dict[int, float]] = {}
    for it in items:
        if it.kind != "final":
            continue
        r = np.exp(it.d) * it.p
        r = r / r.sum()
        for b in set(it.blocs):
            m = np.array([x == b for x in it.blocs])
            out.setdefault(b, {})[it.election] = float(np.log(r[m].sum()) - np.log(it.p[m].sum()))
    return out


def stable_bias(items: list[Item], prior_sd: float = BIAS_PRIOR_SD) -> tuple[dict, dict]:
    """Biais de bloc admis (règle du module) et diagnostic par bloc."""
    errs = bloc_final_errors(items)
    bias, diag = {}, {}
    for b, by_e in errs.items():
        v = np.array(list(by_e.values()))
        same_sign = len(v) >= 3 and (np.all(v > 0) or np.all(v < 0))
        mean = float(v.mean())
        var = float(v.var(ddof=1)) if len(v) > 1 else prior_sd**2
        shrink = len(v) / (len(v) + var / prior_sd**2)
        diag[b] = {"errors": by_e, "mean": mean, "same_sign": bool(same_sign), "shrink": shrink}
        if same_sign:
            bias[b] = mean * shrink
    return bias, diag


# --- modèle


class ErrorModel:
    def __init__(self, params: ErrorParams, info: dict | None = None):
        self.params = params
        self.info = info or {}

    # estimation

    @classmethod
    def fit(
        cls,
        elections=(2002, 2007, 2012, 2017, 2022),
        exclude=(),
        nu: float | str = "auto",
        registry: Registry | None = None,
        gamma: float | None = None,
        items: list[Item] | None = None,
        use_bias: bool = True,
    ) -> ErrorModel:
        """Calibre η sur `elections` privées de `exclude` (leave-one-election-out).

        `nu` : degrés de liberté de l'échelle commune, ou "auto" (validation croisée interne
        sur `NU_GRID`). `gamma` : forme en τ, estimée sur Jennings–Wlezien si None (sans
        l'élection française exclue). `items` : données de calibration déjà calculées (tests)."""
        reg = registry or default_registry()
        excl = {int(e) for e in np.atleast_1d(exclude)} if exclude is not None else set()
        train = [e for e in elections if e not in excl]
        if items is None:
            items = calibration_items(train, reg)
        else:
            items = [it for it in items if it.election in train]
        jw = None
        if gamma is None:
            jw = jw_shape(exclude=tuple(JW_ELECTION_ID[e] for e in excl if e in JW_ELECTION_ID))
            gamma = jw["gamma"]
        bias, bias_diag = stable_bias(items) if use_bias else ({}, {})

        nu_scores = {}
        if nu == "auto":
            if len(train) >= 3:
                for cand in NU_GRID:
                    s = 0.0
                    for held in train:
                        tr = [it for it in items if it.election != held]
                        te = [it for it in items if it.election == held]
                        b_in, _ = stable_bias(tr) if use_bias else ({}, {})
                        x = _fit_params(tr, gamma, cand, b_in)
                        s -= _neg_objective(x, te, gamma, cand, b_in) - _neg_objective(
                            x, [], gamma, cand, b_in
                        )
                    nu_scores[cand] = s
                nu = max(nu_scores, key=nu_scores.get)
            else:
                nu = 8.0
        x = _fit_params(items, gamma, float(nu), bias)
        sb2, sc2, rho, aB, aC, w = _unpack(x)
        finals = [it.final_age for it in items if it.kind == "move"]
        params = ErrorParams(
            sigma_bloc_final=math.sqrt(sb2),
            sigma_cand_final=math.sqrt(sc2),
            rho=rho,
            a_bloc=aB,
            a_cand=aC,
            gamma=float(gamma),
            nu=float(nu),
            final_age=float(np.mean(finals)) if finals else FINAL_WINDOW_DAYS / 2,
            bloc_weight={b: float(v) for b, v in zip(BLOCS, w)},
            bias=bias,
        )
        info = {
            "train": train,
            "exclude": sorted(excl),
            "n_items": len(items),
            "nu_scores": {str(k): v for k, v in nu_scores.items()},
            "bias_diagnostics": bias_diag,
            "jw": jw,
        }
        return cls(params, info)

    # covariance et tirages

    def horizon_lag(self, horizon: float) -> float:
        """h : jours entre une estimation d'âge `horizon` et l'agrégat final."""
        return max(float(horizon) - self.params.final_age, 0.0)

    def variances(self, horizon: float, blocs: list[str], p: np.ndarray) -> tuple:
        """(variance de bloc par candidat (K,), variance propre (…, K)) conditionnelles à λ."""
        pr = self.params
        g = _g(self.horizon_lag(horizon), pr.gamma)
        wb = np.array([pr.bloc_weight.get(b, 1.0) for b in blocs])
        vb = wb * (pr.sigma_bloc_final**2 + pr.a_bloc * g)
        vc = pr.sigma_cand_final**2 + pr.a_cand * g + pr.rho / np.maximum(p, 1e-4)
        return vb, vc

    def covariance(self, horizon: float, blocs: list[str], p: np.ndarray) -> np.ndarray:
        """Covariance marginale de η (K × K) pour des parts de référence p (fractions)."""
        vb, vc = self.variances(horizon, blocs, np.asarray(p))
        same = np.equal.outer(np.array(blocs), np.array(blocs))
        cov = same * np.sqrt(np.outer(vb, vb)) + np.diag(vc)
        nu = self.params.nu
        return cov * (nu / (nu - 2) if math.isfinite(nu) and nu > 2 else 1.0)

    def sample_eta(
        self,
        horizon: float,
        blocs: list[str],
        p: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """η (n, K) pour des parts de référence p (n, K), une ligne par tirage."""
        pr = self.params
        p = np.atleast_2d(p)
        n, K = p.shape
        vb, vc = self.variances(horizon, blocs, p)
        ub = sorted(set(blocs))
        bidx = np.array([ub.index(b) for b in blocs])
        sd_b = np.array([math.sqrt(vb[blocs.index(b)]) for b in ub])
        u = rng.standard_normal((n, len(ub))) * sd_b
        v = rng.standard_normal((n, K)) * np.sqrt(vc)
        eta = u[:, bidx] + v
        if math.isfinite(pr.nu):
            lam = np.sqrt(pr.nu / rng.chisquare(pr.nu, size=n))
            eta = eta * lam[:, None]
        mu = np.array([pr.bias.get(b, 0.0) for b in blocs])
        return eta + mu

    def sample(
        self,
        theta_draws: np.ndarray,
        horizon: float,
        candidates: list[str],
        blocs: list[str],
        rng: np.random.Generator,
        n: int = 10_000,
    ) -> np.ndarray:
        """Tirages de parts au jour du vote (n, K), fractions.

        `theta_draws` : (S, K) logits de la configuration `candidates` (tirages du postérieur à
        la date de prévision, ou une seule ligne pour une estimation ponctuelle), rééchantillonnés
        avec remise à n tirages ; `horizon` : âge de l'estimation en jours avant le vote."""
        theta = np.atleast_2d(np.asarray(theta_draws, dtype=float))
        if theta.shape[1] != len(candidates) or len(blocs) != len(candidates):
            raise ValueError("theta_draws, candidates et blocs de tailles incompatibles")
        idx = rng.integers(0, theta.shape[0], size=n)
        th = theta[idx]
        p = _softmax(th)
        eta = self.sample_eta(horizon, list(blocs), p, rng)
        return _softmax(th + eta)

    # résumés

    def summary(self, horizons=(7, 30, 90, 180)) -> pl.DataFrame:
        """Écarts types sur l'échelle logit par bloc et horizon (marginaux, λ inclus)."""
        pr = self.params
        k = pr.nu / (pr.nu - 2) if math.isfinite(pr.nu) and pr.nu > 2 else 1.0
        rows = []
        for h in horizons:
            g = _g(self.horizon_lag(h), pr.gamma)
            for b in BLOCS:
                vb = pr.bloc_weight.get(b, 1.0) * (pr.sigma_bloc_final**2 + pr.a_bloc * g)
                vc = pr.sigma_cand_final**2 + pr.a_cand * g
                rows.append(
                    {
                        "horizon": h,
                        "bloc": b,
                        "sd_bloc": math.sqrt(k * vb),
                        "sd_cand": math.sqrt(k * vc),
                        "sd_total": math.sqrt(k * (vb + vc)),
                        "corr_within_bloc": vb / (vb + vc),
                        "bias": pr.bias.get(b, 0.0),
                    }
                )
        return pl.DataFrame(rows)

    def points_sd(
        self,
        shares: np.ndarray,
        blocs: list[str],
        horizon: float,
        rng: np.random.Generator,
        n: int = 20_000,
    ) -> np.ndarray:
        """Écart type, en points, de la part au vote de chaque candidat d'une configuration de
        parts `shares` (fractions) supposées exactes à la date de prévision."""
        s = np.asarray(shares, dtype=float)
        th = np.log(s / s.sum())[None, :]
        out = self.sample(th, horizon, [str(i) for i in range(len(s))], blocs, rng, n)
        return 100 * out.std(0)

    def to_dict(self) -> dict:
        return {"params": asdict(self.params), "info": self.info}


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)
