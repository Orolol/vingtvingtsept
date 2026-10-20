"""Modèle d'agrégation des sondages de premier tour (docs/04-modele.md, section 1), en numpyro.

Échelle : logit multinomial. Temps : semaines, index 0 = début de la grille, T-1 = semaine
du vote.

    θ_{c,t} = β_{b(c),t} + δ_{c,t}                       soutien latent
    β_{·,t} = β_{·,t-1} + σ_β ε^β_t                      marche aléatoire de bloc
    δ_{·,t} = δ_{·,t-1} + σ_δ ε^δ_t                      marche aléatoire du candidat
    π_i     = softmax_{c ∈ h_i}(θ_{c,t_i} + α_{p_i,c} + γ_{pop_i,b(c)})
    y_i     ~ DirichletMultinomial(n_i, κ_i π_i),  κ_i tel que n_eff = f · n_i

Identification. Le softmax est invariant à l'ajout d'une constante à tous les candidats
d'une hypothèse ; β et δ ne sont séparés que par convention. On impose donc, à chaque pas :
somme nulle des innovations de β sur les blocs, somme nulle de δ entre candidats d'un même
bloc (β_b est alors la moyenne des logits du bloc), effets maison doublement centrés (somme
nulle sur les instituts pour chaque bloc ou candidat, et sur les blocs ou les candidats d'un
bloc pour chaque institut). Les composantes centrées sont remises à l'échelle pour garder
la variance marginale annoncée. Les marches sont en paramétrisation non centrée par défaut
(centrée en option, voir `fit`).

Vraisemblance. Les comptes sont les parts arrondies sur l'effectif nominal (`sample_size`,
1000 s'il manque). Le facteur de design f ne peut pas modifier des comptes observés : on
l'introduit par une Dirichlet-multinomiale dont la concentration κ_i = (f n_i − 1)/(1 − f)
donne exactement une variance de multinomiale d'effectif f · n_i. C'est la multinomiale
« à effectif réduit » de la spec, écrite comme une loi propre sur des comptes entiers.
Chaque hypothèse pèse 1/H dans la vraisemblance, H étant le nombre d'hypothèses du même
sondage (mêmes répondants, observations non indépendantes).

Plancher d'erreur non due à l'échantillonnage (option `Priors.n_max`, phase 3). Un sondage de
13 000 répondants n'est pas treize fois plus précis qu'un sondage de 1 000 : cadre, questionnaire
et redressement ajoutent une erreur propre à chaque vague qui ne décroît pas avec n. On l'écrit
1/n_eff = 1/(f·n) + 1/n_max, avec n_max estimé (prior log-normal). Sans plancher, la très grande
enquête Ipsos/Cevipof écrase les autres et tire le facteur de design f vers le bas. C'est le
défaut depuis la phase 4 (décision du journal de backtest du 2026-10-20) ; `n_max=None` redonne
le modèle de la phase 2.

Substitution intra-bloc (option `Priors.substitution`, phase 4, désactivée par défaut). Le
softmax restreint suppose l'indépendance des alternatives non pertinentes : retirer un candidat
redistribue ses voix au prorata de tous les autres, et une configuration qui contient plus de
candidats qu'aucun sondage les dilue tous. L'option remplace l'observation par un logit emboîté
par bloc, sur la même décomposition θ = β + δ :

    log π_{i,c} = log P_i(b(c)) + log P_i(c | b(c))
    P_i(c | b)  = softmax_{c' ∈ h_i ∩ b}(D_{i,c'})            D = δ + effet maison candidat
    P_i(b)      = softmax_{b' présent dans h_i}(U_{i,b'})
    U_{i,b}     = B_{i,b} + λ_b · log Σ_{c ∈ h_i ∩ b} exp(D_{i,c})
                                                           B = β + effet maison bloc + γ

λ_b ∈ [0, 1] (prior Beta(lam_a, lam_b)) est le paramètre de dissimilarité du bloc, s_b = 1 − λ_b
sa substitution. λ_b = 1 redonne exactement le softmax restreint (U_b = log Σ_{c∈b} exp(B_b + D_c)).
λ_b = 0 : la part du bloc ne dépend pas de ceux de ses candidats qui sont présents, et retirer un
candidat reporte toutes ses voix sur les candidats présents de son bloc. Entre les deux, retirer
d'une hypothèse des candidats d'un bloc de part P_b dont il reste une part Q_b ramène le poids du
bloc à P_b^{1−λ_b} · Q_b^{λ_b} avant renormalisation entre blocs : le bloc garde une partie des
voix des absents, le reste va aux autres blocs au prorata. Un bloc entièrement absent reporte
ses voix sur tous les autres au prorata (IIA entre blocs).

Pourquoi un logit emboîté plutôt qu'une « part de bloc conservée » sur un champ complet : le
modèle n'a pas de champ complet (l'univers des candidats contient des alternatives jamais testées
ensemble, Juppé et Sarkozy), alors que le logit emboîté donne des parts cohérentes entre elles
pour toute configuration (modèle d'utilité aléatoire), se réduit au modèle existant à λ = 1 et
ne change que l'observation. Identification : δ reste centré dans le bloc, β_b est le niveau du
bloc ; λ_b est identifié par les hypothèses qui diffèrent par la composition d'un bloc, au
premier chef celles d'un même sondage (mêmes répondants, même date) ; il est sans effet pour un
bloc d'un seul candidat (prior seul). La substitution entre blocs voisins (centre et droite) n'est
pas modélisée.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpyro

numpyro.set_host_device_count(min(4, os.cpu_count() or 1))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_uniform
from numpyro.infer.autoguide import AutoNormal

from vvs.model import certainty
from vvs.model import pollster_ratings as ratings
from vvs.model.data import PollData


@dataclass(frozen=True)
class Priors:
    """Hyperparamètres des priors (échelle logit). Les volatilités sont hebdomadaires quel que
    soit le pas de la grille : un pas de k jours a un écart-type σ · sqrt(k / 7)."""

    sigma_beta: float = 0.10  # HalfNormal : volatilité hebdomadaire des blocs
    sigma_delta: float = 0.10  # HalfNormal : volatilité hebdomadaire candidat dans son bloc
    beta0_sd: float = 1.0  # écart-type du prior structurel des blocs au début de la grille
    delta0_sd: float = 1.0  # écart-type des écarts initiaux entre candidats d'un bloc
    house_bloc: float = 0.15  # HalfNormal : échelle des effets maison institut × bloc
    house_cand: float = 0.05  # HalfNormal : échelle du terme institut × candidat
    pop_sd: float = 0.15  # écart-type de l'effet population (par bloc)
    design_a: float = 12.0  # Beta(a, b) du facteur de design, moyenne 0,6
    design_b: float = 8.0
    # Plancher d'erreur non due à l'échantillonnage : médiane du prior log-normal de n_max
    # (None : pas de plancher, modèle de la phase 2) et écart-type de log n_max.
    n_max: float | None = 2500.0
    n_max_sd: float = 0.3
    # Substitution intra-bloc (logit emboîté, voir le docstring du module) : désactivée par
    # défaut ; λ_b ~ Beta(lam_a, lam_b) par bloc.
    substitution: bool = False
    lam_a: float = 2.0
    lam_b: float = 2.0
    # Palmarès des instituts (vvs.model.pollster_ratings), désactivé par défaut : prior appris
    # des effets maison institut × bloc et facteur de précision sur n_eff, sans l'élection.
    pollster_ratings: bool = False
    # Certitude du choix (vvs.model.certainty), désactivée par défaut : volatilité de δ mise à
    # l'échelle par candidat, σ_δ · exp(−κ x_c) ; κ calibré sans l'élection si None.
    certainty: bool = False
    certainty_kappa: float | None = None


# --- entrées statiques


def _within_bloc_projector(cand_bloc: np.ndarray) -> np.ndarray:
    """Matrice C×C qui centre un vecteur candidat dans chaque bloc et rééchelonne.

    Pour un bloc de n candidats, la projection z ↦ z − moyenne_bloc(z) a une variance
    marginale (n−1)/n : on multiplie par sqrt(n/(n−1)). Bloc d'un seul candidat : zéro.
    """
    C = len(cand_bloc)
    M = np.zeros((C, C))
    for b in np.unique(cand_bloc):
        idx = np.flatnonzero(cand_bloc == b)
        n = len(idx)
        if n < 2:
            continue
        block = (np.eye(n) - 1.0 / n) * np.sqrt(n / (n - 1))
        M[np.ix_(idx, idx)] = block
    return M


def _zero_sum_basis(groups: np.ndarray) -> np.ndarray:
    """Base orthonormale (n × d) des vecteurs de somme nulle dans chaque groupe.

    d = n − nombre de groupes ; un groupe d'un seul élément ne contribue aucune colonne.
    """
    cols = []
    n = len(groups)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        k = len(idx)
        if k < 2:
            continue
        # vecteurs propres de la projection I − 11ᵀ/k associés à la valeur propre 1
        _, vecs = np.linalg.eigh(np.eye(k) - 1.0 / k)
        for v in vecs[:, 1:].T:
            col = np.zeros(n)
            col[idx] = v
            cols.append(col)
    return np.stack(cols, axis=1) if cols else np.zeros((n, 0))


def _center(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    n = x.shape[axis]
    if n < 2:
        return jnp.zeros_like(x)
    return (x - x.mean(axis=axis, keepdims=True)) * jnp.sqrt(n / (n - 1))


@dataclass
class ModelInputs:
    """Tableaux jax et dimensions dérivés de `PollData`."""

    T: int
    B: int
    C: int
    P: int
    n_pop: int
    cand_bloc: jnp.ndarray
    M: jnp.ndarray
    prior_mean: jnp.ndarray
    cand_idx: jnp.ndarray
    mask: jnp.ndarray
    counts: jnp.ndarray
    n_nominal: jnp.ndarray
    weight: jnp.ndarray
    week: jnp.ndarray
    pollster_idx: jnp.ndarray
    pop_idx: jnp.ndarray
    Qb: jnp.ndarray  # (B, B-1) base des vecteurs bloc de somme nulle
    Qd: jnp.ndarray  # (C, D) base des vecteurs candidat de somme nulle dans chaque bloc
    scale_b: jnp.ndarray  # () sqrt(B/(B-1))
    scale_d: jnp.ndarray  # (D,) sqrt(n_b/(n_b-1)) du bloc de chaque direction
    centered: bool = False
    step_scale: float = 1.0  # sqrt(step_days / 7)
    priors: Priors = field(default_factory=Priors)
    rating_mu: jnp.ndarray | None = None  # (P, B) palmarès : moyenne du prior des effets maison
    rating_sd: jnp.ndarray | None = None  # (P, B) écart-type de ce prior
    rating_q: jnp.ndarray | None = None  # (P,) facteur de précision sur n_eff
    delta_scale: jnp.ndarray | None = None  # (C,) certitude : échelle de la volatilité de δ

    @classmethod
    def from_data(
        cls, data: PollData, priors: Priors | None = None, centered: bool = False
    ) -> ModelInputs:
        B = len(data.blocs)
        Qd = _zero_sum_basis(data.cand_bloc)
        n_b = np.bincount(data.cand_bloc, minlength=B)
        dir_bloc = np.argmax(np.abs(Qd) > 1e-12, axis=0) if Qd.shape[1] else np.zeros(0, int)
        nd = n_b[data.cand_bloc[dir_bloc]] if Qd.shape[1] else np.zeros(0)
        return cls(
            T=data.n_weeks,
            B=len(data.blocs),
            C=len(data.candidates),
            P=len(data.pollsters),
            n_pop=len(data.populations),
            cand_bloc=jnp.asarray(data.cand_bloc),
            M=jnp.asarray(_within_bloc_projector(data.cand_bloc)),
            prior_mean=jnp.asarray(data.prior_bloc_mean),
            cand_idx=jnp.asarray(data.cand_idx),
            mask=jnp.asarray(data.mask),
            counts=jnp.asarray(data.counts),
            n_nominal=jnp.asarray(data.n_nominal),
            weight=jnp.asarray(data.weight),
            week=jnp.asarray(data.week),
            pollster_idx=jnp.asarray(data.pollster_idx),
            pop_idx=jnp.asarray(data.pop_idx),
            Qb=jnp.asarray(_zero_sum_basis(np.zeros(B, dtype=int))),
            Qd=jnp.asarray(Qd),
            scale_b=jnp.asarray(np.sqrt(B / (B - 1)) if B > 1 else 0.0),
            scale_d=jnp.asarray(np.sqrt(nd / np.maximum(nd - 1, 1))),
            centered=centered,
            step_scale=float(np.sqrt(data.step_days / 7)),
            priors=priors or Priors(),
            **ratings.ratings_inputs(data, priors),
            **certainty.certainty_inputs(data, priors),
        )

    @property
    def D(self) -> int:
        return self.Qd.shape[1]


# --- transformations déterministes (partagées par le modèle et le postérieur)


def latent(s: dict, inp: ModelInputs) -> dict:
    """Variables latentes à partir des sites échantillonnés (un tirage).

    Rend θ (T, C), β (T, B), δ (T, C), α (P, C) et γ (n_pop, C), ainsi que leurs composantes
    de bloc et de candidat (`house_bloc` (P, B), `house_cand` (P, C), `gamma_bloc` (n_pop, B))
    et λ (B,) avec la substitution.
    """
    pr = inp.priors
    if inp.centered:
        beta = inp.prior_mean.mean() + s["u_beta"] @ inp.Qb.T
        delta = s["u_delta"] @ inp.Qd.T
    else:
        beta0 = inp.prior_mean + pr.beta0_sd * _center(s["z_beta0"], -1)
        beta_steps = s["sigma_beta"] * inp.step_scale * _center(s["z_beta"], -1)
        beta = beta0 + jnp.concatenate([jnp.zeros((1, inp.B)), jnp.cumsum(beta_steps, axis=0)])
        delta0 = pr.delta0_sd * (inp.M @ s["z_delta0"])
        z_delta = s["z_delta"] * inp.delta_scale if pr.certainty else s["z_delta"]
        delta_steps = s["sigma_delta"] * inp.step_scale * (z_delta @ inp.M.T)
        delta = delta0 + jnp.concatenate([jnp.zeros((1, inp.C)), jnp.cumsum(delta_steps, axis=0)])

    theta = beta[:, inp.cand_bloc] + delta

    house_bloc = s["sigma_house_bloc"] * _center(_center(s["z_house_bloc"], 0), 1)
    if pr.pollster_ratings:
        house_bloc = ratings.rated_house_bloc(s["z_house_bloc"], inp)
    house_cand = s["sigma_house_cand"] * _center(s["z_house_cand"] @ inp.M.T, 0)
    alpha = house_bloc[:, inp.cand_bloc] + house_cand

    if inp.n_pop > 1:
        g = pr.pop_sd * _center(s["z_pop"], -1)
        gamma_bloc = jnp.concatenate([jnp.zeros((1, inp.B)), g])
    else:
        gamma_bloc = jnp.zeros((1, inp.B))
    gamma = gamma_bloc[:, inp.cand_bloc]
    out = {
        "theta": theta,
        "beta": beta,
        "alpha": alpha,
        "gamma": gamma,
        "delta": delta,
        "house_bloc": house_bloc,
        "house_cand": house_cand,
        "gamma_bloc": gamma_bloc,
    }
    if pr.substitution:
        out["lam"] = s["lam"]
    return out


def masked_log_softmax(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """log-softmax sur les seules cases `mask` (les autres valent 0, à ignorer)."""
    x = jnp.where(mask, x, -1e9)
    m = jax.lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
    z = jnp.where(mask, jnp.exp(x - m), 0.0)
    out = x - m - jnp.log(z.sum(axis=-1, keepdims=True))
    return jnp.where(mask, out, 0.0)


def nested_log_probs(
    bloc_term: jnp.ndarray,
    cand_term: jnp.ndarray,
    slot_bloc: jnp.ndarray,
    mask: jnp.ndarray,
    lam: jnp.ndarray,
) -> jnp.ndarray:
    """log π (N, K) du logit emboîté par bloc (docstring du module), cases masquées à 0.

    `bloc_term` (N, B) : B_{i,b} ; `cand_term` (N, K) : D_{i,c} ; `slot_bloc` (N, K) : bloc de
    chaque case ; `mask` (N, K) ; `lam` (B,) ou (N, B). λ = 1 donne le softmax restreint."""
    B = bloc_term.shape[-1]
    member = mask[..., None] & (slot_bloc[..., None] == jnp.arange(B))  # (N, K, B)
    present = member.any(-2)  # (N, B)
    x = jnp.where(member, cand_term[..., None], -1e9)
    m = jax.lax.stop_gradient(jnp.where(present, x.max(-2), 0.0))
    z = jnp.where(member, jnp.exp(x - m[..., None, :]), 0.0).sum(-2)
    lse = jnp.where(present, m + jnp.log(jnp.where(present, z, 1.0)), 0.0)  # (N, B)
    util = bloc_term + lam * lse
    log_pb = masked_log_softmax(util, present)
    sb = jnp.where(mask, slot_bloc, 0)
    out = (
        jnp.take_along_axis(log_pb, sb, axis=-1)
        + cand_term
        - jnp.take_along_axis(lse, sb, axis=-1)
    )
    return jnp.where(mask, out, 0.0)


def config_log_shares(
    beta: np.ndarray, delta: np.ndarray, cand_bloc: np.ndarray, lam: np.ndarray | None = None
) -> np.ndarray:
    """log-parts d'une configuration de candidats où tous sont présents (numpy, postérieur).

    `beta` (..., B) : niveau des blocs ; `delta` (..., K) : écarts des candidats de la
    configuration ; `cand_bloc` (K,) : bloc de chaque candidat (index dans B) ; `lam` (S, B)
    ou (B,) : λ par bloc, aligné sur la première dimension de `beta` (None : softmax de
    θ = β + δ, modèle sans substitution). Rend (..., K)."""
    beta = np.asarray(beta, dtype=float)
    delta = np.asarray(delta, dtype=float)
    cand_bloc = np.asarray(cand_bloc)
    theta = beta[..., cand_bloc] + delta
    if lam is None:
        return theta - _logsumexp(theta)
    lam = np.asarray(lam, dtype=float)
    if lam.ndim == 2:  # (S, B) -> (S, 1, ..., B)
        lam = lam.reshape(lam.shape[0], *([1] * (beta.ndim - 2)), lam.shape[1])
    present = np.unique(cand_bloc)
    lse = np.concatenate([_logsumexp(delta[..., cand_bloc == b]) for b in present], axis=-1)
    util = beta[..., present] + lam[..., present] * lse
    log_pb = util - _logsumexp(util)
    pos = np.searchsorted(present, cand_bloc)
    return log_pb[..., pos] + delta - lse[..., pos]


def _logsumexp(x: np.ndarray) -> np.ndarray:
    """log Σ exp sur le dernier axe, dimension gardée (…, 1)."""
    m = x.max(-1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(-1, keepdims=True))


def hypothesis_probs(lat: dict, inp: ModelInputs) -> jnp.ndarray:
    """π_i (N, K) : softmax restreint aux candidats de chaque hypothèse, effets inclus ; logit
    emboîté par bloc avec la substitution (`Priors.substitution`)."""
    ci = inp.cand_idx
    if inp.priors.substitution:
        bloc_term = (
            lat["beta"][inp.week]
            + lat["house_bloc"][inp.pollster_idx]
            + lat["gamma_bloc"][inp.pop_idx]
        )
        cand_term = (
            lat["delta"][inp.week[:, None], ci] + lat["house_cand"][inp.pollster_idx[:, None], ci]
        )
        lp = nested_log_probs(bloc_term, cand_term, inp.cand_bloc[ci], inp.mask, lat["lam"])
        return jnp.where(inp.mask, jnp.exp(lp), 0.0)
    logits = (
        lat["theta"][inp.week[:, None], ci]
        + lat["alpha"][inp.pollster_idx[:, None], ci]
        + lat["gamma"][inp.pop_idx[:, None], ci]
    )
    return jnp.where(inp.mask, jnp.exp(masked_log_softmax(logits, inp.mask)), 0.0)


def effective_size(f, n, n_max=None):
    """n_eff = 1 / (1/(f·n) + 1/n_max) ; f·n sans plancher (`n_max` None)."""
    if n_max is None:
        return f * n
    return 1.0 / (1.0 / (f * n) + 1.0 / n_max)


def concentration(f: jnp.ndarray, n: jnp.ndarray, n_max=None) -> jnp.ndarray:
    """κ tel que la Dirichlet-multinomiale d'effectif n ait la variance d'une multinomiale n_eff.

    Variance des parts d'une DM : p(1−p)/n · (n+κ)/(1+κ). En posant r = n/n_eff, l'égalité
    avec p(1−p)/n_eff donne κ = (n − r)/(r − 1) ; sans plancher r = 1/f et
    κ = (f n − 1)/(1 − f).
    """
    if n_max is None:
        return jnp.maximum(f * n - 1.0, 1e-3) / (1.0 - f)
    r = 1.0 / f + n / n_max
    return jnp.maximum(n - r, 1e-3) / (r - 1.0)


def dirmult_loglik(counts, probs, mask, n, kappa) -> jnp.ndarray:
    """Log-vraisemblance Dirichlet-multinomiale par hypothèse (cases masquées ignorées)."""
    a = jnp.where(mask, kappa[:, None] * probs, 1.0)
    a = jnp.maximum(a, 1e-8)
    per_cell = jnp.where(mask, gammaln(counts + a) - gammaln(a) - gammaln(counts + 1.0), 0.0)
    return gammaln(kappa) - gammaln(n + kappa) + gammaln(n + 1.0) + per_cell.sum(-1)


def _walk(name: str, shape: tuple[int, int], loc0, scale0, step) -> jnp.ndarray:
    """Marche aléatoire gaussienne en paramétrisation centrée : x_0 ~ N(loc0, scale0),
    x_t − x_{t−1} ~ N(0, step). Les états sont les paramètres échantillonnés."""
    x = numpyro.sample(name, dist.ImproperUniform(dist.constraints.real_vector, (), shape))
    lp = dist.Normal(loc0, scale0).log_prob(x[0]).sum()
    lp = lp + dist.Normal(0.0, step).log_prob(jnp.diff(x, axis=0)).sum()
    numpyro.factor(f"{name}_prior", lp)
    return x


def _centered_walks(inp: ModelInputs, sigma_beta, sigma_delta) -> dict:
    """Marches centrées dans les sous-espaces de somme nulle (bases orthonormales Qb, Qd).

    Mêmes lois que la version non centrée : la projection de N(0, σ² I) sur un sous-espace
    s'écrit Q u avec u ~ N(0, σ² I), puis on rééchelonne comme `_center`.
    """
    pr = inp.priors
    loc_b = inp.Qb.T @ (inp.prior_mean - inp.prior_mean.mean())
    u_beta = _walk(
        "u_beta",
        (inp.T, inp.B - 1),
        loc_b,
        pr.beta0_sd * inp.scale_b,
        sigma_beta * inp.step_scale * inp.scale_b,
    )
    out = {"u_beta": u_beta}
    if inp.D:
        out["u_delta"] = _walk(
            "u_delta",
            (inp.T, inp.D),
            0.0,
            pr.delta0_sd * inp.scale_d,
            sigma_delta * inp.step_scale * inp.scale_d,
        )
    else:
        out["u_delta"] = jnp.zeros((inp.T, 0))
    return out


def model(inp: ModelInputs) -> None:
    pr = inp.priors
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(pr.sigma_beta))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(pr.sigma_delta))
    sigma_house_bloc = numpyro.sample("sigma_house_bloc", dist.HalfNormal(pr.house_bloc))
    sigma_house_cand = numpyro.sample("sigma_house_cand", dist.HalfNormal(pr.house_cand))
    design = numpyro.sample("design", dist.Beta(pr.design_a, pr.design_b))
    n_max = (
        None
        if pr.n_max is None
        else numpyro.sample("n_max", dist.LogNormal(jnp.log(pr.n_max), pr.n_max_sd))
    )

    s = {
        "sigma_beta": sigma_beta,
        "sigma_delta": sigma_delta,
        "sigma_house_bloc": sigma_house_bloc,
        "sigma_house_cand": sigma_house_cand,
    }
    if inp.centered:
        s |= _centered_walks(inp, sigma_beta, sigma_delta)
    else:
        s |= {
            "z_beta0": numpyro.sample("z_beta0", dist.Normal(0, 1).expand([inp.B]).to_event(1)),
            "z_beta": numpyro.sample(
                "z_beta", dist.Normal(0, 1).expand([inp.T - 1, inp.B]).to_event(2)
            ),
            "z_delta0": numpyro.sample("z_delta0", dist.Normal(0, 1).expand([inp.C]).to_event(1)),
            "z_delta": numpyro.sample(
                "z_delta", dist.Normal(0, 1).expand([inp.T - 1, inp.C]).to_event(2)
            ),
        }
    s |= {
        "z_house_bloc": numpyro.sample(
            "z_house_bloc", dist.Normal(0, 1).expand([inp.P, inp.B]).to_event(2)
        ),
        "z_house_cand": numpyro.sample(
            "z_house_cand", dist.Normal(0, 1).expand([inp.P, inp.C]).to_event(2)
        ),
    }
    if inp.n_pop > 1:
        s["z_pop"] = numpyro.sample(
            "z_pop", dist.Normal(0, 1).expand([inp.n_pop - 1, inp.B]).to_event(2)
        )
    if pr.substitution:
        s["lam"] = numpyro.sample(
            "lam", dist.Beta(pr.lam_a, pr.lam_b).expand([inp.B]).to_event(1)
        )
    lat = latent(s, inp)
    probs = hypothesis_probs(lat, inp)
    kappa = concentration(design, inp.n_nominal, n_max)
    if pr.pollster_ratings:
        kappa = ratings.rated_concentration(
            design, inp.n_nominal, n_max, inp.rating_q[inp.pollster_idx]
        )
    ll = dirmult_loglik(inp.counts, probs, inp.mask, inp.n_nominal, kappa)
    numpyro.factor("polls", jnp.sum(inp.weight * ll))


# --- simulation (tests, contrôles prédictifs)


def simulate_counts(
    inp: ModelInputs,
    lat: dict,
    design: float,
    rng: np.random.Generator,
    n_max: float | None = None,
) -> np.ndarray:
    """Tire des comptes de sondages selon la vraisemblance du modèle, à latents donnés."""
    probs = np.asarray(hypothesis_probs(lat, inp))
    n = np.asarray(inp.n_nominal)
    kappa = np.asarray(concentration(jnp.asarray(design), inp.n_nominal, n_max))
    mask = np.asarray(inp.mask)
    out = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        k = mask[i]
        p = rng.dirichlet(np.maximum(kappa[i] * probs[i, k], 1e-6))
        out[i, k] = rng.multinomial(int(n[i]), p / p.sum())
    return out


# --- inférence


@dataclass
class FitResult:
    method: str
    samples: dict  # sites échantillonnés, tirages à plat (S, ...)
    num_chains: int
    settings: dict
    diagnostics: dict
    duration_s: float

    def by_chain(self, x: np.ndarray) -> np.ndarray:
        """(S, ...) → (chaînes, tirages, ...)."""
        return x.reshape(self.num_chains, -1, *x.shape[1:])


def site_diagnostics(samples_by_chain: dict) -> dict:
    """r_hat et ESS par site (max r_hat, min ESS sur les éléments du site)."""
    out = {}
    summ = summary(samples_by_chain, group_by_chain=True)
    for name, st in summ.items():
        out[name] = {
            "r_hat_max": float(np.nanmax(st["r_hat"])),
            "ess_min": float(np.nanmin(st["n_eff"])),
            "mean": float(np.mean(st["mean"])) if np.size(st["mean"]) == 1 else None,
        }
    return out


def fit(
    data: PollData,
    method: str = "nuts",
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    seed: int = 0,
    target_accept: float = 0.9,
    max_tree_depth: int = 10,
    svi_steps: int = 4000,
    svi_lr: float = 0.01,
    priors: Priors | None = None,
    centered: bool = False,
    progress_bar: bool = False,
) -> FitResult:
    """Ajuste le modèle. `method` : `nuts` (référence) ou `svi` (approximation rapide, tests).

    `centered` : marches aléatoires en paramétrisation non centrée (défaut) ou centrée
    (`_centered_walks`). Sur 2022 au 8 avril (79 semaines, 4 chaînes de 500 + 500), la non
    centrée sature la profondeur d'arbre (1023 pas) mais converge sur tous les sites
    (r_hat ≤ 1,01) ; la centrée est deux fois plus rapide, donne les mêmes parts, mais les
    états sans données (candidat avant son premier sondage) y mélangent mal (r_hat 1,25).
    """
    inp = ModelInputs.from_data(data, priors, centered)
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    settings: dict = {
        "method": method,
        "seed": seed,
        "centered": centered,
        "step_days": data.step_days,
        "priors": asdict(inp.priors),
    }
    if method == "nuts":
        chain_method = "parallel" if jax.local_device_count() >= num_chains else "sequential"
        kernel = NUTS(
            model,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
            init_strategy=init_to_uniform(radius=1.0),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
        )
        mcmc.run(key, inp, extra_fields=("diverging", "num_steps"))
        by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        extra = mcmc.get_extra_fields(group_by_chain=True)
        diag = {
            "sites": site_diagnostics(by_chain),
            "divergences": int(np.asarray(extra["diverging"]).sum()),
            "num_steps_mean": float(np.asarray(extra["num_steps"]).mean()),
            "num_steps_max": int(np.asarray(extra["num_steps"]).max()),
        }
        sites = diag["sites"]
        diag["r_hat_max"] = max(v["r_hat_max"] for v in sites.values())
        diag["ess_min"] = min(v["ess_min"] for v in sites.values())
        samples = {k: v.reshape(-1, *v.shape[2:]) for k, v in by_chain.items()}
        settings |= {
            "num_chains": num_chains,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "target_accept": target_accept,
            "max_tree_depth": max_tree_depth,
            "chain_method": chain_method,
        }
    elif method == "svi":
        guide = AutoNormal(model, init_scale=0.05)
        svi = SVI(model, guide, numpyro.optim.Adam(svi_lr), Trace_ELBO())
        res = svi.run(key, svi_steps, inp, progress_bar=progress_bar)
        post = guide.sample_posterior(
            jax.random.PRNGKey(seed + 1), res.params, sample_shape=(num_samples,)
        )
        samples = {k: np.asarray(v) for k, v in post.items()}
        num_chains = 1
        losses = np.asarray(res.losses)
        diag = {"elbo_final": float(-losses[-100:].mean()), "elbo_start": float(-losses[0])}
        settings |= {"svi_steps": svi_steps, "svi_lr": svi_lr, "num_samples": num_samples}
    else:
        raise ValueError(f"méthode inconnue : {method}")
    jax.block_until_ready(samples)
    return FitResult(
        method=method,
        samples=samples,
        num_chains=num_chains,
        settings=settings,
        diagnostics=diag,
        duration_s=time.perf_counter() - t0,
    )
