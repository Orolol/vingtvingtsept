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
        )

    @property
    def D(self) -> int:
        return self.Qd.shape[1]


# --- transformations déterministes (partagées par le modèle et le postérieur)


def latent(s: dict, inp: ModelInputs) -> dict:
    """Variables latentes à partir des sites échantillonnés (un tirage).

    Rend θ (T, C), β (T, B), α (P, C) et γ (n_pop, C).
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
        delta_steps = s["sigma_delta"] * inp.step_scale * (s["z_delta"] @ inp.M.T)
        delta = delta0 + jnp.concatenate([jnp.zeros((1, inp.C)), jnp.cumsum(delta_steps, axis=0)])

    theta = beta[:, inp.cand_bloc] + delta

    house_bloc = s["sigma_house_bloc"] * _center(_center(s["z_house_bloc"], 0), 1)
    house_cand = s["sigma_house_cand"] * _center(s["z_house_cand"] @ inp.M.T, 0)
    alpha = house_bloc[:, inp.cand_bloc] + house_cand

    if inp.n_pop > 1:
        g = pr.pop_sd * _center(s["z_pop"], -1)
        gamma = jnp.concatenate([jnp.zeros((1, inp.B)), g])[:, inp.cand_bloc]
    else:
        gamma = jnp.zeros((1, inp.C))
    return {"theta": theta, "beta": beta, "alpha": alpha, "gamma": gamma, "delta": delta}


def masked_log_softmax(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """log-softmax sur les seules cases `mask` (les autres valent 0, à ignorer)."""
    x = jnp.where(mask, x, -1e9)
    m = jax.lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
    z = jnp.where(mask, jnp.exp(x - m), 0.0)
    out = x - m - jnp.log(z.sum(axis=-1, keepdims=True))
    return jnp.where(mask, out, 0.0)


def hypothesis_probs(lat: dict, inp: ModelInputs) -> jnp.ndarray:
    """π_i (N, K) : softmax restreint aux candidats de chaque hypothèse, effets inclus."""
    ci = inp.cand_idx
    logits = (
        lat["theta"][inp.week[:, None], ci]
        + lat["alpha"][inp.pollster_idx[:, None], ci]
        + lat["gamma"][inp.pop_idx[:, None], ci]
    )
    return jnp.where(inp.mask, jnp.exp(masked_log_softmax(logits, inp.mask)), 0.0)


def concentration(f: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """κ tel que la Dirichlet-multinomiale d'effectif n ait la variance d'une multinomiale f·n."""
    return jnp.maximum(f * n - 1.0, 1e-3) / (1.0 - f)


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
    lat = latent(s, inp)
    probs = hypothesis_probs(lat, inp)
    kappa = concentration(design, inp.n_nominal)
    ll = dirmult_loglik(inp.counts, probs, inp.mask, inp.n_nominal, kappa)
    numpyro.factor("polls", jnp.sum(inp.weight * ll))


# --- simulation (tests, contrôles prédictifs)


def simulate_counts(
    inp: ModelInputs, lat: dict, design: float, rng: np.random.Generator
) -> np.ndarray:
    """Tire des comptes de sondages selon la vraisemblance du modèle, à latents donnés."""
    probs = np.asarray(hypothesis_probs(lat, inp))
    n = np.asarray(inp.n_nominal)
    kappa = np.asarray(concentration(jnp.asarray(design), inp.n_nominal))
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
