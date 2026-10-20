"""Modèle de premier tour : softmax restreint, identification, retrouvaille de paramètres connus."""

from __future__ import annotations

import datetime as dt

import jax.numpy as jnp
import numpy as np
import pytest

from vvs.model import first_round as fr
from vvs.model.data import PollData
from vvs.model.posterior import Posterior


def test_masked_softmax_is_restricted_and_iia():
    x = jnp.array([[1.0, 0.5, -1.0, 2.0], [1.0, 0.5, 2.0, 0.0]])
    mask = jnp.array([[True, True, True, False], [True, True, False, False]])
    p = np.exp(np.asarray(fr.masked_log_softmax(x, mask))) * np.asarray(mask)
    assert p.sum(1) == pytest.approx([1.0, 1.0])
    assert p[0, 3] == 0 and p[1, 2] == 0
    sub = np.exp([1.0, 0.5, -1.0]) / np.exp([1.0, 0.5, -1.0]).sum()
    assert p[0, :3] == pytest.approx(sub, rel=1e-6)
    # IIA : le rapport entre deux candidats ne dépend pas des autres candidats testés
    assert p[0, 0] / p[0, 1] == pytest.approx(p[1, 0] / p[1, 1], rel=1e-6)


def test_dirichlet_multinomial_matches_design_variance():
    """κ = (f n − 1)/(1 − f) donne la variance d'une multinomiale d'effectif f · n."""
    f, n, p = 0.6, 1500.0, 0.2
    kappa = float(fr.concentration(jnp.asarray(f), jnp.asarray(n)))
    var_share = p * (1 - p) / n * (n + kappa) / (1 + kappa)
    assert var_share == pytest.approx(p * (1 - p) / (f * n), rel=1e-6)  # float32


def test_zero_sum_basis_is_orthonormal_within_groups():
    groups = np.array([0, 0, 1, 2, 2, 2])
    Q = fr._zero_sum_basis(groups)
    assert Q.shape == (6, 3)
    assert Q.T @ Q == pytest.approx(np.eye(3))
    for g in np.unique(groups):
        assert Q[groups == g].sum(0) == pytest.approx(np.zeros(3), abs=1e-12)


# --- données synthétiques générées par le modèle

BLOCS = ["far_right", "right", "centre", "left"]
CANDS = ["a_fr", "b_fr", "c_r", "d_c", "e_l", "f_l"]
CAND_BLOC = np.array([0, 0, 1, 2, 3, 3])
TRUE = {
    "sigma_beta": 0.04,
    "sigma_delta": 0.05,
    "sigma_house_bloc": 0.12,
    "sigma_house_cand": 0.03,
    "design": 0.6,
}


def synthetic(seed: int = 1, n_hyp: int = 160, T: int = 21, P: int = 4):
    """Tire des latents avec des paramètres connus puis des sondages selon la vraisemblance.

    Une hypothèse sur trois omet un candidat (b_fr ou f_l) pour exercer la restriction.
    """
    rng = np.random.default_rng(seed)
    week = rng.integers(0, T - 1, n_hyp).astype(np.int32)
    K = len(CANDS)
    cand_idx = np.tile(np.arange(K, dtype=np.int32), (n_hyp, 1))
    mask = np.ones((n_hyp, K), dtype=bool)
    for i in range(n_hyp):
        if i % 3 == 1:
            mask[i, 1] = False
        elif i % 3 == 2:
            mask[i, 5] = False
    cand_idx = np.where(mask, cand_idx, 0)
    size = rng.integers(900, 2000, n_hyp).astype(float)
    prior_mean = np.log(np.array([0.15, 0.15, 0.25, 0.1]))
    data = PollData(
        election=2099,
        election_date=dt.date(2099, 4, 10),
        asof=dt.date(2099, 4, 10),
        candidates=CANDS,
        blocs=BLOCS,
        cand_bloc=CAND_BLOC,
        pollsters=[f"p{j}" for j in range(P)],
        populations=["unknown"],
        n_weeks=T,
        asof_week=T - 1,
        cand_idx=cand_idx,
        mask=mask,
        counts=np.zeros((n_hyp, K)),
        n_nominal=size,
        weight=np.ones(n_hyp),
        week=week,
        pollster_idx=rng.integers(0, P, n_hyp).astype(np.int32),
        pop_idx=np.zeros(n_hyp, dtype=np.int32),
        poll_id=[f"poll{i}" for i in range(n_hyp)],
        hypothesis_id=["h"] * n_hyp,
        shares=np.zeros((n_hyp, K)),
        prior_bloc_mean=prior_mean,
        reference=CANDS,
    )
    inp = fr.ModelInputs.from_data(data, centered=False)
    s = {k: v for k, v in TRUE.items() if k.startswith("sigma")}
    B, C = len(BLOCS), K
    s |= {
        "z_beta0": jnp.asarray(rng.normal(size=B) * 0.3),
        "z_beta": jnp.asarray(rng.normal(size=(T - 1, B))),
        "z_delta0": jnp.asarray(rng.normal(size=C) * 0.5),
        "z_delta": jnp.asarray(rng.normal(size=(T - 1, C))),
        "z_house_bloc": jnp.asarray(rng.normal(size=(P, B))),
        "z_house_cand": jnp.asarray(rng.normal(size=(P, C))),
    }
    lat = fr.latent(s, inp)
    counts = fr.simulate_counts(inp, lat, TRUE["design"], rng)
    data.counts = counts
    data.n_nominal = counts.sum(1)
    data.shares = counts / counts.sum(1, keepdims=True)
    return data, lat


def _check_recovery(data, lat, post: Posterior, tol_mean: float, tol_max: float):
    true_theta = np.asarray(lat["theta"])
    true_sh = np.exp(true_theta) / np.exp(true_theta).sum(-1, keepdims=True)
    est = np.median(post.shares(list(range(len(CANDS)))), axis=0)
    weeks = np.unique(data.week)
    err = 100 * np.abs(est[weeks] - true_sh[weeks])
    assert err.mean() < tol_mean, err.mean()
    assert err.max() < tol_max, err.max()
    # effets maison : le signe des effets marqués est retrouvé
    alpha_true = np.asarray(lat["alpha"])
    alpha_est = np.median(post.alpha, axis=0)
    big = np.abs(alpha_true) > 0.08
    assert (np.sign(alpha_est[big]) == np.sign(alpha_true[big])).mean() > 0.8


def test_recovers_synthetic_trajectories_svi():
    data, lat = synthetic()
    res = fr.fit(data, method="svi", svi_steps=2500, num_samples=400, seed=0)
    _check_recovery(data, lat, Posterior(data, res), tol_mean=0.8, tol_max=3.0)


@pytest.mark.slow
def test_recovers_synthetic_trajectories_nuts():
    data, lat = synthetic()
    res = fr.fit(data, num_chains=2, num_warmup=300, num_samples=300, seed=0)
    assert res.diagnostics["divergences"] < 10
    post = Posterior(data, res)
    _check_recovery(data, lat, post, tol_mean=0.8, tol_max=3.0)
    assert 0.4 < float(np.median(res.samples["design"])) < 0.8
    # couverture : la vraie part est dans l'intervalle à 90 % la plupart du temps
    true_theta = np.asarray(lat["theta"])
    true_sh = np.exp(true_theta) / np.exp(true_theta).sum(-1, keepdims=True)
    sh = post.shares(list(range(len(CANDS))))
    lo, hi = np.quantile(sh, [0.05, 0.95], axis=0)
    weeks = np.unique(data.week)
    inside = (lo[weeks] <= true_sh[weeks]) & (true_sh[weeks] <= hi[weeks])
    assert inside.mean() > 0.75


@pytest.mark.parametrize("centered", [False, True])
def test_model_branches_give_finite_density(centered):
    """Paramétrisations centrée et non centrée, effet population actif (deux populations)."""
    import jax
    from numpyro import handlers
    from numpyro.infer.util import log_density

    data, _ = synthetic(n_hyp=30, T=6)
    data.populations = ["unknown", "registered"]
    data.pop_idx = (np.arange(data.n_hyp) % 2).astype(np.int32)
    inp = fr.ModelInputs.from_data(data, centered=centered)
    # les états centrés (loi impropre) ne se tirent pas : on les fixe
    fixed = {"u_beta": jnp.zeros((inp.T, inp.B - 1)), "u_delta": jnp.zeros((inp.T, inp.D))}
    seeded = handlers.seed(handlers.substitute(fr.model, data=fixed), jax.random.PRNGKey(0))
    tr = handlers.trace(seeded).get_trace(inp)
    assert "z_pop" in tr and tr["z_pop"]["value"].shape == (1, len(BLOCS))
    assert ("u_beta" in tr) == centered
    params = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}
    lp, _ = log_density(fr.model, (inp,), {}, params)
    assert np.isfinite(float(lp))
    lat = fr.latent(params, inp)
    # identification : β somme constante sur les blocs, δ somme nulle dans chaque bloc
    beta_sum = np.asarray(lat["beta"]).sum(1)
    assert beta_sum == pytest.approx(np.full(data.n_weeks, beta_sum[0]), abs=1e-4)
    delta = np.asarray(lat["delta"])
    for b in range(len(BLOCS)):
        assert delta[:, CAND_BLOC == b].sum(1) == pytest.approx(np.zeros(data.n_weeks), abs=1e-4)
