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


def synthetic(seed: int = 1, n_hyp: int = 160, T: int = 21, P: int = 4, lam=None):
    """Tire des latents avec des paramètres connus puis des sondages selon la vraisemblance.

    Une hypothèse sur trois omet un candidat (b_fr ou f_l) pour exercer la restriction.
    `lam` (B,) : sondages tirés avec la substitution intra-bloc (logit emboîté) de ces λ.
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
    priors = fr.Priors(n_max=None, substitution=lam is not None)
    inp = fr.ModelInputs.from_data(data, priors, centered=False)
    s = {k: v for k, v in TRUE.items() if k.startswith("sigma")}
    if lam is not None:
        s["lam"] = jnp.asarray(lam)
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


# Les sondages synthétiques sont tirés sans plancher d'erreur : on ajuste le modèle qui les a
# générés (`n_max=None`). Avec le plancher, défaut depuis la phase 4, le SVI de la graine 0
# s'arrête sur un optimum local (ELBO −4289 contre −3482 pour la graine 1).
GEN_PRIORS = fr.Priors(n_max=None)


def test_recovers_synthetic_trajectories_svi():
    data, lat = synthetic()
    res = fr.fit(data, method="svi", svi_steps=2500, num_samples=400, seed=0, priors=GEN_PRIORS)
    _check_recovery(data, lat, Posterior(data, res), tol_mean=0.8, tol_max=3.0)


@pytest.mark.slow
def test_recovers_synthetic_trajectories_nuts():
    data, lat = synthetic()
    res = fr.fit(data, num_chains=2, num_warmup=300, num_samples=300, seed=0, priors=GEN_PRIORS)
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


# --- substitution intra-bloc (logit emboîté par bloc)


def _random_hypotheses(seed=0, N=7, K=5, B=3):
    rng = np.random.default_rng(seed)
    bloc_term = jnp.asarray(rng.normal(size=(N, B)))
    cand_term = jnp.asarray(rng.normal(size=(N, K)))
    slot_bloc = jnp.asarray(rng.integers(0, B, size=(N, K)))
    mask = jnp.asarray(rng.random((N, K)) < 0.75).at[:, :2].set(True)
    return bloc_term, cand_term, slot_bloc, mask


def test_nested_logit_reduces_to_iia_at_lambda_one():
    bloc_term, cand_term, slot_bloc, mask = _random_hypotheses()
    nested = fr.nested_log_probs(bloc_term, cand_term, slot_bloc, mask, jnp.ones(3))
    theta = jnp.take_along_axis(bloc_term, slot_bloc, axis=1) + cand_term
    iia = fr.masked_log_softmax(theta, mask)
    assert np.asarray(nested) == pytest.approx(np.asarray(iia), abs=1e-5)
    # même chose pour les parts d'une configuration complète (postérieur)
    beta, delta = np.array([0.3, -0.2, 0.1]), np.array([0.5, -0.5, 0.2, 0.0])
    cb = np.array([0, 0, 2, 1])
    ref = beta[cb] + delta
    ref = ref - np.log(np.exp(ref).sum())
    assert fr.config_log_shares(beta, delta, cb, np.ones(3)) == pytest.approx(ref)
    assert fr.config_log_shares(beta, delta, cb, None) == pytest.approx(ref)


def test_hypothesis_probs_with_substitution_matches_iia_when_lambda_is_one():
    """Câblage du modèle : effets maison et population passent par les bons niveaux."""
    import jax
    from numpyro import handlers

    data, _ = synthetic(n_hyp=30, T=6)
    data.populations = ["unknown", "registered"]
    data.pop_idx = (np.arange(data.n_hyp) % 2).astype(np.int32)
    inp = fr.ModelInputs.from_data(data)
    inp_s = fr.ModelInputs.from_data(data, fr.Priors(substitution=True))
    tr = handlers.trace(handlers.seed(fr.model, jax.random.PRNGKey(1))).get_trace(inp_s)
    s = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}
    assert s["lam"].shape == (len(BLOCS),)
    p_iia = fr.hypothesis_probs(fr.latent(s, inp), inp)
    p_one = fr.hypothesis_probs(fr.latent(s | {"lam": jnp.ones(len(BLOCS))}, inp_s), inp_s)
    assert np.asarray(p_one) == pytest.approx(np.asarray(p_iia), abs=1e-5)
    p_sub = fr.hypothesis_probs(fr.latent(s, inp_s), inp_s)
    assert np.asarray(p_sub).sum(1) == pytest.approx(np.ones(data.n_hyp), abs=1e-5)


def test_substitution_redistributes_within_bloc():
    """Bloc A (a1, a2), bloc B (b) ; on retire a2. Parts complètes : a1 0,3, a2 0,2, b 0,5."""
    full = np.array([0.3, 0.2, 0.5])
    cb = np.array([0, 0, 1])
    delta = np.log(full)
    delta[:2] -= delta[:2].mean()
    delta[2] = 0.0  # δ centré dans chaque bloc
    beta = np.array([np.log(full[:2]).mean(), np.log(0.5)])
    for lam_a, expected_a1 in [
        (1.0, lambda pf: 0.3 / 0.8),  # IIA : a2 se répartit au prorata de tous
        (0.0, lambda pf: pf[:2].sum()),  # le bloc A garde toute sa part, a1 la prend
        (0.5, None),  # entre les deux : forme fermée P^(1−λ) Q^λ ci-dessous
    ]:
        lam = np.array([lam_a, 1.0])
        p_full = np.exp(fr.config_log_shares(beta, delta, cb, lam))
        if lam_a == 1.0:
            assert p_full == pytest.approx(full)
        p = np.exp(fr.config_log_shares(beta, delta[[0, 2]], cb[[0, 2]], lam))
        assert p.sum() == pytest.approx(1.0)
        # la forme fermée suppose les parts complètes au λ testé
        P_A, Q_A = p_full[:2].sum(), p_full[0]
        w = P_A ** (1 - lam_a) * Q_A**lam_a
        assert p[0] == pytest.approx(w / (w + p_full[2]))
        if expected_a1 is not None:
            assert p[0] == pytest.approx(expected_a1(p_full))
        if lam_a < 1:  # b gagne moins qu'à l'IIA (rien à λ = 0)
            assert p_full[2] - 1e-12 <= p[1] < p_full[2] / (1 - p_full[1])
    # en jax, sur une hypothèse masquée : mêmes parts que la configuration réduite
    lam = np.array([0.0, 1.0])
    lp = fr.nested_log_probs(
        jnp.asarray(beta)[None], jnp.asarray(delta)[None], jnp.asarray(cb)[None],
        jnp.array([[True, False, True]]), jnp.asarray(lam),
    )
    ref = np.exp(fr.config_log_shares(beta, delta[[0, 2]], cb[[0, 2]], lam))
    assert np.exp(np.asarray(lp))[0, [0, 2]] == pytest.approx(ref, abs=1e-6)


LAM_TRUE = np.array([0.15, 1.0, 1.0, 0.25])  # far_right et left substituables


def test_posterior_config_shares_arbitrary_configuration():
    data, _ = synthetic(n_hyp=40, T=8, lam=LAM_TRUE)
    for subst in (False, True):
        res = fr.fit(data, method="svi", svi_steps=300, num_samples=50, seed=0,
                     priors=fr.Priors(substitution=subst))
        post = Posterior(data, res)
        assert post.substitution == subst
        config = ["e_l", "a_fr", "d_c"]
        sh = post.config_shares(config, week=data.asof_week)
        assert sh.shape == (50, 3) and sh.sum(1) == pytest.approx(np.ones(50), abs=1e-6)
        idx = [CANDS.index(c) for c in config]
        if subst:
            expected = np.exp(fr.config_log_shares(
                post.beta[:, data.asof_week], post.delta[:, data.asof_week][:, idx],
                CAND_BLOC[idx], post.lam,
            ))
        else:
            th = post.theta[:, data.asof_week][:, idx]
            expected = np.exp(th) / np.exp(th).sum(1, keepdims=True)
        assert sh == pytest.approx(expected, rel=1e-5)
        assert post.config_shares(config).shape == (50, data.n_weeks, 3)
        with pytest.raises(KeyError):
            post.config_shares(["inconnu"])


def test_svi_recovers_known_substitution():
    """Sondages tirés avec λ connus (far_right et left substituables) : l'ajustement avec
    substitution retrouve des λ bas pour ces blocs et prédit mieux l'hypothèse réduite que
    l'IIA."""
    data, lat = synthetic(n_hyp=240, T=12, lam=LAM_TRUE)
    res = fr.fit(data, method="svi", svi_steps=2500, num_samples=300, seed=0,
                 priors=fr.Priors(n_max=None, substitution=True))
    lam = np.median(res.samples["lam"], axis=0)
    assert lam[0] < 0.45 and lam[3] < 0.5, lam
    # parts de la semaine finale sans b_fr ni f_l : substitution contre IIA
    true_lat = dict(lat) | {"lam": jnp.asarray(LAM_TRUE)}
    cfg = [0, 2, 3, 4]
    w = data.n_weeks - 1
    truth = np.exp(fr.config_log_shares(
        np.asarray(true_lat["beta"])[w], np.asarray(true_lat["delta"])[w][cfg], CAND_BLOC[cfg],
        LAM_TRUE,
    ))
    sub = np.median(Posterior(data, res).shares(cfg, w), axis=0)
    res_iia = fr.fit(data, method="svi", svi_steps=2500, num_samples=300, seed=0,
                     priors=fr.Priors(n_max=None))
    iia = np.median(Posterior(data, res_iia).shares(cfg, w), axis=0)
    err_sub, err_iia = np.abs(sub - truth).sum(), np.abs(iia - truth).sum()
    assert err_sub < 0.03, (sub, truth)
    assert err_sub < err_iia, (err_sub, err_iia)
