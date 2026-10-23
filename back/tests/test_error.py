"""Modèle d'erreur η(τ) : covariance structurée, tirages, estimation sur données jouets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvs.model import error as er

BLOCS = ["far_right", "far_right", "right", "centre", "left", "left", "far_left"]
P = np.array([0.20, 0.08, 0.15, 0.25, 0.12, 0.05, 0.15])


def params(nu=math.inf, bias=None) -> er.ErrorParams:
    return er.ErrorParams(
        sigma_bloc_final=0.06,
        sigma_cand_final=0.08,
        rho=1e-3,
        a_bloc=0.05,
        a_cand=0.04,
        gamma=0.5,
        nu=nu,
        final_age=7.0,
        bloc_weight={"far_right": 1.5, "left": 0.8},
        bias=bias or {},
    )


@pytest.mark.parametrize("h", [7, 30, 180])
def test_covariance_positive_definite_and_block_structured(h):
    m = er.ErrorModel(params(nu=8.0))
    cov = m.covariance(h, BLOCS, P)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    # même bloc : corrélation positive ; blocs différents : nulle
    assert corr[0, 1] > 0.1 and corr[4, 5] > 0.1
    assert corr[0, 2] == 0 and corr[3, 6] == 0
    # variance croissante avec l'horizon
    assert np.all(np.diag(m.covariance(h + 60, BLOCS, P)) > np.diag(cov))


@pytest.mark.parametrize("nu", [math.inf, 8.0])
def test_sampled_eta_recovers_covariance(nu):
    m = er.ErrorModel(params(nu=nu))
    rng = np.random.default_rng(1)
    n = 200_000
    eta = m.sample_eta(90, BLOCS, np.tile(P, (n, 1)), rng)
    emp = np.cov(eta, rowvar=False)
    cov = m.covariance(90, BLOCS, P)
    assert np.diag(emp) == pytest.approx(np.diag(cov), rel=0.03)
    assert emp[0, 1] == pytest.approx(cov[0, 1], rel=0.06)
    assert abs(emp[0, 2]) < 0.002


def test_sample_returns_shares_and_bias_shifts_bloc():
    rng = np.random.default_rng(2)
    theta = np.log(P)[None, :]
    cands = [f"c{i}" for i in range(len(P))]
    out = er.ErrorModel(params()).sample(theta, 30, cands, BLOCS, rng, n=5000)
    assert out.shape == (5000, len(P))
    assert out.sum(1) == pytest.approx(np.ones(5000))
    assert np.median(out, 0) == pytest.approx(P, rel=0.15)
    biased = er.ErrorModel(params(bias={"right": 0.3})).sample(theta, 30, cands, BLOCS, rng, 5000)
    assert np.median(biased[:, 2]) > 1.2 * np.median(out[:, 2])


def test_contrast_likelihood_ignores_common_shift():
    rng = np.random.default_rng(3)
    d = rng.normal(size=5)
    cov = np.diag([0.1, 0.2, 0.1, 0.3, 0.2]) + 0.05
    for nu in (math.inf, 6.0):
        assert er._contrast_logpdf(d + 2.7, cov, nu) == pytest.approx(
            er._contrast_logpdf(d, cov, nu)
        )


def toy_items(n_elections: int, true: er.ErrorParams, seed: int = 0) -> list[er.Item]:
    """Erreurs finales et mouvements tirés du modèle lui-même."""
    rng = np.random.default_rng(seed)
    items = []
    for e in range(n_elections):
        fin_cov = er.item_cov(
            er.Item(e, "final", 0.0, [], BLOCS, P, np.zeros(len(P))),
            true.sigma_bloc_final**2, true.sigma_cand_final**2, true.rho,
            true.a_bloc, true.a_cand, true.bloc_weight | {b: 1.0 for b in er.BLOCS
                                                          if b not in true.bloc_weight},
            true.gamma,
        )
        d = rng.multivariate_normal(np.zeros(len(P)), fin_cov)
        items.append(er.Item(e, "final", 0.0, list(map(str, range(len(P)))), BLOCS, P, d))
        moves = []
        for h in (30, 90, 180):
            it = er.Item(e, "move", h, list(map(str, range(len(P)))), BLOCS, P, np.zeros(len(P)))
            w = {b: true.bloc_weight.get(b, 1.0) for b in er.BLOCS}
            cov = er.item_cov(it, 0, 0, true.rho, true.a_bloc, true.a_cand, w, true.gamma)
            it.d = rng.multivariate_normal(np.zeros(len(P)), cov)
            moves.append(it)
        for it in moves:
            it.weight = 1 / len(moves)
        items += moves
    return items


def test_fit_recovers_scales_on_toy_elections():
    true = params()
    true.bloc_weight = {}
    items = toy_items(60, true)
    m = er.ErrorModel.fit(elections=range(60), items=items, nu=math.inf, gamma=true.gamma,
                          use_bias=False)
    p = m.params
    # σ_Cf et ρ sont peu séparables (ρ/p varie peu) : on contrôle la variance propre implicite
    for share in (0.05, 0.15, 0.25):
        assert p.sigma_cand_final**2 + p.rho / share == pytest.approx(
            true.sigma_cand_final**2 + true.rho / share, rel=0.3
        )
    assert p.a_cand == pytest.approx(true.a_cand, rel=0.4)
    assert p.a_bloc == pytest.approx(true.a_bloc, rel=0.5)
    # leave-one-election-out : l'élection exclue ne sert pas
    m2 = er.ErrorModel.fit(elections=range(60), exclude=[0], items=items, nu=math.inf,
                           gamma=true.gamma, use_bias=False)
    assert m2.info["train"] == list(range(1, 60))


def _final(e, d_bloc_right):
    blocs = ["right", "left", "centre"]
    p = np.array([0.3, 0.3, 0.4])
    d = np.array([d_bloc_right, -0.02, 0.0])
    return er.Item(e, "final", 0.0, ["a", "b", "c"], blocs, p, d)


def test_bias_admitted_only_if_stable_and_shrunk():
    stable = [_final(e, 0.08 + 0.01 * e) for e in range(4)]
    bias, diag = er.stable_bias(stable)
    assert "right" in bias and 0 < bias["right"] < diag["right"]["mean"]
    mixed = [_final(e, s) for e, s in enumerate([0.1, -0.05, 0.08, 0.06])]
    bias, diag = er.stable_bias(mixed)
    assert "right" not in bias and not diag["right"]["same_sign"]
    # deux élections seulement : jamais de biais
    bias, _ = er.stable_bias(stable[:2])
    assert bias == {}


@pytest.mark.parametrize("nu", [math.inf, 8.0])
def test_sample_eta_parts_matches_sample_eta_and_exposes_bloc_component(nu):
    """Mêmes tirages que sample_eta (même graine), plus la composante de bloc réduite."""
    m = er.ErrorModel(params(nu=nu))
    n = 50_000
    p = np.tile(P, (n, 1))
    eta = m.sample_eta(90, BLOCS, p, np.random.default_rng(3))
    parts = m.sample_eta_parts(90, BLOCS, p, np.random.default_rng(3))
    np.testing.assert_array_equal(parts.eta, eta)
    z = parts.bloc("far_right")
    assert parts.bloc("other") is None
    assert z.mean() == pytest.approx(0, abs=0.02) and z.std() == pytest.approx(1, rel=0.02)
    # η des candidats far_right = λ (u + v) : corrélé à z ; les autres blocs non
    vb, vc = m.variances(90, BLOCS, P)
    expected = math.sqrt(vb[0] / (vb[0] + vc[0]))
    if math.isinf(nu):
        assert np.corrcoef(eta[:, 0], z)[0, 1] == pytest.approx(expected, abs=0.02)
    else:
        assert np.corrcoef(eta[:, 0], z)[0, 1] > 0.5 * expected
    assert abs(np.corrcoef(eta[:, 2], z)[0, 1]) < 0.02


# --- scrutins de liste (extra_items, with_lists) et biais d'extrême droite depuis 2017


def _toy_with_extra(weight: float, seed: int = 0):
    """Présidentielles jouets 0..19 et dix « scrutins de liste » dont l'erreur finale propre
    est quatre fois plus dispersée."""
    true = params()
    true.bloc_weight = {}
    items = toy_items(20, true, seed=seed)
    rng = np.random.default_rng(seed + 1)
    extra = []
    for k in range(10):
        cov = np.diag(np.full(len(P), (2 * true.sigma_cand_final) ** 2))
        extra.append(er.Item(f"euro{2000 + k}", "final", 0.0, list(map(str, range(len(P)))),
                             BLOCS, P, rng.multivariate_normal(np.zeros(len(P)), cov),
                             weight=weight))
    return true, items, extra


def test_extra_items_weighted_and_never_excluded():
    true, items, extra = _toy_with_extra(1.0)
    kw = {"elections": range(20), "items": items, "nu": math.inf, "gamma": true.gamma,
          "use_bias": False}
    base = er.ErrorModel.fit(**kw)
    sc = {}
    for w in (0.0, 0.5, 1.0):
        for it in extra:
            it.weight = w
        m = er.ErrorModel.fit(**kw, exclude=[3], extra_items=extra)
        # leave-one-election-out : la présidentielle exclue part, les scrutins de liste restent
        assert m.info["train"] == [e for e in range(20) if e != 3]
        assert m.info["extra_elections"] == [it.election for it in extra]
        sc[w] = m.params.sigma_cand_final
    # poids nul : comme sans les items (à l'exclusion près) ; poids croissant : σ_Cf croît
    no_extra = er.ErrorModel.fit(**kw, exclude=[3])
    assert sc[0.0] == pytest.approx(no_extra.params.sigma_cand_final, rel=1e-6)
    assert sc[0.0] < sc[0.5] < sc[1.0]
    assert base.info["extra_elections"] == [] and not base.info["with_lists"]
    # un item supplémentaire ne peut pas porter une élection évaluable
    clash = [er.Item(5, "final", 0.0, ["a", "b", "c"], ["left", "right", "centre"],
                     np.array([0.3, 0.3, 0.4]), np.zeros(3))]
    with pytest.raises(ValueError):
        er.ErrorModel.fit(**kw, extra_items=clash)


def test_with_lists_false_is_current_behaviour_and_true_adds_list_finals():
    true, items, _ = _toy_with_extra(1.0)
    kw = {"elections": range(20), "items": items, "nu": math.inf, "gamma": true.gamma,
          "use_bias": False}
    a = er.ErrorModel.fit(**kw)
    b = er.ErrorModel.fit(**kw, with_lists=False, extra_items=None,
                          far_right_bias_since_2017=False)
    assert a.to_dict()["params"] == b.to_dict()["params"]
    m = er.ErrorModel.fit(**kw, with_lists=True)
    from vvs.model.error_lists import LIST_KEYS, default_list_items

    lst = default_list_items()
    assert [it.election for it in lst] == list(LIST_KEYS)
    assert all(it.kind == "final" and it.weight == 0.5 for it in lst)
    assert m.info["extra_elections"] == list(LIST_KEYS)
    assert m.info["n_items"] == len(items) + len(LIST_KEYS)
    assert m.to_dict()["params"] != a.to_dict()["params"]


def _fr_final(e, d_fr, weight=1.0):
    blocs = ["far_right", "far_right", "right", "left"]
    p = np.array([0.25, 0.05, 0.3, 0.4])
    d = np.array([d_fr, d_fr, 0.0, 0.0])
    return er.Item(e, "final", 0.0, ["a", "b", "c", "d"], blocs, p, d, weight=weight)


def test_far_right_bias_since_2017_estimate_and_shift():
    items = [_fr_final(2012, 0.2), _fr_final(2017, -0.10), _fr_final(2022, -0.12),
             _fr_final("euro2019", -0.08, weight=0.5), _fr_final("euro2014", 0.3, weight=0.5)]
    errs = er.far_right_final_errors(items)
    # log-rapport −0,10 sur un bloc de 30 % : en logit, plus fort en valeur absolue
    assert errs[2017][0] < -0.10
    mu, diag = er.far_right_bias_since(items)
    assert set(diag["errors"]) == {"2017", "2022", "euro2019"}  # rien d'avant 2017
    assert diag["same_sign"] and 0 < diag["shrink"] < 1
    assert mu == pytest.approx(diag["mean"] * diag["shrink"]) and mu < 0
    shifted = er.shift_far_right(items, mu)
    assert np.allclose(shifted[0].d, items[0].d) and np.allclose(shifted[4].d, items[4].d)
    assert np.allclose(shifted[1].d, items[1].d - mu * np.array([1, 1, 0, 0]))
    assert items[1].d[0] == -0.10  # copies : les items d'origine ne bougent pas


def test_far_right_bias_applied_only_to_elections_since_2017():
    items = [_fr_final(e, x) for e, x in ((2007, 0.2), (2012, 0.15), (2017, -0.1), (2022, -0.12))]
    kw = {"elections": (2007, 2012, 2017, 2022), "items": items, "nu": math.inf, "gamma": 0.4,
          "use_bias": False, "far_right_bias_since_2017": True}
    m22 = er.ErrorModel.fit(**kw, exclude=[2022])
    d = m22.info["far_right_bias_since_2017"]
    assert d["applied"] and "2022" not in d["errors"] and "2017" in d["errors"]
    assert m22.params.bias["far_right"] == pytest.approx(d["mu"]) and d["mu"] < 0
    m12 = er.ErrorModel.fit(**kw, exclude=[2012])
    assert not m12.info["far_right_bias_since_2017"]["applied"]
    assert "far_right" not in m12.params.bias
    assert er.ErrorModel.fit(**kw).info["far_right_bias_since_2017"]["applied"]  # prévision
