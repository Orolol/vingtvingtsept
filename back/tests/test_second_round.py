"""Modèle de second tour par reports (vvs.model.second_round) : données jouets, puis
ajustements réels marqués slow."""

import datetime as dt

import numpy as np
import pytest

from vvs.model.second_round import (
    ABSTAIN,
    DuelPolls,
    ErrorParams,
    GroupPrior,
    PairFit,
    analog_prior,
    duel_share,
    proximity_mean,
    sample_pair,
)

ASOF = dt.date(2027, 4, 1)
BLOCS = {"a": "left", "b": "right", "l2": "left", "r2": "right", "x": "far_right"}
ERR = ErrorParams(sigma0=2.0, sigma1=1.0, bias_far_right=0.0, sigma_house=0.5)


def _fit(polls: DuelPolls | None = None, kappa: float = 30.0, errors: ErrorParams = ERR):
    groups = {
        "own:a": GroupPrior("own:a", (0.95, 0.02, 0.03), 200.0, False, "own"),
        "own:b": GroupPrior("own:b", (0.02, 0.95, 0.03), 200.0, False, "own"),
        "bloc:left": GroupPrior("bloc:left", (0.7, 0.1, 0.2), kappa, True, "analog"),
        "bloc:right": GroupPrior("bloc:right", (0.1, 0.7, 0.2), kappa, True, "analog"),
        "bloc:far_right": GroupPrior("bloc:far_right", (0.1, 0.6, 0.3), kappa, True, "analog"),
        ABSTAIN: GroupPrior(ABSTAIN, (0.2, 0.2, 0.6), kappa, True, "analog"),
    }
    return PairFit(
        election=2027,
        asof=ASOF,
        pair=("a", "b"),
        blocs=("left", "right"),
        groups=groups,
        sigma_u=0.2,
        sigma_v=0.2,
        polls=polls or DuelPolls.empty(),
        errors=errors,
        gap_days=14,
        abstain_weight=0.3,
        bloc_of=lambda c: BLOCS[c],
    )


FIRST = {"a": np.array([0.3]), "b": np.array([0.3]), "l2": np.array([0.1]),
         "r2": np.array([0.2]), "x": np.array([0.1])}


def _polls(share: float, k: int = 10) -> DuelPolls:
    return DuelPolls(
        poll_id=[f"p{i}" for i in range(k)],
        pollster=[f"inst{i % 3}" for i in range(k)],
        fieldwork_end=[ASOF - dt.timedelta(days=i) for i in range(k)],
        share=np.full(k, share),
        sample_size=np.full(k, 1000.0),
    )


def test_duel_share_renormalises_on_expressed():
    # deux électorats de poids 0,6 et 0,4 ; le second s'abstient à moitié
    w = np.array([[0.6, 0.4]])
    t = np.array([[[0.9, 0.1, 0.0], [0.1, 0.4, 0.5]]])
    expected = 100 * (0.54 + 0.04) / (0.54 + 0.04 + 0.06 + 0.16)
    assert duel_share(w, t)[0] == pytest.approx(expected)
    # l'abstention totale d'un électorat ne change rien aux autres
    t2 = np.array([[[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]])
    assert duel_share(w, t2)[0] == pytest.approx(90.0)


def test_groups_by_bloc_and_abstain_weight():
    d = sample_pair(_fit(), FIRST, 10, np.random.default_rng(0), n=200)
    assert d.groups == ["own:a", "own:b", "bloc:left", "bloc:right", "bloc:far_right", ABSTAIN]
    # l2 et r2 forment chacun leur bloc ; l'abstention garde son poids propre
    np.testing.assert_allclose(d.weights[0], [0.3, 0.3, 0.1, 0.2, 0.1, 0.3])


def test_prior_only_matches_prior_mean():
    fit = _fit(kappa=1e4)
    fit.sigma_u = fit.sigma_v = 1e-6
    d = sample_pair(fit, FIRST, 10, np.random.default_rng(1), n=4000)
    w = np.array([[0.3, 0.3, 0.1, 0.2, 0.1, 0.3]])
    t = np.array([[g.mean for g in fit.groups.values()]])
    # ordre des groupes du fit = ordre de d.groups
    assert list(fit.groups) == d.groups
    assert np.median(d.share) == pytest.approx(duel_share(w, t)[0], abs=0.2)
    assert d.polls is None and d.ess == pytest.approx(len(d.prior_share))


def test_polls_pull_posterior():
    rng = np.random.default_rng(2)
    prior = sample_pair(_fit(), FIRST, 10, rng, n=4000)
    target = np.median(prior.share) + 8
    post = sample_pair(_fit(_polls(target)), FIRST, 10, rng, n=4000)
    assert np.median(prior.share) < np.median(post.share) < target
    assert np.std(post.share) < np.std(prior.share)
    # reports a posteriori déplacés vers a
    assert post.transfers[:, 2, 0].mean() > prior.transfers[:, 2, 0].mean()


def test_far_right_bias_is_removed_from_polls():
    fit = _fit(_polls(55.0), errors=ErrorParams(2.0, 1.0, bias_far_right=3.0))
    fit.blocs = ("far_right", "right")
    assert fit.poll_summary(10)[2] == pytest.approx(3.0)
    assert fit.swapped().poll_summary(10)[2] == pytest.approx(-3.0)
    fit.use_bias = False
    assert fit.poll_summary(10)[2] == 0.0


def test_symmetry():
    fit = _fit(_polls(53.0))
    a = sample_pair(fit, FIRST, 10, np.random.default_rng(3), n=20000)
    # identité exacte sur un même tirage : inverser i et j donne le complément
    np.testing.assert_allclose(
        a.share + duel_share(a.weights, a.transfers[..., [1, 0, 2]]), 100.0
    )
    # paire inversée : même loi au complément près (bruit Monte-Carlo)
    b = sample_pair(fit.swapped(), FIRST, 10, np.random.default_rng(4), n=20000)
    for q in (0.05, 0.5, 0.95):
        assert np.quantile(a.share, q) + np.quantile(b.share, 1 - q) == pytest.approx(100, abs=0.3)
    s, s2 = fit.poll_summary(10), fit.swapped().poll_summary(10)
    assert s[0] + s2[0] == pytest.approx(100.0)
    assert s[1] == pytest.approx(s2[1])


def test_horizon_error_grows_and_samples():
    e = ErrorParams(sigma0=1.5, sigma1=1.2)
    sd = e.sd([0, 30, 180])
    assert sd[0] == pytest.approx(1.5)
    assert sd[2] == pytest.approx(np.sqrt(1.5**2 + 1.2**2 * 6))
    draws = e.sample(180, np.random.default_rng(4), 200_000)
    assert draws.mean() == pytest.approx(0.0, abs=0.03)
    assert draws.std() == pytest.approx(sd[2], rel=0.01)
    # un seul sondage : V = horizon + maison + échantillonnage ; plus vieux = moins informatif
    near = _fit(_polls(50.0, k=1)).poll_summary(-13)[1]
    far = _fit(_polls(50.0, k=1)).poll_summary(180)[1]
    assert near < far
    s2 = (100 * np.sqrt(0.25 / (0.6 * 1000))) ** 2
    assert near == pytest.approx(ERR.variance(1) + ERR.sigma_house**2 + s2)
    # beaucoup de sondages récents : V plafonne à l'erreur commune, pas à 0
    assert _fit(_polls(50.0, k=200)).poll_summary(-13)[1] > ERR.sigma0**2


def test_proximity_mean_is_a_distribution():
    for e in ("far_left", "left", "centre", "right", "far_right", "other", ABSTAIN):
        m = proximity_mean(e, "far_right", "centre")
        assert sum(m) == pytest.approx(1.0)
        assert all(0 < x < 1 for x in m)
    i, j, _ = proximity_mean("left", "centre", "far_right")
    assert i > j


def test_analog_prior_leaves_target_out():
    # centre contre extrême droite : 2017 et 2022 ; en évaluant 2022, seul 2017 reste
    mean, used, _ = analog_prior("far_left", "centre", "far_right", 2022)
    assert used == [2017]
    assert sum(mean) == pytest.approx(1.0)
    assert mean[0] > mean[1]  # Mélenchon 2017 : Macron devant Le Pen
    assert analog_prior("far_left", "far_left", "far_right", 2027) is None


@pytest.mark.slow
def test_real_fit_2022_eve_of_second_round():
    from vvs.model.second_round import SecondRoundModel
    from vvs.registry import default_registry

    reg = default_registry()
    e = reg.elections[2022]
    pair = ("macron_emmanuel", "le_pen_marine")
    m = SecondRoundModel(reg).fit(2022, e.round_(2).date - dt.timedelta(days=1), [pair])
    fr = {r.candidate_id: np.array([r.share / 100]) for r in e.round_(1).results}
    d = m.sample(pair, fr, -13, np.random.default_rng(0), 4000)
    s = m.summary(d)
    assert s["n_polls"] > 100 and s["poll_bias"] < 0  # biais RN retiré (Macron en i)
    assert 52 < s["median"] < 65 and s["p_i_wins"] > 0.95
    assert m.errors is None and 2022 not in m.pair_fit(pair).errors.calibrated_on


@pytest.mark.slow
def test_forecast_2027():
    from vvs.model.second_round import forecast

    pairs, transfers = forecast(2027, dt.date(2026, 10, 20), n=2000, seed=0)
    assert pairs.height >= 10
    lp_mel = pairs.filter(
        (pairs["candidate_i"] == "le_pen_marine") & (pairs["candidate_j"] == "melenchon_jean_luc")
    )
    assert lp_mel["inedit"][0] and lp_mel["p_i_wins"][0] > 0.8
    assert transfers.height > pairs.height
