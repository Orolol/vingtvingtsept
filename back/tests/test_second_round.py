"""Modèle de second tour par reports (vvs.model.second_round) : données jouets, puis
ajustements réels marqués slow."""

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from vvs.model.second_round import (
    ABSTAIN,
    BIAS_NODES,
    DuelPolls,
    ErrorParams,
    GroupPrior,
    PairFit,
    analog_prior,
    bias_curve,
    duel_share,
    match_turnout,
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


# --- options structurelles : biais par horizon, couplage, participation du second tour


def _rn_fit(polls: DuelPolls, errors: ErrorParams = ERR, **kw) -> PairFit:
    """Duel extrême droite (i) contre droite (j), prior large : les sondages dominent."""
    fit = _fit(polls, kappa=4.0, errors=errors)
    fit.blocs = ("far_right", "right")
    fit.sigma_u = fit.sigma_v = 1.0
    for k, v in kw.items():
        setattr(fit, k, v)
    return fit


def test_coupling_recovers_correlation_with_shared_component():
    """Erreur commune de duel corrélée à z (κ = −ρ σ_e, i d'extrême droite) : sous un prior
    large, part | z ≈ N(ȳ − μ − κ z, V − κ²), donc corr(part, z) ≈ −κ / √V ≈ ρ quand
    l'erreur commune domine V ; la loi marginale de la part ne change pas."""
    rho = 0.6
    fit = _rn_fit(_polls(50.0, k=200), coupling_rho=rho)
    m = 20_000
    z = np.random.default_rng(5).standard_normal(m)
    first = {c: np.full(m, v[0]) for c, v in FIRST.items()}
    d = sample_pair(fit, first, -13, np.random.default_rng(6), n=20_000, shared_error=z)
    sd_e = np.sqrt(fit.common_variance(-13))
    var = fit.poll_summary(-13)[1]
    assert d.coupling == pytest.approx(-rho * sd_e)
    assert 0.9 * rho < rho * sd_e / np.sqrt(var) <= rho  # V ≈ σ_e² avec 200 sondages
    # z > 0 : extrême droite au-dessus des sondages du 1er tour, et donc au second
    assert np.corrcoef(d.share, d.shared)[0, 1] == pytest.approx(rho * sd_e / np.sqrt(var), abs=0.05)
    # finaliste d'extrême droite en j : même corrélation avec la part de j
    sw = sample_pair(fit.swapped(), first, -13, np.random.default_rng(6), n=20_000, shared_error=z)
    assert np.corrcoef(100 - sw.share, sw.shared)[0, 1] == pytest.approx(
        np.corrcoef(d.share, d.shared)[0, 1], abs=0.03)
    # loi marginale inchangée
    base = sample_pair(_rn_fit(_polls(50.0, k=200)), first, -13, np.random.default_rng(6), n=20_000)
    assert np.std(d.share) == pytest.approx(np.std(base.share), rel=0.05)
    assert np.mean(d.share) == pytest.approx(np.mean(base.share), abs=0.1)
    # sans finaliste d'extrême droite (biais de signe nul) : pas de couplage
    nofr = _fit(_polls(50.0, k=200))
    nofr.coupling_rho = rho
    assert sample_pair(nofr, first, -13, np.random.default_rng(6), n=1000, shared_error=z).coupling == 0.0


def test_options_off_reduce_to_previous_behaviour():
    """ρ sans composante partagée, composante sans ρ, courbe de biais constante : mêmes
    tirages que le modèle sans option."""
    polls = _polls(55.0)
    ref = sample_pair(_rn_fit(polls), FIRST, 10, np.random.default_rng(11), n=2000)
    a = sample_pair(_rn_fit(polls, coupling_rho=0.5), FIRST, 10, np.random.default_rng(11), n=2000)
    b = sample_pair(_rn_fit(polls), FIRST, 10, np.random.default_rng(11), n=2000, shared_error=1.0)
    flat = replace(ERR, bias_far_right=0.0, bias_curve=((7.0, 0.0), (195.0, 0.0)))
    c = sample_pair(_rn_fit(polls, errors=flat), FIRST, 10, np.random.default_rng(11), n=2000)
    for x in (a, b, c):
        np.testing.assert_array_equal(x.share, ref.share)
    # sans sondage, le couplage ne fait rien non plus
    e0 = sample_pair(_rn_fit(DuelPolls.empty()), FIRST, 10, np.random.default_rng(12), n=500)
    e1 = sample_pair(_rn_fit(DuelPolls.empty(), coupling_rho=0.9), FIRST, 10,
                     np.random.default_rng(12), n=500, shared_error=2.0)
    np.testing.assert_array_equal(e0.share, e1.share)


def _errs(rows):
    return [{"election": e, "age": BIAS_NODES[k], "error": err, "n_polls": n}
            for e, k, err, n in rows]


def test_bias_curve_is_shrunk_monotone_and_zero_without_data():
    rows = [
        (2017, ("mean_14d", 14), 5.0, 22), (2022, ("mean_14d", 14), 4.0, 33),
        (2017, ("horizon", 7), 4.0, 30), (2022, ("horizon", 7), 4.5, 50),
        # non monotone à 45 et 105 jours : 2022 sous-estime puis surestime
        (2017, ("horizon", 30), 4.0, 25), (2022, ("horizon", 30), 0.5, 38),
        (2017, ("horizon", 90), 1.5, 6), (2022, ("horizon", 90), 3.0, 16),
    ]
    curve = bias_curve(_errs(rows), lambda a: 1.5 + 2.0 * a / 30)
    ages, b = zip(*curve)
    assert list(ages) == sorted(set(BIAS_NODES.values()))
    assert all(x >= y - 1e-12 for x, y in zip(b, b[1:]))  # non croissant avec l'âge
    assert b[-1] == 0.0  # aucun sondage à 180 jours : biais nul
    # rétréci : sous la moyenne brute à chaque nœud
    assert b[0] < 4.5 and b[1] < 4.25
    assert 0 < b[2] <= b[1]
    # plus de sondages et d'élections concordantes : moins de rétrécissement
    more = rows + [(2002, ("mean_14d", 14), 4.5, 30)]
    assert bias_curve(_errs(more), lambda a: 1.5 + 2.0 * a / 30)[0][1] > b[0]
    # leave-one-out : skip retire l'élection
    only17 = bias_curve(_errs(rows), lambda a: 1.5 + 2.0 * a / 30, skip={2022})
    assert only17 == bias_curve(_errs([r for r in rows if r[0] == 2017]), lambda a: 1.5 + 2.0 * a / 30)
    # signe négatif : même forme, vers 0 par le bas
    neg = bias_curve(_errs([(e, k, -x, n) for e, k, x, n in rows]), lambda a: 1.5 + 2.0 * a / 30)
    np.testing.assert_allclose([y for _, y in neg], [-y for y in b])


def test_bias_curve_applied_per_poll_age():
    e = ErrorParams(2.0, 1.0, bias_far_right=3.0, bias_curve=((7.0, 3.0), (45.0, 1.0), (195.0, 0.5)))
    np.testing.assert_allclose(e.bias_at([0.0, 7.0, 26.0, 45.0, 120.0, 500.0]), [3.0, 3.0, 2.0, 1.0, 0.75, 0.5])
    np.testing.assert_allclose(e.scaled_bias(0.5).bias_at([7.0, 195.0]), [1.5, 0.25])
    assert ErrorParams(bias_far_right=2.0).bias_at([10.0, 400.0]).tolist() == [2.0, 2.0]
    # biais d'une moyenne de sondages = moyenne des biais pondérée par les moindres carrés,
    # entre le biais du plus récent et celui du plus ancien ; signe selon le finaliste RN
    fit = _rn_fit(_polls(55.0, k=10), errors=e)
    mu = fit.poll_summary(-13)[2]
    ages = np.array([1.0 + i for i in range(10)])
    assert e.bias_at(ages).min() <= mu <= e.bias_at(ages).max()
    assert fit.swapped().poll_summary(-13)[2] == pytest.approx(-mu)
    old = _rn_fit(_polls(55.0, k=10), errors=e)
    assert old.poll_summary(300)[2] == pytest.approx(0.5)  # sondages de plus de 195 jours


def test_match_turnout_hits_target_and_keeps_splits():
    rng = np.random.default_rng(13)
    m, G = 500, 4
    w = np.tile([0.3, 0.25, 0.2, 0.4], (m, 1))
    t = rng.dirichlet([3, 2, 2], size=(m, G))
    t[:, 0], t[:, 1] = (0.94, 0.02, 0.04), (0.02, 0.94, 0.04)  # finalistes fidèles
    shift = np.array([False, False, True, True])
    target = rng.uniform(0.6, 0.8, m)
    out = match_turnout(w, t, shift, target)
    turnout = (w * (out[..., 0] + out[..., 1])).sum(-1) / w.sum(-1)
    np.testing.assert_allclose(turnout, target, atol=1e-6)
    np.testing.assert_allclose(out.sum(-1), 1.0)
    np.testing.assert_allclose(out[:, ~shift], t[:, ~shift])  # finalistes inchangés
    split = lambda x: x[..., 0] / (x[..., 0] + x[..., 1])  # noqa: E731
    np.testing.assert_allclose(split(out), split(t), atol=1e-9)
    # cible inatteignable : ramenée à la borne
    hi = match_turnout(w, t, shift, np.full(m, 0.999))
    assert np.all((w * (hi[..., 0] + hi[..., 1])).sum(-1) / w.sum(-1) < 0.999)


def test_second_round_turnout_drawn_from_its_law():
    mean, sd = 0.9, 0.3  # logit : participation médiane 71 %
    fit = _fit(kappa=30.0)  # sans sondage : la loi tirée est la loi a posteriori
    fit.turnout2 = (mean, sd)
    d = sample_pair(fit, FIRST, 10, np.random.default_rng(14), n=8000)
    lt = np.log(d.turnout / (1 - d.turnout))
    assert np.median(lt) == pytest.approx(mean, abs=0.03)
    assert lt.std() == pytest.approx(sd, rel=0.05)
    # sans l'option : participation implicite des reports, Σ w (t_i + t_j) / Σ w
    imp = sample_pair(_fit(kappa=30.0), FIRST, 10, np.random.default_rng(14), n=2000)
    w, t = imp.weights, imp.transfers
    np.testing.assert_allclose(imp.turnout, (w * (t[..., 0] + t[..., 1])).sum(-1) / w.sum(-1))
    # les reports « ni l'un ni l'autre » s'ajustent, pas la répartition entre i et j
    assert abs(np.median(d.share) - np.median(imp.share)) < 3.0


def test_turnout2_model_uses_second_rounds():
    from vvs.model.second_round import turnout2_model
    from vvs.registry import default_registry

    reg = default_registry()
    t = turnout2_model(reg, exclude={2022})
    assert set(t["elections"]) == {"2002", "2007", "2012", "2017"}
    r2 = reg.elections[2017].round_(2)
    assert t["elections"]["2017"] == pytest.approx(r2.expressed / r2.registered, abs=1e-4)
    # seconds tours, pas premiers tours : 2017 vaut 66 % (premier tour 76 %)
    assert t["elections"]["2017"] < 0.70


@pytest.mark.slow
def test_real_bias_curve_and_coupling():
    from vvs.model.second_round import calibrate_errors, far_right_coupling

    e = calibrate_errors({2027}, bias_by_horizon=True)
    ages, b = zip(*e.bias_curve)
    assert all(x >= y for x, y in zip(b, b[1:])) and b[0] > 2.0 and b[-1] < b[0]
    c = far_right_coupling({2027})
    assert c["elections"] == [2002, 2017, 2022] and abs(c["rho"]) < abs(c["rho_raw"]) <= 1


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
