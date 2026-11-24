"""Étude du duel RN contre LFI (vvs.eval.duel_rn_lfi) : estimation écologique à lignes
fixées, priors (a) et (b), critère de succès, enquêtes publiées."""

import numpy as np
import pytest

from vvs.eval import duel_rn_lfi as D

OWN = (0.94, 0.02, 0.04)


def _synthetic(n=300, seed=0):
    """Circonscriptions simulées avec une matrice connue (lignes des finalistes = OWN)."""
    rng = np.random.default_rng(seed)
    G = len(D.GROUPS)
    B = np.tile([0.3, 0.3, 0.4], (G, 1))
    B[0], B[1] = [OWN[0], OWN[1], OWN[2]], [OWN[1], OWN[0], OWN[2]]
    B[D.GROUPS.index("centre")] = [0.15, 0.35, 0.50]
    B[D.GROUPS.index("right")] = [0.45, 0.10, 0.45]
    B[D.GROUPS.index("abstain")] = [0.05, 0.05, 0.90]
    X = rng.dirichlet(np.full(G, 2.0), size=n)
    w = rng.uniform(50_000, 100_000, size=n)
    return X, X @ B, w, B


def test_fit_with_fixed_rows_recovers_matrix():
    X, Y, w, B = _synthetic()
    fit = D.fit_transfers(X, Y, w, D.fixed_own_rows(OWN))
    assert np.allclose(fit[:2], B[:2])  # lignes imposées telles quelles
    assert np.allclose(fit, B, atol=1e-3)
    assert np.allclose(fit.sum(1), 1.0)


def test_low_weight_group_falls_back_to_proximity():
    X, _, w, B = _synthetic()
    weights = D.group_weights(X, w)
    weights[D.GROUPS.index("other")] = 0.001
    prior = D.analog_priors(B, weights)
    base = D.proximity_priors()
    k = D.GROUPS.index("other")
    assert prior.groups[k].mean == base.groups[k].mean and prior.groups[k].source == "proximity"
    c = D.GROUPS.index("centre")
    assert prior.groups[c].source == "analog" and prior.groups[c].kappa == D.KAPPA_ANALOG[1]
    assert prior.groups[0] == base.groups[0]  # électorats propres inchangés
    assert (prior.sigma_u, prior.sigma_v) == (D.SIGMA_U["analog"], D.SIGMA_V["analog"])


def test_proximity_prior_is_the_inedit_model_prior():
    p = D.proximity_priors()
    assert (p.sigma_u, p.sigma_v) == (D.SIGMA_U["inedit"], D.SIGMA_V["inedit"])
    assert all(g.source in ("own", "proximity") for g in p.groups)


def test_predictive_draws_reproducible_and_centred():
    X, Y, w, B = _synthetic(n=5)
    prior = D.analog_priors(B, D.group_weights(X, w))
    a = D.predictive_draws(prior, X, n=500)
    b = D.predictive_draws(prior, X, n=500)
    assert np.array_equal(a, b)
    truth = 100 * Y[:, 0] / (Y[:, 0] + Y[:, 1])
    assert np.all(np.abs(np.median(a, axis=1) - truth) < 5)


@pytest.mark.parametrize(
    "b, success",
    [
        ({"mae": 2.0, "log_score": -2.0, "cov90": 0.90}, True),
        ({"mae": 2.0, "log_score": -2.0, "cov90": 0.98}, False),  # trop large
        ({"mae": 2.0, "log_score": -3.5, "cov90": 0.90}, False),  # score log moins bon
        ({"mae": 3.5, "log_score": -2.0, "cov90": 0.90}, False),  # EAM moins bonne
    ],
)  # fmt: skip
def test_verdict(b, success):
    a = {"mae": 3.0, "log_score": -3.0, "cov90": 0.99}
    assert D.verdict(a, b)["success"] is success


def test_survey_means_cover_listed_electorates_only():
    base = D.proximity_priors()
    src = {"rows": [
        {"from": "ensemble", "to": {"nfp": 32, "rn": 18, "blank_or_abstain": 50}},
        {"from": "lr_dvd", "to": {"nfp_lfi": 41, "rn": 59}},  # choix forcé : pas de part « ni »
        {"from": "macron", "to": {"nfp": 77, "rn": 23}},  # pas un électorat législatif : ignoré
    ]}  # fmt: skip
    means, cov = D.survey_means(src, base)
    c, r = D.GROUPS.index("centre"), D.GROUPS.index("right")
    assert cov == ["centre", "right"]
    assert means[c] == pytest.approx([0.18, 0.32, 0.50])
    none = base.means()[r, 2]
    assert means[r] == pytest.approx([(1 - none) * 0.59, (1 - none) * 0.41, none])
    assert np.allclose(np.delete(means, [c, r], axis=0), np.delete(base.means(), [c, r], axis=0))


def test_surveys_registry_rows_are_consistent():
    for s in D.load_surveys():
        assert s["url"].startswith("http") and s["verified"]
        if s["duel"] == "rn_vs_nfp_by_label":
            continue  # colonnes de synthèse et variantes d'étiquette : pas une répartition
        for row in s["rows"]:
            vals = [v for v in row["to"].values() if v is not None]
            assert all(0 <= v <= 100 for v in vals)
            assert sum(vals) <= 102, (s["id"], row["from"])
