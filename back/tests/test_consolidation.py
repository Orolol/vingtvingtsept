"""Vote utile de fin de campagne (vvs.model.consolidation) : forme, dérive, estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvs.model import consolidation as co
from vvs.model import error as er

# parts jouets : gauche radicale en tête du camp de gauche, deux concurrents de gauche
CANDS = ["lepen", "zemmour", "macron", "melenchon", "jadot", "hidalgo", "lassalle"]
BLOCS = ["far_right", "far_right", "centre", "far_left", "left", "left", "other"]
SHARES = np.array([0.22, 0.10, 0.27, 0.16, 0.05, 0.02, 0.03])
SHARES = SHARES / SHARES.sum()


def _camp_sum(p, blocs, camp):
    return p[..., [co.camp_of(b) == camp for b in blocs]].sum(-1)


# --- forme


def test_horizon_weight_and_proximity_shape():
    assert co.horizon_weight(0.0) == pytest.approx(1.0)
    assert co.horizon_weight(180.0) == pytest.approx(0.0)
    assert co.horizon_weight(400.0) == pytest.approx(0.0)
    w = co.horizon_weight(np.linspace(0, 180, 20))
    assert np.all(np.diff(w) < 0)
    # g(0) = 1, décroissante en |m|, symétrique, g = 1 quand m0 = ∞
    assert co.proximity(0.0, m0=5.0) == pytest.approx(1.0)
    g = co.proximity(np.linspace(0, 20, 30), m0=5.0)
    assert np.all(np.diff(g) < 0) and g[-1] < 0.01
    assert co.proximity(-4.0, m0=5.0) == pytest.approx(co.proximity(4.0, m0=5.0))
    assert np.all(co.proximity(np.array([0.0, 30.0]), m0=math.inf) == 1.0)
    assert co.consolidation_phi(0.4, 90.0, 0.0) == pytest.approx(0.2)


def test_qualification_margin():
    s = np.array([[30.0, 20.0, 15.0, 10.0]])
    assert co.qualification_margin(s, 0)[0] == pytest.approx(15.0)  # 1er : contre le 3e
    assert co.qualification_margin(s, 1)[0] == pytest.approx(5.0)  # 2e : contre le 3e
    assert co.qualification_margin(s, 2)[0] == pytest.approx(-5.0)  # 3e : contre le 2e


# --- dérive


@pytest.mark.parametrize("tau,m0", [(7.0, math.inf), (30.0, 5.0), (90.0, 3.0)])
def test_drift_conserves_camp_and_leader_gains(tau, m0):
    mu = co.consolidation_drift(SHARES, BLOCS, tau, 0.5, m0=m0)
    after = co.apply_drift(SHARES, mu)[0]
    for camp in co.CAMPS:
        assert _camp_sum(after, BLOCS, camp) == pytest.approx(_camp_sum(SHARES, BLOCS, camp))
    assert after[6] == pytest.approx(SHARES[6])  # hors camp : intact
    assert after[3] > SHARES[3] and after[4] < SHARES[4] and after[5] < SHARES[5]  # gauche
    assert after[0] > SHARES[0] and after[1] < SHARES[1]  # extrême droite
    # transfert exact : φ des parts des concurrents
    phi = co.consolidation_phi(0.5, tau, 100 * co.qualification_margin(SHARES[None], 3)[0], m0)
    assert after[4] == pytest.approx((1 - phi) * SHARES[4])
    assert after[3] - SHARES[3] == pytest.approx(phi * (SHARES[4] + SHARES[5]))


def test_drift_zero_at_180_days_and_when_phi_zero():
    assert np.all(co.consolidation_drift(SHARES, BLOCS, 180.0, 0.8) == 0.0)
    assert np.all(co.consolidation_drift(SHARES, BLOCS, 7.0, 0.0) == 0.0)


def test_drift_per_row_leader_and_phi_draws():
    """Leader recalculé ligne par ligne ; φ0 tiré par ligne ; φ négatif borné."""
    p = np.array([[0.30, 0.10, 0.60], [0.10, 0.30, 0.60]])
    blocs = ["far_left", "left", "centre"]
    mu = co.consolidation_drift(p, blocs, 0.0, np.array([0.5, 0.5]))
    assert mu[0, 0] > 0 > mu[0, 1] and mu[1, 1] > 0 > mu[1, 0]
    mu_neg = co.consolidation_drift(p[:1], blocs, 0.0, np.array([-50.0]))
    after = co.apply_drift(p[:1], mu_neg)[0]
    assert np.all(np.isfinite(mu_neg)) and after[0] > 0
    assert after[:2].sum() == pytest.approx(0.40)


def test_error_model_sample_reduces_without_drift():
    """Option désactivée (drift=None ou dérive nulle à 180 jours) : mêmes tirages que la
    référence ; dérive active : le leader du camp gagne sur ses concurrents."""
    params = er.ErrorParams(0.08, 0.08, 1e-3, 0.03, 0.03, 0.5, 16.0, 7.0,
                            bloc_weight={b: 1.0 for b in er.BLOCS})
    em = er.ErrorModel(params)
    th = np.log(SHARES)[None, :]
    base = em.sample(th, 30.0, CANDS, BLOCS, np.random.default_rng(1), 4000)
    none = em.sample(th, 30.0, CANDS, BLOCS, np.random.default_rng(1), 4000, drift=None)
    zero = co.consolidation_drift(SHARES, BLOCS, 180.0, 0.6)
    z = em.sample(th, 30.0, CANDS, BLOCS, np.random.default_rng(1), 4000, drift=zero)
    assert np.array_equal(base, none) and np.allclose(base, z)
    mu = co.consolidation_drift(SHARES, BLOCS, 7.0, np.full(4000, 0.5))
    d = em.sample(th, 30.0, CANDS, BLOCS, np.random.default_rng(1), 4000, drift=mu)
    assert np.median(d[:, 3]) > np.median(base[:, 3])
    assert np.median(d[:, 4]) < np.median(base[:, 4])
    assert np.median(_camp_sum(d, BLOCS, "left")) == pytest.approx(
        np.median(_camp_sum(base, BLOCS, "left")), abs=0.003)


def test_variant_drift_falls_back_to_bounce():
    """Hors variante de consolidation : la dérive des rebonds (None pour floor)."""
    import datetime as dt

    assert co.variant_drift("floor", 2022, dt.date(2022, 3, 11), 30, CANDS, 10) is None
    with pytest.raises(ValueError):
        co.variant_drift("consolidation", 2022, dt.date(2022, 3, 11), 30, CANDS, 10)


# --- estimation sur un cas jouet


def _toy_obs(phi_true: float, rng) -> list[co.Observation]:
    """Couples jouets : transfert y = φ0 · w(τ) · f_τ + bruit, sur cinq élections."""
    out = []
    for e in (2002, 2007, 2012, 2017, 2022):
        for camp in ("left", "far_right"):
            f = rng.uniform(0.2, 0.6)
            for h in (0, 7, 30, 90, 180):
                y = phi_true * float(co.horizon_weight(h)) * f + 0.005 * rng.standard_normal()
                L, F = 20.0 * (1 - f), 20.0 * f
                out.append(co.Observation(e, camp, h, "L", ["F"], L, 20.0, f, f - y,
                                          margin_pts=rng.uniform(-8, 8), leader_vote=0.0,
                                          camp_vote=20.0, gap_first_follower=L - F))
    return out


def test_fit_recovers_phi0_and_leader_wins():
    rng = np.random.default_rng(0)
    obs = _toy_obs(0.3, rng)
    pr = co.fit_params(obs, exclude=(2022,), m0=math.inf, n_boot=50)
    assert pr.phi0 == pytest.approx(0.3, abs=0.03)
    assert pr.train == [2002, 2007, 2012, 2017] and len(pr.phi0_draws) == 50
    lo, _, hi = pr.interval()
    assert lo < 0.3 < hi
    auto = co.fit_params(obs, n_boot=10)
    assert auto.m0 in co.M0_GRID
    # le leader du cas jouet gagne au jour du vote
    mu = co.consolidation_drift(SHARES, BLOCS, 7.0, pr.phi0, pr.m0)
    assert co.apply_drift(SHARES, mu)[0, 3] > SHARES[3]


def test_observations_from_shares():
    shares = {"a": 20.0, "b": 8.0, "c": 30.0, "d": 25.0, "e": 2.0}
    votes = {"a": 24.0, "b": 4.0, "c": 31.0, "d": 25.0, "e": 2.0}
    blocs = {"a": "far_left", "b": "left", "c": "centre", "d": "right", "e": "other"}
    (o,) = co.observations_from_shares(2022, 7, shares, votes, blocs)
    assert (o.camp, o.leader, o.followers) == ("left", "a", ["b"])
    assert o.f_tau == pytest.approx(8 / 28) and o.f_vote == pytest.approx(4 / 28)
    assert o.margin_pts == pytest.approx(20.0 - 25.0)
    assert o.phi == pytest.approx(0.5)
