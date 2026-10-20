"""Rebonds d'annonce et de primaire (vvs.model.bounce) : forme, dérive, catalogue."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from vvs.model import bounce as bo
from vvs.model import error as er

# --- forme


def test_decay_shape():
    r, T = 0.3, 4.0
    assert bo.decay(0.0, r, T) == pytest.approx(1.0)
    assert bo.decay(1e6, r, T) == pytest.approx(r)
    d = bo.decay(np.linspace(0, 30, 50), r, T)
    assert np.all(np.diff(d) < 0)
    t = np.array([-3.0, -0.1, 0.0, 2.0])
    y = bo.trajectory(t, pre=-2.0, jump=0.5, r=r, T=T)
    assert y[:2] == pytest.approx([-2.0, -2.0])
    assert y[2] == pytest.approx(-1.5)


@pytest.mark.parametrize("s,horizon", [(0.0, 10.0), (3.0, 13.0), (8.0, 26.0)])
def test_drift_factor_matches_trajectory(s, horizon):
    """D · facteur = y(s + Δ) − y(s) pour D = J · d(s) ; nul si tout le saut est durable."""
    pre, J, r, T = -2.0, 0.4, 0.25, 5.0
    D = bo.trajectory(s, pre, J, r, T) - pre
    mu = D * bo.drift_factor(s, horizon, r, T)
    assert mu == pytest.approx(bo.trajectory(s + horizon, pre, J, r, T)
                               - bo.trajectory(s, pre, J, r, T))
    assert bo.drift_factor(s, horizon, 1.0, T) == pytest.approx(0.0)
    # r hors de [0, 1] tronqué : jamais de dépassement du niveau d'avant
    assert bo.drift_factor(s, horizon, -0.5, T) == pytest.approx(bo.drift_factor(s, horizon, 0.0, T))


def test_fit_shape_recovers_common_form():
    rng = np.random.default_rng(0)
    r, T = 0.4, 3.0
    series = {}
    for k, (pre, J) in enumerate([(-2.5, 0.5), (-1.8, 0.3), (-3.0, 0.6), (-2.2, 0.4)]):
        t = np.sort(rng.uniform(-8, 12, 60))
        y = bo.trajectory(t, pre, J, r, T) + 0.02 * rng.standard_normal(60)
        series[str(k)] = bo.Series(t, y)
    sh = bo.fit_shape(series, "primary", n_boot=20)
    assert sh.r == pytest.approx(r, abs=0.08)
    assert sh.T == pytest.approx(T, rel=0.3)
    assert sh.jumps["2"] == pytest.approx(0.6, abs=0.05)
    assert len(sh.r_draws) > 10
    assert bo.fit_shape({"0": series["0"]}) is None  # un seul événement : pas de forme commune


# --- dérive sur un cas jouet


def _toy_book(dates_shares: list[tuple[dt.date, float]]) -> bo.PollBook:
    """Livre de sondages jouet : deux candidats, A (bloc left) et B (bloc right)."""
    book = bo.PollBook.__new__(bo.PollBook)
    book.election = 2099
    book._bloc = {"a": "left", "b": "right"}
    book.polls = [
        {"poll_id": str(i), "fieldwork_start": d, "fieldwork_end": d, "mid": d,
         "hyps": {"h": {"a": 100 * x, "b": 100 * (1 - x)}}, "config": ["a", "b"]}
        for i, (d, x) in enumerate(dates_shares)
    ]
    return book


def test_candidate_drift_toy_case():
    event = dt.date(2099, 1, 1)
    pre, peak = 0.20, 0.30
    polls = [(event - dt.timedelta(days=7 * k), pre) for k in range(1, 7)]
    polls += [(event + dt.timedelta(days=d), peak) for d in (3, 8, 12, 16)]
    book = _toy_book(polls)
    ev = bo.Event("p", 2099, "primary", event, "a", "left", sources=("http://x",))
    shape = bo.Shape("primary", 0.0, 4.0, np.array([0.0]), np.array([4.0]), ["x", "y"])
    asof = event + dt.timedelta(days=16)
    mu, det = bo.candidate_drift(2099, asof, 90, ["b", "a"], 50, np.random.default_rng(0),
                                 shapes={"primary": shape}, events=[ev], book=book)
    D = math.log(peak) - math.log(pre)
    assert det[0]["D"] == pytest.approx(D)
    assert np.all(mu[:, 0] == 0)  # B n'a pas d'événement
    expected = D * bo.drift_factor(16 / 7, 90 / 7, 0.0, 4.0)
    assert det[0]["mu_point"] == pytest.approx(expected)
    assert expected < 0 and np.median(mu[:, 1]) == pytest.approx(expected, abs=0.05)
    # événement trop ancien, type non retenu, candidat absent : aucune dérive
    for kw in ({"spec": bo.BounceSpec(window_weeks=1)}, {"spec": bo.BounceSpec(types=("shock",))}):
        mu0, _ = bo.candidate_drift(2099, asof, 90, ["b", "a"], 5, np.random.default_rng(0),
                                    shapes={"primary": shape}, events=[ev], book=book, **kw)
        assert np.all(mu0 == 0)
    mu0, _ = bo.candidate_drift(2099, asof, 90, ["b"], 5, np.random.default_rng(0),
                                shapes={"primary": shape}, events=[ev], book=book)
    assert np.all(mu0 == 0)


# --- réduction au comportement actuel


def test_error_model_drift_off_is_identity_and_on_shifts_logits():
    params = er.ErrorParams(sigma_bloc_final=0.06, sigma_cand_final=0.08, rho=1e-3, a_bloc=0.05,
                            a_cand=0.04, gamma=0.5, nu=8.0, final_age=7.0)
    m = er.ErrorModel(params)
    theta = np.log(np.array([[0.3, 0.5, 0.2]]))
    blocs = ["left", "right", "centre"]
    base = m.sample(theta, 90, ["a", "b", "c"], blocs, np.random.default_rng(1), 500)
    same = m.sample(theta, 90, ["a", "b", "c"], blocs, np.random.default_rng(1), 500, drift=None)
    zero = m.sample(theta, 90, ["a", "b", "c"], blocs, np.random.default_rng(1), 500,
                    drift=np.zeros(3))
    assert np.array_equal(base, same) and np.allclose(base, zero)
    mu = np.array([-0.2, 0.0, 0.0])
    shifted = m.sample(theta, 90, ["a", "b", "c"], blocs, np.random.default_rng(1), 500, drift=mu)
    # même η : la dérive se lit exactement dans les log-ratios
    assert np.allclose(np.log(shifted[:, 0] / shifted[:, 1]) - np.log(base[:, 0] / base[:, 1]),
                       -0.2)


def test_variant_drift_only_for_bounce_variants():
    assert bo.variant_drift("floor", 2017, dt.date(2017, 1, 23), 90, ["x"], 10) is None
    assert "bounce" in bo.BOUNCE_VARIANTS


# --- catalogue


def test_events_yaml_loads_and_is_consistent():
    evs = bo.load_events()
    ids = [e.id for e in evs]
    assert len(ids) == len(set(ids)) >= 30
    assert {e.type for e in evs} == set(bo.EVENT_TYPES)
    by_id = {e.id: e for e in evs}
    assert by_id["lr_primary_2016"].date == dt.date(2016, 11, 27)
    assert by_id["ps_primary_2026"].candidate == "glucksmann_raphael"
    assert all(e.sources and e.sources[0].startswith("http") for e in evs)
    assert all(e.sign in (-1, 1) for e in evs if e.type == "shock")
    assert by_id["hollande_withdrawal_2016"].focal == "valls_manuel"


def test_events_yaml_validation_reports_errors(tmp_path):
    bad = tmp_path / "events.yaml"
    bad.write_text(
        "events:\n"
        "  - {id: a, election: 2017, type: rally, date: 2017-01-01, candidate: fillon_francois,"
        " bloc: right, sources: [http://x]}\n"
        "  - {id: b, election: 2017, type: primary, date: 2016-11-27, candidate: fillon_francois,"
        " bloc: left, sources: [http://x]}\n"
        "  - {id: c, election: 2017, type: shock, date: 2017-01-25, candidate: fillon_francois,"
        " bloc: right, sources: []}\n"
        "  - {id: d, election: 2017, type: primary, date: 2017-05-01, candidate: nobody_x,"
        " bloc: right, sources: [http://x]}\n"
    )
    with pytest.raises(ValueError) as exc:
        bo.load_events(bad)
    msg = str(exc.value)
    for frag in ("type 'rally'", "le registre dit 'right'", "un choc doit avoir sign",
                 "sources manquantes", "absent du registre", "postérieure au premier tour"):
        assert frag in msg
