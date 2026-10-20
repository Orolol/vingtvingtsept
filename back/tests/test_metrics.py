"""Métriques du backtest sur des cas calculables à la main."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vvs.eval import metrics as mt


def test_crps_point_mass_is_absolute_error():
    x = np.full(50, 3.0)
    assert mt.crps(x, 5.5) == pytest.approx(2.5)


def test_crps_two_point_distribution_by_hand():
    # X ∈ {0, 1} équiprobable, y = 0 : E|X − y| = 0,5 ; E|X − X'| = 0,5 ; CRPS = 0,25
    x = np.array([0.0, 1.0] * 500)
    assert mt.crps(x, 0.0) == pytest.approx(0.25)
    # y = 0,5 : E|X − y| = 0,5 ; CRPS = 0,25 aussi
    assert mt.crps(x, 0.5) == pytest.approx(0.25)


def test_crps_matches_gaussian_closed_form_and_is_vectorized():
    rng = np.random.default_rng(0)
    sd = 2.0
    x = rng.normal(0, sd, size=(200_000, 2))
    y = np.array([0.0, 1.5])
    z = y / sd
    phi = np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in z]))
    exact = sd * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    assert mt.crps(x, y) == pytest.approx(exact, rel=0.01)


def test_qualification_and_pair_log_score_by_hand():
    # 4 tirages, 3 candidats : paires (0,1), (0,1), (0,2), (1,2)
    shares = np.array(
        [[0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [0.5, 0.1, 0.4], [0.2, 0.4, 0.4]]
    )
    assert mt.qualification_probs(shares) == pytest.approx([0.75, 0.75, 0.5])
    assert mt.pair_probability(shares, (1, 0), smooth=False) == pytest.approx(0.5)
    # lissage de Jeffreys : (k + ½)/(N + ½ M), M = 3 paires
    assert mt.pair_probability(shares, (0, 1)) == pytest.approx(2.5 / 5.5)
    assert mt.pair_log_score(shares, (0, 1)) == pytest.approx(math.log(2.5 / 5.5))
    # paire jamais tirée : probabilité lissée finie
    never = np.array([[0.5, 0.3, 0.2]] * 4)
    assert mt.pair_log_score(never, (1, 2)) == pytest.approx(math.log(0.5 / 5.5))


def test_brier_by_hand():
    assert mt.brier([1.0, 0.5, 0.0], [1, 1, 0]) == pytest.approx(0.25 / 3)


def test_coverage_and_pit():
    x = np.arange(1, 101, dtype=float)[:, None]  # 1..100
    assert mt.covered(x, [50.0], 0.5)[0]
    assert not mt.covered(x, [10.0], 0.5)[0]
    assert mt.covered(x, [10.0], 0.9)[0]
    assert mt.pit(x, [50.0])[0] == pytest.approx(0.495)
    assert mt.abs_error_median(x, [40.0])[0] == pytest.approx(10.5)
