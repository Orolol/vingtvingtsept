import datetime as dt

import numpy as np
import polars as pl
import pytest

from vvs.eval.structure import (
    ErrorMixture,
    fixed_count_scores,
    mix_trajectories,
    mixed_prediction,
    overlap_covariance,
    panel_transitions,
    rolling_summary,
)


def test_mixture_preserves_complete_outcomes_and_scoring_count():
    def draws(value):
        return {"candidates": np.array(["a", "b"]),
                "shares": np.array([[value, 100-value]]), "winner": np.array([int(value < 50)]),
                "qualified": np.array([[0, 1]])}
    mixture = mix_trajectories(draws(90), draws(10))
    assert mixture["shares"].tolist() == [[90, 10], [10, 90]]
    assert mixture["winner"].tolist() == [0, 1]
    score = fixed_count_scores({"n_scoring_candidates": 2, "p_winner": .5, "p_pair": 1}, 40_000)
    assert score["log_score_winner"] == pytest.approx(np.log(.5))
    assert score["log_score_pair"] == 0
    with pytest.raises(ValueError, match="univers"):
        mix_trajectories(draws(90), draws(10) | {"candidates": np.array(["b", "a"])})


def test_calibration_mixture_keeps_one_parameter_draw_for_whole_trajectory():
    # Variance propre nulle : deux candidats d'un même bloc doivent partager leur choc,
    # même quand deux calibrations d'échelles très différentes sont mélangées.
    params = {"sigma_bloc_final": .1, "sigma_cand_final": 0., "rho": 0.,
              "a_bloc": 0., "a_cand": 0., "gamma": .5, "nu": 8.,
              "final_age": 7., "bloc_weight": {"left": 1.}, "bias": {}}
    model = ErrorMixture([params, params | {"sigma_bloc_final": 1.}])
    eta = model.sample_eta(30, ["left", "left"], np.full((500, 2), .5), np.random.default_rng(5))
    assert np.array_equal(eta[:, 0], eta[:, 1])
    assert np.isfinite(eta).all()


def record(start, end, pollster="a", rolling=True):
    return {"sample_size": 1000, "pollster": pollster, "is_rolling": rolling,
            "fieldwork_start": dt.date(2022, 1, start), "fieldwork_end": dt.date(2022, 1, end)}


def test_overlap_covariance_counts_shared_days_and_is_positive_definite():
    records = [record(1, 3), record(2, 4), record(4, 6), record(2, 4, "b")]
    cov = overlap_covariance(records)
    assert cov[0, 1] == pytest.approx((2/3)/600)
    assert cov[0, 2] == 0
    assert cov[0, 3] == 0
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.count_nonzero(overlap_covariance(records, overlap=0)-np.diag(np.diag(cov))) == 0
    duplicate = overlap_covariance([record(1, 3), record(1, 3)])
    independent = overlap_covariance([record(1, 3), record(1, 3)], overlap=0)
    assert duplicate.mean() > independent.mean()


def test_rolling_preserves_constant_signal_and_historical_fallback():
    rows = [record(1, 3) | {"poll_id": "a", "candidate_id": "x", "share": 30.},
            record(2, 4) | {"poll_id": "b", "candidate_id": "x", "share": 30.},
            record(1, 2) | {"poll_id": "old", "candidate_id": "y", "share": 5.}]
    rows[2]["fieldwork_start"] = dt.date(2021, 11, 1)
    rows[2]["fieldwork_end"] = dt.date(2021, 11, 2)
    result, _ = rolling_summary(pl.DataFrame(rows), dt.date(2022, 1, 5),
                                {"rolling_design": .6, "rolling_n_max": 2500}, 1)
    assert result.filter(pl.col("candidate_id") == "x")["share"][0] == pytest.approx(30)
    assert result.filter(pl.col("candidate_id") == "y")["poll_id"][0] == "old"


@pytest.mark.parametrize("tau", [0., .5, 2.])
def test_mixed_substitution_coherent_and_ignores_test_target(tau):
    pair = {"p_full": np.array([.2, .3, .1, .4]), "blocs": ["left", "left", "right", "right"],
            "present": np.array([True, False, True, True])}
    q = mixed_prediction(pair, {"left": .5, "right": .7}, tau, {"left": -1, "right": 1})
    assert q.sum() == pytest.approx(1)
    assert q[1] == 0
    assert (q >= 0).all()
    assert np.array_equal(q, mixed_prediction(pair | {"y_red": np.ones(4)},
                          {"left": .5, "right": .7}, tau, {"left": -1, "right": 1}))
    if tau == 0:
        mass_left = np.sqrt(.5*.2)
        assert q[0] == pytest.approx(mass_left/(mass_left+.5))
    identity = mixed_prediction(pair | {"present": np.ones(4, bool)},
                                {"left": .5, "right": .7}, tau, {"left": -1, "right": 1})
    assert np.allclose(identity, pair["p_full"], atol=1e-8)


def waves():
    def wave(day, choices):
        return pl.DataFrame({"respondent_id": ["1", "2", "3", "4"], "series": ["panel"]*4,
                             "election": [2022]*4, "vote_round1_kind": ["intention"]*4,
                             "fieldwork_start": [dt.date(2022, 1, day)]*4,
                             "fieldwork_end": [dt.date(2022, 1, day)]*4,
                             "weight": [1., 1., 2., 1.], "vote_round1": choices})
    return wave(1, ["undecided", "undecided", "a", "refused"]), wave(8, ["a", "b", "a", "b"])


def test_panel_linkage_weights_and_exclusion_of_refusals():
    before, after = waves()
    matrix, audit = panel_transitions(before, after, ["a", "b", "undecided"],
                                      linked_id_verified=True, min_linked=3)
    assert audit["linked"] == 3
    assert audit["rows"][2]["ess"] == 2
    assert np.allclose(matrix.sum(1), 1)
    assert matrix[2, 0] > matrix[2, 1]  # prior global pondéré : 3/4 vers a
    # Les poids d'enquête ne doivent pas devenir des milliers de pseudo-répondants.
    scaled, _ = panel_transitions(before.with_columns(pl.col("weight")*1000), after,
                                  ["a", "b", "undecided"], linked_id_verified=True, min_linked=3)
    assert np.allclose(matrix, scaled)


@pytest.mark.parametrize("failure", ["unverified", "recall", "duplicate", "reversed", "small"])
def test_panel_rejects_invalid_historical_evidence(failure):
    before, after = waves()
    if failure == "recall":
        before = before.with_columns(pl.lit("recall").alias("vote_round1_kind"))
    elif failure == "duplicate":
        before = pl.concat([before, before.head(1)])
    elif failure == "reversed":
        before, after = after, before
    with pytest.raises(ValueError):
        panel_transitions(before, after, ["a", "b", "undecided"],
                          linked_id_verified=failure != "unverified", min_linked=100 if failure == "small" else 3)
