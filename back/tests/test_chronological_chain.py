"""Contrats temporels, scénarios non-oracles et conservation des probabilités."""

import datetime as dt
import json
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from toy_run import toy_scenarios

from vvs.eval.consolidation import (
    available_polls,
    check_pin,
    digest,
    historical_configurations,
    main,
    report,
    score_chain,
    verify,
)
from vvs.eval.sensitivity import draw_weights, probability_bounds, set_probability
from vvs.registry import default_registry
from vvs.registry.scenarios import enumerate_configurations


def test_availability_uses_publication_and_explicit_missing_delay():
    d = dt.date(2022, 4, 3)
    polls = pl.DataFrame(
        {
            "poll_id": ["published_later", "known", "unknown", "unfinished"],
            "fieldwork_end": [d, d, d, d + dt.timedelta(days=4)],
            "publish_date": [d + dt.timedelta(days=5), d, None, d],
        }
    )
    assert available_polls(polls, d, 2)["poll_id"].to_list() == ["known"]
    assert available_polls(polls, d + dt.timedelta(days=2), 2)["poll_id"].to_list() == [
        "known",
        "unknown",
    ]
    assert available_polls(polls, d, 0)["poll_id"].to_list() == ["known", "unknown"]
    with pytest.raises(ValueError):
        available_polls(polls, d, -1)


def test_fit_rejects_supplied_polls_after_forecast_date(tmp_path):
    from vvs.model.run import run_fit

    asof = dt.date(2022, 4, 3)
    future = pl.DataFrame({"fieldwork_end": [asof + dt.timedelta(days=1)]})
    with pytest.raises(ValueError, match="après la date de prévision"):
        run_fit(2022, asof, tmp_path / "fit", polls=future)


def test_fit_metadata_uses_frozen_commit_instead_of_parent_git_head(monkeypatch):
    from vvs.model.run import _git_commit

    frozen = "a89d304" + "0" * 33
    monkeypatch.setenv("VVS_SOURCE_COMMIT", frozen)
    assert _git_commit() == frozen


def test_historical_configs_weight_polls_not_number_of_hypotheses():
    date = dt.date(2016, 10, 20)
    registry = SimpleNamespace(elections={}, bloc_of=lambda *_: "right")
    rows = []
    for pid, hypotheses in (
        ("one", [("juppe", "lepen"), ("sarkozy", "lepen")]),
        ("two", [("juppe", "lepen")]),
    ):
        for h, cs in enumerate(hypotheses):
            rows += [
                {"poll_id": pid, "hypothesis_id": str(h), "candidate_id": c, "fieldwork_end": date}
                for c in cs
            ]
    configs, info = historical_configurations(
        pl.DataFrame(rows), 2017, date, registry, {"window_days": 30, "max_window_days": 180}
    )
    assert {tuple(c["candidates"]): c["probability"] for c in configs} == {
        ("juppe", "lepen"): 0.75,
        ("lepen", "sarkozy"): 0.25,
    }
    assert info["source"] == "poll_hypotheses"


def test_score_penalizes_absent_winner_without_substitution():
    sim = {
        "candidates": np.array(["bayrou_francois", "le_pen_marine"]),
        "shares": np.tile([60.0, 40.0], (100, 1)),
        "qualified": np.tile([0, 1], (100, 1)),
        "winner": np.zeros(100, dtype=int),
    }
    summary, rows = score_chain(sim, 2017, default_registry())
    assert summary["p_winner"] == summary["p_pair"] == 0
    assert summary["winner_outside_support"] and summary["pair_outside_support"]
    assert summary["brier_win"] == 2.0
    assert summary["unforecast_share"] > 70
    assert np.isfinite(summary["log_score_winner"])
    macron = next(r for r in rows if r["candidate_id"] == "macron_emmanuel")
    assert macron["p_win"] == 0 and not macron["in_support"]


def test_reweighting_matches_states_after_reordering_and_constraints():
    sc = toy_scenarios()
    original = enumerate_configurations(sc, small="marginal")
    target = enumerate_configurations(set_probability(sc, "rn", "bardella", 0.9), small="marginal")
    ids = np.repeat(np.arange(len(original)), 10)
    weights, missing = draw_weights(original, target, ids)
    bardella = np.array([original[int(i)].states["rn"] == "bardella" for i in ids])
    assert missing == 0
    assert weights.sum() == pytest.approx(1)
    assert weights @ bardella == pytest.approx(0.9)
    assert original[0].states != target[0].states
    assert all(
        c.small_presence.get("lisnard_david", 0) == 0 for c in target if c.states["lr"] == "rally"
    )


def test_missing_scenario_mass_is_bounded_not_hidden():
    original = enumerate_configurations(toy_scenarios(), small="marginal")
    ids = np.zeros(20, dtype=int)
    weights, missing = draw_weights(original, original, ids)
    p = original[0].probability
    bounds = probability_bounds(np.ones(20), weights, missing)
    assert missing == pytest.approx(1 - p)
    assert bounds["probability"] == pytest.approx(1)
    assert bounds["lower"] == pytest.approx(p)
    assert bounds["upper"] == pytest.approx(1)


def test_frozen_input_mutation_is_rejected(tmp_path):
    file = tmp_path / "source.txt"
    file.write_text("original")
    (tmp_path / "reference.json").write_text(json.dumps({"files": {file.name: digest(file)}}))
    verify(tmp_path)
    file.write_text("changed")
    with pytest.raises(ValueError, match="modifiée"):
        verify(tmp_path)


def test_reference_pin_allows_recreation_but_rejects_another_source_commit():
    pinned = {"id": "reference", "commit": "original", "spec": {}, "versions": {}, "files": {}}
    check_pin(pinned | {"created_at": "another day"}, pinned)
    with pytest.raises(ValueError, match="commit"):
        check_pin(pinned | {"commit": "current-worktree-head"}, pinned)


def test_resume_rejects_changed_dependencies_before_starting_cases(tmp_path, monkeypatch):
    (tmp_path / "reference.json").write_text(json.dumps({"files": {}, "versions": {}}))
    monkeypatch.setattr(sys, "argv", ["consolidation", "run", "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="versions de dépendances"):
        main()
    assert not (tmp_path / "cases").exists()


def test_report_combines_integer_zero_and_fractional_unforecast_share(tmp_path):
    spec = {"elections": [2012, 2017], "horizons": [7]}
    (tmp_path / "reference.json").write_text(json.dumps({"files": {}, "spec": spec}))
    for election, missing in ((2012, 0), (2017, 0.5)):
        folder = tmp_path / "cases" / f"{election}-007"
        folder.mkdir(parents=True)
        frame = pl.DataFrame(
            [
                {
                    "election": election,
                    "horizon": 7,
                    "method": "reference",
                    "p_winner": 0.8,
                    "log_score_winner": -0.2,
                    "log_score_pair": -0.3,
                    "brier_win": 0.1,
                    "crps": 1.0,
                    "coverage90": 0.9,
                    "fit_acceptable": True,
                    "unforecast_share": missing,
                }
            ]
        )
        file = folder / "metrics.parquet"
        frame.write_parquet(file)
        (folder / "scored.json").write_text(json.dumps({file.name: digest(file)}))
    report(tmp_path)
    assert pl.read_parquet(tmp_path / "metrics.parquet")["unforecast_share"].to_list() == [0, 0.5]
    assert pl.read_parquet(tmp_path / "means.parquet")["n_elections"].item() == 2
