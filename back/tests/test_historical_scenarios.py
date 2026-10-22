"""Mémoire des candidatures : pondération, barrières temporelles et absence d'oracle."""

import datetime as dt
from types import SimpleNamespace

import polars as pl
import pytest

from vvs.eval.historical_scenarios import presence, reconstruct

ASOF = dt.date(2016, 10, 25)
SETTINGS = {"window_days": 365, "half_life_days": 60, "variants": ["memory", "memory_events"]}


def registry():
    # Pas de résultat présidentiel accessible à la construction pré-liste.
    return SimpleNamespace(
        elections={},
        bloc_of=lambda *_: "right",
        candidates={
            c: SimpleNamespace(party_at=lambda year, p=party: p)
            for c, party in {"a": "p", "b": "p", "c": "q", "future": "p"}.items()
        },
    )


def polls(entries):
    return pl.DataFrame(
        [
            {
                "poll_id": pid,
                "hypothesis_id": str(i),
                "candidate_id": c,
                "fieldwork_end": ASOF - dt.timedelta(days=age),
            }
            for pid, age, hyps in entries
            for i, candidates in enumerate(hyps)
            for c in candidates
        ]
    )


def test_memory_weights_polls_and_distinct_hypotheses_with_decay():
    df = polls(
        [
            ("new", 0, [("a", "c"), ("b", "c"), ("b", "c")]),
            ("old", 60, [("a", "c")]),
            ("too_old", 365, [("b", "c")]),
        ]
    )
    cfg, audit = reconstruct(df, 2017, ASOF, registry(), SETTINGS, [], "memory")
    assert presence(cfg) == pytest.approx({"a": 2 / 3, "b": 1 / 3, "c": 1})
    assert audit["n_polls"] == 2
    shuffled = df.sample(fraction=1, shuffle=True, seed=8)
    assert reconstruct(shuffled, 2017, ASOF, registry(), SETTINGS, [], "memory")[0] == cfg


@pytest.mark.parametrize("event_type", ["primary", "withdrawal"])
def test_event_only_applies_next_day_and_never_injects_nominee(event_type):
    df = polls([("p", 1, [("a", "c"), ("b", "c")])])
    event = {
        "id": "e",
        "election": 2017,
        "date": ASOF,
        "type": event_type,
        "candidate": "a" if event_type == "primary" else "b",
    }
    cfg, _ = reconstruct(df, 2017, ASOF, registry(), SETTINGS, [event], "memory_events")
    assert presence(cfg)["b"] == 0.5
    cfg, audit = reconstruct(
        df, 2017, ASOF + dt.timedelta(days=1), registry(), SETTINGS, [event], "memory_events"
    )
    assert presence(cfg) == {"a": 1, "c": 1}
    assert audit["discarded_probability"] == pytest.approx(0.5)
    other_election = event | {"election": 2022, "date": ASOF - dt.timedelta(days=1)}
    cfg, _ = reconstruct(df, 2017, ASOF, registry(), SETTINGS, [other_election], "memory_events")
    assert presence(cfg)["b"] == 0.5
    if event_type == "primary":
        # Un futur vainqueur n'est jamais lu comme candidat avant son événement.
        future = event | {"candidate": "future", "date": ASOF + dt.timedelta(days=1)}
        assert (
            reconstruct(df, 2017, ASOF, registry(), SETTINGS, [future], "memory_events")[0] == cfg
        )


def test_rejects_future_inputs_and_empty_admissible_set():
    df = polls([("p", -1, [("a", "c")])])
    with pytest.raises(ValueError, match="futur"):
        reconstruct(df, 2017, ASOF, registry(), SETTINGS, [], "memory")
    df = polls([("p", 1, [("a", "c")])])
    event = {
        "id": "e",
        "election": 2017,
        "date": ASOF - dt.timedelta(days=1),
        "type": "primary",
        "candidate": "b",
    }
    with pytest.raises(ValueError, match="aucune configuration admissible"):
        reconstruct(df, 2017, ASOF, registry(), SETTINGS, [event], "memory_events")


def test_official_list_unchanged_and_no_early_access():
    from vvs.eval.configurations import list_publication

    reg = registry()
    date = list_publication(2017)
    df = polls([("p", 1, [("a", "c")])])
    cfg, _ = reconstruct(df, 2017, date - dt.timedelta(days=1), reg, SETTINGS, [], "memory")
    assert "b" not in presence(cfg)
    reg.elections[2017] = SimpleNamespace(
        candidates=[SimpleNamespace(candidate_id=c) for c in ["b", "c"]]
    )
    for variant in SETTINGS["variants"]:
        cfg, audit = reconstruct(df, 2017, date, reg, SETTINGS, [], variant)
        assert cfg == [{"candidates": ["b", "c"], "probability": 1.0}]
        assert audit == {"source": "official_list"}
