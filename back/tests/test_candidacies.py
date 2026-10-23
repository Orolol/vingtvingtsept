"""Registre historique des candidatures : validation, états datés, invariance au futur."""

import datetime as dt

import pytest

from vvs.registry.candidacies import (
    ACTIVE_STATES,
    CandidaciesFile,
    assemble,
    load_candidacies,
)

D = dt.date
WIKI = {"url": "https://fr.wikipedia.org/w/index.php?oldid=1", "published": D(2016, 1, 1)}


def ev(eid, person, type_, published, **kw):
    return {
        "id": eid,
        "person": person,
        "type": type_,
        "date": kw.pop("date", published),
        "published": published,
        "published_basis": "source",
        "sources": [{"url": f"https://example.org/{eid}", "published": published}],
        **kw,
    }


def toy(extra_events=(), commitments=()):
    """Primaire à deux tours (a, b, c ; a gagne) ; d même parti hors primaire ; e engagé."""
    persons = [
        {"id": x, "name": x, "first_name": x, "last_name": x, "party": "lr", "bloc": "right",
         "final_status": "declared_absent"}
        for x in ("a", "b", "c", "d", "e")
    ]
    events = [
        ev("a_in", "a", "primary_entry", D(2016, 3, 1), process="p"),
        ev("b_in", "b", "primary_entry", D(2016, 4, 1), process="p"),
        ev("c_in", "c", "primary_entry", D(2016, 5, 1), process="p"),
        ev("d_decl", "d", "declaration", D(2016, 6, 1)),
        ev("e_mention", "e", "mention", D(2016, 2, 1)),
        *extra_events,
    ]
    rounds = [
        {"date": D(2016, 11, 20), "published": D(2016, 11, 20), "qualified": ["a", "b"],
         "sources": ["r1"]},
        {"date": D(2016, 11, 27), "published": D(2016, 11, 28), "qualified": ["a"],
         "sources": ["r2"]},
    ]
    f = CandidaciesFile.model_validate(
        {
            "election": 2017,
            "collection": {"protocol": "jouet", "collected_on": D(2026, 10, 23),
                           "sources": [{"id": "s", "kind": "press", "title": "t"}]},
            "references": {"r1": {"url": "https://example.org/r1", "published": D(2016, 11, 20)},
                           "r2": {"url": "https://example.org/r2", "published": D(2016, 11, 28)}},
            "persons": persons,
            "processes": [{"id": "p", "name": "primaire", "kind": "open_primary",
                           "organizers": ["lr"], "participants": ["a", "b", "c"], "winner": "a",
                           "rounds": rounds, "commitments": list(commitments)}],
            "events": events,
        }
    )
    return assemble(f)


def states(reg, date):
    return {p: s.state for p, s in reg.state_at(date).items()}


def test_no_person_before_first_public_trace():
    reg = toy()
    assert states(reg, D(2016, 2, 1)) == {}  # mention publiée le jour même : pas encore connue
    assert states(reg, D(2016, 2, 2)) == {"e": "possible"}
    assert states(reg, D(2016, 3, 2)) == {"e": "possible", "a": "declared"}


def test_primary_rounds_exclude_only_participants():
    reg = toy()
    assert states(reg, D(2016, 11, 21)) == {
        "a": "declared", "b": "declared", "c": "excluded", "d": "declared", "e": "possible"
    }
    after = reg.state_at(D(2016, 11, 29))
    assert {p: s.state for p, s in after.items()} == {
        "a": "designated", "b": "excluded", "c": "excluded", "d": "declared", "e": "possible"
    }
    # le résultat publié le lendemain du vote ne s'applique qu'après sa publication
    assert states(reg, D(2016, 11, 28))["b"] == "declared"
    assert after["b"].reason == "éliminé de p"
    assert after["d"].trail == ("d_decl",)


def test_documented_commitment_extends_exclusion_only_for_named_winner():
    commit = {"person": "e", "winners": ["a"], "published": D(2016, 9, 1), "sources": ["r1"]}
    assert states(toy(commitments=[commit]), D(2016, 12, 1))["e"] == "excluded"
    other = {**commit, "winners": ["b"]}
    assert states(toy(commitments=[other]), D(2016, 12, 1))["e"] == "possible"


def test_withdrawal_and_return():
    extra = [
        ev("b_back", "b", "mention", D(2017, 3, 1)),
        ev("b_out", "b", "withdrawal", D(2017, 3, 6)),
        ev("d_out", "d", "withdrawal", D(2017, 1, 10)),
        ev("d_again", "d", "declaration", D(2017, 2, 1)),
    ]
    reg = toy(extra)
    at = reg.state_at(D(2017, 3, 2))
    assert at["b"].state == "undetermined" and at["b"].returned
    assert at["d"].state == "declared" and at["d"].returned
    assert states(reg, D(2017, 3, 7))["b"] == "withdrawn"
    assert "undetermined" in ACTIVE_STATES and "withdrawn" not in ACTIVE_STATES


def test_future_events_do_not_change_past_states():
    base = toy()
    future = toy(
        [
            ev("f1", "d", "withdrawal", D(2016, 10, 25)),
            ev("f2", "e", "declaration", D(2016, 12, 1), date=D(2016, 1, 1)),
            ev("f3", "a", "withdrawal", D(2017, 1, 1)),
        ]
    )
    for day in (D(2016, 1, 1), D(2016, 6, 2), D(2016, 9, 30), D(2016, 10, 25)):
        assert base.state_at(day) == future.state_at(day)
    assert base.state_at(D(2016, 10, 26)) != future.state_at(D(2016, 10, 26))


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"published": D(2016, 1, 1), "date": D(2016, 2, 1)}, "avant l'événement"),
        ({"type": "official_list"}, "persons"),
        ({"type": "primary_exit"}, "process"),
        ({"published_basis": "imputed"}, "imputée"),
    ],
)
def test_event_model_rejects_inconsistencies(bad, message):
    base = ev("x", "a", "declaration", D(2016, 1, 1))
    with pytest.raises(ValueError, match=message):
        CandidaciesFile.model_validate(
            {"election": 2017,
             "collection": {"protocol": "p", "collected_on": D(2026, 1, 1),
                            "sources": [{"id": "s", "kind": "press", "title": "t"}]},
             "persons": [{"id": "a", "bloc": "right", "party": "lr",
                          "final_status": "never_declared"}],
             "events": [{**base, **bad}]}
        )


def test_process_rounds_must_shrink_to_winner():
    reg = toy()
    raw = reg.file.model_dump()
    raw["processes"][0]["rounds"][1]["qualified"] = ["b"]
    with pytest.raises(ValueError, match="vainqueur"):
        CandidaciesFile.model_validate(raw)


@pytest.mark.parametrize("election", [2012, 2017, 2022])
def test_registry_files_validate(election):
    reg = load_candidacies(election)
    # la liste officielle s'applique et tout candidat officiel a une trace antérieure
    final = reg.state_at(reg.persons and max(e.published for e in reg.events) + dt.timedelta(1))
    official = [p for p, x in reg.persons.items() if x.final_status == "official"]
    assert sorted(p for p, s in final.items() if s.state == "official") == sorted(official)
    for p in official:
        first = reg.first_trace(p)
        assert first is not None and first.type != "official_list"



@pytest.mark.parametrize("election", [2012, 2017, 2022])
def test_registry_files_no_person_before_first_trace(election):
    reg = load_candidacies(election)
    for p in reg.persons:
        first = reg.first_trace(p).published
        assert p not in reg.state_at(first)
        assert p in reg.state_at(first + dt.timedelta(days=1))
