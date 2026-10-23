"""Scénarios par décision de candidature (lot 2) : états, processus, plafonds, prior, oracle."""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from vvs.eval import decision_scenarios as ds
from vvs.registry.candidacies import CandidaciesFile, assemble

D = dt.date
ASOF = D(2016, 10, 25)


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


PERSONS = {
    # personne : (parti, bloc, statut final, ne sert qu'à l'audit)
    "a": ("lr", "right", "official"),
    "b": ("lr", "right", "declared_absent"),
    "c": ("lr", "right", "declared_absent"),
    "d": ("lr", "right", "never_declared"),  # possible, parti organisateur
    "e": ("modem", "centre", "never_declared"),  # engagé sur b
    "f": ("ps", "left", "official"),  # déclaré
    "g": ("ps", "left", "declared_absent"),  # déclaré
    "h": ("ps", "left", "declared_absent"),  # déclaré
    "w": ("em", "centre", "official"),  # vainqueur réel, possible
}


def toy(extra_events=(), rounds=None, winner="a", participants=("a", "b", "c")):
    """Primaire LR en attente (a, b, c entrés ; d possible du parti), e engagé sur b."""
    events = [
        ev("a_in", "a", "primary_entry", D(2016, 3, 1), process="p"),
        ev("b_in", "b", "primary_entry", D(2016, 4, 1), process="p"),
        ev("c_in", "c", "primary_entry", D(2016, 5, 1), process="p"),
        ev("d_m", "d", "mention", D(2016, 2, 1)),
        ev("e_m", "e", "mention", D(2016, 2, 1)),
        ev("f_d", "f", "declaration", D(2016, 2, 1)),
        ev("g_d", "g", "declaration", D(2016, 2, 1)),
        ev("h_d", "h", "declaration", D(2016, 2, 1)),
        ev("w_m", "w", "mention", D(2016, 5, 11)),
        *extra_events,
    ]
    rounds = rounds or [
        {"date": D(2016, 11, 20), "published": D(2016, 11, 20), "qualified": [winner],
         "sources": ["r1"]},
    ]
    f = CandidaciesFile.model_validate(
        {
            "election": 2017,
            "collection": {"protocol": "jouet", "collected_on": D(2026, 10, 23),
                           "sources": [{"id": "s", "kind": "press", "title": "t"}]},
            "references": {"r1": {"url": "https://example.org/r1", "published": D(2016, 11, 20)},
                           "r2": {"url": "https://example.org/r2", "published": D(2016, 11, 28)}},
            "persons": [
                {"id": x, "name": x, "first_name": x, "last_name": x, "party": party,
                 "bloc": bloc, "final_status": status}
                for x, (party, bloc, status) in PERSONS.items()
            ],
            "processes": [{
                "id": "p", "name": "primaire", "kind": "open_primary", "organizers": ["lr"],
                "participants": list(participants), "winner": winner, "rounds": rounds,
                "commitments": [{"person": "e", "winners": ["b"], "published": D(2015, 7, 1),
                                 "sources": ["r1"]}],
            }],
            "events": events,
        }
    )
    return assemble(f)


CAPS = {"far_left": 5, "left": 4, "centre": 2, "right": 4, "far_right": 2, "other": 1}


def law(cr, settings=None, fitted=frozenset(), caps=CAPS, total=16, asof=ASOF):
    dec = ds.decision_units(cr, asof, settings or ds.Settings())
    ptype = {p: f"type-{dec.blocs[p]}" for p in dec.blocs if p not in fitted}
    return dec, ds.enumerate_law(dec, set(fitted), ptype, caps, total)


def configurations_of(result):
    _, lw = result
    for key, entry in lw.signatures.items():
        for wi in entry["components"]:
            yield key, lw.omegas[wi]


@pytest.mark.parametrize(
    ("state", "expected"),
    [("official", 1.0), ("designated", 1.0), ("declared", 0.9), ("possible", 0.15),
     ("undetermined", 0.3), ("excluded", 0.0), ("withdrawn", 0.0)],
)
def test_state_conversion(state, expected):
    assert ds.candidacy_probability(state, ds.Settings()) == expected


def test_probability_clip_and_unknown_state():
    assert ds.candidacy_probability("declared", ds.Settings(declared=1.0)) == 0.99
    assert ds.candidacy_probability("possible", ds.Settings(possible=0.0)) == 0.01
    assert ds.candidacy_probability("designated", ds.Settings(declared=1.0)) == 1.0
    with pytest.raises(ValueError):
        ds.candidacy_probability("ghost", ds.Settings())


def test_sensitivity_variants_from_protocol():
    import yaml

    from vvs import paths

    protocol = yaml.safe_load(
        (paths.REGISTRY / "historical_decisions_2026-10-23.yaml").read_text()
    )
    names = ds.variants(protocol)
    assert names[0] == "decisions" and len(names) == 5
    assert ds.Settings.from_protocol(protocol, "declared_plus").declared == 0.99
    assert ds.Settings.from_protocol(protocol, "possible_minus").possible == 0.05
    assert ds.Settings.from_protocol(protocol, "decisions") == ds.Settings()


def test_process_members_are_exclusive_and_uniform():
    dec, lw = law(toy())
    (pool,) = dec.pools
    assert pool.members == ("a", "b", "c", "d") and pool.joined_possible == ("d",)
    assert "d" not in dec.independent and dec.independent["w"] == 0.15
    for key, _ in configurations_of((dec, lw)):
        assert sum(m in key[0] for m in "abcd") <= 1
    # Masse des issues : uniforme, l'engagement de e ne pèse pas sur les membres.
    for m in "abcd":
        assert lw.presence[m] == pytest.approx(0.25)


def test_possible_member_joins_only_before_first_round():
    rounds = [
        {"date": D(2016, 10, 1), "published": D(2016, 10, 1), "qualified": ["a", "b"],
         "sources": ["r1"]},
        {"date": D(2016, 11, 27), "published": D(2016, 11, 28), "qualified": ["a"],
         "sources": ["r2"]},
    ]
    dec, _ = law(toy(rounds=rounds))
    (pool,) = dec.pools
    assert pool.members == ("a", "b") and pool.started
    assert dec.independent["d"] == 0.15  # reste une décision indépendante
    assert dec.states["c"].state == "excluded"


def test_commitment_forces_absence_only_for_named_winner():
    dec, lw = law(toy(), fitted={"e"})
    for key, omega in configurations_of((dec, lw)):
        if omega["p"] == "b":
            assert "e" not in key[0]
    assert lw.presence["e"] == pytest.approx(0.15 * 0.75)


def test_caps_condition_independent_decisions_and_certain_raise():
    caps = CAPS | {"left": 1}
    dec, lw = law(toy(), caps=caps, fitted={"f", "g", "h"})
    for key, _ in configurations_of((dec, lw)):
        assert sum(x in key[0] for x in "fgh") <= 1
    p1 = 3 * 0.9 * 0.1**2
    for x in "fgh":
        assert lw.presence[x] == pytest.approx(0.9 * 0.1**2 / (0.1**3 + p1))
    designated = toy([ev("f_i", "f", "investiture", D(2016, 6, 1)),
                      ev("g_i", "g", "investiture", D(2016, 6, 2))])
    dec, lw = law(designated, caps=caps, fitted={"f", "g", "h"})
    assert lw.presence["f"] == lw.presence["g"] == pytest.approx(1.0)
    assert lw.presence.get("h", 0.0) == 0.0


def test_future_events_and_process_outcome_do_not_change_construction():
    base = law(toy())[1]
    future = toy(
        [ev("z_m", "w", "declaration", ASOF), ev("late", "h", "withdrawal", D(2016, 12, 1)),
         ev("ghost", "g", "mention", D(2017, 1, 1))],
        winner="b", participants=("a", "b", "c"),
    )
    other = law(future)[1]
    assert other.signatures.keys() == base.signatures.keys()
    assert other.presence == pytest.approx(base.presence)


def test_no_actual_winner_injection():
    """Changer le statut final ou le vainqueur réel ne change rien ; w reste « possible »."""
    a = law(toy())[1]
    PERSONS_SWAP = dict(PERSONS)
    try:
        PERSONS["w"] = ("em", "centre", "never_declared")
        PERSONS["b"] = ("lr", "right", "official")
        b = law(toy(winner="b"))[1]
    finally:
        PERSONS.clear()
        PERSONS.update(PERSONS_SWAP)
    assert a.presence == pytest.approx(b.presence)
    assert a.presence["w"] == pytest.approx(0.15)


def registry_stub():
    elections = {
        2002: SimpleNamespace(candidates=[SimpleNamespace(bloc=b) for b in
                                          ["left"] * 2 + ["right"] * 3 + ["other"]]),
        2007: SimpleNamespace(candidates=[SimpleNamespace(bloc=b) for b in
                                          ["left"] * 3 + ["centre"]]),
        2017: SimpleNamespace(candidates=[SimpleNamespace(bloc=b) for b in ["left"] * 9]),
    }
    return SimpleNamespace(elections=elections)


def test_caps_use_only_earlier_official_lists():
    caps, total = ds.bloc_caps(registry_stub(), 2012)
    assert caps == {"left": 3, "right": 3, "other": 1, "centre": 1} and total == 6
    with pytest.raises(ValueError):
        ds.bloc_caps(registry_stub(), 2002)


def draws_stub(names):
    rng = np.random.default_rng(0)
    return {"candidates": np.array(names), "theta_asof": rng.normal(0, 0.1, (50, len(names)))}


def build(cr=None, settings=None, n=1000, fitted=("a", "b", "f")):
    return ds.build_case(cr or toy(), registry_stub(), 2012, 180, ASOF, settings or ds.Settings(),
                         draws_stub(list(fitted)), {}, {}, n, 1)


def test_configuration_cap_coverage_and_counts():
    settings = ds.Settings(max_configurations=3, coverage=0.99)
    d = build(settings=settings)
    a = d["audit"]
    assert a["n_top"] == 3 and a["top_coverage"] < 0.99
    assert sum(c["count"] for c in d["configurations"]) == 1000
    assert sum(c["kind"] == "top" for c in d["configurations"]) <= 3
    assert {c["kind"] for c in d["configurations"]} <= {"top", "sampled"}
    assert a["max_presence_shift_from_sampling"] < 0.06
    assert a["pruned_mass_bound"] < 1e-9


def test_prior_support_declared_for_unpolled_candidates():
    d = build()
    rows = {r["person"]: r for r in d["persons"]}
    assert rows["a"]["estimated_support"] and not rows["c"]["estimated_support"]
    assert rows["c"]["prior_centre"] == 0.01 and rows["c"]["prior_source"] == "default"
    assert ds.prior_centre("x", {"x": 12.0}, {"x": 3.0}, 0.01) == (0.12, "cycle")
    assert ds.prior_centre("x", {}, {"x": 3.0}, 0.01) == (0.03, "previous_cycle")
    s = ds.prior_shares([0.01, 0.2], 20000, ds.Settings(), np.random.default_rng(1))
    assert np.median(s[:, 0]) == pytest.approx(0.01, rel=0.05)
    assert s.max() <= 0.5 and s.sum(axis=1).max() <= 0.9 + 1e-12
    x = ds.mix_log_shares(["a", "b", "s"], ["a", "b"], np.log([[0.25, 0.75]] * 3), ["s"],
                          np.array([[0.1], [0.2], [0.3]]))
    p = np.exp(x)
    assert np.allclose(p.sum(axis=1), 1) and np.allclose(p[:, 1] / p[:, 0], 3)


def test_sample_conditional_matches_inclusion():
    persons = [("x", 0.9), ("y", 0.5), ("z", 0.1), ("u", 1.0)]
    out = ds.sample_conditional(persons, 2, 40000, np.random.default_rng(3))
    assert (out[:, 0] != out[:, 1]).all() and (out == "u").any(axis=1).all()
    freq = [(out == name).any(axis=1).mean() for name, _ in persons]
    exact = ds.conditional_inclusion([q for _, q in persons], 2)
    assert np.allclose(freq, exact, atol=0.01)


def test_assign_names_renames_slots_and_keeps_shares():
    d = build(n=400)
    sim_configs, _ = ds.simulation_configurations(d, "v")
    cands = sorted({c for cfg in sim_configs for c in cfg["candidates"]})
    counts = [c["count"] for c in d["configurations"]]
    config = np.repeat(np.arange(len(counts)), counts)
    shares = np.zeros((400, len(cands)))
    for k, cfg in enumerate(sim_configs):
        cols = [cands.index(c) for c in cfg["candidates"]]
        shares[np.ix_(config == k, cols)] = 100 / len(cols)
    order = np.argsort(-shares, axis=1, kind="stable")[:, :2]
    sim = {"candidates": np.array(cands), "shares": shares, "qualified": order,
           "winner": order[:, 0], "share_second": np.full(400, 55.0), "config": config}
    final, expanded = ds.assign_names(d, "v", sim, sim_configs, np.random.default_rng(2))
    assert not any(str(c).startswith(ds.SLOT_PREFIX) for c in final["candidates"])
    assert np.allclose(final["shares"].sum(axis=1), 100)
    assert sum(c["probability"] for c in expanded) == pytest.approx(1)
    present = ds.presence(expanded)
    exact = {r["person"]: r["presence_exact"] for r in d["persons"]}
    for person, p in present.items():
        assert p == pytest.approx(exact[person], abs=0.1)
