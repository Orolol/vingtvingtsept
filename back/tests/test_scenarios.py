import copy
import math

import pytest
import yaml

from vvs.registry.scenarios import (
    ScenarioError,
    enumerate_configurations,
    load_scenarios,
)


@pytest.fixture(scope="module")
def real():
    return load_scenarios()


def _base() -> dict:
    """Petit fichier synthétique, candidats du registre réel."""
    return {
        "election": 2027,
        "as_of": "2026-10-20",
        "references": {"WP": "https://fr.wikipedia.org/"},
        "dimensions": [
            {
                "id": "rn",
                "question": "?",
                "states": [
                    {
                        "id": "le_pen",
                        "probability": 0.8,
                        "description": "-",
                        "candidates": ["le_pen_marine"],
                        "sources": ["WP"],
                    },
                    {
                        "id": "bardella",
                        "probability": 0.2,
                        "description": "-",
                        "candidates": ["bardella_jordan"],
                    },
                ],
            },
            {
                "id": "lr",
                "question": "?",
                "states": [
                    {
                        "id": "retailleau",
                        "probability": 0.7,
                        "description": "-",
                        "candidates": ["retailleau_bruno"],
                    },
                    {"id": "rally", "probability": 0.3, "description": "-", "candidates": []},
                ],
            },
        ],
        "small_candidates": [
            {
                "candidate_id": "arthaud_nathalie",
                "probability": 0.9,
                "bloc": "far_left",
                "rationale": "-",
            },
            {
                "candidate_id": "roussel_fabien",
                "probability": 0.5,
                "bloc": "far_left",
                "rationale": "-",
            },
            {
                "candidate_id": "lisnard_david",
                "probability": 0.4,
                "bloc": "right",
                "rationale": "-",
            },
        ],
        "constraints": [
            {"id": "c", "when": {"lr": "rally"}, "absent": ["lisnard_david"], "rationale": "-"},
        ],
        "revisions": [
            {"date": "2026-10-20", "author": "test", "summary": "-", "rationale": "-"},
        ],
    }


def _load(tmp_path, data):
    p = tmp_path / "scenarios.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_scenarios(p)


def test_real_file_and_marginal_enumeration(real):
    assert real.as_of == real.revisions[-1].date
    e = enumerate_configurations(real, small="marginal")
    # produit des dimensions moins les combinaisons rally × attal_only exclues
    n_states = [len(d.states) for d in real.dimensions]
    excluded = 1
    for d in real.dimensions:
        if d.id not in ("lr", "centre"):
            excluded *= len(d.states)
    assert len(e) == math.prod(n_states) - excluded
    assert e.impossible_mass == pytest.approx(
        real.dimension("lr").state("rally").probability
        * real.dimension("centre").state("attal_only").probability
    )
    assert sum(c.probability for c in e) == pytest.approx(1)
    presence = e.presence()
    # la consolidation à gauche exclut Ruffin : présence = P(fragmented) × p(Ruffin)
    assert presence["ruffin_francois"] == pytest.approx(
        real.dimension("left").state("fragmented").probability
        * real.small("ruffin_francois").probability
    )
    assert presence["retailleau_bruno"] == pytest.approx(
        real.dimension("lr").state("retailleau").probability / (1 - e.impossible_mass)
    )


@pytest.mark.parametrize("mode", ["enumerate", "bloc", "marginal"])
def test_modes_agree_on_presence(tmp_path, mode):
    f = _load(tmp_path, _base())
    e = enumerate_configurations(f, small=mode)
    assert sum(c.probability for c in e) == pytest.approx(1)
    assert e.impossible_mass == 0 and e.pruned_mass == 0
    p = e.presence()
    assert p["le_pen_marine"] == pytest.approx(0.8)
    assert p["arthaud_nathalie"] == pytest.approx(0.9)
    assert p["lisnard_david"] == pytest.approx(0.7 * 0.4)  # absent si ralliement LR
    expected_len = {"enumerate": 2 * (4 * 2 + 4), "bloc": 2 * (2 * 2 + 2), "marginal": 4}
    assert len(e) == expected_len[mode]


def test_bloc_mode_conditional_presence(tmp_path):
    e = enumerate_configurations(_load(tmp_path, _base()), small="bloc")
    c = next(c for c in e if "far_left" in c.small_blocs)
    any_far_left = 1 - 0.1 * 0.5
    assert c.small_presence["arthaud_nathalie"] == pytest.approx(0.9 / any_far_left)


def test_impossible_combination_renormalizes(tmp_path):
    data = _base()
    data["constraints"].append(
        {"id": "x", "when": {"rn": "bardella", "lr": "rally"}, "impossible": True, "rationale": "-"}
    )
    e = enumerate_configurations(_load(tmp_path, data), small="marginal")
    assert e.impossible_mass == pytest.approx(0.2 * 0.3)
    assert len(e) == 3
    assert e.presence()["bardella_jordan"] == pytest.approx(0.2 * 0.7 / (1 - 0.06))


def test_absent_dimension_candidate_makes_configuration_impossible(tmp_path):
    data = _base()
    data["constraints"].append(
        {"id": "x", "when": {"lr": "rally"}, "absent": ["bardella_jordan"], "rationale": "-"}
    )
    e = enumerate_configurations(_load(tmp_path, data), small="marginal")
    assert e.impossible_mass == pytest.approx(0.2 * 0.3)
    assert all(not ({"bardella_jordan"} <= c.candidates and c.states["lr"] == "rally") for c in e)


def test_pruning_and_size_guard(tmp_path):
    f = _load(tmp_path, _base())
    e = enumerate_configurations(f, small="enumerate", min_probability=0.01)
    full = enumerate_configurations(f, small="enumerate")
    assert len(e) < len(full)
    assert e.pruned_mass == pytest.approx(
        sum(c.probability for c in full if c.probability < 0.01), abs=1e-12
    )
    with pytest.raises(ScenarioError, match="configurations"):
        enumerate_configurations(f, small="enumerate", max_configurations=5)


def _mutate(path, value):
    def apply(d):
        target = d
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
        return d

    return apply


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_mutate(["dimensions", 0, "states", 0, "probability"], 0.7), "somment"),
        (_mutate(["dimensions", 0, "states", 0, "candidates"], ["inconnu_x"]), "candidates.yaml"),
        (
            _mutate(["dimensions", 1, "states", 1, "candidates"], ["le_pen_marine"]),
            "deux dimensions",
        ),
        (_mutate(["small_candidates", 0, "candidate_id"], "retailleau_bruno"), "petit candidat"),
        (_mutate(["small_candidates", 0, "bloc"], "left"), "bloc du registre"),
        (_mutate(["constraints", 0, "when"], {"lr": "absent"}), "état inconnu"),
        (_mutate(["dimensions", 0, "states", 0, "sources"], ["NOPE"]), "alias"),
        (_mutate(["as_of"], "2026-10-21"), "as_of"),
    ],
)
def test_validation_errors(tmp_path, mutate, message):
    data = mutate(copy.deepcopy(_base()))
    with pytest.raises(ScenarioError, match=message):
        _load(tmp_path, data)


def test_absent_contradicting_when_state(tmp_path):
    data = _base()
    data["constraints"] = [
        {"id": "c", "when": {"lr": "retailleau"}, "absent": ["retailleau_bruno"], "rationale": "-"}
    ]
    with pytest.raises(ScenarioError, match="forcé"):
        _load(tmp_path, data)


def test_all_configurations_impossible(tmp_path):
    data = _base()
    data["constraints"] = [
        {"id": f"x{i}", "when": {"rn": rn, "lr": lr}, "impossible": True, "rationale": "-"}
        for i, (rn, lr) in enumerate(
            [(a, b) for a in ("le_pen", "bardella") for b in ("retailleau", "rally")]
        )
    ]
    with pytest.raises(ScenarioError, match="aucune configuration"):
        enumerate_configurations(_load(tmp_path, data), small="marginal")
