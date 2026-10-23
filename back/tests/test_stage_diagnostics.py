"""Diagnostics par étape : identité de décomposition, univers commun, conditions vides."""

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from vvs.eval import metrics as mt
from vvs.eval.stage_diagnostics import (
    Outcome,
    compare_study,
    diagnose,
    jeffreys_log,
    presence_matrix,
)

# a et b finalistes réels, a vainqueur ; c officiel éliminé ; x envisagé mais absent.
OUTCOME = Outcome(shares={"a": 40.0, "b": 35.0, "c": 25.0}, finalists=("a", "b"), winner="a")
CANDIDATES = ["a", "b", "c", "x"]
CONFIGS = [
    {"candidates": ["a", "b", "c"], "probability": 0.5},
    {"candidates": ["b", "c", "x"], "probability": 0.5},
]


def toy_sim(rows, configs=CONFIGS, candidates=CANDIDATES):
    """rows : (configuration, parts par candidat, vainqueur parmi les deux premiers)."""
    index = {c: j for j, c in enumerate(candidates)}
    shares = np.zeros((len(rows), len(candidates)))
    for i, (_, part, _) in enumerate(rows):
        for c, s in part.items():
            shares[i, index[c]] = s
    qualified = mt.top2(shares)
    winner = np.array([index[w] for _, _, w in rows])
    assert all(w in q for w, q in zip(winner, qualified))
    return {
        "candidates": np.array(candidates),
        "shares": shares,
        "qualified": qualified,
        "winner": winner,
        "config": np.array([k for k, _, _ in rows]),
    }


ROWS = [
    (0, {"a": 40, "b": 35, "c": 25}, "a"),  # paire réelle, a gagne
    (0, {"a": 45, "b": 30, "c": 25}, "b"),  # paire réelle, b gagne
    (0, {"a": 40, "b": 20, "c": 39.5}, "a"),  # a et c qualifiés
    (0, {"a": 20, "b": 45, "c": 35}, "b"),  # a éliminé
    (1, {"b": 40, "c": 35, "x": 25}, "b"),  # a absent
    (1, {"b": 30, "c": 25, "x": 45}, "x"),
]


def finite(summary):
    return all(
        v is None or isinstance(v, (bool, str)) or math.isfinite(v) for v in summary.values()
    )


def normalize(rows):
    return [(k, {c: 100 * s / sum(p.values()) for c, s in p.items()}, w) for k, p, w in rows]


def test_decomposition_identity_and_counts():
    sim = toy_sim(normalize(ROWS))
    s, _, events = diagnose(sim, OUTCOME, CONFIGS)
    # a présent 4/6, qualifié 3/4, gagne 2/3 ; P(victoire) = 2/6.
    assert (s["p_presence_winner"], s["p_qualify_given_presence_winner"]) == (4 / 6, 3 / 4)
    assert s["p_win_given_qualify_winner"] == pytest.approx(2 / 3)
    assert s["p_winner"] == pytest.approx(2 / 6)
    assert s["identity_gap_winner_raw"] == pytest.approx(0, abs=1e-12)
    # Paire réelle présente 4/6, qualifiée 2/4, a gagne 1/2 de ces duels.
    assert s["p_pair_given_presence"] == 0.5 and s["p_win_given_pair"] == 0.5
    assert s["identity_gap_pair_raw"] == pytest.approx(0, abs=1e-12)
    # Lissage : chaque étape à deux issues ; l'écart au score global est publié.
    stages = jeffreys_log(4, 6) + jeffreys_log(3, 4) + jeffreys_log(2, 3)
    assert s["stage_sum_winner"] == pytest.approx(stages)
    k = 4  # officiels ∪ candidats des configurations
    assert s["log_score_winner"] == pytest.approx(math.log(2.5 / (6 + 0.5 * k)))
    assert s["smoothing_gap_winner"] == pytest.approx(s["log_score_winner"] - stages)
    assert {e["event"] for e in events} >= {"presence_winner", "win_given_pair"}
    assert finite(s)


def test_presence_brier_and_common_universe():
    sim = toy_sim(normalize(ROWS))
    s, table, _ = diagnose(sim, OUTCOME, CONFIGS)
    p = dict(zip(table["candidate_id"], table["p_present"]))
    assert p == pytest.approx({"a": 4 / 6, "b": 1, "c": 1, "x": 2 / 6})
    expected = (1 - 4 / 6) ** 2 + (2 / 6) ** 2
    assert s["brier_presence_sum"] == pytest.approx(expected)
    assert s["brier_presence_mean"] == pytest.approx(expected / 4)
    # Univers commun élargi par une variante qui nomme « y » : la somme ne bouge pas,
    # la moyenne se calcule sur le même dénominateur pour les deux modèles.
    wide, _, _ = diagnose(sim, OUTCOME, CONFIGS, universe={"a", "b", "c", "x", "y"})
    assert wide["brier_presence_sum"] == pytest.approx(expected)
    assert wide["brier_presence_mean"] == pytest.approx(expected / 5)
    with pytest.raises(ValueError, match="univers"):
        diagnose(sim, OUTCOME, CONFIGS, universe={"a", "b", "c"})


def test_conditional_support_uses_present_draws_only():
    sim = toy_sim(normalize(ROWS))
    _, table, _ = diagnose(sim, OUTCOME, CONFIGS)
    row = table.filter(table["candidate_id"] == "a").row(0, named=True)
    present = sim["shares"][:4, 0]
    assert row["crps_conditional"] == pytest.approx(float(mt.crps(present, 40.0)))
    assert row["crps_global"] == pytest.approx(float(mt.crps(sim["shares"][:, 0], 40.0)))
    assert row["crps_global"] > row["crps_conditional"]  # zéros des absents


def test_missing_pair_and_absent_winner_are_explicit_not_nan():
    # a n'est jamais présent : aucune condition aval n'est réalisée.
    rows = [(1, {"b": 40, "c": 35, "x": 25}, "b"), (1, {"b": 30, "c": 25, "x": 45}, "x")]
    sim = toy_sim(normalize(rows))
    s, table, _ = diagnose(sim, OUTCOME, CONFIGS)
    assert finite(s)
    for event in ("qualify_given_presence_winner", "pair_given_presence", "win_given_pair"):
        assert s[f"evaluable_{event}"] is False and s[f"p_{event}"] is None
        assert s[f"log_{event}"] == pytest.approx(math.log(0.5))
    assert s["log_presence_winner"] == pytest.approx(jeffreys_log(0, 2))
    assert s["identity_gap_winner_raw"] is None
    assert s["brier_qualify_given_pair_presence"] is None
    assert s["n_official_absent"] == 1 and s["unforecast_share"] == 40.0
    # L'officiel jamais présent sort des moyennes conditionnelles, de façon comptée.
    assert s["n_conditional"] == 2 and s["conditional_excluded_share"] == 40.0
    assert s["error_stage_winner"] == "presence"
    assert table.filter(table["candidate_id"] == "a")["crps_conditional"].item() is None


def test_presence_must_match_configuration_shares():
    sim = toy_sim(normalize(ROWS))
    sim["shares"][4, 0] = 1.0  # a a une part dans une configuration sans a
    with pytest.raises(ValueError, match="configurations"):
        presence_matrix(sim, CONFIGS)


def _write_case(folder, sim, configs):
    folder.mkdir(parents=True)
    (folder / "case.json").write_text(json.dumps({"configurations": configs}))
    for method in ("reference", "mean14d"):
        np.savez(folder / f"draws_{method}.npz", **sim)


def _hashes(folder, names):
    return {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in names}


def test_compare_study_uses_common_universe(tmp_path):
    reference, study = tmp_path / "reference", tmp_path / "study"
    spec = {"elections": [2017], "horizons": [180]}
    (reference).mkdir()
    (reference / "reference.json").write_text(json.dumps({"files": {}, "spec": spec}))
    ref_case = reference / "cases/2017-180"
    _write_case(ref_case, toy_sim(normalize(ROWS)), CONFIGS)
    names = ["case.json", "draws_reference.npz", "draws_mean14d.npz"]
    (ref_case / "scored.json").write_text(json.dumps(_hashes(ref_case, names)))
    # La variante ajoute « y » et retire x ; a présent partout.
    configs = [
        {"candidates": ["a", "b", "c"], "probability": 0.5},
        {"candidates": ["a", "b", "y"], "probability": 0.5},
    ]
    rows = [
        (0, {"a": 40, "b": 35, "c": 25}, "a"),
        (1, {"a": 40, "b": 35, "y": 25}, "a"),
    ]
    variant = study / "cases/2017-180/memory"
    _write_case(variant, toy_sim(normalize(rows), configs, ["a", "b", "c", "y"]), configs)
    (study / "cases/2017-180/complete.json").write_text(
        json.dumps(_hashes(study / "cases/2017-180", [f"memory/{n}" for n in names]))
    )
    registry = SimpleNamespace(
        elections={
            2017: SimpleNamespace(
                round_=lambda r: SimpleNamespace(
                    results=[
                        SimpleNamespace(candidate_id=c, share=s)
                        for c, s in (
                            OUTCOME.shares.items() if r == 1 else [("a", 60.0), ("b", 40.0)]
                        )
                    ]
                )
            )
        }
    )
    summaries, deltas = compare_study(study, reference, registry)
    assert deltas.height == 2  # deux méthodes
    row = deltas.row(0, named=True)
    assert row["n_universe"] == 5  # a, b, c, x, y
    assert row["delta_log_presence_winner"] > 0 and row["better_log_presence_winner"]
    assert set(summaries["n_universe"]) == {5}
    # Une sortie de variante modifiée est refusée.
    np.savez(variant / "draws_mean14d.npz", **toy_sim(normalize(ROWS)))
    with pytest.raises(ValueError, match="modifiée"):
        compare_study(study, reference, registry)
