"""Backtest : configurations connues à une date, correspondance avec les candidats officiels,
enchaînement ajustement → η → métriques sur une élection jouet."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from vvs import paths
from vvs.eval.configurations import OfficialMap, config_from_polls, known_configuration

HAS_CURATED = (paths.CURATED / "polls.parquet").exists()
needs_curated = pytest.mark.skipif(not HAS_CURATED, reason="data/curated absent")


def test_config_from_polls_keeps_most_tested_of_exclusive_alternatives():
    # deux sondages ; A et B jamais testés ensemble (primaire), A plus souvent
    hyps = {
        "p1": {"h1": {"A": 30, "X": 40, "Y": 30}, "h2": {"B": 25, "X": 45, "Y": 30}},
        "p2": {"h1": {"A": 28, "X": 42, "Y": 30}},
    }
    kept, weight, alt = config_from_polls(hyps)
    assert set(kept) == {"A", "X", "Y"}
    assert alt == {"B": "A"}
    assert weight["A"] == pytest.approx(0.75) and weight["X"] == pytest.approx(1.0)


@needs_curated
@pytest.mark.parametrize(
    ("asof", "present", "absent"),
    [
        # primaire de la droite (20 et 27 novembre 2016)
        ("2016-11-01", {"juppe_alain", "hollande_francois"}, {"fillon_francois", "hamon_benoit"}),
        ("2016-12-20", {"fillon_francois"}, {"juppe_alain", "sarkozy_nicolas", "hamon_benoit"}),
        # primaire de la gauche (22 et 29 janvier 2017)
        ("2017-01-20", {"fillon_francois", "valls_manuel"}, {"hamon_benoit"}),
        ("2017-02-15", {"hamon_benoit", "fillon_francois"}, {"valls_manuel", "juppe_alain"}),
    ],
)
def test_known_configuration_2017_follows_primaries(asof, present, absent):
    k = known_configuration(2017, dt.date.fromisoformat(asof))
    assert k.source == "polls"
    assert present <= set(k.candidates)
    assert not (absent & set(k.candidates))


@needs_curated
def test_known_configuration_after_list_publication_is_official():
    from vvs.registry import default_registry

    official = {c.candidate_id for c in default_registry().elections[2017].candidates}
    k = known_configuration(2017, dt.date(2017, 3, 25))
    assert k.source == "official_list"
    assert set(k.candidates) == official


@needs_curated
def test_official_map_stand_ins_2017():
    from vvs.eval.configurations import official_map

    config = ["le_pen_marine", "juppe_alain", "hollande_francois", "duflot_cecile",
              "macron_emmanuel", "bayrou_francois", "melenchon_jean_luc"]
    m = official_map(config, 2017)
    assert m.rules["juppe_alain"] == ("split", ["fillon_francois"])  # même parti, absent
    assert m.rules["hollande_francois"] == ("split", ["hamon_benoit"])
    assert m.rules["duflot_cecile"] == ("split", ["hamon_benoit"])  # même bloc, absent
    assert m.rules["bayrou_francois"] == ("prorata", ["macron_emmanuel"])  # bloc présent
    shares = np.array([[0.25, 0.30, 0.10, 0.03, 0.12, 0.08, 0.12]])
    out = m.apply(shares)[0]
    idx = {c: i for i, c in enumerate(m.official)}
    assert out.sum() == pytest.approx(1.0)
    assert out[idx["fillon_francois"]] == pytest.approx(0.30)
    assert out[idx["hamon_benoit"]] == pytest.approx(0.13)
    assert out[idx["macron_emmanuel"]] == pytest.approx(0.20)
    assert not m.forecast[idx["lassalle_jean"]]


def test_official_map_drop_renormalizes():
    m = OfficialMap(
        config=["a", "b", "x"],
        official=["a", "b"],
        official_bloc=["right", "left"],
        rules={"a": ("self", ["a"]), "b": ("self", ["b"]), "x": ("drop", [])},
        forecast=np.array([True, True]),
    )
    out = m.apply(np.array([[0.3, 0.5, 0.2]]))[0]
    assert out == pytest.approx([0.375, 0.625])


# --- enchaînement sur une élection jouet


def test_toy_pipeline_svi_error_metrics():
    """Sondages synthétiques → ajustement SVI court → η → métriques contre un résultat jouet."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_model_first_round import BLOCS as TOY_BLOCS
    from test_model_first_round import CAND_BLOC, CANDS, synthetic

    from vvs.eval.configurations import OfficialMap
    from vvs.eval.scoring import Official, case_rows, score_case
    from vvs.model import first_round as fr
    from vvs.model.error import ErrorModel, ErrorParams
    from vvs.model.posterior import Posterior

    data, lat = synthetic(n_hyp=60, T=12)
    res = fr.fit(data, method="svi", svi_steps=600, num_samples=200, seed=0,
                 priors=fr.Priors(n_max=2500.0))
    assert "n_max" in res.samples
    post = Posterior(data, res)
    blocs = [TOY_BLOCS[b] for b in CAND_BLOC]
    true_theta = np.asarray(lat["theta"])[-1]
    truth = 100 * np.exp(true_theta) / np.exp(true_theta).sum()
    off = Official(CANDS, truth, blocs, tuple(sorted(np.argsort(-truth)[:2])))
    omap = OfficialMap(CANDS, CANDS, blocs, {c: ("self", [c]) for c in CANDS},
                       np.ones(len(CANDS), dtype=bool))
    em = ErrorModel(ErrorParams(0.05, 0.08, 1e-3, 0.04, 0.04, 0.4, 8.0, 7.0))
    rng = np.random.default_rng(0)
    case = score_case(post.theta[:, data.asof_week], post.theta[:, -1], CANDS, blocs, 7, em,
                      omap, off, {}, rng, n_draws=2000)
    assert set(case) == {"model", "model_noerror"}
    summ = case["model"][0]
    assert math.isfinite(summ["log_score_pair"]) and summ["log_score_pair"] <= 0
    assert 0 <= summ["brier_qual"] <= 1
    assert 0 < summ["crps"] < 5  # points
    # η élargit les intervalles : la couverture à 90 % ne baisse pas
    assert summ["cov90"] >= case["model_noerror"][0]["cov90"]
    m_rows, f_rows, c_rows = case_rows(case, {"election": 2099, "horizon": 7, "variant": "toy"})
    assert {r["metric"] for r in m_rows if r["level"] == "summary"} >= {"log_score_pair", "crps"}
    assert len(f_rows) == 2 * (len(CANDS) + len(set(blocs)))
    assert {r["nominal"] for r in c_rows} >= {0.5, 0.9}


@pytest.mark.slow
@needs_curated
def test_real_fit_and_score_2022_svi(tmp_path):
    """Ajustement SVI réel (2022 à 180 jours) puis score complet, sans sous-processus."""
    from vvs.eval import backtest as bt
    from vvs.eval.scoring import score_backtest

    settings = bt.FitSettings(method="svi", svi_steps=500, samples=200)
    bt.fit_one(2022, 180, bt.fit_dir(tmp_path, "floor", 2022, 180), bt.VARIANTS["floor"],
               settings)
    res = score_backtest(tmp_path, [2022], [180], n_draws=2000)
    s = res.metrics.filter(res.metrics["level"] == "summary")
    assert set(s["model"].unique().to_list()) == {"model", "model_noerror", "mean14",
                                                  "last_pollster"}
    assert (tmp_path / "metrics.md").exists() and (tmp_path / "calibration.png").exists()
