"""Volatilité par candidat comme échelle de v_c : indice sur cas jouet, κ = 0 identique au modèle
actuel, bornes, leave-one-election-out."""

from __future__ import annotations

import datetime as dt
import math
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from vvs.model import error as er
from vvs.model import volatility as vo
from vvs.model import volatility_candidate as vc

ASOF = dt.date(2022, 3, 1)


def toy_polls(path: dict[str, list[float]], n: int = 10**9) -> pl.DataFrame:
    rows = []
    for k in range(len(next(iter(path.values())))):
        for j in range(2):
            end = ASOF - dt.timedelta(days=7 * k + j)
            for c, s in path.items():
                rows.append({"poll_id": f"p{k}-{j}", "fieldwork_end": end, "sample_size": n,
                             "hypothesis_id": "h", "candidate_id": c, "share": s[k]})
    return pl.DataFrame(rows)


def table(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"election": pl.Int64, "horizon": pl.Int64,
                                      "unit": pl.Utf8, "V": pl.Float64, "p": pl.Float64})


def test_candidate_index_on_toy_case_ranks_the_moving_candidate():
    # a gagne 3 % (log) par semaine, b est stable, c et d absorbent le mouvement
    g = 0.03
    a = [10.0 * math.exp(g * (7 - k)) for k in range(8)]
    path = {"a": a, "b": [30.0] * 8, "c": [0.5 * (70 - x) for x in a],
            "d": [0.5 * (70 - x) for x in a]}
    v = vo.realized_volatility(toy_polls(path), ASOF, list(path), ["left", "right", "centre", "other"])
    assert v.by_unit["b"] < 0.02 < v.by_unit["a"]
    # référence : médiane de la classe de taille de a (8-15 %) aux horizons voisins
    ref = vc.Reference.build(table([
        {"election": 1, "horizon": 30, "unit": "x", "V": 0.01, "p": 0.10},
        {"election": 1, "horizon": 30, "unit": "y", "V": 0.02, "p": 0.12},
        {"election": 1, "horizon": 45, "unit": "z", "V": 0.03, "p": 0.11},
        {"election": 1, "horizon": 30, "unit": "big", "V": 0.50, "p": 0.40},
    ]), [1])
    assert ref.vref(0.10, 30) == pytest.approx(0.02)  # la classe ≥ 15 % n'entre pas
    cv = vc.CandidateVolatility(kappa=0.5, ref=ref)
    s = cv.scales(v.by_unit, v.weights, ["a", "b", "absent"], 30)
    assert s[0] > 1.0 and s[1] < 1.0 and s[2] == 1.0


def test_bounds_on_ratio_and_scale():
    ref = vc.Reference.build(table([{"election": 1, "horizon": 7, "unit": "x", "V": 0.04,
                                     "p": 0.2}]), [1])
    assert vc.ratio(0.0, 0.2, 7, ref) == pytest.approx(vc.RATIO_BOUNDS[0])
    assert vc.ratio(10.0, 0.2, 7, ref) == pytest.approx(vc.RATIO_BOUNDS[1])
    assert math.isnan(vc.ratio(0.04, 0.02, 7, ref))  # classe vide : pas de référence
    s = vc.scale_from_ratio([0.25, 4.0, math.nan, 1.0], 1.0)
    assert s == pytest.approx([vc.S_BOUNDS[0], vc.S_BOUNDS[1], 1.0, 1.0])
    assert vc.scale_from_ratio([0.25, 4.0], 0.0) == pytest.approx([1.0, 1.0])


def _params() -> er.ErrorParams:
    return er.ErrorParams(0.06, 0.08, 1e-3, 0.05, 0.04, 0.5, 8.0, 7.0, {}, {})


def test_kappa_zero_is_the_current_model():
    blocs = ["right", "left", "left", "centre"]
    p = np.array([0.3, 0.2, 0.1, 0.4])
    cands = ["a", "b", "c", "d"]
    ref = vc.Reference.build(table([{"election": 1, "horizon": 30, "unit": "x", "V": 0.04,
                                     "p": 0.2}]), [1])
    by_unit = {"a": 0.0, "b": 0.5, "c": 0.04, "d": 0.08}
    weights = dict(zip(cands, p))
    base = er.ErrorModel(_params())
    out0 = base.sample(np.log(p)[None, :], 30, cands, blocs, np.random.default_rng(3), 2000)
    m = er.ErrorModel(_params())
    cv = vc.CandidateVolatility(kappa=0.0, ref=ref)
    m.cand_scale = lambda c: cv.scales(by_unit, weights, c, 30)
    out1 = m.sample(np.log(p)[None, :], 30, cands, blocs, np.random.default_rng(3), 2000)
    assert np.array_equal(out0, out1)
    # même vraisemblance et même estimation qu'error quand toutes les échelles valent 1
    it = er.Item(1, "final", 0.0, cands, blocs, p, np.array([0.1, -0.2, 0.05, 0.0]))
    x = np.concatenate([[-2.0, -2.5, -7.0, -3.0, -3.0], np.zeros(len(er.BLOCS))])
    assert vc._neg_objective_scaled(x, [it], [None], 0.5, 8.0, {}) == pytest.approx(
        er._neg_objective(x, [it], 0.5, 8.0, {}))
    assert vc._neg_objective_scaled(x, [it], [np.ones(4)], 0.5, 8.0, {}) == pytest.approx(
        er._neg_objective(x, [it], 0.5, 8.0, {}))
    # κ > 0 : seule la composante propre change, plus large pour le candidat volatil
    cv1 = vc.CandidateVolatility(kappa=1.0, ref=ref)
    s = cv1.scales(by_unit, weights, cands, 30)
    assert s[1] == vc.S_BOUNDS[1] and s[0] == vc.S_BOUNDS[0]


def toy_cal(n_elections: int, kappa_true: float, seed: int = 0):
    """Erreurs finales dont l'écart type propre suit (V / V_ref)^κ, et leur table de V."""
    rng = np.random.default_rng(seed)
    K = 12
    p = np.full(K, 0.05)
    blocs = ["left", "right"] * (K // 2)
    rows, items = [], []
    for e in range(n_elections):
        V = 0.04 * np.exp(rng.normal(0, 0.7, K))
        s = np.clip((np.clip(V / 0.04, *vc.RATIO_BOUNDS)) ** kappa_true, *vc.S_BOUNDS)
        d = rng.normal(0, 0.15 * s) + rng.normal(0, 0.02, K)
        units = [f"c{i}" for i in range(K)]
        it = er.Item(e, "final", 0.0, units, blocs, p, d)
        items.append(vo.CalItem(it, 0))
        rows += [{"election": e, "horizon": 1, "unit": u, "V": float(x), "p": 0.05}
                 for u, x in zip(units, V)]
    return items, table(rows)


def test_kappa_recovered_in_internal_cross_validation():
    items, tab = toy_cal(8, 1.0)
    prof = vc.cv_profile(items, [], tab, list(range(8)), 0.5, math.inf, grid=(0.0, 1.0),
                         use_bias=False)
    assert prof[1.0] > prof[0.0] + 2
    items, tab = toy_cal(8, 0.0, seed=1)
    prof = vc.cv_profile(items, [], tab, list(range(8)), 0.5, math.inf, grid=(0.0, 1.0),
                         use_bias=False)
    assert prof[0.0] > prof[1.0]


def test_leave_one_election_out():
    # V_ref ne voit pas l'élection laissée de côté
    rows = [{"election": 1, "horizon": 7, "unit": "a", "V": 0.04, "p": 0.2},
            {"election": 2, "horizon": 7, "unit": "b", "V": 9.0, "p": 0.2}]
    assert vc.Reference.build(table(rows), [1]).vref(0.2, 7) == pytest.approx(0.04)
    # ErrorModel.fit passe à attach les seules élections d'apprentissage
    seen = {}

    def fake_attach(model, train, extra, gamma, bias, reg, use_bias=True):
        seen["train"] = list(train)
        model.cand_volatility = SimpleNamespace(kappa=0.0)

    items = [ci.item for ci in toy_cal(4, 0.0)[0]]
    mp = pytest.MonkeyPatch()
    mp.setattr(vc, "attach", fake_attach)
    try:
        m = er.ErrorModel.fit(elections=range(4), exclude=[2], items=items, nu=math.inf,
                              gamma=0.5, use_bias=False, registry=SimpleNamespace(),
                              candidate_volatility=True)
    finally:
        mp.undo()
    assert seen["train"] == [0, 1, 3]
    assert m.cand_volatility.kappa == 0.0
    # sans l'option : rien n'est attaché
    m0 = er.ErrorModel.fit(elections=range(4), exclude=[2], items=items, nu=math.inf, gamma=0.5,
                           use_bias=False, registry=SimpleNamespace())
    assert m0.cand_volatility is None and m0.cand_scale is None
