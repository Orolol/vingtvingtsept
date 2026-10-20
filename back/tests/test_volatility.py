"""Volatilité réalisée : indice sur un cas jouet, mise à l'échelle de η, plancher, κ en
leave-one-election-out."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from vvs.model import error as er
from vvs.model import volatility as vo

ASOF = dt.date(2022, 3, 1)
BLOCS = ["far_right", "right", "left", "left"]
P = np.array([0.30, 0.25, 0.30, 0.15])


def toy_polls(path: dict[str, list[float]], n: int = 10**9, per_week: int = 2) -> pl.DataFrame:
    """Sondages jouets : `per_week` sondages par semaine (semaine k = 0 la plus récente), parts
    en points données par `path[c][k]`, effectif `n` (quasi infini : pas de bruit)."""
    rows = []
    weeks = len(next(iter(path.values())))
    for k in range(weeks):
        for j in range(per_week):
            end = ASOF - dt.timedelta(days=7 * k + j)
            pid = f"p{k}-{j}"
            for c, s in path.items():
                rows.append({"poll_id": pid, "pollster": f"i{j}", "fieldwork_start": end,
                             "fieldwork_end": end, "sample_size": n, "is_rolling": False,
                             "population": "all", "hypothesis_id": "h", "candidate_id": c,
                             "share": s[k]})
    return pl.DataFrame(rows)


def test_index_flat_series_is_zero_and_trend_is_recovered():
    flat = {"a": [40.0] * 8, "b": [35.0] * 8, "c": [25.0] * 8}
    v = vo.realized_volatility(toy_polls(flat), ASOF, ["a", "b", "c"], ["right", "left", "left"])
    assert v.campaign == pytest.approx(0.0, abs=1e-9)
    # a croît de 2 % (en log) par semaine depuis 8 semaines, b et c décroissent ensemble
    g = 0.02
    a = [40.0 * math.exp(g * (7 - k)) for k in range(8)]
    rest = [100 - x for x in a]
    trend = {"a": a, "b": [0.6 * r for r in rest], "c": [0.4 * r for r in rest]}
    v = vo.realized_volatility(toy_polls(trend), ASOF, ["a", "b", "c"], ["right", "left", "left"])
    # tendance pure : Σ Δ² / Σ écart = g² · Σ écart² / Σ écart sur les paires de la fenêtre
    gaps = [k2 - k1 for k1 in range(8) for k2 in range(k1 + 1, 8)]
    expect_a = g * math.sqrt(sum(x * x for x in gaps) / sum(gaps))
    assert v.by_unit["a"] == pytest.approx(expect_a, rel=1e-6)
    assert v.by_bloc["right"] == pytest.approx(expect_a, rel=1e-6)
    # b et c ont la même variation relative : même volatilité, égale à celle du bloc gauche
    assert v.by_unit["b"] == pytest.approx(v.by_unit["c"], rel=1e-5)
    assert v.by_bloc["left"] == pytest.approx(v.by_unit["b"], rel=1e-5)
    assert v.n_weeks == 8


def test_index_ignores_future_polls_and_needs_three_weeks():
    path = {"a": [40.0, 50.0, 40.0, 50.0], "b": [60.0, 50.0, 60.0, 50.0]}
    df = toy_polls(path)
    # deux semaines visibles seulement (k = 2, 3) : indéfini
    v = vo.realized_volatility(df, ASOF - dt.timedelta(days=14), ["a", "b"], ["left", "right"])
    assert math.isnan(v.campaign)
    v_all = vo.realized_volatility(df, ASOF, ["a", "b"], ["left", "right"])
    assert v_all.campaign > 0


def test_sampling_noise_is_removed():
    rng = np.random.default_rng(0)
    n = 1000
    vals = []
    for _ in range(300):
        weeks = {k: [(float(rng.binomial(n, 0.3)) / n, int(n / vo.DESIGN_FACTOR))
                     for _ in range(3)] for k in range(8)}
        corr, raw, _ = vo._unit_volatility(weeks, 8)
        vals.append((corr**2, raw**2))
    corr2, raw2 = np.mean(vals, axis=0)
    # bruit seul : la version brute vaut environ 2·(1 − p)/(p·n·m), la corrigée environ 0
    assert raw2 == pytest.approx(2 * 0.7 / (0.3 * n * 3), rel=0.25)
    assert corr2 < 0.25 * raw2


def params() -> er.ErrorParams:
    return er.ErrorParams(sigma_bloc_final=0.06, sigma_cand_final=0.08, rho=1e-3, a_bloc=0.05,
                          a_cand=0.04, gamma=0.5, nu=8.0, final_age=7.0)


def test_kappa_zero_or_unit_ratio_is_identical_to_current_model():
    cands = [f"c{i}" for i in range(len(P))]
    theta = np.log(P)[None, :]
    base = er.ErrorModel(params())
    ref = base.sample(theta, 90, cands, BLOCS, np.random.default_rng(3), 4000)
    for vs in (vo.VolatilityScale(kappa=0.0, ratio=2.5, form="total"),
               vo.VolatilityScale(kappa=1.3, ratio=1.0, form="move")):
        m = er.ErrorModel(params())
        m.vol_scale = vs
        out = m.sample(theta, 90, cands, BLOCS, np.random.default_rng(3), 4000)
        assert np.array_equal(out, ref)
        assert np.allclose(m.covariance(90, BLOCS, P), base.covariance(90, BLOCS, P))


@pytest.mark.parametrize("h", [7, 90, 180])
def test_scaling_forms_and_floor(h):
    base = er.ErrorModel(params())
    vb0, vc0 = base.variances(h, BLOCS, P)
    m = er.ErrorModel(params())
    m.vol_scale = vo.VolatilityScale(kappa=1.0, ratio=0.5, form="total")
    vb, vc = m.variances(h, BLOCS, P)
    assert np.allclose(vb, 0.25 * vb0) and np.allclose(vc, 0.25 * vc0)
    # forme move : seule la part de mouvement rétrécit, l'erreur finale reste un plancher
    m.vol_scale = vo.VolatilityScale(kappa=1.0, ratio=0.5, form="move")
    vb, vc = m.variances(h, BLOCS, P)
    pr = base.params
    vb_fin = np.ones(len(P)) * pr.sigma_bloc_final**2
    vc_fin = pr.sigma_cand_final**2 + pr.rho / P
    assert np.all(vb >= vb_fin - 1e-12) and np.all(vc >= vc_fin - 1e-12)
    assert np.allclose(vb - vb_fin, 0.25 * (vb0 - vb_fin))
    m.vol_scale = vo.VolatilityScale(kappa=1.0, ratio=1e-6, form="move")
    vb, vc = m.variances(h, BLOCS, P)
    assert np.allclose(vb, vb_fin, atol=1e-9) and np.allclose(vc, vc_fin, atol=1e-9)
    # par bloc : le bloc à rapport 1 ne bouge pas
    m.vol_scale = vo.VolatilityScale(kappa=1.0, ratio=1.0, form="total",
                                     bloc_ratio={"left": 2.0})
    vb, _ = m.variances(h, BLOCS, P)
    assert np.allclose(vb[:2], vb0[:2]) and np.allclose(vb[2:], 4 * vb0[2:])


def test_reference_and_ratio():
    vols = {(2002, 90): 0.02, (2007, 90): 0.08, (2012, 90): math.nan, (2017, 90): 0.04}
    ref = vo.reference(vols, [2002, 2007, 2012], horizons=(90,))
    assert ref[90] == pytest.approx(0.04)  # moyenne géométrique, NaN ignoré
    assert vo.ratio(0.08, ref[90]) == pytest.approx(2.0)
    assert vo.ratio(math.nan, ref[90]) == 1.0
    assert vo.interp_reference({30: 0.02, 120: 0.08}, 60) == pytest.approx(0.04)


def toy_cal_items(kappa_true: float, seed: int = 0):
    """Élections jouets dont l'ampleur des mouvements suit V^κ ; V varie d'une élection à
    l'autre."""
    rng = np.random.default_rng(seed)
    pr = params()
    blocs = ["far_right", "right", "centre", "left", "far_left", "left"]
    p = np.array([0.22, 0.2, 0.2, 0.15, 0.15, 0.08])
    items, vols = [], {}
    for e, v in zip(range(2000, 2008), (0.02, 0.08, 0.04, 0.03, 0.06, 0.05, 0.025, 0.07)):
        for H in (1, 30, 60, 90, 120, 180):
            vols[(e, H)] = v
        fin = er.Item(e, "final", 0.0, [f"c{i}" for i in range(6)], blocs, p, np.zeros(6))
        fin.d = rng.multivariate_normal(np.zeros(6), er.item_cov(
            fin, pr.sigma_bloc_final**2, pr.sigma_cand_final**2, pr.rho, pr.a_bloc, pr.a_cand,
            {b: 1.0 for b in er.BLOCS}, pr.gamma))
        items.append(vo.CalItem(fin, 0))
        for H in (30, 60, 90, 120, 180):
            it = er.Item(e, "move", float(H), fin.units, blocs, p, np.zeros(6), weight=0.2)
            s2 = (v / 0.04) ** (2 * kappa_true)
            cov = er.item_cov(it, 0, 0, pr.rho, s2 * pr.a_bloc, s2 * pr.a_cand,
                              {b: 1.0 for b in er.BLOCS}, pr.gamma)
            it.d = rng.multivariate_normal(np.zeros(6), cov)
            items.append(vo.CalItem(it, H))
    return items, vols


def test_scale_items_move_form_equals_scaled_covariance():
    items, vols = toy_cal_items(1.0)
    train = sorted({ci.item.election for ci in items})
    ref = vo.reference(vols, train)
    ci = next(c for c in items if c.item.kind == "move")
    s = vols[(ci.item.election, ci.horizon)] / ref[ci.horizon]
    sc_it = vo.scale_items([ci], vols, train, 1.0, "move", 0.5)[0]
    a = er.item_cov(sc_it, 0, 0, 1e-3, 0.05, 0.04, {b: 1.0 for b in er.BLOCS}, 0.5)
    b = er.item_cov(ci.item, 0, 0, 1e-3, s**2 * 0.05, s**2 * 0.04, {b: 1.0 for b in er.BLOCS}, 0.5)
    assert np.allclose(a, b)


def test_fit_kappa_leave_one_out_recovers_sign():
    items, vols = toy_cal_items(1.0, seed=1)
    train = [2001, 2002, 2003, 2004, 2005]  # 2000, 2006, 2007 laissées de côté
    res = vo.fit_kappa(items, vols, train, "move", 0.5, nus=(math.inf,), grid=(0.0, 1.0))
    assert res["kappa"] == 1.0 and res["gain_vs_kappa0"] > 0
    assert res["train"] == train


def test_cv_score_uses_only_training_elections():
    items, vols = toy_cal_items(1.0, seed=2)
    train = [2001, 2002, 2003]
    s = vo.cv_score(items, vols, train, 0.5, "total", 0.5, math.inf)
    # l'élection hors apprentissage n'a aucun effet : la perturber ne change pas le score
    for ci in items:
        if ci.item.election == 2005:
            ci.item.d = ci.item.d * 10
    vols[(2005, 90)] = 10.0
    assert vo.cv_score(items, vols, train, 0.5, "total", 0.5, math.inf) == pytest.approx(s)


def test_volatility_variant_registered():
    from vvs.eval.backtest import VARIANTS

    assert VARIANTS["volatility"] == VARIANTS["floor"]
