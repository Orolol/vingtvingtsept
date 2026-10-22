"""Troupeau des instituts : dispersion relative sur cas jouets, effet de l'effectif, sélection
des sondages, mise à l'échelle de η et option désactivée."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from vvs.model import error as er
from vvs.model import herding as hd

END = dt.date(2022, 4, 9)
UNITS = ["a", "b", "c", "d", "e"]
UBLOCS = ["right", "right", "left", "far_right", "centre"]
TRUE = np.array([0.30, 0.25, 0.20, 0.15, 0.10])


def point(i: int, shares, n: float, day: int = 0, step: float = 0.0, pollster=None,
          end: dt.date = END) -> hd.PollPoint:
    return hd.PollPoint(poll_id=f"p{i}", pollster=pollster or f"i{i}", end=end,
                        mid=float(day), n=float(n), n_imputed=False, step=step,
                        shares=dict(zip(UNITS, shares)))


def random_polls(rng, m: int = 7, herd: float = 0.0) -> list[hd.PollPoint]:
    """m instituts, tirage multinomial aux effectifs déclarés ; `herd` : fraction de l'écart à
    la moyenne retirée (troupeau)."""
    ns = rng.choice([800, 1000, 1500, 3000], size=m)
    x = np.array([rng.multinomial(n, TRUE) / n for n in ns])
    x = x - herd * (x - x.mean(0))
    return [point(i, 100 * x[i], ns[i], day=int(rng.integers(0, 14))) for i in range(m)]


@pytest.mark.parametrize("detrend", [True, False])
def test_relative_dispersion_is_one_under_random_sampling(detrend):
    rng = np.random.default_rng(0)
    ds = [hd.dispersion(random_polls(rng), UNITS, UBLOCS, detrend=detrend, jackknife=False)
          for _ in range(400)]
    assert np.mean([d.D for d in ds]) == pytest.approx(1.0, abs=0.06)
    assert np.mean([d.D_bloc for d in ds]) == pytest.approx(1.0, abs=0.08)
    # troupeau : la moitié de l'écart à la moyenne retirée divise la variance par 4
    rng = np.random.default_rng(1)
    dh = [hd.dispersion(random_polls(rng, herd=0.5), UNITS, UBLOCS, detrend=detrend,
                        jackknife=False).D for _ in range(400)]
    assert np.mean(dh) == pytest.approx(0.25, abs=0.03)


def test_declared_sample_size_scales_the_ratio():
    rng = np.random.default_rng(2)
    pts = random_polls(rng)
    d1 = hd.dispersion(pts, UNITS, UBLOCS, jackknife=False)
    # mêmes parts, effectifs déclarés doublés : variance attendue / 2, rapport × 2 (sans arrondi)
    big = [hd.PollPoint(**{**p.__dict__, "n": 2 * p.n}) for p in pts]
    d2 = hd.dispersion(big, UNITS, UBLOCS, jackknife=False)
    assert d2.D == pytest.approx(2 * d1.D, rel=1e-9)
    for u in d1.by_unit:
        assert d2.by_unit[u] == pytest.approx(2 * d1.by_unit[u], rel=1e-9)
    # tirages à 2 000, effectif déclaré 1 000 : D ≈ 0,5 (sur-dispersion apparente inversée)
    rng = np.random.default_rng(3)
    ds = []
    for _ in range(300):
        x = [rng.multinomial(2000, TRUE) / 2000 for _ in range(6)]
        ds.append(hd.dispersion([point(i, 100 * x[i], 1000, day=i) for i in range(6)], UNITS,
                                UBLOCS, jackknife=False).D)
    assert np.mean(ds) == pytest.approx(0.5, abs=0.05)


def test_rounding_enters_the_expected_variance_and_step_is_read():
    assert hd.rounding_step([30.0, 25.0, 45.0]) == 1.0
    assert hd.rounding_step([30.5, 25.0, 44.5]) == 0.5
    assert hd.rounding_step([30.3, 25.0, 44.7]) == 0.1
    rng = np.random.default_rng(4)
    pts = random_polls(rng)
    d0 = hd.dispersion(pts, UNITS, UBLOCS, jackknife=False)
    rounded = [hd.PollPoint(**{**p.__dict__, "step": 1.0}) for p in pts]
    d1 = hd.dispersion(rounded, UNITS, UBLOCS, jackknife=False)
    assert d1.D < d0.D  # même dispersion observée, attendu plus grand


def test_linear_trend_is_removed_and_too_few_pollsters_is_undefined():
    # parts exactes sur une tendance linéaire : dispersion nulle après retrait de tendance
    pts = []
    for i in range(6):
        a = 0.30 + 0.004 * i
        x = np.array([a, 0.25, 0.20, 0.15, 0.10 - 0.004 * i])
        pts.append(point(i, 100 * x, 1000, day=i))
    assert hd.dispersion(pts, UNITS, UBLOCS).D == pytest.approx(0.0, abs=1e-12)
    assert hd.dispersion(pts, UNITS, UBLOCS, detrend=False).D > 0
    assert math.isnan(hd.dispersion(pts[:3], UNITS, UBLOCS).D)


def test_select_points_latest_per_pollster_and_common_units():
    old = point(0, [30, 25, 20, 15, 10], 1000, pollster="x", end=END - dt.timedelta(days=5))
    new = point(1, [31, 24, 20, 15, 10], 1000, pollster="x")
    del new.shares["e"]  # le dernier sondage de x omet e : on garde le précédent
    others = [point(i, [30, 25, 20, 15, 10], 1000, pollster=f"q{i}") for i in range(2, 6)]
    pts, common = hd.select_points([old, new, *others], UNITS)
    assert common == UNITS
    assert [p.poll_id for p in pts if p.pollster == "x"] == ["p0"]
    assert len(pts) == 5
    # unité testée par moins de 75 % des derniers sondages : écartée, aucun sondage perdu
    for p in others[:2]:
        del p.shares["e"]
    pts, common = hd.select_points([old, new, *others], UNITS)
    assert common == UNITS[:4] and len(pts) == 5


def test_reference_scale_factor_and_bounds():
    dv = {(2002, 30): 0.5, (2007, 30): 2.0, (2012, 30): math.nan}
    ref = hd.reference(dv, [2002, 2007, 2012], horizons=(30,))
    assert ref[30] == pytest.approx(1.0)
    assert hd.scale_factor(0.5, ref[30], 1.0) == pytest.approx(2.0)
    assert hd.scale_factor(0.5, ref[30], -1.0) == pytest.approx(0.5)
    assert hd.scale_factor(0.01, ref[30], 1.0) == hd.S_BOUNDS[1]
    assert hd.scale_factor(math.nan, ref[30], 1.0) == 1.0
    assert hd.interp_reference({1: 9.0, 30: 1.0, 120: 4.0}, 60) == pytest.approx(2.0)
    assert hd.interp_reference({1: 9.0, 30: 1.0, 120: 4.0}, 1) == 9.0


def params() -> er.ErrorParams:
    return er.ErrorParams(sigma_bloc_final=0.06, sigma_cand_final=0.08, rho=1e-3, a_bloc=0.05,
                          a_cand=0.04, gamma=0.5, nu=8.0, final_age=7.0)


BLOCS = ["far_right", "right", "left", "left"]
P = np.array([0.30, 0.25, 0.30, 0.15])


def test_disabled_or_unit_scale_is_identical_to_current_model():
    cands = [f"c{i}" for i in range(len(P))]
    theta = np.log(P)[None, :]
    base = er.ErrorModel(params())
    assert base.herding is None and base.herd_scale is None
    ref = base.sample(theta, 90, cands, BLOCS, np.random.default_rng(3), 4000)
    for form in hd.FORMS:
        m = er.ErrorModel(params())
        m.herd_scale = hd.HerdingScale(s=1.0, form=form)
        out = m.sample(theta, 90, cands, BLOCS, np.random.default_rng(3), 4000)
        assert np.array_equal(out, ref)


@pytest.mark.parametrize("h", [7, 90, 180])
def test_scaling_forms(h):
    base = er.ErrorModel(params())
    vb0, vc0 = base.variances(h, BLOCS, P)
    m = er.ErrorModel(params())
    m.herd_scale = hd.HerdingScale(s=2.0, form="total")
    vb, vc = m.variances(h, BLOCS, P)
    assert np.allclose(vb, 4 * vb0) and np.allclose(vc, 4 * vc0)
    # forme final : seule l'erreur finale (σ²_final, ρ/p) est multipliée, le mouvement reste
    m.herd_scale = hd.HerdingScale(s=2.0, form="final")
    vb, vc = m.variances(h, BLOCS, P)
    pr = base.params
    assert np.allclose(vb - vb0, 3 * pr.sigma_bloc_final**2)
    assert np.allclose(vc - vc0, 3 * (pr.sigma_cand_final**2 + pr.rho / P))


def test_kappa_zero_leaves_items_unchanged_and_jacobian_matches():
    it = er.Item(2002, "final", 0.0, ["a", "b", "c"], ["right", "left", "centre"],
                 np.array([0.3, 0.3, 0.4]), np.array([0.1, -0.05, 0.0]))
    dv = {(2002, 1): 0.5, (2007, 1): 2.0}
    same, jac = hd.scale_items([it], [1], dv, [2002, 2007], 0.0, "total")
    assert same[0] is it and jac == 0.0
    sc, jac = hd.scale_items([it], [1], dv, [2002, 2007], 1.0, "total")
    assert np.allclose(sc[0].d, it.d / 2.0)
    assert jac == pytest.approx(-2 * math.log(2.0))
    # forme final : les mouvements ne sont pas mis à l'échelle
    mv = er.Item(2002, "move", 60.0, it.units, it.blocs, it.p, it.d)
    out, jac = hd.scale_items([mv], [90], {(2002, 90): 0.5, (2007, 90): 2.0}, [2002, 2007],
                              1.0, "final")
    assert out[0] is mv and jac == 0.0


def test_herding_variants_registered():
    from vvs.eval.backtest import VARIANT_ERROR_KWARGS, VARIANTS

    for v in ("herding", "lists_herding", "lists_herding_final"):
        assert VARIANTS[v] == VARIANTS["floor"]
    assert VARIANT_ERROR_KWARGS["lists_herding"] == {"with_lists": True, "herding": True}
    with pytest.raises(ValueError):
        hd.resolve_form("move")
