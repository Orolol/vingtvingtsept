"""Décalage commun des reports vers le RN et centrage injectable (vvs.model.rn_image)."""

import datetime as dt

import numpy as np
import pytest

from vvs.model import rn_image as R
from vvs.model.second_round import (
    ABSTAIN,
    DuelPolls,
    ErrorParams,
    GroupPrior,
    PairFit,
    sample_pair,
)

ERR = ErrorParams(sigma0=2.0, sigma1=1.0, bias_far_right=0.0, sigma_house=0.5)


def _fit(blocs=("far_right", "centre")) -> PairFit:
    groups = {
        "own:x": GroupPrior("own:x", (0.95, 0.02, 0.03), 60.0, False, "own"),
        "own:c": GroupPrior("own:c", (0.02, 0.95, 0.03), 60.0, False, "own"),
        "bloc:left": GroupPrior("bloc:left", (0.1, 0.7, 0.2), 30.0, True, "analog"),
        "bloc:far_right": GroupPrior("bloc:far_right", (0.6, 0.3, 0.1), 30.0, True, "analog"),
        ABSTAIN: GroupPrior(ABSTAIN, (0.15, 0.25, 0.6), 30.0, True, "analog"),
    }
    return PairFit(
        election=2027, asof=dt.date(2027, 4, 1), pair=("x", "c"), blocs=blocs, groups=groups,
        sigma_u=0.0, sigma_v=0.0, polls=DuelPolls.empty(), errors=ERR, gap_days=14,
        abstain_weight=0.3, bloc_of=lambda c: {"x": "far_right", "c": "centre", "l": "left"}[c],
    )


def _split(g: GroupPrior) -> float:
    return g.mean[0] / (g.mean[0] + g.mean[1])


@pytest.mark.parametrize("u", [-1.9, 0.0, 0.43])
def test_common_shift_recovers_a_uniform_shift(u):
    ref = np.array([0.05, 0.3, 0.6, 0.9])
    obs = R.expit(R.logit(ref) + u)
    assert R.common_shift(ref, obs, [0.3, 0.2, 0.1, 0.4]) == pytest.approx(u, abs=1e-6)


def test_common_shift_weights_votes_not_logits():
    # un petit électorat très décalé pèse selon ses voix : u reste près du grand électorat
    ref, obs = [0.3, 0.02], [0.3, 0.5]
    u = R.common_shift(ref, obs, [0.95, 0.05])
    assert 0 < u < 0.5


def test_aggregate_shift_inverts_the_share_and_flags_unreachable_targets():
    ref, w = [0.3, 0.1], [0.4, 0.3]
    fixed_v, fixed_w = 0.2, 0.3
    share = (fixed_v + sum(wi * R.expit(R.logit(r) - 0.4) for wi, r in zip(w, ref))) / (fixed_w + sum(w))
    assert R.aggregate_shift(fixed_v, fixed_w, ref, w, float(share)) == pytest.approx(-0.4, abs=1e-6)
    with pytest.raises(ValueError, match="hors d'atteinte"):
        R.aggregate_shift(fixed_v, fixed_w, ref, w, 0.15)  # sous la part des seuls électorats fixes


@pytest.mark.parametrize("blocs", [("far_right", "centre"), ("centre", "far_right")])
def test_shift_fit_moves_only_barrage_groups_towards_the_rn(blocs):
    f = _fit(blocs)
    g = R.shift_fit(f, 0.5)
    rn_first = blocs[0] == "far_right"
    for k in ("bloc:left", ABSTAIN):
        before, after = _split(f.groups[k]), _split(g.groups[k])
        d = R.logit(after) - R.logit(before)
        assert d == pytest.approx(0.5 if rn_first else -0.5)
        assert g.groups[k].mean[2] == f.groups[k].mean[2]  # « ni l'un ni l'autre » inchangé
        assert sum(g.groups[k].mean) == pytest.approx(1.0)
    for k in ("own:x", "own:c", "bloc:far_right"):
        assert g.groups[k] == f.groups[k]
    assert R.shift_fit(f, 0.0) is f
    assert R.shift_fit(_fit(("left", "centre")), 0.5).groups == _fit(("left", "centre")).groups


def test_shift_raises_the_rn_share_in_the_prior():
    first = {"x": np.array([0.3]), "c": np.array([0.3]), "l": np.array([0.4])}
    shares = []
    for d in (0.0, 0.6):
        draws = sample_pair(R.shift_fit(_fit(), d), first, 30, np.random.default_rng(0), n=4000)
        shares.append(float(np.mean(draws.prior_share)))
    assert shares[1] > shares[0] + 2


def test_fit_line_exact_and_predictive_sd():
    x, y = [70.0, 58.0, 50.0], [-1.9, -0.3, 0.1]
    a = R.fit_line(x[:2], y[:2], tau=0.3)
    # deux points : la droite passe par les deux
    assert a.a + a.b * (x[0] - a.x0) == pytest.approx(y[0])
    assert a.b == pytest.approx((y[1] - y[0]) / (x[1] - x[0]))
    _, sd = a.predict(40.0)
    assert sd > 0.3  # τ plus l'incertitude de la droite, plus grande loin des points
    assert a.predict(a.x0)[1] < sd
    b, se = R.ols_slope(x, y)
    assert b < 0 and se is not None and se > 0
    assert R.ols_slope(x[:2], y[:2])[1] is None


def test_source_cells_average_members_by_first_round_share():
    from vvs.registry import default_registry
    from vvs.registry.transfers import default_transfers, transfer_prior

    reg = default_registry()
    pair = ("le_pen_marine", "macron_emmanuel")
    src = [s for s in default_transfers() if s.id == "harris_2022_expressed"]
    w = R.first_round_weights(reg, 2022)
    cells = R.source_cells(reg, 2022, pair, src, w)
    fl = ("roussel_fabien", "melenchon_jean_luc")
    exp = sum(w[c] * transfer_prior(2022, c, pair, basis="expressed", transfers=src).shares[pair[0]]
              for c in fl) / sum(w[c] for c in fl) / 100
    assert cells["bloc:far_left"].split == pytest.approx(exp)
    assert cells["bloc:far_left"].none is None  # base exprimés : pas de part « ni l'un ni l'autre »
    assert R.barrage_group("bloc:far_left") and not R.barrage_group("bloc:far_right")
    assert R.rn_pair(reg, 2022, pair[::-1]) == pair
