"""Reports publiés pour l'élection en cours (vvs.registry.transfers_current et option
`current_transfers` du second tour, docs/results/2026-11-24-reports-publies.md)."""

import datetime as dt

import numpy as np
import pytest

from vvs.model.second_round import (
    CURRENT_WINDOW,
    KAPPA_CURRENT,
    ErrorParams,
    SecondRoundModel,
    current_group_means,
    current_tables,
)
from vvs.registry import default_registry
from vvs.registry.transfers_current import CurrentTable, load_current_transfers

ERR = ErrorParams(sigma0=2.0, sigma1=1.0, bias_far_right=0.0, sigma_house=0.5)
PAIR = ("macron_emmanuel", "le_pen_marine")
ASOF = dt.date(2022, 3, 11)


def _table(tid="t", pollster="ifop", end=ASOF, published=None, election=2022, pair=PAIR,
           rows=None, hypotheses=None) -> CurrentTable:
    rows = rows or [("zemmour_eric", 10, 70, 20)]
    hyps = hypotheses or [rows]
    return CurrentTable.model_validate({
        "id": tid, "election": election, "pair": list(pair), "pollster": pollster,
        "sponsor": "s", "title": "t", "url": "u", "pages": "p",
        "fieldwork_start": end - dt.timedelta(days=2), "fieldwork_end": end,
        "published": published, "mode": "online", "base": "registered", "base_note": "b",
        "hypotheses": [
            {"label": f"h{k}", "rows": [
                {"from": c, "to": {pair[0]: a, pair[1]: b, "none": n}} for c, a, b, n in h
            ]}
            for k, h in enumerate(hyps)
        ],
    })


def test_registry_file_is_valid():
    tables = load_current_transfers()
    assert {t.election for t in tables} == {2012, 2017, 2022, 2027}
    for t in tables:
        assert t.available >= t.fieldwork_end
        for h in t.hypotheses:
            for r in h.rows:
                assert sum(t.vector(r, t.pair)) == pytest.approx(1.0)


def test_rejects_finalist_row_and_bad_sum():
    with pytest.raises(ValueError, match="finaliste"):
        _table(rows=[("le_pen_marine", 1, 97, 2)])
    with pytest.raises(ValueError, match="somme"):
        _table(rows=[("zemmour_eric", 10, 70, 10)])


def test_selection_has_no_future_leak():
    day = dt.timedelta(days=1)
    t_pub = _table("pub", "a", end=ASOF - 3 * day, published=ASOF)  # publié le jour même : retenu
    t_late = _table("late", "b", end=ASOF - 3 * day, published=ASOF + day)  # publié après
    t_delay = _table("delay", "c", end=ASOF - day)  # terrain + 2 jours = lendemain : exclu
    t_ok = _table("ok", "d", end=ASOF - 2 * day)  # terrain + 2 jours = asof : retenu
    t_old = _table("old", "e", end=ASOF - (CURRENT_WINDOW + 1) * day)
    t_other = _table("other", "f", pair=("macron_emmanuel", "zemmour_eric"),
                     rows=[("pecresse_valerie", 50, 20, 30)])
    got = current_tables(PAIR, 2022, ASOF, [t_pub, t_late, t_delay, t_ok, t_old, t_other])
    assert [t.id for t in got] == ["pub", "ok"]
    # même paire dans l'autre ordre : mêmes tableaux ; autre élection : aucun
    assert [t.id for t in current_tables(PAIR[::-1], 2022, ASOF, [t_pub, t_ok])] == ["pub", "ok"]
    assert current_tables(PAIR, 2017, ASOF, [t_pub, t_ok]) == []


def test_latest_table_per_pollster_and_equal_weights():
    day = dt.timedelta(days=1)
    old = _table("old", "ifop", end=ASOF - 20 * day, rows=[("zemmour_eric", 30, 50, 20)])
    new = _table("new", "ifop", end=ASOF - 5 * day, rows=[("zemmour_eric", 10, 70, 20)])
    other = _table("x", "elabe", end=ASOF - 10 * day, rows=[("zemmour_eric", 20, 60, 20)])
    chosen = current_tables(PAIR, 2022, ASOF, [old, new, other])
    assert [t.id for t in chosen] == ["x", "new"]
    reg = default_registry()
    out = current_group_means(PAIR, chosen, lambda c: reg.bloc_of(c, 2022), lambda c: None)
    assert list(out) == ["bloc:far_right"]
    assert out["bloc:far_right"][0] == pytest.approx((0.15, 0.65, 0.20))
    assert out["bloc:far_right"][1] == ("x", "new")
    # orientation : vecteur (vers pair[0], vers pair[1], ni l'un ni l'autre)
    sw = current_group_means(PAIR[::-1], chosen, lambda c: reg.bloc_of(c, 2022), lambda c: None)
    assert sw["bloc:far_right"][0] == pytest.approx((0.65, 0.15, 0.20))


def test_bloc_members_weighted_by_polls_and_hypotheses_averaged():
    reg = default_registry()
    t = _table(hypotheses=[
        [("jadot_yannick", 60, 10, 30), ("hidalgo_anne", 40, 10, 50)],
        [("jadot_yannick", 80, 0, 20)],
    ])
    weights = {"jadot_yannick": 3.0, "hidalgo_anne": 1.0}
    out = current_group_means(PAIR, [t], lambda c: reg.bloc_of(c, 2022), weights.get)
    h0 = (0.75 * 0.60 + 0.25 * 0.40, 0.10, 0.75 * 0.30 + 0.25 * 0.50)
    expected = tuple((a + b) / 2 for a, b in zip(h0, (0.80, 0.0, 0.20)))
    assert out["bloc:left"][0] == pytest.approx(expected)
    # un membre sans sondage : poids égaux dans l'hypothèse
    out = current_group_means(PAIR, [t], lambda c: reg.bloc_of(c, 2022), {"jadot_yannick": 3.0}.get)
    assert out["bloc:left"][0][0] == pytest.approx((0.50 + 0.80) / 2)


def _fits(tables, **kw):
    reg = default_registry()
    return [
        SecondRoundModel(reg, errors=ERR, use_polls=False, current_transfers=x, **kw)
        .fit(2022, ASOF, [PAIR]).pair_fit(PAIR)
        for x in (None, tables)
    ]


def test_table_moves_only_the_covered_group():
    base, fit = _fits([_table(end=ASOF - dt.timedelta(days=5))])
    changed = [g for g in base.groups if base.groups[g] != fit.groups[g]]
    assert changed == ["bloc:far_right"]
    g = fit.groups["bloc:far_right"]
    assert g.source == "current" and g.kappa == KAPPA_CURRENT and g.shift
    assert g.mean == pytest.approx((0.10, 0.70, 0.20))
    assert (fit.sigma_u, fit.sigma_v, fit.inedit) == (base.sigma_u, base.sigma_v, base.inedit)
    _, fit60 = _fits([_table(end=ASOF - dt.timedelta(days=5))], current_kappa=60.0)
    assert fit60.groups["bloc:far_right"].kappa == 60.0


def test_disabled_or_unavailable_gives_identical_draws():
    """Option désactivée (défaut) ou tableau publié après la date : mêmes priors, mêmes tirages."""
    future = _table(end=ASOF, published=ASOF + dt.timedelta(days=1))
    base, fit = _fits([future])
    assert fit.groups == base.groups and fit.notes == base.notes
    reg = default_registry()
    first = {"macron_emmanuel": np.full(500, 0.28), "le_pen_marine": np.full(500, 0.23),
             "zemmour_eric": np.full(500, 0.12), "melenchon_jean_luc": np.full(500, 0.20),
             "pecresse_valerie": np.full(500, 0.17)}
    draws = [
        SecondRoundModel(reg, errors=ERR, use_polls=False, current_transfers=x)
        .fit(2022, ASOF, [PAIR]).sample(PAIR, first, 30, np.random.default_rng(3), n=500)
        for x in (None, (), [future])
    ]
    for d in draws[1:]:
        assert d.share.tobytes() == draws[0].share.tobytes()
        assert d.transfers.tobytes() == draws[0].transfers.tobytes()
