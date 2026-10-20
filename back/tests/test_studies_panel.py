"""Panel mensuel de l'étude économie (vvs.studies.panel) : agrégation des sondages par mois,
variations entre mois consécutifs seulement, passage des séries macro au mensuel."""

from datetime import date

import polars as pl
import pytest

from vvs.studies import panel as P


def _by_bloc(rows):
    return pl.DataFrame(
        rows,
        schema={"poll_id": pl.Utf8, "election": pl.Int32, "round": pl.Int8,
                "hypothesis_id": pl.Utf8, "bloc": pl.Utf8, "share": pl.Float64,
                "suspect": pl.Boolean},
        orient="row",
    )


def _polls(rows):
    return pl.DataFrame(
        rows,
        schema={"poll_id": pl.Utf8, "pollster": pl.Utf8, "fieldwork_end": pl.Date,
                "is_rolling": pl.Boolean},
        orient="row",
    )


def test_poll_shares_average_hypotheses_and_apply_exclusions():
    by_bloc = _by_bloc([
        ("p1", 2022, 1, "a+b", "far_right", 30.0, False),
        ("p1", 2022, 1, "a+c", "far_right", 34.0, False),
        ("p1", 2022, 1, "a+c", "centre", 25.0, False),  # centre absent de a+b : pas un zéro
        ("p1", 2022, 1, "x+y", "far_right", 90.0, True),  # suspecte
        ("p1", 2022, 2, "a+b", "far_right", 45.0, False),  # second tour
        ("p1", 2022, 1, "a+c", "other", 3.0, False),  # bloc hors étude
        ("p2", 2027, 1, "le_pen+macron_emmanuel", "far_right", 20.0, False),
        ("p2", 2027, 1, "bardella+philippe", "far_right", 35.0, False),
    ])
    polls = _polls([("p1", "ifop", date(2022, 3, 2), False),
                    ("p2", "elabe", date(2023, 4, 30), False)])
    out = P.poll_bloc_shares(by_bloc, polls)
    get = {(r["poll_id"], r["bloc"]): r for r in out.iter_rows(named=True)}
    assert get[("p1", "far_right")]["share"] == pytest.approx(32.0)
    assert get[("p1", "far_right")]["n_hypotheses"] == 2
    assert get[("p1", "centre")]["share"] == pytest.approx(25.0)
    assert get[("p1", "centre")]["share"] == get[("p1", P.INCUMBENT)]["share"]  # 2022 : centre
    assert get[("p2", "far_right")]["share"] == pytest.approx(35.0)  # Macron 2027 écarté
    assert ("p1", "other") not in get
    assert get[("p2", "far_right")]["month"] == date(2023, 4, 1)


def test_month_weights_each_pollster_once():
    ps = pl.DataFrame({
        "poll_id": ["r1", "r2", "r3", "o1"], "election": [2022] * 4, "bloc": ["right"] * 4,
        "share": [10.0, 12.0, 14.0, 20.0], "pollster": ["ifop"] * 3 + ["elabe"],
        "month": [date(2022, 3, 1)] * 4,
    })
    m = P.monthly_bloc_shares(ps).row(0, named=True)
    assert m["share"] == pytest.approx((12.0 + 20.0) / 2)
    assert (m["n_polls"], m["n_pollsters"]) == (4, 2)


def test_panel_changes_only_between_consecutive_observed_months():
    monthly = pl.DataFrame({
        "election": [2012] * 4, "bloc": ["left"] * 4,
        "month": [date(2011, 10, 1), date(2011, 11, 1), date(2012, 1, 1), date(2012, 4, 1)],
        "share": [30.0, 32.0, 35.0, 31.0], "n_polls": [1, 2, 1, 5], "n_pollsters": [1, 2, 1, 3],
    })
    pan = P.bloc_panel(monthly, {2012: date(2012, 4, 22)}).filter(pl.col("bloc") == "left")
    assert pan["month"].to_list() == P._months(date(2011, 10, 1), date(2012, 4, 1))
    d = dict(zip(pan["month"], pan["d_share"]))
    assert d[date(2011, 11, 1)] == pytest.approx(2.0)
    assert d[date(2012, 1, 1)] is None  # décembre sans sondage : pas de variation
    assert d[date(2012, 4, 1)] is None
    assert pan.filter(pl.col("month") == date(2012, 4, 1))["months_to_election"].item() == 0
    assert pan.filter(pl.col("month") == date(2011, 12, 1))["n_polls"].item() == 0
    assert set(P.bloc_panel(monthly, {2012: date(2012, 4, 22)})["bloc"]) == set(P.BLOCS)


def test_monthly_macro_interpolates_quarters_and_averages_weeks():
    macro = pl.DataFrame({
        "series_id": ["u"] * 3 + ["f"] * 3 + ["m"],
        "date": [date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 1),
                 date(2020, 1, 6), date(2020, 1, 13), date(2020, 2, 3), date(2020, 5, 1)],
        "value": [8.0, 9.5, 8.0, 1.0, 2.0, 4.0, 7.0],
        "freq": ["Q"] * 3 + ["W"] * 3 + ["M"],
    })
    w = P.monthly_macro(macro)
    u = dict(zip(w["month"], w["u"]))
    assert u[date(2020, 1, 1)] is None  # avant le premier mois central
    assert u[date(2020, 2, 1)] == pytest.approx(8.0)
    assert u[date(2020, 3, 1)] == pytest.approx(8.5)
    assert u[date(2020, 5, 1)] == pytest.approx(9.5)
    assert u[date(2020, 8, 1)] == pytest.approx(8.0)
    f = dict(zip(w["month"], w["f"]))
    assert f[date(2020, 1, 1)] == pytest.approx(1.5) and f[date(2020, 2, 1)] == 4.0
    assert w["month"].to_list() == P._months(date(2020, 1, 1), date(2020, 8, 1))


def test_indicator_transforms_and_breaks():
    months = P._months(date(2023, 1, 1), date(2025, 2, 1))
    n = len(months)
    wide = pl.DataFrame({
        "month": months,
        "cpi_all": [100.0 * 1.01 ** i for i in range(n)],
        "jobseekers_cat_a": [3000.0] * 24 + [3300.0, 3300.0],
        "approval_president": [40.0] * n,
    })
    ind = P.indicators(wide, {"approval_president": {date(2024, 5, 1)}})
    row = ind.filter(pl.col("month") == date(2024, 3, 1)).row(0, named=True)
    assert row["cpi_all"] == pytest.approx(100 * (1.01 ** 12 - 1))
    assert row["d_cpi_all"] == pytest.approx(0.0, abs=1e-9)
    after = ind.filter(pl.col("month") >= P.JOBSEEKERS_BREAK)
    assert after["d_jobseekers_cat_a"].null_count() == after.height == 2
    assert after["jobseekers_cat_a"].null_count() == 2
    assert ind.filter(pl.col("month") == date(2024, 12, 1))["jobseekers_cat_a"].item() == 3.0
    assert ind.filter(pl.col("month") == date(2024, 5, 1))["d_approval_president"].item() is None
    assert ind.filter(pl.col("month") == date(2024, 6, 1))["d_approval_president"].item() == 0.0


def test_lags_require_complete_calendar():
    ind = pl.DataFrame({"month": P._months(date(2020, 1, 1), date(2020, 4, 1)),
                        "d_x": [1.0, 2.0, 3.0, 4.0]})
    lagged = P.with_lags(ind, ["d_x"], [0, 2])
    assert lagged["d_x_l2"].to_list() == [None, None, 1.0, 2.0]
    assert lagged["d_x_l0"].to_list() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="incomplet"):
        P.with_lags(ind.filter(pl.col("month") != date(2020, 2, 1)), ["d_x"], [1])


def test_officeholder_changes_cut_start_and_next_month():
    from vvs.ingest.macro.registry import Officeholder

    h = Officeholder("valls_manuel", "Manuel Valls", "prime_minister", date(2014, 3, 31), None)
    assert P.officeholder_changes([h]) == {
        "approval_prime_minister": {date(2014, 3, 1), date(2014, 4, 1)}
    }
