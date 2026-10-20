from datetime import date

import pytest
from wikipedia_checks import check_invariants, check_network, counts_by_round, parse_fixture, shares


@pytest.fixture(scope="module")
def parsed():
    return parse_fixture(2027)


def test_counts(parsed):
    polls, results, report = parsed
    assert polls.height == 71
    assert counts_by_round(results) == {1: (71, 294, 3103), 2: (28, 95, 190)}
    reasons = sorted(s.reason.split(" :")[0] for s in report.skipped_rows)
    assert reasons == ["candidature générique (sondage)"] * 3 + ["libellé ambigu"] * 3
    assert [s.table for s in report.skipped_tables] == [6]  # vote blanc compté exprimé


def test_invariants(parsed):
    polls, results, _ = parsed
    check_invariants(2027, polls, results)


def test_split_header_cell(parsed):
    # YouGov 12-15 octobre 2026 : « 9 Hollande ( PS ) » dans la colonne Glucksmann
    polls, results, _ = parsed
    p = polls.filter(polls["poll_id"] == "2027-yougov-20261015-1").row(0, named=True)
    assert (p["fieldwork_start"], p["sample_size"]) == (date(2026, 10, 12), 1094)
    hid = (
        "arthaud_nathalie+dupont_aignan_nicolas+hollande_francois+le_pen_marine+"
        "melenchon_jean_luc+philippe_edouard+retailleau_bruno+roussel_fabien+"
        "tondelier_marine+zemmour_eric"
    )
    s = shares(results, p["poll_id"], 1, hid)
    assert s["hollande_francois"] == 9 and s["philippe_edouard"] == 15
    assert s["le_pen_marine"] == 37
    assert shares(results, p["poll_id"], 2, "melenchon_jean_luc+philippe_edouard") == {
        "melenchon_jean_luc": 34,
        "philippe_edouard": 66,
    }


def test_others_column_and_sponsor(parsed):
    polls, results, _ = parsed
    sub = results.filter(
        (results["poll_id"] == "2027-ifop-20260929-1")
        & results["hypothesis_id"].str.contains("lisnard_david")
    )
    s = dict(zip(sub["candidate_id"], sub["share"], strict=True))
    assert (s["lisnard_david"], s["ruffin_francois"], s["de_villepin_dominique"]) == (3.5, 3, 2)
    sponsors = dict(
        polls.filter(polls["sponsor"].is_not_null()).select("poll_id", "sponsor").iter_rows()
    )
    assert sponsors["2027-cluster17-20240405-1"] == "Picardie Debout"


def test_subsamples_merged(parsed):
    # Odoxa 19-20 novembre 2025 : duels posés à des sous-échantillons (689 à 803 sur 1 300)
    polls, results, _ = parsed
    odoxa = polls.filter(
        (polls["pollster"] == "odoxa") & (polls["fieldwork_end"] == date(2025, 11, 20))
    )
    assert odoxa["sample_size"].to_list() == [1300]
    r2 = results.filter((results["poll_id"] == odoxa["poll_id"][0]) & (results["round"] == 2))
    assert r2["hypothesis_id"].n_unique() == 4


@pytest.mark.slow
def test_network():
    check_network(2027)
