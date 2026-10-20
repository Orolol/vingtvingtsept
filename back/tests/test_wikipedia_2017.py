from datetime import date

import pytest
from wikipedia_checks import check_invariants, check_network, counts_by_round, parse_fixture, shares


@pytest.fixture(scope="module")
def parsed():
    return parse_fixture(2017)


def test_counts(parsed):
    polls, results, report = parsed
    assert polls.height == 188
    assert counts_by_round(results) == {1: (154, 350, 3464), 2: (120, 280, 680)}
    assert report.skipped_rows == []
    assert len(report.skipped_tables) == 10  # reports de voix + sous-échantillons
    assert [s.row for s in report.dropped_values] == [13]  # abstention « 20-24 % »


def test_invariants(parsed):
    polls, results, _ = parsed
    check_invariants(2017, polls, results)


def test_bva_last_first_round(parsed):
    polls, results, _ = parsed
    p = polls.filter(polls["poll_id"] == "2017-bva-20170421-1").row(0, named=True)
    assert (p["fieldwork_start"], p["sample_size"]) == (date(2017, 4, 20), 1494)
    hid = (
        "arthaud_nathalie+asselineau_francois+cheminade_jacques+dupont_aignan_nicolas+"
        "fillon_francois+hamon_benoit+lassalle_jean+le_pen_marine+macron_emmanuel+"
        "melenchon_jean_luc+poutou_philippe"
    )
    s = shares(results, "2017-bva-20170421-1", 1, hid)
    assert s["macron_emmanuel"] == 23 and s["le_pen_marine"] == 23
    assert s["melenchon_jean_luc"] == 19.5 and s["fillon_francois"] == 19
    assert s["cheminade_jacques"] == 0.25  # « <0,5 % »
    assert s["abstention"] == 22 and s["undecided"] == 29


def test_month_from_caption_and_others_column(parsed):
    # tableau « Mars 2017 », date « du 1 er au 4 », colonne « Autres candidats »
    polls, results, _ = parsed
    p = polls.filter(polls["poll_id"] == "2017-ifop-20170304-1").row(0, named=True)
    assert (p["fieldwork_start"], p["sponsor"]) == (date(2017, 3, 1), "Fiducial")
    sub = results.filter(results["poll_id"] == "2017-ifop-20170304-1")
    s = dict(zip(sub["candidate_id"], sub["share"], strict=True))
    assert s["alliot_marie_michele"] == 1 and s["yade_rama"] == 1
    assert s["marchandise_charlotte"] == 0.5


def test_second_round_between_rounds(parsed):
    _, results, _ = parsed
    s = shares(results, "2017-ifop-20170505-1", 2, "le_pen_marine+macron_emmanuel")
    assert s == {"abstention": 24.5, "macron_emmanuel": 63, "le_pen_marine": 37}


@pytest.mark.slow
def test_network():
    check_network(2017)
