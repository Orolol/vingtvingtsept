from datetime import date

import pytest
from wikipedia_checks import check_invariants, check_network, counts_by_round, parse_fixture, shares

FULL_2002 = (
    "bayrou_francois+besancenot_olivier+boutin_christine+chevenement_jean_pierre+chirac_jacques+"
    "gluckstein_daniel+hue_robert+jospin_lionel+laguiller_arlette+le_pen_jean_marie+"
    "lepage_corinne+madelin_alain+mamere_noel+megret_bruno+saint_josse_jean+taubira_christiane"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_fixture(2002)


def test_counts(parsed):
    polls, results, report = parsed
    assert polls.height == 106
    assert counts_by_round(results) == {1: (98, 98, 1405), 2: (29, 29, 58)}
    assert report.skipped_rows == [] and report.skipped_tables == []
    assert report.counters["lignes de résultats"] == 2


def test_invariants(parsed):
    polls, results, _ = parsed
    check_invariants(2002, polls, results)


def test_last_ipsos_before_first_round(parsed):
    polls, results, _ = parsed
    p = polls.filter(polls["poll_id"] == "2002-ipsos-20020418-1").row(0, named=True)
    assert (p["fieldwork_start"], p["sample_size"]) == (date(2002, 4, 17), None)
    s = shares(results, p["poll_id"], 1, FULL_2002)
    assert s["chirac_jacques"] == 20 and s["jospin_lionel"] == 18
    assert s["le_pen_jean_marie"] == 14 and s["bayrou_francois"] == 6
    assert s["gluckstein_daniel"] == 0.5 and s["megret_bruno"] == 2.5


def test_first_and_second_round_same_poll(parsed):
    # Sofres 17-18 avril 2002 : premier tour (« Sofres ») et duel Chirac – Jospin
    # (« TNS Sofres », « 17- 18 avril ») dans deux tableaux, un seul sondage
    _, results, _ = parsed
    s = shares(results, "2002-sofres-20020418-1", 1, FULL_2002)
    assert s["chirac_jacques"] == 19.5 and s["le_pen_jean_marie"] == 12.5
    assert shares(results, "2002-sofres-20020418-1", 2, "chirac_jacques+jospin_lionel") == {
        "chirac_jacques": 50,
        "jospin_lionel": 50,
    }


def test_second_round_chirac_le_pen(parsed):
    _, results, _ = parsed
    s = shares(results, "2002-csa-20020503-1", 2, "chirac_jacques+le_pen_jean_marie")
    assert s == {"chirac_jacques": 80, "le_pen_jean_marie": 20}


def test_absent_candidate_and_year_from_heading(parsed):
    # Ipsos 29-30 mars 2002 : Gluckstein « - » (non testé, absent) ; Ipsos 6-7 mai sous
    # le titre « 1999 »
    polls, results, _ = parsed
    sub = results.filter(results["poll_id"] == "2002-ipsos-20020330-1")
    assert "gluckstein_daniel" not in sub["candidate_id"].to_list()
    assert dict(zip(sub["candidate_id"], sub["share"], strict=True))["pasqua_charles"] == 2
    p = polls.filter(polls["poll_id"] == "2002-ipsos-19990507-1").row(0, named=True)
    assert p["fieldwork_start"] == date(1999, 5, 6)


@pytest.mark.slow
def test_network():
    check_network(2002)
