from datetime import date

import pytest
from wikipedia_checks import check_invariants, check_network, counts_by_round, parse_fixture, shares

FULL_2007 = (
    "bayrou_francois+besancenot_olivier+bove_jose+buffet_marie_george+de_villiers_philippe+"
    "laguiller_arlette+le_pen_jean_marie+nihous_frederic+royal_segolene+sarkozy_nicolas+"
    "schivardi_gerard+voynet_dominique"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_fixture(2007)


def test_counts(parsed):
    polls, results, report = parsed
    assert polls.height == 131
    assert counts_by_round(results) == {1: (105, 107, 1171), 2: (121, 124, 248)}
    reasons = sorted(s.reason.split(" :")[0] for s in report.skipped_rows)
    assert reasons == [
        "hypothèse en double aux parts différentes",  # CSA 21 février, deux lignes
        "somme des intentions hors [95, 105]",  # Sofres 24 mars, Sarkozy « 38 % »
    ]
    assert report.skipped_tables == []


def test_invariants(parsed):
    polls, results, _ = parsed
    check_invariants(2007, polls, results)


def test_last_ipsos_before_first_round(parsed):
    polls, results, _ = parsed
    p = polls.filter(polls["poll_id"] == "2007-ipsos-20070420-1").row(0, named=True)
    # la page ne donne qu'un jour : début = fin
    assert (p["fieldwork_start"], p["fieldwork_end"]) == (date(2007, 4, 20), date(2007, 4, 20))
    s = shares(results, p["poll_id"], 1, FULL_2007)
    assert s["sarkozy_nicolas"] == 30 and s["royal_segolene"] == 23
    assert s["bayrou_francois"] == 18 and s["le_pen_jean_marie"] == 13
    assert s["schivardi_gerard"] == 0.5
    assert shares(results, p["poll_id"], 2, "royal_segolene+sarkozy_nicolas") == {
        "sarkozy_nicolas": 53.5,
        "royal_segolene": 46.5,
    }


def test_one_poll_several_hypotheses(parsed):
    # Ifop 20-21 janvier 2005 (Paris Match) : trois candidats PS testés au premier tour,
    # sur trois lignes sans fusion de cellules, et trois duels contre Sarkozy
    _, results, _ = parsed
    sub = results.filter(results["poll_id"] == "2007-ifop-20050121-1")
    r1 = sub.filter(sub["round"] == 1)
    assert r1["hypothesis_id"].n_unique() == 3
    ps = r1.filter(r1["candidate_id"].is_in(["fabius_laurent", "lang_jack", "hollande_francois"]))
    assert dict(zip(ps["candidate_id"], ps["share"], strict=True)) == {
        "fabius_laurent": 18,
        "lang_jack": 19,
        "hollande_francois": 25,
    }
    assert shares(results, "2007-ifop-20050121-1", 2, "lang_jack+sarkozy_nicolas") == {
        "sarkozy_nicolas": 54,
        "lang_jack": 46,
    }


def test_less_than_and_other_duel(parsed):
    _, results, _ = parsed
    sub = results.filter((results["poll_id"] == "2007-sofres-20070118-1") & (results["round"] == 1))
    s = dict(zip(sub["candidate_id"], sub["share"], strict=True))
    assert s["nihous_frederic"] == 0.25  # « <0,5 % »
    assert s["sarkozy_nicolas"] == 35 and s["royal_segolene"] == 31
    # duel Bayrou – Sarkozy, Ipsos 18-19 avril : sondage distinct du premier tour du 20
    assert shares(results, "2007-ipsos-20070419-1", 2, "bayrou_francois+sarkozy_nicolas") == {
        "bayrou_francois": 52,
        "sarkozy_nicolas": 48,
    }


@pytest.mark.slow
def test_network():
    check_network(2007)
