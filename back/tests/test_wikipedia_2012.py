from datetime import date

import pytest
from wikipedia_checks import check_invariants, check_network, counts_by_round, parse_fixture, shares

FULL_2012 = (
    "arthaud_nathalie+bayrou_francois+cheminade_jacques+dupont_aignan_nicolas+hollande_francois+"
    "joly_eva+le_pen_marine+melenchon_jean_luc+poutou_philippe+sarkozy_nicolas"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_fixture(2012)


def test_counts(parsed):
    polls, results, report = parsed
    assert polls.height == 180
    assert counts_by_round(results) == {1: (115, 140, 1469), 2: (133, 184, 368)}
    reasons = [s.reason.split(" :")[0] for s in report.skipped_rows]
    assert reasons == ["somme des intentions hors [95, 105]"]  # Harris 16 décembre 2011
    # « Second trimestre 2011 » : colonne Chevènement corrompue
    assert [s.table for s in report.skipped_tables] == [6]


def test_invariants(parsed):
    polls, results, _ = parsed
    check_invariants(2012, polls, results)


def test_last_ipsos_before_first_round(parsed):
    polls, results, _ = parsed
    r1 = results.filter(results["round"] == 1)["poll_id"].unique().to_list()
    ipsos = polls.filter((polls["pollster"] == "ipsos") & polls["poll_id"].is_in(r1))
    p = ipsos.sort("fieldwork_end").row(-1, named=True)
    assert p["poll_id"] == "2012-ipsos-20120417-1"
    assert p["fieldwork_start"] == date(2012, 4, 17)  # un seul jour donné par la page
    s = shares(results, p["poll_id"], 1, FULL_2012)
    assert s["hollande_francois"] == 27 and s["sarkozy_nicolas"] == 27
    assert s["le_pen_marine"] == 15.5 and s["melenchon_jean_luc"] == 14.5
    assert s["bayrou_francois"] == 10 and s["cheminade_jacques"] == 0.5
    assert shares(results, p["poll_id"], 2, "hollande_francois+sarkozy_nicolas") == {
        "hollande_francois": 56,
        "sarkozy_nicolas": 44,
    }


def test_second_round_hollande_sarkozy(parsed):
    _, results, _ = parsed
    s = shares(results, "2012-bva-20120502-1", 2, "hollande_francois+sarkozy_nicolas")
    assert s == {"hollande_francois": 53.5, "sarkozy_nicolas": 46.5}


def test_named_candidate_in_generic_column(parsed):
    # Harris 5 septembre 2011 : « 24 % Aubry » sous « Candidat PS », « <1 % Poutou » sous
    # « Candidat NPA »
    _, results, _ = parsed
    sub = results.filter(
        (results["poll_id"] == "2012-harris-20110905-1")
        & results["hypothesis_id"].str.contains("aubry_martine")
    )
    s = dict(zip(sub["candidate_id"], sub["share"], strict=True))
    assert s["aubry_martine"] == 24 and s["poutou_philippe"] == 0.5
    assert "hollande_francois" not in s


def test_le_pen_father_and_daughter(parsed):
    # tableau « Années 2007-2009 », colonne « Candidat FN »
    _, results, _ = parsed
    jm = results.filter(results["poll_id"] == "2012-opinionway-20090618-1")
    assert set(jm.filter(jm["candidate_id"].str.starts_with("le_pen"))["candidate_id"]) == {
        "le_pen_jean_marie"
    }
    m = results.filter(results["poll_id"] == "2012-ifop-20091030-1")
    s = dict(zip(m["candidate_id"], m["share"], strict=True))
    assert s["le_pen_marine"] == 11 and s["aubry_martine"] == 20 and s["duflot_cecile"] == 5
    assert "le_pen_jean_marie" not in s


@pytest.mark.slow
def test_network():
    check_network(2012)
