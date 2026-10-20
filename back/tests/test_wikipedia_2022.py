from datetime import date

import pytest
from wikipedia_checks import check_invariants, check_network, counts_by_round, parse_fixture, shares

FULL_2022 = (
    "arthaud_nathalie+dupont_aignan_nicolas+hidalgo_anne+jadot_yannick+lassalle_jean+"
    "le_pen_marine+macron_emmanuel+melenchon_jean_luc+pecresse_valerie+poutou_philippe+"
    "roussel_fabien+zemmour_eric"
)


@pytest.fixture(scope="module")
def parsed():
    return parse_fixture(2022)


def test_counts(parsed):
    polls, results, report = parsed
    assert polls.height == 302
    assert counts_by_round(results) == {1: (255, 447, 5958), 2: (253, 526, 1052)}
    reasons = [s.reason.split(" :")[0] for s in report.skipped_rows]
    assert reasons.count("candidature générique (cellule)") == 32
    assert reasons.count("candidat non nommé sous un en-tête générique") == 1
    assert len(reasons) == 33


def test_invariants(parsed):
    polls, results, _ = parsed
    check_invariants(2022, polls, results)


def test_last_ifop_before_first_round(parsed):
    polls, results, _ = parsed
    ifop = polls.filter(
        (polls["pollster"] == "ifop") & (polls["fieldwork_end"] < date(2022, 4, 10))
    ).sort("fieldwork_end")
    p = ifop.row(-1, named=True)
    assert p["poll_id"] == "2022-ifop-20220407-1"
    assert (p["fieldwork_start"], p["sample_size"]) == (date(2022, 4, 4), 3010)
    s = shares(results, p["poll_id"], 1, FULL_2022)
    assert s["macron_emmanuel"] == 26.5 and s["le_pen_marine"] == 24
    assert s["melenchon_jean_luc"] == 17.5 and s["zemmour_eric"] == 8.5
    assert s["pecresse_valerie"] == 9 and s["arthaud_nathalie"] == 0.5
    # même sondage : hypothèses de second tour dans d'autres tableaux
    assert shares(results, p["poll_id"], 2, "le_pen_marine+macron_emmanuel") == {
        "macron_emmanuel": 52,
        "le_pen_marine": 48,
    }
    assert shares(results, p["poll_id"], 2, "macron_emmanuel+zemmour_eric") == {
        "macron_emmanuel": 66.5,
        "zemmour_eric": 33.5,
    }


def test_named_candidate_in_generic_column(parsed):
    # « 14 % Bertrand » sous l'en-tête « Candidat LR », Harris 26-29 novembre 2021
    _, results, _ = parsed
    sub = results.filter(
        (results["poll_id"] == "2022-harris-20211129-1")
        & (results["round"] == 1)
        & results["hypothesis_id"].str.contains("bertrand_xavier")
    )
    s = dict(zip(sub["candidate_id"], sub["share"], strict=True))
    assert s["bertrand_xavier"] == 14 and s["zemmour_eric"] == 13
    assert s["lassalle_jean"] == 0.25  # « <0,5 % »


def test_second_round_between_rounds(parsed):
    _, results, _ = parsed
    s = shares(results, "2022-ifop-20220421-1", 2, "le_pen_marine+macron_emmanuel")
    assert s == {"macron_emmanuel": 55.5, "le_pen_marine": 44.5}


@pytest.mark.slow
def test_network():
    check_network(2022)
