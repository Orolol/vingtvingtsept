"""Abstention différentielle : lecture « deux bases » des notices et table turnout_gap.

Deux pages de notices OpinionWay figées dans `fixtures/commission/` (texte pdfplumber tel
quel) : PrésiTrack du 7 février 2017 (neuf colonnes, ligature « ti » perdue) et baromètre
du 31 mars 2022 (onze colonnes). Les valeurs attendues ont été relues sur les PDF.
"""

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from vvs.curate import turnout_gap as tg
from vvs.ingest.commission import archive as ar
from vvs.ingest.commission import notice as nt
from vvs.ingest.commission.aliases import archive_candidates, candidate_id

FIXTURES = Path(__file__).parent / "fixtures" / "commission"
LABELS = {"fillon_francois": ["François Fillon"], "hamon_benoit": ["Benoît Hamon"]}


def _hypotheses(name: str, election: int) -> dict[str, dict[str, float]]:
    page = ar.strip_evolutions((FIXTURES / name).read_text())
    with archive_candidates(LABELS):
        found = ar.classify_blocks(nt.find_blocks([page]), "opinionway", "certain_voters", election)
    assert all(h.round == 1 for h in found.hypotheses)
    return {h.population: {c: s for c, s, _ in h.results} for h in found.hypotheses}


# ------------------------------------------------------------- tableaux à deux bases


def test_dual_2017_nine_columns():
    hyps = _hypotheses("archive_ow_2017_p9.txt", 2017)
    # publié = certains, redressement « SD & T1 & T2 » (9e colonne) ; inscrits même rang (5e)
    assert hyps["certain_voters"]["le_pen_marine"] == 25
    assert hyps["registered"]["le_pen_marine"] == 22
    assert hyps["certain_voters"]["macron_emmanuel"] == 22
    assert hyps["registered"]["macron_emmanuel"] == 24
    assert hyps["certain_voters"]["fillon_francois"] == 20
    assert hyps["registered"]["fillon_francois"] == 19
    # Arthaud : « 0% - - - 1% - - - - » : absente des certains, pas inventée
    assert "arthaud_nathalie" not in hyps["certain_voters"]


def test_dual_2022_eleven_columns():
    hyps = _hypotheses("archive_ow_2022_p9.txt", 2022)
    # publié = certains « SD & PresidT1 » (9e colonne), inscrits même redressement (4e)
    assert hyps["certain_voters"]["le_pen_marine"] == 20
    assert hyps["registered"]["le_pen_marine"] == 21
    assert hyps["certain_voters"]["zemmour_eric"] == 10
    assert hyps["registered"]["zemmour_eric"] == 9
    assert set(hyps["certain_voters"]) == set(hyps["registered"])


def test_same_hypothesis_kept_for_each_population():
    """Le parseur 2027 dédoublonne sur (tour, candidats) : la seconde base disparaissait."""
    page = (FIXTURES / "archive_ow_2022_p9.txt").read_text()
    found = ar.classify_blocks(nt.find_blocks([page]), "opinionway", "certain_voters", 2022)
    assert sorted(h.population for h in found.hypotheses) == ["certain_voters", "registered"]
    assert len({h.label for h in found.hypotheses}) == 1


# ------------------------------------------------------------------- texte, méta


def test_lost_ligatures_repaired_only_when_frequent():
    text = "le suivi de l’élec,on Présiden,elle ; inten>ons ; échanCllon ; No#ce " * 3
    glyphs = ar.lost_ligature_glyphs(text)
    assert {",", ">", "C"} <= glyphs
    assert "élection Présidentielle" in ar.fix_ligatures(text, glyphs)
    assert ar.lost_ligature_glyphs("Paris, Lyon,Marseille et 1,5 point.") == set()


@pytest.mark.parametrize(
    "text, window, deposit_year, expected",
    [
        (
            "Les interviews ont été réalisées les 31 mars, 1er avril et 2 avril 2017.",
            (dt.date(2017, 1, 1), dt.date(2017, 4, 30)),
            None,
            (dt.date(2017, 3, 31), dt.date(2017, 4, 2), False),
        ),
        (
            "échantillons,interrogésdu 30janvier au 2février 2022.",
            (dt.date(2021, 10, 1), dt.date(2022, 2, 28)),
            2022,
            (dt.date(2022, 1, 30), dt.date(2022, 2, 2), False),
        ),
        (  # coquille d'année dans une notice déposée en janvier 2022
            "Lesinterviewsontétéréaliséesdu02au03janvier2021.",
            (dt.date(2021, 9, 1), dt.date(2022, 1, 31)),
            2022,
            (dt.date(2022, 1, 2), dt.date(2022, 1, 3), True),
        ),
        ("Le sondage du 5 au 7 février 2017.", None, None, None),  # pas d'indice de terrain
    ],
)
def test_find_fieldwork_loose(text, window, deposit_year, expected):
    assert ar.find_fieldwork_loose(text, window, deposit_year) == expected


def test_strip_evolutions():
    page = "Emmanuel Macron 27,5% +/-2,9 points -1\nJean-Luc Mélenchon 12\n+2\nFabien Roussel 3 ="
    assert ar.strip_evolutions(page).splitlines() == [
        "Emmanuel Macron 27,5% +/-2,9 points",
        "Jean-Luc Mélenchon 12",
        "Fabien Roussel 3",
    ]


def test_population_reference_overrides_base_line():
    ifop = [
        "Base : Ensemble des personnes inscrites",
        (
            "1 Sur toutes les personnes inscrites sur les"
            " listes électorales et certaines d’aller voter interrogées, 1 450 personnes"
        ),
    ]
    assert ar.notice_population(ifop, "registered") == "certain_voters"
    assert (
        ar.notice_population(["Base : Ensemble des personnes inscrites"], "registered")
        == "registered"
    )
    block = nt.Block([], ["Base : Ensemble des personnes inscrites sur les listes électorales"], 1)
    assert ar.block_population(block, "certain_voters") == "certain_voters"
    assert ar.block_population(block, "registered") == "registered"


@pytest.mark.parametrize(
    "pollster, page, expected",
    [
        (  # OpinionWay 2022 : les certains en premier dans la mise en page
            "opinionway",
            (
                "Ensemble des personnes inscrites sur les listes\nCertains d’aller voter\nélectorales\n"
                "(Notes 9 et 10 : 761 personnes)\n(1059 personnes)"
            ),
            [(1, 100 * 761 / 1059, "certain")],
        ),
        (
            "harris",
            (
                "CERTITUDE D’ALLER VOTER AU PREMIER TOUR\nTout à fait certain(e) d'aller voter 69 70\n"
                "CERTITUDE D’ALLER VOTER AU SECOND TOUR\nTout à fait certain(e) d'aller voter 71 73"
            ),
            [(1, 70.0, "certain"), (2, 73.0, "certain")],
        ),
        (
            "ifop",
            (
                "Question : [Aux personnes inscrites] Vous personnellement,\nenvisagez-vous d’aller voter ?\n"
                "Oui 95,1 95,4\nNon 4,9 4,6\nEt diriez-vous que… ?\nC’est tout à fait certain 74,8 74,5"
            ),
            [(1, 95.4 * 74.5 / 100, "certain")],
        ),
        (
            "kantar",
            "33% des répondants ne sont pas certains d’aller voter ou n’ont pas exprimé de choix",
            [(1, 67.0, "certain_expressed")],
        ),
    ],
)
def test_find_turnout_intent(pollster, page, expected):
    got = [(t.round, t.certain_share, t.kind) for t in ar.find_turnout_intent([page], pollster)]
    assert [(r, k) for r, _, k in got] == [(r, k) for r, _, k in expected]
    for (_, a, _), (_, b, _) in zip(got, expected, strict=True):
        assert a == pytest.approx(b)


def test_archive_candidates_inactive_by_default():
    assert candidate_id("François Fillon") is None  # le parseur 2027 ne voit pas 2017
    with archive_candidates(LABELS):
        assert candidate_id("François Fillon") == "fillon_francois"
        assert candidate_id("fillon_francois") == "fillon_francois"
    assert candidate_id("François Fillon") is None
    assert candidate_id("Yannick Jadot, soutenu par Europe Ecologie-Les Verts") == "jadot_yannick"


# -------------------------------------------------------------------- turnout_gap


def _frames():
    polls = pl.DataFrame(
        {
            "poll_id": ["p1", "p2"],
            "election": [2022, 2022],
            "pollster": ["opinionway", "ifop"],
            "fieldwork_start": [dt.date(2022, 3, 1)] * 2,
            "fieldwork_end": [dt.date(2022, 3, 3)] * 2,
            "is_rolling": [True, True],
        }
    )
    rows = [
        # p1 : deux bases ; Arthaud seulement chez les inscrits
        ("p1", 1, "h", "registered", "melenchon_jean_luc", 15.0),
        ("p1", 1, "h", "certain_voters", "melenchon_jean_luc", 14.0),
        ("p1", 1, "h", "registered", "pecresse_valerie", 9.0),
        ("p1", 1, "h", "certain_voters", "pecresse_valerie", 10.5),
        ("p1", 1, "h", "registered", "arthaud_nathalie", 1.0),
        ("p1", 2, "d", "registered", "macron_emmanuel", 55.0),
        ("p1", 2, "d", "certain_voters", "macron_emmanuel", 56.0),
        # p2 : une seule base
        ("p2", 1, "h", "certain_voters", "melenchon_jean_luc", 13.0),
    ]
    results = pl.DataFrame(
        rows,
        schema=["poll_id", "round", "hypothesis_id", "population", "candidate_id", "share"],
        orient="row",
    ).with_columns(pl.lit(None, pl.Float64).alias("raw_share"), pl.lit(False).alias("suspect"))
    intent = pl.DataFrame(
        {
            "poll_id": ["p1"],
            "round": [1],
            "certain_share": [75.0],
            "kind": ["certain"],
            "n_registered": [1000],
            "n_certain": [750],
        }
    ).with_columns(pl.col("round").cast(pl.Int8))
    return polls, results, intent


def test_turnout_gap_table():
    polls, results, intent = _frames()
    table = tg.build_table([(polls, results)], intent, lambda c, e: "x")
    assert table.columns == list(tg.SCHEMA)
    got = {(r["round"], r["candidate_id"]): r for r in table.iter_rows(named=True)}
    assert set(got) == {(1, "melenchon_jean_luc"), (1, "pecresse_valerie"), (2, "macron_emmanuel")}
    assert got[(1, "melenchon_jean_luc")]["gap"] == pytest.approx(-1.0)
    assert got[(1, "pecresse_valerie")]["gap"] == pytest.approx(1.5)
    # part de certains : même tour, sinon premier tour
    assert got[(2, "macron_emmanuel")]["certain_share_sample"] == 75.0
    assert got[(1, "melenchon_jean_luc")]["n_certain"] == 750
