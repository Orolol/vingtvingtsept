"""Questions d'anticipation : lecture des tableaux déclarés, contrôles, registre.

Pages figées dans tests/fixtures/expectations/ (extraites avec pdfseparate), une par forme de
tableau rencontrée :

- Kantar pour Le Figaro, février 2017, p. 29 : liste « Candidat valeur évolution » mêlée aux
  étiquettes d'un graphique (noms répétés, valeur placée avant le nom), « <0,5 » ;
- OpinionWay Présitrack, septembre 2021, p. 19 : deux questions côte à côte, colonnes brut et
  redressé, lignes « NSP » des deux questions ;
- rapport du rolling Ifop du 1er avril 2022, p. 17 : prénom, valeur, NOM, évolution intercalée ;
- Ipsos pour France Info, 19 mars 2022, p. 19 : « qualifié et élu / qualifié mais battu / pas
  qualifié » par candidat ;
- OpinionWay Présitrack, juillet 2026, p. 21 : un bloc par candidat, une ligne par qualité ;
- Ifop pour le JDD, décembre 2016, p. 7 : « pensez-vous que François Fillon va remporter… »
  (oui certainement / oui probablement…).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vvs.ingest import expectations as ex
from vvs.ingest.commission.certainty import candidate_lookup
from vvs.ingest.commission.notice import extract_text

FIXTURES = Path(__file__).parent / "fixtures" / "expectations"

LOOKUP = candidate_lookup({
    "macron_emmanuel": ["Emmanuel Macron"],
    "le_pen_marine": ["Marine Le Pen"],
    "fillon_francois": ["François Fillon"],
    "hamon_benoit": ["Benoît Hamon"],
    "melenchon_jean_luc": ["Jean-Luc Mélenchon"],
    "poutou_philippe": ["Philippe Poutou"],
    "dupont_aignan_nicolas": ["Nicolas Dupont-Aignan"],
    "jadot_yannick": ["Yannick Jadot"],
    "arthaud_nathalie": ["Nathalie Arthaud"],
    "cheminade_jacques": ["Jacques Cheminade"],
    "bertrand_xavier": ["Xavier Bertrand"],
    "zemmour_eric": ["Éric Zemmour"],
    "hidalgo_anne": ["Anne Hidalgo"],
    "pecresse_valerie": ["Valérie Pécresse"],
    "barnier_michel": ["Michel Barnier"],
    "montebourg_arnaud": ["Arnaud Montebourg"],
    "roussel_fabien": ["Fabien Roussel"],
})


def _page(name: str) -> str:
    return extract_text(FIXTURES / name, column_pages=0).pages[0]


def _src(**kw) -> ex.Source:
    base = {"election": 2022, "kind": "commission", "question_type": "winner", "layout": "list",
                "population": "all", "verified": True}
    return ex.Source(**(base | kw))


def test_kantar_list_ignores_chart_labels():
    vals, undecided = ex.read_table(_page("kantar_288_p29.pdf"), _src(n_candidates=10), LOOKUP)
    assert vals["win"] == {
        "macron_emmanuel": 26.0, "le_pen_marine": 17.0, "fillon_francois": 16.0,
        "hamon_benoit": 6.0, "melenchon_jean_luc": 3.0, "poutou_philippe": 1.0,
        "dupont_aignan_nicolas": 0.25, "jadot_yannick": 0.25, "arthaud_nathalie": 0.0,
        "cheminade_jacques": 0.0,
    }
    # par différence : « Sans réponse 29 » plus l'écart d'arrondi des parts publiées
    assert undecided == pytest.approx(30.5)


def test_opinionway_side_by_side_redressed_column():
    s = _src(anchor="qui sera elu", measures=[ex.Measure("win", (1,))], n_candidates=13)
    vals, undecided = ex.read_table(_page("opinionway_1143_p19.pdf"), s, LOOKUP)
    assert vals["win"]["macron_emmanuel"] == 35.0
    assert vals["win"]["bertrand_xavier"] == 11.0
    assert vals["win"]["barnier_michel"] == 1.0
    # par différence : « Aucun » 12 et « NSP » 3 de la question, pas le « NSP » voisin
    assert undecided == 15.0


def test_ifop_rolling_split_names():
    s = _src(kind="ifop_report", layout="split_name", population="registered", n_candidates=7)
    vals, undecided = ex.read_table(_page("ifop_rol22_0401_p17.pdf"), s, LOOKUP)
    assert vals["win"] == {
        "melenchon_jean_luc": 8.0, "hidalgo_anne": 1.0, "jadot_yannick": 0.0,
        "macron_emmanuel": 47.0, "pecresse_valerie": 2.0, "le_pen_marine": 17.0,
        "zemmour_eric": 3.0,
    }
    assert undecided == 22.0  # aucun 2 + ne se prononce pas 20


def test_ipsos_qualification_and_win():
    s = _src(question_type="chances",
             measures=[ex.Measure("win", (0,)), ex.Measure("qualify", (0, 1))],
             row_total=(0, 1, 2), n_candidates=6)
    vals, undecided = ex.read_table(_page("ipsos_1418_p19.pdf"), s, LOOKUP)
    assert undecided is None
    assert vals["win"]["macron_emmanuel"] == 72.0
    assert vals["qualify"]["macron_emmanuel"] == 93.0
    assert vals["qualify"]["le_pen_marine"] == 76.0
    assert vals["win"]["jadot_yannick"] == 2.0


def test_row_total_failure_and_explicit_drop():
    s = _src(question_type="chances", measures=[ex.Measure("win", (0,))], row_total=(0, 1, 3))
    with pytest.raises(ex.TableError, match="ne somment pas"):
        ex.read_table(_page("ipsos_1418_p19.pdf"), s, LOOKUP)
    s = _src(question_type="chances", n_candidates=6, drop=("jadot_yannick",))
    with pytest.raises(ex.TableError, match="5 candidats lus, 6 attendus"):
        ex.read_table(_page("ipsos_1418_p19.pdf"), s, LOOKUP)


def test_opinionway_attribute_blocks():
    lookup = candidate_lookup({"le_pen_marine": ["Marine Le Pen"], "zemmour_eric": ["Éric Zemmour"]})
    s = _src(election=2027, question_type="chances", layout="attribute", item="peut remporter",
             measures=[ex.Measure("win", (1,))], n_candidates=2)
    vals, _ = ex.read_table(_page("opinionway_2229_p21.pdf"), s, lookup)
    assert vals["win"] == {"le_pen_marine": 54.0, "zemmour_eric": 7.0}
    s.item, s.question_type = "ferait (un bon president|une bonne presidente)", "good_president"
    vals, _ = ex.read_table(_page("opinionway_2229_p21.pdf"), s, lookup)
    assert vals["win"] == {"le_pen_marine": 37.0, "zemmour_eric": 14.0}


def test_single_candidate_yes_rows():
    s = _src(election=2017, question_type="chances", layout="labels",
             candidate="fillon_francois", label="^oui")
    vals, _ = ex.read_table(_page("ifop_162_p7.pdf"), s, LOOKUP)
    assert vals["win"] == {"fillon_francois": 55.0}  # 7 certainement + 48 probablement


def test_missing_anchor_is_an_error():
    with pytest.raises(ex.TableError, match="ancre"):
        ex.read_table(_page("kantar_288_p29.pdf"), _src(anchor="qui sera elu"), LOOKUP)


@pytest.mark.parametrize(
    ("text", "ok"),
    [("Macron 72 % ; Le Pen 26 %, sans opinion 2", True), ("Macron 72 ; Le Pen 62", False)],
)
def test_verify_numbers_against_raw_copy(text, ok):
    s = _src(layout="manual", values={"macron_emmanuel": 72, "le_pen_marine": 26})
    assert (ex.verify_web(s, text) == []) is ok


@pytest.mark.parametrize(
    ("text", "typ"),
    [
        ("Selon vous, qui sera élu Président de la République en avril prochain ?", "winner"),
        ("Selon vous, quel candidat l’emportera à l’issue du second tour ?", "winner"),
        ("…peut remporter l’élection présidentielle", "chances"),
        ("Quel candidat souhaiteriez-vous voir remporter l’élection présidentielle 2022 ?", "wish"),
        ("Selon vous, chacun des candidats suivants ferait-il un bon président ?", None),
        ("…ferait un bon Président / une bonne Présidente", "good_president"),
        ("pour lequel y aurait-il le plus de chances que vous votiez ?", None),
    ],
)
def test_page_cues(text, typ):
    cues = ex.page_cues(text)
    found = [t for t in ex.QUESTION_TYPES if cues[t]]
    assert found == ([typ] if typ else [])


def test_registry_loads_and_names_known_candidates():
    """Le registre réel se lit : types, formes, populations et ids de candidats valides."""
    sources = ex.load_sources()
    assert {s.kind for s in sources} == {"commission", "ifop_report", "web"}
    web = [s for s in sources if s.kind == "web"]
    assert all(s.url and s.file and s.fieldwork for s in web)
