"""Registre presse : panel de titres (outlets.yaml) et requêtes (press_queries.yaml)."""

import pytest

from vvs.ingest.press import ina
from vvs.ingest.press.outlets import (
    PressQuery,
    PressRegistryError,
    gdelt_query,
    load_outlets,
    load_queries,
    mediacloud_query,
)

CAMPAIGN_CANDIDATES = [
    "macron_emmanuel",
    "le_pen_marine",
    "melenchon_jean_luc",
    "fillon_francois",
    "hamon_benoit",
    "zemmour_eric",
    "pecresse_valerie",
    "jadot_yannick",
    "bardella_jordan",
    "philippe_edouard",
    "attal_gabriel",
    "retailleau_bruno",
    "glucksmann_raphael",
    "tondelier_marine",
]


@pytest.fixture(scope="module")
def queries():
    return load_queries()


@pytest.fixture(scope="module")
def outlets():
    return load_outlets()


def test_queries_cover_campaign_candidates(queries):
    assert set(CAMPAIGN_CANDIDATES) <= set(queries.candidates)
    assert queries.mediacloud_collection == 34412146
    assert str(queries.gdelt_start) == "2017-01-01"


@pytest.mark.parametrize(
    "cid, ambiguous",
    [
        ("philippe_edouard", "Philippe"),
        ("le_pen_marine", "Le Pen"),
        ("attal_gabriel", "Attal"),
        ("jadot_yannick", "Jadot"),
        ("glucksmann_raphael", "Glucksmann"),
        ("hamon_benoit", "Hamon"),
    ],
)
def test_ambiguous_surnames_are_rejected(queries, cid, ambiguous):
    assert ambiguous not in queries.candidate(cid).terms
    assert ambiguous in queries.rejected[cid]


def test_rejected_alias_reused_fails(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text(
        "defaults: {gdelt_filters: '', gdelt_start: 2017-01-01, mediacloud_collection: 1}\n"
        "candidates:\n  philippe_edouard:\n    aliases: [Philippe]\n"
        "    rejected: {Philippe: trop courant}\nthemes: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(PressRegistryError, match="alias rejeté"):
        load_queries(p)


def test_ina_rubrics_exist(queries):
    for t in queries.themes.values():
        assert set(t.ina_rubrics) <= set(ina.RUBRICS), t.topic
        assert (t.ina_fit == "none") == (not t.ina_rubrics), t.topic


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.lemonde.fr/politique/article/x.html", "le_monde"),
        ("abonnes.lemonde.fr", "le_monde"),
        ("https://www.lefigaro.fr/politique/x.php", "le_figaro"),
        ("https://www.lefigaro.fr/flash-actu/2017/01/05/x.php", "afp"),
        ("https://rmc.bfmtv.com/actualites/x.html", "rmc"),
        ("https://www.bfmtv.com/politique/x.html", "bfmtv"),
        ("https://www.huffingtonpost.fr/2017/x", "huffpost"),
        ("lci.fr", "tf1info"),
        ("https://www.closermag.fr/people/x", None),
        ("https://notlemonde.fr/x", None),
        ("", None),
    ],
)
def test_outlet_match(outlets, url, expected):
    assert outlets.match(url) == expected


def test_panel_excludes_regional_and_international(outlets):
    panel = {o.outlet_id for o in outlets.panel()}
    assert {"le_monde", "afp", "bfmtv", "rtl"} <= panel
    assert not {"ouest_france", "france24", "rfi", "rmc"} & panel


def test_duplicate_domain_fails(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text(
        "outlets:\n"
        "  a: {name: A, kind: daily, domains: [a.fr], panel: true}\n"
        "  b: {name: B, kind: daily, domains: [a.fr], panel: true}\n",
        encoding="utf-8",
    )
    with pytest.raises(PressRegistryError, match="déjà attribué"):
        load_outlets(p)


def test_gdelt_query_syntax(queries):
    f = queries.gdelt_filters
    # un seul terme : pas de parenthèses (GDELT les refuse autour d'un terme unique)
    assert (
        gdelt_query(queries.candidate("zemmour_eric"), f)
        == "Zemmour sourcelang:french sourcecountry:FR"
    )
    assert (
        gdelt_query(queries.candidate("philippe_edouard"), f)
        == '("Édouard Philippe" OR "Edouard Philippe" OR "Edward Philippe") '
        "sourcelang:french sourcecountry:FR"
    )
    # GDELT cherche dans la traduction anglaise : « Le Pen » y devient « the Pen »
    assert gdelt_query(queries.candidate("le_pen_marine"), f).startswith("Pen ")
    assert mediacloud_query(queries.candidate("le_pen_marine")) == '"Marine Le Pen"'
    q = PressQuery("x", terms=("Jadot",), exclude=("Louis Jadot",))
    assert gdelt_query(q, "") == 'Jadot -"Louis Jadot"'
    t = gdelt_query(queries.theme_query("immigration"), f)
    assert "theme:IMMIGRATION" in t and t.startswith("(") and "sourcecountry:FR" in t


def test_mediacloud_query_syntax(queries):
    assert mediacloud_query(queries.candidate("melenchon_jean_luc")) == "(Mélenchon OR Melenchon)"
    q = PressQuery("x", terms=("Jadot",), exclude=("Louis Jadot", "vin"))
    assert mediacloud_query(q) == 'Jadot AND NOT ("Louis Jadot" OR vin)'


def test_gdelt_aliases_override_terms():
    q = PressQuery("x", terms=("Marine Le Pen",), gdelt_terms=("Pen", "Le Pen"))
    assert gdelt_query(q, "") == '(Pen OR "Le Pen")'
    assert mediacloud_query(q) == '"Marine Le Pen"'
