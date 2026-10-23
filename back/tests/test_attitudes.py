"""Séries d'attitudes (vvs.ingest.attitudes) : lecteurs de graphiques, tableaux, classeurs et
microdonnées sur extraits figés des documents réels (tests/fixtures/attitudes/), registre et
table curated. Les pages PDF sont extraites des bruts par `pdfseparate`, sauf la page 40 des
deux éditions Verian (chiffrement non recopiable : `pdftocairo -pdf`, mêmes positions et
couleurs de texte) et celle de la CNCDH (Ghostscript, polices sous-ensemble, 4,2 Mo -> 49 Ko,
mêmes objets texte et vectoriels). Les extraits Eurobaromètre sont les premières lignes de
feuilles réelles (mise en page conservée)."""

import json
from datetime import date
from pathlib import Path

import pdfplumber
import polars as pl
import pytest

from vvs.ingest import attitudes
from vvs.ingest.attitudes import cncdh, drees, eurobarometer, extract, pdfchart, registry

FIX = Path(__file__).parent / "fixtures" / "attitudes"
REG = registry.load()


def _ex(ex_id: str) -> registry.Extraction:
    return next(e for e in REG.extractions if e.id == ex_id)


def _chart(fixture: str, ex_id: str) -> dict[str, dict[str, float]]:
    e = _ex(ex_id)
    with pdfplumber.open(FIX / fixture) as pdf:
        got = pdfchart.extract_chart(pdf.pages[0], extract.chart_spec(e.spec))
    if e.spec.get("exclusive"):
        extract.check_exclusive(ex_id, got)
    return {sid: dict(vals) for sid, vals in got.items()}


def _table(fixture: str, ex_id: str) -> dict[str, dict[str, float]]:
    with pdfplumber.open(FIX / fixture) as pdf:
        got = pdfchart.extract_table(pdf.pages[0], extract.table_spec(_ex(ex_id).spec))
    return {sid: dict(vals) for sid, vals in got.items()}


# ---------------------------------------------------------------------------- registre


def test_registry_covers_the_six_families_and_documents_every_series():
    fams = {s.family for s in REG.series.values()}
    assert fams == {"rn_image", "cevipof_confidence", "cncdh", "eurobarometer", "credoc", "drees"}
    for s in REG.series.values():
        assert s.question and s.source and s.population, s.series_id
    for d in REG.documents.values():
        assert d.url.startswith("https://"), d.id
    # toute date de terrain du calendrier cite son document
    for fam, waves in REG.waves.items():
        for label, w in waves.items():
            assert (w.document is None) == (w.quote == ""), (fam, label)


def test_registry_rejects_series_without_extraction(tmp_path):
    raw = (registry.registry_file()).read_text(encoding="utf-8")
    raw = raw.replace("series:\n", "series:\n  - {series_id: orphan, family: x, label: x, "
                      "question: x, source: x, freq: A, unit: percent, "
                      "population: fr_adults_18plus, since: 2000}\n", 1)
    p = tmp_path / "reg.yaml"
    p.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="sans extraction"):
        registry.load(p)


# ------------------------------------------------------------------- graphiques et tableaux


def test_rn_danger_chart_reads_all_waves_since_1983():
    got = _chart("rn_image_2026_01_p7.pdf", "rn2026_p7_danger")
    yes, no = got["rn_danger_democracy_yes"], got["rn_danger_democracy_no"]
    assert len(yes) == len(no) == 37
    assert (yes["1983-10"], no["1983-10"]) == (38, 43)
    assert (yes["1997-03"], no["1997-03"]) == (75, 19)
    # janvier 2010 : l'étiquette « 42 » est décalée d'une colonne vers la droite sur la page
    assert (yes["2010-01"], no["2010-01"]) == (52, 42)
    assert (yes["2024-11"], no["2024-11"]) == (38, 51)
    assert (yes["2026-01"], no["2026-01"]) == (41, 44)


def test_rn_chart_rejects_a_wrong_wave_count():
    e = _ex("rn2026_p7_danger")
    spec = extract.chart_spec({**e.spec, "waves": e.spec["waves"][:-1]})
    with (
        pdfplumber.open(FIX / "rn_image_2026_01_p7.pdf") as pdf,
        pytest.raises(pdfchart.ChartError, match="colonnes"),
    ):
        pdfchart.extract_chart(pdf.pages[0], spec)


def test_mlp_image_chart_reads_text_axis_and_incomplete_series():
    got = _chart("rn_image_2026_01_p40.pdf", "rn2026_p40_mlp_image")
    pres = got["mlp_good_president"]
    assert min(pres) == "2016-02" and len(pres) == 11
    assert (pres["2016-02"], pres["2023-11"], pres["2026-01"]) == (22, 34, 34)
    # étiquette « 51 » posée entre février 2020 et mai 2021 : série complète lue dans l'ordre
    assert (got["mlp_able_to_decide"]["2020-02"], got["mlp_able_to_decide"]["2021-05"]) == (51, 53)
    assert len(got["mlp_honest_trustworthy"]) == 13  # une vague sans étiquette


def test_rn_opinion_table_2000_2024():
    got = _table("rn_image_2024_11_p40.pdf", "rn2024_p40_opinions")
    imm = got["opinion_too_many_immigrants"]
    assert (imm["2000-05"], imm["2010-01"], imm["2023-11"], imm["2024-11"]) == (59, 44, 52, 48)
    # droit du sol : « - » en novembre 2023 (non posé)
    assert got["opinion_abolish_birthright_citizenship"] == {"2022-12": 29, "2024-11": 40}
    # libellé sur deux lignes autour de la ligne de valeurs
    assert got["opinion_hiring_priority_french"] == {"2024-11": 44}


def test_cevipof_democracy_chart_with_grey_wave_6b():
    got = _chart("cevipof_confidence_v17_p34.pdf", "cevipof_v17_p34_democracy")
    bad, good = got["cevipof_democracy_works_badly"], got["cevipof_democracy_works_well"]
    assert len(bad) == len(good) == 18
    assert (bad["2015-02"], good["2015-02"]) == (61, 39)  # étiquettes grises de la vague 6 bis
    assert (bad["2009-12"], good["2009-12"]) == (48, 50)
    assert (bad["2026-01"], good["2026-01"]) == (76, 23)


def test_credoc_table_with_vertical_year_headers():
    got = _table("credoc_note_2015_p53.pdf", "credoc2015_p53_concerns")
    assert all(len(v) == 25 for v in got.values())
    une = got["credoc_concern_unemployment"]
    assert (une["1991"], une["1994"], une["2014"], une["2015"]) == (27, 52, 55, 48)
    assert got["credoc_concern_immigration"]["2015"] == 15


def test_align_skips_the_right_column():
    cols = [0.0, 10.0, 20.0, 30.0]
    assert pdfchart.align([1.0, 21.0, 29.0], cols) == [0, 2, 3]
    assert pdfchart.align([9.0, 19.0], cols) == [1, 2]


def test_exclusive_check_catches_misplaced_labels():
    with pytest.raises(extract.ExtractionError, match="100 %"):
        extract.check_exclusive("t", {"a": [("2011-01", 72.0)], "b": [("2011-01", 31.0)]})


# ------------------------------------------------------------------------------------ CNCDH


def test_cncdh_index_from_vector_markers_matches_printed_labels():
    years, labelled = cncdh.read_chart(
        FIX / "cncdh_racism_report_2025_p258.pdf", 1, _ex("cncdh2025_p258_ilt").spec
    )
    assert len(years) == 33 and not {1992, 2020, 2021} & set(years)
    assert len(labelled) == 19
    assert (years[1990], years[1991], years[2013], years[2025]) == (52, 46, 51, 64)
    assert years[2008] == 62 and 2008 not in labelled  # année sans étiquette


# ----------------------------------------------------------------------------- Eurobaromètre


@pytest.fixture(scope="module")
def eb_rows():
    return json.loads((FIX / "eurobarometer_volume_a_extracts.json").read_text(encoding="utf-8"))


def test_eurobarometer_modern_sheet(eb_rows):
    rows = eb_rows["eb_std104:QA3"]
    total, items = eurobarometer.parse_modern(rows[2:])
    assert total == 1001
    assert items["immigration"][:2] == (112.0, 11.0)
    assert round(100 * items["rising_prices"][0] / total, 1) == 35.8
    assert eurobarometer.fieldwork(rows) == (date(2025, 10, 9), date(2025, 11, 5), date(2025, 10, 1))
    assert "unemployment" in items and "terrorism" not in items  # item retiré de la liste


def test_eurobarometer_half_samples_have_different_item_lists(eb_rows):
    _, a = eurobarometer.parse_modern(eb_rows["eb_std074:QA6a1"][2:])
    _, b = eurobarometer.parse_modern(eb_rows["eb_std074:QA6a2"][2:])
    assert "government_debt" in b and "government_debt" not in a
    # « Climate change » (demi-échantillon B) n'est pas fondu dans l'environnement
    assert b["environment_climate"][2] == "The environment"


def test_eurobarometer_legacy_text_sheet(eb_rows):
    total, items = eurobarometer.parse_legacy(eb_rows["eb_std060:EB601VOLA"])
    assert total == 1015
    assert items["unemployment"][:2] == (546.0, 53.8)
    assert items["rising_prices"][2].startswith("4.Rising prices/")  # libellé sur deux lignes
    assert {"pensions", "education", "health"} <= set(items)  # après le saut de page


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Terrain/Fieldwork : 9/10 - 5/11/2025", (date(2025, 10, 9), date(2025, 11, 5))),
        ("Terrain/Fieldwork : 12-27/05/2012", (date(2012, 5, 12), date(2012, 5, 27))),
        ("Terrain/Fieldwork : 08 - 17/11/2014", (date(2014, 11, 8), date(2014, 11, 17))),
        ("Terrain/Fieldwork : 05-06/2005", (None, None)),
        ("Terrain/Fieldwork : 10/2008", (None, None)),
    ],
)
def test_eurobarometer_fieldwork_formats(text, expected):
    assert eurobarometer.fieldwork([[None, text]])[:2] == expected


# ------------------------------------------------------------------------------------ DREES


def test_drees_shares_keep_dk_in_base_and_skip_unasked(tmp_path):
    csv = tmp_path / "b.csv"
    csv.write_text(
        "ident;annee;poids;in01\n"
        "1;2024;1,5;2\n"
        "2;2024;0,5;1\n"
        "3;2024;1,0;999999999\n"
        "4;2024;2,0;\n"  # question non posée : hors calcul
        "5;2025;1,0;2\n",
        encoding="utf-8",
    )
    got = drees.shares(csv, "in01", [2])
    assert got == {2024: pytest.approx(50.0), 2025: pytest.approx(100.0)}


# ---------------------------------------------------------------------- extraits et table


def test_number_in_quote():
    assert extract.number_in("l'immigration (32 %, +8 points)", 32)
    assert not extract.number_in("132 %", 32)
    assert extract.number_in("taux de 7,5 %", 7.5)


def test_validate_rejects_duplicates_and_out_of_range():
    row = {c: None for c in attitudes.ATTITUDES_SCHEMA}
    row.update(series_id="x", date=date(2020, 1, 1), value=50.0, freq="A", source="s",
               unit="percent", population="fr_adults_18plus", wave="2020", method="table",
               document="d", raw_sha256="h", notes="")
    df = pl.DataFrame([row, row], schema=attitudes.ATTITUDES_SCHEMA)
    with pytest.raises(ValueError, match="double"):
        attitudes.validate_attitudes(df)
    df = pl.DataFrame([{**row, "value": 101.0}], schema=attitudes.ATTITUDES_SCHEMA)
    with pytest.raises(ValueError, match="hors"):
        attitudes.validate_attitudes(df)


@pytest.mark.slow
def test_full_parse_from_raw_documents():
    """Lecture complète depuis data/raw/attitudes (après `vvs ingest attitudes`)."""
    if not (attitudes.raw.root() / "rn_image").exists():
        pytest.skip("bruts absents")
    df, report = attitudes.parse(write=False)
    assert not report.disagreements
    assert report.checks > 300
    assert df["series_id"].n_unique() == len(REG.series)
