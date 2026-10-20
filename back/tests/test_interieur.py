"""Résultats officiels du ministère de l'Intérieur (vvs.ingest.interieur).

Tests rapides : extraits figés de chaque format dans tests/fixtures/interieur/ (lignes
recopiées telles quelles des fichiers du ministère, en latin-1 ; pr22_fe_t2.txt,
leg2022_fe_t1.txt et leg2024_fe_t1.xlsx sont les fichiers d'origine complets). Les deux
classeurs pr02_dept.xls et euro2014_dept.xlsx ont été régénérés par LibreOffice à partir de
lignes réelles (2002 : départements 01, ZA, ZZ ; 2014 : 01, ZA, ZZ), faute de pouvoir couper
un .xls d'origine.

Tests lents (`-m slow`) : parsing des vrais fichiers de data/raw/results (téléchargés au
besoin) et comparaison aux chiffres proclamés par le Conseil constitutionnel.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from vvs import paths
from vvs.ingest import interieur
from vvs.ingest.interieur import formats, geo
from vvs.ingest.interieur.candidates import (
    CANDIDATE_IDS,
    UnknownCandidateError,
    candidate_id,
    normalize_label,
)
from vvs.ingest.interieur.xls import XlsError, read_xls
from vvs.schemas import RESULTS_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures" / "interieur"


# --- Codes géographiques ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("dept", "commune", "expected"),
    [
        ("1", None, "01"),
        (1.0, None, "01"),
        ("2A", None, "2A"),
        ("95", None, "95"),
        ("ZA", "101", "971"),
        ("ZM", "601", "976"),
        ("ZN", "801", "988"),
        ("ZP", "711", "987"),
        ("ZS", "501", "975"),
        ("ZW", "001", "986"),
        ("ZX", "701", "977"),
        ("ZX", "801", "978"),
        ("ZZ", "001", "ZZ"),
    ],
)
def test_department_codes(dept, commune, expected):
    assert geo.department(dept, commune) == expected


@pytest.mark.parametrize(
    ("dept", "commune", "expected"),
    [
        ("01", "001", "01001"),
        ("2B", "033", "2B033"),
        ("75", "056", "75056"),
        ("ZA", "101", "97101"),
        ("ZC", "352", "97352"),
        ("ZM", "501", "97601"),  # Mayotte : 5xx jusqu'en 2017
        ("ZM", "601", "97601"),  # 6xx en 2022
        ("ZP", "011", "98711"),  # Polynésie : 0xx jusqu'en 2017
        ("ZP", "711", "98711"),  # 7xx en 2022
        ("ZX", "701", "97701"),
        ("ZX", "801", "97801"),
        ("ZS", "502", "97502"),
        ("ZW", "001", "ZW001"),
        ("ZZ", "009", "ZZ009"),
    ],
)
def test_commune_codes(dept, commune, expected):
    assert geo.commune(dept, commune) == expected


@pytest.mark.parametrize(("dept", "commune"), [("ZY", "001"), ("96", None), ("ZX", "123")])
def test_unknown_geo_codes_fail(dept, commune):
    with pytest.raises(geo.GeoCodeError):
        geo.department(dept, commune)


# --- Candidats ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("last", "first", "expected"),
    [
        ("MÉLENCHON", "Jean-Luc", "melenchon_jean_luc"),
        ("LE PEN", "JEAN-MARIE", "le_pen_jean_marie"),
        ("de VILLIERS", "Philippe", "de_villiers_philippe"),
        ("ZEMMOUR", "Éric", "zemmour_eric"),
        ("PÉCRESSE", "Valérie", "pecresse_valerie"),
        ("ROYAL", "Ségolène", "royal_segolene"),
    ],
)
def test_candidate_labels(last, first, expected):
    assert candidate_id(normalize_label(last, first)) == expected


def test_unknown_candidate_fails():
    with pytest.raises(UnknownCandidateError):
        candidate_id(normalize_label("DUPONT", "Jean"))


def test_candidate_ids_follow_registry_convention():
    for label, cid in CANDIDATE_IDS.items():
        assert cid == label.lower().replace("-", "_").replace(" ", "_"), label


# --- Lecteurs de formats ------------------------------------------------------------------


def test_bvot_long_2002_comments_and_padded_numbers():
    df = formats.read_bvot_long(FIXTURES / "pr02_bvot_long.txt", (1, 2))
    assert df.schema == formats.UNITS_SCHEMA
    assert df.filter(pl.col("round") == 1)["unit"].n_unique() == 2
    assert set(df.filter(pl.col("round") == 2)["label"]) == {"CHIRAC JACQUES", "LE PEN JEAN-MARIE"}
    # « ␣␣1003 » dans le fichier d'origine
    fourmies = df.filter(pl.col("unit") == "59|249|0001", pl.col("label") == "CHIRAC JACQUES")
    assert fourmies["votes"].to_list() == [1003]
    first = df.filter(pl.col("unit") == "01|001|0001", pl.col("round") == 1)
    assert first["votes"].sum() == first["expressed"][0] == 427


def test_bvot_long_2012_has_two_extra_fields():
    df = formats.read_bvot_long(FIXTURES / "pr12_bvot_long.txt", (1, 2))
    assert set(df["dept_src"]) == {"ZX", "ZZ"}
    st_barth = df.filter(pl.col("unit") == "ZX|701|0001", pl.col("round") == 1)
    assert st_barth["votes"].sum() == st_barth["expressed"][0] == 444
    assert "MELENCHON JEAN-LUC" in set(df["label"])


def test_bvot_long_rejects_unexpected_round():
    with pytest.raises(formats.FormatError):
        formats.read_bvot_long(FIXTURES / "pr02_bvot_long.txt", (1,))


def test_wide_txt_2017_polling_stations():
    df = formats.read_wide_txt(FIXTURES / "pr17_bvot_t1.txt", (1,))
    assert df["unit"].n_unique() == 4
    assert df.group_by("unit").len()["len"].to_list() == [11] * 4
    ain = df.filter(pl.col("dept_src") == "01")
    assert ain["votes"].sum() == ain["expressed"][0] == 495
    assert ain.filter(pl.col("label") == "LE PEN MARINE")["votes"].item() == 126


def test_wide_txt_requires_single_round():
    with pytest.raises(formats.FormatError):
        formats.read_wide_txt(FIXTURES / "pr17_bvot_t1.txt", (1, 2))


def test_xls_reader_and_department_sheets():
    book = read_xls(FIXTURES / "pr02_dept.xls")
    assert {"Départements T1", "Départements T2"} <= set(book)
    df = formats.read_xls_dept(FIXTURES / "pr02_dept.xls", (1, 2))
    assert sorted(set(df["dept_src"])) == ["01", "ZA", "ZZ"]  # ligne total ignorée
    ain = df.filter(pl.col("dept_src") == "01", pl.col("round") == 1)
    assert ain["registered"][0] == 338220
    assert ain.filter(pl.col("label") == "LE PEN JEAN-MARIE")["votes"].item() == 52617
    t2 = df.filter(pl.col("round") == 2)
    assert (
        t2.group_by("dept_src")
        .agg(pl.col("votes").sum() == pl.col("expressed").first())["votes"]
        .all()
    )


def test_xls_reader_rejects_non_ole(tmp_path):
    p = tmp_path / "x.xls"
    p.write_bytes(b"pas un classeur")
    with pytest.raises(XlsError):
        read_xls(p)


# --- Parsing complet d'une présidentielle sur extraits -----------------------------------


def _sources_2022(tmp_path: Path, fe_name: str | None) -> dict[str, interieur.ElectionSources]:
    files = [
        interieur.SourceFile(
            name="pr22_subcom_t2.txt", url="", format="wide_txt", role="commune",
            rounds=(2,), sha256=None,
        )
    ]  # fmt: skip
    if fe_name:
        files.append(
            interieur.SourceFile(
                name=fe_name, url="", format="wide_txt", role="national", rounds=(2,), sha256=None
            )
        )
    return {
        "t": interieur.ElectionSources(key="t", kind="presidential", year=2022, files=tuple(files))
    }


def _synthetic_fe(tmp_path: Path) -> str:
    """Ligne France entière égale à la somme des communes de l'extrait (construite ici, à
    partir du vrai en-tête, pour tester le contrôle national)."""
    units = formats.read_wide_txt(FIXTURES / "pr22_subcom_t2.txt", (2,))
    tot = units.select("unit", "registered", "voters", "expressed").unique().drop("unit").sum()
    votes = units.group_by("label").agg(pl.col("votes").sum())
    v = dict(zip(votes["label"], votes["votes"]))
    header = (FIXTURES / "pr22_fe_t2.txt").read_bytes().decode("latin-1").splitlines()[0]
    row = (
        f"FE;France entière;Complet;{tot['registered'][0]};0;0;{tot['voters'][0]};0;0;0;0;0;0;0;"
        f"{tot['expressed'][0]};0;0;1;M;MACRON;Emmanuel;{v['MACRON EMMANUEL']};0;0;"
        f"2;F;LE PEN;Marine;{v['LE PEN MARINE']};0;0"
    )
    (tmp_path / "fe.txt").write_bytes((header + "\r\n" + row + "\r\n").encode("latin-1"))
    return "fe.txt"


def test_parse_presidential_levels_and_codes(tmp_path):
    shutil.copy(FIXTURES / "pr22_subcom_t2.txt", tmp_path)
    fe = _synthetic_fe(tmp_path)
    df = interieur.parse_presidential("t", base=tmp_path, sources=_sources_2022(tmp_path, fe))
    assert df.schema == RESULTS_SCHEMA
    assert set(df["geo_level"]) == {"national", "department", "commune"}
    depts = set(df.filter(pl.col("geo_level") == "department")["geo_code"])
    assert depts == {"01", "2A", "976", "977", "978", "986", "987", "ZZ"}
    communes = set(df.filter(pl.col("geo_level") == "commune")["geo_code"])
    assert {"01001", "2A004", "97601", "98711", "97701", "97801", "ZW001", "ZZ001"} <= communes
    assert set(df["candidate_id"]) == {"macron_emmanuel", "le_pen_marine"}
    # Le national est la somme des départements, qui est la somme des communes
    nat = df.filter(pl.col("geo_level") == "national")
    dep = df.filter(pl.col("geo_level") == "department")
    assert nat["votes"].sum() == dep["votes"].sum() == nat["expressed"][0]
    st_barth = df.filter(pl.col("geo_code") == "977", pl.col("candidate_id") == "macron_emmanuel")
    assert st_barth["registered"].item() == 5270


def test_parse_presidential_national_mismatch_fails(tmp_path):
    shutil.copy(FIXTURES / "pr22_subcom_t2.txt", tmp_path)
    shutil.copy(FIXTURES / "pr22_fe_t2.txt", tmp_path)  # vrai France entière : ≠ extrait
    with pytest.raises(interieur.ConsistencyError):
        interieur.parse_presidential(
            "t", base=tmp_path, sources=_sources_2022(tmp_path, "pr22_fe_t2.txt")
        )


def test_parse_presidential_unknown_candidate_fails(tmp_path):
    text = (FIXTURES / "pr22_subcom_t2.txt").read_bytes().decode("latin-1")
    (tmp_path / "pr22_subcom_t2.txt").write_bytes(
        text.replace("MACRON;Emmanuel", "MACRON;Emanuel").encode("latin-1")
    )
    with pytest.raises(UnknownCandidateError):
        interieur.parse_presidential("t", base=tmp_path, sources=_sources_2022(tmp_path, None))


def test_parse_checks_sha256(tmp_path):
    shutil.copy(FIXTURES / "pr22_subcom_t2.txt", tmp_path)
    src = _sources_2022(tmp_path, None)
    f = src["t"].files[0]
    bad = interieur.ElectionSources(
        key="t", kind="presidential", year=2022,
        files=(interieur.SourceFile(**{**f.__dict__, "sha256": "0" * 64}),),
    )  # fmt: skip
    with pytest.raises(interieur.ConsistencyError):
        interieur.parse_presidential("t", base=tmp_path, sources={"t": bad})


# --- Scrutins de listes -------------------------------------------------------------------


def test_euro2014_aggregates_by_nuance():
    df = formats.read_euro2014_xlsx(FIXTURES / "euro2014_dept.xlsx", (1,))
    assert sorted(set(df["unit"])) == ["01", "ZA", "ZZ"]
    ain = df.filter(pl.col("unit") == "01")
    assert ain["votes"].sum() == ain["expressed"][0] == 163020
    assert set(df["key"]) >= {
        "LFN|Liste Front National",
        "LUMP|Liste Union pour un Mouvement Populaire",
    }


def test_euro2019_lists_without_nuance():
    df = formats.read_euro_dept_txt(FIXTURES / "euro2019_dept.txt", (1,))
    assert df["key"].n_unique() == 34
    assert df["nuance"].is_null().all()
    assert (
        df.group_by("unit").agg(pl.col("votes").sum() == pl.col("expressed").first())["votes"].all()
    )


def test_euro2024_quoted_csv():
    df = formats.read_euro_dept_csv(FIXTURES / "euro2024_dept.csv", (1,))
    assert df["key"].n_unique() == 38
    rn = df.filter(pl.col("unit") == "01", pl.col("nuance") == "LRN")
    assert rn["votes"].item() == 86768
    assert rn["label"].item() == "LA FRANCE REVIENT ! AVEC JORDAN BARDELLA ET MARINE LE PEN"


def test_legislative_national_files():
    t1 = formats.read_leg_fe_txt(FIXTURES / "leg2022_fe_t1.txt", (1,))
    assert t1["votes"].sum() == t1["expressed"][0] == 22744708
    assert t1.filter(pl.col("nuance") == "RN")["votes"].item() == 4248537
    x = formats.read_leg_fe_xlsx(FIXTURES / "leg2024_fe_t1.xlsx", (1,))
    assert x.filter(pl.col("nuance") == "RN")["votes"].item() == 9379092
    assert x["expressed"][0] == 32057946


def _leg2024_sources(gaps: dict[int, int]) -> dict[str, interieur.ElectionSources]:
    f = interieur.SourceFile(
        name="leg2024_fe_t1.xlsx", url="", format="leg_fe_xlsx", role="national",
        rounds=(1,), sha256=None,
    )  # fmt: skip
    return {
        "l": interieur.ElectionSources(
            key="l", kind="legislative", year=2024, files=(f,), vote_sum_gaps=gaps
        )
    }


def test_parse_lists_requires_declared_vote_sum_gap():
    # Le fichier France entière 2024 du ministère a 2 voix de moins que les exprimés au T1
    with pytest.raises(interieur.ConsistencyError):
        interieur.parse_lists("l", base=FIXTURES, sources=_leg2024_sources({}))
    df = interieur.parse_lists("l", base=FIXTURES, sources=_leg2024_sources({1: -2}))
    assert df.schema == interieur.RESULTS_LISTS_SCHEMA
    assert df["votes"].sum() == df["expressed"][0] - 2
    assert set(df["geo_code"]) == {interieur.NATIONAL_CODE}


def test_registry_is_complete():
    sources = interieur.load_sources()
    assert {"2002", "2007", "2012", "2017", "2022"} <= set(sources)
    for e in sources.values():
        for f in e.files:
            assert f.sha256 and len(f.sha256) == 64, (e.key, f.name)
            fmt = formats.PRESIDENTIAL_READERS if e.kind == "presidential" else formats.LIST_READERS
            assert f.format in fmt, (e.key, f.name)


# --- Fichiers réels contre les chiffres du Conseil constitutionnel (lents) -----------------

# Résultats proclamés par le Conseil constitutionnel (déclarations du premier tour,
# décisions de proclamation du second), relevés sur conseil-constitutionnel.fr le
# 2026-10-20 : 2002 (déclaration du 24 avril, décision du 8 mai), 2007-139 et 2007 du 10 mai,
# 2012-152 et 2012-154, 2017-169 et 2017-171, 2022-195 et 2022-197 PDR.
CC = {
    (2002, 1): {
        "registered": 41194689, "voters": 29495733, "expressed": 28498471,
        "votes": {
            "chirac_jacques": 5665855, "le_pen_jean_marie": 4804713, "jospin_lionel": 4610113,
            "bayrou_francois": 1949170, "laguiller_arlette": 1630045,
            "chevenement_jean_pierre": 1518528, "mamere_noel": 1495724,
            "besancenot_olivier": 1210562, "saint_josse_jean": 1204689, "madelin_alain": 1113484,
            "hue_robert": 960480, "megret_bruno": 667026, "taubira_christiane": 660447,
            "lepage_corinne": 535837, "boutin_christine": 339112, "gluckstein_daniel": 132686,
        },
    },
    (2002, 2): {
        "registered": 41191169, "voters": 32832295, "expressed": 31062988,
        "votes": {"chirac_jacques": 25537956, "le_pen_jean_marie": 5525032},
    },
    (2007, 1): {
        "registered": 44472834, "voters": 37254242, "expressed": 36719396,
        "votes": {
            "besancenot_olivier": 1498581, "buffet_marie_george": 707268,
            "schivardi_gerard": 123540, "bayrou_francois": 6820119, "bove_jose": 483008,
            "voynet_dominique": 576666, "de_villiers_philippe": 818407,
            "royal_segolene": 9500112, "nihous_frederic": 420645, "le_pen_jean_marie": 3834530,
            "laguiller_arlette": 487857, "sarkozy_nicolas": 11448663,
        },
    },
    (2007, 2): {
        "registered": 44472733, "voters": 37342004, "expressed": 35773578,
        "votes": {"sarkozy_nicolas": 18983138, "royal_segolene": 16790440},
    },
    (2012, 1): {
        "registered": 46028542, "voters": 36584399, "expressed": 35883209,
        "votes": {
            "joly_eva": 828345, "le_pen_marine": 6421426, "sarkozy_nicolas": 9753629,
            "melenchon_jean_luc": 3984822, "poutou_philippe": 411160, "arthaud_nathalie": 202548,
            "cheminade_jacques": 89545, "bayrou_francois": 3275122,
            "dupont_aignan_nicolas": 643907, "hollande_francois": 10272705,
        },
    },
    (2012, 2): {
        "registered": 46066307, "voters": 37016309, "expressed": 34861353,
        "votes": {"hollande_francois": 18000668, "sarkozy_nicolas": 16860685},
    },
    (2017, 1): {
        "registered": 47582183, "voters": 37003728, "expressed": 36054394,
        "votes": {
            "dupont_aignan_nicolas": 1695000, "le_pen_marine": 7678491,
            "macron_emmanuel": 8656346, "hamon_benoit": 2291288, "arthaud_nathalie": 232384,
            "poutou_philippe": 394505, "cheminade_jacques": 65586, "lassalle_jean": 435301,
            "melenchon_jean_luc": 7059951, "asselineau_francois": 332547,
            "fillon_francois": 7212995,
        },
    },
    (2017, 2): {
        "registered": 47568693, "voters": 35467327, "expressed": 31381603,
        "votes": {"macron_emmanuel": 20743128, "le_pen_marine": 10638475},
    },
    (2022, 1): {
        "registered": 48747876, "voters": 35923707, "expressed": 35132947,
        "votes": {
            "arthaud_nathalie": 197094, "roussel_fabien": 802422, "macron_emmanuel": 9783058,
            "lassalle_jean": 1101387, "le_pen_marine": 8133828, "zemmour_eric": 2485226,
            "melenchon_jean_luc": 7712520, "hidalgo_anne": 616478, "jadot_yannick": 1627853,
            "pecresse_valerie": 1679001, "poutou_philippe": 268904,
            "dupont_aignan_nicolas": 725176,
        },
    },
    (2022, 2): {
        "registered": 48752339, "voters": 35096478, "expressed": 32057325,
        "votes": {"macron_emmanuel": 18768639, "le_pen_marine": 13288686},
    },
}  # fmt: skip

# Écarts connus (fichier du ministère - Conseil constitutionnel), voir les notes du registre :
# - 2002 : fichier départemental antérieur aux rectifications du Conseil ;
# - 2007 T1 : la somme des départements du classeur s'écarte de sa propre feuille « France
#   entière » (qui reprend le Conseil) de quelques voix entre candidats ;
# - 2012 : inscrits seulement (voix, votants et exprimés identiques).
KNOWN_DIFFS = {
    (2002, 1): {
        "registered": 3275, "voters": 2155, "expressed": 1016,
        "votes": {
            "chirac_jacques": 166, "le_pen_jean_marie": 59, "jospin_lionel": 154,
            "bayrou_francois": 49, "laguiller_arlette": 73, "chevenement_jean_pierre": 40,
            "mamere_noel": 50, "saint_josse_jean": 112, "madelin_alain": 67, "hue_robert": 68,
            "megret_bruno": 17, "taubira_christiane": 68, "lepage_corinne": 38,
            "boutin_christine": 45, "gluckstein_daniel": 10,
        },
    },
    (2002, 2): {
        "registered": 1103, "voters": 537, "expressed": -60,
        "votes": {"chirac_jacques": -62, "le_pen_jean_marie": 2},
    },
    (2007, 1): {"votes": {"bayrou_francois": -2, "bove_jose": 1, "laguiller_arlette": 1}},
    (2012, 1): {"registered": 29},
    (2012, 2): {"registered": -92},
}  # fmt: skip


def _curated(election: str) -> pl.DataFrame:
    """Parse les vrais fichiers (téléchargés si absents)."""
    sources = interieur.load_sources()
    if not all((interieur.raw_dir(election) / f.name).exists() for f in sources[election].files):
        interieur.ingest(election, sources)
    return (
        interieur.parse_presidential(election, sources=sources)
        if sources[election].kind == "presidential"
        else interieur.parse_lists(election, sources=sources)
    )


@pytest.mark.slow
@pytest.mark.parametrize("year", [2002, 2007, 2012, 2017, 2022])
def test_national_matches_conseil_constitutionnel(year):
    df = _curated(str(year))
    assert df.schema == RESULTS_SCHEMA
    for rnd in (1, 2):
        ref, known = CC[year, rnd], KNOWN_DIFFS.get((year, rnd), {})
        nat = df.filter(pl.col("geo_level") == "national", pl.col("round") == rnd)
        got = dict(zip(nat["candidate_id"], nat["votes"]))
        assert set(got) == set(ref["votes"])
        diffs = {c: got[c] - v for c, v in ref["votes"].items() if got[c] != v}
        assert diffs == known.get("votes", {}), (year, rnd)
        for col in ("registered", "voters", "expressed"):
            assert nat[col][0] - ref[col] == known.get(col, 0), (year, rnd, col)
        # Départements : métropole + outre-mer + étranger, chacun une fois
        dep = df.filter(pl.col("geo_level") == "department", pl.col("round") == rnd)
        codes = set(dep["geo_code"])
        assert "ZZ" in codes and "75" in codes and "2A" in codes and "971" in codes
        assert len(codes) == (106 if year <= 2007 else 108)


@pytest.mark.slow
def test_lists_national_figures():
    e14 = _curated("euro2014")
    # Feuille « Résultats » du même classeur : national par nuance
    assert dict(zip(e14["nuance"], e14["votes"]))["LFN"] == 4712461
    assert e14["expressed"][0] == 18955761
    e19 = _curated("euro2019")
    assert e19["votes"].max() == 5286939 and e19["expressed"][0] == 22655174
    e24 = _curated("euro2024")
    assert dict(zip(e24["nuance"], e24["votes"]))["LRN"] == 7765936
    leg22 = _curated("leg2022")
    assert set(leg22["round"]) == {1, 2}
    leg24 = _curated("leg2024")
    assert leg24.filter(pl.col("round") == 1, pl.col("nuance") == "RN")["votes"].item() == 9379092


@pytest.mark.slow
def test_curated_files_written():
    for election in ("2022", "leg2022"):
        out, df = interieur.parse(election)
        assert out.parent == paths.CURATED and out.exists()
        assert pl.read_parquet(out).equals(df)
