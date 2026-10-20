"""Séries macro et popularité (vvs.ingest.macro) : parseurs sur extraits figés des sources
réelles (tests/fixtures/macro/), registre, table curated. Le réseau est marqué `slow`."""

import itertools
import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from vvs import paths
from vvs.ingest import macro
from vvs.ingest.macro import dares, fuel, ifop, insee, raw, registry

FIX = Path(__file__).parent / "fixtures" / "macro"
REG = registry.load()
HOLDERS = REG.officeholders


# ---------------------------------------------------------------------------- registre


def test_registry_series_cover_the_spec():
    ids = {s.series_id for s in REG.series}
    assert {
        "unemployment_rate_ilo",
        "jobseekers_cat_a",
        "cpi_all",
        "cpi_energy",
        "cpi_food",
        "consumer_confidence",
        "consumer_confidence_future_living",
        "consumer_confidence_unemployment_expectations",
        "business_climate",
        "fuel_price_sp95",
        "fuel_price_diesel",
        "approval_president",
        "approval_prime_minister",
    } <= ids
    for s in REG.series:
        assert s.label and s.unit and s.since and s.url.startswith("https://")
        if s.source == "insee":
            assert s.source_id.isdigit() and len(s.source_id) == 9, s.series_id


@pytest.mark.parametrize("role", registry.ROLES)
def test_officeholders_do_not_overlap(role):
    """Un seul titulaire par rôle à une date donnée : sinon l'attribution d'un mois est ambiguë."""
    terms = sorted((h for h in HOLDERS if h.role == role), key=lambda h: h.start)
    for prev, nxt in itertools.pairwise(terms):
        assert prev.end is not None and prev.end <= nxt.start, (prev.person_id, nxt.person_id)
    assert terms[-1].end is None


# ---------------------------------------------------------------------------- INSEE


def test_insee_monthly_and_quarterly_periods():
    freq, obs = insee.parse((FIX / "insee_cpi_all.xml").read_bytes(), "011814630")
    assert freq == "M"
    assert obs[-1] == (date(2026, 9, 1), 103.07)
    assert [d for d, _ in obs] == sorted(d for d, _ in obs)
    freq, obs = insee.parse((FIX / "insee_unemployment_rate_ilo.xml").read_bytes(), "001688527")
    assert freq == "Q"
    assert obs[-1] == (date(2026, 4, 1), 8.3)  # 2026-Q2
    assert {d.month for d, _ in obs} <= {1, 4, 7, 10}


def test_insee_skips_missing_values_and_checks_idbank():
    content = (FIX / "insee_cpi_all.xml").read_bytes()
    nan = content.replace(b'OBS_VALUE="103.07"', b'OBS_VALUE="NaN"')
    _, obs = insee.parse(nan, "011814630")
    assert date(2026, 9, 1) not in dict(obs)
    with pytest.raises(ValueError, match="porte sur"):
        insee.parse(content, "001587668")


def test_insee_error_message_is_raised():
    err = (
        b'<?xml version="1.0"?><mes:Error xmlns:mes="http://www.sdmx.org/resources/sdmxml/'
        b'schemas/v2_1/message"><mes:ErrorMessage code="510"><com:Text xmlns:com="x">trop'
        b" volumineuse</com:Text></mes:ErrorMessage></mes:Error>"
    )
    with pytest.raises(ValueError, match="volumineuse"):
        insee.parse(err, "011814630")


# ---------------------------------------------------------------------------- DARES


def test_dares_keeps_requested_scope_only():
    obs = dares.parse((FIX / "dares_defm_cat_a.csv").read_bytes(), "France")
    assert obs == [
        (date(2025, 12, 1), 3323400.0),
        (date(2026, 1, 1), 3289700.0),
        (date(2026, 2, 1), 3279200.0),
    ]
    metro = dict(dares.parse((FIX / "dares_defm_cat_a.csv").read_bytes(), "France métropolitaine"))
    assert metro[date(2026, 1, 1)] == 3082000.0


def test_dares_rejects_duplicate_months():
    content = (FIX / "dares_defm_cat_a.csv").read_bytes()
    lines = content.splitlines()
    with pytest.raises(ValueError, match="double"):
        dares.parse(b"\n".join(lines + lines[1:2]), "France")


# ---------------------------------------------------------------------------- carburants


@pytest.mark.parametrize(
    "code, first",
    [("sp95", 1.0392), ("gazole", 0.8051)],
)
def test_fuel_weekly_prices_dated_on_monday(code, first):
    obs = fuel.parse((FIX / "mte_fuel_export.xlsx").read_bytes(), code, "Carburants - hebdo TTC")
    # vendredi 7 janvier 2000 -> lundi 3 janvier 2000
    assert obs[0] == (date(2000, 1, 3), first)
    assert all(d.weekday() == 0 for d, _ in obs)
    assert len(obs) == 5


def test_fuel_unknown_sheet_or_fuel():
    content = (FIX / "mte_fuel_export.xlsx").read_bytes()
    with pytest.raises(KeyError):
        fuel.parse(content, "sp95", "Carburants - quotidien")
    with pytest.raises(KeyError):
        fuel.parse(content, "e85", "Carburants - hebdo TTC")


# ---------------------------------------------------------------------------- Ifop


def test_ifop_post_month():
    pub = date(2025, 9, 1)
    assert ifop.post_month("Les indices de popularité – Août 2025", pub) == date(2025, 8, 1)
    assert ifop.post_month(
        "Les indices de popularité – vague de novembre", date(2023, 11, 20)
    ) == date(2023, 11, 1)
    assert ifop.post_month("Les indices de popularité", date(2022, 9, 26)) == date(2022, 9, 1)


def test_ifop_tables_assign_role_by_name():
    text = (FIX / "ifop_2014-03_tables.txt").read_text(encoding="utf-8")
    report = ifop.TextReport()
    obs = ifop.parse_text(text, date(2014, 3, 1), HOLDERS, report)
    by = {(o.person_id, o.month): o for o in obs}
    assert by[("hollande_francois", date(2012, 5, 1))].satisfied == 61
    assert by[("hollande_francois", date(2012, 7, 1))].satisfied == 56  # « 56 44 - 100 »
    assert by[("hollande_francois", date(2014, 3, 1))].satisfied == 23
    assert by[("ayrault_jean_marc", date(2012, 5, 1))].role == "prime_minister"
    assert by[("ayrault_jean_marc", date(2014, 3, 1))].satisfied == 26
    assert all(o.kind == "table" for o in obs)
    assert report.rows_without_person == 0 and not report.rows_out_of_tenure


def test_ifop_rows_outside_tenure_are_rejected():
    text = (FIX / "ifop_2014-03_tables.txt").read_text(encoding="utf-8")
    shifted = tuple(
        registry.Officeholder(h.person_id, h.name, h.role, date(2013, 1, 1), h.end)
        if h.person_id == "ayrault_jean_marc"
        else h
        for h in HOLDERS
    )
    report = ifop.TextReport()
    obs = ifop.parse_text(text, date(2014, 3, 1), shifted, report)
    assert not [o for o in obs if o.person_id == "ayrault_jean_marc" and o.month.year == 2012]
    assert any(r.startswith("ayrault_jean_marc 2012-05") for r in report.rows_out_of_tenure)


@pytest.mark.parametrize(
    "rest, expected",
    [
        (" ........ 61 33 6 100", (61, 33, 6)),
        (" ........ 56 44 - 100", (56, 44, 0)),
        (" ........ 1 55 28 17 100", (55, 28, 17)),  # renvoi de note parasite
        (" ........ 49 45 9 100", (49, 45, 9)),  # arrondi Ifop, Fillon mars 2010
        (" 2013", None),  # « Octobre 2013 2013 » des en-têtes de rappel
        (" ........ 40 20 10 100", None),
    ],
)
def test_ifop_row_values(rest, expected):
    assert ifop._row_values(rest) == expected


def test_ifop_slides_headline_per_person():
    text = (FIX / "ifop_2026-10_slides.txt").read_text(encoding="utf-8")
    report = ifop.TextReport()
    obs = ifop.parse_text(text, date(2026, 10, 1), HOLDERS, report)
    got = {(o.role, o.person_id, o.month, o.satisfied, o.kind) for o in obs}
    assert got == {
        ("president", "macron_emmanuel", date(2026, 10, 1), 17.0, "headline"),
        ("prime_minister", "lecornu_sebastien", date(2026, 10, 1), 25.0, "headline"),
    }
    assert not report.headlines_unreadable


def test_ifop_pptx_chart_cache():
    text = ifop.pptx_text((FIX / "ifop_2021-04_slide.pptx").read_bytes())
    assert "TOTAL Satisfaits TOTAL Mécontents\n37 60" in text
    obs = ifop.parse_text(text, date(2021, 4, 1), HOLDERS)
    assert [(o.person_id, o.satisfied) for o in obs] == [("macron_emmanuel", 37.0)]


def test_ifop_merge_prefers_latest_publication_and_reports_conflicts():
    def o(pub, value, kind="table"):
        return ifop.Observation("president", "macron_emmanuel", date(2020, 6, 1), value, kind, pub)

    chosen, conflicts = ifop.merge([o(date(2020, 6, 1), 38), o(date(2020, 10, 1), 38)])
    assert [c.publication for c in chosen] == [date(2020, 10, 1)] and not conflicts
    chosen, conflicts = ifop.merge([o(date(2020, 6, 1), 38), o(date(2020, 10, 1), 39)])
    assert chosen[0].satisfied == 39 and len(conflicts) == 1


def test_ifop_select_posts_keeps_last_before_each_new_officeholder():
    def post(pid, published, month):
        return ifop.Post(pid, published, "t", "l", month)

    posts = [
        post(1, date(2014, 2, 17), date(2014, 2, 1)),
        post(2, date(2014, 3, 17), date(2014, 3, 1)),  # dernier d'Ayrault (Valls : 31 mars)
        post(3, date(2014, 4, 14), date(2014, 4, 1)),
        post(4, date(2020, 12, 21), date(2020, 12, 1)),  # dernier de l'ère des tableaux
        post(5, date(2021, 1, 25), date(2021, 1, 1)),  # diapositives : tous gardés
        post(6, date(2021, 2, 22), date(2021, 2, 1)),
    ]
    holders = tuple(h for h in HOLDERS if h.person_id in ("ayrault_jean_marc", "valls_manuel"))
    kept = [p.post_id for p in ifop.select_posts(posts, holders)]
    assert kept == [2, 4, 5, 6]


def test_ifop_collection_mode_and_sponsor():
    assert (
        ifop.collection_mode("Mode de recueil Les interviews ont eu lieu par téléphone.") == "phone"
    )
    assert (
        ifop.collection_mode(
            "Les interviews ont été réalisées par questionnaire auto-\n"
            "administreenligneetpartelephonedu11au18fevrier2021."
        )
        == "mixed"
    )
    slides = (FIX / "ifop_2026-10_slides.txt").read_text(encoding="utf-8")
    assert ifop.sponsor(slides) == "L'Opinion"


# ---------------------------------------------------------------------------- table curated


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Arborescence data/ temporaire avec un fichier brut par source, tirés des fixtures."""
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "CURATED", tmp_path / "curated")
    day = date(2026, 10, 20)
    raw.write_raw("cpi_all", "xml", (FIX / "insee_cpi_all.xml").read_bytes(), "u", day)
    raw.write_raw(
        "unemployment_rate_ilo",
        "xml",
        (FIX / "insee_unemployment_rate_ilo.xml").read_bytes(),
        "u",
        day,
    )
    raw.write_raw("jobseekers_cat_a", "csv", (FIX / "dares_defm_cat_a.csv").read_bytes(), "u", day)
    raw.write_raw("fuel_price_sp95", "xlsx", (FIX / "mte_fuel_export.xlsx").read_bytes(), "u", day)
    doc = raw.macro_dir() / "ifop_popularity" / "documents" / "2026-10_1.pdf"
    raw.write_once(doc, b"%PDF fixture", "u")
    listing = {"posts": [{"month": "2026-10", "file": "documents/2026-10_1.pdf"}]}
    raw.write_raw("ifop_popularity", "json", json.dumps(listing).encode(), "u", day)
    slides = (FIX / "ifop_2026-10_slides.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(ifop, "document_text", lambda path: slides)
    return tmp_path


SAMPLE = [
    "cpi_all",
    "unemployment_rate_ilo",
    "jobseekers_cat_a",
    "fuel_price_sp95",
    "approval_president",
    "approval_prime_minister",
]


def test_parse_writes_schema_and_period_starts(tmp_data):
    df, report = macro.parse(SAMPLE)
    assert df.schema == macro.MACRO_SCHEMA
    out = pl.read_parquet(paths.CURATED / macro.OUTPUT_NAME)
    assert out.equals(df)
    freqs = dict(df.group_by("series_id").agg(pl.col("freq").first()).iter_rows())
    assert freqs["unemployment_rate_ilo"] == "Q" and freqs["fuel_price_sp95"] == "W"
    assert freqs["approval_prime_minister"] == "M"
    approval = df.filter(pl.col("source") == "ifop").select("series_id", "date", "value").rows()
    assert sorted(approval) == [
        ("approval_president", date(2026, 10, 1), 17.0),
        ("approval_prime_minister", date(2026, 10, 1), 25.0),
    ]
    assert df["retrieved_at"].dtype == pl.Datetime("us")
    assert report.rows["jobseekers_cat_a"] == (3, date(2025, 12, 1), date(2026, 2, 1))


def test_partial_parse_keeps_other_series(tmp_data):
    macro.parse(SAMPLE)
    df, _ = macro.parse(["cpi_all"])
    assert set(df["series_id"]) == set(SAMPLE)


def test_raw_files_are_never_overwritten(tmp_data):
    assert raw.write_raw("cpi_all", "xml", b"autre", "u", date(2026, 10, 20)) is None
    latest = raw.latest_raw("cpi_all", "xml")
    assert latest.path.read_bytes() == (FIX / "insee_cpi_all.xml").read_bytes()
    latest.path.write_bytes(b"corrompu")
    with pytest.raises(ValueError, match="sha256"):
        raw.latest_raw("cpi_all", "xml")


def test_validate_macro_rejects_bad_period_start():
    bad = pl.DataFrame(
        {
            "series_id": ["fuel_price_sp95"],
            "date": [date(2000, 1, 7)],
            "value": [1.0],
            "freq": ["W"],
            "source": ["mte"],
            "retrieved_at": [datetime(2026, 10, 20)],
        },
        schema=macro.MACRO_SCHEMA,
    )
    with pytest.raises(ValueError, match="début de période"):
        macro.validate_macro(bad)


def test_missing_periods():
    assert macro.missing_periods([date(2021, 1, 1), date(2021, 4, 1)], "M") == [
        date(2021, 2, 1),
        date(2021, 3, 1),
    ]
    assert macro.missing_periods([date(2021, 1, 1), date(2021, 7, 1)], "Q") == [date(2021, 4, 1)]


# ---------------------------------------------------------------------------- réseau


@pytest.mark.slow
def test_network_insee_series():
    with raw.client() as http:
        content = insee.fetch(http, REG.get("unemployment_rate_ilo"))
    freq, obs = insee.parse(content, "001688527")
    assert freq == "Q" and obs[0][0] <= date(2000, 1, 1)


@pytest.mark.slow
def test_network_dares_and_fuel(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    with raw.client() as http:
        written = dares.ingest(http, REG.get("jobseekers_cat_a"))
        sp95 = REG.get("fuel_price_sp95")
        content, _ = fuel.fetch_export(http, "sp95", date(2024, 1, 1), date(2024, 3, 1))
    obs = dares.parse(written.path.read_bytes(), "France")
    assert obs[0][0] <= date(2000, 1, 1)
    prices = fuel.parse(content, "sp95", sp95.params["sheet"])
    assert 7 <= len(prices) <= 10 and all(1.0 < v < 3.0 for _, v in prices)


@pytest.mark.slow
def test_network_ifop_listing():
    with raw.client() as http:
        posts = ifop.list_posts(http)
        selected = ifop.select_posts(posts, HOLDERS)
        url = ifop.document_url(http, selected[0])
    assert min(p.published for p in posts) <= date(2006, 1, 31)
    assert url.endswith((".pdf", ".pptx"))
