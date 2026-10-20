"""Base de Jennings et Wlezien (vvs.ingest.jennings_wlezien).

Tests rapides : extraits figés dans tests/fixtures/jennings_wlezien/.
- raw_extract.csv : lignes recopiées telles quelles du fichier Stata
  LONG_MI_NATURE_20180111.dta (France 2017 complète, France 1965 T1 et 1969 T2, deux lignes
  de grille sans sondage, États-Unis 2016 présidentielle et législative, Finlande 2011,
  Hongrie 2010, Chili 2009-2010 dernier jour de chaque tour) ;
- codebook_extract.txt : lignes du texte extrait du codebook PDF par pdfplumber.

Tests lents (`-m slow`) : téléchargement et parsing de la vraie base.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from vvs import paths
from vvs.ingest import jennings_wlezien as jw
from vvs.registry import default_registry

FIXTURES = Path(__file__).parent / "fixtures" / "jennings_wlezien"


@pytest.fixture(scope="module")
def codebook() -> pl.DataFrame:
    return jw.parse_codebook((FIXTURES / "codebook_extract.txt").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw() -> pl.DataFrame:
    return pl.read_csv(FIXTURES / "raw_extract.csv", try_parse_dates=True)


@pytest.fixture(scope="module")
def curated(raw, codebook) -> tuple[pl.DataFrame, jw.ParseReport]:
    return jw.curate(raw, codebook, default_registry())


def _label(cb: pl.DataFrame, **key) -> str | None:
    rows = cb.filter(*[pl.col(k) == v for k, v in key.items()])
    assert rows.height == 1, rows
    return rows["label"][0]


# --- Codebook ------------------------------------------------------------------------------


def test_codebook_sections(codebook):
    assert codebook.schema == jw.CODEBOOK_SCHEMA
    leg = codebook.filter(pl.col("election_type") == "legislative")
    assert set(leg["country"]) == {"Australia", "Finland", "Germany", "Hungary", "United States"}
    assert leg["cycle_year"].is_null().all() and leg["round"].is_null().all()
    # la liste « two-party preference » n'écrase pas celle de l'Australie
    assert leg.filter(pl.col("country") == "Australia").height == 6
    assert _label(codebook, country="France", cycle_year=2017, round=1, party_id=13) == (
        "En Marche! Macron"
    )
    # tirets de remplissage retirés ; id sans nom -> null
    assert _label(codebook, country="France", cycle_year=2017, round=1, party_id=4) == "UDF"
    assert _label(codebook, country="France", cycle_year=1969, round=2, party_id=2) is None


def test_codebook_layout_rules(codebook):
    # libellé coupé sur deux lignes
    assert _label(codebook, country="Germany", party_id=6) == (
        "Alternative for Germany [Alternative für Deutschland] (AfD)"
    )
    # en-tête « Cyprus: 2013 (Round 1) » répété : la seconde occurrence est le tour 2
    assert _label(codebook, country="Cyprus", cycle_year=2013, round=1, party_id=3) == (
        "EDEK Lillikas"
    )
    assert _label(codebook, country="Cyprus", cycle_year=2013, round=2, party_id=3) == "EDEK"
    # le codebook du fichier secondaire, après le titre de fin, n'est pas lu
    assert _label(codebook, country="United States", cycle_year=2016, party_id=2) == (
        "Republican Party Donald Trump"
    )


def test_codebook_rejects_repeated_heading():
    text = f"{jw.PRESIDENTIAL_HEADER}\nPeru: 2006\n1 APRA García\nPeru: 2006\n1 APRA García"
    with pytest.raises(ValueError, match="en-tête répété"):
        jw.parse_codebook(text)


# --- Curation ------------------------------------------------------------------------------


def test_curated_schema_and_ids(curated, raw):
    df, report = curated
    assert df.schema == jw.JW_POLLS_SCHEMA
    assert df.height == raw.filter(pl.col("poll_").is_not_null()).height
    ids = set(df["election_id"])
    assert {"FR-2017-04-23", "FR-2017-05-07", "US-2016-11-08", "US-2016-11-08-L"} <= ids
    assert report.elections == len(ids) == 10
    assert report.countries == 5
    # pays de la base sans libellé au codebook : US 2016 n'a que les ids 1 et 2
    assert report.unlabelled == []


def test_france_candidates(curated):
    df, report = curated
    fr17 = df.filter(pl.col("election_id") == "FR-2017-04-23")
    assert dict(fr17.select("party_id", "candidate_id").unique().iter_rows()) == {
        1: "fillon_francois",
        2: "hamon_benoit",
        3: "le_pen_marine",
        5: "melenchon_jean_luc",
        13: "macron_emmanuel",
    }
    # 1965 : libellé conservé, candidats hors registre
    fr65 = df.filter(pl.col("election_id") == "FR-1965-12-05")
    assert fr65["candidate_id"].is_null().all()
    assert "FGDS Mitterand" in set(fr65["label"])
    # hors France : pas de candidate_id
    assert df.filter(pl.col("country") != "France")["candidate_id"].is_null().all()
    assert ("FR-2017-05-07", 2, 156, "13=macron_emmanuel, 3=le_pen_marine") in report.france


def test_france_unknown_label_fails(raw, codebook):
    # id 7 en 2017 = « LO Sarkozy », erreur de recopie du codebook absente des alias
    bad = raw.with_columns(
        pl.when((pl.col("elecdate") == date(2017, 4, 23)) & (pl.col("partyid") == 2))
        .then(7)
        .otherwise(pl.col("partyid"))
        .alias("partyid")
    )
    with pytest.raises(jw.UnknownLabelError, match="LO Sarkozy"):
        jw.curate(bad, codebook)


def test_france_checked_against_registry(raw, codebook):
    bad = raw.with_columns(
        pl.when((pl.col("elecdate") == date(2017, 5, 7)) & (pl.col("partyid") == 13))
        .then(65.0)
        .otherwise(pl.col("vote_"))
        .alias("vote_")
    )
    jw.curate(bad, codebook)  # sans registre : pas de contrôle
    with pytest.raises(jw.ConsistencyError, match="macron_emmanuel 65.00 ≠ registre 66.10"):
        jw.curate(bad, codebook, default_registry())


def test_france_aliases_in_registry():
    candidates = default_registry().candidates
    assert {c for c in jw.FRANCE_ALIASES.values() if c} <= set(candidates)


def test_days_before_rules(curated):
    df, report = curated
    # Finlande 2011 : dates recalculées depuis daysbeforeED
    fi = df.filter(pl.col("election_id") == "FI-2011-04-17-L")
    assert (fi["days_before"] == fi["days_before_jw"]).all()
    assert fi["poll_date"].unique().to_list() == [date(2011, 4, 15)]
    # Hongrie 2010 : les dates font foi, l'écart est signalé
    hu = df.filter(pl.col("election_id") == "HU-2010-04-11-L")
    assert hu["days_before"].unique().to_list() == [13]
    assert report.days_mismatch == {"HU-2010-04-11-L": 12}
    assert (df["days_before"] == (df["election_date"] - df["poll_date"]).dt.total_days()).all()


def test_flags_and_totals(curated):
    df, report = curated
    fi = df.filter(pl.col("election_id") == "FI-2011-04-17-L").sort("party_id")
    # gov_ = 7 pour la Green League : mis à null
    assert report.bad_flags == 1
    assert fi.filter(pl.col("party_id") == 5)["gov_party"].to_list() == [None]
    assert fi.filter(pl.col("party_id") == 1)["gov_party"].to_list() == [True]
    fr69 = df.filter(pl.col("election_id") == "FR-1969-06-15")
    # 1969 : les sondages laissent 24 points d'indécis, le résultat somme à 100
    assert fr69["poll_total"].to_list() == [76.0, 76.0]
    assert fr69["result_total"].to_list() == pytest.approx([100.0, 100.0])
    # taille d'échantillon imputée (1000 par sondage) en 1965, réelle en 2017
    by_id = dict(df.group_by("election_id").agg(pl.col("sample_size_is_default").all()).iter_rows())
    assert by_id["FR-1965-12-05"] and not by_id["FR-2017-04-23"]


def test_second_round_label_uses_first_round_year(curated):
    df, _ = curated
    cl2 = df.filter(pl.col("election_id") == "CL-2010-01-17")
    assert cl2["cycle_year"].unique().to_list() == [2009]
    assert set(cl2["label"]) == {"RN/CFC Piñera", "PDC/CPD Frei"}


# --- Erreur par horizon -------------------------------------------------------------------


def _synthetic() -> pl.DataFrame:
    """Deux élections, deux partis ; erreurs connues."""
    rows = []
    for eid, results, polls in [
        # (days_before, part A, part B)
        ("XX-2000-01-01", (50.0, 30.0), [(200, 40.0, 30.0), (95, 52.0, 30.0), (10, 51.0, 29.0)]),
        ("XX-2004-01-01", (40.0, 40.0), [(40, 44.0, 40.0), (7, 42.0, 34.0)]),
    ]:
        for days, a, b in polls:
            for party_id, share, result in [(1, a, results[0]), (2, b, results[1])]:
                rows.append(
                    {
                        "election_id": eid,
                        "country": "Nowhere",
                        "country_iso2": "XX",
                        "election_type": "legislative",
                        "round": 1,
                        "election_date": date.fromisoformat(eid[3:]),
                        "party_id": party_id,
                        "label": None,
                        "candidate_id": None,
                        "poll_date": date(1999, 1, 1),
                        "days_before": days,
                        "n_polls": 1,
                        "poll_share": share,
                        "result_share": result,
                        "poll_total": a + b,
                        "result_total": sum(results),
                    }
                )
    return pl.DataFrame(rows)


def test_horizon_errors_takes_last_poll_within_window():
    err = jw.horizon_errors(_synthetic(), horizons=(180, 90, 30, 7), max_staleness=30)
    got = {(h, e[3:7], p): x for h, e, p, x in err.select("horizon", "election_id", "party_id",
                                                         "error").iter_rows()}  # fmt: skip
    # h=180 : le sondage à 200 jours est dans [180, 210] ; h=90 : celui à 95 jours
    assert got[(180, "2000", 1)] == pytest.approx(-10.0)
    assert got[(90, "2000", 1)] == pytest.approx(2.0)
    # h=30 : 2000 n'a rien dans [30, 60] (le sondage à 95 jours est trop ancien)
    assert (30, "2000", 1) not in got
    assert got[(30, "2004", 1)] == pytest.approx(4.0)
    assert got[(7, "2000", 2)] == pytest.approx(-1.0)
    assert got[(7, "2004", 2)] == pytest.approx(-6.0)


def test_error_summary_components():
    s = jw.error_summary(horizons=(7,), df=_synthetic()).row(0, named=True)
    # erreurs à 7 jours : 2000 -> (+1, -1), 2004 -> (+2, -6)
    assert (s["n_elections"], s["n_obs"]) == (2, 4)
    assert s["mean"] == pytest.approx(-1.0)
    assert s["mae"] == pytest.approx(2.5)
    # moyennes par élection 0 et -2 -> écart type sqrt(2)
    assert s["sd_election_mean"] == pytest.approx(2**0.5)
    # normalisé : 2004 ramené au total du résultat (80 / 76), moyenne par élection nulle
    n = jw.error_summary(horizons=(7,), normalize=True, df=_synthetic()).row(0, named=True)
    assert n["sd_election_mean"] == pytest.approx(0.0, abs=1e-9)
    # leave-one-election-out
    e = jw.error_summary(horizons=(7,), exclude=["XX-2004-01-01"], df=_synthetic())
    assert e["n_elections"].to_list() == [1]


def test_error_summary_filters(curated):
    df, _ = curated
    fr = jw.error_summary(country="FR", round=2, horizons=(7, 500), df=df)
    assert fr["horizon"].to_list() == [7, 500]
    assert fr["n_elections"].to_list() == [2, None]  # 1969 et 2017 ; rien à 500 jours
    assert jw.error_summary(country="France", round=2, horizons=(7,), df=df).equals(fr.head(1))


# --- Ingestion -----------------------------------------------------------------------------


def test_ingest_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW", tmp_path)
    monkeypatch.setattr(jw, "_download", lambda *a, **k: pytest.fail("pas de téléchargement"))
    d = tmp_path / jw.RAW_DIRNAME
    d.mkdir()
    (d / "a.dta").write_bytes(b"contenu")
    digest = jw.sha256_of(d / "a.dta")
    ok = (jw.SourceFile(name="a.dta", role="main", url="https://x", sha256=digest),)
    assert jw.ingest(ok) == [("a.dta", "present", digest)]
    bad = (jw.SourceFile(name="a.dta", role="main", url="https://x", sha256="0" * 64),)
    with pytest.raises(jw.ConsistencyError):
        jw.ingest(bad)


def test_source_registry_complete():
    dataset, files = jw.load_source()
    assert dataset["license"] == "CC0-1.0"
    assert {f.role for f in files} == {"main", "codebook", "erratum"}
    assert all(f.sha256 and len(f.sha256) == 64 for f in files)


# --- Données réelles -----------------------------------------------------------------------


@pytest.mark.slow
def test_real_base():
    jw.ingest()
    out, df, report = jw.parse()
    assert out.exists()
    assert (report.rows, report.elections, report.countries) == (88089, 337, 45)
    fr = df.filter(pl.col("country") == "France")
    assert fr["election_id"].n_unique() == 19
    assert fr.filter(pl.col("election_date").dt.year() >= 2002)["candidate_id"].is_not_null().all()
    s = jw.error_summary(df=df)
    assert s["n_elections"].to_list() == sorted(s["n_elections"].to_list())
