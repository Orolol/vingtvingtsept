"""Baromètre INA des JT sur un extrait figé du CSV data.gouv (octets d'origine : latin-1,
CRLF, sans en-tête) : les 12 premières lignes et 30 lignes de fin avril 2017."""

import hashlib
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from vvs import paths
from vvs.ingest.press import ina

EXTRACT = (Path(__file__).parent / "fixtures" / "press" / "ina_extract.csv").read_bytes()


def test_parse_extract():
    df = ina.parse_bytes(EXTRACT)
    assert df.schema == ina.INA_SCHEMA and df.height == 42
    first = df.filter(date=date(2000, 1, 2), channel="France 3", theme="Catastrophes").row(
        0, named=True
    )
    assert (first["n_subjects"], first["duration_s"]) == (2, 235)
    assert "Santé" in df["theme"].to_list()  # latin-1 décodé
    assert df.filter(pl.col("date").dt.year() == 2017).height == 30


@pytest.mark.parametrize(
    "line, match",
    [
        ("02/01/2000;France 3;matin;Sport;1;10\r\n", "colonne 3"),
        ("02/01/2000;France 3;;Météo;1;10\r\n", "rubriques inconnues"),
        ("02/01/2000;Canal+;;Sport;1;10\r\n", "chaînes inconnues"),
        ("02/01/2000;France 3;;Catastrophes;2;235\r\n", "en double"),
    ],
)
def test_format_drift_fails(line, match):
    with pytest.raises(ValueError, match=match):
        ina.parse_bytes(EXTRACT + line.encode("latin-1"))


def test_theme_shares_sum_to_one():
    shares = ina.theme_shares(ina.parse_bytes(EXTRACT), every="1w")
    sums = shares.group_by("date").agg(pl.col("share").sum())
    assert sums["share"].to_list() == pytest.approx([1.0] * sums.height)
    assert shares.group_by("date").len()["len"].unique().to_list() == [len(ina.RUBRICS)]


def datagouv_handler(sha1):
    url = "https://static.data.gouv.fr/resources/x/ina-extrait.csv"

    def handler(request):
        if request.url.host == "www.data.gouv.fr":
            return httpx.Response(
                200,
                json={
                    "resources": [
                        {
                            "format": "csv",
                            "url": url,
                            "title": "ina-extrait.csv",
                            "last_modified": "2024-10-15",
                            "checksum": {"type": "sha1", "value": sha1},
                        },
                        {"format": "pdf", "url": "https://x/doc.pdf", "title": "doc"},
                    ]
                },
            )
        assert str(request.url) == url
        return httpx.Response(200, content=EXTRACT)

    return handler


def test_ingest_writes_once_with_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW", tmp_path)
    http = httpx.Client(
        transport=httpx.MockTransport(datagouv_handler(hashlib.sha1(EXTRACT).hexdigest()))
    )
    path, status = ina.ingest(http)
    assert status == "écrit" and path == tmp_path / "press" / "ina" / "ina-extrait.csv"
    assert path.read_bytes() == EXTRACT
    assert ina.latest_raw() == path  # vérifie le sha256 de la méta
    _, status = ina.ingest(http)
    assert status.startswith("déjà présent")


def test_ingest_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW", tmp_path)
    http = httpx.Client(transport=httpx.MockTransport(datagouv_handler("0" * 40)))
    with pytest.raises(ValueError, match="sha1"):
        ina.ingest(http)
    assert not (tmp_path / "press" / "ina").exists()


@pytest.mark.slow
def test_live_dataset_has_one_csv():
    with httpx.Client(follow_redirects=True, timeout=60) as http:
        res = ina.find_resource(http)
    assert res.url.endswith(".csv") and res.checksum
