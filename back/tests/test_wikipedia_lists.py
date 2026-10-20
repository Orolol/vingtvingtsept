"""Adaptateurs Wikipédia des scrutins de liste et des législatives (HTML figés, gzip)."""

from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

from vvs.ingest.wikipedia import scrutins
from vvs.ingest.wikipedia.common import UnknownLabelError

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia"


@pytest.fixture(scope="module", params=sorted(scrutins.SCRUTINS))
def parsed(request):
    key = request.param
    with gzip.open(FIXTURES / f"{key}.html.gz", "rt", encoding="utf-8") as f:
        html = f.read()
    return (key, *scrutins.parse_with_report(key, html))


def shares(results: pl.DataFrame, poll_id: str) -> dict[str, float]:
    sub = results.filter(pl.col("poll_id") == poll_id)
    assert sub["hypothesis_id"].n_unique() == 1, sub["hypothesis_id"].unique()
    return dict(zip(sub["candidate_id"], sub["share"], strict=True))


# sondages, hypothèses, motifs des hypothèses écartées (révisions figées du 2026-10-20)
EXPECTED = {
    "euro2014": (54, 54, {}),
    "euro2019": (72, 88, {"candidat non nommé sous un en-tête générique": 18}),
    "euro2024": (119, 120, {"libellé ambigu": 5, "hypothèse en double aux parts différentes": 3,
                            "candidat non nommé sous un en-tête générique": 1}),
    "leg2022": (30, 32, {"candidat non nommé sous un en-tête générique": 8}),
    "leg2024": (23, 23, {"candidat non nommé sous un en-tête générique": 19}),
}


def test_counts(parsed):
    key, polls, results, report = parsed
    n_polls, n_hyps, reasons = EXPECTED[key]
    assert polls.height == n_polls
    assert results.select(pl.struct("poll_id", "hypothesis_id").n_unique()).item() == n_hyps
    assert dict(Counter(s.reason.split(" :")[0] for s in report.skipped_rows)) == reasons


def test_invariants(parsed):
    key, polls, results, _ = parsed
    assert polls.schema == scrutins.POLLS_LISTS_SCHEMA
    assert (polls["election"] == key).all()
    assert polls["poll_id"].str.starts_with(f"{key}-").all()
    assert polls["poll_id"].is_unique().all()
    assert set(results["poll_id"]) == set(polls["poll_id"])
    assert (results["round"] == 1).all()
    assert results.select("poll_id", "hypothesis_id", "candidate_id").is_unique().all()
    vote = scrutins.scrutin(key).vote
    assert (polls["fieldwork_end"] < vote).all() and (polls["fieldwork_start"] <= polls["fieldwork_end"]).all()
    sums = results.group_by("poll_id", "hypothesis_id").agg(pl.col("share").sum())
    assert sums["share"].is_between(95, 105).all()
    # tous les ids sont des listes du scrutin ou `other`
    from vvs.curate.lists import load_lists

    known = set(load_lists().for_election(key)) | {"other"}
    assert set(results["candidate_id"]) <= known


def test_leg2024_merged_cells():
    """Cellules fusionnées : en-tête commun (NFP), groupe (DVD + LR, RN + Ciotti)."""
    with gzip.open(FIXTURES / "leg2024.html.gz", "rt", encoding="utf-8") as f:
        polls, results, report = scrutins.parse_with_report("leg2024", f.read())
    # Ipsos 27-28 juin : NFP sur quatre colonnes, DVD et LR fusionnés, Ciotti et RN séparés
    assert shares(results, "leg2024-ipsos-20240628-1") == {
        "exg_2024": 1, "nfp_2024": 29, "dvg_2024": 1, "dvc_2024": 1.5, "ensemble_2024": 20,
        "lr_dvd_2024": 8, "uxd_2024": 4, "rn_2024": 32, "rec_2024": 1, "other": 2.5,
    }
    # Ifop 25-28 juin : « RN et alliés » sur les deux colonnes
    s = shares(results, "leg2024-ifop-20240628-1")
    assert s["rn_allies_2024"] == 36.5 and "rn_2024" not in s
    # LR et Ciotti fusionnés (avant l'alliance) : pas de groupe commun, hypothèse écartée
    reasons = [s.reason for s in report.skipped_rows]
    assert sum("LR/RN et alliés LR ( Ciotti )" in r for r in reasons) == 18
    assert polls["fieldwork_end"].min().isoformat() == "2024-06-17"


def test_euro2014_not_tested_and_group():
    with gzip.open(FIXTURES / "euro2014.html.gz", "rt", encoding="utf-8") as f:
        _, results, _ = scrutins.parse_with_report("euro2014", f.read())
    s = shares(results, "euro2014-ifop-20140523-1")
    assert "nc_2014" not in s  # « N.t. » : non testé, pas zéro
    assert s["udi_modem_2014"] == 9.5 and s["ps_prg_2014"] == 16  # cellules UDI|MoDem, PS|PRG
    assert s["fn_2014"] == 23.5 and s["other"] == 4
    # 2013 : MoDem et UDI testés séparément (composantes)
    s = shares(results, "euro2014-ifop-20130531-1")
    assert (s["modem_2014"], s["udi_2014"]) == (7, 6.5)


def test_euro2019_last_ifop_and_sponsor():
    with gzip.open(FIXTURES / "euro2019.html.gz", "rt", encoding="utf-8") as f:
        polls, results, _ = scrutins.parse_with_report("euro2019", f.read())
    s = shares(results, "euro2019-ifop-20190524-1")
    assert (s["rn_2019"], s["lrem_2019"], s["eelv_2019"], s["lr_2019"]) == (25.5, 23, 8, 13.5)
    p = polls.filter(pl.col("poll_id") == "euro2019-ifop-20190524-1").row(0, named=True)
    assert p["sponsor"] == "Fiducial" and p["sample_size"] == 2773
    elabe = shares(results, "euro2019-elabe-20190524-1")
    assert elabe["prc_2019"] == 0.25  # « <0,5 % »


def test_euro2024_heads_and_ambiguous():
    with gzip.open(FIXTURES / "euro2024.html.gz", "rt", encoding="utf-8") as f:
        _, results, report = scrutins.parse_with_report("euro2024", f.read())
    s = shares(results, "euro2024-ipsos-20240607-1")
    assert (s["rn_eu2024"], s["ensemble_eu2024"], s["ps_pp_eu2024"], s["lfi_eu2024"]) == (32, 15, 14.5, 9.5)
    # Ifop 7-8 février : tête de liste nommée ; Hayer -> la liste, les autres -> hypothèse
    sub = results.filter(pl.col("poll_id") == "euro2024-ifop-20240208-1")
    assert set(sub["candidate_id"]) >= {"ensemble_eu2024", "ensemble_alt_eu2024"}
    ambiguous = [s for s in report.skipped_rows if s.reason.startswith("libellé ambigu")]
    assert {s.reason.split("'")[1] for s in ambiguous} == {"Royal", "Toussaint", "Aubry"}


def test_unknown_list_label_fails(parsed):
    key, *_ = parsed
    aliases = scrutins.load_aliases(key)
    with pytest.raises(UnknownLabelError, match="lists.yaml"):
        aliases.candidate("Liste inventée")


def test_latest_raw_ignores_other_titles(tmp_path, monkeypatch):
    """data/raw/wikipedia/euro2019 garde l'article de l'élection téléchargé par erreur."""
    monkeypatch.setattr(scrutins, "RAW_DIR", tmp_path)
    d = tmp_path / "euro2019"
    d.mkdir()
    head = '<!-- vvs:wikipedia election=euro2019 revid=1 fetched_at=2026-10-20T16:1{m}:00+00:00 title="{t}" -->\n'
    (d / "2026-10-20T1611.html").write_text(head.format(m=1, t="Sondages sur les élections européennes de 2019"))
    (d / "2026-10-20T1612.html").write_text(head.format(m=2, t="Élections européennes de 2019 en France"))
    assert scrutins.latest_raw("euro2019").name == "2026-10-20T1611.html"
