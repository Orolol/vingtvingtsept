"""Ingestion MieuxVoter/presidentielle2027 sur un extrait figé du dépôt (commit 7646466).

L'extrait (`fixtures/mieuxvoter/`) garde six notices telles quelles : Harris juillet 2024,
Ifop « Hexagone » (une notice, deux terrains, duel Le Pen–Philippe inversé dans la source),
Verian novembre 2025 (sans sous-échantillon), Ifop pour Têtu (sous-population exclue),
Cluster17 septembre 2026 (décimales, de Villepin), Ipsos BVA enquête électorale Cevipof.
"""

import dataclasses
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from vvs import schemas
from vvs.ingest import mieuxvoter as mv
from vvs.ingest.wikipedia.common import UnknownLabelError

FIXTURES = Path(__file__).parent / "fixtures" / "mieuxvoter"
SHA = "7646466c2e9484b3fc6560f33f945561426cfb88"


@pytest.fixture(scope="module")
def snap():
    return mv.latest_snapshot(FIXTURES)


@pytest.fixture(scope="module")
def notice_urls():
    return mv.load_notice_urls(FIXTURES / "commission_index.jsonl")


@pytest.fixture(scope="module")
def parsed(snap, notice_urls):
    return mv.parse(snap, notice_urls=notice_urls)


def shares(results, poll_id, rnd, hid):
    sub = results.filter(
        (pl.col("poll_id") == poll_id) & (pl.col("round") == rnd) & (pl.col("hypothesis_id") == hid)
    )
    return dict(zip(sub["candidate_id"], sub["share"], strict=True))


def test_counts_and_ids(parsed):
    polls, results, report = parsed
    assert polls["poll_id"].to_list() == [
        "2027-harris-20240708-1",
        "2027-ifop-20250418-1",  # même notice 9930 que la suivante, terrain distinct
        "2027-ifop-20250430-1",
        "2027-verian-20251120-1",
        "2027-cluster17-20260901-1",
        "2027-ipsos-20260909-1",
    ]
    assert results.height == 243
    assert report.counters["hypothèses tour 1"] == 22
    assert report.counters["hypothèses tour 2"] == 5
    assert set(polls["source"]) == {"mieuxvoter"} and set(polls["mode"]) == {"unknown"}
    assert set(results["share_base"]) == {"expressed"}
    assert polls["ingested_at"].dt.date().unique().to_list() == [date(2026, 10, 20)]
    # sommes des hypothèses entre 97 et 103 (règle de curation)
    sums = results.group_by("poll_id", "round", "hypothesis_id").agg(pl.col("share").sum())
    assert sums["share"].is_between(97, 103).all()
    # conformes aux schémas (parse valide déjà ; on vérifie que la validation est idempotente)
    assert schemas.validate_polls(polls).equals(polls)
    assert schemas.validate_poll_results(results).equals(results)


def test_poll_metadata(parsed):
    polls, _, _ = parsed
    p = polls.filter(pl.col("poll_id") == "2027-ipsos-20260909-1").row(0, named=True)
    assert (p["pollster"], p["fieldwork_start"], p["fieldwork_end"], p["sample_size"]) == (
        "ipsos",
        date(2026, 9, 3),
        date(2026, 9, 9),
        13060,
    )
    assert p["sponsor"] == "CEVIPOF, Le Monde, Fondation Jean Jaurès, Institut Montaigne"
    assert p["publish_date"] is None


def test_particle_hypothesis_and_decimals(parsed):
    _, results, _ = parsed
    ids = [
        "arthaud_nathalie", "de_villepin_dominique", "dupont_aignan_nicolas", "glucksmann_raphael",
        "le_pen_marine", "melenchon_jean_luc", "philippe_edouard", "retailleau_bruno",
        "roussel_fabien", "tondelier_marine", "zemmour_eric",
    ]  # fmt: skip
    hid = schemas.hypothesis_id(ids)
    s = shares(results, "2027-cluster17-20260901-1", 1, hid)
    assert set(s) == set(ids)
    assert (s["de_villepin_dominique"], s["le_pen_marine"], s["arthaud_nathalie"]) == (
        3.2,
        30.0,
        0.9,
    )


def test_second_round_kept_as_published(parsed):
    # La source inverse le duel Le Pen–Philippe de la notice Ifop 9930 (publié : Philippe 52) ;
    # le parseur ne corrige rien, l'écart relève de la curation inter-sources.
    _, results, _ = parsed
    pid = "2027-ifop-20250418-1"
    assert shares(results, pid, 2, "le_pen_marine+philippe_edouard") == {
        "le_pen_marine": 52.0,
        "philippe_edouard": 48.0,
    }
    assert results.filter((pl.col("poll_id") == pid) & (pl.col("round") == 2)).height == 8


@pytest.mark.parametrize(
    ("subs", "expected"),
    [
        (["Inscrits sur les listes électorales", "certains d'aller voter"], "certain_voters"),
        (
            ["Inscrits sur les listes électorales, certains d'aller voter, exprimés"],
            "certain_voters",
        ),
        (["Inscrits sur les listes électorales", "votants décidés (base brute)"], "certain_voters"),
        (["comptant aller voter et ayant exprimé une intention de vote"], "certain_voters"),
        (["Inscrits sur les listes électorales", "exprimé une intention de vote"], "registered"),
        (["représentatif de la population française"], "unknown"),
        ([], "unknown"),
    ],
)
def test_population_of(subs, expected):
    assert mv.population_of(subs) == expected


def test_population_in_results(parsed):
    _, results, report = parsed
    pop = dict(results.group_by("poll_id").agg(pl.col("population").unique()).iter_rows())
    assert pop["2027-ipsos-20260909-1"] == ["certain_voters"]
    assert pop["2027-harris-20240708-1"] == ["registered"]
    assert pop["2027-verian-20251120-1"] == ["unknown"]
    assert report.population == {"certain_voters": 6, "registered": 20, "unknown": 1}


def test_subpopulation_notice_excluded(parsed):
    polls, _, report = parsed
    assert polls.filter(pl.col("fieldwork_start") == date(2026, 2, 1)).height == 0
    assert report.excluded == [
        "20260201_0206_if_A : sous-population : électeurs LGBT+ (Ifop pour Têtu)"
    ]


def test_source_url(parsed):
    polls, _, _ = parsed
    urls = dict(polls.select("poll_id", "source_url").iter_rows())
    commission = "https://www.commission-des-sondages.fr/notices/medias/fichiers/add/"
    assert urls["2027-cluster17-20260901-1"] == commission + "2263"
    assert urls["2027-ipsos-20260909-1"] == commission + "2275"
    # notice hors de l'index Commission : lien vers le JSON au commit téléchargé
    assert urls["2027-harris-20240708-1"] == (
        f"https://github.com/MieuxVoter/presidentielle2027/blob/{SHA}/presidentielle2027.json"
    )


def _edited(snap, field, old, new):
    data = json.loads(snap.json_text)
    for p in data["polls"]:
        if field == "institut" and p["institut"] == old:
            p["institut"] = new
        for c in p["candidats"]:
            if field == "candidat" and c["candidat"] == old:
                c["candidat"] = new
    return dataclasses.replace(snap, json_text=json.dumps(data))


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [("candidat", "Nathalie Arthaud", "N. Arthaud"), ("institut", "Verian", "Kantar Public")],
)
def test_unknown_label_fails(snap, notice_urls, field, old, new):
    with pytest.raises(UnknownLabelError):
        mv.parse(_edited(snap, field, old, new), notice_urls=notice_urls)


def test_alias_inconsistent_with_json_names_fails(snap, notice_urls):
    aliases = mv.load_aliases()
    bad = dict(aliases.candidates, **{"dominique de villepin": "villepin_dominique"})
    with pytest.raises(ValueError, match="de_villepin_dominique"):
        mv.parse(snap, dataclasses.replace(aliases, candidates=bad), notice_urls)


def test_store_snapshot_never_rewrites(snap, tmp_path):
    path, written = mv.store_snapshot(snap, tmp_path)
    assert written and path.name == "2026-10-20.json"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "2026-10-20.json", "2026-10-20.meta.json", "2026-10-20.polls.csv",
    ]  # fmt: skip
    other = dataclasses.replace(snap, json_text="{}")
    assert mv.store_snapshot(other, tmp_path) == (path, False)
    assert path.read_text(encoding="utf-8") == snap.json_text
    assert mv.latest_snapshot(tmp_path).sha == SHA


def test_store_snapshot_refuses_partial(snap, tmp_path):
    (tmp_path / "2026-10-20.polls.csv").write_text("x")
    with pytest.raises(FileExistsError):
        mv.store_snapshot(snap, tmp_path)


def test_compare_with_wikipedia(parsed):
    """Wikipédia simulé à partir de l'extrait : décalages, écarts, sondages d'un seul côté."""
    polls, results, _ = parsed
    w_polls = polls.with_columns(
        source=pl.lit("wikipedia"),
        poll_id=pl.col("poll_id") + "-w",
        fieldwork_start=pl.when(pl.col("pollster") == "harris")
        .then(pl.col("fieldwork_start") + timedelta(days=1))  # un jour : apparié
        .when(pl.col("pollster") == "verian")
        .then(pl.col("fieldwork_start") - timedelta(days=4))  # quatre jours : non apparié
        .otherwise(pl.col("fieldwork_start")),
    ).filter(pl.col("pollster") != "ipsos")  # absent de Wikipédia
    cluster_hid = results.filter(pl.col("poll_id") == "2027-cluster17-20260901-1")["hypothesis_id"][
        0
    ]

    def bump(cid, delta):
        return (
            (pl.col("poll_id") == "2027-cluster17-20260901-1-w")
            & (pl.col("hypothesis_id") == cluster_hid)
            & (pl.col("candidate_id") == cid),
            delta,
        )

    w_results = results.with_columns(poll_id=pl.col("poll_id") + "-w")
    for cond, delta in (bump("le_pen_marine", 0.6), bump("zemmour_eric", 0.5)):
        w_results = w_results.with_columns(
            share=pl.when(cond).then(pl.col("share") + delta).otherwise(pl.col("share"))
        )
    w_results = pl.concat(  # abstention côté Wikipédia : ignorée
        [
            w_results,
            w_results.head(1).with_columns(candidate_id=pl.lit("abstention"), share=pl.lit(30.0)),
        ]
    ).filter(~((pl.col("poll_id") == "2027-ifop-20250430-1-w") & (pl.col("round") == 2)))
    extra = w_polls.head(1).with_columns(
        poll_id=pl.lit("2027-elabe-20240101-1-w"), pollster=pl.lit("elabe"),
        fieldwork_start=pl.lit(date(2023, 12, 30)), fieldwork_end=pl.lit(date(2024, 1, 1)),
    )  # fmt: skip
    w_polls = pl.concat([w_polls, extra])

    c = mv.compare_with_wikipedia(polls, results, w_polls, w_results)
    assert c.matches.height == 4
    assert c.matches.filter(pl.col("date_shift") == 1)["pollster"].to_list() == ["harris"]
    assert sorted(c.only_mieuxvoter["poll_id"]) == [
        "2027-ipsos-20260909-1",
        "2027-verian-20251120-1",
    ]
    only_w = dict(c.only_wikipedia.select("poll_id", "in_window").iter_rows())
    assert only_w == {"2027-verian-20251120-1-w": True, "2027-elabe-20240101-1-w": False}
    assert c.probable_matches.select(
        "poll_id_mv", "poll_id_wiki", "identical_hypotheses"
    ).rows() == [("2027-verian-20251120-1", "2027-verian-20251120-1-w", 1)]
    assert c.share_gaps.select("candidate_id", "share_mv", "share_wiki").rows() == [
        ("le_pen_marine", 30.0, 30.6)
    ]  # 0,5 point pile n'est pas un écart
    h = c.hypotheses.filter(pl.col("poll_id_mv") == "2027-ifop-20250430-1")
    assert h.select("round", "common", "only_mieuxvoter", "only_wikipedia").rows() == [
        (1, 2, 0, 0),
        (2, 0, 1, 0),
    ]
    assert "parts différant de plus de 0.5 point: 1" in c.summary()


@pytest.mark.slow
def test_network_snapshot_parses():
    snap = mv.fetch_snapshot()
    assert len(snap.sha) == 40
    polls, results, _ = mv.parse(snap)
    assert polls.height >= 50 and results.height >= 2000
