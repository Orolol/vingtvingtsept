"""Chargement et encodage des sondages pour le modèle de premier tour (vvs.model.data)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from vvs.model.data import DEFAULT_SAMPLE_SIZE, encode, load_polls, weeks_before_vote
from vvs.schemas import hypothesis_id

D = dt.date
VOTE = D(2022, 4, 10)  # premier tour 2022 dans elections.yaml


def _poll(pid, pollster, start, end, size, **extra):
    return {
        "poll_id": pid,
        "election": 2022,
        "pollster": pollster,
        "sponsor": None,
        "fieldwork_start": start,
        "fieldwork_end": end,
        "publish_date": None,
        "sample_size": size,
        "mode": "unknown",
        "source": "wikipedia",
        "source_url": "",
        "ingested_at": dt.datetime(2026, 1, 1),
        **extra,
    }


def _results(pid, shares: dict[str, float], round_=1, **extra):
    hid = hypothesis_id(list(shares))
    return [
        {
            "poll_id": pid,
            "round": round_,
            "hypothesis_id": hid,
            "population": "unknown",
            "candidate_id": c,
            "share": s,
            "share_base": "expressed",
            "raw_share": None,
            **extra,
        }
        for c, s in shares.items()
    ]


FULL = {
    "macron_emmanuel": 30,
    "le_pen_marine": 20,
    "melenchon_jean_luc": 15,
    "zemmour_eric": 15,
    "pecresse_valerie": 20,
}
NO_ZEMMOUR = {
    "macron_emmanuel": 32,
    "le_pen_marine": 28,
    "melenchon_jean_luc": 18,
    "pecresse_valerie": 22,
}


@pytest.fixture
def wiki_dir(tmp_path):
    polls = [
        _poll("a", "ifop", D(2022, 1, 3), D(2022, 1, 5), 1500),
        _poll("b", "elabe", D(2022, 3, 1), D(2022, 3, 2), None),
        _poll("c", "ipsos", D(2022, 4, 5), D(2022, 4, 7), 2000),
    ]
    res = (
        _results("a", FULL)
        + _results("a", NO_ZEMMOUR)
        + _results("b", FULL | {"personne_inconnue": 0})  # hors registre
        + [
            {**r, "candidate_id": "undecided", "share": 25.0, "share_base": "all_respondents"}
            for r in _results("b", {"x": 0})
        ]
        + _results("c", FULL)
        + _results("c", {"macron_emmanuel": 55, "le_pen_marine": 45}, round_=2)
    )
    pl.DataFrame(polls).write_parquet(tmp_path / "polls_wikipedia_2022.parquet")
    pl.DataFrame(res).write_parquet(tmp_path / "poll_results_wikipedia_2022.parquet")
    return tmp_path


def test_load_polls_asof_and_reserved(wiki_dir):
    df = load_polls(2022, curated=wiki_dir)
    assert load_polls.last_source == "wikipedia"
    assert set(df["poll_id"]) == {"a", "b", "c"}
    assert "undecided" not in df["candidate_id"].to_list()
    assert df.filter(pl.col("poll_id") == "c").height == len(FULL)  # second tour exclu

    df = load_polls(2022, asof=D(2022, 4, 6), curated=wiki_dir)
    assert set(df["poll_id"]) == {"a", "b"}  # c finit le 7 avril
    df = load_polls(2022, asof="2022-04-07", curated=wiki_dir)
    assert set(df["poll_id"]) == {"a", "b", "c"}


def test_load_polls_prefers_merged_table(wiki_dir):
    """Table fusionnée : suspects exclus, doublons canoniques, vagues de rolling éclaircies."""
    polls = [
        _poll(
            "m1",
            "ifop",
            D(2022, 2, 28),
            D(2022, 3, 2),
            1500,
            is_rolling=True,
            canonical_poll_id="m1",
            sources=["wikipedia"],
        ),
        _poll(
            "m2",
            "ifop",
            D(2022, 3, 2),
            D(2022, 3, 4),
            1500,
            is_rolling=True,
            canonical_poll_id="m2",
            sources=["wikipedia"],
        ),  # recouvre m3 : écartée
        _poll(
            "m3",
            "ifop",
            D(2022, 3, 3),
            D(2022, 3, 5),
            1500,
            is_rolling=True,
            canonical_poll_id="m3",
            sources=["wikipedia"],
        ),
        _poll(
            "e1",
            "elabe",
            D(2022, 3, 1),
            D(2022, 3, 2),
            1500,
            is_rolling=False,
            canonical_poll_id="e1",
            sources=["commission", "wikipedia"],
        ),
        _poll(
            "e1dup",
            "elabe",
            D(2022, 3, 1),
            D(2022, 3, 2),
            1500,
            is_rolling=False,
            canonical_poll_id="e1",
            sources=["nsppolls"],
        ),
        _poll(
            "s",
            "bva",
            D(2022, 3, 1),
            D(2022, 3, 2),
            1000,
            is_rolling=False,
            canonical_poll_id="s",
            sources=["wikipedia"],
        ),
        # OpinionWay : chaque tableau publié sur les certains d'aller voter et les inscrits
        _poll("o", "opinionway", D(2022, 3, 1), D(2022, 3, 2), 1000, is_rolling=False,
              canonical_poll_id="o", sources=["commission"]),
        _poll("o2", "opinionway", D(2022, 2, 1), D(2022, 2, 2), 1000, is_rolling=False,
              canonical_poll_id="o2", sources=["commission"]),
    ]
    res = []
    registered = FULL | {"le_pen_marine": 17, "zemmour_eric": 18}
    for pid in ("o", "o2"):
        res += _results(pid, FULL, suspect=pid == "o2", population="certain_voters")
        res += _results(pid, registered, suspect=False, population="registered")
    for pid in ("m1", "m2", "m3", "e1", "e1dup"):
        res += _results(pid, FULL, suspect=False)
    res += _results("s", FULL, suspect=True) + _results("s", NO_ZEMMOUR, suspect=False)
    pl.DataFrame(polls).write_parquet(wiki_dir / "polls.parquet")
    pl.DataFrame(res).write_parquet(wiki_dir / "poll_results.parquet")

    df = load_polls(2022, curated=wiki_dir)
    assert load_polls.last_source == "curated"
    # m3 gardée (la plus récente), m2 recouvre m3, m1 finit avant le début de m3
    assert set(df["poll_id"]) == {"m1", "m3", "e1", "s", "o", "o2"}
    hyps = df.filter(pl.col("poll_id") == "s")["hypothesis_id"].unique().to_list()
    assert hyps == [hypothesis_id(list(NO_ZEMMOUR))]
    # une base par hypothèse : les certains d'aller voter, sauf s'ils sont suspects
    o = df.filter(pl.col("poll_id") == "o")
    assert o.height == len(FULL) and o["population"].unique().to_list() == ["certain_voters"]
    assert dict(o.select("candidate_id", "share").iter_rows())["le_pen_marine"] == 20
    o2 = df.filter(pl.col("poll_id") == "o2")
    assert o2.height == len(FULL) and o2["population"].unique().to_list() == ["registered"]


def test_encode_hypotheses(wiki_dir):
    df = load_polls(2022, curated=wiki_dir)
    data = encode(df, 2022, D(2022, 4, 8), n_weeks=20)
    assert data.n_polls == 3 and data.n_hyp == 4
    assert data.dropped_candidates == {"personne_inconnue": 1}
    # candidats : ceux des sondages et toute la liste officielle (référence)
    assert set(FULL) <= set(data.candidates)
    assert len(data.reference) == 12 and set(data.reference) <= set(data.candidates)
    # restriction : chaque ligne ne porte que les candidats de son hypothèse
    for i in range(data.n_hyp):
        got = {data.candidates[c] for c in data.cand_idx[i][data.mask[i]]}
        expected = set(NO_ZEMMOUR) if "zemmour_eric" not in data.hypothesis_id[i] else set(FULL)
        assert got == expected
        assert data.shares[i][data.mask[i]].sum() == pytest.approx(1.0)
        assert data.counts[i][~data.mask[i]].sum() == 0
    # effectif : sample_size, ou 1000 s'il manque ; poids 1/H par sondage
    by_poll = {p: i for i, p in enumerate(data.poll_id)}
    assert data.n_nominal[by_poll["b"]] == pytest.approx(DEFAULT_SAMPLE_SIZE, abs=3)
    assert data.n_nominal[by_poll["c"]] == pytest.approx(2000, abs=3)
    w = dict(zip(data.poll_id, data.weight))
    assert w["a"] == 0.5 and w["c"] == 1.0
    # temps : la dernière case de la grille est la semaine du vote
    assert data.n_weeks == 21
    i_c = by_poll["c"]
    assert data.week[i_c] == 20 - weeks_before_vote(D(2022, 4, 6), VOTE)
    assert data.asof_week == 20
    # blocs : Zemmour et Le Pen partagent l'extrême droite
    zb = data.cand_bloc[data.candidates.index("zemmour_eric")]
    assert zb == data.cand_bloc[data.candidates.index("le_pen_marine")]
    assert data.blocs[zb] == "far_right"


def test_encode_drops_polls_outside_grid(wiki_dir):
    df = load_polls(2022, curated=wiki_dir)
    data = encode(df, 2022, D(2022, 4, 8), n_weeks=10)  # le sondage de janvier sort
    assert set(data.poll_id) == {"b", "c"}
    two_weeks = encode(df, 2022, D(2022, 4, 8), n_weeks=10, step_days=14)
    assert two_weeks.n_weeks == 6 and two_weeks.week.max() == 5
