"""Titres Media Cloud pour la classification NLP (mediacloud_titles) et échantillon plafonné
(vvs.nlp.features.cap_sample), sur réponses story-list figées."""

from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl
import pytest

from vvs.ingest.press import mediacloud as mc
from vvs.ingest.press import mediacloud_titles as MT
from vvs.ingest.press.cache import DiskCache
from vvs.ingest.press.outlets import load_queries
from vvs.nlp.features import cap_sample

FIX = Path(__file__).parent / "fixtures" / "press"


@pytest.fixture
def queries():
    return load_queries()


def counts_frame(cid: str, cycle: int, per_day: int) -> pl.DataFrame:
    a, b = MT.WINDOWS[cycle]
    days = [a + timedelta(days=k) for k in range((b - a).days + 1)]
    return pl.DataFrame({"scope": "candidate", "id": cid, "election": cycle, "date": days,
                         "count": per_day})


def test_plan_estimates_pages_and_enforces_budget(queries):
    counts = pl.concat([counts_frame("melenchon_jean_luc", 2022, 100),
                        counts_frame("le_pen_marine", 2022, 10)])
    reqs = MT.plan(queries, counts, cycles=(2022,),
                   candidates={2022: ("melenchon_jean_luc", "le_pen_marine")})
    n_days = (MT.WINDOWS[2022][1] - MT.WINDOWS[2022][0]).days + 1
    assert [(r.candidate_id, r.est_stories) for r in reqs] == [
        ("melenchon_jean_luc", 100 * n_days), ("le_pen_marine", 10 * n_days)]
    assert reqs[0].q == "(Mélenchon OR Melenchon)"
    assert reqs[0].est_pages == -(-100 * n_days // 1000)
    with pytest.raises(ValueError, match="budget"):
        MT.plan(queries, counts, cycles=(2022,),
                candidates={2022: ("melenchon_jean_luc", "le_pen_marine")}, max_requests=5)
    # échantillon quotidien : le plus gros candidat passe en une page par jour
    counts = pl.concat([counts_frame("melenchon_jean_luc", 2022, 2000),
                        counts_frame("le_pen_marine", 2022, 10)])
    reqs = MT.plan(queries, counts, cycles=(2022,),
                   candidates={2022: ("melenchon_jean_luc", "le_pen_marine")},
                   max_requests=n_days + 5, sample_per_day=50)
    daily = [r for r in reqs if r.candidate_id == "melenchon_jean_luc"]
    assert len(daily) == n_days and all(r.max_pages == 1 and r.page_size == 50 for r in daily)
    assert sum(r.est_pages for r in reqs) <= n_days + 5


def handler(calls):
    def h(request):
        calls.append(request)
        page = 2 if request.url.params.get("pagination_token") == "tok2" else 1
        return httpx.Response(
            200, content=(FIX / f"mediacloud_titles_page{page}_synthetic.json").read_bytes())
    return h


def test_collect_resumes_from_cache_and_parses(tmp_path, queries):
    calls = []
    cache = DiskCache(MT.SOURCE, root=tmp_path / "mct")

    def client():
        return mc.MediaCloudClient(
            key="cle-de-test", cache=cache, sleep=lambda s: None, log=lambda _: None,
            http=httpx.Client(transport=httpx.MockTransport(handler(calls))))

    reqs = MT.plan(queries, None, cycles=(2022,), candidates={2022: ("melenchon_jean_luc",)})
    stats = MT.collect(reqs, client(), log=lambda _: None)
    assert stats["stories"] == 6 and len(calls) == 2
    assert calls[0].url.params["page_size"] == "1000"
    MT.collect(reqs, client(), log=lambda _: None)  # relance : tout vient du cache
    assert len(calls) == 2
    df = MT.parse_titles(queries, cache=cache, write=False, log=lambda _: None)
    assert df.columns == list(MT.SCHEMA)
    # s1 en double sur deux pages, s5 sans titre : écartés
    assert df["story_id"].to_list() == ["s1", "s2", "s3", "s4"]
    assert df["cycle"].unique().to_list() == [2022]
    assert df["publish_date"][1] == date(2022, 1, 11)
    # alias sans accent et mot entier (« mélenchonistes » ne compte pas)
    assert df["in_title"].to_list() == [True, True, False, False]


def test_title_names_add_surname_except_ambiguous(queries):
    names = MT.title_names(queries)
    assert "Hamon" in names["hamon_benoit"] and "Benoît Hamon" in names["hamon_benoit"]
    assert names["philippe_edouard"] == ("Édouard Philippe", "Edouard Philippe")
    p = MT.alias_pattern(names["le_pen_marine"])
    assert p.search(MT._norm("Le Pen promet un référendum"))


def test_alias_pattern_hyphen_and_case():
    p = MT.alias_pattern(["Dupont-Aignan", "Dupont Aignan"])
    assert p.search(MT._norm("NICOLAS DUPONT-AIGNAN SE PRÉSENTE"))
    assert p.search(MT._norm("Dupont Aignan rallie"))
    assert not p.search(MT._norm("Dupontaignan"))


def test_cap_sample_spreads_over_days():
    rows = []
    week = date(2022, 1, 10)  # lundi
    for d in range(7):
        n = 100 if d == 0 else 5  # un lundi chargé, des jours creux
        rows += [{"story_id": f"{d}-{k}", "candidate_id": "c", "cycle": 2022,
                  "publish_date": week + timedelta(days=d), "in_title": True}
                 for k in range(n)]
    rows.append({"story_id": "x", "candidate_id": "c", "cycle": 2022,
                 "publish_date": week, "in_title": False})
    s = cap_sample(pl.DataFrame(rows), cap=40)
    assert s.height == 40 and "x" not in s["story_id"].to_list()
    by_day = dict(s.group_by("publish_date").len().iter_rows())
    # les six jours creux sont pris en entier, le lundi complète
    assert by_day[week] == 10 and all(by_day[week + timedelta(days=d)] == 5 for d in range(1, 7))
    assert cap_sample(pl.DataFrame(rows), cap=40).equals(s)  # déterministe
