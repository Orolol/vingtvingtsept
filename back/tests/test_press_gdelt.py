"""Client GDELT DOC 2.0 sur réponses figées.

`gdelt_timelinevolraw_le_pen_2017-01.json` est une vraie réponse de l'API (20 octobre 2026,
`"Marine Le Pen" sourcelang:french sourcecountry:FR`, janvier 2017) ; les fichiers
`*_synthetic.json` sont construits à la main au format de l'API, et
`gdelt_error_text.txt` reproduit une erreur de requête renvoyée en texte avec un statut 200.
"""

import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from vvs.ingest.press import gdelt
from vvs.ingest.press.cache import DiskCache
from vvs.ingest.press.outlets import load_outlets, load_queries

FIX = Path(__file__).parent / "fixtures" / "press"
VOLRAW = (FIX / "gdelt_timelinevolraw_le_pen_2017-01.json").read_bytes()
TONE = (FIX / "gdelt_timelinetone_synthetic.json").read_bytes()
ARTLIST = (FIX / "gdelt_artlist_synthetic.json").read_bytes()


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make_client(tmp_path, handler, **kw):
    t = FakeTime()
    c = gdelt.GdeltClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        cache=DiskCache("gdelt", root=tmp_path / "gdelt"),
        sleep=t.sleep,
        clock=t.clock,
        log=lambda _: None,
        **kw,
    )
    return c, t


@pytest.fixture(scope="module")
def queries():
    return load_queries()


def by_mode(request):
    mode = request.url.params["mode"]
    return httpx.Response(200, content={"timelinevolraw": VOLRAW, "timelinetone": TONE}[mode])


def test_parse_real_timeline():
    pts = gdelt.parse_timeline(json.loads(VOLRAW), "timelinevolraw")
    assert len(pts) == 32 and pts[0][0] == date(2017, 1, 1) and pts[-1][0] == date(2017, 2, 1)
    assert pts[4] == (date(2017, 1, 5), 3.0, 830028.0)


def test_parse_timeline_rejects_non_daily():
    weekly = {
        "timeline": [
            {
                "series": "s",
                "data": [{"date": f"201701{d:02d}T000000Z", "value": 1} for d in (1, 8, 15, 22)],
            }
        ]
    }
    with pytest.raises(gdelt.GdeltError, match="réduire window_months"):
        gdelt.parse_timeline(weekly, "timelinevolraw")
    hourly = {"timeline": [{"series": "s", "data": [{"date": "20170101T010000Z", "value": 1}]}]}
    with pytest.raises(gdelt.GdeltError, match="infra-journalier"):
        gdelt.parse_timeline(hourly, "timelinevolraw")
    assert gdelt.parse_timeline({}, "timelinetone") == []
    monthly = dict(json.loads(VOLRAW), query_details={"date_resolution": "month"})
    with pytest.raises(gdelt.GdeltError, match="résolution 'month'"):
        gdelt.parse_timeline(monthly, "timelinevolraw")


def test_windows():
    assert list(gdelt.windows(date(2017, 1, 1), date(2019, 3, 10))) == [
        (date(2017, 1, 1), date(2017, 12, 31)),
        (date(2018, 1, 1), date(2018, 12, 31)),
        (date(2019, 1, 1), date(2019, 3, 10)),
    ]
    assert list(gdelt.windows(date(2017, 2, 15), date(2017, 7, 1), months=3)) == [
        (date(2017, 2, 15), date(2017, 3, 31)),
        (date(2017, 4, 1), date(2017, 6, 30)),
        (date(2017, 7, 1), date(2017, 7, 1)),
    ]
    assert list(gdelt.windows(date(2017, 1, 1), date(2026, 10, 19), months=120)) == [
        (date(2017, 1, 1), date(2026, 10, 19))
    ]


def test_retry_on_429_then_cache(tmp_path, queries):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= 2:
            return httpx.Response(429, text="Please limit requests to one every 5 seconds")
        return httpx.Response(200, content=VOLRAW)

    c, t = make_client(tmp_path, handler)
    q = queries.candidate("le_pen_marine")
    got = c.timeline(q, "timelinevolraw", date(2017, 1, 1), date(2017, 2, 1), queries.gdelt_filters)
    assert not got.from_cache and len(calls) == 3 and c.n_429 == 2
    assert [s for s in t.sleeps if s >= 300] == list(gdelt.BACKOFF_S[:2])
    assert got.path.read_bytes() == VOLRAW
    assert calls[0].url.params["query"] == "Pen sourcelang:french sourcecountry:FR"
    assert calls[0].url.params["enddatetime"] == "20170201235959"
    # deuxième appel : cache, aucune requête
    again = c.timeline(
        q, "timelinevolraw", date(2017, 1, 1), date(2017, 2, 1), queries.gdelt_filters
    )
    assert again.from_cache and len(calls) == 3
    journal = (tmp_path / "gdelt" / "requests.jsonl").read_text().splitlines()
    assert [json.loads(line)["status"] for line in journal] == [429, 429, 200]


def test_throttle_spaces_requests(tmp_path, queries):
    c, t = make_client(tmp_path, by_mode)
    q = queries.candidate("le_pen_marine")
    c.timeline(q, "timelinevolraw", date(2017, 1, 1), date(2017, 2, 1), "")
    c.timeline(q, "timelinetone", date(2017, 1, 1), date(2017, 2, 1), "")
    assert t.sleeps == [pytest.approx(gdelt.MIN_INTERVAL_S)]
    assert gdelt.MIN_INTERVAL_S >= 5


def test_persistent_429_raises_without_caching(tmp_path, queries):
    c, _ = make_client(tmp_path, lambda r: httpx.Response(429), backoff=(1, 2))
    with pytest.raises(gdelt.GdeltRateLimited):
        c.timeline(
            queries.candidate("zemmour_eric"),
            "timelinetone",
            date(2017, 1, 1),
            date(2017, 2, 1),
            "",
        )
    assert not list((tmp_path / "gdelt").rglob("*.json"))
    stats = gdelt.ingest_timelines(
        queries,
        ["zemmour_eric"],
        start=date(2017, 1, 1),
        end=date(2018, 6, 1),
        client=c,
        log=lambda _: None,
    )
    assert stats.rate_limited and len(stats.failed) == 1  # arrêt au premier 429 persistant


def test_text_error_is_not_cached(tmp_path, queries):
    err = (FIX / "gdelt_error_text.txt").read_bytes()
    c, _ = make_client(tmp_path, lambda r: httpx.Response(200, content=err))
    with pytest.raises(gdelt.GdeltQueryError, match="too short"):
        c.timeline(
            queries.candidate("zemmour_eric"),
            "timelinetone",
            date(2017, 1, 1),
            date(2017, 2, 1),
            "",
        )
    assert not list((tmp_path / "gdelt").rglob("*.json"))


def test_ingest_and_parse_timelines(tmp_path, queries):
    c, _ = make_client(tmp_path, by_mode)
    stats = gdelt.ingest_timelines(
        queries,
        ["le_pen_marine"],
        start=date(2017, 1, 1),
        end=date(2017, 2, 1),
        client=c,
        log=lambda _: None,
    )
    assert (stats.fetched, stats.cached, stats.failed) == (2, 0, [])
    df = gdelt.parse_timelines(queries, cache=c.cache, write=False)
    assert df.schema == gdelt.TIMELINE_SCHEMA and df.height == 32
    row = df.filter(date=date(2017, 1, 5)).row(0, named=True)
    assert row["source"] == "gdelt" and row["candidate_id"] == "le_pen_marine"
    assert row["volume"] == 3 and row["total_monitored"] == 830028
    assert row["volume_share"] == pytest.approx(100 * 3 / 830028)
    assert row["tone"] == pytest.approx(-2.5397)
    assert df.filter(date=date(2017, 1, 20))["tone"].item() is None  # ton absent ce jour-là
    # le 6 janvier, GDELT donne un ton 0 sans aucun article : nul, pas neutre
    assert df.filter(date=date(2017, 1, 6))["tone"].item() is None
    # relance : tout vient du cache
    again = gdelt.ingest_timelines(
        queries,
        ["le_pen_marine"],
        start=date(2017, 1, 1),
        end=date(2017, 2, 1),
        client=c,
        log=lambda _: None,
    )
    assert (again.fetched, again.cached) == (0, 2)


def test_artlist_splits_saturated_windows(tmp_path, queries):
    one = json.loads(ARTLIST)["articles"][0]
    calls = []

    def handler(request):
        p = request.url.params
        calls.append((p["startdatetime"], p["enddatetime"]))
        n = 250 if len(calls) == 1 else 2
        return httpx.Response(
            200,
            json={"articles": [dict(one, url=f"{one['url']}?{len(calls)}-{i}") for i in range(n)]},
        )

    c, _ = make_client(tmp_path, handler)
    got = c.articles(
        queries.candidate("le_pen_marine"), datetime(2017, 1, 5), datetime(2017, 1, 6), ""
    )
    assert calls == [
        ("20170105000000", "20170105235959"),
        ("20170105000000", "20170105115959"),
        ("20170105120000", "20170105235959"),
    ]
    assert [len(f.data["articles"]) for f in got] == [2, 2]


def test_parse_article_items(tmp_path, queries):
    c, _ = make_client(tmp_path, lambda r: httpx.Response(200, content=ARTLIST))
    c.articles(queries.candidate("le_pen_marine"), datetime(2017, 1, 5), datetime(2017, 1, 6), "")
    c.articles(queries.candidate("bardella_jordan"), datetime(2017, 1, 5), datetime(2017, 1, 6), "")
    items = gdelt.parse_article_items(queries, load_outlets(), cache=c.cache)
    assert items.height == 4  # dédoublonné par URL
    assert items["outlet"].to_list() == ["le_monde", "afp", "rmc", None]
    assert items["query_ids"][0].to_list() == ["bardella_jordan", "le_pen_marine"]
    assert items["lang"].unique().to_list() == ["fr"]
    assert items["published_at"][0] == datetime(2017, 1, 5, 8, 15)
    assert items["item_id"].n_unique() == 4


def test_non_daily_response_is_not_cached(tmp_path, queries):
    monthly = dict(json.loads(VOLRAW), query_details={"date_resolution": "month"})
    c, _ = make_client(tmp_path, lambda r: httpx.Response(200, json=monthly))
    stats = gdelt.ingest_timelines(
        queries,
        ["zemmour_eric"],
        modes=["timelinevolraw"],
        start=date(2017, 1, 1),
        end=date(2026, 10, 19),
        window_months=120,
        client=c,
        log=lambda _: None,
    )
    assert len(stats.failed) == 1 and "résolution" in stats.failed[0][2]
    assert not list((tmp_path / "gdelt").rglob("*.json"))


def test_parse_ignores_responses_from_old_aliases(tmp_path, queries):
    import dataclasses

    c, _ = make_client(tmp_path, by_mode)
    gdelt.ingest_timelines(
        queries,
        ["le_pen_marine"],
        start=date(2017, 1, 1),
        end=date(2017, 2, 1),
        client=c,
        log=lambda _: None,
    )
    q = dataclasses.replace(queries.candidates["le_pen_marine"], gdelt_terms=("Marine Pen",))
    changed = dataclasses.replace(queries, candidates={**queries.candidates, "le_pen_marine": q})
    msgs = []
    df = gdelt.parse_timelines(changed, cache=c.cache, write=False, log=msgs.append)
    assert df.height == 0 and "2 réponse(s)" in msgs[0]
