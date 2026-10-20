"""Signaux d'attention (vvs.ingest.attention) et panel hebdomadaire par candidat.

Réponses figées dans fixtures/attention/ :

- `pageviews_macron_2016-11-14.json` : vraie réponse Pageviews (Emmanuel_Macron, 14 au 20
  novembre 2016, relevée le 20 octobre 2026) ; `pageviews_404.json` : vraie réponse 404
  (Marine_Tondelier en juillet 2015, avant la création de l'article) ;
- `castex_redirects.json`, `castex_moves.json` : vraies réponses MediaWiki (redirections vers
  « Jean Castex », journal de renommage de « Jean Castex (homme politique) ») ;
- `trends_explore_*.txt`, `trends_multiline_*.txt` : vraies réponses Trends (mots-clés
  « Emmanuel Macron » et « François Fillon », France, 2016), préfixe anti-JSON compris.

Les tests réseau sont marqués `slow`.
"""

import itertools
import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest

from vvs.ingest.attention import RawCache, load_registry, with_cycle_share
from vvs.ingest.attention import google_trends as gt
from vvs.ingest.attention import wikipedia_pageviews as wp
from vvs.studies import panel as P

FIX = Path(__file__).parent / "fixtures" / "attention"
PV_MACRON = (FIX / "pageviews_macron_2016-11-14.json").read_bytes()
PV_404 = (FIX / "pageviews_404.json").read_bytes()
EXPLORE = (FIX / "trends_explore_macron_fillon_2016.txt").read_text(encoding="utf-8")
MULTILINE = (FIX / "trends_multiline_macron_fillon_2016.txt").read_text(encoding="utf-8")


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


@pytest.fixture(scope="module")
def reg():
    return load_registry()


# ---------------------------------------------------------------------------- registre


def test_registry_cycles(reg):
    assert {2007, 2012, 2017, 2022, 2027} <= set(reg.cycles)
    assert "macron_emmanuel" in reg.members(2017) and "macron_emmanuel" not in reg.members(2027)
    for e in (2017, 2022):
        officials = {"le_pen_marine", "melenchon_jean_luc"}
        assert officials <= set(reg.members(e))
    assert reg.cycle_of(date(2017, 4, 23)) == 2017
    assert reg.cycle_of(date(2017, 4, 24)) == 2022
    assert reg.cycle_of(date(2026, 10, 20)) == 2027
    assert reg.candidates[reg.trends_pivot].trends_term


def test_registry_rejects_unknown_ids(tmp_path, reg):
    bad = tmp_path / "a.yaml"
    bad.write_text(
        "wikipedia: {start: 2015-07-01}\n"
        "trends: {start: 2004-01-01, pivot: x_y}\n"
        "candidates:\n  x_y: {name: X Y, wikipedia: {title: X}, trends: {keyword: X}}\n"
        "cycles:\n  2017: [x_y, z_w]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent de candidates.yaml"):
        load_registry(bad)


# ---------------------------------------------------------------------------- Wikipédia


def test_former_titles_follow_move_chains():
    moves = {
        "A (homme politique)": [("A (homme politique)", "A", "2020")],
        "A ancien": [("A ancien", "A (homme politique)", "2015")],
        "Faute de frappe": [],
        "Autre": [("Autre", "Ailleurs", "2019")],
    }
    assert wp.former_titles("A", moves) == ["A (homme politique)", "A ancien"]


def test_titles_from_real_snapshot(tmp_path, reg):
    d = tmp_path / "castex_jean"
    d.mkdir()
    (d / "redirects.json").write_bytes((FIX / "castex_redirects.json").read_bytes())
    (d / f"moves_{wp.title_key('Jean Castex (homme politique)')}.json").write_bytes(
        (FIX / "castex_moves.json").read_bytes())
    q = reg.candidates["castex_jean"]
    assert wp.titles_from_snapshot(q, d) == ["Jean Castex", "Jean Castex (homme politique)"]


def test_parse_pageviews_real_and_404():
    rows = wp.parse_pageviews(PV_MACRON, 200)
    assert rows[0][0] == date(2016, 11, 14) and len(rows) == 7
    assert all(v > 0 for _, v in rows)
    assert wp.parse_pageviews(PV_404, 404) == []


def make_wiki_client(tmp_path, handler, **kw):
    t = FakeTime()
    c = wp.WikiClient(http=httpx.Client(transport=httpx.MockTransport(handler)),
                      cache=RawCache("wikipedia", root=tmp_path / "wikipedia"),
                      sleep=t.sleep, clock=t.clock, log=lambda _: None, **kw)
    return c, t


def test_wiki_client_caches_retries_and_keeps_404(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "Emmanuel_Macron" in str(request.url):
            if len([c for c in calls if "Emmanuel_Macron" in c]) == 1:
                return httpx.Response(429)
            return httpx.Response(200, content=PV_MACRON)
        return httpx.Response(404, content=PV_404)

    c, t = make_wiki_client(tmp_path, handler)
    p, from_cache = wp.fetch_pageviews(c, "Emmanuel Macron", date(2016, 11, 14),
                                       date(2016, 11, 20))
    assert not from_cache and t.sleeps[0] == wp.BACKOFF_S[0]
    assert p.read_bytes() == PV_MACRON
    _, from_cache = wp.fetch_pageviews(c, "Emmanuel Macron", date(2016, 11, 14),
                                       date(2016, 11, 20))
    assert from_cache and len(calls) == 2
    wp.fetch_pageviews(c, "Marine Tondelier", date(2015, 7, 1), date(2015, 7, 10))
    assert wp.covered_end(c.cache, "Marine Tondelier") == date(2015, 7, 10)
    with pytest.raises(FileExistsError):
        c.cache.put(p, b"{}", {})


def test_wiki_client_raises_on_other_status(tmp_path):
    c, _ = make_wiki_client(tmp_path, lambda r: httpx.Response(403, content=b"no"))
    with pytest.raises(wp.WikimediaError, match="403"):
        wp.fetch_pageviews(c, "X", date(2020, 1, 1), date(2020, 1, 2))


def test_ingest_is_incremental_and_parse_sums_titles(tmp_path, reg):
    requested = []

    def handler(request):
        url = str(request.url)
        if "w/api.php" in url:
            if request.url.params.get("prop") == "redirects":
                return httpx.Response(200, content=(FIX / "castex_redirects.json").read_bytes())
            return httpx.Response(200, content=(FIX / "castex_moves.json").read_bytes())
        requested.append(url)
        start, end = url.rsplit("/", 2)[-2:]
        d0 = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
        d1 = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
        art = url.split("/user/")[1].split("/")[0]
        items = [{"article": art, "timestamp": (d0 + timedelta(days=i)).strftime("%Y%m%d00"),
                  "views": 10 if "homme" in art else 100}
                 for i in range((d1 - d0).days + 1) if i % 3 != 2]  # un jour sur 3 omis
        return httpx.Response(200, json={"items": items})

    c, _ = make_wiki_client(tmp_path, handler)
    small = type(reg)(candidates={"castex_jean": reg.candidates["castex_jean"]}, cycles={},
                      first_rounds=reg.first_rounds, trends_pivot=reg.trends_pivot,
                      trends_start=reg.trends_start, wikipedia_start=date(2020, 1, 1))
    s1 = wp.ingest(small, end=date(2020, 1, 10), client=c, snapshot=date(2026, 10, 20),
                   log=lambda _: None)
    assert s1.downloaded == 2 and s1.former_titles == {"castex_jean": [
        "Jean Castex (homme politique)"]}
    s2 = wp.ingest(small, end=date(2020, 1, 15), client=c, snapshot=date(2026, 10, 20),
                   log=lambda _: None)
    assert s2.downloaded == 2 and all("20200111" in u for u in requested[-2:])
    df, notes = wp.parse_cached(small, c.cache)
    assert df["date"].to_list() == [date(2020, 1, d) for d in range(1, 16)]
    v = dict(zip(df["date"], df["value"]))
    assert v[date(2020, 1, 1)] == 110 and v[date(2020, 1, 3)] == 0  # jour omis = 0
    assert not df["suspect"].any() and notes == []


def test_flag_plateaus_marks_flat_runs_only():
    base = np.full(40, 300.0)
    plateau = base.copy()
    plateau[30:38] = [160_000, 170_000, 165_000, 168_000, 172_000, 166_000, 161_000, 158_000]
    v, flag = wp.flag_plateaus(plateau)
    assert flag[30:38].all() and not flag[:30].any() and (v[30:38] == 300).all()
    spike = base.copy()
    spike[30:38] = [500_000, 200_000, 90_000, 40_000, 20_000, 9_000, 5_000, 4_000]
    assert not wp.flag_plateaus(spike)[1].any()
    small = base.copy()
    small[30:38] = 4_000.0  # plat mais sous PLATEAU_MIN_VIEWS
    assert not wp.flag_plateaus(small)[1].any()


def test_cycle_share_sums_to_one_among_members(reg):
    d = date(2017, 1, 2)
    df = pl.DataFrame({
        "source": ["wikipedia"] * 3, "candidate_id": ["macron_emmanuel", "fillon_francois",
                                                    "tondelier_marine"],
        "date": [d] * 3, "freq": ["D"] * 3, "value": [300.0, 100.0, 50.0]})
    out = with_cycle_share(df, reg)
    s = dict(zip(out["candidate_id"], out["share"]))
    assert s["macron_emmanuel"] == pytest.approx(0.75) and s["tondelier_marine"] is None
    assert set(out["election"]) == {2017}


# ---------------------------------------------------------------------------- Trends


def test_strip_prefix_and_parse_real_multiline():
    ex = gt.strip_prefix(EXPLORE)
    assert any(w["id"] == "TIMESERIES" for w in ex["widgets"])
    df = gt.parse_multiline(gt.strip_prefix(MULTILINE), ["a", "b"])
    assert df.filter(pl.col("term") == "a").height == 53
    assert df["date"].min() == date(2015, 12, 27)  # dimanche
    assert df["value"].max() == 100


def test_parse_multiline_partial_and_missing():
    data = {"default": {"timelineData": [
        {"time": "1451174400", "value": [10, 0], "hasData": [True, False]},
        {"time": "1451779200", "value": [20, 5], "hasData": [True, True], "isPartial": True},
    ]}}
    df = gt.parse_multiline(data, ["a", "b"])
    assert df.height == 2 and df.filter(pl.col("term") == "b")["value"].item() == 0.0


def test_window_plan_and_batches(reg):
    plan = gt.window_plan(date(2004, 1, 1), date(2026, 10, 19))
    assert plan[0] == gt.Window(date(2004, 1, 1), date(2006, 12, 31))
    assert plan[-1].end == date(2026, 10, 19)
    for a, b in itertools.pairwise(plan):
        assert b.start <= a.end - timedelta(days=300)  # recouvrement d'environ un an
    for w in plan:
        for batch in gt.batches(reg, w):
            assert batch[0] == reg.trends_pivot and 2 <= len(batch) <= 5
    w2017 = next(w for w in plan if w.start == date(2016, 1, 1))
    assert "macron_emmanuel" in gt.window_members(reg, w2017)
    assert "tondelier_marine" not in gt.window_members(reg, plan[0])


def test_normalize_window_uses_reference_batch():
    d = [date(2010, 1, 3) + timedelta(weeks=i) for i in range(5)]
    # lot 1 : pivot bien mesuré ; lot 2 : un candidat écrase le pivot (échelle / 10)
    b1 = pl.DataFrame({"term": ["p"] * 5 + ["a"] * 5, "date": d * 2,
                       "value": [10.0, 20, 30, 40, 100, 5, 5, 5, 5, 5]})
    b2 = pl.DataFrame({"term": ["p"] * 5 + ["b"] * 5, "date": d * 2,
                       "value": [1.0, 2, 3, 4, 10, 100, 50, 50, 50, 50]})
    out, diag = gt.normalize_window([b1, b2], "p")
    assert diag[0][1] == pytest.approx(1.0) and diag[1][1] == pytest.approx(0.1)
    v = {(r["term"], r["date"]): r["value"] for r in out.iter_rows(named=True)}
    assert v[("p", d[0])] == pytest.approx(10 / 40)  # moyenne du pivot de référence = 40
    assert v[("b", d[0])] == pytest.approx(100 / 0.1 / 40)
    assert v[("a", d[0])] == pytest.approx(5 / 40)


def test_chain_windows():
    d = [date(2010, 1, 3) + timedelta(weeks=i) for i in range(6)]
    # fenêtre 1 : semaines 0-3 ; fenêtre 2 : semaines 2-5, unités différentes
    n1 = pl.DataFrame({"term": ["p"] * 4 + ["c"] * 4, "date": d[:4] * 2,
                       "value": [10.0, 10, 20, 20, 5, 5, 5, 5]}).with_columns(
        pl.col("value") / 15)
    n2 = pl.DataFrame({"term": ["p"] * 4 + ["c"] * 4, "date": d[2:] * 2,
                       "value": [40.0, 40, 40, 40, 20, 20, 30, 30]}).with_columns(
        pl.col("value") / 40)
    out, factors = gt.chain_windows({"2010_a": n1, "2010_b": n2}, "p")
    # recouvrement : pivot 20/15 dans la fenêtre 1 et 40/40 dans la 2 -> facteur 4/3
    assert factors["2010_b"] == pytest.approx(4 / 3)
    c = dict(zip(out.filter(pl.col("term") == "c")["date"],
                 out.filter(pl.col("term") == "c")["value"]))
    assert c[d[0]] == pytest.approx(100 * 5 / 15)
    assert c[d[5]] == pytest.approx(100 * 30 / 40 * 4 / 3)
    assert c[d[2]] == pytest.approx((100 * 5 / 15 + 100 * 20 / 40 * 4 / 3) / 2)


def make_trends_client(tmp_path, handler, **kw):
    t = FakeTime()
    c = gt.TrendsClient(http=httpx.Client(transport=httpx.MockTransport(handler)),
                        cache=RawCache("trends", root=tmp_path / "trends"), sleep=t.sleep,
                        clock=t.clock, rand=lambda: 0.0, log=lambda _: None, **kw)
    return c, t


def test_trends_client_flow_and_cache(tmp_path):
    calls = []

    def handler(request):
        path = request.url.path
        calls.append(path)
        if path == "/trends/explore":
            return httpx.Response(429, headers={"set-cookie": "NID=abc; Path=/"})
        if path == "/trends/api/explore":
            req = json.loads(request.url.params["req"])
            assert req["comparisonItem"][0]["time"] == "2016-01-01 2016-12-31"
            return httpx.Response(200, text=EXPLORE)
        if path == "/trends/api/widgetdata/multiline":
            assert request.url.params["token"]
            return httpx.Response(200, text=MULTILINE)
        return httpx.Response(404)

    c, t = make_trends_client(tmp_path, handler)
    w = gt.Window(date(2016, 1, 1), date(2016, 12, 31))
    data, from_cache = c.timeseries(["Emmanuel Macron", "François Fillon"], w)
    assert not from_cache and calls == ["/trends/explore", "/trends/api/explore",
                                        "/trends/api/widgetdata/multiline"]
    assert t.sleeps and min(t.sleeps) >= gt.MIN_INTERVAL_S - 1e-9
    data2, from_cache = c.timeseries(["Emmanuel Macron", "François Fillon"], w)
    assert from_cache and data2 == data and len(calls) == 3
    files = sorted(p.name for p in (tmp_path / "trends" / w.key).iterdir())
    assert len([f for f in files if f.endswith(".multiline.json")]) == 1


def test_trends_rate_limited_after_backoff(tmp_path):
    c, t = make_trends_client(tmp_path, lambda r: httpx.Response(429), backoff=(1, 2))
    with pytest.raises(gt.TrendsRateLimited):
        c.timeseries(["x"], gt.Window(date(2016, 1, 1), date(2016, 12, 31)))
    assert 1 in t.sleeps and 2 in t.sleeps


def test_trends_parse_cached_maps_terms(tmp_path, reg):
    cache = RawCache("trends", root=tmp_path / "trends")
    pivot = reg.candidates[reg.trends_pivot].trends_term
    other = reg.candidates["macron_emmanuel"].trends_term
    w = gt.Window(date(2016, 1, 1), date(2016, 12, 31))
    cache.put(cache.root / w.key / "k.multiline.json", MULTILINE.encode("utf-8"),
              {"terms": [pivot, other], "window": w.time, "resolution": "WEEK"})
    df, _ = gt.parse_cached(reg, cache)
    assert set(df["candidate_id"]) == {reg.trends_pivot, "macron_emmanuel"}
    assert df.filter(pl.col("candidate_id") == reg.trends_pivot)["value"].mean() == (
        pytest.approx(100.0))


# ---------------------------------------------------------------------------- panel


def test_weekly_candidate_panel_rules():
    polls = pl.DataFrame({
        "poll_id": ["a", "b", "c", "d"], "election": [2027] * 4,
        "pollster": ["ifop", "ifop", "elabe", "elabe"],
        "fieldwork_end": [date(2026, 9, 7), date(2026, 9, 9), date(2026, 9, 10),
                          date(2026, 9, 10)],
        "is_rolling": [False] * 4})
    rows = [  # (poll, hypothèse, candidat, part, suspect)
        ("a", "le_pen_marine+philippe_edouard", "le_pen_marine", 34.0, False),
        ("a", "le_pen_marine+philippe_edouard", "philippe_edouard", 20.0, False),
        ("b", "le_pen_marine+philippe_edouard", "le_pen_marine", 36.0, False),
        ("c", "le_pen_marine+philippe_edouard", "le_pen_marine", 30.0, False),
        ("c", "le_pen_marine+zemmour_eric", "le_pen_marine", 32.0, False),
        ("c", "le_pen_marine+macron_emmanuel", "le_pen_marine", 10.0, False),
        ("d", "le_pen_marine+philippe_edouard", "le_pen_marine", 99.0, True),
        ("d", "le_pen_marine+philippe_edouard", "abstention", 30.0, False),
    ]
    results = pl.DataFrame(rows, schema=["poll_id", "hypothesis_id", "candidate_id", "share",
                                         "suspect"], orient="row").with_columns(
        pl.lit(1, dtype=pl.Int8).alias("round"))
    ps = P.poll_candidate_shares(results, polls)
    assert set(ps["candidate_id"]) == {"le_pen_marine", "philippe_edouard"}
    w = P.weekly_candidate_shares(ps)
    lp = w.filter(pl.col("candidate_id") == "le_pen_marine").row(0, named=True)
    # ifop : moyenne de a et b = 35 ; elabe : moyenne des hypothèses de c sans Macron = 31
    assert lp["week"] == date(2026, 9, 7) and lp["share"] == pytest.approx(33.0)
    assert lp["n_polls"] == 3 and lp["n_pollsters"] == 2


def test_weekly_attention_full_weeks_and_trends_shift():
    days = [date(2017, 1, 2) + timedelta(days=i) for i in range(10)]
    daily = pl.DataFrame({"source": "wikipedia", "candidate_id": "x", "date": days,
                          "freq": "D", "value": [1.0] * 10})
    w = P.weekly_attention(daily, "wikipedia")
    assert w["week"].to_list() == [date(2017, 1, 2)] and w["value"].item() == 7.0
    weekly = pl.DataFrame({"source": "trends", "candidate_id": "x", "date": [date(2017, 1, 1)],
                           "freq": "W", "value": [5.0]})
    assert P.weekly_attention(weekly, "trends")["week"].item() == date(2017, 1, 2)


# ---------------------------------------------------------------------------- réseau


@pytest.mark.slow
def test_live_pageviews(tmp_path):
    c = wp.WikiClient(cache=RawCache("wikipedia", root=tmp_path), log=lambda _: None)
    p, _ = wp.fetch_pageviews(c, "Emmanuel Macron", date(2016, 11, 14), date(2016, 11, 20))
    content, meta = c.cache.get(p)
    assert wp.parse_pageviews(content, meta["status"]) == wp.parse_pageviews(PV_MACRON, 200)


@pytest.mark.slow
def test_live_trends(tmp_path):
    c = gt.TrendsClient(cache=RawCache("trends", root=tmp_path), log=lambda _: None,
                        min_interval=5, jitter=2)
    data, _ = c.timeseries(["Emmanuel Macron", "François Fillon"],
                           gt.Window(date(2016, 1, 1), date(2016, 12, 31)))
    assert len(data["default"]["timelineData"]) >= 52
