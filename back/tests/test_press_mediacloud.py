"""Client Media Cloud sur réponses factices (format de mediacloud/api.py 5.2.0)."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from vvs.cli import app
from vvs.ingest.press import mediacloud as mc
from vvs.ingest.press.cache import DiskCache
from vvs.ingest.press.outlets import load_outlets, load_queries

FIX = Path(__file__).parent / "fixtures" / "press"
KEY = "cle-de-test-0123456789"


def fixture(name):
    return (FIX / name).read_bytes()


def make_client(tmp_path, handler):
    return mc.MediaCloudClient(
        key=KEY,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        cache=DiskCache("mediacloud", root=tmp_path / "mc"),
        sleep=lambda s: None,
        log=lambda _: None,
    )


def router(calls):
    def handler(request):
        calls.append(request)
        path = request.url.path
        if path.endswith("search/count-over-time"):
            return httpx.Response(200, content=fixture("mediacloud_count_over_time_synthetic.json"))
        if path.endswith("search/total-count"):
            return httpx.Response(200, json={"count": {"relevant": 52, "total": 13100}})
        if path.endswith("search/story-list"):
            page = 2 if request.url.params.get("pagination_token") == "tok2" else 1
            return httpx.Response(
                200, content=fixture(f"mediacloud_story_list_page{page}_synthetic.json")
            )
        if path.endswith("sources/sources/"):
            return httpx.Response(200, content=fixture("mediacloud_sources_synthetic.json"))
        return httpx.Response(404, json={"note": "inconnu"})

    return handler


def test_missing_key_explains(monkeypatch):
    monkeypatch.delenv(mc.ENV_KEY, raising=False)
    with pytest.raises(mc.MediaCloudKeyMissing) as e:
        mc.MediaCloudClient()
    assert "search.mediacloud.org" in str(e.value) and mc.ENV_KEY in str(e.value)
    result = CliRunner().invoke(app, ["ingest", "press", "--source", "mediacloud"])
    assert result.exit_code == 2
    assert "search.mediacloud.org" in result.output


def test_counts_auth_and_cache(tmp_path):
    calls = []
    c = make_client(tmp_path, router(calls))
    pts = c.story_count_over_time("Macron", date(2017, 1, 1), date(2017, 2, 28))
    assert [p["date"] for p in pts] == [date(2017, 1, 1), date(2017, 1, 2), date(2017, 2, 1)]
    r = calls[0]
    assert r.headers["Authorization"] == f"Token {KEY}"
    assert r.url.params["cs"] == "34412146" and r.url.params["platform"] == "onlinenews-mediacloud"
    assert c.story_count("Macron", date(2017, 1, 1), date(2017, 2, 28)) == {
        "relevant": 52,
        "total": 13100,
    }
    c.story_count_over_time("Macron", date(2017, 1, 1), date(2017, 2, 28))
    assert len(calls) == 2  # la troisième vient du cache
    # la clé n'est écrite nulle part sur disque
    for f in (tmp_path / "mc").rglob("*"):
        if f.is_file():
            assert KEY not in f.read_text(encoding="utf-8")


def test_story_list_pagination_and_items(tmp_path):
    calls = []
    c = make_client(tmp_path, router(calls))
    queries = load_queries()
    n = mc.ingest_story_lists(
        queries,
        ["macron_emmanuel"],
        date(2017, 1, 1),
        date(2017, 1, 31),
        client=c,
        log=lambda _: None,
    )
    assert n == 4 and len(calls) == 2
    assert calls[1].url.params["pagination_token"] == "tok2"
    assert calls[0].url.params["q"] == "Macron" and calls[0].url.params["page_size"] == "1000"
    items = mc.parse_story_items(queries, load_outlets(), cache=c.cache)
    assert items.height == 3  # l'article sans date est écarté
    assert dict(zip(items["title"], items["outlet"], strict=True)) == {
        "Macron candidat": "le_monde",
        "Brigitte et Emmanuel Macron en vacances": None,  # closermag.fr, hors registre
        "Macron et la gauche": "huffpost",
    }
    assert items["query_ids"].to_list()[0] == ["macron_emmanuel"]


def test_measure_coverage_monthly(tmp_path):
    calls = []
    c = make_client(tmp_path, router(calls))
    df = mc.measure_coverage([2017], client=c, outlets=load_outlets())
    assert set(df["scope"]) == {"collection:34412146", "le_monde", "le_figaro"}  # panel seulement
    coll = df.filter(scope="collection:34412146").sort("month")
    assert coll["stories"].to_list() == [3100 + 5200, 4800]
    assert coll["days_with_data"].to_list() == [2, 1]
    q = [r for r in calls if r.url.path.endswith("count-over-time")]
    assert {r.url.params["q"] for r in q} == {"*"}
    assert {r.url.params.get("ss") for r in q} == {None, "1", "2"}
    out = mc.write_coverage(df, tmp_path / "run")
    assert out.read_text().startswith("scope,month,stories")


def test_unauthorized_is_explicit(tmp_path):
    c = make_client(tmp_path, lambda r: httpx.Response(401, json={"note": "Invalid token"}))
    with pytest.raises(mc.MediaCloudError, match="401"):
        c.story_count("Macron", date(2017, 1, 1), date(2017, 1, 2))
    assert not list((tmp_path / "mc").rglob("*.json"))
    journal = [json.loads(x) for x in (tmp_path / "mc" / "requests.jsonl").read_text().splitlines()]
    assert journal[0]["status"] == 401


@pytest.mark.slow
def test_live_version_endpoint():
    r = httpx.get(mc.API_URL + "version", timeout=30)
    assert r.status_code == 200 and "version" in r.json()
