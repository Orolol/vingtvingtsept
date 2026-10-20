"""Comptages Media Cloud par candidat et thème (mediacloud_counts) et panel hebdomadaire
presse (vvs.studies.press_panel), sur réponses count-over-time figées."""

from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from vvs.ingest.press import mediacloud as mc
from vvs.ingest.press import mediacloud_counts as MC
from vvs.ingest.press.cache import DiskCache
from vvs.ingest.press.outlets import load_queries
from vvs.studies import press_panel as PP

FIX = Path(__file__).parent / "fixtures" / "press"
BY_QUERY = {"Macron": "macron", '"Marine Le Pen"': "lepen"}


def handler(calls):
    def h(request):
        calls.append(request)
        q = request.url.params["q"]
        name = BY_QUERY.get(q, "theme")
        return httpx.Response(200, content=(FIX / f"mediacloud_cot_{name}_synthetic.json")
                              .read_bytes())
    return h


@pytest.fixture
def setup(tmp_path):
    calls = []
    cache = DiskCache("mediacloud", root=tmp_path / "mc")
    client = mc.MediaCloudClient(key="cle-de-test", cache=cache, sleep=lambda s: None,
                                 log=lambda _: None,
                                 http=httpx.Client(transport=httpx.MockTransport(handler(calls))))
    queries = load_queries()
    reqs = MC.plan(queries, elections=(2017,),
                   candidates={2017: ("macron_emmanuel", "le_pen_marine")},
                   themes=("immigration",), co_themes=())
    return calls, cache, client, queries, reqs


def test_windows():
    assert MC.windows(2017) == [(date(2016, 4, 23), date(2017, 4, 22))]
    assert MC.windows(2022) == [(date(2021, 4, 10), date(2022, 4, 9))]
    assert MC.windows(2027, today=date(2026, 10, 20)) == [
        (date(2025, 1, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 10, 19))]
    with pytest.raises(ValueError):
        MC.windows(2012)


def test_plan_uses_registry_aliases(setup):
    *_, reqs = setup
    assert [(r.scope, r.id) for r in reqs] == [
        ("candidate", "macron_emmanuel"), ("candidate", "le_pen_marine"),
        ("theme", "immigration")]
    assert reqs[1].q == '"Marine Le Pen"'
    assert reqs[0].params()["cs"] == str(mc.FRANCE_NATIONAL)


def test_collect_then_parse_from_cache_only(setup):
    calls, cache, client, _, reqs = setup
    stats = MC.collect(reqs, client=client, log=lambda _: None)
    assert (stats["done"], stats["cached"], stats["failed"]) == (3, 0, [])
    stats = MC.collect(reqs, client=client, log=lambda _: None)
    assert stats["cached"] == 3 and len(calls) == 3  # deuxième passage : tout en cache
    df = MC.counts_frame(reqs, MC._cached_index(cache), log=lambda _: None)
    assert df.schema == MC.SCHEMA
    assert df.height == 3 * 16
    m = df.filter(pl.col("id") == "macron_emmanuel").sort("date")
    assert m["count"][0] == 30 and m["total"][0] == 1000
    assert m["share"][0] == pytest.approx(0.03)


def test_latest_volatile_response_wins(tmp_path):
    cache = DiskCache("mediacloud", root=tmp_path / "mc")
    r = MC.CountRequest("candidate", "macron_emmanuel", 2027, date(2026, 1, 1),
                        date(2026, 1, 16), "Macron")
    body = (FIX / "mediacloud_cot_macron_synthetic.json").read_bytes()
    newer = body.replace(b'"count": 30', b'"count": 31')
    cache.put(cache.path_for(MC.ENDPOINT_KIND, 2026, "k_asof20261020"), body,
              {"params": r.params()})
    # même requête, relancée le lendemain : la réponse la plus récente l'emporte
    p = cache.path_for(MC.ENDPOINT_KIND, 2026, "k_asof20261021")
    cache.put(p, newer, {"params": r.params()})
    meta = p.with_name(p.stem + ".meta.json")
    meta.write_text(meta.read_text().replace('"retrieved_at": "', '"retrieved_at": "9'))
    df = MC.counts_frame([r], MC._cached_index(cache), log=lambda _: None)
    assert set(df["count"]) == {31}


def test_weekly_share_of_voice(setup):
    _, cache, client, _, reqs = setup
    MC.collect(reqs, client=client, log=lambda _: None)
    df = MC.counts_frame(reqs, MC._cached_index(cache), log=lambda _: None)
    w = PP.weekly_counts(df)
    # 2017-01-01 (dimanche) et 2017-01-16 (lundi seul) : semaines incomplètes retirées
    assert sorted(set(w["week"])) == [date(2017, 1, 2), date(2017, 1, 9)]
    sov = PP.share_of_voice(w, {2017: ("macron_emmanuel", "le_pen_marine")})
    first = sov.filter(pl.col("week") == date(2017, 1, 2))
    x = dict(zip(first["candidate_id"], first["sov"]))
    # semaine du 2 janvier : Macron 7 × 30, Le Pen 6 × 10 + 1 × 20 (bascule le 8 janvier)
    assert x["macron_emmanuel"] == pytest.approx(100 * 210 / 290)
    assert x["macron_emmanuel"] + x["le_pen_marine"] == pytest.approx(100)
    assert (sov.group_by("week").agg(pl.col("sov").sum())["sov"] - 100).abs().max() < 1e-9
    # part de la collection : 210 articles sur 1 010 + … + 1 070 (jours 2 à 8)
    tot = sum(1000 + 10 * i for i in range(1, 8))
    assert first.filter(pl.col("candidate_id") == "macron_emmanuel")["volume_share"][0] == \
        pytest.approx(100 * 210 / tot)
    th = PP.theme_salience(w)
    assert th["theme"].unique().to_list() == ["immigration"]
    assert th["salience"][0] == pytest.approx(100 * 350 / tot)


def test_gdelt_weekly_tone_weighted_and_ina_shares():
    days = pl.date_range(date(2017, 1, 2), date(2017, 1, 8), eager=True)
    tl = pl.DataFrame({"candidate_id": ["x"] * 7, "date": days,
                       "volume": [0, 10, 10, 10, 10, 10, 30],
                       "total_monitored": [100] * 7,
                       "tone": [None, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]})
    g = PP.gdelt_weekly(tl)
    assert g.height == 1
    assert g["tone"][0] == pytest.approx((-50 + 30) / 80)  # jour sans article ignoré
    assert g["volume_share"][0] == pytest.approx(100 * 80 / 700)
    ina = pl.DataFrame({"date": list(days) * 2, "channel": ["TF1"] * 14,
                        "theme": ["Faits divers"] * 7 + ["Sport"] * 7,
                        "n_subjects": [1] * 14, "duration_s": [30] * 7 + [90] * 7})
    s = PP.ina_weekly(ina, {"security": ["Faits divers", "Justice"]})
    assert s["share"][0] == pytest.approx(25.0)
