import datetime as dt
import json
import shutil
import subprocess

import polars as pl
import pytest
from toy_run import ASOF, FIT_CANDIDATES, run_toy, toy_scenarios

from vvs import paths
from vvs.export.export import compatible, export_run

FRONT = paths.ROOT / "front"


def _toy_polls() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    end = ASOF - dt.timedelta(days=5)
    polls = pl.DataFrame(
        {
            "poll_id": ["2027-ifop-1", "2027-elabe-1"],
            "pollster": ["ifop", "elabe"],
            "sponsor": ["Le JDD", None],
            "fieldwork_start": [end - dt.timedelta(days=3), end - dt.timedelta(days=10)],
            "fieldwork_end": [end, end - dt.timedelta(days=7)],
            "publish_date": [end + dt.timedelta(days=1), None],
            "sample_size": [1000, None],
            "mode": ["online", "unknown"],
            "source_url": ["https://example.org/notice", None],
        }
    )
    rows = []
    for pid, pol, d in zip(polls["poll_id"], polls["pollster"], polls["fieldwork_end"]):
        cands = sorted(c for c in FIT_CANDIDATES if c != "bardella_jordan")
        tot = sum(FIT_CANDIDATES[c][1] for c in cands)
        for c in cands:
            rows.append({"poll_id": pid, "hypothesis_id": "+".join(cands), "round": 1,
                         "population": "registered", "candidate_id": c, "pollster": pol,
                         "fieldwork_end": d, "share": round(100 * FIT_CANDIDATES[c][1] / tot, 1)})
        for c, s in (("le_pen_marine", 53.0), ("philippe_edouard", 47.0)):
            rows.append({"poll_id": pid, "hypothesis_id": "le_pen_marine+philippe_edouard",
                         "round": 2, "population": "registered", "candidate_id": c,
                         "pollster": pol, "fieldwork_end": d, "share": s})
    results = pl.DataFrame(rows).with_columns(pl.col("round").cast(pl.Int8))
    return polls, results, results.filter(pl.col("round") == 1)


def _toy_explanatory(root):
    """Études et parquet explicatifs jouets : (étude économie, palmarès, curated)."""
    eco, ratings, curated = root / "economy", root / "ratings", root / "curated"
    for d in (eco, ratings, curated):
        d.mkdir()
    pl.DataFrame(
        {
            "bloc": ["far_right", "incumbent"],
            "indicator": ["unemployment_rate_ilo", "cpi_all"],
            "sd_d1": [0.1, 0.3],
            "best_beta": [2.0, -1.0],
            "best_se": [1.0, 0.5],
            "best_lag": [3, 0],
            "p_placebo": [0.01, 0.5],
            "best_n": [80, 79],
            "best_n_cycles": [6, 6],
        }
    ).write_parquet(eco / "coef_pairs.parquet")
    shutil.copy(paths.CURATED / "macro.parquet", curated / "macro.parquet")

    # attention : trois semaines complètes (lundi 2026-09-28 au dimanche 2026-10-18) et un
    # lundi isolé (semaine incomplète, écartée) ; arthaud n'est pas dans l'univers jouet
    days = [dt.date(2026, 9, 28) + dt.timedelta(days=i) for i in range(22)]
    rows = []
    for d in days:
        for cid, v in (("le_pen_marine", 300.0), ("melenchon_jean_luc", 100.0),
                       ("zemmour_eric", 100.0)):
            rows.append({"source": "wikipedia", "candidate_id": cid, "date": d, "freq": "D",
                         "election": 2027, "value": v, "raw_value": v, "suspect": False,
                         "share": v / 500})
    pl.DataFrame(rows).with_columns(pl.col("election").cast(pl.Int32)).write_parquet(
        curated / "attention.parquet")
    pl.DataFrame(
        {
            "date": [dt.date(2002, 3, 1), dt.date(2002, 3, 1), dt.date(2007, 3, 1),
                     dt.date(2010, 3, 1)],
            "channel": ["TF1"] * 4,
            "theme": ["Société", "Sport", "Société", "Sport"],
            "n_subjects": [3, 1, 2, 4],
            "duration_s": [60, 60, 60, 60],
        }
    ).with_columns(pl.col("n_subjects", "duration_s").cast(pl.Int32)).write_parquet(
        curated / "press_ina.parquet")
    pl.DataFrame(
        {
            "source": ["gdelt"] * 4,
            "query_id": ["le_pen_marine", "le_pen_marine", "macron_emmanuel", "macron_emmanuel"],
            "candidate_id": ["le_pen_marine", "le_pen_marine", "macron_emmanuel",
                             "macron_emmanuel"],
            "date": [dt.date(2026, 8, 3), dt.date(2026, 9, 3)] * 2,
            "volume": [10, 30, 50, 20],
            "total_monitored": [1000, 1000, 1000, 1000],
            "volume_share": [1.0, 3.0, 5.0, 2.0],
            "tone": [0.0] * 4,
        }
    ).write_parquet(curated / "press_timelines.parquet")

    lineages = ["ifop", "elabe", "verian"]
    pl.DataFrame(
        {
            "lineage": lineages, "elections": ["2012,2017,2022", "2017,2022", "2012,2022"],
            "n_polls": [30, 20, 10], "n_final": [5, 4, 3], "mae_cand": [1.0, 0.8, 1.2],
            "mae_bloc": [2.0, 1.5, 2.2], "pm_cand": [0.0, -0.2, 0.15], "x_raw": [0.1, -0.3, 0.2],
            "n_better": [2, 2, 1], "n_x": [3, 2, 2], "far_left": [1.0, -2.0, 0.5],
            "right": [0.1, 1.6, 0.2], "far_right": [-0.8, 1.6, -0.8], "left": [0.3, 0.1, 1.3],
            "centre": [0.2, -0.9, -1.0],
        }
    ).write_parquet(ratings / "palmares.parquet")
    per_e = [{"election": e, "lineage": lid, "x": x, "pm_cand": x / 2}
             for e, xs in ((2012, (0.1, 0.4, -0.2)), (2017, (0.3, -0.1, 0.0)),
                           (2022, (-0.2, 0.2, 0.1)))
             for lid, x in zip(lineages, xs)]
    pl.DataFrame(per_e).write_parquet(ratings / "palmares_by_election.parquet")
    pl.DataFrame(
        [{"election": e, "pollster": lid, "lineage": lid, "bloc": b, "alpha": 0.01 * (i - j)}
         for e in (2012, 2017, 2022) for i, lid in enumerate(lineages)
         for j, b in enumerate(("far_right", "left", "other"))]
    ).write_parquet(ratings / "house_effects.parquet")
    return {"economy_study": eco, "pollster_ratings": ratings, "curated": curated}


def _toy_backtest(root):
    first, second = root / "bt1", root / "bt2"
    first.mkdir()
    second.mkdir()
    rows = []
    for e in (2017, 2022):
        for h in (180, 30):
            for m in ("model", "mean14"):
                for metric, v in (("log_score_pair", -1.0), ("brier_qual", 0.1), ("crps", 2.0),
                                  ("mae", 2.5), ("cov50", 0.5), ("cov90", 0.9)):
                    rows.append({"election": e, "horizon": h, "variant": "floor", "model": m,
                                 "level": "summary", "unit": "", "metric": metric, "value": v})
    pl.DataFrame(rows).write_parquet(first / "metrics.parquet")
    pl.DataFrame(
        {
            "election": [2022, 2022],
            "horizon_days": [-13, 30],
            "method": ["model", "model"],
            "log_score": [-0.1, None],
            "cover50": [True, False],
            "cover90": [True, True],
        }
    ).write_parquet(second / "metrics.parquet")
    return first, second


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    run = root / "2026-10-20-simulate"
    run_toy(root, n=2000, out=run)
    m = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps(m | {"polls_hash": "0123456789abcdef"}))
    out = root / "data"
    out.mkdir()
    # données factices d'avant : l'historique factice doit disparaître
    (out / "history").mkdir()
    (out / "history" / "2026-06-01.json").write_text("{}")
    (out / "manifest.json").write_text(json.dumps({"mock": True}))
    first, second = _toy_backtest(root)
    extra = _toy_explanatory(root)
    rep = export_run(run, out, scenarios=toy_scenarios(), backtest_first=first,
                     backtest_second=second, polls_data=_toy_polls(), **extra)
    return run, out, rep, (first, second, extra)


def test_compatible_rule():
    f = toy_scenarios()
    hyp = ["le_pen_marine", "philippe_edouard", "melenchon_jean_luc"]
    assert compatible(hyp, {"rn": "le_pen"}, f)
    assert not compatible(hyp, {"rn": "bardella"}, f)
    assert compatible(hyp, {"lr": "rally"}, f)  # aucun candidat d'un autre état
    assert not compatible(hyp + ["retailleau_bruno"], {"lr": "rally"}, f)
    assert not compatible(hyp + ["bardella_jordan"], {"rn": "le_pen"}, f)
    assert compatible(hyp, {"rn": "le_pen", "centre": "philippe"}, f)


def test_export_files_and_history(exported):
    _, out, _, _ = exported
    names = {p.name for p in out.glob("*.json")}
    assert names == {"manifest.json", "summary.json", "first_round.json", "second_round.json",
                     "scenarios.json", "polls.json", "economy.json", "press.json", "backtest.json"}
    assert sorted(p.name for p in (out / "history").glob("*.json")) == ["2026-10-20.json",
                                                                       "index.json"]
    m = json.loads((out / "manifest.json").read_text())
    assert m["mock"] is False and m["run_date"] == "2026-10-20" and m["n_polls"] == 2
    s = json.loads((out / "summary.json").read_text())
    assert s["conditions"][0]["id"] == "mix"
    assert {c["id"] for c in s["conditions"]} >= {"rn.le_pen", "rn.bardella", "lr.rally"}
    fr = json.loads((out / "first_round.json").read_text())
    poll = fr["polls"][0]
    assert {"rn.le_pen", "lr.retailleau", "centre.philippe"} <= set(poll["conditions"])
    assert not {"rn.bardella", "lr.rally", "centre.other"} & set(poll["conditions"])
    bt = json.loads((out / "backtest.json").read_text())
    assert min(bt["horizons_days"]) >= 0  # la veille du second tour n'est pas exportée


def test_history_is_never_rewritten(exported, tmp_path):
    run, out, _, (first, second, extra) = exported
    hist = out / "history" / "2026-10-20.json"
    before = hist.read_text()
    tampered = json.loads(before) | {"n_polls": 999}
    hist.write_text(json.dumps(tampered))
    rep = export_run(run, out, scenarios=toy_scenarios(), backtest_first=first,
                     backtest_second=second, polls_data=_toy_polls(), **extra)
    assert not rep["history_written"]
    assert json.loads(hist.read_text())["n_polls"] == 999
    idx = json.loads((out / "history" / "index.json").read_text())
    assert idx["entries"][-1]["n_polls"] == 999
    hist.write_text(before)
    export_run(run, out, scenarios=toy_scenarios(), backtest_first=first,
               backtest_second=second, polls_data=_toy_polls(), **extra)


def test_explanatory_sections(exported):
    _, out, rep, _ = exported
    m = json.loads((out / "manifest.json").read_text())
    assert m["polls_hash"] == rep["polls_hash"] == "0123456789abcdef"
    assert "macron_emmanuel" in {c["id"] for c in m["candidates"]}  # cité par GDELT

    eco = json.loads((out / "economy.json").read_text())
    c = {x["target"]: x for x in eco["correlations"]}
    assert [x["series_id"] for x in eco["correlations"]] == ["unemployment_rate_ilo", "cpi_all"]
    assert c["far_right"]["effect"]["q50"] == pytest.approx(0.2)  # β × écart type
    assert c["far_right"]["effect"]["q05"] == pytest.approx((2.0 - 1.6449) * 0.1, abs=0.01)
    assert c["incumbent"]["target_label"] == "Bloc du président sortant"
    assert "1 paire sur 2 passe le placebo" in eco["correlations_note"]
    assert eco["study"]["verdict"] == "negative" and eco["study"]["findings"]

    press = json.loads((out / "press.json").read_text())
    att = press["attention"]
    assert att["in_forecast"] is False
    assert att["weeks"] == ["2026-09-28", "2026-10-05", "2026-10-12"]  # semaine incomplète écartée
    tracks = {t["candidate_id"]: t["values"] for t in att["tracks"]}
    assert set(tracks) == {"le_pen_marine", "melenchon_jean_luc"}  # zemmour hors univers jouet
    assert tracks["le_pen_marine"] == [60.0] * 3 and tracks["melenchon_jean_luc"] == [20.0] * 3
    ina = press["ina"]
    assert ina["years"] == [2002, 2007] and ina["all_years_label"] == "2002-2010"
    shares = {t["label"]: t["shares"] for t in ina["topics"]}
    assert shares == {"Sport": [25.0, 0.0], "Société": [75.0, 100.0]}
    gd = press["gdelt"]
    assert gd["months"] == ["2026-08-01", "2026-09-01"]
    assert {t["candidate_id"]: t["values"] for t in gd["tracks"]} == {
        "macron_emmanuel": [5.0, 2.0], "le_pen_marine": [1.0, 3.0]}

    ratings = json.loads((out / "polls.json").read_text())["pollster_ratings"]
    assert ratings["note"].startswith("Classement non reproductible d'une élection à l'autre")
    assert ratings["elections"] == [2012, 2017, 2022]
    row = {r["pollster"]: r for r in ratings["rows"]}
    assert row["verian"]["name"] == "Verian (Sofres, Kantar)"
    assert row["elabe"]["bias"]["far_right"] == 1.6 and row["elabe"]["elections"] == [2017, 2022]
    st = {x["id"]: x for x in ratings["stability"]}
    assert st["precision"]["n_pairs"] == 6 and st["house_effects"]["n_pairs"] == 12  # sans other
    assert -1 <= st["precision"]["pearson"] <= 1

    bt = json.loads((out / "backtest.json").read_text())
    tested = {v["id"]: v for v in bt["variables_tested"]}
    assert len(tested) == 6 and not any(v["admitted"] for v in tested.values())
    assert [k for k, v in tested.items() if v["published"]] == ["substitution"]


@pytest.mark.skipif(
    shutil.which("node") is None or not (FRONT / "node_modules" / "ajv").exists(),
    reason="node ou les dépendances du front manquent",
)
def test_export_validates_against_schemas(exported):
    _, out, _, _ = exported
    r = subprocess.run(["node", "scripts/validate-data.js", str(out)], cwd=FRONT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr + r.stdout
