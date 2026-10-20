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
    out = root / "data"
    out.mkdir()
    # données factices d'avant : l'historique factice doit disparaître
    (out / "history").mkdir()
    (out / "history" / "2026-06-01.json").write_text("{}")
    (out / "manifest.json").write_text(json.dumps({"mock": True}))
    first, second = _toy_backtest(root)
    rep = export_run(run, out, scenarios=toy_scenarios(), backtest_first=first,
                     backtest_second=second, polls_data=_toy_polls())
    return run, out, rep, (first, second)


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
    run, out, _, (first, second) = exported
    hist = out / "history" / "2026-10-20.json"
    before = hist.read_text()
    tampered = json.loads(before) | {"n_polls": 999}
    hist.write_text(json.dumps(tampered))
    rep = export_run(run, out, scenarios=toy_scenarios(), backtest_first=first,
                     backtest_second=second, polls_data=_toy_polls())
    assert not rep["history_written"]
    assert json.loads(hist.read_text())["n_polls"] == 999
    idx = json.loads((out / "history" / "index.json").read_text())
    assert idx["entries"][-1]["n_polls"] == 999
    hist.write_text(before)
    export_run(run, out, scenarios=toy_scenarios(), backtest_first=first,
               backtest_second=second, polls_data=_toy_polls())


@pytest.mark.skipif(
    shutil.which("node") is None or not (FRONT / "node_modules" / "ajv").exists(),
    reason="node ou les dépendances du front manquent",
)
def test_export_validates_against_schemas(exported):
    _, out, _, _ = exported
    r = subprocess.run(["node", "scripts/validate-data.js", str(out)], cwd=FRONT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr + r.stdout
