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

    _toy_press_nlp(curated, root / "press_nlp")

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
    _toy_opinion(curated)
    return {"economy_study": eco, "pollster_ratings": ratings, "curated": curated,
            "press_nlp_study": root / "press_nlp", "dispersion": TOY_DISPERSION}


# Dispersion des instituts déjà calculée (le calcul réel relit les sondages curated).
TOY_DISPERSION = {
    "in_forecast": False, "note": "n", "reading": "r", "window_days": 30,
    "weeks": [(ASOF - dt.timedelta(weeks=1)).isoformat(), ASOF.isoformat()],
    "D": [None, 3.5], "reference": [2.0, 2.0], "n_pollsters": [2, 5],
    "current": {"asof": ASOF.isoformat(), "horizon_days": 180, "D": 3.5, "reference": 2.0,
                "pollsters": ["elabe", "ifop"]},
    "past_same_horizon": [{"election": 2022, "D": 2.4}],
}


def _toy_opinion(curated):
    """Anticipations (« qui sera élu ? ») et écarts certains − inscrits jouets."""
    rows = []

    def measure(e, pollster, end, qtype, shares, undecided=None, party=False, verified=True,
                pair=None):
        for k, s in shares.items():
            rows.append({
                "election": e, "pollster": pollster, "fieldwork_end": end, "question_type": qtype,
                "outcome": "win", "candidate_id": None if party else k,
                "party": k if party else None, "pair": pair, "share": s, "undecided": undecided,
                "sample_size": 1000, "source_url": f"https://x/{pollster}/{end}",
                "file_id": 1, "page": 1, "verified": verified,
            })

    d = dt.date
    measure(2027, "cluster17", d(2026, 7, 24), "winner",
            {"le_pen_marine": 40.0, "philippe_edouard": 20.0, "melenchon_jean_luc": 20.0}, 20.0)
    measure(2027, "opinionway", d(2026, 7, 9), "chances", {"le_pen_marine": 54.0,
                                                          "philippe_edouard": 41.0})
    measure(2027, "harris", d(2026, 3, 22), "chances", {"rn": 63.0, "lfi": 19.0}, party=True)
    measure(2027, "opinionway", d(2026, 7, 9), "good_president", {"le_pen_marine": 37.0})
    measure(2027, "elabe", ASOF + dt.timedelta(days=2), "winner", {"le_pen_marine": 99.0})
    # 2022 : un favori juste, une exception, un pronostic d'entre-deux-tours et une mesure non
    # vérifiée, écartés
    measure(2022, "ifop", d(2022, 1, 10), "winner",
            {"macron_emmanuel": 40.0, "le_pen_marine": 20.0, "zemmour_eric": 20.0}, 20.0)
    measure(2022, "elabe", d(2022, 2, 10), "winner",
            {"le_pen_marine": 30.0, "macron_emmanuel": 25.0})
    measure(2022, "ipsos", d(2022, 4, 15), "winner", {"macron_emmanuel": 70.0})
    measure(2022, "bva", d(2022, 3, 1), "winner", {"macron_emmanuel": 90.0}, verified=False)
    pl.DataFrame(rows).write_parquet(curated / "poll_expectations.parquet")

    gaps = []
    for pid, end, values in (
        ("p1", d(2026, 9, 1), {"le_pen_marine": [0.5, 1.5], "melenchon_jean_luc": [-1.0]}),
        ("p2", d(2026, 10, 1), {"le_pen_marine": [2.0], "melenchon_jean_luc": [-0.5]}),
        ("p3", ASOF + dt.timedelta(days=3), {"le_pen_marine": [9.0]}),
    ):
        for cid, gs in values.items():
            for i, g in enumerate(gs):
                gaps.append({"poll_id": pid, "election": 2027, "pollster": "opinionway",
                             "fieldwork_end": end, "round": 1, "hypothesis_id": f"h{i}",
                             "candidate_id": cid, "share_registered": 20.0,
                             "share_certain": 20.0 + g, "gap": g})
    gaps.append({**gaps[0], "round": 2, "gap": 5.0})
    pl.DataFrame(gaps).with_columns(pl.col("round").cast(pl.Int8)).write_parquet(
        curated / "turnout_gap.parquet")


NLP_WEEKS = [dt.date(2026, 9, 21) + dt.timedelta(weeks=i) for i in range(4)]  # lundis
# ton par (candidat, semaine) : (−1, 0, +1) titres classés ; zemmour hors univers jouet
NLP_TONES = {
    "le_pen_marine": [(10, 0, 0), (0, 0, 10), (2, 0, 2), (5, 0, 0)],
    "melenchon_jean_luc": [(0, 4, 0), (0, 3, 0), (0, 0, 2), (0, 30, 0)],
    "zemmour_eric": [(0, 20, 0)] * 4,
}


def _toy_press_nlp(curated, study):
    """Titres, étiquettes et variables jouets de 2027 (quatre semaines complètes jusqu'au
    dimanche 2026-10-18 et le lundi 2026-10-19, semaine incomplète) et de 2017 (repère Fillon),
    plus l'étude (accords, dates de rupture)."""
    from vvs.nlp.prompts import load_prompts

    model = load_prompts().model["id"]
    titles, labels, feats = [], [], []

    def add(cid, cycle, day, tone, stance, themes, k, version="title-v3"):
        sid = f"{cid}-{day}-{k}"
        titles.append({"story_id": sid, "candidate_id": cid, "cycle": cycle, "publish_date": day,
                       "outlet": "exemple.fr", "title": f"Titre {sid}",
                       "url": f"https://example.org/{sid}", "language": "fr", "in_title": True})
        labels.append({"story_id": sid, "candidate_id": cid, "prompt_version": version,
                       "model": model, "tone": tone, "stance": stance, "themes": themes})

    for cid, per_week in NLP_TONES.items():
        for w, counts in zip(NLP_WEEKS, per_week):
            k = 0
            for tone, n in zip((-1, 0, 1), counts):
                for _ in range(n):
                    stance = {-1: "attack", 0: "announcement", 1: "support"}[tone]
                    themes = ["immigration"] if k % 2 == 0 else ["security", "europe"]
                    add(cid, 2027, w + dt.timedelta(days=k % 7), tone, stance, themes, k)
                    k += 1
            feats.append({"cycle": 2027, "candidate_id": cid, "week": w, "mentions": k,
                          "n_labeled": k, "days": 7})
    # une étiquette favorable d'une autre version de prompt sur un titre classé : ignorée
    labels.append(labels[-1] | {"story_id": f"le_pen_marine-{NLP_WEEKS[-1]}-0",
                                 "candidate_id": "le_pen_marine", "prompt_version": "title-v2",
                                 "tone": 1, "stance": "support"})
    # semaine du lundi 2026-10-19 : incomplète, hors grille
    feats.append({"cycle": 2027, "candidate_id": "le_pen_marine", "week": dt.date(2026, 10, 19),
                  "mentions": 1, "n_labeled": 0, "days": 1})
    # repère Fillon 2017 : neutre, puis « affaire » hostile à partir du 25 janvier
    for i in range(-30, 43):
        day = dt.date(2017, 1, 25) + dt.timedelta(days=i)
        for k in range(4):
            if i < 0:
                add("fillon_francois", 2017, day, 0, "announcement", [], k)
            else:
                add("fillon_francois", 2017, day, -1, "scandal", ["scandals"], k)
    pl.DataFrame(titles).with_columns(pl.col("cycle").cast(pl.Int32)).write_parquet(
        curated / "press_titles.parquet")
    pl.DataFrame(labels).with_columns(pl.col("tone").cast(pl.Int8)).write_parquet(
        curated / "press_labels.parquet")
    pl.DataFrame(feats).with_columns(pl.col("cycle").cast(pl.Int32),
                                     pl.col("n_labeled", "days").cast(pl.UInt32)).write_parquet(
        curated / "press_features.parquet")
    study.mkdir()
    (study / "agreement.json").write_text(json.dumps({
        "sample": {"n": 300, "tone_kappa": 0.5958, "stance_kappa": 0.739, "themes_f1": 0.546},
        "thresholds": {"tone_kappa": 0.6, "stance_kappa": 0.5, "themes_f1": 0.6},
        "passes": {"tone_kappa": False, "stance_kappa": True, "themes_f1": False}}))
    (study / "shocks.csv").write_text(
        "event,candidate_id,date,sign,poll_break,break_tone_3d,break_scandal_share_3d\n"
        "fillon_affair_2017,fillon_francois,2017-01-25,-1,2017-01-27,2017-01-26,2017-01-24\n"
        "pecresse_zenith_2022,pecresse_valerie,2022-02-13,-1,2022-02-05,2022-02-11,\n")


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
    assert len(tested) == 12 and not any(v["admitted"] for v in tested.values())
    assert tested["cevipof"]["verdict"] == "en attente d'accès"
    assert tested["herding"]["wave"] == "2026-10-22" and tested["bounce"]["wave"] == "2026-10-20"
    assert [k for k, v in tested.items() if v["published"]] == ["substitution"]
    assert tested["press_nlp"]["verdict"] == "non retenue"


def test_opinion_indicators(exported):
    """Anticipations, dispersion et électorats fragiles : hors prévision, bornés à asof."""
    _, out, _, _ = exported
    exp = json.loads((out / "summary.json").read_text())["expectations"]
    assert exp["in_forecast"] is False
    # candidats d'abord (`winner` puis `chances`), partis ensuite ; « bon président » et mesure
    # postérieure à asof écartés
    assert [(m["pollster"], m["question_type"], m["kind"]) for m in exp["measures"]] == [
        ("cluster17", "winner", "candidate"), ("opinionway", "chances", "candidate"),
        ("harris", "chances", "party")]
    c17 = exp["measures"][0]["items"][0]
    assert (c17["candidate_id"], c17["share"], c17["share_named"]) == ("le_pen_marine", 40.0, 50.0)
    assert exp["measures"][1]["items"][0]["share_named"] is None
    assert exp["measures"][2]["items"][0]["label"] == "Un candidat du RN"
    [h] = exp["history"]
    assert h["election"] == 2022 and h["winner"] == "macron_emmanuel"
    assert [(p["days_before"], p["winner_share_named"], p["top_is_winner"]) for p in h["points"]] \
        == [(90, 50.0, True), (59, 45.5, False)]
    texts = {x["id"]: x["text"] for x in exp["landmarks"]}
    assert "dans 1 mesures sur 2 depuis 2022" in texts["top_is_winner"]
    assert "Marine Le Pen" in texts["top_is_winner"]
    assert "plus de 50 %" in texts["max_winner_share"]

    polls = json.loads((out / "polls.json").read_text())
    assert polls["dispersion"]["current"]["D"] == 3.5
    tg = polls["turnout_gap"]
    assert tg["in_forecast"] is False and tg["n_polls"] == 2
    assert [(r["candidate_id"], r["gap"], r["se"], r["n_polls"]) for r in tg["rows"]] == [
        ("melenchon_jean_luc", -0.75, 0.25, 2), ("le_pen_marine", 1.5, 0.5, 2)]


def test_press_nlp_sections(exported):
    _, out, _, _ = exported
    press = json.loads((out / "press.json").read_text())
    assert press["nlp"]["in_forecast"] is False and len(press["nlp"]["verdict"]) == 2
    weeks = [w.isoformat() for w in NLP_WEEKS]

    tone = press["tone"]
    assert tone["weeks"] == weeks  # semaine du 19 octobre incomplète, écartée
    tr = {t["candidate_id"]: t for t in tone["tracks"]}
    assert set(tr) == {"le_pen_marine", "melenchon_jean_luc"}  # zemmour hors univers
    lp, me = tr["le_pen_marine"], tr["melenchon_jean_luc"]
    # deux semaines glissantes : 10 hostiles, puis 10 + 10, puis 10 favorables + 2 + 2, etc.
    assert lp["n"] == [10, 20, 14, 9]
    assert lp["q50"][0] == -1.0 and lp["q05"][0] == lp["q95"][0] == -1.0
    assert lp["q05"][1] < lp["q50"][1] < lp["q95"][1] and abs(lp["q50"][1]) <= 0.15
    assert lp["q50"][3] is None and lp["q05"][3] is None  # 9 titres : masqué
    assert me["n"] == [4, 7, 5, 32] and me["q50"][:3] == [None] * 3 and me["q50"][3] is not None

    st = press["stance"]
    ids = [c["id"] for c in st["categories"]]
    rows = {r["candidate_id"]: r for r in st["last_weeks"]["rows"]}
    assert rows["le_pen_marine"]["n"] == 29 and st["last_weeks"]["end"] == "2026-10-18"
    shares = dict(zip(ids, rows["le_pen_marine"]["shares"]))
    assert shares["attack"] == pytest.approx(17 / 29, abs=1e-3)
    assert shares["support"] == pytest.approx(12 / 29, abs=1e-3)
    assert sum(shares.values()) == pytest.approx(1, abs=0.01)
    wk = {t["candidate_id"]: t for t in st["tracks"]}["melenchon_jean_luc"]
    assert wk["n"] == [4, 3, 2, 30] and wk["shares"][:3] == [None] * 3

    hl = {c["candidate_id"]: c for c in press["headlines"]["candidates"]}
    assert press["headlines"]["week"] == weeks[-1]
    neg = hl["le_pen_marine"]["negative"]
    assert len(neg) == 3 and [h["date"] for h in neg] == sorted((h["date"] for h in neg),
                                                                reverse=True)
    assert hl["le_pen_marine"]["positive"] == [] and hl["melenchon_jean_luc"]["negative"] == []

    sov = press["share_of_voice"]
    v = {t["candidate_id"]: t["values"] for t in sov["tracks"]}
    assert v["le_pen_marine"][0] == pytest.approx(10 / 14, abs=1e-4)
    assert v["melenchon_jean_luc"][3] == pytest.approx(30 / 35, abs=1e-4)
    att = {t["candidate_id"]: t["values"] for t in sov["attention"]}
    # pages vues jouets : 300 contre 100 sur les semaines complètes du 28/09 au 12/10
    assert att["le_pen_marine"] == [None, 0.75, 0.75, 0.75]

    ag = press["agenda_nlp"]
    assert [t["id"] for t in ag["themes"]] == ["immigration", "purchasing_power",
                                               "international", "security"]
    th = {t["id"]: t["values"] for t in ag["themes"]}
    assert ag["n"][0] == 14 and th["immigration"][0] == pytest.approx(round(7 / 14, 4))
    assert th["purchasing_power"][0] == 0.0

    lm = {e["id"]: e for e in press["landmarks"]["events"]}
    f = lm["fillon_affair_2017"]
    assert f["cycle"] == 2017 and f["dates"][0] == "2016-12-28" and f["dates"][-1] == "2017-03-08"
    i = f["dates"].index("2017-01-25")
    assert f["tone"][i - 2] == 0.0 and f["tone"][i + 2] == -1.0 and f["scandal_share"][i + 2] == 1.0
    assert f["breaks"] == {"polls": "2017-01-27", "tone": "2017-01-26", "scandal": "2017-01-24"}
    p = lm["pecresse_zenith_2022"]
    assert p["breaks"]["scandal"] is None and all(x is None for x in p["tone"])

    q = press["quality"]
    assert q["prompt_version"] == "title-v3" and q["tone_kappa"] == 0.596
    assert q["passes"] == {"tone_kappa": False, "stance_kappa": True, "themes_f1": False}
    m = json.loads((out / "manifest.json").read_text())
    assert {"fillon_francois", "pecresse_valerie"} <= {c["id"] for c in m["candidates"]}


@pytest.mark.skipif(
    shutil.which("node") is None or not (FRONT / "node_modules" / "ajv").exists(),
    reason="node ou les dépendances du front manquent",
)
def test_export_validates_against_schemas(exported):
    _, out, _, _ = exported
    r = subprocess.run(["node", "scripts/validate-data.js", str(out)], cwd=FRONT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr + r.stdout
