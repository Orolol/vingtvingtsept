"""Export JSON du site (docs/08-front.md, contrat de back/src/vvs/export/schemas/).

`export_run(run, out)` lit un run de simulation (`vvs simulate`, voir vvs.model.simulate) et
écrit dans `out` (front/static/data/) exactement les fichiers du contrat :

- manifest.json, summary.json, first_round.json, second_round.json, scenarios.json,
  polls.json, economy.json, press.json, backtest.json ;
- history/<run_date>.json, jamais réécrit une fois publié (un fichier existant et différent
  est laissé tel quel et signalé), et history/index.json, régénéré à chaque export. Les
  entrées d'historique des données factices (`manifest.mock` vrai dans le dossier de sortie)
  sont retirées au premier export réel.

Sources en plus du run : registres (candidats, instituts, scénarios), sondages curated 2027,
macro.parquet, et les runs de backtest (premier et second tour). La validation contre les
schémas se fait côté front (`npm run validate-data`, ajv) : jsonschema n'est pas une
dépendance du back.

Arrondis : probabilités à 4 décimales, parts et quantiles à 0,1 point, reports au point.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import Registry, default_registry
from vvs.registry.scenarios import ScenariosFile, enumerate_configurations, load_scenarios

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
MODEL_VERSION = "0.4.0-iia"  # phase 4 : mélange de scénarios, softmax restreint (IIA)
QNAMES = ("q05", "q25", "q50", "q75", "q95")
EXPORT_CONFIGS = 10  # configurations résumées dans summary.json
SCENARIO_CONFIGS = 12  # configurations listées dans scenarios.json
FIRST_ROUND_CONFIGS = 3  # configurations proposées au filtre du premier tour
SERIES_MIN_PRESENCE = 0.1  # trajectoires des candidats présents au moins une fois sur dix
SERIES_MAX_CANDIDATES = 12  # par condition autre que `mix`
PAIR_MIN_PROBABILITY = 0.002
TRANSFER_MIN_PAIR = 0.03
TRANSFER_MIN_SHARE = 1.5  # points : électorats affichés dans les reports
ECONOMY_SINCE = dt.date(2012, 1, 1)
BACKTEST_FIRST_ROUND = "2026-10-20-backtest"
BACKTEST_SECOND_ROUND = "2026-10-20-second-round"

ECONOMY_NOTE = (
    "Ces séries n'entrent pas dans la prévision. L'étude du 20 octobre 2026 "
    "(docs/05-economie.md) ne trouve aucune variable économique qui fasse bouger les "
    "intentions de vote par bloc de façon distinguable du hasard, et la popularité du "
    "président est liée aux sondages sans les précéder. Elles sont montrées à titre explicatif."
)
PRESS_NOTE = (
    "La presse n'entre pas dans la prévision : le volet presse (part de voix, ton, agenda) "
    "arrive en phase 7, et une variable n'est admise qu'après un gain au backtest."
)
UNIT_LABELS = {
    "percent": "%",
    "persons": "personnes",
    "index_2025_100": "indice 100 = 2025",
    "index_mean_100": "indice, moyenne = 100",
    "balance_points": "solde, points",
    "eur_per_litre": "€ par litre",
}
FREQUENCY = {"M": "monthly", "Q": "quarterly", "W": "weekly"}
SOURCE_LABELS = {
    "insee": "Insee",
    "dares": "Dares",
    "mte": "Ministère de la Transition écologique",
    "ifop": "Ifop",
}
FIRST_ROUND_MODELS = {
    "model": ("Modèle + η", "model",
              ("Agrégation bayésienne des sondages (effets maison, plancher d'erreur), projection "
              "au vote par le modèle d'erreur η calibré sans l'élection évaluée.")),
    "mean14": ("Moyenne 14 jours + η", "baseline",
               "Moyenne simple des sondages des 14 derniers jours, même modèle d'erreur η."),
    "last_pollster": ("Dernier par institut + η", "baseline",
                      "Dernier sondage de chaque institut, moyenné, même modèle d'erreur η."),
    "model_noerror": ("Modèle sans η", "ablation",
                      ("Modèle de la phase 2 : projection par la seule marche aléatoire, sans "
                      "modèle d'erreur.")),
}
SECOND_ROUND_MODELS = {
    "model": ("2d tour : reports + sondages", "model",
              ("Second tour par reports de voix et sondages de duel (biais RN corrigé). Seuls "
              "le log-score (log P du vainqueur réel, duel réel connu) et la couverture du score "
              "du vainqueur sont remplis. La veille du second tour n'est pas représentée ici "
              "(horizon négatif) : erreur absolue 1,6 point (docs/backtest-log.md).")),
    "model_no_bias": ("2d tour sans biais RN", "ablation",
                      "Même modèle sans correction du biais d'extrême droite des sondages de duel."),
    "polls_14d": ("2d tour : sondages 14 jours", "baseline",
                  "Moyenne des sondages de duel des 14 derniers jours."),
    "transfers_only": ("2d tour : reports seuls", "baseline",
                       "Reports de voix historiques, sans sondages de duel."),
}


# --- utilitaires


def _r(x: float | None, nd: int = 1) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def _interval(row: dict, nd: int = 1, scale: float = 1.0, prefix: str = "") -> dict:
    q = [float(row[f"{prefix}{k}"]) * scale for k in QNAMES]
    q = np.maximum.accumulate(q)  # quantiles croissants malgré les arrondis flottants
    return {k: round(float(v), nd) for k, v in zip(QNAMES, q)}


def _write(out: Path, rel: str, obj) -> None:
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True,
                           check=True)
        return r.stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def clean_commit() -> str | None:
    """Commit courant si le code du back et les registres sont sans modification, sinon None."""
    head = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain", "--", "back/src", "data/registry")
    if head is None or status is None or status.strip():
        return None
    return head.strip()


def run_date_of(run: Path, manifest: dict) -> str:
    name = run.name
    try:
        return dt.date.fromisoformat(name[:10]).isoformat()
    except ValueError:
        return manifest["created_at"][:10]


# --- candidats


def candidate_entry(cid: str, reg: Registry, election: int, bloc_hint: dict[str, str]) -> dict:
    if cid.startswith("placeholder_"):
        bloc = cid.removeprefix("placeholder_")
        names = {
            "centre": ("Autre candidat du bloc central", "Autre (centre)"),
        }
        name, short = names.get(bloc, (f"Autre candidat ({bloc})", f"Autre ({bloc})"))
        return {"id": cid, "name": name, "short_name": short, "bloc": bloc, "party": None,
                "is_placeholder": True}
    c = reg.candidates[cid]
    bloc = bloc_hint.get(cid)
    if bloc is None:
        try:
            bloc = reg.bloc_of(cid, election)
        except KeyError:
            bloc = "other"
            for y in sorted(reg.elections, reverse=True):
                try:
                    bloc = reg.bloc_of(cid, y)
                    break
                except KeyError:
                    continue
    return {
        "id": cid,
        "name": c.name,
        "short_name": c.last_name,
        "bloc": bloc,
        "party": c.party_at(election),
        "is_placeholder": False,
    }


# --- conditions


def load_conditions(run: Path) -> list[dict]:
    return json.loads((run / "conditions.json").read_text())


def condition_json(c: dict) -> dict:
    return {
        "id": c["id"],
        "kind": c["kind"],
        "label": c["label"],
        "states": c["states"],
        "probability": round(float(c["probability"]), 4),
    }


def compatible(candidates: Iterable[str], states: dict[str, str], f: ScenariosFile) -> bool:
    """Règle de docs/08 : l'hypothèse contient les candidats de chaque état de la condition et
    aucun candidat d'un autre état de la même dimension."""
    present = set(candidates)
    for dim_id, state_id in states.items():
        d = f.dimension(dim_id)
        need = set(d.state(state_id).candidates)
        others = d.candidate_ids() - need
        if not need <= present or present & others:
            return False
    return True


# --- fichiers


def outcome_set(summary: pl.DataFrame, cond_id: str) -> dict:
    sub = summary.filter(pl.col("condition_id") == cond_id).sort(
        ["p_win", "p_qualify", "q50"], descending=True
    )
    cands = []
    for r in sub.iter_rows(named=True):
        cands.append(
            {
                "candidate_id": r["candidate_id"],
                "p_present": round(r["p_present"], 4),
                "p_qualify": round(r["p_qualify"], 4),
                "p_win": round(r["p_win"], 4),
                "first_round_share": _interval(r),
            }
        )
    return {"condition_id": cond_id, "candidates": cands}


def build_summary(run_manifest: dict, summary: pl.DataFrame, conditions: list[dict]) -> dict:
    return {
        "asof": run_manifest["asof"],
        "n_draws": run_manifest["n_draws"],
        "conditions": [condition_json(c) for c in conditions],
        "mixture": outcome_set(summary, "mix"),
        "by_condition": [outcome_set(summary, c["id"]) for c in conditions if c["id"] != "mix"],
    }


def build_first_round(
    run_manifest: dict,
    summary_json: dict,
    traj: pl.DataFrame,
    conditions: list[dict],
    polls_r1: pl.DataFrame,
    f: ScenariosFile,
) -> dict:
    asof = dt.date.fromisoformat(run_manifest["asof"])
    dates = (
        traj.filter((pl.col("condition_id") == "mix") & (pl.col("date") <= asof))
        .select("week", "date")
        .unique()
        .sort("week")
    )
    traj = traj.join(dates.select("week"), on="week")
    n_weeks = dates.height
    outcomes = {o["condition_id"]: o for o in [summary_json["mixture"], *summary_json["by_condition"]]}
    series = []
    for c in conditions:
        o = outcomes[c["id"]]
        kept = [x for x in o["candidates"] if x["p_present"] >= SERIES_MIN_PRESENCE]
        if c["id"] != "mix":
            kept = kept[:SERIES_MAX_CANDIDATES]
        t = traj.filter(pl.col("condition_id") == c["id"])
        for x in kept:
            tc = t.filter(pl.col("candidate_id") == x["candidate_id"]).sort("week")
            if tc.height != n_weeks:
                continue
            tracks = {}
            for k in QNAMES:
                tracks[k] = [_r(v) for v in tc[k].to_list()]
            # quantiles croissants semaine par semaine malgré l'arrondi
            for w in range(n_weeks):
                vals = [tracks[k][w] for k in QNAMES]
                if vals[0] is not None:
                    acc = np.maximum.accumulate(vals)
                    for k, v in zip(QNAMES, acc):
                        tracks[k][w] = round(float(v), 1)
            series.append(
                {
                    "condition_id": c["id"],
                    "candidate_id": x["candidate_id"],
                    **tracks,
                    "election_day": x["first_round_share"],
                }
            )
    other_conds = [c for c in conditions if c["id"] != "mix"]
    polls = []
    for (poll_id, hyp_id), g in polls_r1.group_by(["poll_id", "hypothesis_id"], maintain_order=True):
        ids = g["candidate_id"].to_list()
        r0 = g.row(0, named=True)
        polls.append(
            {
                "poll_id": poll_id,
                "hypothesis_id": hyp_id,
                "pollster": r0["pollster"],
                "fieldwork_end": r0["fieldwork_end"].isoformat(),
                "population": r0["population"] or "unknown",
                "conditions": [c["id"] for c in other_conds if compatible(ids, c["states"], f)],
                "shares": {cid: _r(s) for cid, s in zip(ids, g["share"].to_list())},
            }
        )
    return {
        "asof": run_manifest["asof"],
        "election_date": run_manifest["first_round"],
        "dates": [d.isoformat() for d in dates["date"].to_list()],
        "conditions": [condition_json(c) for c in conditions],
        "series": series,
        "polls": polls,
    }


def build_second_round(
    run_manifest: dict,
    pairs: pl.DataFrame,
    transfers: pl.DataFrame,
    draws: pl.DataFrame,
    polls_r2: pl.DataFrame,
    bloc_of: dict[str, str],
) -> dict:
    mix = pairs.filter((pl.col("condition_id") == "mix") & (pl.col("p_pair") >= PAIR_MIN_PROBABILITY))
    out_pairs, out_transfers = [], []
    orient = {}
    for r in mix.sort("p_pair", descending=True).iter_rows(named=True):
        a, b = r["candidate_a"], r["candidate_b"]
        orient[frozenset((a, b))] = (a, b)
        out_pairs.append(
            {
                "candidate_a": a,
                "candidate_b": b,
                "p_pair": round(r["p_pair"], 4),
                "p_a_wins": round(r["p_a_wins"], 4),
                "share_a": _interval(r),
            }
        )
        if r["p_pair"] < TRANSFER_MIN_PAIR:
            continue
        sub = draws.filter((pl.col("qualified_a") == a) & (pl.col("qualified_b") == b))
        tr = transfers.filter((pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
        groups = {t["group"]: t for t in tr.iter_rows(named=True)}
        cands = [c for c in bloc_of if c in sub.columns and c not in (a, b)]
        means = {c: float(sub[c].fill_null(0.0).mean()) for c in cands}
        for c in sorted(cands, key=lambda x: -means[x]):
            if means[c] < TRANSFER_MIN_SHARE:
                continue
            g = groups.get(f"bloc:{bloc_of[c]}")
            if g is None:
                continue
            out_transfers.append(
                {
                    "candidate_a": a,
                    "candidate_b": b,
                    "from": c,
                    "to_a": _interval(g, 0, 100.0, "to_a_"),
                    "to_b": _interval(g, 0, 100.0, "to_b_"),
                    "to_none": _interval(g, 0, 100.0, "none_"),
                }
            )
    polls = []
    for (poll_id, _hyp), g in polls_r2.group_by(["poll_id", "hypothesis_id"], maintain_order=True):
        if g.height != 2:
            continue
        x, y = g["candidate_id"].to_list()
        sx, sy = g["share"].to_list()
        a, b = orient.get(frozenset((x, y))) or (
            (x, y) if bloc_of.get(x) == "far_right" or sx >= sy else (y, x)
        )
        sa = sx if a == x else sy
        r0 = g.row(0, named=True)
        polls.append(
            {
                "poll_id": poll_id,
                "pollster": r0["pollster"],
                "fieldwork_end": r0["fieldwork_end"].isoformat(),
                "candidate_a": a,
                "candidate_b": b,
                "share_a": round(100 * sa / (sx + sy), 1),
            }
        )
    polls.sort(key=lambda p: p["fieldwork_end"], reverse=True)
    return {
        "asof": run_manifest["asof"],
        "election_date": run_manifest["second_round"],
        "pairs": out_pairs,
        "transfers": out_transfers,
        "polls": polls,
    }


def _source_ref(s: str, refs: dict[str, str]) -> dict:
    if s in refs:
        return {"label": s, "href": refs[s]}
    try:
        host = urlparse(s).hostname
    except ValueError:
        host = None
    return {"label": (host or s).removeprefix("www."), "href": s}


def scenario_history(f: ScenariosFile) -> list[dict]:
    """Probabilités des états à chaque version du registre de scénarios : versions commitées
    (git) puis version de travail ; une entrée par `as_of`, la plus récente l'emporte."""
    import yaml

    def probs(data: dict) -> dict[str, float]:
        return {
            f"{d['id']}.{s['id']}": float(s["probability"])
            for d in data["dimensions"]
            for s in d["states"]
        }

    points: dict[str, dict[str, float]] = {}
    rel = "data/registry/scenarios_2027.yaml"
    revs = (_git("log", "--format=%H", "--reverse", "--", rel) or "").split()
    for h in revs:
        txt = _git("show", f"{h}:{rel}")
        if not txt:
            continue
        try:
            data = yaml.safe_load(txt)
            points[str(data["as_of"])] = probs(data)
        except (yaml.YAMLError, KeyError, TypeError):
            continue
    points[f.as_of.isoformat()] = {
        f"{d.id}.{s.id}": s.probability for d in f.dimensions for s in d.states
    }
    return [{"date": k, "probabilities": v} for k, v in sorted(points.items())]


def build_scenarios(f: ScenariosFile) -> dict:
    refs = f.references
    enum = enumerate_configurations(f, small="marginal")
    dims = [d.id for d in f.dimensions]
    configs = []
    for c in list(enum)[:SCENARIO_CONFIGS]:
        states = dict(c.states)
        configs.append(
            {
                "condition_id": "cfg." + "-".join(states[d] for d in dims),
                "states": states,
                "probability": round(c.probability, 4),
                "candidates": sorted(c.candidates),
                "placeholders": list(c.placeholders),
                "small_presence": {k: round(v, 4) for k, v in c.small_presence.items()},
            }
        )
    return {
        "as_of": f.as_of.isoformat(),
        "election": f.election,
        "dimensions": [
            {
                "id": d.id,
                "question": d.question,
                "review": [str(x) for x in d.review],
                "states": [
                    {
                        "id": s.id,
                        "probability": s.probability,
                        "candidates": list(s.candidates),
                        "placeholders": list(s.placeholders),
                        "description": s.description,
                        "sources": [_source_ref(x, refs) for x in s.sources],
                        "condition_id": f"{d.id}.{s.id}",
                    }
                    for s in d.states
                ],
            }
            for d in f.dimensions
        ],
        "small_candidates": [
            {
                "candidate_id": s.candidate_id,
                "probability": s.probability,
                "bloc": s.bloc,
                "rationale": s.rationale,
                "sources": [_source_ref(x, refs) for x in s.sources],
            }
            for s in f.small_candidates
        ],
        "constraints": [
            {
                "id": c.id,
                "when": dict(c.when),
                "absent": list(c.absent),
                "impossible": c.impossible,
                "rationale": c.rationale,
            }
            for c in f.constraints
        ],
        "revisions": [
            {"date": r.date.isoformat(), "author": r.author, "summary": r.summary,
             "rationale": r.rationale}
            for r in sorted(f.revisions, key=lambda r: r.date, reverse=True)
        ],
        "configurations": configs,
        "n_configurations": len(enum),
        "impossible_mass": round(enum.impossible_mass, 4),
        "probability_history": scenario_history(f),
    }


def build_polls(
    run_manifest: dict, polls: pl.DataFrame, results: pl.DataFrame, house: pl.DataFrame,
    reg: Registry,
) -> dict:
    out = []
    for r in polls.sort("fieldwork_end", "poll_id", descending=[True, False]).iter_rows(named=True):
        res = results.filter(pl.col("poll_id") == r["poll_id"])
        hyps = []
        for (hyp_id, rnd), g in res.group_by(["hypothesis_id", "round"], maintain_order=True):
            g = g.sort("share", descending=True)
            if g.height < 2:
                continue
            hyps.append(
                {
                    "hypothesis_id": hyp_id,
                    "round": int(rnd),
                    "population": g["population"][0] or "unknown",
                    "shares": [
                        {"candidate_id": c, "share": _r(s)}
                        for c, s in zip(g["candidate_id"].to_list(), g["share"].to_list())
                    ],
                }
            )
        if not hyps:
            continue
        size = r["sample_size"]
        out.append(
            {
                "poll_id": r["poll_id"],
                "pollster": r["pollster"],
                "sponsor": r["sponsor"],
                "fieldwork_start": r["fieldwork_start"].isoformat(),
                "fieldwork_end": r["fieldwork_end"].isoformat(),
                "publish_date": r["publish_date"].isoformat() if r["publish_date"] else None,
                "sample_size": int(size) if size and size > 0 else None,
                "mode": r["mode"] if r["mode"] in ("online", "phone", "mixed") else "unknown",
                "source_url": r["source_url"],
                "hypotheses": hyps,
            }
        )
    pollsters = sorted({p["pollster"] for p in out} | set(house["pollster"].to_list()))
    return {
        "asof": run_manifest["asof"],
        "pollsters": [
            {"id": p, "name": reg.pollsters[p].name if p in reg.pollsters else p} for p in pollsters
        ],
        "polls": out,
        "house_effects": [
            {
                "pollster": h["pollster"],
                "bloc": h["bloc"],
                "effect": _interval(h, 2),
                "n_polls": int(h["n_polls"]),
            }
            for h in house.sort("pollster", "bloc").iter_rows(named=True)
        ],
    }


def build_economy(asof: str) -> dict:
    import yaml

    reg = yaml.safe_load((paths.REGISTRY / "macro_series.yaml").read_text(encoding="utf-8"))
    macro = pl.read_parquet(paths.CURATED / "macro.parquet")
    series = []
    for s in reg["series"]:
        sub = macro.filter(
            (pl.col("series_id") == s["series_id"])
            & (pl.col("date") >= ECONOMY_SINCE)
            & (pl.col("date") <= dt.date.fromisoformat(asof))
        ).sort("date")
        if sub.is_empty():
            continue
        series.append(
            {
                "id": s["series_id"],
                "label": s["label"],
                "unit": UNIT_LABELS.get(s["unit"], s["unit"]),
                "frequency": FREQUENCY[s["freq"]],
                "source": {"label": SOURCE_LABELS.get(s["source"], s["source"]), "href": s["url"]},
                "in_forecast": False,
                "points": [
                    {"date": d.isoformat(), "value": round(float(v), 3)}
                    for d, v in zip(sub["date"].to_list(), sub["value"].to_list())
                ],
            }
        )
    return {
        "asof": asof,
        "in_forecast": False,
        "in_forecast_note": ECONOMY_NOTE,
        "series": series,
        "correlations": [],
    }


def build_press(asof: str) -> dict:
    return {
        "asof": asof,
        "in_forecast": False,
        "in_forecast_note": PRESS_NOTE,
        "weeks": [],
        "share_of_voice": [],
        "tone": [],
        "agenda": [],
    }


def build_backtest(generated_at: str, first: Path, second: Path) -> dict:
    fr = pl.read_parquet(first / "metrics.parquet").filter(pl.col("level") == "summary")
    variant = "floor" if "floor" in fr["variant"].unique().to_list() else fr["variant"][0]
    fr = fr.filter(pl.col("variant") == variant)
    sr = pl.read_parquet(second / "metrics.parquet").filter(pl.col("horizon_days") >= 0)
    rows = []
    metric = {
        "log_score_pair": "log_score_pair",
        "brier_qualify": "brier_qual",
        "crps_first_round": "crps",
        "mae_first_round": "mae",
        "coverage_50": "cov50",
        "coverage_90": "cov90",
    }
    for (model, e, h), g in fr.group_by(["model", "election", "horizon"], maintain_order=True):
        if model not in FIRST_ROUND_MODELS:
            continue
        vals = dict(zip(g["metric"].to_list(), g["value"].to_list()))
        row = {"model_id": model, "election": int(e), "horizon_days": int(h),
               "log_score_winner": None}
        for k, m in metric.items():
            v = vals.get(m)
            row[k] = None if v is None or not np.isfinite(v) else round(float(v), 4)
        rows.append(row)
    for r in sr.iter_rows(named=True):
        if r["method"] not in SECOND_ROUND_MODELS:
            continue
        rows.append(
            {
                "model_id": f"r2_{r['method']}",
                "election": int(r["election"]),
                "horizon_days": int(r["horizon_days"]),
                "log_score_pair": None,
                "log_score_winner": _r(r["log_score"], 4),
                "brier_qualify": None,
                "crps_first_round": None,
                "mae_first_round": None,
                "coverage_50": None if r["cover50"] is None else float(r["cover50"]),
                "coverage_90": None if r["cover90"] is None else float(r["cover90"]),
            }
        )
    rows.sort(key=lambda x: (x["horizon_days"], x["election"], x["model_id"]))
    models = [
        {"id": k, "label": lab, "kind": kind, "description": desc}
        for k, (lab, kind, desc) in FIRST_ROUND_MODELS.items()
    ] + [
        {"id": f"r2_{k}", "label": lab, "kind": kind, "description": desc}
        for k, (lab, kind, desc) in SECOND_ROUND_MODELS.items()
    ]
    return {
        "generated_at": generated_at,
        "model_version": MODEL_VERSION,
        "elections": sorted({r["election"] for r in rows}),
        "horizons_days": sorted({r["horizon_days"] for r in rows}, reverse=True),
        "models": models,
        "rows": rows,
    }


# --- sondages intégrés


def integrated_polls(run_manifest: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(polls, résultats longs des deux tours, hypothèses de premier tour de la grille) des
    sondages intégrés au run : premier tour dans la grille de l'ajustement, second tour dans la
    fenêtre du modèle de duel (`MAX_POLL_AGE`)."""
    from vvs.model.data import load_polls
    from vvs.model.second_round import MAX_POLL_AGE

    election = run_manifest["election"]
    asof = dt.date.fromisoformat(run_manifest["asof"])
    fit_dir = Path(run_manifest["fit"]["dir"])
    observed = pl.read_parquet(fit_dir / "observed.parquet").select("poll_id", "hypothesis_id").unique()
    r1 = load_polls(election, asof, round_=1).join(observed, on=["poll_id", "hypothesis_id"])
    r1 = r1.with_columns(pl.lit(1, dtype=pl.Int8).alias("round"))
    gap = (dt.date.fromisoformat(run_manifest["second_round"])
           - dt.date.fromisoformat(run_manifest["first_round"])).days
    oldest = asof - dt.timedelta(days=MAX_POLL_AGE - run_manifest["horizon_days"] - gap)
    r2 = load_polls(election, asof, round_=2).filter(pl.col("fieldwork_end") >= oldest)
    r2 = r2.with_columns(pl.lit(2, dtype=pl.Int8).alias("round"))
    results = pl.concat([r1, r2])
    polls = pl.read_parquet(paths.CURATED / "polls.parquet").filter(
        pl.col("poll_id").is_in(results["poll_id"].unique().to_list())
    )
    return polls, results, r1


# --- export


def export_run(
    run: Path,
    out: Path,
    registry: Registry | None = None,
    scenarios: ScenariosFile | None = None,
    backtest_first: Path | None = None,
    backtest_second: Path | None = None,
    polls_data: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None = None,
    export_configs: int = EXPORT_CONFIGS,
) -> dict:
    """Écrit les fichiers du contrat JSON dans `out` ; rend un petit compte rendu."""
    run, out = Path(run), Path(out)
    reg = registry or default_registry()
    f = scenarios or load_scenarios(registry=reg)
    rm = json.loads((run / "manifest.json").read_text())
    summary = pl.read_parquet(run / "summary.parquet")
    pairs = pl.read_parquet(run / "pairs.parquet")
    transfers = pl.read_parquet(run / "transfers.parquet")
    traj = pl.read_parquet(run / "trajectories.parquet")
    house = pl.read_parquet(run / "house_effects.parquet")
    draws = pl.read_parquet(run / "draws.parquet")
    run_date = run_date_of(run, rm)
    generated_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    all_conditions = load_conditions(run)
    have = set(summary["condition_id"].unique().to_list())
    all_conditions = [c for c in all_conditions if c["id"] in have]
    cfg_conditions = [c for c in all_conditions if c["kind"] == "configuration"]
    conditions = [c for c in all_conditions if c["kind"] != "configuration"]
    conditions += cfg_conditions[:export_configs]
    fr_conditions = [c for c in all_conditions if c["kind"] != "configuration"]
    fr_conditions += cfg_conditions[:FIRST_ROUND_CONFIGS]

    polls, results, r1 = polls_data or integrated_polls(rm)
    r2 = results.filter(pl.col("round") == 2)
    bloc_hint = {u["candidate_id"]: u["bloc"] for u in rm["universe"]}

    summary_json = build_summary(rm, summary, conditions)
    first_round = build_first_round(rm, summary_json, traj, fr_conditions, r1, f)
    second_round = build_second_round(rm, pairs, transfers, draws, r2, bloc_hint)
    scenarios_json = build_scenarios(f)
    polls_json = build_polls(rm, polls, results, house, reg)
    economy = build_economy(rm["asof"])
    press = build_press(rm["asof"])
    backtest = build_backtest(
        generated_at,
        backtest_first or paths.RUNS / BACKTEST_FIRST_ROUND,
        backtest_second or paths.RUNS / BACKTEST_SECOND_ROUND,
    )

    cited = set(bloc_hint)
    cited |= set(results["candidate_id"].unique().to_list())
    for p in second_round["pairs"]:
        cited |= {p["candidate_a"], p["candidate_b"]}
    cited |= {t["from"] for t in second_round["transfers"]}
    candidates = [candidate_entry(c, reg, rm["election"], bloc_hint) for c in sorted(cited)]

    last_poll = max(p["fieldwork_end"] for p in polls_json["polls"])
    commit = clean_commit()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "generated_at": generated_at,
        "asof": rm["asof"],
        "model_version": MODEL_VERSION,
        "commit": commit,
        "last_poll_date": last_poll,
        "n_polls": len(polls_json["polls"]),
        "n_draws": rm["n_draws"],
        "mock": False,
        "election": {
            "year": rm["election"],
            "first_round": rm["first_round"],
            "second_round": rm["second_round"],
        },
        "candidates": candidates,
    }

    # historique : on retire les entrées factices, on n'écrase jamais une entrée réelle
    hist_dir = out / "history"
    old_manifest = out / "manifest.json"
    if old_manifest.exists():
        try:
            was_mock = json.loads(old_manifest.read_text()).get("mock", False)
        except json.JSONDecodeError:
            was_mock = False
        if was_mock and hist_dir.exists():
            for p in hist_dir.glob("*.json"):
                p.unlink()
    entry = {
        "run_date": run_date,
        "asof": rm["asof"],
        "model_version": MODEL_VERSION,
        "last_poll_date": last_poll,
        "n_polls": manifest["n_polls"],
        "candidates": [
            {k: c[k] for k in ("candidate_id", "p_qualify", "p_win", "first_round_share")}
            for c in summary_json["mixture"]["candidates"]
        ],
    }
    hist_file = hist_dir / f"{run_date}.json"
    history_written = False
    if hist_file.exists():
        if json.loads(hist_file.read_text()) != entry:
            log.warning("%s existe déjà et n'est pas réécrit", hist_file)
    else:
        _write(out, f"history/{run_date}.json", entry)
        history_written = True

    _write(out, "manifest.json", manifest)
    _write(out, "summary.json", summary_json)
    _write(out, "first_round.json", first_round)
    _write(out, "second_round.json", second_round)
    _write(out, "scenarios.json", scenarios_json)
    _write(out, "polls.json", polls_json)
    _write(out, "economy.json", economy)
    _write(out, "press.json", press)
    _write(out, "backtest.json", backtest)
    entries = [
        json.loads(p.read_text())
        for p in sorted(hist_dir.glob("*.json"))
        if p.name != "index.json"
    ]
    _write(out, "history/index.json", {"entries": sorted(entries, key=lambda e: e["run_date"])})
    sizes = {p.name: p.stat().st_size for p in out.glob("*.json")}
    return {
        "out": str(out),
        "run_date": run_date,
        "commit": commit,
        "n_polls": manifest["n_polls"],
        "n_candidates": len(candidates),
        "history_written": history_written,
        "history_entries": len(entries),
        "sizes": sizes,
    }
