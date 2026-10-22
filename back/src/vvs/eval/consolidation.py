"""Référence immuable et backtest chronologique jusqu'au vainqueur (docs/10).

Entrée : python -m vvs.eval.consolidation {freeze,run,report,verify} --out runs/...
Les sous-processus utilisent exclusivement la copie du code et des données du gel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import polars as pl
import yaml

from vvs import paths

REFERENCE = "data/registry/reference_2026-10-22.yaml"
TABLES = (
    "polls.parquet",
    "poll_results.parquet",
    "poll_errors.parquet",
    "jw_polls.parquet",
    "poll_errors_lists.parquet",
    "polls_lists.parquet",
    "poll_results_lists.parquet",
    "results_lists_by_bloc.parquet",
)
PRESIDENTIALS = (2002, 2007, 2012, 2017, 2022)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def load_spec() -> dict:
    return yaml.safe_load((paths.ROOT / REFERENCE).read_text())


def freeze(out: Path) -> None:
    """Archive Git du code/données suivis, plus les tirages publiés ; jamais d'écrasement."""
    if (out / "reference.json").exists():
        verify(out)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"gel incomplet ou dossier non vide : {out}")
    spec = load_spec()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=paths.ROOT, text=True
    ).strip()
    members = ["back/src/vvs", "back/uv.lock", "back/pyproject.toml", "data/registry"]
    members += [f"data/curated/{name}" for name in TABLES]
    # Refuse un gel trompeur : les sources exécutées doivent être celles du commit archivé.
    dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", *members], cwd=paths.ROOT, text=True
    ).splitlines()
    relevant_dirty = [p for p in dirty if p != "back/src/vvs/cli.py"]
    if relevant_dirty:
        raise ValueError(f"committer les entrées du gel d'abord : {relevant_dirty}")
    archive = subprocess.check_output(["git", "archive", commit, *members], cwd=paths.ROOT)
    source = out / "source"
    source.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(source, filter="data")
    published = out / "published"
    published.mkdir()
    for name in ("manifest.json", "draws.parquet", "summary.parquet"):
        shutil.copy2(paths.ROOT / spec["published_run"] / name, published / name)
    run = json.loads((published / "manifest.json").read_text())
    fit = json.loads((paths.ROOT / spec["published_fit"] / "manifest.json").read_text())
    if fit["parameters"]["priors"] != spec["priors"]:
        raise ValueError("priors de la référence différents du run publié")
    if run["second_round_model"]["bias_mode"] != "curve":
        raise ValueError("le run publié n'utilise pas la courbe de biais attendue")
    if run["scenarios"]["sha256"] != digest(source / "data/registry/scenarios_2027.yaml")[:16]:
        raise ValueError("le registre de scénarios diffère de celui du run publié")
    shutil.copy2(
        paths.ROOT / spec["published_fit"] / "manifest.json", published / "fit_manifest.json"
    )
    files = {
        str(p.relative_to(out)): digest(p)
        for base in (source, published)
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }
    from vvs.model.simulate import _versions

    write_json(
        out / "reference.json",
        {
            "id": spec["id"],
            "commit": commit,
            "spec": spec,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "versions": _versions(),
            "files": files,
            "ignored_worktree_changes": dirty,
        },
    )
    print(f"Référence figée : {out} ({len(files)} fichiers)", flush=True)


def verify(out: Path) -> dict:
    manifest = json.loads((out / "reference.json").read_text())
    bad = [
        rel
        for rel, sha in manifest["files"].items()
        if not (out / rel).is_file() or digest(out / rel) != sha
    ]
    if bad:
        raise ValueError(f"référence modifiée ou incomplète : {bad}")
    return manifest


def available_polls(polls: pl.DataFrame, asof: dt.date, delay: int) -> pl.DataFrame:
    """Date connue prioritaire ; délai explicite seulement pour les publications manquantes."""
    if delay < 0:
        raise ValueError("délai de publication négatif")
    if "publish_date" not in polls.columns:
        raise ValueError("colonne publish_date requise, même si ses valeurs sont nulles")
    available = pl.coalesce(
        pl.col("publish_date"), pl.col("fieldwork_end") + pl.duration(days=delay)
    )
    return polls.filter((pl.col("fieldwork_end") <= asof) & (available <= asof))


def historical_configurations(
    df: pl.DataFrame, election: int, asof: dt.date, registry, settings: dict
) -> tuple[list[dict], dict]:
    """Aucun rapprochement a posteriori avec les candidats finalement présents."""
    from vvs.eval.configurations import list_publication

    if asof >= list_publication(election, registry):
        candidates = sorted(c.candidate_id for c in registry.elections[election].candidates)
        return [{"candidates": candidates, "probability": 1.0}], {"source": "official_list"}
    window = settings["window_days"]
    while True:
        recent = df.filter(pl.col("fieldwork_end") > asof - dt.timedelta(days=window))
        if recent.height or window >= settings["max_window_days"]:
            break
        window += settings["window_days"]
    if recent.is_empty():
        raise ValueError(f"aucune configuration disponible pour {election} au {asof}")
    hyps: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for (pid, _), grp in recent.group_by("poll_id", "hypothesis_id"):
        candidates = tuple(sorted(set(grp["candidate_id"])))
        for cid in candidates:
            registry.bloc_of(cid, election)  # échec explicite si inconnu
        if len(candidates) >= 2:
            hyps[pid].add(candidates)
    mass: dict[tuple, float] = defaultdict(float)
    for configs in hyps.values():
        for c in configs:
            mass[c] += 1 / len(hyps) / len(configs)
    configs = [{"candidates": list(c), "probability": p} for c, p in sorted(mass.items())]
    return configs, {"source": "poll_hypotheses", "window_days": window, "n_polls": len(hyps)}


def prepare_case(out: Path, election: int, horizon: int, spec: dict) -> tuple[Path, dict]:
    from vvs.model.data import load_polls
    from vvs.registry import default_registry

    reg = default_registry()
    asof = reg.elections[election].round_(1).date - dt.timedelta(days=horizon)
    folder = out / "cases" / f"{election}-{horizon:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "case.json").exists():
        return folder, json.loads((folder / "case.json").read_text())
    raw = pl.read_parquet(paths.CURATED / "polls.parquet").filter(pl.col("election") == election)
    audit = []
    for delay in spec["availability"]["audit_delays"]:
        p = available_polls(raw, asof, delay)
        audit.append(
            {
                "delay_days": delay,
                "n_polls": p.height,
                "missing_publish_date": p["publish_date"].null_count(),
            }
        )
    p = available_polls(raw, asof, spec["availability"]["missing_publication_delay_days"])
    curated = folder / "curated"
    curated.mkdir(exist_ok=True)
    p.write_parquet(curated / "polls.parquet")
    results = pl.read_parquet(paths.CURATED / "poll_results.parquet").filter(
        pl.col("poll_id").is_in(p["poll_id"].implode())
    )
    results.write_parquet(curated / "poll_results.parquet")
    df = load_polls(election, asof, curated=curated)
    configs, info = historical_configurations(df, election, asof, reg, spec["historical_scenarios"])
    data = {
        "election": election,
        "horizon": horizon,
        "asof": str(asof),
        "configurations": configs,
        "scenario_rule": info,
        "availability_audit": audit,
        "input_hashes": {p.name: digest(p) for p in sorted(curated.glob("*.parquet"))},
    }
    write_json(folder / "case.json", data)
    return folder, data


def check_case(folder: Path, case: dict) -> None:
    for name, sha in case["input_hashes"].items():
        if digest(folder / "curated" / name) != sha:
            raise ValueError(f"entrée du cas modifiée : {folder}/{name}")


def fit_case(out: Path, election: int, horizon: int) -> None:
    from vvs.model.data import load_polls
    from vvs.model.first_round import Priors
    from vvs.model.run import run_fit

    spec = load_spec()
    folder, case = prepare_case(out, election, horizon, spec)
    check_case(folder, case)
    fit_dir = folder / "fit"
    if (fit_dir / "complete.json").exists():
        done = json.loads((fit_dir / "complete.json").read_text())
        for name, sha in done.items():
            if digest(fit_dir / name) != sha:
                raise ValueError(f"ajustement modifié : {fit_dir}/{name}")
        return
    df = load_polls(election, case["asof"], curated=folder / "curated")
    candidates = sorted({c for cfg in case["configurations"] for c in cfg["candidates"]})
    manifest = run_fit(
        election,
        dt.date.fromisoformat(case["asof"]),
        fit_dir,
        **spec["fit"],
        seed=spec["seed"],
        reference=candidates,
        priors=Priors(**spec["priors"]),
        save_draws=True,
        figures=False,
        polls=df,
    )
    diag = manifest["diagnostics"]
    share = diag["derived"]["share_asof"]
    limits = spec["diagnostics"]
    acceptable = (
        diag["divergences"] <= limits["max_divergences"]
        and diag["r_hat_max"] <= limits["r_hat_max"]
        and share["r_hat_max"] <= limits["r_hat_max"]
        and share["ess_min"] >= limits["share_ess_min"]
    )
    write_json(fit_dir / "quality.json", {"acceptable": acceptable, "diagnostics": diag})
    write_json(
        fit_dir / "complete.json",
        {name: digest(fit_dir / name) for name in ("draws.npz", "manifest.json", "quality.json")},
    )
    print(f"Ajustement {election}/{horizon} : acceptable={acceptable}", flush=True)


def calibration(asof: dt.date, spec: dict):
    from vvs.curate.lists import vote_date
    from vvs.ingest.jennings_wlezien import load_polls as load_jw
    from vvs.model.error import ErrorModel, jw_shape
    from vvs.model.error_lists import LIST_KEYS, list_items
    from vvs.model.second_round import calibrate_errors
    from vvs.model.simulate import turnout_model
    from vvs.registry import default_registry
    from vvs.registry.transfers import default_transfers

    reg = default_registry()
    past = [e for e in PRESIDENTIALS if reg.elections[e].round_(2).date < asof]
    if len(past) < 2:
        raise ValueError("deux présidentielles antérieures requises pour la calibration")
    keys = tuple(k for k in LIST_KEYS if vote_date(k) < asof)
    extras = list_items(weight=spec["error"]["list_weight"], keys=keys) if keys else []
    jw = load_jw().filter(pl.col("election_date") < asof)
    shape = jw_shape(df=jw)
    errors = ErrorModel.fit(
        elections=past,
        gamma=shape["gamma"],
        extra_items=extras,
        use_bias=spec["error"]["use_bias"],
        nu=spec["error"]["nu"],
    )
    excluded = set(reg.elections) - set(past)
    errors2 = calibrate_errors(
        exclude=excluded, registry=reg, bias_by_horizon=spec["second_round"]["bias_by_horizon"]
    )
    transfers = tuple(
        s
        for s in default_transfers()
        if s.election in past and (s.fieldwork_end is None or s.fieldwork_end < asof)
    )
    turnout = turnout_model(reg, elections=past)
    audit = {
        "asof": str(asof),
        "presidentials": past,
        "list_elections": list(keys),
        "jw_max_election_date": str(jw["election_date"].max()),
        "jw_shape": shape,
        "transfer_sources": [s.id for s in transfers],
        "transfer_dates_missing": [s.id for s in transfers if s.fieldwork_end is None],
        "error_model": errors.to_dict(),
        "second_round": asdict(errors2),
        "turnout": turnout,
    }
    return errors, errors2, transfers, turnout, excluded, audit


def simulate_chain(
    case: dict,
    draws,
    error_model,
    model2,
    turnout: dict,
    spec: dict,
    polls: pl.DataFrame,
    method: str = "reference",
) -> tuple[dict, list[dict]]:
    """Les finalistes sont tirés du premier tour ; aucun résultat réel n'entre ici."""
    from vvs.eval.metrics import top2
    from vvs.model.run import draws_log_shares
    from vvs.model.simulate import allocate, sample_turnout
    from vvs.registry import default_registry

    reg = default_registry()
    rng = np.random.default_rng(spec["seed"])
    n = spec["n_draws"]
    configs = case["configurations"]
    candidates = sorted({c for cfg in configs for c in cfg["candidates"]})
    index = {c: i for i, c in enumerate(candidates)}
    shares = np.zeros((n, len(candidates)))
    config_ids = np.repeat(
        np.arange(len(configs)), allocate(np.array([cfg["probability"] for cfg in configs]), n)
    )
    for k, cfg in enumerate(configs):
        rows = np.flatnonzero(config_ids == k)
        if not len(rows):
            continue
        cands = cfg["candidates"]
        if method == "reference":
            logp = draws_log_shares(draws, cands)
            x = logp[rng.integers(0, len(logp), len(rows))]
        else:
            recent = polls.filter(
                pl.col("fieldwork_end")
                > dt.date.fromisoformat(case["asof"]) - dt.timedelta(days=14)
            )
            if recent.is_empty():
                recent = polls.filter(pl.col("fieldwork_end") == polls["fieldwork_end"].max())
            # Une mesure par sondage/candidat avant moyenne ; les hypothèses ne sont pas
            # autant de sondages indépendants. Les absents récents utilisent leurs mesures passées.
            means = recent.group_by("poll_id", "candidate_id").agg(pl.col("share").mean())
            means = dict(means.group_by("candidate_id").agg(pl.col("share").mean()).iter_rows())
            older = dict(polls.group_by("candidate_id").agg(pl.col("share").median()).iter_rows())
            p = np.array([max(means.get(c, older.get(c, 1.0)), 0.01) for c in cands])
            x = np.broadcast_to(np.log(p / p.sum()), (len(rows), len(cands))).copy()
        p = np.exp(x)
        eta = error_model.sample_eta(
            case["horizon"], [reg.bloc_of(c, case["election"]) for c in cands], p, rng
        )
        x = x + eta
        p = np.exp(x - x.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        shares[np.ix_(rows, [index[c] for c in cands])] = p * 100
    qualified = top2(shares)
    pair_codes, pair_idx = np.unique(qualified, axis=0, return_inverse=True)
    pairs = [(candidates[a], candidates[b]) for a, b in pair_codes]
    model2.fit(case["election"], case["asof"], pairs)
    tau = sample_turnout(turnout, rng, n)
    winner = np.full(n, -1, dtype=int)
    share2 = np.zeros(n)
    details = []
    # Comme la projection publiée, les reports sont conditionnels au scénario et au duel.
    for k in np.unique(config_ids):
        for code in np.unique(pair_idx[config_ids == k]):
            rows = np.flatnonzero((config_ids == k) & (pair_idx == code))
            pair = pairs[code]
            fr = {c: shares[rows, index[c]] / 100 for c in configs[k]["candidates"]}
            fr["abstain"] = (1 - tau[rows]) / tau[rows]
            second = model2.sample(pair, fr, case["horizon"], rng, n=len(rows))
            share2[rows] = second.share
            winner[rows] = np.where(second.share > 50, pair_codes[code, 0], pair_codes[code, 1])
            details.append(
                {
                    "config": int(k),
                    "candidate_a": pair[0],
                    "candidate_b": pair[1],
                    "n": len(rows),
                    "ess_ratio": second.ess / len(rows),
                    "n_duel_polls": model2.pair_fit(pair).polls.n,
                }
            )
    return {
        "candidates": np.array(candidates),
        "shares": shares,
        "qualified": qualified,
        "winner": winner,
        "share_second": share2,
        "config": config_ids,
    }, details


def score_chain(sim: dict, election: int, registry) -> tuple[dict, list[dict]]:
    """Score sans correspondance oracle : les candidats absents reçoivent zéro."""
    from vvs.eval import metrics as mt

    candidates = list(sim["candidates"])
    e = registry.elections[election]
    actual = {r.candidate_id: r.share for r in e.round_(1).results}
    universe = sorted(set(candidates) | set(actual))
    finals = sorted(r.candidate_id for r in e.round_(2).results)
    winner = max(e.round_(2).results, key=lambda r: r.share).candidate_id
    n = len(sim["winner"])
    qual = np.array(candidates)[sim["qualified"]]
    wins = np.array(candidates)[sim["winner"]]
    k_pair = int(np.sum(np.all(np.sort(qual, axis=1) == finals, axis=1)))
    k_win = int(np.sum(wins == winner))
    pwin = np.array([(wins == c).mean() for c in universe])
    pq = np.array([(qual == c).any(axis=1).mean() for c in universe])
    cols = {c: j for j, c in enumerate(candidates)}
    x = np.column_stack([sim["shares"][:, cols[c]] if c in cols else np.zeros(n) for c in actual])
    y = np.array(list(actual.values()))
    k = len(universe)
    summary = {
        "p_winner": k_win / n,
        "p_pair": k_pair / n,
        "log_score_winner": math.log((k_win + 0.5) / (n + 0.5 * k)),
        "log_score_pair": math.log((k_pair + 0.5) / (n + 0.5 * k * (k - 1) / 2)),
        "winner_outside_support": winner not in candidates,
        "pair_outside_support": any(c not in candidates for c in finals),
        "brier_win": float(np.sum((pwin - np.array([c == winner for c in universe])) ** 2)),
        "brier_qualify": mt.brier(pq, np.array([c in finals for c in universe])),
        "crps": float(mt.crps(x, y).mean()),
        "coverage50": float(mt.covered(x, y, 0.5).mean()),
        "coverage90": float(mt.covered(x, y, 0.9).mean()),
        "unforecast_share": sum(s for c, s in actual.items() if c not in candidates),
        "n_draws": n,
        "n_candidates": len(candidates),
        "n_scoring_candidates": k,
    }
    rows = [
        {
            "candidate_id": c,
            "p_win": float(pwin[i]),
            "p_qualify": float(pq[i]),
            "actual_winner": c == winner,
            "actual_qualified": c in finals,
            "in_support": c in candidates,
        }
        for i, c in enumerate(universe)
    ]
    return summary, rows


def score_case(out: Path, election: int, horizon: int) -> None:
    from vvs.model.data import load_polls
    from vvs.model.second_round import SecondRoundModel
    from vvs.registry import default_registry

    spec = load_spec()
    folder, case = prepare_case(out, election, horizon, spec)
    check_case(folder, case)
    fit_case(out, election, horizon)  # valide/reprend seulement si incomplet
    asof = dt.date.fromisoformat(case["asof"])
    errors, errors2, transfers, turnout, excluded, audit = calibration(asof, spec)
    write_json(folder / "calibration.json", audit)
    quality = json.loads((folder / "fit/quality.json").read_text())
    polls = load_polls(election, asof, curated=folder / "curated")
    metrics, outcomes, cells = [], [], []
    with np.load(folder / "fit/draws.npz") as draws:
        for method in ("reference", "mean14d"):
            model2 = SecondRoundModel(
                curated=folder / "curated",
                transfers=transfers,
                errors=errors2,
                exclude=excluded,
                **spec["second_round"],
            )
            sim, detail = simulate_chain(case, draws, errors, model2, turnout, spec, polls, method)
            summary, rows = score_chain(sim, election, default_registry())
            key = {"election": election, "horizon": horizon, "method": method}
            metrics.append(key | summary | {"fit_acceptable": quality["acceptable"]})
            outcomes += [key | row for row in rows]
            cells += [key | row for row in detail]
            np.savez_compressed(folder / f"draws_{method}.npz", **sim)
    pl.DataFrame(metrics).write_parquet(folder / "metrics.parquet")
    pl.DataFrame(outcomes).write_parquet(folder / "outcomes.parquet")
    pl.DataFrame(cells).write_parquet(folder / "cells.parquet")
    outputs = [
        "case.json",
        "curated/polls.parquet",
        "curated/poll_results.parquet",
        "fit/draws.npz",
        "fit/manifest.json",
        "fit/quality.json",
        "metrics.parquet",
        "outcomes.parquet",
        "cells.parquet",
        "calibration.json",
        "draws_reference.npz",
        "draws_mean14d.npz",
    ]
    write_json(folder / "scored.json", {p: digest(folder / p) for p in outputs})
    print(
        f"Score {election}/{horizon} : {metrics[0]['p_winner']:.3f} pour le vainqueur réel",
        flush=True,
    )


def complete_score(folder: Path) -> bool:
    if not (folder / "scored.json").exists():
        return False
    for name, sha in json.loads((folder / "scored.json").read_text()).items():
        if not (folder / name).exists() or digest(folder / name) != sha:
            raise ValueError(f"sortie modifiée : {folder}/{name}")
    return True


def run_all(out: Path, jobs: int) -> None:
    spec = load_spec()
    cpus = sorted(os.sched_getaffinity(0))
    workers = max(1, min(jobs, len(cpus) // 4 or 1))
    cases = [(e, h) for e in spec["elections"] for h in spec["horizons"]]
    for e, h in cases:
        prepare_case(out, e, h, spec)

    def worker(group: int):
        own_cpus = cpus[group * 4 : (group + 1) * 4] or cpus
        for e, h in cases[group::workers]:
            folder = out / "cases" / f"{e}-{h:03d}"
            if complete_score(folder):
                print(f"Réutilisation vérifiée : {e}/{h}", flush=True)
                continue
            command = [
                "taskset",
                "-c",
                ",".join(map(str, own_cpus)),
                sys.executable,
                "-m",
                "vvs.eval.consolidation",
                "case",
                "--out",
                str(out),
                "--election",
                str(e),
                "--horizon",
                str(h),
            ]
            print(f"Début {e}/{h} ; journal {folder / 'run.log'}", flush=True)
            with (folder / "run.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"Terminé {e}/{h}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, i) for i in range(workers)]
        for future in as_completed(futures):
            future.result()
    from vvs.eval.sensitivity import analyze

    analyze(out, spec)
    report(out)


def report(out: Path) -> None:
    spec = load_spec()
    frames = []
    for e in spec["elections"]:
        for h in spec["horizons"]:
            folder = out / "cases" / f"{e}-{h:03d}"
            if complete_score(folder):
                frames.append(pl.read_parquet(folder / "metrics.parquet"))
    if not frames:
        raise ValueError("aucun cas terminé")
    metrics = pl.concat(frames).sort("election", "horizon", "method")
    metrics.write_parquet(out / "metrics.parquet")
    cols = ["p_winner", "log_score_winner", "log_score_pair", "brier_win", "crps", "coverage90"]
    lines = [
        "# Consolidation chronologique — référence du 22 octobre 2026",
        "",
        "Protocole : docs/10-consolidation.md. Les probabilités sont brutes ; les log-scores",
        "utilisent le lissage déclaré. Aucun candidat n'est remplacé après coup.",
        "",
        "| Élection | Horizon | Modèle | P(vainqueur) | Log victoire | Log paire | Brier victoire | CRPS | Couv. 90 % | Fit valide |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in metrics.iter_rows(named=True):
        vals = [str(row[c]) for c in ("election", "horizon", "method")]
        vals += [f"{row[c]:.4f}" for c in cols] + [str(row["fit_acceptable"])]
        lines.append("| " + " | ".join(vals) + " |")
    means = (
        metrics.group_by("horizon", "method")
        .agg(pl.len().alias("n_elections"), *[pl.col(c).mean() for c in cols])
        .sort("horizon", "method")
    )
    means.write_parquet(out / "means.parquet")
    lines += [
        "",
        "## Moyennes à poids égal par élection",
        "",
        "| Horizon | Modèle | n | Log victoire | Log paire | CRPS |",
        "|---|---|---|---|---|---|",
    ]
    for r in means.iter_rows(named=True):
        lines.append(
            f"| {r['horizon']} | {r['method']} | {r['n_elections']} | {r['log_score_winner']:.4f} | {r['log_score_pair']:.4f} | {r['crps']:.4f} |"
        )
    failures = metrics.filter(~pl.col("fit_acceptable")).select("election", "horizon").unique()
    lines += [
        "",
        f"Cas avec diagnostic d'inférence non conforme : {failures.to_dicts()}.",
        "Les scores sont conservés même dans ce cas, et ne valent pas validation de convergence.",
        "",
        "Limites : trois élections, dates de publication imputées, scénarios historiques",
        "approchés par les hypothèses sondées, classifications et méthode élaborées aujourd'hui.",
        "Les périodes des observations sont filtrées, pas la date de publication des bases",
        "rétrospectives (notamment Jennings–Wlezien et Cevipof).",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "verify", "run", "case", "report", "sensitivity")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--election", type=int)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "freeze":
        freeze(out)
        return
    verify(out)
    if args.command == "verify":
        print("Empreintes de la référence vérifiées.")
        return
    source = out / "source"
    if paths.ROOT.resolve() != source:
        env = os.environ | {
            "VVS_ROOT": str(source),
            "PYTHONPATH": str(source / "back/src"),
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "2",
            "MPLCONFIGDIR": "/tmp/vvs-mpl-consolidation",
        }
        subprocess.run(
            [sys.executable, "-m", "vvs.eval.consolidation", *sys.argv[1:]], env=env, check=True
        )
        return
    if args.command == "case":
        score_case(out, args.election, args.horizon)
    elif args.command == "run":
        run_all(out, args.jobs)
    elif args.command == "report":
        report(out)
    else:
        from vvs.eval.sensitivity import analyze

        analyze(out, load_spec())


if __name__ == "__main__":
    main()
