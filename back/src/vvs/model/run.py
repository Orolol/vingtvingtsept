"""Un ajustement complet : chargement, encodage, inférence, sorties et figures dans un dossier.

Sorties (`out/`) : manifest.json, trajectories.parquet, house_effects.parquet,
final.parquet, observed.parquet, trajectories.png, house_effects.png, final.png ; avec
`save_draws`, draws.npz (tirages de θ à la semaine asof et à la semaine du vote, tous les
candidats, pour le backtest : vvs.eval.backtest) ; avec `save_trajectories` en plus, draws.npz
contient aussi θ à toutes les semaines et les effets maison α (simulation 2027 :
vvs.model.simulate). Avec la substitution intra-bloc (`Priors.substitution`), draws.npz porte en
plus β, λ et l'ordre des blocs (`substitution_draws`), de quoi recalculer les parts de toute
configuration (`draws_log_shares`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.model.data import encode, hypotheses_frame, load_polls
from vvs.model.first_round import Priors, fit
from vvs.model.plots import plot_final, plot_house_effects, plot_trajectories
from vvs.model.posterior import Posterior
from vvs.registry import default_registry

log = logging.getLogger(__name__)


def official_shares(election: int) -> dict[str, float] | None:
    e = default_registry().elections.get(election)
    if e is None or not e.rounds[0].results:
        return None
    return {r.candidate_id: r.share for r in e.round_(1).results}


def _git_commit() -> str | None:
    # Un gel de calcul n'est pas un worktree Git : son dossier parent peut avancer.
    if frozen_commit := os.environ.get("VVS_SOURCE_COMMIT"):
        return frozen_commit
    try:
        out = subprocess.run(
            ["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hash_frame(df: pl.DataFrame) -> str:
    return hashlib.sha256(df.write_csv().encode()).hexdigest()[:16]


def substitution_draws(post: Posterior, data, trajectories: bool = False) -> dict:
    """Tirages propres à la substitution pour draws.npz (vide sans substitution) : β à asof et
    au vote (S, B), λ (S, B), `bloc_ids` (ordre de β) ; avec `trajectories`, β à toutes les
    semaines (S, T, B) et les effets maison de bloc et de candidat (S, P, B), (S, P, C)."""
    if not post.substitution:
        return {}
    out = {
        "beta_asof": post.beta[:, data.asof_week].astype(np.float32),
        "beta_vote": post.beta[:, data.n_weeks - 1].astype(np.float32),
        "lam": post.lam.astype(np.float32),
        "bloc_ids": np.array(data.blocs),
    }
    if trajectories:
        out |= {
            "beta": post.beta.astype(np.float32),
            "house_bloc": post.house_bloc.astype(np.float32),
            "house_cand": post.house_cand.astype(np.float32),
        }
    return out


def draws_log_shares(draws, config: list[str], when: str = "asof") -> np.ndarray:
    """log-parts (S, K) de la configuration `config` à partir d'un draws.npz (`when` : asof ou
    vote) : softmax de θ, ou logit emboîté si les tirages portent λ (substitution)."""
    from vvs.model.first_round import config_log_shares

    cands = [str(c) for c in draws["candidates"]]
    idx = [cands.index(c) for c in config]
    theta = np.asarray(draws[f"theta_{when}"], dtype=float)
    if "lam" not in draws:
        th = theta[:, idx]
        m = th.max(1, keepdims=True)
        return th - m - np.log(np.exp(th - m).sum(1, keepdims=True))
    bloc_ids = [str(b) for b in draws["bloc_ids"]]
    cb = np.array([bloc_ids.index(str(b)) for b in draws["blocs"]])
    beta = np.asarray(draws[f"beta_{when}"], dtype=float)
    delta = theta - beta[:, cb]
    return config_log_shares(beta, delta[:, idx], cb[idx], np.asarray(draws["lam"], dtype=float))


def run_fit(
    election: int,
    asof: dt.date,
    out: Path,
    weeks: int = 78,
    step_days: int = 7,
    method: str = "nuts",
    chains: int = 4,
    warmup: int = 1000,
    samples: int = 1000,
    seed: int = 0,
    centered: bool = False,
    target_accept: float = 0.9,
    max_tree_depth: int = 10,
    progress_bar: bool = False,
    reference: list[str] | None = None,
    priors: Priors | None = None,
    save_draws: bool = False,
    figures: bool = True,
    save_trajectories: bool = False,
    polls: pl.DataFrame | None = None,
) -> dict:
    """Ajuste le modèle de premier tour à la date `asof` et écrit les sorties dans `out`.

    `reference` : configuration des sorties (défaut : candidats officiels, voir `encode`) ;
    le backtest passe la configuration connue à la date (vvs.eval.configurations).
    `priors` : hyperparamètres (défaut `Priors()`)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    df = load_polls(election, asof) if polls is None else polls
    source = load_polls.last_source if polls is None else "provided"
    if df.filter(pl.col("fieldwork_end") > asof).height:
        raise ValueError("sondages fournis après la date de prévision")
    data = encode(df, election, asof, n_weeks=weeks, step_days=step_days, reference=reference)
    data.source = source
    if priors is not None and priors.pollster_ratings:
        from vvs.model.pollster_ratings import attach_ratings

        attach_ratings(data, out)  # notes calibrées sans l'élection ajustée

    res = fit(
        data,
        method=method,
        num_chains=chains,
        num_warmup=warmup,
        num_samples=samples,
        seed=seed,
        centered=centered,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        progress_bar=progress_bar,
        priors=priors,
    )
    post = Posterior(data, res)
    official = official_shares(election)
    traj = post.trajectories()
    house = post.house_effects()
    final = post.final(official)
    observed = hypotheses_frame(data)

    traj.write_parquet(out / "trajectories.parquet")
    house.write_parquet(out / "house_effects.parquet")
    final.write_parquet(out / "final.parquet")
    observed.write_parquet(out / "observed.parquet")

    if save_draws:
        extra = {}
        if save_trajectories:
            extra = {
                "theta": post.theta.astype(np.float32),  # (S, T, C)
                "alpha": post.alpha.astype(np.float32),  # (S, P, C)
                "pollsters": np.array(data.pollsters),
                "week_dates": np.array([d.isoformat() for d in data.week_date(np.arange(data.n_weeks))]),
                "asof_week": np.array(data.asof_week),
            }
        np.savez_compressed(
            out / "draws.npz",
            theta_asof=post.theta[:, data.asof_week].astype(np.float32),
            theta_vote=post.theta[:, data.n_weeks - 1].astype(np.float32),
            candidates=np.array(data.candidates),
            blocs=np.array([data.blocs[b] for b in data.cand_bloc]),
            **extra,
            **substitution_draws(post, data, save_trajectories),
        )
    if figures:
        plot_trajectories(
            traj, observed, final, data.asof, data.election_date, out / "trajectories.png"
        )
        plot_house_effects(house, final, out / "house_effects.png")
        plot_final(final, out / "final.png", f"Estimation au jour du vote, {election}, au {asof}")

    s = res.samples
    posterior_params = {
        k: {
            "median": float(np.median(s[k])),
            "q05": float(np.quantile(s[k], 0.05)),
            "q95": float(np.quantile(s[k], 0.95)),
        }
        for k in (
            "sigma_beta",
            "sigma_delta",
            "sigma_house_bloc",
            "sigma_house_cand",
            "design",
            "n_max",
        )
        if k in s
    }
    if "lam" in s:
        posterior_params["lam"] = {
            b: {
                "median": float(np.median(s["lam"][:, j])),
                "q05": float(np.quantile(s["lam"][:, j], 0.05)),
                "q95": float(np.quantile(s["lam"][:, j], 0.95)),
            }
            for j, b in enumerate(data.blocs)
        }
    diag = dict(res.diagnostics)
    diag["derived"] = post.derived_diagnostics()
    manifest = {
        "election": election,
        "asof": asof.isoformat(),
        "election_date": data.election_date.isoformat(),
        "created_at": started.isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "data": {
            "source": source,
            "rows_hash": _hash_frame(df),
            "n_polls": data.n_polls,
            "n_hypotheses": data.n_hyp,
            "n_candidates": len(data.candidates),
            "candidates": data.candidates,
            "blocs": data.blocs,
            "pollsters": data.pollsters,
            "populations": data.populations,
            "reference": data.reference,
            "dropped_candidates": data.dropped_candidates,
            "first_poll_week": int(data.week.min()),
            "n_steps": data.n_weeks,
            "asof_step": data.asof_week,
            "prior_election": data.prior_election,
            "prior_bloc_mean_logit": dict(zip(data.blocs, map(float, data.prior_bloc_mean))),
        },
        "parameters": {"weeks": weeks, **res.settings},
        "posterior": posterior_params,
        "diagnostics": diag,
        "duration_s": {
            "inference": round(res.duration_s, 1),
            "total": round((dt.datetime.now() - started).total_seconds(), 1),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
