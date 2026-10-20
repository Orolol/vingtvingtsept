"""Un ajustement complet : chargement, encodage, inférence, sorties et figures dans un dossier.

Sorties (`out/`) : manifest.json, trajectories.parquet, house_effects.parquet,
final.parquet, observed.parquet, trajectories.png, house_effects.png, final.png ; avec
`save_draws`, draws.npz (tirages de θ à la semaine asof et à la semaine du vote, tous les
candidats, pour le backtest : vvs.eval.backtest).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
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
) -> dict:
    """Ajuste le modèle de premier tour à la date `asof` et écrit les sorties dans `out`.

    `reference` : configuration des sorties (défaut : candidats officiels, voir `encode`) ;
    le backtest passe la configuration connue à la date (vvs.eval.configurations).
    `priors` : hyperparamètres (défaut `Priors()`)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now()
    df = load_polls(election, asof)
    source = load_polls.last_source
    data = encode(df, election, asof, n_weeks=weeks, step_days=step_days, reference=reference)
    data.source = source

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
        np.savez_compressed(
            out / "draws.npz",
            theta_asof=post.theta[:, data.asof_week].astype(np.float32),
            theta_vote=post.theta[:, data.n_weeks - 1].astype(np.float32),
            candidates=np.array(data.candidates),
            blocs=np.array([data.blocs[b] for b in data.cand_bloc]),
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
