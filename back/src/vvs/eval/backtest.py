"""Backtest du premier tour (docs/07-evaluation.md) : prévisions « comme à l'époque ».

Pour chaque (élection, horizon) : asof = premier tour − horizon ; configuration connue à cette
date (`configurations.known_configuration`) ; ajustement du modèle de premier tour à asof
(`vvs.model.run.run_fit`, tirages de θ sauvegardés) ; modèle d'erreur η(τ) calibré sans
l'élection évaluée (`vvs.model.error.ErrorModel`) ; 10 000 tirages de parts au jour du vote,
ramenés aux candidats officiels (`configurations.OfficialMap`) ; métriques (`metrics`) contre
les baselines (`baselines`).

Deux étapes, rejouables séparément :

- `fit` : un sous-processus par ajustement (au plus `jobs` en parallèle, 4 chaînes NUTS
  chacun), sorties dans `<out>/fits/<variant>/<election>-<horizon>/` (draws.npz, manifest,
  configuration), journal dans `<out>/backtest.log`. Un ajustement déjà présent est sauté.
- `score` : lit les tirages, applique le modèle d'erreur et les baselines, écrit
  `metrics.parquet`, `forecasts.parquet`, les tableaux markdown et les figures de calibration.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from vvs import paths
from vvs.model.first_round import Priors

log = logging.getLogger(__name__)

# Variantes du modèle de premier tour. `floor` est la référence de la phase 3 : plancher
# d'erreur non due à l'échantillonnage, n_max ~ LogNormal(log 2500, 0,3) (voir
# first_round.Priors, défaut de Priors depuis la phase 4) ; `phase2` est le modèle sans
# plancher de la phase 2 ; `subst` ajoute à `floor` la substitution intra-bloc (logit emboîté
# par bloc, phase 4, voir first_round).
VARIANTS: dict[str, Priors] = {
    "floor": Priors(n_max=2500.0),
    "phase2": Priors(n_max=None),
    "subst": Priors(n_max=2500.0, substitution=True),
    # palmarès des instituts : prior appris des effets maison et facteur de précision sur n_eff
    # (vvs.model.pollster_ratings), calibrés sans l'élection ajustée
    "ratings": Priors(n_max=2500.0, pollster_ratings=True),
    # rebonds de primaire : mêmes ajustements que floor, dérive μ_c(τ) ajoutée à la projection
    # au score (vvs.model.bounce.BOUNCE_VARIANTS)
    "bounce": Priors(n_max=2500.0),
    # certitude du choix : volatilité de δ par candidat selon la part de ses électeurs sûrs de
    # leur choix, κ calibré sans l'élection ajustée (vvs.model.certainty) ; la mise à l'échelle
    # de v_c dans η se fait au score (studies/certainty_backtest.py)
    "certainty": Priors(n_max=2500.0, certainty=True),
    # volatilité réalisée : mêmes ajustements que floor, écart type de η mis à l'échelle par
    # (V / V_ref)^κ à la date de prévision, κ et η recalibrés sans l'élection évaluée
    # (vvs.model.volatility, au score : studies/volatility.py)
    "volatility": Priors(n_max=2500.0),
    # vote utile de fin de campagne : mêmes ajustements que floor, dérive μ_c(τ) de
    # consolidation d'un camp vers son leader ajoutée à la projection au score
    # (vvs.model.consolidation.CONSOLIDATION_VARIANTS, studies/consolidation.py)
    "consolidation": Priors(n_max=2500.0),
}
DEFAULT_VARIANT = "floor"
DEFAULT_ELECTIONS = (2012, 2017, 2022)
DEFAULT_HORIZONS = (180, 90, 30, 7)
# Poids des élections dans la synthèse : 2002 et 2007 ont moins de sondages, sans effectifs
# (docs/07-evaluation.md, « avec un poids réduit »).
ELECTION_WEIGHT = {2002: 0.5, 2007: 0.5}


def fit_dir(out: Path, variant: str, election: int, horizon: int) -> Path:
    return Path(out) / "fits" / variant / f"{election}-{horizon:03d}"


def asof_for(election: int, horizon: int) -> dt.date:
    from vvs.model.data import election_date
    from vvs.registry import default_registry

    return election_date(default_registry(), election) - dt.timedelta(days=horizon)


# --- étape 1 : ajustements


@dataclass
class FitSettings:
    method: str = "nuts"
    chains: int = 4
    warmup: int = 500
    samples: int = 500
    weeks: int = 78
    seed: int = 0
    svi_steps: int = 3000


def fit_one(
    election: int,
    horizon: int,
    out_dir: Path,
    priors: Priors,
    settings: FitSettings,
) -> dict:
    """Un ajustement du backtest (exécuté dans un sous-processus par `run_fits`)."""
    from vvs.eval.configurations import known_configuration
    from vvs.model.run import run_fit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asof = asof_for(election, horizon)
    known = known_configuration(election, asof)
    (out_dir / "configuration.json").write_text(
        json.dumps(
            {
                "election": election,
                "horizon": horizon,
                "asof": asof.isoformat(),
                "candidates": known.candidates,
                "source": known.source,
                "window_days": known.window_days,
                "n_polls": known.n_polls,
                "alternatives": known.alternatives,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if settings.method == "svi":
        return _fit_svi(election, asof, out_dir, known.candidates, priors, settings)
    return run_fit(
        election,
        asof,
        out_dir,
        weeks=settings.weeks,
        method=settings.method,
        chains=settings.chains,
        warmup=settings.warmup,
        samples=settings.samples,
        seed=settings.seed,
        reference=known.candidates,
        priors=priors,
        save_draws=True,
    )


def _fit_svi(election, asof, out_dir, reference, priors, settings) -> dict:
    """Ajustement SVI court (tests) : mêmes sorties utiles au score que `run_fit`."""
    import numpy as np

    from vvs.model.data import encode, load_polls
    from vvs.model.first_round import fit
    from vvs.model.posterior import Posterior
    from vvs.model.run import substitution_draws

    data = encode(load_polls(election, asof), election, asof, n_weeks=settings.weeks,
                  reference=reference)
    res = fit(data, method="svi", svi_steps=settings.svi_steps, num_samples=settings.samples,
              seed=settings.seed, priors=priors)
    post = Posterior(data, res)
    np.savez_compressed(
        out_dir / "draws.npz",
        theta_asof=post.theta[:, data.asof_week].astype(np.float32),
        theta_vote=post.theta[:, data.n_weeks - 1].astype(np.float32),
        candidates=np.array(data.candidates),
        blocs=np.array([data.blocs[b] for b in data.cand_bloc]),
        **substitution_draws(post, data),
    )
    manifest = {
        "election": election,
        "asof": asof.isoformat(),
        "data": {"n_polls": data.n_polls, "reference": data.reference},
        "parameters": res.settings,
        "diagnostics": res.diagnostics,
        "duration_s": {"inference": round(res.duration_s, 1)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _worker_main(argv: list[str]) -> None:
    """Point d'entrée d'un sous-processus : `python -m vvs.eval.backtest fit-one <json>`."""
    spec = json.loads(argv[0])
    priors = Priors(**spec["priors"])
    settings = FitSettings(**spec["settings"])
    m = fit_one(spec["election"], spec["horizon"], Path(spec["out_dir"]), priors, settings)
    diag = m.get("diagnostics", {})
    print(
        json.dumps(
            {
                "n_polls": m["data"]["n_polls"],
                "inference_s": m["duration_s"]["inference"],
                "r_hat_max": diag.get("r_hat_max"),
                "divergences": diag.get("divergences"),
                "derived": diag.get("derived"),
            }
        )
    )


@dataclass
class FitJob:
    election: int
    horizon: int
    variant: str
    priors: Priors
    out_dir: Path

    def done(self) -> bool:
        return (self.out_dir / "draws.npz").exists() and (self.out_dir / "manifest.json").exists()


def plan_fits(
    out: Path,
    elections,
    horizons,
    variant: str = DEFAULT_VARIANT,
    priors: Priors | dict[int, Priors] | None = None,
) -> list[FitJob]:
    """Ajustements à faire, du plus long au plus court (horizon croissant : plus de sondages)."""
    jobs = []
    for h in sorted(horizons):
        for e in elections:
            p = priors.get(e) if isinstance(priors, dict) else priors
            p = p if p is not None else VARIANTS[variant]
            jobs.append(FitJob(e, h, variant, p, fit_dir(out, variant, e, h)))
    return jobs


def _log(path: Path, msg: str) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    with path.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_fits(
    jobs: list[FitJob],
    out: Path,
    settings: FitSettings | None = None,
    n_parallel: int = 3,
) -> list[FitJob]:
    """Lance les ajustements en sous-processus (au plus `n_parallel` à la fois).

    Chaque sous-processus garde JAX sur CPU (`JAX_PLATFORMS=cpu`, le GPU est partagé) et voit
    4 périphériques hôtes (une chaîne chacun). Rend les ajustements en échec."""
    settings = settings or FitSettings()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "backtest.log"
    todo = [j for j in jobs if not j.done()]
    _log(logfile, f"ajustements : {len(todo)} à faire, {len(jobs) - len(todo)} déjà présents")
    env = os.environ | {"JAX_PLATFORMS": "cpu"}
    running: list[tuple[FitJob, subprocess.Popen, float, Path]] = []
    failed: list[FitJob] = []
    queue = list(todo)
    while queue or running:
        while queue and len(running) < n_parallel:
            j = queue.pop(0)
            j.out_dir.mkdir(parents=True, exist_ok=True)
            spec = {
                "election": j.election,
                "horizon": j.horizon,
                "out_dir": str(j.out_dir),
                "priors": asdict(j.priors),
                "settings": asdict(settings),
            }
            jlog = j.out_dir / "fit.log"
            fh = jlog.open("w")
            proc = subprocess.Popen(
                [sys.executable, "-m", "vvs.eval.backtest", "fit-one", json.dumps(spec)],
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append((j, proc, time.time(), jlog))
            _log(logfile, f"début {j.variant} {j.election} τ={j.horizon}")
        time.sleep(2)
        still = []
        for j, proc, t0, jlog in running:
            if proc.poll() is None:
                still.append((j, proc, t0, jlog))
                continue
            dur = time.time() - t0
            tail = jlog.read_text().strip().splitlines()[-1:] if jlog.exists() else []
            if proc.returncode == 0:
                _log(logfile, f"fin {j.variant} {j.election} τ={j.horizon} ({dur:.0f} s) {tail}")
            else:
                failed.append(j)
                _log(logfile, f"ÉCHEC {j.variant} {j.election} τ={j.horizon} code "
                     f"{proc.returncode} ({dur:.0f} s), voir {jlog}")
        running = still
    _log(logfile, f"ajustements terminés, {len(failed)} en échec")
    return failed


# --- étape 2 : score


def score(
    out: Path,
    elections,
    horizons,
    variant: str = DEFAULT_VARIANT,
    n_draws: int = 10_000,
    seed: int = 0,
    calibration_elections=None,
    write: bool = True,
):
    """Applique le modèle d'erreur et les baselines aux ajustements, calcule les métriques."""
    from vvs.eval.scoring import score_backtest

    return score_backtest(
        Path(out),
        list(elections),
        list(horizons),
        variant=variant,
        n_draws=n_draws,
        seed=seed,
        calibration_elections=calibration_elections,
        write=write,
    )


def default_out() -> Path:
    return paths.RUNS / f"{dt.date.today().isoformat()}-backtest"


def with_overrides(p: Priors, **kw) -> Priors:
    return replace(p, **kw)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "fit-one":
        _worker_main(sys.argv[2:])
    else:
        raise SystemExit("usage : python -m vvs.eval.backtest fit-one '<json>'")
