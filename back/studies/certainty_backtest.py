"""Backtest de la certitude du choix (docs/07-evaluation.md, règle d'admission).

    uv run python studies/certainty_backtest.py fit   [--out runs/<date>-backtest-certainty] [--jobs 2]
    uv run python studies/certainty_backtest.py score [--out …] [--ref runs/2026-10-20-backtest]

Quatre versions, 2017 et 2022 (les seules élections avec des mesures de certitude par
candidat), horizons 180, 90, 30 et 7 jours, NUTS 4 chaînes de 500 + 500 sur CPU :

- `floor` : la référence, ajustements repris de `--ref` (mêmes données : empreinte des lignes
  vérifiée), η inchangé ;
- `floor+eta` : mêmes ajustements, composante candidat v_c de η mise à l'échelle par la
  certitude à la date de prévision (`model.certainty.CertaintyScaler`) ;
- `certainty` : ajustements de la variante `certainty` de `eval.backtest.VARIANTS`
  (volatilité de δ par candidat), η inchangé ;
- `certainty+eta` : les deux.

κ est estimé sans l'élection évaluée (`model.certainty.fit_kappa`) : pour 2017 sur 2022 et
inversement, η calibré sans l'élection évaluée. Les tirages de η partagent la même graine
d'une version à l'autre (nombres aléatoires communs) : les écarts viennent de la seule mise à
l'échelle. 2012, sans mesure de certitude, donnerait exactement la référence (échelle 1).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.eval import scoring as sc
from vvs.eval.backtest import VARIANTS, FitSettings, asof_for, fit_dir, plan_fits, run_fits
from vvs.eval.baselines import last_by_pollster, poll_mean
from vvs.eval.configurations import official_map
from vvs.model import certainty as mc
from vvs.model.data import load_polls
from vvs.model.error import ErrorModel
from vvs.model.run import _hash_frame, draws_log_shares
from vvs.registry import default_registry

ELECTIONS = (2017, 2022)
HORIZONS = (180, 90, 30, 7)
VERSIONS = {  # nom -> (variante d'ajustement, η mis à l'échelle)
    "floor": ("floor", False),
    "floor+eta": ("floor", True),
    "certainty": ("certainty", False),
    "certainty+eta": ("certainty", True),
}
DEFAULT_REF = paths.RUNS / "2026-10-20-backtest"


def kappas(out: Path) -> dict[int, dict]:
    """κ leave-one-election-out, mis en cache dans `out/kappa.json`."""
    path = out / "kappa.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    res = {}
    for e in ELECTIONS:
        em = ErrorModel.fit(exclude=[e])
        res[e] = mc.fit_kappa(e, error_model=em).to_dict()
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    return res


def cmd_fit(out: Path, jobs: int) -> None:
    k = kappas(out)
    priors = {e: replace(VARIANTS["certainty"], certainty_kappa=float(k[e]["kappa"]))
              for e in ELECTIONS}
    plan = plan_fits(out, ELECTIONS, HORIZONS, variant="certainty", priors=priors)
    failed = run_fits(plan, out, FitSettings(warmup=500, samples=500), n_parallel=jobs)
    if failed:
        raise SystemExit(f"{len(failed)} ajustements en échec")


def _fit_path(out: Path, ref: Path, variant: str, e: int, h: int) -> Path:
    return fit_dir(ref if variant == "floor" else out, variant, e, h)


def check_ref(ref: Path) -> list[str]:
    """Ajustements de référence dont les données ne sont plus celles d'aujourd'hui."""
    stale = []
    for e in ELECTIONS:
        for h in HORIZONS:
            m = json.loads((fit_dir(ref, "floor", e, h) / "manifest.json").read_text())
            if m["data"]["rows_hash"] != _hash_frame(load_polls(e, asof_for(e, h))):
                stale.append(f"{e}-{h}")
    return stale


def cmd_score(out: Path, ref: Path, n_draws: int = 10_000, seed: int = 0) -> sc.ScoreResult:
    stale = check_ref(ref)
    if stale:
        raise SystemExit(f"ajustements de référence périmés (données changées) : {stale}")
    reg = default_registry()
    k = kappas(out)
    metric_rows, fc_rows, cal_rows, ems = [], [], [], {}
    for e in ELECTIONS:
        em = ErrorModel.fit(elections=sc.CALIBRATION_ELECTIONS, exclude=[e], registry=reg)
        ems[e] = em
        off = sc.official(reg, e)
        for h in HORIZONS:
            asof = asof_for(e, h)
            polls = load_polls(e, asof)
            scaler = mc.CertaintyScaler(e, asof, float(k[e]["kappa"]))
            for name, (variant, eta) in VERSIONS.items():
                fd = _fit_path(out, ref, variant, e, h)
                if not (fd / "draws.npz").exists():
                    print(f"manquant : {fd}")
                    continue
                conf = json.loads((fd / "configuration.json").read_text())
                dr = np.load(fd / "draws.npz")
                omap = official_map(conf["candidates"], e, reg, substitution="lam" in dr.files)
                K = omap.config
                cands = [str(c) for c in dr["candidates"]]
                cblocs = [str(b) for b in dr["blocs"]]
                kblocs = [cblocs[cands.index(c)] for c in K]
                points = {}
                if name == "floor":
                    points["mean14"] = poll_mean(e, asof, K, days=14, registry=reg, polls=polls)
                    points["last_pollster"] = last_by_pollster(e, asof, K, registry=reg,
                                                               polls=polls)
                em.cand_scale = scaler if eta else None
                rng = np.random.default_rng([seed, e, h])
                case = sc.score_case(
                    draws_log_shares(dr, K, "asof"), draws_log_shares(dr, K, "vote"), K, kblocs,
                    h, em, omap, off, points, rng, n_draws,
                )
                em.cand_scale = None
                m_rows, f_rows, c_rows = sc.case_rows(
                    case, {"election": e, "horizon": h, "variant": name})
                metric_rows += m_rows
                fc_rows += f_rows
                cal_rows += c_rows
            scales = scaler(K)
            print(e, h, "κ", round(k[e]["kappa"], 2),
                  {c: round(float(s), 2) for c, s in zip(K, scales) if abs(s - 1) > 1e-9})
    res = sc.ScoreResult(
        out=out,
        metrics=pl.DataFrame(metric_rows),
        forecasts=pl.DataFrame(fc_rows, infer_schema_length=None),
        calibration=pl.DataFrame(cal_rows),
        error_models=ems,
    )
    sc.write_outputs(res, list(VERSIONS))
    admission_report(res.metrics, out, k)
    return res


def admission_report(metrics: pl.DataFrame, out: Path, k: dict) -> None:
    """Écarts à la référence, par élection et horizon, et règle d'admission de docs/07."""
    s = metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model"))
    val = {(r["variant"], r["election"], r["horizon"], r["metric"]): r["value"]
           for r in s.iter_rows(named=True)}
    lines = ["# Certitude du choix : backtest avant/après", "",
             f"Généré le {dt.date.today().isoformat()}. Modèle + η, 2017 et 2022, NUTS 500 + 500. "
             "κ sans l'élection évaluée : "
             + ", ".join(f"{e} : {v['kappa']:+.2f} (appris sur {v['train']})" for e, v in k.items())
             + ".", ""]
    for metric, better in (("log_score_pair", 1), ("crps", -1), ("cov50", 0), ("cov90", 0),
                           ("brier_qual", -1), ("bloc_crps", -1)):
        lines += [f"## {sc.METRIC_TITLE.get(metric, metric)}", "",
                  "| élection | horizon | " + " | ".join(VERSIONS) + " |",
                  "|---|---|" + "---|" * len(VERSIONS)]
        for e in ELECTIONS:
            for h in HORIZONS:
                cells = []
                for v in VERSIONS:
                    x = val.get((v, e, h, metric))
                    base = val.get(("floor", e, h, metric))
                    if x is None:
                        cells.append("–")
                    elif v == "floor" or base is None:
                        cells.append(sc._fmt(metric, x))
                    else:
                        d = x - base
                        cells.append(f"{sc._fmt(metric, x)} ({d:+.3f})")
                lines.append(f"| {e} | {h} | " + " | ".join(cells) + " |")
        for h in HORIZONS:
            cells = []
            for v in VERSIONS:
                xs = [val.get((v, e, h, metric)) for e in ELECTIONS]
                cells.append("–" if None in xs else sc._fmt(metric, float(np.mean(xs))))
            lines.append(f"| **moyenne** | {h} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Règle d'admission (docs/07-evaluation.md)", ""]
    for v in [x for x in VERSIONS if x != "floor"]:
        verdict = []
        ok_all = True
        for metric, sign in (("log_score_pair", 1), ("crps", -1)):
            for h in (180, 90):
                d = [sign * (val[(v, e, h, metric)] - val[("floor", e, h, metric)])
                     for e in ELECTIONS if (v, e, h, metric) in val]
                mean_gain = float(np.mean(d)) if d else float("nan")
                n_pos = sum(x > 0 for x in d)
                verdict.append(f"{metric} à {h} j : gain moyen {mean_gain:+.3f}, "
                               f"positif sur {n_pos}/{len(d)} élections")
                ok_all &= mean_gain > 0 and n_pos >= 2
        cov = []
        for h in HORIZONS:
            for metric in ("cov50", "cov90"):
                d = [val[(v, e, h, metric)] - val[("floor", e, h, metric)]
                     for e in ELECTIONS if (v, e, h, metric) in val]
                cov.append(float(np.mean(d)) if d else 0.0)
        cov_ok = min(cov) >= -0.02
        lines.append(f"- `{v}` : " + " ; ".join(verdict)
                     + f" ; pire écart de couverture {min(cov):+.2f}. "
                     + ("**Critères 1 à 3 remplis.**" if ok_all and cov_ok
                        else "**Non admis.**"))
    lines += ["", ("Gain = amélioration (log-score plus haut, CRPS plus bas). La règle demande "
                   "un gain moyen aux horizons 180 et 90 jours sur les deux métriques, de même "
                   "signe sur au moins deux des trois élections de référence (2012 n'a pas de "
                   "mesure : écart nul), sans dégrader la couverture."), ""]
    (out / "admission.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "score", "kappa"])
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-backtest-certainty")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "kappa":
        print(json.dumps(kappas(args.out), indent=2))
    elif args.cmd == "fit":
        cmd_fit(args.out, args.jobs)
    else:
        cmd_score(args.out, args.ref)


if __name__ == "__main__":
    main()
