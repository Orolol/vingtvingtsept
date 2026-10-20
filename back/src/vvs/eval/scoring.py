"""Score du backtest : tirages de parts au jour du vote, métriques, tableaux et figures.

Appelé par `vvs.eval.backtest.score`. Pour chaque (élection, horizon) ajusté :

- `model` : θ à la semaine asof (postérieur de l'agrégateur) + η(τ) calibré sans l'élection ;
- `model_noerror` : θ prolongé jusqu'à la semaine du vote par la seule marche aléatoire, sans
  η : la projection de la phase 2 ;
- `mean14` : moyenne simple des sondages des 14 derniers jours avant asof + η ;
- `last_pollster` : dernier sondage de chaque institut, moyenné + η.

Les parts sont tirées sur la configuration connue à la date puis ramenées aux candidats
officiels (`configurations.OfficialMap`). Les métriques par candidat ne portent que sur les
candidats officiels qui reçoivent une part ; les autres sont comptés dans `unforecast_share`.
Un ajustement avec substitution intra-bloc (tirages portant λ) est évalué sur la configuration
privée des tenants lieu absorbés (`official_map(..., substitution=True)`), parts du logit
emboîté (`vvs.model.run.draws_log_shares`).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from vvs.eval import metrics as mt
from vvs.eval.backtest import ELECTION_WEIGHT, VARIANT_ERROR_KWARGS, asof_for, fit_dir
from vvs.eval.baselines import last_by_pollster, poll_mean
from vvs.eval.configurations import official_map
from vvs.model.data import load_polls
from vvs.model.consolidation import variant_drift  # rebonds et consolidation
from vvs.model.error import ErrorModel
from vvs.model.run import draws_log_shares
from vvs.registry import BLOCS, Registry, default_registry

log = logging.getLogger(__name__)

MODELS = ("model", "model_noerror", "mean14", "last_pollster")
MODEL_LABEL = {
    "model": "modèle + η",
    "model_noerror": "modèle sans η (phase 2)",
    "mean14": "moyenne 14 j + η",
    "last_pollster": "dernier par institut + η",
}
CALIBRATION_ELECTIONS = (2002, 2007, 2012, 2017, 2022)
SUMMARY_METRICS = (
    "log_score_pair",
    "brier_qual",
    "crps",
    "mae",
    "cov50",
    "cov90",
    "bloc_crps",
    "bloc_mae",
    "bloc_cov50",
    "bloc_cov90",
    "unforecast_share",
    "p_pair",
)


@dataclass
class Official:
    ids: list[str]
    shares: np.ndarray  # points
    blocs: list[str]
    pair: tuple[int, int]


def official(reg: Registry, election: int) -> Official:
    e = reg.elections[election]
    res = {r.candidate_id: r.share for r in e.round_(1).results}
    ids = [c.candidate_id for c in e.candidates]
    s = np.array([res[c] for c in ids])
    s = 100 * s / s.sum()
    top = np.argsort(-s)[:2]
    return Official(ids, s, [reg.bloc_of(c, election) for c in ids], tuple(sorted(top)))


def _softmax(x):
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def evaluate(
    shares: np.ndarray, forecast: np.ndarray, off: Official
) -> tuple[dict, list[dict], list[dict]]:
    """Métriques d'un jeu de tirages (n, O) en points sur les candidats officiels.

    Rend (résumé, lignes par candidat, lignes par bloc)."""
    f = np.flatnonzero(forecast)
    y = off.shares
    qual = mt.qualification_probs(shares)
    outcome = np.zeros(len(y))
    outcome[list(off.pair)] = 1
    cr = mt.crps(shares[:, f], y[f])
    ae = mt.abs_error_median(shares[:, f], y[f])
    c50 = mt.covered(shares[:, f], y[f], 0.5)
    c90 = mt.covered(shares[:, f], y[f], 0.9)
    pits = mt.pit(shares[:, f], y[f])
    cand_rows = []
    q = np.quantile(shares, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    for j, i in enumerate(f):
        cand_rows.append(
            {
                "unit": off.ids[i],
                "bloc": off.blocs[i],
                "official": float(y[i]),
                "q05": float(q[0, i]),
                "q25": float(q[1, i]),
                "median": float(q[2, i]),
                "q75": float(q[3, i]),
                "q95": float(q[4, i]),
                "p_qualify": float(qual[i]),
                "qualified": bool(outcome[i]),
                "crps": float(cr[j]),
                "abs_error": float(ae[j]),
                "in50": bool(c50[j]),
                "in90": bool(c90[j]),
                "pit": float(pits[j]),
            }
        )
    # blocs : somme des parts des candidats officiels du bloc (tirages ramenés aux officiels)
    bloc_rows = []
    ub = [b for b in BLOCS if b in off.blocs]
    for b in ub:
        m = np.array([x == b for x in off.blocs])
        if not (m & forecast).any():
            continue
        xb = shares[:, m].sum(1)
        yb = float(y[m].sum())
        qb = np.quantile(xb, [0.05, 0.25, 0.5, 0.75, 0.95])
        bloc_rows.append(
            {
                "unit": b,
                "bloc": b,
                "official": yb,
                "q05": float(qb[0]),
                "q25": float(qb[1]),
                "median": float(qb[2]),
                "q75": float(qb[3]),
                "q95": float(qb[4]),
                "crps": float(mt.crps(xb, yb)),
                "abs_error": float(abs(qb[2] - yb)),
                "in50": bool(qb[1] <= yb <= qb[3]),
                "in90": bool(qb[0] <= yb <= qb[4]),
                "pit": float(mt.pit(xb, yb)),
            }
        )
    summary = {
        "log_score_pair": mt.pair_log_score(shares, off.pair),
        "p_pair": mt.pair_probability(shares, off.pair, smooth=False),
        "brier_qual": mt.brier(qual[f], outcome[f]),
        "crps": float(cr.mean()),
        "mae": float(ae.mean()),
        "cov50": float(c50.mean()),
        "cov90": float(c90.mean()),
        "bloc_crps": float(np.mean([r["crps"] for r in bloc_rows])),
        "bloc_mae": float(np.mean([r["abs_error"] for r in bloc_rows])),
        "bloc_cov50": float(np.mean([r["in50"] for r in bloc_rows])),
        "bloc_cov90": float(np.mean([r["in90"] for r in bloc_rows])),
        "unforecast_share": float(y[~forecast].sum()),
        "n_units": len(f),
    }
    return summary, cand_rows, bloc_rows


def calibration_rows(shares: np.ndarray, forecast: np.ndarray, off: Official) -> list[dict]:
    """Couverture observée des intervalles centraux à chaque niveau nominal."""
    f = np.flatnonzero(forecast)
    rows = []
    for lvl in mt.LEVELS:
        cov = mt.covered(shares[:, f], off.shares[f], lvl)
        rows.append({"level": "candidate", "nominal": lvl, "hits": int(cov.sum()), "n": len(f)})
        hits = n = 0
        for b in sorted(set(off.blocs)):
            m = np.array([x == b for x in off.blocs])
            if not (m & forecast).any():
                continue
            hits += int(mt.covered(shares[:, m].sum(1), off.shares[m].sum(), lvl))
            n += 1
        rows.append({"level": "bloc", "nominal": lvl, "hits": hits, "n": n})
    return rows


def score_case(
    theta_asof: np.ndarray,
    theta_vote: np.ndarray,
    config: list[str],
    blocs: list[str],
    horizon: float,
    em: ErrorModel,
    omap,
    off: Official,
    points: dict,
    rng: np.random.Generator,
    n_draws: int = 10_000,
    drift: np.ndarray | None = None,
) -> dict:
    """Tirages et métriques d'un cas (élection, horizon) pour chaque modèle.

    `theta_asof`, `theta_vote` : (S, K) logits de la configuration ; `points` : estimations
    ponctuelles des baselines (`baselines.PointForecast`) par nom de modèle ; `drift` : dérive
    μ (n_draws, K) de la projection du modèle (variantes de rebond), None par défaut. Rend
    {modèle: (résumé, lignes candidats, lignes blocs, lignes de calibration)}."""
    sims = {"model": em.sample(theta_asof, horizon, config, blocs, rng, n_draws, drift=drift)}
    idx = rng.integers(0, theta_vote.shape[0], n_draws)
    sims["model_noerror"] = _softmax(np.asarray(theta_vote, dtype=float)[idx])
    for name, pf in points.items():
        sims[name] = em.sample(np.log(pf.shares)[None, :], pf.age_days, config, blocs, rng, n_draws)
    out = {}
    for model, sh in sims.items():
        shares = 100 * omap.apply(sh)
        summ, crow, brow = evaluate(shares, omap.forecast, off)
        out[model] = (summ, crow, brow, calibration_rows(shares, omap.forecast, off))
    return out


def case_rows(case: dict, key: dict) -> tuple[list, list, list]:
    """Lignes longues (métriques, prévisions, calibration) d'un cas."""
    metric_rows, fc_rows, cal_rows = [], [], []
    for model, (summ, crow, brow, crows) in case.items():
        k = key | {"model": model}
        for name, val in summ.items():
            metric_rows.append(k | {"level": "summary", "unit": "", "metric": name,
                                    "value": float(val)})
        for lvl, rows in (("candidate", crow), ("bloc", brow)):
            for r in rows:
                for name in ("crps", "abs_error", "in50", "in90", "pit"):
                    metric_rows.append(k | {"level": lvl, "unit": r["unit"], "metric": name,
                                            "value": float(r[name])})
                fc_rows.append(k | {"level": lvl} | r)
        cal_rows += [k | r for r in crows]
    return metric_rows, fc_rows, cal_rows


@dataclass
class ScoreResult:
    out: Path
    metrics: pl.DataFrame
    forecasts: pl.DataFrame
    calibration: pl.DataFrame
    error_models: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    # modèles d'erreur des variantes qui n'ont pas celui de la référence, par (variante, élection)
    variant_error_models: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = self.metrics.filter(pl.col("level") == "summary")
        if s.is_empty():
            return "aucun ajustement à scorer"
        piv = (
            s.filter(pl.col("metric").is_in(["log_score_pair", "crps", "cov90"]))
            .group_by("variant", "horizon", "model", "metric")
            .agg(pl.col("value").mean())
            .pivot(on="metric", index=["variant", "horizon", "model"], values="value")
            .sort("variant", "horizon", "model", descending=[False, True, False])
        )
        lines = [f"{len(self.missing)} ajustements manquants" if self.missing else ""]
        lines.append(str(piv))
        return "\n".join(x for x in lines if x)


def score_backtest(
    out: Path,
    elections: list[int],
    horizons: list[int],
    variant: str = "floor",
    n_draws: int = 10_000,
    seed: int = 0,
    calibration_elections=None,
    write: bool = True,
    variants: list[str] | None = None,
    error_kwargs: dict | None = None,
    absorb_stand_ins: bool = True,
) -> ScoreResult:
    """Score des ajustements de `out/fits/<variant>/`. `variants` : plusieurs variantes à
    scorer ensemble (la première est la référence des tableaux). `absorb_stand_ins` : pour un
    ajustement avec substitution, retirer de la configuration les tenants lieu que le modèle
    redistribue (défaut) ; faux : garder la correspondance par règles de la phase 3 (parts du
    logit emboîté sur toute la configuration), pour séparer l'effet du modèle de celui de la
    correspondance. Une variante de `backtest.VARIANT_ERROR_KWARGS` a son propre modèle
    d'erreur (options ajoutées à `error_kwargs`) ; les autres partagent celui de la
    référence."""
    reg = default_registry()
    variants = variants or [variant]
    cal = tuple(calibration_elections or CALIBRATION_ELECTIONS)
    metric_rows, fc_rows, cal_rows, missing = [], [], [], []
    error_models = {}
    variant_models: dict[tuple[str, int], ErrorModel] = {}
    for e in elections:
        fitted: dict[str, ErrorModel] = {}
        for v in variants:
            kw = (error_kwargs or {}) | VARIANT_ERROR_KWARGS.get(v, {})
            key = repr(sorted(kw.items()))
            if key not in fitted:
                fitted[key] = ErrorModel.fit(elections=cal, exclude=[e], registry=reg, **kw)
            variant_models[(v, e)] = fitted[key]
        error_models[e] = variant_models[(variants[0], e)]
        off = official(reg, e)
        for h in horizons:
            asof = asof_for(e, h)
            polls = load_polls(e, asof)
            for v in variants:
                fd = fit_dir(out, v, e, h)
                if not (fd / "draws.npz").exists():
                    missing.append((v, e, h))
                    continue
                conf = json.loads((fd / "configuration.json").read_text())
                dr = np.load(fd / "draws.npz")
                omap = official_map(
                    conf["candidates"], e, reg,
                    substitution=absorb_stand_ins and "lam" in dr.files,
                )
                K = omap.config
                cands = [str(c) for c in dr["candidates"]]
                cblocs = [str(b) for b in dr["blocs"]]
                kblocs = [cblocs[cands.index(c)] for c in K]
                rng = np.random.default_rng([seed, e, h])
                points = {}
                if v == variants[0]:
                    points["mean14"] = poll_mean(e, asof, K, days=14, registry=reg, polls=polls)
                    points["last_pollster"] = last_by_pollster(e, asof, K, registry=reg, polls=polls)
                case = score_case(
                    draws_log_shares(dr, K, "asof"), draws_log_shares(dr, K, "vote"), K, kblocs,
                    h, variant_models[(v, e)], omap, off, points, rng, n_draws,
                    drift=variant_drift(v, e, asof, h, K, n_draws, seed, reg,
                                        log_shares=draws_log_shares(dr, K, "asof")),
                )
                key = {"election": e, "horizon": h, "variant": v}
                m_rows, f_rows, c_rows = case_rows(case, key)
                metric_rows += m_rows
                fc_rows += f_rows
                cal_rows += c_rows
    res = ScoreResult(
        out=Path(out),
        metrics=pl.DataFrame(metric_rows),
        forecasts=pl.DataFrame(fc_rows, infer_schema_length=None),
        calibration=pl.DataFrame(cal_rows),
        error_models=error_models,
        missing=missing,
        variant_error_models={k: m for k, m in variant_models.items()
                              if m is not error_models[k[1]]},
    )
    if write and not res.metrics.is_empty():
        write_outputs(res, variants)
    return res


# --- sorties


def weighted_mean(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    w = pl.col("election").replace_strict(ELECTION_WEIGHT, default=1.0, return_dtype=pl.Float64)
    return df.with_columns(w.alias("_w")).group_by(by).agg(
        ((pl.col("value") * pl.col("_w")).sum() / pl.col("_w").sum()).alias("value")
    )


def _fmt(metric: str, x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if metric.startswith(("cov", "bloc_cov")):
        return f"{100 * x:.0f} %"
    if metric == "p_pair":
        return f"{100 * x:.1f} %"
    return f"{x:.2f}"


METRIC_TITLE = {
    "log_score_pair": "Log-score de la paire de qualifiés réelle (plus haut = mieux)",
    "p_pair": "Probabilité prévue de la paire réelle (sans lissage)",
    "brier_qual": "Brier de qualification, moyenne par candidat (plus bas = mieux)",
    "crps": "CRPS des parts par candidat, points (plus bas = mieux)",
    "mae": "Erreur absolue de la médiane par candidat, points",
    "cov50": "Couverture de l'intervalle à 50 % (candidats)",
    "cov90": "Couverture de l'intervalle à 90 % (candidats)",
    "bloc_crps": "CRPS des parts par bloc, points",
    "bloc_mae": "Erreur absolue de la médiane par bloc, points",
    "bloc_cov50": "Couverture à 50 % (blocs)",
    "bloc_cov90": "Couverture à 90 % (blocs)",
    "unforecast_share": "Part des candidats officiels hors configuration connue, points",
}


def metric_table(s: pl.DataFrame, metric: str, models: list[str]) -> str:
    """Tableau élection × horizon × modèle d'une métrique, avec la moyenne pondérée."""
    sub = s.filter(pl.col("metric") == metric)
    elections = sorted(sub["election"].unique().to_list())
    horizons = sorted(sub["horizon"].unique().to_list(), reverse=True)
    head = "| élection | horizon | " + " | ".join(MODEL_LABEL.get(m, m) for m in models) + " |"
    lines = [f"### {METRIC_TITLE.get(metric, metric)}", "", head,
             "|---|---|" + "---|" * len(models)]
    val = {(r["election"], r["horizon"], r["model"]): r["value"] for r in sub.iter_rows(named=True)}
    for e in elections:
        for h in horizons:
            cells = [_fmt(metric, val.get((e, h, m))) for m in models]
            lines.append(f"| {e} | {h} | " + " | ".join(cells) + " |")
    wm = weighted_mean(sub, ["horizon", "model"])
    wv = {(r["horizon"], r["model"]): r["value"] for r in wm.iter_rows(named=True)}
    for h in horizons:
        cells = [_fmt(metric, wv.get((h, m))) for m in models]
        lines.append(f"| **moyenne pondérée** | {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(res: ScoreResult, variants: list[str]) -> None:
    out = res.out
    tag = "" if variants == ["floor"] else "_" + "_".join(variants)
    res.metrics.write_parquet(out / f"metrics{tag}.parquet")
    res.forecasts.write_parquet(out / f"forecasts{tag}.parquet")
    res.calibration.write_parquet(out / f"calibration{tag}.parquet")
    edir = out / "error_models"
    edir.mkdir(exist_ok=True)
    for e, em in res.error_models.items():
        (edir / f"exclude_{e}.json").write_text(
            json.dumps(em.to_dict(), indent=2, ensure_ascii=False, default=str)
        )
    for (v, e), em in res.variant_error_models.items():
        (edir / f"exclude_{e}_{v}.json").write_text(
            json.dumps(em.to_dict(), indent=2, ensure_ascii=False, default=str)
        )
    s = res.metrics.filter(pl.col("level") == "summary")
    ref = variants[0]
    md = [f"# Backtest premier tour : métriques (variante `{ref}`)", ""]
    md.append(
        f"Généré le {dt.date.today().isoformat()}. Élections pondérées "
        f"{ {k: v for k, v in ELECTION_WEIGHT.items()} } (1 sinon). Horizons en jours."
    )
    md.append("")
    sref = s.filter(pl.col("variant") == ref)
    models = [m for m in MODELS if m in sref["model"].unique().to_list()]
    for metric in SUMMARY_METRICS:
        md.append(metric_table(sref, metric, models))
    if len(variants) > 1:
        md.append("## Variantes du modèle\n")
        for metric in ("log_score_pair", "crps", "cov50", "cov90", "bloc_crps"):
            sv = s.filter(pl.col("model").is_in(["model", "model_noerror"])).with_columns(
                (pl.col("model") + "@" + pl.col("variant")).alias("model")
            )
            cols = [f"{m}@{v}" for m in ("model", "model_noerror") for v in variants]
            md.append(metric_table(sv, metric, cols))
    (out / f"metrics{tag}.md").write_text("\n".join(md))
    plot_calibration(res.calibration.filter(pl.col("variant") == ref), out / f"calibration{tag}.png")
    plot_scores(sref, out / f"scores_by_horizon{tag}.png")


def plot_calibration(cal: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vvs.model.plots import INK, MUTED, _style

    colors = {"model": "#2a78d6", "model_noerror": "#8a8984", "mean14": "#eb6834",
              "last_pollster": "#3a9b5c"}
    horizons = sorted(cal["horizon"].unique().to_list(), reverse=True)
    levels = ["candidate", "bloc"]
    fig, axes = plt.subplots(len(levels), len(horizons), figsize=(3.3 * len(horizons), 6.4),
                             sharex=True, sharey=True, squeeze=False)
    w = pl.col("election").replace_strict(ELECTION_WEIGHT, default=1.0, return_dtype=pl.Float64)
    agg = (
        cal.with_columns(w.alias("w"))
        .group_by("level", "horizon", "model", "nominal")
        .agg(((pl.col("hits") * pl.col("w")).sum() / (pl.col("n") * pl.col("w")).sum()).alias("obs"))
    )
    for i, lvl in enumerate(levels):
        for j, h in enumerate(horizons):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
            for m, col in colors.items():
                d = agg.filter((pl.col("level") == lvl) & (pl.col("horizon") == h)
                               & (pl.col("model") == m)).sort("nominal")
                if d.height:
                    ax.plot(d["nominal"], d["obs"], marker="o", markersize=3, color=col,
                            label=MODEL_LABEL[m], linewidth=1.4)
            ax.set_title(f"{'candidats' if lvl == 'candidate' else 'blocs'}, τ = {h} j",
                         fontsize=9, loc="left", color=INK)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            _style(ax)
            if i == len(levels) - 1:
                ax.set_xlabel("couverture nominale", fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel("couverture observée", fontsize=8, color=MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Calibration des intervalles centraux (élections pondérées)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scores(s: pl.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vvs.model.plots import INK, MUTED, _style

    colors = {"model": "#2a78d6", "model_noerror": "#8a8984", "mean14": "#eb6834",
              "last_pollster": "#3a9b5c"}
    metrics = ["log_score_pair", "crps", "brier_qual", "bloc_crps"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.2))
    for ax, metric in zip(axes, metrics):
        wm = weighted_mean(s.filter(pl.col("metric") == metric), ["horizon", "model"])
        for m, col in colors.items():
            d = wm.filter(pl.col("model") == m).sort("horizon")
            if d.height:
                ax.plot(d["horizon"], d["value"], marker="o", markersize=3, color=col,
                        label=MODEL_LABEL[m], linewidth=1.4)
        ax.set_title(METRIC_TITLE[metric].split(" (")[0], fontsize=8, loc="left", color=INK)
        ax.invert_xaxis()
        ax.set_xlabel("horizon (jours)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
