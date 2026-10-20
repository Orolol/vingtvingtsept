"""Étude : backtest du premier tour (phase 3), variantes du modèle et rapport.

    uv run python studies/backtest_first_round.py fits-cv [--out runs/<date>-backtest]
    uv run python studies/backtest_first_round.py report  [--out runs/<date>-backtest]

Les ajustements de référence (`floor`, plancher d'erreur) et sans plancher (`phase2`) se lancent
par `vvs backtest --stage fit` ; cette étude ajoute :

- `fits-cv` : validation croisée 2017 contre 2022 des priors du facteur de design et des effets
  maison candidat. Pour évaluer 2022, le prior est recentré sur le postérieur de l'ajustement
  2017 à 7 jours (variante `floor`), et inversement. Deux variantes séparées pour attribuer
  l'effet : `cv_design` (Beta de moyenne le postérieur de l'autre élection, a + b = 20 comme le
  prior de la phase 2) et `cv_house` (demi-normale dont la médiane est le postérieur de l'autre
  élection : échelle = médiane / 0,6745).
- `report` : score de toutes les variantes présentes (`vvs.eval.scoring`), tableaux du modèle
  d'erreur (paramètres par pli, écarts types par bloc et horizon, comparaison aux tableaux de
  Jennings–Wlezien), figures, et `tables.md` que le rapport cite.
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
from vvs.eval import backtest as bt

CV_ELECTIONS = (2017, 2022)
ALL_ELECTIONS = (2002, 2007, 2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
# Jennings–Wlezien, écart type de l'erreur sondage − résultat en points (premier tour) :
# France sans 1965 et 1969, et toutes présidentielles de la base (vvs.ingest.jennings_wlezien).
JW_REFERENCE = {
    "France, présidentielle, 1er tour": {7: 2.6, 30: 3.2, 90: 4.4, 180: 5.9},
    "Tous pays, présidentielles": {7: 4.6, 30: 6.7, 90: 7.4, 180: 8.9},
}


def _posterior(out: Path, election: int) -> dict:
    m = json.loads((bt.fit_dir(out, "floor", election, 7) / "manifest.json").read_text())
    return m["posterior"]


def cv_priors(out: Path) -> dict[str, dict[int, bt.Priors]]:
    base = bt.VARIANTS["floor"]
    other = {2017: 2022, 2022: 2017}
    design, house = {}, {}
    for e, o in other.items():
        post = _posterior(out, o)
        mean = post["design"]["median"]
        design[e] = replace(base, design_a=20 * mean, design_b=20 * (1 - mean))
        house[e] = replace(base, house_cand=post["sigma_house_cand"]["median"] / 0.6745)
    return {"cv_design": design, "cv_house": house}


def fits_cv(out: Path, jobs: int) -> None:
    for name, priors in cv_priors(out).items():
        (out / f"priors_{name}.json").write_text(
            json.dumps({str(e): p.__dict__ for e, p in priors.items()}, indent=2)
        )
        plan = bt.plan_fits(out, CV_ELECTIONS, HORIZONS, name, priors)
        bt.run_fits(plan, out, bt.FitSettings(), n_parallel=jobs)


# --- rapport


def error_model_tables(ems: dict, rng: np.random.Generator) -> str:
    from vvs.eval.scoring import official
    from vvs.registry import default_registry

    reg = default_registry()
    lines = ["## Modèle d'erreur η(τ)", ""]
    lines.append(
        "Paramètres par pli (élection exclue de la calibration). Variances sur l'échelle logit ; "
        "σ : écarts types de l'erreur finale (bloc, candidat), ρ : terme des petits candidats "
        "(variance ρ/p), a : variances du mouvement à 90 jours, γ : forme en τ "
        "(Jennings–Wlezien), ν : degrés de liberté de l'échelle commune."
    )
    lines.append("")
    lines.append("| exclue | σ_Bf | σ_Cf | ρ | a_B | a_C | γ | ν | âge final | poids de bloc | biais admis |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e, em in sorted(ems.items()):
        p = em.params
        w = ", ".join(f"{b} {v:.2f}" for b, v in p.bloc_weight.items())
        bias = ", ".join(f"{b} {v:+.3f}" for b, v in p.bias.items()) or "aucun"
        lines.append(
            f"| {e} | {p.sigma_bloc_final:.3f} | {p.sigma_cand_final:.3f} | {p.rho:.4f} | "
            f"{p.a_bloc:.3f} | {p.a_cand:.3f} | {p.gamma:.2f} | {p.nu:g} | {p.final_age:.1f} | "
            f"{w} | {bias} |"
        )
    lines.append("")
    lines.append("Validation croisée interne de ν (log-vraisemblance des élections laissées) :")
    lines.append("")
    for e, em in sorted(ems.items()):
        sc = em.info.get("nu_scores", {})
        lines.append(f"- exclue {e} : " + ", ".join(f"ν={k} : {v:.1f}" for k, v in sc.items()))
    lines.append("")
    lines.append("Erreurs finales des blocs (log résultat − log agrégat 14 j), règle du biais :")
    lines.append("")
    em_all = next(iter(ems.values()))
    lines.append("| exclue | bloc | erreurs par élection | même signe | biais retenu |")
    lines.append("|---|---|---|---|---|")
    for e, em in sorted(ems.items()):
        for b, d in em.info.get("bias_diagnostics", {}).items():
            errs = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(d["errors"].items()))
            lines.append(
                f"| {e} | {b} | {errs} | {'oui' if d['same_sign'] else 'non'} | "
                f"{em.params.bias.get(b, 0.0):+.3f} |"
            )
    lines.append("")
    # écarts types par bloc et horizon (pli sans 2022, le plus proche de l'usage 2027)
    ref = ems.get(2022) or em_all
    s = ref.summary(HORIZONS)
    lines.append(
        "Écarts types marginaux de η sur l'échelle logit (pli sans 2022, ν inclus) et "
        "corrélation intra-bloc :"
    )
    lines.append("")
    lines.append("| horizon | bloc | σ bloc | σ candidat | σ total | corrélation intra-bloc |")
    lines.append("|---|---|---|---|---|---|")
    for r in s.sort("horizon", descending=True).iter_rows(named=True):
        lines.append(
            f"| {r['horizon']} | {r['bloc']} | {r['sd_bloc']:.3f} | {r['sd_cand']:.3f} | "
            f"{r['sd_total']:.3f} | {r['corr_within_bloc']:.2f} |"
        )
    lines.append("")
    # échelle en points comparée à Jennings–Wlezien
    lines.append(
        "Échelle en points : écart type de la part au vote, à parts exactes à la date de "
        "prévision égales au résultat officiel (moyenne quadratique sur les candidats ; entre "
        "parenthèses, candidats de plus de 5 %). Comparaison : écart type de l'erreur du dernier "
        "sondage dans Jennings–Wlezien (un sondage, pas un agrégat)."
    )
    lines.append("")
    head = "| pli | " + " | ".join(f"τ = {h}" for h in HORIZONS) + " |"
    lines += [head, "|---|" + "---|" * len(HORIZONS)]
    for e, em in sorted(ems.items()):
        off = official(reg, e)
        cells = []
        for h in HORIZONS:
            sd = em.points_sd(off.shares / 100, off.blocs, h, rng, n=20_000)
            big = off.shares > 5
            cells.append(f"{np.sqrt((sd**2).mean()):.1f} ({np.sqrt((sd[big]**2).mean()):.1f})")
        lines.append(f"| sans {e}, config. {e} | " + " | ".join(cells) + " |")
    for name, ref_sd in JW_REFERENCE.items():
        lines.append(f"| JW : {name} | " + " | ".join(f"{ref_sd[h]:.1f}" for h in HORIZONS) + " |")
    jw = (ems.get(2022) or em_all).info.get("jw")
    if jw:
        lines.append("")
        lines.append(
            f"Forme en τ (Jennings–Wlezien, log-parts) : v(h) = {jw['v7']:.3f} + "
            f"{jw['a83']:.3f}·((h − 7)/83)^{jw['gamma']:.2f}."
        )
    return "\n".join(lines) + "\n"


def report(out: Path, n_draws: int) -> None:
    from vvs.eval.scoring import score_backtest

    variants = [v for v in ("floor", "phase2", "cv_design", "cv_house")
                if (out / "fits" / v).exists()]
    # 1) référence : toutes les élections, variante floor -> metrics.parquet, metrics.md
    res = score_backtest(out, list(ALL_ELECTIONS), list(HORIZONS), variant="floor",
                         n_draws=n_draws)
    print(res.summary())
    # 2) variantes : élections ajustées dans chaque variante
    others = [v for v in variants if v != "floor"]
    res_v = None
    if others:
        res_v = score_backtest(out, [2012, 2017, 2022], list(HORIZONS), n_draws=n_draws,
                               variants=["floor", *others])
    rng = np.random.default_rng(0)
    md = [f"# Backtest premier tour : tableaux ({dt.date.today().isoformat()})", ""]
    md.append(error_model_tables(res.error_models, rng))
    md.append(variant_tables(res_v) if res_v is not None else "")
    md.append(forecast_tables(res))
    md.append(fit_table(out, variants))
    md.append(posterior_table(out, variants))
    (out / "tables.md").write_text("\n".join(md))
    plot_qualification(res, out / "qualification_by_horizon.png")
    print(f"écrit : {out}/tables.md")


def variant_tables(res) -> str:
    from vvs.eval.scoring import weighted_mean

    s = res.metrics.filter(pl.col("level") == "summary")
    lines = ["## Variantes du modèle (2012, 2017, 2022)", ""]
    lines.append(
        "Modèle + η ; écarts à la variante `floor` sur les mêmes ajustements, élections "
        "communes seulement. Log-score : plus haut = mieux ; CRPS : plus bas = mieux."
    )
    lines.append("")
    for metric in ("log_score_pair", "crps", "bloc_crps", "cov50", "cov90", "brier_qual"):
        sub = s.filter((pl.col("metric") == metric) & (pl.col("model") == "model"))
        lines.append(f"### {metric}")
        lines.append("")
        variants = sub["variant"].unique().sort().to_list()
        lines.append("| élection | horizon | " + " | ".join(variants) + " |")
        lines.append("|---|---|" + "---|" * len(variants))
        val = {(r["election"], r["horizon"], r["variant"]): r["value"]
               for r in sub.iter_rows(named=True)}
        for e in sorted(sub["election"].unique().to_list()):
            for h in HORIZONS:
                cells = [f"{val[(e, h, v)]:.3f}" if (e, h, v) in val else "–" for v in variants]
                lines.append(f"| {e} | {h} | " + " | ".join(cells) + " |")
        for es, label in (((2017, 2022), "moyenne 2017-2022"), ((2012, 2017, 2022), "moyenne 2012-2022")):
            wm = weighted_mean(sub.filter(pl.col("election").is_in(list(es))), ["variant"])
            wv = dict(zip(wm["variant"].to_list(), wm["value"].to_list()))
            ok = [v for v in variants
                  if all((e, h, v) in val for e in es for h in HORIZONS)]
            cells = [f"{wv[v]:.3f}" if v in ok else "–" for v in variants]
            lines.append(f"| **{label}** | tous | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def forecast_tables(res) -> str:
    """Prévision du modèle + η par candidat officiel et horizon, contre le résultat."""
    from vvs.model.plots import display_name

    fc = res.forecasts.filter(
        (pl.col("variant") == "floor") & (pl.col("model") == "model")
        & (pl.col("level") == "candidate")
    )
    lines = ["## Prévisions rejouées du modèle + η (médiane [90 %], P(qualifié))", ""]
    for e in sorted(fc["election"].unique().to_list()):
        sub = fc.filter(pl.col("election") == e)
        hs = sorted(sub["horizon"].unique().to_list(), reverse=True)
        lines.append(f"### {e}")
        lines.append("")
        lines.append("| candidat | résultat | " + " | ".join(f"τ = {h}" for h in hs) + " |")
        lines.append("|---|---|" + "---|" * len(hs))
        order = (sub.group_by("unit").agg(pl.col("official").first())
                 .sort("official", descending=True))
        for cid, off in order.iter_rows():
            cells = []
            for h in hs:
                r = sub.filter((pl.col("unit") == cid) & (pl.col("horizon") == h))
                if r.is_empty():
                    cells.append("hors config.")
                    continue
                r = r.row(0, named=True)
                mark = "" if r["in90"] else " ✗"
                cells.append(f"{r['median']:.1f} [{r['q05']:.1f}–{r['q95']:.1f}]{mark}, "
                             f"{100 * r['p_qualify']:.0f} %")
            lines.append(f"| {display_name(cid)} | {off:.2f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("✗ : résultat hors de l'intervalle à 90 %.")
    return "\n".join(lines) + "\n"


def plot_qualification(res, path: Path) -> None:
    """P(qualifié) rejouée par horizon (modèle + η), candidats officiels ayant dépassé 5 %."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vvs.model.plots import INK, MUTED, _style, display_name

    fc = res.forecasts.filter(
        (pl.col("variant") == "floor") & (pl.col("model") == "model")
        & (pl.col("level") == "candidate")
    )
    elections = sorted(fc["election"].unique().to_list())
    fig, axes = plt.subplots(1, len(elections), figsize=(3.3 * len(elections), 3.4),
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, e in zip(axes[0], elections):
        sub = fc.filter(pl.col("election") == e)
        keep = (sub.group_by("unit").agg(pl.col("p_qualify").max(), pl.col("qualified").first())
                .filter(pl.col("p_qualify") > 0.05))
        for i, (cid, _, q) in enumerate(keep.sort("p_qualify", descending=True).iter_rows()):
            d = sub.filter(pl.col("unit") == cid).sort("horizon")
            ax.plot(d["horizon"], d["p_qualify"], marker="o", markersize=3,
                    linewidth=2.0 if q else 1.0, linestyle="-" if q else "--",
                    color=cmap(i % 10), label=display_name(cid))
        ax.invert_xaxis()
        ax.set_title(str(e), fontsize=10, loc="left", color=INK)
        ax.set_xlabel("horizon (jours)", fontsize=8, color=MUTED)
        ax.legend(fontsize=6, frameon=False)
        _style(ax)
    axes[0, 0].set_ylabel("P(qualifié)", fontsize=8, color=MUTED)
    fig.suptitle("Probabilité de qualification rejouée (trait plein : qualifiés réels)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fit_table(out: Path, variants: list[str]) -> str:
    lines = ["## Ajustements", ""]
    lines.append("| variante | élection | horizon | asof | sondages | config. | NUTS (s) | r_hat max (parts) | ESS min (parts) | divergences |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        for d in sorted((out / "fits" / v).glob("*-*")):
            mf = d / "manifest.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text())
            conf = json.loads((d / "configuration.json").read_text())
            der = m["diagnostics"].get("derived", {})
            rh = max((x["r_hat_max"] for x in der.values()), default=float("nan"))
            ess = min((x["ess_min"] for x in der.values()), default=float("nan"))
            lines.append(
                f"| {v} | {m['election']} | {conf['horizon']} | {m['asof']} | "
                f"{m['data']['n_polls']} | {conf['source']} ({len(conf['candidates'])}) | "
                f"{m['duration_s']['inference']} | {rh:.3f} | {ess:.0f} | "
                f"{m['diagnostics'].get('divergences', '–')} |"
            )
    return "\n".join(lines) + "\n"


def posterior_table(out: Path, variants: list[str]) -> str:
    keys = ("design", "n_max", "sigma_house_bloc", "sigma_house_cand", "sigma_beta", "sigma_delta")
    lines = ["## Postérieurs des hyperparamètres (médiane [5 %, 95 %])", ""]
    lines.append("| variante | élection | horizon | " + " | ".join(keys) + " |")
    lines.append("|---|---|---|" + "---|" * len(keys))
    for v in variants:
        for d in sorted((out / "fits" / v).glob("*-*")):
            mf = d / "manifest.json"
            if not mf.exists():
                continue
            m = json.loads(mf.read_text())
            post = m["posterior"]
            cells = []
            for k in keys:
                if k in post:
                    p = post[k]
                    fmt = "{:.0f}" if k == "n_max" else "{:.3f}"
                    cells.append(
                        f"{fmt.format(p['median'])} [{fmt.format(p['q05'])}, {fmt.format(p['q95'])}]"
                    )
                else:
                    cells.append("–")
            h = int(d.name.split("-")[1])
            lines.append(f"| {v} | {m['election']} | {h} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["fits-cv", "report"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--draws", type=int, default=10_000)
    args = ap.parse_args()
    out = args.out or paths.RUNS / f"{dt.date.today().isoformat()}-backtest"
    if args.stage == "fits-cv":
        fits_cv(out, args.jobs)
    else:
        report(out, args.draws)


if __name__ == "__main__":
    main()
