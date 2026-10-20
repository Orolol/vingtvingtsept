"""Backtest de la calibration de η étendue aux scrutins de liste, et du biais d'extrême droite
depuis 2017.

    uv run python studies/backtest_lists.py [--out runs/<date>-backtest-lists]
        [--fits-from runs/2026-10-20-backtest] [--sim-old …] [--sim-new …]

Mêmes ajustements floor (liens vers `--fits-from`, aucun nouvel ajustement NUTS), même graine
([0, élection, horizon]) : seule la calibration de η change.

- `floor` : référence, η calibré sur les présidentielles 2002-2022 sans l'élection évaluée ;
- `lists` : `ErrorModel.fit(with_lists=True)`, erreurs finales des européennes 2014-2024 et
  législatives 2022-2024 ajoutées, poids 0,5 (`vvs.model.error_lists`) ;
- `fr2017`, `lists_fr2017` : biais d'extrême droite depuis 2017 dans l'erreur finale
  (`far_right_bias_since_2017=True`), sans puis avec les scrutins de liste ; le biais n'agit
  que sur 2017 et 2022.

Contrôle : `floor` et `lists` doivent reproduire les calibrations « présidentielles » et
« étendue, finales seules » de runs/2026-10-20-calibration-extension/coverage.parquet.

Sorties : metrics_*.parquet / .md, forecasts, calibration (scoring), comparison.parquet,
comparison.md, admission.json, calibration_<variante>.png, report.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.eval.backtest import ELECTION_WEIGHT, fit_dir
from vvs.eval.scoring import plot_calibration, score_backtest

ELECTIONS = (2002, 2007, 2012, 2017, 2022)
REFERENCE = (2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
VARIANTS = ("floor", "lists", "fr2017", "lists_fr2017")
# (variante, base de comparaison)
PAIRS = (("lists", "floor"), ("fr2017", "floor"), ("lists_fr2017", "lists"))
METRICS = ("log_score_pair", "crps", "cov50", "cov90", "bloc_cov50", "bloc_cov90", "bloc_crps",
           "brier_qual")
COV_NOMINAL = {"cov50": 0.5, "cov90": 0.9, "bloc_cov50": 0.5, "bloc_cov90": 0.9}
STUDY = paths.RUNS / "2026-10-20-calibration-extension" / "coverage.parquet"
STUDY_CALIB = {"floor": "présidentielles", "lists": "étendue, finales seules"}
# horizons où la variante agit par construction (docs/07, règle du jour) : l'erreur finale et
# les échelles de mouvement réestimées conjointement entrent dans η à tous les horizons
ACTS = {"lists": HORIZONS, "fr2017": HORIZONS, "lists_fr2017": HORIZONS}
TOL = 0.01  # tolérance « ne dégrade pas » (log-score, CRPS), comme studies/volatility.py


def link_fits(out: Path, fits_from: Path) -> None:
    for v in VARIANTS:
        for e in ELECTIONS:
            for h in HORIZONS:
                src, dst = fit_dir(fits_from, "floor", e, h), fit_dir(out, v, e, h)
                if not (src / "draws.npz").exists():
                    raise FileNotFoundError(f"ajustement floor absent : {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src.resolve())


def wmean(df: pl.DataFrame, col: str, elections) -> float:
    sub = df.filter(pl.col("election").is_in(list(elections)))
    w = np.array([ELECTION_WEIGHT.get(e, 1.0) for e in sub["election"]])
    return float((w * sub[col].to_numpy()).sum() / w.sum())


def fmt(metric: str, x: float, diff: bool = False) -> str:
    if metric in COV_NOMINAL:
        return f"{100 * x:+.0f}" if diff else f"{100 * x:.0f} %"
    return f"{x:+.3f}" if diff else f"{x:.3f}"


def reproduction(metrics: pl.DataFrame) -> list[str]:
    """Écart maximal aux valeurs de l'étude, par variante."""
    if not STUDY.exists():
        return ["Étude absente : contrôle de reproduction sauté.", ""]
    st = pl.read_parquet(STUDY).filter(pl.col("level") == "summary")
    lines = ["| variante | calibration de l'étude | valeurs comparées | écart absolu max |",
             "|---|---|---|---|"]
    for v, calib in STUDY_CALIB.items():
        a = metrics.filter((pl.col("variant") == v) & (pl.col("level") == "summary")).select(
            "election", "horizon", "model", "metric", "value")
        b = st.filter(pl.col("calibration") == calib).select(
            "election", "horizon", "model", "metric", pl.col("value").alias("study"))
        j = a.join(b, on=["election", "horizon", "model", "metric"], how="inner").filter(
            pl.col("model") == "model")
        d = (j["value"] - j["study"]).abs()
        lines.append(f"| {v} | {calib} | {j.height} | {d.max():.2e} |")
    return lines + [""]


def comparison(metrics: pl.DataFrame) -> pl.DataFrame:
    s = metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model")
                       & pl.col("metric").is_in(METRICS))
    return s.pivot(on="variant", index=["election", "horizon", "metric"], values="value").sort(
        "metric", "election", "horizon", descending=[False, False, True])


def comparison_md(wide: pl.DataFrame, v: str, base: str) -> list[str]:
    md = [f"## `{v}` contre `{base}`", ""]
    for m in METRICS:
        sub = wide.filter(pl.col("metric") == m).with_columns((pl.col(v) - pl.col(base)).alias("d"))
        md += [f"### {m}", "", "| élection | 180 j | 90 j | 30 j | 7 j |", "|---|---|---|---|---|"]
        for e in ELECTIONS:
            cells = []
            for h in HORIZONS:
                r = sub.filter((pl.col("election") == e) & (pl.col("horizon") == h)).row(0, named=True)
                cells.append(f"{fmt(m, r[base])} → {fmt(m, r[v])} ({fmt(m, r['d'], True)})")
            md.append(f"| {e} | " + " | ".join(cells) + " |")
        for lab, els in (("moyenne 2012-2022", REFERENCE), ("moyenne pondérée 2002-2022", ELECTIONS)):
            cells = []
            for h in HORIZONS:
                g = sub.filter(pl.col("horizon") == h)
                a, b = wmean(g, base, els), wmean(g, v, els)
                cells.append(f"{fmt(m, a)} → {fmt(m, b)} ({fmt(m, b - a, True)})")
            md.append(f"| **{lab}** | " + " | ".join(cells) + " |")
        md.append("")
    return md


def admission(wide: pl.DataFrame, v: str, base: str) -> tuple[dict, list[str]]:
    """Règle de docs/07 (version du jour), sur les moyennes 2012-2022 et pondérée 2002-2022."""
    rows = []
    for h in HORIZONS:
        g = wide.filter(pl.col("horizon") == h)
        r = {"horizon": h}
        for m, sign in (("log_score_pair", 1), ("crps", -1)):
            gm = g.filter(pl.col("metric") == m).with_columns(
                (sign * (pl.col(v) - pl.col(base))).alias("gain"))
            r[f"{m}_gain_ref"] = wmean(gm, "gain", REFERENCE)
            r[f"{m}_gain_w"] = wmean(gm, "gain", ELECTIONS)
            ref = gm.filter(pl.col("election").is_in(list(REFERENCE)))
            r[f"{m}_n_pos"] = int((ref["gain"] > 1e-9).sum())
            r[f"{m}_n_neg"] = int((ref["gain"] < -1e-9).sum())
        for m, nom in COV_NOMINAL.items():
            gm = g.filter(pl.col("metric") == m)
            for lab, els in (("ref", REFERENCE), ("w", ELECTIONS)):
                before = abs(wmean(gm, base, els) - nom)
                after = abs(wmean(gm, v, els) - nom)
                r[f"{m}_gap_{lab}"] = (before, after)
        rows.append(r)
    acts = ACTS[v]
    crit = {}
    for lab in ("ref", "w"):
        gain = all(r[f"{m}_gain_{lab}"] > 0 for r in rows if r["horizon"] in acts
                   for m in ("log_score_pair", "crps"))
        no_harm = all(r[f"{m}_gain_{lab}"] >= -TOL for r in rows if r["horizon"] not in acts
                      for m in ("log_score_pair", "crps"))
        crit[f"c1_{lab}"] = bool(gain and no_harm)
    # même signe sur au moins deux élections de référence, là où la variante agit
    crit["c2"] = all(r[f"{m}_n_pos"] >= 2 for r in rows if r["horizon"] in acts
                     for m in ("log_score_pair", "crps"))
    # lecture agrégée (studies/volatility.py) : écart moyen à la nominale sur les horizons et les
    # niveaux 50 et 90 %, non accru
    for lab in ("ref", "w"):
        for lvl, ms in (("cand", ("cov50", "cov90")), ("bloc", ("bloc_cov50", "bloc_cov90"))):
            before = float(np.mean([r[f"{m}_gap_{lab}"][0] for r in rows for m in ms]))
            after = float(np.mean([r[f"{m}_gap_{lab}"][1] for r in rows for m in ms]))
            crit[f"c3_mean_{lvl}_{lab}"] = (before, after)
    for lab in ("ref", "w"):
        crit[f"c3_cand_{lab}"] = all(r[f"{m}_gap_{lab}"][1] <= r[f"{m}_gap_{lab}"][0] + 1e-9
                                     for r in rows for m in ("cov50", "cov90"))
        crit[f"c3_bloc_{lab}"] = all(r[f"{m}_gap_{lab}"][1] <= r[f"{m}_gap_{lab}"][0] + 1e-9
                                     for r in rows for m in ("bloc_cov50", "bloc_cov90"))
    md = [f"### Règle d'admission : `{v}` contre `{base}`", "",
          ("Gains (log-score : écart ; CRPS : baisse ; positif = mieux) en moyenne 2012-2022 / "
           "pondérée 2002-2022 ; entre parenthèses, élections de référence où le gain est "
           "positif / négatif. Couverture : écart à la nominale avant → après (2012-2022), "
           "candidats puis blocs."), "",
          "| horizon | log-score | CRPS | couv. 50 % | couv. 90 % | blocs 50 % | blocs 90 % |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for m in ("log_score_pair", "crps"):
            cells.append(f"{r[f'{m}_gain_ref']:+.3f} / {r[f'{m}_gain_w']:+.3f} "
                         f"({r[f'{m}_n_pos']}+ / {r[f'{m}_n_neg']}−)")
        for m in COV_NOMINAL:
            b, a = r[f"{m}_gap_ref"]
            cells.append(f"{100 * b:.0f} → {100 * a:.0f} pts")
        md.append(f"| {r['horizon']} | " + " | ".join(cells) + " |")
    md += ["", (f"Horizons d'action retenus : {', '.join(map(str, acts))} j. Critère 1 (gain à "
           f"ces horizons, aucun autre dégradé de plus de {TOL}) : 2012-2022 "
           f"**{'oui' if crit['c1_ref'] else 'non'}**, pondérée **{'oui' if crit['c1_w'] else 'non'}** ; "
           f"critère 2 (au moins 2 élections de référence sur 3 à chaque horizon d'action, "
           f"log-score et CRPS) : **{'oui' if crit['c2'] else 'non'}** ; critère 3 (écart à la "
           f"nominale non accru à aucun horizon) : candidats **{'oui' if crit['c3_cand_ref'] else 'non'}** "
           f"(pondérée {'oui' if crit['c3_cand_w'] else 'non'}), blocs "
           f"**{'oui' if crit['c3_bloc_ref'] else 'non'}** (pondérée {'oui' if crit['c3_bloc_w'] else 'non'}). "
           "Écart moyen à la nominale (horizons × 50 et 90 %), 2012-2022 / pondérée : candidats "
           + " / ".join(f"{100 * crit[f'c3_mean_cand_{lab}'][0]:.1f} → {100 * crit[f'c3_mean_cand_{lab}'][1]:.1f}"
                        for lab in ("ref", "w"))
           + " pts ; blocs "
           + " / ".join(f"{100 * crit[f'c3_mean_bloc_{lab}'][0]:.1f} → {100 * crit[f'c3_mean_bloc_{lab}'][1]:.1f}"
                        for lab in ("ref", "w"))
           + " pts."), ""]
    return {"rows": rows, "criteria": crit, "acts": list(acts)}, md


def error_models(out: Path) -> dict[tuple[str, int], dict]:
    """Modèles d'erreur écrits par le score, par (variante, élection exclue)."""
    ems = {}
    for e in ELECTIONS:
        for v in VARIANTS:
            f = out / "error_models" / (f"exclude_{e}.json" if v == "floor" else f"exclude_{e}_{v}.json")
            ems[(v, e)] = json.loads(f.read_text())
    return ems


def params_table(ems: dict) -> list[str]:
    keys = ("sigma_bloc_final", "sigma_cand_final", "rho", "a_bloc", "a_cand", "nu")
    md = ["| élection exclue | variante | " + " | ".join(keys) + " | biais appliqués |",
          "|---|---|" + "---|" * len(keys) + "---|"]
    for e in ELECTIONS:
        for v in ("floor", "lists"):
            pr = ems[(v, e)]["params"]
            bias = ", ".join(f"{b} {x:+.3f}" for b, x in pr["bias"].items()) or "—"
            md.append(f"| {e} | {v} | " + " | ".join(f"{pr[k]:.3g}" for k in keys) + f" | {bias} |")
    return md + [""]


def far_right_table(ems: dict) -> list[str]:
    md = ["| variante | élection exclue | μ estimé (logit) | appliqué | erreurs depuis 2017 (logit) |",
          "|---|---|---|---|---|"]
    for (v, e), em in sorted(ems.items(), key=lambda x: (x[0][0], x[0][1])):
        d = em["info"].get("far_right_bias_since_2017")
        if not d:
            continue
        errs = ", ".join(f"{k} {x:+.3f}" for k, x in d["errors"].items())
        md.append(f"| {v} | {e} | {d['mu']:+.3f} | {'oui' if d['applied'] else 'non'} | {errs} |")
    return md + [""]


def candidate_rows(forecasts: pl.DataFrame, units: dict[int, list[str]]) -> list[str]:
    fc = forecasts.filter((pl.col("model") == "model") & (pl.col("level") == "bloc"))
    md = ["| élection | horizon | bloc | résultat | " + " | ".join(VARIANTS) + " |",
          "|---|---|---|---|" + "---|" * len(VARIANTS)]
    for e, blocs in units.items():
        for h in HORIZONS:
            for b in blocs:
                cells, off = [], None
                for v in VARIANTS:
                    r = fc.filter((pl.col("election") == e) & (pl.col("horizon") == h)
                                  & (pl.col("variant") == v) & (pl.col("unit") == b))
                    if r.is_empty():
                        cells.append("–")
                        continue
                    r = r.row(0, named=True)
                    off = r["official"]
                    cells.append(f"{r['median']:.1f} [{r['q05']:.1f}, {r['q95']:.1f}]")
                md.append(f"| {e} | {h} | {b} | {off:.1f} | " + " | ".join(cells) + " |")
    return md + [""]


DUELS = (("le_pen_marine", "philippe_edouard"), ("le_pen_marine", "melenchon_jean_luc"),
         ("le_pen_marine", "glucksmann_raphael"), ("bardella_jordan", "philippe_edouard"))


def simulation_md(old: Path, new: Path) -> list[str]:
    """Mélange et duels principaux de deux simulations 2027."""
    so, sn = (pl.read_parquet(d / "summary.parquet").filter(pl.col("condition_id") == "mix")
              for d in (old, new))
    j = so.join(sn, on="candidate_id", suffix="_new").filter(
        (pl.col("p_qualify") >= 0.01) | (pl.col("p_qualify_new") >= 0.01)).sort("p_win", descending=True)
    md = [f"Avant : `{old.name}` ; après : `{new.name}`. Mélange des configurations.", "",
          "| candidat | P(qualif.) | P(victoire) | part médiane [5-95 %] |", "|---|---|---|---|"]
    for r in j.iter_rows(named=True):
        md.append(f"| {r['candidate_id']} | {100 * r['p_qualify']:.1f} → {100 * r['p_qualify_new']:.1f} % "
                  f"| {100 * r['p_win']:.1f} → {100 * r['p_win_new']:.1f} % "
                  f"| {r['q50']:.1f} [{r['q05']:.1f}, {r['q95']:.1f}] → "
                  f"{r['q50_new']:.1f} [{r['q05_new']:.1f}, {r['q95_new']:.1f}] |")
    po, pn = (pl.read_parquet(d / "pairs.parquet").filter(pl.col("condition_id") == "mix")
              for d in (old, new))
    md += ["", "| duel | P(duel) | P(victoire du premier) | part médiane du premier [5-95 %] |",
           "|---|---|---|---|"]
    for a, b in DUELS:
        cells = []
        for p in (po, pn):
            r = p.filter(((pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
                         | ((pl.col("candidate_a") == b) & (pl.col("candidate_b") == a)))
            if r.is_empty():
                cells.append(None)
                continue
            r = r.row(0, named=True)
            flip = r["candidate_a"] != a
            pw = 1 - r["p_a_wins"] if flip else r["p_a_wins"]
            if flip:  # part du premier nommé : 100 − part de l'autre, quantiles inversés
                q = [100 - r["q50"], 100 - r["q95"], 100 - r["q05"]]
            else:
                q = [r["q50"], r["q05"], r["q95"]]
            cells.append((r["p_pair"], pw, q))
        if None in cells:
            md.append(f"| {a} – {b} | – | – | – |")
            continue
        (p0, w0, q0), (p1, w1, q1) = cells
        md.append(f"| {a} – {b} | {100 * p0:.1f} → {100 * p1:.1f} % | {100 * w0:.1f} → {100 * w1:.1f} % "
                  f"| {q0[0]:.1f} [{q0[1]:.1f}, {q0[2]:.1f}] → {q1[0]:.1f} [{q1[1]:.1f}, {q1[2]:.1f}] |")
    return md + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-backtest-lists")
    ap.add_argument("--fits-from", type=Path, default=paths.RUNS / "2026-10-20-backtest")
    ap.add_argument("--n-draws", type=int, default=10_000)
    ap.add_argument("--reuse", action="store_true",
                    help="relire les sorties du score au lieu de rescorer (tableaux seulement)")
    ap.add_argument("--sim-old", type=Path, default=None, help="simulation 2027 de référence")
    ap.add_argument("--sim-new", type=Path, default=None, help="simulation 2027 avec les listes")
    a = ap.parse_args()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    tag = "_" + "_".join(VARIANTS)
    if not a.reuse:
        link_fits(out, a.fits_from)
        res = score_backtest(out, list(ELECTIONS), list(HORIZONS), variants=list(VARIANTS),
                             n_draws=a.n_draws, seed=0)
        for v in VARIANTS[1:]:
            plot_calibration(res.calibration.filter(pl.col("variant") == v),
                             out / f"calibration_{v}.png")
    metrics = pl.read_parquet(out / f"metrics{tag}.parquet")
    forecasts = pl.read_parquet(out / f"forecasts{tag}.parquet")
    ems = error_models(out)
    wide = comparison(metrics)
    wide.write_parquet(out / "comparison.parquet")

    md = [f"# Backtest : η calibré avec les scrutins de liste ({dt.date.today().isoformat()})", "",
          (f"Tirages θ : {a.fits_from} (floor, réutilisés tels quels), {a.n_draws} tirages, graine "
           "[0, élection, horizon] identique pour toutes les variantes. 2002 et 2007 pondérées 0,5 "
           "dans la moyenne pondérée ; élections de référence 2012, 2017, 2022."), "",
          "## Reproduction de l'étude (runs/2026-10-20-calibration-extension)", "",
          *reproduction(metrics),
          "## Paramètres de η par pli (floor contre lists)", "", *params_table(ems)]
    adm = {}
    for v, base in PAIRS:
        info, lines = admission(wide, v, base)
        adm[f"{v}_vs_{base}"] = info
        md += comparison_md(wide, v, base) + lines
    md += ["## Biais d'extrême droite depuis 2017 : estimations par pli", "", *far_right_table(ems)]
    md += ["## Extrême droite (bloc) en 2017 et 2022 : médiane [5-95 %], points", "",
           *candidate_rows(forecasts, {2017: ["far_right"], 2022: ["far_right"]})]
    if a.sim_old and a.sim_new:
        md += ["## Simulation 2027", "", *simulation_md(a.sim_old, a.sim_new)]
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "admission.json").write_text(json.dumps(adm, indent=2, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
