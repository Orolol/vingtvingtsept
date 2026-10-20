"""Étude : étendre la calibration de η aux européennes et aux législatives.

    uv run python studies/calibration_extension.py [--out runs/<date>-calibration-extension]
        [--weight 0.5] [--skip-backtest]

docs/04-modele.md, section 2 : « avec cinq présidentielles, on complète la calibration avec
les législatives et européennes pour les composantes de bloc ». Entrées : `poll_errors`
(présidentielles 2002-2022) et `poll_errors_lists` (européennes 2014, 2019, 2024,
législatives 2022, 2024 ; `vvs.curate.lists`). Rien n'entre dans le modèle ici.

Questions, fixées avant de lire les résultats :

(a) Erreurs de bloc (sondages moins résultat, points d'exprimés) par scrutin et horizon :
    moyenne des 14 derniers jours et moyennes à 180, 90, 30, 7 jours (fenêtre de 30 jours),
    à côté des présidentielles. Même ordre de grandeur ? même signe (RN surestimé aux
    législatives 2024, sous-estimé aux européennes 2019) ? Aux législatives, NUPES et NFP
    sont rangés en gauche radicale (bloc dominant) : on donne aussi la somme gauche + gauche
    radicale.

(b) Paramètres de η calibrés sur les présidentielles seules, puis sur les présidentielles et
    les scrutins de liste pondérés `--weight` (0,5), en laissant de côté une présidentielle à
    la fois : σ de bloc et de candidat, échelles du mouvement, poids de bloc, ν, biais,
    corrélation intra-bloc, et corrélation empirique des erreurs de bloc entre élections.
    γ (forme en τ) est estimé sur Jennings–Wlezien et ne dépend pas de ces données.

    Entrée des scrutins de liste dans `ErrorModel.fit` : argument `extra_items`, items
    construits par `vvs.model.error_lists.list_items` (déplacé de cette étude dans le paquet),
    sous la clé du scrutin (`Item.election`, par exemple "euro2019"), jamais exclus ; la
    pondération passe par `Item.weight` (final : w ; mouvements : w / nombre d'horizons).
    Unités : les unités de résultat de lists.yaml dont la part finale dans les sondages est
    d'au moins `MIN_SHARE` points (les petites listes, parfois sondées à 0 ou « <0,5 »,
    donneraient des log-rapports dominés par l'arrondi). Erreur finale : log(résultat) −
    log(moyenne des 14 derniers jours) ; mouvements : log(moyenne finale) − log(moyenne des
    30 jours avant l'horizon), h = écart des âges moyens, horizons de `MOVE_HORIZONS`.

(c) Couverture du backtest 2002-2022 (ajustements de runs/2026-10-20-backtest/fits/floor,
    même graine) avec les deux calibrations, par `vvs.eval.scoring.score_backtest`
    (`error_kwargs={"items": ...}`), sans rien écrire dans le dossier du backtest.

Sorties : `<out>/report.md` (tableaux), `<out>/bloc_errors.parquet`, `<out>/eta_params.parquet`,
`<out>/coverage.parquet`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.model import error as er
from vvs.model import error_lists as el
from vvs.registry import BLOCS, default_registry

log = logging.getLogger("calibration_extension")

PRESIDENTIALS = (2002, 2007, 2012, 2017, 2022)
LIST_KEYS = el.LIST_KEYS
LIST_KINDS = ("final", "move")  # l'étude compare aussi la variante avec mouvements
BLOC_LABEL = {
    "far_left": "gauche rad.", "left": "gauche", "centre": "centre", "right": "droite",
    "far_right": "ext. droite", "other": "autres", "left_union": "gauches (somme)",
}
BLOC_ORDER = ["far_left", "left", "left_union", "centre", "right", "far_right", "other"]
BACKTEST_DIR = paths.RUNS / "2026-10-20-backtest"


# ---------------------------------------------------------------------------
# (a) erreurs de bloc
# ---------------------------------------------------------------------------


def bloc_errors() -> pl.DataFrame:
    """(election, type, method, horizon, bloc, error, n_polls), premier tour, avec la somme
    gauche + gauche radicale (`left_union`)."""
    pres = pl.read_parquet(paths.CURATED / "poll_errors.parquet").filter(
        (pl.col("round") == 1) & (pl.col("level") == "bloc") & pl.col("method").is_in(["mean_14d", "horizon"])
    ).with_columns(pl.col("election").cast(pl.Utf8), pl.lit("pres").alias("election_type"))
    lst = pl.read_parquet(paths.CURATED / "poll_errors_lists.parquet").filter(
        (pl.col("level") == "bloc") & pl.col("method").is_in(["mean_14d", "horizon"])
    )
    cols = ["election", "election_type", "method", "horizon_days", "unit", "poll_share",
            "result_share", "error", "n_polls"]
    df = pl.concat([pres.select(cols), lst.select(cols)]).rename({"unit": "bloc"})
    union = (
        df.filter(pl.col("bloc").is_in(["left", "far_left"]))
        .group_by("election", "election_type", "method", "horizon_days")
        .agg(pl.col("poll_share").sum(), pl.col("result_share").sum(), pl.col("n_polls").max(),
             pl.len().alias("k"))
        .filter(pl.col("k") == 2)
        .with_columns(pl.lit("left_union").alias("bloc"),
                      (pl.col("poll_share") - pl.col("result_share")).alias("error"))
        .select(cols[:4] + ["bloc"] + cols[5:])
    )
    return pl.concat([df, union]).sort("election_type", "election", "method", "horizon_days", "bloc")


def _fmt(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:+.{digits}f}".replace(".", ",")
    return s.replace("-", "−")


def bloc_error_table(be: pl.DataFrame, method: str, horizon: int | None) -> str:
    sub = be.filter(pl.col("method") == method)
    if horizon is not None:
        sub = sub.filter(pl.col("horizon_days") == horizon)
    order = [str(e) for e in PRESIDENTIALS] + list(LIST_KEYS)
    head = "| scrutin | n | " + " | ".join(BLOC_LABEL[b] for b in BLOC_ORDER) + " |"
    lines = [head, "|" + "---|" * (len(BLOC_ORDER) + 2)]
    for e in order:
        r = sub.filter(pl.col("election") == e)
        if r.is_empty():
            lines.append(f"| {e} | 0 | " + " | ".join("—" for _ in BLOC_ORDER) + " |")
            continue
        vals = dict(zip(r["bloc"], r["error"]))
        n = int(r["n_polls"].max())
        lines.append(f"| {e} | {n} | " + " | ".join(_fmt(vals.get(b)) for b in BLOC_ORDER) + " |")
    return "\n".join(lines)


def rms_table(be: pl.DataFrame) -> str:
    """Racine de l'erreur quadratique moyenne par famille de scrutins, bloc et horizon."""
    rows = []
    for method, h in [("mean_14d", 14), ("horizon", 7), ("horizon", 30), ("horizon", 90), ("horizon", 180)]:
        sub = be.filter((pl.col("method") == method) & (pl.col("horizon_days") == h))
        for fam, types in (("présidentielles", ["pres"]), ("européennes", ["euro"]),
                           ("législatives", ["leg"])):
            f = sub.filter(pl.col("election_type").is_in(types))
            vals = {}
            for b in BLOC_ORDER:
                x = f.filter(pl.col("bloc") == b)["error"].to_numpy()
                vals[b] = (float(np.sqrt(np.mean(x**2))), len(x)) if len(x) else (None, 0)
            n = f["election"].n_unique()
            label = "14 j" if method == "mean_14d" else f"{h} j"
            rows.append(f"| {label} | {fam} ({n}) | " + " | ".join(
                "—" if v is None else f"{v:.1f}".replace(".", ",") for v, _ in vals.values()) + " |")
    head = "| horizon | famille (scrutins) | " + " | ".join(BLOC_LABEL[b] for b in BLOC_ORDER) + " |"
    return "\n".join([head, "|" + "---|" * (len(BLOC_ORDER) + 2), *rows])


def list_level_errors(keys=("euro2019", "leg2024", "leg2022", "euro2024", "euro2014")) -> pl.DataFrame:
    e = pl.read_parquet(paths.CURATED / "poll_errors_lists.parquet")
    return (
        e.filter((pl.col("level") == "candidate") & (pl.col("method") == "mean_14d")
                 & pl.col("election").is_in(list(keys)) & (pl.col("result_share") >= 2))
        .select("election", "unit", "poll_share", "result_share", "error", "n_polls")
        .sort("election", "result_share", descending=[False, True])
    )


# ---------------------------------------------------------------------------
# (b) items de calibration des scrutins de liste
# ---------------------------------------------------------------------------


def fit_pair(pres_items, lst_items, exclude):
    t0 = time.time()
    a = er.ErrorModel.fit(elections=PRESIDENTIALS, exclude=exclude, items=pres_items)
    b = er.ErrorModel.fit(elections=PRESIDENTIALS, exclude=exclude, items=pres_items,
                          extra_items=lst_items)
    log.info("ajustements sans %s : %.1f s", exclude, time.time() - t0)
    return a, b


def params_rows(em: er.ErrorModel, calib: str, excluded: str) -> list[dict]:
    pr = em.params
    base = {"calibration": calib, "excluded": excluded}
    rows = [base | {"param": k, "bloc": "", "horizon": None, "value": float(getattr(pr, k))}
            for k in ("sigma_bloc_final", "sigma_cand_final", "rho", "a_bloc", "a_cand", "gamma", "nu")]
    rows += [base | {"param": "bloc_weight", "bloc": b, "horizon": None, "value": v}
             for b, v in pr.bloc_weight.items()]
    rows += [base | {"param": "bias", "bloc": b, "horizon": None, "value": v} for b, v in pr.bias.items()]
    for s in em.summary(horizons=(7, 30, 90, 180)).iter_rows(named=True):
        for k in ("sd_bloc", "sd_cand", "sd_total", "corr_within_bloc"):
            rows.append(base | {"param": k, "bloc": s["bloc"], "horizon": s["horizon"], "value": s[k]})
    return rows


def bloc_error_correlation(items: list[er.Item]) -> tuple[np.ndarray, list[str], int]:
    """Corrélation empirique, entre élections, des erreurs finales de bloc (log, centrées
    par élection sur les blocs présents) ; blocs présents partout."""
    errs = er.bloc_final_errors(items)
    elections = sorted({it.election for it in items if it.kind == "final"}, key=str)
    blocs = [b for b in BLOCS if b in errs and all(e in errs[b] for e in elections)]
    m = np.array([[errs[b][e] for b in blocs] for e in elections])
    m = m - m.mean(1, keepdims=True)
    return np.corrcoef(m, rowvar=False), blocs, len(elections)


# ---------------------------------------------------------------------------
# (c) couverture du backtest
# ---------------------------------------------------------------------------


def calibrations(pres_items, lst_items) -> dict[str, tuple[tuple[int, ...], list, dict]]:
    """Calibrations comparées : (élections, items, arguments de fit). Les variantes « sans
    biais » isolent l'effet de la largeur de η de celui du biais de bloc, que la règle de
    stabilité admet parfois sur les présidentielles seules et jamais avec les scrutins de
    liste ; « finales seules » n'utilise des scrutins de liste que l'erreur finale (pas les
    mouvements, dont la dynamique de campagne diffère)."""
    P = PRESIDENTIALS
    finals = [it for it in lst_items if it.kind == "final"]
    return {
        "présidentielles": (P, pres_items, {}),
        "étendue": (P, pres_items, {"extra_items": lst_items}),
        "étendue, finales seules": (P, pres_items, {"extra_items": finals}),
        "présidentielles, sans biais": (P, pres_items, {"use_bias": False}),
        "étendue, sans biais": (P, pres_items, {"extra_items": lst_items, "use_bias": False}),
        "étendue, finales seules, sans biais": (P, pres_items, {"extra_items": finals,
                                                                "use_bias": False}),
    }


def coverage(configs, elections, horizons, n_draws) -> pl.DataFrame:
    from vvs.eval.scoring import score_backtest

    out = []
    for calib, (cal, items, kw) in configs.items():
        t0 = time.time()
        res = score_backtest(BACKTEST_DIR, list(elections), list(horizons), variant="floor",
                             n_draws=n_draws, seed=0, calibration_elections=cal, write=False,
                             error_kwargs={"items": items, **kw})
        log.info("score %s : %.0f s, %d manquants", calib, time.time() - t0, len(res.missing))
        out.append(res.metrics.filter(pl.col("level") == "summary").with_columns(pl.lit(calib).alias("calibration")))
    return pl.concat(out, how="diagonal")


def coverage_table(cov: pl.DataFrame, elections: list[int]) -> str:
    from vvs.eval.backtest import ELECTION_WEIGHT

    metrics = ("cov50", "cov90", "bloc_cov50", "bloc_cov90", "crps", "log_score_pair")
    sub = cov.filter((pl.col("model") == "model") & pl.col("election").is_in(elections)
                     & pl.col("metric").is_in(metrics))
    sub = sub.with_columns(
        pl.col("election").replace_strict(ELECTION_WEIGHT, default=1.0, return_dtype=pl.Float64).alias("w")
    )
    agg = sub.group_by("calibration", "horizon", "metric").agg(
        ((pl.col("value") * pl.col("w")).sum() / pl.col("w").sum()).alias("v")
    )
    lines = ["| horizon | calibration | couv. 50 % | couv. 90 % | couv. blocs 50 % | couv. blocs 90 % | CRPS | log-score paire |",
             "|---|---|---|---|---|---|---|---|"]
    for h in sorted(agg["horizon"].unique(), reverse=True):
        for calib in cov["calibration"].unique(maintain_order=True):
            v = dict(agg.filter((pl.col("horizon") == h) & (pl.col("calibration") == calib))
                     .select("metric", "v").iter_rows())
            if not v:
                continue
            pct = {m: f"{100 * v[m]:.0f} %" for m in ("cov50", "cov90", "bloc_cov50", "bloc_cov90")}
            lines.append(
                f"| {h} j | {calib} | {pct['cov50']} | {pct['cov90']} | {pct['bloc_cov50']} | "
                f"{pct['bloc_cov90']} | {v['crps']:.2f} | {v['log_score_pair']:.2f} |".replace(".", ",")
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# rapport
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.RUNS / f"{dt.date.today().isoformat()}-calibration-extension")
    ap.add_argument("--weight", type=float, default=0.5)
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--n-draws", type=int, default=10_000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    be = bloc_errors()
    be.write_parquet(out / "bloc_errors.parquet")

    reg = default_registry()
    pres_items = er.calibration_items(PRESIDENTIALS, reg)
    lst_items = el.list_items(args.weight, kinds=LIST_KINDS)
    n_units = {it.election: len(it.units) for it in lst_items if it.kind == "final"}
    n_moves = Counter(it.election for it in lst_items if it.kind == "move")

    rows = []
    fits = {}
    for excl in [None, *PRESIDENTIALS]:
        a, b = fit_pair(pres_items, lst_items, [] if excl is None else [excl])
        tag = "aucune" if excl is None else str(excl)
        fits[tag] = (a, b)
        rows += params_rows(a, "présidentielles", tag) + params_rows(b, "étendue", tag)
    # sensibilité au poids, ajustement complet
    for w in (0.25, 1.0):
        b = er.ErrorModel.fit(elections=PRESIDENTIALS, items=pres_items,
                              extra_items=el.list_items(w, kinds=LIST_KINDS))
        rows += params_rows(b, f"étendue w={w:g}", "aucune")
    configs = calibrations(pres_items, lst_items)
    for calib, (cal, items, kw) in configs.items():
        if calib not in ("présidentielles", "étendue"):
            rows += params_rows(er.ErrorModel.fit(elections=cal, items=items, **kw), calib, "aucune")
    params = pl.DataFrame(rows)
    params.write_parquet(out / "eta_params.parquet")

    cov = None
    if not args.skip_backtest:
        cov = coverage(configs, PRESIDENTIALS, (180, 90, 30, 7), args.n_draws)
        cov.write_parquet(out / "coverage.parquet")

    write_report(out, be, params, fits, pres_items, lst_items, n_units, n_moves, cov, args.weight)


def _param_table(params: pl.DataFrame) -> str:
    keys = ("sigma_bloc_final", "sigma_cand_final", "rho", "a_bloc", "a_cand", "gamma", "nu")
    lines = ["| laissée de côté | calibration | " + " | ".join(keys) + " | biais admis |",
             "|---|---|" + "---|" * len(keys) + "---|"]
    for excl in ["aucune", *map(str, PRESIDENTIALS)]:
        for calib in ("présidentielles", "étendue", "étendue w=0.25", "étendue w=1",
                      "étendue, finales seules", "présidentielles, sans biais",
                      "étendue, sans biais", "étendue, finales seules, sans biais"):
            sub = params.filter((pl.col("excluded") == excl) & (pl.col("calibration") == calib))
            if sub.is_empty():
                continue
            v = {(p, b): x for p, b, x in sub.select("param", "bloc", "value").iter_rows()}
            bias = ", ".join(f"{b} {x:+.3f}" for (p, b), x in v.items() if p == "bias") or "—"
            cells = [f"{v[(k, '')]:.3g}" for k in keys]
            lines.append(f"| {excl} | {calib} | " + " | ".join(cells) + f" | {bias} |")
    return "\n".join(lines)


def _sd_table(params: pl.DataFrame, excl: str = "aucune") -> str:
    """Écart type total de η (logit) par bloc et horizon, corrélation intra-bloc."""
    lines = ["| horizon | calibration | " + " | ".join(BLOC_LABEL[b] for b in BLOCS) + " | corr. intra-bloc (centre) |",
             "|---|---|" + "---|" * (len(BLOCS) + 1)]
    for h in (7, 30, 90, 180):
        for calib in ("présidentielles", "étendue", "étendue, finales seules"):
            sub = params.filter((pl.col("excluded") == excl) & (pl.col("calibration") == calib)
                                & (pl.col("horizon") == h))
            sd = dict(sub.filter(pl.col("param") == "sd_total").select("bloc", "value").iter_rows())
            corr = dict(sub.filter(pl.col("param") == "corr_within_bloc").select("bloc", "value").iter_rows())
            lines.append(f"| {h} j | {calib} | " + " | ".join(f"{sd[b]:.3f}" for b in BLOCS)
                         + f" | {corr['centre']:.2f} |")
    return "\n".join(lines).replace(".", ",")


def _weight_table(params: pl.DataFrame) -> str:
    lines = ["| laissée de côté | calibration | " + " | ".join(BLOC_LABEL[b] for b in BLOCS) + " |",
             "|---|---|" + "---|" * len(BLOCS)]
    for excl in ["aucune", *map(str, PRESIDENTIALS)]:
        for calib in ("présidentielles", "étendue"):
            sub = params.filter((pl.col("excluded") == excl) & (pl.col("calibration") == calib)
                                & (pl.col("param") == "bloc_weight"))
            w = dict(sub.select("bloc", "value").iter_rows())
            lines.append(f"| {excl} | {calib} | " + " | ".join(f"{w[b]:.2f}" for b in BLOCS) + " |")
    return "\n".join(lines).replace(".", ",")


def _corr_md(c: np.ndarray, blocs: list[str]) -> str:
    lines = ["| | " + " | ".join(BLOC_LABEL[b] for b in blocs) + " |", "|---|" + "---|" * len(blocs)]
    for i, b in enumerate(blocs):
        lines.append(f"| {BLOC_LABEL[b]} | " + " | ".join(f"{c[i, j]:+.2f}" for j in range(len(blocs))) + " |")
    return "\n".join(lines).replace(".", ",").replace("-", "−")


def write_report(out, be, params, fits, pres_items, lst_items, n_units, n_moves, cov, weight):
    polls = pl.read_parquet(paths.CURATED / "polls_lists.parquet")
    counts = polls.group_by("election").agg(
        pl.len().alias("n"), pl.col("pollster").n_unique().alias("instituts"),
        pl.col("fieldwork_end").min().alias("debut"), pl.col("fieldwork_end").max().alias("fin"),
    ).sort("election")
    ll = list_level_errors()
    c_pres, b_pres, n_pres = bloc_error_correlation(pres_items)
    c_all, b_all, n_all = bloc_error_correlation(pres_items + lst_items)

    md = [f"# Extension de la calibration de η aux scrutins de liste ({dt.date.today().isoformat()})", ""]
    md += [("Script : `back/studies/calibration_extension.py` (poids des scrutins de liste "
            f"w = {weight:g}). Données : `poll_errors.parquet`, `poll_errors_lists.parquet`, "
            "`results_lists_by_bloc.parquet` (`vvs.curate.lists`). Erreurs en points d'exprimés, "
            "sondages moins résultat (positif = surestimé)."), ""]
    md += ["## Sondages obtenus", "", "| scrutin | sondages | instituts | premier | dernier | unités η | mouvements η |",
           "|---|---|---|---|---|---|---|"]
    for r in counts.iter_rows(named=True):
        md.append(f"| {r['election']} | {r['n']} | {r['instituts']} | {r['debut']} | {r['fin']} | "
                  f"{n_units.get(r['election'], 0)} | {n_moves.get(r['election'], 0)} |")
    md += [""]
    md += ["## (a) Erreurs de bloc", ""]
    md += ["Moyenne des sondages des 14 derniers jours :", "", bloc_error_table(be, "mean_14d", 14), ""]
    for h in (7, 30, 90, 180):
        md += [f"Moyenne à {h} jours (sondages terminés dans les 30 jours avant l'horizon) :", "",
               bloc_error_table(be, "horizon", h), ""]
    md += ["Racine de l'erreur quadratique moyenne par famille :", "", rms_table(be), ""]
    md += ["Erreurs par liste (14 derniers jours, listes à 2 % ou plus) :", "",
           "| scrutin | liste | sondages | résultat | erreur | n |", "|---|---|---|---|---|---|"]
    for r in ll.iter_rows(named=True):
        md.append(f"| {r['election']} | {r['unit']} | {r['poll_share']:.1f} | {r['result_share']:.1f} | "
                  f"{_fmt(r['error'])} | {r['n_polls']} |".replace(".", ","))
    md += [""]
    md += ["## (b) Paramètres de η", "",
           ("Paramètres (échelle logit ; σ de bloc et de candidat de l'erreur finale, a : variance "
            "du mouvement à 90 jours, ρ : terme ρ/p, γ : forme en τ estimée sur Jennings–Wlezien) :"), "",
           _param_table(params), "",
           "Écart type marginal de η par bloc et horizon (λ inclus), ajustement sans exclusion :", "",
           _sd_table(params), "",
           "Poids relatifs de bloc w_b :", "", _weight_table(params), "",
           (f"Corrélation empirique des erreurs finales de bloc (log, centrées par élection), "
            f"présidentielles seules ({n_pres} élections) :"), "", _corr_md(c_pres, b_pres), "",
           f"Avec les scrutins de liste ({n_all} élections, poids égaux) :", "", _corr_md(c_all, b_all), ""]
    if cov is not None:
        md += ["## (c) Couverture du backtest", "",
               ("Ajustements `floor` de runs/2026-10-20-backtest, mêmes tirages θ, graine 0 ; seule "
                "la calibration de η change. 2012-2022 (poids 1) :"), "",
               coverage_table(cov, [2012, 2017, 2022]), "",
               "2002-2022 (2002 et 2007 pondérées 0,5) :", "", coverage_table(cov, list(PRESIDENTIALS)), ""]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("écrit %s", out / "report.md")


if __name__ == "__main__":
    main()
