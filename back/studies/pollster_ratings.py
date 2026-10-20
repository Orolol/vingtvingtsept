"""Étude : palmarès des instituts de sondage (à la manière des notes de FiveThirtyEight).

    uv run python studies/pollster_ratings.py [--out runs/<date>-pollster-ratings]
        [--backtest runs/<date>-backtest-ratings --reference runs/2026-10-20-backtest]

Questions, dans l'ordre :

1. Palmarès 2002-2022 par institut (lignée, successions comprises) : erreur absolue moyenne par
   candidat et par bloc du dernier sondage avant le vote (`poll_errors`, `last_by_pollster`,
   terrain fini dans les 21 derniers jours, configuration officielle exacte), biais signé par
   bloc, nombre de sondages, et l'écart à l'« institut moyen » (moyenne des instituts de
   l'élection). Précision relative x_{p,e} sur tous les sondages des 21 derniers jours et
   effets maison de campagne α_{p,b,e} : définitions dans `vvs.model.pollster_ratings`.
2. Stabilité : un institut bon (ou biaisé d'un côté) à une élection l'est-il à la suivante ?
   Corrélations entre élections des x et des α, composantes de variance (part persistante),
   et prédiction hors échantillon : x et α d'une élection prédits par les autres (ou par les
   seules élections passées), comparés à la prédiction « institut moyen » (0).
3. Successions : Sofres → Kantar → Verian mis en commun, BVA gardé à part d'Ipsos ; on
   montre ce que change la mise en commun.
4. Avec `--backtest` : comparaison de la variante `ratings` à la référence `floor` sur les
   mêmes ajustements, et règle d'admission de docs/07-evaluation.md.

Rien n'entre dans le modèle ici : l'option `Priors.pollster_ratings` reste désactivée par
défaut quoi qu'il arrive dans ce script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from vvs import paths
from vvs.model import pollster_ratings as pr
from vvs.model.plots import DIVERGING, GRID, INK, MUTED, RESULT, SERIES, _style
from vvs.registry import default_registry

ELECTIONS = pr.ELECTIONS
MAIN_BLOCS = ("far_right", "right", "centre", "left", "far_left")
BLOC_LABEL = {"far_right": "ext. droite", "right": "droite", "centre": "centre",
              "left": "gauche", "far_left": "gauche rad.", "other": "autres"}
ELECTION_COLORS = {2002: "#8a8984", 2007: "#3a9b5c", 2012: "#eb6834", 2017: "#2a78d6",
                   2022: "#9b4dca"}
EVAL_ELECTIONS = (2012, 2017, 2022)
BACKTEST_HORIZONS = (180, 90, 30, 7)


# --- données


def load_tables() -> dict[int, tuple[pl.DataFrame, pl.DataFrame]]:
    return {e: pr.election_tables(e) for e in ELECTIONS}


def last_poll_errors(reg) -> pl.DataFrame:
    """`poll_errors` : dernier sondage de chaque institut, premier tour, terrain fini dans les
    21 derniers jours et configuration exacte (les « derniers » sondages de 2013 ou 2015 que
    la table garde pour quelques instituts de 2017 sont écartés)."""
    pe = pl.read_parquet(paths.CURATED / "poll_errors.parquet").filter(
        (pl.col("method") == "last_by_pollster") & (pl.col("round") == 1)
        & pl.col("exact_config")
    )
    vote = {e: reg.elections[e].round_(1).date for e in ELECTIONS}
    pe = pe.filter(
        pl.struct("election", "fieldwork_end").map_elements(
            lambda s: (vote[s["election"]] - s["fieldwork_end"]).days <= pr.FINAL_WINDOW_DAYS,
            return_dtype=pl.Boolean,
        )
    )
    return pe.with_columns(
        pl.col("pollster").map_elements(pr.lineage, return_dtype=pl.Utf8).alias("lineage")
    )


def campaign_poll_counts(reg) -> pl.DataFrame:
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    vote = {e: reg.elections[e].round_(1).date for e in ELECTIONS}
    rows = []
    for e in ELECTIONS:
        sub = polls.filter((pl.col("election") == e) & (pl.col("fieldwork_end") < vote[e]))
        for p, n in sub.group_by("pollster").len().iter_rows():
            rows.append({"election": e, "pollster": p, "lineage": pr.lineage(p), "n_polls": n})
    return pl.DataFrame(rows)


# --- palmarès


def palmares(pe: pl.DataFrame, acc: pl.DataFrame, counts: pl.DataFrame,
             ratings: pr.PollsterRatings) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(une ligne par lignée, une ligne par lignée × élection)."""
    cand = pe.filter(pl.col("level") == "candidate")
    bloc = pe.filter(pl.col("level") == "bloc")
    per_e = (
        cand.group_by("election", "lineage")
        .agg(pl.col("error").abs().mean().alias("mae_cand"))
        .join(bloc.filter(pl.col("unit").is_in(MAIN_BLOCS)).group_by("election", "lineage")
              .agg(pl.col("error").abs().mean().alias("mae_bloc")), on=["election", "lineage"])
    )
    per_e = per_e.with_columns(
        (pl.col("mae_cand") - pl.col("mae_cand").mean().over("election")).alias("pm_cand"),
        (pl.col("mae_bloc") - pl.col("mae_bloc").mean().over("election")).alias("pm_bloc"),
    )
    acc = acc.with_columns(
        pl.col("pollster").map_elements(pr.lineage, return_dtype=pl.Utf8).alias("lineage")
    )
    per_e = per_e.join(acc.select("election", "lineage", "x", "rms_z",
                                  pl.col("n_polls").alias("n_final")),
                       on=["election", "lineage"], how="full", coalesce=True)
    per_e = per_e.join(counts.group_by("election", "lineage").agg(pl.col("n_polls").sum()),
                       on=["election", "lineage"], how="left")
    bias = (
        bloc.filter(pl.col("unit").is_in(MAIN_BLOCS))
        .group_by("lineage", "unit").agg(pl.col("error").mean().alias("bias"))
        .pivot(on="unit", index="lineage", values="bias")
    )
    summ = per_e.group_by("lineage").agg(
        pl.col("election").sort().cast(pl.Utf8).str.join(",").alias("elections"),
        pl.col("n_polls").sum().alias("n_polls"),
        pl.col("n_final").sum().alias("n_final"),
        pl.col("mae_cand").mean(), pl.col("mae_bloc").mean(),
        pl.col("pm_cand").mean(), pl.col("x").mean().alias("x_raw"),
        (pl.col("x") < 0).sum().alias("n_better"), pl.col("x").count().alias("n_x"),
    )
    summ = summ.join(bias, on="lineage", how="left").join(
        ratings.precision.select("lineage", "x_mean", "factor"), on="lineage", how="left"
    )
    return summ.sort("x_mean", nulls_last=True), per_e.sort("election", "lineage")


def average_pollster(pe: pl.DataFrame) -> dict:
    """Institut moyen : moyenne sur les instituts de l'élection, puis sur les élections."""
    cand = pe.filter(pl.col("level") == "candidate")
    bloc = pe.filter((pl.col("level") == "bloc") & pl.col("unit").is_in(MAIN_BLOCS))
    out = {
        "mae_cand": cand.group_by("election", "lineage").agg(pl.col("error").abs().mean())
        .group_by("election").agg(pl.col("error").mean())["error"].mean(),
        "mae_bloc": bloc.group_by("election", "lineage").agg(pl.col("error").abs().mean())
        .group_by("election").agg(pl.col("error").mean())["error"].mean(),
    }
    b = bloc.group_by("election", "unit").agg(pl.col("error").mean()).group_by("unit").agg(
        pl.col("error").mean())
    out |= dict(zip(b["unit"].to_list(), b["error"].to_list()))
    return out


# --- stabilité


def pairwise(values: pl.DataFrame, key: list[str], col: str) -> pl.DataFrame:
    """Corrélations de `col` entre paires d'élections, sur les clés communes."""
    rows = []
    for a, b in itertools.combinations(ELECTIONS, 2):
        va = values.filter(pl.col("election") == a).select(*key, pl.col(col).alias("a"))
        vb = values.filter(pl.col("election") == b).select(*key, pl.col(col).alias("b"))
        j = va.join(vb, on=key).drop_nulls()
        if j.height < 3:
            rows.append({"e1": a, "e2": b, "n": j.height, "pearson": None, "spearman": None})
            continue
        rows.append({"e1": a, "e2": b, "n": j.height,
                     "pearson": float(stats.pearsonr(j["a"], j["b"])[0]),
                     "spearman": float(stats.spearmanr(j["a"], j["b"])[0])})
    return pl.DataFrame(rows)


def consecutive_pooled(values: pl.DataFrame, key: list[str], col: str) -> dict:
    """Corrélation poolée des paires d'élections consécutives (a, a+5)."""
    a_all, b_all = [], []
    for a, b in itertools.pairwise(ELECTIONS):
        va = values.filter(pl.col("election") == a).select(*key, pl.col(col).alias("a"))
        vb = values.filter(pl.col("election") == b).select(*key, pl.col(col).alias("b"))
        j = va.join(vb, on=key).drop_nulls()
        a_all += j["a"].to_list()
        b_all += j["b"].to_list()
    r, p = stats.pearsonr(a_all, b_all)
    return {"n": len(a_all), "pearson": float(r), "p": float(p),
            "spearman": float(stats.spearmanr(a_all, b_all)[0])}


def oos_skill(tables, mode: str) -> tuple[pl.DataFrame, dict]:
    """Prédiction hors échantillon de x et des α d'une élection, par les notes calibrées sans
    elle (`loo`) ou sur les seules élections antérieures (`past`). Skill = 1 − SSE/SSE(0) sur
    les instituts ayant un historique (0 : institut moyen)."""
    rows = []
    for e in ELECTIONS:
        excl = (e,) if mode == "loo" else tuple(f for f in ELECTIONS if f >= e)
        if mode == "past" and e == ELECTIONS[0]:
            continue
        r = pr.PollsterRatings.fit(ELECTIONS, exclude=excl, tables=tables)
        acc = pr.relative_accuracy(tables[e][0])
        prec = dict(zip(r.precision["lineage"].to_list(), r.precision["x_mean"].to_list())) \
            if not r.precision.is_empty() else {}
        raw = dict(zip(r.precision["lineage"].to_list(), r.precision["raw_mean"].to_list())) \
            if not r.precision.is_empty() else {}
        for p, x in zip(acc["pollster"].to_list(), acc["x"].to_list()):
            ln = pr.lineage(p)
            if ln in prec:
                rows.append({"election": e, "kind": "precision", "unit": ln, "obs": x,
                             "pred": prec[ln], "pred_raw": raw[ln]})
        house = {(a, b): (m, rm) for a, b, m, rm in
                 r.house.select("lineage", "bloc", "mu", "raw_mean").iter_rows()} \
            if not r.house.is_empty() else {}
        for p, b, a in tables[e][1].select("pollster", "bloc", "alpha").iter_rows():
            k = (pr.lineage(p), b)
            if k in house and b in MAIN_BLOCS:
                rows.append({"election": e, "kind": f"house_{b}", "unit": k[0], "obs": a,
                             "pred": house[k][0], "pred_raw": house[k][1]})
    df = pl.DataFrame(rows)

    def skill(d):
        sse0 = (d["obs"] ** 2).sum()
        return {
            "n": d.height,
            "skill_shrunk": float(1 - ((d["obs"] - d["pred"]) ** 2).sum() / sse0),
            "skill_raw": float(1 - ((d["obs"] - d["pred_raw"]) ** 2).sum() / sse0),
        }

    summ = {}
    for kind in ["precision", "house"]:
        d = df.filter(pl.col("kind").str.starts_with(kind))
        if d.height:
            summ[kind] = skill(d) | {
                "by_election": {e: skill(d.filter(pl.col("election") == e))
                                for e in sorted(d["election"].unique().to_list())}}
    return df, summ


# --- figures


def fig_palmares(per_e: pl.DataFrame, summ: pl.DataFrame, path: Path) -> None:
    s = summ.filter(pl.col("n_x") >= 1).sort("x_mean", descending=True, nulls_last=True)
    names = s["lineage"].to_list()
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(names) + 1.2))
    for i, ln in enumerate(names):
        d = per_e.filter((pl.col("lineage") == ln) & pl.col("x").is_not_null())
        for e, x in d.select("election", "x").iter_rows():
            ax.scatter(x, i, s=22, color=ELECTION_COLORS[e], zorder=3,
                       label=str(e) if i == 0 else None)
        xm = s.filter(pl.col("lineage") == ln)["x_mean"].item()
        if xm is not None:
            ax.scatter(xm, i, marker="D", s=30, color=INK, zorder=4)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=str(e))
               for e, c in ELECTION_COLORS.items()]
    handles.append(plt.Line2D([], [], marker="D", linestyle="", color=INK,
                              label="note rétrécie"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("log de l'écart quadratique relatif à l'institut moyen (← plus précis)",
                  fontsize=8, color=MUTED)
    ax.set_title("Précision des sondages des 21 derniers jours, par élection", loc="left",
                 fontsize=10, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_scatter_consecutive(values: pl.DataFrame, key: list[str], col: str, path: Path,
                            title: str, color_by: str | None = None) -> None:
    pairs = list(itertools.pairwise(ELECTIONS))
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.0 * len(pairs), 3.1))
    bloc_colors = dict(zip(MAIN_BLOCS, ["#3b3a8f", "#2a78d6", "#eb6834", "#e34948",
                                        "#9b1c1c"]))
    for ax, (a, b) in zip(axes, pairs):
        va = values.filter(pl.col("election") == a).select(*key, pl.col(col).alias("a"))
        vb = values.filter(pl.col("election") == b).select(*key, pl.col(col).alias("b"))
        j = va.join(vb, on=key).drop_nulls()
        if color_by:
            for bl, c in bloc_colors.items():
                d = j.filter(pl.col(color_by) == bl)
                ax.scatter(d["a"], d["b"], s=14, color=c, label=BLOC_LABEL[bl])
        else:
            ax.scatter(j["a"], j["b"], s=18, color=SERIES)
            for row in j.iter_rows(named=True):
                ax.annotate(row[key[0]], (row["a"], row["b"]), fontsize=6, color=MUTED,
                            xytext=(2, 2), textcoords="offset points")
        if j.height >= 3:
            r = stats.pearsonr(j["a"], j["b"])[0]
            ax.set_title(f"{a} → {b}  (r = {r:.2f}, n = {j.height})", fontsize=8, loc="left",
                         color=INK)
        lim = max(0.05, float(np.nanmax(np.abs(np.r_[j["a"], j["b"]])))) * 1.1 if j.height \
            else 1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.6, linestyle=":")
        _style(ax)
    if color_by:
        axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_house(house: pl.DataFrame, path: Path) -> None:
    lins = sorted(house.filter(pl.col("election").n_unique().over("lineage") >= 2)["lineage"]
                  .unique().to_list())
    fig, axes = plt.subplots(1, len(ELECTIONS), figsize=(2.3 * len(ELECTIONS), 0.3 * len(lins)
                                                         + 1.4), sharey=True)
    for ax, e in zip(axes, ELECTIONS):
        m = np.full((len(lins), len(MAIN_BLOCS)), np.nan)
        for ln, b, a in house.filter(pl.col("election") == e).select(
                "lineage", "bloc", "alpha").iter_rows():
            if ln in lins and b in MAIN_BLOCS:
                m[lins.index(ln), MAIN_BLOCS.index(b)] = a
        ax.imshow(m, cmap=DIVERGING, vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(MAIN_BLOCS)), [BLOC_LABEL[b] for b in MAIN_BLOCS], rotation=60,
                      fontsize=7)
        ax.set_yticks(range(len(lins)), lins, fontsize=7)
        ax.set_title(str(e), fontsize=9, color=INK, loc="left")
        ax.tick_params(colors=MUTED)
    fig.suptitle("Effets maison de campagne α (logit, écart à l'institut moyen ; rouge = "
                 "surestime le bloc)", x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_bias(pe: pl.DataFrame, path: Path) -> None:
    bloc = pe.filter((pl.col("level") == "bloc") & pl.col("unit").is_in(MAIN_BLOCS))
    fig, axes = plt.subplots(1, len(MAIN_BLOCS), figsize=(2.4 * len(MAIN_BLOCS), 3.0),
                             sharey=True)
    for ax, b in zip(axes, MAIN_BLOCS):
        d = bloc.filter(pl.col("unit") == b)
        for i, e in enumerate(ELECTIONS):
            de = d.filter(pl.col("election") == e)
            if de.height:
                ax.scatter(np.full(de.height, i) + np.linspace(-0.15, 0.15, de.height),
                           de["error"], s=10, color=ELECTION_COLORS[e], alpha=0.8)
                ax.hlines(de["error"].mean(), i - 0.3, i + 0.3, color=INK, linewidth=1.4)
        ax.axhline(0, color=RESULT, linewidth=0.8)
        ax.set_xticks(range(len(ELECTIONS)), [str(e)[2:] for e in ELECTIONS], fontsize=7)
        ax.set_title(BLOC_LABEL[b], fontsize=9, loc="left", color=INK)
        _style(ax)
    axes[0].set_ylabel("dernier sondage − résultat (points)", fontsize=8, color=MUTED)
    fig.suptitle("Biais signé par bloc du dernier sondage de chaque institut (trait : institut "
                 "moyen)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- backtest


def _link_reference(bt: Path, ref: Path) -> None:
    """Lien symbolique vers les ajustements `floor` de la référence (mêmes tirages)."""
    dst = bt / "fits" / "floor"
    dst.mkdir(parents=True, exist_ok=True)
    for e in EVAL_ELECTIONS:
        for h in BACKTEST_HORIZONS:
            name = f"{e}-{h:03d}"
            link = dst / name
            if not link.exists():
                os.symlink(os.path.relpath(ref / "fits" / "floor" / name, dst), link)


def backtest_comparison(bt: Path, ref: Path) -> tuple[str, dict]:
    from vvs.eval.scoring import score_backtest

    _link_reference(bt, ref)
    res = score_backtest(bt, list(EVAL_ELECTIONS), list(BACKTEST_HORIZONS),
                         variants=["floor", "ratings"])
    s = res.metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model"))
    val = {(r["variant"], r["election"], r["horizon"], r["metric"]): r["value"]
           for r in s.iter_rows(named=True)}
    md, verdict = [], {}
    better_high = {"log_score_pair": True, "crps": False, "bloc_crps": False, "mae": False}
    for metric in ("log_score_pair", "crps", "bloc_crps", "cov50", "cov90", "bloc_cov90"):
        md += [f"### {metric}", "", "| élection | horizon | floor | ratings | Δ ratings − floor |",
               "|---|---|---|---|---|"]
        for e in EVAL_ELECTIONS:
            for h in BACKTEST_HORIZONS:
                a, b = val.get(("floor", e, h, metric)), val.get(("ratings", e, h, metric))
                d = "–" if a is None or b is None else f"{b - a:+.3f}"
                md.append(f"| {e} | {h} | {a if a is None else f'{a:.3f}'} | "
                          f"{b if b is None else f'{b:.3f}'} | {d} |")
        for h in BACKTEST_HORIZONS:
            a = [val.get(("floor", e, h, metric)) for e in EVAL_ELECTIONS]
            b = [val.get(("ratings", e, h, metric)) for e in EVAL_ELECTIONS]
            if None in a or None in b:
                continue
            ma, mb = np.mean(a), np.mean(b)
            md.append(f"| moy | {h} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} |")
            gains = [(y - x) if better_high.get(metric, True) else (x - y) for x, y in zip(a, b)]
            verdict[(metric, h)] = {"floor": ma, "ratings": mb, "delta": mb - ma,
                                    "n_gain": sum(g > 0 for g in gains),
                                    "n_loss": sum(g < 0 for g in gains)}
        md.append("")
    # couverture : distance au nominal, moyenne des 12 cas
    cov = {}
    for v in ("floor", "ratings"):
        cov[v] = {
            m: float(np.mean([abs(val[(v, e, h, m)] - nom) for e in EVAL_ELECTIONS
                              for h in BACKTEST_HORIZONS if (v, e, h, m) in val]))
            for m, nom in (("cov50", 0.5), ("cov90", 0.9), ("bloc_cov90", 0.9))
        }
    verdict["coverage_gap"] = cov
    return "\n".join(md), verdict


def admission(verdict: dict) -> list[str]:
    lines = []
    ok1 = all(verdict[(m, h)]["delta"] * (1 if m == "log_score_pair" else -1) > 0
              for m in ("log_score_pair", "crps") for h in (180, 90))
    ok2 = all(verdict[(m, h)]["n_gain"] >= 2 for m in ("log_score_pair", "crps")
              for h in (180, 90))
    cg = verdict["coverage_gap"]
    ok3 = all(cg["ratings"][m] <= cg["floor"][m] + 1e-9 for m in cg["floor"])
    for m in ("log_score_pair", "crps"):
        for h in (180, 90):
            v = verdict[(m, h)]
            lines.append(f"- {m} à {h} j : floor {v['floor']:.3f}, ratings {v['ratings']:.3f} "
                         f"(Δ {v['delta']:+.3f}) ; gain sur {v['n_gain']} élection(s) sur 3, "
                         f"perte sur {v['n_loss']}")
    lines.append(
        "- couverture (écart moyen au nominal, 12 cas) : "
        + ", ".join(f"{m} {cg['floor'][m]:.3f} → {cg['ratings'][m]:.3f}" for m in cg["floor"])
    )
    lines.append(f"- critère 1 (log-score et CRPS améliorés à 180 et 90 j) : "
                 f"{'oui' if ok1 else 'non'}")
    lines.append(f"- critère 2 (même signe sur au moins 2 élections sur 3) : "
                 f"{'oui' if ok2 else 'non'}")
    lines.append(f"- critère 3 (couverture non dégradée) : {'oui' if ok3 else 'non'}")
    lines.append(f"- **admission : {'oui' if ok1 and ok2 and ok3 else 'non'}**")
    return lines


# --- rapport

# Lecture rédigée après le passage du 2026-10-20 (les chiffres cités sont ceux des tableaux).
LECTURE = """## Lecture (2026-10-20)

- **Le palmarès ne tient pas d'une élection à l'autre.** Sur les paires d'élections
  consécutives, la précision relative se corrèle à 0,11 (p = 0,6), l'erreur absolue relative
  du dernier sondage à 0,21 (p = 0,4), le biais relatif par bloc à −0,08. Prédire la précision
  d'une élection par les autres fait pire que l'institut moyen (skill −0,17 en leave-one-out,
  −0,02 avec les seules élections passées) ; aucun institut ne bat l'institut moyen à toutes
  ses élections sauf Elabe (2 sur 2). Les écarts entre instituts sont petits devant l'erreur
  commune (biais par bloc : ce que tous les instituts ratent ensemble, déjà dans η).
- **Effets maison** : pas de composante persistante à l'extrême droite, à droite ni au centre
  (ICC ≈ 0) ; une composante nette pour la gauche radicale (ICC 0,73 : OpinionWay et Ifop la
  sous-estiment de façon répétée) et faible pour la gauche (0,28). Le prior appris est donc
  presque partout centré sur 0 avec un écart-type de 0,04 à 0,06 logit, ce que le modèle
  estimait déjà seul ; la prédiction hors échantillon des α reste sans gain (skill −0,03).
- **Successions** : la lignée Sofres → Kantar → Verian a la quatrième note sur 14 (facteur
  1,05 avec toutes les élections) mais ne bat l'institut moyen qu'à 2 élections sur 4 ; ses
  effets maison ne se suivent pas de Sofres 2012 à Kantar 2017. Le profil d'effets maison de
  BVA ne ressemble pas plus à celui d'Ipsos qu'à un autre (corrélation moyenne sur les cinq
  élections 0,03, contre 0,04 en moyenne pour les paires d'instituts), ce qui conforte le choix
  de ne pas transmettre l'historique de BVA à Ipsos.
- **Facteur de précision** : la part persistante estimée tombe à 0 dès que 2012 est dans
  l'apprentissage (facteurs tous à 1 pour évaluer 2017 et 2022) ; sans 2012, les facteurs vont
  de 0,76 (OpinionWay) à 1,21 (BVA) et ne prédisent pas 2012 (skill −0,32).
- **Backtest** : écarts de l'ordre du bruit Monte-Carlo d'un réajustement (|Δ| ≤ 0,06 en
  log-score, ≤ 0,03 point en CRPS), signes mêlés. Règle d'admission : non. La variable reste
  dans le volet explicatif ; `Priors.pollster_ratings` reste désactivé.
- Note technique : une première version du facteur (n_eff multiplié par q, plafonné sous n)
  créait une zone plate de la vraisemblance et 2012 à 7 jours n'a pas convergé (r_hat 2,96) ;
  les quatre ajustements 2012 ont été refaits avec la forme lisse actuelle (ancienne version
  conservée dans `runs/2026-10-20-backtest-ratings/superseded_clamped_q/`), avant tout score.
"""


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def main(out: Path, backtest: Path | None, reference: Path) -> None:
    reg = default_registry()
    tables = load_tables()
    pe = last_poll_errors(reg)
    counts = campaign_poll_counts(reg)
    acc = pr.relative_accuracy(pl.concat([t[0] for t in tables.values()]))
    house = pl.concat([t[1] for t in tables.values()]).with_columns(
        pl.col("pollster").map_elements(pr.lineage, return_dtype=pl.Utf8).alias("lineage")
    )
    full = pr.PollsterRatings.fit(ELECTIONS, exclude=(), tables=tables)
    summ, per_e = palmares(pe, acc, counts, full)
    avg = average_pollster(pe)
    summ.write_parquet(out / "palmares.parquet")
    per_e.write_parquet(out / "palmares_by_election.parquet")
    house.write_parquet(out / "house_effects.parquet")
    (out / "ratings_all_elections.json").write_text(
        json.dumps(full.to_dict(), indent=2, ensure_ascii=False, default=str))

    # stabilité
    acc_l = acc.with_columns(
        pl.col("pollster").map_elements(pr.lineage, return_dtype=pl.Utf8).alias("lineage"))
    pw_x = pairwise(acc_l, ["lineage"], "x")
    pw_pm = pairwise(per_e, ["lineage"], "pm_cand")
    house_main = house.filter(pl.col("bloc").is_in(MAIN_BLOCS))
    pw_h = pairwise(house_main, ["lineage", "bloc"], "alpha")
    bias_rel = pe.filter((pl.col("level") == "bloc") & pl.col("unit").is_in(MAIN_BLOCS)) \
        .with_columns((pl.col("error") - pl.col("error").mean().over("election", "unit"))
                      .alias("rel_bias"))
    pw_b = pairwise(bias_rel, ["lineage", "unit"], "rel_bias")
    cons = {
        "précision x": consecutive_pooled(acc_l, ["lineage"], "x"),
        "erreur absolue relative (dernier sondage)": consecutive_pooled(per_e, ["lineage"],
                                                                        "pm_cand"),
        "effets maison α (5 blocs)": consecutive_pooled(house_main, ["lineage", "bloc"],
                                                         "alpha"),
        "biais relatif par bloc (dernier sondage)": consecutive_pooled(
            bias_rel, ["lineage", "unit"], "rel_bias"),
    }
    oos_loo, sk_loo = oos_skill(tables, "loo")
    oos_past, sk_past = oos_skill(tables, "past")
    oos_loo.write_parquet(out / "oos_loo.parquet")
    oos_past.write_parquet(out / "oos_past.parquet")
    # composantes de variance par bloc
    var_rows = []
    for b in MAIN_BLOCS + ("other",):
        g = house.filter(pl.col("bloc") == b)
        groups = [(x["alpha"].to_numpy(), x["se"].to_numpy() ** 2)
                  for _, x in sorted(g.group_by("lineage"), key=lambda kg: kg[0])]
        sm2, se2 = pr.fit_random_effects(groups)
        var_rows.append({"bloc": b, "s_m": sm2**0.5, "s_e": se2**0.5,
                         "icc": sm2 / (sm2 + se2), "se_median": float(g["se"].median())})
    loo_ratings = {e: pr.PollsterRatings.fit(ELECTIONS, exclude=(e,), tables=tables)
                   for e in EVAL_ELECTIONS}

    # successions : Sofres et Kantar séparés contre mis en commun
    succ = house_main.filter(pl.col("pollster").is_in(["sofres", "kantar", "bva", "ipsos"]))
    succ_tab = succ.pivot(on="bloc", index=["election", "pollster"], values="alpha").sort(
        "pollster", "election")

    fig_palmares(per_e, summ, out / "palmares.png")
    fig_scatter_consecutive(acc_l, ["lineage"], "x", out / "stability_accuracy.png",
                            "Précision relative x d'une élection à la suivante")
    fig_scatter_consecutive(house_main, ["lineage", "bloc"], "alpha",
                            out / "stability_house.png",
                            "Effets maison de campagne α d'une élection à la suivante",
                            color_by="bloc")
    fig_house(house, out / "house_effects.png")
    fig_bias(pe, out / "bias_by_bloc.png")

    md = [
        "# Palmarès des instituts de sondage, 2002-2022",
        "",
        (f"Généré le {dt.date.today().isoformat()} par `back/studies/pollster_ratings.py`. "
        "Définitions dans `back/src/vvs/model/pollster_ratings.py`. Lignées : Sofres → Kantar → "
        "Verian mis en commun sous `verian` ; BVA gardé à part (rachat par Ipsos en 2024)."),
        "",
        "## 1. Palmarès",
        "",
        ("Dernier sondage de chaque institut (`poll_errors`, terrain fini dans les 21 derniers "
        "jours, configuration exacte) : erreur absolue moyenne par candidat et par bloc "
        "(points), « ± moyen » = écart à l'institut moyen de l'élection (négatif = meilleur), "
        "biais signé moyen par bloc (points, sondage − résultat). Précision x : tous les "
        "sondages des 21 derniers jours, log de l'écart quadratique des erreurs standardisées "
        "autour de l'institut moyen, centré par élection (négatif = plus précis) ; « x brut » "
        "= moyenne sur les élections, « note » = moyenne rétrécie (effets aléatoires), "
        "« facteur » = exp(−note) borné à [0,5 ; 1,5], celui qu'appliquerait le modèle avec "
        "toutes les élections. « meilleur » = élections où l'institut bat l'institut moyen."),
        "",
        "| institut | élections | sondages | sondages 21 j | EAM cand. | EAM bloc | ± moyen "
        "| " + " | ".join(f"biais {BLOC_LABEL[b]}" for b in MAIN_BLOCS)
        + " | x brut | note | facteur | meilleur |",
        "|---|---|---|---|---|---|---|" + "---|" * len(MAIN_BLOCS) + "---|---|---|---|",
    ]
    for r in summ.iter_rows(named=True):
        md.append(
            f"| {r['lineage']} | {r['elections']} | {r['n_polls'] or 0} | {r['n_final'] or 0} "
            f"| {fmt(r['mae_cand'])} | {fmt(r['mae_bloc'])} | {fmt(r['pm_cand'], 2)} | "
            + " | ".join(fmt(r.get(b), 1) for b in MAIN_BLOCS)
            + f" | {fmt(r['x_raw'])} | {fmt(r['x_mean'])} | {fmt(r['factor'])} "
            f"| {r['n_better']}/{r['n_x']} |"
        )
    md.append(
        f"| **institut moyen** | | | | {fmt(avg['mae_cand'])} | {fmt(avg['mae_bloc'])} | 0 | "
        + " | ".join(fmt(avg.get(b), 1) for b in MAIN_BLOCS) + " | 0 | 0 | 1 | |"
    )
    md += [
        "",
        "![palmarès](palmares.png)",
        "",
        "![biais par bloc](bias_by_bloc.png)",
        "",
        "## 2. Stabilité d'une élection à l'autre",
        "",
        "### Paires d'élections consécutives, poolées",
        "",
        "| quantité | paires | Pearson | p | Spearman |",
        "|---|---|---|---|---|",
    ]
    for k, v in cons.items():
        md.append(f"| {k} | {v['n']} | {v['pearson']:.2f} | {v['p']:.2f} | "
                  f"{v['spearman']:.2f} |")
    md += ["", "### Toutes les paires (Pearson, n)", "",
           "| paire | précision x | erreur abs. relative | α (5 blocs) | biais relatif bloc |",
           "|---|---|---|---|---|"]
    for i in range(pw_x.height):
        cells = []
        for t in (pw_x, pw_pm, pw_h, pw_b):
            r = t.row(i, named=True)
            cells.append("–" if r["pearson"] is None else f"{r['pearson']:+.2f} (n={r['n']})")
        r0 = pw_x.row(i, named=True)
        md.append(f"| {r0['e1']} → {r0['e2']} | " + " | ".join(cells) + " |")
    md += ["", "![stabilité précision](stability_accuracy.png)", "",
           "![stabilité effets maison](stability_house.png)", "",
           "### Composantes de variance des effets maison (logit)", "",
           ("s_m : effet persistant de l'institut ; s_e : variation d'une élection à l'autre "
           "(nette du bruit d'estimation) ; ICC = s_m² / (s_m² + s_e²)."), "",
           "| bloc | s_m | s_e | ICC | erreur type médiane |", "|---|---|---|---|---|"]
    for r in var_rows:
        md.append(f"| {BLOC_LABEL[r['bloc']]} | {r['s_m']:.3f} | {r['s_e']:.3f} | "
                  f"{r['icc']:.2f} | {r['se_median']:.3f} |")
    md += ["", f"Précision : s_m = {full.prec_sm:.3f}, s_e = {full.prec_se:.3f} (toutes "
           "élections) ; sans 2012 / 2017 / 2022 : "
           + " / ".join(f"s_m = {loo_ratings[e].prec_sm:.3f}" for e in EVAL_ELECTIONS) + ".",
           "", "### Prédiction hors échantillon (skill = 1 − SSE / SSE de l'institut moyen)", "",
           "| quantité | calibration | n | skill note rétrécie | skill moyenne brute |",
           "|---|---|---|---|---|"]
    for name, sk in (("leave-one-out", sk_loo), ("élections passées seules", sk_past)):
        for kind in ("precision", "house"):
            if kind in sk:
                v = sk[kind]
                md.append(f"| {kind} | {name} | {v['n']} | {v['skill_shrunk']:+.3f} | "
                          f"{v['skill_raw']:+.3f} |")
    md += ["", "Par élection évaluée (leave-one-out, note rétrécie) :", "",
           "| élection | précision : n, skill | effets maison : n, skill |", "|---|---|---|"]
    for e in ELECTIONS:
        p = sk_loo.get("precision", {}).get("by_election", {}).get(e)
        h = sk_loo.get("house", {}).get("by_election", {}).get(e)
        md.append(f"| {e} | {p['n'] if p else 0}, {fmt(p['skill_shrunk']) if p else '–'} | "
                  f"{h['n'] if h else 0}, {fmt(h['skill_shrunk']) if h else '–'} |")
    md += ["", "![effets maison](house_effects.png)", "",
           "## 3. Successions", "",
           "Effets maison de campagne α (logit) des instituts concernés :", "",
           "| élection | institut | " + " | ".join(BLOC_LABEL[b] for b in MAIN_BLOCS) + " |",
           "|---|---|" + "---|" * len(MAIN_BLOCS)]
    for r in succ_tab.iter_rows(named=True):
        md.append(f"| {r['election']} | {r['pollster']} | "
                  + " | ".join(fmt(r.get(b), 3) for b in MAIN_BLOCS) + " |")
    md += ["", "## 4. Notes appliquées au backtest (leave-one-election-out)", "",
           ("| élection laissée | s_m précision | facteurs ≠ 1 | max |μ| effets maison | "
           "sd du prior (médiane) |"), "|---|---|---|---|---|"]
    for e, r in loo_ratings.items():
        f = r.precision.filter((pl.col("factor") - 1).abs() > 0.005) if not \
            r.precision.is_empty() else r.precision
        fl = ", ".join(f"{a} {b:.2f}" for a, b in f.select("lineage", "factor").iter_rows())
        md.append(f"| {e} | {r.prec_sm:.3f} | {fl or 'aucun'} | "
                  f"{r.house['mu'].abs().max():.3f} | {r.house['sd'].median():.3f} |")
    if backtest is not None:
        table, verdict = backtest_comparison(backtest, reference)
        (backtest / "comparison.md").write_text(table)
        md += ["", "## 5. Backtest : variante `ratings` contre `floor`", "",
               (f"Ajustements `ratings` dans `{backtest.relative_to(paths.ROOT)}`, référence "
               f"`floor` de `{reference.relative_to(paths.ROOT)}` (liens symboliques, mêmes "
               "tirages), même modèle d'erreur η, 10 000 tirages."), "", table, "",
               "### Règle d'admission (docs/07-evaluation.md)", "", *admission(verdict)]
    md += ["", LECTURE]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-pollster-ratings")
    ap.add_argument("--backtest", type=Path, default=None)
    ap.add_argument("--reference", type=Path, default=paths.RUNS / "2026-10-20-backtest")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    main(a.out, a.backtest.resolve() if a.backtest else None, a.reference.resolve())
