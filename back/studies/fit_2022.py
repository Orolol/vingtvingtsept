"""Étude : agrégateur de premier tour sur 2022 à trois dates, comparé au résultat officiel.

    uv run python studies/fit_2022.py [--out runs/<date>-fit-2022-study] [--warmup 1000 …]

Ajuste le modèle (docs/04-modele.md §1) au 10 octobre 2021, au 10 janvier 2022 et au
8 avril 2022 (veille de la fin de campagne officielle), puis compare, par candidat,
l'estimation au jour du vote (marche aléatoire prolongée jusqu'au vote, sans modèle
d'erreur) au résultat du Conseil constitutionnel : erreur de la médiane et couverture des
intervalles à 50 et 90 %. Écrit `comparison.parquet`, `comparison.md` et une figure.

Le jeu de candidats final est connu ici a posteriori : c'est la configuration officielle,
pas celle qu'on aurait pu anticiper à la date (les scénarios arrivent en phase 4).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from vvs import paths
from vvs.model.plots import INK, MUTED, RESULT, SERIES, _style, display_name
from vvs.model.run import run_fit

ASOFS = [dt.date(2021, 10, 10), dt.date(2022, 1, 10), dt.date(2022, 4, 8)]


def comparison_table(out: Path, asofs: list[dt.date]) -> pl.DataFrame:
    frames = []
    for a in asofs:
        f = pl.read_parquet(out / a.isoformat() / "final.parquet")
        frames.append(f.with_columns(pl.lit(a).alias("asof")))
    return pl.concat(frames, how="diagonal_relaxed")


def to_markdown(cmp: pl.DataFrame, manifests: dict[str, dict]) -> str:
    lines = ["# Agrégateur 2022 : estimation au jour du vote contre résultat officiel", ""]
    lines.append(
        "| asof | sondages | hypothèses | durée NUTS (s) | r_hat max (parts) | ESS min (parts) | divergences |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for a, m in manifests.items():
        der = m["diagnostics"].get("derived", {})
        rh = max((v["r_hat_max"] for v in der.values()), default=float("nan"))
        ess = min((v["ess_min"] for v in der.values()), default=float("nan"))
        lines.append(
            f"| {a} | {m['data']['n_polls']} | {m['data']['n_hypotheses']} | "
            f"{m['duration_s']['inference']} | {rh:.3f} | {ess:.0f} | "
            f"{m['diagnostics'].get('divergences', '–')} |"
        )
    lines += ["", "Parts en % des exprimés. Intervalle : 5–95 %. Erreur = médiane − résultat.", ""]
    asofs = sorted(cmp["asof"].unique().to_list())
    head = (
        "| candidat | résultat | "
        + " | ".join(f"{a} médiane [90 %] | erreur" for a in asofs)
        + " |"
    )
    lines += [head, "|---|---|" + "---|---|" * len(asofs)]
    order = cmp.filter(pl.col("asof") == asofs[-1]).sort("official", descending=True)[
        "candidate_id"
    ]
    for cid in order:
        sub = cmp.filter(pl.col("candidate_id") == cid)
        off = sub["official"][0]
        cells = []
        for a in asofs:
            r = sub.filter(pl.col("asof") == a).row(0, named=True)
            mark = "" if r["in_90"] else " ✗"
            cells.append(
                f"{r['vote_median']:.1f} [{r['vote_q05']:.1f}–{r['vote_q95']:.1f}]{mark} | "
                f"{r['error']:+.1f}"
            )
        lines.append(f"| {display_name(cid)} | {off:.2f} | " + " | ".join(cells) + " |")
    lines += ["", "✗ : résultat hors de l'intervalle à 90 %.", ""]
    lines.append(
        "| asof | erreur absolue moyenne | erreur max | couverture 50 % | couverture 90 % |"
    )
    lines.append("|---|---|---|---|---|")
    for a in asofs:
        s = cmp.filter(pl.col("asof") == a)
        lines.append(
            f"| {a} | {s['error'].abs().mean():.2f} | {s['error'].abs().max():.2f} | "
            f"{s['in_50'].mean():.0%} | {s['in_90'].mean():.0%} |"
        )
    return "\n".join(lines) + "\n"


def plot_comparison(cmp: pl.DataFrame, path: Path) -> None:
    asofs = sorted(cmp["asof"].unique().to_list())
    order = cmp.filter(pl.col("asof") == asofs[-1]).sort("official")["candidate_id"].to_list()
    fig, axes = plt.subplots(
        1, len(asofs), figsize=(4.2 * len(asofs), 0.38 * len(order) + 1.4), sharey=True, sharex=True
    )
    for ax, a in zip(np.atleast_1d(axes), asofs):
        s = cmp.filter(pl.col("asof") == a)
        r = {row["candidate_id"]: row for row in s.iter_rows(named=True)}
        y = np.arange(len(order))

        def get(k, r=r):
            return [r[c][k] for c in order]

        ax.hlines(y, get("vote_q05"), get("vote_q95"), color=SERIES, alpha=0.35, linewidth=6)
        ax.hlines(y, get("vote_q25"), get("vote_q75"), color=SERIES, linewidth=6)
        ax.scatter(get("vote_median"), y, color="white", edgecolor=SERIES, s=26, zorder=4)
        ax.scatter(get("official"), y, color=RESULT, marker="D", s=26, zorder=5)
        ax.set_yticks(y, [display_name(c) for c in order], fontsize=8)
        ax.set_title(f"au {a}", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("% exprimés", fontsize=9, color=MUTED)
        _style(ax)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    fig.suptitle(
        "2022 : estimation au jour du vote (50 et 90 %) et résultat officiel (losange)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--weeks", type=int, default=78)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--report-only", action="store_true", help="Relit les ajustements existants")
    args = ap.parse_args()
    out = args.out or paths.RUNS / f"{dt.date.today().isoformat()}-fit-2022-study"
    out.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for a in ASOFS:
        d = out / a.isoformat()
        if not args.report_only:
            m = run_fit(
                2022,
                a,
                d,
                weeks=args.weeks,
                step_days=args.step_days,
                chains=args.chains,
                warmup=args.warmup,
                samples=args.samples,
            )
            print(
                f"{a} : {m['data']['n_polls']} sondages, {m['duration_s']['inference']} s, "
                f"r_hat max {m['diagnostics']['r_hat_max']:.3f}"
            )
        manifests[a.isoformat()] = json.loads((d / "manifest.json").read_text())

    cmp = comparison_table(out, ASOFS)
    cmp.write_parquet(out / "comparison.parquet")
    md = to_markdown(cmp, manifests)
    (out / "comparison.md").write_text(md)
    plot_comparison(cmp, out / "comparison.png")
    print(md)
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
