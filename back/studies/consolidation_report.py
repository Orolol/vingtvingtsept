"""Exporte le bilan compact d'un gel terminé ; figures et tables restent dans runs/.

Usage depuis back/ : uv run python studies/consolidation_report.py --run ../runs/2026-10-22-consolidation
Copier ensuite review/ dans docs/results/ pour conserver le bilan dans Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from vvs.eval.consolidation import complete_score, digest, verify, write_json


def validate_draws(file: Path, expected: int) -> None:
    with np.load(file) as draws:
        shares, qualified, winner = draws["shares"], draws["qualified"], draws["winner"]
        valid = (
            len(shares) == expected
            and np.isfinite(shares).all()
            and (shares >= 0).all()
            and np.allclose(shares.sum(axis=1), 100)
            and (qualified[:, 0] != qualified[:, 1]).all()
            and ((winner == qualified[:, 0]) | (winner == qualified[:, 1])).all()
            and np.isfinite(draws["share_second"]).all()
            and ((draws["share_second"] >= 0) & (draws["share_second"] <= 100)).all()
        )
        if not valid:
            raise ValueError(f"tirages incohérents : {file}")


def export(run: Path) -> None:
    manifest = verify(run)
    spec = manifest["spec"]
    cases, metrics, outcomes = [], [], []
    for election in spec["elections"]:
        for horizon in spec["horizons"]:
            folder = run / "cases" / f"{election}-{horizon:03d}"
            if not complete_score(folder):
                raise ValueError(f"cas incomplet : {folder}")
            cells = pl.read_parquet(folder / "cells.parquet")
            m = pl.read_parquet(folder / "metrics.parquet")
            o = pl.read_parquet(folder / "outcomes.parquet")
            metrics.append(m)
            outcomes.append(o)
            diagnostics = []
            for method in ("reference", "mean14d"):
                validate_draws(folder / f"draws_{method}.npz", spec["n_draws"])
                sub = cells.filter(pl.col("method") == method)
                low = sub.filter(pl.col("ess_ratio") < 0.1)
                diagnostics.append(
                    {
                        "method": method,
                        "n_cells": sub.height,
                        "min_ess_ratio": sub["ess_ratio"].min(),
                        # Des cellules minuscules peuvent avoir une ESS faible sans peser sur le mélange.
                        "draw_mass_ess_ratio_below_0_1": low["n"].sum() / spec["n_draws"],
                        "draw_mass_without_duel_polls": sub.filter(pl.col("n_duel_polls") == 0)[
                            "n"
                        ].sum()
                        / spec["n_draws"],
                    }
                )
            cases.append(
                {
                    "case": json.loads((folder / "case.json").read_text()),
                    "calibration": json.loads((folder / "calibration.json").read_text()),
                    "quality": json.loads((folder / "fit/quality.json").read_text()),
                    "fit_manifest_git_hint": json.loads(
                        (folder / "fit/manifest.json").read_text()
                    ).get("commit"),
                    "statistical_source_commit": manifest["commit"],
                    "second_round_diagnostics": diagnostics,
                    "output_hashes": json.loads((folder / "scored.json").read_text()),
                }
            )
    target = run / "review"
    target.mkdir(exist_ok=True)
    write_json(
        target / "audit.json",
        {
            "reference": manifest,
            "reference_sha256": digest(run / "reference.json"),
            "postprocessing_sources": {
                "back/studies/consolidation_report.py": digest(Path(__file__)),
                "back/src/vvs/eval/consolidation.py": digest(
                    Path(__file__).resolve().parents[1] / "src/vvs/eval/consolidation.py"
                ),
            },
            "cases": cases,
            "sensitivity_outputs": {
                name: digest(run / "sensitivity" / name)
                for name in ("probabilities.parquet", "variants.json")
            },
        },
    )
    frame = pl.concat(metrics, how="vertical_relaxed").sort("election", "horizon", "method")
    frame.write_csv(target / "metrics.csv", float_precision=8)
    pl.concat(outcomes).sort("election", "horizon", "method", "candidate_id").write_csv(
        target / "outcomes.csv", float_precision=8
    )
    sensitivity = pl.read_parquet(run / "sensitivity/probabilities.parquet")
    sensitivity.select(pl.exclude("label")).write_csv(target / "sensitivity.csv", float_precision=8)
    (target / "variants.json").write_text((run / "sensitivity/variants.json").read_text())
    score_plot(frame, target / "backtest.png")
    sensitivity_plot(sensitivity, target / "sensitivity.png")
    print(f"Bilan compact vérifié : {target}")


def score_plot(frame: pl.DataFrame, file: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, year in zip(axes, sorted(frame["election"].unique())):
        for method, label, color in (
            ("reference", "Référence", "#2563a6"),
            ("mean14d", "Moyenne 14 j", "#b66b32"),
        ):
            data = frame.filter((pl.col("election") == year) & (pl.col("method") == method)).sort(
                "horizon", descending=True
            )
            ax.plot(data["horizon"], data["log_score_winner"], "o-", label=label, color=color)
            invalid = (
                data.filter(~pl.col("fit_acceptable")) if method == "reference" else data.head(0)
            )
            ax.scatter(
                invalid["horizon"],
                invalid["log_score_winner"],
                s=90,
                marker="x",
                color="#b02c36",
                label="Diagnostic insuffisant" if invalid.height else None,
            )
        ax.set_title(str(year))
        ax.set_xticks([180, 90, 30, 7])
        ax.invert_xaxis()
        ax.set_xlabel("Jours avant le premier tour")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Log-score du vainqueur (plus haut = mieux)")
    axes[-1].legend(fontsize=9)
    fig.suptitle("Chaîne complète, calibration chronologique — trois élections")
    fig.savefig(file, dpi=180)
    plt.close(fig)


def sensitivity_plot(frame: pl.DataFrame, file: Path) -> None:
    local = frame.filter(
        pl.col("variant").str.ends_with("minus") | pl.col("variant").str.ends_with("plus")
    )
    base = frame.filter(pl.col("variant") == "reference")
    labels = [
        ("le_pen_marine", "Marine Le Pen"),
        ("bardella_jordan", "Jordan Bardella"),
        ("philippe_edouard", "Édouard Philippe"),
        ("glucksmann_raphael", "Raphaël Glucksmann"),
        ("melenchon_jean_luc", "Jean-Luc Mélenchon"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i, (candidate, _) in enumerate(labels):
        sub = local.filter(pl.col("candidate_id") == candidate)
        center = base.filter(pl.col("candidate_id") == candidate)["p_win"].item() * 100
        lo, hi = sub["p_win_lower"].min() * 100, sub["p_win_upper"].max() * 100
        ax.plot([lo, hi], [i, i], lw=6, color="#88b7dc", solid_capstyle="round")
        ax.scatter(center, i, color="#16466f", zorder=3)
        ax.text(hi + 1, i, f"{lo:.1f}–{hi:.1f} %", va="center", fontsize=9)
    ax.set_yticks(np.arange(len(labels)), [name for _, name in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 90)
    ax.set_xlabel("Probabilité de victoire (%)")
    ax.set_title("Sensibilité aux poids de candidature : ±10 points, un état à la fois")
    ax.grid(axis="x", alpha=0.2)
    fig.supxlabel(
        "Point : référence. Barre : extrêmes, bornes de masse manquante incluses.\nCe ne sont pas des intervalles de confiance.",
        fontsize=9,
    )
    fig.savefig(file, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    export(parser.parse_args().run.resolve())
