"""Bilan de docs/11 depuis les cas vérifiés, sans recalcul ni écriture dans la référence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from historical_scenarios import digest, inventory, validate_simulation, verify_inventory

METRICS = [
    "log_score_winner",
    "log_score_pair",
    "brier_win",
    "brier_qualify",
    "crps",
    "coverage50",
    "coverage90",
    "unforecast_share",
]


def export(run, reference):
    study = json.loads((run / "study.json").read_text())
    verify_inventory(reference, study["reference_files"])
    verify_inventory(run / "source", study["inputs"])
    pin = json.loads((run / "source/reference.lock.json").read_text())
    spec = pin["spec"]
    variants = ["frozen", *study["protocol"]["variants"]]
    metrics, outcomes, presence_rows, case_audits, diagnostics = [], [], [], [], []
    for election in spec["elections"]:
        for horizon in spec["horizons"]:
            name = f"{election}-{horizon:03d}"
            folder, original = run / "cases" / name, reference / "cases" / name
            hashes = json.loads((folder / "complete.json").read_text())
            actual = inventory(folder)
            actual.pop("complete.json")
            if hashes != actual:
                raise ValueError(f"cas d'étude modifié : {name}")
            for variant in variants:
                dest = original if variant == "frozen" else folder / variant
                metrics.append(
                    pl.read_parquet(dest / "metrics.parquet").with_columns(
                        pl.lit(variant).alias("variant")
                    )
                )
                outcomes.append(
                    pl.read_parquet(dest / "outcomes.parquet").with_columns(
                        pl.lit(variant).alias("variant")
                    )
                )
                case = json.loads((dest / "case.json").read_text())
                key = {"election": election, "horizon": horizon, "variant": variant}
                probabilities = {}
                for cfg in case["configurations"]:
                    for c in cfg["candidates"]:
                        probabilities[c] = probabilities.get(c, 0.0) + cfg["probability"]
                presence_rows.extend(
                    key | {"candidate_id": c, "p_presence": p}
                    for c, p in sorted(probabilities.items())
                )
                cells = pl.read_parquet(dest / "cells.parquet")
                for method in ("reference", "mean14d"):
                    with np.load(dest / f"draws_{method}.npz") as sim:
                        validate_simulation(sim, spec["n_draws"])
                        counts = np.bincount(sim["config"], minlength=len(case["configurations"]))
                        undrawn = sum(
                            cfg["probability"]
                            for cfg, n in zip(case["configurations"], counts, strict=True)
                            if n == 0
                        )
                    sub = cells.filter(pl.col("method") == method)
                    diagnostics.append(
                        key
                        | {
                            "method": method,
                            "n_configurations": len(counts),
                            "min_draws_per_config": int(counts.min()),
                            "undrawn_probability": float(undrawn),
                            "min_ess_ratio": sub["ess_ratio"].min(),
                            "draw_mass_ess_ratio_below_0_1": sub.filter(pl.col("ess_ratio") < 0.1)[
                                "n"
                            ].sum()
                            / spec["n_draws"],
                        }
                    )
                case_audits.append(key | {"case": case})
    metrics = pl.concat(metrics, how="diagonal_relaxed").sort(
        "election", "horizon", "method", "variant"
    )
    if metrics.height != 72:
        raise ValueError("72 lignes de scores attendues")
    base = metrics.filter(pl.col("variant") == "frozen").select(
        "election", "horizon", "method", *[pl.col(m).alias(f"{m}_frozen") for m in METRICS]
    )
    comparisons = (
        metrics.filter(pl.col("variant") != "frozen")
        .join(base, on=["election", "horizon", "method"], validate="m:1")
        .with_columns(*[(pl.col(m) - pl.col(f"{m}_frozen")).alias(f"delta_{m}") for m in METRICS])
    )
    means = (
        metrics.group_by("variant", "method", "horizon")
        .agg(*[pl.col(m).mean() for m in METRICS])
        .sort("method", "horizon", "variant")
    )
    delta_means = (
        comparisons.group_by("variant", "method", "horizon")
        .agg(*[pl.col(f"delta_{m}").mean() for m in METRICS])
        .sort("method", "horizon", "variant")
    )
    candidate_outcomes = (
        pl.concat(outcomes, how="diagonal_relaxed")
        .join(
            pl.DataFrame(presence_rows),
            on=["election", "horizon", "variant", "candidate_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(pl.col("p_presence").fill_null(0.0))
    )
    review = run / "review"
    review.mkdir(exist_ok=True)
    tables = {
        "metrics": metrics,
        "comparisons": comparisons,
        "means": means,
        "delta_means": delta_means,
        "outcomes": candidate_outcomes,
        "diagnostics": pl.DataFrame(diagnostics),
    }
    for name, table in tables.items():
        table.write_csv(review / f"{name}.csv", float_precision=10)
    audit = {
        "study": study,
        "reference_unchanged": True,
        "cases": case_audits,
        "case_outputs": {
            p.parent.name: json.loads(p.read_text())
            for p in sorted((run / "cases").glob("*/complete.json"))
        },
        "replays": {
            p.parent.name: json.loads(p.read_text())
            for p in sorted((run / "cases").glob("*/replay.json"))
        },
        "report_sha256": digest(Path(__file__)),
    }
    (review / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    shutil.copy2(run / "source/protocol.yaml", review / "protocol.yaml")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    colors = {"frozen": "#777777", "memory": "#d28b20", "memory_events": "#127e8c"}
    labels = {
        "frozen": "Référence figée",
        "memory": "Mémoire seule",
        "memory_events": "Mémoire + événements",
    }
    for j, horizon in enumerate((180, 90)):
        for i, (metric, label) in enumerate(
            (("log_score_pair", "Log-score de la paire (↑)"), ("crps", "CRPS du premier tour (↓)"))
        ):
            ax = axes[i, j]
            for k, variant in enumerate(variants):
                sub = metrics.filter(
                    (pl.col("horizon") == horizon)
                    & (pl.col("method") == "reference")
                    & (pl.col("variant") == variant)
                ).sort("election")
                ax.bar(
                    np.arange(3) + (k - 1) * 0.25,
                    sub[metric],
                    width=0.24,
                    color=colors[variant],
                    label=labels[variant],
                )
            ax.set_xticks(np.arange(3), [str(e) for e in spec["elections"]])
            ax.set_title(f"À {horizon} jours — {label}")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Reconstruction historique : même chaîne, scénarios différents\nAgrégateur bayésien ; trois élections, sans promotion du modèle",
        fontsize=12,
    )
    fig.savefig(review / "comparison.png", dpi=170)
    plt.close(fig)
    protocol = yaml.safe_load((review / "protocol.yaml").read_text())
    (review / "README.md").write_text(
        "# Reconstruction des scénarios historiques — 22 octobre 2026\n\n"
        "Protocole et interprétation : [11-scenarios-historiques.md](../../11-scenarios-historiques.md).\n\n"
        "Comparaison exploratoire, référence statistique `a89d304` intacte. "
        f"Variante principale : `{protocol['primary_variant']}`.\n\n"
        "- `protocol.yaml` : paramètres déclarés avant calcul.\n"
        "- `metrics.csv` : 72 scores (12 cas × 2 agrégateurs × 3 versions de scénarios).\n"
        "- `comparisons.csv` : écarts à la référence, par cas ; `means.csv` et `delta_means.csv` : moyennes à poids égal par élection.\n"
        "- `outcomes.csv` : probabilités de présence, qualification et victoire ; zéro de présence pour les candidats hors support.\n"
        "- `diagnostics.csv` : configurations sans tirage et qualité du second tour.\n"
        "- `audit.json` : empreintes, scénarios, événements appliqués, masse exclue, contrôles des rejeux.\n"
        "- `comparison.png` : log-score de paire et CRPS, par élection et horizon long.\n\n"
        "Les écarts sont variante moins référence : positif préférable pour les log-scores, "
        "négatif pour le CRPS et les Brier. Les quatre dates d'une même élection ne sont pas indépendantes. "
        "Les tirages complets restent dans `runs/2026-10-22-historical-scenarios/`.\n\n"
        "Reproduction depuis la racine : `scripts/run_historical_scenarios.sh`, puis depuis `back/` :\n\n"
        "```sh\nuv run python studies/historical_scenarios_report.py --run ../runs/2026-10-22-historical-scenarios --reference ../runs/2026-10-22-consolidation\n```\n"
    )
    print(
        delta_means.filter(pl.col("method") == "reference").select(
            "variant",
            "horizon",
            "delta_log_score_winner",
            "delta_log_score_pair",
            "delta_crps",
            "delta_coverage90",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    export(args.run.resolve(), args.reference.resolve())
