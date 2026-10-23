"""Bilan compact de docs/13, à partir des seules étapes terminées et vérifiées."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import polars as pl


def verify(folder):
    expected = json.loads((folder / "complete.json").read_text())
    actual = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "complete.json"}
    if actual != expected:
        raise ValueError(f"étape modifiée : {folder}")


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
    for row in frame.select(columns).iter_rows():
        values = [f"{x:+.4f}" if isinstance(x, float) else str(x) for x in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "study.json").read_text())
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (run / "source").iterdir() if p.is_file()}
    if source_hashes != manifest["inputs"]:
        raise ValueError("sources modifiées")
    metrics, outcomes, rolling, bootstrap = [], [], [], []
    for folder in sorted((run / "cases").iterdir()):
        verify(folder)
        metrics.append(pl.read_csv(folder / "metrics.csv"))
        outcomes.append(pl.read_csv(folder / "outcomes.csv"))
        election, horizon = map(int, folder.name.split("-"))
        for variant, rows in json.loads((folder / "rolling.json").read_text()).items():
            rolling.extend({"election": election, "horizon": horizon, "variant": variant, **r}
                           for r in rows)
        calibration = json.loads((folder / "bootstrap.json").read_text())
        for i, params in enumerate(calibration["parameters"]):
            bootstrap.append({"election": election, "horizon": horizon, "replicate": i,
                              **{k: v for k, v in params.items() if isinstance(v, (float, int))}})
    if len(metrics) != 12:
        raise ValueError("douze cas requis")
    frame = pl.concat(metrics, how="vertical_relaxed")
    frame.write_csv(out / "metrics.csv")
    pl.concat(outcomes).write_csv(out / "outcomes.csv")
    pl.DataFrame(rolling).write_csv(out / "rolling_diagnostics.csv")
    pl.DataFrame(bootstrap).write_csv(out / "calibration_parameters.csv")
    cols = ["log_score_winner", "log_score_pair", "crps", "coverage90", "brier_win"]
    comparisons = []
    for base in ("reference", "mean14d", "gls_thinned", "rolling_independent"):
        baseline = frame.filter(pl.col("variant") == base).select(
            "election", "horizon", *[pl.col(c).alias(c+"_base") for c in cols])
        comparisons.append(frame.join(baseline, on=["election", "horizon"]).with_columns(
            pl.lit(base).alias("baseline"),
            *[(pl.col(c)-pl.col(c+"_base")).alias("delta_"+c) for c in cols]).select(
                "election", "horizon", "variant", "baseline", *["delta_"+c for c in cols]))
    differences = pl.concat(comparisons)
    differences.write_csv(out / "comparisons.csv")
    means = differences.group_by("variant", "baseline", "horizon").agg(
        *[pl.col("delta_"+c).mean() for c in cols]).sort("baseline", "variant", "horizon")
    means.write_csv(out / "means.csv")
    for stage in ("panel", "substitution"):
        verify(run / stage)
    shutil.copy2(run / "panel/audit.json", out / "panel_audit.json")
    for name in ("means.csv", "fits.json", "inventory.json"):
        shutil.copy2(run / "substitution" / name, out / ("substitution_"+name))
    shutil.copy2(run / "source/protocol.yaml", out / "protocol.yaml")
    audit = json.loads((run / "review/audit.json").read_text())
    audit["source_manifest"] = manifest
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n")
    selected = ["ensemble50", "calibration_bootstrap", "rolling_overlap"]
    small = means.filter((pl.col("baseline") == "reference") & pl.col("variant").is_in(selected))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    labels = {"ensemble50": "Mélange 50/50", "calibration_bootstrap": "Calibration rééchantillonnée",
              "rolling_overlap": "Moyenne avec recouvrements"}
    for ax, metric, title in zip(axes, ("log_score_winner", "log_score_pair", "crps"),
                                ("Log-score victoire · plus haut préférable",
                                 "Log-score paire · plus haut préférable", "CRPS · plus bas préférable")):
        for variant in selected:
            rows = small.filter(pl.col("variant") == variant).sort("horizon")
            ax.plot(rows["horizon"], rows["delta_"+metric], "o-", label=labels[variant])
        ax.axhline(0, color="black", lw=.8)
        ax.set(title=title, xlabel="Jours avant le premier tour", ylabel="Écart à la référence")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    text = ["# Expériences de structure — 23 octobre 2026", "",
            "Moyennes de trois élections, poids égaux. Résultats exploratoires ; aucune admission.", "",
            "![Comparaison](comparison.png)", "",
            table(small, ["variant", "horizon", "delta_log_score_winner", "delta_log_score_pair",
                          "delta_crps", "delta_coverage90"]), "",
            "Les écarts positifs favorisent les log-scores ; les écarts négatifs favorisent le CRPS.",
            "La couverture est une fraction, pas un nombre de points.", "",
            "Les témoins mean14d, GLS éclairci et GLS indépendant figurent dans comparisons.csv.",
            "La covariance des rollings est une hypothèse fondée sur les terrains, pas une mesure",
            "du chevauchement des répondants. La calibration rééchantillonnée est conditionnelle",
            "aux biais, à gamma et à nu fixés. Les scénarios incomplets du gel restent identiques.", "",
            "## Diagnostics", "",
            f"- {audit['cases']} cas ; {audit['rows']} lignes de scores ; référence inchangée",
            f"  ({audit['reference_files']} fichiers contrôlés).",
            f"- {len(bootstrap)} calibrations pondérées, optimisations convergées.",
            "- Substitution : substitution_means.csv et substitution_fits.json, tests séparés",
            "  leave-one-election-out et chronologiques ; pas de projection électorale si le pilote échoue.",
            "- Indécis : panel_audit.json ; aucun identifiant individuel ni microdonnée dans ce bilan.", "",
            "## Reproduction", "",
            "Depuis la racine, avec le gel disponible :", "",
            "```sh", "bash scripts/run_structure_study.sh", "cd back",
            ("uv run --no-sync python studies/structure_report.py --run ../runs/2026-10-23-structure-v2 "
             "--out ../docs/results/structure-2026-10-23"), "```", "",
            "Les trajectoires complètes restent dans runs/, hors Git. Le script de reprise vérifie",
            "les empreintes des sources et des sorties avant de sauter une étape terminée."]
    (out / "README.md").write_text("\n".join(text)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.out)
