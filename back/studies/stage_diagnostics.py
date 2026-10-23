"""Diagnostics par étape de la chaîne sur les tirages du gel (docs/12 § 3), sans simulation.

Usage depuis back/ :
    uv run python studies/stage_diagnostics.py --reference ../runs/2026-10-22-consolidation \
        --study ../runs/2026-10-22-historical-scenarios --out ../runs/2026-10-23-stage-diagnostics

Lit seulement le gel et, si fourni, une étude de variantes au même format ; écrit dans
--out des CSV par cas et par étape, la synthèse par élection × horizon, les moyennes à
poids égal par élection, les écarts des variantes et les figures. Définitions et règles
de score : docstring de vvs.eval.stage_diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from vvs.eval import stage_diagnostics as sd
from vvs.eval.consolidation import complete_score, digest, verify, write_json
from vvs.registry import default_registry

# Colonnes reproduites à l'identique depuis metrics.parquet du gel.
FROZEN = {
    "p_winner": "p_winner",
    "p_pair": "p_pair",
    "log_score_winner": "log_score_winner",
    "log_score_pair": "log_score_pair",
    "crps_global": "crps",
    "coverage90_global": "coverage90",
}
SYNTHESIS = [
    # Présence
    "p_presence_winner",
    "log_presence_winner",
    "p_presence_pair",
    "log_presence_pair",
    "brier_presence_sum",
    "brier_presence_mean",
    "n_universe",
    "unforecast_share",
    # Qualification | présence
    "p_qualify_given_presence_winner",
    "log_qualify_given_presence_winner",
    "p_pair_given_presence",
    "log_pair_given_presence",
    "brier_qualify_given_presence",
    "phantom_qualify_given_presence_winner",
    # Victoire | qualification
    "p_win_given_qualify_winner",
    "log_win_given_qualify_winner",
    "p_win_given_pair",
    "log_win_given_pair",
    "phantom_opponent_given_qualify_winner",
    # Soutien | présence
    "crps_conditional",
    "crps_global_on_conditional",
    "coverage90_conditional",
    "n_conditional",
    # Global et décomposition
    "log_score_winner",
    "log_score_pair",
    "crps_global",
    "coverage90_global",
    "smoothing_gap_winner",
    "smoothing_gap_pair",
    "identity_gap_winner_raw",
    "identity_gap_pair_raw",
    "error_stage_winner",
]
COLORS = {
    "presence": "#8c564b",
    "qualification|presence": "#2563a6",
    "victory|qualification": "#b66b32",
}


def reference_diagnostics(reference: Path, registry):
    spec = verify(reference)["spec"]
    summaries, stages, candidates, checks = [], [], [], []
    for election, horizon, name in sd.case_names(spec):
        folder = reference / "cases" / name
        if not complete_score(folder):
            raise ValueError(f"cas incomplet : {folder}")
        outcome = sd.election_outcome(election, registry)
        frozen = pl.read_parquet(folder / "metrics.parquet")
        for method, (summary, table, events) in sd.diagnose_folder(folder, outcome).items():
            key = {"election": election, "horizon": horizon, "method": method}
            row = frozen.filter(pl.col("method") == method).row(0, named=True)
            gaps = {mine: abs(summary[mine] - row[theirs]) for mine, theirs in FROZEN.items()}
            if max(gaps.values()) > 1e-12:
                raise ValueError(f"score global non reproduit : {name}/{method} {gaps}")
            checks.append(key | {"max_abs_gap_vs_frozen": max(gaps.values())})
            summaries.append(key | summary)
            stages += [key | e for e in events]
            candidates += [key | r for r in table.to_dicts()]
    return (
        spec,
        pl.DataFrame(summaries, infer_schema_length=None),
        pl.DataFrame(stages, infer_schema_length=None),
        pl.DataFrame(candidates, infer_schema_length=None),
        checks,
    )


def horizon_means(frame: pl.DataFrame, keys: list[str], columns: list[str]) -> pl.DataFrame:
    """Moyenne à poids égal par élection ; effectif des valeurs définies publié."""
    numeric = [c for c in columns if frame.schema[c].is_numeric()]
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("n_elections"),
            *[pl.col(c).mean() for c in numeric],
            *[pl.col(c).is_not_null().sum().alias(f"n_{c}") for c in numeric],
        )
        .sort(keys)
    )


def stage_plot(summary: pl.DataFrame, file: Path) -> None:
    years = sorted(summary["election"].unique())
    fig, axes = plt.subplots(
        2, len(years), figsize=(13, 7.5), sharey="row", constrained_layout=True
    )
    chains = [
        (
            "Chaîne du vainqueur : −log P(vainqueur réel)",
            [
                ("log_presence_winner", "presence"),
                ("log_qualify_given_presence_winner", "qualification|presence"),
                ("log_win_given_qualify_winner", "victory|qualification"),
            ],
        ),
        (
            "Chaîne de la paire : −log P(paire réelle, vainqueur réel)",
            [
                ("log_presence_pair", "presence"),
                ("log_pair_given_presence", "qualification|presence"),
                ("log_win_given_pair", "victory|qualification"),
            ],
        ),
    ]
    labels = {
        "presence": "Présence",
        "qualification|presence": "Qualification | présence",
        "victory|qualification": "Victoire | qualification",
    }
    horizons = [180, 90, 30, 7]
    for r, (title, parts) in enumerate(chains):
        for ax, year in zip(axes[r], years):
            for m, (method, hatch) in enumerate((("reference", ""), ("mean14d", "//"))):
                data = summary.filter((pl.col("election") == year) & (pl.col("method") == method))
                bottom = np.zeros(len(horizons))
                for column, stage in parts:
                    values = np.array(
                        [-data.filter(pl.col("horizon") == h)[column].item() for h in horizons]
                    )
                    ax.bar(
                        np.arange(len(horizons)) + (m - 0.5) * 0.38,
                        values,
                        0.36,
                        bottom=bottom,
                        color=COLORS[stage],
                        hatch=hatch,
                        edgecolor="white",
                        label=labels[stage] if (year == years[0] and m == 0) else None,
                    )
                    bottom += values
            ax.set_xticks(np.arange(len(horizons)), [f"{h} j" for h in horizons])
            ax.set_title(f"{year}" if r == 0 else "")
            ax.grid(axis="y", alpha=0.2)
        axes[r][0].set_ylabel(title, fontsize=9)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Perte de log-score par étape (barre pleine : référence ; hachurée : moyenne 14 j)\n"
        "Lissage de Jeffreys à deux issues par étape ; plus haut = pire"
    )
    fig.savefig(file, dpi=170)
    plt.close(fig)


def support_plot(summary: pl.DataFrame, file: Path) -> None:
    data = summary.filter(pl.col("method") == "reference").sort(
        "election", "horizon", descending=[False, True]
    )
    names = [f"{e}\n{h} j" for e, h in zip(data["election"], data["horizon"])]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    ax.bar(
        x - 0.2,
        data["crps_global_on_conditional"],
        0.38,
        color="#9db7d5",
        label="CRPS global (zéro si absent)",
    )
    ax.bar(x + 0.2, data["crps_conditional"], 0.38, color="#2563a6", label="CRPS | présence")
    for i, cov in enumerate(data["coverage90_conditional"]):
        ax.text(
            x[i] + 0.2, data["crps_conditional"][i] + 0.05, f"{cov:.0%}", ha="center", fontsize=7
        )
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("CRPS moyen (points), officiels présents")
    ax.set_title(
        "Soutien conditionnel à la présence, référence (étiquette : couverture 90 % conditionnelle)"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(file, dpi=170)
    plt.close(fig)


def delta_plot(means: pl.DataFrame, file: Path) -> None:
    rows = [(c, s) for c, s in sd.DELTA_COLUMNS]
    groups = [(v, m) for v in sorted(means["variant"].unique()) for m in sd.METHODS]
    fig, axes = plt.subplots(
        1, len(groups), figsize=(4.2 * len(groups), 6), constrained_layout=True, sharey=True
    )
    for ax, (variant, method) in zip(np.atleast_1d(axes), groups):
        sub = means.filter((pl.col("variant") == variant) & (pl.col("method") == method)).sort(
            "horizon", descending=True
        )
        horizons = sub["horizon"].to_list()
        values = np.array(
            [[sub[f"delta_{c}"][j] for j in range(len(horizons))] for c, _ in rows], dtype=float
        )
        signed = np.array([s for _, s in rows])[:, None] * values
        scale = np.nanmax(np.abs(signed)) or 1
        ax.imshow(
            np.sign(signed) * np.sqrt(np.abs(signed) / scale),
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            aspect="auto",
        )
        for i in range(len(rows)):
            for j in range(len(horizons)):
                v = values[i, j]
                ax.text(
                    j,
                    i,
                    "—" if math.isnan(v) else f"{v:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_xticks(range(len(horizons)), [f"{h} j" for h in horizons])
        ax.set_yticks(range(len(rows)), [c for c, _ in rows], fontsize=8)
        ax.set_title(f"{variant} / {method}", fontsize=10)
    fig.suptitle("Écarts variante − référence, moyenne à poids égal par élection (bleu = mieux)")
    fig.savefig(file, dpi=170)
    plt.close(fig)


def main(reference: Path, study: Path | None, out: Path) -> None:
    for other in (reference, study):
        if other is not None and (out == other or out.is_relative_to(other)):
            raise ValueError("le dossier de sortie doit rester hors des dossiers lus")
    registry = default_registry()
    spec, summary, stages, candidates, checks = reference_diagnostics(reference, registry)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["election", "horizon", "method"]
    summary = summary.sort(keys)
    summary.write_csv(out / "summary_full.csv", float_precision=8)
    summary.select(keys + SYNTHESIS).write_csv(out / "summary.csv", float_precision=6)
    stages.sort(keys).write_csv(out / "stages.csv", float_precision=8)
    candidates.sort(keys + ["candidate_id"]).write_csv(out / "candidates.csv", float_precision=8)
    means = horizon_means(
        summary, ["horizon", "method"], [c for c in SYNTHESIS if c != "error_stage_winner"]
    )
    means.sort("horizon", "method", descending=[True, False]).write_csv(
        out / "means.csv", float_precision=6
    )
    stage_plot(summary, out / "stage_losses.png")
    support_plot(summary, out / "support.png")
    record = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "reference": str(reference),
        "reference_sha256": digest(reference / "reference.json"),
        "reference_case_outputs": {
            name: digest(reference / "cases" / name / "scored.json")
            for _, _, name in sd.case_names(spec)
        },
        "sources": {
            "back/src/vvs/eval/stage_diagnostics.py": digest(Path(sd.__file__)),
            "back/studies/stage_diagnostics.py": digest(Path(__file__)),
        },
        "frozen_score_reproduction": checks,
    }
    if study is not None:
        var_summary, deltas = sd.compare_study(study, reference, registry)
        # Les scores globaux des variantes doivent aussi reproduire leurs metrics.parquet.
        for row in var_summary.filter(pl.col("variant") != "reference_frozen").iter_rows(
            named=True
        ):
            name = f"{row['election']}-{row['horizon']:03d}"
            frozen = pl.read_parquet(study / "cases" / name / row["variant"] / "metrics.parquet")
            theirs = frozen.filter(pl.col("method") == row["method"]).row(0, named=True)
            gap = max(abs(row[m] - theirs[t]) for m, t in FROZEN.items())
            if gap > 1e-12:
                raise ValueError(f"score de variante non reproduit : {name}/{row['variant']}")
        target = out / "variants"
        target.mkdir(exist_ok=True)
        var_summary.sort(keys + ["variant", "against"]).write_csv(
            target / "summary_full.csv", float_precision=8
        )
        deltas = deltas.sort(keys + ["variant"])
        deltas.write_csv(target / "deltas.csv", float_precision=6)
        delta_cols = [f"delta_{c}" for c, _ in sd.DELTA_COLUMNS]
        dmeans = horizon_means(deltas, ["variant", "horizon", "method"], delta_cols)
        dmeans.sort("variant", "method", "horizon", descending=[False, False, True]).write_csv(
            target / "delta_means.csv", float_precision=6
        )
        # Signe par élection : nombre d'élections où la variante fait mieux ou moins bien
        # (écart strictement non nul), par étape ; les égalités exactes ne comptent pas.
        signs = (
            deltas.group_by("variant", "horizon", "method")
            .agg(
                *[
                    pl.col(f"better_{c}").cast(pl.Int32).sum().alias(f"n_better_{c}")
                    for c, _ in sd.DELTA_COLUMNS
                ],
                *[
                    ((pl.col(f"delta_{c}") * s) < 0).cast(pl.Int32).sum().alias(f"n_worse_{c}")
                    for c, s in sd.DELTA_COLUMNS
                ],
            )
            .sort("variant", "method", "horizon", descending=[False, False, True])
        )
        signs.write_csv(target / "signs.csv")
        delta_plot(dmeans, target / "deltas.png")
        record["study"] = str(study)
        record["study_sha256"] = (
            digest(study / "study.json") if (study / "study.json").exists() else None
        )
    write_json(out / "manifest.json", record)
    print(f"Diagnostics écrits : {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(
        args.reference.resolve(),
        args.study.resolve() if args.study else None,
        args.out.resolve(),
    )
