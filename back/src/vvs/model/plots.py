"""Figures matplotlib d'un ajustement : trajectoires, effets maison, estimation au vote."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

from vvs.registry import default_registry

# Palette (références du projet : bleu = estimation, orange = résultat officiel)
SERIES = "#2a78d6"
RESULT = "#eb6834"
POINTS = "#8a8984"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("house", ["#2a78d6", "#f0efec", "#e34948"])


def display_name(cid: str) -> str:
    c = default_registry().candidates.get(cid)
    return c.name if c is not None else cid


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_trajectories(
    traj: pl.DataFrame,
    observed: pl.DataFrame,
    final: pl.DataFrame,
    asof: dt.date,
    vote_date: dt.date,
    path: Path,
    start: dt.date | None = None,
) -> Path:
    """Une vignette par candidat de la référence : médiane, intervalles 50 et 90 %, sondages
    (parts normalisées dans leur hypothèse), date asof et résultat officiel s'il existe."""
    cands = final["candidate_id"].to_list()
    official = dict(zip(cands, final["official"].to_list()))
    ref = traj.filter((pl.col("level") == "candidate") & (pl.col("config") == "reference"))
    if start is None and observed.height:
        start = observed["date"].min()
    if start is not None:
        ref = ref.filter(pl.col("date") >= start - dt.timedelta(days=7))
        observed = observed.filter(pl.col("date") >= start - dt.timedelta(days=7))
    ncol = 4
    nrow = math.ceil(len(cands) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    for ax, cid in zip(axes.flat, cands):
        t = ref.filter(pl.col("id") == cid).sort("date")
        x = t["date"].to_list()
        ax.fill_between(x, t["q05"], t["q95"], color=SERIES, alpha=0.15, linewidth=0, label="90 %")
        ax.fill_between(x, t["q25"], t["q75"], color=SERIES, alpha=0.3, linewidth=0, label="50 %")
        ax.plot(x, t["median"], color=SERIES, linewidth=2, label="médiane")
        o = observed.filter(pl.col("candidate_id") == cid)
        ax.scatter(
            o["date"].to_list(),
            100 * o["share"].to_numpy(),
            s=8,
            color=POINTS,
            alpha=0.5,
            linewidths=0,
            label="sondages",
        )
        ax.axvline(asof, color=MUTED, linestyle="--", linewidth=1)
        if official.get(cid) is not None:
            ax.scatter(
                [vote_date],
                [official[cid]],
                s=50,
                color=RESULT,
                zorder=5,
                edgecolor="white",
                linewidth=1.5,
                label="résultat",
            )
        ax.set_title(display_name(cid), fontsize=10, color=INK, loc="left")
        ax.set_ylim(bottom=0)
        _style(ax)
        ax.tick_params(axis="x", labelrotation=30)
    for ax in list(axes.flat)[len(cands) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(
        f"Intentions de vote au premier tour (% exprimés), estimation au {asof.isoformat()}",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_house_effects(house: pl.DataFrame, final: pl.DataFrame, path: Path) -> Path:
    """Carte institut × candidat de l'effet maison médian, en points de pourcentage."""
    cands = final["candidate_id"].to_list()
    pollsters = (
        house.select("pollster", "n_polls").unique().sort("n_polls", descending=True)["pollster"]
    ).to_list()
    grid = np.zeros((len(pollsters), len(cands)))
    for row in house.iter_rows(named=True):
        if row["candidate_id"] in cands:
            grid[pollsters.index(row["pollster"]), cands.index(row["candidate_id"])] = row[
                "effect_pts_median"
            ]
    lim = max(0.5, float(np.abs(grid).max()))
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(cands), 0.9 + 0.45 * len(pollsters)))
    im = ax.imshow(grid, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    n_polls = dict(house.select("pollster", "n_polls").unique().iter_rows())
    ax.set_yticks(range(len(pollsters)), [f"{p} ({n_polls[p]})" for p in pollsters], fontsize=8)
    ax.set_xticks(
        range(len(cands)), [display_name(c) for c in cands], rotation=40, ha="right", fontsize=8
    )
    for i in range(len(pollsters)):
        for j in range(len(cands)):
            ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("effet maison (points)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(
        "Effets maison médians : écart de l'institut à l'agrégat (points, nb de sondages)",
        fontsize=10,
        loc="left",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_final(final: pl.DataFrame, path: Path, title: str) -> Path:
    """Estimation au jour du vote (intervalles 50 et 90 %) contre résultat officiel."""
    df = final.sort("vote_median")
    y = np.arange(df.height)
    fig, ax = plt.subplots(figsize=(7, 0.4 * df.height + 1.2))
    ax.hlines(
        y, df["vote_q05"], df["vote_q95"], color=SERIES, alpha=0.35, linewidth=6, label="90 %"
    )
    ax.hlines(y, df["vote_q25"], df["vote_q75"], color=SERIES, linewidth=6, label="50 %")
    ax.scatter(
        df["vote_median"], y, color="white", edgecolor=SERIES, s=30, zorder=4, label="médiane"
    )
    if df["official"].is_not_null().any():
        ax.scatter(df["official"], y, color=RESULT, marker="D", s=30, zorder=5, label="résultat")
    ax.set_yticks(y, [display_name(c) for c in df["candidate_id"]], fontsize=8)
    ax.set_xlabel("% exprimés", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
