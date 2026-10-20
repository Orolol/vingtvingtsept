"""Étude : backtest leave-one-election-out du modèle de second tour (2002 à 2022).

    uv run python studies/second_round_backtest.py [--out runs/<date>-second-round] [--n 8000]

Rejoue le duel réellement joué à la veille du second tour et à 30, 90 et 180 jours du premier
tour (vvs.eval.second_round), contre deux baselines : moyenne des sondages de duel des
14 derniers jours, reports historiques sans sondages de duel. Écrit `metrics.parquet`,
`errors.parquet`, `transfers.parquet`, `summaries.parquet`, `proximity_check.parquet`,
`report.md` et deux figures (intervalles prédictifs, reports estimés contre observés).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from vvs import paths
from vvs.eval import second_round as ev
from vvs.model import second_round as sr
from vvs.model.plots import INK, MUTED, RESULT, SERIES, _style, display_name
from vvs.model.run import _git_commit
from vvs.registry import BLOCS

METHOD_LABEL = {
    "model": "modèle",
    "model_no_bias": "modèle sans biais RN",
    "transfers_only": "reports seuls",
    "polls_14d": "sondages 14 j",
}
METHOD_COLOR = {
    "model": SERIES,
    "model_no_bias": "#7fb0ea",
    "transfers_only": "#8a8984",
    "polls_14d": "#3a9b6b",
}
HORIZON_LABEL = {
    "r2_eve": "veille du 2d tour",
    "r1_30": "30 j avant le 1er tour",
    "r1_90": "90 j avant le 1er tour",
    "r1_180": "180 j avant le 1er tour",
}


def _f(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def plot_intervals(metrics: pl.DataFrame, path: Path) -> Path:
    hz = list(ev.HORIZONS)
    elections = sorted(metrics["election"].unique().to_list())
    fig, axes = plt.subplots(1, len(hz), figsize=(4.2 * len(hz), 5.2), sharey=True)
    for ax, h in zip(axes, hz):
        for k, e in enumerate(elections):
            sub = metrics.filter((pl.col("election") == e) & (pl.col("horizon") == h))
            if sub.is_empty():
                continue
            y0 = len(elections) - 1 - k
            res = sub["result"][0]
            ax.plot([res, res], [y0 - 0.4, y0 + 0.4], color=RESULT, lw=2, zorder=3)
            for m_i, m in enumerate(ev.METHODS):
                r = sub.filter(pl.col("method") == m)
                if r.is_empty() or r["median"][0] is None:
                    continue
                y = y0 + 0.3 - 0.2 * m_i
                ax.plot([r["q05"][0], r["q95"][0]], [y, y], color=METHOD_COLOR[m], lw=1.2)
                ax.plot([r["q25"][0], r["q75"][0]], [y, y], color=METHOD_COLOR[m], lw=3.5)
                ax.plot(r["median"][0], y, "o", color="white", mec=METHOD_COLOR[m], ms=4)
        ax.axvline(50, color=MUTED, lw=0.6, ls=":")
        ax.set_title(HORIZON_LABEL[h], fontsize=10, color=INK)
        ax.set_xlabel("part du vainqueur (%)")
        _style(ax)
    axes[0].set_yticks(range(len(elections)))
    axes[0].set_yticklabels([str(e) for e in reversed(elections)])
    handles = [plt.Line2D([], [], color=METHOD_COLOR[m], lw=3) for m in ev.METHODS]
    handles.append(plt.Line2D([], [], color=RESULT, lw=2))
    fig.legend(
        handles, [METHOD_LABEL[m] for m in ev.METHODS] + ["résultat"],
        loc="lower center", ncol=5, frameon=False, fontsize=9,
    )
    fig.suptitle("Second tour : intervalles prédictifs 50 et 90 %, leave-one-election-out",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_transfers(tr: pl.DataFrame, path: Path) -> Path:
    """Reports estimés (veille du second tour) contre reports observés dans les enquêtes."""
    tr = tr.filter((pl.col("weight") >= 0.01) & pl.col("obs_split").is_not_null())
    elections = sorted(tr["election"].unique().to_list())
    colors = dict(zip(elections, ["#2a78d6", "#3a9b6b", "#e3a33b", "#e34948", "#7d55c7"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    panels = [
        ("split", "part vers le vainqueur parmi les exprimés de l'électorat"),
        ("none", "part abstention, blanc ou nul"),
    ]
    for ax, (what, title) in zip(axes, panels):
        for e in elections:
            sub = tr.filter(pl.col("election") == e)
            if what == "none":
                sub = sub.filter(pl.col("obs_none").is_not_null())
            obs = sub[f"obs_{what}"].to_numpy() * 100
            post = sub[f"post_{what}"].to_numpy() * 100
            lo = sub[f"post_{what}_q05"].to_numpy() * 100
            hi = sub[f"post_{what}_q95"].to_numpy() * 100
            prior = sub[f"prior_{what}"].to_numpy() * 100
            c = colors[e]
            ax.vlines(obs, lo, hi, color=c, lw=1, alpha=0.7)
            ax.plot(obs, post, "o", color=c, ms=5, label=str(e))
            ax.plot(obs, prior, "o", mfc="none", mec=c, ms=5, alpha=0.8)
            for o, p, g in zip(obs, post, sub["group"].to_list()):
                if g.startswith("own:"):
                    continue
                ax.annotate(g.split(":")[-1].replace("_", " "), (o, p), fontsize=6,
                            color=MUTED, xytext=(3, 2), textcoords="offset points")
        ax.plot([0, 100], [0, 100], color=MUTED, lw=0.6, ls=":")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("observé (enquêtes post-électorales, %)")
        ax.set_ylabel("estimé (%)")
        ax.set_title(title, fontsize=10, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, title="plein : a posteriori\ncreux : a priori",
                   title_fontsize=7)
    fig.suptitle("Reports estimés à la veille du second tour contre reports observés",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def proximity_table() -> list[str]:
    cols = list(BLOCS)
    lines = ["| électorat \\ finaliste | " + " | ".join(cols) + " | ni (min–max) |"]
    lines.append("|---" * (len(cols) + 2) + "|")
    for row, vals in sr.PROXIMITY.items():
        lo, hi = sr.PROXIMITY_NONE.get(row, sr.PROXIMITY_NONE_DEFAULT)
        lines.append(
            f"| {row} | " + " | ".join(_f(float(vals[c]), 1) for c in cols)
            + f" | {_f(lo, 2)}–{_f(hi, 2)} |"
        )
    return lines


def to_markdown(out: dict[str, pl.DataFrame], prox: pl.DataFrame, settings: dict) -> str:
    m = out["metrics"]
    L = ["# Second tour par reports de voix : backtest leave-one-election-out", ""]
    L.append(
        f"Run du {settings['date']} (commit {settings['commit']}), {settings['n']} tirages par "
        f"cas, graine {settings['seed']}. Duel réellement joué, orienté sur le vainqueur. "
        "Tout ce qui est calibré sur l'histoire (analogues de reports, erreur des sondages de "
        "duel, bruit du premier tour) exclut l'élection évaluée."
    )
    L += ["", "## Paramètres d'erreur des sondages de duel par pli", ""]
    L.append("| élection exclue | σ0 | σ1 (par √mois) | biais RN | σ maison | calibré sur |")
    L.append("|---|---|---|---|---|---|")
    for r in out["errors"].iter_rows(named=True):
        L.append(
            f"| {r['election']} | {_f(r['sigma0'], 2)} | {_f(r['sigma1'], 2)} | "
            f"{_f(r['bias_far_right'], 2)} | {_f(r['sigma_house'], 2)} | {r['calibrated_on']} |"
        )
    L += [
        "",
        ("Erreur d'un sondage terminé `a` jours avant le second tour : écart-type "
        "√(σ0² + σ1² a/30), commune à tous les sondages (marche aléatoire vue depuis le vote). "
        "Biais RN : surestimation moyenne du finaliste d'extrême droite, retirée des sondages."),
    ]
    for common, title in ((False, "tous les cas"), (True, "cas communs à toutes les méthodes")):
        agg = ev.aggregate(m, common=common)
        L += ["", f"## Synthèse par horizon ({title})", ""]
        L.append("| horizon | méthode | n | erreur abs. | CRPS | couv. 50 % | couv. 90 % | log-score |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in agg.iter_rows(named=True):
            L.append(
                f"| {HORIZON_LABEL[r['horizon']]} | {METHOD_LABEL[r['method']]} | {r['n']} | "
                f"{_f(r['mae'], 2)} | {_f(r['crps'], 2)} | {_f(r['cover50'], 2)} | "
                f"{_f(r['cover90'], 2)} | {_f(r['log_score'], 3)} |"
            )
    L += [
        "",
        ("Log-score : log P(vainqueur gagne), fréquence lissée (k + ½)/(N + 1). La baseline « sondages 14 j » "
        "manque quand aucun sondage du duel n'a moins de 14 jours (2002 avant le premier "
        "tour) : le second tableau compare les méthodes sur les mêmes cas."),
        "",
        "## Par élection et horizon",
        "",
        "Médiane [5 % – 95 %] de la part du vainqueur ; entre parenthèses, P(vainqueur gagne).",
        "",
        ("| élection | horizon | résultat | sondages duel | modèle | modèle sans biais | "
        "reports seuls | sondages 14 j |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for (e, h), sub in m.group_by(["election", "horizon"], maintain_order=True):
        cells = []
        for meth in ev.METHODS:
            r = sub.filter(pl.col("method") == meth).row(0, named=True)
            if r["median"] is None:
                cells.append("–")
                continue
            cells.append(
                f"{_f(r['median'])} [{_f(r['q05'])}–{_f(r['q95'])}] ({_f(r['p_winner'], 2)})"
            )
        r0 = sub.row(0, named=True)
        n_polls = sub.filter(pl.col("method") == "model")["n_polls"][0]
        L.append(
            f"| {e} | {HORIZON_LABEL[h]} | {_f(r0['result'])} ({display_name(r0['winner'])}) | "
            f"{n_polls} | " + " | ".join(cells) + " |"
        )
    L += ["", "Entrées de premier tour :", ""]
    for r in out["summaries"].iter_rows(named=True):
        L.append(f"- {r['election']}, {HORIZON_LABEL[r['horizon']]} : {r['first_round']}"
                 + (f" — {r['notes']}" if r["notes"] else ""))
    tr = out["transfers"].filter(pl.col("weight") >= 0.01)
    L += [
        "",
        "## Reports estimés à la veille du second tour contre reports observés",
        "",
        ("Part vers le vainqueur parmi les exprimés de l'électorat (a priori, a posteriori "
        "médiane [5–95 %], observé dans les enquêtes de l'élection, exclues du prior) et part "
        "abstention-blanc-nul. Poids : part de l'électorat rapportée aux exprimés du premier "
        "tour (abstentionnistes : inscrits non exprimés / exprimés)."),
        "",
        "| élection | électorat | poids | source du prior | répartition a priori | a posteriori | observée | ni a priori | ni a posteriori | ni observée |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in tr.iter_rows(named=True):
        src = r["source"] + (f" ({r['analogs']})" if r["analogs"] else "")
        L.append(
            f"| {r['election']} | {r['group']} | {_f(100 * r['weight'])} | {src} | "
            f"{_f(100 * r['prior_split'])} | {_f(100 * r['post_split'])} "
            f"[{_f(100 * r['post_split_q05'])}–{_f(100 * r['post_split_q95'])}] | "
            f"{_f(None if r['obs_split'] is None else 100 * r['obs_split'])} | "
            f"{_f(100 * r['prior_none'])} | {_f(100 * r['post_none'])} | "
            f"{_f(None if r['obs_none'] is None else 100 * r['obs_none'])} |"
        )
    L += ["", "## Matrice de proximité des blocs (prior sans analogue)", ""]
    L += proximity_table()
    L += [
        "",
        (f"Répartition entre i et j : logistique de {sr.PROXIMITY_LAMBDA} × (a_i − a_j) ; part "
        "ni l'un ni l'autre : n_min + (n_max − n_min)(1 − max(a_i, a_j)/10). Concentration "
        f"Dirichlet {sr.KAPPA_PROXIMITY:g}, décalages communs σ_u = {sr.SIGMA_U['inedit']}, "
        f"σ_v = {sr.SIGMA_V['inedit']} (logit)."),
        "",
        "Recoupement avec les reports observés (ne sert à aucun calibrage) :",
        "",
        "| élection | électorat | répartition matrice | observée | ni matrice | ni observée |",
        "|---|---|---|---|---|---|",
    ]
    for r in prox.iter_rows(named=True):
        L.append(
            f"| {r['election']} | {r['group']} | {_f(100 * r['matrix_split'])} | "
            f"{_f(100 * r['obs_split'])} | {_f(100 * r['matrix_none'])} | "
            f"{_f(None if r['obs_none'] is None else 100 * r['obs_none'])} |"
        )
    L += ["", "Figures : `intervals.png`, `transfers.png`.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    today = dt.date.today().isoformat()
    out_dir = args.out or paths.RUNS / f"{today}-second-round"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = ev.run(n=args.n, seed=args.seed)
    prox = ev.proximity_check()
    for name, df in out.items():
        df.write_parquet(out_dir / f"{name}.parquet")
    prox.write_parquet(out_dir / "proximity_check.parquet")
    plot_intervals(out["metrics"], out_dir / "intervals.png")
    plot_transfers(out["transfers"], out_dir / "transfers.png")
    settings = {"date": today, "commit": _git_commit(), "n": args.n, "seed": args.seed}
    (out_dir / "report.md").write_text(to_markdown(out, prox, settings), encoding="utf-8")
    print(f"écrit : {out_dir}")


if __name__ == "__main__":
    main()
