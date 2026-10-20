"""Étude : la part d'électeurs « sûrs de leur choix » prédit-elle la volatilité d'un candidat ?

    uv run python studies/certainty.py [--out runs/<date>-certainty] [--perm 10000] [--coverage]

Données : `data/curated/poll_certainty.parquet` (`vvs ingest certainty`), sondages curated,
résultats officiels. Élections 2017 et 2022 (les seules dont les notices de la Commission
ventilent la certitude par candidat ; 2027 n'a pas de résultat).

Protocole, fixé avant de lire les résultats :

1. Tranches d'horizon h ∈ {180, 120, 90, 60, 30, 7} jours : fenêtre de terrain
   ]vote − h − 15, vote − h + 15] (]vote − 14, vote − 1] pour h = 7). Dans chaque tranche,
   certitude par candidat x_c par le modèle additif part = a_sondage + b_c
   (`vvs.model.certainty.two_way_effects`, les instituts ne mesurent pas sur la même base),
   centrée ; candidats officiels seulement.
2. Parts : moyenne des sondages de la même fenêtre (`baselines.poll_mean`, hypothèse la plus
   proche des candidats officiels), moyenne des 14 derniers jours (agrégat final), résultat.
   Sur l'échelle des log-parts renormalisées sur les candidats retenus, centrées (moyenne
   pondérée par les parts) :
   - mouvement ultérieur m_c = log(final) − log(horizon), sauf h = 7 ;
   - erreur finale des sondages f_c = log(résultat) − log(final) ;
   - erreur totale t_c = m_c + f_c, et sa version standardisée z_c = t_c / sd_η,c, sd_η de
     η calibré sans l'élection (`ErrorModel`), qui contient déjà un terme ρ/p de taille.
3. Statistiques par (élection, horizon) : Spearman(x, |m|), Spearman(x, |f|), Spearman(x, |z|),
   et pour |m| et |f| la corrélation partielle de rangs à log(part) fixée (les gros candidats
   ont plus d'électeurs sûrs et bougent moins en relatif). Signe attendu : négatif.
4. Placebo : permutation des candidats. Par cellule, permutation des x entre candidats ;
   pour la moyenne sur toutes les cellules, une même permutation des identités de candidats
   par élection, appliquée à toutes ses tranches (les tranches d'une élection partagent les
   mêmes candidats et des mouvements emboîtés : les permuter indépendamment surestimerait la
   significativité). p unilatéral : proportion de permutations au moins aussi négatives.
5. Paramètre d'intégration : κ de s_c = exp(−κ x_c) (`vvs.model.certainty`), estimé sur chaque
   élection seule et en leave-one-election-out, avec le gain de log-vraisemblance contre κ = 0.

Critère de passage vers l'intégration, fixé d'avance : moyenne des Spearman(x, |z|) négative
avec p placebo < 0,10, et même signe dans les deux élections. Sinon l'intégration est quand
même rejouée au backtest (la règle d'admission est celle de docs/07), mais le résultat de
l'étude est consigné comme négatif.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from vvs import paths
from vvs.eval.baselines import poll_mean
from vvs.model import certainty as mc
from vvs.model.data import election_date, load_polls
from vvs.model.error import ErrorModel
from vvs.model.plots import INK, MUTED, _style
from vvs.registry import default_registry

ELECTIONS = (2017, 2022)
HORIZONS = (180, 120, 90, 60, 30, 7)
HALF_WINDOW = 15
STATS = ("rho_mov", "pcor_mov", "rho_fin", "pcor_fin", "rho_z")
STAT_LABEL = {
    "rho_mov": "Spearman(x, |mouvement|)",
    "pcor_mov": "partielle(x, |mouvement| | log part)",
    "rho_fin": "Spearman(x, |erreur finale|)",
    "pcor_fin": "partielle(x, |erreur finale| | log part)",
    "rho_z": "Spearman(x, |erreur totale / sd η|)",
}


@dataclass
class Cell:
    election: int
    horizon: int
    candidates: list[str]
    x: np.ndarray
    share: np.ndarray  # parts à l'horizon (fractions renormalisées)
    mov: np.ndarray | None
    fin: np.ndarray
    z: np.ndarray
    n_polls: int
    pollsters: list[str]


def _center(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return v - np.average(v, weights=w)


def build_cells(cert: pl.DataFrame, reg, ems: dict[int, ErrorModel]) -> list[Cell]:
    cells = []
    for e in ELECTIONS:
        vote = election_date(reg, e)
        el = reg.elections[e]
        res = {r.candidate_id: r.share for r in el.round_(1).results}
        ids = [c.candidate_id for c in el.candidates]
        polls = load_polls(e, vote - dt.timedelta(days=1))
        fin = poll_mean(e, vote - dt.timedelta(days=1), ids, days=14, registry=reg, polls=polls)
        for h in HORIZONS:
            if h == 7:
                lo, hi = vote - dt.timedelta(days=14), vote - dt.timedelta(days=1)
            else:
                lo = vote - dt.timedelta(days=h + HALF_WINDOW)
                hi = vote - dt.timedelta(days=h - HALF_WINDOW)
            sub = cert.filter(
                (pl.col("election") == e)
                & (pl.col("fieldwork_end") > lo)
                & (pl.col("fieldwork_end") <= hi)
            )
            b = mc.two_way_effects(sub)
            pm = poll_mean(e, hi, ids, days=(hi - lo).days, registry=reg, polls=polls)
            keep = [
                i
                for i, c in enumerate(ids)
                if c in b and c not in pm.missing and c not in fin.missing and pm.shares[i] > 0
            ]
            if len(keep) < 4:
                continue
            cs = [ids[i] for i in keep]
            a = pm.shares[keep] / pm.shares[keep].sum()
            f = fin.shares[keep] / fin.shares[keep].sum()
            r = np.array([res[c] for c in cs])
            r = r / r.sum()
            mov = None if h == 7 else _center(np.log(f) - np.log(a), a)
            fe = _center(np.log(r) - np.log(f), f)
            tot = _center(np.log(r) - np.log(a), a)
            blocs = [reg.bloc_of(c, e) for c in cs]
            vb, vc = ems[e].variances(pm.age_days, blocs, a)
            z = tot / np.sqrt(np.asarray(vb) + np.asarray(vc))
            xs = np.array([b[c] for c in cs]) / 100
            cells.append(
                Cell(e, h, cs, xs - xs.mean(), a, mov, fe, z, sub["poll_id"].n_unique(),
                     sorted(set(sub["pollster"].to_list())))
            )
    return cells


def _rank(v):
    return stats.rankdata(v)


def partial_rank_corr(x, y, z) -> float:
    """Corrélation de rangs de x et y à z fixé (résidus des rangs sur le rang de z)."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    d = np.sqrt((ex**2).sum() * (ey**2).sum())
    return float((ex * ey).sum() / d) if d > 0 else float("nan")


def cell_stats(c: Cell, x: np.ndarray | None = None) -> dict[str, float]:
    x = c.x if x is None else x
    logp = np.log(c.share)
    out = {
        "rho_fin": float(stats.spearmanr(x, np.abs(c.fin))[0]),
        "pcor_fin": partial_rank_corr(x, np.abs(c.fin), logp),
        "rho_z": float(stats.spearmanr(x, np.abs(c.z))[0]),
        "rho_mov": float("nan"),
        "pcor_mov": float("nan"),
    }
    if c.mov is not None:
        out["rho_mov"] = float(stats.spearmanr(x, np.abs(c.mov))[0])
        out["pcor_mov"] = partial_rank_corr(x, np.abs(c.mov), logp)
    return out


def permutation_tests(cells: list[Cell], n_perm: int, seed: int = 0):
    """p par cellule (permutation des x) et p de la moyenne par élection et globale
    (permutation des identités de candidats, la même pour toutes les tranches d'une élection)."""
    rng = np.random.default_rng(seed)
    obs = [cell_stats(c) for c in cells]
    per_cell = []
    for c, o in zip(cells, obs):
        sims = np.array([[cell_stats(c, rng.permutation(c.x))[s] for s in STATS]
                         for _ in range(min(n_perm, 2000))])
        row = {"election": c.election, "horizon": c.horizon, "n": len(c.x)}
        for j, s in enumerate(STATS):
            if np.isnan(o[s]):
                row[f"p_{s}"] = float("nan")
            else:
                row[f"p_{s}"] = float((1 + np.sum(sims[:, j] <= o[s] + 1e-12)) / (1 + len(sims)))
        per_cell.append(row)

    groups = {"toutes": cells} | {str(e): [c for c in cells if c.election == e] for e in ELECTIONS}
    pooled = {}
    universe = {e: sorted({k for c in cells if c.election == e for k in c.candidates})
                for e in ELECTIONS}
    sims_all = {g: [] for g in groups}
    for _ in range(n_perm):
        perm = {e: dict(zip(u, rng.permutation(u))) for e, u in universe.items()}
        vals = []
        for c in cells:
            # x permuté : le candidat k reçoit la certitude de perm[k] si ce dernier est mesuré
            # dans la tranche ; sinon il garde la sienne (rare, petits candidats en bord)
            idx = {k: i for i, k in enumerate(c.candidates)}
            xp = np.array([c.x[idx[perm[c.election][k]]] if perm[c.election][k] in idx
                           else c.x[i] for i, k in enumerate(c.candidates)])
            vals.append(cell_stats(c, xp))
        for g, gc in groups.items():
            sel = [v for v, c in zip(vals, cells) if c in gc]
            sims_all[g].append({s: np.nanmean([v[s] for v in sel]) for s in STATS})
    for g, gc in groups.items():
        o = {s: float(np.nanmean([obs[cells.index(c)][s] for c in gc])) for s in STATS}
        sims = sims_all[g]
        pooled[g] = {
            s: {
                "mean": o[s],
                "p": float((1 + sum(v[s] <= o[s] + 1e-12 for v in sims)) / (1 + len(sims))),
                "n_cells": int(sum(not np.isnan(obs[cells.index(c)][s]) for c in gc)),
            }
            for s in STATS
        }
    return obs, per_cell, pooled


def kappa_estimates(cert: pl.DataFrame, ems: dict[int, ErrorModel]) -> dict:
    """κ par élection seule (η calibré sans elle) et en leave-one-election-out."""
    out = {}
    cells = {e: mc.error_cells(e, ems[e], data=cert) for e in ELECTIONS}
    for e in ELECTIONS:
        k, gain, se = mc.fit_kappa_cells(cells[e])
        out[f"seule_{e}"] = {"kappa": k, "gain": gain, "se": se,
                             "n_cells": len(cells[e]), "n_obs": sum(len(c.t) for c in cells[e])}
    for e in ELECTIONS:
        # pour évaluer e : κ appris sur l'autre élection, η de l'autre élection calibré sans e
        kf = mc.fit_kappa(e, data=cert, error_model=ems[e])
        out[f"sans_{e}"] = kf.to_dict()
    curves = {}
    grid = np.linspace(-10, 10, 201)
    for e in ELECTIONS:
        curves[e] = [mc._loglik(k, cells[e]) - mc._loglik(0.0, cells[e]) for k in grid]
    return out, grid, curves, cells


# --- sorties


def fig_cells(cells: list[Cell], obs: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(len(ELECTIONS), len(HORIZONS), figsize=(2.6 * len(HORIZONS), 5.6),
                             squeeze=False)
    for i, e in enumerate(ELECTIONS):
        for j, h in enumerate(HORIZONS):
            ax = axes[i, j]
            _style(ax)
            k = next((n for n, c in enumerate(cells) if c.election == e and c.horizon == h), None)
            ax.set_title(f"{e}, {h} j", fontsize=8, loc="left", color=INK)
            if k is None:
                ax.text(0.5, 0.5, "pas de mesure", ha="center", fontsize=7, color=MUTED,
                        transform=ax.transAxes)
                continue
            c = cells[k]
            ax.scatter(100 * c.x, np.abs(c.z), s=14, color="#2a78d6")
            for xx, zz, name in zip(100 * c.x, np.abs(c.z), c.candidates):
                ax.annotate(name.split("_")[0][:6], (xx, zz), fontsize=5, color=MUTED)
            ax.text(0.02, 0.92, f"ρ = {obs[k]['rho_z']:+.2f}", fontsize=7, color=INK,
                    transform=ax.transAxes)
            if i == len(ELECTIONS) - 1:
                ax.set_xlabel("certitude centrée (points)", fontsize=7, color=MUTED)
            if j == 0:
                ax.set_ylabel("|erreur totale| / sd η", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_kappa(grid, curves, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for e, col in zip(ELECTIONS, ("#2a78d6", "#eb6834")):
        ax.plot(grid, curves[e], color=col, label=str(e))
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("κ", fontsize=8, color=MUTED)
    ax.set_ylabel("log-vraisemblance − celle de κ = 0", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def md_table(header, rows) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def f2(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}"


def fp(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def coverage_tables(cert_all: pl.DataFrame) -> list[str]:
    t = cert_all.filter(pl.col("candidate_id") != mc.TOTAL_ID)
    tot = cert_all.filter(pl.col("candidate_id") == mc.TOTAL_ID)
    reg = default_registry()
    lines = ["## Couverture des données", ""]
    rows = []
    for (e, p), g in sorted(t.group_by("election", "pollster"), key=lambda kv: kv[0]):
        vote = reg.elections[e].round_(1).date if e in reg.elections else None
        days = [(vote - d).days for d in g["fieldwork_end"].to_list()] if vote else []
        rows.append([
            str(e), p, str(g["poll_id"].n_unique()), str(g.height),
            str(g["candidate_id"].n_unique()),
            f"{g['fieldwork_end'].min()} → {g['fieldwork_end'].max()}",
            f"{max(days)} → {min(days)}" if days else "–",
            ", ".join(sorted(set(g["fmt"].to_list()))),
            f"{100 * g['aligned'].mean():.0f} %",
        ])
    lines += md_table(
        ["élection", "institut", "sondages", "lignes", "candidats", "terrain",
         "jours avant le vote", "format", "alignés sur polls"], rows)
    lines.append("")
    if tot.height:
        rows = []
        for (e, p), g in sorted(tot.group_by("election", "pollster"), key=lambda kv: kv[0]):
            only = g.join(t.select("poll_id").unique(), on="poll_id", how="anti").height
            rows.append([str(e), p, str(g.height), str(only),
                         f"{g['certain_share'].min():.0f} à {g['certain_share'].max():.0f}"])
        lines += ["Part des sûrs dans l'ensemble (`_all`) :", ""]
        lines += md_table(["élection", "institut", "sondages", "dont sans ventilation",
                           "part des sûrs"], rows)
        lines.append("")
    return lines


def write_report(out: Path, cert_all, cells, obs, per_cell, pooled, kappas, extra_cov) -> None:
    lines = ["# Certitude du choix et volatilité des candidats", "",
             f"Généré le {dt.date.today().isoformat()} par `back/studies/certainty.py`.", ""]
    lines += coverage_tables(cert_all)
    if extra_cov is not None:
        lines += ["Notices lues et notices posant la question (`vvs.ingest.certainty`) :", ""]
        rows = [[str(r["election"]), str(r["pollster"]), str(r["notices"]),
                 str(r["notices_asking"]), str(r["polls_by_candidate"] or 0)]
                for r in extra_cov.iter_rows(named=True)]
        lines += md_table(["élection", "institut", "notices", "posant la question",
                           "sondages ventilés par candidat"], rows)
        lines.append("")
    lines += ["## Cellules (élection × horizon)", ""]
    rows = []
    for c, o, p in zip(cells, obs, per_cell):
        rows.append([str(c.election), str(c.horizon), str(len(c.x)), str(c.n_polls),
                     ", ".join(c.pollsters)]
                    + [f"{f2(o[s])} (p {fp(p['p_' + s])})" for s in STATS])
    lines += md_table(["élection", "h", "candidats", "sondages", "instituts"]
                      + [STAT_LABEL[s] for s in STATS], rows)
    lines += ["", ("p : permutation des certitudes entre candidats de la cellule, unilatéral "
                   "(corrélation négative attendue)."), ""]
    lines += ["## Moyenne des cellules et placebo par permutation des candidats", ""]
    rows = []
    for g, d in pooled.items():
        rows.append([g] + [f"{f2(d[s]['mean'])} (p {fp(d[s]['p'])}, {d[s]['n_cells']} c.)"
                           for s in STATS])
    lines += md_table(["élections"] + [STAT_LABEL[s] for s in STATS], rows)
    lines += ["", "## κ de s_c = exp(−κ x_c)", ""]
    rows = [[k, f"{v['kappa']:+.2f}", f"{v.get('se', float('nan')):.2f}",
             f"{v.get('gain', v.get('loglik_gain', float('nan'))):.2f}",
             str(v.get("n_cells")), str(v.get("n_obs"))] for k, v in kappas.items()]
    lines += md_table(["estimation", "κ", "écart type", "gain de log-vrais. vs κ = 0",
                       "cellules", "observations"], rows)
    lines.append("")
    rz = pooled["toutes"]["rho_z"]
    same_sign = all(pooled[str(e)]["rho_z"]["mean"] < 0 for e in ELECTIONS)
    passed = rz["mean"] < 0 and rz["p"] < 0.10 and same_sign
    lines += ["## Critère de passage (fixé d'avance)", "",
              (f"Moyenne de Spearman(x, |z|) = {f2(rz['mean'])}, p placebo = {fp(rz['p'])}, "
               f"même signe négatif dans les deux élections : {'oui' if same_sign else 'non'}. "
               f"Critère {'rempli' if passed else 'non rempli'}."), ""]
    (out / "certainty.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-certainty")
    ap.add_argument("--perm", type=int, default=10_000)
    ap.add_argument("--coverage", action="store_true",
                    help="relire toutes les notices pour compter celles qui posent la question")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    reg = default_registry()
    cert_all = pl.read_parquet(mc.CERTAINTY_PATH)
    cert = cert_all.filter(pl.col("candidate_id") != mc.TOTAL_ID)
    extra_cov = None
    if args.coverage:
        from vvs.ingest import certainty as ic

        rep = ic.run(write=False)
        extra_cov = rep.coverage()
        extra_cov.write_csv(out / "coverage_notices.csv")
    ems = {e: ErrorModel.fit(exclude=[e], registry=reg) for e in ELECTIONS}
    cells = build_cells(cert, reg, ems)
    obs, per_cell, pooled = permutation_tests(cells, args.perm)
    kappas, grid, curves, _ = kappa_estimates(cert, ems)
    fig_cells(cells, obs, out / "certainty_cells.png")
    fig_kappa(grid, curves, out / "kappa_loglik.png")
    rows = []
    for c in cells:
        for i, k in enumerate(c.candidates):
            rows.append({"election": c.election, "horizon": c.horizon, "candidate_id": k,
                         "x": float(c.x[i]), "share": float(c.share[i]),
                         "mov": None if c.mov is None else float(c.mov[i]),
                         "fin": float(c.fin[i]), "z": float(c.z[i])})
    pl.DataFrame(rows).write_parquet(out / "cells.parquet")
    (out / "results.json").write_text(json.dumps(
        {"cells": [dict(o, **p) for o, p in zip(obs, per_cell)], "pooled": pooled,
         "kappa": kappas}, indent=2, ensure_ascii=False, default=float))
    write_report(out, cert_all, cells, obs, per_cell, pooled, kappas, extra_cov)
    print((out / "certainty.md").read_text())


if __name__ == "__main__":
    main()
