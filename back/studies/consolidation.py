"""Étude du vote utile de fin de campagne : consolidation tardive d'un camp vers son leader.

    uv run python studies/consolidation.py [--out runs/<date>-consolidation] [--n-perm 20000]
    uv run python studies/consolidation.py --backtest [--out runs/<date>-backtest-consolidation]

Camps (`vvs.model.consolidation.CAMP_OF`) : gauche radicale et gauche réunies, centre, droite,
extrême droite ; `other` exclu. Couples camp × élection : camps d'au moins deux candidats
officiels sondés, 2002-2022. Parts des sondages : moyenne des 14 jours précédant vote − τ,
configuration connue à la date, ramenée aux candidats officiels comme au backtest.

Mesures, pour le leader L du camp (premier du camp dans les sondages) et ses concurrents F :

- part du leader dans le camp s_L = p_L / p_camp, transfert y = s_L(vote) − s_L(sondages),
  fraction transférée φ = y / f (f = 1 − s_L, part des concurrents dans le camp) ;
- écart d'erreurs finales en log : d = [log r_L − log p_L] − [log r_F − log p_F] (F groupés) ;
- consolidation déjà visible dans les sondages : s_L(7 j) − s_L(30 j), même leader (celui de
  l'erreur finale) ; et après les derniers sondages : y final (14 derniers jours → vote).

Variables candidates pour l'ampleur : marge de qualification du leader m (sa part moins la 2e
plus haute des autres candidats, |m| petit = vote utile attendu fort), écart au premier
concurrent, taille du camp, part des concurrents f, certitude du choix (2017 et 2022
seulement, `poll_certainty`).

Placebos : (1) permutation des rôles : un candidat du camp tiré au hasard (part ≥ 1 point)
joue le leader, statistique moyenne de y et de d sur les couples ; (2) taille : les erreurs
finales en log de tous les candidats sont régressées sur la log-part (effets fixes d'élection),
d est recalculé sur les résidus, pour séparer le vote utile d'un simple effet « les gros
candidats sont sous-estimés ».

Critères, fixés avant les placebos, la validation croisée et le backtest (après une première
lecture descriptive des erreurs finales, déjà publiées dans docs/03-donnees.md) : la
consolidation est un phénomène général si (a) y final > 0 dans au moins 8 couples sur ~11,
(b) la moyenne de y final dépasse le 95e centile du placebo de permutation (p < 0,05) et (c) la
moyenne reste positive quel que soit l'élection retirée. Sinon elle n'est pas générale ; une
restriction au seul camp de gauche, suggérée par les données, est exploratoire et ne compte
pas pour l'admission. La forme (`vvs.model.consolidation`) est estimée et backtestée dans tous
les cas (variante `consolidation`, plus `consolidation_left` exploratoire).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from vvs import paths
from vvs.model import consolidation as co
from vvs.model.plots import GRID, INK, MUTED, _style, display_name
from vvs.registry import default_registry

ELECTIONS = co.CALIBRATION_ELECTIONS
TRAJ_HORIZONS = (120, 90, 60, 45, 30, 21, 14, 7, 0)
MIN_PLACEBO_SHARE = 1.0  # points
CAMP_COLOR = {"left": "#d64545", "far_right": "#3b3b8f", "right": "#2a78d6", "centre": "#e0a100"}
CAMP_LABEL = {"left": "gauche", "far_right": "extrême droite", "right": "droite", "centre": "centre"}
ASOF_2027 = dt.date(2026, 10, 20)
CONFIG_2027 = ["le_pen_marine", "zemmour_eric", "dupont_aignan_nicolas", "retailleau_bruno",
               "philippe_edouard", "attal_gabriel", "glucksmann_raphael", "tondelier_marine",
               "melenchon_jean_luc", "roussel_fabien", "arthaud_nathalie"]


def fmt(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def name(c: str) -> str:
    try:
        return display_name(c)
    except Exception:  # noqa: BLE001
        return c


# --- mesures par couple


def leader_share(shares: dict[str, float], camp_members: list[str], leader: str) -> float | None:
    cs = [c for c in camp_members if c in shares]
    if leader not in shares or not cs:
        return None
    return shares[leader] / sum(shares[c] for c in cs)


def certainty_by_candidate(election: int, vote: dt.date) -> dict[str, float]:
    """Part moyenne des électeurs sûrs de leur choix, sondages des 30 derniers jours."""
    df = pl.read_parquet(paths.CURATED / "poll_certainty.parquet").filter(
        (pl.col("election") == election)
        & (pl.col("fieldwork_end") > vote - dt.timedelta(days=30))
    )
    if df.is_empty():
        return {}
    g = df.group_by("candidate_id").agg(pl.col("certain_share").mean())
    return dict(zip(g["candidate_id"].to_list(), g["certain_share"].to_list()))


def size_residuals(finals: list[co.Observation], reg) -> dict[tuple[int, str], float]:
    """Résidus de log(r) − log(p) sur log(p) avec effets fixes d'élection (tous candidats)."""
    from vvs.model.consolidation import _official_results, poll_shares

    rows = []
    for e in ELECTIONS:
        sh = poll_shares(e, 0, reg)
        res = _official_results(reg, e)
        for c, p in sh.items():
            if c in res and res[c] > 0:
                rows.append((e, c, math.log(p), math.log(res[c]) - math.log(p)))
    E = sorted({r[0] for r in rows})
    X = np.array([[1.0 * (r[0] == e) for e in E] + [r[2]] for r in rows])
    y = np.array([r[3] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out = {(r[0], r[1]): float(z) for r, z in zip(rows, resid)}
    out["slope"] = float(beta[-1])
    return out


def couple_table(obs: list[co.Observation], reg) -> tuple[list[dict], dict]:
    """Une ligne par couple camp × élection (leader de l'erreur finale)."""
    from vvs.model.data import election_date

    by = {(o.election, o.horizon, o.camp): o for o in obs}
    finals = [o for o in obs if o.horizon == 0]
    resid = size_residuals(finals, reg)
    rows = []
    for o in finals:
        vote = election_date(reg, o.election)
        members = list(o.shares)
        L = o.leader
        traj = {}
        for h in TRAJ_HORIZONS:
            oh = by.get((o.election, h, o.camp))
            traj[h] = None if oh is None else leader_share(oh.shares, list(oh.shares), L)
        s_vote = o.votes[L] / o.camp_vote
        pF = sum(o.shares[c] for c in o.followers)
        rF = sum(o.votes.get(c, 0.0) for c in o.followers)
        d = (math.log(o.leader_vote) - math.log(o.leader_share)) - (math.log(rF) - math.log(pF))
        rF_res = np.average([resid[(o.election, c)] for c in o.followers if (o.election, c) in resid],
                            weights=[o.shares[c] for c in o.followers if (o.election, c) in resid])
        d_res = resid[(o.election, L)] - float(rF_res)
        cert = certainty_by_candidate(o.election, vote)
        cert_L = cert.get(L)
        cf = [(cert[c], o.shares[c]) for c in o.followers if c in cert]
        cert_F = float(np.average([x for x, _ in cf], weights=[w for _, w in cf])) if cf else None
        o30 = by.get((o.election, 30, o.camp))
        rows.append({
            "election": o.election, "camp": o.camp, "leader": L, "followers": o.followers,
            "n": len(members), "leader_poll": o.leader_share, "leader_vote": o.leader_vote,
            "followers_poll": pF, "followers_vote": rF, "camp_poll": o.camp_share,
            "camp_vote": o.camp_vote, "f_final": o.f_tau, "y_final": o.y, "phi_final": o.phi,
            "d_final": d, "d_size_resid": d_res, "margin": o.margin_pts,
            "gap_first": o.gap_first_follower,
            "s30": traj[30], "s7": traj[7], "s0": traj[0], "s_vote": s_vote,
            "y_poll_30_7": None if traj[30] is None or traj[7] is None else traj[7] - traj[30],
            "y_7_0": None if traj[7] is None else traj[0] - traj[7],
            "leader_30": None if o30 is None else o30.leader,
            "certain_leader": cert_L, "certain_followers": cert_F,
            "traj": traj,
        })
    return rows, {"size_slope": resid["slope"]}


def pair_table(obs: list[co.Observation]) -> list[dict]:
    """Leader contre chaque concurrent : écart des erreurs finales en log et mouvement 30 → 7 j
    du rapport des parts dans les sondages."""
    by = {(o.election, o.horizon, o.camp): o for o in obs}
    rows = []
    for o in obs:
        if o.horizon != 0:
            continue
        o30, o7 = by.get((o.election, 30, o.camp)), by.get((o.election, 7, o.camp))
        for c in o.followers:
            if c not in o.votes:
                continue
            e_L = math.log(o.leader_vote) - math.log(o.leader_share)
            e_c = math.log(o.votes[c]) - math.log(o.shares[c])
            mv = None
            if o30 and o7 and all(x in s.shares for s in (o30, o7) for x in (o.leader, c)):
                mv = (math.log(o7.shares[o.leader] / o7.shares[c])
                      - math.log(o30.shares[o.leader] / o30.shares[c]))
            rows.append({"election": o.election, "camp": o.camp, "leader": o.leader,
                         "follower": c, "leader_poll": o.leader_share, "follower_poll": o.shares[c],
                         "leader_vote": o.leader_vote, "follower_vote": o.votes[c],
                         "d_log": e_L - e_c, "poll_move_30_7": mv})
    return rows


# Couples nommés dans la commande de l'étude (leader, concurrent), y compris hors camp
NAMED = [
    (2002, "jospin_lionel", "chevenement_jean_pierre"), (2002, "jospin_lionel", "mamere_noel"),
    (2002, "jospin_lionel", "taubira_christiane"), (2002, "le_pen_jean_marie", "megret_bruno"),
    (2007, "royal_segolene", "besancenot_olivier"), (2007, "royal_segolene", "buffet_marie_george"),
    (2007, "le_pen_jean_marie", "de_villiers_philippe"),
    (2012, "hollande_francois", "joly_eva"), (2012, "hollande_francois", "melenchon_jean_luc"),
    (2012, "le_pen_marine", "dupont_aignan_nicolas"),
    (2017, "melenchon_jean_luc", "hamon_benoit"), (2017, "fillon_francois", "dupont_aignan_nicolas"),
    (2017, "le_pen_marine", "dupont_aignan_nicolas"),
    (2022, "melenchon_jean_luc", "jadot_yannick"), (2022, "melenchon_jean_luc", "hidalgo_anne"),
    (2022, "melenchon_jean_luc", "roussel_fabien"), (2022, "le_pen_marine", "zemmour_eric"),
]


def named_pairs(reg) -> list[dict]:
    from vvs.model.consolidation import _official_results, poll_shares

    rows = []
    for e, a, b in NAMED:
        s0, s7, s30 = (poll_shares(e, h, reg) for h in (0, 7, 30))
        res = _official_results(reg, e)
        mv = None
        if all(x in s for s in (s7, s30) for x in (a, b)):
            mv = math.log(s7[a] / s7[b]) - math.log(s30[a] / s30[b])
        rows.append({
            "election": e, "leader": a, "follower": b, "leader_poll": s0[a], "leader_vote": res[a],
            "follower_poll": s0[b], "follower_vote": res[b],
            "d_log": (math.log(res[a] / s0[a]) - math.log(res[b] / s0[b])),
            "poll_move_30_7": mv,
            "pts_leader": res[a] - s0[a], "pts_follower": res[b] - s0[b],
        })
    return rows


# --- placebos


def permutation_placebo(rows: list[dict], n_perm: int, seed: int = 0) -> dict:
    """Leader tiré au hasard dans le camp (part ≥ 1 point) : moyenne de y et de d."""
    rng = np.random.default_rng(seed)
    cand = []
    for r in rows:
        members = [r["leader"], *r["followers"]]
        poll = {r["leader"]: r["leader_poll"]}
        o = r["_obs"]
        poll.update(o.shares)
        vote = o.votes
        ys, ds = [], []
        for c in members:
            if poll[c] < MIN_PLACEBO_SHARE or vote.get(c, 0) <= 0:
                continue
            s_p = poll[c] / o.camp_share
            s_v = vote[c] / o.camp_vote
            rest_p, rest_v = o.camp_share - poll[c], o.camp_vote - vote[c]
            ys.append(s_v - s_p)
            ds.append(math.log(vote[c] / poll[c]) - math.log(rest_v / rest_p))
        cand.append((np.array(ys), np.array(ds)))
    obs_y = float(np.mean([r["y_final"] for r in rows]))
    obs_d = float(np.mean([r["d_final"] for r in rows]))
    null_y = np.zeros(n_perm)
    null_d = np.zeros(n_perm)
    for k, (ys, ds) in enumerate(cand):
        idx = rng.integers(0, len(ys), n_perm)
        null_y += ys[idx]
        null_d += ds[idx]
    null_y /= len(cand)
    null_d /= len(cand)
    return {
        "obs_y": obs_y, "obs_d": obs_d,
        "p_y": float((null_y >= obs_y).mean()), "p_d": float((null_d >= obs_d).mean()),
        "q95_y": float(np.quantile(null_y, 0.95)), "q95_d": float(np.quantile(null_d, 0.95)),
        "null_y": null_y, "null_d": null_d,
    }


# --- figures


def plot_trajectories(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = [-h for h in TRAJ_HORIZONS]
    for r in rows:
        ys = [r["traj"][h] for h in TRAJ_HORIZONS]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        col = CAMP_COLOR[r["camp"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=1.3, alpha=0.85)
        ax.plot([6], [r["s_vote"]], marker="o", color=col, markersize=4)
        ax.plot([pts[-1][0], 6], [pts[-1][1], r["s_vote"]], color=col, linestyle=":",
                linewidth=1.0)
        ax.annotate(f"{r['election']} {name(r['leader'])}", (6, r["s_vote"]), fontsize=6.5,
                    color=col, xytext=(4, 0), textcoords="offset points", va="center")
    ax.axvline(0, color=GRID, linewidth=0.8)
    ax.set_xlim(-125, 40)
    ax.set_xlabel("jours avant le vote (fin de la fenêtre de 14 jours) ; point : résultat",
                  fontsize=8, color=MUTED)
    ax.set_ylabel("part du leader dans son camp", fontsize=8, color=MUTED)
    ax.set_title("Part du leader de l'erreur finale dans son camp, sondages puis vote",
                 fontsize=10, loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(rows: list[dict], path: Path) -> None:
    vars_ = [("margin", "marge de qualification m (pts)"), ("gap_first", "leader − 1er concurrent (pts)"),
             ("camp_poll", "taille du camp (pts)"), ("f_final", "part des concurrents dans le camp")]
    fig, axes = plt.subplots(1, len(vars_), figsize=(3.2 * len(vars_), 3.2), sharey=True)
    for ax, (v, lab) in zip(axes, vars_):
        for r in rows:
            ax.scatter(r[v], r["y_final"], color=CAMP_COLOR[r["camp"]], s=18)
            ax.annotate(str(r["election"])[2:], (r[v], r["y_final"]), fontsize=6, color=MUTED,
                        xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_xlabel(lab, fontsize=8, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("y final : gain du leader dans le camp", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_placebo(pl_: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax, k, lab in ((axes[0], "y", "moyenne de y (part dans le camp)"),
                       (axes[1], "d", "moyenne de d (log)")):
        ax.hist(pl_[f"null_{k}"], bins=60, color="#b8b7b2")
        ax.axvline(pl_[f"obs_{k}"], color="#d64545", linewidth=1.6)
        ax.set_title(f"{lab} ; p = {pl_[f'p_{k}']:.3f}", fontsize=8, loc="left", color=INK)
        _style(ax)
    fig.suptitle("Placebo : leader tiré au hasard dans le camp", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_phi(obs: list[co.Observation], folds: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, camps, title in ((axes[0], None, "tous camps"), (axes[1], ("left",), "gauche")):
        for o in obs:
            if o.horizon not in co.STUDY_HORIZONS or (camps and o.camp not in camps):
                continue
            ax.scatter(o.horizon, o.y, color=CAMP_COLOR[o.camp], s=14)
        tau = np.linspace(0, 180, 100)
        for pr in folds[title].values():
            f = 0.4  # part typique des concurrents dans le camp
            ax.plot(tau, pr.phi0 * co.horizon_weight(tau, pr.kappa) * co.proximity(5.0, pr.m0) * f,
                    color=MUTED, linewidth=0.8, alpha=0.7)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.invert_xaxis()
        ax.set_title(f"transfert y(τ) observé, {title} ; lignes : forme LOO (f = 0,4, |m| = 5)",
                     fontsize=7.5, loc="left", color=INK)
        ax.set_xlabel("horizon τ (jours)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("y = f_τ − f_vote", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- étude


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() >= 4 else float("nan")


def main(out: Path, n_perm: int, n_boot: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    reg = default_registry()
    obs = co.couple_observations(ELECTIONS, (*TRAJ_HORIZONS, 180), reg)
    rows, extra = couple_table(obs, reg)
    for r in rows:
        r["_obs"] = next(o for o in obs if o.horizon == 0 and o.election == r["election"]
                         and o.camp == r["camp"])
    pairs = pair_table(obs)
    named = named_pairs(reg)
    perm = permutation_placebo(rows, n_perm)
    left = [r for r in rows if r["camp"] == "left"]
    perm_left = permutation_placebo(left, n_perm, seed=1)

    # critères
    n_pos = sum(r["y_final"] > 0 for r in rows)
    loo_means = {e: float(np.mean([r["y_final"] for r in rows if r["election"] != e]))
                 for e in ELECTIONS}
    crit = {
        "a_n_positive": n_pos, "a_n_couples": len(rows), "a_ok": n_pos >= 8,
        "b_p_perm_y": perm["p_y"], "b_ok": perm["p_y"] < 0.05,
        "c_loo_means": loo_means, "c_ok": all(v > 0 for v in loo_means.values()),
    }
    crit["general"] = crit["a_ok"] and crit["b_ok"] and crit["c_ok"]

    # variables explicatives (rang de Spearman, n ≈ 11)
    expl = {}
    for v in ("margin", "abs_margin", "gap_first", "camp_poll", "f_final", "n", "certain_gap"):
        xs = []
        for r in rows:
            if v == "abs_margin":
                xs.append(abs(r["margin"]))
            elif v == "certain_gap":
                xs.append(np.nan if r["certain_leader"] is None or r["certain_followers"] is None
                          else r["certain_leader"] - r["certain_followers"])
            else:
                xs.append(r[v])
        expl[v] = {"y_final": spearman(xs, [r["y_final"] for r in rows]),
                   "phi_final": spearman(xs, [r["phi_final"] for r in rows]),
                   "y_30": spearman(
                       [x for x, r in zip(xs, rows)],
                       [next((o.y for o in obs if o.horizon == 30 and o.election == r["election"]
                              and o.camp == r["camp"]), np.nan) for r in rows])}

    # forme : paramètres par pli (tous camps et gauche seule)
    folds: dict[str, dict] = {"tous camps": {}, "gauche": {}}
    for title, camps in (("tous camps", None), ("gauche", ("left",))):
        ob = [o for o in obs if camps is None or o.camp in camps]
        for ex in (None, *ELECTIONS):
            folds[title][ex] = co.fit_params(ob, exclude=() if ex is None else (ex,),
                                             n_boot=n_boot)
    # sensibilité à κ (validation croisée par élection, m0 choisi dans chaque pli)
    kappa_cv = {}
    for kappa in (0.5, 1.0, 2.0):
        s = 0.0
        for ex in ELECTIONS:
            tr = [o for o in obs if o.election != ex and o.horizon in co.FIT_HORIZONS]
            te = [o for o in obs if o.election == ex and o.horizon in co.FIT_HORIZONS]
            m0, _ = co.select_m0(tr, kappa)
            s += co._sse(te, co._ls_phi0(tr, m0, kappa, co.PROX_SCALE), m0, kappa, co.PROX_SCALE)
        kappa_cv[kappa] = s
    zero_cv = sum((o.y ** 2) for o in obs if o.horizon in co.FIT_HORIZONS)
    form_cv = form_cross_validation(obs)
    # φ libre par horizon (estimateur de ratio Σ y / Σ f)
    free = {}
    for h in co.STUDY_HORIZONS:
        for title, camps in (("tous camps", None), ("gauche", ("left",))):
            ob = [o for o in obs if o.horizon == h and (camps is None or o.camp in camps)]
            free[(title, h)] = (sum(o.y for o in ob) / sum(o.f_tau for o in ob), len(ob))

    # 2027
    out27 = study_2027(folds, reg)

    # sorties
    table = pl.DataFrame([{k: v for k, v in r.items() if k not in ("traj", "_obs", "followers")}
                          | {"followers": ",".join(r["followers"])} for r in rows])
    table.write_parquet(out / "couples.parquet")
    pl.DataFrame(pairs).write_parquet(out / "pairs.parquet")
    pl.DataFrame([o.to_dict() | {"followers": ",".join(o.followers), "shares": None, "votes": None}
                  for o in obs]).drop("shares", "votes").write_parquet(out / "observations.parquet")
    plot_trajectories(rows, out / "trajectories.png")
    plot_scatter(rows, out / "scatter.png")
    plot_placebo(perm, out / "placebo.png")
    plot_phi(obs, folds, out / "transfer_by_horizon.png")
    results = {
        "criteria": crit, "placebo": {k: v for k, v in perm.items() if not k.startswith("null")},
        "placebo_left": {k: v for k, v in perm_left.items() if not k.startswith("null")},
        "size_slope": extra["size_slope"], "explanatory_spearman": expl,
        "folds": {t: {str(k): v.to_dict() for k, v in f.items()} for t, f in folds.items()},
        "kappa_cv_sse": {str(k): v for k, v in kappa_cv.items()}, "zero_cv_sse": zero_cv,
        "free_phi": {f"{t}@{h}": v for (t, h), v in free.items()}, "y2027": out27,
        "form_cv": form_cv,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    report(out, rows, pairs, named, perm, perm_left, crit, expl, folds, kappa_cv, zero_cv, free,
           out27, extra, form_cv)


def form_cross_validation(obs: list[co.Observation]) -> list[dict]:
    """Erreur quadratique de prévision du transfert y, élection par élection laissée de côté :
    dérive nulle, φ constant (m0 = ∞), φ modulé par la proximité (m0 choisi dans le pli)."""
    rows = []
    for title, camps in (("tous camps", None), ("gauche", ("left",))):
        for hs in (co.FIT_HORIZONS, (7, 30)):
            ob = [o for o in obs if o.horizon in hs and (camps is None or o.camp in camps)]
            res = {"zéro": 0.0, "m0 = ∞": 0.0, "m0 choisi": 0.0}
            per_e = {}
            for ex in ELECTIONS:
                tr = [o for o in ob if o.election != ex]
                te = [o for o in ob if o.election == ex]
                z = sum(o.y**2 for o in te)
                a = co._sse(te, co._ls_phi0(tr, math.inf, 1.0, co.PROX_SCALE), math.inf, 1.0,
                            co.PROX_SCALE)
                m0, _ = co.select_m0(tr)
                b = co._sse(te, co._ls_phi0(tr, m0, 1.0, co.PROX_SCALE), m0, 1.0, co.PROX_SCALE)
                res["zéro"] += z
                res["m0 = ∞"] += a
                res["m0 choisi"] += b
                per_e[ex] = (z, a, b)
            rows.append({"camps": title, "horizons": list(hs), "sse": res, "per_election": per_e})
    return rows


def study_2027(folds: dict, reg) -> dict:
    """Dérive impliquée par la forme (tous camps, toutes élections) sur la moyenne actuelle des
    sondages 2027, si les sondages restaient où ils sont, à 90, 30 et 7 jours."""
    from vvs.eval.baselines import poll_mean
    from vvs.model.data import load_polls

    polls = load_polls(2027, ASOF_2027)
    pf = poll_mean(2027, ASOF_2027, CONFIG_2027, days=45, registry=reg, polls=polls)
    p = pf.shares
    blocs = [co._bloc(reg, c, 2027) for c in CONFIG_2027]
    out = {"asof": ASOF_2027.isoformat(), "n_polls": pf.n_polls, "window": pf.window_days,
           "shares": {c: float(100 * s) for c, s in zip(CONFIG_2027, p)}, "rows": []}
    for title in ("tous camps", "gauche"):
        pr = folds[title][None]
        camps = None if title == "tous camps" else ("left",)
        for tau in (180, 90, 30, 7):
            det: list = []
            rng = np.random.default_rng(0)
            phi0 = pr.phi0_draws[rng.integers(0, len(pr.phi0_draws), 4000)]
            mu = co.consolidation_drift(p, blocs, tau, phi0, pr.m0, pr.kappa, pr.scale, camps, det)
            after = co.apply_drift(p, mu)
            q = np.quantile(after, [0.05, 0.5, 0.95], axis=0)
            for d in det:
                d["leader"] = CONFIG_2027[d["leader_index"]]
            out["rows"].append({"variant": title, "tau": tau, "detail": det,
                                "median": {c: float(100 * q[1, k]) for k, c in enumerate(CONFIG_2027)},
                                "q05": {c: float(100 * q[0, k]) for k, c in enumerate(CONFIG_2027)},
                                "q95": {c: float(100 * q[2, k]) for k, c in enumerate(CONFIG_2027)}})
    return out


def report(out, rows, pairs, named, perm, perm_left, crit, expl, folds, kappa_cv, zero_cv, free,
           y27, extra, form_cv) -> None:
    md = [(f"# Vote utile de fin de campagne : consolidation d'un camp vers son leader "
           f"({dt.date.today().isoformat()})"), ""]
    md.append("Script : `back/studies/consolidation.py` ; forme : `vvs.model.consolidation`. "
              "Camps : gauche radicale + gauche, centre, droite, extrême droite. Parts des "
              "sondages : moyenne des 14 jours avant la date, candidats officiels (non-candidats "
              "ramenés comme au backtest). y : gain de la part du leader dans son camp ; "
              "φ = y / f ; d : écart des erreurs finales en log, leader contre concurrents "
              "groupés ; m : marge de qualification du leader.\n")
    md.append("## 1. Couples camp × élection (erreur finale : 14 derniers jours → vote)\n")
    md.append("| élection | camp | leader | concurrents | leader sond. → vote | concurrents sond. → vote "
              "| y final | φ final | d (log) | d résidu taille | s_L 30 j → 7 j | y 30→7 (sondages) "
              "| y 7 j→14 derniers | m | leader − 1er conc. | camp |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["camp"], r["election"])):
        fol = ", ".join(name(c) for c in r["followers"][:4]) + ("…" if len(r["followers"]) > 4 else "")
        md.append(
            f"| {r['election']} | {CAMP_LABEL[r['camp']]} | {name(r['leader'])} | {fol} "
            f"| {r['leader_poll']:.1f} → {r['leader_vote']:.1f} "
            f"| {r['followers_poll']:.1f} → {r['followers_vote']:.1f} "
            f"| {fmt(r['y_final'], 3, True)} | {fmt(r['phi_final'], 2, True)} "
            f"| {fmt(r['d_final'], 2, True)} | {fmt(r['d_size_resid'], 2, True)} "
            f"| {fmt(r['s30'], 2)} → {fmt(r['s7'], 2)} | {fmt(r['y_poll_30_7'], 3, True)} "
            f"| {fmt(r['y_7_0'], 3, True)} | {r['margin']:+.1f} | {r['gap_first']:.1f} "
            f"| {r['camp_poll']:.1f} |")
    md.append("")
    md.append("Note : « s_L 30 j → 7 j » suit le leader de l'erreur finale ; à 30 jours il "
              "n'était pas toujours en tête du camp (colonne leader_30 de `couples.parquet`).\n")
    md.append("## 2. Couples nommés (leader contre un concurrent, points)\n")
    md.append("| élection | leader | concurrent | leader sond. → vote | concurrent sond. → vote "
              "| d (log, final) | mouvement 30 → 7 j du rapport (log) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in named:
        md.append(f"| {r['election']} | {name(r['leader'])} | {name(r['follower'])} "
                  f"| {r['leader_poll']:.1f} → {r['leader_vote']:.1f} "
                  f"| {r['follower_poll']:.1f} → {r['follower_vote']:.1f} "
                  f"| {r['d_log']:+.2f} | {fmt(r['poll_move_30_7'], 2, True)} |")
    md.append("")
    md.append("Fillon / Dupont-Aignan 2017 est hors camp (droite / extrême droite) : donné pour "
              "mémoire.\n")
    md.append("## 3. Le leader gagne-t-il ? Critères fixés d'avance\n")
    md.append(f"- (a) y final > 0 : **{crit['a_n_positive']} couples sur {crit['a_n_couples']}** "
              f"(seuil 8) → {'rempli' if crit['a_ok'] else 'non rempli'} ;")
    md.append(f"- (b) placebo de permutation (leader tiré au hasard dans le camp, part ≥ 1 pt) : "
              f"moyenne observée de y {perm['obs_y']:+.3f}, 95e centile placebo "
              f"{perm['q95_y']:+.3f}, **p = {perm['p_y']:.3f}** ; sur d : {perm['obs_d']:+.3f}, "
              f"p = {perm['p_d']:.3f} → {'rempli' if crit['b_ok'] else 'non rempli'} ;")
    md.append("- (c) moyenne de y sans chaque élection : "
              + ", ".join(f"{e} {v:+.3f}" for e, v in crit["c_loo_means"].items())
              + f" → {'rempli' if crit['c_ok'] else 'non rempli'}.")
    md.append(f"\nConsolidation générale au sens des critères : **{'oui' if crit['general'] else 'non'}**.\n")
    md.append(f"Camp de gauche seul (exploratoire, choisi après lecture) : moyenne de y "
              f"{perm_left['obs_y']:+.3f}, p = {perm_left['p_y']:.3f} (d : p = "
              f"{perm_left['p_d']:.3f}), {sum(r['y_final'] > 0 for r in rows if r['camp'] == 'left')} "
              f"couples positifs sur {sum(r['camp'] == 'left' for r in rows)}.\n")
    md.append(f"Placebo de taille : pente des erreurs finales en log sur la log-part, effets "
              f"fixes d'élection : {extra['size_slope']:+.3f} (une pente positive voudrait dire que les "
              f"gros candidats gagnent sur les petits en général, quel que soit le camp). Colonne "
              f"« d résidu taille » du tableau 1 : d recalculé sur les résidus.\n")
    md.append("## 4. Avant ou après les derniers sondages ?\n")
    vis = [r["y_poll_30_7"] for r in rows if r["y_poll_30_7"] is not None]
    md.append(f"Gain du leader (part dans le camp) déjà visible dans les sondages entre 30 et "
              f"7 jours : moyenne {np.mean(vis):+.3f} ({sum(v > 0 for v in vis)} / {len(vis)} "
              f"positifs) ; entre la fenêtre de 7 jours et les 14 derniers jours : "
              f"{np.mean([r['y_7_0'] for r in rows if r['y_7_0'] is not None]):+.3f} ; après les "
              f"derniers sondages (erreur finale) : {np.mean([r['y_final'] for r in rows]):+.3f}. "
              "Figure `trajectories.png`.\n")
    md.append("## 5. Variables explicatives (corrélation de rang de Spearman, n ≈ 11)\n")
    md.append("| variable | y final | φ final | y à 30 j |")
    md.append("|---|---|---|---|")
    lab = {"margin": "marge m", "abs_margin": "|m|", "gap_first": "leader − 1er concurrent",
           "camp_poll": "taille du camp", "f_final": "part des concurrents f", "n": "nombre de candidats",
           "certain_gap": "certitude leader − concurrents (2017, 2022)"}
    for v, d in expl.items():
        md.append(f"| {lab[v]} | {fmt(d['y_final'], 2, True)} | {fmt(d['phi_final'], 2, True)} "
                  f"| {fmt(d['y_30'], 2, True)} |")
    md.append("\nAvec 11 couples, |ρ| doit dépasser environ 0,6 pour être distinguable de zéro "
              "au seuil de 5 %. Figure `scatter.png`.\n")
    md.append("## 6. Forme estimée, par pli (leave-one-election-out)\n")
    md.append("φ(τ, m) = φ0 · (1 − τ/180)^κ · g(m), g(0) = 1 ; φ0 par moindres carrés sur y aux "
              "horizons 7, 30, 90 jours ; m0 choisi par validation interne ; intervalle : "
              "rééchantillonnage des couples. φ à |m| = 5 pts et τ = 7 j donné pour lecture.\n")
    md.append("| variante | élection laissée | φ0 [5-95 %] | m0 (pts) | φ(τ = 7, |m| = 5) | φ(τ = 30, |m| = 5) | couples |")
    md.append("|---|---|---|---|---|---|---|")
    for title, f in folds.items():
        for ex, pr in f.items():
            lo, _, hi = pr.interval()
            m0 = "∞" if math.isinf(pr.m0) else f"{pr.m0:.0f}"
            p7 = float(co.consolidation_phi(pr.phi0, 7, 5.0, pr.m0))
            p30 = float(co.consolidation_phi(pr.phi0, 30, 5.0, pr.m0))
            md.append(f"| {title} | {ex or 'aucune'} | {pr.phi0:.2f} [{lo:.2f}, {hi:.2f}] | {m0} "
                      f"| {p7:.2f} | {p30:.2f} | {pr.n_couples} |")
    md.append("")
    md.append("Sensibilité à κ (somme des erreurs quadratiques de prévision de y, validation par "
              "élection, m0 choisi dans chaque pli ; sans dérive : "
              f"{zero_cv:.4f}) : " + ", ".join(f"κ = {k} : {v:.4f}" for k, v in kappa_cv.items()) + ".\n")
    md.append("Validation croisée de la forme (somme des erreurs quadratiques de prévision de y "
              "sur l'élection laissée ; plus bas = mieux) :\n")
    md.append("| camps | horizons | dérive nulle | φ constant (m0 = ∞) | φ · proximité (m0 choisi) "
              "| par élection laissée (nulle / constant / proximité) |")
    md.append("|---|---|---|---|---|---|")
    for r in form_cv:
        pe = " ; ".join(f"{e} {z:.3f} / {a:.3f} / {b:.3f}" for e, (z, a, b) in r["per_election"].items())
        md.append(f"| {r['camps']} | {', '.join(map(str, r['horizons']))} | {r['sse']['zéro']:.4f} "
                  f"| {r['sse']['m0 = ∞']:.4f} | {r['sse']['m0 choisi']:.4f} | {pe} |")
    md.append("")
    md.append("φ libre par horizon (Σ y / Σ f, sans forme) :\n")
    md.append("| horizon | tous camps | gauche |")
    md.append("|---|---|---|")
    for h in co.STUDY_HORIZONS:
        a, na = free[("tous camps", h)]
        b, nb = free[("gauche", h)]
        md.append(f"| {h} | {a:+.3f} (n = {na}) | {b:+.3f} (n = {nb}) |")
    md.append("\nFigure `transfer_by_horizon.png`.\n")
    md.append("## 7. Ce que la forme impliquerait pour 2027\n")
    md.append(f"Moyenne des sondages au {y27['asof']} ({y27['n_polls']} sondages, fenêtre "
              f"{y27['window']} j), hypothèse Le Pen, Zemmour, Dupont-Aignan, Retailleau, "
              f"Philippe, Attal, Glucksmann, Tondelier, Mélenchon, Roussel, Arthaud. Si les "
              "sondages restaient où ils sont, la dérive serait (paramètres sur 2002-2022, "
              "médiane des parts après dérive, sans η) :\n")
    show = ["le_pen_marine", "zemmour_eric", "philippe_edouard", "attal_gabriel",
            "melenchon_jean_luc", "glucksmann_raphael", "tondelier_marine", "roussel_fabien",
            "retailleau_bruno"]
    md.append("| variante | τ | " + " | ".join(name(c) for c in show) + " |")
    md.append("|---|---|" + "---|" * len(show))
    md.append("| sondages | – | " + " | ".join(f"{y27['shares'][c]:.1f}" for c in show) + " |")
    for r in y27["rows"]:
        md.append(f"| {r['variant']} | {r['tau']} | "
                  + " | ".join(f"{r['median'][c]:.1f}" for c in show) + " |")
    md.append("")
    md.append("| variante | τ | camp | leader | marge m | φ médian [5-95 %] | transfert médian (pts) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in y27["rows"]:
        for d in r["detail"]:
            md.append(f"| {r['variant']} | {r['tau']} | {CAMP_LABEL[d['camp']]} | {name(d['leader'])} "
                      f"| {d['margin_pts']:+.1f} | {d['phi_median']:.2f} [{d['phi_q05']:.2f}, "
                      f"{d['phi_q95']:.2f}] | {d['transfer_pts_median']:.1f} |")
    md.append("")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# --- backtest des variantes `consolidation`


BT_ELECTIONS = (2002, 2007, 2012, 2017, 2022)
BT_HORIZONS = (180, 90, 30, 7)
BT_VARIANTS = ("floor", "consolidation", "consolidation_left")
BT_METRICS = ("log_score_pair", "crps", "cov50", "cov90", "bloc_crps", "brier_qual")
WEIGHT = {2002: 0.5, 2007: 0.5}


def backtest(out: Path, fits_from: Path, n_draws: int = 10_000) -> None:
    """Score floor et consolidation sur les mêmes tirages θ (liens vers `fits_from`), même η
    (même graine), tableau d'admission (docs/07-evaluation.md)."""
    from vvs.eval.backtest import fit_dir
    from vvs.eval.configurations import official_map
    from vvs.eval.scoring import score_backtest
    from vvs.model.run import draws_log_shares

    out.mkdir(parents=True, exist_ok=True)
    logf = out / "backtest.log"

    def say(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        with logf.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    for v in BT_VARIANTS:
        for e in BT_ELECTIONS:
            for h in BT_HORIZONS:
                src, dst = fit_dir(fits_from, "floor", e, h), fit_dir(out, v, e, h)
                if not (src / "draws.npz").exists():
                    raise FileNotFoundError(f"ajustement floor absent : {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src.resolve())
    say(f"score {', '.join(BT_VARIANTS)}, tirages de {fits_from}")
    res = score_backtest(out, list(BT_ELECTIONS), list(BT_HORIZONS), variants=list(BT_VARIANTS),
                         n_draws=n_draws)
    say("score terminé")

    # dérives appliquées (médianes, sur la référence du postérieur)
    reg = default_registry()
    drows = []
    for v in BT_VARIANTS[1:]:
        spec = co.CONSOLIDATION_VARIANTS[v]
        for e in BT_ELECTIONS:
            pr = co.params_for(e, spec, reg)
            for h in BT_HORIZONS:
                fd = fit_dir(out, v, e, h)
                conf = json.loads((fd / "configuration.json").read_text())
                dr = np.load(fd / "draws.npz")
                K = official_map(conf["candidates"], e, reg).config
                ref = co.reference_shares(draws_log_shares(dr, K, "asof"))
                blocs = [co._bloc(reg, c, e) for c in K]
                rng = np.random.default_rng([0, e, h, 11])
                phi0 = pr.phi0_draws[rng.integers(0, len(pr.phi0_draws), 4000)]
                det: list = []
                co.consolidation_drift(ref, blocs, h, phi0, pr.m0, pr.kappa, pr.scale, spec.camps, det)
                for d in det:
                    drows.append({"variant": v, "election": e, "horizon": h,
                                  "leader": K[d["leader_index"]], "phi0": pr.phi0,
                                  "m0": None if math.isinf(pr.m0) else pr.m0} | d)
    dtab = pl.DataFrame(drows, infer_schema_length=None)
    dtab.write_parquet(out / "drifts.parquet")

    s = res.metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model")
                           & pl.col("metric").is_in(BT_METRICS))
    wide = s.pivot(on="variant", index=["election", "horizon", "metric"], values="value")
    for v in BT_VARIANTS[1:]:
        wide = wide.with_columns((pl.col(v) - pl.col("floor")).alias(f"diff_{v}"))
    wide.write_parquet(out / "comparison.parquet")
    fc = res.forecasts.filter((pl.col("model") == "model") & (pl.col("level") == "candidate"))
    fc.write_parquet(out / "candidate_forecasts.parquet")

    md = [f"# Backtest des variantes de consolidation ({dt.date.today().isoformat()})", "",
          (f"Tirages θ : {fits_from} (variante floor, réutilisés tels quels). Même η "
           "(graine [0, élection, horizon]) ; φ0 tiré avec une graine propre. Paramètres "
           "estimés sans l'élection évaluée. `consolidation` : tous camps (variante de la règle "
           "d'admission) ; `consolidation_left` : camp de gauche seul, exploratoire. Moyennes : "
           "pondérée (2002 et 2007 à 0,5) et sur 2012-2022 seules."), ""]
    md.append("## Dérives appliquées (médiane sur la référence du postérieur)\n")
    md.append("| variante | élection | horizon | camp | leader | m (pts) | φ médian [5-95 %] | transfert médian (pts) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for d in drows:
        if d["horizon"] == 180:
            continue
        md.append(f"| {d['variant']} | {d['election']} | {d['horizon']} | {CAMP_LABEL[d['camp']]} "
                  f"| {name(d['leader'])} | {d['margin_pts']:+.1f} | {d['phi_median']:.2f} "
                  f"[{d['phi_q05']:.2f}, {d['phi_q95']:.2f}] | {d['transfer_pts_median']:.1f} |")
    md.append("\nÀ 180 jours, w(180) = 0 : dérive nulle, scores identiques à floor par construction.\n")
    for m in BT_METRICS:
        md.append(f"### {m}\n")
        md.append("| élection | horizon | floor | consolidation | écart | gauche seule | écart |")
        md.append("|---|---|---|---|---|---|---|")
        sub = wide.filter(pl.col("metric") == m).sort("election", "horizon", descending=[False, True])
        for r in sub.iter_rows(named=True):
            md.append(f"| {r['election']} | {r['horizon']} | {r['floor']:.3f} | {r['consolidation']:.3f} "
                      f"| {r['diff_consolidation']:+.3f} | {r['consolidation_left']:.3f} "
                      f"| {r['diff_consolidation_left']:+.3f} |")
        for lab, flt, wexpr in (
            ("moyenne pondérée", pl.lit(True),
             pl.col("election").replace_strict(WEIGHT, default=1.0, return_dtype=pl.Float64)),
            ("moyenne 2012-2022", pl.col("election") >= 2012, pl.lit(1.0)),
        ):
            mean = (sub.filter(flt).with_columns(wexpr.alias("w")).group_by("horizon")
                    .agg(*[((pl.col(c) * pl.col("w")).sum() / pl.col("w").sum()).alias(c)
                           for c in ("floor", "consolidation", "diff_consolidation",
                                     "consolidation_left", "diff_consolidation_left")])
                    .sort("horizon", descending=True))
            for r in mean.iter_rows(named=True):
                md.append(f"| **{lab}** | {r['horizon']} | {r['floor']:.3f} | {r['consolidation']:.3f} "
                          f"| {r['diff_consolidation']:+.3f} | {r['consolidation_left']:.3f} "
                          f"| {r['diff_consolidation_left']:+.3f} |")
        md.append("")
    # prévisions des candidats clés
    keys = {2002: ["jospin_lionel", "le_pen_jean_marie", "chevenement_jean_pierre"],
            2007: ["royal_segolene", "le_pen_jean_marie", "bayrou_francois"],
            2012: ["hollande_francois", "melenchon_jean_luc", "le_pen_marine"],
            2017: ["melenchon_jean_luc", "hamon_benoit", "fillon_francois", "le_pen_marine"],
            2022: ["melenchon_jean_luc", "jadot_yannick", "le_pen_marine", "zemmour_eric"]}
    md.append("## Candidats clés à 30 et 7 jours (médiane [5-95 %], points)\n")
    md.append("| élection | horizon | candidat | résultat | floor | consolidation | gauche seule |")
    md.append("|---|---|---|---|---|---|---|")
    for e, cs in keys.items():
        for h in (30, 7):
            for c in cs:
                cells = []
                off = None
                for v in BT_VARIANTS:
                    r = fc.filter((pl.col("election") == e) & (pl.col("horizon") == h)
                                  & (pl.col("variant") == v) & (pl.col("unit") == c))
                    if r.is_empty():
                        cells.append("–")
                        continue
                    r = r.row(0, named=True)
                    off = r["official"]
                    cells.append(f"{r['median']:.1f} [{r['q05']:.1f}, {r['q95']:.1f}]")
                md.append(f"| {e} | {h} | {name(c)} | {fmt(off, 1)} | " + " | ".join(cells) + " |")
    md.append("")
    # règle d'admission
    adm = {}
    for v in BT_VARIANTS[1:]:
        rows_ = []
        for h in BT_HORIZONS:
            g = wide.filter((pl.col("horizon") == h) & (pl.col("election") >= 2012))
            def dm(metric, g=g, v=v):
                return g.filter(pl.col("metric") == metric)[f"diff_{v}"]
            ls, cr = dm("log_score_pair"), dm("crps")
            rows_.append({
                "horizon": h, "ls_gain": float(ls.mean()), "crps_gain": float(-cr.mean()),
                "n_ls": int((ls > 1e-9).sum()), "n_crps": int((cr < -1e-9).sum()),
                "cov90": float(dm("cov90").mean()), "cov50": float(dm("cov50").mean()),
            })
        ok = all(r["ls_gain"] > 0 and r["crps_gain"] > 0 and r["n_ls"] >= 2 and r["n_crps"] >= 2
                 and r["cov90"] >= -0.02 for r in rows_ if r["horizon"] in (180, 90))
        adm[v] = {"rows": rows_, "criteria_1_3_at_180_90": ok}
        md.append(f"## Règle d'admission (docs/07), variante `{v}`, 2012-2022\n")
        md.append("| horizon | gain log-score moyen | gain CRPS moyen | élections où le log-score gagne "
                  "| où le CRPS gagne | Δ couverture 90 % | Δ couverture 50 % |")
        md.append("|---|---|---|---|---|---|---|")
        for r in rows_:
            md.append(f"| {r['horizon']} | {r['ls_gain']:+.3f} | {r['crps_gain']:+.3f} | {r['n_ls']} / 3 "
                      f"| {r['n_crps']} / 3 | {100 * r['cov90']:+.1f} pts | {100 * r['cov50']:+.1f} pts |")
        md.append(f"\nCritères 1 à 3 remplis à 180 et 90 jours : **{'oui' if ok else 'non'}**.\n")
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    (out / "admission.json").write_text(json.dumps(adm, indent=2))
    say("comparaison écrite")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=20_000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--backtest", action="store_true",
                    help="score floor contre consolidation sur les ajustements floor existants")
    ap.add_argument("--fits-from", type=Path, default=paths.RUNS / "2026-10-20-backtest")
    a = ap.parse_args()
    today = dt.date.today().isoformat()
    if a.backtest:
        backtest(a.out or paths.RUNS / f"{today}-backtest-consolidation", a.fits_from)
    else:
        main(a.out or paths.RUNS / f"{today}-consolidation", a.n_perm, a.n_boot)
