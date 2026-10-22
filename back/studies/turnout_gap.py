"""Étude : l'abstention différentielle (écart certains d'aller voter − inscrits) par candidat.

    uv run python studies/turnout_gap.py [--out runs/2026-10-22-turnout-gap] [--perm 5000]

Données : `data/curated/turnout_gap.parquet` (`vvs.curate.turnout_gap`, reconstruit ici),
lui-même tiré des notices de la Commission (`vvs parse commission-archive --election …`) :
tableaux OpinionWay publiant les deux bases avec le même redressement (PrésiTrack 2017,
baromètres et rolling 2022, 2027). Sondages des autres instituts (une seule base) :
`poll_results_archive.parquet`. Agrégat tous instituts : sondages curated (`load_polls`,
`poll_mean`) et `poll_errors.parquet`. Résultats et participation : `elections.yaml`.

Notations : g_c = part sur certains − part sur inscrits (points d'exprimés), moyenne sur les
hypothèses d'un sondage puis sur les sondages ; e_c = résultat − agrégat final (points).

Protocole et critères, fixés avant de calculer les tests (les moyennes brutes de g par
candidat ont été vues pendant la validation du parseur, pas les statistiques ci-dessous) :

(a) Stabilité. Par élection, sondages OpinionWay du premier tour coupés en deux moitiés de
    dates ; candidats présents dans au moins 5 sondages de chaque moitié ; r de Pearson entre
    les g_c des deux moitiés. Placebo : permutation des candidats de la seconde moitié
    (p unilatéral, r ≥ r observé). Critère : r ≥ 0,5 et p ≤ 0,05 en 2017 et en 2022, sinon
    l'écart est traité comme du bruit. Entre instituts (pas d'autre institut à deux bases) :
    écart publié certains − inscrits entre instituts de bases différentes sur les mêmes
    quinzaines (2022 : Harris, inscrits, contre Ifop, Elabe, Ipsos, OpinionWay, certains ;
    2017 : Ifop jusqu'au 29 mars, inscrits, contre Elabe, Kantar, OpinionWay, certains),
    corrélé aux g_c d'OpinionWay ; confondu avec les effets maison, lu comme indication
    seulement (critère : r > 0 avec p ≤ 0,10).
(b) Erreur finale. b1 : quelle base est la plus proche du résultat ? Moyenne OpinionWay
    sur chaque base, sondages terminés dans ]vote − h − 7, vote − h + 7] (h = 7 : ]vote − 14,
    vote − 1]) pour h ∈ {7, 30, 60, 90}, hypothèse la plus proche des candidats officiels ;
    erreur absolue moyenne sur les candidats et sur les blocs ; même comparaison au second tour
    (duel réel, 7 derniers jours). Lecture : la base de plus faible erreur à 7 jours dans les
    deux élections. b2 : g_c prédit-il e_c ? g_c moyen sur les 60 jours avant le vote, e_c
    contre la base certains d'OpinionWay et contre la moyenne tous instituts des 14 derniers
    jours ; pente β de e sur g par élection et poolée ; placebo : permutation des g entre
    candidats dans chaque élection (p bilatéral). Critère : β de même signe dans les deux
    élections et p poolé ≤ 0,05. Validation croisée : λ estimé sur une élection, appliqué à
    l'autre (part = certains + λ g), gain d'erreur absolue.
(c) Mouvement ultérieur. Points t toutes les deux semaines de vote − 180 à vote − 28 jours ;
    g_c(t) moyen des sondages OpinionWay terminés dans ]t − 14, t] ; mouvement de l'agrégat
    tous instituts m_c = A_c(t + 28) − A_c(t) (moyenne 14 jours, `poll_mean`) ; candidats
    officiels à au moins 1 % ; g et m centrés par date. Placebo : une même permutation des
    candidats appliquée à toutes les dates d'une élection (les dates partagent les candidats).
    Critère : corrélation de même signe en 2017 et 2022 et p poolé ≤ 0,05.
(d) Participation. Part de certains d'aller voter publiée (`poll_turnout_intent`) dans les
    30 jours avant chaque tour, par institut, contre la participation réelle (votants /
    inscrits). Critère d'usage comme prédicteur : pour chaque institut présent les deux années
    avec la même définition, décalage participation − part de certains stable à 2 points près
    entre 2017 et 2022, et variation de la part de certains de même signe que celle de la
    participation. Scrutins de liste : pas de notices ingérées, non testé.
(e) 2027 : g_c actuels par candidat et par bloc (OpinionWay), comparés aux g par bloc de 2017
    et 2022.
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
from vvs.curate import turnout_gap as tg
from vvs.eval.baselines import poll_mean
from vvs.eval.configurations import closest_hypothesis
from vvs.model.data import load_polls
from vvs.registry import default_registry

ELECTIONS = (2017, 2022)
HORIZONS = (7, 30, 60, 90)
BLOCS = ("far_left", "left", "centre", "right", "far_right", "other")
REG = default_registry()


# ------------------------------------------------------------------------- données


def official(election: int, rnd: int = 1) -> list[str]:
    return [r.candidate_id for r in REG.elections[election].round_(rnd).results]


def result_shares(election: int, rnd: int = 1) -> dict[str, float]:
    return {r.candidate_id: r.share for r in REG.elections[election].round_(rnd).results}


def turnout(election: int, rnd: int) -> float:
    r = REG.elections[election].round_(rnd)
    return 100 * r.voters / r.registered


def vote_date(election: int, rnd: int = 1) -> dt.date:
    return REG.elections[election].round_(rnd).date


def bloc_of(cid: str, election: int) -> str:
    try:
        return REG.bloc_of(cid, election)
    except KeyError:
        return "other"


def poll_gaps(gap: pl.DataFrame, rnd: int = 1) -> pl.DataFrame:
    """Un g par (sondage, candidat) : moyenne sur les hypothèses du sondage."""
    return (
        gap.filter(pl.col("round") == rnd)
        .group_by("poll_id", "election", "pollster", "fieldwork_end", "candidate_id")
        .agg(
            pl.col("gap").mean(),
            pl.col("share_certain").mean(),
            pl.col("share_registered").mean(),
            pl.col("certain_share_sample").first(),
        )
    )


def closest_by_base(gap: pl.DataFrame, election: int, rnd: int, config: list[str]) -> pl.DataFrame:
    """Une hypothèse par sondage (la plus proche de `config`), parts sur les deux bases."""
    sub = gap.filter((pl.col("election") == election) & (pl.col("round") == rnd))
    rows = []
    for (pid,), g in sub.group_by(["poll_id"]):
        hyps = {
            h: dict(zip(gg["candidate_id"], gg["share_certain"], strict=True))
            for (h,), gg in g.group_by(["hypothesis_id"])
        }
        hid, dist = closest_hypothesis(hyps, config)
        rows.append(
            g.filter(pl.col("hypothesis_id") == hid).with_columns(pl.lit(dist).alias("distance"))
        )
    return pl.concat(rows) if rows else sub.clear()


def mean_by_base(sub: pl.DataFrame, config: list[str]) -> dict[str, dict[str, float]]:
    """Moyenne des parts renormalisées sur les candidats de `config` présents, par base."""
    out = {}
    for base, col in (("certain", "share_certain"), ("registered", "share_registered")):
        acc: dict[str, list[float]] = {}
        for (_,), g in sub.group_by(["poll_id"]):
            s = {c: v for c, v in zip(g["candidate_id"], g[col], strict=True) if c in config}
            tot = sum(s.values())
            for c, v in s.items():
                acc.setdefault(c, []).append(100 * v / tot)
        out[base] = {c: float(np.mean(v)) for c, v in acc.items()}
    return out


def by_bloc(shares: dict[str, float], election: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for c, v in shares.items():
        b = bloc_of(c, election)
        out[b] = out.get(b, 0.0) + v
    return out


def perm_pvalue(stat_fn, x_groups, y_groups, rng, n_perm: int, two_sided: bool) -> float:
    """Permutation des x entre candidats dans chaque groupe (élection) ; stat_fn(xs, ys)."""
    obs = stat_fn(x_groups, y_groups)
    hits = 0
    for _ in range(n_perm):
        xp = [rng.permutation(x) for x in x_groups]
        s = stat_fn(xp, y_groups)
        hits += abs(s) >= abs(obs) - 1e-12 if two_sided else s >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def pearson_pooled(xs, ys) -> float:
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def slope_pooled(xs, ys) -> float:
    """Pente de y sur x, x et y centrés dans chaque groupe."""
    x = np.concatenate([a - a.mean() for a in xs])
    y = np.concatenate([b - b.mean() for b in ys])
    return float((x @ y) / (x @ x)) if x @ x > 0 else float("nan")


def fmt(v, nd=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.{nd}f}" if isinstance(v, float) else str(v)


def md_table(df: pl.DataFrame, nd: int = 2) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("–" if np.isnan(v) else f"{v:.{nd}f}")
            else:
                cells.append("–" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- (a) stabilité


def stability(pg: pl.DataFrame, election: int, rng, n_perm: int) -> dict:
    sub = pg.filter(pl.col("election") == election).sort("fieldwork_end")
    dates = sorted(sub["fieldwork_end"].unique().to_list())
    if len(dates) < 4:
        return {"election": election, "n_candidates": 0}
    cut = dates[len(dates) // 2]
    h1 = sub.filter(pl.col("fieldwork_end") < cut)
    h2 = sub.filter(pl.col("fieldwork_end") >= cut)
    m1 = h1.group_by("candidate_id").agg(pl.col("gap").mean().alias("g1"), pl.len().alias("n1"))
    m2 = h2.group_by("candidate_id").agg(pl.col("gap").mean().alias("g2"), pl.len().alias("n2"))
    m = (
        m1.join(m2, on="candidate_id")
        .filter((pl.col("n1") >= 5) & (pl.col("n2") >= 5))
        .sort("candidate_id")
    )
    x, y = m["g1"].to_numpy(), m["g2"].to_numpy()
    r = pearson_pooled([x], [y])
    p = perm_pvalue(pearson_pooled, [x], [y], rng, n_perm, two_sided=False)
    # bruit sondage à sondage : écart-type des g d'un candidat, rapporté à la dispersion entre
    # candidats des moyennes
    within = sub.group_by("candidate_id").agg(pl.col("gap").std().alias("sd"), pl.len().alias("n"))
    big = m.filter((pl.col("g1").abs() >= 0.5) | (pl.col("g2").abs() >= 0.5))
    same_sign = (
        float((np.sign(big["g1"]) == np.sign(big["g2"])).mean()) if big.height else float("nan")
    )
    return {
        "election": election,
        "cut": str(cut),
        "n_candidates": m.height,
        "r": r,
        "p": p,
        "same_sign_share": same_sign,
        "median_within_sd": float(within["sd"].median()),
        "table": m.with_columns(pl.lit(election).alias("election")),
        "passes": bool(r >= 0.5 and p <= 0.05),
    }


def published_by_base(election: int, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Parts publiées (une base par institut) des sondages archivés, R1, hypothèse la plus
    proche des candidats officiels, par quinzaine."""
    polls = pl.read_parquet(paths.CURATED / "polls_archive.parquet").filter(
        pl.col("election") == election
    )
    res = pl.read_parquet(paths.CURATED / "poll_results_archive.parquet").filter(
        pl.col("round") == 1
    )
    df = res.join(polls.select("poll_id", "pollster", "fieldwork_end"), on="poll_id").filter(
        pl.col("fieldwork_end").is_between(start, end)
    )
    config = official(election)
    rows = []
    for (pid, pop), g in df.group_by(["poll_id", "population"]):
        hyps = {
            h: dict(zip(gg["candidate_id"], gg["share"], strict=True))
            for (h,), gg in g.group_by(["hypothesis_id"])
        }
        hid, _ = closest_hypothesis(hyps, config)
        s = {c: v for c, v in hyps[hid].items() if c in config}
        tot = sum(s.values())
        for c, v in s.items():
            rows.append(
                {
                    "poll_id": pid,
                    "pollster": g["pollster"][0],
                    "population": pop,
                    "fieldwork_end": g["fieldwork_end"][0],
                    "candidate_id": c,
                    "share": 100 * v / tot,
                }
            )
    out = pl.DataFrame(rows)
    return out.with_columns(
        ((pl.col("fieldwork_end") - pl.lit(start)).dt.total_days() // 14).alias("fortnight")
    )


def inter_institute(pg: pl.DataFrame, election: int, rng, n_perm: int) -> dict:
    """Écart publié certains − inscrits entre instituts de bases différentes, par quinzaine,
    contre le g_c d'OpinionWay sur la même période."""
    if election == 2022:
        reg_set, start, end = {"harris"}, dt.date(2021, 9, 1), dt.date(2022, 4, 9)
    else:
        reg_set, start, end = {"ifop"}, dt.date(2017, 1, 15), dt.date(2017, 3, 29)
    pub = published_by_base(election, start, end)
    # OpinionWay entre dans le côté « certains » par sa base publiée
    reg_side = pub.filter(
        pl.col("pollster").is_in(list(reg_set)) & (pl.col("population") == "registered")
    )
    cert_side = pub.filter(
        ~pl.col("pollster").is_in(list(reg_set)) & (pl.col("population") == "certain_voters")
    )
    a = reg_side.group_by("fortnight", "candidate_id").agg(pl.col("share").mean().alias("reg"))
    b = cert_side.group_by("fortnight", "candidate_id").agg(pl.col("share").mean().alias("cert"))
    d = (
        a.join(b, on=["fortnight", "candidate_id"])
        .group_by("candidate_id")
        .agg(
            (pl.col("cert") - pl.col("reg")).mean().alias("published_diff"),
            pl.len().alias("fortnights"),
        )
    )
    ow = (
        pg.filter((pl.col("election") == election) & pl.col("fieldwork_end").is_between(start, end))
        .group_by("candidate_id")
        .agg(pl.col("gap").mean().alias("ow_gap"))
    )
    m = d.join(ow, on="candidate_id").filter(pl.col("fortnights") >= 2).sort("candidate_id")
    if m.height < 4:
        return {"election": election, "n_candidates": m.height, "table": m}
    x, y = m["ow_gap"].to_numpy(), m["published_diff"].to_numpy()
    return {
        "election": election,
        "registered_side": sorted(reg_set),
        "n_candidates": m.height,
        "r": pearson_pooled([x], [y]),
        "p": perm_pvalue(pearson_pooled, [x], [y], rng, n_perm, two_sided=False),
        "table": m.with_columns(pl.lit(election).alias("election")),
    }


# -------------------------------------------------------------------- (b) erreur finale


def base_errors(gap: pl.DataFrame, election: int) -> list[dict]:
    rows = []
    config = official(election)
    res = result_shares(election)
    vote = vote_date(election)
    best = closest_by_base(gap, election, 1, config)
    for h in HORIZONS:
        lo, hi = (
            (vote - dt.timedelta(days=14), vote - dt.timedelta(days=1))
            if h == 7
            else (
                vote - dt.timedelta(days=h + 7),
                vote - dt.timedelta(days=h - 7),
            )
        )
        sub = best.filter(pl.col("fieldwork_end").is_between(lo, hi, closed="right"))
        if sub["poll_id"].n_unique() == 0:
            continue
        means = mean_by_base(sub, config)
        row = {"election": election, "round": 1, "horizon": h, "polls": sub["poll_id"].n_unique()}
        for base, m in means.items():
            cs = [c for c in config if c in m]
            row[f"mae_cand_{base}"] = float(np.mean([abs(res[c] - m[c]) for c in cs]))
            bm, br = by_bloc(m, election), by_bloc({c: res[c] for c in cs}, election)
            row[f"mae_bloc_{base}"] = float(np.mean([abs(br[b] - bm[b]) for b in bm]))
        bm_c, bm_r = by_bloc(means["certain"], election), by_bloc(means["registered"], election)
        br = by_bloc({c: res[c] for c in means["certain"]}, election)
        row["blocs_certain_closer"] = sum(abs(br[b] - bm_c[b]) < abs(br[b] - bm_r[b]) for b in bm_c)
        row["blocs"] = len(bm_c)
        rows.append(row)
    # second tour : duel réel, 7 derniers jours
    fin = official(election, 2)
    vote2 = vote_date(election, 2)
    sub = gap.filter(
        (pl.col("election") == election)
        & (pl.col("round") == 2)
        & pl.col("fieldwork_end").is_between(
            vote2 - dt.timedelta(days=8), vote2 - dt.timedelta(days=1)
        )
    )
    sub = sub.filter(pl.col("hypothesis_id") == "+".join(sorted(fin)))
    if sub.height:
        means = mean_by_base(sub, fin)
        res2 = result_shares(election, 2)
        row = {"election": election, "round": 2, "horizon": 7, "polls": sub["poll_id"].n_unique()}
        for base, m in means.items():
            row[f"mae_cand_{base}"] = float(np.mean([abs(res2[c] - m[c]) for c in fin if c in m]))
        rows.append(row)
    return rows


def final_vectors(gap: pl.DataFrame, pg: pl.DataFrame, election: int) -> pl.DataFrame:
    """Par candidat officiel : g moyen sur 60 jours, parts finales OpinionWay par base (7 j),
    moyenne tous instituts 14 j (`poll_errors`), résultat."""
    config = official(election)
    res = result_shares(election)
    vote = vote_date(election)
    g60 = (
        pg.filter(
            (pl.col("election") == election)
            & pl.col("fieldwork_end").is_between(
                vote - dt.timedelta(days=60), vote - dt.timedelta(days=1)
            )
        )
        .group_by("candidate_id")
        .agg(pl.col("gap").mean().alias("gap60"), pl.len().alias("n_polls"))
    )
    best = closest_by_base(gap, election, 1, config).filter(
        pl.col("fieldwork_end").is_between(
            vote - dt.timedelta(days=14), vote - dt.timedelta(days=1), closed="right"
        )
    )
    means = mean_by_base(best, config)
    errs = pl.read_parquet(paths.CURATED / "poll_errors.parquet").filter(
        (pl.col("election") == election)
        & (pl.col("round") == 1)
        & (pl.col("method") == "mean_14d")
        & (pl.col("level") == "candidate")
    )
    allp = dict(zip(errs["unit"], errs["poll_share"], strict=True))
    rows = []
    for c in config:
        rows.append(
            {
                "election": election,
                "candidate_id": c,
                "bloc": bloc_of(c, election),
                "result": res[c],
                "ow_certain": means["certain"].get(c),
                "ow_registered": means["registered"].get(c),
                "all_pollsters": allp.get(c),
            }
        )
    out = pl.DataFrame(rows).join(g60, on="candidate_id", how="left")
    return out.with_columns(
        (pl.col("result") - pl.col("ow_certain")).alias("err_ow_certain"),
        (pl.col("result") - pl.col("ow_registered")).alias("err_ow_registered"),
        (pl.col("result") - pl.col("all_pollsters")).alias("err_all"),
    )


def gap_vs_error(fv: pl.DataFrame, target: str, rng, n_perm: int) -> dict:
    xs, ys, per = [], [], {}
    for e in ELECTIONS:
        sub = fv.filter(
            (pl.col("election") == e) & pl.col("gap60").is_not_null() & pl.col(target).is_not_null()
        )
        x, y = sub["gap60"].to_numpy(), sub[target].to_numpy()
        xs.append(x)
        ys.append(y)
        per[e] = {
            "n": len(x),
            "r": pearson_pooled([x], [y]),
            "beta": slope_pooled([x], [y]),
            "p": perm_pvalue(slope_pooled, [x], [y], rng, n_perm, two_sided=True),
        }
    pooled = {
        "beta": slope_pooled(xs, ys),
        "r": pearson_pooled([a - a.mean() for a in xs], [b - b.mean() for b in ys]),
        "p": perm_pvalue(slope_pooled, xs, ys, rng, n_perm, two_sided=True),
    }
    same_sign = np.sign(per[2017]["beta"]) == np.sign(per[2022]["beta"])
    # validation croisée : λ d'une élection appliqué à l'autre (part = certains + λ g)
    cv = {}
    for train, test in ((2017, 2022), (2022, 2017)):
        lam = per[train]["beta"]
        sub = fv.filter(
            (pl.col("election") == test)
            & pl.col("gap60").is_not_null()
            & pl.col(target).is_not_null()
        )
        base = float(np.mean(np.abs(sub[target].to_numpy())))
        adj = float(np.mean(np.abs(sub[target].to_numpy() - lam * sub["gap60"].to_numpy())))
        cv[test] = {"lambda": lam, "mae_before": base, "mae_after": adj}
    return {
        "target": target,
        "per_election": per,
        "pooled": pooled,
        "passes": bool(same_sign and pooled["p"] <= 0.05),
        "cv": cv,
    }


# -------------------------------------------------------------------- (c) mouvement


def movement(pg: pl.DataFrame, election: int) -> pl.DataFrame:
    vote = vote_date(election)
    config = official(election)
    polls = load_polls(election)
    rows = []
    for k in range(180, 27, -14):
        t = vote - dt.timedelta(days=k)
        t2 = t + dt.timedelta(days=28)
        g = (
            pg.filter(
                (pl.col("election") == election)
                & pl.col("fieldwork_end").is_between(t - dt.timedelta(days=13), t)
                & pl.col("candidate_id").is_in(config)
            )
            .group_by("candidate_id")
            .agg(pl.col("gap").mean())
        )
        if g.height < 4:
            continue
        a1 = poll_mean(election, t, config, polls=polls)
        a2 = poll_mean(election, t2, config, polls=polls)
        s1 = dict(zip(config, 100 * a1.shares, strict=True))
        s2 = dict(zip(config, 100 * a2.shares, strict=True))
        for c, gv in zip(g["candidate_id"], g["gap"], strict=True):
            if c in a1.missing or c in a2.missing or s1[c] < 1.0:
                continue
            rows.append(
                {
                    "election": election,
                    "t": t,
                    "days_before": k,
                    "candidate_id": c,
                    "gap": gv,
                    "share_t": s1[c],
                    "move": s2[c] - s1[c],
                }
            )
    df = pl.DataFrame(rows)
    if df.height:
        df = df.with_columns(
            (pl.col("gap") - pl.col("gap").mean().over("t")).alias("gap_c"),
            (pl.col("move") - pl.col("move").mean().over("t")).alias("move_c"),
        )
    return df


def movement_test(mv: pl.DataFrame, rng, n_perm: int) -> dict:
    out = {}
    stats_by_e = {}
    for e in ELECTIONS:
        sub = mv.filter(pl.col("election") == e)
        if sub.height < 6:
            continue
        r = pearson_pooled([sub["gap_c"].to_numpy()], [sub["move_c"].to_numpy()])
        cands = sorted(sub["candidate_id"].unique().to_list())
        hits = 0
        for _ in range(n_perm):
            perm = dict(zip(cands, rng.permutation(cands), strict=True))
            # même permutation des identités à toutes les dates : g du candidat perm[c]
            gp = sub.select("t", "candidate_id", "gap").with_columns(
                pl.col("candidate_id").replace_strict(perm).alias("candidate_id")
            )
            j = sub.select("t", "candidate_id", "move_c").join(gp, on=["t", "candidate_id"])
            if j.height < 6:
                continue
            j = j.with_columns((pl.col("gap") - pl.col("gap").mean().over("t")).alias("gp"))
            rp = pearson_pooled([j["gp"].to_numpy()], [j["move_c"].to_numpy()])
            hits += abs(rp) >= abs(r) - 1e-12
        stats_by_e[e] = {
            "n": sub.height,
            "dates": sub["t"].n_unique(),
            "r": r,
            "p": (hits + 1) / (n_perm + 1),
        }
    out["per_election"] = stats_by_e
    pooled = mv.filter(pl.col("election").is_in(list(stats_by_e)))
    out["pooled_r"] = pearson_pooled([pooled["gap_c"].to_numpy()], [pooled["move_c"].to_numpy()])
    # p poolé : combinaison de Fisher des deux p bilatéraux
    from scipy import stats as st

    ps = [v["p"] for v in stats_by_e.values()]
    out["pooled_p"] = float(st.combine_pvalues(ps, method="fisher").pvalue) if ps else float("nan")
    rs = [v["r"] for v in stats_by_e.values()]
    out["passes"] = bool(
        len(rs) == 2 and np.sign(rs[0]) == np.sign(rs[1]) and out["pooled_p"] <= 0.05
    )
    return out


# -------------------------------------------------------------------- (d) participation


def turnout_table(intent: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for e in (*ELECTIONS, 2027):
        for rnd in (1, 2):
            if e == 2027:
                lo, hi, actual = dt.date(2026, 9, 1), dt.date(2026, 12, 31), None
            else:
                vote = vote_date(e, rnd)
                lo, hi, actual = (
                    vote - dt.timedelta(days=30),
                    vote - dt.timedelta(days=1),
                    turnout(e, rnd),
                )
            sub = intent.filter(
                (pl.col("election") == e)
                & (pl.col("round") == rnd)
                & pl.col("fieldwork_end").is_between(lo, hi)
            )
            for (pollster, kind), g in sub.group_by(["pollster", "kind"]):
                m = float(g["certain_share"].mean())
                rows.append(
                    {
                        "election": e,
                        "round": rnd,
                        "pollster": pollster,
                        "kind": kind,
                        "polls": g.height,
                        "certain_share": m,
                        "min": float(g["certain_share"].min()),
                        "max": float(g["certain_share"].max()),
                        "actual_turnout": actual,
                        "offset": None if actual is None else actual - m,
                    }
                )
    return pl.DataFrame(rows).sort("election", "round", "pollster")


def turnout_verdict(tt: pl.DataFrame) -> dict:
    out = {}
    for (pollster, kind, rnd), g in tt.filter(pl.col("election").is_in(list(ELECTIONS))).group_by(
        ["pollster", "kind", "round"]
    ):
        if g.height != 2:
            continue
        g = g.sort("election")
        d_off = float(g["offset"][1] - g["offset"][0])
        d_cert = float(g["certain_share"][1] - g["certain_share"][0])
        d_turn = float(g["actual_turnout"][1] - g["actual_turnout"][0])
        out[f"{pollster}/{kind}/T{rnd}"] = {
            "offset_change": d_off,
            "certain_change": d_cert,
            "turnout_change": d_turn,
            "passes": bool(abs(d_off) <= 2 and np.sign(d_cert) == np.sign(d_turn)),
        }
    return out


# ------------------------------------------------------------------------- figures


def fig_timeseries(gap: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, e in zip(axes, (*ELECTIONS, 2027), strict=True):
        sub = (
            gap.filter((pl.col("election") == e) & (pl.col("round") == 1))
            .group_by("fieldwork_end", "bloc")
            .agg(
                pl.col("share_certain").sum(),
                pl.col("share_registered").sum(),
                pl.col("poll_id").n_unique(),
            )
        )
        # g par bloc et par sondage (somme des candidats du bloc, moyenne des hypothèses)
        bl = (
            gap.filter((pl.col("election") == e) & (pl.col("round") == 1))
            .group_by("poll_id", "fieldwork_end", "hypothesis_id", "bloc")
            .agg(pl.col("gap").sum())
            .group_by("poll_id", "fieldwork_end", "bloc")
            .agg(pl.col("gap").mean())
            .sort("fieldwork_end")
        )
        for b in BLOCS[:5]:
            s = bl.filter(pl.col("bloc") == b)
            if s.height < 3:
                continue
            y = s["gap"].rolling_mean(5, min_samples=1).to_numpy()
            ax.plot(s["fieldwork_end"].to_list(), y, label=b)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(str(e))
        ax.tick_params(axis="x", rotation=30)
        _ = sub
    axes[0].set_ylabel("points d'exprimés")
    axes[0].legend(fontsize=8)
    fig.suptitle("Écart certains − inscrits par bloc (OpinionWay, moyenne mobile sur 5 sondages)")
    fig.tight_layout()
    fig.savefig(out / "gap_by_bloc.png", dpi=110)
    plt.close(fig)


def fig_gap_error(fv: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, target, title in (
        (axes[0], "err_ow_certain", "erreur OpinionWay (certains, 14 j)"),
        (axes[1], "err_all", "erreur tous instituts (14 j)"),
    ):
        for e, mk in ((2017, "o"), (2022, "s")):
            s = fv.filter((pl.col("election") == e) & pl.col("gap60").is_not_null())
            ax.scatter(s["gap60"], s[target], marker=mk, label=str(e))
            for c, x, y in zip(s["candidate_id"], s["gap60"], s[target], strict=True):
                if y is not None and abs(y) > 1 or abs(x) > 0.5:
                    ax.annotate(REG.candidates[c].last_name, (x, y), fontsize=7)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("g (certains − inscrits), 60 jours avant le vote")
        ax.set_ylabel("résultat − sondages (points)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gap_vs_error.png", dpi=110)
    plt.close(fig)


def fig_turnout(intent: pl.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, e in zip(axes, ELECTIONS, strict=True):
        sub = intent.filter((pl.col("election") == e) & (pl.col("round") == 1)).sort(
            "fieldwork_end"
        )
        for (pollster, kind), g in sub.group_by(["pollster", "kind"]):
            ax.plot(
                g["fieldwork_end"].to_list(),
                g["certain_share"].to_list(),
                ".-",
                label=f"{pollster} ({kind})",
            )
        ax.axhline(turnout(e, 1), color="black", ls="--", label="participation T1")
        ax.axhline(turnout(e, 2), color="grey", ls=":", label="participation T2")
        ax.set_title(f"{e} — part de certains d'aller voter publiée")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("% des inscrits interrogés")
    fig.tight_layout()
    fig.savefig(out / "turnout_intent.png", dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.RUNS / f"{dt.date.today()}-turnout-gap"))
    ap.add_argument("--perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gap = tg.build()
    pg = poll_gaps(gap)
    intent = pl.read_parquet(paths.CURATED / "poll_turnout_intent.parquet")
    coverage = pl.read_parquet(paths.CURATED / "poll_archive_coverage.parquet")
    gap.write_csv(out / "turnout_gap.csv")

    # couverture
    cov_pairs = (
        gap.group_by("election", "pollster")
        .agg(
            pl.col("poll_id").n_unique().alias("polls_both_bases"),
            pl.col("fieldwork_end").min().alias("first"),
            pl.col("fieldwork_end").max().alias("last"),
            pl.col("hypothesis_id").n_unique().alias("hypotheses"),
        )
        .sort("election")
    )
    pa = pl.concat(
        [
            pl.read_parquet(paths.CURATED / "polls_archive.parquet").select(
                "poll_id", "election", "pollster"
            ),
            pl.read_parquet(paths.CURATED / "polls_commission_dual.parquet").select(
                "poll_id", "election", "pollster"
            ),
        ]
    )
    ra = pl.concat(
        [
            pl.read_parquet(paths.CURATED / "poll_results_archive.parquet"),
            pl.read_parquet(paths.CURATED / "poll_results_commission_dual.parquet"),
        ],
        how="diagonal_relaxed",
    )
    cov_pop = (
        ra.join(pa, on="poll_id")
        .group_by("election", "pollster", "population")
        .agg(pl.col("poll_id").n_unique().alias("polls"))
        .pivot(on="population", index=["election", "pollster"], values="polls")
        .sort("election", "pollster")
    )

    # (a)
    stab = [stability(pg, e, rng, args.perm) for e in (*ELECTIONS, 2027)]
    inter = [inter_institute(pg, e, rng, args.perm) for e in ELECTIONS]
    # entre élections, par bloc
    bloc_gap = (
        gap.filter(pl.col("round") == 1)
        .group_by("election", "poll_id", "hypothesis_id", "bloc")
        .agg(pl.col("gap").sum())
        .group_by("election", "poll_id", "bloc")
        .agg(pl.col("gap").mean())
        .group_by("election", "bloc")
        .agg(
            pl.col("gap").mean().alias("gap"),
            (pl.col("gap").std() / pl.len().sqrt()).alias("se_naive"),
            pl.len().alias("polls"),
        )
        .sort("bloc", "election")
    )

    # (b)
    be = pl.DataFrame([r for e in ELECTIONS for r in base_errors(gap, e)])
    fv = pl.concat([final_vectors(gap, pg, e) for e in ELECTIONS])
    gve = {t: gap_vs_error(fv, t, rng, args.perm) for t in ("err_ow_certain", "err_all")}
    fb = (
        fv.filter(pl.col("ow_certain").is_not_null())
        .group_by("election", "bloc")
        .agg(pl.col("result", "ow_certain", "ow_registered", "all_pollsters", "gap60").sum())
        .with_columns(
            (pl.col("result") - pl.col("ow_certain")).alias("err_certain"),
            (pl.col("result") - pl.col("ow_registered")).alias("err_registered"),
        )
        .with_columns(
            pl.when(pl.col("err_certain").abs() < pl.col("err_registered").abs())
            .then(pl.lit("certains"))
            .otherwise(pl.lit("inscrits"))
            .alias("closer")
        )
        .sort("election", "bloc")
    )

    # (c)
    mv = pl.concat([movement(pg, e) for e in ELECTIONS], how="diagonal_relaxed")
    mt = movement_test(mv, rng, max(1000, args.perm // 5))

    # (d)
    tt = turnout_table(intent)
    tv = turnout_verdict(tt)

    # (e) 2027
    g27 = (
        pg.filter(pl.col("election") == 2027)
        .group_by("candidate_id")
        .agg(
            pl.col("gap").mean().alias("gap"),
            (pl.col("gap").std() / pl.len().sqrt()).alias("se"),
            pl.col("share_certain").mean().alias("share_certain"),
            pl.col("share_registered").mean().alias("share_registered"),
            pl.len().alias("polls"),
            pl.col("fieldwork_end").max().alias("last"),
        )
        .with_columns(
            pl.col("candidate_id")
            .map_elements(lambda c: bloc_of(c, 2027), return_dtype=pl.Utf8)
            .alias("bloc")
        )
        .sort("gap")
    )

    fig_timeseries(gap, out)
    fig_gap_error(fv, out)
    fig_turnout(intent, out)

    for name, df in (
        ("coverage_pairs", cov_pairs),
        ("coverage_population", cov_pop),
        ("coverage_parse", coverage),
        ("bloc_gap", bloc_gap),
        ("base_errors", be),
        ("final_vectors", fv),
        ("final_blocs", fb),
        ("movement", mv),
        ("turnout_intent", tt),
        ("gap_2027", g27),
    ):
        df.write_csv(out / f"{name}.csv")
    results = {
        "stability": [{k: v for k, v in s.items() if k != "table"} for s in stab],
        "inter_institute": [{k: v for k, v in s.items() if k != "table"} for s in inter],
        "gap_vs_error": gve,
        "movement": mt,
        "turnout": tv,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # rapport : tableaux et verdicts des critères ; la lecture est consignée dans docs/
    lines = [
        "# Abstention différentielle : écart certains d'aller voter − inscrits",
        "",
        f"Généré le {dt.date.today()} par `back/studies/turnout_gap.py` (permutations : {args.perm}).",
        "Protocole et critères : docstring du script, recopiés en fin de rapport.",
        "",
        "## Couverture",
        "",
        "Notices lues par institut (`poll_archive_coverage.parquet`) :",
        "",
        md_table(
            coverage.select(
                "election",
                "pollster",
                "iv_notices",
                "parsed",
                "dual",
                "errors",
                "turnout_intent",
                "main_population",
                "parse_rate",
            ).sort("election", "pollster")
        ),
        "",
        "Sondages par base publiée (archives 2017-2022, relecture 2027) :",
        "",
        md_table(cov_pop),
        "",
        "Sondages publiant les deux bases sur la même hypothèse (`turnout_gap.parquet`) :",
        "",
        md_table(cov_pairs),
        "",
        "## (a) Stabilité",
        "",
    ]
    for s in stab:
        if not s.get("n_candidates"):
            continue
        lines.append(
            f"- {s['election']} : coupure {s['cut']}, {s['n_candidates']} candidats, r = {s['r']:.2f}"
            f" (placebo p = {s['p']:.3f}), signe conservé pour {s['same_sign_share']:.0%} des |g| ≥ 0,5,"
            f" écart-type médian d'un sondage à l'autre {s['median_within_sd']:.2f} pt"
            f" — critère {'rempli' if s['passes'] else 'non rempli'}"
        )
    lines += [
        "",
        md_table(
            pl.concat(
                [s["table"] for s in stab if s.get("n_candidates")], how="diagonal_relaxed"
            ).select("election", "candidate_id", "g1", "n1", "g2", "n2")
        ),
        "",
    ]
    lines.append("Entre instituts (écart publié certains − inscrits, quinzaines communes) :")
    lines.append("")
    for s in inter:
        if "r" in s:
            lines.append(
                f"- {s['election']} (inscrits : {', '.join(s['registered_side'])}) : {s['n_candidates']}"
                f" candidats, r = {s['r']:.2f} (p = {s['p']:.3f})"
            )
    lines += [
        "",
        md_table(
            pl.concat([s["table"] for s in inter if s["table"].height], how="diagonal_relaxed")
        ),
        "",
    ]
    lines += [
        "Par bloc et par élection (somme des candidats du bloc) :",
        "",
        md_table(bloc_gap),
        "",
    ]
    lines += [
        "## (b) Erreur finale",
        "",
        "Erreur absolue moyenne par base (OpinionWay) :",
        "",
        md_table(be),
        "",
    ]
    lines += ["Par bloc, 14 derniers jours (OpinionWay) :", "", md_table(fb), ""]
    lines += [
        "Vecteurs finaux par candidat :",
        "",
        md_table(
            fv.select(
                "election",
                "candidate_id",
                "bloc",
                "result",
                "ow_certain",
                "ow_registered",
                "all_pollsters",
                "gap60",
                "err_ow_certain",
                "err_ow_registered",
                "err_all",
            )
        ),
        "",
    ]
    for t, r in gve.items():
        pe = r["per_election"]
        lines.append(
            f"- e = {t} : β 2017 = {pe[2017]['beta']:+.2f} (r {pe[2017]['r']:+.2f}, p {pe[2017]['p']:.3f}),"
            f" β 2022 = {pe[2022]['beta']:+.2f} (r {pe[2022]['r']:+.2f}, p {pe[2022]['p']:.3f}),"
            f" poolé β = {r['pooled']['beta']:+.2f} (p {r['pooled']['p']:.3f})"
            f" — critère {'rempli' if r['passes'] else 'non rempli'} ; validation croisée :"
            + "; ".join(
                f" {k} λ={v['lambda']:+.2f} MAE {v['mae_before']:.2f} → {v['mae_after']:.2f}"
                for k, v in r["cv"].items()
            )
        )
    lines += ["", "## (c) Mouvement ultérieur", ""]
    for e, v in mt["per_election"].items():
        lines.append(
            f"- {e} : {v['n']} points (candidat × date, {v['dates']} dates), r = {v['r']:+.2f}, p = {v['p']:.3f}"
        )
    lines.append(
        f"- poolé : r = {mt['pooled_r']:+.2f}, p (Fisher) = {mt['pooled_p']:.3f} — critère"
        f" {'rempli' if mt['passes'] else 'non rempli'}"
    )
    lines += ["", "## (d) Participation", "", md_table(tt), ""]
    for k, v in tv.items():
        lines.append(
            f"- {k} : décalage {v['offset_change']:+.1f} pt entre 2017 et 2022, part de certains"
            f" {v['certain_change']:+.1f}, participation {v['turnout_change']:+.1f} — critère"
            f" {'rempli' if v['passes'] else 'non rempli'}"
        )
    lines += ["", "## (e) 2027", "", md_table(g27), ""]
    lines += ["## Protocole", "", "```", __doc__.strip(), "```", ""]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:400]))


if __name__ == "__main__":
    main()
