"""Étude : l'attention en ligne annonce-t-elle les mouvements des sondages par candidat ?

    uv run python studies/attention.py [--out runs/<date>-attention] [--shifts 200]

Protocole de docs/06-presse.md (part de voix contre émergence, Granger dans les deux sens,
chocs), appliqué aux pages vues Wikipédia (2017, 2022, 2027) et, en réplication quand le
cache existe, à Google Trends (2007 à 2027). Rien n'entre dans le modèle ici.

Données (vvs.studies.panel, vvs.ingest.attention) :

- attention hebdomadaire par candidat (lundi à dimanche) : pages vues de la semaine
  (Wikipédia) ou indice Trends de la semaine Google ; `x` = part d'attention en points
  (100 × part du candidat parmi les membres du cycle, data/registry/attention_queries.yaml) ;
  `x4` = moyenne mobile de `x` sur les 4 dernières semaines ; `datt` = x4(t) − x4(t − 4),
  variation de la part d'attention sur un mois connue à la fin de la semaine t ;
- sondages : part hebdomadaire de premier tour par candidat (moyenne des instituts, chacun
  moyenné sur ses sondages de la semaine de fin de terrain) ; semaine sans sondage vide.

Questions et tests, fixés avant de lire les résultats :

(a) **L'attention précède-t-elle les sondages ?**
    - spécification principale (prévision « comme à l'époque ») : pour chaque semaine t où le
      candidat est sondé, y = sondage(t + h) − sondage(t), h = 4 semaines (cible : semaine
      t + 4, sinon t + 5, sinon t + 3), régressé sur `datt`(t) et sur l'élan des sondages
      sondage(t) − sondage(t − 4) (même tolérance), effet fixe par cycle, écarts types par
      grappe candidat × cycle ; h = 2 et 8 en complément ;
    - Granger dans les deux sens sur les semaines consécutives sondées (ordre 2, 4 en
      complément) : variations hebdomadaires de sondage sur leurs retards et ceux de la part
      d'attention, et l'inverse ;
    - corrélations croisées des variations hebdomadaires, retards de −8 à +8 semaines
      (descriptif, bande placebo) ;
    - épisodes d'émergence détectés par une règle : hausse d'au moins 5 points de la moyenne
      mensuelle des sondages en 6 mois au plus (la plus forte par candidat et cycle) ; date de
      mi-hausse des sondages (interpolation linéaire entre semaines sondées) contre date de
      mi-hausse de `x4` (de sa moyenne des 8 semaines avant l'épisode à son maximum) ; avance
      = différence en semaines.
    Placebo : la part d'attention de tous les candidats d'un cycle est décalée
    circulairement de s semaines (13 ≤ s ≤ T − 13), les autocorrélations sont conservées et
    tout lien avec les sondages est fortuit ; p placebo = rang de la statistique observée
    (|t| ou Wald) parmi les décalages. La part des décalages où le test nominal rejette à 5 %
    mesure sa taille réelle. Critère « tient » : β > 0, p placebo < 0,05 en commun 2017-2022,
    β > 0 dans chacun des deux cycles, signe inchangé en retirant chaque candidat, Granger
    attention → sondages à p placebo < 0,05. Les épisodes sont descriptifs : « l'attention
    mène » si sa mi-hausse précède celle des sondages d'au moins 2 semaines dans la majorité
    des épisodes.

(b) **Les pics d'attention annoncent-ils des mouvements ?** Pic : semaine dont les pages vues
    sont au moins 3 fois la médiane des 8 semaines précédentes, à un z-score d'au moins 3
    (log des vues contre les 8 semaines précédentes), avec au moins 10 000 vues ; un pic au
    plus par fenêtre de 4 semaines. Mouvement : dernier sondage des semaines t − 4 à t − 1
    contre premier sondage des semaines t + 1 à t + 4. Comparaison à des semaines tirées au
    hasard pour le même candidat, à 12 semaines au plus du pic et à plus de 4 semaines de
    tout pic (5 000 tirages). Critère : |mouvement| moyen après pic plus grand qu'au hasard,
    p < 0,05 en commun, même sens en 2017 et en 2022. Le sens du mouvement est rapporté sans
    critère (l'attention ne dit pas le signe). Cas Fillon en figure.

(c) **L'attention prévoit-elle l'erreur finale ?** Surperformance = résultat − moyenne des
    sondages des 14 derniers jours (poll_errors, `mean_14d`). Prédicteur principal : log du
    rapport part d'attention / part dans les sondages sur les 14 derniers jours, parmi les
    candidats officiels ; secondaires : écart en points, élan d'attention (14 derniers jours
    contre les 4 semaines d'avant). Régression avec effet fixe par élection ; p par
    permutation des prédicteurs entre candidats d'une même élection (20 000 tirages).
    Critère : p < 0,05 et même signe dans chaque élection. Robustesse : sans le président
    sortant candidat (Macron 2022). Deux élections seulement en Wikipédia, quatre avec Trends.

    Ajout du 2026-10-20, après lecture des résultats Wikipédia et avant toute lecture des
    données Trends : l'élan d'attention (prédicteur secondaire) y ressort à la marge
    (β > 0 en 2017 et en 2022, p = 0,035 sans le sortant, une variante sur six). Test
    confirmatoire déclaré d'avance sur des élections non vues, 2007 et 2012 avec Trends :
    β de l'élan > 0 dans chacune, p permutation < 0,05 sur les deux réunies.
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
import pandas as pd
import polars as pl
from scipy import stats

from vvs import paths
from vvs.ingest.attention import OUTPUT_NAME, load_registry
from vvs.ingest.attention.google_trends import cycle_span
from vvs.model.plots import GRID, INK, MUTED, RESULT, SERIES, _style, display_name
from vvs.studies import panel as P

CYCLES = {"wikipedia": (2017, 2022, 2027), "trends": (2007, 2012, 2017, 2022, 2027)}
TRAIN = {"wikipedia": (2017, 2022), "trends": (2007, 2012, 2017, 2022)}
H_MAIN, HORIZONS, TOL = 4, (2, 4, 8), 1
ATT_WINDOW = 4
GRANGER_ORDERS = (2, 4)
XCORR_LAGS = tuple(range(-8, 9))
MIN_SHIFT = 13
PEAK_RATIO, PEAK_Z, PEAK_MIN_VIEWS, PEAK_BASE, PEAK_GAP = 3.0, 3.0, 10_000, 8, 4
NULL_RADIUS, N_PERM_PEAKS = 12, 5000
EPISODE_RISE, EPISODE_MONTHS = 5.0, 6
FINAL_DAYS, MOMENTUM_DAYS = 14, 28
N_PERM_FINAL = 20_000
INCUMBENT_CANDIDATES = {2022: "macron_emmanuel"}
HOLDOUT_FINAL = (2007, 2012)  # test confirmatoire de l'élan (Trends), voir la docstring
# Changements de configuration (robustesse ajoutée après la première lecture) : primaires,
# déclarations et retraits qui modifient à la fois l'attention et les hypothèses testées.
EVENTS = {
    2017: [dt.date(2016, 8, 30), dt.date(2016, 11, 16), dt.date(2016, 11, 20),
           dt.date(2016, 11, 27), dt.date(2016, 12, 1), dt.date(2017, 1, 22),
           dt.date(2017, 1, 25), dt.date(2017, 1, 29), dt.date(2017, 2, 22)],
    2022: [dt.date(2021, 9, 12), dt.date(2021, 9, 28), dt.date(2021, 11, 30),
           dt.date(2021, 12, 4), dt.date(2022, 1, 15), dt.date(2022, 1, 30),
           dt.date(2022, 3, 2)],
}
SEED = 20261020


# ---------------------------------------------------------------------------- données


def load_inputs():
    reg = load_registry()
    att = pl.read_parquet(paths.CURATED / OUTPUT_NAME)
    results = pl.read_parquet(paths.CURATED / "poll_results.parquet")
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    errors = pl.read_parquet(paths.CURATED / "poll_errors.parquet")
    elections = sorted({e for c in CYCLES.values() for e in c})
    poll_shares = P.poll_candidate_shares(results, polls, elections)
    weekly_polls = P.weekly_candidate_shares(poll_shares)
    return reg, att, poll_shares, weekly_polls, errors


class Cycle:
    """Matrices hebdomadaires d'un cycle : semaines × membres (attention, sondages)."""

    def __init__(self, election: int, weeks: list[dt.date], cands: list[str],
                 views: np.ndarray, polls: np.ndarray, n_polls: np.ndarray):
        self.election, self.weeks, self.cands = election, weeks, cands
        self.views, self.polls, self.n_polls = views, polls, n_polls
        tot = views.sum(axis=1, keepdims=True)
        self.x = np.where(tot > 0, 100 * views / np.where(tot > 0, tot, 1), np.nan)

    @property
    def T(self) -> int:
        return len(self.weeks)

    def shifted_x(self, s: int) -> np.ndarray:
        return np.roll(self.x, s, axis=0) if s else self.x

    def datt(self, s: int = 0) -> np.ndarray:
        return delta_rolling(self.shifted_x(s), ATT_WINDOW)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def delta_rolling(x: np.ndarray, w: int) -> np.ndarray:
    m = rolling_mean(x, w)
    out = np.full_like(m, np.nan)
    out[w:] = m[w:] - m[:-w]
    return out


def build_cycles(reg, att: pl.DataFrame, weekly_polls: pl.DataFrame, source: str
                 ) -> dict[int, Cycle]:
    aw = P.weekly_attention(att, source)
    if aw.is_empty():
        return {}
    first_week, last_week = aw["week"].min(), aw["week"].max()
    out = {}
    for e in CYCLES[source]:
        d0, vote = cycle_span(reg, e)
        w0 = max(_monday(d0 + dt.timedelta(days=6)), first_week)
        w1 = min(_monday(vote - dt.timedelta(days=6)), last_week)
        if w1 <= w0:
            continue
        weeks = list(pl.date_range(w0, w1, "1w", eager=True))
        have = set(aw.filter(pl.col("week").is_between(w0, w1))["candidate_id"].unique())
        cands = [c for c in reg.members(e) if c in have]
        if not cands:
            continue
        grid = pl.DataFrame({"week": weeks}).join(
            pl.DataFrame({"candidate_id": cands}), how="cross")
        a = grid.join(aw, on=["candidate_id", "week"], how="left").with_columns(
            pl.col("value").fill_null(0.0))
        p = grid.join(weekly_polls.filter(pl.col("election") == e)
                      .select("candidate_id", "week", "share", "n_polls"),
                      on=["candidate_id", "week"], how="left")
        V = a.pivot(on="candidate_id", index="week", values="value").sort("week")
        S = p.pivot(on="candidate_id", index="week", values="share").sort("week")
        N = p.with_columns(pl.col("n_polls").fill_null(0)).pivot(
            on="candidate_id", index="week", values="n_polls").sort("week")
        out[e] = Cycle(e, weeks, cands, V.select(cands).to_numpy().astype(float),
                       S.select(cands).to_numpy().astype(float),
                       N.select(cands).to_numpy().astype(float))
    return out


def _monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def panel_frame(cycles: dict[int, Cycle]) -> pd.DataFrame:
    rows = []
    for e, c in cycles.items():
        x4 = rolling_mean(c.x, ATT_WINDOW)
        da = c.datt()
        for j, cid in enumerate(c.cands):
            for t, w in enumerate(c.weeks):
                rows.append((e, cid, w, t, c.views[t, j], c.x[t, j], x4[t, j], da[t, j],
                             c.polls[t, j], c.n_polls[t, j]))
    return pd.DataFrame(rows, columns=["election", "candidate_id", "week", "t", "views", "x",
                                       "x4", "datt", "poll", "n_polls"])


# ---------------------------------------------------------------------------- outils


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """MCO, covariance robuste par grappe (correction G/(G−1)·(N−1)/(N−K)).
    Renvoie (beta, cov, ddl = G − 1) ou None si dégénéré."""
    n, k = X.shape
    ug, g = np.unique(groups, return_inverse=True)
    G = len(ug)
    if n <= k + 2 or G < 3:
        return None
    XtX = X.T @ X
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    b = inv @ X.T @ y
    u = y - X @ b
    S = np.zeros((G, k))
    np.add.at(S, g, X * u[:, None])
    meat = S.T @ S
    cov = inv @ meat @ inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, cov, G - 1


def fe_design(cols: list[np.ndarray], fe: np.ndarray) -> np.ndarray:
    levels = np.unique(fe)
    dummies = [(fe == lv).astype(float) for lv in levels]
    return np.column_stack([*cols, *dummies])


def t_and_p(fit, i: int) -> tuple[float, float, float, float]:
    if fit is None:
        return (np.nan,) * 4
    b, cov, ddl = fit
    se = float(np.sqrt(cov[i, i]))
    t = b[i] / se if se > 0 else np.nan
    return float(b[i]), se, float(t), float(2 * stats.t.sf(abs(t), ddl))


def wald(fit, idx: list[int]) -> tuple[float, float]:
    if fit is None:
        return np.nan, np.nan
    b, cov, ddl = fit
    bb, V = b[idx], cov[np.ix_(idx, idx)]
    try:
        w = float(bb @ np.linalg.solve(V, bb))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    q = len(idx)
    return w, float(stats.f.sf(w / q, q, ddl))


def shift_grid(cycles: dict[int, Cycle], n: int) -> list[dict[int, int]]:
    """n jeux de décalages (un par cycle), répartis régulièrement dans [13, T − 13]."""
    out = []
    for u in (np.arange(n) + 0.5) / n:
        out.append({e: int(MIN_SHIFT + round(u * (c.T - 2 * MIN_SHIFT))) for e, c in cycles.items()
                    if c.T > 2 * MIN_SHIFT + 1})
    return out


def placebo_p(obs: float, null: np.ndarray) -> float:
    null = null[~np.isnan(null)]
    if np.isnan(obs) or len(null) == 0:
        return np.nan
    return float((1 + (null >= obs).sum()) / (1 + len(null)))


# ---------------------------------------------------------------------------- (a) prévision


def _observed(p: np.ndarray, t: int, offsets: tuple[int, ...]) -> int | None:
    for o in offsets:
        u = t + o
        if 0 <= u < len(p) and not np.isnan(p[u]):
            return u
    return None


def anchor_rows(cycles: dict[int, Cycle], h: int) -> pd.DataFrame:
    """Une ligne par (cycle, candidat, semaine sondée t) avec cible t + h et élan t − 4."""
    rows = []
    for e, c in cycles.items():
        for j, cid in enumerate(c.cands):
            p = c.polls[:, j]
            for t in np.flatnonzero(~np.isnan(p)):
                tgt = _observed(p, t, (h, h + TOL, h - TOL))
                past = _observed(p, t, (-ATT_WINDOW, -ATT_WINDOW - TOL, -ATT_WINDOW + TOL))
                if tgt is None or tgt <= t:
                    continue
                rows.append({"election": e, "candidate_id": cid, "j": j, "t": int(t),
                             "y": p[tgt] - p[t],
                             "mom": np.nan if past is None else p[t] - p[past]})
    return pd.DataFrame(rows)


def gather(rows: pd.DataFrame, mats: dict[int, np.ndarray], lag: int = 0) -> np.ndarray:
    """mats[cycle][t − lag, j] pour chaque ligne (NaN avant le début du cycle)."""
    out = np.full(len(rows), np.nan)
    e_arr, t_arr, j_arr = (rows[k].to_numpy() for k in ("election", "t", "j"))
    for e, m in mats.items():
        sel = e_arr == e
        tt = t_arr[sel] - lag
        ok = tt >= 0
        vals = np.full(sel.sum(), np.nan)
        vals[ok] = m[tt[ok], j_arr[sel][ok]]
        out[sel] = vals
    return out


def anchor_fit(rows: pd.DataFrame, cycles: dict[int, Cycle], shifts: dict[int, int] | None,
               control: bool = True, lag: int = 0):
    if rows.empty:
        return None, 0
    x = gather(rows, {e: c.datt((shifts or {}).get(e, 0)) for e, c in cycles.items()}, lag)
    ok = ~np.isnan(x) & ~np.isnan(rows["y"].to_numpy())
    if control:
        ok &= ~np.isnan(rows["mom"].to_numpy())
    r = rows[ok]
    cols = [x[ok]] + ([r["mom"].to_numpy()] if control else [])
    X = fe_design(cols, r["election"].to_numpy())
    groups = (r["election"].astype(str) + ":" + r["candidate_id"]).to_numpy()
    return ols_cluster(r["y"].to_numpy(), X, groups), int(ok.sum())


def anchor_analysis(cycles, train, shift_sets) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coefficients (h, échantillon, contrôle) et taille réelle du test principal."""
    res, size_rows = [], []
    sub = {"commun": [e for e in train if e in cycles]}
    sub |= {str(e): [e] for e in cycles}
    for h in HORIZONS:
        rows_all = anchor_rows(cycles, h)
        for name, els in sub.items():
            cyc = {e: cycles[e] for e in els}
            rows = rows_all[rows_all["election"].isin(els)]
            for control in (True, False):
                fit, n = anchor_fit(rows, cyc, None, control)
                b, se, t, p = t_and_p(fit, 0)
                out = {"h": h, "sample": name, "control": control, "beta": b, "se": se, "t": t,
                       "p": p, "n": n, "n_cand": rows["candidate_id"].nunique() if n else 0}
                if h == H_MAIN or (name == "commun" and control):
                    null_t, null_p = [], []
                    for s in shift_sets:
                        f0, _ = anchor_fit(rows, cyc, s, control)
                        _, _, tn, pn = t_and_p(f0, 0)
                        null_t.append(abs(tn))
                        null_p.append(pn)
                    null_t, null_p = np.array(null_t), np.array(null_p)
                    out["p_placebo"] = placebo_p(abs(t), null_t)
                    out["size_05"] = float(np.nanmean(null_p < 0.05))
                    if name == "commun" and control and h == H_MAIN:
                        size_rows = [{"shift": i, "abs_t": a, "p": q}
                                     for i, (a, q) in enumerate(zip(null_t, null_p))]
                res.append(out)
    return pd.DataFrame(res), pd.DataFrame(size_rows)


def event_mask(rows: pd.DataFrame, cycles: dict[int, Cycle], h: int) -> np.ndarray:
    """Vrai si aucun événement de `EVENTS` ne tombe entre t − 4 semaines et la fin de t + h."""
    keep = np.ones(len(rows), dtype=bool)
    for i, (e, t) in enumerate(rows[["election", "t"]].itertuples(index=False)):
        w = cycles[e].weeks[t]
        lo, hi = w - dt.timedelta(weeks=ATT_WINDOW), w + dt.timedelta(weeks=h, days=6)
        keep[i] = not any(lo <= d <= hi for d in EVENTS.get(e, ()))
    return keep


def anchor_robustness(cycles, train, shift_sets) -> pd.DataFrame:
    """Variantes de la spécification principale (h = 4, commun, avec élan), ajoutées après la
    première lecture pour interpréter le résultat, avec leur propre placebo."""
    els = [e for e in train if e in cycles]
    cyc = {e: cycles[e] for e in els}
    rows = anchor_rows(cyc, H_MAIN)
    da = gather(rows, {e: c.datt() for e, c in cyc.items()})
    big = np.abs(da) > np.nanpercentile(np.abs(da), 95)
    variants = {
        "principale": (rows, 0),
        "attention décalée d'une semaine": (rows, 1),
        "sans les 5 % plus fortes variations d'attention": (rows[~big], 0),
        "hors fenêtres d'événements de configuration": (rows[event_mask(rows, cyc, H_MAIN)], 0),
    }
    out = []
    for name, (r, lag) in variants.items():
        fit, n = anchor_fit(r, cyc, None, True, lag)
        b, se, t, p = t_and_p(fit, 0)
        null = np.array([abs(t_and_p(anchor_fit(r, cyc, sh, True, lag)[0], 0)[2])
                         for sh in shift_sets])
        x = gather(r, {e: c.datt() for e, c in cyc.items()}, lag)
        sd = float(np.nanstd(x[~np.isnan(r["mom"].to_numpy())]))
        out.append({"variant": name, "beta": b, "se": se, "p": p,
                    "p_placebo": placebo_p(abs(t), null), "n": n, "sd_datt": sd,
                    "effect_sd": b * sd, "sd_y": float(r["y"].std())})
    return pd.DataFrame(out)


def leave_one_candidate_out(cycles, train) -> pd.DataFrame:
    els = [e for e in train if e in cycles]
    cyc = {e: cycles[e] for e in els}
    rows = anchor_rows(cyc, H_MAIN)
    out = []
    for cid in sorted(rows["candidate_id"].unique()):
        fit, n = anchor_fit(rows[rows["candidate_id"] != cid], cyc, None, True)
        b, _, _, p = t_and_p(fit, 0)
        out.append({"without": cid, "beta": b, "p": p, "n": n})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- (a) Granger


def granger_rows(cycles: dict[int, Cycle], order: int) -> pd.DataFrame:
    """Semaines t où le sondage est observé de t − order − 1 à t (variations consécutives)."""
    rows = []
    for e, c in cycles.items():
        for j, cid in enumerate(c.cands):
            p = c.polls[:, j]
            for t in range(order + 1, c.T):
                if np.all(~np.isnan(p[t - order - 1:t + 1])):
                    rows.append({"election": e, "candidate_id": cid, "j": j, "t": t})
    return pd.DataFrame(rows)


def granger_fit(rows: pd.DataFrame, cycles: dict[int, Cycle], order: int,
                shifts: dict[int, int] | None) -> dict:
    if rows.empty:
        return {"W_att_to_poll": np.nan, "p_att_to_poll": np.nan, "W_poll_to_att": np.nan,
                "p_poll_to_att": np.nan, "n": 0, "sum_att_to_poll": np.nan}
    dx = {e: np.diff(c.shifted_x((shifts or {}).get(e, 0)), axis=0, prepend=np.nan)
          for e, c in cycles.items()}
    dy = {e: np.diff(c.polls, axis=0, prepend=np.nan) for e, c in cycles.items()}
    Y, Xd = gather(rows, dy), gather(rows, dx)
    Ylag = np.column_stack([gather(rows, dy, k) for k in range(1, order + 1)])
    Xlag = np.column_stack([gather(rows, dx, k) for k in range(1, order + 1)])
    ok = ~np.isnan(np.column_stack([Y, Xd, Ylag, Xlag])).any(axis=1)
    fe = rows["election"].to_numpy()[ok]
    groups = (rows["election"].astype(str) + ":" + rows["candidate_id"]).to_numpy()[ok]
    X1 = fe_design([*Ylag[ok].T, *Xlag[ok].T], fe)
    f1 = ols_cluster(Y[ok], X1, groups)
    w1, p1 = wald(f1, list(range(order, 2 * order)))
    X2 = fe_design([*Xlag[ok].T, *Ylag[ok].T], fe)
    f2 = ols_cluster(Xd[ok], X2, groups)
    w2, p2 = wald(f2, list(range(order, 2 * order)))
    s1 = float(f1[0][order:2 * order].sum()) if f1 is not None else np.nan
    return {"W_att_to_poll": w1, "p_att_to_poll": p1, "W_poll_to_att": w2,
            "p_poll_to_att": p2, "n": int(ok.sum()), "sum_att_to_poll": s1}


def granger_analysis(cycles, train, shift_sets) -> pd.DataFrame:
    out = []
    sub = {"commun": [e for e in train if e in cycles]} | {str(e): [e] for e in cycles}
    for order in GRANGER_ORDERS:
        for name, els in sub.items():
            cyc = {e: cycles[e] for e in els}
            rows = granger_rows(cyc, order)
            r = granger_fit(rows, cyc, order, None) | {"order": order, "sample": name}
            nulls = [granger_fit(rows, cyc, order, s) for s in shift_sets]
            for k in ("att_to_poll", "poll_to_att"):
                nw = np.array([z[f"W_{k}"] for z in nulls])
                r[f"p_placebo_{k}"] = placebo_p(r[f"W_{k}"], nw)
                r[f"size_{k}"] = float(np.nanmean(np.array([z[f"p_{k}"] for z in nulls]) < 0.05))
            out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- (a) corrélations


def xcorr(cycles: dict[int, Cycle], shifts: dict[int, int] | None) -> dict[int, np.ndarray]:
    out = {}
    for e, c in cycles.items():
        x = c.shifted_x((shifts or {}).get(e, 0))
        dx = np.diff(x, axis=0, prepend=np.nan)
        dy = np.diff(c.polls, axis=0, prepend=np.nan)
        r = []
        for k in XCORR_LAGS:
            dxk = np.full_like(dx, np.nan)
            if k >= 0:
                dxk[k:] = dx[:c.T - k]
            else:
                dxk[:k] = dx[-k:]
            ok = ~np.isnan(dy) & ~np.isnan(dxk)
            r.append(np.corrcoef(dy[ok], dxk[ok])[0, 1] if ok.sum() > 10 else np.nan)
        out[e] = np.array(r)
    return out


# ---------------------------------------------------------------------------- (a) épisodes


def monthly_polls(poll_shares: pl.DataFrame) -> pd.DataFrame:
    m = poll_shares.with_columns(pl.col("fieldwork_end").dt.truncate("1mo").alias("month"))
    by_p = m.group_by("election", "candidate_id", "month", "pollster").agg(pl.col("share").mean())
    return (by_p.group_by("election", "candidate_id", "month").agg(pl.col("share").mean())
            .sort("election", "candidate_id", "month").to_pandas())


def detect_episodes(monthly: pd.DataFrame, elections) -> pd.DataFrame:
    rows = []
    for (e, cid), g in monthly[monthly["election"].isin(elections)].groupby(
            ["election", "candidate_id"]):
        g = g.sort_values("month")
        mi = np.array([P.month_index(pd.Timestamp(m).date()) for m in g["month"]])
        s = g["share"].to_numpy()
        best = None
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                if (mi[b] - mi[a] <= EPISODE_MONTHS and s[b] - s[a] >= EPISODE_RISE
                        and (best is None or s[b] - s[a] > best[2])):
                    best = (a, b, s[b] - s[a])
        if best:
            a, b, rise = best
            rows.append({"election": e, "candidate_id": cid,
                         "month_start": pd.Timestamp(g["month"].iloc[a]).date(),
                         "month_end": pd.Timestamp(g["month"].iloc[b]).date(),
                         "poll_start": s[a], "poll_end": s[b], "rise": rise})
    return pd.DataFrame(rows)


def half_rise(weeks: list[dt.date], y: np.ndarray, lo: dt.date, hi: dt.date, level: float,
              interpolate: bool) -> dt.date | None:
    """Première semaine de [lo, hi] où y atteint `level` (y observé, ou interpolé linéairement
    entre semaines observées si `interpolate`)."""
    idx = [i for i, w in enumerate(weeks) if lo <= w <= hi]
    if not idx:
        return None
    yy = y[idx]
    if interpolate:
        ok = ~np.isnan(yy)
        if ok.sum() < 2:
            return None
        pos = np.arange(len(yy))
        yy = np.where(ok, yy, np.interp(pos, pos[ok], yy[ok]))
        first, last = np.flatnonzero(ok)[[0, -1]]
        yy[:first] = np.nan
        yy[last + 1:] = np.nan
    hit = np.flatnonzero(yy >= level)
    return weeks[idx[hit[0]]] if len(hit) else None


def episode_leads(episodes: pd.DataFrame, cycles: dict[int, Cycle]) -> pd.DataFrame:
    out = []
    for r in episodes.itertuples(index=False):
        c = cycles.get(r.election)
        if c is None or r.candidate_id not in c.cands:
            continue
        j = c.cands.index(r.candidate_id)
        lo = r.month_start
        hi = (pd.Timestamp(r.month_end) + pd.offsets.MonthEnd(0)).date()
        lo_att = (pd.Timestamp(lo) - pd.DateOffset(months=3)).date()
        poll_half = half_rise(c.weeks, c.polls[:, j], lo, hi,
                              (r.poll_start + r.poll_end) / 2, interpolate=True)
        x4 = rolling_mean(c.x, ATT_WINDOW)[:, j]
        base_idx = [i for i, w in enumerate(c.weeks)
                    if lo - dt.timedelta(weeks=8) <= w < lo]
        win_idx = [i for i, w in enumerate(c.weeks) if lo_att <= w <= hi]
        if not base_idx or not win_idx:
            continue
        a0, a1 = np.nanmean(x4[base_idx]), np.nanmax(x4[win_idx])
        att_half = half_rise(c.weeks, x4, lo_att, hi, (a0 + a1) / 2, interpolate=False)
        lead = ((poll_half - att_half).days / 7 if poll_half and att_half else np.nan)
        out.append({**r._asdict(), "att_base": a0, "att_peak": a1, "poll_half": poll_half,
                    "att_half": att_half, "lead_weeks": lead})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- (b) pics


def detect_peaks(views: np.ndarray) -> list[int]:
    peaks, last = [], -10**9
    lv = np.log1p(views)
    for t in range(PEAK_BASE, len(views)):
        base = views[t - PEAK_BASE:t]
        lb = lv[t - PEAK_BASE:t]
        sd = lb.std(ddof=1)
        z = (lv[t] - lb.mean()) / sd if sd > 0 else np.inf
        if (views[t] >= PEAK_RATIO * max(np.median(base), 1) and z >= PEAK_Z
                and views[t] >= PEAK_MIN_VIEWS and t - last > PEAK_GAP):
            peaks.append(t)
            last = t
    return peaks


def poll_move(p: np.ndarray, t: int) -> float:
    pre = _observed(p, t, (-1, -2, -3, -4))
    post = _observed(p, t, (1, 2, 3, 4))
    return np.nan if pre is None or post is None else p[post] - p[pre]


def peak_analysis(cycles: dict[int, Cycle], train, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = []
    for e, c in cycles.items():
        for j, cid in enumerate(c.cands):
            p = c.polls[:, j]
            if np.isnan(p).all():
                continue
            peaks = detect_peaks(c.views[:, j])
            near = set()
            for t in peaks:
                near |= set(range(t - PEAK_GAP, t + PEAK_GAP + 1))
            for t in peaks:
                mv = poll_move(p, t)
                pool = [u for u in range(max(0, t - NULL_RADIUS), min(c.T, t + NULL_RADIUS + 1))
                        if u not in near and not np.isnan(poll_move(p, u))]
                events.append({"election": e, "candidate_id": cid, "week": c.weeks[t],
                               "views": c.views[t, j],
                               "ratio": c.views[t, j] / max(np.median(c.views[t - PEAK_BASE:t, j]), 1),
                               "move": mv, "null_moves": [poll_move(p, u) for u in pool]})
    ev = pd.DataFrame(events)
    tests = []
    if ev.empty:
        return ev, pd.DataFrame()
    usable = ev[ev["move"].notna() & (ev["null_moves"].str.len() > 0)]
    samples = {"commun": [e for e in train if e in cycles]} | {str(e): [e] for e in cycles}
    for name, els in samples.items():
        u = usable[usable["election"].isin(els)]
        if u.empty:
            continue
        obs_abs, obs_signed = u["move"].abs().mean(), u["move"].mean()
        nulls = [np.array(v) for v in u["null_moves"]]
        draws = np.array([[z[rng.integers(len(z))] for z in nulls] for _ in range(N_PERM_PEAKS)])
        null_abs = np.abs(draws).mean(axis=1)
        tests.append({"sample": name, "peaks": int((ev["election"].isin(els)).sum()),
                      "usable": len(u), "mean_abs_move": obs_abs,
                      "null_mean_abs_move": float(null_abs.mean()),
                      "p_abs": float((1 + (null_abs >= obs_abs).sum()) / (1 + N_PERM_PEAKS)),
                      "mean_signed_move": obs_signed,
                      "share_down": float((u["move"] < 0).mean())})
    return ev.drop(columns="null_moves"), pd.DataFrame(tests)


# ---------------------------------------------------------------------------- (c) erreur finale


def final_attention(att: pl.DataFrame, reg, source: str, e: int) -> pd.DataFrame:
    """Part d'attention parmi les officiels : 14 derniers jours et 28 jours d'avant."""
    vote = reg.first_rounds[e]
    officials = [c.candidate_id for c in _official(e)]
    a = att.filter((pl.col("source") == source) & pl.col("candidate_id").is_in(officials))
    if a.is_empty():
        return pd.DataFrame()
    if a["freq"][0] == "D":
        last = (vote - dt.timedelta(days=FINAL_DAYS), vote - dt.timedelta(days=1))
        prev = (vote - dt.timedelta(days=FINAL_DAYS + MOMENTUM_DAYS),
                vote - dt.timedelta(days=FINAL_DAYS + 1))
    else:  # semaines Google (dimanche) entièrement avant le vote
        last = (vote - dt.timedelta(days=FINAL_DAYS + 6), vote - dt.timedelta(days=7))
        prev = (vote - dt.timedelta(days=FINAL_DAYS + MOMENTUM_DAYS + 6),
                vote - dt.timedelta(days=FINAL_DAYS + 7))
    def share(lo, hi):
        g = a.filter(pl.col("date").is_between(lo, hi)).group_by("candidate_id").agg(
            pl.col("value").sum())
        tot = g["value"].sum()
        return {r[0]: 100 * r[1] / tot for r in g.iter_rows()} if tot > 0 else {}
    s_last, s_prev = share(*last), share(*prev)
    return pd.DataFrame([{"candidate_id": c, "att_last": s_last.get(c, np.nan),
                          "att_prev": s_prev.get(c, np.nan)} for c in officials])


def _official(e: int):
    from vvs.registry import default_registry

    return default_registry().elections[e].candidates


def final_error_frame(att: pl.DataFrame, reg, errors: pl.DataFrame, source: str
                      ) -> pd.DataFrame:
    pe = errors.filter((pl.col("method") == "mean_14d") & (pl.col("level") == "candidate")
                       & (pl.col("round") == 1)).to_pandas()
    rows = []
    for e in TRAIN[source]:
        fa = final_attention(att, reg, source, e)
        if fa.empty:
            continue
        g = pe[pe["election"] == e].rename(columns={"unit": "candidate_id"})
        d = fa.merge(g[["candidate_id", "poll_share", "result_share"]], on="candidate_id")
        d["poll_norm"] = 100 * d["poll_share"] / d["poll_share"].sum()
        d["surperf"] = d["result_share"] - d["poll_share"]
        d["log_ratio"] = np.log(d["att_last"] / d["poll_norm"])
        d["gap"] = d["att_last"] - d["poll_norm"]
        d["momentum"] = d["att_last"] - d["att_prev"]
        d["election"] = e
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fe_beta(d: pd.DataFrame, x: str) -> float:
    dm = d.groupby("election")[[x, "surperf"]].transform(lambda s: s - s.mean())
    v = (dm[x] ** 2).sum()
    return float((dm[x] * dm["surperf"]).sum() / v) if v > 0 else np.nan


def final_error_tests(d: pd.DataFrame, rng) -> pd.DataFrame:
    out = []
    for x in ("log_ratio", "gap", "momentum"):
        for name, sub in (("tous", d),
                          ("sans sortant", d[~d.apply(lambda r: INCUMBENT_CANDIDATES.get(
                              r["election"]) == r["candidate_id"], axis=1)])):
            s = sub.dropna(subset=[x, "surperf"]).replace([np.inf, -np.inf], np.nan).dropna(
                subset=[x])
            if s["election"].nunique() == 0 or len(s) < 5:
                continue
            b = fe_beta(s, x)
            # permutation dans chaque élection : la moyenne d'élection ne change pas, donc
            # β permuté = Σ x̃ permuté · ỹ / Σ x̃² (x̃, ỹ centrés par élection)
            num = np.zeros(N_PERM_FINAL)
            den = 0.0
            for _, g in s.groupby("election"):
                xt = (g[x] - g[x].mean()).to_numpy()
                yt = (g["surperf"] - g["surperf"].mean()).to_numpy()
                order = np.argsort(rng.random((N_PERM_FINAL, len(xt))), axis=1)
                num += xt[order] @ yt
                den += float(xt @ xt)
            null = num / den
            per = {str(e): fe_beta(g, x) for e, g in s.groupby("election")}
            r = {"predictor": x, "sample": name, "beta": b, "n": len(s),
                 "p_perm": float((1 + (np.abs(null) >= abs(b)).sum()) / (1 + N_PERM_FINAL)),
                 "same_sign_all": bool(all(np.sign(v) == np.sign(b) for v in per.values()))}
            r |= {f"beta_{k}": v for k, v in per.items()}
            out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- figures


def fig_xcorr(xc: dict[int, np.ndarray], bands: dict[int, np.ndarray], out: Path, source: str):
    es = sorted(xc)
    fig, axes = plt.subplots(1, len(es), figsize=(4.2 * len(es), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, es):
        lo, hi = np.nanpercentile(bands[e], [2.5, 97.5], axis=0)
        ax.fill_between(XCORR_LAGS, lo, hi, color=GRID, label="placebo 95 %")
        ax.plot(XCORR_LAGS, xc[e], "o-", color=SERIES, ms=3, label="observé")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.5, ls=":")
        ax.set_title(str(e), color=INK)
        ax.set_xlabel("retard k (semaines) : attention en t − k")
        _style(ax)
    axes[0].set_ylabel("corrélation Δsondage(t), Δattention(t − k)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Corrélations croisées des variations hebdomadaires ({source})", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"xcorr_{source}.png", dpi=150)
    plt.close(fig)


def fig_episodes(leads: pd.DataFrame, cycles: dict[int, Cycle], out: Path, source: str):
    if leads.empty:
        return
    n = len(leads)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow), squeeze=False)
    for ax, r in zip(axes.flat, leads.itertuples(index=False)):
        c = cycles[r.election]
        j = c.cands.index(r.candidate_id)
        lo = pd.Timestamp(r.month_start) - pd.DateOffset(months=6)
        hi = pd.Timestamp(r.month_end) + pd.DateOffset(months=3)
        w = pd.to_datetime(pd.Series(c.weeks))
        m = (w >= lo) & (w <= hi)
        ax.plot(w[m], c.polls[m.to_numpy(), j], "o", color=RESULT, ms=3, label="sondages")
        ax2 = ax.twinx()
        ax2.plot(w[m], rolling_mean(c.x, ATT_WINDOW)[m.to_numpy(), j], color=SERIES, lw=1.2,
                 label="attention (4 sem.)")
        for d, col in ((r.poll_half, RESULT), (r.att_half, SERIES)):
            if d is not None and not pd.isna(d):
                ax.axvline(pd.Timestamp(d), color=col, ls="--", lw=0.8)
        ax.set_title(f"{display_name(r.candidate_id)} {r.election} : avance "
                     f"{'–' if np.isnan(r.lead_weeks) else f'{r.lead_weeks:+.0f}'} sem.",
                     fontsize=9, color=INK)
        ax.set_ylabel("sondages (%)", color=RESULT, fontsize=8)
        ax2.set_ylabel("part d'attention (pts)", color=SERIES, fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        _style(ax)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"episodes_{source}.png", dpi=150)
    plt.close(fig)


def fig_fillon(att: pl.DataFrame, weekly_polls: pl.DataFrame, out: Path):
    a = att.filter((pl.col("source") == "wikipedia") & (pl.col("candidate_id") == "fillon_francois")
                   & pl.col("date").is_between(dt.date(2016, 10, 1), dt.date(2017, 4, 23)))
    p = weekly_polls.filter((pl.col("election") == 2017)
                            & (pl.col("candidate_id") == "fillon_francois")
                            & pl.col("week").is_between(dt.date(2016, 10, 1), dt.date(2017, 4, 23)))
    if a.is_empty():
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(a["date"].to_list(), a["value"].to_list(), color=SERIES, width=1.0, alpha=0.6,
           label="pages vues quotidiennes")
    ax.set_ylabel("pages vues / jour", color=SERIES)
    ax2 = ax.twinx()
    ax2.plot(p["week"].to_list(), p["share"].to_list(), "o-", color=RESULT, ms=3,
             label="sondages (semaine)")
    ax2.set_ylabel("intentions de vote (%)", color=RESULT)
    for d, lab in ((dt.date(2016, 11, 27), "primaire"), (dt.date(2017, 1, 25), "Canard enchaîné"),
                   (dt.date(2017, 3, 1), "convocation des juges")):
        ax.axvline(d, color=MUTED, ls=":", lw=0.8)
        ax.text(d, ax.get_ylim()[1] * 0.95, lab, fontsize=7, color=MUTED, rotation=90,
                va="top", ha="right")
    _style(ax)
    ax.set_title("François Fillon, 2016-2017 : attention Wikipédia et sondages", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fillon.png", dpi=150)
    plt.close(fig)


def fig_final(d: pd.DataFrame, out: Path, source: str):
    if d.empty:
        return
    es = sorted(d["election"].unique())
    fig, axes = plt.subplots(1, len(es), figsize=(4.2 * len(es), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, es):
        g = d[d["election"] == e]
        ax.scatter(g["log_ratio"], g["surperf"], color=SERIES, s=18)
        for r in g.itertuples(index=False):
            ax.annotate(display_name(r.candidate_id).split()[-1], (r.log_ratio, r.surperf),
                        fontsize=7, color=MUTED, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(str(e), color=INK)
        ax.set_xlabel("log(part d'attention / part sondages), 14 derniers jours")
        _style(ax)
    axes[0].set_ylabel("résultat − sondages 14 jours (pts)")
    fig.tight_layout()
    fig.savefig(out / f"final_error_{source}.png", dpi=150)
    plt.close(fig)


def fig_2027(cycles: dict[int, Cycle], out: Path, source: str):
    c = cycles.get(2027)
    if c is None:
        return
    top = np.argsort(-(~np.isnan(c.polls)).sum(axis=0), kind="stable")[:8]  # les plus sondés
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), sharex=True)
    w = pd.to_datetime(pd.Series(c.weeks))
    x4 = rolling_mean(c.x, ATT_WINDOW)
    for ax, j in zip(axes.flat, top):
        ax.plot(w, x4[:, j], color=SERIES, lw=1, label="part d'attention (4 sem.)")
        ax.plot(w, c.polls[:, j], "o", color=RESULT, ms=3, label="sondages")
        ax.set_title(display_name(c.cands[j]), fontsize=9, color=INK)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        _style(ax)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Cycle 2027 : part d'attention ({source}) et sondages, points", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"cycle2027_{source}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- rapport


def f(v, fmt="{:+.3f}") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def fp(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "<0,001" if v < 0.001 else f"{v:.3f}".replace(".", ",")


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def coverage_table(att: pl.DataFrame, cycles_by_source: dict) -> list[list[str]]:
    rows = []
    for s, cycles in cycles_by_source.items():
        a = att.filter(pl.col("source") == s)
        for e, c in cycles.items():
            tested = int((~np.isnan(c.polls)).any(axis=0).sum())
            weeks_polled = int((~np.isnan(c.polls)).any(axis=1).sum())
            rows.append([s, str(e), f"{c.weeks[0]} → {c.weeks[-1]}", str(c.T), str(len(c.cands)),
                         str(tested), str(weeks_polled)])
        if not a.is_empty():
            rows.append([s, "tout", f"{a['date'].min()} → {a['date'].max()}", "–",
                         str(a["candidate_id"].n_unique()), "–", "–"])
    return rows


def run_source(source, reg, att, poll_shares, weekly_polls, errors, n_shifts, out, rng):
    cycles = build_cycles(reg, att, weekly_polls, source)
    if not cycles:
        return None
    train = [e for e in TRAIN[source] if e in cycles]
    shifts = shift_grid({e: cycles[e] for e in cycles}, n_shifts)
    res = {"cycles": cycles}
    res["anchor"], res["size"] = anchor_analysis(cycles, train, shifts)
    res["loco"] = leave_one_candidate_out(cycles, train)
    res["robust"] = anchor_robustness(cycles, train, shifts)
    res["granger"] = granger_analysis(cycles, train, shifts)
    xc = xcorr(cycles, None)
    bands = {e: np.array([xcorr({e: cycles[e]}, s)[e] for s in shifts]) for e in cycles}
    res["xcorr"], res["xcorr_bands"] = xc, bands
    monthly = monthly_polls(poll_shares)
    res["episodes"] = episode_leads(detect_episodes(monthly, list(cycles)), cycles)
    res["peaks"], res["peak_tests"] = (peak_analysis(cycles, train, rng) if source == "wikipedia"
                                       else (pd.DataFrame(), pd.DataFrame()))
    res["final"] = final_error_frame(att, reg, errors, source)
    res["final_tests"] = (final_error_tests(res["final"], rng) if not res["final"].empty
                          else pd.DataFrame())
    res["final_tests_holdout"] = pd.DataFrame()
    if source == "trends" and not res["final"].empty:
        hold = res["final"][res["final"]["election"].isin(HOLDOUT_FINAL)]
        if hold["election"].nunique() == len(HOLDOUT_FINAL):
            res["final_tests_holdout"] = final_error_tests(hold, rng)
    fig_xcorr(xc, bands, out, source)
    fig_episodes(res["episodes"], cycles, out, source)
    fig_final(res["final"], out, source)
    fig_2027(cycles, out, source)
    panel_frame(cycles).to_parquet(out / f"panel_{source}.parquet")
    for k in ("anchor", "size", "loco", "robust", "granger", "episodes", "peaks", "peak_tests", "final",
              "final_tests", "final_tests_holdout"):
        if isinstance(res[k], pd.DataFrame) and not res[k].empty:
            res[k].to_csv(out / f"{k}_{source}.csv", index=False)
    return res


def verdicts(r: dict) -> dict:
    a = r["anchor"]
    main = a[(a["h"] == H_MAIN) & a["control"] & (a["sample"] == "commun")].iloc[0]
    per = a[(a["h"] == H_MAIN) & a["control"] & a["sample"].isin(["2017", "2022", "2007", "2012"])]
    g = r["granger"]
    g2 = g[(g["order"] == GRANGER_ORDERS[0]) & (g["sample"] == "commun")].iloc[0]
    loco_stable = bool((np.sign(r["loco"]["beta"]) == np.sign(main["beta"])).all())
    ep = r["episodes"]
    lead_ok = ep["lead_weeks"].dropna()
    out = {
        "anchor_beta": main["beta"], "anchor_p_placebo": main.get("p_placebo"),
        "anchor_size": main.get("size_05"),
        "anchor_each_positive": bool((per["beta"] > 0).all()) if len(per) else False,
        "loco_stable": loco_stable,
        "granger_att_to_poll_p_placebo": g2["p_placebo_att_to_poll"],
        "granger_poll_to_att_p_placebo": g2["p_placebo_poll_to_att"],
        "episodes": len(ep),
        "episodes_attention_leads": int((lead_ok >= 2).sum()),
        "episodes_with_lead": len(lead_ok),
    }
    out["a_holds"] = bool(main["beta"] > 0 and (main.get("p_placebo") or 1) < 0.05
                          and out["anchor_each_positive"] and loco_stable
                          and (g2["p_placebo_att_to_poll"] or 1) < 0.05)
    pt = r["peak_tests"]
    if not pt.empty:
        com = pt[pt["sample"] == "commun"].iloc[0]
        each = pt[pt["sample"].isin(["2017", "2022"])]
        out["b_holds"] = bool(com["p_abs"] < 0.05 and (each["mean_abs_move"]
                                                       > each["null_mean_abs_move"]).all())
    ft = r["final_tests"]
    if not ft.empty:
        m = ft[(ft["predictor"] == "log_ratio") & (ft["sample"] == "tous")].iloc[0]
        out["c_holds"] = bool(m["p_perm"] < 0.05 and m["same_sign_all"])
    fh = r.get("final_tests_holdout", pd.DataFrame())
    if not fh.empty:
        m = fh[(fh["predictor"] == "momentum") & (fh["sample"] == "tous")].iloc[0]
        out["c_momentum_holdout_holds"] = bool(m["beta"] > 0 and m["p_perm"] < 0.05
                                               and m["same_sign_all"])
    return out


def write_report(out: Path, att: pl.DataFrame, results: dict, n_shifts: int, ingest_notes: str):
    L = ["# Attention en ligne et dynamique des sondages par candidat", ""]
    L += [(f"Run du {dt.date.today().isoformat()}, `back/studies/attention.py`. Protocole : "
           "docstring du script (critères fixés avant lecture), d'après docs/06-presse.md. "
           "Rien n'entre dans le modèle à ce stade."), ""]
    L += ["## Couverture", ""]
    L += md_table(["source", "cycle", "semaines", "T", "membres avec données",
                   "candidats sondés", "semaines sondées"],
                  coverage_table(att, {s: r["cycles"] for s, r in results.items()}))
    L += ["", ingest_notes, ""]
    for s, r in results.items():
        v = r["verdicts"]
        L += [f"## {s} — (a) l'attention précède-t-elle les sondages ?", ""]
        L += [(f"Spécification principale : y = sondage(t + h) − sondage(t), régressé sur la "
               "variation sur 4 semaines de la part d'attention (points) et sur l'élan des "
               "sondages ; effet fixe par cycle ; écarts types par grappe candidat × cycle ; "
               f"p placebo sur {n_shifts} décalages circulaires ; « taille » = part des "
               "décalages où le test nominal rejette à 5 %."), ""]
        rows = []
        for x in r["anchor"].itertuples(index=False):
            rows.append([str(x.h), x.sample, "oui" if x.control else "non", f(x.beta),
                         f(x.se, "{:.3f}"), fp(x.p), fp(getattr(x, "p_placebo", np.nan)),
                         f(getattr(x, "size_05", np.nan), "{:.2f}"), str(x.n), str(x.n_cand)])
        L += md_table(["h", "échantillon", "élan", "β (pts/pt)", "é.-t.", "p nominal",
                       "p placebo", "taille", "n", "candidats"], rows)
        rb = r["robust"]
        L += ["", ("Variantes de la spécification principale (h = 4, commun, avec élan), "
                   "**ajoutées après la première lecture** pour l'interpréter : attention "
                   "décalée d'une semaine (strictement antérieure à la semaine du sondage de "
                   "départ), sans les 5 % plus fortes variations d'attention, hors fenêtres "
                   "d'événements de configuration (primaires, déclarations, retraits : "
                   "`EVENTS`). Effet 1 é.-t. = β × écart type de la variation d'attention, en "
                   "points de sondage sur 4 semaines ; à comparer à l'écart type de y."), ""]
        L += md_table(["variante", "β", "é.-t.", "p nominal", "p placebo", "n",
                       "é.-t. Δattention", "effet 1 é.-t. (pts)", "é.-t. de y (pts)"],
                      [[x.variant, f(x.beta), f(x.se, "{:.3f}"), fp(x.p), fp(x.p_placebo),
                        str(x.n), f(x.sd_datt, "{:.2f}"), f(x.effect_sd, "{:+.2f}"),
                        f(x.sd_y, "{:.2f}")] for x in rb.itertuples(index=False)])
        lo = r["loco"]
        L += ["", (f"Retrait d'un candidat à la fois (h = {H_MAIN}, commun) : β de "
                   f"{f(lo['beta'].min())} à {f(lo['beta'].max())} ; signe stable : "
                   f"{'oui' if v['loco_stable'] else 'non'}. Détail : `loco_{s}.csv`."), ""]
        L += ["Granger (semaines consécutives sondées) :", ""]
        rows = []
        for x in r["granger"].itertuples(index=False):
            rows.append([str(x.order), x.sample, str(x.n), f(x.sum_att_to_poll),
                         fp(x.p_att_to_poll), fp(x.p_placebo_att_to_poll),
                         f(x.size_att_to_poll, "{:.2f}"), fp(x.p_poll_to_att),
                         fp(x.p_placebo_poll_to_att), f(x.size_poll_to_att, "{:.2f}")])
        L += md_table(["ordre", "échantillon", "n", "Σβ att→sond.", "p att→sond.",
                       "p placebo", "taille", "p sond.→att.", "p placebo", "taille"], rows)
        L += ["", f"Corrélations croisées : `xcorr_{s}.png`.", ""]
        ep = r["episodes"]
        L += ["Épisodes d'émergence (règle : +5 points en 6 mois au plus) :", ""]
        rows = [[str(x.election), display_name(x.candidate_id), f"{x.month_start} → {x.month_end}",
                 f"{x.poll_start:.1f} → {x.poll_end:.1f}", str(x.poll_half), str(x.att_half),
                 f(x.lead_weeks, "{:+.0f}")] for x in ep.itertuples(index=False)]
        L += md_table(["cycle", "candidat", "mois", "sondages", "mi-hausse sondages",
                       "mi-hausse attention", "avance (sem.)"], rows)
        L += ["", (f"L'attention mène d'au moins 2 semaines dans {v['episodes_attention_leads']} "
                   f"épisodes sur {v['episodes_with_lead']} datables. Figure : "
                   f"`episodes_{s}.png`."), ""]
        if not r["peak_tests"].empty:
            L += [f"## {s} — (b) pics d'attention et mouvements des sondages", ""]
            rows = [[x.sample, str(x.peaks), str(x.usable), f(x.mean_abs_move, "{:.2f}"),
                     f(x.null_mean_abs_move, "{:.2f}"), fp(x.p_abs), f(x.mean_signed_move, "{:+.2f}"),
                     f(x.share_down, "{:.2f}")] for x in r["peak_tests"].itertuples(index=False)]
            L += md_table(["échantillon", "pics", "avec sondages avant/après",
                           "|mouvement| moyen (pts)", "au hasard", "p", "mouvement moyen",
                           "part en baisse"], rows)
            pk = r["peaks"].dropna(subset=["move"]).sort_values("ratio", ascending=False).head(15)
            L += ["", "Plus forts pics (avec sondages avant et après) :", ""]
            L += md_table(["cycle", "candidat", "semaine", "vues", "× médiane 8 sem.",
                           "mouvement (pts)"],
                          [[str(x.election), display_name(x.candidate_id), str(x.week),
                            f"{x.views:,.0f}".replace(",", " "), f"{x.ratio:.1f}",
                            f(x.move, "{:+.1f}")] for x in pk.itertuples(index=False)])
            L += ["", "Cas Fillon : `fillon.png`.", ""]
        ft = r["final_tests"]
        if not ft.empty:
            L += [f"## {s} — (c) attention et erreur finale des sondages", ""]
            bcols = sorted(c for c in ft.columns if c.startswith("beta_"))
            rows = [[x["predictor"], x["sample"], str(x["n"]), f(x["beta"]), fp(x["p_perm"]),
                     *[f(x[c]) for c in bcols], "oui" if x["same_sign_all"] else "non"]
                    for _, x in ft.iterrows()]
            L += md_table(["prédicteur", "échantillon", "n", "β", "p permutation",
                           *[c.replace("beta_", "β ") for c in bcols], "même signe"], rows)
            L += ["", f"Figure : `final_error_{s}.png` ; données : `final_{s}.csv`.", ""]
            fh = r.get("final_tests_holdout", pd.DataFrame())
            if not fh.empty:
                m = fh[(fh["predictor"] == "momentum") & (fh["sample"] == "tous")].iloc[0]
                L += [(f"Test confirmatoire de l'élan sur 2007 et 2012 (déclaré avant lecture "
                       f"des données Trends) : β = {f(m['beta'])}, p permutation = "
                       f"{fp(m['p_perm'])}, β 2007 = {f(m.get('beta_2007', np.nan))}, "
                       f"β 2012 = {f(m.get('beta_2012', np.nan))} ; critère (β > 0 dans "
                       "chacune, p < 0,05) : "
                       f"{'rempli' if m['beta'] > 0 and m['p_perm'] < 0.05 and m['same_sign_all'] else 'non rempli'}."), ""]
        L += [f"Verdicts selon les critères fixés : `{json.dumps(v, default=str)}`", ""]
    L += ["## Conclusion", "", CONCLUSION, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


CONCLUSION = """\
*Rédigée au vu de la run du 2026-10-20 (Wikipédia et Trends) ; à relire si les données
changent.*

**Ce qui passe formellement, et pourquoi je ne le retiens pas.** Sur les pages vues
Wikipédia, la spécification principale (a) remplit le critère fixé d'avance : une hausse de la
part d'attention sur quatre semaines précède une hausse des sondages dans les quatre semaines
suivantes (β = +0,085 point de sondage par point de part d'attention, p placebo = 0,010,
positif en 2017 et en 2022, stable au retrait de chaque candidat, Granger attention →
sondages p placebo = 0,005). Mais l'effet est petit et tient à une poignée d'événements :
0,35 point pour un écart type de variation d'attention, face à un écart type de 1,8 point
des variations de sondage sur quatre semaines et à une erreur de 4 à 6 points à 90 et
180 jours. Il disparaît sans les 5 % plus fortes variations d'attention (β = +0,048,
p placebo = 0,56) et hors des fenêtres de primaires, déclarations et retraits (β = +0,034,
p placebo = 0,37) ; avec l'attention de la semaine précédente seulement, il tombe à
p placebo = 0,11. Les corrélations croisées culminent au retard d'une semaine, pas au-delà.
Lecture : l'attention et les sondages réagissent aux mêmes événements (primaire de la
droite, primaire de la gauche, congrès LR de Pécresse, entrée de Zemmour), l'attention le
jour même, les sondages une à trois semaines plus tard parce qu'ils sont rares et publiés
avec retard. C'est un indicateur avancé de la *mesure*, pas de l'*opinion*. Le Granger
inverse (sondages → attention) passe aussi (p placebo = 0,025) : la causalité va dans les
deux sens. En 2027 le signe s'inverse (β = −0,058 à 4 semaines). Google Trends ne réplique
pas : β = +0,004 (p placebo = 0,96) sur 2007-2027, positif mais non significatif en 2007 et
2012 (seules fenêtres où l'échelle Trends est propre), signe instable d'un candidat à l'autre.

**Émergences.** Macron fin 2016 : son attention monte avec la démission du gouvernement
(août) et la déclaration (novembre), quatre à cinq mois avant les sondages de janvier-février
2017, mais elle retombe entre les deux : un signal qu'on n'aurait pas su lire comme une
montée. Zemmour 2021 : attention et sondages montent la même semaine (septembre). Bayrou
2007 et 2012, Mélenchon 2012 et 2017 (Trends) : l'attention accompagne les sondages ou les
suit d'une à deux semaines.
Sur les épisodes détectés par la règle, l'attention mène de deux semaines ou plus dans 10
sur 18 (Wikipédia) et 12 sur 27 (Trends), soit à peu près une fois sur deux, et les grandes
avances (Macron 2018, Juppé et Sarkozy 2015, Ruffin 2023) relient un pic d'actualité à une
hausse de sondage sans rapport évident. Pas d'émergence annoncée de façon exploitable.

**Pics et chocs (b).** Les mouvements de sondage autour des pics d'attention sont plus
grands qu'au hasard sur 2017-2022 réunis (1,44 point contre 1,19, p = 0,026), mais pas en
2017 (1,50 contre 1,55) : critère non rempli. L'attention ne dit pas le sens : 77 % des
mouvements après un pic sont des hausses (déclarations, primaires), et l'affaire Fillon,
le choc négatif le plus net, produit un pic d'attention modeste (40 000 vues par jour, contre
330 000 pour sa victoire à la primaire) la semaine même où les sondages commencent à baisser.

**Erreur finale (c).** Rien sur le prédicteur principal (log du rapport attention /
sondages : signes opposés en 2017 et 2022, p = 0,87) : l'attention des deux dernières
semaines va aux petits candidats que le public découvre (Lassalle, Cheminade, Arthaud).
L'élan d'attention ressortait à la marge sur Wikipédia (Mélenchon et Pécresse 2022) ; le
test confirmatoire déclaré avant de lire Trends échoue sur 2007 et 2012 (β 2012 = −0,003,
p = 0,25).

**Recommandation : ne pas tester au backtest.** Ni dérive μ_c(τ), ni mise à l'échelle de
l'erreur candidat. Le seul effet est un rattrapage de mesure à une à quatre semaines, de
quelques dixièmes de point, déjà absorbé par le modèle dès que les sondages suivants
arrivent, et nul aux horizons de 180 et 90 jours qui fondent la règle d'admission. Si Gaétan
veut malgré tout un essai, une seule forme a un sens : μ_c(τ) = γ · Δatt_c (variation sur
quatre semaines de la part d'attention Wikipédia), appliquée sur les quatre premières
semaines de la projection puis éteinte, γ estimé en laissant l'élection évaluée de côté
(environ 0,08) ; amplitude attendue inférieure à 0,5 point, gain de log-score attendu nul.
Usage plus utile hors prévision : un témoin d'actualité pour le site (pic d'attention = « les
sondages n'ont peut-être pas encore vu cet événement »), et la collecte quotidienne continue
pour 2027. Résultat négatif à consigner dans docs/06-presse.md ou un document dédié."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-attention")
    ap.add_argument("--shifts", type=int, default=200, help="décalages placebo par test")
    ap.add_argument("--sources", default="wikipedia,trends")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    reg, att, poll_shares, weekly_polls, errors = load_inputs()
    fig_fillon(att, weekly_polls, out)
    results = {}
    for s in args.sources.split(","):
        r = run_source(s, reg, att, poll_shares, weekly_polls, errors, args.shifts, out, rng)
        if r is None:
            print(f"{s} : pas de données, ignoré")
            continue
        r["verdicts"] = verdicts(r)
        results[s] = r
        print(s, json.dumps(r["verdicts"], default=str))
    notes = (out / "ingest_notes.md").read_text(encoding="utf-8") if (
        out / "ingest_notes.md").exists() else ""
    write_report(out, att, results, args.shifts, notes)
    (out / "summary.json").write_text(json.dumps({s: r["verdicts"] for s, r in results.items()},
                                                 indent=2, default=str), encoding="utf-8")
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
