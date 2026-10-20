"""Étude phase 6 : la conjoncture a-t-elle fait bouger les intentions de vote par bloc ?

    uv run python studies/economy.py [--out runs/<date>-economy] [--placebo-step 1]

Protocole de docs/05-economie.md, partie analyse (rien n'entre dans le modèle ici) :

1. panel mensuel (vvs.studies.panel) : part moyenne de chaque bloc dans les sondages du mois
   contre 13 séries macro et de popularité, en niveau et en variation, retards de 0 à 6 mois ;
2. régression par bloc des variations mensuelles de part sur les variations d'un indicateur
   (un retard à la fois, puis les 7 retards ensemble), effets fixes par cycle électoral,
   erreurs HAC par cycle (Bartlett, 3 retards ; un cluster par cycle en complément) ; test
   de Granger dans les deux sens ;
3. robustesse : 12 derniers mois avant le vote, retrait d'un cycle à la fois, niveaux,
   popularité du président en contrôle ; plus un placebo par décalage circulaire des séries
   macro, qui mesure combien le choix du meilleur retard parmi 7 gonfle la significativité ;
4. niveau contre dynamique : une observation par élection (niveau à environ 6 mois du vote,
   dérive de la campagne jusqu'au résultat) ;
5. amplitude : points de part par écart type de la variation de l'indicateur sur 1, 3 et 6
   mois, face à l'erreur des sondages (2,6 points à 7 jours, 4 à 6 points à 90 et 180 jours).

Critères fixés avant de lire les résultats :

- une paire (bloc, indicateur) « ressort » si le test joint des 7 retards a p < 0,05, ou si
  son meilleur retard a p < 0,05 après Bonferroni sur les 7 retards (p brut < 0,0071) ;
- elle « tient » si, en plus, le signe du meilleur retard ne change dans aucun retrait de
  cycle, reste le même sur les 12 derniers mois, en niveau et avec le contrôle de
  popularité, et si le placebo donne p < 0,05.

Les paires a priori (`A_PRIORI`, question de Gaétan et littérature) sont tabulées quoi
qu'il arrive.
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
import statsmodels.api as sm
from scipy import stats

from vvs import paths
from vvs.ingest.macro.registry import load as load_macro_registry
from vvs.model.plots import GRID, INK, MUTED, POINTS, RESULT, SERIES, _style
from vvs.registry import default_registry
from vvs.studies import panel as P

LAGS = tuple(range(7))
HAC_LAGS = 3
HAC_LAGS_LEVEL = 6
GRANGER_ORDERS = (1, 3)
START_WINDOW = (240, 120)  # sondages terminés entre 240 et 120 jours avant le vote (~6 mois)
POLL_ERROR = {"7 jours": 2.6, "90 jours": 4.0, "180 jours": 6.0}
PLACEBO_MIN_SHIFT = 24
CAMPAIGN_MONTHS = 12

BLOC_LABEL = {
    "far_left": "Gauche radicale", "left": "Gauche", "centre": "Centre", "right": "Droite",
    "far_right": "Extrême droite", "incumbent": "Bloc du président sortant",
}
A_PRIORI = [
    ("far_right", "unemployment_rate_ilo"), ("right", "unemployment_rate_ilo"),
    ("incumbent", "unemployment_rate_ilo"), ("far_right", "jobseekers_cat_a"),
    ("right", "jobseekers_cat_a"), ("incumbent", "jobseekers_cat_a"),
    ("far_right", "cpi_all"), ("incumbent", "cpi_all"),
    ("incumbent", "consumer_confidence"), ("incumbent", "approval_president"),
]
CYCLE_COLORS = {2002: "#8a8984", 2007: "#9b6bd6", 2012: "#e34948", 2017: "#2a78d6",
                2022: "#eb6834", 2027: "#1f9e74"}


# ---------------------------------------------------------------------------- données


def build(out: Path):
    reg = default_registry()
    first_round = {e: el.round_(1).date for e, el in reg.elections.items()}
    by_bloc = pl.read_parquet(paths.CURATED / "poll_results_by_bloc.parquet")
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    macro = pl.read_parquet(paths.CURATED / "macro.parquet")

    poll_shares = P.poll_bloc_shares(by_bloc, polls)
    panel = P.bloc_panel(P.monthly_bloc_shares(poll_shares), first_round)
    panel = panel.with_columns(
        [pl.col("d_share").shift(j).over("election", "bloc").alias(f"d_share_l{j}")
         for j in range(1, max(GRANGER_ORDERS) + 1)]
    )
    ind = P.indicators(P.monthly_macro(macro),
                       P.officeholder_changes(load_macro_registry().officeholders))
    ids = [i for i in P.INDICATORS if i in ind.columns]
    data = join_panel(panel, ind, ids)

    panel.write_parquet(out / "panel_blocs.parquet")
    ind.write_parquet(out / "indicators_monthly.parquet")
    return reg, first_round, poll_shares, panel, ind, ids, data


def join_panel(panel: pl.DataFrame, ind: pl.DataFrame, ids: list[str]) -> pd.DataFrame:
    lagged = P.with_lags(ind, ids + [f"d_{i}" for i in ids], LAGS)
    data = panel.join(lagged, on="month", how="left").to_pandas()
    return data.sort_values(["bloc", "election", "month"]).reset_index(drop=True)


def shifted(ind: pl.DataFrame, shift: int) -> pl.DataFrame:
    """Toutes les séries décalées circulairement de `shift` mois (placebo)."""
    return ind.select("month", *[pl.Series(c, np.roll(ind[c].to_numpy(), shift))
                                 for c in ind.columns if c != "month"])


# ---------------------------------------------------------------------------- régressions


def fe_fit(df: pd.DataFrame, y: str, xs: list[str], *, cov: str = "hac",
           maxlags: int = HAC_LAGS):
    """MCO avec un effet fixe par cycle ; None si l'échantillon est trop petit."""
    d = df[["election", "month", y, *xs]].dropna().sort_values(["election", "month"])
    n_cycles = d["election"].nunique()
    if len(d) < len(xs) + n_cycles + 5 or d[xs].std().min() == 0:
        return None, d
    fe = pd.get_dummies(d["election"], prefix="c", dtype=float)
    X = pd.concat([d[xs], fe], axis=1)
    groups = pd.factorize(d["election"])[0]
    model = sm.OLS(d[y].astype(float), X.astype(float))
    if cov == "hac":
        res = model.fit(cov_type="hac-panel",
                        cov_kwds={"groups": groups, "maxlags": maxlags, "use_correction": "hac"})
    elif cov == "cluster":
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    else:
        raise ValueError(cov)
    return res, d


def coef(res, x: str) -> dict:
    if res is None:
        return {"beta": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    lo, hi = res.conf_int().loc[x]
    return {"beta": res.params[x], "se": res.bse[x], "ci_lo": lo, "ci_hi": hi,
            "p": res.pvalues[x]}


def joint_wald(res, xs: list[str]) -> tuple[float, float, float, float]:
    """(chi2, p) du test joint, somme des coefficients et son écart type."""
    b = res.params[xs].to_numpy()
    V = res.cov_params().loc[xs, xs].to_numpy()
    w = float(b @ np.linalg.solve(V, b))
    ones = np.ones(len(xs))
    return w, float(stats.chi2.sf(w, len(xs))), float(b.sum()), float(np.sqrt(ones @ V @ ones))


def bh(p: pd.Series) -> pd.Series:
    """q-valeurs de Benjamini-Hochberg (NaN conservés)."""
    q = pd.Series(np.nan, index=p.index)
    ok = p.dropna().sort_values()
    m = len(ok)
    if m == 0:
        return q
    vals = ok.to_numpy() * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q.loc[ok.index] = np.minimum(vals, 1.0)
    return q


def main_regressions(data: pd.DataFrame, ind: pl.DataFrame, ids: list[str],
                     cluster: bool = True):
    sd = {}
    indf = ind.sort("month")
    for i in ids:
        sd[i] = {
            "sd_d1": float(indf[f"d_{i}"].std()),
            "sd_d3": float(P.cumulative_change(indf, i, 3).std()),
            "sd_d6": float(P.cumulative_change(indf, i, 6).std()),
            "sd_level": float(indf[i].std()),
        }
    by_lag, pairs = [], []
    for b in P.BLOCS:
        db = data[data["bloc"] == b]
        for i in ids:
            for k in LAGS:
                x = f"d_{i}_l{k}"
                res, d = fe_fit(db, "d_share", [x])
                resc = fe_fit(db, "d_share", [x], cov="cluster")[0] if cluster else None
                c = coef(res, x)
                by_lag.append({"bloc": b, "indicator": i, "lag": k, **c,
                               "p_cluster": coef(resc, x)["p"], "n": len(d),
                               "n_cycles": d["election"].nunique()})
            xs = [f"d_{i}_l{k}" for k in LAGS]
            res, d = fe_fit(db, "d_share", xs)
            if res is None:
                w = pj = s = ss = np.nan
            else:
                w, pj, s, ss = joint_wald(res, xs)
            pairs.append({"bloc": b, "indicator": i, "joint_chi2": w, "p_joint": pj,
                          "sum_beta": s, "sum_beta_se": ss, "n_joint": len(d), **sd[i]})
    by_lag = pd.DataFrame(by_lag)
    pairs = pd.DataFrame(pairs)
    best = (by_lag.dropna(subset=["p"]).sort_values("p")
            .groupby(["bloc", "indicator"], as_index=False).first())
    best = best.rename(columns={c: f"best_{c}" for c in
                                ["lag", "beta", "se", "ci_lo", "ci_hi", "p", "p_cluster", "n",
                                 "n_cycles"]})
    pairs = pairs.merge(best, on=["bloc", "indicator"], how="left")
    pairs["p_best_bonf"] = np.minimum(pairs["best_p"] * len(LAGS), 1.0)
    pairs["q_joint"] = bh(pairs["p_joint"])
    pairs["q_best_bonf"] = bh(pairs["p_best_bonf"])
    pairs["emerges"] = (pairs["p_joint"] < 0.05) | (pairs["p_best_bonf"] < 0.05)
    for h, col in (("1m", "sd_d1"), ("3m", "sd_d3"), ("6m", "sd_d6")):
        pairs[f"effect_sd_{h}"] = pairs["best_beta"] * pairs[col]
    return by_lag, pairs


# ---------------------------------------------------------------------------- placebo


def _demeaned_t(y: np.ndarray, x: np.ndarray, g: np.ndarray) -> tuple[float, int] | None:
    """(t, degrés de liberté) de la pente en MCO à effets fixes par groupe, ou None."""
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x, g = y[ok], x[ok], g[ok]
    ug, inv = np.unique(g, return_inverse=True)
    if len(y) < 10 or len(y) <= len(ug) + 3:
        return None
    ym = y - (np.bincount(inv, y) / np.bincount(inv))[inv]
    xm = x - (np.bincount(inv, x) / np.bincount(inv))[inv]
    sxx = xm @ xm
    if sxx <= 1e-12:
        return None
    beta = (xm @ ym) / sxx
    resid = ym - beta * xm
    df = len(y) - len(ug) - 1
    return float(beta / np.sqrt(resid @ resid / df / sxx)), df


def _min_p_shifted(y, g, mi, x, shift: int) -> float:
    """Meilleur p (MCO à effets fixes) sur les 7 retards, série décalée de `shift` mois."""
    best = 1.0
    for k in LAGS:
        r = _demeaned_t(y, x[(mi - k - shift) % len(x)], g)
        if r is not None:
            best = min(best, 2 * stats.t.sf(abs(r[0]), r[1]))
    return best


def _abs_t_shifted(y, g, mi, x, lag: int, shift: int) -> float:
    r = _demeaned_t(y, x[(mi - lag - shift) % len(x)], g)
    return np.nan if r is None else abs(r[0])


def _panel_arrays(data: pd.DataFrame, bloc: str, ycol: str, m0: int):
    db = data[(data["bloc"] == bloc) & data[ycol].notna()]
    mi = np.array([P.month_index(m) for m in pd.to_datetime(db["month"]).dt.date]) - m0
    return db[ycol].to_numpy(), db["election"].to_numpy(), mi


def placebo(data: pd.DataFrame, ind: pl.DataFrame, ids: list[str], step: int) -> pd.DataFrame:
    """Décalage circulaire de chaque série macro de s mois (|s| >= 24) : distribution du
    meilleur p sur 7 retards sous l'hypothèse nulle, autocorrélations conservées."""
    m0, N = P.month_index(ind["month"][0]), ind.height
    rows = []
    for b in P.BLOCS:
        y, g, mi = _panel_arrays(data, b, "d_share", m0)
        for i in ids:
            x = ind[f"d_{i}"].to_numpy().astype(float)
            obs = _min_p_shifted(y, g, mi, x, 0)
            null = np.array([_min_p_shifted(y, g, mi, x, s)
                             for s in range(PLACEBO_MIN_SHIFT, N - PLACEBO_MIN_SHIFT, step)])
            rows.append({"bloc": b, "indicator": i, "ols_min_p": obs, "n_shifts": len(null),
                         "p_placebo": (1 + (null <= obs).sum()) / (1 + len(null))})
    return pd.DataFrame(rows)


def hac_size(panel: pl.DataFrame, ind: pl.DataFrame, ids: list[str], step: int
             ) -> pd.DataFrame:
    """Refait les tests HAC (retard par retard, joint, critère « ressort », Granger) sur des
    séries macro décalées circulairement : sous l'hypothèse nulle, quelle part rejette ?"""
    N = ind.height
    rows = []
    for sh in range(PLACEBO_MIN_SHIFT, N - PLACEBO_MIN_SHIFT, step):
        d = join_panel(panel, shifted(ind, sh), ids)
        by_lag, pairs = main_regressions(d, ind, ids, cluster=False)
        gr = granger(d, pairs)
        rows.append({
            "shift": sh,
            "single_p05": float((by_lag["p"] < 0.05).mean()),
            "joint_p05": float((pairs["p_joint"] < 0.05).mean()),
            "emerges": float(pairs["emerges"].mean()),
            "q_joint_10": float((pairs["q_joint"] < 0.10).mean()),
            "granger_x_to_bloc": float(((gr["p_x_to_bloc_1"] < 0.05)
                                        | (gr["p_x_to_bloc_3"] < 0.05)).mean()),
            "granger_bloc_to_x": float(((gr["p_bloc_to_x_1"] < 0.05)
                                        | (gr["p_bloc_to_x_3"] < 0.05)).mean()),
        })
    return pd.DataFrame(rows)


def level_placebo(data: pd.DataFrame, ind: pl.DataFrame, rob: pd.DataFrame) -> pd.Series:
    """p placebo de la spécification en niveau (c), au retard retenu : rang de |t| à la vraie
    date parmi les décalages circulaires de la série en niveau."""
    m0, N = P.month_index(ind["month"][0]), ind.height
    out = []
    for r in rob.itertuples(index=False):
        y, g, mi = _panel_arrays(data, r.bloc, "share", m0)
        x = ind[r.indicator].to_numpy().astype(float)
        obs = _abs_t_shifted(y, g, mi, x, r.lag, 0)
        null = np.array([_abs_t_shifted(y, g, mi, x, r.lag, s)
                         for s in range(PLACEBO_MIN_SHIFT, N - PLACEBO_MIN_SHIFT)])
        null = null[~np.isnan(null)]
        out.append(np.nan if np.isnan(obs) else (1 + (null >= obs).sum()) / (1 + len(null)))
    return pd.Series(out, index=rob.index)


# ---------------------------------------------------------------------------- Granger


def granger(data: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b, i in pairs[["bloc", "indicator"]].itertuples(index=False):
        db = data[data["bloc"] == b]
        row = {"bloc": b, "indicator": i}
        for p in GRANGER_ORDERS:
            ylags = [f"d_share_l{j}" for j in range(1, p + 1)]
            xlags = [f"d_{i}_l{j}" for j in range(1, p + 1)]
            # indicateur -> bloc
            res, d = fe_fit(db, "d_share", ylags + xlags)
            row[f"p_x_to_bloc_{p}"] = np.nan if res is None else joint_wald(res, xlags)[1]
            row[f"n_{p}"] = len(d)
            # bloc -> indicateur (sur les mêmes mois)
            res, d = fe_fit(db.dropna(subset=["d_share"] + ylags), f"d_{i}_l0", xlags + ylags)
            row[f"p_bloc_to_x_{p}"] = np.nan if res is None else joint_wald(res, ylags)[1]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- robustesse


def robustness(data: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in pairs.itertuples(index=False):
        b, i, k = r.bloc, r.indicator, int(r.best_lag)
        db = data[data["bloc"] == b]
        x = f"d_{i}_l{k}"
        out = {"bloc": b, "indicator": i, "lag": k, "beta": r.best_beta, "p": r.best_p}
        res, d = fe_fit(db[db["months_to_election"] <= CAMPAIGN_MONTHS], "d_share", [x])
        c = coef(res, x)
        out |= {"camp_beta": c["beta"], "camp_lo": c["ci_lo"], "camp_hi": c["ci_hi"],
                "camp_p": c["p"], "camp_n": len(d)}
        loo = []
        for e in sorted(db["election"].unique()):
            res, _ = fe_fit(db[db["election"] != e], "d_share", [x])
            loo.append((e, coef(res, x)["beta"], coef(res, x)["p"]))
        betas = np.array([v for _, v, _ in loo], dtype=float)
        out |= {"loo_min": np.nanmin(betas), "loo_max": np.nanmax(betas),
                "loo_sign_stable": bool(np.all(np.sign(betas[~np.isnan(betas)])
                                               == np.sign(r.best_beta))),
                "loo_detail": "; ".join(f"sans {e} : {v:+.3f} (p={p:.2f})" for e, v, p in loo)}
        xl = f"{i}_l{k}"
        res, d = fe_fit(db, "share", [xl], maxlags=HAC_LAGS_LEVEL)
        c = coef(res, xl)
        out |= {"level_beta": c["beta"], "level_lo": c["ci_lo"], "level_hi": c["ci_hi"],
                "level_p": c["p"], "level_n": len(d)}
        if i == "approval_president":
            out |= {"ctrl_beta": np.nan, "ctrl_lo": np.nan, "ctrl_hi": np.nan, "ctrl_p": np.nan}
        else:
            ctrl = sorted({"d_approval_president_l0", f"d_approval_president_l{k}"})
            res, d = fe_fit(db, "d_share", [x, *ctrl])
            c = coef(res, x)
            out |= {"ctrl_beta": c["beta"], "ctrl_lo": c["ci_lo"], "ctrl_hi": c["ci_hi"],
                    "ctrl_p": c["p"], "ctrl_n": len(d)}
        s = np.sign(r.best_beta)
        out["holds_sign_checks"] = bool(
            out["loo_sign_stable"]
            and np.sign(out["camp_beta"]) == s
            and np.sign(out["level_beta"]) == s
            and (np.isnan(out["ctrl_beta"]) or np.sign(out["ctrl_beta"]) == s)
        )
        rows.append(out)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- niveau / dynamique


def election_level(reg, first_round, poll_shares: pl.DataFrame, ind: pl.DataFrame,
                   ids: list[str]) -> pd.DataFrame:
    """Une ligne par (élection, bloc) : part à ~6 mois, résultat, indicateurs."""
    lv = ind.sort("month").with_columns(
        [pl.col(i).forward_fill(limit=6).alias(i) for i in ids]
    ).to_pandas().set_index("month")
    lv.index = pd.to_datetime(lv.index)
    rows = []
    for e, vote in sorted(first_round.items()):
        lo, hi = vote - dt.timedelta(days=START_WINDOW[0]), vote - dt.timedelta(days=START_WINDOW[1])
        w = poll_shares.filter((pl.col("election") == e)
                               & pl.col("fieldwork_end").is_between(lo, hi))
        start = (w.group_by("bloc", "pollster").agg(pl.col("share").mean())
                 .group_by("bloc").agg(pl.col("share").mean(), pl.len().alias("n_pollsters")))
        start = {r["bloc"]: r for r in start.iter_rows(named=True)}
        el = reg.elections[e]
        result = {}
        r1 = el.round_(1)
        for c in r1.results:
            bl = el.candidacy(c.candidate_id).bloc
            result[bl] = result.get(bl, 0.0) + c.share
        if result:
            result[P.INCUMBENT] = result.get(P.INCUMBENT_BLOC[e], np.nan)
        vm = pd.Timestamp(vote.replace(day=1))
        m6, m18, m1 = (vm - pd.DateOffset(months=k) for k in (6, 18, 1))
        n_polls = w.select("poll_id").n_unique()
        for b in P.BLOCS:
            row = {"election": e, "bloc": b,
                   "start_share": start.get(b, {}).get("share", np.nan),
                   "start_pollsters": start.get(b, {}).get("n_pollsters", 0),
                   "start_polls": n_polls,
                   "result_share": result.get(b, np.nan) if result else np.nan}
            for i in ids:
                x6 = lv[i].get(m6, np.nan)
                row[f"{i}__start"] = x6
                row[f"{i}__chg12"] = x6 - lv[i].get(m18, np.nan)
                row[f"{i}__campaign"] = lv[i].get(m1, np.nan) - x6 if e < 2027 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df["drift"] = df["result_share"] - df["start_share"]
    return df


def level_regressions(lvl: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    specs = [("start_share", "start", "niveau à ~6 mois ~ indicateur à 6 mois"),
             ("drift", "chg12", "dérive campagne ~ variation sur 12 mois connue à 6 mois"),
             ("drift", "campaign", "dérive campagne ~ variation pendant la campagne")]
    rows = []
    for b in P.BLOCS:
        d = lvl[lvl["bloc"] == b]
        for i in ids:
            for y, suffix, label in specs:
                dd = d[[y, f"{i}__{suffix}"]].dropna()
                if len(dd) < 4 or dd.iloc[:, 1].std() == 0:
                    continue
                sl = stats.linregress(dd.iloc[:, 1], dd[y])
                rows.append({"bloc": b, "indicator": i, "spec": suffix, "label": label,
                             "n": len(dd), "slope": sl.slope, "r": sl.rvalue, "p": sl.pvalue,
                             "elections": ",".join(str(e) for e in d.loc[dd.index, "election"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- figures


def fig_coverage(panel: pl.DataFrame, path: Path) -> None:
    p = panel.filter(pl.col("bloc") == "far_left").to_pandas()
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for e, g in p.groupby("election"):
        ax.bar(pd.to_datetime(g["month"]), g["n_polls"], width=25, color=CYCLE_COLORS[e],
               label=str(e))
    _style(ax)
    ax.set_ylabel("sondages par mois", color=MUTED, fontsize=8)
    ax.set_title("Couverture : sondages de premier tour retenus, par mois et par cycle",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_heatmap(pairs: pd.DataFrame, path: Path) -> None:
    t = pairs.assign(t=pairs["best_beta"] / pairs["best_se"])
    mat = t.pivot(index="indicator", columns="bloc", values="t").reindex(
        index=list(P.INDICATORS), columns=list(P.BLOCS))
    pj = pairs.pivot(index="indicator", columns="bloc", values="p_joint").reindex_like(mat)
    em = pairs.pivot(index="indicator", columns="bloc", values="emerges").reindex_like(mat)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
    for (r, c), v in np.ndenumerate(mat.to_numpy(dtype=float)):
        if np.isnan(v):
            continue
        mark = "*" if bool(em.iloc[r, c]) else ""
        ax.text(c, r, f"{v:+.1f}{mark}\np={pj.iloc[r, c]:.2f}", ha="center", va="center",
                fontsize=6.5, color=INK)
    ax.set_xticks(range(len(mat.columns)), [BLOC_LABEL[b] for b in mat.columns], fontsize=7,
                  rotation=20)
    ax.set_yticks(range(len(mat.index)), [P.INDICATORS[i].label for i in mat.index], fontsize=7)
    ax.set_title("t du meilleur retard (variations, effets fixes par cycle, HAC)\n"
                 "p = test joint des 7 retards ; * = ressort (critère fixé d'avance)",
                 fontsize=9, color=INK, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bloc(b: str, row, data: pd.DataFrame, ind: pl.DataFrame, first_round, path: Path):
    i, k = row.indicator, int(row.best_lag)
    meta = P.INDICATORS[i]
    db = data[data["bloc"] == b]
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, :])
    obs = db[db["share"].notna()]
    for e, g in obs.groupby("election"):
        ax.scatter(pd.to_datetime(g["month"]), g["share"], s=8 + 6 * np.sqrt(g["n_polls"]),
                   color=CYCLE_COLORS[e], alpha=0.8, label=f"cycle {e}", zorder=3)
    for e, v in first_round.items():
        ax.axvline(pd.Timestamp(v), color=GRID, lw=1, ls="--", zorder=1)
    ax.set_ylabel(f"{BLOC_LABEL[b]} (% exprimés)", color=MUTED, fontsize=8)
    _style(ax)
    ax2 = ax.twinx()
    ip = ind.to_pandas()
    ax2.plot(pd.to_datetime(ip["month"]), ip[i], color=POINTS, lw=1)
    ax2.set_ylabel(f"{meta.label} ({meta.unit})", color=POINTS, fontsize=8)
    ax2.tick_params(colors=POINTS, labelsize=8)
    for side in ("top",):
        ax2.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper left")
    ax.set_title(f"{BLOC_LABEL[b]} : part mensuelle des sondages (points, taille ~ nombre de "
                 f"sondages) et « {meta.label} » (ligne grise, niveau)",
                 fontsize=9, color=INK, loc="left")

    ax3 = fig.add_subplot(gs[1, 0])
    x = f"d_{i}_l{k}"
    d = db[["election", "d_share", x]].dropna()
    dm = d.groupby("election")[["d_share", x]].transform(lambda s: s - s.mean())
    for e in sorted(d["election"].unique()):
        sel = d["election"] == e
        ax3.scatter(dm.loc[sel, x], dm.loc[sel, "d_share"], s=14, color=CYCLE_COLORS[e],
                     label=str(e))
    xs = np.linspace(dm[x].min(), dm[x].max(), 10)
    ax3.plot(xs, row.best_beta * xs, color=RESULT, lw=1.5)
    ax3.axhline(0, color=GRID, lw=0.8)
    _style(ax3)
    ax3.set_xlabel(f"variation « {meta.label} » il y a {k} mois (centrée par cycle)",
                   color=MUTED, fontsize=8)
    ax3.set_ylabel("variation mensuelle de part (centrée)", color=MUTED, fontsize=8)
    ax3.set_title(
        f"pente {row.best_beta:+.3f} [{row.best_ci_lo:+.3f} ; {row.best_ci_hi:+.3f}], "
        f"p={row.best_p:.3f}, n={int(row.best_n)}\n"
        f"meilleur retard parmi 7 ; test joint p={row.p_joint:.2f}",
        fontsize=8, color=INK, loc="left")

    ax4 = fig.add_subplot(gs[1, 1])
    lag_rows = row.lag_table
    ax4.errorbar(lag_rows["lag"], lag_rows["beta"],
                 yerr=[lag_rows["beta"] - lag_rows["ci_lo"], lag_rows["ci_hi"] - lag_rows["beta"]],
                 fmt="o", color=SERIES, ms=4, capsize=2)
    ax4.axhline(0, color=MUTED, lw=0.8)
    _style(ax4)
    ax4.set_xlabel("retard (mois)", color=MUTED, fontsize=8)
    ax4.set_title("coefficient par retard, IC 95 %", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- rapport


def f(v, fmt="{:+.3f}") -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def fp(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "<0,001" if v < 0.001 else f"{v:.3f}".replace(".", ",")


def lbl(i: str) -> str:
    return P.INDICATORS[i].label


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


CONCLUSION = """\
*Rédigée au vu de la run du 2026-10-20 ; à relire si les données changent.*

**Ce qui tient.** {holds}

**Ce qui ne tient pas.** {fails}

**Ce que je recommande pour le backtest de la phase 6.** {reco}
"""


def write_report(out: Path, cov: pl.DataFrame, pairs: pd.DataFrame, by_lag: pd.DataFrame,
                 gr: pd.DataFrame, rob: pd.DataFrame, lvl: pd.DataFrame, lreg: pd.DataFrame,
                 focus: pd.DataFrame, best_by_bloc: dict, size: pd.DataFrame,
                 conclusion: dict) -> None:
    L = ["# Économie et intentions de vote par bloc (phase 6, analyse)", ""]
    L += [(f"Run du {dt.date.today().isoformat()}, `back/studies/economy.py`. Protocole : "
          "`docs/05-economie.md`. Aucune variable n'entre dans le modèle à ce stade."), ""]
    L += ["## Données et choix", "",
          ("- Parts de bloc : premier tour, base exprimés, hypothèses suspectes exclues. Un "
          "sondage = moyenne de ses hypothèses ; un bloc absent d'une hypothèse est absent, pas "
          "nul. Un mois = moyenne des instituts présents (fin de terrain dans le mois), chaque "
          "institut comptant pour la moyenne de ses sondages du mois (un rolling ne pèse pas "
          "plus qu'un sondage ponctuel)."),
          ("- Mois sans sondage : laissés vides, rien n'est interpolé. Une variation mensuelle "
          "n'existe qu'entre deux mois consécutifs observés tous les deux."),
          ("- 2027 : sondages jusqu'au 16 octobre 2026 ; les hypothèses avec Emmanuel Macron "
          "(octobre 2022 à octobre 2023) sont écartées, il ne peut pas se représenter. Le cycle "
          "2002 commence en 2000 (les séries macro commencent en janvier 2000)."),
          ("- Pseudo-bloc « président sortant » : droite en 2002 (Chirac, malgré la cohabitation), "
          "2007 et 2012 ; gauche en 2017 ; centre en 2022 et 2027. Le bloc « autres » est "
          "laissé de côté (présence trop variable selon les hypothèses)."),
          ("- Chômage BIT trimestriel : valeur placée au mois central du trimestre, **interpolée "
          "linéairement** entre deux trimestres (ce qui donne au mois t une information du "
          "trimestre suivant : regarder les retards d'au moins 2 mois pour une lecture "
          "prédictive). Carburants hebdomadaires : moyenne du mois."),
          ("- Transformations : prix (IPC, énergie, alimentation, carburants) en glissement annuel "
          "(%) pour le niveau, variation = différence mensuelle du glissement annuel (les indices "
          "bruts ne sont pas CVS) ; demandeurs d'emploi : niveau en milliers jusqu'en décembre "
          "2024, variation = % mensuel jusqu'en décembre 2024 ; rien après la rupture de janvier "
          "2025 (loi plein emploi, labellisation suspendue, reclassements comme le -5 % d'avril "
          "2025). Ce dernier point a été écarté après l'avoir vu dans la figure de l'extrême "
          "droite, où il renforçait la pente : choix conservateur, signalé ici ; enquêtes et popularité : "
          "niveau tel quel, variation = différence mensuelle. Les variations de popularité du "
          "mois de prise de fonction d'un nouveau titulaire et du mois suivant sont vides."),
          ("- Régression : variation mensuelle de part sur la variation de l'indicateur retardée "
          "de k mois (k = 0 à 6), un effet fixe par cycle, erreurs HAC par cycle (Bartlett, "
          "3 retards ; 6 en niveau). La p-valeur « cluster » (un cluster par cycle, loi de "
          "Student à 5 degrés de liberté) est donnée en complément : avec 6 grappes, elle est "
          "fragile. Les mois sont consécutifs dans la grille mais les variations observées ne le "
          "sont pas toujours ; le HAC traite les lignes successives d'un cycle comme voisines."),
          ("- Placebo : chaque série macro est décalée circulairement de s mois (|s| ≥ 24, "
          "toutes les valeurs), on recalcule le meilleur p sur les 7 retards (MCO à effets fixes, "
          "écarts types classiques) ; p placebo = part des décalages qui font aussi bien que la "
          "vraie date. Il corrige le choix du meilleur retard et l'autocorrélation des séries."),
          ""]
    L += ["## Couverture", "",
          ("La série est creuse hors campagne avant 2017, et encore en 2017-2022 loin du vote : "
          "l'essentiel des variations mensuelles vient des 12 derniers mois de chaque cycle."), ""]
    L += md_table(["cycle", "premier mois", "mois de la grille", "mois avec sondage",
                   "dont 12 derniers mois", "variations mensuelles", "dont 12 derniers mois",
                   "sondages"],
                  [[str(r["election"]), str(r["first_month"]), str(r["grid_months"]),
                    str(r["months_with_polls"]), str(r["months_with_polls_last12"]),
                    str(r["monthly_changes"]), str(r["monthly_changes_last12"]), str(r["polls"])]
                   for r in cov.iter_rows(named=True)])
    tot = cov["monthly_changes"].sum()
    L += ["", (f"Au total {tot} variations mensuelles par bloc, réparties sur 6 cycles. "
          "Figure : `coverage.png`."), ""]

    n_pairs = len(pairs)
    n_em = int(pairs["emerges"].sum())
    L += ["## Coefficients par bloc et indicateur", "",
          (f"{n_pairs} paires (6 blocs × 13 indicateurs) × 7 retards = {len(by_lag)} "
          "régressions. Sous l'hypothèse nulle, on attend environ "
          f"{0.05 * len(by_lag):.0f} p < 0,05 au hasard sur les retards pris un à un, et "
          f"{0.05 * n_pairs:.0f} paires avec un test joint à p < 0,05, si les tests avaient "
          "leur taille nominale (ce n'est pas le cas, voir la section suivante). Observé : "
          f"{int((by_lag['p'] < 0.05).sum())} régressions à p < 0,05, "
          f"{int((pairs['p_joint'] < 0.05).sum())} tests joints à p < 0,05, "
          f"{n_em} paires qui ressortent selon le critère fixé d'avance, "
          f"{int((pairs['q_joint'] < 0.10).sum())} avec q de Benjamini-Hochberg < 0,10 sur le "
          f"test joint, {int((pairs['p_placebo'] < 0.05).sum())} avec un placebo à p < 0,05 "
          f"(attendu au hasard : {0.05 * n_pairs:.0f}), "
          f"{int((pairs['q_placebo'] < 0.10).sum())} avec q de Benjamini-Hochberg < 0,10 sur le "
          "placebo."), "",
          ("Lecture : β = points de part de bloc par unité de variation mensuelle de l'indicateur, "
          "au meilleur retard (choisi a posteriori : son p brut est optimiste, voir p Bonferroni "
          "et placebo). Effet 1 é.-t. = β × écart type de la variation mensuelle (points par "
          "mois) ; 6 mois = β × écart type de la variation sur 6 mois, soit la dérive cumulée "
          "qu'impliquerait un choc typique de l'indicateur à 180 jours."), ""]
    rows = []
    for r in pairs.sort_values(["bloc", "p_joint"]).itertuples(index=False):
        rows.append([BLOC_LABEL[r.bloc], lbl(r.indicator), str(int(r.best_lag)),
                     f"{f(r.best_beta)} [{f(r.best_ci_lo)} ; {f(r.best_ci_hi)}]",
                     fp(r.best_p), fp(r.p_best_bonf), fp(r.best_p_cluster), fp(r.p_joint),
                     fp(r.q_joint), fp(r.p_placebo), fp(r.q_placebo),
                     f(r.effect_sd_1m, "{:+.2f}"),
                     f(r.effect_sd_6m, "{:+.2f}"), str(int(r.best_n)),
                     "**oui**" if r.emerges else ""])
    L += md_table(["bloc", "indicateur", "retard", "β [IC 95 %]", "p", "p Bonf. 7",
                   "p cluster", "p joint 0-6", "q joint", "p placebo", "q placebo",
                   "effet 1 é.-t. (pts/mois)",
                   "effet 6 mois (pts)", "n", "ressort"], rows)
    L += ["", ("Tableau complet par retard : `coef_by_lag.parquet` ; par paire : "
          "`coef_pairs.parquet`. Vue d'ensemble : `summary_heatmap.png`."), ""]

    L += ["## Taille réelle des tests HAC (diagnostic)", "",
          (f"Les mêmes tests refaits sur les séries macro décalées circulairement de 24 mois et "
          f"plus ({len(size)} décalages, tous les {size['shift'].diff().median():.0f} mois) : "
          "toute relation y est fortuite, donc la part de rejets à 5 % mesure la taille réelle "
          "du test sur ces données."), ""]
    obs = {
        "single_p05": float((by_lag["p"] < 0.05).mean()),
        "joint_p05": float((pairs["p_joint"] < 0.05).mean()),
        "emerges": float(pairs["emerges"].mean()),
        "q_joint_10": float((pairs["q_joint"] < 0.10).mean()),
        "granger_x_to_bloc": float(((gr["p_x_to_bloc_1"] < 0.05)
                                    | (gr["p_x_to_bloc_3"] < 0.05)).mean()),
        "granger_bloc_to_x": float(((gr["p_bloc_to_x_1"] < 0.05)
                                    | (gr["p_bloc_to_x_3"] < 0.05)).mean()),
    }
    names = {"single_p05": ("retard par retard, p < 0,05", "5 %"),
             "joint_p05": ("test joint des 7 retards, p < 0,05", "5 %"),
             "emerges": ("paires qui « ressortent » (critère fixé d'avance)", "≈ 10 %"),
             "q_joint_10": ("test joint, q BH < 0,10", "≈ 0 %"),
             "granger_x_to_bloc": ("Granger indicateur → bloc, p < 0,05 à 1 ou 3 retards",
                                   "≈ 10 %"),
             "granger_bloc_to_x": ("Granger bloc → indicateur, p < 0,05 à 1 ou 3 retards",
                                   "≈ 10 %")}
    pct = lambda v: f"{100 * v:.0f} %"
    L += md_table(["test", "taille nominale", "sous le nul (médiane des décalages)",
                   "sous le nul (90e centile)", "vraies dates"],
                  [[names[k][0], names[k][1], pct(size[k].median()),
                    pct(size[k].quantile(0.9)), pct(obs[k])] for k in names])
    L += ["", ("Lecture : si la colonne « vraies dates » ne dépasse pas nettement la distribution "
          "sous le nul, le nombre de résultats significatifs ne dit rien. Le placebo par paire "
          "(colonne p placebo plus haut) est la référence ; les p HAC sont trop optimistes."), ""]

    L += ["## Tests de Granger", "",
          ("Chaque direction avec 1 et 3 retards, effets fixes par cycle, Wald HAC. "
          "« indicateur → bloc » : les variations passées de l'indicateur aident-elles à prévoir "
          "la variation de part, au-delà des variations passées de la part ? « bloc → "
          "indicateur » : l'inverse, sur les mêmes mois. Il faut 2 (resp. 4) mois consécutifs "
          "avec sondages, d'où des n faibles. Paires examinées : celles qui ressortent et les "
          "paires a priori ; toutes les paires sont dans `granger.parquet`."), ""]
    g = gr.merge(focus[["bloc", "indicator", "why"]], on=["bloc", "indicator"])
    L += md_table(["bloc", "indicateur", "pourquoi", "ind. → bloc (1)", "ind. → bloc (3)",
                   "bloc → ind. (1)", "bloc → ind. (3)", "n (1)", "n (3)"],
                  [[BLOC_LABEL[r.bloc], lbl(r.indicator), r.why, fp(r.p_x_to_bloc_1),
                    fp(r.p_x_to_bloc_3), fp(r.p_bloc_to_x_1), fp(r.p_bloc_to_x_3),
                    str(int(r.n_1)), str(int(r.n_3))] for r in g.itertuples(index=False)])
    sig_g = gr[(gr["p_x_to_bloc_1"] < 0.05) | (gr["p_x_to_bloc_3"] < 0.05)]
    sig_r = gr[(gr["p_bloc_to_x_1"] < 0.05) | (gr["p_bloc_to_x_3"] < 0.05)]
    L += ["", (f"Sur les {len(gr)} paires : {len(sig_g)} ont un test indicateur → bloc à p < 0,05 "
          f"(1 ou 3 retards), {len(sig_r)} un test bloc → indicateur à p < 0,05 ; on en attend "
          f"environ {0.0975 * len(gr):.0f} par direction au hasard (deux tests à 5 %)."), ""]

    L += ["## Robustesse", "",
          ("Au meilleur retard de chaque paire : (a) 12 derniers mois avant le vote ; (b) retrait "
          "d'un cycle à la fois (min et max de β, signe stable ou non) ; (c) en niveau (part du "
          "mois sur le niveau de l'indicateur retardé, effets fixes par cycle, HAC 6) ; (d) avec "
          "la variation de popularité du président (même mois et même retard) en contrôle. "
          "Le niveau a son propre placebo (décalages circulaires de la série en niveau, |t| au "
          "même retard), parce qu'une régression de niveaux persistants rejette trop souvent. "
          "« Tient » = ressort, signe stable partout, placebo p < 0,05."), ""]
    rr = rob.merge(focus[["bloc", "indicator", "why"]], on=["bloc", "indicator"]).merge(
        pairs[["bloc", "indicator", "p_placebo", "emerges"]], on=["bloc", "indicator"])
    rows = []
    for r in rr.itertuples(index=False):
        holds = bool(r.emerges and r.holds_sign_checks and r.p_placebo < 0.05)
        rows.append([BLOC_LABEL[r.bloc], lbl(r.indicator), r.why, str(r.lag),
                     f"{f(r.beta)} (p {fp(r.p)})",
                     f"{f(r.camp_beta)} [{f(r.camp_lo)} ; {f(r.camp_hi)}] p {fp(r.camp_p)}",
                     f"{f(r.loo_min)} à {f(r.loo_max)} {'stable' if r.loo_sign_stable else '**change**'}",
                     f"{f(r.level_beta)} [{f(r.level_lo)} ; {f(r.level_hi)}] p {fp(r.level_p)}",
                     fp(r.level_p_placebo),
                     f"{f(r.ctrl_beta)} [{f(r.ctrl_lo)} ; {f(r.ctrl_hi)}] p {fp(r.ctrl_p)}",
                     fp(r.p_placebo), "**oui**" if holds else "non"])
    L += md_table(["bloc", "indicateur", "pourquoi", "retard", "β principal",
                   "(a) 12 derniers mois", "(b) retrait d'un cycle", "(c) niveau",
                   "(c) p placebo", "(d) contrôle popularité", "p placebo", "tient"], rows)
    L += ["", "Détail des retraits de cycle : `robustness.parquet` (colonne `loo_detail`).", ""]

    L += ["## Niveau contre dynamique (une observation par élection)", "",
          (f"Niveau de départ : moyenne des instituts sur les sondages terminés entre "
          f"{START_WINDOW[0]} et {START_WINDOW[1]} jours avant le premier tour (2027 : jusqu'au "
          "16 octobre 2026, fenêtre incomplète). Indicateur au mois du vote moins 6 (dernière "
          "valeur connue dans les 6 mois précédents si le mois manque). Dérive = résultat − "
          "niveau de départ (2002 à 2022). Six points au plus pour le niveau, cinq pour la "
          "dérive : on lit le signe et l'ordre de grandeur de r, pas la p-valeur."), ""]
    lv_show = lvl[["election", "bloc", "start_share", "start_pollsters", "result_share", "drift"]]
    rows = []
    for e, gdf in lv_show.groupby("election"):
        d = gdf.set_index("bloc")
        rows.append([str(e)] + [
            f"{f(d.loc[b, 'start_share'], '{:.1f}')} → {f(d.loc[b, 'result_share'], '{:.1f}')}"
            for b in P.BLOCS] + [str(int(d["start_pollsters"].max()))])
    L += md_table(["élection"] + [BLOC_LABEL[b] for b in P.BLOCS] + ["instituts"], rows)
    L += [""]
    for spec, title in (("start", ("r(niveau à ~6 mois, indicateur à 6 mois), n = 6 (5 pour "
                                  "les demandeurs d'emploi)")),
                        ("chg12", ("r(dérive de campagne, variation sur 12 mois connue à 6 mois), "
                                  "n = 5")),
                        ("campaign", ("r(dérive de campagne, variation de l'indicateur pendant la "
                                     "campagne), n = 5"))):
        m = lreg[lreg["spec"] == spec].pivot(index="indicator", columns="bloc", values="r")
        m = m.reindex(index=[i for i in P.INDICATORS if i in m.index], columns=list(P.BLOCS))
        L += [f"**{title}**", ""]
        L += md_table(["indicateur"] + [BLOC_LABEL[b] for b in P.BLOCS],
                      [[lbl(i)] + [f(v, "{:+.2f}") for v in m.loc[i]] for i in m.index])
        L += [""]
    big = lreg[(lreg["r"].abs() >= 0.9)]
    L += [(f"Avec 5 ou 6 points, |r| ≥ 0,9 arrive souvent par hasard ({len(big)} cas sur "
          f"{len(lreg)} ici ; sous l'hypothèse nulle, P(|r| ≥ 0,9) vaut environ 0,04 à n = 5 et "
          "0,015 à n = 6, soit plusieurs cas attendus sur ce nombre de tests). Détail : "
          "`election_level.parquet`, `election_level_regressions.parquet`."), ""]

    L += ["## Amplitude face à l'erreur des sondages", "",
          "Pour les paires examinées : dérive de part impliquée par un choc d'un écart type de "
          "l'indicateur, cumulé sur 1, 3 et 6 mois (β × écart type de la variation sur 1, 3, 6 "
          "mois, séries 2000-2026). À comparer à l'erreur des sondages : "
          + ", ".join(f"{v:.1f} points à {k}".replace(".", ",") for k, v in POLL_ERROR.items())
          + ".", ""]
    rows = []
    for r in pairs.merge(focus[["bloc", "indicator", "why"]], on=["bloc", "indicator"]).itertuples(
            index=False):
        e6 = abs(r.effect_sd_6m)
        verdict = ("négligeable (< 0,5 pt)" if e6 < 0.5 else
                   "faible face à 4-6 pts" if e6 < 2 else "comparable à l'erreur des sondages")
        rows.append([BLOC_LABEL[r.bloc], lbl(r.indicator), r.why,
                     f"{f(r.sd_d1, '{:.3g}')} / {f(r.sd_d3, '{:.3g}')} / {f(r.sd_d6, '{:.3g}')}",
                     f(r.effect_sd_1m, "{:+.2f}"), f(r.effect_sd_3m, "{:+.2f}"),
                     f(r.effect_sd_6m, "{:+.2f}"), verdict])
    L += md_table(["bloc", "indicateur", "pourquoi", "é.-t. variation 1/3/6 mois",
                   "1 mois (pts)", "3 mois (pts)", "6 mois (pts)", "lecture"], rows)
    L += ["", ("Ces amplitudes supposent que β est vrai ; pour les paires qui ne tiennent pas, "
          "ce sont des bornes de ce qu'on pourrait gagner, pas des effets."), ""]

    L += ["## Figures", ""]
    for b, (i, k) in best_by_bloc.items():
        L += [(f"- `bloc_{b}.png` : {BLOC_LABEL[b]} contre « {lbl(i)} » (indicateur au plus "
              f"petit p, retard {k} mois).")]
    L += ["- `summary_heatmap.png` : t du meilleur retard pour chaque paire.",
          "- `coverage.png` : sondages par mois.", ""]

    L += ["## Conclusion", "", CONCLUSION.format(**conclusion)]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-economy")
    ap.add_argument("--placebo-step", type=int, default=1,
                    help="pas des décalages du placebo (1 = tous)")
    ap.add_argument("--size-step", type=int, default=12,
                    help="pas des décalages pour la taille empirique des tests HAC")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    reg, first_round, poll_shares, panel, ind, ids, data = build(out)
    cov = P.coverage(panel)
    cov.write_parquet(out / "coverage.parquet")
    fig_coverage(panel, out / "coverage.png")

    by_lag, pairs = main_regressions(data, ind, ids)
    plc = placebo(data, ind, ids, args.placebo_step)
    pairs = pairs.merge(plc, on=["bloc", "indicator"], how="left")
    pairs["q_placebo"] = bh(pairs["p_placebo"])
    by_lag.to_parquet(out / "coef_by_lag.parquet")
    pairs.to_parquet(out / "coef_pairs.parquet")
    fig_heatmap(pairs, out / "summary_heatmap.png")

    gr = granger(data, pairs)
    gr.to_parquet(out / "granger.parquet")

    em = pairs[pairs["emerges"]][["bloc", "indicator"]].assign(why="ressort")
    ap_ = pd.DataFrame(A_PRIORI, columns=["bloc", "indicator"]).assign(why="a priori")
    focus = (pd.concat([em, ap_]).groupby(["bloc", "indicator"], as_index=False)
             .agg(why=("why", lambda s: " + ".join(sorted(set(s))))))
    all_rob = robustness(data, pairs)
    all_rob["level_p_placebo"] = level_placebo(data, ind, all_rob)
    all_rob.to_parquet(out / "robustness_all_pairs.parquet")
    rob = all_rob.merge(focus[["bloc", "indicator"]], on=["bloc", "indicator"])
    rob.to_parquet(out / "robustness.parquet")

    size = hac_size(panel, ind, ids, args.size_step)
    size.to_parquet(out / "hac_size.parquet")

    lvl = election_level(reg, first_round, poll_shares, ind, ids)
    lvl.to_parquet(out / "election_level.parquet")
    lreg = level_regressions(lvl, ids)
    lreg.to_parquet(out / "election_level_regressions.parquet")

    best_by_bloc = {}
    for b in P.BLOCS:
        pb = pairs[pairs["bloc"] == b].sort_values("best_p").iloc[0]
        best_by_bloc[b] = (pb.indicator, int(pb.best_lag))
        lag_table = by_lag[(by_lag["bloc"] == b) & (by_lag["indicator"] == pb.indicator)]
        row = pb.copy()
        row["lag_table"] = lag_table
        fig_bloc(b, row, data, ind, first_round, out / f"bloc_{b}.png")

    holds = all_rob.merge(pairs[["bloc", "indicator", "emerges", "p_placebo"]],
                          on=["bloc", "indicator"])
    holds = holds[holds["emerges"] & holds["holds_sign_checks"] & (holds["p_placebo"] < 0.05)]
    summary = {
        "n_pairs": len(pairs), "n_regressions": len(by_lag),
        "n_p05_single": int((by_lag["p"] < 0.05).sum()),
        "n_joint_p05": int((pairs["p_joint"] < 0.05).sum()),
        "n_emerges": int(pairs["emerges"].sum()),
        "n_q10_joint": int((pairs["q_joint"] < 0.10).sum()),
        "n_placebo_p05": int((pairs["p_placebo"] < 0.05).sum()),
        "emerges": pairs[pairs["emerges"]][["bloc", "indicator", "best_lag", "best_beta",
                                            "best_p", "p_joint", "p_placebo"]].to_dict("records"),
        "holds": holds[["bloc", "indicator", "lag", "beta"]].to_dict("records"),
        "placebo_p05": pairs[pairs["p_placebo"] < 0.05][["bloc", "indicator", "best_lag",
                                                         "best_beta", "p_placebo"]
                                                        ].to_dict("records"),
        "best_by_bloc": best_by_bloc,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str,
                                                 ensure_ascii=False))
    write_report(out, cov, pairs, by_lag, gr, rob, lvl, lreg, focus, best_by_bloc, size,
                 conclusion_text(pairs, all_rob, size, lreg, cov, gr, covid_check(data)))
    print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    print(f"écrit {out}")


def covid_check(data: pd.DataFrame) -> dict:
    """Extrême droite ~ demandeurs d'emploi (retard 1) sans les mois de mars 2020 à juillet
    2021, où la série bouge de 3 à 20 % par mois."""
    db = data[data["bloc"] == "far_right"]
    m = pd.to_datetime(db["month"])
    x = "d_jobseekers_cat_a_l1"
    out = {}
    for name, d in (("all", db), ("no_covid", db[~m.between("2020-03-01", "2021-07-01")])):
        res, used = fe_fit(d, "d_share", [x])
        out[name] = coef(res, x) | {"n": len(used)}
    return out


def conclusion_text(pairs: pd.DataFrame, rob: pd.DataFrame, size: pd.DataFrame,
                    lreg: pd.DataFrame, cov: pl.DataFrame, gr: pd.DataFrame,
                    covid: dict) -> dict:
    """Conclusion rédigée à la main, chiffres relus dans les tableaux de la run."""
    def pr(b, i):
        return pairs.set_index(["bloc", "indicator"]).loc[(b, i)]

    def rb(b, i):
        return rob.set_index(["bloc", "indicator"]).loc[(b, i)]

    def r_el(b, i, spec):
        m = lreg[(lreg["bloc"] == b) & (lreg["indicator"] == i) & (lreg["spec"] == spec)]
        return float(m["r"].iloc[0]) if len(m) else np.nan

    def num(v, fmt="{:.2f}"):
        return fmt.format(v).replace(".", ",")

    ru, fu = pr("right", "unemployment_rate_ilo"), pr("far_right", "unemployment_rate_ilo")
    fj, ff = pr("far_right", "jobseekers_cat_a"), pr("far_right", "cpi_food")
    ia, iar = pr("incumbent", "approval_president"), rb("incumbent", "approval_president")
    hits = int((pairs["p_placebo"] < 0.05).sum())
    g = gr.set_index(["bloc", "indicator"])
    last12 = cov["monthly_changes_last12"].sum() / cov["monthly_changes"].sum()
    joint_obs = (pairs["p_joint"] < 0.05).mean()
    holds = (
        "Presque rien, et rien d'économique. Le seul lien net est un témoin positif, non "
        "économique et en niveau seulement : dans un même cycle, le bloc du président "
        "sortant est plus haut les mois où le président est plus populaire "
        f"(+{num(iar['level_beta'])} point de part par point de satisfaits Ifop, placebo en "
        f"niveau p = {num(iar['level_p_placebo'], '{:.3f}')}), et le même signe se voit d'une "
        f"élection à l'autre (r = {num(r_el('incumbent', 'approval_president', 'start'), '{:+.2f}')} "
        "entre la popularité à six mois du vote et le niveau du bloc sortant, six points). "
        "C'est le résultat classique de Lewis-Beck et Nadeau, mais il est contemporain : en "
        "variations, la popularité ne précède pas les intentions de vote (placebo "
        f"p = {num(ia['p_placebo'])}), et rien ne dit qu'elle ne fait pas que refléter la même "
        "opinion que les sondages. Formellement, une paire économique passe le critère fixé "
        "d'avance : l'inflation alimentaire et l'extrême droite (retard de "
        f"{int(ff['best_lag'])} mois, β = {num(ff['best_beta'])}, placebo "
        f"p = {num(ff['p_placebo'], '{:.3f}')}). Je ne la retiens pas : son signe va contre "
        "l'intuition (une accélération des prix alimentaires suivie d'un recul de l'extrême "
        f"droite), elle est nulle en niveau, et avec {len(pairs)} paires on attend environ "
        f"{0.05 * len(pairs):.0f} placebos à p < 0,05 par hasard pour {hits} observés "
        f"(q de Benjamini-Hochberg minimal {num(pairs['q_placebo'].min())})."
    )
    fails = (
        "Tout le reste, y compris la question de départ. Les p-valeurs HAC donnent "
        f"l'illusion d'effets nombreux ({int((pairs['p_joint'] < 0.05).sum())} tests joints "
        "à p < 0,05 sur 78), mais refaits sur des séries macro décalées de deux ans ou plus, "
        f"les mêmes tests rejettent dans {num(100 * size['joint_p05'].median(), '{:.0f}')} % "
        f"des cas en médiane, contre {num(100 * joint_obs, '{:.0f}')} % aux vraies dates : la "
        "taille réelle du test, pas un signal. Avec "
        f"{int(cov['monthly_changes'].sum())} variations mensuelles par bloc, dont "
        f"{num(100 * last12, '{:.0f}')} % dans les douze derniers mois avant le vote, des mois "
        "à un seul sondage et des "
        "changements d'hypothèses de candidats, le panel ne peut pas isoler un effet de "
        "quelques dixièmes de point par mois. Chômage : quand le chômage BIT monte, la droite "
        f"baisse trois mois plus tard (β = {num(ru['best_beta'], '{:.1f}')} point par point de "
        f"chômage, placebo p = {num(ru['p_placebo'])}) et l'extrême droite monte "
        f"(β = {num(fu['best_beta'], '{:+.1f}')}, placebo p = {num(fu['p_placebo'])}), signes "
        "stables quand on retire un cycle et sur les douze derniers mois, mais nuls en niveau "
        "(placebo en niveau p = "
        f"{num(rb('right', 'unemployment_rate_ilo')['level_p_placebo'])} et "
        f"{num(rb('far_right', 'unemployment_rate_ilo')['level_p_placebo'])}), peu clairs au "
        "test de Granger (chômage → droite p = "
        f"{num(g.loc[('right', 'unemployment_rate_ilo'), 'p_x_to_bloc_3'])}, → extrême droite "
        f"p = {num(g.loc[('far_right', 'unemployment_rate_ilo'), 'p_x_to_bloc_3'], '{:.3f}')} "
        "à trois retards, pour un test qui rejette environ "
        f"{num(100 * size['granger_x_to_bloc'].median(), '{:.0f}')} % du temps sous le nul), "
        "et sans trace d'une élection à l'autre "
        f"(r = {num(r_el('right', 'unemployment_rate_ilo', 'start'), '{:+.2f}')} et "
        f"{num(r_el('far_right', 'unemployment_rate_ilo', 'start'), '{:+.2f}')} entre chômage et "
        "niveau à six mois). Le lien le plus fort en apparence, demandeurs d'emploi et extrême "
        f"droite (p HAC < 0,01), tient aux mois de sortie du choc Covid (sans mars 2020 à "
        f"juillet 2021, β passe de {num(covid['all']['beta'])} à "
        f"{num(covid['no_covid']['beta'])}, p = {num(covid['no_covid']['p'])}) et tombe au "
        f"placebo (p = {num(fj['p_placebo'])}) ; la série est de toute façon inutilisable "
        "pour 2027 après la rupture de janvier 2025. Prix, carburants, confiance des ménages et "
        "climat des affaires : les deux seuls placebos à p < 0,05 (inflation alimentaire et "
        "SP95 pour l'extrême droite) ont des signes opposés pour deux mesures de prix, ce qui "
        "ressemble à du bruit plus qu'à un mécanisme ; rien d'autre ne passe, et les corrélations "
        "entre élections (|r| ≥ 0,9 fréquents) suivent surtout la montée continue de l'extrême "
        "droite et la hausse tendancielle des prix : six points ne séparent pas une cause "
        "d'une tendance."
    )
    reco = (
        "Ne pas mettre de dérive économique μ_b(τ) au backtest sur la foi de cette étude : "
        "aucun indicateur ne précède les mouvements de bloc de façon distinguable du hasard, "
        "et prévoir les indicateurs jusqu'au vote (Timestral) ajouterait une incertitude pour "
        "un gain attendu nul. Si Gaétan veut un test confirmatoire, en faire un seul, déclaré "
        "avant de lancer le backtest, sur la question posée : variation du chômage BIT sur "
        "trois mois, retard de trois mois, sur la droite et l'extrême droite, γ estimé en "
        "laissant l'élection évaluée de côté ; l'amplitude impliquée (environ "
        f"{num(abs(ru['effect_sd_6m']), '{:.1f}')} points pour la droite et "
        f"{num(abs(fu['effect_sd_6m']), '{:.1f}')} pour l'extrême droite à 180 jours pour un "
        "écart type) est de l'ordre de l'erreur des sondages, donc le test n'est pas futile, "
        "mais je m'attends à un échec. Pour le prior structurel, la seule variable qui mérite "
        "un essai est la popularité du président (niveau, à 180 et 90 jours) pour le bloc "
        "sortant, en sachant que les sondages la contiennent déjà en grande partie et que "
        "le bloc sortant de 2027 (centre, sans Macron) n'a pas de précédent. Ne pas tester : "
        "inflation et sous-indices, carburants, confiance des ménages et ses soldes, climat "
        "des affaires, demandeurs d'emploi (rupture de 2025) ; les garder dans le tableau de "
        "bord explicatif du site avec la mention qu'ils n'entrent pas dans la prévision, et "
        "consigner ce résultat négatif dans docs/05-economie.md."
    )
    return {"holds": holds, "fails": fails, "reco": reco}


if __name__ == "__main__":
    main()
