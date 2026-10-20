"""Étude : la presse (part de voix, ton, agenda) annonce-t-elle les mouvements des sondages ?

    uv run python studies/press.py collect   # comptages Media Cloud (réseau, clé requise)
    uv run python studies/press.py study [--out runs/<date>-press-study] [--shifts 200]

Protocole de docs/06-presse.md (part de voix contre émergence, Granger dans les deux sens,
chocs, agenda), par comptages, sur 2017, 2022 et 2027. Même méthode que l'étude attention
(back/studies/attention.py, dont on réutilise les fonctions) : régressions « comme à
l'époque », placebos par décalage circulaire, taille réelle des tests, variantes hors
événements. Rien n'entre dans le modèle ici.

Données :

- Media Cloud (`press_mediacloud.parquet`, vvs.ingest.press.mediacloud_counts) : articles
  par jour de la collection France - National qui citent chaque candidat (alias du
  registre), chaque thème (mots-clés), et chaque candidat avec un mot du thème « affaires » ;
  douze mois avant le premier tour 2017 et 2022, depuis janvier 2025 pour 2027. Semaine
  complète (lundi à dimanche) ; `sov` = part de voix (points) = 100 × articles du candidat /
  somme des articles des membres du cycle (`MC.CANDIDATES`) ; saillance d'un thème = 100 ×
  articles du thème / articles de la collection ; part « affaires » d'un candidat = articles
  candidat ∧ affaires / articles du candidat.
- GDELT (`press_timelines.parquet`, reconstruit depuis le cache) : volume et ton quotidiens
  des candidats téléchargés (Macron, Le Pen, Mélenchon, Zemmour au 20 octobre 2026), depuis
  janvier 2017. Part de voix GDELT = part parmi ces seuls candidats ; ton = moyenne
  hebdomadaire pondérée par le volume.
- INA (`press_ina.parquet`) : part de la durée des JT du soir par rubrique, 2000-2020.
- Sondages : part hebdomadaire par candidat et par bloc (moyenne des instituts, chacun
  moyenné sur ses sondages de la semaine) ; part mensuelle par bloc et séries macro
  (vvs.studies.panel) pour la médiation.
- Attention Wikipédia (`attention.parquet`) : pages vues hebdomadaires, pour la comparaison.

Questions et tests, fixés avant de lire les résultats (seuls les volumes totaux par série
étaient visibles au moment de la rédaction) :

(a) **La part de voix précède-t-elle les sondages ?** Spécification principale de l'étude
    attention à l'identique, avec `x` = part de voix Media Cloud : y = sondage(t + 4) −
    sondage(t) sur `datt` = variation sur 4 semaines de la moyenne mobile sur 4 semaines de
    la part de voix, et l'élan des sondages ; effet fixe par cycle ; écarts types par grappe
    candidat × cycle ; h = 2 et 8 en complément ; part de voix retardée de 0 à 8 semaines
    (tableau des retards) ; Granger dans les deux sens (ordres 2 et 4) ; corrélations
    croisées de −8 à +8 semaines ; épisodes d'émergence ; variantes de l'étude attention
    (retard d'une semaine, sans les 5 % plus fortes variations, hors fenêtres d'événements).
    Placebo : décalage circulaire de la série de presse de tous les candidats d'un cycle de s
    semaines, 13 ≤ s ≤ T − 13, tiré indépendamment par cycle (les fenêtres de 52 semaines
    n'offrent que 26 décalages par cycle ; le tirage indépendant donne 676 combinaisons sur
    deux cycles) ; p placebo = rang de |t| (ou du Wald) parmi les décalages distincts tirés
    (un cycle seul : p ≥ 1/27) ; taille = part des décalages où le test nominal rejette à
    5 %. Critère « tient » (celui de l'attention) : β > 0, p placebo < 0,05 en commun
    2017-2022, β > 0 en 2017 et en 2022, signe stable au retrait de chaque candidat, Granger
    presse → sondages à p placebo < 0,05.
    GDELT : même spécification avec la part de voix GDELT entre les candidats disponibles,
    fenêtre longue (cycle 2022 depuis mai 2017, cycle 2027 depuis avril 2022) ; descriptif
    (trop peu de candidats pour un critère), comparé à Media Cloud restreint aux mêmes
    candidats et semaines.
    **Presse contre attention** (question de Gaétan) : sur les mêmes candidats et semaines,
    régression conjointe y ~ datt presse + datt Wikipédia + élan ; placebo en décalant la
    presse seule (et Wikipédia seule). La presse « apporte quelque chose » si son β reste
    > 0 avec p placebo < 0,05 en présence de l'attention. Corrélation des deux `datt`.

(b) **Ton et chocs.**
    - Ton GDELT : même spécification avec `x` = ton hebdomadaire (Δ sur 4 semaines de la
      moyenne mobile), cycle 2022 en fenêtre longue (principal), 2027 en réplication.
      Critère : β > 0, p placebo < 0,05 en 2022, même signe en 2027. Granger dans les deux
      sens.
    - Couverture « affaires » Media Cloud (tous les candidats) : même spécification avec `x`
      = part « affaires » du candidat (points). Critère : β < 0, p placebo < 0,05 en commun
      2017-2022, β < 0 dans chaque cycle.
    - Pics de volume (règle de l'étude attention, seuil minimal 50 articles dans la semaine
      au lieu de 10 000 vues) : |mouvement| des sondages après un pic contre semaines tirées
      au hasard (même critère que l'attention). Pic « négatif » : pic dont la part
      « affaires » est au moins le double de sa médiane des 8 semaines précédentes et d'au
      moins 20 % ; mouvement signé moyen après un pic négatif contre le hasard (attendu
      < 0). Affaire Fillon et chocs d'events.yaml en descriptif.

(c) **Agenda.** Propriété des enjeux, deux hypothèses principales :
    H1 extrême droite ← saillance immigration + sécurité ; H2 gauche totale (gauche radicale
    + gauche) ← saillance pouvoir d'achat. Spécification : y = part du bloc(t + 4) − part(t)
    sur la variation sur 4 semaines de la saillance (moyenne mobile 4 semaines), élan du bloc,
    effet fixe par cycle, erreurs HAC (Bartlett, 4 retards) par cycle ; placebo par
    décalage circulaire de la saillance. Critère : β > 0, p placebo < 0,05 en commun
    2017-2022, β > 0 dans chaque cycle. Réplication INA (déclarée d'avance) sur les douze
    mois avant 2002, 2007, 2012 et 2017 : H1 avec « Faits divers » + « Justice », H2 avec
    « Economie » (rubriques partielles, pas d'immigration à l'INA) ; critère β > 0, p placebo
    < 0,05 sur les quatre cycles réunis. Tableau secondaire thème × bloc (descriptif, on
    attend 5 % de faux positifs).
    Médiation (docs/05 point 5, « le chômage joue quand on en parle ») : panel mensuel
    2002-2020, d_share(m) d'un bloc sur ΔU3 = variation sur trois mois du chômage BIT
    retardée de 3 mois (le test confirmatoire de l'étude économie), la saillance de
    l'économie à l'INA en m − 3 (centrée réduite) et leur produit ; effet fixe par cycle,
    HAC par cycle. Critère : produit > 0 pour l'extrême droite et < 0 pour la droite, chacun
    à p placebo < 0,05 (décalage circulaire de la saillance sur le calendrier, 24 mois au
    moins). Version Media Cloud (thème chômage, 2017 et 2022) en descriptif.

(d) **Erreur finale des sondages** (deux élections, à prendre avec des pincettes) :
    surperformance = résultat − moyenne des sondages des 14 derniers jours ; prédicteurs :
    log du rapport part de voix / part dans les sondages sur les 14 derniers jours
    (principal, comme l'attention), écart, élan de la part de voix (14 jours contre les
    28 précédents), part « affaires » sur les 28 derniers jours, ton GDELT (descriptif) ;
    effet fixe par élection, p par permutation dans chaque élection. Critère : p < 0,05 et
    même signe en 2017 et en 2022.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention as A

from vvs import paths
from vvs.ingest.macro.registry import load as load_macro_registry
from vvs.ingest.press import gdelt
from vvs.ingest.press import mediacloud_counts as MC
from vvs.ingest.press.outlets import load_queries
from vvs.model.bounce import load_events
from vvs.model.plots import GRID, INK, MUTED, RESULT, SERIES, _style, display_name
from vvs.registry import default_registry
from vvs.studies import panel as P
from vvs.studies import press_panel as PP

TRAIN = (2017, 2022)
CYCLES = (2017, 2022, 2027)
PRESS_LAGS = (0, 1, 2, 3, 4, 6, 8)
PRESS_PEAK_MIN = 50  # articles dans la semaine (seuil de l'attention : 10 000 vues)
NEG_PEAK_RATIO, NEG_PEAK_MIN = 2.0, 0.20
HAC_LAGS = 4
H1 = ("far_right", ("immigration", "security"))
H2 = ("left_total", ("purchasing_power",))
INA_KEYS = {"security": ("Faits divers", "Justice"), "economy": ("Economie",)}
INA_H = {"H1": ("far_right", "security"), "H2": ("left_total", "economy")}
INA_CYCLES = (2002, 2007, 2012, 2017)
BLOCS_ALL = ("far_left", "left", "left_total", "centre", "right", "far_right", "incumbent")
MED_MIN_SHIFT = 24
GDELT_LONG_START = {2022: dt.date(2017, 5, 1), 2027: dt.date(2022, 4, 11)}
OFFICIALS_EXCLUDED_2027 = ("macron_emmanuel",)
SEED = 20261020


def rolling_mean_nan(x: np.ndarray, w: int) -> np.ndarray:
    """Moyenne mobile sur `w` semaines, vide si une valeur de la fenêtre manque.

    Remplace `A.rolling_mean` (sommes cumulées) : avec des séries trouées (ton GDELT, part
    « affaires » d'une semaine sans article), une seule valeur manquante y rendait vide toute
    la suite de la série. Identique à l'original sur une série complète."""
    out = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        out[w - 1:] = np.lib.stride_tricks.sliding_window_view(x, w, axis=0).mean(axis=-1)
    return out


A.rolling_mean = rolling_mean_nan


def placebo_p_distinct(obs: float, null: np.ndarray) -> float:
    """p placebo sur les décalages distincts : les tirages répétés d'un même décalage donnent
    la même statistique et sont comptés une fois. Avec 26 décalages possibles par cycle,
    compter les répétitions ferait descendre p sous 1/27, sa vraie résolution."""
    null = np.unique(null[~np.isnan(null)])
    if np.isnan(obs) or len(null) == 0:
        return np.nan
    return float((1 + (null >= obs).sum()) / (1 + len(null)))


A.placebo_p = placebo_p_distinct


def log(msg: str) -> None:
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------------------- collecte


def cmd_collect(args) -> None:
    queries = load_queries()
    reqs = MC.plan(queries)
    log(f"plan : {len(reqs)} séries-fenêtres")
    stats_ = MC.collect(reqs, log=log)
    log(f"bilan : {stats_['done']} requêtes, {stats_['cached']} en cache, "
        f"{len(stats_['failed'])} échecs {stats_['failed']}")
    df = MC.parse_counts(queries, log=log)
    log(f"{MC.OUTPUT_NAME} : {df.height} lignes, {df.select('scope', 'id').n_unique()} séries")


# ---------------------------------------------------------------------------- données


def first_rounds() -> dict[int, dt.date]:
    return {e: el.round_(1).date for e, el in default_registry().elections.items()}


def load_all(from_cache: bool = False):
    reg, att, poll_shares, weekly_polls, errors = A.load_inputs()
    queries = load_queries()
    mc = (MC.parse_counts(queries, write=False, log=log) if from_cache
          else pl.read_parquet(paths.CURATED / MC.OUTPUT_NAME))
    wk = PP.weekly_counts(mc)
    sov = PP.share_of_voice(wk, MC.CANDIDATES)
    themes = PP.theme_salience(wk)
    co = (wk.filter(pl.col("scope") == "candidate_theme")
          .with_columns(pl.col("id").str.split(":").list.get(0).alias("candidate_id"))
          .select("election", "candidate_id", "week", pl.col("count").alias("n_scandal")))
    sov = sov.join(co, on=["election", "candidate_id", "week"], how="left").with_columns(
        pl.when(pl.col("count") > 0).then(100 * pl.col("n_scandal") / pl.col("count"))
        .otherwise(None).alias("scandal_share"))
    tl = gdelt.parse_timelines(queries, write=False, log=log)
    gw = PP.gdelt_weekly(tl)
    ina = pl.read_parquet(paths.CURATED / "press_ina.parquet")
    by_bloc = pl.read_parquet(paths.CURATED / "poll_results_by_bloc.parquet")
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    pbs = P.poll_bloc_shares(by_bloc, polls)
    left = (pbs.filter(pl.col("bloc").is_in(["far_left", "left"]))
            .group_by("poll_id", "election", "pollster", "fieldwork_end", "is_rolling", "month")
            .agg(pl.col("share").sum(), pl.len().alias("_n"))
            .filter(pl.col("_n") == 2).drop("_n")
            .with_columns(pl.lit("left_total").alias("bloc")))
    pbs = pl.concat([pbs.select(left.columns), left])
    wbloc = PP.weekly_bloc_shares(pbs)
    wiki = P.weekly_attention(att, "wikipedia")
    return {"reg": reg, "att": att, "poll_shares": poll_shares, "weekly_polls": weekly_polls,
            "errors": errors, "mc": mc, "wk": wk, "sov": sov, "themes": themes, "tl": tl,
            "gw": gw, "ina": ina, "pbs": pbs, "wbloc": wbloc, "wiki": wiki, "queries": queries}


def _monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def make_cycle(e: int, weeks: list[dt.date], cands: list[str], series: pl.DataFrame,
               value: str, polls: pl.DataFrame, key: str = "candidate_id",
               fill: float | None = 0.0) -> A.Cycle:
    """A.Cycle : `views` = colonne `value` de `series` (key, week), sondages de `polls`
    (key, week, share, n) ; une valeur absente vaut `fill` (None : NaN)."""
    grid = pl.DataFrame({"week": weeks}).join(pl.DataFrame({key: cands}), how="cross")
    a = grid.join(series.select(key, "week", value), on=[key, "week"], how="left")
    if fill is not None:
        a = a.with_columns(pl.col(value).fill_null(fill))
    p = grid.join(polls.select(key, "week", "share", "n"), on=[key, "week"], how="left")
    V = a.pivot(on=key, index="week", values=value).sort("week").select(cands)
    S = p.pivot(on=key, index="week", values="share").sort("week").select(cands)
    N = p.with_columns(pl.col("n").fill_null(0)).pivot(on=key, index="week", values="n").sort(
        "week").select(cands)
    return A.Cycle(e, weeks, cands, V.to_numpy().astype(float), S.to_numpy().astype(float),
                   N.to_numpy().astype(float))


def cand_polls(weekly_polls: pl.DataFrame, e: int) -> pl.DataFrame:
    return (weekly_polls.filter(pl.col("election") == e)
            .select("candidate_id", "week", "share", pl.col("n_polls").alias("n")))


def bloc_polls(wbloc: pl.DataFrame, e: int) -> pl.DataFrame:
    return (wbloc.filter(pl.col("election") == e)
            .select(pl.col("bloc").alias("candidate_id"), "week", "share",
                    pl.col("n_pollsters").alias("n")))


def mc_weeks(sov: pl.DataFrame, e: int) -> list[dt.date]:
    w = sorted(sov.filter(pl.col("election") == e)["week"].unique().to_list())
    if w and (w[-1] - w[0]).days // 7 + 1 != len(w):
        raise ValueError(f"semaines Media Cloud non contiguës en {e}")
    return w


def press_cycles(D, value: str = "count", members: dict[int, list[str]] | None = None,
                 weeks_from=None, fill: float | None = 0.0) -> dict[int, A.Cycle]:
    """Cycles Media Cloud : `views` = colonne `value` de `sov` (count, scandal_share…)."""
    out = {}
    for e in CYCLES:
        weeks = mc_weeks(D["sov"], e) if weeks_from is None else weeks_from(e)
        if not weeks:
            continue
        have = set(D["sov"].filter(pl.col("election") == e)["candidate_id"])
        cands = [c for c in (members or MC.CANDIDATES)[e] if c in have]
        out[e] = make_cycle(e, weeks, cands, D["sov"].filter(pl.col("election") == e), value,
                            cand_polls(D["weekly_polls"], e), fill=fill)
    return out


def with_x(cycles: dict[int, A.Cycle], mats: dict[int, np.ndarray]) -> dict[int, A.Cycle]:
    """Remplace la variable de presse `x` (sinon part des `views` parmi les membres)."""
    for e, m in mats.items():
        cycles[e].x = m
    return cycles


def wiki_like(D, ref: dict[int, A.Cycle]) -> dict[int, A.Cycle]:
    """Attention Wikipédia sur les mêmes candidats et semaines que `ref`."""
    out = {}
    for e, c in ref.items():
        out[e] = make_cycle(e, c.weeks, c.cands, D["wiki"], "value",
                            cand_polls(D["weekly_polls"], e))
    return out


def gdelt_members(D, e: int) -> list[str]:
    have = set(D["gw"]["candidate_id"].unique())
    polled = set(D["weekly_polls"].filter(pl.col("election") == e)["candidate_id"])
    excl = OFFICIALS_EXCLUDED_2027 if e == 2027 else ()
    return sorted(c for c in have & polled if c not in excl)


def gdelt_cycles(D, value: str, long: bool, fill: float | None = 0.0) -> dict[int, A.Cycle]:
    out = {}
    last = _monday(D["gw"]["week"].max())
    for e in (2022, 2027):
        cands = gdelt_members(D, e)
        if len(cands) < 2:
            continue
        if long:
            vote = first_rounds().get(e)
            w1 = _monday(vote - dt.timedelta(days=6)) if e == 2022 else last
            weeks = list(pl.date_range(GDELT_LONG_START[e], w1, "1w", eager=True))
        else:
            weeks = mc_weeks(D["sov"], e)
        out[e] = make_cycle(e, weeks, cands, D["gw"], value, cand_polls(D["weekly_polls"], e),
                            fill=fill)
    return out


# ---------------------------------------------------------------------------- outils


def shift_sets(cycles: dict[int, A.Cycle], n: int, rng, min_shift: int = A.MIN_SHIFT
               ) -> list[dict[int, int]]:
    """n jeux de décalages circulaires, tirés indépendamment par cycle dans [13, T − 13]."""
    ok = {e: c.T for e, c in cycles.items() if c.T > 2 * min_shift + 1}
    return [{e: int(rng.integers(min_shift, T - min_shift + 1)) for e, T in ok.items()}
            for _ in range(n)]


def hac_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray, lags: int = HAC_LAGS):
    """MCO, covariance de Newey-West (Bartlett) calculée série par série (`groups`, lignes
    dans l'ordre du temps) puis sommée. Renvoie (beta, cov, ddl) ou None."""
    n, k = X.shape
    if n <= k + 2:
        return None
    try:
        inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    b = inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        s = X[groups == g] * u[groups == g, None]
        meat += s.T @ s
        for j in range(1, lags + 1):
            if j >= len(s):
                break
            w = 1 - j / (lags + 1)
            G = s[j:].T @ s[:-j]
            meat += w * (G + G.T)
    return b, inv @ meat @ inv, n - k


def fe_rows_fit(rows: pd.DataFrame, x: np.ndarray, control: bool = True, hac: bool = True,
                extra: list[np.ndarray] | None = None):
    """y sur x (+ élan, + extra) avec effet fixe par cycle ; HAC par série ou grappes."""
    y = rows["y"].to_numpy()
    ok = ~np.isnan(x) & ~np.isnan(y)
    if control:
        ok &= ~np.isnan(rows["mom"].to_numpy())
    for z in extra or []:
        ok &= ~np.isnan(z)
    r = rows[ok]
    cols = [x[ok]] + [z[ok] for z in extra or []] + ([r["mom"].to_numpy()] if control else [])
    X = A.fe_design(cols, r["election"].to_numpy())
    g = (r["election"].astype(str) + ":" + r["candidate_id"]).to_numpy()
    fit = hac_ols(y[ok], X, g) if hac else A.ols_cluster(y[ok], X, g)
    return fit, int(ok.sum())


# ---------------------------------------------------------------------------- (a) part de voix


def lag_table(cycles, shifts) -> pd.DataFrame:
    """Spécification principale (h = 4, commun, élan) avec la presse retardée de k semaines."""
    cyc = {e: cycles[e] for e in TRAIN if e in cycles}
    rows = A.anchor_rows(cyc, A.H_MAIN)
    out = []
    for k in PRESS_LAGS:
        fit, n = A.anchor_fit(rows, cyc, None, True, k)
        b, se, t, p = A.t_and_p(fit, 0)
        null = [A.t_and_p(A.anchor_fit(rows, cyc, s, True, k)[0], 0) for s in shifts]
        out.append({"lag": k, "beta": b, "se": se, "p": p, "n": n,
                    "p_placebo": A.placebo_p(abs(t), np.array([abs(z[2]) for z in null])),
                    "size_05": float(np.nanmean(np.array([z[3] for z in null]) < 0.05))})
    return pd.DataFrame(out)


def run_anchor_family(cycles, shifts, train, rng=None, with_peaks=False, label=""):
    """Batterie de l'étude attention sur des cycles de presse."""
    res = {"cycles": cycles}
    res["anchor"], res["size"] = A.anchor_analysis(cycles, train, shifts)
    res["loco"] = A.leave_one_candidate_out(cycles, train)
    res["robust"] = A.anchor_robustness(cycles, train, shifts)
    res["granger"] = A.granger_analysis(cycles, train, shifts)
    res["xcorr"] = A.xcorr(cycles, None)
    res["xcorr_bands"] = {e: np.array([A.xcorr({e: cycles[e]}, s)[e] for s in shifts])
                          for e in cycles}
    if with_peaks:
        A.PEAK_MIN_VIEWS = PRESS_PEAK_MIN
        res["peaks"], res["peak_tests"] = A.peak_analysis(cycles, train, rng)
    log(f"  batterie {label} faite")
    return res


def encompassing(press: dict, wiki: dict, shifts, els=None, h: int = A.H_MAIN
                 ) -> pd.DataFrame:
    """y ~ datt presse + datt Wikipédia + élan (même lignes) ; placebo presse seule, puis
    Wikipédia seule. `els` : cycles réunis (défaut : 2017 et 2022)."""
    els = [e for e in (els or TRAIN) if e in press]
    pc, wc = {e: press[e] for e in els}, {e: wiki[e] for e in els}
    rows = A.anchor_rows(pc, h)

    def fit(sp, sw):
        xp = A.gather(rows, {e: c.datt((sp or {}).get(e, 0)) for e, c in pc.items()})
        xw = A.gather(rows, {e: c.datt((sw or {}).get(e, 0)) for e, c in wc.items()})
        f, n = fe_rows_fit(rows, xp, True, hac=False, extra=[xw])
        return f, n, xp, xw

    f0, n, xp, xw = fit(None, None)
    bp, sep, tp, pp = A.t_and_p(f0, 0)
    bw, sew, tw, pw = A.t_and_p(f0, 1)
    null_p = np.array([abs(A.t_and_p(fit(s, None)[0], 0)[2]) for s in shifts])
    null_w = np.array([abs(A.t_and_p(fit(None, s)[0], 1)[2]) for s in shifts])
    ok = ~np.isnan(xp) & ~np.isnan(xw)
    out = [{"term": "presse", "beta": bp, "se": sep, "p": pp,
            "p_placebo": A.placebo_p(abs(tp), null_p), "n": n},
           {"term": "Wikipédia", "beta": bw, "se": sew, "p": pw,
            "p_placebo": A.placebo_p(abs(tw), null_w), "n": n}]
    # seules, sur les mêmes lignes
    for name, xs, idx in (("presse seule", xp, 0), ("Wikipédia seule", xw, 0)):
        f1, n1 = fe_rows_fit(rows[ok], xs[ok], True, hac=False)
        b, se, _t, p = A.t_and_p(f1, idx)
        out.append({"term": name, "beta": b, "se": se, "p": p, "p_placebo": np.nan, "n": n1})
    df = pd.DataFrame(out)
    df.attrs["corr"] = float(np.corrcoef(xp[ok], xw[ok])[0, 1])
    return df


# ---------------------------------------------------------------------------- (b) ton, affaires


def neg_peaks(cycles: dict[int, A.Cycle], scandal: dict[int, np.ndarray], train, rng
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pics de volume dont la part « affaires » est anormale ; mouvement signé contre hasard."""
    ev = []
    for e, c in cycles.items():
        sc = scandal[e]
        for j, cid in enumerate(c.cands):
            p = c.polls[:, j]
            peaks = A.detect_peaks(c.views[:, j])
            near = set()
            for t in peaks:
                near |= set(range(t - A.PEAK_GAP, t + A.PEAK_GAP + 1))
            for t in peaks:
                base = np.nanmedian(sc[max(0, t - A.PEAK_BASE):t, j])
                if not (sc[t, j] >= max(NEG_PEAK_RATIO * base, NEG_PEAK_MIN * 100)):
                    continue
                mv = A.poll_move(p, t)
                pool = [u for u in range(max(0, t - A.NULL_RADIUS),
                                         min(c.T, t + A.NULL_RADIUS + 1))
                        if u not in near and not np.isnan(A.poll_move(p, u))]
                ev.append({"election": e, "candidate_id": cid, "week": c.weeks[t],
                           "articles": c.views[t, j], "scandal_share": sc[t, j],
                           "scandal_base": base, "move": mv,
                           "null_moves": [A.poll_move(p, u) for u in pool]})
    ev = pd.DataFrame(ev)
    tests = []
    if ev.empty:
        return ev, pd.DataFrame()
    u = ev[ev["move"].notna() & (ev["null_moves"].str.len() > 0)]
    for name, els in {"commun": list(train), **{str(e): [e] for e in cycles}}.items():
        s = u[u["election"].isin(els)]
        if s.empty:
            continue
        nulls = [np.array(v) for v in s["null_moves"]]
        draws = np.array([[z[rng.integers(len(z))] for z in nulls]
                          for _ in range(A.N_PERM_PEAKS)]).mean(axis=1)
        obs = s["move"].mean()
        tests.append({"sample": name, "peaks": int(ev["election"].isin(els).sum()),
                      "usable": len(s), "mean_move": obs, "null_mean_move": float(draws.mean()),
                      "p_lower": float((1 + (draws <= obs).sum()) / (1 + A.N_PERM_PEAKS))})
    return ev.drop(columns="null_moves"), pd.DataFrame(tests)


def shock_events(D) -> pd.DataFrame:
    """Chocs et retraits d'events.yaml (2017, 2022, 2027) : presse et sondages autour."""
    sov, gw, wp = D["sov"], D["gw"], D["weekly_polls"]
    rows = []
    for ev in load_events():
        if ev.election not in CYCLES or ev.type != "shock":
            continue
        w = _monday(ev.date)
        s = sov.filter((pl.col("election") == ev.election)
                       & (pl.col("candidate_id") == ev.candidate)).sort("week")
        if s.is_empty():
            continue
        cur = s.filter(pl.col("week") == w)
        base = s.filter(pl.col("week").is_between(w - dt.timedelta(weeks=8),
                                                  w - dt.timedelta(weeks=1)))
        nxt = s.filter(pl.col("week").is_between(w, w + dt.timedelta(weeks=3)))
        g = gw.filter(pl.col("candidate_id") == ev.candidate)
        tone_pre = g.filter(pl.col("week").is_between(w - dt.timedelta(weeks=8),
                                                      w - dt.timedelta(weeks=1)))["tone"].mean()
        tone_post = g.filter(pl.col("week").is_between(w, w + dt.timedelta(weeks=3)))["tone"].mean()
        p = wp.filter((pl.col("election") == ev.election)
                      & (pl.col("candidate_id") == ev.candidate)).sort("week")
        pre = p.filter(pl.col("week").is_between(w - dt.timedelta(weeks=4),
                                                 w - dt.timedelta(weeks=1)))["share"]
        post = p.filter(pl.col("week").is_between(w + dt.timedelta(weeks=1),
                                                  w + dt.timedelta(weeks=6)))["share"]
        rows.append({"event": ev.id, "date": ev.date, "candidate_id": ev.candidate,
                     "sign": ev.sign,
                     "articles_ratio": (cur["count"].sum() / base["count"].mean()
                                        if base.height and cur.height else np.nan),
                     "sov_pre": base["sov"].mean(), "sov_post": nxt["sov"].mean(),
                     "scandal_pre": base["scandal_share"].mean(),
                     "scandal_post": nxt["scandal_share"].mean(),
                     "tone_pre": tone_pre, "tone_post": tone_post,
                     "poll_pre": pre[-1] if len(pre) else np.nan,
                     "poll_post_min": post.min() if len(post) else np.nan,
                     "poll_post_last": post[-1] if len(post) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- (c) agenda


def theme_mat(themes: pl.DataFrame, e: int, weeks: list[dt.date], keys: tuple[str, ...]
              ) -> np.ndarray:
    """Saillance cumulée des thèmes `keys`, en colonne, sur `weeks` (NaN si absente)."""
    t = (themes.filter((pl.col("election") == e) & pl.col("theme").is_in(list(keys)))
         .group_by("week").agg(pl.col("salience").sum(), pl.len().alias("_k"))
         .filter(pl.col("_k") == len(keys)))
    m = pl.DataFrame({"week": weeks}).join(t, on="week", how="left").sort("week")
    return m["salience"].to_numpy().astype(float)[:, None]


def bloc_cycles(D, bloc: str, series: dict[int, tuple[list[dt.date], np.ndarray]]
                ) -> dict[int, A.Cycle]:
    out = {}
    for e, (weeks, x) in series.items():
        c = make_cycle(e, weeks, [bloc], pl.DataFrame(
            {"candidate_id": [bloc] * len(weeks), "week": weeks, "v": np.ones(len(weeks))}),
            "v", bloc_polls(D["wbloc"], e))
        c.x = x
        out[e] = c
    return out


def agenda_test(cycles: dict[int, A.Cycle], shifts, train) -> list[dict]:
    """Spécification agenda (h = 4, élan, FE cycle, HAC) en commun et par cycle."""
    out = []
    for name, els in {"commun": [e for e in train if e in cycles],
                      **{str(e): [e] for e in cycles}}.items():
        if not els:
            continue
        cyc = {e: cycles[e] for e in els}
        rows = A.anchor_rows(cyc, A.H_MAIN)
        if rows.empty:
            continue

        def fit(s, rows=rows, cyc=cyc):
            x = A.gather(rows, {e: c.datt((s or {}).get(e, 0)) for e, c in cyc.items()})
            return fe_rows_fit(rows, x, True, hac=True)

        f0, n = fit(None)
        b, se, t, p = A.t_and_p(f0, 0)
        null = [A.t_and_p(fit(s)[0], 0) for s in shifts]
        x = A.gather(rows, {e: c.datt() for e, c in cyc.items()})
        out.append({"sample": name, "beta": b, "se": se, "p": p, "n": n,
                    "p_placebo": A.placebo_p(abs(t), np.array([abs(z[2]) for z in null])),
                    "size_05": float(np.nanmean(np.array([z[3] for z in null]) < 0.05)),
                    "effect_sd": b * float(np.nanstd(x)) if n else np.nan})
    return out


def agenda_analysis(D, shifts_n, rng) -> dict:
    res = {}
    hyp_rows, table = [], []
    weeks = {e: mc_weeks(D["sov"], e) for e in CYCLES}
    for hname, (bloc, keys) in (("H1", H1), ("H2", H2)):
        cyc = bloc_cycles(D, bloc, {e: (weeks[e], theme_mat(D["themes"], e, weeks[e], keys))
                                    for e in CYCLES if weeks[e]})
        sh = shift_sets(cyc, shifts_n, rng)
        for r in agenda_test(cyc, sh, TRAIN):
            hyp_rows.append({"hyp": hname, "bloc": bloc, "themes": "+".join(keys), **r})
    res["hyp"] = pd.DataFrame(hyp_rows)
    for theme in sorted(D["themes"]["theme"].unique()):
        for bloc in BLOCS_ALL:
            cyc = bloc_cycles(D, bloc, {e: (weeks[e], theme_mat(D["themes"], e, weeks[e],
                                                                 (theme,)))
                                        for e in TRAIN if weeks[e]})
            sh = shift_sets(cyc, shifts_n, rng)
            r = [z for z in agenda_test(cyc, sh, TRAIN) if z["sample"] == "commun"]
            if r:
                table.append({"theme": theme, "bloc": bloc, **r[0]})
    res["table"] = pd.DataFrame(table)
    # INA : douze mois avant chaque vote 2002-2017
    fr = first_rounds()
    iw = PP.ina_weekly(D["ina"], INA_KEYS)
    ina_rows = []
    for hname, (bloc, key) in INA_H.items():
        series = {}
        for e in INA_CYCLES:
            vote = fr[e]
            w0 = _monday(vote - dt.timedelta(days=365)) + dt.timedelta(days=7)
            w1 = _monday(vote - dt.timedelta(days=6))
            wks = list(pl.date_range(w0, w1, "1w", eager=True))
            s = pl.DataFrame({"week": wks}).join(iw.filter(pl.col("key") == key), on="week",
                                                 how="left").sort("week")
            series[e] = (wks, s["share"].to_numpy().astype(float)[:, None])
        cyc = bloc_cycles(D, bloc, series)
        sh = shift_sets(cyc, shifts_n, rng)
        for r in agenda_test(cyc, sh, INA_CYCLES):
            ina_rows.append({"hyp": hname, "bloc": bloc, "rubric": key, **r})
    res["ina"] = pd.DataFrame(ina_rows)
    return res


def mediation(D, shifts_n, rng) -> dict:
    """Panel mensuel : d_share ~ ΔU3(m − 3) × saillance(m − 3), effet fixe par cycle."""
    fr = first_rounds()
    by_bloc = pl.read_parquet(paths.CURATED / "poll_results_by_bloc.parquet")
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    macro = pl.read_parquet(paths.CURATED / "macro.parquet")
    panel = P.bloc_panel(P.monthly_bloc_shares(P.poll_bloc_shares(by_bloc, polls)), fr)
    ind = P.indicators(P.monthly_macro(macro),
                       P.officeholder_changes(load_macro_registry().officeholders))
    u = ind.sort("month").select("month", "unemployment_rate_ilo").with_columns(
        (pl.col("unemployment_rate_ilo") - pl.col("unemployment_rate_ilo").shift(3))
        .shift(3).alias("du3_l3"))
    ina = D["ina"].with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    tot = ina.group_by("month").agg(pl.col("duration_s").sum().alias("tot"))
    eco = ina.filter(pl.col("theme") == "Economie").group_by("month").agg(
        pl.col("duration_s").sum().alias("eco"))
    sal = {"ina": tot.join(eco, on="month", how="left").select(
        "month", (100 * pl.col("eco").fill_null(0) / pl.col("tot")).alias("s")).sort("month")}
    sal["mediacloud"] = (
        D["mc"].filter((pl.col("scope") == "theme") & (pl.col("id") == "unemployment"))
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month").agg(pl.col("count").sum(), pl.col("total").sum(),
                               pl.col("date").n_unique().alias("_days"))
        .filter(pl.col("_days") >= 28)
        .select("month", (100 * pl.col("count") / pl.col("total")).alias("s")).sort("month"))
    out = {}
    for src, ser in sal.items():
        lag = ser.select(pl.col("month").dt.offset_by("3mo"), pl.col("s").alias("S"))
        cal, months = ser["s"].to_numpy(), ser["month"].to_list()
        idx = {m: i for i, m in enumerate(months)}
        rows = []
        for bloc in ("far_right", "right", "incumbent"):
            d = (panel.filter(pl.col("bloc") == bloc).join(u, on="month", how="left")
                 .join(lag, on="month", how="left"))
            pdf = d.to_pandas().dropna(subset=["d_share", "du3_l3", "S"])
            if len(pdf) < 12:
                continue
            mu, sd = pdf["S"].mean(), pdf["S"].std()
            y, du, g = (pdf["d_share"].to_numpy(), pdf["du3_l3"].to_numpy(),
                        pdf["election"].to_numpy())

            def fit(Svals, mu=mu, sd=sd, y=y, du=du, g=g):
                z = (Svals - mu) / sd
                return hac_ols(y, A.fe_design([du, z, du * z], g), g, 3)

            f0 = fit(pdf["S"].to_numpy())
            r = {"source": src, "bloc": bloc, "n": len(pdf),
                 "cycles": ",".join(str(e) for e in sorted(pdf["election"].unique()))}
            for i, nm in enumerate(("du3", "salience", "interaction")):
                b, se, _t, p = A.t_and_p(f0, i)
                r |= {f"beta_{nm}": b, f"se_{nm}": se, f"p_{nm}": p}
            L = len(cal)
            if src == "ina" and L > 2 * MED_MIN_SHIFT:
                pos = np.array([idx[_add_months(pd.Timestamp(m).date(), -3)]
                                for m in pdf["month"]])
                tobs = A.t_and_p(f0, 2)[2]
                null = [A.t_and_p(fit(np.roll(cal, int(sh))[pos]), 2)
                        for sh in rng.integers(MED_MIN_SHIFT, L - MED_MIN_SHIFT + 1, shifts_n)]
                r["p_placebo_interaction"] = A.placebo_p(abs(tobs),
                                                         np.array([abs(z[2]) for z in null]))
                r["size_05"] = float(np.nanmean(np.array([z[3] for z in null]) < 0.05))
            rows.append(r)
        out[src] = pd.DataFrame(rows)
    return out


def _add_months(d: dt.date, k: int) -> dt.date:
    i = P.month_index(d) + k
    return dt.date(i // 12, i % 12 + 1, 1)


# ---------------------------------------------------------------------------- (d) erreur finale


def final_press_frame(D) -> pd.DataFrame:
    reg, mc, errors = D["reg"], D["mc"], D["errors"]
    pe = errors.filter((pl.col("method") == "mean_14d") & (pl.col("level") == "candidate")
                       & (pl.col("round") == 1)).to_pandas()
    rows = []
    for e in TRAIN:
        vote = reg.first_rounds[e]
        officials = [c.candidate_id for c in A._official(e)]
        c = mc.filter((pl.col("scope") == "candidate") & (pl.col("election") == e)
                      & pl.col("id").is_in(officials))
        co = mc.filter((pl.col("scope") == "candidate_theme") & (pl.col("election") == e)) \
            .with_columns(pl.col("id").str.split(":").list.get(0))

        def share(lo, hi, c=c):
            g = c.filter(pl.col("date").is_between(lo, hi)).group_by("id").agg(
                pl.col("count").sum())
            tot = g["count"].sum()
            return {r[0]: 100 * r[1] / tot for r in g.iter_rows()} if tot else {}

        last = share(vote - dt.timedelta(days=A.FINAL_DAYS), vote - dt.timedelta(days=1))
        prev = share(vote - dt.timedelta(days=A.FINAL_DAYS + A.MOMENTUM_DAYS),
                     vote - dt.timedelta(days=A.FINAL_DAYS + 1))
        lo28 = vote - dt.timedelta(days=28)
        n28 = c.filter(pl.col("date") >= lo28).group_by("id").agg(pl.col("count").sum())
        s28 = co.filter(pl.col("date") >= lo28).group_by("id").agg(
            pl.col("count").sum().alias("sc"))
        sc = {r[0]: r[1] for r in n28.join(s28, on="id", how="left").select(
            "id", (100 * pl.col("sc") / pl.col("count")).alias("x")).iter_rows()}
        g = D["gw"].filter(pl.col("week").is_between(lo28, vote))
        tone = {r[0]: r[1] for r in g.group_by("candidate_id").agg(
            pl.col("tone").mean()).iter_rows()}
        fa = pd.DataFrame([{"candidate_id": cid, "att_last": last.get(cid, np.nan),
                            "att_prev": prev.get(cid, np.nan), "scandal": sc.get(cid, np.nan),
                            "tone": tone.get(cid, np.nan)} for cid in officials])
        p = pe[pe["election"] == e].rename(columns={"unit": "candidate_id"})
        d = fa.merge(p[["candidate_id", "poll_share", "result_share"]], on="candidate_id")
        d["poll_norm"] = 100 * d["poll_share"] / d["poll_share"].sum()
        d["surperf"] = d["result_share"] - d["poll_share"]
        d["log_ratio"] = np.log(d["att_last"] / d["poll_norm"])
        d["gap"] = d["att_last"] - d["poll_norm"]
        d["momentum"] = d["att_last"] - d["att_prev"]
        d["election"] = e
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def final_tests(d: pd.DataFrame, predictors, rng) -> pd.DataFrame:
    """Algorithme de A.final_error_tests, pour une liste de prédicteurs quelconque."""
    out = []
    for x in predictors:
        s = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, "surperf"])
        if len(s) < 5:
            continue
        b = A.fe_beta(s, x)
        num, den = np.zeros(A.N_PERM_FINAL), 0.0
        for _, g in s.groupby("election"):
            xt = (g[x] - g[x].mean()).to_numpy()
            yt = (g["surperf"] - g["surperf"].mean()).to_numpy()
            order = np.argsort(rng.random((A.N_PERM_FINAL, len(xt))), axis=1)
            num += xt[order] @ yt
            den += float(xt @ xt)
        null = num / den
        per = {str(e): A.fe_beta(g, x) for e, g in s.groupby("election")}
        r = {"predictor": x, "beta": b, "n": len(s),
             # tolérance : la permutation identité doit compter (petits n, arrondis)
             "p_perm": float((1 + (np.abs(null) >= abs(b) - 1e-9).sum()) / (1 + A.N_PERM_FINAL)),
             "same_sign_all": bool(all(np.sign(v) == np.sign(b) for v in per.values()))}
        out.append(r | {f"beta_{k}": v for k, v in per.items()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- figures


def fig_xcorr(xc, bands, out: Path, name: str, label: str):
    es = sorted(xc)
    fig, axes = plt.subplots(1, len(es), figsize=(4.2 * len(es), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, es):
        lo, hi = np.nanpercentile(bands[e], [2.5, 97.5], axis=0)
        ax.fill_between(A.XCORR_LAGS, lo, hi, color=GRID, label="placebo 95 %")
        ax.plot(A.XCORR_LAGS, xc[e], "o-", color=SERIES, ms=3, label="observé")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.5, ls=":")
        ax.set_title(str(e), color=INK)
        ax.set_xlabel(f"retard k (semaines) : {label} en t − k")
        _style(ax)
    axes[0].set_ylabel("corrélation Δsondage(t), Δpresse(t − k)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Corrélations croisées des variations hebdomadaires ({name})", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"xcorr_{name}.png", dpi=150)
    plt.close(fig)


def fig_sov(cycles: dict[int, A.Cycle], out: Path):
    for e, c in cycles.items():
        top = np.argsort(-np.nan_to_num(np.nanmean(c.polls, axis=0)), kind="stable")[:8]
        fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), sharex=True)
        w = pd.to_datetime(pd.Series(c.weeks))
        x4 = A.rolling_mean(c.x, A.ATT_WINDOW)
        for ax, j in zip(axes.flat, top):
            ax.plot(w, x4[:, j], color=SERIES, lw=1, label="part de voix (4 sem.)")
            ax.plot(w, c.polls[:, j], "o", color=RESULT, ms=3, label="sondages")
            ax.set_title(display_name(c.cands[j]), fontsize=9, color=INK)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            _style(ax)
        axes.flat[0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Cycle {e} : part de voix Media Cloud et sondages, points", color=INK)
        fig.tight_layout()
        fig.savefig(out / f"sov_{e}.png", dpi=150)
        plt.close(fig)


def fig_agenda(D, out: Path):
    th = D["themes"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    cmap = plt.get_cmap("tab10")
    for ax, e in zip(axes, CYCLES):
        g = th.filter(pl.col("election") == e)
        for i, t in enumerate(sorted(g["theme"].unique())):
            s = g.filter(pl.col("theme") == t).sort("week")
            y = pd.Series(s["salience"].to_numpy()).rolling(4, min_periods=1).mean()
            ax.plot(pd.to_datetime(pd.Series(s["week"].to_list())), y, lw=1, color=cmap(i),
                    label=t)
        ax.set_title(f"{e}", color=INK)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        _style(ax)
    axes[0].set_ylabel("saillance (% des articles, 4 sem.)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Agenda Media Cloud par thème", color=INK)
    fig.tight_layout()
    fig.savefig(out / "agenda.png", dpi=150)
    plt.close(fig)


def fig_fillon(D, out: Path):
    mc = D["mc"].filter(pl.col("election") == 2017)
    a = mc.filter((pl.col("scope") == "candidate") & (pl.col("id") == "fillon_francois")
                  & (pl.col("date") >= dt.date(2016, 10, 1))).sort("date")
    s = mc.filter((pl.col("scope") == "candidate_theme")
                  & (pl.col("id") == "fillon_francois:scandals")
                  & (pl.col("date") >= dt.date(2016, 10, 1))).sort("date")
    p = D["weekly_polls"].filter((pl.col("election") == 2017)
                                 & (pl.col("candidate_id") == "fillon_francois")
                                 & (pl.col("week") >= dt.date(2016, 10, 1)))
    if a.is_empty():
        return
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(a["date"].to_list(), a["count"].to_list(), color=SERIES, width=1.0, alpha=0.5,
           label="articles citant Fillon")
    if not s.is_empty():
        ax.bar(s["date"].to_list(), s["count"].to_list(), color="#e34948", width=1.0,
               alpha=0.8, label="… et un mot du thème affaires")
    ax.set_ylabel("articles / jour")
    ax2 = ax.twinx()
    ax2.plot(p["week"].to_list(), p["share"].to_list(), "o-", color=RESULT, ms=3,
             label="sondages (semaine)")
    ax2.set_ylabel("intentions de vote (%)", color=RESULT)
    for d, lab in ((dt.date(2016, 11, 27), "primaire"), (dt.date(2017, 1, 25), "Canard enchaîné"),
                   (dt.date(2017, 3, 1), "convocation des juges")):
        ax.axvline(d, color=MUTED, ls=":", lw=0.8)
        ax.text(d, ax.get_ylim()[1] * 0.95, lab, fontsize=7, color=MUTED, rotation=90,
                va="top", ha="right")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _style(ax)
    ax.set_title("François Fillon, 2016-2017 : articles Media Cloud et sondages", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fillon.png", dpi=150)
    plt.close(fig)


def fig_final(d: pd.DataFrame, out: Path):
    es = sorted(d["election"].unique())
    fig, axes = plt.subplots(1, len(es), figsize=(4.2 * len(es), 3.6), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), es):
        g = d[d["election"] == e]
        ax.scatter(g["log_ratio"], g["surperf"], color=SERIES, s=18)
        for r in g.itertuples(index=False):
            ax.annotate(display_name(r.candidate_id).split()[-1], (r.log_ratio, r.surperf),
                        fontsize=7, color=MUTED, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(str(e), color=INK)
        ax.set_xlabel("log(part de voix / part sondages), 14 derniers jours")
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("résultat − sondages 14 jours (pts)")
    fig.tight_layout()
    fig.savefig(out / "final_error.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- rapport


f, fp, md_table = A.f, A.fp, A.md_table


def anchor_md(a: pd.DataFrame) -> list[str]:
    rows = [[str(x.h), x.sample, "oui" if x.control else "non", f(x.beta), f(x.se, "{:.3f}"),
             fp(x.p), fp(getattr(x, "p_placebo", np.nan)),
             f(getattr(x, "size_05", np.nan), "{:.2f}"), str(x.n), str(x.n_cand)]
            for x in a.itertuples(index=False)]
    return md_table(["h", "échantillon", "élan", "β (pts/pt)", "é.-t.", "p nominal",
                     "p placebo", "taille", "n", "candidats"], rows)


def robust_md(rb: pd.DataFrame) -> list[str]:
    return md_table(["variante", "β", "é.-t.", "p nominal", "p placebo", "n", "é.-t. Δx",
                     "effet 1 é.-t. (pts)", "é.-t. de y (pts)"],
                    [[x.variant.replace("d'attention", "de presse").replace("attention", "presse"),
                      f(x.beta), f(x.se, "{:.3f}"),
                      fp(x.p), fp(x.p_placebo), str(x.n), f(x.sd_datt, "{:.2f}"),
                      f(x.effect_sd, "{:+.2f}"), f(x.sd_y, "{:.2f}")]
                     for x in rb.itertuples(index=False)])


def granger_md(g: pd.DataFrame) -> list[str]:
    rows = [[str(x.order), x.sample, str(x.n), f(x.sum_att_to_poll), fp(x.p_att_to_poll),
             fp(x.p_placebo_att_to_poll), f(x.size_att_to_poll, "{:.2f}"), fp(x.p_poll_to_att),
             fp(x.p_placebo_poll_to_att), f(x.size_poll_to_att, "{:.2f}")]
            for x in g.itertuples(index=False)]
    return md_table(["ordre", "échantillon", "n", "Σβ presse→sond.", "p presse→sond.",
                     "p placebo", "taille", "p sond.→presse", "p placebo", "taille"], rows)


def coverage_rows(D) -> list[list[str]]:
    """Une ligne par série : articles par cycle et part de la collection (%)."""
    g = (D["mc"].group_by("scope", "id", "election")
         .agg(pl.col("count").sum(), pl.col("total").sum(), pl.len().alias("days")))
    rows = []
    for (scope, i), s in sorted(g.group_by("scope", "id"), key=lambda k: k[0]):
        cells = []
        for e in CYCLES:
            x = s.filter(pl.col("election") == e)
            cells.append("–" if x.is_empty() else
                         f"{x['count'][0]:,} ({100 * x['count'][0] / x['total'][0]:.2f})"
                         .replace(",", " "))
        rows.append([scope, i, *cells])
    return rows


def verdict_a(r) -> dict:
    a = r["anchor"]
    main = a[(a["h"] == A.H_MAIN) & a["control"] & (a["sample"] == "commun")].iloc[0]
    per = a[(a["h"] == A.H_MAIN) & a["control"] & a["sample"].isin([str(e) for e in TRAIN])]
    g = r["granger"]
    g2 = g[(g["order"] == A.GRANGER_ORDERS[0]) & (g["sample"] == "commun")].iloc[0]
    loco = bool((np.sign(r["loco"]["beta"]) == np.sign(main["beta"])).all())
    v = {"beta": main["beta"], "p_placebo": main.get("p_placebo"), "size": main.get("size_05"),
         "each_positive": bool((per["beta"] > 0).all()) and len(per) == len(TRAIN),
         "loco_stable": loco, "granger_p_placebo": g2["p_placebo_att_to_poll"],
         "granger_reverse_p_placebo": g2["p_placebo_poll_to_att"]}
    v["holds"] = bool(main["beta"] > 0 and (v["p_placebo"] or 1) < 0.05 and v["each_positive"]
                      and loco and (v["granger_p_placebo"] or 1) < 0.05)
    return v


def write_report(out: Path, D, R: dict, n_shifts: int):
    L = ["# Presse et dynamique des sondages : part de voix, ton, agenda", ""]
    L += [(f"Run du {dt.date.today().isoformat()}, `back/studies/press.py`. Protocole et "
           "critères : docstring du script, fixés avant lecture ; méthode de l'étude attention "
           "(`runs/2026-10-20-attention/report.md`). Rien n'entre dans le modèle."), ""]
    L += ["## Collecte et couverture", "", R["collect_notes"], ""]
    L += md_table(["scope", "série", *[f"{e} : articles (% collection)" for e in CYCLES]],
                  coverage_rows(D))
    L += [""]
    # (a)
    r = R["mc"]
    v = R["verdict_mc"]
    L += ["## (a) La part de voix précède-t-elle les sondages ?", "",
          "### Media Cloud, part de voix parmi les candidats du cycle", "",
          (f"y = sondage(t + h) − sondage(t) sur la variation sur 4 semaines de la part de voix "
           f"(points), élan, effet fixe par cycle, grappes candidat × cycle ; {n_shifts} "
           "décalages placebo."), ""]
    L += anchor_md(r["anchor"])
    L += ["", "Retards de la part de voix (h = 4, commun 2017-2022, avec élan) :", ""]
    L += md_table(["retard (sem.)", "β", "é.-t.", "p nominal", "p placebo", "taille", "n"],
                  [[str(x.lag), f(x.beta), f(x.se, "{:.3f}"), fp(x.p), fp(x.p_placebo),
                    f(x.size_05, "{:.2f}"), str(x.n)] for x in R["lags"].itertuples(index=False)])
    L += ["", "Variantes (h = 4, commun, avec élan), reprises de l'étude attention :", ""]
    L += robust_md(r["robust"])
    lo = r["loco"]
    L += ["", (f"Retrait d'un candidat à la fois : β de {f(lo['beta'].min())} à "
               f"{f(lo['beta'].max())} ; signe stable : {'oui' if v['loco_stable'] else 'non'}."),
          "", "Granger (semaines consécutives sondées) :", ""]
    L += granger_md(r["granger"])
    L += ["", "Corrélations croisées : `xcorr_mediacloud.png` ; séries : `sov_<cycle>.png`.", ""]
    ep = R["episodes"]
    if not ep.empty:
        L += ["Épisodes d'émergence (règle de l'étude attention) dans les fenêtres Media Cloud :",
              ""]
        L += md_table(["cycle", "candidat", "mois", "sondages", "mi-hausse sondages",
                       "mi-hausse presse", "avance (sem.)"],
                      [[str(x.election), display_name(x.candidate_id),
                        f"{x.month_start} → {x.month_end}",
                        f"{x.poll_start:.1f} → {x.poll_end:.1f}", str(x.poll_half),
                        str(x.att_half), f(x.lead_weeks, "{:+.0f}")]
                       for x in ep.itertuples(index=False)])
        L += [""]
    L += [f"Verdict selon le critère fixé : `{json.dumps(v, default=str)}`", ""]
    # presse contre attention
    L += ["### Presse contre attention Wikipédia (mêmes candidats, mêmes semaines)", ""]
    wa = R["wiki"]["anchor"]
    wmain = wa[(wa["h"] == A.H_MAIN) & wa["control"]]
    L += ["Attention Wikipédia, spécification principale sur l'échantillon Media Cloud :", ""]
    L += anchor_md(wmain)
    enc = R["enc"]
    L += ["", (f"Régression conjointe (h = 4, commun, élan) ; corrélation des deux variations : "
               f"{enc.attrs['corr']:+.2f}. p placebo : décalage de la seule variable testée."), ""]
    L += md_table(["terme", "β", "é.-t.", "p nominal", "p placebo", "n"],
                  [[x.term, f(x.beta), f(x.se, "{:.3f}"), fp(x.p), fp(x.p_placebo), str(x.n)]
                   for x in enc.itertuples(index=False)])
    L += ["", ("Par cycle (**ajouté après la première lecture**, pour voir si le signe de la "
               "presse en présence de l'attention tient d'un cycle à l'autre ; un seul cycle : "
               "26 décalages possibles, p placebo ≥ 0,04) :"), ""]
    L += md_table(["cycle", "terme", "β", "é.-t.", "p nominal", "p placebo", "n"],
                  [[x.sample, x.term, f(x.beta), f(x.se, "{:.3f}"), fp(x.p), fp(x.p_placebo),
                    str(x.n)] for x in R["enc_cycles"].itertuples(index=False)
                   if x.term in ("presse", "Wikipédia")])
    L += [""]
    # GDELT
    for key, title in (("gdelt_long", ("GDELT, part de voix entre les candidats disponibles, "
                                      "fenêtre longue")),
                       ("gdelt_mcwin", "GDELT, fenêtres Media Cloud"),
                       ("mc_gdelt_members", "Media Cloud restreint aux candidats GDELT")):
        if key not in R:
            continue
        g = R[key]
        L += [f"### {title}", "",
              "Candidats : " + "; ".join(f"{e} : {', '.join(display_name(c) for c in cc.cands)}"
                                         f" ({cc.T} sem.)" for e, cc in g["cycles"].items()), ""]
        L += anchor_md(g["anchor"][g["anchor"]["h"] == A.H_MAIN])
        L += ["", "Granger :", ""]
        L += granger_md(g["granger"])
        L += [""]
    # (b)
    L += ["## (b) Ton, affaires et chocs", ""]
    if "tone" in R:
        t = R["tone"]
        L += ["### Ton GDELT (Δ sur 4 semaines du ton moyen pondéré)", "",
              "Candidats : " + "; ".join(f"{e} : {', '.join(display_name(c) for c in cc.cands)}"
                                         for e, cc in t["cycles"].items()), ""]
        L += anchor_md(t["anchor"])
        L += ["", "Granger :", ""]
        L += granger_md(t["granger"])
        L += ["", f"Verdict : `{json.dumps(R['verdict_tone'], default=str)}`", ""]
    if "scandal" in R:
        s = R["scandal"]
        L += [("### Couverture « affaires » Media Cloud (part des articles du candidat qui citent "
              "un mot du thème affaires, points)"), ""]
        L += anchor_md(s["anchor"])
        L += ["", "Variantes :", ""]
        L += robust_md(s["robust"])
        L += ["", "Granger :", ""]
        L += granger_md(s["granger"])
        L += ["", f"Verdict : `{json.dumps(R['verdict_scandal'], default=str)}`", ""]
    pt = r.get("peak_tests", pd.DataFrame())
    if not pt.empty:
        L += ["### Pics de volume Media Cloud", ""]
        L += md_table(["échantillon", "pics", "avec sondages avant/après",
                       "|mouvement| moyen (pts)", "au hasard", "p", "mouvement moyen",
                       "part en baisse"],
                      [[x.sample, str(x.peaks), str(x.usable), f(x.mean_abs_move, "{:.2f}"),
                        f(x.null_mean_abs_move, "{:.2f}"), fp(x.p_abs),
                        f(x.mean_signed_move, "{:+.2f}"), f(x.share_down, "{:.2f}")]
                       for x in pt.itertuples(index=False)])
        pk = r["peaks"].dropna(subset=["move"]).sort_values("ratio", ascending=False).head(12)
        L += ["", "Plus forts pics (avec sondages avant et après) :", ""]
        L += md_table(["cycle", "candidat", "semaine", "articles", "× médiane 8 sem.",
                       "mouvement (pts)"],
                      [[str(x.election), display_name(x.candidate_id), str(x.week),
                        f"{x.views:,.0f}".replace(",", " "), f"{x.ratio:.1f}",
                        f(x.move, "{:+.1f}")] for x in pk.itertuples(index=False)])
        L += [""]
    npk, npt = R.get("neg_peaks", (pd.DataFrame(), pd.DataFrame()))
    if not npt.empty:
        L += ["Pics négatifs (part « affaires » ≥ 2 × sa médiane des 8 semaines et ≥ 20 %) :", ""]
        L += md_table(["échantillon", "pics", "utilisables", "mouvement moyen", "au hasard",
                       "p (plus bas qu'au hasard)"],
                      [[x.sample, str(x.peaks), str(x.usable), f(x.mean_move, "{:+.2f}"),
                        f(x.null_mean_move, "{:+.2f}"), fp(x.p_lower)]
                       for x in npt.itertuples(index=False)])
        L += [""]
        L += md_table(["cycle", "candidat", "semaine", "articles", "part affaires (%)",
                       "médiane avant", "mouvement"],
                      [[str(x.election), display_name(x.candidate_id), str(x.week),
                        f"{x.articles:.0f}", f"{x.scandal_share:.0f}", f"{x.scandal_base:.0f}",
                        f(x.move, "{:+.1f}")] for x in npk.itertuples(index=False)])
        L += [""]
    se = R["shocks"]
    if not se.empty:
        L += [("Chocs d'events.yaml (semaine de l'événement ; presse : 8 semaines avant contre "
              "les 4 semaines à partir de l'événement ; sondages : dernier des 4 semaines avant, "
              "minimum et dernier des 6 semaines après) :"), ""]
        L += md_table(["événement", "date", "candidat", "signe", "articles × moy. 8 sem.",
                       "part de voix avant → après", "part affaires avant → après",
                       "ton GDELT avant → après", "sondage avant", "min après", "dernier après"],
                      [[x.event, str(x.date), display_name(x.candidate_id), f(x.sign, "{:+.0f}"),
                        f(x.articles_ratio, "{:.1f}"),
                        f"{f(x.sov_pre, '{:.1f}')} → {f(x.sov_post, '{:.1f}')}",
                        f"{f(x.scandal_pre, '{:.0f}')} → {f(x.scandal_post, '{:.0f}')}",
                        f"{f(x.tone_pre, '{:+.2f}')} → {f(x.tone_post, '{:+.2f}')}",
                        f(x.poll_pre, "{:.1f}"), f(x.poll_post_min, "{:.1f}"),
                        f(x.poll_post_last, "{:.1f}")] for x in se.itertuples(index=False)])
        L += ["", "Cas Fillon : `fillon.png`.", ""]
    # (c)
    ag = R["agenda"]
    L += ["## (c) Agenda et parts des blocs", "",
          ("Hypothèses principales (h = 4, élan, effet fixe par cycle, HAC 4 retards ; placebo par "
          "décalage de la saillance ; effet 1 é.-t. en points de bloc sur 4 semaines) :"), ""]
    L += md_table(["hyp.", "bloc", "thèmes", "échantillon", "β (pts/pt)", "é.-t.", "p HAC",
                   "p placebo", "taille", "n", "effet 1 é.-t."],
                  [[x.hyp, x.bloc, x.themes, x.sample, f(x.beta), f(x.se, "{:.3f}"), fp(x.p),
                    fp(x.p_placebo), f(x.size_05, "{:.2f}"), str(x.n),
                    f(x.effect_sd, "{:+.2f}")] for x in ag["hyp"].itertuples(index=False)])
    L += ["", f"Verdicts : `{json.dumps(R['verdict_agenda'], default=str)}`", "",
          "Réplication INA (douze mois avant 2002, 2007, 2012, 2017) :", ""]
    L += md_table(["hyp.", "bloc", "rubriques", "échantillon", "β", "é.-t.", "p HAC",
                   "p placebo", "taille", "n", "effet 1 é.-t."],
                  [[x.hyp, x.bloc, x.rubric, x.sample, f(x.beta), f(x.se, "{:.3f}"), fp(x.p),
                    fp(x.p_placebo), f(x.size_05, "{:.2f}"), str(x.n),
                    f(x.effect_sd, "{:+.2f}")] for x in ag["ina"].itertuples(index=False)])
    tb = ag["table"]
    if not tb.empty:
        L += ["", (f"Tableau secondaire, thème × bloc (commun 2017-2022) : {int((tb['p_placebo'] < 0.05).sum())} "
                   f"paires sur {len(tb)} à p placebo < 0,05 (attendu au hasard : "
                   f"{0.05 * len(tb):.1f}). Paires à p placebo < 0,10 :"), ""]
        sel = tb[tb["p_placebo"] < 0.10].sort_values("p_placebo")
        L += md_table(["thème", "bloc", "β", "p HAC", "p placebo", "taille", "effet 1 é.-t."],
                      [[x.theme, x.bloc, f(x.beta), fp(x.p), fp(x.p_placebo),
                        f(x.size_05, "{:.2f}"), f(x.effect_sd, "{:+.2f}")]
                       for x in sel.itertuples(index=False)])
        L += ["", "Détail : `agenda_table.csv` ; figure : `agenda.png`.", ""]
    med = R["mediation"]
    L += ["### Médiation : le chômage joue-t-il quand on en parle ?", "",
          ("d_share(m) ~ ΔU3(m − 3) + saillance(m − 3) + produit, effet fixe par cycle, HAC "
           "3 retards ; saillance centrée réduite ; INA : rubrique Economie ; Media Cloud : thème "
           "chômage (fenêtres de l'étude seulement)."), ""]
    for m in med.values():
        if m.empty:
            continue
        L += md_table(["source", "bloc", "cycles", "n", "β ΔU3", "p", "β saillance", "p",
                       "β produit", "p", "p placebo produit", "taille"],
                      [[x["source"], x["bloc"], x["cycles"], str(x["n"]), f(x["beta_du3"]),
                        fp(x["p_du3"]), f(x["beta_salience"]), fp(x["p_salience"]),
                        f(x["beta_interaction"]), fp(x["p_interaction"]),
                        fp(x.get("p_placebo_interaction", np.nan)),
                        f(x.get("size_05", np.nan), "{:.2f}")] for _, x in m.iterrows()])
        L += [""]
    L += [f"Verdict médiation : `{json.dumps(R['verdict_mediation'], default=str)}`", ""]
    # (d)
    ft = R["final_tests"]
    L += ["## (d) Presse et erreur finale des sondages", ""]
    bcols = sorted(c for c in ft.columns if c.startswith("beta_2"))
    L += md_table(["prédicteur", "n", "β", "p permutation", *[c.replace("beta_", "β ")
                                                             for c in bcols], "même signe"],
                  [[x["predictor"], str(x["n"]), f(x["beta"]), fp(x["p_perm"]),
                    *[f(x[c]) for c in bcols], "oui" if x["same_sign_all"] else "non"]
                   for _, x in ft.iterrows()])
    fd = R["final"]
    L += ["", "Données (`final_press.csv`), 14 derniers jours :", ""]
    L += md_table(["élection", "candidat", "part de voix", "sondages (norm.)", ("résultat − "
                   "sondages"), "part affaires 28 j (%)", "ton GDELT 28 j", "log ratio Wikipédia"],
                  [[str(x.election), display_name(x.candidate_id), f(x.att_last, "{:.1f}"),
                    f(x.poll_norm, "{:.1f}"), f(x.surperf, "{:+.1f}"), f(x.scandal, "{:.0f}"),
                    f(x.tone, "{:+.2f}"), f(getattr(x, "log_ratio_wiki", np.nan), "{:+.2f}")]
                   for x in fd.sort_values(["election", "poll_norm"], ascending=[True, False])
                   .itertuples(index=False)])
    L += ["", (f"Corrélation des log ratios presse et Wikipédia : "
               f"{R['final_corr_wiki']:+.2f}. Figure : `final_error.png`."), "",
          f"Verdict : `{json.dumps(R['verdict_final'], default=str)}`", ""]
    L += ["## Conclusion", "", CONCLUSION, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


CONCLUSION = """\
*Rédigée au vu de la run du 2026-10-20 (Media Cloud 2017, 2022, 2027 ; GDELT pour quatre
candidats ; INA 2002-2017) ; à relire si les données changent.*

**Aucun critère fixé d'avance n'est rempli.** Six tests principaux, six échecs, sans
exception, et sans variante qui sauve l'un d'eux.

**Part de voix (a).** La variation de la part de voix Media Cloud ne précède pas les
sondages : β = −0,009 point de sondage par point de part de voix (p placebo = 0,96), nul en
2017 comme en 2022 et en 2027, signe instable au retrait d'un candidat, rien à aucun retard
de 0 à 8 semaines (tous négatifs, p placebo ≥ 0,17), Granger presse → sondages à p placebo
= 0,40 (le Granger nominal rejette 69 à 84 % du temps sous le placebo à l'ordre 4 : il ne
veut rien dire ici). GDELT ne fait pas mieux sur ses quatre candidats (β = +0,007 en
fenêtre longue, p placebo = 0,84). Les épisodes d'émergence ne montrent pas d'avance
exploitable : la presse mène de deux semaines ou plus dans 5 épisodes sur 11, et les
grandes avances (Mélenchon +25 semaines, Pécresse +22) rattachent une rentrée ou une
déclaration à une hausse sans rapport ; Macron 2017 et Bardella 2025 montent dans les
sondages avant la presse. Zemmour 2021 : presse et sondages montent la même semaine.

**Presse contre attention.** Sur les mêmes candidats et les mêmes semaines, l'attention
Wikipédia réplique son résultat (β = +0,083, p placebo = 0,015) quand la presse ne donne
rien. Les deux variations sont corrélées à +0,71 et les log ratios de fin de campagne à
+0,88 : elles mesurent les mêmes événements. Dans la régression conjointe, l'attention garde
son signe (+0,158) et la presse devient négative (−0,174, p placebo = 0,006 chacune), de même
signe en 2017 et en 2022 mais pas en 2027 (+0,086, non significatif). Lecture, **après coup
et dans le sens opposé à l'hypothèse** : ce qui annonce un rattrapage des sondages, c'est
l'intérêt du public au-delà de la couverture, et une couverture que le public ne suit pas
annonce plutôt une baisse (non vérifié candidat par candidat) ; environ un demi-point sur
quatre semaines par écart type. C'est le même rattrapage de mesure de court terme que
l'étude attention, pas une information nouvelle sur l'opinion, et le signe s'inverse sur le
cycle en cours. **La presse n'apporte rien de plus que l'attention** ; elle en est au mieux
un dénominateur.

**Ton et chocs (b).** Ton GDELT : rien (β = +0,22, p placebo = 0,83 sur 2022 ; 2027 trop
court). Part « affaires » Media Cloud : rien (β = +0,006, p placebo = 0,78, signe contraire à
l'attendu). Les pics de volume ne précèdent pas de mouvement plus grand qu'au hasard en
commun (0,82 point contre 0,77, p = 0,34) ni en 2017 ; seulement en 2022. L'affaire Fillon
n'est pas un pic de volume (articles × 2,1, il était déjà très couvert depuis la primaire)
mais un changement de composition : la part « affaires » de sa couverture passe de 2 à 22 %
la semaine du Canard enchaîné, et les sondages décrochent la même semaine (25,7 → 21,5
puis 18,5), sans avance de la presse (`fillon.png`). Pécresse 2022 perd cinq points après le
Zénith sans aucun signal de presse (volume × 1,2, part « affaires » inchangée). Les deux pics
négatifs nets sont ceux de Marine Le Pen en 2027 (jugement du 31 mars 2025, appel du 6
juillet 2026), suivis de −3,5 et −2 points : une affaire, deux dates, hors échantillon
d'apprentissage, et des hypothèses de sondage qui changent en même temps (Bardella).

**Agenda (c).** Propriété des enjeux : H1 (immigration et sécurité → extrême droite) a le
mauvais signe dans chaque cycle (β = −0,18, p placebo = 0,63) ; H2 (pouvoir d'achat → gauche)
est positif mais nul (p placebo = 0,43) ; la réplication INA sur 2002-2017 échoue pour les
deux (p placebo 0,89 et 0,56). Tableau thème × bloc : 2 paires sur 70 sous 5 %, moins que
les 3,5 attendues au hasard ; elles disent que la saillance des affaires précède une baisse
du bloc sortant (2017 : la gauche, d'où la seconde paire), plausible, non déclaré, et à
compter comme bruit tant qu'il ne se réplique pas. Médiation : le produit chômage ×
saillance de l'économie au JT a le **signe opposé** à l'hypothèse pour l'extrême droite
(−2,7, p placebo = 0,045, taille réelle du test HAC 21 %) et pour la droite (+3,2, p placebo
= 0,09) : rien ne soutient « le chômage joue quand on en parle ».

**Erreur finale (d).** Comme pour l'attention : log ratio part de voix / sondages de signes
opposés en 2017 et 2022 (p = 0,43), élan de même signe mais non significatif (p = 0,14). Le
ton GDELT sort à p = 0,007, mais sur 7 candidats (3 + 4, 144 permutations possibles) et tient
à Mélenchon 2022 (ton le moins négatif, +5,9 points) : anecdote, pas un résultat.

**Recommandation : ne pas tester la presse au backtest**, sous aucune des formes étudiées
(dérive μ_c(τ) par la part de voix, le ton ou la part « affaires » ; dérive μ_b(τ) par
l'agenda ; médiation de l'économie). Aucune ne passe son critère, et aucune n'a d'effet aux
horizons de 180 et 90 jours de la règle d'admission. La seule forme qui aurait un sens est
le terme conjoint « attention moins presse » sur quatre semaines, testé aux horizons de 30
et 7 jours que la règle admet désormais pour les effets de fin de campagne : je ne la
recommande pas (découverte après coup, signe inversé en 2027, amplitude d'un demi-point
qu'un modèle absorbe dès le sondage suivant). Usages hors prévision : la part de voix et
l'agenda Media Cloud pour la section « Presse » du site, avec la mention « n'entre pas dans
la prévision » ; la mise à jour coûte une requête par série 2027 (44 séries : candidats,
thèmes, co-mentions), soit environ 300 requêtes par semaine en quotidien, loin du quota. Résultat négatif
à consigner dans docs/06-presse.md."""


# ---------------------------------------------------------------------------- étude


def cmd_study(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = load_all(args.from_cache)
    R: dict = {}
    n = args.shifts
    # (a) Media Cloud
    mc = press_cycles(D)
    sh = shift_sets(mc, n, rng)
    R["mc"] = run_anchor_family(mc, sh, list(TRAIN), rng, with_peaks=True, label="Media Cloud")
    R["verdict_mc"] = verdict_a(R["mc"])
    R["lags"] = lag_table(mc, sh)
    monthly = A.monthly_polls(D["poll_shares"])
    R["episodes"] = A.episode_leads(A.detect_episodes(monthly, list(mc)), mc)
    wk = wiki_like(D, mc)
    R["wiki"] = {"anchor": A.anchor_analysis(wk, list(TRAIN), sh)[0]}
    R["enc"] = encompassing(mc, wk, sh)
    R["enc_cycles"] = pd.concat([encompassing(mc, wk, sh, [e]).assign(sample=str(e))
                                 for e in CYCLES if e in mc], ignore_index=True)
    # GDELT
    g_long = gdelt_cycles(D, "volume", long=True)
    if g_long:
        R["gdelt_long"] = run_anchor_family(g_long, shift_sets(g_long, n, rng), [2022],
                                            label="GDELT long")
        g_mc = gdelt_cycles(D, "volume", long=False)
        R["gdelt_mcwin"] = run_anchor_family(g_mc, shift_sets(g_mc, n, rng), [2022],
                                             label="GDELT fenêtres MC")
        mem = {e: gdelt_members(D, e) for e in CYCLES}
        mcg = {e: c for e, c in press_cycles(D, members=mem).items() if e in g_mc}
        R["mc_gdelt_members"] = run_anchor_family(mcg, shift_sets(mcg, n, rng), [2022],
                                                  label="MC membres GDELT")
        tone = gdelt_cycles(D, "volume", long=True)
        tmat = gdelt_cycles(D, "tone", long=True, fill=None)
        tone = with_x(tone, {e: c.views for e, c in tmat.items()})
        R["tone"] = run_anchor_family(tone, shift_sets(tone, n, rng), [2022], label="ton")
        a = R["tone"]["anchor"]
        m = a[(a["h"] == A.H_MAIN) & a["control"]]
        b22 = m[m["sample"] == "2022"]
        b27 = m[m["sample"] == "2027"]
        R["verdict_tone"] = {
            "beta_2022": float(b22["beta"].iloc[0]) if len(b22) else None,
            "p_placebo_2022": float(b22["p_placebo"].iloc[0]) if len(b22) else None,
            "beta_2027": float(b27["beta"].iloc[0]) if len(b27) else None}
        R["verdict_tone"]["holds"] = bool(
            len(b22) and b22["beta"].iloc[0] > 0 and b22["p_placebo"].iloc[0] < 0.05
            and len(b27) and b27["beta"].iloc[0] > 0)
    # (b) affaires
    if (D["mc"]["scope"] == "candidate_theme").any():
        sc = press_cycles(D)
        smat = press_cycles(D, value="scandal_share", fill=None)
        sc = with_x(sc, {e: c.views for e, c in smat.items()})
        R["scandal"] = run_anchor_family(sc, sh, list(TRAIN), label="affaires")
        a = R["scandal"]["anchor"]
        main = a[(a["h"] == A.H_MAIN) & a["control"] & (a["sample"] == "commun")].iloc[0]
        per = a[(a["h"] == A.H_MAIN) & a["control"] & a["sample"].isin(["2017", "2022"])]
        R["verdict_scandal"] = {"beta": main["beta"], "p_placebo": main["p_placebo"],
                                "each_negative": bool((per["beta"] < 0).all()),
                                "holds": bool(main["beta"] < 0 and main["p_placebo"] < 0.05
                                              and (per["beta"] < 0).all())}
        R["neg_peaks"] = neg_peaks(mc, {e: c.views for e, c in smat.items()}, TRAIN, rng)
    R["shocks"] = shock_events(D)
    # (c)
    R["agenda"] = agenda_analysis(D, n, rng)
    h = R["agenda"]["hyp"]
    vag = {}
    for hy in ("H1", "H2"):
        s = h[h["hyp"] == hy]
        com = s[s["sample"] == "commun"]
        per = s[s["sample"].isin(["2017", "2022"])]
        ina = R["agenda"]["ina"]
        ic = ina[(ina["hyp"] == hy) & (ina["sample"] == "commun")]
        vag[hy] = {"holds": bool(len(com) and com["beta"].iloc[0] > 0
                                 and com["p_placebo"].iloc[0] < 0.05 and (per["beta"] > 0).all()),
                   "ina_replicates": bool(len(ic) and ic["beta"].iloc[0] > 0
                                          and ic["p_placebo"].iloc[0] < 0.05)}
    R["verdict_agenda"] = vag
    R["mediation"] = mediation(D, n, rng)
    mi = R["mediation"]["ina"].set_index("bloc") if not R["mediation"]["ina"].empty else None
    R["verdict_mediation"] = {} if mi is None else {
        "far_right": bool(mi.loc["far_right", "beta_interaction"] > 0
                          and mi.loc["far_right", "p_placebo_interaction"] < 0.05),
        "right": bool(mi.loc["right", "beta_interaction"] < 0
                      and mi.loc["right", "p_placebo_interaction"] < 0.05)}
    # (d)
    fd = final_press_frame(D)
    fw = A.final_error_frame(D["att"], D["reg"], D["errors"], "wikipedia")
    fd = fd.merge(fw[["election", "candidate_id", "log_ratio"]].rename(
        columns={"log_ratio": "log_ratio_wiki"}), on=["election", "candidate_id"], how="left")
    R["final"] = fd
    R["final_tests"] = final_tests(fd, ("log_ratio", "gap", "momentum", "scandal", "tone"), rng)
    ok = fd[["log_ratio", "log_ratio_wiki"]].replace([np.inf, -np.inf], np.nan).dropna()
    R["final_corr_wiki"] = float(np.corrcoef(ok["log_ratio"], ok["log_ratio_wiki"])[0, 1])
    m = R["final_tests"].set_index("predictor").loc["log_ratio"]
    R["verdict_final"] = {"beta": m["beta"], "p_perm": m["p_perm"],
                          "holds": bool(m["p_perm"] < 0.05 and m["same_sign_all"])}
    # collecte
    reqs = D["mc"].select("scope", "id", "election").unique().height
    R["collect_notes"] = collect_notes(D)
    # sorties
    fig_xcorr(R["mc"]["xcorr"], R["mc"]["xcorr_bands"], out, "mediacloud", "part de voix")
    fig_sov(mc, out)
    fig_agenda(D, out)
    fig_fillon(D, out)
    fig_final(fd, out)
    for k in ("anchor", "loco", "robust", "granger", "peaks", "peak_tests"):
        if isinstance(R["mc"].get(k), pd.DataFrame) and not R["mc"][k].empty:
            R["mc"][k].to_csv(out / f"{k}_mediacloud.csv", index=False)
    R["lags"].to_csv(out / "lags_mediacloud.csv", index=False)
    pd.concat([R["enc"].assign(sample="commun"), R["enc_cycles"]]).to_csv(
        out / "encompassing_wikipedia.csv", index=False)
    R["agenda"]["hyp"].to_csv(out / "agenda_hypotheses.csv", index=False)
    R["agenda"]["table"].to_csv(out / "agenda_table.csv", index=False)
    R["agenda"]["ina"].to_csv(out / "agenda_ina.csv", index=False)
    for src, mm in R["mediation"].items():
        mm.to_csv(out / f"mediation_{src}.csv", index=False)
    fd.to_csv(out / "final_press.csv", index=False)
    R["final_tests"].to_csv(out / "final_tests.csv", index=False)
    R["shocks"].to_csv(out / "shocks.csv", index=False)
    for k in ("gdelt_long", "tone", "scandal"):
        if k in R:
            R[k]["anchor"].to_csv(out / f"anchor_{k}.csv", index=False)
            R[k]["granger"].to_csv(out / f"granger_{k}.csv", index=False)
    write_report(out, D, R, n)
    summary = {k: R[k] for k in R if k.startswith("verdict")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")
    log(json.dumps(summary, default=str))
    log(f"écrit : {out} ({reqs} séries Media Cloud)")


def collect_notes(D) -> str:
    """Bilan de collecte tiré du journal des requêtes Media Cloud (sans la clé)."""
    import collections

    path = paths.RAW / "press" / "mediacloud" / "requests.jsonl"
    ev = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    mine = [e for e in ev if e.get("endpoint") == "search/count-over-time"
            and "cs" in e.get("params", {}) and e["params"].get("q") != "*"]
    st = collections.Counter(e.get("status") for e in mine)
    tot = collections.Counter(e.get("status") for e in ev)
    return (f"Media Cloud, count-over-time de l'étude (collection 34412146, hors requêtes « * » "
            f"de la mesure de couverture) : "
            f"{st.get(200, 0)} réponses 200, {st.get(429, 0)} réponses 429 (reprises), "
            f"{sum(v for k, v in st.items() if k not in (200, 429))} autres ; journal complet "
            f"du client (mesure de couverture comprise) : {sum(tot.values())} requêtes HTTP "
            f"dont {tot.get(200, 0)} réussies. Séries : "
            f"{D['mc'].select('scope', 'id', 'election').unique().height} (scope × id × cycle). "
            "GDELT : candidats en cache au moment de la run : "
            + ", ".join(sorted(D["gw"]["candidate_id"].unique())) + ".")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect")
    s = sub.add_parser("study")
    s.add_argument("--out", type=Path,
                   default=paths.RUNS / f"{dt.date.today().isoformat()}-press-study")
    s.add_argument("--shifts", type=int, default=200, help="décalages placebo par test")
    s.add_argument("--from-cache", action="store_true",
                   help="relire le cache Media Cloud au lieu de press_mediacloud.parquet")
    args = ap.parse_args()
    {"collect": cmd_collect, "study": cmd_study}[args.cmd](args)


if __name__ == "__main__":
    main()
