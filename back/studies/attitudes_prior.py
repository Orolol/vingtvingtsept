"""Étude : les préoccupations et attitudes améliorent-elles le prior structurel de bloc ?

    uv run python studies/attitudes_prior.py [--out runs/<date>-attitudes-prior] [--no-leverage]

Question (suite de docs/05-economie.md) : la seule entrée encore ouverte pour une série
annuelle ou semestrielle est le prior structurel de `β_b` au début de la série
(docs/04-modele.md, section 1). Ce prior est aujourd'hui, dans le code
(`vvs.model.data.prior_bloc_shares`), la part du bloc au premier tour de la présidentielle
précédente, avec un écart-type de 1 sur l'échelle logit (`Priors.beta0_sd`). Rien n'entre dans
le modèle ici.

Séries (data/curated/attitudes.parquet, registre data/registry/attitude_series.yaml) :
saillance de l'immigration et de la sécurité (Eurobaromètre « deux problèmes les plus
importants », CREDOC), du chômage et des prix ; indice longitudinal de tolérance de la CNCDH ;
confiance politique (Cevipof) ; baromètre d'image du FN/RN (Verian, ex-Sofres). Valeur
« connue au début de la série » : la dernière observation disponible 180 jours avant le
premier tour, disponibilité = fin du terrain (`FIELDWORK_END_OFFSET` quand la table ne la
donne pas) + délai de publication de la source
(`PUBLICATION_LAG_BY_SOURCE`, sauf colonne `published` dans la table), observation de moins
de deux ans (`MAX_STALENESS_DAYS`).

(a) Niveau, une observation par élection (2002 à 2022 : cinq au plus par bloc). Résidu
r_{e,b} = part du bloc au premier tour − prior actuel (points). Deux spécifications : l'attitude
à J−180 (`level`) et sa variation depuis J−180 de l'élection précédente (`change`) ; le prior
étant le résultat précédent, la seconde est le régresseur naturel du résidu. Pour 2002, le registre n'a pas
de présidentielle précédente et le modèle part d'un prior uniforme : l'étude reconstitue le
prior de 2002 avec le premier tour de 1995 (`PRIOR_1995`, Conseil constitutionnel, blocs selon
data/registry/blocs.yaml). Trois prévisions du résidu en leave-one-election-out :

- M0 : prior actuel tel quel (résidu prévu nul) ;
- M1 : prior + biais moyen du bloc (moyenne des résidus des autres élections) ;
- M2 : M1 + pente sur l'attitude (MCO sur les autres élections).

Statistique : gain = RMSE hors échantillon de M1 − RMSE de M2 (points). Placebo : toutes les
permutations des valeurs de l'attitude entre élections (n! ≤ 720), p = part des permutations
dont le gain atteint le gain observé.

(b) Dynamique : panel mensuel de l'étude économie (`vvs.studies.panel`). Pour chaque vague
d'une série infra-annuelle (Eurobaromètre, deux par an), part du bloc dans les sondages du mois
de la vague (à défaut, moyenne des mois voisins observés) ; variations entre deux vagues
consécutives du même cycle électoral. Régression de Δpart sur Δattitude de la même période
(retard 0, contemporain) et de la période précédente (retard 1, seul utilisable pour prévoir).
Placebo : décalage circulaire de la série de vagues d'au moins `PLACEBO_MIN_WAVES` vagues
(autocorrélation conservée), p = part des décalages dont |t| atteint le |t| observé.

Paires a priori (`LEVEL_PAIRS`, signe attendu entre parenthèses), rôles des séries fixés
dans `SERIES_ROLES` avant lecture des valeurs : extrême droite et saillance de l'immigration
(+), de la sécurité (+), attitudes du baromètre d'image Verian « trop d'immigrés », « plus
chez soi » (+), « plus de pouvoir à la police », « justice pas assez sévère » (+), tolérance
CNCDH (−), adhésion aux idées du FN/RN (+), FN/RN danger pour la démocratie (−) ; gauche et
gauche radicale et saillance du chômage (+), des prix (+) ; bloc du président sortant et
confiance dans le gouvernement (Cevipof, +). Chaque série qui remplit un rôle forme sa propre
paire ; les autres séries de la table sont décrites, pas testées.

Critères fixés avant de lire les résultats :

- (a) un test (paire × spécification) est « testable » s'il couvre au moins 4 élections ; il
  « tient » si :
  1. la pente sur toutes les élections a le signe attendu ;
  2. la pente garde ce signe dans chacun des n ajustements leave-one-out ;
  3. RMSE hors échantillon de M2 < celles de M1 et de M0 ;
  4. placebo par permutation p < 0,05, sans correction : avec 4 ou 5 élections, le plus petit
     p atteignable est 1/24 ou 1/120, ce qui rend une correction de Bonferroni sur K tests
     impossible à passer ; le seuil non corrigé est le choix favorable à la variable, le
     nombre de faux positifs attendus (0,05 × K) est rapporté et le backtest reste le filtre ;
  5. gain ≥ 1 point de part sur la meilleure de M0 et M1 (en deçà, sans effet face à
     l'écart-type du prior, 1 sur l'échelle logit, et à l'erreur des sondages à 180 jours,
     environ 6 points).
  Une paire qui tient déclenche l'implémentation (terme dans le prior, désactivé par défaut,
  variante et backtest) ; sinon rien n'est implémenté.
- (b) une paire est testable si elle a au moins 12 variations et 3 cycles ; elle « tient » si,
  au retard 1 : signe attendu, placebo p < 0,05 / K_b (K_b paires testables), signe inchangé
  en retirant chaque cycle, et effet d'un écart type de Δattitude ≥ 0,5 point de part. Le
  retard 0 est rapporté sans verdict : contemporain, il ne prévoit rien.
- Complément sans verdict (`leverage`) : sensibilité de la part a posteriori à l'horizon à un
  décalage de +0,3 sur l'échelle logit du prior d'un bloc (ajustements SVI courts, sur CPU).
  Elle borne l'effet de tout terme dans ce prior, quel que soit le résultat de (a).
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy import stats

from vvs import paths
from vvs.model.data import prior_bloc_shares
from vvs.model.plots import GRID, INK, MUTED, POINTS, RESULT, SERIES, _style
from vvs.registry import default_registry
from vvs.studies import panel as P

ELECTIONS = (2002, 2007, 2012, 2017, 2022)
HORIZON_DAYS = 180
# Premier tour de 1995 (Conseil constitutionnel, % exprimés), par bloc de blocs.yaml : Chirac
# et Balladur droite, Le Pen et de Villiers (MPF) extrême droite, Jospin et Voynet gauche,
# Hue et Laguiller gauche radicale, Cheminade autres.
PRIOR_1995 = {"right": 20.84 + 18.58, "far_right": 15.00 + 4.74, "left": 23.30 + 3.32,
              "far_left": 8.64 + 5.30, "centre": 0.0, "other": 0.28}
DEFAULT_LAG_DAYS = 180
MIN_LEVEL_ELECTIONS = 4
P_PERM = 0.05
MIN_GAIN_POINTS = 1.0
MIN_DYN_CHANGES = 12
MIN_DYN_CYCLES = 3
MIN_DYN_EFFECT = 0.5
PLACEBO_MIN_WAVES = 4
LEVERAGE_SHIFT = 0.3
# 3 000 pas (réglage des tests du backtest) laissent des écarts d'optimisation de plusieurs
# points entre deux ajustements appariés ; 15 000 pas les ramènent sous 0,05 point.
LEVERAGE_SVI_STEPS = 15000
VOTE_1995 = dt.date(1995, 4, 23)
# level : valeur à J−180 ; change : variation depuis J−180 de l'élection précédente (le prior
# étant le résultat précédent, c'est le régresseur naturel du résidu)
SPECS = ("level", "change")
LEVERAGE_HORIZONS = (180, 90)
LEVERAGE_ELECTIONS = (2007, 2012, 2017, 2022)

BLOC_LABEL = {
    "far_left": "Gauche radicale", "left": "Gauche", "centre": "Centre", "right": "Droite",
    "far_right": "Extrême droite", "incumbent": "Bloc du président sortant",
}
ROLE_LABEL = {
    "salience_immigration": "saillance de l'immigration",
    "salience_security": "saillance de la sécurité",
    "salience_unemployment": "saillance du chômage",
    "salience_prices": "saillance des prix",
    "immigration_attitude": "attitude envers l'immigration (Verian)",
    "security_attitude": "attitude sécuritaire (Verian)",
    "tolerance": "indice de tolérance CNCDH",
    "rn_ideas": "adhésion aux idées du FN/RN",
    "rn_danger": "FN/RN danger pour la démocratie",
    "political_trust": "confiance dans le gouvernement (Cevipof)",
}
# (bloc, rôle, signe attendu)
LEVEL_PAIRS = [
    ("far_right", "salience_immigration", +1),
    ("far_right", "salience_security", +1),
    ("far_right", "immigration_attitude", +1),
    ("far_right", "security_attitude", +1),
    ("far_right", "tolerance", -1),
    ("far_right", "rn_ideas", +1),
    ("far_right", "rn_danger", -1),
    ("left", "salience_unemployment", +1),
    ("left", "salience_prices", +1),
    ("far_left", "salience_unemployment", +1),
    ("far_left", "salience_prices", +1),
    ("incumbent", "political_trust", +1),
]
# Rôles des séries nommées du registre (fixés avant lecture des valeurs) ; les autres séries
# du baromètre d'image et du Cevipof sont descriptives, hors test.
SERIES_ROLES = {
    "opinion_too_many_immigrants": "immigration_attitude",
    "opinion_not_at_home_anymore": "immigration_attitude",
    "opinion_more_power_to_police": "security_attitude",
    "opinion_justice_too_lenient": "security_attitude",
    "cncdh_tolerance_index": "tolerance",
    "rn_ideas_agree": "rn_ideas",
    "rn_danger_democracy_yes": "rn_danger",
    "cevipof_trust_government": "political_trust",
}
# Séries de préoccupations (Eurobaromètre, CREDOC, autres) : rôle par motif dans le series_id,
# hors des familles nommées ci-dessus.
NAMED_SOURCES = ("verian", "cevipof", "cncdh")
ROLE_PATTERNS = {
    "salience_immigration": ("immigration",),
    "salience_security": ("security", "crime", "insecurity", "delinquency"),
    "salience_unemployment": ("unemployment", "chomage", "jobs"),
    "salience_prices": ("prices", "inflation", "cost_of_living", "purchasing_power"),
}
# Fin du terrain quand la table ne la donne pas : `date`, sauf CNCDH et DREES où `date` est le
# 1er janvier de l'année d'enquête pour un terrain de fin d'année (CNCDH 2025 : 25 novembre au
# 12 décembre) : `date` + 335 jours.
FIELDWORK_END_OFFSET = {"cncdh": 335, "drees": 335}
# Délai entre la fin du terrain et la publication (jours), par source ; défaut prudent.
# CNCDH : rapport annuel au printemps ou en été suivant (rapport 2025 paru le 2026-06-25).
PUBLICATION_LAG_BY_SOURCE = {"verian": 30, "cevipof": 30, "eurobarometer": 90, "credoc": 180,
                             "cncdh": 200, "drees": 200}
# Une observation plus vieille que cela à la date cible ne compte pas (série arrêtée).
MAX_STALENESS_DAYS = 730


# ---------------------------------------------------------------------------- données


def load_attitudes() -> tuple[pl.DataFrame, list[dict]]:
    att = pl.read_parquet(paths.CURATED / "attitudes.parquet")
    reg_path = paths.REGISTRY / "attitude_series.yaml"
    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    series = reg.get("series", []) if isinstance(reg, dict) else []
    known = {s["series_id"] for s in series}
    for sid in att["series_id"].unique().sort().to_list():
        if sid not in known:
            series.append({"series_id": sid, "label": sid})
    return att, series


def role_of(series_id: str, meta: dict) -> str | None:
    if series_id in SERIES_ROLES:
        return SERIES_ROLES[series_id]
    if meta.get("source") in NAMED_SOURCES or meta.get("family") in ("rn_image", "cncdh",
                                                                   "cevipof_confidence"):
        return None
    for role, pats in ROLE_PATTERNS.items():
        if any(p in series_id for p in pats):
            return role
    return None


def available_on(att: pl.DataFrame) -> pl.DataFrame:
    """Ajoute `obs_end` (fin du terrain) et `available` (date de publication si connue, sinon
    fin du terrain + délai de publication de la source)."""
    pub = pl.col("published").cast(pl.Date) if "published" in att.columns else pl.lit(None, pl.Date)
    off = pl.col("source").replace_strict(FIELDWORK_END_OFFSET, default=0, return_dtype=pl.Int64)
    end = pl.col("date") + pl.duration(days=off)
    if "fieldwork_end" in att.columns:
        end = pl.coalesce(pl.col("fieldwork_end"), end)
    lag = pl.col("source").replace_strict(PUBLICATION_LAG_BY_SOURCE, default=DEFAULT_LAG_DAYS,
                                          return_dtype=pl.Int64)
    att = att.with_columns(end.alias("obs_end"))
    return att.with_columns(
        pl.coalesce(pub, pl.col("obs_end") + pl.duration(days=lag)).alias("available"))


def value_at(att: pl.DataFrame, sid: str, day: dt.date) -> tuple[float, dt.date | None]:
    s = att.filter((pl.col("series_id") == sid) & (pl.col("available") <= day)
                   & (pl.col("obs_end") >= day - dt.timedelta(days=MAX_STALENESS_DAYS))
                   & pl.col("value").is_not_null()).sort("date")
    if s.is_empty():
        return np.nan, None
    r = s.row(-1, named=True)
    return float(r["value"]), r["date"]


def bloc_results(reg) -> pd.DataFrame:
    """(élection, bloc) : part au premier tour, prior actuel (points)."""
    rows = []
    for e in ELECTIONS:
        el = reg.elections[e]
        res: dict[str, float] = {}
        for c in el.round_(1).results:
            b = el.candidacy(c.candidate_id).bloc
            res[b] = res.get(b, 0.0) + c.share
        prior, prev = prior_bloc_shares(reg, e)
        prior = {b: 100 * v for b, v in prior.items()} if prev is not None else dict(PRIOR_1995)
        inc = P.INCUMBENT_BLOC[e]
        for b in (*P.MAIN_BLOCS, P.INCUMBENT):
            src = inc if b == P.INCUMBENT else b
            rows.append({"election": e, "bloc": b, "result": res.get(src, 0.0),
                         "prior": prior.get(src, 0.0),
                         "prior_source": "1995 (reconstitué)" if prev is None else str(prev)})
    df = pd.DataFrame(rows)
    df["residual"] = df["result"] - df["prior"]
    return df


# ---------------------------------------------------------------------------- (a) niveau


def _loo(y: np.ndarray, x: np.ndarray) -> dict:
    n = len(y)
    e0, e1, e2, slopes = [], [], [], []
    for i in range(n):
        m = np.arange(n) != i
        e0.append(y[i])
        e1.append(y[i] - y[m].mean())
        if np.ptp(x[m]) == 0:
            slopes.append(np.nan)
            e2.append(e1[-1])
            continue
        b, a = np.polyfit(x[m], y[m], 1)
        slopes.append(b)
        e2.append(y[i] - (a + b * x[i]))
    rm = lambda v: float(np.sqrt(np.mean(np.square(v))))
    return {"rmse_m0": rm(e0), "rmse_m1": rm(e1), "rmse_m2": rm(e2),
            "loo_slopes": np.array(slopes), "err_m1": np.array(e1), "err_m2": np.array(e2)}


def level_test(lvl: pd.DataFrame, pairs: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    rows = []
    for (b, role, sid, sign), spec in itertools.product(pairs, SPECS):
        d = lvl[(lvl["bloc"] == b) & (lvl["series_id"] == sid) & (lvl["spec"] == spec)]
        d = d.dropna(subset=["x"]).sort_values("election")
        n = len(d)
        row = {"bloc": b, "role": role, "series_id": sid, "spec": spec, "expected_sign": sign,
               "n": n,
               "elections": ",".join(str(e) for e in d["election"])}
        if n < 3 or d["x"].std() == 0:
            rows.append(row | {"testable": False})
            continue
        y, x = d["residual"].to_numpy(), d["x"].to_numpy()
        sl = stats.linregress(x, y)
        lo = _loo(y, x)
        gain = lo["rmse_m1"] - lo["rmse_m2"]
        null = []
        for perm in itertools.permutations(range(n)):
            lp = _loo(y, x[list(perm)])
            null.append(lp["rmse_m1"] - lp["rmse_m2"])
        null = np.array(null)
        row |= {
            "testable": n >= MIN_LEVEL_ELECTIONS,
            "slope": sl.slope, "slope_per_sd": sl.slope * x.std(ddof=1), "r": sl.rvalue,
            "p_ols": sl.pvalue,
            "sign_ok": bool(np.sign(sl.slope) == sign),
            "loo_sign_stable": bool(np.all(np.sign(lo["loo_slopes"]) == sign)),
            "loo_slope_min": float(np.nanmin(lo["loo_slopes"])),
            "loo_slope_max": float(np.nanmax(lo["loo_slopes"])),
            "rmse_m0": lo["rmse_m0"], "rmse_m1": lo["rmse_m1"], "rmse_m2": lo["rmse_m2"],
            "gain_vs_m1": gain, "gain_vs_best": min(lo["rmse_m0"], lo["rmse_m1"]) - lo["rmse_m2"],
            "folds_m2_better": int((np.abs(lo["err_m2"]) < np.abs(lo["err_m1"])).sum()),
            "n_perm": len(null), "p_perm": float((null >= gain - 1e-12).mean()),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    k = int(out["testable"].sum())
    out["expected_false_positives"] = P_PERM * k
    out["holds"] = (
        out["testable"] & out["sign_ok"].fillna(False).astype(bool)
        & out["loo_sign_stable"].fillna(False).astype(bool)
        & (out["rmse_m2"] < out["rmse_m1"]) & (out["rmse_m2"] < out["rmse_m0"])
        & (out["p_perm"] < P_PERM) & (out["gain_vs_best"] >= MIN_GAIN_POINTS)
    )
    return out


# ---------------------------------------------------------------------------- (b) dynamique


def wave_panel(att: pl.DataFrame, sid: str, panel: pd.DataFrame, bloc: str) -> pd.DataFrame:
    """Une ligne par vague de la série : valeur, part du bloc au mois de la vague (à défaut
    moyenne des mois voisins observés du même cycle), cycle électoral."""
    s = att.filter(pl.col("series_id") == sid).sort("date").to_pandas()
    s["month"] = pd.to_datetime(s["date"]).dt.to_period("M").dt.to_timestamp()
    s = s.groupby("month", as_index=False)["value"].mean()
    pb = panel[panel["bloc"] == bloc][["election", "month", "share"]].copy()
    pb["month"] = pd.to_datetime(pb["month"])
    rows = []
    for w, (m, v) in enumerate(zip(s["month"], s["value"])):
        for e, g in pb[pb["month"].between(m - pd.DateOffset(months=1),
                                           m + pd.DateOffset(months=1))].groupby("election"):
            exact = g[(g["month"] == m) & g["share"].notna()]
            sh = exact["share"].mean() if len(exact) else g["share"].mean()
            rows.append({"wave": w, "month": m, "x": v, "election": e, "share": sh})
    return pd.DataFrame(rows, columns=["wave", "month", "x", "election", "share"])


def _changes(wp: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    """Variations entre vagues consécutives (indices de vague w−1 → w) du même cycle, avec la
    valeur de la série `x` (indexée par vague, éventuellement décalée)."""
    rows = []
    for e, g in wp.dropna(subset=["share"]).sort_values("wave").groupby("election"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            w0, w1 = int(g.loc[i - 1, "wave"]), int(g.loc[i, "wave"])
            if w1 - w0 != 1 or w0 < 1:
                continue
            rows.append({"election": e, "wave": w1, "d_share": g.loc[i, "share"] - g.loc[i - 1, "share"],
                         "dx0": x[w1] - x[w0], "dx1": x[w0] - x[w0 - 1]})
    return pd.DataFrame(rows, columns=["election", "wave", "d_share", "dx0", "dx1"])


def _t(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    ok = ~(np.isnan(y) | np.isnan(x))
    y, x = y[ok], x[ok]
    if len(y) < 4 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    sl = stats.linregress(x, y)
    return sl.slope, sl.slope / sl.stderr if sl.stderr > 0 else np.nan, sl.pvalue


def dynamic_test(att: pl.DataFrame, panel: pd.DataFrame,
                 pairs: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    rows = []
    for b, role, sid, sign in pairs:
        wp = wave_panel(att, sid, panel, b)
        if wp.empty:
            continue
        xw = wp.drop_duplicates("wave").set_index("wave")["x"].sort_index()
        x = xw.reindex(range(int(xw.index.max()) + 1)).to_numpy(dtype=float)
        ch = _changes(wp, x)
        nw = len(x)
        for lag, col in ((0, "dx0"), (1, "dx1")):
            y, xv = ch["d_share"].to_numpy(), ch[col].to_numpy()
            slope, t, p = _t(y, xv)
            n = int((~np.isnan(y) & ~np.isnan(xv)).sum())
            cycles = ch.loc[~np.isnan(xv), "election"].nunique()
            row = {"bloc": b, "role": role, "series_id": sid, "expected_sign": sign, "lag": lag,
                   "n": n, "n_cycles": int(cycles), "n_waves": nw, "slope": slope, "t": t,
                   "p_ols": p}
            if np.isnan(t):
                rows.append(row)
                continue
            null = []
            for k in range(PLACEBO_MIN_WAVES, nw - PLACEBO_MIN_WAVES + 1):
                cs = _changes(wp, np.roll(x, k))
                tt = _t(cs["d_share"].to_numpy(), cs[col].to_numpy())[1]
                if not np.isnan(tt):
                    null.append(abs(tt))
            null = np.array(null)
            drop = [_t(ch.loc[ch["election"] != e, "d_share"].to_numpy(),
                       ch.loc[ch["election"] != e, col].to_numpy())[0]
                    for e in sorted(ch["election"].unique())]
            sd = np.nanstd(xv, ddof=1)
            row |= {"n_shifts": len(null),
                    "p_placebo": (1 + (null >= abs(t)).sum()) / (1 + len(null)) if len(null) else np.nan,
                    "sign_ok": bool(np.sign(slope) == sign),
                    "drop_cycle_sign_stable": bool(np.all(np.sign(drop) == sign)),
                    "effect_per_sd": slope * sd}
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    testable = (out["lag"] == 1) & (out["n"] >= MIN_DYN_CHANGES) & (out["n_cycles"] >= MIN_DYN_CYCLES)
    out["testable"] = testable
    k = int(testable.sum())
    out["alpha_bonferroni"] = 0.05 / max(k, 1)
    out["holds"] = (
        testable & out["sign_ok"].fillna(False).astype(bool)
        & out["drop_cycle_sign_stable"].fillna(False).astype(bool)
        & (out["p_placebo"] < out["alpha_bonferroni"])
        & (out["effect_per_sd"].abs() >= MIN_DYN_EFFECT)
    )
    return out


# ---------------------------------------------------------------------------- levier du prior


def leverage(reg) -> pd.DataFrame:
    """Part a posteriori du bloc à la date de prévision, prior décalé de +LEVERAGE_SHIFT
    (logit) sur ce bloc contre prior inchangé ; SVI, même graine, `LEVERAGE_SVI_STEPS` pas."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from vvs.eval.backtest import asof_for
    from vvs.eval.configurations import known_configuration
    from vvs.model.data import encode, load_polls
    from vvs.model.first_round import Priors, fit
    from vvs.model.posterior import Posterior

    rows = []
    for e in LEVERAGE_ELECTIONS:
        for h in LEVERAGE_HORIZONS:
            asof = asof_for(e, h)
            known = known_configuration(e, asof)
            polls = load_polls(e, asof)
            blocs = ("far_right", "left", "far_left", P.INCUMBENT_BLOC[e])
            base = None
            for target in (None, *dict.fromkeys(blocs)):
                d = encode(polls, e, asof, n_weeks=78, reference=known.candidates)
                if target is not None:
                    if target not in d.blocs:
                        continue
                    d.prior_bloc_mean[d.blocs.index(target)] += LEVERAGE_SHIFT
                res = fit(d, method="svi", svi_steps=LEVERAGE_SVI_STEPS, num_samples=1000,
                          seed=0, priors=Priors())
                post = Posterior(d, res)
                sh = post.shares(post.ref_idx, d.asof_week).mean(0) * 100
                cb = [d.blocs[d.cand_bloc[k]] for k in post.ref_idx]
                by = {b: float(sum(s for s, c in zip(sh, cb) if c == b)) for b in set(cb)}
                if target is None:
                    base = by
                    continue
                if target not in by:
                    continue
                rows.append({"election": e, "horizon": h, "bloc": target, "n_polls": d.n_polls,
                             "posterior": base[target], "posterior_shifted": by[target],
                             "delta_points": by[target] - base[target]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- figures


def fig_level(lvl: pd.DataFrame, res: pd.DataFrame, labels: dict, path: Path) -> None:
    r = res[res["n"] >= 3].reset_index(drop=True)
    if r.empty:
        return
    nc = 3
    nr = int(np.ceil(len(r) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, 3.2 * nr), squeeze=False)
    for ax, row in zip(axes.flat, r.itertuples()):
        d = lvl[(lvl["bloc"] == row.bloc) & (lvl["series_id"] == row.series_id)
                & (lvl["spec"] == row.spec)].dropna(subset=["x"])
        ax.scatter(d["x"], d["residual"], color=SERIES, s=24, zorder=3)
        for e, xx, yy in zip(d["election"], d["x"], d["residual"]):
            ax.annotate(str(e), (xx, yy), fontsize=7, color=MUTED, xytext=(3, 3),
                        textcoords="offset points")
        if len(d) >= 3 and d["x"].std() > 0:
            b, a = np.polyfit(d["x"], d["residual"], 1)
            xs = np.linspace(d["x"].min(), d["x"].max(), 10)
            ax.plot(xs, a + b * xs, color=RESULT, lw=1.2)
        ax.axhline(0, color=GRID, lw=0.8)
        _style(ax)
        ax.set_title(f"{BLOC_LABEL[row.bloc]} ~ {row.series_id}\n"
                     f"gain LOO {row.gain_vs_m1:+.2f} pt, p perm. {row.p_perm:.2f}"
                     if row.testable or not np.isnan(getattr(row, "gain_vs_m1", np.nan))
                     else f"{BLOC_LABEL[row.bloc]} ~ {row.series_id}", fontsize=7, color=INK,
                     loc="left")
        ax.set_xlabel("valeur à J−180" if row.spec == "level"
                      else "variation depuis J−180 de l'élection précédente", fontsize=7,
                      color=MUTED)
        ax.set_ylabel("résultat − prior (points)", fontsize=7, color=MUTED)
    for ax in axes.flat[len(r):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_series(att: pl.DataFrame, sid: str, label: str, bloc: str | None, panel: pd.DataFrame,
               first_round: dict, path: Path) -> None:
    """Figure pour le site : série longue et part du bloc associé dans les sondages."""
    s = att.filter(pl.col("series_id") == sid).sort("date").to_pandas()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(pd.to_datetime(s["date"]), s["value"], color=POINTS, lw=1.2, marker="o", ms=2.5)
    ax.set_ylabel(sid, color=POINTS, fontsize=8)
    for v in first_round.values():
        ax.axvline(pd.Timestamp(v), color=GRID, lw=1, ls="--")
    _style(ax)
    if bloc is not None:
        ax2 = ax.twinx()
        pb = panel[(panel["bloc"] == bloc) & panel["share"].notna()]
        ax2.scatter(pd.to_datetime(pb["month"]), pb["share"], s=6, color=SERIES, alpha=0.7)
        ax2.set_ylabel(f"{BLOC_LABEL[bloc]}, sondages (%)", color=SERIES, fontsize=8)
        ax2.tick_params(colors=SERIES, labelsize=8)
        ax2.spines["top"].set_visible(False)
    ax.set_title(f"{label} (gris)\n{BLOC_LABEL.get(bloc, '')} dans les sondages (bleu) — "
                 "n'entre pas dans la prévision", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- rapport


def fnum(v, fmt="{:+.2f}") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return fmt.format(v).replace(".", ",")


def fp(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "<0,001" if v < 0.001 else f"{v:.3f}".replace(".", ",")


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def write_report(out: Path, cover: pd.DataFrame, base: pd.DataFrame, res_a: pd.DataFrame,
                 res_b: pd.DataFrame, lev: pd.DataFrame | None, labels: dict,
                 missing_roles: list[str]) -> None:
    L = ["# Attitudes et prior structurel de bloc (premier tour)", "",
         (f"Run du {dt.date.today().isoformat()}, `back/studies/attitudes_prior.py`. Protocole et "
         "critères dans la docstring, fixés avant lecture. Rien n'entre dans le modèle."), ""]
    L += ["## Séries et couverture", ""]
    L += md_table(["Série", "Rôle", "Première", "Dernière", "Obs.", "Valeur à J−180 pour"],
                  [[f"`{r.series_id}` {labels.get(r.series_id, '')}", ROLE_LABEL[r.role] if isinstance(r.role, str) else "–",
                    str(r.first), str(r.last), str(r.n_obs), r.elections or "–"]
                   for r in cover.itertuples()])
    if missing_roles:
        L += ["", "Rôles sans série disponible : " + ", ".join(ROLE_LABEL[r] for r in missing_roles) + "."]
    L += ["", "## Prior actuel et résidus", ""]
    L += md_table(["Élection", "Prior de"] + [BLOC_LABEL[b] for b in (*P.MAIN_BLOCS, P.INCUMBENT)],
                  [[str(e), g["prior_source"].iloc[0]] + [
                      f"{fnum(r.result, '{:.1f}')} − {fnum(r.prior, '{:.1f}')} = {fnum(r.residual, '{:+.1f}')}"
                      for r in g.set_index("bloc").loc[[*P.MAIN_BLOCS, P.INCUMBENT]].reset_index().itertuples()]
                   for e, g in base.groupby("election")])
    L += ["", "## (a) Niveau : résidu ~ attitude à J−180, leave-one-election-out", ""]
    L += md_table(["Bloc", "Série", "Spéc.", "n", "Pente (pt / ét)", "Signe attendu", "Signe LOO stable",
                   "RMSE M0", "RMSE M1", "RMSE M2", "Gain vs M1", "p perm.", "Tient"],
                  [[BLOC_LABEL[r.bloc], f"`{r.series_id}`",
                    "niveau" if r.spec == "level" else "variation", str(r.n),
                    fnum(getattr(r, "slope_per_sd", np.nan)),
                    "+" if r.expected_sign > 0 else "−",
                    {True: "oui", False: "non"}.get(getattr(r, "loo_sign_stable", None), "–"),
                    fnum(getattr(r, "rmse_m0", np.nan), "{:.2f}"),
                    fnum(getattr(r, "rmse_m1", np.nan), "{:.2f}"),
                    fnum(getattr(r, "rmse_m2", np.nan), "{:.2f}"),
                    fnum(getattr(r, "gain_vs_m1", np.nan)), fp(getattr(r, "p_perm", np.nan)),
                    ("oui" if r.holds else "non") if r.testable else "non testable"]
                   for r in res_a.itertuples()])
    k = int(res_a["testable"].sum()) if len(res_a) else 0
    L += ["", (f"K = {k} tests testables au seuil de placebo {fp(P_PERM)} non corrigé : "
          f"environ {fnum(P_PERM * k, '{:.1f}')} faux positifs attendus par hasard. "
          "M0 : prior actuel ; M1 : prior + biais moyen du bloc ; M2 : M1 + pente.")]
    L += ["", "## (b) Dynamique : Δ part du bloc ~ Δ attitude entre vagues", ""]
    if res_b is None or res_b.empty:
        L += ["Aucune série infra-annuelle utilisable."]
    else:
        L += md_table(["Bloc", "Série", "Retard", "n", "Cycles", "Pente", "Effet / ét (pt)",
                       "p MCO", "p placebo", "Signe stable", "Tient"],
                      [[BLOC_LABEL[r.bloc], f"`{r.series_id}`", str(r.lag), str(r.n),
                        str(r.n_cycles), fnum(r.slope, "{:+.3f}"),
                        fnum(getattr(r, "effect_per_sd", np.nan)), fp(r.p_ols),
                        fp(getattr(r, "p_placebo", np.nan)),
                        {True: "oui", False: "non"}.get(getattr(r, "drop_cycle_sign_stable", None), "–"),
                        ("oui" if r.holds else "non") if r.testable else
                        ("contemporain" if r.lag == 0 else "non testable")]
                       for r in res_b.itertuples()])
    if lev is not None and not lev.empty:
        L += ["", "## Levier du prior (sans verdict)", "",
              (f"Décalage du prior d'un bloc de +{fnum(LEVERAGE_SHIFT, '{:.1f}')} sur l'échelle "
              "logit (environ +30 % relatifs, par exemple 20 → 25 %), effet sur la part a "
              "posteriori du bloc (configuration de référence) à la date de prévision ; SVI à "
              f"{LEVERAGE_SVI_STEPS} pas, même graine, 1 000 tirages."), ""]
        L += md_table(["Élection", "Horizon", "Bloc", "Sondages", "Part a posteriori", "Δ (points)"],
                      [[str(r.election), f"{r.horizon} j", BLOC_LABEL[r.bloc], str(r.n_polls),
                        fnum(r.posterior, "{:.1f}"), fnum(r.delta_points, "{:+.2f}")]
                       for r in lev.itertuples()])
    n_hold_a = int(res_a["holds"].sum()) if len(res_a) else 0
    n_hold_b = int(res_b["holds"].sum()) if res_b is not None and len(res_b) else 0
    L += ["", "## Conclusion", "",
          (f"Niveau : {n_hold_a} test sur {k} tient ; dynamique : {n_hold_b} paire sur "
          f"{int(res_b['testable'].sum()) if res_b is not None and len(res_b) else 0} tient.")]
    if lev is not None and not lev.empty:
        L += ["", (f"Levier du prior : un décalage de +{fnum(LEVERAGE_SHIFT, '{:.1f}')} logit "
              f"déplace la part a posteriori de {fnum(lev['delta_points'].abs().max(), '{:.3f}')} "
              "point au plus (sur tous les cas ci-dessus) : avec `beta0_sd` = 1 et la marche "
              "aléatoire depuis le début de la grille (78 semaines), les sondages fixent le "
              "niveau des blocs dès 180 jours, même avec quatre sondages (2007). Un terme "
              "d'attitude dans ce prior serait sans effet mesurable sur le backtest, quel que "
              "soit son coefficient raisonnable.")]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-attitudes-prior")
    ap.add_argument("--no-leverage", action="store_true")
    ap.add_argument("--reuse-leverage", action="store_true",
                    help="relit <out>/leverage.csv au lieu de refaire les ajustements")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    reg = default_registry()
    first_round = {e: el.round_(1).date for e, el in reg.elections.items()}
    att, series = load_attitudes()
    att = available_on(att)
    labels = {s["series_id"]: s.get("label", s["series_id"]) for s in series}
    src = dict(att.select("series_id", "source").unique("series_id").iter_rows())
    roles = {s["series_id"]: role_of(s["series_id"], s | {"source": src.get(s["series_id"])})
             for s in series if s["series_id"] in src}

    pairs = [(b, role, sid, sign) for b, role, sign in LEVEL_PAIRS
             for sid, r in roles.items() if r == role]
    missing_roles = sorted({role for _, role, _ in LEVEL_PAIRS} - set(roles.values()))

    res = bloc_results(reg)
    rows = []
    vote_prev = {e: first_round.get(e - 5, VOTE_1995) for e in ELECTIONS}
    for b, role, sid, sign in pairs:
        for e in ELECTIONS:
            x, xd = value_at(att, sid, first_round[e] - dt.timedelta(days=HORIZON_DAYS))
            x0, x0d = value_at(att, sid, vote_prev[e] - dt.timedelta(days=HORIZON_DAYS))
            r = res[(res["election"] == e) & (res["bloc"] == b)].iloc[0].to_dict()
            base = {**r, "series_id": sid, "role": role}
            rows.append(base | {"spec": "level", "x": x, "x_date": xd, "x_prev_date": None})
            rows.append(base | {"spec": "change", "x": x - x0, "x_date": xd, "x_prev_date": x0d})
    lvl = pd.DataFrame(rows) if rows else res.assign(series_id=None, role=None, spec=None,
                                                     x=np.nan, x_date=None)
    lvl.to_csv(out / "level_data.csv", index=False)
    res_a = level_test(lvl, pairs) if pairs else pd.DataFrame()
    res_a.to_csv(out / "level_results.csv", index=False)

    cover = []
    for sid in sorted(roles):
        s = att.filter(pl.col("series_id") == sid)
        el = lvl[(lvl["series_id"] == sid) & (lvl["spec"] == "level")
                 & lvl["x"].notna()]["election"].unique()
        cover.append({"series_id": sid, "role": roles[sid], "first": s["date"].min(),
                      "last": s["date"].max(), "n_obs": s.height,
                      "elections": ",".join(str(e) for e in sorted(el))})
    cover = pd.DataFrame(cover)
    cover.to_csv(out / "coverage.csv", index=False)

    by_bloc = pl.read_parquet(paths.CURATED / "poll_results_by_bloc.parquet")
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    panel = P.bloc_panel(P.monthly_bloc_shares(P.poll_bloc_shares(by_bloc, polls)),
                         first_round).to_pandas()
    per_year = (att.with_columns(pl.col("date").dt.year().alias("y"))
                .group_by("series_id", "y").len().group_by("series_id")
                .agg(pl.col("len").median().alias("per_year")))
    sub_annual = set(per_year.filter(pl.col("per_year") >= 2)["series_id"].to_list())
    dyn_pairs = [p for p in pairs if p[2] in sub_annual]
    res_b = dynamic_test(att, panel, dyn_pairs) if dyn_pairs else pd.DataFrame()
    res_b.to_csv(out / "dynamic_results.csv", index=False)

    lev = None
    if args.reuse_leverage and (out / "leverage.csv").exists():
        lev = pd.read_csv(out / "leverage.csv")
    elif not args.no_leverage:
        lev = leverage(reg)
        lev.to_csv(out / "leverage.csv", index=False)

    fig_level(lvl, res_a, labels, out / "level_scatter.png")
    bloc_of_role: dict[str, str] = {}
    for b, role, _ in LEVEL_PAIRS:
        bloc_of_role.setdefault(role, b)
    for sid, role in roles.items():
        if role is None:
            continue
        b = bloc_of_role.get(role)
        fig_series(att, sid, labels.get(sid, sid), None if b == P.INCUMBENT else b, panel,
                   first_round, out / f"series_{sid}.png")

    write_report(out, cover, res, res_a, res_b, lev, labels, missing_roles)
    summary = {
        "level_testable": int(res_a["testable"].sum()) if len(res_a) else 0,
        "level_holds": res_a[res_a["holds"]][["bloc", "series_id", "spec"]].to_dict("records")
        if len(res_a) else [],
        "dynamic_testable": int(res_b["testable"].sum()) if len(res_b) else 0,
        "dynamic_holds": res_b[res_b["holds"]][["bloc", "series_id", "lag"]].to_dict("records")
        if len(res_b) else [],
        "missing_roles": missing_roles,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                 default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"écrit {out}")


if __name__ == "__main__":
    main()
