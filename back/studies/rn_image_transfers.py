"""Étude : normalisation du RN et érosion du front républicain (décalage commun u).

    uv run python studies/rn_image_transfers.py measure  [--out runs/<date>-rn-image]
    uv run python studies/rn_image_transfers.py relation [--out ...]
    uv run python studies/rn_image_transfers.py prior    [--out ...] [--n 20000]
    uv run python studies/rn_image_transfers.py eval     [--out ...] [--seed 0]
    uv run python studies/rn_image_transfers.py report   [--out ...]

Question : la normalisation du RN mesurée par le baromètre d'image Kantar/Verian (« danger pour
la démocratie », « capable de gouverner », « adhésion aux idées ») explique-t-elle l'érosion du
front républicain, c'est-à-dire le décalage commun u des reports vers le finaliste RN
(vvs.model.second_round, docstring du module), et que donnerait-elle comme centre du prior de u
en 2027 ? Outils : vvs.model.rn_image (conventions : u orienté vers le RN, u > 0 = barrage plus
faible que la référence).

Protocole et critères fixés avant la lecture du baromètre (2026-10-23). Les valeurs de u de
l'étude écologique (runs/2026-10-22-ecological) et la tendance générale du baromètre (recul du
« danger » depuis 2002) étaient connues avant d'écrire ces critères ; les valeurs de la série
ingérée (data/curated/attitudes.parquet) ne l'étaient pas.

(1) Cible. Pour chaque second tour avec un finaliste FN/RN (2002, 2017, 2022), décalage commun
    des électorats du barrage (non finalistes hors extrême droite, abstention comprise) :
    - `u_survey` (cible principale) : enquêtes post-électorales de transfers.yaml (FES 2017
      comprise), un institut = la moyenne de ses sources ; référence = matrice de proximité
      (fixe d'une élection à l'autre, donc comparable) ; poids = part de premier tour × part
      d'exprimés observée (sinon celle de la matrice) ;
    - `u_result` : le décalage qui, appliqué à la matrice pour ces électorats, reproduit le
      résultat officiel, les finalistes et l'extrême droite non finaliste étant pris dans les
      enquêtes (absorbe les biais de déclaration des enquêtes) ;
    - 2022 contre l'analogue 2017 (même type de duel) : écart par électorat et commun.
    Intervalles à 90 % : effet institut (dispersion entre instituts d'une même élection,
    poolée sur les élections à plusieurs instituts, divisée par le nombre d'instituts) et
    échantillonnage des cellules (effectif × part de l'électorat, facteur de design 1,5),
    combinés en quadrature.
(2) Points supplémentaires, déclarés comme tels : duels hypothétiques RN sondés avant le
    premier tour (2012 : Marine Le Pen contre Sarkozy, Aubry, Strauss-Kahn, Hollande, avril-juin
    2011 ; aucun duel FN sondé en 2007), u implicite : le décalage qui fait rendre à la matrice
    (parts de premier tour des sondages du moment, fidélité moyenne des finalistes) la part du
    duel, corrigée du biais RN par âge (courbe calibrée hors élection). Contrôle de méthode : le
    même calcul sur les duels réels 2017 et 2022 (veille du second tour, et 60 à 120 jours avant
    le premier tour) doit retrouver `u_result` à l'intervalle près.
(3) Prédicteur principal : « danger pour la démocratie » (% oui) à la dernière vague avant
    l'élection (avant le sondage pour les points hypothétiques). Secondaires, rapportés sans
    servir à la décision (choisir le meilleur de trois items sur trois points serait de la
    sélection) : « capable de gouverner », « adhésion aux idées ». Variations depuis l'élection
    précédente rapportées (deux différences seulement).
(4) Relation : droite de u_survey sur le prédicteur, (a) moindres carrés ordinaires (n = 3,
    un degré de liberté : intervalle de Student à 1 ddl), (b) pondérée par la variance de
    mesure plus un écart résiduel τ = σ_u analogue 0,30 (le jugement du modèle sur la
    variation d'une élection à l'autre).
(5) Critères pour proposer un centrage du prior (règle d'admission à deux niveaux de
    docs/07-evaluation.md : une relation non admissible statistiquement peut centrer un prior
    si on le dit ; elle n'entre pas dans la prévision tant qu'elle n'est pas adoptée) :
    C1 signe : pente négative sur « danger » (plus de danger perçu, barrage plus fort), et
       pentes positives sur les deux items secondaires ;
    C2 chronologie : 2022 prédit depuis 2002 et 2017 seuls (u_2017 + b (X_2022 − X_2017))
       fait mieux que la persistance (u_2022 = u_2017) et tombe à moins de σ_u = 0,30 ;
    C3 cohérence : même signe de pente avec `u_result` qu'avec `u_survey` ;
    C4 points hypothétiques : le point 2012 (corrigé du biais) tombe dans l'intervalle
       prédictif à 90 % de la droite (b) ;
    C5 matérialité : le centrage 2027 qui en résulte s'écarte du centre actuel d'au moins
       0,15 en logit pour au moins un des duels principaux.
    Centrage proposable si C1 à C4 tiennent ; sinon affichage seul. C5 dit seulement s'il y a
    quelque chose à changer.
(6) Prior 2027 : u_2027 = u_2022 + b (X_2026 − X_2022) (ancré sur 2022), écart-type
    √(ΔX² var(b) + var(u_2022) + τ²) ; centre injecté par duel δ = u_2027 − u_centre, u_centre
    étant le décalage commun du prior actuel du modèle par rapport à la matrice (analogue
    2017/2022 pour un duel centre – RN, matrice pour les duels inédits). Duels : Le Pen –
    Philippe, Le Pen – Glucksmann, Le Pen – Mélenchon, Bardella – Philippe, avec sondages de
    duel (configuration publiée : biais RN par âge) et prior seul. Comparé à la persistance
    (centre u_2022, sans baromètre) et au défaut.
(7) Évaluation pour mémoire au protocole apparié du second tour (vvs.eval.second_round :
    élection exclue de tout ce qui est calibré, mêmes tirages de premier tour et même graine
    par variante, graines 0, 1, 2) sur 2002, 2017, 2022, relation réestimée sans l'élection
    évaluée (droite par les deux autres points) : n = 3, ce n'est pas un test. Variante oracle
    (centre = u observé) pour situer le plafond, signalée comme telle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.model import rn_image as R
from vvs.model.plots import display_name
from vvs.model.second_round import (
    ErrorParams,
    SecondRoundModel,
    calibrate_errors,
    first_round_mean,
    load_duel_polls,
    own_prior,
)
from vvs.registry import default_registry
from vvs.registry.transfers import default_transfers

log = logging.getLogger(__name__)

RN_ELECTIONS = (2002, 2017, 2022)
DEFF = 1.5
DEFAULT_N = 1000
Z90 = 1.6449
TAU = 0.30  # σ_u analogue du modèle : variation d'une élection à l'autre
OUT = paths.RUNS / f"{dt.date.today().isoformat()}-rn-image"


def _f(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def _s(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def md(df: pl.DataFrame, nd: int = 2) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(_f(v, nd))
            else:
                cells.append("–" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- (1) mesure de la cible


def finalists(reg, election: int) -> tuple[str, str]:
    res = [r.candidate_id for r in reg.elections[election].round_(2).results]
    return R.rn_pair(reg, election, res)


def pollster_sources(election: int) -> dict[str, list]:
    out: dict[str, list] = {}
    for s in default_transfers():
        if s.election == election and s.base != "other":
            out.setdefault(s.pollster, []).append(s)
    return out


def respondents_share(reg, election: int) -> float:
    r1 = reg.elections[election].round_(1)
    return r1.expressed / r1.registered


def measure_election(reg, election: int, rng: np.random.Generator, n_boot: int = 4000) -> dict:
    """Cellules par groupe et par institut, u_survey, u_result, intervalles d'échantillonnage."""
    pair = finalists(reg, election)
    rn, other = pair
    other_bloc = reg.bloc_of(other, election)
    weights = R.first_round_weights(reg, election)
    gw = R.group_weights(reg, election, pair, weights)
    by_p = {p: R.source_cells(reg, election, pair, ss, weights) for p, ss in pollster_sources(election).items()}
    size = {p: float(np.mean([s.sample_size or DEFAULT_N for s in ss]))
            for p, ss in pollster_sources(election).items()}
    mean = R.mean_cells(by_p)
    barrage = [g for g in sorted(gw) if R.barrage_group(g)]
    rows = []
    resp = respondents_share(reg, election)
    for g in barrage + [g for g in sorted(gw) if not R.barrage_group(g)]:
        m_split, m_none = (R.matrix_cell(g, other_bloc) if R.barrage_group(g) or g == "bloc:far_right"
                           else (np.nan, np.nan))
        c = mean.get(g)
        k = sum(1 for p in by_p if g in by_p[p])
        none = c.none if c and c.none is not None else m_none
        # variance d'échantillonnage du logit de la répartition (moyenne de k instituts)
        var = np.nan
        if c is not None:
            ns = [size[p] * by_p[p][g].weight * resp * (1 - (none if not np.isnan(none) else 0.3))
                  for p in by_p if g in by_p[p]]
            s = min(max(c.split, 0.01), 0.99)
            var = float(np.mean([DEFF / (n * s * (1 - s)) for n in ns])) / k
        rows.append({
            "election": election, "group": g, "weight": gw[g], "k_pollsters": k,
            "survey_split": c.split if c else np.nan, "survey_none": c.none if c and c.none is not None else np.nan,
            "matrix_split": m_split, "matrix_none": m_none, "none_used": none,
            "dlogit": float(R.logit(c.split) - R.logit(m_split)) if c and not np.isnan(m_split) else np.nan,
            "se_logit_sampling": math.sqrt(var) if not np.isnan(var) else np.nan,
            "barrage": R.barrage_group(g),
        })
    cells = pl.DataFrame(rows)
    obs = cells.filter(pl.col("barrage") & pl.col("survey_split").is_not_nan())
    W = (obs["weight"] * (1 - obs["none_used"])).to_numpy()
    u_survey = R.common_shift(obs["matrix_split"].to_numpy(), obs["survey_split"].to_numpy(), W)
    # effet institut : u de chaque institut sur les groupes qu'il couvre, contre la moyenne des
    # instituts sur ces mêmes groupes
    house = []
    if len(by_p) >= 2:
        for p, cs in by_p.items():
            sub = obs.filter(pl.col("group").is_in(list(cs)))
            if sub.height == 0:
                continue
            Wp = (sub["weight"] * (1 - sub["none_used"])).to_numpy()
            up = R.common_shift(sub["matrix_split"].to_numpy(), [cs[g].split for g in sub["group"]], Wp)
            um = R.common_shift(sub["matrix_split"].to_numpy(), sub["survey_split"].to_numpy(), Wp)
            house.append({"election": election, "pollster": p, "groups": sub.height, "d_u": up - um})
    # échantillonnage : tirages des logits des cellules du barrage
    L = R.logit(obs["survey_split"].to_numpy())
    se = obs["se_logit_sampling"].to_numpy()
    draws = [R.common_shift(obs["matrix_split"].to_numpy(), R.expit(L + rng.normal(0, se)), W)
             for _ in range(n_boot // 8)]
    # u_result : finalistes et extrême droite non finaliste pris dans les enquêtes
    own = own_prior(election, transfers=default_transfers(), registry=reg)
    fixed_v = fixed_w = 0.0
    for g in (f"own:{rn}", f"own:{other}", "bloc:far_right"):
        if g not in gw:
            continue
        c = mean.get(g)
        if c is not None:
            split, none = c.split, (c.none if c.none is not None else own[2])
        elif g.startswith("own:"):
            split = own[0] / (own[0] + own[1]) if g == f"own:{rn}" else own[1] / (own[0] + own[1])
            none = own[2]
        else:
            split, none = R.matrix_cell(g, other_bloc)
        fixed_v += gw[g] * (1 - none) * split
        fixed_w += gw[g] * (1 - none)
    bar = cells.filter(pl.col("barrage"))
    Wb = (bar["weight"] * (1 - bar["none_used"])).to_numpy()
    res = {r.candidate_id: r.share for r in reg.elections[election].round_(2).results}
    target = res[rn] / 100
    u_result = R.aggregate_shift(fixed_v, fixed_w, bar["matrix_split"].to_numpy(), Wb, target)
    return {
        "election": election, "pair": pair, "cells": cells, "house": house,
        "u_survey": u_survey, "se_sampling": float(np.std(draws)), "k": len(by_p),
        "u_result": u_result, "rn_result": target, "by_pollster": by_p, "mean": mean,
    }


def analog_shift(m17: dict, m22: dict) -> dict:
    """2022 contre l'analogue 2017 : écart de logit par groupe et décalage commun."""
    c17 = m17["cells"].filter(pl.col("barrage") & pl.col("survey_split").is_not_nan())
    c22 = m22["cells"].filter(pl.col("barrage") & pl.col("survey_split").is_not_nan())
    j = c22.join(c17.select("group", pl.col("survey_split").alias("split_2017"),
                            pl.col("se_logit_sampling").alias("se_2017")), on="group")
    W = (j["weight"] * (1 - j["none_used"])).to_numpy()
    u = R.common_shift(j["split_2017"].to_numpy(), j["survey_split"].to_numpy(), W)
    rows = j.select("group", "weight", pl.col("split_2017"), pl.col("survey_split").alias("split_2022"),
                    (pl.col("survey_split").map_elements(lambda x: float(R.logit(x)), return_dtype=pl.Float64)
                     - pl.col("split_2017").map_elements(lambda x: float(R.logit(x)), return_dtype=pl.Float64)
                     ).alias("dlogit"),
                    ((pl.col("se_logit_sampling") ** 2 + pl.col("se_2017") ** 2).sqrt()).alias("se_sampling"))
    return {"u": u, "rows": rows, "n": j.height}


def run_measure(out: Path) -> dict:
    reg = default_registry()
    rng = np.random.default_rng(20261023)
    ms = {e: measure_election(reg, e, rng) for e in RN_ELECTIONS}
    # effet institut poolé : élections à au moins deux instituts
    house = pl.DataFrame([h for m in ms.values() for h in m["house"]])
    dof = sum(m["k"] - 1 for m in ms.values() if m["k"] >= 2)
    tau_house = math.sqrt(float((house["d_u"] ** 2).sum()) / dof) if dof else float("nan")
    summary = []
    for e, m in ms.items():
        se = math.sqrt(m["se_sampling"] ** 2 + tau_house**2 / m["k"])
        m["se"] = se
        summary.append({
            "election": e, "duel": "–".join(m["pair"]), "pollsters": m["k"],
            "u_survey": m["u_survey"], "se_sampling": m["se_sampling"], "se_house": tau_house / math.sqrt(m["k"]),
            "se": se, "lo90": m["u_survey"] - Z90 * se, "hi90": m["u_survey"] + Z90 * se,
            "u_result": m["u_result"], "rn_result": 100 * m["rn_result"],
        })
    summ = pl.DataFrame(summary)
    an = analog_shift(ms[2017], ms[2022])
    out.mkdir(parents=True, exist_ok=True)
    pl.concat([m["cells"] for m in ms.values()]).write_csv(out / "u_cells.csv")
    summ.write_csv(out / "u_observed.csv")
    house.write_csv(out / "u_house.csv")
    an["rows"].write_csv(out / "u_analog_2022_vs_2017.csv")
    meta = {"tau_house": tau_house, "house_dof": dof, "u_2022_vs_2017": an["u"],
            "u_2022_vs_2017_groups": an["n"]}
    (out / "measure.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {"ms": ms, "summary": summ, "analog": an, "meta": meta}


# --- (2) points hypothétiques : u implicite des sondages de duel


def poll_implied(
    reg, election: int, pair: tuple[str, str], start: dt.date, end: dt.date, errors: ErrorParams,
    profile: dict[str, tuple[float | None, float]], fr_asof: dt.date | None = None,
    official_first_round: bool = False,
) -> dict | None:
    """u implicite d'une moyenne de sondages de duel terminés dans [start, end].

    `profile` : groupe -> (répartition fixe ou None, part ni l'un ni l'autre) d'un second tour RN
    typique (moyenne des enquêtes 2002, 2017, 2022 hors élection évaluée) : la part « ni l'un ni
    l'autre » de la matrice est bien trop forte pour un duel RN (2002 : gauche 38 % contre 14 %
    observé), et l'électorat d'extrême droite non finaliste n'y suit pas la matrice."""
    rn, other = R.rn_pair(reg, election, pair)
    polls = load_duel_polls(election, end, (rn, other))
    if polls.n == 0:
        return None
    keep = np.array([start <= d <= end for d in polls.fieldwork_end])
    if not keep.any():
        return None
    share = polls.share[keep]
    sizes = polls.sample_size[keep]
    ends = [d for d, k in zip(polls.fieldwork_end, keep) if k]
    d2 = reg.elections[election].round_(2).date
    age = float(np.mean([(d2 - d).days for d in ends]))
    raw = float(share.mean()) / 100
    bias = float(errors.bias_at(age)) / 100
    if official_first_round:
        fr = {r.candidate_id: r.share / 100 for r in reg.elections[election].round_(1).results}
        fr_note = "premier tour officiel"
    else:
        fr, (s0, s1), nh = first_round_mean(election, fr_asof or end, (rn, other), 30)
        fr_note = f"{nh} hypothèses de premier tour du {s0 + dt.timedelta(days=1)} au {s1}"
    weights = R.first_round_weights(reg, election, fr)
    gw = R.group_weights(reg, election, (rn, other), weights)
    other_bloc = reg.bloc_of(other, election)
    own = own_prior(election, transfers=default_transfers(), registry=reg)
    fixed_v = fixed_w = 0.0
    ref, W = [], []
    for g, w in gw.items():
        if g == f"own:{rn}":
            s, n = own[0] / (own[0] + own[1]), own[2]
        elif g == f"own:{other}":
            s, n = own[1] / (own[0] + own[1]), own[2]
        else:
            s, n = R.matrix_cell(g, other_bloc)
            fs, fn = profile.get(g, (None, n))
            s, n = (fs if fs is not None and not R.barrage_group(g) else s), fn
        if R.barrage_group(g):
            ref.append(s)
            W.append(w * (1 - n))
        else:
            fixed_v += w * (1 - n) * s
            fixed_w += w * (1 - n)
    out = {"election": election, "rn": rn, "other": other, "other_bloc": other_bloc,
           "start": start, "end": end, "n_polls": int(keep.sum()), "age_days": age,
           "poll_rn": 100 * raw, "bias": 100 * bias, "first_round": fr_note}
    for name, tgt in (("u_raw", raw), ("u_corrected", raw - bias)):
        try:
            out[name] = R.aggregate_shift(fixed_v, fixed_w, ref, W, tgt)
        except ValueError:
            out[name] = float("nan")  # part hors d'atteinte avec ces fidélités et abstentions
    if math.isnan(out["u_corrected"]):
        out["se"] = float("nan")
        return out
    # écart-type : erreur d'horizon (σ0² + σ1² âge / 30) et échantillonnage de la moyenne,
    # ramenés en logit par la dérivée numérique de la part
    sd_share = math.sqrt(float(errors.variance(age))
                         + float(np.mean(100**2 * raw * (1 - raw) / (0.6 * sizes))) / keep.sum()) / 100
    eps = 0.005
    up = R.aggregate_shift(fixed_v, fixed_w, ref, W, raw - bias + eps)
    out["se"] = sd_share * (up - out["u_corrected"]) / eps
    return out


def typical_profile(cells: pl.DataFrame, exclude: int | None = None) -> dict:
    """Groupe -> (répartition de l'extrême droite non finaliste ou None, part ni l'un ni
    l'autre) : moyennes des enquêtes des seconds tours RN, hors `exclude`."""
    c = cells.filter(pl.col("election") != (exclude or 0))
    out = {}
    for g in c["group"].unique().to_list():
        if g.startswith("own:"):
            continue
        sub = c.filter(pl.col("group") == g)
        none = sub["survey_none"].drop_nans().mean()
        none = float(none) if none is not None else float(sub["matrix_none"].mean())
        split = None if R.barrage_group(g) else float(sub["survey_split"].drop_nans().mean())
        out[g] = (split, none)
    return out


def run_hypothetical(out: Path) -> pl.DataFrame:
    reg = default_registry()
    cells = pl.read_csv(out / "u_cells.csv")
    rows = []
    err = {e: calibrate_errors(exclude={e}, registry=reg, bias_by_horizon=True)
           for e in (2002, 2012, 2017, 2022)}
    # 2012 : duels hypothétiques d'avril-juin 2011 (seuls duels RN sondés avant 2017)
    for other in ("sarkozy_nicolas", "aubry_martine", "strauss_kahn_dominique", "hollande_francois"):
        r = poll_implied(reg, 2012, ("le_pen_marine", other), dt.date(2011, 1, 1), dt.date(2011, 12, 31),
                         err[2012], typical_profile(cells))
        if r:
            rows.append(r | {"kind": "hypothétique"})
    # contrôle de méthode : duels réels, veille du second tour et 60-120 jours avant le premier
    for e in (2002, 2017, 2022):
        el = reg.elections[e]
        d1, d2 = el.round_(1).date, el.round_(2).date
        pair = finalists(reg, e)
        prof = typical_profile(cells, exclude=e)
        r = poll_implied(reg, e, pair, d1 + dt.timedelta(days=1), d2 - dt.timedelta(days=1), err[e],
                         prof, official_first_round=True)
        if r:
            rows.append(r | {"kind": "réel, entre les tours"})
        try:
            r = poll_implied(reg, e, pair, d1 - dt.timedelta(days=120), d1 - dt.timedelta(days=60), err[e],
                             prof)
        except KeyError:
            r = None
        if r:
            rows.append(r | {"kind": "réel, 60-120 j avant le 1er tour"})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(out / "u_poll_implied.csv")
    return df


# --- (3)-(4) baromètre d'image et relation

# séries du baromètre (data/registry/attitude_series.yaml) retenues pour chaque rôle. « Capable de
# gouverner » : la seule mesure du baromètre est « a la capacité de participer à un gouvernement »,
# posée depuis janvier 2011 (absente pour 2002).
ITEMS = {
    "danger": "rn_danger_democracy_yes",
    "capable": "rn_can_join_government",
    "adhesion": "rn_ideas_agree",
}
ITEM_LABEL = {"danger": "danger pour la démocratie", "capable": "capable de gouverner",
              "adhesion": "adhésion aux idées"}
EXPECTED_SIGN = {"danger": -1, "capable": +1, "adhesion": +1}
HYPO_DATE = dt.date(2011, 4, 7)  # premier duel hypothétique 2012 (Le Pen – Sarkozy, Aubry)
ASOF_2027_DEFAULT = dt.date(2026, 10, 23)
# dernière vague en face-à-face : la vague de janvier 2026 passe en ligne sur panel (Verian :
# « évolutions à considérer avec précaution »)
FACE_TO_FACE_END = dt.date(2025, 12, 31)


def load_barometer(parquet: Path, registry: Path | None = None) -> dict[str, pl.DataFrame]:
    """Séries du baromètre d'image FN/RN par rôle (`ITEMS`) : (date, value, series_id)."""
    att = pl.read_parquet(parquet)
    out = {}
    for role, sid in ITEMS.items():
        df = (att.filter((pl.col("series_id") == sid) & pl.col("value").is_not_null())
              .select(pl.col("date").cast(pl.Date), pl.col("value").cast(pl.Float64)).sort("date"))
        if df.height:
            out[role] = df.with_columns(pl.lit(sid).alias("series_id"))
    return out


def wave_before(s: pl.DataFrame, day: dt.date) -> tuple[float, dt.date] | None:
    sub = s.filter(pl.col("date") <= day)
    if sub.is_empty():
        return None
    return float(sub["value"][-1]), sub["date"][-1]


def hypothetical_2012(out: Path) -> dict:
    """Point 2012 combiné : moyenne des duels pondérée par la précision ; écart-type du plus
    précis (erreur d'horizon commune aux duels d'une même période : pas d'indépendance)."""
    h = pl.read_csv(out / "u_poll_implied.csv").filter(pl.col("kind") == "hypothétique")
    h = h.filter(pl.col("u_corrected").is_not_nan())
    w = 1 / h["se"].to_numpy() ** 2
    u = float((w * h["u_corrected"].to_numpy()).sum() / w.sum())
    ur = float((w * h["u_raw"].to_numpy()).sum() / w.sum())
    return {"u": u, "u_raw": ur, "se": float(h["se"].min()), "n_duels": h.height}


def barometer_points(bar: dict, reg, cutoff_round: int = 1, max_2027: dt.date = ASOF_2027_DEFAULT
                     ) -> pl.DataFrame:
    """Valeur du baromètre à la dernière vague avant le tour `cutoff_round` de chaque élection,
    avant le premier duel hypothétique de 2011 (point 2012) et avant `max_2027` (prior 2027)."""
    dates = {e: reg.elections[e].round_(cutoff_round).date for e in RN_ELECTIONS}
    dates[2012] = HYPO_DATE
    dates[2027] = max_2027
    rows = []
    for role, srs in bar.items():
        for e, d in sorted(dates.items()):
            wb = wave_before(srs, d)
            rows.append({"item": role, "series_id": srs["series_id"][0], "point": e, "cutoff": d,
                         "wave": wb[1] if wb else None, "value": wb[0] if wb else None})
    return pl.DataFrame(rows)


def relation_core(xs: pl.DataFrame, u: dict, h12: dict) -> dict:
    def X(role: str, e: int) -> float | None:
        r = xs.filter((pl.col("item") == role) & (pl.col("point") == e))
        return None if r.is_empty() else r["value"][0]

    rel: dict = {"items": {}, "h2012": h12, "tau": TAU}
    for role in xs["item"].unique(maintain_order=True).to_list():
        xe = [X(role, e) for e in RN_ELECTIONS]
        ys = [u[e]["u_survey"] for e in RN_ELECTIONS]
        if any(v is None for v in xe):
            ok = [(x, y) for x, y in zip(xe, ys) if x is not None]
            item = {"missing": [e for e, v in zip(RN_ELECTIONS, xe) if v is None],
                    "x": {e: v for e, v in zip(RN_ELECTIONS, xe) if v is not None}}
            if len(ok) == 2:
                item["slope_2pts"] = (ok[1][1] - ok[0][1]) / (ok[1][0] - ok[0][0])
            x27 = X(role, 2027)
            if x27 is not None and xe[-1] is not None:
                item["x2027"], item["dx_2027"] = x27, x27 - xe[-1]
            rel["items"][role] = item
            continue
        yr = [u[e]["u_result"] for e in RN_ELECTIONS]
        se = [u[e]["se"] for e in RN_ELECTIONS]
        b, b_se = R.ols_slope(xe, ys)
        br, _ = R.ols_slope(xe, yr)
        lf = R.fit_line(xe, ys, se, tau=TAU)
        item = {
            "x": dict(zip(RN_ELECTIONS, xe)), "ols_slope": b, "ols_se": b_se,
            # Student à 1 ddl (n = 3) : t_0,95 = 6,314
            "ols_lo90": b - 6.314 * b_se, "ols_hi90": b + 6.314 * b_se,
            "ols_slope_result": br, "wls_slope": lf.b, "wls_se": math.sqrt(lf.cov[1, 1]),
            "wls_a": lf.a, "wls_x0": lf.x0, "wls_cov": lf.cov.tolist(), "wls_rss": lf.rss,
            "r2": float(np.corrcoef(xe, ys)[0, 1] ** 2),
            "d_x": {"2017-2002": xe[1] - xe[0], "2022-2017": xe[2] - xe[1]},
            "d_u": {"2017-2002": ys[1] - ys[0], "2022-2017": ys[2] - ys[1]},
            "slope_2017_2022": (ys[2] - ys[1]) / (xe[2] - xe[1]) if xe[2] != xe[1] else None,
        }
        # C2 : 2022 depuis 2002 et 2017 seuls
        b0217 = (ys[1] - ys[0]) / (xe[1] - xe[0])
        item["c2_pred_2022"] = ys[1] + b0217 * (xe[2] - xe[1])
        item["c2_err"] = item["c2_pred_2022"] - ys[2]
        item["c2_persistence_err"] = ys[1] - ys[2]
        # C4 : point 2012 contre la droite pondérée
        x12 = X(role, 2012)
        if x12 is not None:
            p, sd = lf.predict(x12)
            item["c4_x2012"] = x12
            item["c4_pred"] = p
            item["c4_sd"] = math.sqrt(sd**2 + h12["se"] ** 2)
            item["c4_z"] = (h12["u"] - p) / item["c4_sd"]
        # prior 2027, ancré sur 2022
        x27 = X(role, 2027)
        if x27 is not None:
            dx = x27 - xe[2]
            item["x2027"] = x27
            item["dx_2027"] = dx
            item["prior_2027"] = {
                "center": ys[2] + lf.b * dx,
                "sd": math.sqrt(dx**2 * lf.cov[1, 1] + se[2] ** 2 + TAU**2),
                "center_ols": ys[2] + b * dx,
                "center_2017_2022": ys[2] + item["slope_2017_2022"] * dx
                if item["slope_2017_2022"] is not None else None,
                # pente locale 2017 → 2022 : erreur type de Δu (mesures 2017 et 2022) / Δx
                "sd_2017_2022": math.sqrt((dx * math.hypot(se[1], se[2]) / (xe[2] - xe[1])) ** 2
                                          + se[2] ** 2 + TAU**2) if xe[2] != xe[1] else None,
                "line_pred": lf.predict(x27),
            }
        # LOEO : droite par les deux autres points
        loeo = {}
        for k, e in enumerate(RN_ELECTIONS):
            idx = [i for i in range(3) if i != k]
            bb = (ys[idx[1]] - ys[idx[0]]) / (xe[idx[1]] - xe[idx[0]])
            loeo[e] = {"u_hat": ys[idx[0]] + bb * (xe[k] - xe[idx[0]]), "slope": bb}
        item["loeo"] = loeo
        rel["items"][role] = item
    d = rel["items"]["danger"]
    sec = [r for r in ("capable", "adhesion") if r in rel["items"] and "ols_slope" in rel["items"][r]]
    crit = {
        # item secondaire sans 2002 (« capable », depuis 2011) : pente sur les deux points restants
        "C1": d["ols_slope"] < 0 and all(
            (it.get("ols_slope") if it.get("ols_slope") is not None else it.get("slope_2pts", 0.0)) > 0
            for r, it in rel["items"].items() if r != "danger"),
        "C1_secondary_available": sec,
        "C2": abs(d["c2_err"]) < abs(d["c2_persistence_err"]) and abs(d["c2_err"]) < TAU,
        "C3": np.sign(d["ols_slope_result"]) == np.sign(d["ols_slope"]),
        "C4": ("c4_z" in d) and abs(d["c4_z"]) <= Z90,
    }
    crit = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in crit.items()}
    crit["centering_proposable"] = all(crit[k] for k in ("C1", "C2", "C3", "C4"))
    rel["criteria"] = crit
    rel["prior_2027"] = d.get("prior_2027")
    rel["loeo"] = {str(k): v for k, v in d["loeo"].items()}
    return rel


def run_relation(out: Path, parquet: Path, registry: Path) -> dict:
    reg = default_registry()
    bar = load_barometer(parquet, registry)
    if "danger" not in bar:
        raise SystemExit("baromètre : série « danger pour la démocratie » introuvable")
    obs = pl.read_csv(out / "u_observed.csv")
    meta = json.loads((out / "measure.json").read_text())
    u = {r["election"]: r for r in obs.iter_rows(named=True)}
    h12 = hypothetical_2012(out)
    xs = barometer_points(bar, reg)
    xs.write_csv(out / "barometer_values.csv")
    rel = relation_core(xs, u, h12)
    rel["tau_house"] = meta["tau_house"]
    # sensibilités (rapportées, sans servir à la décision)
    sens = {}
    xs2 = barometer_points(bar, reg, cutoff_round=2)
    xs2.write_csv(out / "barometer_values_round2.csv")
    sens["vague avant le second tour (2002 : mai 2002)"] = relation_core(xs2, u, h12)
    xs3 = barometer_points(bar, reg, max_2027=FACE_TO_FACE_END)
    sens["2027 : dernière vague en face-à-face (novembre 2024)"] = relation_core(xs3, u, h12)
    rel["sensitivity"] = {k: {"criteria": v["criteria"], "danger": {
        kk: v["items"]["danger"].get(kk) for kk in ("x", "ols_slope", "wls_slope", "c2_pred_2022", "c2_err",
                                                    "x2027", "dx_2027", "prior_2027", "loeo")}}
        for k, v in sens.items()}
    (out / "relation.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False, default=str))
    return rel


# --- centre actuel du modèle et centrage injecté


def model_center(model: SecondRoundModel, reg, election: int, pair: tuple[str, str],
                 fr: dict[str, float]) -> dict:
    """Décalage commun du prior actuel (moyennes des groupes du barrage ajustés par `model`)
    par rapport à la matrice, poids = parts de premier tour `fr` × part d'exprimés du prior."""
    rn, other = R.rn_pair(reg, election, pair)
    f = model.pair_fit((rn, other))
    other_bloc = reg.bloc_of(other, election)
    gw = R.group_weights(reg, election, (rn, other), R.first_round_weights(reg, election, fr)
                         if reg.elections[election].round_(1).results else
                         {**fr, "abstain": f.abstain_weight})
    ref, obs, W, src = [], [], [], {}
    for g, w in gw.items():
        if not R.barrage_group(g) or g not in f.groups:
            continue
        s, n = R.prior_cell(f.groups[g], rn_first=True)
        m, _ = R.matrix_cell(g, other_bloc)
        ref.append(m)
        obs.append(s)
        W.append(w * (1 - n))
        src[g] = f.groups[g].source
    return {"u_center": R.common_shift(ref, obs, W), "sources": src, "inedit": f.inedit,
            "sigma_u": f.sigma_u}


DUELS = (
    ("le_pen_marine", "philippe_edouard"),
    ("le_pen_marine", "glucksmann_raphael"),
    ("le_pen_marine", "melenchon_jean_luc"),
    ("bardella_jordan", "philippe_edouard"),
)
ASOF_2027 = dt.date(2026, 10, 23)


def run_prior(out: Path, n: int, seed: int) -> pl.DataFrame:
    """Duels 2027 sous trois centrages de u (défaut, persistance de 2022, relation du
    baromètre), avec et sans sondages de duel, mêmes tirages de premier tour par duel."""
    from vvs.model.second_round import (
        SecondRoundSettings,
        first_round_concentration,
        noisy_first_round,
    )

    reg = default_registry()
    rel = json.loads((out / "relation.json").read_text())
    obs = pl.read_csv(out / "u_observed.csv")
    u22 = float(obs.filter(pl.col("election") == 2022)["u_survey"][0])
    u27, sd27 = rel["prior_2027"]["center"], rel["prior_2027"]["sd"]
    u27_loc, sd27_loc = rel["prior_2027"]["center_2017_2022"], rel["prior_2027"]["sd_2017_2022"]
    d1 = reg.elections[2027].round_(1).date
    horizon = (d1 - ASOF_2027).days
    errors = calibrate_errors(exclude={2027}, registry=reg, bias_by_horizon=True)
    kappa = first_round_concentration(horizon, exclude={2027})
    base = SecondRoundSettings()
    wide = base.with_(sigma_u={"analog": max(sd27, base.sigma_u["analog"]),
                               "inedit": max(sd27, base.sigma_u["inedit"])})
    wide_loc = base.with_(sigma_u={"analog": max(sd27_loc, base.sigma_u["analog"]),
                                   "inedit": max(sd27_loc, base.sigma_u["inedit"])})
    rows = []
    for k, pair in enumerate(DUELS):
        fr_mean, (start, end), nh = first_round_mean(2027, ASOF_2027, pair, 14)
        fr = noisy_first_round(fr_mean, kappa, np.random.default_rng([seed, k, 1]), 4000)
        center = model_center(SecondRoundModel(reg, errors=errors).fit(2027, ASOF_2027, [pair]),
                              reg, 2027, pair, fr_mean)
        rn, other = R.rn_pair(reg, 2027, pair)
        variants = {
            "défaut": (0.0, base),
            "persistance 2022": (u22 - center["u_center"], base),
            "relation baromètre": (u27 - center["u_center"], wide),
            "pente 2017-2022": (u27_loc - center["u_center"], wide_loc),
        }
        for use_polls in (True, False):
            for name, (delta, st) in variants.items():
                m = R.ShiftedSecondRoundModel(reg, errors=errors, settings=st, use_polls=use_polls,
                                              u_shift=delta).fit(2027, ASOF_2027, [(rn, other)])
                d = m.sample((rn, other), fr, horizon, np.random.default_rng([seed, k, 2]), n)
                sh = d.share
                rows.append({
                    "duel": f"{rn}–{other}", "polls": "avec sondages" if use_polls else "prior seul",
                    "variant": name, "u_center": center["u_center"], "delta": delta,
                    "sigma_u": m.pair_fit((rn, other)).sigma_u, "inedit": center["inedit"],
                    "n_polls": m.pair_fit((rn, other)).polls.n,
                    "median": float(np.median(sh)), "q05": float(np.quantile(sh, 0.05)),
                    "q95": float(np.quantile(sh, 0.95)), "p_rn": float(np.mean(sh > 50)),
                    "prior_median": float(np.median(d.prior_share)), "ess": d.ess,
                    "first_round": f"{nh} hypothèses, {start + dt.timedelta(days=1)} – {end}",
                })
            log.info("%s %s fait", pair, use_polls)
    df = pl.DataFrame(rows)
    df.write_csv(out / "prior_2027_duels.csv")
    return df


def run_eval(out: Path, seed: int, n: int = 8000) -> pl.DataFrame:
    """Protocole apparié du second tour sur 2002, 2017, 2022 (pour mémoire, n = 3)."""
    from vvs.eval import second_round as ev

    reg = default_registry()
    rel = json.loads((out / "relation.json").read_text())
    loo = {int(k): v for k, v in rel["loeo"].items()}
    obs = pl.read_csv(out / "u_observed.csv")
    rows = []
    for e in RN_ELECTIONS:
        errors_h = calibrate_errors(exclude={e}, registry=reg, bias_by_horizon=True)
        u_res = float(obs.filter(pl.col("election") == e)["u_result"][0])
        for label in ev.HORIZONS:
            hseed = list(ev.HORIZONS).index(label)
            try:
                c = ev.case(e, label, np.random.default_rng([seed, e, hseed, 1]), reg)
            except KeyError as err:
                log.warning("%s %s : %s", e, label, err)
                continue
            base = SecondRoundModel(reg, errors=errors_h).fit(e, c.asof, [c.pair])
            fr_mean = {k: float(np.mean(v)) for k, v in c.first_round.items()}
            center = model_center(base, reg, e, c.pair, fr_mean)["u_center"]
            variants = {"référence": 0.0, "relation (sans l'élection)": loo[e]["u_hat"] - center,
                        "oracle (u_result)": u_res - center}
            for name, delta in variants.items():
                m = R.ShiftedSecondRoundModel(reg, errors=errors_h, u_shift=delta).fit(e, c.asof, [c.pair])
                d = m.sample(c.pair, c.first_round, c.horizon, np.random.default_rng([seed, e, hseed, 3]), n)
                rows.append({"election": e, "horizon": label, "variant": name, "seed": seed,
                             "u_center": center, "delta": delta, "result": c.result,
                             "n_polls": m.pair_fit(c.pair).polls.n} | ev.score(d.share, c.result))
            log.info("%s %s fait", e, label)
    df = pl.DataFrame(rows)
    df.write_csv(out / f"eval_{seed}.csv")
    return df


# --- rapport

ECOLOGICAL = {2002: -1.85, 2007: -0.47, 2012: -0.58, 2017: -0.53, 2022: -0.28}
GROUP_LABEL = {"abstain": "abstention, blancs, nuls", "bloc:far_left": "gauche radicale",
               "bloc:left": "gauche", "bloc:centre": "centre", "bloc:right": "droite",
               "bloc:other": "autres", "bloc:far_right": "extrême droite non finaliste"}
HORIZON_LABEL = {"r2_eve": "veille du 2d tour", "r1_30": "30 j", "r1_90": "90 j", "r1_180": "180 j"}


def figures(out: Path, bar: dict[str, pl.DataFrame], rel: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vvs.model.plots import INK, MUTED, POINTS, RESULT, SERIES, _style

    reg = default_registry()
    files = []
    # (a) séries du baromètre
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = {"danger": RESULT, "capable": SERIES, "adhesion": POINTS}
    for role, srs in bar.items():
        ax.plot(srs["date"].to_list(), srs["value"].to_list(), marker="o", ms=2.5, lw=1.2,
                color=colors[role], label=ITEM_LABEL[role])
    for e in (2002, 2007, 2012, 2017, 2022):
        ax.axvline(reg.elections[e].round_(1).date, color=MUTED, lw=0.6, ls=":")
    _style(ax)
    ax.set_ylabel("% des personnes interrogées", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("Baromètre d'image du FN/RN (Kantar, Verian) — n'entre pas dans la prévision",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_barometer.png", dpi=150)
    plt.close(fig)
    files.append("fig_barometer.png")
    # (b) u contre « danger »
    d = rel["items"]["danger"]
    obs = pl.read_csv(out / "u_observed.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [d["x"][str(e)] if str(e) in d["x"] else d["x"][e] for e in RN_ELECTIONS]
    ys = obs["u_survey"].to_list()
    se = obs["se"].to_list()
    ax.errorbar(xs, ys, yerr=[Z90 * v for v in se], fmt="o", color=SERIES, capsize=3,
                label="u observé (enquêtes), 90 %")
    ax.scatter(xs, obs["u_result"].to_list(), marker="x", color=RESULT, label="u du résultat agrégé")
    for x, y, e in zip(xs, ys, RN_ELECTIONS):
        ax.annotate(str(e), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    if "c4_x2012" in d:
        h = rel["h2012"]
        ax.errorbar([d["c4_x2012"]], [h["u"]], yerr=[Z90 * h["se"]], fmt="s", mfc="white",
                    color=POINTS, capsize=3, label="2012 hypothétique (sondages 2011)")
    grid = np.linspace(min(xs + [d.get("x2027", xs[-1])]) - 3, max(xs) + 3, 50)
    lf = R.LineFit(d["wls_a"], d["wls_slope"], d["wls_x0"], np.array(d["wls_cov"]), TAU, 3, d["wls_rss"])
    pr = np.array([lf.predict(x) for x in grid])
    ax.plot(grid, pr[:, 0], color=MUTED, lw=1)
    ax.fill_between(grid, pr[:, 0] - Z90 * pr[:, 1], pr[:, 0] + Z90 * pr[:, 1], color=MUTED, alpha=0.12,
                    lw=0)
    if rel.get("prior_2027"):
        p = rel["prior_2027"]
        ax.errorbar([d["x2027"]], [p["center"]], yerr=[Z90 * p["sd"]], fmt="D", color=RESULT, mfc="white",
                    capsize=3, label="2027 proposé (ancré sur 2022)")
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    _style(ax)
    ax.set_xlabel("« danger pour la démocratie » (% oui), dernière vague avant le vote", fontsize=8,
                  color=MUTED)
    ax.set_ylabel("u (logit, vers le RN ; référence : matrice)", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Érosion du front républicain et image du RN (n = 3)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_u_vs_danger.png", dpi=150)
    plt.close(fig)
    files.append("fig_u_vs_danger.png")
    # (c) écarts par électorat
    cells = pl.read_csv(out / "u_cells.csv").filter(pl.col("barrage") & pl.col("dlogit").is_not_nan())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    groups = [g for g in GROUP_LABEL if g in cells["group"].to_list() and g != "bloc:far_right"]
    for k, e in enumerate(RN_ELECTIONS):
        c = cells.filter(pl.col("election") == e)
        m = dict(zip(c["group"], c["dlogit"]))
        sd = dict(zip(c["group"], c["se_logit_sampling"]))
        xs_ = [i + (k - 1) * 0.25 for i, g in enumerate(groups) if g in m]
        ax.errorbar(xs_, [m[g] for g in groups if g in m], yerr=[Z90 * sd[g] for g in groups if g in m],
                    fmt="o", ms=4, capsize=2, color=[POINTS, SERIES, RESULT][k], label=str(e))
    ax.set_xticks(range(len(groups)), [GROUP_LABEL[g] for g in groups], fontsize=7)
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    _style(ax)
    ax.set_ylabel("logit(observé) − logit(matrice)", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Écart à la matrice de proximité par électorat (enquêtes post-électorales)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_u_by_electorate.png", dpi=150)
    plt.close(fig)
    files.append("fig_u_by_electorate.png")
    return files


def name(duel: str) -> str:
    return " – ".join(display_name(c) for c in duel.split("–"))


def report(out: Path, parquet: Path, registry: Path) -> str:
    reg = default_registry()
    rel = json.loads((out / "relation.json").read_text())
    meta = json.loads((out / "measure.json").read_text())
    obs = pl.read_csv(out / "u_observed.csv")
    cells = pl.read_csv(out / "u_cells.csv")
    an = pl.read_csv(out / "u_analog_2022_vs_2017.csv")
    hyp = pl.read_csv(out / "u_poll_implied.csv")
    xs = pl.read_csv(out / "barometer_values.csv", try_parse_dates=True)
    bar = load_barometer(parquet, registry)
    figs = figures(out, bar, rel)
    d = rel["items"]["danger"]
    L: list[str] = [
        f"# Normalisation du RN et érosion du front républicain ({dt.date.today().isoformat()})",
        "",
        ("Étude `back/studies/rn_image_transfers.py`, outils `vvs.model.rn_image`. Protocole et critères "
        "fixés avant la lecture du baromètre : voir l'en-tête du script. **Rien de cette étude n'entre "
        "dans la prévision** : les défauts de `vvs.model.second_round` sont inchangés, le centrage "
        "2027 est une proposition."),
        "",
        ("Convention : u en logit, orienté vers le finaliste FN/RN ; u > 0 = reports plus favorables "
        "au RN que la référence (barrage plus faible). Électorats du barrage : non finalistes hors "
        "extrême droite, abstention, blancs et nuls compris."),
        "",
        "## 1. Décalage commun observé",
        "",
        ("Référence : la matrice de proximité du modèle (fixe, donc comparable d'une élection à "
        "l'autre). `u_survey` : enquêtes post-électorales (un institut = la moyenne de ses sources, "
        "FES 2017 comprise) ; `u_result` : décalage qui reproduit le résultat officiel, finalistes "
        "et extrême droite non finaliste pris dans les enquêtes. Intervalle à 90 % : effet institut "
        f"(τ = {_f(meta['tau_house'])} logit par institut, {meta['house_dof']} degrés de liberté "
        "poolés) et échantillonnage des cellules (facteur de design 1,5). Étude écologique : "
        "moyenne pondérée des écarts de logit, électorat d'extrême droite non finaliste compris, "
        "poids des électorats de l'inférence écologique."),
        "",
        "| élection | duel | instituts | u_survey | 90 % | u_result | part RN | étude écologique |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in obs.iter_rows(named=True):
        L.append(f"| {r['election']} | {name(r['duel'])} | {r['pollsters']} | "
                 f"{_s(r['u_survey'])} | [{_s(r['lo90'])} ; {_s(r['hi90'])}] | {_s(r['u_result'])} | "
                 f"{_f(r['rn_result'], 1)} | {_s(ECOLOGICAL[r['election']])} |")
    L += [
        "",
        (f"2022 contre l'analogue 2017 (même type de duel, {meta['u_2022_vs_2017_groups']} électorats "
        f"communs, poids 2022) : **u = {_s(meta['u_2022_vs_2017'])}** (étude écologique, électorat "
        "Zemmour/Dupont-Aignan compris : +0,82)."),
        "",
        ("Les valeurs 2007 (−0,47) et 2012 (−0,58) de l'étude écologique portent sur des duels "
        "droite – gauche : elles mesurent l'écart de la matrice dans un duel sans RN et ne sont pas "
        "des points de la relation."),
        "",
        "Par électorat (écart de logit à la matrice ; écart-type d'échantillonnage) :",
        "",
        "| électorat | " + " | ".join(str(e) for e in RN_ELECTIONS) + " | 2022 − 2017 |",
        "|---|" + "---|" * (len(RN_ELECTIONS) + 1),
    ]
    anm = {r["group"]: r for r in an.iter_rows(named=True)}
    for g, lab in GROUP_LABEL.items():
        vals = []
        for e in RN_ELECTIONS:
            c = cells.filter((pl.col("election") == e) & (pl.col("group") == g))
            if c.is_empty() or c["dlogit"][0] is None or math.isnan(c["dlogit"][0]):
                vals.append("–")
            else:
                vals.append(f"{_s(c['dlogit'][0])} ({_f(c['se_logit_sampling'][0])}) "
                            f"[{_f(100 * c['survey_split'][0], 0)} %]")
        a = anm.get(g)
        vals.append(f"{_s(a['dlogit'])} ({_f(a['se_sampling'])})" if a else "–")
        L.append(f"| {lab} | " + " | ".join(vals) + " |")
    L += [
        "",
        ("Entre crochets : part du RN parmi les exprimés de l'électorat. La ligne extrême droite est "
        "hors du décalage commun (matrice : 98 % vers le RN face au centre)."),
        "",
        f"![écarts par électorat]({figs[2]})",
        "",
        "## 2. Points hypothétiques (sondages de duel) et contrôle de la méthode",
        "",
        ("u implicite : décalage qui fait rendre à la matrice la part du duel dans les sondages, "
        "parts de premier tour des sondages du moment, fidélité moyenne des finalistes, parts « ni "
        "l'un ni l'autre » et électorat d'extrême droite non finaliste moyens des seconds tours RN "
        "(hors élection). Corrigé : moins le biais RN des sondages de duel à cet âge (courbe calibrée "
        "hors élection). Écart-type : erreur d'horizon calibrée et échantillonnage."),
        "",
        "| élection | duel | type | sondages | âge (j) | RN sondé | biais | u brut | u corrigé | é.-t. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in hyp.iter_rows(named=True):
        L.append(f"| {r['election']} | {display_name(r['rn'])} – {display_name(r['other'])} | "
                 f"{r['kind']} | {r['n_polls']} | {_f(r['age_days'], 0)} | {_f(r['poll_rn'], 1)} | "
                 f"{_f(r['bias'], 1)} | {_s(r['u_raw'])} | {_s(r['u_corrected'])} | {_f(r['se'])} |")
    h = rel["h2012"]
    ctrl = hyp.filter(pl.col("kind") == "réel, entre les tours").join(
        obs.select("election", "u_result"), on="election")
    off = (ctrl["u_corrected"] - ctrl["u_result"]).drop_nans()
    L += [
        "",
        (f"Point 2012 combiné ({h['n_duels']} duels, pondéré par la précision) : u = {_s(h['u'])} "
        f"(brut {_s(h['u_raw'])}), écart-type {_f(h['se'])} (celui du duel le plus précis : l'erreur "
        "d'horizon est commune aux duels d'une même période). Aucun duel FN n'est sondé avant le premier tour en 2002 ni en 2007 dans les données (poll_results, archives, Wikipédia)."),
        "",
        "Contrôle (duels réels, entre les tours, premier tour officiel) : u implicite moins u_result = "
        + ", ".join(f"{e} {_s(v)}" for e, v in zip(ctrl["election"], ctrl["u_corrected"] - ctrl["u_result"]))
        + (f" ; décalage moyen {_s(float(off.mean()))}." if off.len() else ".")
        + " En 2002 la part corrigée (17,0 %) est hors d'atteinte avec la fidélité moyenne des "
        "finalistes : l'électorat Le Pen de 2002 n'a reporté qu'à 89 % (enquête Ipsos), contre 96 à "
        "97 % depuis.",
        "",
        "## 3. Baromètre d'image du FN/RN",
        "",
        ("Source : data/curated/attitudes.parquet, registre data/registry/attitude_series.yaml "
        "(ingestion d'un autre agent). Dernière vague avant la date indiquée."),
        "",
        "| item | série | point | date limite | vague | valeur |",
        "|---|---|---|---|---|---|",
    ]
    for r in xs.iter_rows(named=True):
        L.append(f"| {ITEM_LABEL[r['item']]} | `{r['series_id']}` | {r['point']} | {r['cutoff']} | "
                 f"{r['wave'] or '–'} | {_f(r['value'], 0) if r['value'] is not None else '–'} |")
    L += ["", f"![baromètre]({figs[0]})", "", "## 4. Relation entre l'image du RN et u", "",
          ("| item | pente MCO (par point) | 90 % (Student, 1 ddl) | R² | pente avec u_result | "
          "pente pondérée (τ = 0,30) | é.-t. | Δx 2002→2017 / 2017→2022 | Δu |"),
          "|---|---|---|---|---|---|---|---|---|"]
    for role, it in rel["items"].items():
        if "ols_slope" not in it:
            sl = it.get("slope_2pts")
            L.append(f"| {ITEM_LABEL[role]} | absent en {', '.join(map(str, it.get('missing')))} ; "
                     f"pente 2017 → 2022 seule : {_s(sl, 3) if sl is not None else '–'} "
                     "| – | – | – | – | – | – | – |")
            continue
        L.append(
            f"| {ITEM_LABEL[role]} | {_s(it['ols_slope'], 3)} | [{_s(it['ols_lo90'], 3)} ; "
            f"{_s(it['ols_hi90'], 3)}] | {_f(it['r2'])} | {_s(it['ols_slope_result'], 3)} | "
            f"{_s(it['wls_slope'], 3)} | {_f(it['wls_se'], 3)} | "
            f"{_s(it['d_x']['2017-2002'], 0)} / {_s(it['d_x']['2022-2017'], 0)} | "
            f"{_s(it['d_u']['2017-2002'])} / {_s(it['d_u']['2022-2017'])} |")
    c = rel["criteria"]
    yes = {True: "oui", False: "non"}
    L += [
        "",
        f"![u contre danger]({figs[1]})",
        "",
        "Critères (fixés avant lecture, prédicteur principal « danger ») :",
        "",
        (f"- C1 signe (danger < 0, items secondaires > 0) : **{yes[c['C1']]}** — danger "
         f"{_s(d['ols_slope'], 3)}, adhésion {_s(rel['items']['adhesion'].get('ols_slope'), 3)}, mais « capable "
         "de gouverner », absent en 2002, recule de 38 à 34 % entre 2017 et 2022 quand u monte de "
         f"0,35 : pente {_s(rel['items']['capable'].get('slope_2pts'), 3)}, signe contraire ; l'adhésion "
         "recule aussi entre 2017 et 2022 (33 → 29 %) : son signe d'ensemble tient au seul point 2002 ;"),
        (f"- C2 chronologie : 2022 prédit depuis 2002 et 2017 = {_s(d['c2_pred_2022'])} pour "
        f"{_s(obs.filter(pl.col('election') == 2022)['u_survey'][0])} observé (erreur {_s(d['c2_err'])}, "
        f"persistance {_s(d['c2_persistence_err'])}) : **{yes[c['C2']]}** ;"),
        f"- C3 même signe avec u_result : **{yes[c['C3']]}** ;",
        (f"- C4 point 2012 : prédit {_s(d['c4_pred'])} ± {_f(Z90 * d['c4_sd'])} (90 %), observé "
         f"{_s(rel['h2012']['u'])}, z = {_s(d['c4_z'])} : **{yes[c['C4']]}** ;" if "c4_z" in d else
         "- C4 point 2012 : pas de vague du baromètre avant avril 2011 : **non** ;"),
        f"- centrage proposable (C1 à C4) : **{yes[c['centering_proposable']]}**.",
        "",
        ("Le point 2002 commande tout : de 2002 à 2017, « danger » ne perd que 4 points (62 → 58, vague "
         "de mai 2000 faute de vague 2001) quand u gagne 1,65 ; de 2017 à 2022 il perd 10 points pour "
         f"0,35 de u (pente locale {_s(d['slope_2017_2022'], 3)} par point, trois fois plus faible). Une "
         "droite unique ne décrit pas ces deux intervalles : 2002 est un choc (21 avril, Le Pen père, "
         "duel droite – extrême droite) plus qu'un point d'une relation continue."),
        "",
        "Sensibilités (sans rôle dans la décision) :",
        "",
    ]
    for k, v in rel.get("sensitivity", {}).items():
        sd_ = v["danger"]
        p_ = sd_.get("prior_2027") or {}
        L.append(f"- {k} : x = {sd_['x']}, pente MCO {_s(sd_['ols_slope'], 3)}, 2022 prédit depuis "
                 f"2002-2017 {_s(sd_['c2_pred_2022'])} (C2 {yes[v['criteria']['C2']]}), x 2027 = "
                 f"{_f(sd_['x2027'], 0)}, u 2027 = {_s(p_.get('center'))} ± {_f(p_.get('sd'))} "
                 f"(pente locale 2017-2022 : {_s(p_.get('center_2017_2022'))}) ;")
    L += [
        "",
        "Validation croisée par élection (droite par les deux autres points) : "
        + ", ".join(f"{e} prédit {_s(v['u_hat'])} (pente {_s(v['slope'], 3)})" for e, v in rel["loeo"].items())
        + ".",
        "",
    ]
    # prior 2027
    pr = pl.read_csv(out / "prior_2027_duels.csv") if (out / "prior_2027_duels.csv").exists() else None
    p27 = rel.get("prior_2027")
    L += ["## 5. Prior 2027", ""]
    if p27:
        u22 = float(obs.filter(pl.col("election") == 2022)["u_survey"][0])
        L += [
            (f"Dernière vague : {_f(d['x2027'], 0)} % « danger » ({_s(d['dx_2027'], 0)} depuis la vague "
            f"de 2022). u_2027 = u_2022 + b Δx = {_s(u22)} {_s(d['wls_slope'] * d['dx_2027'])} = "
            f"**{_s(p27['center'])}**, écart-type **{_f(p27['sd'])}** (pente, mesure de 2022, τ = 0,30) ; "
            f"avec la pente MCO : {_s(p27['center_ols'])} ; prédiction par la droite seule (non ancrée) : "
            f"{_s(p27['line_pred'][0])} ± {_f(p27['line_pred'][1])}."),
            "",
            (f"Variante « pente 2017-2022 » (pente locale, la seule paire d'élections du même type de duel) : "
             f"u_2027 = {_s(p27['center_2017_2022'])} ± {_f(p27['sd_2017_2022'])}. La vague de janvier 2026 "
             "est la première en ligne sur panel (face-à-face auparavant ; Verian : « évolutions à considérer "
             "avec précaution ») : elle remonte à 41 % après 38 % en novembre 2024 ; avec la dernière vague "
             "en face-à-face, le centre passe à "
             f"{_s(rel['sensitivity']['2027 : dernière vague en face-à-face (novembre 2024)']['danger']['prior_2027']['center'])}."),
            "",
            ("Centre actuel du modèle (décalage commun de son prior, par rapport à la matrice, poids du "
            "premier tour 2027 sondé) : analogue 2017/2022 pour les duels centre – RN (σ_u 0,30), "
            "matrice pour les duels inédits (σ_u 0,60)."),
            "",
        ]
    if pr is not None:
        L += [("| duel | sondages | centrage | u centre actuel | δ injecté | σ_u | médiane RN | 90 % | "
              "P(RN gagne) | médiane du prior |"), "|---|---|---|---|---|---|---|---|---|---|"]
        for r in pr.iter_rows(named=True):
            L.append(f"| {name(r['duel'])} | {r['polls']} ({r['n_polls']}) | {r['variant']} | "
                     f"{_s(r['u_center'])} | {_s(r['delta'])} | {_f(r['sigma_u'])} | {_f(r['median'], 1)} | "
                     f"{_f(r['q05'], 1)} – {_f(r['q95'], 1)} | {_f(100 * r['p_rn'], 0)} % | "
                     f"{_f(r['prior_median'], 1)} |")
        L += ["", (f"Date : {ASOF_2027}, horizon {(reg.elections[2027].round_(1).date - ASOF_2027).days} "
              "jours ; premier tour : moyenne des sondages de 14 jours bruitée (Dirichlet), comme "
              "`second_round.forecast` ; biais RN par âge (configuration publiée). Probabilités "
              "conditionnelles au duel, pas la probabilité de victoire de la simulation complète."), ""]
    # évaluation
    ev_files = sorted(out.glob("eval_*.csv"))
    if ev_files:
        m = pl.concat([pl.read_csv(f) for f in ev_files])
        agg = (m.group_by("horizon", "variant").agg(
            pl.col("abs_error").mean(), pl.col("crps").mean(), pl.col("log_score").mean(),
            pl.col("cover50").mean(), pl.col("cover90").mean(), pl.len())
            .with_columns(pl.col("horizon").replace(HORIZON_LABEL).alias("horizon")))
        order = {v: k for k, v in enumerate(HORIZON_LABEL.values())}
        agg = agg.with_columns(pl.col("horizon").replace_strict(order, default=9).alias("o")).sort("o", "variant")
        L += ["## 6. Évaluation pour mémoire (protocole apparié du second tour, 2002, 2017, 2022)", "",
              (f"Graines : {sorted(m['seed'].unique().to_list())}, 8 000 tirages par cas. Relation réestimée "
              "sans l'élection évaluée (droite par les deux autres points). **n = 3 : ce n'est pas un "
              "test**, et la relation est estimée sur les mêmes trois élections. L'oracle (centre = "
              "u_result de l'élection) situe le plafond de ce qu'un centrage parfait apporterait."), "",
              "| horizon | variante | erreur abs. | CRPS | log-score | couv. 50 | couv. 90 | cas |",
              "|---|---|---|---|---|---|---|---|"]
        for r in agg.iter_rows(named=True):
            L.append(f"| {r['horizon']} | {r['variant']} | {_f(r['abs_error'])} | {_f(r['crps'])} | "
                     f"{_f(r['log_score'], 4)} | {_f(r['cover50'])} | {_f(r['cover90'])} | {r['len']} |")
        per = (m.group_by("election", "horizon", "variant").agg(pl.col("crps").mean(), pl.col("median").mean(),
                                                               pl.col("delta").first(), pl.col("result").first())
               .with_columns(pl.col("horizon").replace_strict({k: i for i, k in enumerate(HORIZON_LABEL)},
                                                              default=9).alias("o"))
               .sort("election", "o", "variant"))
        L += ["", "Par élection (CRPS moyen sur les graines, médiane prévue, δ injecté) :", "",
              "| élection | horizon | variante | δ | médiane | résultat | CRPS |", "|---|---|---|---|---|---|---|"]
        for r in per.iter_rows(named=True):
            L.append(f"| {r['election']} | {HORIZON_LABEL[r['horizon']]} | {r['variant']} | {_s(r['delta'])} | "
                     f"{_f(r['median'], 1)} | {_f(r['result'], 1)} | {_f(r['crps'])} |")
        L += ["", ("Lecture : la relation sans l'élection gagne en CRPS moyen à 30, 90 et 180 jours parce que "
                   "le centre actuel est trop favorable au RN en 2002 (pas de sondage de duel avant le premier "
                   "tour) et en 2017 ; elle perd lourdement en 2022 (droite tirée par 2002 et 2017, δ ≈ +4) et "
                   "perd en log-score du vainqueur à 90 et 180 jours. L'oracle montre que le centre de u "
                   "compte surtout quand les sondages de duel manquent (2002) et peu quand ils abondent "
                   "(2022 : 0,48 → 0,43 de CRPS à la veille). Trois élections, une relation estimée sur "
                   "les mêmes : aucune conclusion d'admission."), ""]
    L += [
        "## 7. Volet site (proposition)",
        "",
        ("Page second tour, encadré « Front républicain et image du RN », marqué **n'entre pas dans "
        "la prévision** tant que le centrage n'est pas adopté :"),
        "",
        ("1. la série du baromètre d'image (trois items, 1984-2026, `fig_barometer.png`), élections "
        "présidentielles marquées ;"),
        ("2. les reports observés par élection : part du finaliste RN dans chaque électorat (2002, "
        "2017, 2022, enquêtes post-électorales) et le décalage commun u avec son intervalle "
        "(`u_observed.csv`, `u_cells.csv`), plus le point 2012 hypothétique distingué ;"),
        ("3. le nuage u contre « danger » avec la droite et sa bande, en écrivant n = 3 sur la figure "
        "(`fig_u_vs_danger.png`) ;"),
        ("4. pour chaque duel principal, la part du RN avec le prior actuel et avec le centrage "
        "proposé, côte à côte (`prior_2027_duels.csv`, avec sondages), la différence étant "
        "présentée comme une sensibilité, pas comme une prévision alternative."),
        "",
    ]
    if p27:
        u22 = float(obs.filter(pl.col("election") == 2022)["u_survey"][0])
        L += [
            "## 8. Avis",
            "",
            (f"**Ne pas centrer le prior sur cette relation ; l'afficher.** Les critères fixés avant "
             f"lecture ne sont pas remplis (C1 {yes[c['C1']]}, C2 {yes[c['C2']]}) ; C5 (matérialité) "
             f"l'est largement, ce qui aggrave le problème : la droite pondérée déplacerait u de "
             f"{_s(p27['center'] - u22)} logit au-delà de 2022 (δ ≈ +0,9 sur le centre actuel), soit "
             "+2 à +2,5 points de médiane RN avec sondages et environ +8 points sur le prior seul, "
             "sur la foi de trois points dont un (2002) fait toute la pente."),
            "",
            ("Ce qui tient : (i) l'érosion elle-même est mesurée proprement et dans le même sens par "
             "les enquêtes et par le résultat agrégé (u ≈ −1,9 en 2002, −0,3/−0,45 en 2017, +0,05 en "
             "2022 par rapport à la matrice ; +0,43 de 2017 à 2022 sur les électorats communs) ; "
             "(ii) le recul du « danger » va avec elle dans le temps ; (iii) le point 2012 hypothétique "
             "ne la contredit pas, avec une incertitude énorme. Ce qui ne tient pas : une relation "
             "linéaire stable, la reproduction de 2022 depuis 2002 et 2017, le signe des items "
             "secondaires sur le seul intervalle comparable (2017 → 2022), la comparabilité de la "
             "vague 2026 (changement de mode)."),
            "",
            (f"Si l'on veut un centrage prudent malgré tout, la seule version défendable est la pente "
             f"locale 2017 → 2022 (u_2027 = {_s(p27['center_2017_2022'])} ± {_f(p27['sd_2017_2022'])}, "
             "δ ≈ +0,3 à +0,4 sur le centre actuel, +0,5 à +1 point de médiane RN avec sondages) : elle "
             "revient presque à prolonger la tendance 2017 → 2022, sans le baromètre. Le défaut actuel "
             "(moyenne des analogues 2017 et 2022, u ≈ −0,09 sur la matrice) est plus favorable au "
             "barrage que 2022 ; la persistance de 2022 seule (δ ≈ +0,15) est le changement minimal "
             "cohérent avec les données, à évaluer comme réglage injectable au protocole du second "
             "tour, pas comme effet du baromètre. Aucun des deux n'est admis ici."),
            "",
        ]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


# --- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["measure", "relation", "prior", "eval", "report"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--attitudes", default=str(paths.CURATED / "attitudes.parquet"))
    ap.add_argument("--series", default=str(paths.REGISTRY / "attitude_series.yaml"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "measure":
        r = run_measure(out)
        print(md(r["summary"]))
        print(r["meta"])
        print(md(r["analog"]["rows"]))
        for e, m in r["ms"].items():
            print(e)
            print(md(m["cells"].drop("election")))
        h = run_hypothetical(out)
        print(md(h.drop("first_round")))
    elif a.step == "relation":
        rel = run_relation(out, Path(a.attitudes), Path(a.series))
        print(json.dumps(rel, indent=1, ensure_ascii=False, default=str))
    elif a.step == "prior":
        print(md(run_prior(out, a.n, a.seed).drop("first_round")))
    elif a.step == "eval":
        print(md(run_eval(out, a.seed)))
    elif a.step == "report":
        report(out, Path(a.attitudes), Path(a.series))
        print(out / "report.md")


if __name__ == "__main__":
    main()
