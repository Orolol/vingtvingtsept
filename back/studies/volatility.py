"""Étude : la volatilité réalisée de la campagne annonce-t-elle l'erreur de projection ?

    uv run python studies/volatility.py [all|study|kappa|score|current] [--out runs/<date>-volatility]

Indice : `vvs.model.volatility` (variance hebdomadaire du mouvement des log-parts agrégées par
semaine, bruit d'échantillonnage retiré, moyenne pondérée par les parts ; fenêtres de 4, 8 et
12 semaines, 8 par défaut). Quatre volets :

1. France 2002-2022, horizons 7 à 180 jours : relation entre l'indice à la date et (a) le
   mouvement ultérieur (log-parts, de l'agrégat des 30 jours avant la date à l'agrégat final des
   14 derniers jours, et jusqu'au résultat), au niveau de la campagne (écart quadratique moyen
   des log-erreurs centrées, pondéré par les parts), des candidats et des blocs ; (b) le z-score
   du backtest de référence (runs/2026-10-20-backtest, variante floor, modèle + η) : écart sur
   l'échelle logit entre médiane projetée et résultat, rapporté à l'écart type projeté
   ((q95 − q05) / 3,29), écart quadratique moyen par cas.
2. Jennings–Wlezien hors France (présidentielles et législatives, premier tour) : même indice
   sur les parts quotidiennes des partis, même relation avec le mouvement ultérieur ; beaucoup
   plus de cas, sert de forme a priori (élasticité β).
3. Forme σ_eff = σ · (V / V_ref)^κ, κ et ν estimés en validation croisée interne sur les
   données de calibration de η, η recalibré par pli (`vvs.model.volatility.fit_kappa`) ;
   formes `move` (erreur finale non mise à l'échelle) et `total`, échelle globale ou par bloc,
   et κ fixé à l'élasticité de Jennings–Wlezien.
4. Backtest sur les tirages floor existants (aucun nouvel ajustement NUTS) : 2012, 2017, 2022,
   et 2002, 2007 à poids 0,5, quatre horizons, nombres aléatoires communs avec floor.

Critères fixés avant de lire les relations (l'estimateur de l'indice a été révisé une fois sur
la seule inspection des indices, voir le docstring de `vvs.model.volatility`) :

- C1 (France, mouvement) : corrélation de Spearman, centrée par horizon, entre log V et le log
  de l'écart quadratique moyen du mouvement jusqu'au résultat, sur la grille fine
  d'horizons ; positive et au-delà du 95e centile des 120 permutations des élections.
- C2 (Jennings–Wlezien) : élasticité β du mouvement ultérieur (vers les derniers sondages) à V
  positive, intervalle à 95 % par bootstrap des pays excluant 0, à au moins trois des quatre
  horizons 7 (vers le résultat), 30, 90, 180.
- C3 (backtest) : Spearman centrée par horizon entre log V et log z quadratique moyen sur les
  20 cas, positive et au-delà du 90e centile des permutations des élections.
- C4 (adoption) : règle de docs/07 (log-score et CRPS améliorés aux horizons où l'option agit,
  même signe sur deux des trois élections de référence, aucun autre horizon dégradé,
  couverture non dégradée) ; la volatilité agit à tous les horizons, on teste les paires
  (180, 90) et (30, 7). Correction de structure de l'erreur : si l'écart de couverture à 50 et
  90 % à la nominale se réduit en moyenne pondérée sans perte de log-score (≥ −0,01 à chaque
  horizon) ni de CRPS (≤ +0,01), on recommande l'adoption même sans gain strict.

n honnête : cinq élections françaises ; les horizons d'une même élection ne sont pas
indépendants (marche emboîtée), les permutations se font donc par élection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from vvs import paths
from vvs.eval import scoring as sc
from vvs.eval.backtest import ELECTION_WEIGHT, asof_for, fit_dir
from vvs.eval.baselines import last_by_pollster, poll_mean
from vvs.eval.configurations import known_configuration, official_map
from vvs.model import error as er
from vvs.model import volatility as vo
from vvs.model.data import election_date, load_polls
from vvs.model.plots import GRID, INK, MUTED, _style
from vvs.model.run import _hash_frame, draws_log_shares
from vvs.registry import default_registry

ELECTIONS = (2002, 2007, 2012, 2017, 2022)
REFERENCE = (2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
FINE = (7, 14, 21, 30, 45, 60, 90, 120, 150, 180)
JW_HORIZONS = (7, 14, 30, 60, 90, 120, 180)
DEFAULT_REF = paths.RUNS / "2026-10-20-backtest"
COLORS = {2002: "#8a8984", 2007: "#3a9b5c", 2012: "#eb6834", 2017: "#2a78d6", 2022: "#b03a8f"}
N_PLACEBO = 2000
VERSIONS = ("floor", "vol_move", "vol_total", "vol_bloc", "vol_move_jw")
VERSION_LABEL = {
    "floor": "floor (référence)",
    "vol_move": "mouvement × s, κ LOEO",
    "vol_total": "η entier × s, κ LOEO",
    "vol_bloc": "mouvement × s_bloc, κ LOEO",
    "vol_move_jw": "mouvement × s, κ = β JW",
}


def _rms(d: np.ndarray, w: np.ndarray) -> float:
    """Écart quadratique moyen des log-erreurs centrées (moyenne pondérée par w) : invariant au
    décalage commun, comme les contrastes du modèle d'erreur."""
    w = w / w.sum()
    c = d - (w * d).sum()
    return float(np.sqrt((w * c**2).sum()))


def _centered(d: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    return d - (w * d).sum()


# --- 1. France : indice, mouvement ultérieur


def french_volatility(reg) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(indice par élection × horizon × fenêtre, volatilités par unité à la fenêtre principale)."""
    rows, units = [], []
    for e in ELECTIONS:
        for h in sorted(set(FINE) | set(vo.REF_HORIZONS)):
            for w in vo.WINDOWS:
                v = vo.volatility_at_horizon(e, h, w, reg)
                rows.append({"election": e, "horizon": h, "window": w, "V": v.campaign,
                             "V_raw": v.campaign_raw, "V_points": v.campaign_points,
                             "V_bloc": v.bloc_campaign, "n_polls": v.n_polls,
                             "n_weeks": v.n_weeks})
                if w != vo.PRIMARY_WINDOW:
                    continue
                for u, x in v.by_unit.items():
                    units.append({"election": e, "horizon": h, "level": "candidate", "unit": u,
                                  "V": x, "weight": v.weights[u]})
                for b, x in v.by_bloc.items():
                    units.append({"election": e, "horizon": h, "level": "bloc", "unit": b,
                                  "V": x, "weight": math.nan})
    return pl.DataFrame(rows), pl.DataFrame(units)


def french_moves(reg) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mouvement ultérieur par cas (campagne) et par unité.

    d_move = log(agrégat final 14 j) − log(agrégat des 30 j avant la date) (`error.move_items`),
    d_result = d_move + erreur finale (`error.final_item`), sur les candidats officiels communs."""
    cases, units = [], []
    for e in ELECTIONS:
        fin = er.final_item(e, reg)
        dfin = dict(zip(fin.units, fin.d))
        for h in FINE:
            its = er.move_items(e, reg, (h,))
            if not its:
                continue
            it = its[0]
            keep = [i for i, u in enumerate(it.units) if u in dfin]
            p = it.p[keep] / it.p[keep].sum()
            dm = it.d[keep]
            dr = dm + np.array([dfin[it.units[i]] for i in keep])
            blocs = [it.blocs[i] for i in keep]
            row = {"election": e, "horizon": h, "n_units": len(keep),
                   "rms_move": _rms(dm, p), "rms_result": _rms(dr, p)}
            # blocs : parts sommées à la date, à l'agrégat final et au résultat
            ub = sorted(set(blocs))
            pb = np.array([p[[b == x for x in blocs]].sum() for b in ub])
            for name, d in (("move", dm), ("result", dr)):
                q = p * np.exp(d)
                q = q / q.sum()
                qb = np.array([q[[b == x for x in blocs]].sum() for b in ub])
                db = np.log(qb) - np.log(pb)
                row[f"bloc_rms_{name}"] = _rms(db, pb)
                cb = _centered(db, pb)
                for b, x, s in zip(ub, cb, pb):
                    units.append({"election": e, "horizon": h, "level": "bloc", "unit": b,
                                  "target": name, "abs_move": abs(float(x)), "share": float(s)})
                cc = _centered(d, p)
                for i, x in enumerate(cc):
                    units.append({"election": e, "horizon": h, "level": "candidate",
                                  "unit": it.units[keep[i]], "target": name,
                                  "abs_move": abs(float(x)), "share": float(p[i])})
            cases.append(row)
    return pl.DataFrame(cases), pl.DataFrame(units)


def backtest_z(ref: Path) -> pl.DataFrame:
    """z-scores du backtest floor (modèle + η) par cas, candidats et blocs, échelle logit."""
    f = pl.read_parquet(ref / "forecasts.parquet").filter(
        (pl.col("variant") == "floor") & (pl.col("model") == "model")
    )

    def lg(x):
        x = np.clip(np.asarray(x, dtype=float) / 100, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))

    rows = []
    for (e, h, lvl), g in f.group_by("election", "horizon", "level"):
        sd = (lg(g["q95"]) - lg(g["q05"])) / (2 * 1.6449)
        z = (lg(g["official"]) - lg(g["median"])) / sd
        w = g["official"].to_numpy()
        rows.append({"election": e, "horizon": h, "level": lvl, "rms_z": float(np.sqrt(np.mean(z**2))),
                     "rms_z_w": float(np.sqrt((w * z**2).sum() / w.sum())),
                     "cov50": float(g["in50"].mean()), "cov90": float(g["in90"].mean()),
                     "n": g.height})
    return pl.DataFrame(rows).sort("level", "election", "horizon")


# --- relations et placebos


def _center(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = x.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = x[m] - x[m].mean()
    return out


def relation(df: pl.DataFrame, x: str, y: str, n_date: int = N_PLACEBO, seed: int = 0) -> dict:
    """Spearman et pente (log y sur log x, centrés par horizon), placebos par permutation des
    élections (exacte, profils entiers d'une élection à l'autre) et des dates (V d'un autre
    horizon de la même élection)."""
    d = df.filter(pl.col(x).is_finite() & (pl.col(x) > 0) & pl.col(y).is_finite() & (pl.col(y) > 0))
    hs = d["horizon"].to_numpy()
    es = d["election"].to_numpy()
    lx = np.log(d[x].to_numpy())
    ly = np.log(d[y].to_numpy())
    cy = _center(ly, hs)

    def stat(lx_):
        cx = _center(lx_, hs)
        return float(spearmanr(cx, cy).statistic), float(np.polyfit(cx, cy, 1)[0])

    rho, slope = stat(lx)
    key = {(e, h): v for e, h, v in zip(es, hs, lx)}
    els = sorted(set(es.tolist()))
    perm_rhos = []
    for perm in itertools.permutations(els):
        if list(perm) == els:
            continue
        m = dict(zip(els, perm))
        lx_p = np.array([key.get((m[e], h), np.nan) for e, h in zip(es, hs)])
        ok = np.isfinite(lx_p)
        if ok.sum() < 4:
            continue
        cx = _center(lx_p[ok], hs[ok])
        perm_rhos.append(float(spearmanr(cx, _center(ly[ok], hs[ok])).statistic))
    rng = np.random.default_rng(seed)
    by_e = {e: [(h, v) for (ee, h), v in key.items() if ee == e] for e in els}
    date_rhos = []
    for _ in range(n_date):
        lx_p = np.array([
            rng.choice([v for hh, v in by_e[e] if hh != h] or [v0])
            for e, h, v0 in zip(es, hs, lx)
        ])
        date_rhos.append(stat(lx_p)[0])
    perm_rhos = np.array(perm_rhos)
    return {
        "n": len(d),
        "n_elections": len(els),
        "spearman": rho,
        "slope": slope,
        "perm_p": float((np.sum(perm_rhos >= rho) + 1) / (len(perm_rhos) + 1)),
        "perm_q95": float(np.quantile(perm_rhos, 0.95)),
        "perm_q90": float(np.quantile(perm_rhos, 0.90)),
        "date_p": float((np.sum(np.array(date_rhos) >= rho) + 1) / (n_date + 1)),
    }


def unit_relation(vols: pl.DataFrame, moves: pl.DataFrame, level: str, target: str,
                  seed: int = 0) -> dict:
    """Unités (candidats ou blocs) : log |mouvement centré| sur log V de l'unité, contrôle log
    part et effets d'horizon ; placebo : V permutée entre unités d'un même horizon."""
    d = moves.filter((pl.col("level") == level) & (pl.col("target") == target)).join(
        vols.filter(pl.col("level") == level).select("election", "horizon", "unit", "V"),
        on=["election", "horizon", "unit"], how="inner",
    ).filter((pl.col("V") > 0) & (pl.col("abs_move") > 0))
    if d.height < 10:
        return {"n": d.height}
    hs = d["horizon"].to_numpy()
    y = _center(np.log(d["abs_move"].to_numpy()), hs)
    lp = _center(np.log(d["share"].to_numpy()), hs)
    lv = np.log(d["V"].to_numpy())

    def beta(lv_):
        X = np.column_stack([_center(lv_, hs), lp])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][0])

    b = beta(lv)
    rng = np.random.default_rng(seed)
    pl_b = []
    for _ in range(N_PLACEBO):
        lv_p = lv.copy()
        for h in np.unique(hs):
            m = np.flatnonzero(hs == h)
            lv_p[m] = lv[rng.permutation(m)]
        pl_b.append(beta(lv_p))
    pl_b = np.array(pl_b)
    return {"n": d.height, "n_elections": d["election"].n_unique(), "beta": b,
            "placebo_p": float((np.sum(pl_b >= b) + 1) / (N_PLACEBO + 1))}


# --- 2. Jennings–Wlezien


def jw_cases(window: int = vo.PRIMARY_WINDOW, min_share: float = 2.0) -> pl.DataFrame:
    """Indice et mouvement ultérieur par élection × horizon, hors France, premier tour.

    Parts quotidiennes normalisées (poll_share / poll_total), effectif sample_size × n_polls ;
    niveau à la date : moyenne des jours [h, h + 14) ; final : jours [0, 14) ; résultat
    normalisé sur les partis listés. Partis de plus de `min_share` % au résultat."""
    from vvs.ingest.jennings_wlezien import load_polls as jw_load

    d = jw_load().filter((pl.col("round") == 1) & (pl.col("country_iso2") != "FR"))
    d = d.with_columns((pl.col("poll_share") / pl.col("poll_total")).alias("s")).filter(
        pl.col("s") > 0
    )
    rows = []
    for (eid,), g in d.group_by("election_id"):
        country = g["country_iso2"][0]
        etype = g["election_type"][0]
        res = (g.group_by("party_id").agg(pl.col("result_share").first(), pl.col("result_total").first()))
        res = {r["party_id"]: r["result_share"] / r["result_total"] for r in res.iter_rows(named=True)
               if r["result_share"] >= min_share}
        if len(res) < 3:
            continue
        g = g.filter(pl.col("party_id").is_in(list(res)))
        fin = g.filter(pl.col("days_before") < 14).group_by("party_id").agg(pl.col("s").mean())
        fin = dict(zip(fin["party_id"].to_list(), fin["s"].to_list()))
        for h in JW_HORIZONS:
            win = g.filter(pl.col("days_before").is_between(h, h + 7 * window - 1))
            weeks: dict = {}
            for pid, db, s, n, m in win.select("party_id", "days_before", "s", "sample_size",
                                               "n_polls").iter_rows():
                weeks.setdefault(pid, {}).setdefault((db - h) // 7, []).append(
                    (s, (n or vo.DEFAULT_N) * (m or 1)))
            vol, wts = {}, {}
            for pid, wk in weeks.items():
                v, _, _ = vo._unit_volatility(wk, window)
                if np.isfinite(v):
                    vol[pid] = v
                    wts[pid] = float(np.mean([s for vals in wk.values() for s, _ in vals]))
            if len(vol) < 2:
                continue
            V = sum(wts[p] * vol[p] for p in vol) / sum(wts[p] for p in vol)
            at = g.filter(pl.col("days_before").is_between(h, h + 13)).group_by("party_id").agg(
                pl.col("s").mean())
            at = dict(zip(at["party_id"].to_list(), at["s"].to_list()))
            common = [p for p in res if p in at and at[p] > 0]
            if len(common) < 3:
                continue
            a = np.array([at[p] for p in common])
            w = a / a.sum()
            r = np.array([res[p] for p in common])
            row = {"election_id": eid, "country": country, "type": etype, "horizon": h,
                   "V": V, "rms_result": _rms(np.log(r) - np.log(a), w)}
            cf = [p for p in common if fin.get(p, 0) > 0]
            if h >= 14 and len(cf) >= 3:
                af = np.array([at[p] for p in cf])
                row["rms_move"] = _rms(np.log([fin[p] for p in cf]) - np.log(af), af / af.sum())
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def jw_relation(df: pl.DataFrame, y: str, seed: int = 0, n_boot: int = 1000) -> pl.DataFrame:
    """Élasticité β de y à V par horizon et toutes horizons confondues (effets d'horizon),
    intervalle par bootstrap des pays, placebo par permutation de V dans l'horizon."""
    d = df.filter(pl.col(y).is_not_null() & (pl.col(y) > 0) & (pl.col("V") > 0))
    rng = np.random.default_rng(seed)
    out = []
    for h in [*JW_HORIZONS, "all"]:
        s = d if h == "all" else d.filter(pl.col("horizon") == h)
        if s.height < 10:
            continue
        hs = s["horizon"].to_numpy()
        lx = np.log(s["V"].to_numpy())
        ly = np.log(s[y].to_numpy())
        countries = s["country"].to_numpy()

        def slope(idx, lx=lx, ly=ly, hs=hs):
            cx, cy = _center(lx[idx], hs[idx]), _center(ly[idx], hs[idx])
            return float((cx * cy).sum() / (cx**2).sum())

        allidx = np.arange(len(lx))
        b = slope(allidx)
        uc = np.unique(countries)
        groups = {c: np.flatnonzero(countries == c) for c in uc}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uc, size=len(uc))
            boots.append(slope(np.concatenate([groups[c] for c in pick])))
        perm = []
        for _ in range(n_boot):
            lx_p = lx.copy()
            for hh in np.unique(hs):
                m = np.flatnonzero(hs == hh)
                lx_p[m] = lx[rng.permutation(m)]
            cx = _center(lx_p, hs)
            perm.append(float((cx * _center(ly, hs)).sum() / (cx**2).sum()))
        out.append({"horizon": str(h), "target": y, "n": s.height,
                    "n_countries": len(uc), "beta": b,
                    "lo95": float(np.quantile(boots, 0.025)), "hi95": float(np.quantile(boots, 0.975)),
                    "spearman": float(spearmanr(_center(lx, hs), _center(ly, hs)).statistic),
                    "perm_p": float((np.sum(np.array(perm) >= b) + 1) / (n_boot + 1))})
    return pl.DataFrame(out)


# --- 3. κ par pli


def vol_tables(reg) -> tuple[dict, dict]:
    """(V global, V par bloc) par (élection, horizon), fenêtre principale."""
    vols, bvols = {}, {}
    for e in ELECTIONS:
        for h in sorted(set(vo.REF_HORIZONS) | set(HORIZONS)):
            v = vo.volatility_at_horizon(e, h, vo.PRIMARY_WINDOW, reg)
            vols[(e, h)] = v.campaign
            bvols[(e, h)] = v.by_bloc
    return vols, bvols


def kappa_fits(out: Path, kappa_jw: float, reg) -> dict:
    """κ, ν par pli (élection laissée) et forme, mis en cache dans `out/kappa.json`."""
    path = out / "kappa.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    items = vo.calibration_items(ELECTIONS, reg)
    vols, _ = vol_tables(reg)
    for e in [*ELECTIONS, "all"]:
        key = str(e)
        train = [x for x in ELECTIONS if x != e]
        if key in cache and abs(cache[key].get("kappa_jw_value", math.nan) - kappa_jw) < 1e-9:
            continue
        gamma = er.jw_shape(exclude=tuple(er.JW_ELECTION_ID[x] for x in [e]
                                          if x in er.JW_ELECTION_ID))["gamma"]
        res = {"train": train, "gamma": gamma, "kappa_jw_value": kappa_jw}
        for form in vo.FORMS:
            res[form] = vo.fit_kappa(items, vols, train, form, gamma)
            print(e, form, res[form]["kappa"], res[form]["nu"], round(res[form]["gain_vs_kappa0"], 3),
                  flush=True)
        res["move_jw"] = vo.fit_kappa(items, vols, train, "move", gamma, grid=(kappa_jw,))
        cache[key] = res
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2))
    return cache


def fold_models(e, k: dict, reg) -> dict:
    """Modèles d'erreur d'un pli : floor (inchangé) et recalibrés par version."""
    items = vo.calibration_items(ELECTIONS, reg)
    vols, _ = vol_tables(reg)
    train = k["train"]
    ems = {"floor": er.ErrorModel.fit(elections=sc.CALIBRATION_ELECTIONS,
                                      exclude=[e] if e != "all" else (), registry=reg)}
    for name, key, form in (("vol_move", "move", "move"), ("vol_total", "total", "total"),
                            ("vol_move_jw", "move_jw", "move")):
        f = k[key]
        ems[name] = vo.fit_error_model(items, vols, train, f["kappa"], form, k["gamma"], f["nu"],
                                       reg)
    ems["vol_bloc"] = ems["vol_move"]
    return ems


def version_scale(name: str, e: int, h: int, k: dict, reg) -> vo.VolatilityScale | None:
    if name == "floor":
        return None
    vols, bvols = vol_tables(reg)
    key, form = {"vol_move": ("move", "move"), "vol_total": ("total", "total"),
                 "vol_bloc": ("move", "move"), "vol_move_jw": ("move_jw", "move")}[name]
    return vo.scale_for(e, h, vols, k["train"], k[key]["kappa"], form,
                        bloc_vols=bvols if name == "vol_bloc" else None, registry=reg)


# --- 4. backtest


def check_ref(ref: Path) -> list[str]:
    stale = []
    for e in ELECTIONS:
        for h in HORIZONS:
            m = json.loads((fit_dir(ref, "floor", e, h) / "manifest.json").read_text())
            if m["data"]["rows_hash"] != _hash_frame(load_polls(e, asof_for(e, h))):
                stale.append(f"{e}-{h}")
    return stale


def cmd_score(out: Path, ref: Path, kap: dict, reg, n_draws: int = 10_000, seed: int = 0):
    stale = check_ref(ref)
    if stale:
        raise SystemExit(f"ajustements de référence périmés (données changées) : {stale}")
    metric_rows, fc_rows, cal_rows, ems_floor, scales = [], [], [], {}, []
    for e in ELECTIONS:
        k = kap[str(e)]
        ems = fold_models(e, k, reg)
        ems_floor[e] = ems["floor"]
        off = sc.official(reg, e)
        for h in HORIZONS:
            asof = asof_for(e, h)
            polls = load_polls(e, asof)
            fd = fit_dir(ref, "floor", e, h)
            conf = json.loads((fd / "configuration.json").read_text())
            dr = np.load(fd / "draws.npz")
            omap = official_map(conf["candidates"], e, reg, substitution="lam" in dr.files)
            K = omap.config
            cands = [str(c) for c in dr["candidates"]]
            cblocs = [str(b) for b in dr["blocs"]]
            kblocs = [cblocs[cands.index(c)] for c in K]
            for name in VERSIONS:
                em = ems[name]
                vs = version_scale(name, e, h, k, reg)
                em.vol_scale = vs
                points = {}
                if name == "floor":
                    points["mean14"] = poll_mean(e, asof, K, days=14, registry=reg, polls=polls)
                    points["last_pollster"] = last_by_pollster(e, asof, K, registry=reg, polls=polls)
                rng = np.random.default_rng([seed, e, h])
                case = sc.score_case(
                    draws_log_shares(dr, K, "asof"), draws_log_shares(dr, K, "vote"), K, kblocs,
                    h, em, omap, off, points, rng, n_draws,
                )
                em.vol_scale = None
                m_rows, f_rows, c_rows = sc.case_rows(case, {"election": e, "horizon": h,
                                                              "variant": name})
                metric_rows += m_rows
                fc_rows += f_rows
                cal_rows += c_rows
                if vs is not None:
                    fac = vs.factor(kblocs)
                    scales.append({"election": e, "horizon": h, "version": name,
                                   "kappa": vs.kappa, "ratio": vs.ratio,
                                   "s_min": float(fac.min()), "s_max": float(fac.max()),
                                   "s_mean": float(fac.mean())})
            print(e, h, "fait", flush=True)
    res = sc.ScoreResult(out=out, metrics=pl.DataFrame(metric_rows),
                         forecasts=pl.DataFrame(fc_rows, infer_schema_length=None),
                         calibration=pl.DataFrame(cal_rows), error_models=ems_floor)
    sc.write_outputs(res, list(VERSIONS))
    pl.DataFrame(scales).write_parquet(out / "scales.parquet")
    return res


def _wmean(vals: dict[int, float]) -> float:
    w = np.array([ELECTION_WEIGHT.get(e, 1.0) for e in vals])
    return float((w * np.array(list(vals.values()))).sum() / w.sum())


def admission(metrics: pl.DataFrame) -> tuple[list[str], dict]:
    """Tableaux avant/après et règle de docs/07, plus le critère de structure (C4)."""
    s = metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model"))
    val = {(r["variant"], r["election"], r["horizon"], r["metric"]): r["value"]
           for r in s.iter_rows(named=True)}
    lines = []
    for metric in ("log_score_pair", "crps", "cov50", "cov90", "bloc_cov90", "brier_qual"):
        lines += [f"### {sc.METRIC_TITLE.get(metric, metric)}", "",
                  "| élection | horizon | " + " | ".join(VERSIONS) + " |",
                  "|---|---|" + "---|" * len(VERSIONS)]
        for e in ELECTIONS:
            for h in HORIZONS:
                cells = []
                for v in VERSIONS:
                    x = val.get((v, e, h, metric))
                    base = val.get(("floor", e, h, metric))
                    if x is None:
                        cells.append("–")
                    elif v == "floor":
                        cells.append(sc._fmt(metric, x))
                    else:
                        d = x - base
                        dd = f"{100 * d:+.0f}" if metric.startswith(("cov", "bloc_cov")) else f"{d:+.3f}"
                        cells.append(f"{sc._fmt(metric, x)} ({dd})")
                lines.append(f"| {e} | {h} | " + " | ".join(cells) + " |")
        for h in HORIZONS:
            cells = [sc._fmt(metric, _wmean({e: val[(v, e, h, metric)] for e in ELECTIONS}))
                     for v in VERSIONS]
            lines.append(f"| **moyenne pondérée** | {h} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Couverture par élection (moyenne des quatre horizons, candidats)", "",
              "| élection | " + " | ".join(f"{v} 50 / 90" for v in VERSIONS) + " |",
              "|---|" + "---|" * len(VERSIONS)]
    for e in ELECTIONS:
        cells = []
        for v in VERSIONS:
            c50 = np.mean([val[(v, e, h, "cov50")] for h in HORIZONS])
            c90 = np.mean([val[(v, e, h, "cov90")] for h in HORIZONS])
            cells.append(f"{100 * c50:.0f} / {100 * c90:.0f} %")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines.append("")
    verdicts = {}
    lines += ["### Règle d'admission (docs/07) et critère de structure", "",
              ("Chiffres = gains contre floor en moyenne pondérée (log-score : écart ; CRPS : "
              "baisse, positif = mieux), entre parenthèses le nombre d'élections de référence "
              "(2012, 2017, 2022) où le gain est positif."), ""]
    for v in VERSIONS[1:]:
        # règle de docs/07 (version du 2026-10-20) : gain aux horizons où la variable agit par
        # construction, aucun autre horizon dégradé. La volatilité agit à tous les horizons :
        # on teste les deux paires (180, 90) et (30, 7), l'autre paire ne devant pas se
        # dégrader (log-score ≥ −0,01, CRPS ≤ +0,01 en moyenne pondérée).
        parts, gains = [], {}
        for metric, sign in (("log_score_pair", 1), ("crps", -1)):
            for h in HORIZONS:
                d = {e: sign * (val[(v, e, h, metric)] - val[("floor", e, h, metric)])
                     for e in ELECTIONS}
                gains[(metric, h)] = (_wmean(d), sum(d[e] > 0 for e in REFERENCE))
                parts.append(f"{metric} {h} j : {gains[(metric, h)][0]:+.3f} "
                             f"({gains[(metric, h)][1]}/3)")
        ok_rule = False
        for act, other in (((180, 90), (30, 7)), ((30, 7), (180, 90))):
            gain = all(gains[(m, h)][0] > 0 and gains[(m, h)][1] >= 2
                       for m in ("log_score_pair", "crps") for h in act)
            no_harm = all(gains[(m, h)][0] >= -0.01 for m in ("log_score_pair", "crps")
                          for h in other)
            ok_rule |= gain and no_harm
        gap_before, gap_after = [], []
        for h in HORIZONS:
            for metric, nom in (("cov50", 0.5), ("cov90", 0.9)):
                gap_before.append(_wmean({e: abs(val[("floor", e, h, metric)] - nom)
                                          for e in ELECTIONS}))
                gap_after.append(_wmean({e: abs(val[(v, e, h, metric)] - nom) for e in ELECTIONS}))
        gb, ga = float(np.mean(gap_before)), float(np.mean(gap_after))
        ls = [_wmean({e: val[(v, e, h, "log_score_pair")] - val[("floor", e, h, "log_score_pair")]
                      for e in ELECTIONS}) for h in HORIZONS]
        cr = [_wmean({e: val[(v, e, h, "crps")] - val[("floor", e, h, "crps")]
                      for e in ELECTIONS}) for h in HORIZONS]
        # couverture « dégradée » = plus loin de la nominale (ici, une baisse de 100 % vers
        # 90 % est une amélioration) : écart moyen à la nominale non accru
        cov_ok = ga <= gb + 0.005
        struct = ga < gb and min(ls) >= -0.01 and max(cr) <= 0.01
        verdicts[v] = {"rule": bool(ok_rule and cov_ok), "structure": bool(struct),
                       "gap_before": gb, "gap_after": ga, "logscore_by_horizon": ls}
        lines.append(
            f"- `{v}` : " + " ; ".join(parts)
            + f" ; écart de couverture moyen à la nominale {100 * gb:.1f} → {100 * ga:.1f} pts"
            + f" ; règle docs/07 : **{'remplie' if ok_rule and cov_ok else 'non remplie'}**"
            + f" ; critère de structure (C4) : **{'rempli' if struct else 'non rempli'}**."
        )
    lines.append("")
    return lines, verdicts


# --- 2027


def current_2027(out: Path, kap: dict, reg) -> dict:
    """Indice de la campagne 2027 au jour de l'étude et effet sur les intervalles."""
    today = dt.date.today()
    vote = election_date(reg, 2027)
    h = (vote - today).days
    polls = load_polls(2027, today)
    conf = known_configuration(2027, today, reg, polls=polls).candidates
    blocs = [reg.bloc_of(c, 2027) for c in conf]
    res = {"asof": today.isoformat(), "horizon": h, "config": conf, "windows": {}}
    for w in vo.WINDOWS:
        v = vo.realized_volatility(polls, today, conf, blocs, w)
        res["windows"][w] = {"V": v.campaign, "V_raw": v.campaign_raw, "V_bloc": v.bloc_campaign,
                             "n_polls": v.n_polls, "n_weeks": v.n_weeks, "by_bloc": v.by_bloc}
    v = vo.realized_volatility(polls, today, conf, blocs, vo.PRIMARY_WINDOW)
    res["by_unit"] = v.by_unit
    vols, _bvols = vol_tables(reg)
    ref = vo.reference(vols, ELECTIONS)
    vref = vo.interp_reference(ref, h)
    res["V_ref"] = vref
    res["ratio"] = vo.ratio(v.campaign, vref)
    res["past_at_horizon"] = {e: vols.get((e, 180)) for e in ELECTIONS}
    # trajectoire de l'indice 2027 sur les derniers mois
    traj = []
    for back in range(0, 25, 2):
        d = today - dt.timedelta(weeks=back)
        try:
            c = known_configuration(2027, d, reg, polls=polls).candidates
        except ValueError:
            continue
        x = vo.realized_volatility(polls, d, c, [reg.bloc_of(y, 2027) for y in c])
        traj.append({"asof": d.isoformat(), "horizon": (vote - d).days, "V": x.campaign})
    res["trajectory"] = traj
    # intervalles : parts de la moyenne des sondages des 30 derniers jours
    pm = poll_mean(2027, today, conf, days=30, registry=reg, polls=polls)
    k = kap["all"]
    ems = fold_models("all", k, reg)
    widths = {}
    for name in ("floor", "vol_move", "vol_total", "vol_move_jw"):
        em = ems[name]
        if name != "floor":
            key = {"vol_move": "move", "vol_total": "total", "vol_move_jw": "move_jw"}[name]
            em.vol_scale = vo.VolatilityScale(kappa=k[key]["kappa"], ratio=res["ratio"],
                                              form="total" if key == "total" else "move")
        sims = em.sample(np.log(pm.shares)[None, :], h, conf, blocs,
                         np.random.default_rng(7), 40_000)
        q = np.quantile(100 * sims, [0.05, 0.95], axis=0)
        widths[name] = {c: float(q[1, i] - q[0, i]) for i, c in enumerate(conf)}
        em.vol_scale = None
    res["shares"] = {c: float(100 * s) for c, s in zip(conf, pm.shares)}
    res["width90"] = widths
    res["kappa"] = {f: k[f]["kappa"] for f in ("move", "total", "move_jw")}
    (out / "current_2027.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                      default=str))
    return res


# --- figures


def fig_index(vol: pl.DataFrame, path: Path, cur: dict | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, w in zip(axes, vo.WINDOWS):
        for e in ELECTIONS:
            d = vol.filter((pl.col("election") == e) & (pl.col("window") == w)
                           & pl.col("horizon").is_in(list(FINE))).sort("horizon")
            ax.plot(d["horizon"], d["V"], marker="o", ms=3, color=COLORS[e], label=str(e), lw=1.3)
        if cur is not None and w == vo.PRIMARY_WINDOW:
            t = [x for x in cur["trajectory"] if x["horizon"] <= 360]
            ax.plot([x["horizon"] for x in t], [x["V"] for x in t], color=INK, ls="--", lw=1.2,
                    marker="s", ms=3, label="2027")
        ax.invert_xaxis()
        ax.set_title(f"fenêtre de {w} semaines", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("jours avant le vote", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("V (log-part / semaine)", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Volatilité réalisée des sondages (moyenne pondérée par les parts)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_relations(fr: pl.DataFrame, z: pl.DataFrame, jw: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax = axes[0]
    d = fr.filter(pl.col("V") > 0)
    for e in ELECTIONS:
        s = d.filter(pl.col("election") == e)
        ax.scatter(s["V"], s["rms_result"], color=COLORS[e], s=14, label=str(e))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("France : V → mouvement jusqu'au résultat", fontsize=9, loc="left", color=INK)
    ax.set_xlabel("V à la date", fontsize=8, color=MUTED)
    ax.set_ylabel("écart quadratique moyen (log-parts)", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    d = z.filter((pl.col("level") == "candidate") & (pl.col("V") > 0))
    for e in ELECTIONS:
        s = d.filter(pl.col("election") == e)
        ax.scatter(s["V_ratio"], s["rms_z"], color=COLORS[e], s=18)
        for r in s.iter_rows(named=True):
            ax.annotate(str(r["horizon"]), (r["V_ratio"], r["rms_z"]), fontsize=6, color=MUTED)
    ax.axhline(1, color=GRID, lw=1)
    ax.set_xscale("log")
    ax.set_title("Backtest floor : V / V_ref → z quadratique moyen", fontsize=9, loc="left",
                 color=INK)
    ax.set_xlabel("V / V_ref (même horizon)", fontsize=8, color=MUTED)
    ax = axes[2]
    d = jw.filter((pl.col("V") > 0) & (pl.col("rms_move").is_not_null()) & (pl.col("rms_move") > 0)
                  & (pl.col("horizon") >= 30))
    ax.scatter(d["V"], d["rms_move"], s=5, color="#2a78d6", alpha=0.35)
    # moyennes par déciles de V
    lv = np.log(d["V"].to_numpy())
    ly = np.log(d["rms_move"].to_numpy())
    qs = np.quantile(lv, np.linspace(0, 1, 11))
    mx, my = [], []
    for a, b in itertools.pairwise(qs):
        m = (lv >= a) & (lv <= b)
        mx.append(np.exp(lv[m].mean()))
        my.append(np.exp(ly[m].mean()))
    ax.plot(mx, my, color=INK, marker="o", ms=4, lw=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Jennings–Wlezien hors France (h ≥ 30 j)", fontsize=9, loc="left", color=INK)
    ax.set_xlabel("V à la date", fontsize=8, color=MUTED)
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_coverage(metrics: pl.DataFrame, path: Path) -> None:
    s = metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, metric, nom in ((axes[0], "cov50", 0.5), (axes[1], "cov90", 0.9)):
        x = np.arange(len(ELECTIONS))
        wbar = 0.8 / len(VERSIONS)
        for i, v in enumerate(VERSIONS):
            y = []
            for e in ELECTIONS:
                d = s.filter((pl.col("variant") == v) & (pl.col("election") == e)
                             & (pl.col("metric") == metric))
                y.append(float(d["value"].mean()))
            ax.bar(x + i * wbar - 0.4 + wbar / 2, y, wbar, label=VERSION_LABEL[v],
                   color=["#8a8984", "#2a78d6", "#eb6834", "#3a9b5c", "#b03a8f"][i])
        ax.axhline(nom, color=INK, lw=0.8, ls=":")
        ax.set_xticks(x, [str(e) for e in ELECTIONS])
        ax.set_title(f"couverture à {int(100 * nom)} % (moyenne des 4 horizons)", fontsize=9,
                     loc="left", color=INK)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- rapport


def _md_table(df: pl.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows(named=True):
        cells = []
        for c in cols:
            x = r[c]
            if isinstance(x, float):
                cells.append("–" if not math.isfinite(x) else fmt.get(c, "{:.3f}").format(x))
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_study(out: Path, ref: Path) -> dict:
    reg = default_registry()
    out.mkdir(parents=True, exist_ok=True)
    vol, vunits = french_volatility(reg)
    vol.write_parquet(out / "volatility_france.parquet")
    vunits.write_parquet(out / "volatility_units.parquet")
    moves, munits = french_moves(reg)
    moves.write_parquet(out / "moves_france.parquet")
    munits.write_parquet(out / "moves_units.parquet")
    z = backtest_z(ref)
    jw = jw_cases()
    jw.write_parquet(out / "jw_cases.parquet")
    res = {"france": {}, "units": {}, "z": {}, "jw": {}}
    for w in vo.WINDOWS:
        fr = moves.join(vol.filter(pl.col("window") == w), on=["election", "horizon"], how="left")
        for x in ("V", "V_raw", "V_points", "V_bloc"):
            for y in ("rms_result", "rms_move", "bloc_rms_result", "bloc_rms_move"):
                s = fr if y.endswith("result") else fr.filter(pl.col("horizon") >= 14)
                res["france"][f"{w}|{x}|{y}"] = relation(s, x, y)
    for level in ("candidate", "bloc"):
        for target in ("result", "move"):
            res["units"][f"{level}|{target}"] = unit_relation(vunits, munits, level, target)
    vw = vol.filter(pl.col("window") == vo.PRIMARY_WINDOW)
    zz = z.join(vw, on=["election", "horizon"], how="left")
    ref_h = {h: float(np.exp(np.nanmean(np.log(
        vw.filter((pl.col("horizon") == h) & (pl.col("V") > 0))["V"].to_numpy())))) for h in HORIZONS}
    zz = zz.with_columns((pl.col("V") / pl.col("horizon").replace_strict(ref_h)).alias("V_ratio"))
    zz.write_parquet(out / "backtest_z.parquet")
    for level in ("candidate", "bloc"):
        for y in ("rms_z", "rms_z_w"):
            for x in ("V", "V_raw", "V_bloc"):
                res["z"][f"{level}|{x}|{y}"] = relation(zz.filter(pl.col("level") == level), x, y)
    for y in ("rms_move", "rms_result"):
        res["jw"][y] = jw_relation(jw, y).to_dicts()
    (out / "relations.json").write_text(json.dumps(res, indent=2))
    fig_relations(moves.join(vw, on=["election", "horizon"], how="left"), zz, jw,
                  out / "relations.png")
    return res


def kappa_jw_from(res: dict) -> float:
    """Élasticité JW retenue : mouvement vers les derniers sondages, horizons ≥ 30 j, mise en
    commun (β estimé sur les seuls horizons 30 à 180)."""
    rows = [r for r in res["jw"]["rms_move"] if r["horizon"] not in ("all", "7", "14")]
    w = np.array([r["n"] for r in rows], dtype=float)
    return float((w * np.array([r["beta"] for r in rows])).sum() / w.sum())


def write_report(out: Path, res: dict, kap: dict, metrics: pl.DataFrame, cur: dict,
                 kappa_jw: float) -> None:
    vol = pl.read_parquet(out / "volatility_france.parquet")
    moves = pl.read_parquet(out / "moves_france.parquet")
    zz = pl.read_parquet(out / "backtest_z.parquet")
    lines = [
        "# Volatilité réalisée et échelle de l'erreur de projection",
        "",
        (f"Généré le {dt.date.today().isoformat()} par `back/studies/volatility.py`. Indice et "
        "forme : `back/src/vvs/model/volatility.py`. Critères fixés avant lecture : docstring "
        "de l'étude."),
        "",
        "## 1. Indice de volatilité (fenêtre de 8 semaines, log-parts par semaine)",
        "",
    ]
    piv = (vol.filter((pl.col("window") == 8) & pl.col("horizon").is_in(list(FINE)))
           .pivot(on="election", index="horizon", values="V").sort("horizon", descending=True))
    lines += [_md_table(piv), "",
              "Indice brut (variations consécutives, sans correction du bruit), même fenêtre :", ""]
    piv = (vol.filter((pl.col("window") == 8) & pl.col("horizon").is_in(list(FINE)))
           .pivot(on="election", index="horizon", values="V_raw").sort("horizon", descending=True))
    lines += [_md_table(piv), "", ("Figure : `index.png`. Indéfini (–) : moins de trois semaines "
              "avec sondages dans la fenêtre (2012 à 180 jours, tout à 240 jours sauf 2022)."), ""]
    lines += ["## 2. Relation avec le mouvement ultérieur (France, grille fine)", "",
              ("Spearman sur les valeurs centrées par horizon (log V, log écart quadratique moyen "
              "pondéré par les parts) ; pente = élasticité ; p de permutation exacte des "
              "élections (119 permutations) et des dates (V d'un autre horizon de la même "
              "élection, 2 000 tirages). Mouvement « move » : de l'agrégat à la date à l'agrégat "
              "final (horizons ≥ 14 j) ; « result » : jusqu'au résultat."), "",
              "| fenêtre | indice | cible | n | ρ | pente | p élections | 95e centile | p dates |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["france"].items():
        w, x, y = key.split("|")
        if x not in ("V", "V_raw") and not (x == "V_bloc" and y.startswith("bloc")):
            continue
        lines.append(f"| {w} | {x} | {y} | {r['n']} | {r['spearman']:+.2f} | {r['slope']:+.2f} | "
                     f"{r['perm_p']:.2f} | {r['perm_q95']:+.2f} | {r['date_p']:.2f} |")
    lines += ["", ("Par unité (log |mouvement centré| sur log V de l'unité, contrôle log part, "
              "effets d'horizon ; placebo : V permutée entre unités du même horizon) :"), "",
              "| niveau | cible | n | élections | β | p placebo |", "|---|---|---|---|---|---|"]
    for key, r in res["units"].items():
        lv, t = key.split("|")
        if "beta" in r:
            lines.append(f"| {lv} | {t} | {r['n']} | {r['n_elections']} | {r['beta']:+.2f} | "
                         f"{r['placebo_p']:.3f} |")
    lines += ["", "Mouvement jusqu'au résultat, écart quadratique moyen (log-parts) :", ""]
    piv = moves.pivot(on="election", index="horizon", values="rms_result").sort(
        "horizon", descending=True)
    lines += [_md_table(piv), ""]
    lines += ["## 3. Jennings–Wlezien hors France (forme a priori)", "",
              ("Élasticité β (log écart quadratique moyen sur log V, effets d'horizon), intervalle "
              "à 95 % par bootstrap des pays, p de permutation de V dans l'horizon."), "",
              "| cible | horizon | n | pays | β | IC 95 % | Spearman | p perm. |",
              "|---|---|---|---|---|---|---|---|"]
    for y, rows in res["jw"].items():
        for r in rows:
            lines.append(f"| {y} | {r['horizon']} | {r['n']} | {r['n_countries']} | "
                         f"{r['beta']:+.2f} | [{r['lo95']:+.2f}, {r['hi95']:+.2f}] | "
                         f"{r['spearman']:+.2f} | {r['perm_p']:.3f} |")
    lines += ["", (f"β retenu comme forme a priori (mouvement vers les derniers sondages, "
              f"horizons 30 à 180, moyenne pondérée par n) : **{kappa_jw:+.2f}**. Avec du bruit "
              "de mesure sur V, β est atténué vers 0."), ""]
    lines += ["## 4. z-scores du backtest de référence", "",
              ("z = (logit résultat − logit médiane) / ((logit q95 − logit q05) / 3,29), écart "
              "quadratique moyen par cas (1 attendu si l'échelle est juste ; la loi de Student "
              "rend la queue un peu plus épaisse)."), ""]
    piv = zz.filter(pl.col("level") == "candidate").select(
        "election", "horizon", "rms_z", "cov50", "cov90", "V", "V_ratio").sort(
        "election", pl.col("horizon"), descending=[False, True])
    lines += [_md_table(piv), "", ("| niveau | indice | z | n | ρ | pente | p élections | "
              "90e centile |"), "|---|---|---|---|---|---|---|---|"]
    for key, r in res["z"].items():
        lv, x, y = key.split("|")
        lines.append(f"| {lv} | {x} | {y} | {r['n']} | {r['spearman']:+.2f} | {r['slope']:+.2f} | "
                     f"{r['perm_p']:.2f} | {r['perm_q90']:+.2f} |")
    lines += ["", ("Pente = élasticité du z à V/V_ref : c'est, en échantillon, le κ dont η aurait "
              "eu besoin. Figure : `relations.png`."), ""]
    lines += ["## 5. κ par pli (validation croisée interne sur les données de calibration de η)",
              "", ("| pli (élection laissée) | γ | κ move | ν move | gain move | κ total | ν total | "
              "gain total | ν (κ JW) |"), "|---|---|---|---|---|---|---|---|---|"]
    for key, k in kap.items():
        lines.append(
            f"| {key} | {k['gamma']:.2f} | {k['move']['kappa']:+.2f} | {k['move']['nu']} | "
            f"{k['move']['gain_vs_kappa0']:+.2f} | {k['total']['kappa']:+.2f} | "
            f"{k['total']['nu']} | {k['total']['gain_vs_kappa0']:+.2f} | {k['move_jw']['nu']} |")
    lines += ["", ("Gain = log-vraisemblance en validation croisée interne contre κ = 0 (même ν "
              "optimisé). Profils complets : `kappa.json`."), ""]
    lines += ["## 6. Backtest avant/après (tirages floor, modèle + η)", "",
              "Versions : " + " ; ".join(f"`{v}` {VERSION_LABEL[v]}" for v in VERSIONS) + ".",
              "Facteurs d'échelle par cas : `scales.parquet`. Figure : `coverage.png`.", ""]
    sc_df = pl.read_parquet(out / "scales.parquet")
    piv = sc_df.filter(pl.col("version") == "vol_move").select(
        "election", "horizon", "ratio", "kappa", "s_mean").sort("election", "horizon")
    lines += ["Rapport V/V_ref et facteur moyen s (version `vol_move`) :", "", _md_table(piv), ""]
    adm, verdicts = admission(metrics)
    lines += adm
    lines += ["## 7. Campagne 2027", "",
              (f"Au {cur['asof']} (τ = {cur['horizon']} jours), configuration "
              f"{', '.join(cur['config'])}."), "",
              "| fenêtre | V | V brut | V blocs | sondages | semaines |", "|---|---|---|---|---|---|"]
    for w, r in cur["windows"].items():
        lines.append(f"| {w} | {r['V']:.3f} | {r['V_raw']:.3f} | {r['V_bloc']:.3f} | "
                     f"{r['n_polls']} | {r['n_weeks']} |")
    lines += ["", (f"V_ref à {cur['horizon']} jours (2002-2022) : {cur['V_ref']:.3f} ; rapport "
              f"V / V_ref = **{cur['ratio']:.2f}**. κ estimé sur les cinq élections : move "
              f"{cur['kappa']['move']:+.2f}, total {cur['kappa']['total']:+.2f}."), "",
              ("Largeur de l'intervalle à 90 % (points) sur la moyenne des sondages des 30 "
              "derniers jours, η seul (sans l'incertitude d'agrégation) :"), "",
              "| candidat | part | floor | move | total | move, κ JW |", "|---|---|---|---|---|---|"]
    for c in sorted(cur["shares"], key=lambda c: -cur["shares"][c]):
        lines.append(f"| {c} | {cur['shares'][c]:.1f} | {cur['width90']['floor'][c]:.1f} | "
                     f"{cur['width90']['vol_move'][c]:.1f} | {cur['width90']['vol_total'][c]:.1f} | "
                     f"{cur['width90']['vol_move_jw'][c]:.1f} |")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "study", "kappa", "score", "current"], nargs="?",
                    default="all")
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-volatility")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    args = ap.parse_args()
    reg = default_registry()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rel_path = out / "relations.json"
    if args.cmd in ("all", "study") or not rel_path.exists():
        res = run_study(out, args.ref)
    else:
        res = json.loads(rel_path.read_text())
    if args.cmd == "study":
        return
    kappa_jw = kappa_jw_from(res)
    kap = kappa_fits(out, kappa_jw, reg)
    if args.cmd == "kappa":
        return
    mpath = out / ("metrics_" + "_".join(VERSIONS) + ".parquet")
    if args.cmd in ("all", "score") or not mpath.exists():
        cmd_score(out, args.ref, kap, reg)
    metrics = pl.read_parquet(mpath)
    fig_coverage(metrics, out / "coverage.png")
    cur = current_2027(out, kap, reg)
    vol = pl.read_parquet(out / "volatility_france.parquet")
    fig_index(vol, out / "index.png", cur)
    write_report(out, res, kap, metrics, cur, kappa_jw)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
