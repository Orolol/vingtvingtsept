"""Étude : le troupeau des instituts en fin de campagne, et son usage comme signal sur l'échelle
d'erreur de l'élection.

    uv run python studies/herding.py [all|study|score|current|report] [--out runs/<date>-herding]

Contexte (docs/backtest-log.md) : la largeur des intervalles par élection relève de l'échelle λ
commune à l'élection, que la volatilité observée avant la date n'anticipe pas ; les z-scores du
backtest sont structurés par élection (2002 et 2022 à 1,4-1,7, 2007, 2012 et 2017 à 0,5-0,9).
Le troupeau propose un autre signal : si les instituts se ressemblent plus que ne le permet le
tirage aléatoire, leurs erreurs sont corrélées et l'agrégat est moins fiable qu'il n'en a l'air.

Mesure (`vvs.model.herding`) : dispersion relative D = variance des parts entre instituts (un
sondage par institut, le dernier de la fenêtre, hypothèse la plus proche de la configuration,
tendance linéaire retirée) / variance attendue sous tirage aléatoire aux effectifs déclarés,
arrondi des parts publiées compris ; par candidat (R_u), par bloc et agrégée (moyenne des R_u
des candidats d'au moins 2 %). D = 1 sous tirage aléatoire ; effets de plan, effets maison et
sous-échantillons d'exprimés la portent au-dessus ; un troupeau la tire vers le bas.

Élections : présidentielles 2002-2022 et scrutins de liste (européennes 2014, 2019, 2024,
législatives 2022, 2024), soit dix. Fenêtres : 14 et 30 derniers jours (veille du vote), puis
30 jours avant vote − h pour h = 7, 30, 60, 90, 120, 180.

Critères, fixés avant de lire les relations (seuls les indices D ont été regardés, pour régler
la sélection des sondages : une unité commune est désormais jugée sur les derniers sondages des
instituts, l'institut gardant son dernier sondage qui la teste) :

- (a) troupeau : Δ_e = log D(14 derniers jours) − log D(30 jours avant vote − 30). La France
  « a un troupeau » si Δ_e < 0 dans au moins 75 % des élections où il est défini et si la
  moyenne des Δ_e est négative au-delà du hasard (test exact par inversion des signes, p < 0,05).
  Par élection, troupeau marqué si Δ_e < −1,645 · se (jackknife sur les instituts), et
  dispersion « sous le hasard » si la borne haute à 90 % de D(14 j) est sous 1. Lecture
  secondaire : D_bloc, et D sans retrait de tendance.
- (b) erreur finale : E_e = écart quadratique moyen des log-erreurs centrées (résultat contre
  moyenne des 14 derniers jours, `poll_errors`, unités d'au moins 1 point, pondéré par les
  parts). Relation tenue si le Spearman entre log D(14 j) et log E_e sur les dix élections est
  négatif avec p < 0,10 (20 000 permutations des élections, unilatéral). Secondaires : D_bloc et
  E_bloc ; dans l'élection, blocs (log R_b centré contre log |erreur centrée du bloc|, Spearman
  regroupé, permutation des R_b dans l'élection, p < 0,10) ; z quadratique moyen du backtest de
  référence (`lists`, moyenne des quatre horizons) sur les cinq présidentielles, même critère
  (n = 5 : p < 0,10 exige ρ ≤ −0,9).
- (c) prévision : sur les 20 cas présidentiels (5 élections × 180, 90, 30, 7 jours), Spearman
  centré par horizon entre log D(h) (30 jours avant la date) et log z quadratique moyen du
  backtest `lists` ; tenue si négatif au-delà du 10e centile des 119 permutations exactes des
  élections (p ≤ 0,10). Meilleure que la volatilité réalisée si, en plus, la même statistique
  calculée avec log V (indice de `vvs.model.volatility`, 8 semaines, signe attendu positif) a un
  p plus grand.
- Forme et backtest : faits dans tous les cas, pour documenter κ (pas de nouvel ajustement NUTS,
  tirages floor de runs/2026-10-20-backtest, même graine) ; l'admission n'est envisagée que si
  (b) ou (c) tient. Règle de docs/07 contre `lists` (tous horizons d'action), plus le critère de
  structure de studies/volatility.py : écart moyen de couverture à la nominale réduit en moyenne
  pondérée sans perte de log-score (≥ −0,01 à chaque horizon) ni de CRPS (≤ +0,01).

n honnête : dix élections pour (b), cinq présidentielles pour le backtest ; les effectifs sont
imputés (1 000) en 2002, 2007, 2012 et aux européennes 2014 : le niveau de D de ces scrutins en
dépend linéairement, pas sa trajectoire.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from vvs import paths
from vvs.eval import scoring as sc
from vvs.eval.backtest import ELECTION_WEIGHT, fit_dir
from vvs.model import herding as hd
from vvs.model.plots import INK, MUTED, _style
from vvs.registry import default_registry

PRES = (2002, 2007, 2012, 2017, 2022)
LISTS = ("euro2014", "euro2019", "euro2024", "leg2022", "leg2024")
ELECTIONS = PRES + LISTS
REFERENCE = (2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
# fenêtres : (nom, horizon de fin, jours)
WINDOWS = (("final14", 1, 14), ("final30", 1, 30), ("h7", 7, 30), ("h30", 30, 30),
           ("h60", 60, 30), ("h90", 90, 30), ("h120", 120, 30), ("h180", 180, 30))
REF_BACKTEST = paths.RUNS / "2026-10-20-backtest-lists"
FITS_FROM = paths.RUNS / "2026-10-20-backtest"
VOL_STUDY = paths.RUNS / "2026-10-20-volatility" / "volatility_france.parquet"
VARIANTS = ("lists", "lists_herding", "lists_herding_final", "floor", "herding")
PAIRS = (("lists_herding", "lists"), ("lists_herding_final", "lists"), ("herding", "floor"))
N_PERM = 20_000
COLORS = {2002: "#8a8984", 2007: "#3a9b5c", 2012: "#eb6834", 2017: "#2a78d6", 2022: "#b03a8f",
          "euro2014": "#c7a33a", "euro2019": "#5aa6a6", "euro2024": "#7a5cc2",
          "leg2022": "#d4645c", "leg2024": "#4a4a8a"}


def ekey(e) -> str:
    return str(e)


# --- 1. dispersion


def dispersion_tables(reg) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(D par élection × fenêtre × retrait de tendance, rapports par unité)."""
    rows, units = [], []
    for e in ELECTIONS:
        for name, h, days in WINDOWS:
            for detrend in (True, False):
                d = hd.herding_at(e, h, days=days, detrend=detrend, registry=reg)
                rows.append({"election": ekey(e), "window": name, "horizon": h, "days": days,
                             "detrend": detrend, "D": d.D, "D_bloc": d.D_bloc, "m": d.m,
                             "se_log_D": d.se_log_D, "n_imputed": d.n_imputed,
                             "n_mean": d.n_mean})
                if not detrend or name not in ("final14", "h30"):
                    continue
                for u, r in d.by_unit.items():
                    units.append({"election": ekey(e), "window": name, "level": "candidate",
                                  "unit": u, "R": r, "share": d.shares[u],
                                  "sd": d.sd_points[u], "sd_expected": d.sd_expected[u]})
                for b, r in d.by_bloc.items():
                    k = f"bloc:{b}"
                    units.append({"election": ekey(e), "window": name, "level": "bloc",
                                  "unit": b, "R": r, "share": d.shares[k], "sd": d.sd_points[k],
                                  "sd_expected": d.sd_expected[k]})
    return pl.DataFrame(rows), pl.DataFrame(units)


def disp(tab: pl.DataFrame, e, window: str, col: str = "D", detrend: bool = True) -> float:
    r = tab.filter((pl.col("election") == ekey(e)) & (pl.col("window") == window)
                   & (pl.col("detrend") == detrend))
    return float(r[col][0]) if r.height else math.nan


# --- (a) troupeau


def herding_trend(tab: pl.DataFrame) -> tuple[list[dict], dict]:
    rows = []
    for e in ELECTIONS:
        r = {"election": ekey(e)}
        for col, det, tag, wf in (("D", True, "", "final14"), ("D_bloc", True, "_bloc", "final14"),
                                  ("D", False, "_raw", "final14"), ("D", True, "_eq", "final30")):
            d14, d30 = disp(tab, e, wf, col, det), disp(tab, e, "h30", col, det)
            r[f"delta{tag}"] = (math.log(d14) - math.log(d30)
                                if d14 > 0 and d30 > 0 else math.nan)
        se14 = disp(tab, e, "final14", "se_log_D")
        se30 = disp(tab, e, "h30", "se_log_D")
        r["se_delta"] = math.sqrt(se14**2 + se30**2) if np.isfinite(se14 + se30) else math.nan
        r["marked"] = bool(np.isfinite(r["delta"]) and np.isfinite(r["se_delta"])
                           and r["delta"] < -1.645 * r["se_delta"])
        d14 = disp(tab, e, "final14")
        r["D14"], r["D14_hi90"] = d14, d14 * math.exp(1.645 * se14) if np.isfinite(se14) else math.nan
        r["below_random"] = bool(np.isfinite(r["D14_hi90"]) and r["D14_hi90"] < 1)
        rows.append(r)
    out = {}
    for tag in ("", "_bloc", "_raw", "_eq"):
        x = np.array([r[f"delta{tag}"] for r in rows])
        x = x[np.isfinite(x)]
        k = len(x)
        flips = np.array(list(itertools.product((1, -1), repeat=k)))
        means = (flips * np.abs(x)).mean(1)
        out[tag or "cand"] = {
            "n": k, "n_neg": int((x < 0).sum()), "mean": float(x.mean()),
            "p_signflip": float((means <= x.mean() + 1e-12).mean()),
            "holds": bool((x < 0).mean() >= 0.75
                          and (means <= x.mean() + 1e-12).mean() < 0.05),
        }
    return rows, out


# --- erreurs finales et z


def _rms_centered(d: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    c = d - (w * d).sum()
    return float(np.sqrt((w * c**2).sum()))


def final_errors() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(E par élection, candidats et blocs ; erreurs centrées par unité)."""
    pe = pl.read_parquet(paths.CURATED / "poll_errors.parquet").with_columns(
        pl.col("election").cast(pl.Utf8))
    pel = pl.read_parquet(paths.CURATED / "poll_errors_lists.parquet")
    cols = ["election", "round", "method", "level", "unit", "poll_share", "result_share"]
    allp = pl.concat([pe.select(cols), pel.select(cols)]).filter(
        (pl.col("round") == 1) & (pl.col("method") == "mean_14d"))
    rows, units = [], []
    for e in ELECTIONS:
        r = {"election": ekey(e)}
        for lvl in ("candidate", "bloc"):
            g = allp.filter((pl.col("election") == ekey(e)) & (pl.col("level") == lvl)
                            & (pl.col("poll_share") >= 1.0) & (pl.col("result_share") > 0))
            p = g["poll_share"].to_numpy()
            y = g["result_share"].to_numpy()
            p, y = p / p.sum(), y / y.sum()
            d = np.log(y) - np.log(p)
            r["E" if lvl == "candidate" else "E_bloc"] = _rms_centered(d, p)
            c = d - (p / p.sum() * d).sum()
            for u, x, s in zip(g["unit"].to_list(), c, p):
                units.append({"election": ekey(e), "level": lvl, "unit": u, "err": float(x),
                              "share": float(s)})
        rows.append(r)
    return pl.DataFrame(rows), pl.DataFrame(units)


def backtest_z(ref: Path = REF_BACKTEST, variant: str = "lists") -> pl.DataFrame:
    """z quadratique moyen du backtest de référence par cas (candidats), échelle logit."""
    f = pl.read_parquet(ref / "forecasts_floor_lists_fr2017_lists_fr2017.parquet").filter(
        (pl.col("variant") == variant) & (pl.col("model") == "model")
        & (pl.col("level") == "candidate"))

    def lg(x):
        x = np.clip(np.asarray(x, dtype=float) / 100, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))

    rows = []
    for (e, h), g in f.group_by("election", "horizon"):
        sd = (lg(g["q95"]) - lg(g["q05"])) / (2 * 1.6449)
        z = (lg(g["official"]) - lg(g["median"])) / sd
        rows.append({"election": int(e), "horizon": int(h),
                     "rms_z": float(np.sqrt(np.mean(z**2)))})
    return pl.DataFrame(rows).sort("election", "horizon")


# --- relations


def spearman_perm(x: np.ndarray, y: np.ndarray, n: int = N_PERM, seed: int = 0) -> dict:
    """Spearman et p unilatéral (corrélation négative) par permutation de x."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    perm = np.array([spearmanr(rng.permutation(x), y).statistic for _ in range(n)])
    return {"n": int(ok.sum()), "rho": rho, "p_neg": float((np.sum(perm <= rho) + 1) / (n + 1)),
            "p_pos": float((np.sum(perm >= rho) + 1) / (n + 1))}


def within_relation(R: pl.DataFrame, err: pl.DataFrame, level: str, n: int = N_PERM,
                    seed: int = 0) -> dict:
    """Unités d'une même élection : log R centré contre log |erreur centrée| centré, Spearman
    regroupé ; placebo : R permuté entre unités de l'élection."""
    d = R.filter((pl.col("window") == "final14") & (pl.col("level") == level)).join(
        err.filter(pl.col("level") == level).select("election", "unit", "err"),
        on=["election", "unit"], how="inner").filter(
        (pl.col("R") > 0) & (pl.col("err").abs() > 0) & (pl.col("share") >= 0.02))
    if d.height < 6:
        return {"n": d.height}
    es = d["election"].to_numpy()
    lr = np.log(d["R"].to_numpy())
    le = np.log(np.abs(d["err"].to_numpy()))

    def center(v):
        out = v.copy()
        for g in np.unique(es):
            m = es == g
            out[m] = v[m] - v[m].mean()
        return out

    ce = center(le)
    rho = float(spearmanr(center(lr), ce).statistic)
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n):
        lp = lr.copy()
        for g in np.unique(es):
            idx = np.flatnonzero(es == g)
            lp[idx] = lr[rng.permutation(idx)]
        perm.append(spearmanr(center(lp), ce).statistic)
    perm = np.array(perm)
    return {"n": d.height, "n_elections": len(np.unique(es)), "rho": rho,
            "p_neg": float((np.sum(perm <= rho) + 1) / (n + 1))}


def case_relation(cases: pl.DataFrame, x: str, y: str = "rms_z") -> dict:
    """Spearman centré par horizon, p exacts par permutation des élections (profils entiers)."""
    d = cases.filter(pl.col(x).is_finite() & (pl.col(x) > 0))
    hs = d["horizon"].to_numpy()
    lx, ly = np.log(d[x].to_numpy()), np.log(d[y].to_numpy())

    def center(v, h):
        out = v.copy()
        for g in np.unique(h):
            m = h == g
            out[m] = v[m] - v[m].mean()
        return out

    rho = float(spearmanr(center(lx, hs), center(ly, hs)).statistic)
    full = cases.filter(pl.col(x).is_finite() & (pl.col(x) > 0))
    key = {(int(e), int(h)): math.log(v) for e, h, v in full.select("election", "horizon", x).iter_rows()}
    yk = {(int(e), int(h)): math.log(v) for e, h, v in cases.select("election", "horizon", y).iter_rows()}
    els = sorted(set(PRES))
    perm = []
    for p in itertools.permutations(els):
        if list(p) == els:
            continue
        m = dict(zip(els, p))
        pairs = [(key[(m[e], h)], yk[(e, h)], h) for (e, h) in yk if (m[e], h) in key]
        if len(pairs) < 4:
            continue
        a = np.array(pairs)
        perm.append(spearmanr(center(a[:, 0], a[:, 2]), center(a[:, 1], a[:, 2])).statistic)
    perm = np.array(perm)
    return {"n": d.height, "rho": rho, "p_neg": float((np.sum(perm <= rho) + 1) / (len(perm) + 1)),
            "p_pos": float((np.sum(perm >= rho) + 1) / (len(perm) + 1)),
            "q10": float(np.quantile(perm, 0.10)), "q90": float(np.quantile(perm, 0.90))}


def run_study(out: Path) -> dict:
    reg = default_registry()
    tab, units = dispersion_tables(reg)
    tab.write_parquet(out / "dispersion.parquet")
    units.write_parquet(out / "units.parquet")
    trend_rows, trend = herding_trend(tab)
    E, err_units = final_errors()
    E.write_parquet(out / "final_errors.parquet")
    z = backtest_z()
    z.write_parquet(out / "backtest_z.parquet")
    # (b)
    d14 = np.array([disp(tab, e, "final14") for e in ELECTIONS])
    d14b = np.array([disp(tab, e, "final14", "D_bloc") for e in ELECTIONS])
    d30 = np.array([disp(tab, e, "final30") for e in ELECTIONS])
    Ev = np.array([E.filter(pl.col("election") == ekey(e))["E"][0] for e in ELECTIONS])
    Eb = np.array([E.filter(pl.col("election") == ekey(e))["E_bloc"][0] for e in ELECTIONS])
    zbar = {e: float(z.filter(pl.col("election") == e)["rms_z"].mean()) for e in PRES}
    b = {
        "D14_E": spearman_perm(np.log(d14), np.log(Ev)),
        "D14bloc_Ebloc": spearman_perm(np.log(d14b), np.log(Eb)),
        "D30_E": spearman_perm(np.log(d30), np.log(Ev)),
        "D14_E_pres": spearman_perm(np.log(d14[:5]), np.log(Ev[:5])),
        "D14_zbar": spearman_perm(np.log(d14[:5]), np.log([zbar[e] for e in PRES])),
        "within_bloc": within_relation(units, err_units, "bloc"),
        "within_candidate": within_relation(units, err_units, "candidate"),
    }
    # (c)
    vol = pl.read_parquet(VOL_STUDY).filter(pl.col("window") == 8).select(
        pl.col("election").cast(pl.Int64), pl.col("horizon").cast(pl.Int64), "V")
    cases = []
    for e in PRES:
        for h in HORIZONS:
            dd = hd.herding_at(e, h, registry=reg)
            zz = z.filter((pl.col("election") == e) & (pl.col("horizon") == h))["rms_z"][0]
            vv = vol.filter((pl.col("election") == e) & (pl.col("horizon") == h))["V"]
            cases.append({"election": e, "horizon": h, "D": dd.D, "D_bloc": dd.D_bloc,
                          "m": dd.m, "rms_z": zz, "V": float(vv[0]) if vv.len() else math.nan})
    cases = pl.DataFrame(cases)
    cases.write_parquet(out / "cases.parquet")
    c = {"D": case_relation(cases, "D"), "D_bloc": case_relation(cases, "D_bloc"),
         "V": case_relation(cases, "V")}
    verdict = {
        "a": trend["cand"]["holds"],
        "b": b["D14_E"]["rho"] < 0 and b["D14_E"]["p_neg"] < 0.10,
        "c": c["D"]["rho"] < 0 and c["D"]["p_neg"] <= 0.10,
    }
    verdict["c_better_than_V"] = bool(verdict["c"] and c["D"]["p_neg"] < c["V"]["p_pos"])
    res = {"trend_rows": trend_rows, "trend": trend, "b": b, "c": c, "zbar": zbar,
           "verdict": verdict}
    (out / "study.json").write_text(json.dumps(res, indent=2, default=float))
    fig_dispersion(tab, out / "dispersion.png")
    fig_relations(tab, E, cases, out / "relations.png")
    return res


# --- figures


def fig_dispersion(tab: pl.DataFrame, path: Path) -> None:
    order = ["h180", "h120", "h90", "h60", "h30", "h7", "final30", "final14"]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, group, title in ((axes[0], PRES, "présidentielles"), (axes[1], LISTS, "scrutins de liste")):
        for e in group:
            y = [disp(tab, e, w) for w in order]
            ax.plot(x, y, marker="o", markersize=3, color=COLORS[e], label=str(e), linewidth=1.3)
        ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xticks(x, ["180 j", "120 j", "90 j", "60 j", "30 j", "7 j", "30 der.", "14 der."],
                      fontsize=7)
        ax.set_yscale("log")
        ax.set_title(f"Dispersion relative D, {title}", fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        _style(ax)
    axes[0].set_ylabel("observée / attendue (échelle log)", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_relations(tab: pl.DataFrame, E: pl.DataFrame, cases: pl.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for e in ELECTIONS:
        d = disp(tab, e, "final14")
        y = E.filter(pl.col("election") == ekey(e))["E"][0]
        ax.scatter(d, y, color=COLORS[e], s=22)
        ax.annotate(str(e), (d, y), fontsize=7, color=MUTED, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("D, 14 derniers jours", fontsize=8, color=MUTED)
    ax.set_ylabel("erreur finale E (log-parts)", fontsize=8, color=MUTED)
    ax.set_title("(b) dispersion finale → erreur finale", fontsize=9, loc="left", color=INK)
    _style(ax)
    ax = axes[1]
    for e in PRES:
        s = cases.filter(pl.col("election") == e)
        ax.scatter(s["D"], s["rms_z"], color=COLORS[e], s=18, label=str(e))
        for r in s.iter_rows(named=True):
            if np.isfinite(r["D"]):
                ax.annotate(str(r["horizon"]), (r["D"], r["rms_z"]), fontsize=6, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("D, 30 jours avant la date", fontsize=8, color=MUTED)
    ax.set_ylabel("z quadratique moyen (backtest lists)", fontsize=8, color=MUTED)
    ax.set_title("(c) dispersion à la date → z du backtest", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- 2-3. forme et backtest


def link_fits(out: Path) -> None:
    for v in VARIANTS:
        for e in PRES:
            for h in HORIZONS:
                src, dst = fit_dir(FITS_FROM, "floor", e, h), fit_dir(out, v, e, h)
                if not (src / "draws.npz").exists():
                    raise FileNotFoundError(f"ajustement floor absent : {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src.resolve())


def _score_one(args):
    out, e, n_draws = args
    return sc.score_backtest(Path(out), [e], list(HORIZONS), variants=list(VARIANTS),
                             n_draws=n_draws, seed=0, write=False)


def run_score(out: Path, n_draws: int = 10_000) -> sc.ScoreResult:
    link_fits(out)
    with ProcessPoolExecutor(max_workers=len(PRES)) as ex:
        parts = list(ex.map(_score_one, [(str(out), e, n_draws) for e in PRES]))
    res = sc.ScoreResult(
        out=out,
        metrics=pl.concat([p.metrics for p in parts]),
        forecasts=pl.concat([p.forecasts for p in parts], how="diagonal_relaxed"),
        calibration=pl.concat([p.calibration for p in parts]),
        error_models={k: v for p in parts for k, v in p.error_models.items()},
        missing=[m for p in parts for m in p.missing],
        variant_error_models={k: v for p in parts for k, v in p.variant_error_models.items()},
    )
    sc.write_outputs(res, list(VARIANTS))
    for v in VARIANTS[1:]:
        sc.plot_calibration(res.calibration.filter(pl.col("variant") == v),
                            out / f"calibration_{v}.png")
    kap = []
    for (v, e), em in res.variant_error_models.items():
        hf = em.info.get("herding")
        if hf:
            kap.append({"variant": v, "exclude": e, "kappa": hf["kappa"], "form": hf["form"],
                        "nu_kappa": hf["nu"], "nu": em.params.nu, "gain": hf["gain_vs_kappa0"],
                        "profile": json.dumps(hf["profile"]),
                        "d_ref": json.dumps({str(k): x for k, x in hf["d_ref"].items()})})
    pl.DataFrame(kap).sort("variant", "exclude").write_parquet(out / "kappa.parquet")
    return res


def scales_table(out: Path) -> pl.DataFrame:
    """D, D_ref et facteur s par cas et variante (relus des modèles d'erreur écrits)."""
    reg = default_registry()
    rows = []
    for v in VARIANTS:
        for e in PRES:
            f = out / "error_models" / f"exclude_{e}_{v}.json"
            if not f.exists():
                continue
            hf = json.loads(f.read_text())["info"].get("herding")
            if not hf:
                continue
            fit = hd.HerdingFit(kappa=hf["kappa"], form=hf["form"], nu=hf["nu"],
                                d_ref={int(k): x for k, x in hf["d_ref"].items()},
                                train=hf["train"])
            for h in HORIZONS:
                d = hd.herding_at(e, h, registry=reg).D
                s = fit.scale(d, h)
                rows.append({"variant": v, "election": e, "horizon": h, "D": d,
                             "D_ref": s.d_ref, "kappa": s.kappa, "s": s.s})
    df = pl.DataFrame(rows)
    df.write_parquet(out / "scales.parquet")
    return df


# --- 2027


def current_2027(out: Path) -> dict:
    from vvs.eval.baselines import poll_mean
    from vvs.eval.configurations import known_configuration
    from vvs.model.data import election_date, load_polls
    from vvs.model.error import ErrorModel

    reg = default_registry()
    today = dt.date.today()
    vote = election_date(reg, 2027)
    h = (vote - today).days
    polls = load_polls(2027, today)
    conf = known_configuration(2027, today, reg, polls=polls).candidates
    blocs = [reg.bloc_of(c, 2027) for c in conf]
    res = {"asof": today.isoformat(), "horizon": h, "config": conf, "windows": {}}
    for days in (14, 30, 60):
        for det in (True, False):
            d = hd.presidential_dispersion(2027, today, conf, blocs, polls, days, det)
            res["windows"][f"{days}{'' if det else '_raw'}"] = {
                "D": d.D, "D_bloc": d.D_bloc, "m": d.m, "se_log_D": d.se_log_D,
                "pollsters": d.pollsters}
    d = hd.presidential_dispersion(2027, today, conf, blocs, polls, hd.WINDOW_DAYS)
    res["by_unit"] = {u: {"R": r, "share": d.shares[u], "sd": d.sd_points[u],
                          "sd_expected": d.sd_expected[u]} for u, r in d.by_unit.items()}
    res["by_bloc"] = d.by_bloc
    traj = []
    for back in range(0, 27, 2):
        a = today - dt.timedelta(weeks=back)
        try:
            c = known_configuration(2027, a, reg, polls=polls).candidates
        except ValueError:
            continue
        x = hd.presidential_dispersion(2027, a, c, [reg.bloc_of(y, 2027) for y in c],
                                       polls.filter(pl.col("fieldwork_end") <= a), hd.WINDOW_DAYS)
        traj.append({"asof": a.isoformat(), "horizon": (vote - a).days, "D": x.D, "m": x.m})
    res["trajectory"] = traj
    cal = sc.CALIBRATION_ELECTIONS
    pm = poll_mean(2027, today, conf, days=30, registry=reg, polls=polls)
    widths, fits = {}, {}
    for name, kw in (("lists", {"with_lists": True}),
                     ("lists_herding", {"with_lists": True, "herding": True}),
                     ("lists_herding_final", {"with_lists": True, "herding": "final"})):
        em = ErrorModel.fit(elections=cal, exclude=(), registry=reg, **kw)
        if em.herding is not None:
            em.herd_scale = em.herding.scale_at(2027, today, h, conf, blocs, polls)
            fits[name] = {"kappa": em.herding.kappa, "form": em.herding.form,
                          "d_ref": em.herding.d_ref, "D": em.herd_scale.d_obs,
                          "D_ref_h": em.herd_scale.d_ref, "s": em.herd_scale.s,
                          "profile": em.herding.profile}
        sims = em.sample(np.log(pm.shares)[None, :], h, conf, blocs, np.random.default_rng(7),
                         40_000)
        q = np.quantile(100 * sims, [0.05, 0.95], axis=0)
        widths[name] = {c: float(q[1, i] - q[0, i]) for i, c in enumerate(conf)}
    res["shares"] = {c: float(100 * s) for c, s in zip(conf, pm.shares)}
    res["width90"] = widths
    res["fits"] = fits
    # dispersion des présidentielles passées au même horizon (30 jours avant vote − 180)
    res["past_at_180"] = {str(e): hd.herding_at(e, 180, registry=reg).D for e in PRES}
    (out / "current_2027.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                      default=float))
    return res


# --- rapport


def _f(x, nd=2) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_report(out: Path, res: dict, cur: dict | None) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import backtest_lists as bl

    tab = pl.read_parquet(out / "dispersion.parquet")
    E = pl.read_parquet(out / "final_errors.parquet")
    units = pl.read_parquet(out / "units.parquet")
    L = ["# Troupeau des instituts et échelle de l'erreur de projection", "",
         (f"Généré le {dt.date.today().isoformat()} par `back/studies/herding.py`. Mesure et forme : "
         "`back/src/vvs/model/herding.py`. Critères fixés avant lecture : docstring de l'étude."), "",
         "## 1. Dispersion relative D (observée / attendue sous tirage aléatoire)", "",
         ("Un sondage par institut, tendance linéaire retirée ; D = 1 sous tirage aléatoire aux "
         "effectifs déclarés. Entre parenthèses : nombre d'instituts. * : effectifs imputés "
         "(1 000, ou médiane de l'institut au scrutin le plus proche)."), ""]
    order = [("final14", "14 der. j"), ("final30", "30 der. j"), ("h7", "7 j"), ("h30", "30 j"),
             ("h60", "60 j"), ("h90", "90 j"), ("h120", "120 j"), ("h180", "180 j")]
    for col, title in (("D", "candidats"), ("D_bloc", "blocs")):
        L += [f"### {title}", "", "| élection | " + " | ".join(t for _, t in order) + " |",
              "|---|" + "---|" * len(order)]
        for e in ELECTIONS:
            cells = []
            for w, _ in order:
                r = tab.filter((pl.col("election") == ekey(e)) & (pl.col("window") == w)
                               & pl.col("detrend"))
                x, m = r[col][0], r["m"][0]
                cells.append(f"{_f(x)} ({m})" if np.isfinite(x) else f"– ({m})")
            imp = tab.filter((pl.col("election") == ekey(e)) & (pl.col("window") == "final14")
                             & pl.col("detrend"))["n_imputed"][0]
            L.append(f"| {e}{'*' if imp > 0.5 else ''} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Sans retrait de tendance (écart à la moyenne des instituts), candidats :", "",
          "| élection | " + " | ".join(t for _, t in order) + " |", "|---|" + "---|" * len(order)]
    for e in ELECTIONS:
        L.append(f"| {e} | " + " | ".join(_f(disp(tab, e, w, "D", False)) for w, _ in order) + " |")
    L += ["", ("Présidentielles : les 14 et 30 derniers jours retiennent les mêmes sondages (chaque "
               "institut publie dans les deux dernières semaines, on garde son dernier sondage)."),
          "", "Figure : `dispersion.png`.", "",
          "### Par candidat, 14 derniers jours (présidentielles)", "",
          "R = variance observée / attendue ; écarts types entre instituts en points.", "",
          "| élection | candidat | part | σ observé | σ attendu | R |", "|---|---|---|---|---|---|"]
    u = units.filter((pl.col("window") == "final14") & (pl.col("level") == "candidate")
                     & (pl.col("share") >= 0.02)).sort("election", "share", descending=[False, True])
    for r in u.iter_rows(named=True):
        if r["election"] in [str(e) for e in PRES]:
            L.append(f"| {r['election']} | {r['unit']} | {100 * r['share']:.1f} | {r['sd']:.2f} "
                     f"| {r['sd_expected']:.2f} | {r['R']:.2f} |")
    L += ["", "## 2. (a) La dispersion se resserre-t-elle à l'approche du vote ?", "",
          ("Δ = log D(14 derniers jours) − log D(30 jours avant vote − 30) ; se par jackknife sur les "
          "instituts ; « sous le hasard » : borne haute à 90 % de D(14 j) sous 1."), "",
          "| élection | D(14 j) [haut 90 %] | Δ candidats | se | troupeau marqué | Δ blocs | Δ sans tendance | Δ 30 der. j | sous le hasard |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["trend_rows"]:
        L.append(f"| {r['election']} | {_f(r['D14'])} [{_f(r['D14_hi90'])}] | {_f(r['delta'], 2)} "
                 f"| {_f(r['se_delta'])} | {'oui' if r['marked'] else 'non'} | {_f(r['delta_bloc'])} "
                 f"| {_f(r['delta_raw'])} | {_f(r['delta_eq'])} | {'oui' if r['below_random'] else 'non'} |")
    L.append("")
    for k, t in res["trend"].items():
        k = {"cand": "candidats (critère)", "_bloc": "blocs", "_raw": "sans retrait de tendance",
             "_eq": "fenêtres égales (30 derniers jours contre 30 jours avant vote − 30)"}[k]
        L.append(f"- {k} : {t['n_neg']}/{t['n']} élections avec Δ < 0, moyenne {t['mean']:+.2f}, "
                 f"p (inversion des signes) = {t['p_signflip']:.3f} → critère "
                 f"**{'tenu' if t['holds'] else 'non tenu'}**.")
    L += ["", "## 3. (b) Dispersion finale et erreur finale", "",
          "| élection | D(14 j) | D_bloc(14 j) | E candidats | E blocs | z moyen backtest |",
          "|---|---|---|---|---|---|"]
    for e in ELECTIONS:
        r = E.filter(pl.col("election") == ekey(e)).row(0, named=True)
        zb = res["zbar"].get(e, res["zbar"].get(str(e)))
        L.append(f"| {e} | {_f(disp(tab, e, 'final14'))} | {_f(disp(tab, e, 'final14', 'D_bloc'))} "
                 f"| {r['E']:.3f} | {r['E_bloc']:.3f} | {_f(zb) if zb is not None else '–'} |")
    L += ["", "| relation | n | ρ | p (ρ < 0) |", "|---|---|---|---|"]
    lab = {"D14_E": "log D(14 j) → log E, dix élections (critère)",
           "D14bloc_Ebloc": "log D_bloc(14 j) → log E_bloc",
           "D30_E": "log D(30 der. j) → log E",
           "D14_E_pres": "log D(14 j) → log E, présidentielles",
           "D14_zbar": "log D(14 j) → log z moyen du backtest (n = 5)",
           "within_bloc": "dans l'élection : log R_bloc → log |erreur du bloc|",
           "within_candidate": "dans l'élection : log R_candidat → log |erreur du candidat|"}
    for k, t in res["b"].items():
        L.append(f"| {lab[k]} | {t['n']} | {t.get('rho', math.nan):+.2f} | {t.get('p_neg', math.nan):.3f} |")
    L += ["", (f"Critère (b) : **{'tenu' if res['verdict']['b'] else 'non tenu'}**. Figure : "
          "`relations.png`."), "",
          "## 4. (c) Dispersion à la date et erreur de projection du backtest", "",
          ("z = (logit résultat − logit médiane) / ((logit q95 − logit q05) / 3,29), backtest `lists` "
          "(runs/2026-10-20-backtest-lists), écart quadratique moyen par cas."), "",
          "| élection | horizon | D (instituts) | rms_z | V |", "|---|---|---|---|---|"]
    cases = pl.read_parquet(out / "cases.parquet")
    for r in cases.iter_rows(named=True):
        L.append(f"| {r['election']} | {r['horizon']} | {_f(r['D'])} ({r['m']}) | {r['rms_z']:.2f} "
                 f"| {_f(r['V'], 3)} |")
    L += ["", "| indice | n | ρ centré par horizon | p (ρ < 0) | p (ρ > 0) | 10e centile | 90e centile |",
          "|---|---|---|---|---|---|---|"]
    for k, t in res["c"].items():
        L.append(f"| {k} | {t['n']} | {t['rho']:+.2f} | {t['p_neg']:.3f} | {t['p_pos']:.3f} "
                 f"| {t['q10']:+.2f} | {t['q90']:+.2f} |")
    L += ["", (f"Critère (c) : **{'tenu' if res['verdict']['c'] else 'non tenu'}** ; meilleur que la "
          f"volatilité : **{'oui' if res['verdict']['c_better_than_V'] else 'non'}**."), ""]
    # backtest
    mfile = out / f"metrics_{'_'.join(VARIANTS)}.parquet"
    if mfile.exists():
        metrics = pl.read_parquet(mfile)
        kap = pl.read_parquet(out / "kappa.parquet")
        L += ["## 5. Forme : κ par pli (validation croisée interne)", "",
              ("s = clip((D_ref / D)^κ, [0,5 ; 2]) ; κ choisi sur la grille −1 à 2 (pas 0,25) avec ν, "
              "dans les élections d'apprentissage (scrutins de liste compris pour `lists_*`)."), "",
              "| variante | élection laissée | forme | κ | ν (κ) | ν final | gain CV contre κ = 0 |",
              "|---|---|---|---|---|---|---|"]
        for r in kap.iter_rows(named=True):
            L.append(f"| {r['variant']} | {r['exclude']} | {r['form']} | {r['kappa']:+.2f} "
                     f"| {r['nu_kappa']:g} | {r['nu']:g} | {r['gain']:+.3f} |")
        sct = scales_table(out)
        L += ["", "Facteur appliqué par cas (`lists_herding`) :", "",
              "| élection | horizon | D | D_ref | κ | s |", "|---|---|---|---|---|---|"]
        for r in sct.filter(pl.col("variant") == "lists_herding").iter_rows(named=True):
            L.append(f"| {r['election']} | {r['horizon']} | {_f(r['D'])} | {_f(r['D_ref'])} "
                     f"| {r['kappa']:+.2f} | {r['s']:.3f} |")
        L += ["", "## 6. Backtest (tirages floor, même graine, 10 000 tirages)", ""]
        wide = bl.comparison(metrics)
        wide.write_parquet(out / "comparison.parquet")
        adm = {}
        for v, base in PAIRS:
            bl.ACTS[v] = HORIZONS
            info, lines = bl.admission(wide, v, base)
            struct = structure(wide, v, base)
            info["structure"] = struct
            adm[f"{v}_vs_{base}"] = info
            L += bl.comparison_md(wide, v, base) + lines
            L.append(f"Critère de structure (couverture rapprochée de la nominale sans perte de "
                     f"log-score ni de CRPS, moyenne pondérée) : écart moyen {100 * struct['gap_before']:.1f} "
                     f"→ {100 * struct['gap_after']:.1f} pts, log-score min {struct['min_logscore']:+.3f}, "
                     f"CRPS max {struct['max_crps']:+.3f} → **{'rempli' if struct['ok'] else 'non rempli'}**.")
            L.append("")
        (out / "admission.json").write_text(json.dumps(adm, indent=2, default=str))
    if cur:
        L += ["## 7. Campagne 2027 au jour de l'étude", "",
              (f"Date {cur['asof']}, vote dans {cur['horizon']} jours. Configuration connue : "
              f"{', '.join(cur['config'])}."), "",
              "| fenêtre | D candidats | D blocs | instituts | se log D |", "|---|---|---|---|---|"]
        for k, w in cur["windows"].items():
            L.append(f"| {k} j | {_f(w['D'])} | {_f(w['D_bloc'])} | {w['m']} | {_f(w['se_log_D'])} |")
        L += ["", "Par candidat (30 jours) :", "", "| candidat | part | σ observé | σ attendu | R |",
              "|---|---|---|---|---|"]
        for c, x in sorted(cur["by_unit"].items(), key=lambda kv: -kv[1]["share"]):
            L.append(f"| {c} | {100 * x['share']:.1f} | {x['sd']:.2f} | {x['sd_expected']:.2f} | {x['R']:.2f} |")
        L += ["", "Trajectoire (30 jours glissants) : "
              + ", ".join(f"{t['asof']} {_f(t['D'])} ({t['m']})" for t in cur["trajectory"]), "",
              "Présidentielles passées au même horizon (30 jours avant vote − 180) : "
              + ", ".join(f"{e} {_f(x)}" for e, x in cur["past_at_180"].items()), ""]
        for name, f in cur["fits"].items():
            L.append(f"- `{name}` (sans élection exclue) : κ = {f['kappa']:+.2f} ({f['form']}), "
                     f"D = {_f(f['D'])}, D_ref(h) = {_f(f['D_ref_h'])}, s = {f['s']:.3f}.")
        L += ["", "Largeur de l'intervalle à 90 % (points), parts de la moyenne des 30 derniers jours :",
              "", "| candidat | part | " + " | ".join(cur["width90"]) + " |",
              "|---|---|" + "---|" * len(cur["width90"])]
        for c in sorted(cur["shares"], key=lambda c: -cur["shares"][c]):
            L.append(f"| {c} | {cur['shares'][c]:.1f} | "
                     + " | ".join(f"{cur['width90'][n][c]:.1f}" for n in cur["width90"]) + " |")
        L.append("")
    L[4:4] = synthesis(res, out, cur)
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def synthesis(res: dict, out: Path, cur: dict | None) -> list[str]:
    """Synthèse en tête du rapport (chiffres relus des sorties)."""
    t, b, c, v = res["trend"]["cand"], res["b"]["D14_E"], res["c"], res["verdict"]
    S = ["## Synthèse", "",
         (f"- (a) troupeau : la dispersion relative baisse entre 30-60 jours et les 14 derniers "
          f"jours dans {t['n_neg']} élections sur {t['n']} (moyenne de Δ = {t['mean']:+.2f} en log, "
          f"p = {t['p_signflip']:.3f}) → critère **{'tenu' if v['a'] else 'non tenu'}** ; "
          f"{sum(r['marked'] for r in res['trend_rows'])} élection(s) le montrent seules au-delà "
          f"de leur bruit (jackknife) ; {sum(r['below_random'] for r in res['trend_rows'])} sous "
          "le hasard en fin de campagne."),
         (f"- (b) erreur finale : Spearman(log D 14 j, log E) = {b['rho']:+.2f} sur dix élections "
          f"(p unilatéral négatif {b['p_neg']:.2f}, positif {b['p_pos']:.2f}) → critère "
          f"**{'tenu' if v['b'] else 'non tenu'}**"
          + (" ; le signe observé est l'inverse (dispersion forte, erreur forte)."
             if b["rho"] > 0 else ".")),
         (f"- (c) prévision : Spearman centré(log D à la date, log z) = {c['D']['rho']:+.2f} "
          f"(p négatif {c['D']['p_neg']:.2f}, positif {c['D']['p_pos']:.2f}) contre "
          f"{c['V']['rho']:+.2f} pour la volatilité → critère **{'tenu' if v['c'] else 'non tenu'}**."),
         ]
    kf = out / "kappa.parquet"
    af = out / "admission.json"
    if kf.exists() and af.exists():
        kap = pl.read_parquet(kf)
        adm = json.loads(af.read_text())
        for var in ("lists_herding", "lists_herding_final", "herding"):
            k = kap.filter(pl.col("variant") == var)["kappa"].to_list()
            a = next((x for key, x in adm.items() if key.startswith(var + "_vs_")), None)
            if a is None:
                continue
            cr = a["criteria"]
            S.append(f"- `{var}` : κ par pli {', '.join(f'{x:+.2f}' for x in k)} ; règle de "
                     f"docs/07 : critère 1 {'oui' if cr['c1_ref'] else 'non'} / "
                     f"{'oui' if cr['c1_w'] else 'non'}, critère 2 {'oui' if cr['c2'] else 'non'}, "
                     f"critère 3 {'oui' if cr['c3_cand_ref'] else 'non'} ; structure "
                     f"{'remplie' if a['structure']['ok'] else 'non remplie'}.")
    if cur and "lists_herding" in cur.get("fits", {}):
        f = cur["fits"]["lists_herding"]
        S.append(f"- 2027 ({cur['asof']}, {cur['horizon']} jours) : D = {_f(f['D'])} sur les 30 "
                 f"derniers jours contre D_ref = {_f(f['D_ref_h'])} au même horizon ; facteur "
                 f"`lists_herding` s = {f['s']:.2f}.")
    return S + [""]


def structure(wide: pl.DataFrame, v: str, base: str) -> dict:
    """Critère de structure (studies/volatility.py, C4), moyenne pondérée 2002-2022."""
    def wm(metric, col, h, f=lambda x: x):
        g = wide.filter((pl.col("metric") == metric) & (pl.col("horizon") == h))
        w = np.array([ELECTION_WEIGHT.get(e, 1.0) for e in g["election"]])
        return float((w * f(g[col].to_numpy())).sum() / w.sum())

    gb, ga, ls, cr = [], [], [], []
    for h in HORIZONS:
        for m, nom in (("cov50", 0.5), ("cov90", 0.9)):
            gb.append(abs(wm(m, base, h) - nom))
            ga.append(abs(wm(m, v, h) - nom))
        ls.append(wm("log_score_pair", v, h) - wm("log_score_pair", base, h))
        cr.append(wm("crps", v, h) - wm("crps", base, h))
    b, a = float(np.mean(gb)), float(np.mean(ga))
    return {"gap_before": b, "gap_after": a, "min_logscore": min(ls), "max_crps": max(cr),
            "ok": bool(a < b and min(ls) >= -0.01 and max(cr) <= 0.01)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "study", "score", "current", "report"])
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-herding")
    ap.add_argument("--n-draws", type=int, default=10_000)
    a = ap.parse_args()
    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("all", "study"):
        res = run_study(out)
        print(json.dumps({"trend": res["trend"], "b": res["b"], "c": res["c"],
                          "verdict": res["verdict"]}, indent=2, default=float))
    if a.cmd in ("all", "score"):
        run_score(out, a.n_draws)
    if a.cmd in ("all", "current"):
        current_2027(out)
    if a.cmd in ("all", "report", "study", "score", "current"):
        res = json.loads((out / "study.json").read_text())
        res["zbar"] = {int(k): v for k, v in res["zbar"].items()}
        cf = out / "current_2027.json"
        write_report(out, res, json.loads(cf.read_text()) if cf.exists() else None)


if __name__ == "__main__":
    main()
