"""Étude : la volatilité réalisée d'un candidat comme échelle de sa composante propre v_c de η.

    uv run python studies/volatility_candidate.py [all|study|kappa|score|current|report]
        [--out runs/<date>-volatility-candidate] [--fits-from runs/2026-10-20-backtest]

Forme et estimation : `vvs.model.volatility_candidate` (s_c = clip((V_c / V_ref)^κ, 0,5, 2),
V_ref médiane des candidats de même taille aux horizons voisins, κ en validation croisée
interne, η réestimé conjointement). Volets :

1. Reproduction du β = +0,28 de runs/2026-10-20-volatility (mêmes fonctions,
   `studies/volatility.py`), puis relation par candidat, France 2002-2022, entre log V_c
   (fenêtres de 4, 8, 12 semaines) et (a) log |mouvement centré| jusqu'au résultat, à log part
   et horizon contrôlés, grille fine et quatre horizons du backtest ; placebo : V permutée
   entre candidats de la même élection au même horizon ; β sans chaque élection ; (b) log |z|
   du candidat dans les prévisions du backtest de référence (runs/2026-10-20-backtest-lists,
   variante `lists`), mêmes contrôles et placebo, et pente de log |z| sur log (V / V_ref)
   (V_ref sans l'élection) : le κ dont η aurait eu besoin.
2. Jennings–Wlezien hors France, par parti : même relation (forme a priori), bootstrap des pays,
   placebo dans l'élection.
3. κ par pli : `ErrorModel.fit(exclude=[e], candidate_volatility=True)` avec et sans scrutins
   de liste ; profil de vraisemblance en validation croisée interne.
4. Backtest sur les tirages floor existants (liens vers --fits-from), même graine
   [0, élection, horizon] : `lists` (référence, défaut actuel) contre `lists_volcand`, et `floor`
   contre `volcand` ; sensibilité hors règle `lists_volcand_beta` : κ fixé au β de (1a) estimé
   sans l'élection évaluée (fenêtre 8, quatre horizons, V nulle au plancher), η réestimé. Règle de docs/07, horizons d'action : tous.
5. Indice actuel des candidats 2027.

Critères fixés avant les résultats : la variante n'est admise que par la règle de docs/07
appliquée comme pour `lists` (studies/backtest_lists.py : gain de log-score et de CRPS à tous
les horizons en moyenne 2012-2022, même signe sur deux élections de référence sur trois,
couverture non dégradée). La forme (fenêtre de 8 semaines, classes de taille, bornes du rapport
et de s) est fixée dans le module avant le backtest ; les fenêtres 4 et 12 ne sont que
descriptives.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.eval import scoring as sc
from vvs.eval.backtest import ELECTION_WEIGHT, asof_for, fit_dir
from vvs.eval.baselines import poll_mean
from vvs.eval.configurations import known_configuration, official_map
from vvs.model import error as er
from vvs.model import volatility as vo
from vvs.model import volatility_candidate as vc
from vvs.model.data import election_date, load_polls
from vvs.model.run import _hash_frame, draws_log_shares
from vvs.registry import default_registry

STUDIES = Path(__file__).resolve().parent
ELECTIONS = (2002, 2007, 2012, 2017, 2022)
REFERENCE = (2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
FINE = (7, 14, 21, 30, 45, 60, 90, 120, 150, 180)
JW_HORIZONS = (7, 14, 30, 60, 90, 120, 180)
REF_BACKTEST = paths.RUNS / "2026-10-20-backtest-lists"
DEFAULT_FITS = paths.RUNS / "2026-10-20-backtest"
N_PLACEBO = 2000
N_BOOT = 1000
VARIANTS = ("lists", "lists_volcand", "floor", "volcand")
SENSITIVITY = "lists_volcand_beta"
FOCUS = (("melenchon_jean_luc", 2022), ("zemmour_eric", 2022), ("pecresse_valerie", 2022),
         ("fillon_francois", 2017))


def _load_study(name: str):
    spec = importlib.util.spec_from_file_location(f"study_{name}", STUDIES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _center(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = x[m] - x[m].mean()
    return out


def _codes(*cols) -> np.ndarray:
    keys = list(zip(*[np.asarray(c).tolist() for c in cols]))
    idx = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([idx[k] for k in keys])


def regression(y: np.ndarray, lv: np.ndarray, lp: np.ndarray | None, fe: np.ndarray,
               perm_groups: np.ndarray, n_perm: int = N_PLACEBO, seed: int = 0) -> dict:
    """β de y sur lv, contrôles lp et effets fixes `fe` (centrage), placebo : lv permutée dans
    `perm_groups`."""
    yc = _center(y, fe)
    cols = [] if lp is None else [_center(lp, fe)]

    def beta(lv_):
        X = np.column_stack([_center(lv_, fe), *cols])
        return float(np.linalg.lstsq(X, yc, rcond=None)[0][0])

    b = beta(lv)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(perm_groups == g) for g in np.unique(perm_groups)]
    pb = []
    for _ in range(n_perm):
        lv_p = lv.copy()
        for m in groups:
            lv_p[m] = lv[rng.permutation(m)]
        pb.append(beta(lv_p))
    pb = np.array(pb)
    if not n_perm:
        return {"beta": b, "n": len(y)}
    return {"beta": b, "placebo_p": float((np.sum(pb >= b) + 1) / (n_perm + 1)),
            "placebo_q95": float(np.quantile(pb, 0.95)), "n": len(y)}


# --- 1. France


def reproduce_beta(reg) -> dict:
    """β de runs/2026-10-20-volatility, avec les fonctions de studies/volatility.py."""
    vs = _load_study("volatility")
    _, units = vs.french_volatility(reg)
    _, moves = vs.french_moves(reg)
    return {t: vs.unit_relation(units, moves, "candidate", t) for t in ("result", "move")}


def candidate_moves(reg, horizons) -> pl.DataFrame:
    """|mouvement centré| (log-parts) de chaque candidat officiel, de l'agrégat des 30 jours
    avant la date au résultat (`studies/volatility.french_moves`, cible result)."""
    vs = _load_study("volatility")
    out = []
    for e in ELECTIONS:
        fin = er.final_item(e, reg)
        dfin = dict(zip(fin.units, fin.d))
        for h in horizons:
            its = er.move_items(e, reg, (h,))
            if not its:
                continue
            it = its[0]
            keep = [i for i, u in enumerate(it.units) if u in dfin]
            p = it.p[keep] / it.p[keep].sum()
            dr = it.d[keep] + np.array([dfin[it.units[i]] for i in keep])
            cc = vs._centered(dr, p)
            for i, x in enumerate(cc):
                out.append({"election": e, "horizon": h, "unit": it.units[keep[i]],
                            "abs_move": abs(float(x)), "share": float(p[i])})
    return pl.DataFrame(out)


def vol_tables(reg, horizons) -> pl.DataFrame:
    """V par candidat, fenêtres 4, 8, 12 semaines (colonne `window`)."""
    return pl.concat([vc.candidate_table(ELECTIONS, horizons, w, reg).with_columns(
        pl.lit(w).alias("window")) for w in vo.WINDOWS])


def move_relations(moves: pl.DataFrame, vols: pl.DataFrame) -> tuple[list[dict], pl.DataFrame]:
    rows = []
    d_all = moves.join(vols, on=["election", "horizon", "unit"], how="inner")
    for w in vo.WINDOWS:
        for grid, hs in (("fine", FINE), ("backtest", HORIZONS)):
            for zeros in ("drop", "floor"):
                d = d_all.filter((pl.col("window") == w) & pl.col("horizon").is_in(list(hs))
                                 & (pl.col("abs_move") > 0) & pl.col("V").is_finite())
                if zeros == "drop":
                    d = d.filter(pl.col("V") > 0)
                v = d["V"].to_numpy()
                if zeros == "floor":  # V nulle : plancher au 5e centile des V positives
                    v = np.maximum(v, np.quantile(v[v > 0], 0.05))
                e, h = d["election"].to_numpy(), d["horizon"].to_numpy()
                r = regression(np.log(d["abs_move"].to_numpy()), np.log(v),
                               np.log(d["share"].to_numpy()), h, _codes(e, h))
                # β sans chaque élection
                loo = {}
                for x in ELECTIONS:
                    m = e != x
                    rr = regression(np.log(d["abs_move"].to_numpy()[m]), np.log(v[m]),
                                    np.log(d["share"].to_numpy()[m]), h[m], _codes(e[m], h[m]),
                                    n_perm=0)
                    loo[str(x)] = rr["beta"]
                rows.append({"window": w, "grid": grid, "zeros": zeros, **r,
                             "n_elections": len(set(e.tolist())), "loo_beta": loo})
    return rows, d_all


def z_scores(ref_dir: Path) -> pl.DataFrame:
    """z = (logit résultat − logit médiane) / ((logit q95 − logit q05) / 3,29), variante lists."""
    f = pl.concat([pl.read_parquet(p) for p in sorted(ref_dir.glob("forecasts*.parquet"))],
                  how="diagonal_relaxed")
    f = f.filter((pl.col("variant") == "lists") & (pl.col("model") == "model")
                 & (pl.col("level") == "candidate")).unique(["election", "horizon", "unit"])

    def lg(x):
        x = np.clip(np.asarray(x, dtype=float) / 100, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))

    sd = (lg(f["q95"]) - lg(f["q05"])) / (2 * 1.6449)
    z = (lg(f["official"]) - lg(f["median"])) / sd
    return f.select("election", "horizon", "unit", "official", "median", "crps", "in50",
                    "in90").with_columns(pl.Series("z", z))


def z_relations(z: pl.DataFrame, vols: pl.DataFrame) -> tuple[list[dict], pl.DataFrame]:
    rows = []
    d_all = z.join(vols, on=["election", "horizon", "unit"], how="inner")
    for w in vo.WINDOWS:
        d = d_all.filter((pl.col("window") == w) & (pl.col("V") > 0) & (pl.col("z").abs() > 0))
        e, h = d["election"].to_numpy(), d["horizon"].to_numpy()
        r = regression(np.log(d["z"].abs().to_numpy()), np.log(d["V"].to_numpy()),
                       np.log(d["p"].to_numpy()), h, _codes(e, h))
        rows.append({"window": w, "x": "log V (log p contrôlé)", **r})
    # κ « nécessaire » : pente de log |z| sur log (V / V_ref), V_ref sans l'élection, forme du
    # module (fenêtre principale, rapport borné, V nulle comprise)
    d = d_all.filter((pl.col("window") == vc.WINDOW) & (pl.col("z").abs() > 0)
                     & pl.col("V").is_finite())
    table = vols.filter(pl.col("window") == vc.WINDOW)
    lr = []
    for e, h, v, p in d.select("election", "horizon", "V", "p").iter_rows():
        ref = vc.Reference.build(table, [x for x in ELECTIONS if x != e])
        lr.append(math.log(vc.ratio(v, p, h, ref)))
    d = d.with_columns(pl.Series("log_ratio", lr)).filter(pl.col("log_ratio").is_finite())
    e, h = d["election"].to_numpy(), d["horizon"].to_numpy()
    ly, lx = np.log(d["z"].abs().to_numpy()), d["log_ratio"].to_numpy()
    r = regression(ly, lx, None, h, _codes(e, h))
    loo = {str(x): regression(ly[e != x], lx[e != x], None, h[e != x],
                              _codes(e[e != x], h[e != x]), n_perm=0)["beta"] for x in ELECTIONS}
    rows.append({"window": vc.WINDOW, "x": "log V/V_ref (forme du module)", **r, "loo_beta": loo})
    return rows, d


# --- 2. Jennings–Wlezien par parti


def jw_party(window: int = vc.WINDOW, min_share: float = 2.0) -> pl.DataFrame:
    from vvs.ingest.jennings_wlezien import load_polls as jw_load

    d = jw_load().filter((pl.col("round") == 1) & (pl.col("country_iso2") != "FR"))
    d = d.with_columns((pl.col("poll_share") / pl.col("poll_total")).alias("s")).filter(
        pl.col("s") > 0)
    rows = []
    for (eid,), g in d.group_by("election_id"):
        res = g.group_by("party_id").agg(pl.col("result_share").first(),
                                         pl.col("result_total").first())
        res = {r["party_id"]: r["result_share"] / r["result_total"]
               for r in res.iter_rows(named=True) if r["result_share"] >= min_share}
        if len(res) < 3:
            continue
        g = g.filter(pl.col("party_id").is_in(list(res)))
        for h in JW_HORIZONS:
            win = g.filter(pl.col("days_before").is_between(h, h + 7 * window - 1))
            weeks: dict = {}
            for pid, db, s, n, m in win.select("party_id", "days_before", "s", "sample_size",
                                               "n_polls").iter_rows():
                weeks.setdefault(pid, {}).setdefault((db - h) // 7, []).append(
                    (s, (n or vo.DEFAULT_N) * (m or 1)))
            at = g.filter(pl.col("days_before").is_between(h, h + 13)).group_by("party_id").agg(
                pl.col("s").mean())
            at = {p: s for p, s in zip(at["party_id"].to_list(), at["s"].to_list()) if s > 0}
            common = [p for p in res if p in at]
            if len(common) < 3:
                continue
            a = np.array([at[p] for p in common])
            w = a / a.sum()
            dm = np.log([res[p] for p in common]) - np.log(a)
            cm = dm - (w * dm).sum()
            for p, x, s in zip(common, cm, w):
                if p not in weeks:
                    continue
                v, _, _ = vo._unit_volatility(weeks[p], window)
                if np.isfinite(v):
                    rows.append({"election_id": eid, "country": g["country_iso2"][0],
                                 "horizon": h, "party": p, "V": v, "share": float(s),
                                 "abs_move": abs(float(x))})
    return pl.DataFrame(rows)


def jw_relation(df: pl.DataFrame, seed: int = 0) -> list[dict]:
    out = []
    for h in [*JW_HORIZONS, "all"]:
        d = df.filter((pl.col("V") > 0) & (pl.col("abs_move") > 0))
        if h != "all":
            d = d.filter(pl.col("horizon") == h)
        y, lv, lp = (np.log(d["abs_move"].to_numpy()), np.log(d["V"].to_numpy()),
                     np.log(d["share"].to_numpy()))
        hs, eid, ctry = d["horizon"].to_numpy(), d["election_id"].to_numpy(), d["country"].to_numpy()
        r = regression(y, lv, lp, hs, _codes(eid, hs), n_perm=N_BOOT, seed=seed)
        rng = np.random.default_rng(seed)
        uc = np.unique(ctry)
        groups = {c: np.flatnonzero(ctry == c) for c in uc}
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([groups[c] for c in rng.choice(uc, size=len(uc))])
            boots.append(regression(y[idx], lv[idx], lp[idx], hs[idx], hs[idx], n_perm=0)["beta"])
        out.append({"horizon": str(h), **r, "n_countries": len(uc),
                    "lo95": float(np.quantile(boots, 0.025)),
                    "hi95": float(np.quantile(boots, 0.975))})
    return out


# --- 3. κ par pli


def fold_models(out: Path, reg) -> dict:
    """κ et profils par pli et calibration, mis en cache dans out/kappa.json."""
    path = out / "kappa.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for lists in (True, False):
        for e in [*ELECTIONS, "all"]:
            key = f"{'lists' if lists else 'floor'}:{e}"
            if key in cache:
                continue
            em = er.ErrorModel.fit(elections=ELECTIONS, exclude=[e] if e != "all" else (),
                                   registry=reg, with_lists=lists, candidate_volatility=True)
            info = em.info["candidate_volatility"]
            cache[key] = {"kappa": info["kappa"], "gain_vs_kappa0": info["gain_vs_kappa0"],
                          "profile": info["profile"], "nu": em.params.nu,
                          "reference": info["reference"]}
            print(key, info["kappa"], round(info["gain_vs_kappa0"], 3), flush=True)
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2))
    return cache


# --- 4. backtest


def link_fits(out: Path, fits_from: Path, variants) -> None:
    for v in variants:
        for e in ELECTIONS:
            for h in HORIZONS:
                src, dst = fit_dir(fits_from, "floor", e, h), fit_dir(out, v, e, h)
                if not (src / "draws.npz").exists():
                    raise FileNotFoundError(f"ajustement floor absent : {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src.resolve())


def check_ref(fits_from: Path) -> list[str]:
    stale = []
    for e in ELECTIONS:
        for h in HORIZONS:
            m = json.loads((fit_dir(fits_from, "floor", e, h) / "manifest.json").read_text())
            if m["data"]["rows_hash"] != _hash_frame(load_polls(e, asof_for(e, h))):
                stale.append(f"{e}-{h}")
    return stale


def beta_kappas(out: Path) -> dict[int, float]:
    """κ de la sensibilité : β de (1a) sans l'élection évaluée (fenêtre principale, quatre
    horizons du backtest, V nulle comprise au plancher, comme dans la forme du module)."""
    rel = json.loads((out / "relations.json").read_text())
    r = next(x for x in rel["moves"] if x["window"] == vc.WINDOW and x["grid"] == "backtest"
             and x["zeros"] == "floor")
    return {int(e): float(b) for e, b in r["loo_beta"].items()}


def score_sensitivity(out: Path, reg, n_draws: int = 10_000, seed: int = 0) -> sc.ScoreResult:
    """`lists_volcand_beta` : κ fixé, η réestimé, même graine que score_backtest."""
    from vvs.model.error_lists import default_list_items

    kb = beta_kappas(out)
    metric_rows, fc_rows, cal_rows, scales = [], [], [], []
    for e in ELECTIONS:
        em = er.ErrorModel.fit(elections=ELECTIONS, exclude=[e], registry=reg, with_lists=True)
        train = [x for x in ELECTIONS if x != e]
        bias = er.stable_bias(er.calibration_items(train, reg) + default_list_items())[0]
        vc.attach(em, train, default_list_items(), em.params.gamma, bias, reg,
                  grid=(round(kb[e], 3),))
        off = sc.official(reg, e)
        for h in HORIZONS:
            asof = asof_for(e, h)
            polls = load_polls(e, asof)
            fd = fit_dir(out, SENSITIVITY, e, h)
            conf = json.loads((fd / "configuration.json").read_text())
            dr = np.load(fd / "draws.npz")
            omap = official_map(conf["candidates"], e, reg, substitution="lam" in dr.files)
            K = omap.config
            cands = [str(c) for c in dr["candidates"]]
            cblocs = [str(b) for b in dr["blocs"]]
            kblocs = [cblocs[cands.index(c)] for c in K]
            rng = np.random.default_rng([seed, e, h])
            em.cand_scale = em.cand_volatility.scaler(e, asof, polls, reg)
            case = sc.score_case(draws_log_shares(dr, K, "asof"), draws_log_shares(dr, K, "vote"),
                                 K, kblocs, h, em, omap, off, {}, rng, n_draws)
            for c, s in zip(K, em.cand_scale(K)):
                scales.append({"election": e, "horizon": h, "variant": SENSITIVITY, "unit": c,
                               "kappa": em.cand_volatility.kappa, "s": float(s)})
            em.cand_scale = None
            m_rows, f_rows, c_rows = sc.case_rows(case, {"election": e, "horizon": h,
                                                          "variant": SENSITIVITY})
            metric_rows += m_rows
            fc_rows += f_rows
            cal_rows += c_rows
    pl.DataFrame(scales).write_parquet(out / f"scales_{SENSITIVITY}.parquet")
    return sc.ScoreResult(out=out, metrics=pl.DataFrame(metric_rows),
                          forecasts=pl.DataFrame(fc_rows, infer_schema_length=None),
                          calibration=pl.DataFrame(cal_rows))


def case_scales(out: Path, kap: dict, reg) -> pl.DataFrame:
    """Échelles s_c appliquées par `lists_volcand` et `volcand` (κ du pli), par candidat."""
    rows = []
    for calib, v in (("lists", "lists_volcand"), ("floor", "volcand")):
        for e in ELECTIONS:
            k = kap[f"{calib}:{e}"]["kappa"]
            train = [x for x in ELECTIONS if x != e]
            table = vc.candidate_table(train, registry=reg)
            cv = vc.CandidateVolatility(k, vc.Reference.build(table, train))
            for h in HORIZONS:
                asof = asof_for(e, h)
                conf = json.loads((fit_dir(out, v, e, h) / "configuration.json").read_text())
                scaler = cv.scaler(e, asof, load_polls(e, asof), reg)
                idx = scaler.index(conf["candidates"])
                for c in conf["candidates"]:
                    V = idx.by_unit.get(c, math.nan) if idx else math.nan
                    p = idx.weights.get(c, math.nan) if idx else math.nan
                    r = vc.ratio(V, p, h, cv.ref) if np.isfinite(V) else math.nan
                    rows.append({"election": e, "horizon": h, "variant": v, "unit": c, "V": V,
                                 "p": p, "ratio": r, "kappa": k,
                                 "s": float(vc.scale_from_ratio([r], k)[0])})
    return pl.DataFrame(rows)


def cmd_score(out: Path, fits_from: Path, reg) -> None:
    stale = check_ref(fits_from)
    if stale:
        raise SystemExit(f"ajustements de référence périmés (données changées) : {stale}")
    link_fits(out, fits_from, [*VARIANTS, SENSITIVITY])
    res = sc.score_backtest(out, list(ELECTIONS), list(HORIZONS), variants=list(VARIANTS))
    sens = score_sensitivity(out, reg)
    metrics = pl.concat([res.metrics, sens.metrics], how="diagonal_relaxed")
    forecasts = pl.concat([res.forecasts, sens.forecasts], how="diagonal_relaxed")
    metrics.write_parquet(out / "metrics_all.parquet")
    forecasts.write_parquet(out / "forecasts_all.parquet")
    for (v, e), em in res.variant_error_models.items():
        if em.cand_volatility is not None:
            print(v, e, "κ =", em.cand_volatility.kappa)


# --- 5. 2027


def current_2027(out: Path, kap: dict, reg) -> dict:
    today = dt.date.today()
    vote = election_date(reg, 2027)
    h = (vote - today).days
    polls = load_polls(2027, today)
    conf = known_configuration(2027, today, reg, polls=polls).candidates
    table = vc.candidate_table(ELECTIONS, registry=reg)
    ref = vc.Reference.build(table, ELECTIONS)
    k_all = kap["lists:all"]["kappa"]
    kb = json.loads((out / "relations.json").read_text())
    kb = next(x for x in kb["moves"] if x["window"] == vc.WINDOW and x["grid"] == "backtest"
              and x["zeros"] == "floor")["beta"]
    blocs = [reg.bloc_of(c, 2027) for c in conf]
    res = {"asof": today.isoformat(), "horizon": h, "kappa_all": k_all, "kappa_beta": kb,
           "windows": {}}
    for w in vo.WINDOWS:
        v = vo.realized_volatility(polls, today, conf, blocs, w)
        res["windows"][w] = {c: v.by_unit.get(c) for c in conf}
    v = vo.realized_volatility(polls, today, conf, blocs, vc.WINDOW)
    pm = poll_mean(2027, today, conf, days=30, registry=reg, polls=polls)
    cands = []
    for i, c in enumerate(conf):
        V = v.by_unit.get(c, math.nan)
        p = v.weights.get(c, math.nan)
        r = vc.ratio(V, p, h, ref) if np.isfinite(V) else math.nan
        cands.append({"unit": c, "bloc": blocs[i], "share30": float(100 * pm.shares[i]),
                      "V": V, "p_window": p, "V_ref": ref.vref(p, h) if np.isfinite(p) else None,
                      "ratio": r, "s_kappa_all": float(vc.scale_from_ratio([r], k_all)[0]),
                      "s_kappa_beta": float(vc.scale_from_ratio([r], kb)[0])})
    res["candidates"] = cands
    res["n_polls"] = v.n_polls
    (out / "current_2027.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                      default=str))
    return res


# --- rapport


def _md(df: pl.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.iter_rows():
        cells = []
        for c, x in zip(df.columns, r):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                cells.append("–")
            elif isinstance(x, float):
                cells.append(fmt.get(c, "{:.3f}").format(x))
            else:
                cells.append(str(x))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _wmean(vals: dict) -> float:
    w = np.array([ELECTION_WEIGHT.get(e, 1.0) for e in vals])
    return float((w * np.array(list(vals.values()))).sum() / w.sum())


def coverage_by_volatility(out: Path, scales: pl.DataFrame, metrics: pl.DataFrame,
                           variants: list[str]) -> pl.DataFrame:
    """Couverture et CRPS par tiers du rapport V / V_ref (candidats officiels prévus)."""
    s = scales.filter(pl.col("variant") == "lists_volcand").select(
        "election", "horizon", "unit", "ratio")
    q = s.filter(pl.col("ratio").is_finite())["ratio"].quantile(1 / 3), \
        s.filter(pl.col("ratio").is_finite())["ratio"].quantile(2 / 3)
    s = s.with_columns(
        pl.when(~pl.col("ratio").is_finite()).then(pl.lit("V indéfinie"))
        .when(pl.col("ratio") <= q[0]).then(pl.lit("1 bas"))
        .when(pl.col("ratio") <= q[1]).then(pl.lit("2 moyen"))
        .otherwise(pl.lit("3 haut")).alias("tier"))
    m = metrics.filter((pl.col("level") == "candidate") & (pl.col("model") == "model")
                       & pl.col("variant").is_in(variants)
                       & pl.col("metric").is_in(["crps", "in50", "in90"]))
    j = m.join(s, on=["election", "horizon", "unit"], how="left").with_columns(
        pl.col("tier").fill_null("V indéfinie"))
    return (j.group_by("variant", "tier", "metric").agg(pl.col("value").mean(), pl.len())
            .pivot(on="metric", index=["tier", "variant", "len"], values="value")
            .sort("tier", "variant"))


def write_report(out: Path, reg) -> None:
    bl = _load_study("backtest_lists")
    bl.ACTS.update({v: HORIZONS for v in ("lists_volcand", "volcand", SENSITIVITY)})
    rel = json.loads((out / "relations.json").read_text())
    kap = json.loads((out / "kappa.json").read_text())
    metrics = pl.read_parquet(out / "metrics_all.parquet")
    forecasts = pl.read_parquet(out / "forecasts_all.parquet")
    cur = json.loads((out / "current_2027.json").read_text())
    scales = case_scales(out, kap, reg)
    scales.write_parquet(out / "scales.parquet")
    L = [f"# Volatilité réalisée par candidat et composante v_c de η ({dt.date.today()})", "",
         ("Généré par `back/studies/volatility_candidate.py` ; forme et estimation : "
         "`back/src/vvs/model/volatility_candidate.py`. Critères fixés avant les résultats : "
         "docstring de l'étude."), ""]
    # 1. reproduction
    rp = rel["reproduction"]
    L += ["## 1. Reproduction du β de l'étude précédente", "",
          ("Mêmes fonctions (`studies/volatility.py` : `french_volatility`, `french_moves`, "
          "`unit_relation`), fenêtre de 8 semaines, grille fine, V nulle exclue, placebo : V "
          "permutée entre candidats d'un même horizon."), "",
          "| cible | n | élections | β | p placebo |", "|---|---|---|---|---|"]
    for t, r in rp.items():
        L.append(f"| {t} | {r['n']} | {r['n_elections']} | {r['beta']:+.3f} | {r['placebo_p']:.3f} |")
    L += ["", "## 2. Relation par candidat (France 2002-2022)", "",
          "### (a) Mouvement ultérieur jusqu'au résultat", "",
          ("log |mouvement centré| sur log V_c, contrôle log part, effets d'horizon ; placebo : V "
          "permutée entre candidats de la même élection au même horizon (2 000 tirages). "
          "« zeros » : V nulle exclue (drop) ou au 5e centile des V positives (floor). β sans "
          "chaque élection en dernière colonne."), "",
          "| fenêtre | grille | V nulle | n | β | p placebo | 95e centile placebo | β sans 2002 / 2007 / 2012 / 2017 / 2022 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rel["moves"]:
        loo = " / ".join(f"{r['loo_beta'][str(e)]:+.2f}" for e in ELECTIONS)
        L.append(f"| {r['window']} | {r['grid']} | {r['zeros']} | {r['n']} | {r['beta']:+.3f} | "
                 f"{r['placebo_p']:.3f} | {r['placebo_q95']:+.3f} | {loo} |")
    L += ["", "### (b) z-score du candidat dans le backtest de référence (`lists`)", "",
          ("z = (logit résultat − logit médiane) / ((logit q95 − logit q05) / 3,29). Ligne « forme "
          "du module » : pente de log |z| sur log (V / V_ref) (V_ref sans l'élection, rapport "
          "borné, V nulle comprise), sans autre contrôle que l'horizon : c'est le κ dont η aurait "
          "eu besoin."), "",
          "| fenêtre | régresseur | n | pente | p placebo | 95e centile placebo | sans 2002 / 2007 / 2012 / 2017 / 2022 |",
          "|---|---|---|---|---|---|---|"]
    for r in rel["z"]:
        loo = (" / ".join(f"{r['loo_beta'][str(e)]:+.2f}" for e in ELECTIONS)
               if "loo_beta" in r else "–")
        L.append(f"| {r['window']} | {r['x']} | {r['n']} | {r['beta']:+.3f} | "
                 f"{r['placebo_p']:.3f} | {r['placebo_q95']:+.3f} | {loo} |")
    L += ["", "## 3. Jennings–Wlezien hors France, par parti (forme a priori)", "",
          ("Même régression (fenêtre de 8 semaines, mouvement jusqu'au résultat, part contrôlée, "
          "effets d'horizon), IC à 95 % par bootstrap des pays, placebo dans l'élection."), "",
          "| horizon | n | pays | β | IC 95 % | p placebo |", "|---|---|---|---|---|---|"]
    for r in rel["jw"]:
        L.append(f"| {r['horizon']} | {r['n']} | {r['n_countries']} | {r['beta']:+.3f} | "
                 f"[{r['lo95']:+.2f}, {r['hi95']:+.2f}] | {r['placebo_p']:.3f} |")
    # 4. κ
    L += ["", "## 4. κ par pli (validation croisée interne, η réestimé à chaque κ)", "",
          ("Gain = log-vraisemblance en validation croisée interne contre κ = 0 (ν du pli). "
          "Profil complet : `kappa.json`."), "",
          "| pli | calibration | ν | κ | gain | profil (κ : log-vraisemblance) |",
          "|---|---|---|---|---|---|"]
    for key, r in kap.items():
        calib, e = key.split(":")
        prof = ", ".join(f"{float(k):+.2f} : {v:.1f}" for k, v in r["profile"].items())
        L.append(f"| {e} | {calib} | {r['nu']:g} | {r['kappa']:+.2f} | "
                 f"{r['gain_vs_kappa0']:+.2f} | {prof} |")
    kb = beta_kappas(out)
    L += ["", "κ de la sensibilité `lists_volcand_beta` (β de 2a sans l'élection évaluée, "
          "fenêtre 8, quatre horizons, V nulle au plancher) : "
          + ", ".join(f"{e} : {k:+.2f}" for e, k in kb.items()) + ".", ""]
    # 5. backtest
    s = metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model")
                       & pl.col("metric").is_in(list(bl.METRICS)))
    wide = s.pivot(on="variant", index=["election", "horizon", "metric"], values="value")
    L += ["## 5. Backtest apparié (tirages floor, même graine)", ""]
    ref_m = pl.concat([pl.read_parquet(p) for p in sorted(REF_BACKTEST.glob("metrics*.parquet"))],
                      how="diagonal_relaxed").filter(pl.col("level") == "summary")
    j = s.join(ref_m.select("election", "horizon", "variant", "model", "metric",
                            pl.col("value").alias("ref")),
               on=["election", "horizon", "variant", "model", "metric"], how="inner")
    L += [(f"Contrôle : `lists` et `floor` reproduisent runs/2026-10-20-backtest-lists sur "
          f"{j.height} valeurs, écart absolu max {(j['value'] - j['ref']).abs().max():.2e}."), ""]
    adm = {}
    for v, base in (("lists_volcand", "lists"), ("volcand", "floor"), (SENSITIVITY, "lists")):
        md = bl.comparison_md(wide, v, base)
        crit, amd = bl.admission(wide, v, base)
        adm[v] = crit
        keep = []
        skip = False
        for line in md:  # tableaux principaux seulement
            if line.startswith("### "):
                skip = line[4:] not in ("log_score_pair", "crps", "cov50", "cov90")
            if not skip:
                keep.append(line)
        L += keep + amd + [""]
    (out / "admission.json").write_text(json.dumps(adm, indent=2, default=str))
    # échelles et candidats volatils
    sm = scales.filter(pl.col("variant") == "lists_volcand")
    L += ["### Échelles appliquées (`lists_volcand`)", "",
          f"s ≠ 1 dans {int((sm['s'] != 1).sum())} cas sur {sm.height} ; κ par pli : "
          + ", ".join(f"{e} : {kap[f'lists:{e}']['kappa']:+.2f}" for e in ELECTIONS) + ".", ""]
    vt = coverage_by_volatility(out, scales, metrics, ["lists", "lists_volcand", SENSITIVITY])
    L += ["### Candidats par tiers de V / V_ref (tous horizons, toutes élections, non pondéré)", "",
          _md(vt.rename({"len": "n"}), {"crps": "{:.2f}", "in50": "{:.2f}", "in90": "{:.2f}"}), ""]
    fc = forecasts.filter((pl.col("model") == "model") & (pl.col("level") == "candidate")
                          & pl.col("variant").is_in(["lists", "lists_volcand", SENSITIVITY]))
    sens_s = pl.read_parquet(out / f"scales_{SENSITIVITY}.parquet")
    L += ["### Candidats volatils", "",
          "Résultat, médiane [q05–q95] et CRPS par variante ; V/V_ref et s de la sensibilité.", "",
          "| candidat | horizon | résultat | V/V_ref | s (β) | lists | lists_volcand | lists_volcand_beta |",
          "|---|---|---|---|---|---|---|---|"]
    for u, e in FOCUS:
        for h in HORIZONS:
            cells = []
            for v in ("lists", "lists_volcand", SENSITIVITY):
                r = fc.filter((pl.col("variant") == v) & (pl.col("election") == e)
                              & (pl.col("horizon") == h) & (pl.col("unit") == u))
                if r.is_empty():
                    cells.append("–")
                    continue
                r = r.row(0, named=True)
                cells.append(f"{r['median']:.1f} [{r['q05']:.1f}–{r['q95']:.1f}] CRPS {r['crps']:.2f}")
            rr = sm.filter((pl.col("election") == e) & (pl.col("horizon") == h) & (pl.col("unit") == u))
            ratio = rr["ratio"][0] if rr.height else math.nan
            ss = sens_s.filter((pl.col("election") == e) & (pl.col("horizon") == h)
                               & (pl.col("unit") == u))
            sb = ss["s"][0] if ss.height else math.nan
            off = fc.filter((pl.col("election") == e) & (pl.col("unit") == u))["official"]
            L.append(f"| {u} {e} | {h} | {off[0] if off.len() else math.nan:.1f} | "
                     f"{ratio:.2f} | {sb:.2f} | " + " | ".join(cells) + " |")
    verdict = {v: c["criteria"]["c1_ref"] and c["criteria"]["c2"]
               and c["criteria"]["c3_cand_ref"] for v, c in adm.items()}
    L += ["### Verdict", "",
          ("Règle de docs/07 (comme pour `lists`) : `lists_volcand` "
          f"**{'admise' if verdict['lists_volcand'] else 'non admise'}**, `volcand` "
          f"**{'admise' if verdict['volcand'] else 'non admise'}** ; la sensibilité "
          f"`{SENSITIVITY}` (κ hors validation croisée) "
          f"{'remplirait' if verdict[SENSITIVITY] else 'ne remplirait pas'} la règle."), ""]
    # 6. 2027
    L += ["", f"## 6. Indice actuel des candidats 2027 ({cur['asof']}, {cur['horizon']} jours)", "",
          (f"Configuration connue à la date, {cur['n_polls']} sondages dans la fenêtre de 8 "
          f"semaines ; V_ref sur 2002-2022 ; κ (toutes élections, lists) = "
          f"{cur['kappa_all']:+.2f}, κ β = {cur['kappa_beta']:+.2f}. V_ref nulle (plus de la "
          "moitié des candidats de la classe sans mouvement au-delà du bruit à cet horizon) : "
          "rapport indéfini, s = 1."), "",
          "| candidat | bloc | part 30 j | V (4 sem.) | V (8 sem.) | V (12 sem.) | V_ref | V/V_ref | s (κ retenu) | s (κ β) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cur["candidates"], key=lambda x: -x["share30"]):
        ws = [cur["windows"][str(w)].get(c["unit"]) for w in vo.WINDOWS]
        f = (lambda x: "–" if x is None or not math.isfinite(x) else f"{x:.3f}")
        L.append(f"| {c['unit']} | {c['bloc']} | {c['share30']:.1f} | " + " | ".join(f(x) for x in ws)
                 + f" | {f(c['V_ref'])} | {f(c['ratio'])} | {c['s_kappa_all']:.2f} | "
                 f"{c['s_kappa_beta']:.2f} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# --- commandes


def cmd_study(out: Path, reg) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = {"reproduction": reproduce_beta(reg)}
    print("reproduction", res["reproduction"], flush=True)
    hs = sorted(set(FINE) | set(HORIZONS))
    vols = vol_tables(reg, hs)
    vols.write_parquet(out / "candidate_volatility.parquet")
    moves = candidate_moves(reg, hs)
    res["moves"], dm = move_relations(moves, vols)
    dm.write_parquet(out / "moves.parquet")
    res["z"], dz = z_relations(z_scores(REF_BACKTEST), vols)
    dz.write_parquet(out / "z.parquet")
    jw = jw_party()
    jw.write_parquet(out / "jw_party.parquet")
    res["jw"] = jw_relation(jw)
    (out / "relations.json").write_text(json.dumps(res, indent=2, default=str))
    for k in ("moves", "z", "jw"):
        print(pl.DataFrame([{a: b for a, b in r.items() if a != "loo_beta"} for r in res[k]]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "study", "kappa", "score", "current", "report"])
    ap.add_argument("--out", type=Path,
                    default=paths.RUNS / f"{dt.date.today().isoformat()}-volatility-candidate")
    ap.add_argument("--fits-from", type=Path, default=DEFAULT_FITS)
    args = ap.parse_args()
    reg = default_registry()
    if args.cmd in ("all", "study"):
        cmd_study(args.out, reg)
    if args.cmd in ("all", "kappa"):
        fold_models(args.out, reg)
    if args.cmd in ("all", "score"):
        cmd_score(args.out, args.fits_from, reg)
    if args.cmd in ("all", "current"):
        current_2027(args.out, fold_models(args.out, reg), reg)
    if args.cmd in ("all", "report"):
        write_report(args.out, reg)


if __name__ == "__main__":
    main()
