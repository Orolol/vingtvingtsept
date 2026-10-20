"""Étude des rebonds d'annonce et de primaire (« convention bounce » de Silver).

    uv run python studies/bounce.py [--out runs/<date>-bounce] [--n-placebo 400]

Catalogue : data/registry/events.yaml (vvs.model.bounce.load_events). Pour chaque événement :

1. trajectoire de la part de l'unité suivie (candidat ; vainqueur de primaire peu testé
   avant : son bloc ; retrait : premier bénéficiaire), de son bloc et de ses voisins (autres
   candidats du bloc, bénéficiaires), de 8 semaines avant à 12 semaines après, alignée sur la
   date, sondages bruts (hypothèse la plus proche de la configuration connue) ;
2. amplitude (pic des moyennes sur [0, 2), [2, 4), [4, 6) semaines moins le niveau des 8
   semaines d'avant), délai du pic, part du saut conservée à 4, 8, 12 semaines (moyennes
   sur ± 2 semaines) et au vote (résultat officiel) ;
3. forme commune par type : y = pre + J·(r + (1 − r)·exp(−s/T)), (r, T) partagés,
   intervalles par rééchantillonnage des événements, stabilité en retirant une élection,
   sensibilité à la coupure au prochain événement visant la même unité ;
4. placebos : dates tirées au hasard pour les candidats suffisamment sondés, loin de tout
   événement catalogué de leur bloc ; mêmes statistiques. La question est de savoir si la
   retombée après un saut dépasse le simple retour à la moyenne d'une série bruitée : on
   compare les événements aux placebos qui ont un saut de même taille, et la forme (r, T)
   estimée sur des jeux de placebos sélectionnés sur leur saut ;
5. nouveaux entrants (Montalvo et al. 2019) : erreur log(résultat) − log(moyenne des sondages)
   à 180, 120, 90, 60, 30 et 7 jours pour Bayrou 2007, Macron 2017, Hamon 2017, Zemmour 2022,
   rapportée à l'écart type de η à l'horizon, contre les autres candidats ;
6. 2027 : saut observé au 2026-10-20 et dérive impliquée par la forme des primaires pour
   les événements 2026.

Critère, fixé avant de lire les résultats : une forme « tient » pour un type si (a) au moins
cinq événements ont assez de sondages, (b) l'intervalle à 90 % de r par rééchantillonnage des
événements exclut 1 (une partie du saut se résorbe), (c) la retombée moyenne à 8 semaines est
plus forte que celle des placebos de saut comparable (au-delà du 90e centile de la
distribution placebo), et (d) le signe de r − 1 ne change dans aucun retrait d'élection.
Sinon, chaque cas est traité comme singulier et rien n'entre dans la projection.
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
from vvs.model import bounce as bo
from vvs.model.plots import GRID, INK, MUTED, POINTS, RESULT, SERIES, _style, display_name
from vvs.registry import default_registry

ELECTIONS = (2007, 2012, 2017, 2022)
CHECKPOINTS = (4, 8, 12)
NEW_ENTRANT_HORIZONS = (180, 120, 90, 60, 30, 7)
PLACEBO_MIN_POLLS = 20
PLACEBO_EXCLUSION_WEEKS = 6
ASOF_2027 = dt.date(2026, 10, 20)
VOTE_2027 = dt.date(2027, 4, 18)


# --- résultats officiels


def official_log_share(reg, election: int, unit: str, kind: str) -> float | None:
    e = reg.elections[election]
    if not e.rounds or not e.round_(1).results:
        return None
    res = {r.candidate_id: r.share for r in e.round_(1).results}
    tot = sum(res.values())
    if kind == "candidate":
        return math.log(res[unit] / tot) if unit in res else None
    x = sum(v for c, v in res.items() if reg.bloc_of(c, election) == unit)
    return math.log(x / tot) if x > 0 else None


# --- statistiques d'un événement


def event_stats(s: bo.Series, vote_week: float, result: float | None) -> dict:
    pre = s.y[s.t < 0]
    row = {"n_pre": len(pre), "n_post": s.n_post}
    if len(pre) == 0:
        return row
    p = float(pre.mean())
    bins = [(0, 2), (2, 4), (4, 6)]
    means = [(lo, s.window(lo, hi)) for lo, hi in bins]
    means = [(lo + 1, float(x.mean())) for lo, x in means if len(x)]
    row |= {"pre": p, "pre_pts": 100 * math.exp(p)}
    if not means:
        return row
    # pic : le plus grand écart au niveau d'avant, dans le sens du premier bin
    sign = 1.0 if means[0][1] >= p else -1.0
    delay, peak = max(means, key=lambda m: sign * (m[1] - p))
    jump = peak - p
    row |= {
        "jump": jump,
        "jump_pts": 100 * (math.exp(peak) - math.exp(p)),
        "peak_pts": 100 * math.exp(peak),
        "delay_weeks": delay,
    }
    for c in CHECKPOINTS:
        x = s.window(c - 2, c + 2)
        if len(x):
            row[f"level_{c}"] = float(x.mean())
            row[f"retained_{c}"] = (float(x.mean()) - p) / jump if abs(jump) > 1e-9 else None
    if result is not None:
        row["result_pts"] = 100 * math.exp(result)
        row["retained_vote"] = (result - p) / jump if abs(jump) > 1e-9 else None
        row["vote_weeks"] = vote_week
    return row


def neighbours(book: bo.PollBook, ev: bo.Event) -> list[str]:
    """Bénéficiaires et autres candidats du bloc présents dans la configuration un mois après."""
    after = [p for p in book.polls if p["fieldwork_end"] >= ev.date + dt.timedelta(days=21)]
    before = [p for p in book.polls if p["fieldwork_end"] < ev.date]
    cfg = set(after[0]["config"] if after else []) | set(before[-1]["config"] if before else [])
    out = list(ev.beneficiaries)
    for c in sorted(cfg):
        if c not in out and c not in (ev.candidate, ev.focal) and book.bloc_of(c) == ev.bloc:
            out.append(c)
    return out[:4]


# --- placebos


def full_series(book: bo.PollBook, unit: str, kind: str = "candidate") -> dict:
    """Valeurs d'une unité sur toute la campagne (indépendantes de la date de l'événement)."""
    rows = []
    for p in book.polls:
        v = book.value(p, unit, kind)
        if v is not None:
            rows.append((p["fieldwork_start"], p["fieldwork_end"], p["mid"], v))
    return {
        "start": np.array([r[0] for r in rows]),
        "end": np.array([r[1] for r in rows]),
        "mid": np.array([r[2] for r in rows]),
        "y": np.array([r[3] for r in rows]),
    }


def slice_series(fs: dict, date: dt.date, unit: str) -> bo.Series:
    keep = ~((fs["start"] < date) & (fs["end"] >= date))
    t = np.array([(m - date).days / 7.0 for m in fs["mid"]])
    t = np.where(fs["end"] < date, np.minimum(t, -1e-6), t)
    keep &= (t >= -bo.BEFORE_WEEKS) & (t <= bo.AFTER_WEEKS)
    return bo.Series(t[keep], fs["y"][keep], unit, "candidate")


def placebos(events, reg, n: int, seed: int = 0) -> list[dict]:
    """Dates au hasard pour les candidats sondés au moins `PLACEBO_MIN_POLLS` fois, à plus de
    `PLACEBO_EXCLUSION_WEEKS` semaines de tout événement visant leur bloc."""
    rng = np.random.default_rng(seed)
    pool = []
    for e in ELECTIONS:
        book = bo.poll_book(e, reg)
        counts: dict[str, int] = {}
        for p in book.polls:
            for c in {c for h in p["hyps"].values() for c in h}:
                counts[c] = counts.get(c, 0) + 1
        vote = reg.elections[e].round_(1).date
        for c, k in counts.items():
            if k < PLACEBO_MIN_POLLS:
                continue
            fs = full_series(book, c)
            b = book.bloc_of(c)
            ev_dates = [x.date for x in events if x.election == e
                        and (x.bloc == b or x.candidate == c or x.focal == c)]
            pool.append((e, c, fs, ev_dates, vote, official_log_share(reg, e, c, "candidate")))
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        e, c, fs, ev_dates, vote, result = pool[rng.integers(len(pool))]
        lo = min(fs["mid"]) + dt.timedelta(weeks=bo.BEFORE_WEEKS)
        hi = vote - dt.timedelta(weeks=2)
        if hi <= lo:
            continue
        d = lo + dt.timedelta(days=int(rng.integers(0, (hi - lo).days)))
        if any(abs((d - x).days) < 7 * PLACEBO_EXCLUSION_WEEKS for x in ev_dates):
            continue
        s = slice_series(fs, d, c)
        st = event_stats(s, (vote - d).days / 7, result)
        if st.get("n_pre", 0) < bo.PRE_MIN_POLLS or s.n_post < bo.POST_MIN_POLLS \
                or "level_8" not in st:
            continue
        out.append({"election": e, "unit": c, "date": d, "series": s} | st)
    return out


def placebo_shapes(pl_rows: list[dict], n_events: int, min_jump: float, n_sets: int,
                   seed: int = 1) -> np.ndarray:
    """r estimé sur `n_sets` jeux de `n_events` placebos dont le saut dépasse `min_jump`."""
    rng = np.random.default_rng(seed)
    cand = [r for r in pl_rows if r.get("jump") is not None and r["jump"] >= min_jump]
    if len(cand) < n_events:
        return np.array([])
    rs = []
    for _ in range(n_sets):
        idx = rng.choice(len(cand), n_events, replace=False)
        sh = bo.fit_shape({str(i): cand[i]["series"] for i in idx}, n_boot=0)
        if sh is not None:
            rs.append(sh.r)
    return np.array(rs)


# --- nouveaux entrants


def entrant_errors(reg) -> pl.DataFrame:
    """log(résultat) − log(moyenne 14 j) par candidat officiel et horizon, en unités de l'écart
    type de η (modèle d'erreur sans l'élection)."""
    from vvs.eval.baselines import poll_mean
    from vvs.eval.configurations import known_configuration, official_map
    from vvs.model.error import ErrorModel

    entrants = {(2007, "bayrou_francois"), (2017, "macron_emmanuel"), (2017, "hamon_benoit"),
                (2022, "zemmour_eric")}
    rows = []
    for e in ELECTIONS:
        em = ErrorModel.fit(exclude=[e], registry=reg)
        vote = reg.elections[e].round_(1).date
        res = {r.candidate_id: r.share for r in reg.elections[e].round_(1).results}
        tot = sum(res.values())
        for h in NEW_ENTRANT_HORIZONS:
            asof = vote - dt.timedelta(days=h)
            try:
                known = known_configuration(e, asof, reg)
            except ValueError:
                continue
            pf = poll_mean(e, asof, known.candidates, days=14, registry=reg)
            if pf.n_polls == 0:
                continue
            m = official_map(known.candidates, e, reg)
            sh = m.apply(pf.shares[None, :])[0]
            blocs = [reg.bloc_of(c, e) for c in m.official]
            vb, vc = em.variances(pf.age_days, blocs, np.maximum(sh, 1e-3))
            sd = np.sqrt(vb + vc)
            for i, c in enumerate(m.official):
                if not m.forecast[i] or sh[i] <= 0 or res[c] / tot < 0.01:
                    continue
                err = math.log(res[c] / tot) - math.log(sh[i])
                rows.append({"election": e, "horizon": h, "candidate": c,
                             "new_entrant": (e, c) in entrants, "poll_pts": 100 * sh[i],
                             "result_pts": 100 * res[c] / tot, "log_error": err,
                             "z": err / sd[i]})
    return pl.DataFrame(rows)


# --- figures


def plot_events(cases: list[dict], path: Path, title: str, shape: bo.Shape | None) -> None:
    n = len(cases)
    if n == 0:
        return
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for ax, c in zip(axes.flat, cases):
        s: bo.Series = c["series"]
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.scatter(s.t, 100 * np.exp(s.y), s=9, color=SERIES, alpha=0.8, zorder=3)
        if c.get("bloc_series") is not None and s.kind == "candidate":
            b = c["bloc_series"]
            ax.scatter(b.t, 100 * np.exp(b.y), s=5, color=POINTS, alpha=0.5, marker="s")
        for nb in c.get("neighbour_series", []):
            ax.plot(nb.t[np.argsort(nb.t)], 100 * np.exp(nb.y[np.argsort(nb.t)]),
                    color=GRID, linewidth=0.8)
        if shape is not None and c["id"] in shape.jumps:
            t = np.linspace(-bo.BEFORE_WEEKS, bo.AFTER_WEEKS, 200)
            ax.plot(t, 100 * np.exp(bo.trajectory(t, shape.pres[c["id"]], shape.jumps[c["id"]],
                                                  shape.r, shape.T)), color=INK, linewidth=1.2)
        if c.get("result_pts") is not None:
            ax.axhline(c["result_pts"], color=RESULT, linewidth=1.0)
        ax.set_xlim(-bo.BEFORE_WEEKS - 0.5, bo.AFTER_WEEKS + 0.5)
        unit = display_name(s.unit) if s.kind == "candidate" else f"bloc {s.unit}"
        ax.set_title(f"{c['id']}\n{unit}", fontsize=7.5, loc="left", color=INK)
        _style(ax)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title + " (points ; bleu : unité suivie, gris : bloc, orange : résultat, "
                 "noir : forme commune)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_aligned(by_type: dict[str, list[dict]], shapes: dict, pl_rows: list[dict],
                 path: Path) -> None:
    types = [t for t in bo.EVENT_TYPES if by_type.get(t)]
    fig, axes = plt.subplots(1, len(types) + 1, figsize=(3.6 * (len(types) + 1), 3.3))
    for ax, t in zip(axes, types):
        for c in by_type[t]:
            s, st = c["series"], c["stats"]
            if "pre" not in st or "jump" not in st or abs(st["jump"]) < 1e-6:
                continue
            o = np.argsort(s.t)
            ax.plot(s.t[o], (s.y[o] - st["pre"]) / abs(st["jump"]), color=SERIES, alpha=0.5,
                    linewidth=0.9, marker="o", markersize=2)
        sh = shapes.get(t)
        if sh is not None:
            x = np.linspace(0, bo.AFTER_WEEKS, 100)
            ax.plot(x, bo.decay(x, sh.r, sh.T), color=INK, linewidth=1.6)
        ax.axhline(0, color=MUTED, linewidth=0.7)
        ax.axhline(1, color=MUTED, linewidth=0.7, linestyle=":")
        ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=":")
        ax.set_ylim(-3, 3)
        ax.set_title(f"{t} : (y − pre) / |saut|", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("semaines depuis l'événement", fontsize=8, color=MUTED)
        _style(ax)
    ax = axes[-1]
    j = np.array([r["jump"] for r in pl_rows if "jump" in r and "level_8" in r])
    r8 = np.array([r["level_8"] - r["pre"] - r["jump"] for r in pl_rows if "jump" in r and "level_8" in r])
    ax.scatter(j, r8, s=5, color=POINTS, alpha=0.5, label="placebos")
    for t, col in (("primary", RESULT), ("declaration", SERIES)):
        pts = [(c["stats"]["jump"], c["stats"]["level_8"] - c["stats"]["pre"] - c["stats"]["jump"])
               for c in by_type.get(t, []) if "level_8" in c["stats"] and "jump" in c["stats"]]
        if pts:
            ax.scatter(*zip(*pts), s=22, color=col, label=t, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.set_xlabel("saut (log-part)", fontsize=8, color=MUTED)
    ax.set_title("retombée à 8 semaines contre saut", fontsize=8, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- main


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{d}f}"


def main(out: Path, n_placebo: int, n_boot: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    reg = default_registry()
    events = bo.load_events(registry=reg)
    cases = []
    for ev in events:
        book = bo.poll_book(ev.election, reg)
        s = bo.focal_series(book, ev)
        vote = reg.elections[ev.election].round_(1).date
        result = official_log_share(reg, ev.election, s.unit, s.kind)
        st = event_stats(s, (vote - ev.date).days / 7, result)
        bloc_s = book.series(ev.bloc, "bloc", ev.date, pre_end=ev.first_round)
        bst = event_stats(bloc_s, (vote - ev.date).days / 7,
                          official_log_share(reg, ev.election, ev.bloc, "bloc"))
        nbs = [book.series(c, "candidate", ev.date, pre_end=ev.first_round)
               for c in neighbours(book, ev)]
        cases.append({"id": ev.id, "event": ev, "series": s, "stats": st, "bloc_series": bloc_s,
                      "bloc_stats": bst, "neighbour_series": nbs,
                      "result_pts": st.get("result_pts")})

    # tableau par événement
    rows = []
    for c in cases:
        ev, st, bst = c["event"], c["stats"], c["bloc_stats"]
        nb_txt = []
        for nb in c["neighbour_series"]:
            a, b = nb.window(-bo.BEFORE_WEEKS, 0), nb.window(0, 4)
            if len(a) and len(b):
                nb_txt.append(f"{nb.unit}:{100 * (math.exp(b.mean()) - math.exp(a.mean())):+.1f}")
        rows.append({
            "id": ev.id, "election": ev.election, "type": ev.type, "date": ev.date,
            "unit": c["series"].unit, "kind": c["series"].kind,
            **{k: st.get(k) for k in ("n_pre", "n_post", "pre_pts", "peak_pts", "jump_pts",
                                      "jump", "delay_weeks", "retained_4", "retained_8",
                                      "retained_12", "result_pts", "retained_vote")},
            "bloc_jump_pts": bst.get("jump_pts"), "bloc_retained_8": bst.get("retained_8"),
            "bloc_retained_vote": bst.get("retained_vote"),
            "neighbours_0_4w": " ".join(nb_txt),
        })
    table = pl.DataFrame(rows, infer_schema_length=None)
    table.write_parquet(out / "events.parquet")

    # formes par type
    calib = [c for c in cases if c["event"].election in bo.CALIBRATION_ELECTIONS]
    by_type = {t: [c for c in calib if c["event"].type == t] for t in bo.EVENT_TYPES}
    shapes, shapes_trunc, loo = {}, {}, {}
    for t in bo.EVENT_TYPES:
        ser = {c["id"]: c["series"] for c in by_type[t]}
        shapes[t] = bo.fit_shape(ser, t, n_boot=n_boot)
        shapes_trunc[t] = bo.fit_shape(
            bo.type_series(events, t, bo.CALIBRATION_ELECTIONS, truncate=True, registry=reg),
            t, n_boot=n_boot)
        loo[t] = {}
        for e in ELECTIONS:
            sh = bo.fit_shape({k: v for k, v in ser.items()
                               if next(c for c in calib if c["id"] == k)["event"].election != e},
                              t, n_boot=0)
            loo[t][e] = None if sh is None else {"r": sh.r, "T": sh.T, "n": len(sh.events)}

    # placebos
    pl_rows = placebos(events, reg, n_placebo)
    pl_table = pl.DataFrame([{k: v for k, v in r.items() if k != "series"} for r in pl_rows],
                            infer_schema_length=None)
    pl_table.write_parquet(out / "placebos.parquet")

    def fall8(st):
        return st["level_8"] - st["pre"] - st["jump"]

    pj = np.array([r["jump"] for r in pl_rows])
    pf = np.array([fall8(r) for r in pl_rows])
    slope, icpt = np.polyfit(pj, pf, 1)
    verdict = {}
    for t in bo.EVENT_TYPES:
        sts = [c["stats"] for c in by_type[t] if "level_8" in c["stats"] and "jump" in c["stats"]]
        if not sts:
            continue
        ej = np.array([s["jump"] for s in sts])
        ef = np.array([fall8(s) for s in sts])
        # excès de retombée par rapport au retour à la moyenne des placebos (même saut)
        excess = ef - (icpt + slope * ej)
        # distribution placebo de l'excès moyen pour des jeux de même taille
        res_pl = pf - (icpt + slope * pj)
        rng = np.random.default_rng(3)
        null = np.array([res_pl[rng.choice(len(res_pl), len(sts), replace=False)].mean()
                         for _ in range(4000)])
        sh = shapes[t]
        r_pl = placebo_shapes(pl_rows, len(sh.events) if sh else len(sts),
                              min_jump=float(np.median(np.abs(ej))) * 0.5, n_sets=200) \
            if t in ("primary", "declaration") else np.array([])
        iv = sh.interval() if sh is not None and len(sh.r_draws) else None
        # part conservée : événements contre placebos de saut comparable (même signe, au moins
        # la moitié du saut médian du type)
        thr = 0.5 * float(np.median(np.abs(ej)))
        sgn = 1.0 if np.median(ej) >= 0 else -1.0
        comp = [r for r in pl_rows if sgn * r["jump"] >= thr]
        ret = {}
        for key in ("retained_8", "retained_vote"):
            ev_v = [s[key] for s in sts if s.get(key) is not None]
            pl_v = [r[key] for r in comp if r.get(key) is not None]
            ret[key] = {
                "events_median": float(np.median(ev_v)) if ev_v else None, "n_events": len(ev_v),
                "placebo_median": float(np.median(pl_v)) if pl_v else None, "n_placebo": len(pl_v),
                "placebo_q10_q90": [float(np.quantile(pl_v, 0.1)), float(np.quantile(pl_v, 0.9))]
                if pl_v else None,
            }
        verdict[t] = {
            "n_events_fit": len(sh.events) if sh else 0,
            "r": sh.r if sh else None, "T": sh.T if sh else None,
            "r_interval90": iv["r"] if iv else None, "T_interval90": iv["T"] if iv else None,
            "trunc_r": shapes_trunc[t].r if shapes_trunc[t] else None,
            "trunc_T": shapes_trunc[t].T if shapes_trunc[t] else None,
            "mean_jump": float(ej.mean()), "mean_fall8": float(ef.mean()),
            "mean_excess_fall8": float(excess.mean()),
            "placebo_excess_q10": float(np.quantile(null, 0.10)),
            "placebo_excess_pctile": float((null <= excess.mean()).mean()),
            "placebo_r_selected_median": float(np.median(r_pl)) if len(r_pl) else None,
            "placebo_r_selected_q10_q90": [float(np.quantile(r_pl, 0.1)),
                                           float(np.quantile(r_pl, 0.9))] if len(r_pl) else None,
            "loo": loo[t],
            "retention_vs_placebo": ret,
        }
        v = verdict[t]
        signs = [np.sign(x["r"] - 1) for x in loo[t].values() if x is not None]
        v["criteria"] = {
            "a_five_events": v["n_events_fit"] >= 5,
            "b_r_excludes_1": bool(iv is not None and iv["r"][2] < 0.95),
            "c_beyond_placebo": bool(v["placebo_excess_pctile"] <= 0.10) if np.mean(ej) > 0
            else bool(v["placebo_excess_pctile"] >= 0.90),
            "d_loo_stable": bool(len(signs) > 0 and len(set(signs)) == 1),
        }
        v["holds"] = all(v["criteria"].values())

    shapes_json = {t: (s.to_dict() if s else None) for t, s in shapes.items()}
    (out / "shapes.json").write_text(json.dumps(
        {"shapes": shapes_json,
         "shapes_truncated": {t: (s.to_dict() if s else None) for t, s in shapes_trunc.items()},
         "verdict": verdict, "placebo_regression": {"slope": slope, "intercept": icpt,
                                                    "n": len(pl_rows)}},
        indent=2, default=str, ensure_ascii=False))

    # nouveaux entrants
    ent = entrant_errors(reg)
    ent.write_parquet(out / "new_entrants.parquet")

    # 2027
    rows27 = []
    prim = bo.shape_for("primary", None, bo.BounceSpec(n_boot=n_boot), events, reg)
    book27 = bo.poll_book(2027, reg)
    rng = np.random.default_rng(0)
    for ev in events:
        if ev.election != 2027:
            continue
        obs = bo.observed_jump(book27, ev, ASOF_2027)
        s = bo.focal_series(book27, ev, after=60)
        row = {"id": ev.id, "type": ev.type, "unit": s.unit, "kind": s.kind,
               "n_pre": s.n_pre, "n_post": s.n_post}
        if obs is not None:
            row |= {"D": obs["D"], "age_weeks": obs["age_weeks"], "n_now": obs["n_now"]}
            pre = s.y[s.t < 0].mean()
            row["pre_pts"] = 100 * math.exp(pre)
            row["now_pts"] = 100 * math.exp(pre + obs["D"])
            if prim is not None:
                k = rng.integers(0, len(prim.r_draws), 20000)
                D = obs["D"] + obs["se"] * rng.standard_normal(20000)
                mu = D * bo.drift_factor(obs["age_weeks"], (VOTE_2027 - ASOF_2027).days / 7,
                                         prim.r_draws[k], prim.T_draws[k])
                row |= {"mu_median": float(np.median(mu)), "mu_q05": float(np.quantile(mu, .05)),
                        "mu_q95": float(np.quantile(mu, .95)),
                        "vote_pts_if_primary_shape": 100 * math.exp(pre + obs["D"] + np.median(mu))}
        rows27.append(row)
    t27 = pl.DataFrame(rows27, infer_schema_length=None)
    t27.write_parquet(out / "events_2027.parquet")

    # figures
    for t in bo.EVENT_TYPES:
        plot_events([c for c in cases if c["event"].type == t], out / f"events_{t}.png",
                    f"Événements « {t} »", shapes.get(t))
    plot_aligned(by_type, shapes, pl_rows, out / "aligned.png")

    # rapport
    md = [f"# Étude des rebonds ({dt.date.today().isoformat()})", ""]
    md.append("## Événements (unité suivie, log-parts sur sondages bruts)\n")
    md.append("| id | type | unité | n avant / après | avant | pic | saut (pts) | délai (sem.) "
              "| conservé 4 / 8 / 12 sem. | résultat | conservé au vote | bloc : saut, conservé 8 sem. / vote | voisins 0-4 sem. |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['id']} | {r['type']} | {r['unit']}{' (bloc)' if r['kind'] == 'bloc' else ''} "
            f"| {r['n_pre']} / {r['n_post']} | {fmt(r['pre_pts'], 1)} | {fmt(r['peak_pts'], 1)} "
            f"| {fmt(r['jump_pts'], 1)} | {fmt(r['delay_weeks'], 0)} "
            f"| {fmt(r['retained_4'])} / {fmt(r['retained_8'])} / {fmt(r['retained_12'])} "
            f"| {fmt(r['result_pts'], 1)} | {fmt(r['retained_vote'])} "
            f"| {fmt(r['bloc_jump_pts'], 1)}, {fmt(r['bloc_retained_8'])} / {fmt(r['bloc_retained_vote'])} "
            f"| {r['neighbours_0_4w']} |")
    md.append("\n## Forme commune par type (2002-2022)\n")
    md.append("| type | événements | r [90 %] | T semaines [90 %] | coupé au prochain événement r, T "
              "| saut moyen | retombée 8 sem. | excès sur placebo (centile) | r placebos sélectionnés [10-90 %] "
              "| r sans 2007 / 2012 / 2017 / 2022 | tient ? |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, v in verdict.items():
        loo_txt = " / ".join(fmt(v["loo"][e]["r"]) if v["loo"].get(e) else "–" for e in ELECTIONS)
        md.append(
            f"| {t} | {v['n_events_fit']} | {fmt(v['r'])} {v['r_interval90'] and '[' + fmt(v['r_interval90'][0]) + ', ' + fmt(v['r_interval90'][2]) + ']'} "
            f"| {fmt(v['T'], 1)} {v['T_interval90'] and '[' + fmt(v['T_interval90'][0], 1) + ', ' + fmt(v['T_interval90'][2], 1) + ']'} "
            f"| {fmt(v['trunc_r'])}, {fmt(v['trunc_T'], 1)} | {fmt(v['mean_jump'])} | {fmt(v['mean_fall8'])} "
            f"| {fmt(v['mean_excess_fall8'])} ({fmt(100 * v['placebo_excess_pctile'], 0)} %) "
            f"| {fmt(v['placebo_r_selected_median'])} {v['placebo_r_selected_q10_q90'] and '[' + fmt(v['placebo_r_selected_q10_q90'][0]) + ', ' + fmt(v['placebo_r_selected_q10_q90'][1]) + ']'} "
            f"| {loo_txt} | {'oui' if v['holds'] else 'non'} {v['criteria']} |")
    md.append("\nPart du saut conservée, médianes : événements contre placebos de saut comparable "
              "(même signe, au moins la moitié du saut médian du type).\n")
    md.append("| type | conservé 8 sem. : événements (n) | placebos (n) [10-90 %] | conservé au vote : événements (n) | placebos (n) [10-90 %] |")
    md.append("|---|---|---|---|---|")
    for t, v in verdict.items():
        a, b = v["retention_vs_placebo"]["retained_8"], v["retention_vs_placebo"]["retained_vote"]
        def cell(x, ev):
            if ev:
                return f"{fmt(x['events_median'])} ({x['n_events']})"
            q = x["placebo_q10_q90"]
            return f"{fmt(x['placebo_median'])} ({x['n_placebo']}) [{fmt(q[0]) if q else '–'}, {fmt(q[1]) if q else '–'}]"
        md.append(f"| {t} | {cell(a, True)} | {cell(a, False)} | {cell(b, True)} | {cell(b, False)} |")
    md.append(f"\nPlacebos : {len(pl_rows)} dates ; régression retombée 8 sem. = "
              f"{icpt:.3f} + {slope:.3f} × saut (retour à la moyenne d'une série bruitée).\n")
    md.append("## Nouveaux entrants (log résultat − log moyenne 14 j, en écarts types de η)\n")
    if not ent.is_empty():
        agg = (ent.group_by("new_entrant", "horizon").agg(
            pl.col("z").mean().alias("z_moyen"), pl.col("log_error").mean().alias("err_moy"),
            pl.len().alias("n")).sort("new_entrant", "horizon", descending=[False, True]))
        md.append("| nouvel entrant | horizon | z moyen | erreur log moyenne | n |")
        md.append("|---|---|---|---|---|")
        for r in agg.iter_rows(named=True):
            md.append(f"| {r['new_entrant']} | {r['horizon']} | {r['z_moyen']:+.2f} | "
                      f"{r['err_moy']:+.3f} | {r['n']} |")
        md.append("\n| élection | candidat | " + " | ".join(f"{h} j" for h in NEW_ENTRANT_HORIZONS)
                  + " | résultat |")
        md.append("|---|---|" + "---|" * (len(NEW_ENTRANT_HORIZONS) + 1))
        for (e, c), g in ent.filter(pl.col("new_entrant")).group_by("election", "candidate",
                                                                   maintain_order=True):
            d = {r["horizon"]: r for r in g.iter_rows(named=True)}
            cells = [f"{d[h]['poll_pts']:.1f} (z {d[h]['z']:+.1f})" if h in d else "–"
                     for h in NEW_ENTRANT_HORIZONS]
            md.append(f"| {e} | {c} | " + " | ".join(cells) + f" | {g['result_pts'][0]:.1f} |")
    md.append("\n## 2027 au 2026-10-20 (forme des primaires, Δ jusqu'au 18 avril 2027)\n")
    md.append("| id | unité | n avant / après | avant | maintenant | âge (sem.) | μ médiane [5-95 %] (log) | part au vote si forme primaire |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows27:
        md.append(f"| {r['id']} | {r['unit']}{' (bloc)' if r['kind'] == 'bloc' else ''} "
                  f"| {r['n_pre']} / {r['n_post']} | {fmt(r.get('pre_pts'), 1)} | {fmt(r.get('now_pts'), 1)} "
                  f"| {fmt(r.get('age_weeks'), 1)} | {fmt(r.get('mu_median'), 3)} "
                  f"[{fmt(r.get('mu_q05'), 3)}, {fmt(r.get('mu_q95'), 3)}] "
                  f"| {fmt(r.get('vote_pts_if_primary_shape'), 1)} |")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# --- backtest de la variante `bounce`


BT_ELECTIONS = (2012, 2017, 2022)
BT_HORIZONS = (180, 90, 30, 7)
BT_METRICS = ("log_score_pair", "crps", "cov50", "cov90", "bloc_crps", "brier_qual")


def backtest(out: Path, fits_from: Path) -> None:
    """Score des variantes floor et bounce sur les mêmes tirages θ (liens vers `fits_from`),
    même η (même graine), et tableau d'admission (docs/07-evaluation.md)."""
    from vvs.eval.backtest import asof_for, fit_dir
    from vvs.eval.scoring import score_backtest

    out.mkdir(parents=True, exist_ok=True)
    log = out / "backtest.log"

    def say(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        with log.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    for v in ("floor", "bounce"):
        for e in BT_ELECTIONS:
            for h in BT_HORIZONS:
                src, dst = fit_dir(fits_from, "floor", e, h), fit_dir(out, v, e, h)
                if not (src / "draws.npz").exists():
                    raise FileNotFoundError(f"ajustement floor absent : {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src.resolve())
    say(f"score floor et bounce, tirages de {fits_from}")
    res = score_backtest(out, list(BT_ELECTIONS), list(BT_HORIZONS),
                         variants=["floor", "bounce"])
    say("score terminé")

    # détail des dérives appliquées
    reg = default_registry()
    events = bo.load_events(registry=reg)
    drows = []
    for e in BT_ELECTIONS:
        for h in BT_HORIZONS:
            conf = json.loads((fit_dir(out, "bounce", e, h) / "configuration.json").read_text())
            _, det = bo.candidate_drift(e, asof_for(e, h), h, conf["candidates"], 4000,
                                        np.random.default_rng(0), bo.BOUNCE_VARIANTS["bounce"],
                                        events=events, registry=reg)
            sh = bo.shape_for("primary", e, bo.BOUNCE_VARIANTS["bounce"], events, reg)
            for d in det:
                drows.append({"election": e, "horizon": h, "r": sh.r, "T": sh.T} | d)
    drift_table = pl.DataFrame(drows, infer_schema_length=None)
    if not drift_table.is_empty():
        drift_table.write_parquet(out / "drifts.parquet")

    s = res.metrics.filter((pl.col("level") == "summary") & (pl.col("model") == "model")
                           & pl.col("metric").is_in(BT_METRICS))
    wide = s.pivot(on="variant", index=["election", "horizon", "metric"], values="value")
    wide = wide.with_columns((pl.col("bounce") - pl.col("floor")).alias("diff"))
    wide.write_parquet(out / "comparison.parquet")
    md = [f"# Backtest de la variante `bounce` ({dt.date.today().isoformat()})", "",
          (f"Tirages θ : {fits_from} (variante floor, réutilisés tels quels). Même η "
          "(graine [0, élection, horizon]) ; la dérive est tirée avec une graine propre. "
          "Forme des primaires estimée sans l'élection évaluée."), ""]
    md.append("## Dérives appliquées (log-part du candidat)\n")
    md.append("| élection | horizon | événement | unité | âge (sem.) | D observé | μ médiane [5-95 %] | r, T (LOO) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for d in drows:
        md.append(f"| {d['election']} | {d['horizon']} | {d['event']} | {d['unit']} ({d['kind']}) "
                  f"| {d['age_weeks']:.1f} | {d['D']:+.3f} | {d['mu_median']:+.3f} "
                  f"[{d['mu_q05']:+.3f}, {d['mu_q95']:+.3f}] | {d['r']:.2f}, {d['T']:.1f} |")
    md.append("")
    for m in BT_METRICS:
        md.append(f"### {m}\n")
        md.append("| élection | horizon | floor | bounce | écart |")
        md.append("|---|---|---|---|---|")
        sub = wide.filter(pl.col("metric") == m).sort("election", "horizon",
                                                     descending=[False, True])
        for r in sub.iter_rows(named=True):
            md.append(f"| {r['election']} | {r['horizon']} | {r['floor']:.3f} | {r['bounce']:.3f} "
                      f"| {r['diff']:+.3f} |")
        mean = sub.group_by("horizon").agg(pl.col("floor").mean(), pl.col("bounce").mean(),
                                           pl.col("diff").mean()).sort("horizon", descending=True)
        for r in mean.iter_rows(named=True):
            md.append(f"| **moyenne** | {r['horizon']} | {r['floor']:.3f} | {r['bounce']:.3f} "
                      f"| {r['diff']:+.3f} |")
        md.append("")
    # règle d'admission (180 et 90 jours)
    adm = []
    for h in (180, 90):
        ls = wide.filter((pl.col("metric") == "log_score_pair") & (pl.col("horizon") == h))
        cr = wide.filter((pl.col("metric") == "crps") & (pl.col("horizon") == h))
        c90 = wide.filter((pl.col("metric") == "cov90") & (pl.col("horizon") == h))
        c50 = wide.filter((pl.col("metric") == "cov50") & (pl.col("horizon") == h))
        n_ls = int((ls["diff"] > 1e-9).sum())
        n_cr = int((cr["diff"] < -1e-9).sum())
        adm.append({
            "horizon": h,
            "1_log_score_gain": float(ls["diff"].mean()), "1_crps_gain": float(-cr["diff"].mean()),
            "2_elections_ls_gain": n_ls, "2_elections_crps_gain": n_cr,
            "3_cov90_change": float(c90["diff"].mean()), "3_cov50_change": float(c50["diff"].mean()),
        })
    md.append("## Règle d'admission (docs/07-evaluation.md), horizons 180 et 90\n")
    md.append("| horizon | gain log-score moyen | gain CRPS moyen | élections où le log-score gagne | où le CRPS gagne | Δ couverture 90 % | Δ couverture 50 % |")
    md.append("|---|---|---|---|---|---|---|")
    for a in adm:
        md.append(f"| {a['horizon']} | {a['1_log_score_gain']:+.3f} | {a['1_crps_gain']:+.3f} "
                  f"| {a['2_elections_ls_gain']} / 3 | {a['2_elections_crps_gain']} / 3 "
                  f"| {100 * a['3_cov90_change']:+.1f} pts | {100 * a['3_cov50_change']:+.1f} pts |")
    ok = all(a["1_log_score_gain"] > 0 and a["1_crps_gain"] > 0 and a["2_elections_ls_gain"] >= 2
             and a["2_elections_crps_gain"] >= 2 and a["3_cov90_change"] >= -0.02 for a in adm)
    md.append(f"\nCritères 1 à 3 remplis aux deux horizons : **{'oui' if ok else 'non'}** "
              "(le critère 4, mécanisme documenté, est jugé dans le rapport).")
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    (out / "admission.json").write_text(json.dumps({"rows": adm, "criteria_1_3": ok}, indent=2))
    say("comparaison écrite")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-placebo", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--backtest", action="store_true",
                    help="score floor contre bounce sur les ajustements floor existants")
    ap.add_argument("--fits-from", type=Path, default=paths.RUNS / "2026-10-20-backtest")
    a = ap.parse_args()
    today = dt.date.today().isoformat()
    if a.backtest:
        backtest(a.out or paths.RUNS / f"{today}-backtest-bounce", a.fits_from)
    else:
        main(a.out or paths.RUNS / f"{today}-bounce", a.n_placebo, a.n_boot)
