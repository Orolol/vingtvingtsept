"""Étude : les questions d'anticipation (« qui va gagner ? ») contre les intentions de vote
et contre la probabilité de victoire du backtest.

    uv run python studies/expectations.py [--out runs/<date>-expectations] [--refresh]

Données : `data/curated/poll_expectations.parquet` (`vvs parse expectations`), sondages
curés (`polls.parquet`, `poll_results.parquet`), backtest de la calibration étendue
(`runs/2026-10-20-backtest-lists`, variante `lists`, celle de la publication) et modèle de
second tour (courbe de biais RN par horizon, retenue pour la publication).

Probabilité de victoire du backtest (le backtest ne la publie pas) : tirages de premier tour
du modèle à chaque (élection, horizon), rejoués par `scoring.score_backtest` sans écriture et
capturés ; probabilité de chaque paire de qualifiés ; pour chaque paire de probabilité ≥ 0,5 %,
P(a bat b | paire) par `SecondRoundModel` (calibré sans l'élection évaluée) sur les tirages de
premier tour qui donnent cette paire ; P(victoire c) = Σ P(paire) · P(c gagne | paire). Mis en
cache dans `<out>/backtest_pwin.parquet` (`--refresh` pour recalculer).

Critères fixés avant lecture des résultats (consignés dans le rapport) :

- mesure : pour un `winner`, part normalisée p = part du candidat / part des répondants qui
  nomment un candidat (lecture naïve en probabilité, celle de Graefe) ; mesures d'avant le
  premier tour seulement pour (a), sans les pronostics de duel conditionnels ;
- appariement : à chaque horizon h (180, 90, 30, 7 jours), dernière mesure de chaque institut
  terminée dans [asof − 30 j, asof] (h ≥ 90) ou [asof − 14 j, asof] (h ≤ 30), moyenne des
  instituts ; intentions de vote : moyenne des sondages de premier tour des 14 jours avant
  asof (par sondage, moyenne des hypothèses où le candidat figure) ;
- C1 : le favori des anticipations est-il le vainqueur, comparé au favori du modèle ;
- C2 : log-score du vainqueur, anticipations (p normalisée) contre modèle (P(victoire)), par
  élection et horizon ; « information en plus » seulement si les anticipations font mieux
  sur au moins deux des élections 2012, 2017, 2022 à 90 ou 180 jours et pas moins bien en
  moyenne (règle 2 de docs/07) ;
- C3 (qualification, Ipsos 2022) : Brier des « sera qualifié » contre p_qualify du modèle
  aux horizons 30 et 7 jours ;
- aucune de ces comparaisons n'est un test d'admission : une intégration éventuelle passe
  par le backtest de docs/07.
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
from vvs.registry import default_registry

BACKTEST = paths.RUNS / "2026-10-20-backtest-lists"
VARIANT = "lists"
SIM_2027 = paths.RUNS / "2026-10-20-simulate-subst-v3" / "summary.parquet"
SECOND_ROUND = paths.RUNS / "2026-10-20-second-round-v2" / "metrics.parquet"
ELECTIONS = (2007, 2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
MIN_PAIR = 0.005
N_DRAWS = 4000
N_SECOND = 2000


def window(h: int) -> int:
    return 30 if h >= 90 else 14


# ------------------------------------------------------------------ probabilité du backtest

def backtest_pwin(elections=ELECTIONS, horizons=HORIZONS, seed: int = 0) -> pl.DataFrame:
    """P(qualification) et P(victoire) par candidat, élection et horizon (voir le module)."""
    from vvs.eval import scoring
    from vvs.eval.backtest import asof_for
    from vvs.model.second_round import SecondRoundModel

    captured: dict[tuple[int, int], tuple[list[str], np.ndarray]] = {}
    orig_case, orig_eval = scoring.score_case, scoring.evaluate
    state: dict = {}

    def evaluate(shares, forecast, off):
        if state.get("want"):
            captured[state["key"]] = (list(off.ids), np.asarray(shares))
            state["want"] = False
        return orig_eval(shares, forecast, off)

    def score_case(*args, **kw):
        state["want"] = True  # premier appel d'evaluate : le modèle (« model »)
        return orig_case(*args, **kw)

    scoring.evaluate, scoring.score_case = evaluate, score_case
    try:
        for e in elections:
            for h in horizons:
                state["key"] = (e, h)
                scoring.score_backtest(BACKTEST, [e], [h], variant=VARIANT, n_draws=N_DRAWS,
                                       seed=seed, write=False)
    finally:
        scoring.score_case, scoring.evaluate = orig_case, orig_eval

    reg = default_registry()
    rows = []
    for (e, h), (ids, shares) in sorted(captured.items()):
        asof = asof_for(e, h)
        top2 = np.argsort(-shares, axis=1)[:, :2]
        top2.sort(axis=1)
        keys, counts = np.unique(top2, axis=0, return_counts=True)
        p_pair = counts / counts.sum()
        pairs = [(ids[a], ids[b]) for a, b in keys]
        keep = [k for k, p in enumerate(p_pair) if p >= MIN_PAIR]
        model = SecondRoundModel(reg, bias_by_horizon=True)
        model.fit(e, asof, [pairs[k] for k in keep])
        rng = np.random.default_rng([seed, e, h])
        win = dict.fromkeys(ids, 0.0)
        covered = 0.0
        for k in keep:
            a, b = pairs[k]
            sel = np.all(top2 == keys[k], axis=1)
            fr = {c: shares[sel, i] / 100 for i, c in enumerate(ids)}
            d = model.predict((a, b), fr, h, rng, N_SECOND)
            pa = float(np.mean(d > 50))
            win[a] += p_pair[k] * pa
            win[b] += p_pair[k] * (1 - pa)
            covered += p_pair[k]
        qual = (np.argsort(-shares, axis=1)[:, :2][..., None] == np.arange(len(ids))).any(1).mean(0)
        for i, c in enumerate(ids):
            rows.append({"election": e, "horizon": h, "asof": asof, "candidate_id": c,
                         "p_qualify": float(qual[i]), "p_win": win[c] / covered,
                         "pair_mass": covered})
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ données

def winners() -> dict[int, str]:
    reg = default_registry()
    out = {}
    for e, el in reg.elections.items():
        res = el.round_(2).results if el.round_(2) else []
        if res:
            out[e] = max(res, key=lambda r: r.share).candidate_id
    return out


def qualified() -> dict[int, set[str]]:
    reg = default_registry()
    return {e: {r.candidate_id for r in el.round_(2).results}
            for e, el in reg.elections.items() if el.round_(2) and el.round_(2).results}


def first_round_date(e: int) -> dt.date:
    return default_registry().elections[e].round_(1).date


def iv_mean(polls: pl.DataFrame, res: pl.DataFrame, e: int, d: dt.date, days: int = 14) -> dict:
    """Intentions de vote de premier tour : moyenne des sondages terminés dans [d − days, d]
    (par sondage, moyenne des hypothèses où le candidat figure)."""
    p = polls.filter((pl.col("election") == e) & (pl.col("fieldwork_end") <= d)
                     & (pl.col("fieldwork_end") > d - dt.timedelta(days=days)))
    r = (res.filter(pl.col("poll_id").is_in(p["poll_id"].implode()) & (pl.col("round") == 1)
                    & (pl.col("share_base") == "expressed"))
         .group_by("poll_id", "candidate_id").agg(pl.col("share").mean())
         .group_by("candidate_id").agg(pl.col("share").mean()))
    return dict(zip(r["candidate_id"], r["share"], strict=True))


def normalized(t: pl.DataFrame) -> pl.DataFrame:
    """Part normalisée : parmi les répondants qui nomment un candidat, 100 − `undecided` quand
    la mesure le donne, sinon la somme des candidats relevés."""
    keys = ["election", "pollster", "fieldwork_end", "question_type", "outcome", "pair",
            "file_id", "page", "source_url"]
    named = pl.coalesce(100 - pl.col("undecided").max().over(keys), pl.col("share").sum().over(keys))
    return t.with_columns((pl.col("share") / named).alias("p_norm"))


# ------------------------------------------------------------------ (a) horizons

def horizon_table(exp: pl.DataFrame, bt: pl.DataFrame, polls, res) -> pl.DataFrame:
    from vvs.eval.backtest import asof_for

    win = winners()
    w = normalized(exp.filter((pl.col("question_type") == "winner") & pl.col("pair").is_null()
                              & pl.col("party").is_null()))
    rows = []
    for e in ELECTIONS:
        for h in HORIZONS:
            asof = asof_for(e, h)
            sub = w.filter((pl.col("election") == e) & (pl.col("fieldwork_end") <= asof)
                           & (pl.col("fieldwork_end") > asof - dt.timedelta(days=window(h))))
            if sub.is_empty():
                continue
            last = sub.group_by("pollster").agg(pl.col("fieldwork_end").max())
            sub = sub.join(last, on=["pollster", "fieldwork_end"])
            n_inst = sub["pollster"].n_unique()
            avg = (sub.group_by("candidate_id")
                   .agg((pl.col("p_norm").sum() / n_inst).alias("p"),
                        (pl.col("share").sum() / n_inst).alias("raw"))
                   .sort("p", descending=True))
            fav = avg["candidate_id"][0]
            pw = dict(zip(avg["candidate_id"], avg["p"], strict=True))
            raw = dict(zip(avg["candidate_id"], avg["raw"], strict=True))
            b = bt.filter((pl.col("election") == e) & (pl.col("horizon") == h))
            pm = dict(zip(b["candidate_id"], b["p_win"], strict=True))
            mfav = max(pm, key=pm.get) if pm else None
            iv = iv_mean(polls, res, e, asof)
            ivfav = max(iv, key=iv.get) if iv else None
            ww = win[e]
            p_e = min(max(pw.get(ww, 0.0), 1e-3), 1 - 1e-3)
            p_m = min(max(pm.get(ww, 0.0), 1e-3), 1 - 1e-3) if pm else None
            cands = set(pw) | set(pm)
            brier_e = sum((pw.get(c, 0) - (c == ww)) ** 2 for c in cands)
            brier_m = sum((pm.get(c, 0) - (c == ww)) ** 2 for c in cands) if pm else None
            rows.append({
                "election": e, "horizon": h, "asof": asof,
                "measures": ", ".join(f"{r['pollster']} {r['fieldwork_end']}" for r in
                                      sub.select("pollster", "fieldwork_end").unique()
                                      .sort("fieldwork_end").iter_rows(named=True)),
                "winner": ww, "exp_raw_winner": raw.get(ww, 0.0), "exp_p_winner": pw.get(ww, 0.0),
                "exp_favourite": fav, "iv_winner": iv.get(ww), "iv_leader": ivfav,
                "model_p_winner": pm.get(ww) if pm else None, "model_favourite": mfav,
                "logscore_exp": float(np.log(p_e)),
                "logscore_model": float(np.log(p_m)) if p_m is not None else None,
                "brier_exp": brier_e, "brier_model": brier_m,
            })
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ (b) qualification

def qualification_table(exp: pl.DataFrame, bt: pl.DataFrame) -> pl.DataFrame:
    """Ipsos 2022, « sera qualifié pour le second tour » contre p_qualify du modèle à l'horizon
    le plus proche (même fenêtre que (a))."""
    from vvs.eval.backtest import asof_for

    q = qualified()
    rows = []
    sub = exp.filter(pl.col("outcome") == "qualify")
    for e in sub["election"].unique().to_list():
        for h in HORIZONS:
            asof = asof_for(e, h)
            s = sub.filter((pl.col("election") == e) & (pl.col("fieldwork_end") <= asof)
                           & (pl.col("fieldwork_end") > asof - dt.timedelta(days=window(h))))
            if s.is_empty():
                continue
            last = s.group_by("sponsor").agg(pl.col("fieldwork_end").max())
            s = s.join(last, on=["sponsor", "fieldwork_end"])
            avg = s.group_by("candidate_id").agg(pl.col("share").mean() / 100)
            b = bt.filter((pl.col("election") == e) & (pl.col("horizon") == h))
            pm = dict(zip(b["candidate_id"], b["p_qualify"], strict=True))
            for c, p in zip(avg["candidate_id"], avg["share"], strict=True):
                y = float(c in q[e])
                rows.append({"election": e, "horizon": h, "candidate_id": c,
                             "exp_qualify": p, "model_p_qualify": pm.get(c), "qualified": bool(y),
                             "brier_exp": (p - y) ** 2,
                             "brier_model": (pm[c] - y) ** 2 if c in pm else None})
    return pl.DataFrame(rows)


def leader_dates(exp: pl.DataFrame, polls, res, e: int, pollster: str) -> pl.DataFrame:
    """Série d'un institut : favori des anticipations et meneur des intentions de vote (14 j)."""
    t = normalized(exp.filter((pl.col("election") == e) & (pl.col("pollster") == pollster)
                              & (pl.col("question_type") == "winner") & pl.col("pair").is_null()))
    rows = []
    for d in sorted(t["fieldwork_end"].unique().to_list()):
        x = t.filter(pl.col("fieldwork_end") == d).sort("share", descending=True)
        iv = iv_mean(polls, res, e, d)
        top_iv = sorted(iv.items(), key=lambda kv: -kv[1])[:3]
        rows.append({"fieldwork_end": d,
                     "anticipation": ", ".join(f"{c.split('_')[0]} {s:g}" for c, s in
                                               zip(x["candidate_id"][:3], x["share"][:3],
                                                   strict=True)),
                     "iv_top3": ", ".join(f"{c.split('_')[0]} {s:.1f}" for c, s in top_iv)})
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ entre les deux tours

def between_rounds(exp: pl.DataFrame) -> pl.DataFrame:
    """Pronostics du duel réel publiés entre les deux tours, contre P(vainqueur) du modèle de
    second tour à la veille du vote (courbe de biais RN par horizon, `bias_h`)."""
    sr = (pl.read_parquet(SECOND_ROUND)
          .filter((pl.col("horizon") == "r2_eve") & (pl.col("method") == "bias_h"))
          .group_by("election").agg(pl.col("p_winner").mean(), pl.col("median").mean()))
    pm = {r["election"]: (r["p_winner"], r["median"]) for r in sr.iter_rows(named=True)}
    win = winners()
    t = normalized(exp.filter((pl.col("question_type") == "winner") & pl.col("pair").is_not_null()))
    rows = []
    for (e, pollster, d), g in t.group_by("election", "pollster", "fieldwork_end"):
        if d < first_round_date(e):
            continue
        x = g.filter(pl.col("candidate_id") == win[e])
        rows.append({"election": e, "pollster": pollster, "fieldwork_end": d,
                     "winner": win[e], "exp_raw": x["share"][0], "exp_p": x["p_norm"][0],
                     "model_p_winner": pm.get(e, (None, None))[0],
                     "model_median_share": pm.get(e, (None, None))[1]})
    return pl.DataFrame(rows).sort("election", "fieldwork_end")


# ------------------------------------------------------------------ figures

def plot_series(exp: pl.DataFrame, bt: pl.DataFrame, polls, res, out: Path) -> None:
    win = winners()
    w = normalized(exp.filter((pl.col("question_type") == "winner") & pl.col("pair").is_null()
                              & pl.col("party").is_null()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, e in zip(axes.flat, ELECTIONS, strict=True):
        d1 = first_round_date(e)
        ww = win[e]
        x = w.filter((pl.col("election") == e) & (pl.col("candidate_id") == ww))
        days = [(d1 - d).days for d in x["fieldwork_end"]]
        ax.scatter(days, 100 * x["p_norm"], s=18, color="#1f77b4",
                   label="anticipation (part normalisée)")
        ivx = []
        for d in sorted(set(x["fieldwork_end"].to_list())):
            v = iv_mean(polls, res, e, d).get(ww)
            if v is not None:
                ivx.append(((d1 - d).days, v))
        if ivx:
            ax.plot(*zip(*sorted(ivx), strict=True), "s", ms=4, color="#ff7f0e",
                    label="intentions de vote (14 j)")
        b = bt.filter((pl.col("election") == e) & (pl.col("candidate_id") == ww)).sort("horizon")
        if b.height:
            ax.plot(b["horizon"], 100 * b["p_win"], "D-", color="#2ca02c",
                    label="P(victoire), backtest")
        ax.set_title(f"{e} : {ww}")
        ax.invert_xaxis()
        ax.set_xlabel("jours avant le premier tour")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("%")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "winner_series.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ rapport

def fmt(x, nd=1, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def coverage(exp: pl.DataFrame) -> pl.DataFrame:
    m = (exp.with_columns(
            pl.struct("election").map_elements(lambda r: first_round_date(r["election"]).isoformat(),
                                               return_dtype=pl.Utf8).alias("d1"))
         .with_columns((pl.col("d1").str.to_date() - pl.col("fieldwork_end")).dt.total_days()
                       .alias("days"))
         .group_by("election", "pollster", "question_type", "outcome", "pair", "fieldwork_end",
                   "file_id", "page", "source_url")
         .agg(pl.col("days").first(), pl.col("verified").all()))
    return (m.group_by("election", "question_type", "outcome", "pollster")
            .agg(pl.len().alias("mesures"), pl.col("days").max().alias("j_max"),
                 pl.col("days").min().alias("j_min"), pl.col("verified").all().alias("vérifiées"))
            .sort("election", "question_type", "outcome", "pollster"))


def report(out: Path, exp, cov, ht, qt, series17, series22, br, now2027) -> str:
    L = ["# Anticipations du vainqueur (« qui va gagner ? ») — étude", "",
         (f"Généré par `back/studies/expectations.py` le {dt.date.today().isoformat()}. "
         "Données : `data/curated/poll_expectations.parquet`, registre "
         "`data/registry/expectations_sources.yaml`."), "",
         "## Critères fixés avant lecture", ""]
    L += [ln.strip() for ln in (__doc__ or "").split("Critères fixés avant lecture des "
          "résultats (consignés dans le rapport) :")[1].strip().splitlines()]
    L += ["", "## Couverture", "",
          ("Jours avant le premier tour : `j_max` (plus ancienne) à `j_min` (plus récente ; "
          "négatif entre les deux tours)."), "",
          "| élection | type | issue | institut | mesures | j_max | j_min | vérifiées |",
          "|---|---|---|---|---|---|---|---|"]
    for r in cov.iter_rows(named=True):
        L.append(f"| {r['election']} | {r['question_type']} | {r['outcome']} | {r['pollster']} "
                 f"| {r['mesures']} | {r['j_max']} | {r['j_min']} | "
                 f"{'oui' if r['vérifiées'] else 'non'} |")
    L += ["", "## (a) Part qui anticipe le vainqueur, intentions de vote, P(victoire) du backtest",
          "",
          ("| élection | h | mesures | vainqueur | anticip. brute | anticip. norm. | IV (14 j) | "
          "P(victoire) modèle | favori anticip. | meneur IV | favori modèle | log-score anticip. "
          "| log-score modèle | Brier anticip. | Brier modèle |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ht.iter_rows(named=True):
        L.append(
            f"| {r['election']} | {r['horizon']} | {r['measures']} | {r['winner']} | "
            f"{fmt(r['exp_raw_winner'])} | {fmt(r['exp_p_winner'], pct=True)} | "
            f"{fmt(r['iv_winner'])} | {fmt(r['model_p_winner'], pct=True)} | "
            f"{r['exp_favourite']} | {r['iv_leader']} | {r['model_favourite']} | "
            f"{fmt(r['logscore_exp'], 3)} | {fmt(r['logscore_model'], 3)} | "
            f"{fmt(r['brier_exp'], 3)} | {fmt(r['brier_model'], 3)} |")
    L += ["", "## (b) Qualification", "", "### Ipsos 2022, « sera qualifié pour le second tour »",
          "", ("| h | candidat | anticipé qualifié | p_qualify modèle | qualifié | Brier anticip. | "
          "Brier modèle |"), "|---|---|---|---|---|---|---|"]
    for r in qt.sort("horizon", "exp_qualify", descending=[True, True]).iter_rows(named=True):
        L.append(f"| {r['horizon']} | {r['candidate_id']} | {fmt(r['exp_qualify'], pct=True)} | "
                 f"{fmt(r['model_p_qualify'], pct=True)} | {'oui' if r['qualified'] else 'non'} | "
                 f"{fmt(r['brier_exp'], 3)} | {fmt(r['brier_model'], 3)} |")
    if qt.height:
        agg = qt.group_by("horizon").agg(pl.col("brier_exp").mean(), pl.col("brier_model").mean())
        L += ["", "Brier moyen par horizon : " + " ; ".join(
            f"{r['horizon']} j : anticipations {r['brier_exp']:.3f}, modèle {r['brier_model']:.3f}"
            for r in agg.sort("horizon", descending=True).iter_rows(named=True))]
    for title, s in (("Kantar 2017 (« lequel remportera l'élection ? »)", series17),
                     ("Harris Interactive 2022", series22)):
        L += ["", f"### {title} : favori des anticipations et meneur des intentions de vote", "",
              "| fin de terrain | anticipations (3 premiers, brut) | IV 14 j (3 premiers) |",
              "|---|---|---|"]
        for r in s.iter_rows(named=True):
            L.append(f"| {r['fieldwork_end']} | {r['anticipation']} | {r['iv_top3']} |")
    L += ["", "## Entre les deux tours : pronostic du duel réel", "",
          ("| élection | institut | fin de terrain | vainqueur | anticip. brute | anticip. norm. | "
          "P(vainqueur) modèle, veille | médiane du modèle (part du vainqueur) |"),
          "|---|---|---|---|---|---|---|---|"]
    for r in br.iter_rows(named=True):
        L.append(f"| {r['election']} | {r['pollster']} | {r['fieldwork_end']} | {r['winner']} | "
                 f"{fmt(r['exp_raw'])} | {fmt(r['exp_p'], pct=True)} | "
                 f"{fmt(r['model_p_winner'], pct=True)} | {fmt(r['model_median_share'])} |")
    L += ["", "## (c) 2027", "", now2027, ""]
    return "\n".join(L)


def text_2027(exp: pl.DataFrame) -> str:
    sim = pl.read_parquet(SIM_2027).filter(pl.col("condition_id") == "mix")
    pw = dict(zip(sim["candidate_id"], sim["p_win"], strict=True))
    pq = dict(zip(sim["candidate_id"], sim["p_qualify"], strict=True))
    e = normalized(exp.filter(pl.col("election") == 2027))
    L = [("| mesure | candidat ou parti | part | part normalisée | P(victoire) modèle | "
         "P(qualif.) modèle |"), "|---|---|---|---|---|---|"]
    for r in e.sort("question_type", "fieldwork_end", "share", descending=[False, False, True]) \
            .iter_rows(named=True):
        who = r["candidate_id"] or f"parti {r['party']}"
        pn = fmt(r["p_norm"], pct=True) if r["question_type"] == "winner" else "–"
        L.append(f"| {r['pollster']} {r['fieldwork_end']} {r['question_type']} | {who} | "
                 f"{r['share']:g} | {pn} | {fmt(pw.get(r['candidate_id']), pct=True)} | "
                 f"{fmt(pq.get(r['candidate_id']), pct=True)} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.RUNS / f"{dt.date.today()}-expectations")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    exp = pl.read_parquet(paths.CURATED / "poll_expectations.parquet")
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    res = pl.read_parquet(paths.CURATED / "poll_results.parquet")
    cache = out / "backtest_pwin.parquet"
    if a.refresh or not cache.exists():
        backtest_pwin().write_parquet(cache)
    bt = pl.read_parquet(cache)
    cov = coverage(exp)
    ht = horizon_table(exp, bt, polls, res)
    qt = qualification_table(exp, bt)
    s17 = leader_dates(exp, polls, res, 2017, "kantar")
    s22 = leader_dates(exp, polls, res, 2022, "harris")
    plot_series(exp, bt, polls, res, out)
    for name, df in (("coverage", cov), ("horizons", ht), ("qualification", qt)):
        df.write_parquet(out / f"{name}.parquet")
    inv = paths.CURATED / "poll_expectations_inventory.parquet"
    if inv.exists():
        pl.read_parquet(inv).write_csv(out / "inventory.csv")
    br = between_rounds(exp)
    text = report(out, exp, cov, ht, qt, s17, s22, br, text_2027(exp))
    (out / "report.md").write_text(text)
    (out / "manifest.json").write_text(json.dumps(
        {"backtest": str(BACKTEST), "variant": VARIANT, "sim_2027": str(SIM_2027),
         "n_draws": N_DRAWS, "n_second": N_SECOND, "min_pair": MIN_PAIR}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
