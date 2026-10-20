"""Étude : le potentiel électoral (« pourrait voter pour ») et le rejet comme information de
plafond pour le second tour.

    uv run python studies/potential.py [--out runs/<date>-potential]

Entrée : data/curated/electoral_potential.parquet (`vvs parse potential`), normalisé par
mesure (vvs.model.potential, référence Kantar « five_level »). Trois questions :

(a) le potentiel d'un finaliste mesuré avant le premier tour borne-t-il son score de second
    tour ? Unités : part des INSCRITS (le potentiel est mesuré sur les inscrits), v2 = voix du
    finaliste au second tour / inscrits ; aussi, pour les sondages de duel hypothétiques, la
    part sondée × (exprimés / inscrits du second tour de référence) ;
(b) le potentiel et le rejet par électorat expliquent-ils les reports observés
    (transfers.yaml) mieux que la matrice de proximité par blocs ?
(c) que disent les mesures 2027 pour les principales personnalités et les duels inédits ?

Écrit coverage, offsets, ceiling_finalists, ceiling_duels, transfers_compare, table_2027 en
parquet et report.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.eval.second_round import observed_transfers
from vvs.model import potential as P
from vvs.model.second_round import ABSTAIN, proximity_mean
from vvs.registry import default_registry

ELECTIONS = (2012, 2017, 2022)
MAIN_2027 = (
    "le_pen_marine", "bardella_jordan", "philippe_edouard", "melenchon_jean_luc",
    "glucksmann_raphael", "attal_gabriel", "retailleau_bruno",
)
DUELS_2027 = (
    ("le_pen_marine", "philippe_edouard"),
    ("le_pen_marine", "melenchon_jean_luc"),
    ("le_pen_marine", "glucksmann_raphael"),
    ("bardella_jordan", "philippe_edouard"),
    ("le_pen_marine", "attal_gabriel"),
    ("le_pen_marine", "retailleau_bruno"),
    ("bardella_jordan", "melenchon_jean_luc"),
    ("bardella_jordan", "glucksmann_raphael"),
    ("melenchon_jean_luc", "philippe_edouard"),
)
ASOF_2027 = dt.date(2026, 10, 20)
HORIZONS = {"r1_7": 7, "r1_30": 30, "r1_90": 90, "r1_180": 180}


def _f(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",")


def _name(cid: str) -> str:
    from vvs.model.plots import display_name

    return display_name(cid)


def load_normalized() -> tuple[pl.DataFrame, P.Offsets, P.Offsets]:
    """Relevés vérifiés seulement (comme le modèle), normalisés sur les mesures de vote."""
    df = P.load()
    vote = df.filter(pl.col("scale") != "attitude")
    po = P.fit_offsets(vote, "potential_share")
    ro = P.fit_offsets(vote, "reject_share")
    return P.normalize(df, po, ro), po, ro


# --------------------------------------------------------------------------- couverture


def coverage(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("election", "pollster", "measure", "scale")
        .agg(
            pl.col("source_id").n_unique().alias("n_sources"),
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
            (pl.col("electorate") == "all").sum().alias("n_all"),
            (pl.col("electorate") != "all").sum().alias("n_electorate"),
            pl.col("potential_share").is_not_null().any().alias("has_potential"),
            pl.col("reject_share").is_not_null().any().alias("has_reject"),
            pl.col("verified").mean().alias("verified"),
        )
        .sort("election", "pollster", "measure")
    )


def offsets_table(po: P.Offsets, ro: P.Offsets) -> pl.DataFrame:
    rows = []
    for m in sorted(set(po.by_measure) | set(ro.by_measure)):
        rows.append(
            {
                "measure": m,
                "scale": po.scale_of.get(m) or ro.scale_of.get(m),
                "potential_offset": po.by_measure.get(m),
                "potential_pairs": po.n_pairs.get(m, 0),
                "reject_offset": ro.by_measure.get(m),
                "reject_pairs": ro.n_pairs.get(m, 0),
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- (a) plafond


def _round2(reg, election: int) -> dict:
    e = reg.elections[election]
    r2 = e.round_(2)
    return {
        "registered": r2.registered,
        "expressed": r2.expressed,
        "tau": r2.expressed / r2.registered,
        "share": {r.candidate_id: r.share for r in r2.results},
        "d1": e.round_(1).date,
        "d2": r2.date,
    }


def ceiling_finalists(df: pl.DataFrame, reg) -> pl.DataFrame:
    """Potentiel et rejet de chaque finaliste à chaque horizon, mesure par mesure (brut et
    normalisé), contre v2 = voix au second tour / inscrits."""
    rows = []
    for e in ELECTIONS:
        if e not in reg.elections:
            continue
        r2 = _round2(reg, e)
        sub = df.filter(
            (pl.col("election") == e) & (pl.col("electorate") == "all")
            & (pl.col("scale") != "attitude")
        )
        for cid, share in r2["share"].items():
            v2 = share * r2["tau"]
            labels = list(HORIZONS.items()) + [("r2_between", None)]
            for label, h in labels:
                if h is None:
                    win = sub.filter((pl.col("date") > r2["d1"]) & (pl.col("date") < r2["d2"]))
                else:
                    asof = r2["d1"] - dt.timedelta(days=h)
                    win = sub.filter(
                        (pl.col("date") <= asof) & (pl.col("date") > asof - dt.timedelta(days=45))
                        & (pl.col("round") == 1)
                    )
                win = win.filter(pl.col("candidate_id") == cid)
                for r in win.iter_rows(named=True):
                    rows.append(
                        {
                            "election": e, "candidate_id": cid, "horizon": label,
                            "date": r["date"], "measure": r["measure"], "source_id": r["source_id"],
                            "share2": share, "v2": v2, "tau2": r2["tau"],
                            "potential": r["potential_share"], "potential_norm": r["potential_norm"],
                            "reject": r["reject_share"], "reject_norm": r["reject_norm"],
                        }
                    )
    out = pl.DataFrame(rows, infer_schema_length=None)
    if out.is_empty():
        return out
    return out.with_columns(
        (pl.col("potential") - pl.col("v2")).alias("slack_potential"),
        (pl.col("potential_norm") - pl.col("v2")).alias("slack_potential_norm"),
        (100 - pl.col("reject") - pl.col("v2")).alias("slack_reject"),
        (100 - pl.col("reject_norm") - pl.col("v2")).alias("slack_reject_norm"),
    )


def ceiling_duels(df: pl.DataFrame, reg, elections=(2017, 2022, 2027)) -> pl.DataFrame:
    """Sondages de duel hypothétiques avant le premier tour : part sondée de chaque
    personnalité convertie en part des inscrits (× exprimés/inscrits du second tour de
    l'élection, ou de 2022 pour 2027), contre son potentiel et son rejet normalisés publiés à
    moins de 45 jours avant la fin de terrain du duel."""
    from vvs.model.data import load_polls

    rows = []
    for e in elections:
        tau_e = _round2(reg, e if e != 2027 else 2022)["tau"]
        d1 = reg.elections[e].round_(1).date if e != 2027 else ASOF_2027 + dt.timedelta(days=1)
        polls = load_polls(e, min(d1 - dt.timedelta(days=1), ASOF_2027), round_=2)
        hyp = polls.group_by("poll_id", "hypothesis_id").agg(pl.len()).filter(pl.col("len") == 2)
        polls = polls.join(hyp.select("poll_id", "hypothesis_id"), on=["poll_id", "hypothesis_id"])
        polls = polls.with_columns(
            (100 * pl.col("share") / pl.col("share").sum().over("poll_id", "hypothesis_id")).alias("s")
        )
        pot = df.filter(
            (pl.col("election") == e) & (pl.col("electorate") == "all")
            & (pl.col("scale") != "attitude") & (pl.col("round") == 1)
        )
        for (cid,), g in polls.group_by("candidate_id"):
            pc = pot.filter(pl.col("candidate_id") == cid).sort("date")
            if pc.is_empty():
                continue
            pdates = pc["date"].to_list()
            for r in g.iter_rows(named=True):
                end = r["fieldwork_end"]
                ok = [k for k, d in enumerate(pdates) if dt.timedelta(0) <= end - d <= dt.timedelta(days=45)]
                if not ok:
                    continue
                w = pc[ok]
                pn = w["potential_norm"].drop_nulls().to_numpy()
                rn = w["reject_norm"].drop_nulls().to_numpy()
                rows.append(
                    {
                        "election": e, "candidate_id": cid, "poll_id": r["poll_id"],
                        "hypothesis_id": r["hypothesis_id"], "fieldwork_end": end,
                        "duel_share": r["s"], "v_implied": r["s"] * tau_e, "tau": tau_e,
                        "potential_norm": float(P.expit(np.mean(P.logit(pn)))) if len(pn) else None,
                        "reject_norm": float(P.expit(np.mean(P.logit(rn)))) if len(rn) else None,
                        "days_before_r1": (d1 - end).days,
                    }
                )
    out = pl.DataFrame(rows, infer_schema_length=None)
    if out.is_empty():
        return out
    return out.with_columns(
        (pl.col("potential_norm") - pl.col("v_implied")).alias("slack_potential"),
        (100 - pl.col("reject_norm") - pl.col("v_implied")).alias("slack_reject"),
    )


# --------------------------------------------------------------------------- (b) reports


def electorate_values(
    df: pl.DataFrame, election: int, candidate: str, bloc: str, asof: dt.date,
    finalist_blocs: tuple[str, str], reg, max_age: int = 120,
) -> tuple[float | None, float | None, int, str]:
    """(potentiel, rejet) normalisés de `candidate` dans le bloc d'électeurs `bloc` :
    intentions de vote de premier tour d'abord (électorats exacts), sinon proximité
    partisane (sauf si le bloc est celui d'un finaliste : les sympathisants y sont surtout ses
    propres électeurs), sinon vote passé. Dernière vague disponible."""
    x = df.filter(
        (pl.col("election") == election) & (pl.col("candidate_id") == candidate)
        & (pl.col("electorate") != "all") & (pl.col("scale") != "attitude")
        & (pl.col("date") <= asof) & (pl.col("date") > asof - dt.timedelta(days=max_age))
    )
    if x.is_empty():
        return None, None, 0, ""
    blocs = [P.bloc_of_electorate(c, election, reg) for c in x["electorate"].to_list()]
    x = x.with_columns(pl.Series("bloc", blocs, dtype=pl.Utf8)).filter(pl.col("bloc") == bloc)
    kinds = ["vote_intention_r1", "party_proximity", "past_vote"]
    for kind in kinds:
        y = x.filter(pl.col("electorate_kind") == kind)
        if kind == "party_proximity" and bloc in finalist_blocs:
            continue
        if kind == "vote_intention_r1":
            y = y.filter(~pl.col("electorate").str.ends_with(f":{candidate}"))
        if y.is_empty():
            continue
        y = y.filter(pl.col("date") == y["date"].max())
        pn = y["potential_norm"].drop_nulls().to_numpy()
        rn = y["reject_norm"].drop_nulls().to_numpy()
        return (
            float(P.expit(np.mean(P.logit(pn)))) if len(pn) else None,
            float(P.expit(np.mean(P.logit(rn)))) if len(rn) else None,
            y.height,
            kind,
        )
    return None, None, 0, ""


def split_from_potential(pi, pj, ri, rj) -> tuple[float | None, float | None]:
    """Répartition vers i parmi les exprimés : à partir des potentiels, p_i / (p_i + p_j) ; à
    partir des rejets, (100 − r_i) / ((100 − r_i) + (100 − r_j))."""
    sp = None if pi is None or pj is None else pi / (pi + pj)
    sr_ = None if ri is None or rj is None else (100 - ri) / ((100 - ri) + (100 - rj))
    return sp, sr_


def transfers_compare(df: pl.DataFrame, reg) -> pl.DataFrame:
    rows = []
    for e in ELECTIONS:
        el = reg.elections[e]
        pair = tuple(r.candidate_id for r in el.round_(2).results)
        blocs = tuple(reg.bloc_of(c, e) for c in pair)
        asof = el.round_(1).date - dt.timedelta(days=1)
        for g in ("far_left", "left", "centre", "right", "far_right", "other", ABSTAIN):
            group = ABSTAIN if g == ABSTAIN else f"bloc:{g}"
            split_obs, none_obs = observed_transfers(e, group, pair, reg)
            if split_obs is None:
                continue
            m = proximity_mean(g, *blocs)
            row = {
                "election": e, "candidate_i": pair[0], "candidate_j": pair[1], "group": group,
                "obs_split": split_obs, "obs_none": none_obs,
                "matrix_split": m[0] / (m[0] + m[1]), "matrix_none": m[2],
            }
            if g != ABSTAIN:
                pi, ri, _, ki = electorate_values(df, e, pair[0], g, asof, blocs, reg)
                pj, rj, _, kj = electorate_values(df, e, pair[1], g, asof, blocs, reg)
                sp, srj = split_from_potential(pi, pj, ri, rj)
                row |= {
                    "pot_i": pi, "pot_j": pj, "rej_i": ri, "rej_j": rj,
                    "kind": ki or kj, "pot_split": sp, "rej_split": srj,
                }
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def calibrated_split(t: pl.DataFrame) -> pl.DataFrame:
    """Variante calibrée : logit(répartition) = κ · logit(répartition via potentiel ou rejet),
    κ estimé par moindres carrés sur les AUTRES élections (leave-one-election-out)."""
    x = t.with_columns(pl.coalesce("rej_split", "pot_split").alias("raw"))
    have = x.filter(pl.col("raw").is_not_null())
    preds = []
    for r in x.iter_rows(named=True):
        if r["raw"] is None:
            preds.append(None)
            continue
        other = have.filter(pl.col("election") != r["election"])
        if other.is_empty():
            preds.append(None)
            continue
        a = np.log(other["raw"].to_numpy() / (1 - other["raw"].to_numpy()))
        b = np.log(other["obs_split"].to_numpy() / (1 - other["obs_split"].to_numpy()))
        kappa = float(a @ b / (a @ a))
        z = kappa * np.log(r["raw"] / (1 - r["raw"]))
        preds.append(float(1 / (1 + np.exp(-z))))
    return x.with_columns(pl.Series("cal_split", preds, dtype=pl.Float64)).drop("raw")


def transfers_scores(t: pl.DataFrame) -> pl.DataFrame:
    """Erreur absolue moyenne de la répartition (points de %) par prédicteur, sur les cas où
    chaque prédicteur existe, et sur les cas communs."""
    out = []
    for pred in ("matrix_split", "pot_split", "rej_split", "cal_split"):
        if pred not in t.columns:
            continue
        x = t.filter(pl.col(pred).is_not_null())
        if x.is_empty():
            continue
        out.append(
            {
                "predictor": pred, "cases": "own", "n": x.height,
                "mae": float((100 * (x[pred] - x["obs_split"]).abs()).mean()),
                "mae_matrix_same_cases": float((100 * (x["matrix_split"] - x["obs_split"]).abs()).mean()),
            }
        )
    return pl.DataFrame(out)


# --------------------------------------------------------------------------- (c) 2027


def table_2027(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    x = df.filter((pl.col("election") == 2027) & (pl.col("electorate") == "all"))
    for cid in MAIN_2027:
        c = x.filter(pl.col("candidate_id") == cid)
        row = {"candidate_id": cid}
        for m in ("harris_binary", "harris_zero_ten", "opinionway_four_level", "ipsos_four_level",
                  "cluster17_zero_ten", "odoxa_attitude"):
            cm = c.filter(pl.col("measure") == m).sort("date")
            if cm.is_empty():
                continue
            last = cm.row(-1, named=True)
            row[f"{m}_date"] = last["date"]
            row[f"{m}_pot"] = last["potential_share"]
            row[f"{m}_rej"] = last["reject_share"]
        s = P.summary_at(df.filter(pl.col("scale") != "attitude"), 2027, cid, ASOF_2027)
        if s is not None:
            row |= {"pot_norm": s.potential, "rej_norm": s.reject, "n_recent": s.n, "last": s.last_date}
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def parties_2027(df: pl.DataFrame) -> pl.DataFrame:
    """Potentiel par sympathisants (suivi Toluna Harris Interactive, dernière vague) ; seuls
    les dix premiers noms de chaque parti sont publiés : une absence veut dire « moins que le
    dixième nom » (colonne `floor_if_absent`)."""
    x = df.filter(
        (pl.col("election") == 2027) & (pl.col("electorate_kind") == "party_proximity")
        & (pl.col("measure") == "harris_binary")
    )
    last = x["date"].max()
    x = x.filter(pl.col("date") == last)
    floors = x.group_by("electorate").agg(pl.col("potential_share").min().alias("floor"))
    rows = []
    for party, floor in floors.sort("electorate").iter_rows():
        for cid in MAIN_2027:
            v = x.filter((pl.col("electorate") == party) & (pl.col("candidate_id") == cid))
            rows.append(
                {"date": last, "party": party, "candidate_id": cid,
                 "potential": v["potential_share"][0] if v.height else None,
                 "floor_if_absent": floor}
            )
    return pl.DataFrame(rows)


def duels_2027(df: pl.DataFrame, reg) -> pl.DataFrame:
    """Pour chaque duel : plafonds des inscrits (potentiel et 100 − rejet normalisés) de chaque
    finaliste, convertis en part des exprimés avec exprimés/inscrits du second tour 2022."""
    tau = _round2(reg, 2022)["tau"]
    vote = df.filter(pl.col("scale") != "attitude")
    rows = []
    for i, j in DUELS_2027:
        si = P.summary_at(vote, 2027, i, ASOF_2027)
        sj = P.summary_at(vote, 2027, j, ASOF_2027)
        row = {"candidate_i": i, "candidate_j": j, "tau_2022": tau}
        for k, s in (("i", si), ("j", sj)):
            if s is None:
                continue
            row[f"pot_{k}"] = s.potential
            row[f"rej_{k}"] = s.reject
            row[f"ceil_reg_{k}"] = s.ceiling
            row[f"pot_expr_{k}"] = None if s.potential is None else s.potential / tau
            row[f"ceil_expr_{k}"] = None if s.ceiling is None else s.ceiling / tau
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


# --------------------------------------------------------------------------- rapport


def report(out: Path, res: dict) -> str:
    L = [
        "# Potentiel électoral et rejet : plafond du second tour ?",
        "",
        (f"Étude du {dt.date.today().isoformat()}, `back/studies/potential.py`. Données : "
        "`data/curated/electoral_potential.parquet` (`vvs parse potential`). Potentiel et "
        "rejet en % des inscrits (sauf mention), normalisés par mesure sur l'échelle Kantar "
        "« five_level » (potentiel = possible + fortes chances + certain ; rejet = « je ne "
        "voterai jamais pour lui/elle »)."),
        "",
        "## Couverture",
        "",
        "| élection | institut | mesure | sources | du | au | lignes ensemble | lignes électorats | potentiel | rejet | vérifié |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["coverage"].iter_rows(named=True):
        L.append(
            f"| {r['election']} | {r['pollster']} | {r['measure']} | {r['n_sources']} | {r['first']} | "
            f"{r['last']} | {r['n_all']} | {r['n_electorate']} | {'oui' if r['has_potential'] else ''} | "
            f"{'oui' if r['has_reject'] else ''} | {_f(100 * r['verified'], 0)} % |"
        )
    L += [
        "",
        ("Instituts sans potentiel par candidat trouvable : Elabe, Ifop (hors septembre-octobre "
         "2021 et 2011), Harris (hors 2027 et un sondage DSK 2011), Odoxa (résultats 2017 non "
         "retrouvés ; en 2027, seulement « vous la rejetez », mesure d'attitude), TNS Sofres, "
         "CSA, OpinionWay (hors 2027), LH2, YouGov, Verian. Relevés non vérifiés (exclus du "
         "modèle et des analyses) : Ifop 2011 et 2012 (ifop.com sert un certificat invalide, "
         "brut non téléchargé), ventilations Ifop 2021 (extraits paraphrasés), Harris 2011."),
        "",
        "## Normalisation par mesure",
        "",
        ("Décalage logit de la mesure par rapport à la référence (positif : la mesure donne des "
        "parts plus élevées), estimé sur les couples de mesures d'une même personnalité à moins "
        "de 21 jours. Paires = nombre de couples utilisés ; une mesure sans paire garde le "
        "décalage de son échelle."),
        "",
        "| mesure | échelle | potentiel | paires | rejet | paires |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["offsets"].iter_rows(named=True):
        L.append(
            f"| {r['measure']} | {r['scale']} | {_f(r['potential_offset'], 2)} | {r['potential_pairs']} | "
            f"{_f(r['reject_offset'], 2)} | {r['reject_pairs']} |"
        )
    cf = res["ceiling_finalists"]
    L += [
        "",
        "## (a) Le potentiel borne-t-il le score de second tour ?",
        "",
        ("Finalistes : v2 = voix au second tour en % des inscrits. Marge = valeur − v2 (négative "
        "= plafond franchi). Une ligne par mesure publiée dans les 45 jours précédant l'horizon."),
        "",
        "| élection | finaliste | horizon | mesure | date | v2 | part exprimés | potentiel | norm. | marge | rejet | 100 − rejet norm. | marge |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cf.sort("election", "candidate_id", "horizon", "date").iter_rows(named=True):
        L.append(
            f"| {r['election']} | {_name(r['candidate_id'])} | {r['horizon']} | {r['measure']} | {r['date']} | "
            f"{_f(r['v2'])} | {_f(r['share2'])} | {_f(r['potential'])} | {_f(r['potential_norm'])} | "
            f"{_f(r['slack_potential_norm'])} | {_f(r['reject'])} | "
            f"{_f(None if r['reject_norm'] is None else 100 - r['reject_norm'])} | {_f(r['slack_reject_norm'])} |"
        )
    agg = res["ceiling_finalists_agg"]
    L += ["", "Résumé par finaliste (moyenne des mesures avant le premier tour, normalisées) :", "",
          "| élection | finaliste | v2 | potentiel norm. | marge | 100 − rejet norm. | marge | n |",
          "|---|---|---|---|---|---|---|---|"]
    for r in agg.iter_rows(named=True):
        L.append(
            f"| {r['election']} | {_name(r['candidate_id'])} | {_f(r['v2'])} | {_f(r['pot'])} | "
            f"{_f(r['slack_pot'])} | {_f(r['ceil'])} | {_f(r['slack_ceil'])} | {r['n']} |"
        )
    cd = res["ceiling_duels_agg"]
    L += [
        "",
        ("Sondages de duel hypothétiques (part sondée × exprimés/inscrits du second tour ; 2022 "
        "pour 2027), contre le potentiel et le rejet normalisés publiés dans les 45 jours "
        "précédents :"),
        "",
        "| élection | personnalité | sondages | part sondée moy. | v impliquée | potentiel norm. | > potentiel | 100 − rejet | > 100 − rejet |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cd.iter_rows(named=True):
        L.append(
            f"| {r['election']} | {_name(r['candidate_id'])} | {r['n']} | {_f(r['duel_share'])} | "
            f"{_f(r['v_implied'])} | {_f(r['pot'])} | {_f(100 * r['over_pot'], 0)} % | {_f(r['ceil'])} | "
            f"{_f(100 * r['over_ceil'], 0) if r['over_ceil'] is not None else '–'} % |"
        )
    L += ["", res["reading_a"], ""]
    t = res["transfers"]
    L += [
        "## (b) Potentiel et rejet par électorat contre reports observés",
        "",
        ("Répartition vers le finaliste i parmi les exprimés de l'électorat (observée dans les "
        "enquêtes de reports de l'élection) ; prédicteurs : matrice de proximité, potentiels "
        "p_i / (p_i + p_j), rejets (100 − r_i) / (200 − r_i − r_j), mesurés dans le bloc "
        "(intentions de premier tour, à défaut proximité partisane hors blocs des finalistes) "
        "la veille du premier tour."),
        "",
        "| élection | i | j | électorat | observée | matrice | potentiel i / j | via potentiel | rejet i / j | via rejet | calibré (LOO) | source |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in t.iter_rows(named=True):
        L.append(
            f"| {r['election']} | {_name(r['candidate_i'])} | {_name(r['candidate_j'])} | {r['group']} | "
            f"{_f(100 * r['obs_split'])} | {_f(100 * r['matrix_split'])} | "
            f"{_f(r.get('pot_i'))} / {_f(r.get('pot_j'))} | "
            f"{_f(None if r.get('pot_split') is None else 100 * r['pot_split'])} | "
            f"{_f(r.get('rej_i'))} / {_f(r.get('rej_j'))} | "
            f"{_f(None if r.get('rej_split') is None else 100 * r['rej_split'])} | "
            f"{_f(None if r.get('cal_split') is None else 100 * r['cal_split'])} | {r.get('kind') or ''} |"
        )
    L += ["", "| prédicteur | n | erreur abs. moyenne | matrice sur les mêmes cas |", "|---|---|---|---|"]
    for r in res["transfers_scores"].iter_rows(named=True):
        L.append(f"| {r['predictor']} | {r['n']} | {_f(r['mae'])} | {_f(r['mae_matrix_same_cases'])} |")
    L += ["", res["reading_b"], ""]
    L += [
        "## (c) 2027",
        "",
        ("Dernière mesure de chaque série (potentiel / rejet, en %) et moyenne normalisée des "
        f"mesures des 60 jours précédant la dernière, au {ASOF_2027}."),
        "",
        "| personnalité | Harris oui/non | Harris 0-10 (7-10 / 0-3) | OpinionWay | Ipsos bva | Cluster17 | Odoxa « rejette » | potentiel norm. | rejet norm. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def cell(r: dict, m: str) -> str:
        if r.get(f"{m}_date") is None:
            return "–"
        return f"{_f(r.get(f'{m}_pot'), 0)} / {_f(r.get(f'{m}_rej'), 0)} ({r[f'{m}_date']:%d/%m/%y})"

    for r in res["table_2027"].iter_rows(named=True):
        L.append(
            f"| {_name(r['candidate_id'])} | {cell(r, 'harris_binary')} | {cell(r, 'harris_zero_ten')} | "
            f"{cell(r, 'opinionway_four_level')} | {cell(r, 'ipsos_four_level')} | "
            f"{cell(r, 'cluster17_zero_ten')} | {cell(r, 'odoxa_attitude')} | {_f(r.get('pot_norm'))} | {_f(r.get('rej_norm'))} |"
        )
    pt = res["parties_2027"]
    if not pt.is_empty():
        parties = pt["party"].unique(maintain_order=True).to_list()
        L += ["", (f"Potentiel par sympathisants (Toluna Harris Interactive, vague du {pt['date'][0]}) ; "
              "« < x » : hors des dix premiers noms du parti, donc sous le dixième (x)."), "",
              "| personnalité | " + " | ".join(parties) + " |", "|---|" + "---|" * len(parties)]
        for cid in MAIN_2027:
            cells = []
            for p in parties:
                v = pt.filter((pl.col("party") == p) & (pl.col("candidate_id") == cid)).row(0, named=True)
                cells.append(_f(v["potential"], 0) if v["potential"] is not None else f"< {_f(v['floor_if_absent'], 0)}")
            L.append(f"| {_name(cid)} | " + " | ".join(cells) + " |")
    L += ["", ("Duels : plafond des inscrits (100 − rejet normalisé) et potentiel normalisé, "
          "convertis en part des exprimés avec exprimés/inscrits du second tour 2022 "
          f"({_f(100 * res['duels_2027']['tau_2022'][0])} %) :"), "",
          "| i | j | potentiel i | plafond i (inscrits) | plafond i (exprimés) | potentiel j | plafond j (inscrits) | plafond j (exprimés) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["duels_2027"].iter_rows(named=True):
        L.append(
            f"| {_name(r['candidate_i'])} | {_name(r['candidate_j'])} | {_f(r.get('pot_i'))} | "
            f"{_f(r.get('ceil_reg_i'))} | {_f(r.get('ceil_expr_i'))} | {_f(r.get('pot_j'))} | "
            f"{_f(r.get('ceil_reg_j'))} | {_f(r.get('ceil_expr_j'))} |"
        )
    L += ["", res["reading_c"], ""]
    if "backtest_agg" in res:
        L += backtest_section(res)
    if "duels_before_after" in res:
        L += duels_section(res["duels_before_after"])
    if res.get("reading_verdict"):
        L += ["## Verdict d'admission", "", res["reading_verdict"], ""]
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


HORIZON_LABEL = {
    "r2_eve": "veille du 2d tour",
    "r1_30": "30 j avant le 1er tour",
    "r1_90": "90 j avant le 1er tour",
    "r1_180": "180 j avant le 1er tour",
}


def backtest_section(res: dict) -> list[str]:
    from vvs.eval import second_round as ev

    agg = res["backtest_agg"].filter(pl.col("method").is_in([ev.PAIRED_METHOD, ev.POTENTIAL_METHOD]))
    label = {ev.PAIRED_METHOD: "modèle (apparié)", ev.POTENTIAL_METHOD: "modèle + potentiel"}
    L = [
        "## Évaluation : backtest leave-one-election-out du second tour",
        "",
        ("`vvs.eval.second_round.run(with_potential=True)`, 8000 tirages, graine 0. Comparaison "
         "appariée : `model_paired` et `model_potential` partagent la graine de chaque cas (mêmes "
         "tirages du prior, seuls les poids diffèrent) ; `model` (graine commune du run) "
         "redonne runs/2026-10-20-second-round au bruit Monte-Carlo près. Le potentiel n'existe "
         "qu'en 2012 (veille du second tour seulement), 2017 et 2022 : 2002 et 2007 sont "
         "identiques par construction."),
        "",
        "| horizon | méthode | n | erreur abs. | CRPS | couv. 50 % | couv. 90 % | log-score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in ("r1_180", "r1_90", "r1_30", "r2_eve"):
        for r in agg.filter(pl.col("horizon") == h).sort("method").iter_rows(named=True):
            L.append(
                f"| {HORIZON_LABEL[h]} | {label[r['method']]} | {r['n']} | {_f(r['mae'], 2)} | "
                f"{_f(r['crps'], 3)} | {_f(r['cover50'], 2)} | {_f(r['cover90'], 2)} | {_f(r['log_score'], 4)} |"
            )
    m = res["backtest_metrics"].filter(
        pl.col("method").is_in([ev.PAIRED_METHOD, ev.POTENTIAL_METHOD]) & (pl.col("election") >= 2012)
    )
    w = m.pivot(on="method", index=["election", "horizon", "result"], values=["median", "crps"])
    L += [
        "",
        "Par cas (2012-2022 ; médiane de la part du vainqueur, CRPS ; Δ < 0 = gain) :",
        "",
        "| élection | horizon | résultat | médiane sans / avec | CRPS sans / avec | Δ CRPS |",
        "|---|---|---|---|---|---|",
    ]
    for r in w.sort("horizon", "election").iter_rows(named=True):
        a, b = r[f"crps_{ev.PAIRED_METHOD}"], r[f"crps_{ev.POTENTIAL_METHOD}"]
        L.append(
            f"| {r['election']} | {HORIZON_LABEL[r['horizon']]} | {_f(r['result'])} | "
            f"{_f(r[f'median_{ev.PAIRED_METHOD}'])} / {_f(r[f'median_{ev.POTENTIAL_METHOD}'])} | "
            f"{_f(a, 3)} / {_f(b, 3)} | {_f(b - a, 3)} |"
        )
    return [*L, ""]


def duels_section(t: pl.DataFrame) -> list[str]:
    L = [
        "## 2027 : tableau par paire avant / après (`vvs second-round --asof 2026-10-20`)",
        "",
        ("Part de i en % des exprimés, médiane [5–95 %] et P(i gagne) ; sans l'option (même "
         "graine, identique à runs/2026-10-20-second-round-2027) et avec `--potential` "
         "(runs/2026-10-20-second-round-2027-potential). Plafonds en % des inscrits ; τ = "
         "exprimés / inscrits du second tour 2022."),
        "",
        "| i | j | sans | P sans | avec | P avec | plafond i | plafond j |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in t.sort("p0", descending=True).iter_rows(named=True):
        L.append(
            f"| {_name(r['candidate_i'])} | {_name(r['candidate_j'])} | "
            f"{_f(r['m0'])} [{_f(r['lo0'])}–{_f(r['hi0'])}] | {_f(r['p0'], 2)} | "
            f"{_f(r['m1'])} [{_f(r['lo1'])}–{_f(r['hi1'])}] | {_f(r['p1'], 2)} | "
            f"{_f(r['ceiling_i'])} | {_f(r['ceiling_j'])} |"
        )
    return [*L, ""]


def duels_before_after(before: Path, after: Path) -> pl.DataFrame:
    k = ["candidate_i", "candidate_j"]
    a = pl.read_parquet(before / "pairs.parquet").select(
        *k, pl.col("median").alias("m0"), pl.col("q05").alias("lo0"), pl.col("q95").alias("hi0"),
        pl.col("p_i_wins").alias("p0"),
    )
    b = pl.read_parquet(after / "pairs.parquet").select(
        *k, pl.col("median").alias("m1"), pl.col("q05").alias("lo1"), pl.col("q95").alias("hi1"),
        pl.col("p_i_wins").alias("p1"), "ceiling_i", "ceiling_j",
    )
    return a.join(b, on=k)


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    reg = default_registry()
    df, po, ro = load_normalized()
    res: dict = {"coverage": coverage(P.load(verified_only=False)), "offsets": offsets_table(po, ro)}
    cf = ceiling_finalists(df, reg)
    res["ceiling_finalists"] = cf
    res["ceiling_finalists_agg"] = (
        cf.filter(pl.col("horizon") != "r2_between")
        .group_by("election", "candidate_id")
        .agg(
            pl.col("v2").first(),
            pl.col("potential_norm").drop_nulls().mean().alias("pot"),
            (100 - pl.col("reject_norm")).drop_nulls().mean().alias("ceil"),
            pl.len().alias("n"),
        )
        .with_columns((pl.col("pot") - pl.col("v2")).alias("slack_pot"),
                      (pl.col("ceil") - pl.col("v2")).alias("slack_ceil"))
        .sort("election", "candidate_id")
    )
    cd = ceiling_duels(df, reg)
    res["ceiling_duels"] = cd
    res["ceiling_duels_agg"] = (
        cd.group_by("election", "candidate_id")
        .agg(
            pl.len().alias("n"),
            pl.col("duel_share").mean(),
            pl.col("v_implied").mean(),
            pl.col("potential_norm").mean().alias("pot"),
            (100 - pl.col("reject_norm")).mean().alias("ceil"),
            (pl.col("slack_potential") < 0).mean().alias("over_pot"),
            (pl.col("slack_reject") < 0).mean().alias("over_ceil"),
        )
        .sort("election", "candidate_id")
    )
    t = calibrated_split(transfers_compare(df, reg))
    res["transfers"] = t
    res["transfers_scores"] = transfers_scores(t)
    res["table_2027"] = table_2027(df)
    res["parties_2027"] = parties_2027(df)
    res["duels_2027"] = duels_2027(df, reg)
    for k in ("coverage", "offsets", "ceiling_finalists", "ceiling_duels", "transfers",
              "transfers_scores", "table_2027", "parties_2027", "duels_2027"):
        res[k].write_parquet(out / f"{k}.parquet")
    res.setdefault("reading_a", "")
    res.setdefault("reading_b", "")
    res.setdefault("reading_c", "")
    return res


# --------------------------------------------------------------------------- évaluation


def backtest(out: Path, n: int = 8000, seed: int = 0) -> dict[str, pl.DataFrame]:
    """Backtest du second tour (vvs.eval.second_round, leave-one-election-out, 2002-2022,
    veille du second tour et 30, 90, 180 jours avant le premier tour) avec la méthode
    `model_potential` en plus ; mêmes tirages et graine que runs/2026-10-20-second-round."""
    from vvs.eval import second_round as ev

    res = ev.run(n=n, seed=seed, with_potential=True)
    for k, v in res.items():
        v.write_parquet(out / f"backtest_{k}.parquet")
    m = res["metrics"].filter(
        pl.col("method").is_in(["model", ev.PAIRED_METHOD, ev.POTENTIAL_METHOD])
    )
    res["by_case"] = (
        m.select("election", "horizon", "method", "median", "abs_error", "crps", "cover50",
                 "cover90", "log_score", "result")
        .pivot(on="method", index=["election", "horizon", "result"],
               values=["median", "abs_error", "crps", "log_score"])
        .sort("horizon", "election")
    )
    res["agg"] = ev.aggregate(m)
    res["by_case"].write_parquet(out / "backtest_by_case.parquet")
    res["agg"].write_parquet(out / "backtest_agg.parquet")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(paths.RUNS / f"{dt.date.today().isoformat()}-potential"))
    ap.add_argument("--reading", default=None, help="Fichier markdown des lectures (a), (b), (c)")
    ap.add_argument("--backtest", action="store_true", help="Rejoue aussi le backtest du second tour")
    ap.add_argument(
        "--after", default=str(paths.RUNS / "2026-10-20-second-round-2027-potential"),
        help="Sortie de `vvs second-round --potential` (référence sans option attendue dans "
        "<out>/second-round-2027-baseline)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    res = run(out)
    if args.backtest:
        bt = backtest(out)
        res["backtest_agg"], res["backtest_metrics"] = bt["agg"], bt["metrics"]
    before, after = out / "second-round-2027-baseline", Path(args.after)
    if (before / "pairs.parquet").exists() and (after / "pairs.parquet").exists():
        res["duels_before_after"] = duels_before_after(before, after)
        res["duels_before_after"].write_parquet(out / "duels_before_after.parquet")
    if args.reading:
        text = Path(args.reading).read_text(encoding="utf-8")
        parts = {
            blk.split("\n", 1)[0].strip(): blk.split("\n", 1)[1].strip()
            for blk in text.split("@@")[1:]
        }
        res.update({f"reading_{k}": v for k, v in parts.items()})
    print(report(out, res))


if __name__ == "__main__":
    main()
