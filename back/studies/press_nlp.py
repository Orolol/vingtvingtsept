"""Étude : le ton et la posture des titres de presse, classés par un modèle local, annoncent-ils
les mouvements des sondages mieux que les comptages ?

    uv run python studies/press_nlp.py plan                 # budget de la collecte
    uv run python studies/press_nlp.py collect              # Media Cloud story-list (clé)
    uv run python studies/press_nlp.py parse                # cache -> press_titles.parquet
    uv run python studies/press_nlp.py classify [--cycles 2017 2022 2027]  # llama-server
    uv run python studies/press_nlp.py sample               # échantillon de 300, deux prompts
    uv run python studies/press_nlp.py features             # -> press_features.parquet
    uv run python studies/press_nlp.py study [--out runs/<date>-press-nlp] [--shifts 200]

Suite de l'étude par comptages (`studies/press.py`, runs/2026-10-20-press-study, négative sur
six tests) et de l'étude attention (runs/2026-10-20-attention). Rien n'entre dans le modèle.

Données :

- titres Media Cloud (`press_titles.parquet`, vvs.ingest.press.mediacloud_titles) : tous les
  articles de la collection France - National qui citent un des candidats suivis (alias du
  registre), du 26 décembre 2016 au 22 avril 2017 (11 candidats officiels), du 10 janvier au
  9 avril 2022 (12 candidats officiels), du 1er janvier au 18 octobre 2026 (12 candidats les
  plus sondés) ;
- étiquettes (`press_labels.parquet`, vvs.nlp) : titres qui citent le candidat dans le titre
  lui-même, au plus 300 par candidat et par semaine (répartis sur les jours), classés par
  Qwen3.8-27B (GGUF Q5_K_XL, llama.cpp) avec le prompt `title-v3`
  (data/registry/nlp_prompts.yaml) : ton envers le candidat (−1, 0, 1), posture (annonce,
  attaque subie, soutien, affaire, sondage, programme, autre), thèmes de la liste fermée ;
- variables hebdomadaires (`press_features.parquet`, vvs.nlp.features) ; une semaine compte
  pour un candidat si au moins `MIN_LABELED` titres y sont classés ;
- sondages : part hebdomadaire par candidat et par bloc (mêmes règles que les études presse et
  attention) ; sondages individuels pour les chocs ;
- comparaisons : part de voix par comptage (articles qui citent le candidat, titre ou corps,
  parmi les candidats suivis : la mesure de l'étude par comptages), part « affaires » par
  mots-clés (press_mediacloud.parquet), attention Wikipédia (part des pages vues parmi les
  mêmes candidats).

Qualité de l'étiquetage, seuils fixés avant toute classification de masse : échantillon de
300 titres stratifié par cycle (100 chacun) et par candidat (parts égales, dans l'univers
classé), étiqueté avec `title-v3` et `title-v3-alt` (autre formulation, autre ordre des
champs, autres voisins dans les lots de huit). Accord : κ de Cohen pour le ton et la posture,
F1 micro pour les thèmes (v3 en référence). Seuils : κ ton ≥ 0,6, κ posture ≥ 0,5, F1 thèmes
≥ 0,6. Une dimension sous son seuil reste dans l'étude mais est signalée comme non fiable et
ne peut pas « tenir ». Contrôle supplémentaire : accord entre `title-v1` (un titre par
requête, sortie JSON) et `title-v3` (lots de huit) sur les 500 titres de la mesure de débit.

Contrainte des fenêtres : 13 à 17 semaines en 2017 et 2022 (consigne : douze semaines avant
le vote), 42 en 2027. Les spécifications de l'étude par comptages (variation sur quatre
semaines d'une moyenne sur quatre semaines, placebo à 13 semaines au moins) n'y tiennent
pas ; on les raccourcit ici, avant lecture :

- x2(t) = moyenne de la variable sur les semaines t − 1 et t (une seule si l'autre manque) ;
- élan des sondages : sondage(t) − sondage(t − 2) (tolérance d'une semaine) ;
- effet fixe candidat × cycle (on compare chaque candidat à lui-même), grappes candidat ×
  cycle ;
- placebo : décalage circulaire de la série de presse de tous les candidats d'un cycle de s
  semaines, `MIN_SHIFT` ≤ s ≤ T − `MIN_SHIFT`, tiré indépendamment par cycle ; p placebo sur
  les décalages distincts ; taille = part des décalages où le test nominal rejette à 5 %.
  Placebo complémentaire : permutation des séries de presse entre candidats d'un même cycle.

Questions et critères (fixés avant de lire les résultats ; seuls les volumes de titres par
candidat étaient visibles) :

(a) **Ton et attaques → sondages, 1 à 4 semaines.** y = sondage(t + h) − sondage(t), h = 1 à
    4 (cible t + h, sinon t + h + 1, sinon t + h − 1, toujours après t), sur x2(t) et l'élan.
    Variables : ton (attendu β > 0), part d'attaques subies et part d'affaires (attendu
    β < 0), part de titres négatifs (β < 0). Principal : ton, h = 2, commun 2017-2022.
    Critère « tient » : β > 0, p placebo (décalage) < 0,05 en commun, β > 0 en 2017 et en
    2022, et Granger ton → sondages (ordre 2 : Δsondage(t) sur ses deux retards et le ton des
    semaines t − 1 et t − 2) à p placebo < 0,05. 2027 en réplication (même signe attendu).
    Comparaisons sur les mêmes lignes : part de voix par comptage, part « affaires » par
    mots-clés, attention Wikipédia ; régression conjointe ton + part de voix + Wikipédia.
    « Le NLP apporte un signal que les comptages n'ont pas » si le ton garde β > 0 à
    p placebo < 0,05 dans la régression conjointe.

(b) **Chocs (affaire Fillon, Zénith Pécresse).** Descriptif. Séries quotidiennes : ton moyen
    (moyenne mobile sur 3 jours), part d'affaires (NLP), part « affaires » par mots-clés,
    pages vues Wikipédia, sondages individuels. Référence : les 28 jours avant le choc.
    Rupture de presse : premier jour, à partir de 14 jours avant le choc, où la série passe
    au-delà de la moyenne de référence ± 2 écarts types (dans le sens défavorable). Rupture
    des sondages : premier sondage (fin de terrain), à partir de 14 jours avant le choc, à
    plus de 2 points sous la moyenne de référence. Avance = rupture des sondages − rupture de
    presse, en jours. « Le ton devance » si l'avance du ton dépasse 3 jours dans les deux cas.
    Les autres chocs d'events.yaml dans les fenêtres sont traités de même.

(c) **Agenda.** Part hebdomadaire des titres classés (tous candidats du cycle) portant les
    thèmes de chaque hypothèse de l'étude par comptages : H1 extrême droite ← immigration +
    sécurité ; H2 gauche totale ← pouvoir d'achat. y = part du bloc(t + 2) − part(t) sur
    x2(t), élan du bloc sur 2 semaines, effet fixe par cycle, HAC (2 retards) ; placebo par
    décalage. Critère : β > 0, p placebo < 0,05 en commun 2017-2022, β > 0 dans chaque
    cycle. Même spécification avec la saillance par mots-clés (press_mediacloud) sur les
    mêmes semaines, et corrélation hebdomadaire des deux mesures par thème (validité).

(d) **Erreur finale** (2017, 2022, candidats officiels) : surperformance = résultat −
    moyenne des sondages des 14 derniers jours ; prédicteurs : ton sur 14 et 28 jours, part
    d'attaques, part d'affaires et part de titres négatifs sur 28 jours ; comparaison : log du
    rapport part de voix (comptage) / part dans les sondages sur 14 jours. Effet fixe par
    élection, p par permutation dans chaque élection (20 000). Critère : p < 0,05 et même
    signe en 2017 et en 2022. Deux élections : à prendre avec des pincettes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention as A
import press as PS  # remplace aussi A.placebo_p (décalages distincts)

from vvs import paths
from vvs.ingest.press import mediacloud_counts as MC
from vvs.ingest.press import mediacloud_titles as MT
from vvs.ingest.press.outlets import load_queries
from vvs.model.bounce import load_events
from vvs.model.plots import INK, MUTED, RESULT, SERIES, _style, display_name
from vvs.nlp import features as F
from vvs.nlp import labels as L
from vvs.nlp import server as S
from vvs.nlp.client import ChatClient
from vvs.nlp.prompts import load_prompts
from vvs.studies import panel as P
from vvs.studies import press_panel as PP

VERSION, ALT, V1 = "title-v3", "title-v3-alt", "title-v1"
CYCLES, TRAIN = (2017, 2022, 2027), (2017, 2022)
MIN_LABELED = 10
HORIZONS, H_MAIN = (1, 2, 3, 4), 2
MIN_SHIFT = 3
GRANGER_ORDER = 2
SAMPLE_PER_CYCLE = 100
THRESHOLDS = {"tone_kappa": 0.6, "stance_kappa": 0.5, "themes_f1": 0.6}
SHOCKS = {"fillon_affair_2017": ("fillon_francois", dt.date(2017, 1, 25)),
          "pecresse_zenith_2022": ("pecresse_valerie", dt.date(2022, 2, 13))}
REF_DAYS, SEARCH_FROM, POLL_DROP = 28, 14, 2.0
H1 = ("far_right", ("immigration", "security"))
H2 = ("left_total", ("purchasing_power",))
SEED = 20261022
TODAY = dt.date.today().isoformat()
RUN = paths.RUNS / f"{TODAY}-press-nlp"
VARS = {"tone": +1, "attack_share": -1, "scandal_share": -1, "neg_share": -1}


def log(msg: str) -> None:
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------------------- collecte


def title_plan():
    counts = pl.read_parquet(paths.CURATED / MC.OUTPUT_NAME)
    return MT.plan(load_queries(), counts)


def cmd_plan(args) -> None:
    reqs = title_plan()
    s = MT.plan_summary(reqs)
    with pl.Config(tbl_rows=60):
        print(s)
    print(s.group_by("cycle").agg(pl.col("est_stories", "est_pages").sum()).sort("cycle"))
    n = sum(r.est_pages for r in reqs)
    print(f"total : {len(reqs)} séries, {n} requêtes estimées, "
          f"{n * MT.MIN_INTERVAL_S / 3600:.1f} h à {MT.MIN_INTERVAL_S:.0f} s")


def cmd_collect(args) -> None:
    reqs = title_plan()
    log(f"plan : {len(reqs)} séries, {sum(r.est_pages for r in reqs)} requêtes estimées")
    stats = MT.collect(reqs, log=log)
    log(f"bilan : {stats['stories']} titres, {stats['requests']} requêtes réseau, "
        f"{len(stats['failed'])} échecs {stats['failed']}")
    cmd_parse(args)


def cmd_parse(args) -> None:
    df = MT.parse_titles(load_queries(), log=log)
    log(f"{MT.OUTPUT_NAME} : {df.height} lignes, {df['in_title'].sum()} avec le nom dans le titre")


# ---------------------------------------------------------------------------- classification


def load_titles() -> pl.DataFrame:
    return pl.read_parquet(paths.CURATED / MT.OUTPUT_NAME)


def universe(titles: pl.DataFrame) -> pl.DataFrame:
    return F.cap_sample(titles)


def server_client() -> ChatClient:
    c = ChatClient()
    if not c.health():
        raise SystemExit("llama-server injoignable sur 127.0.0.1:8765 : le lancer avec\n  "
                         + S.shell_command())
    return c


def record(kind: str, payload: dict) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "classify_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"),
                            "kind": kind, **payload}, default=str) + "\n")


def cmd_classify(args) -> None:
    P_ = load_prompts()
    model = P_.model["id"]
    u = universe(load_titles()).filter(pl.col("cycle").is_in(args.cycles))
    used, _ = S.gpu_memory_mib()
    log(f"univers : {u.height} titres ({args.cycles}), VRAM utilisée {used} Mio")
    c = server_client()
    t0 = time.perf_counter()
    st = L.classify_missing(u, P_.get(args.version), c, model, workers=args.workers, log=log)
    wall = time.perf_counter() - t0
    log(f"bilan : {st['labeled']} classés, {st['errors']} erreurs, {st['rate']:.2f} titres/s "
        f"(attente serveur), {wall / 60:.1f} min au total")
    record("classify", {"version": args.version, "cycles": args.cycles, "todo": st["todo"],
                        "labeled": st["labeled"], "errors": st["errors"],
                        "seconds": st["seconds"], "wall_s": wall, "rate": st["rate"],
                        "workers": args.workers, "vram_used_mib": used})
    L.consolidate()


def draw_sample(u: pl.DataFrame, per_cycle: int = SAMPLE_PER_CYCLE) -> pl.DataFrame:
    """`per_cycle` titres par cycle, parts égales entre candidats (le reste aux premiers dans
    l'ordre du registre), ordre tiré par empreinte de story_id."""
    out = []
    for e in CYCLES:
        g = u.filter(pl.col("cycle") == e)
        cands = [c for c in MT.CANDIDATES[e] if c in set(g["candidate_id"])]
        if not cands:
            continue
        avail = dict(g.group_by("candidate_id").len().iter_rows())
        quota = {c: 0 for c in cands}
        left = per_cycle
        while left > 0 and any(quota[c] < avail[c] for c in cands):
            for c in cands:
                if left and quota[c] < avail[c]:
                    quota[c] += 1
                    left -= 1
        for c, q in quota.items():
            gc = (g.filter(pl.col("candidate_id") == c)
                  .with_columns(pl.col("story_id").map_elements(
                      lambda s: F._h("sample:" + s), return_dtype=pl.Utf8).alias("_k"))
                  .sort("_k").head(q).drop("_k"))
            out.append(gc)
    return pl.concat(out)


def kappa(a: list, b: list) -> float:
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    cats = sorted(set(a) | set(b), key=str)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def agreement(x: pd.DataFrame, y: pd.DataFrame, themes) -> dict:
    """x, y : (story_id, candidate_id, tone, stance, themes) ; x en référence."""
    m = x.merge(y, on=["story_id", "candidate_id"], suffixes=("_a", "_b"))
    tp = fp_ = fn = 0
    per_theme = {}
    for t in themes:
        ia = m["themes_a"].map(lambda s, t=t: t in list(s)).to_numpy()
        ib = m["themes_b"].map(lambda s, t=t: t in list(s)).to_numpy()
        tp += int((ia & ib).sum())
        fp_ += int((~ia & ib).sum())
        fn += int((ia & ~ib).sum())
        per_theme[t] = {"n_a": int(ia.sum()), "n_b": int(ib.sum()),
                        "kappa": kappa(list(ia), list(ib)) if ia.any() or ib.any() else None}
    return {"n": len(m),
            "tone_agree": float((m["tone_a"] == m["tone_b"]).mean()),
            "tone_kappa": kappa(list(m["tone_a"]), list(m["tone_b"])),
            "tone_opposite": float(((m["tone_a"] * m["tone_b"]) < 0).mean()),
            "stance_agree": float((m["stance_a"] == m["stance_b"]).mean()),
            "stance_kappa": kappa(list(m["stance_a"]), list(m["stance_b"])),
            "themes_exact": float((m["themes_a"].map(lambda s: tuple(sorted(s)))
                                   == m["themes_b"].map(lambda s: tuple(sorted(s)))).mean()),
            "themes_f1": 2 * tp / (2 * tp + fp_ + fn) if tp + fp_ + fn else float("nan"),
            "per_theme": per_theme}


def labels_of(version: str, model: str) -> pl.DataFrame:
    c = L.load_cache().filter((pl.col("prompt_version") == version) & (pl.col("model") == model)
                              & pl.col("error").is_null())
    return c.unique(["story_id", "candidate_id"], keep="first")


def cmd_sample(args) -> None:
    P_ = load_prompts()
    model = P_.model["id"]
    titles = load_titles()
    s = draw_sample(universe(titles))
    log(f"échantillon : {s.height} titres")
    c = server_client()
    for v in (VERSION, ALT):
        st = L.classify_missing(s, P_.get(v), c, model, workers=args.workers, log=log)
        record("sample", {"version": v, "todo": st["todo"], "labeled": st["labeled"],
                          "errors": st["errors"], "seconds": st["seconds"], "rate": st["rate"]})
    L.consolidate()
    a, b = labels_of(VERSION, model), labels_of(ALT, model)
    keys = s.select("story_id", "candidate_id")
    cols = ["story_id", "candidate_id", "tone", "stance", "themes"]
    la = a.join(keys, on=["story_id", "candidate_id"]).select(cols).to_pandas()
    lb = b.join(keys, on=["story_id", "candidate_id"]).select(cols).to_pandas()
    ag = agreement(la, lb, P_.themes)
    # contrôle de format : v1 (JSON, un titre) contre v3 (lots) sur les titres de la mesure
    v1 = labels_of(V1, model).select(cols).to_pandas()
    ag_fmt = agreement(v1, a.select(cols).to_pandas(), P_.themes)
    RUN.mkdir(parents=True, exist_ok=True)
    out = (s.join(a.select("story_id", "candidate_id", pl.col("tone").alias("tone_v3"),
                           pl.col("stance").alias("stance_v3"),
                           pl.col("themes").list.join(",").alias("themes_v3")),
                  on=["story_id", "candidate_id"], how="left")
           .join(b.select("story_id", "candidate_id", pl.col("tone").alias("tone_alt"),
                          pl.col("stance").alias("stance_alt"),
                          pl.col("themes").list.join(",").alias("themes_alt")),
                 on=["story_id", "candidate_id"], how="left")
           .with_columns(pl.col("candidate_id").map_elements(display_name, return_dtype=pl.Utf8)
                         .alias("candidate"),
                         pl.lit("").alias("tone_manual"), pl.lit("").alias("stance_manual"),
                         pl.lit("").alias("themes_manual"), pl.lit("").alias("comment"))
           .select("cycle", "candidate", "candidate_id", "publish_date", "outlet", "title",
                   "tone_v3", "tone_alt", "stance_v3", "stance_alt", "themes_v3", "themes_alt",
                   "tone_manual", "stance_manual", "themes_manual", "comment", "url",
                   "story_id")
           .sort("cycle", "candidate_id", "publish_date"))
    out.write_csv(RUN / "sample_300.csv")
    res = {"sample": ag, "format_v1_v3": ag_fmt, "thresholds": THRESHOLDS,
           "passes": {k: bool(ag[k] >= v) for k, v in THRESHOLDS.items()}}
    (RUN / "agreement.json").write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                                   default=float), encoding="utf-8")
    log(json.dumps({k: v for k, v in ag.items() if k != "per_theme"}, default=float))
    log(f"format v1/v3 : {json.dumps({k: v for k, v in ag_fmt.items() if k != 'per_theme'})}")


def universe_labels(titles: pl.DataFrame, model: str) -> pl.DataFrame:
    """Étiquettes `VERSION` des seuls titres de l'univers plafonné (les titres classés hors
    univers, mesure de débit par exemple, n'entrent pas dans les variables)."""
    keys = universe(titles).select("story_id", "candidate_id")
    return labels_of(VERSION, model).join(keys, on=["story_id", "candidate_id"], how="semi")


def cmd_features(args) -> None:
    P_ = load_prompts()
    titles = load_titles()
    lab = universe_labels(titles, P_.model["id"])
    f = F.weekly_features(titles, lab, P_.stances, P_.themes, MT.WINDOWS)
    F.write_features(f)
    log(f"{F.OUTPUT_NAME} : {f.height} lignes, {lab.height} étiquettes {VERSION}")


# ---------------------------------------------------------------------------- panel


def x2(m: np.ndarray) -> np.ndarray:
    """Moyenne des semaines t − 1 et t (une seule si l'autre manque)."""
    prev = np.vstack([np.full((1, m.shape[1]), np.nan), m[:-1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # deux semaines vides : NaN voulu
        return np.nanmean(np.stack([m, prev]), axis=0)


class Cyc:
    """Semaines × candidats d'un cycle : sondages et variables de presse (matrices)."""

    def __init__(self, e, weeks, cands, polls, mats):
        self.e, self.weeks, self.cands, self.polls, self.mats = e, weeks, cands, polls, mats

    @property
    def T(self) -> int:
        return len(self.weeks)


def load_study():
    P_ = load_prompts()
    reg, att, poll_shares, weekly_polls, errors = A.load_inputs()
    feat = pl.read_parquet(paths.CURATED / F.OUTPUT_NAME)
    titles = load_titles()
    lab = universe_labels(titles, P_.model["id"])
    mc = pl.read_parquet(paths.CURATED / MC.OUTPUT_NAME)
    wiki = P.weekly_attention(att, "wikipedia")
    # sondages par bloc (mêmes règles que studies/press.py)
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
    # mots-clés « affaires » par candidat (comptages), à la semaine
    wk = PP.weekly_counts(mc)
    co = (wk.filter(pl.col("scope") == "candidate_theme")
          .with_columns(pl.col("id").str.split(":").list.get(0).alias("candidate_id"))
          .select("election", "candidate_id", "week", pl.col("count").alias("_sc")))
    kw = (wk.filter(pl.col("scope") == "candidate").rename({"id": "candidate_id"})
          .join(co, on=["election", "candidate_id", "week"], how="left")
          .with_columns(pl.when(pl.col("count") > 0).then(100 * pl.col("_sc") / pl.col("count"))
                        .alias("kw_scandal_share"))
          .select(pl.col("election").alias("cycle"), "candidate_id", "week",
                  "kw_scandal_share"))
    return {"P": P_, "reg": reg, "att": att, "poll_shares": poll_shares,
            "weekly_polls": weekly_polls, "errors": errors, "feat": feat, "titles": titles,
            "lab": lab, "mc": mc, "wiki": wiki, "wbloc": wbloc, "kw": kw,
            "themes_kw": PP.theme_salience(wk)}


def build_cycles(D) -> dict[int, Cyc]:
    feat = D["feat"].with_columns(
        pl.when(pl.col("n_labeled") >= MIN_LABELED).then(pl.col(c)).otherwise(None).alias(c)
        for c in ["tone", "attack_share", "scandal_share", "neg_share", "pos_share"])
    feat = feat.join(D["kw"], on=["cycle", "candidate_id", "week"], how="left")
    out = {}
    for e in CYCLES:
        f = feat.filter(pl.col("cycle") == e)
        if f.is_empty():
            continue
        weeks = sorted(f["week"].unique().to_list())
        polled = set(D["weekly_polls"].filter(pl.col("election") == e)["candidate_id"])
        cands = [c for c in MT.CANDIDATES[e] if c in set(f["candidate_id"]) and c in polled]
        grid = pl.DataFrame({"week": weeks}).join(pl.DataFrame({"candidate_id": cands}),
                                                  how="cross")

        def mat(df, col, key="candidate_id", grid=grid, cands=cands):
            m = grid.join(df.select(key, "week", col), on=[key, "week"], how="left")
            return (m.pivot(on=key, index="week", values=col).sort("week").select(cands)
                    .to_numpy().astype(float))

        polls = mat(D["weekly_polls"].filter(pl.col("election") == e), "share")
        mats = {c: mat(f, c) for c in ["tone", "attack_share", "scandal_share", "neg_share",
                                       "articles", "mentions", "kw_scandal_share"]}
        art = np.nan_to_num(mats["articles"])
        mats["sov_counts"] = 100 * art / np.where(art.sum(1, keepdims=True) > 0,
                                                  art.sum(1, keepdims=True), 1)
        w = np.nan_to_num(mat(D["wiki"], "value"))
        mats["wiki_share"] = np.where(w.sum(1, keepdims=True) > 0,
                                      100 * w / np.where(w.sum(1, keepdims=True) > 0,
                                                         w.sum(1, keepdims=True), 1), np.nan)
        out[e] = Cyc(e, weeks, cands, polls, mats)
    return out


# ---------------------------------------------------------------------------- (a) régressions


def rows_for(cycles: dict[int, Cyc], h: int) -> pd.DataFrame:
    rows = []
    for e, c in cycles.items():
        for j, cid in enumerate(c.cands):
            p = c.polls[:, j]
            for t in np.flatnonzero(~np.isnan(p)):
                tgt = A._observed(p, t, (h, h + 1, h - 1) if h > 1 else (1, 2))
                past = A._observed(p, t, (-2, -3, -1))
                if tgt is None or tgt <= t:
                    continue
                rows.append({"election": e, "candidate_id": cid, "j": j, "t": int(t),
                             "y": p[tgt] - p[t],
                             "mom": np.nan if past is None else p[t] - p[past]})
    return pd.DataFrame(rows)


def transformed(c: Cyc, var: str, shift: int = 0, perm=None) -> np.ndarray:
    m = x2(c.mats[var])
    if perm is not None:
        m = m[:, perm]
    return np.roll(m, shift, axis=0) if shift else m


def fit_rows(rows: pd.DataFrame, cycles: dict[int, Cyc], vars_: list[str],
             shifts: dict | None = None, perms: dict | None = None, shift_only: str | None = None):
    """y ~ vars (x2) + élan + effet fixe candidat × cycle ; grappes candidat × cycle."""
    if rows.empty:
        return None, 0
    xs = []
    for v in vars_:
        sh = shifts if (shift_only is None or shift_only == v) else None
        pm = perms if (shift_only is None or shift_only == v) else None
        xs.append(A.gather(rows, {e: transformed(c, v, (sh or {}).get(e, 0),
                                                 (pm or {}).get(e)) for e, c in cycles.items()}))
    y, mom = rows["y"].to_numpy(), rows["mom"].to_numpy()
    ok = ~np.isnan(y) & ~np.isnan(mom)
    for x in xs:
        ok &= ~np.isnan(x)
    r = rows[ok]
    g = (r["election"].astype(str) + ":" + r["candidate_id"]).to_numpy()
    X = A.fe_design([x[ok] for x in xs] + [mom[ok]], g)
    return A.ols_cluster(y[ok], X, g), int(ok.sum())


def size05(ps) -> float:
    """Taille réelle : part des tirages placebo où le test nominal rejette à 5 %."""
    ps = np.asarray(ps, dtype=float)
    ps = ps[~np.isnan(ps)]
    return float((ps < 0.05).mean()) if len(ps) else np.nan


def shift_draws(cycles, n, rng) -> list[dict]:
    ok = {e: c.T for e, c in cycles.items() if c.T > 2 * MIN_SHIFT + 1}
    return [{e: int(rng.integers(MIN_SHIFT, T - MIN_SHIFT + 1)) for e, T in ok.items()}
            for _ in range(n)]


def perm_draws(cycles, n, rng) -> list[dict]:
    return [{e: rng.permutation(len(c.cands)) for e, c in cycles.items()} for _ in range(n)]


def test_var(cycles, var, h, shifts, perms, extra=()) -> list[dict]:
    """β de `var` (et des `extra` en conjoint) en commun 2017-2022 et par cycle."""
    out = []
    samples = {"commun": [e for e in TRAIN if e in cycles], **{str(e): [e] for e in cycles}}
    rows_all = rows_for(cycles, h)
    for name, els in samples.items():
        cyc = {e: cycles[e] for e in els}
        rows = rows_all[rows_all["election"].isin(els)] if not rows_all.empty else rows_all
        vs = [var, *extra]
        fit, n = fit_rows(rows, cyc, vs)
        for i, v in enumerate(vs):
            b, se, t, p = A.t_and_p(fit, i)
            r = {"var": v, "h": h, "sample": name, "joint": "+".join(vs) if extra else "",
                 "beta": b, "se": se, "p": p, "n": n,
                 "n_cand": rows["candidate_id"].nunique() if n else 0}
            if shifts is not None:
                ns = [A.t_and_p(fit_rows(rows, cyc, vs, shifts=s, shift_only=v)[0], i)
                      for s in shifts]
                r["p_placebo"] = A.placebo_p(abs(t), np.array([abs(z[2]) for z in ns]))
                r["size_05"] = size05([z[3] for z in ns])
            if perms is not None:
                nm = [abs(A.t_and_p(fit_rows(rows, cyc, vs, perms=pm, shift_only=v)[0], i)[2])
                      for pm in perms]
                r["p_perm"] = A.placebo_p(abs(t), np.array(nm))
            out.append(r)
    return out


def granger(cycles: dict[int, Cyc], var: str, shifts) -> list[dict]:
    """Ordre 2 : Δsondage(t) ~ Δsondage(t−1, t−2) + var(t−1, t−2) et l'inverse."""
    k = GRANGER_ORDER
    out = []
    samples = {"commun": [e for e in TRAIN if e in cycles], **{str(e): [e] for e in cycles}}

    def fit(cyc, s):
        Ys, Xs, YL, XL, G = [], [], [], [], []
        for e, c in cyc.items():
            x = c.mats[var]
            if s:
                x = np.roll(x, s.get(e, 0), axis=0)
            dp = np.diff(c.polls, axis=0, prepend=np.nan)
            for j, cid in enumerate(c.cands):
                for t in range(k + 1, c.T):
                    yl = dp[t - k:t, j][::-1]
                    xl = x[t - k:t, j][::-1]
                    if np.isnan(dp[t, j]) or np.isnan(x[t, j]) or np.isnan(yl).any() \
                            or np.isnan(xl).any():
                        continue
                    Ys.append(dp[t, j]); Xs.append(x[t, j]); YL.append(yl); XL.append(xl)
                    G.append(f"{e}:{cid}")
        if len(Ys) < 10:
            return {"W_fwd": np.nan, "p_fwd": np.nan, "W_rev": np.nan, "p_rev": np.nan,
                    "n": len(Ys), "sum_fwd": np.nan}
        Ys, Xs, YL, XL, G = map(np.asarray, (Ys, Xs, YL, XL, G))
        f1 = A.ols_cluster(Ys, A.fe_design([*YL.T, *XL.T], G), G)
        w1, p1 = A.wald(f1, list(range(k, 2 * k)))
        f2 = A.ols_cluster(Xs, A.fe_design([*XL.T, *YL.T], G), G)
        w2, p2 = A.wald(f2, list(range(k, 2 * k)))
        s1 = float(f1[0][k:2 * k].sum()) if f1 is not None else np.nan
        return {"W_fwd": w1, "p_fwd": p1, "W_rev": w2, "p_rev": p2, "n": len(Ys), "sum_fwd": s1}

    for name, els in samples.items():
        cyc = {e: cycles[e] for e in els}
        r = fit(cyc, None) | {"var": var, "sample": name}
        nulls = [fit(cyc, s) for s in shifts]
        for d in ("fwd", "rev"):
            r[f"p_placebo_{d}"] = A.placebo_p(r[f"W_{d}"], np.array([z[f"W_{d}"] for z in nulls]))
            r[f"size_{d}"] = size05([z[f"p_{d}"] for z in nulls])
        out.append(r)
    return out


def verdict_a(anchor: pd.DataFrame, gr: pd.DataFrame, var: str = "tone") -> dict:
    a = anchor[(anchor["var"] == var) & (anchor["h"] == H_MAIN) & (anchor["joint"] == "")]
    main = a[a["sample"] == "commun"].iloc[0]
    sign = VARS[var]
    per = {s: float(a[a["sample"] == s]["beta"].iloc[0]) for s in ("2017", "2022", "2027")
           if (a["sample"] == s).any()}
    g = gr[(gr["var"] == var) & (gr["sample"] == "commun")].iloc[0]
    v = {"var": var, "beta": float(main["beta"]), "p_placebo": float(main["p_placebo"]),
         "size": float(main["size_05"]), "p_perm": float(main.get("p_perm", np.nan)),
         "beta_by_cycle": per,
         "each_expected_sign": all(np.sign(per.get(s, np.nan)) == sign for s in ("2017", "2022")),
         "granger_p_placebo": float(g["p_placebo_fwd"]),
         "granger_reverse_p_placebo": float(g["p_placebo_rev"])}
    v["holds"] = bool(np.sign(v["beta"]) == sign and v["p_placebo"] < 0.05
                      and v["each_expected_sign"] and v["granger_p_placebo"] < 0.05)
    v["replicates_2027"] = bool(np.sign(per.get("2027", np.nan)) == sign)
    return v


# ---------------------------------------------------------------------------- (b) chocs


def daily_press(D, cid: str, cycle: int) -> pd.DataFrame:
    t = D["titles"].filter((pl.col("candidate_id") == cid) & (pl.col("cycle") == cycle))
    lab = D["lab"].join(t.select("story_id", "candidate_id", "publish_date"),
                        on=["story_id", "candidate_id"])
    d = (lab.group_by("publish_date")
         .agg(pl.len().alias("n_labeled"), pl.col("tone").cast(pl.Float64).mean().alias("tone"),
              (100 * (pl.col("stance") == "scandal").mean()).alias("scandal_share"),
              (100 * (pl.col("stance") == "attack").mean()).alias("attack_share"))
         .join(t.group_by("publish_date").agg(pl.len().alias("articles"),
                                              pl.col("in_title").sum().alias("mentions")),
               on="publish_date", how="full", coalesce=True)
         .rename({"publish_date": "date"}).sort("date"))
    mc = D["mc"].filter(pl.col("election") == cycle)
    kw = (mc.filter((pl.col("scope") == "candidate") & (pl.col("id") == cid))
          .select("date", pl.col("count").alias("_c"))
          .join(mc.filter((pl.col("scope") == "candidate_theme")
                          & (pl.col("id") == f"{cid}:scandals"))
                .select("date", pl.col("count").alias("_s")), on="date", how="left")
          .select("date", (100 * pl.col("_s") / pl.col("_c")).alias("kw_scandal_share")))
    wiki = (D["att"].filter((pl.col("source") == "wikipedia") & (pl.col("candidate_id") == cid))
            .select("date", pl.col("value").alias("wiki_views")))
    d = d.join(kw, on="date", how="left").join(wiki, on="date", how="left").to_pandas()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    for c in ("tone", "scandal_share", "attack_share", "kw_scandal_share", "wiki_views"):
        w = d[c].where(d["n_labeled"].fillna(0) >= 3) if c in ("tone", "scandal_share",
                                                               "attack_share") else d[c]
        d[c + "_3d"] = w.rolling(3, min_periods=2).mean()
    return d


def first_break(d: pd.DataFrame, col: str, shock: dt.date, sign: int) -> dt.date | None:
    ref = d[(d["date"] >= shock - dt.timedelta(days=REF_DAYS)) & (d["date"] < shock)][col]
    ref = ref.dropna()
    if len(ref) < 7:
        return None
    lim = ref.mean() + sign * 2 * ref.std()
    s = d[d["date"] >= shock - dt.timedelta(days=SEARCH_FROM)].dropna(subset=[col])
    hit = s[(s[col] - lim) * sign > 0]
    return hit["date"].iloc[0] if len(hit) else None


def poll_break(D, cid: str, cycle: int, shock: dt.date, sign: int = -1
               ) -> tuple[dt.date | None, float]:
    """Premier sondage, à partir de 14 jours avant le choc, à plus de 2 points de la moyenne
    des 28 jours d'avant, dans le sens du choc (`sign`)."""
    p = (D["poll_shares"].filter((pl.col("election") == cycle) & (pl.col("candidate_id") == cid))
         .select("fieldwork_end", "share").sort("fieldwork_end").to_pandas())
    p["fieldwork_end"] = pd.to_datetime(p["fieldwork_end"]).dt.date
    ref = p[(p["fieldwork_end"] >= shock - dt.timedelta(days=REF_DAYS))
            & (p["fieldwork_end"] < shock)]["share"]
    if ref.empty:
        return None, np.nan
    s = p[p["fieldwork_end"] >= shock - dt.timedelta(days=SEARCH_FROM)]
    hit = s[(s["share"] - ref.mean()) * sign > POLL_DROP]
    return (hit["fieldwork_end"].iloc[0] if len(hit) else None), float(ref.mean())


def shocks(D) -> pd.DataFrame:
    """Chocs d'events.yaml dans les fenêtres (au moins 28 jours de référence avant), pour un
    candidat suivi. Le sens attendu suit `sign` (−1 par défaut) : ton et sondages dans ce sens,
    parts d'affaires et d'attaques, mots-clés « affaires » et pages vues à la hausse."""
    evs = {ev.id: (ev.candidate, ev.date, ev.election, ev.sign or -1) for ev in load_events()
           if ev.type == "shock" and ev.election in CYCLES}
    rows = []
    for k, (cid, d, e, sign) in evs.items():
        a, b = MT.WINDOWS[e]
        if cid not in MT.CANDIDATES.get(e, ()) or not (a + dt.timedelta(days=REF_DAYS) <= d <= b):
            continue
        dd = daily_press(D, cid, e)
        pb, pref = poll_break(D, cid, e, d, sign)
        r = {"event": k, "candidate_id": cid, "date": d, "sign": sign, "poll_break": pb,
             "poll_ref": pref}
        for col, sg in (("tone_3d", sign), ("scandal_share_3d", 1), ("attack_share_3d", 1),
                        ("kw_scandal_share_3d", 1), ("wiki_views_3d", 1)):
            br = first_break(dd, col, d, sg)
            r[f"break_{col}"] = br
            r[f"lead_{col}"] = (pb - br).days if (pb and br) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- (c) agenda


def weekly_topics(D) -> pl.DataFrame:
    """(cycle, week, theme) : part (points) des titres classés de la semaine portant le thème."""
    t = D["titles"].select("story_id", "candidate_id", "cycle",
                           P.week_start("publish_date").alias("week"))
    lab = D["lab"].join(t, on=["story_id", "candidate_id"])
    n = lab.group_by("cycle", "week").len()
    out = []
    for th in D["P"].themes:
        out.append(lab.group_by("cycle", "week").agg(
            (100 * pl.col("themes").list.contains(th).mean()).alias("share"))
            .with_columns(pl.lit(th).alias("theme")))
    return pl.concat(out).join(n, on=["cycle", "week"]).sort("cycle", "theme", "week")


def agenda(D, cycles: dict[int, Cyc], n_shifts: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    wt = weekly_topics(D)
    kw = D["themes_kw"]
    res, corr = [], []
    for hname, (bloc, keys) in (("H1", H1), ("H2", H2)):
        for src in ("nlp", "keywords"):
            series = {}
            for e, c in cycles.items():
                if src == "nlp":
                    s = (wt.filter((pl.col("cycle") == e) & pl.col("theme").is_in(list(keys)))
                         .group_by("week").agg(pl.col("share").sum()))
                else:
                    s = (kw.filter((pl.col("election") == e) & pl.col("theme").is_in(list(keys)))
                         .group_by("week").agg(pl.col("salience").sum().alias("share"),
                                               pl.len().alias("_k"))
                         .filter(pl.col("_k") == len(keys)))
                x = (pl.DataFrame({"week": c.weeks}).join(s, on="week", how="left")
                     .sort("week")["share"].to_numpy().astype(float))
                p = (pl.DataFrame({"week": c.weeks})
                     .join(D["wbloc"].filter((pl.col("election") == e)
                                             & (pl.col("bloc") == bloc)), on="week", how="left")
                     .sort("week")["share"].to_numpy().astype(float))
                series[e] = Cyc(e, c.weeks, [bloc], p[:, None], {"x": x[:, None]})
            for name, els in {"commun": list(TRAIN), **{str(e): [e] for e in series}}.items():
                cyc = {e: series[e] for e in els if e in series}
                rows = rows_for(cyc, H_MAIN)

                def fit(s, rows=rows, cyc=cyc):
                    if rows.empty:
                        return None, 0
                    x = A.gather(rows, {e: transformed(c, "x", (s or {}).get(e, 0))
                                        for e, c in cyc.items()})
                    y, mom = rows["y"].to_numpy(), rows["mom"].to_numpy()
                    ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(mom)
                    g = rows["election"].to_numpy()[ok]
                    X = A.fe_design([x[ok], mom[ok]], g)
                    return PS.hac_ols(y[ok], X, g, 2), int(ok.sum())

                f0, n = fit(None)
                b, se, t, p = A.t_and_p(f0, 0)
                sh = shift_draws(cyc, n_shifts, rng)
                ns = [A.t_and_p(fit(s)[0], 0) for s in sh]
                res.append({"hyp": hname, "bloc": bloc, "themes": "+".join(keys), "source": src,
                            "sample": name, "beta": b, "se": se, "p": p, "n": n,
                            "p_placebo": A.placebo_p(abs(t), np.array([abs(z[2]) for z in ns])),
                            "size_05": size05([z[3] for z in ns])})
    for e in cycles:
        for th in D["P"].themes:
            a = wt.filter((pl.col("cycle") == e) & (pl.col("theme") == th)).select("week", "share")
            b = kw.filter((pl.col("election") == e) & (pl.col("theme") == th)).select(
                "week", "salience")
            m = a.join(b, on="week").drop_nulls()
            if m.height >= 5:
                corr.append({"cycle": e, "theme": th, "weeks": m.height,
                             "corr": float(np.corrcoef(m["share"], m["salience"])[0, 1]),
                             "nlp_mean": float(m["share"].mean()),
                             "kw_mean": float(m["salience"].mean())})
    return pd.DataFrame(res), pd.DataFrame(corr)


# ---------------------------------------------------------------------------- (d) erreur finale


def final_frame(D) -> pd.DataFrame:
    reg, errors = D["reg"], D["errors"]
    pe = errors.filter((pl.col("method") == "mean_14d") & (pl.col("level") == "candidate")
                       & (pl.col("round") == 1)).to_pandas()
    rows = []
    for e in TRAIN:
        vote = reg.first_rounds[e]
        officials = [c.candidate_id for c in A._official(e)]
        t = D["titles"].filter((pl.col("cycle") == e) & pl.col("candidate_id").is_in(officials))
        lab = D["lab"].join(t.select("story_id", "candidate_id", "publish_date"),
                            on=["story_id", "candidate_id"])

        def agg(days, lab=lab, vote=vote):
            lo = vote - dt.timedelta(days=days)
            g = lab.filter(pl.col("publish_date") >= lo).group_by("candidate_id").agg(
                pl.col("tone").cast(pl.Float64).mean().alias("tone"),
                (100 * (pl.col("stance") == "attack").mean()).alias("attack"),
                (100 * (pl.col("stance") == "scandal").mean()).alias("scandal"),
                (100 * (pl.col("tone") == -1).mean()).alias("neg"), pl.len().alias("n"))
            return {r["candidate_id"]: r for r in g.iter_rows(named=True)}

        a14, a28 = agg(14), agg(28)
        lo = vote - dt.timedelta(days=A.FINAL_DAYS)
        art = t.filter(pl.col("publish_date") >= lo).group_by("candidate_id").len()
        tot = art["len"].sum()
        sov = {r[0]: 100 * r[1] / tot for r in art.iter_rows()}
        p = pe[pe["election"] == e].rename(columns={"unit": "candidate_id"})
        d = pd.DataFrame([{"candidate_id": c, "tone14": a14.get(c, {}).get("tone"),
                           "tone28": a28.get(c, {}).get("tone"),
                           "attack28": a28.get(c, {}).get("attack"),
                           "scandal28": a28.get(c, {}).get("scandal"),
                           "neg28": a28.get(c, {}).get("neg"),
                           "n28": a28.get(c, {}).get("n"), "sov14": sov.get(c)}
                          for c in officials])
        num = ["tone14", "tone28", "attack28", "scandal28", "neg28", "n28", "sov14"]
        d[num] = d[num].astype(float)
        d = d.merge(p[["candidate_id", "poll_share", "result_share"]], on="candidate_id")
        d["surperf"] = d["result_share"] - d["poll_share"]
        d["poll_norm"] = 100 * d["poll_share"] / d["poll_share"].sum()
        d["log_ratio_counts"] = np.log(d["sov14"].astype(float) / d["poll_norm"])
        d["election"] = e
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------- figures


def fig_tone(cycles: dict[int, Cyc], out: Path) -> None:
    for e, c in cycles.items():
        top = np.argsort(-np.nan_to_num(np.nanmean(c.polls, axis=0)), kind="stable")[:8]
        fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), sharex=True)
        w = pd.to_datetime(pd.Series(c.weeks))
        for ax, j in zip(axes.flat, top):
            ax.plot(w, x2(c.mats["tone"])[:, j], color=SERIES, lw=1.2, label="ton (2 sem.)")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax2 = ax.twinx()
            ax2.plot(w, c.polls[:, j], "o", color=RESULT, ms=3, label="sondages")
            ax.set_title(display_name(c.cands[j]), fontsize=9, color=INK)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            _style(ax)
            ax2.tick_params(labelsize=7, colors=RESULT)
        axes.flat[0].legend(frameon=False, fontsize=7, loc="lower left")
        fig.suptitle(f"Cycle {e} : ton moyen des titres (NLP, −1 à 1) et sondages (points, "
                     "axe de droite)", color=INK)
        fig.tight_layout()
        fig.savefig(out / f"tone_{e}.png", dpi=150)
        plt.close(fig)


def fig_shock(D, out: Path) -> None:
    for k, (cid, d) in SHOCKS.items():
        e = 2017 if d.year <= 2017 else 2022
        dd = daily_press(D, cid, e)
        p = (D["poll_shares"].filter((pl.col("election") == e) & (pl.col("candidate_id") == cid))
             .select("fieldwork_end", "share").to_pandas())
        fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
        x = pd.to_datetime(dd["date"])
        axes[0].plot(x, dd["tone_3d"], color=SERIES, label="ton NLP (3 jours)")
        axes[0].axhline(0, color=MUTED, lw=0.6)
        axes[0].set_ylabel("ton")
        axes[1].plot(x, dd["scandal_share_3d"], color="#e34948", label="posture « affaire » (NLP)")
        axes[1].plot(x, dd["kw_scandal_share_3d"], color=MUTED, ls="--",
                     label="mots-clés « affaires »")
        axes[1].set_ylabel("% des titres / articles")
        axes[2].plot(pd.to_datetime(p["fieldwork_end"]), p["share"], "o", color=RESULT, ms=3,
                     label="sondages")
        ax3 = axes[2].twinx()
        ax3.plot(x, dd["wiki_views_3d"], color=SERIES, lw=0.8, alpha=0.6,
                 label="Wikipédia (vues, 3 j)")
        axes[2].set_ylabel("intentions (%)")
        ax3.legend(frameon=False, fontsize=7, loc="upper right")
        for ax in axes:
            ax.axvline(pd.Timestamp(d), color=MUTED, ls=":", lw=1)
            ax.legend(frameon=False, fontsize=7, loc="upper left")
            _style(ax)
        axes[2].set_xlim(pd.Timestamp(d - dt.timedelta(days=REF_DAYS + 3)),
                         pd.Timestamp(d + dt.timedelta(days=42)))
        fig.suptitle(f"{display_name(cid)} autour du {d:%d/%m/%Y}", color=INK)
        fig.tight_layout()
        fig.savefig(out / f"shock_{k}.png", dpi=150)
        plt.close(fig)


def fig_agenda(D, cycles, out: Path) -> None:
    wt = weekly_topics(D)
    fig, axes = plt.subplots(1, len(cycles), figsize=(5.2 * len(cycles), 4))
    cmap = plt.get_cmap("tab10")
    for ax, e in zip(np.atleast_1d(axes), cycles):
        g = wt.filter(pl.col("cycle") == e)
        for i, th in enumerate(D["P"].themes):
            s = g.filter(pl.col("theme") == th).sort("week")
            ax.plot(pd.to_datetime(pd.Series(s["week"].to_list())), s["share"].to_list(),
                    lw=1, color=cmap(i), label=th)
        ax.set_title(str(e), color=INK)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("% des titres classés de la semaine")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7, loc="upper left",
                                   bbox_to_anchor=(1, 1))
    fig.suptitle("Agenda des titres de candidats, thèmes classés par le modèle", color=INK)
    fig.tight_layout()
    fig.savefig(out / "agenda_nlp.png", dpi=150)
    plt.close(fig)


def fig_final(d: pd.DataFrame, out: Path) -> None:
    es = sorted(d["election"].unique())
    fig, axes = plt.subplots(1, len(es), figsize=(4.4 * len(es), 3.6), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), es):
        g = d[(d["election"] == e) & d["tone28"].notna()]
        ax.scatter(g["tone28"], g["surperf"], color=SERIES, s=18)
        for r in g.itertuples(index=False):
            ax.annotate(display_name(r.candidate_id).split()[-1], (r.tone28, r.surperf),
                        fontsize=7, color=MUTED, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(str(e), color=INK)
        ax.set_xlabel("ton moyen des titres, 28 derniers jours")
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("résultat − sondages 14 jours (pts)")
    fig.tight_layout()
    fig.savefig(out / "final_error_tone.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- rapport

f, fp, md_table = A.f, A.fp, A.md_table


def anchor_md(a: pd.DataFrame) -> list[str]:
    return md_table(["variable", "h", "échantillon", "β", "é.-t.", "p nominal", "p placebo",
                     "taille", "p permutation", "n", "candidats"],
                    [[x.var, str(x.h), x.sample, f(x.beta), f(x.se, "{:.3f}"), fp(x.p),
                      fp(getattr(x, "p_placebo", np.nan)), f(getattr(x, "size_05", np.nan),
                                                             "{:.2f}"),
                      fp(getattr(x, "p_perm", np.nan)), str(x.n), str(x.n_cand)]
                     for x in a.itertuples(index=False)])


def cmd_study(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = load_study()
    cyc = build_cycles(D)
    log(f"cycles : { {e: (c.T, len(c.cands)) for e, c in cyc.items()} }")
    sh = shift_draws(cyc, args.shifts, rng)
    pm = perm_draws(cyc, args.shifts, rng)
    R: dict = {}
    # (a)
    rows = []
    for var in ("tone", "attack_share", "scandal_share", "neg_share"):
        for h in HORIZONS:
            main = var == "tone" or h == H_MAIN
            rows += test_var(cyc, var, h, sh if main else None, pm if h == H_MAIN else None)
        log(f"  (a) {var} fait")
    for var in ("sov_counts", "kw_scandal_share", "wiki_share"):
        rows += test_var(cyc, var, H_MAIN, sh, None)
    rows += test_var(cyc, "tone", H_MAIN, sh, None, extra=("sov_counts", "wiki_share"))
    rows += test_var(cyc, "attack_share", H_MAIN, sh, None, extra=("sov_counts", "wiki_share"))
    R["anchor"] = pd.DataFrame(rows)
    R["anchor"].to_csv(out / "anchor.csv", index=False)
    gr = []
    for var in (*VARS, "sov_counts", "kw_scandal_share", "wiki_share"):
        gr += granger(cyc, var, sh)
    R["granger"] = pd.DataFrame(gr)
    R["granger"].to_csv(out / "granger.csv", index=False)
    R["verdicts"] = {v: verdict_a(R["anchor"], R["granger"], v) for v in VARS}
    log("  (a) fait")
    # (b)
    R["shocks"] = shocks(D)
    R["shocks"].to_csv(out / "shocks.csv", index=False)
    # (c)
    R["agenda"], R["agenda_corr"] = agenda(D, cyc, args.shifts, rng)
    R["agenda"].to_csv(out / "agenda.csv", index=False)
    R["agenda_corr"].to_csv(out / "agenda_corr.csv", index=False)
    log("  (c) fait")
    # (d)
    fin = final_frame(D)
    fin.to_csv(out / "final.csv", index=False)
    R["final"] = PS.final_tests(fin, ["tone14", "tone28", "attack28", "scandal28", "neg28",
                                      "log_ratio_counts"], rng)
    R["final"].to_csv(out / "final_tests.csv", index=False)
    # figures
    fig_tone(cyc, out)
    fig_shock(D, out)
    fig_agenda(D, cyc, out)
    fig_final(fin, out)
    (out / "summary.json").write_text(json.dumps(
        {"verdicts": R["verdicts"], "cycles": {e: [c.T, len(c.cands)] for e, c in cyc.items()},
         "shifts": args.shifts}, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    write_tables(out, D, cyc, R)
    log(f"tables écrites dans {out}")


def write_tables(out: Path, D, cyc, R) -> None:
    """Tableaux en Markdown (tables.md) ; le rapport rédigé les reprend."""
    L_ = ["# Tableaux de l'étude NLP", ""]
    feat = D["feat"]
    cov = (feat.group_by("cycle", "candidate_id")
           .agg(pl.col("articles").sum(), pl.col("mentions").sum(), pl.col("n_labeled").sum(),
                pl.col("tone").filter(pl.col("n_labeled") > 0).mean().alias("tone_w"))
           .sort("cycle", "articles", descending=[False, True]))
    tone_all = D["lab"].join(D["titles"].select("story_id", "candidate_id", "cycle"),
                             on=["story_id", "candidate_id"]).group_by(
        "cycle", "candidate_id").agg(pl.col("tone").cast(pl.Float64).mean().alias("tone"),
                                     (100 * (pl.col("stance") == "attack").mean()).alias("att"),
                                     (100 * (pl.col("stance") == "scandal").mean()).alias("sc"))
    cov = cov.join(tone_all, on=["cycle", "candidate_id"], how="left")
    L_ += ["## Couverture", ""]
    L_ += md_table(["cycle", "candidat", "articles", "titres nommant", "classés", "ton moyen",
                    "% attaques", "% affaires"],
                   [[str(r["cycle"]), display_name(r["candidate_id"]), f"{r['articles']:,}"
                     .replace(",", " "), f"{r['mentions']:,}".replace(",", " "),
                     f"{r['n_labeled']:,}".replace(",", " "), f(r["tone"], "{:+.2f}"),
                     f(r["att"], "{:.1f}"), f(r["sc"], "{:.1f}")]
                    for r in cov.iter_rows(named=True)])
    L_ += ["", "## (a) Régressions", ""]
    L_ += anchor_md(R["anchor"])
    L_ += ["", "### Granger (ordre 2)", ""]
    L_ += md_table(["variable", "échantillon", "n", "Σβ presse→sond.", "p", "p placebo",
                    "taille", "p sond.→presse", "p placebo", "taille"],
                   [[x.var, x.sample, str(x.n), f(x.sum_fwd), fp(x.p_fwd), fp(x.p_placebo_fwd),
                     f(x.size_fwd, "{:.2f}"), fp(x.p_rev), fp(x.p_placebo_rev),
                     f(x.size_rev, "{:.2f}")] for x in R["granger"].itertuples(index=False)])
    L_ += ["", "### Verdicts", "", "```json",
           json.dumps(R["verdicts"], indent=2, default=float, ensure_ascii=False), "```"]
    L_ += ["", "## (b) Chocs", ""]
    s = R["shocks"]
    if not s.empty:
        cols = [c for c in s.columns if c.startswith("lead_")]
        L_ += md_table(["événement", "candidat", "date", "rupture sondages", "réf. sondages"]
                       + [c.removeprefix("lead_").removesuffix("_3d") for c in cols],
                       [[r["event"], display_name(r["candidate_id"]), str(r["date"]),
                         str(r["poll_break"]), f(r["poll_ref"], "{:.1f}")]
                        + [f(r[c], "{:+.0f}") for c in cols] for _, r in s.iterrows()])
        L_ += ["", ("Avance en jours (rupture des sondages − rupture de la série ; positif = "
                    "la série rompt avant les sondages)."), ""]
        L_ += md_table(["événement"] + [c for c in s.columns if c.startswith("break_")],
                       [[r["event"]] + [str(r[c]) for c in s.columns if c.startswith("break_")]
                        for _, r in s.iterrows()])
    L_ += ["", "## (c) Agenda", ""]
    L_ += md_table(["hyp.", "bloc", "thèmes", "mesure", "échantillon", "β", "é.-t.", "p",
                    "p placebo", "taille", "n"],
                   [[x.hyp, x.bloc, x.themes, x.source, x.sample, f(x.beta), f(x.se, "{:.3f}"),
                     fp(x.p), fp(x.p_placebo), f(x.size_05, "{:.2f}"), str(x.n)]
                    for x in R["agenda"].itertuples(index=False)])
    L_ += ["", "Corrélation hebdomadaire thèmes classés / mots-clés :", ""]
    L_ += md_table(["cycle", "thème", "semaines", "r", "% titres (NLP)", "% articles (mots-clés)"],
                   [[str(x.cycle), x.theme, str(x.weeks), f(x.corr, "{:+.2f}"),
                     f(x.nlp_mean, "{:.1f}"), f(x.kw_mean, "{:.2f}")]
                    for x in R["agenda_corr"].itertuples(index=False)])
    L_ += ["", "## (d) Erreur finale", ""]
    fin = R["final"]
    L_ += md_table(["prédicteur", "β", "β 2017", "β 2022", "p permutation", "même signe", "n"],
                   [[x["predictor"], f(x["beta"]), f(x.get("beta_2017")), f(x.get("beta_2022")),
                     fp(x["p_perm"]), "oui" if x["same_sign_all"] else "non", str(x["n"])]
                    for _, x in fin.iterrows()])
    (out / "tables.md").write_text("\n".join(L_) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("collect", cmd_collect), ("parse", cmd_parse),
                     ("features", cmd_features)):
        sub.add_parser(name).set_defaults(fn=fn)
    c = sub.add_parser("classify")
    c.add_argument("--cycles", type=int, nargs="+", default=list(CYCLES))
    c.add_argument("--version", default=VERSION)
    c.add_argument("--workers", type=int, default=8)
    c.set_defaults(fn=cmd_classify)
    s = sub.add_parser("sample")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_sample)
    st = sub.add_parser("study")
    st.add_argument("--out", type=Path, default=RUN)
    st.add_argument("--shifts", type=int, default=200)
    st.set_defaults(fn=cmd_study)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
