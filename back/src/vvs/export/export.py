"""Export JSON du site (docs/08-front.md, contrat de back/src/vvs/export/schemas/).

`export_run(run, out)` lit un run de simulation (`vvs simulate`, voir vvs.model.simulate) et
écrit dans `out` (front/static/data/) exactement les fichiers du contrat :

- manifest.json, summary.json, first_round.json, second_round.json, scenarios.json,
  polls.json, economy.json, press.json, backtest.json ;
- history/<run_date>.json, jamais réécrit une fois publié (un fichier existant et différent
  est laissé tel quel et signalé), et history/index.json, régénéré à chaque export. Les
  entrées d'historique des données factices (`manifest.mock` vrai dans le dossier de sortie)
  sont retirées au premier export réel.

Sources en plus du run : registres (candidats, instituts, scénarios), sondages curated 2027,
macro.parquet, attention.parquet, press_ina.parquet, press_timelines.parquet, les titres
classés (press_titles, press_labels, press_features), les runs de backtest (premier et second
tour) et trois études : économie (`runs/<date>-economy/`, corrélations), palmarès des instituts
(`runs/<date>-pollster-ratings/`) et presse classée (`runs/<date>-press-nlp/` : accords et dates
de rupture). Ces volets sont
explicatifs : tous portent `in_forecast: false`. La validation contre les
schémas se fait côté front (`npm run validate-data`, ajv) : jsonschema n'est pas une
dépendance du back.

Arrondis : probabilités à 4 décimales, parts et quantiles à 0,1 point, reports au point.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import Registry, default_registry
from vvs.registry.scenarios import ScenariosFile, enumerate_configurations, load_scenarios

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
MODEL_VERSION = "0.4.1"  # phase 4 : mélange de scénarios ; règle de parts selon l'ajustement (IIA ou substitution, voir manifest)
QNAMES = ("q05", "q25", "q50", "q75", "q95")
EXPORT_CONFIGS = 10  # configurations résumées dans summary.json
SCENARIO_CONFIGS = 12  # configurations listées dans scenarios.json
FIRST_ROUND_CONFIGS = 3  # configurations proposées au filtre du premier tour
SERIES_MIN_PRESENCE = 0.1  # trajectoires des candidats présents au moins une fois sur dix
SERIES_MAX_CANDIDATES = 12  # par condition autre que `mix`
PAIR_MIN_PROBABILITY = 0.002
TRANSFER_MIN_PAIR = 0.03
TRANSFER_MIN_SHARE = 1.5  # points : électorats affichés dans les reports
ECONOMY_SINCE = dt.date(2012, 1, 1)
BACKTEST_FIRST_ROUND = "2026-10-20-backtest"
BACKTEST_SECOND_ROUND = "2026-10-20-second-round"
ECONOMY_STUDY = "2026-10-20-economy"
POLLSTER_RATINGS = "2026-10-20-pollster-ratings"
ATTENTION_WEEKS = 78  # profondeur du témoin d'actualité, comme la grille du premier tour
ATTENTION_MAX_CANDIDATES = 12
INA_ELECTIONS = (2002, 2007, 2012, 2017)  # sujets de JT disponibles de 2000 à 2020
GDELT_SINCE = dt.date(2017, 1, 1)
PRESS_NLP_STUDY = "2026-10-22-press-nlp"
NLP_VERSION = "title-v3"  # prompt retenu (data/registry/nlp_prompts.yaml)
NLP_CYCLE = 2027
NLP_MIN_TITLES = 10  # sous ce nombre de titres classés, ton, postures et agenda sont masqués
NLP_TONE_WEEKS = 2  # fenêtre glissante du ton
NLP_STANCE_WEEKS = 4  # « dernier mois » des postures
NLP_BOOTSTRAP = 1000
NLP_SEED = 20261022
NLP_HEADLINES = 3
NLP_STANCES = (
    ("announcement", "Annonce"), ("programme", "Programme"), ("support", "Soutien"),
    ("attack", "Attaques subies"), ("scandal", "Affaires"), ("poll", "Sondages"),
    ("other", "Autre"),
)
NLP_AGENDA_THEMES = (
    ("immigration", "Immigration"), ("purchasing_power", "Pouvoir d'achat"),
    ("international", "International"), ("security", "Sécurité"),
)
NLP_LANDMARKS = (
    # (événement de shocks.csv, libellé, résumé) : runs/<PRESS_NLP_STUDY>/report.md, partie (b)
    ("fillon_affair_2017", "Fillon 2017 : l'affaire des emplois",
     ("Le Canard enchaîné publie le 25 janvier 2017. La posture « affaire » passe de 2 % à 78 % "
      "des titres entre le 24 et le 26 janvier, le ton de −0,1 à −0,7 ; les sondages décrochent "
      "le 27, après un trou du 20 au 27. Presse et sondages enregistrent le choc ensemble.")),
    ("pecresse_zenith_2022", "Pécresse 2022 : le meeting du Zénith",
     ("Les sondages décrochent le 5 février 2022, huit jours avant le meeting du Zénith du 13 ; le "
      "ton suit le 11 et la posture « affaire » le 21. La presse suit les sondages.")),
)
NLP_LANDMARK_DAYS = (28, 42)  # jours avant et après le choc
Z90 = {"q05": -1.6449, "q25": -0.6745, "q50": 0.0, "q75": 0.6745, "q95": 1.6449}
STUDY_BLOCS = ("far_right", "right", "centre", "incumbent", "left", "far_left")
STUDY_BLOC_LABELS = {
    "far_right": "Extrême droite", "right": "Droite", "centre": "Centre",
    "incumbent": "Bloc du président sortant", "left": "Gauche", "far_left": "Gauche radicale",
    "other": "Autres",
}
MAIN_BLOCS = ("far_right", "right", "centre", "left", "far_left")

ECONOMY_NOTE = (
    "Ces séries n'entrent pas dans la prévision. L'étude du 20 octobre 2026 "
    "(docs/05-economie.md) ne trouve aucune variable économique qui fasse bouger les "
    "intentions de vote par bloc de façon distinguable du hasard, et la popularité du "
    "président est liée aux sondages sans les précéder. Elles sont montrées à titre explicatif."
)
ECONOMY_STUDY_SUMMARY = (
    "Sur 80 variations mensuelles par bloc (six cycles, de 2002 à 2027), aucune "
    "variable économique ne fait bouger les intentions de vote par bloc de façon distinguable "
    "du hasard une fois corrigés le choix du retard et l'autocorrélation des séries (placebo "
    "par décalage circulaire). Critères fixés avant lecture."
)
ECONOMY_STUDY_FINDINGS = (
    ("Chômage BIT, la question de départ : quand il monte, la droite baisse trois mois plus tard "
     "et l'extrême droite monte, mais l'effet s'efface en niveau, le profil par retard est "
     "instable et ne se retrouve pas d'une élection à l'autre (placebos p = 0,08 et 0,13)."),
    ("Prix, carburants, confiance des ménages, climat des affaires : rien. La seule paire qui "
     "passe formellement (inflation alimentaire et extrême droite) a un signe contre-intuitif, "
     "compatible avec les faux positifs attendus sur 78 paires."),
    ("Popularité du président : liée au niveau du bloc sortant (+0,32 point par point de "
     "satisfaits, même signe sur six élections), mais elle ne précède pas les sondages "
     "(placebo p = 0,59) : lien contemporain, pas prédictif."),
)
PRESS_NOTE = (
    "La presse et l'attention en ligne n'entrent pas dans la prévision. L'attention Wikipédia "
    "a été testée (docs/06-presse.md) : elle précède les sondages d'une à quatre semaines "
    "autour des événements, mais l'effet disparaît hors événements et la causalité inverse est "
    "aussi significative ; elle n'est pas retenue. Les titres classés par un modèle local (ton, "
    "posture, thèmes) ont été testés à leur tour : résultat négatif, ils sont montrés à titre "
    "descriptif."
)
NLP_NOTE = (
    "Environ 58 000 titres de presse (Media Cloud, campagnes 2017, 2022 et 2027) classés par un "
    "modèle de langue local : ton, posture, thèmes. Volet "
    "descriptif : il sépare une couverture abondante et favorable d'une couverture abondante et "
    "hostile, ce que les comptages confondent."
)
NLP_VERDICT = (
    ("Le ton a le signe attendu dans les trois campagnes, mais son effet sur les sondages des "
     "deux semaines suivantes ne se distingue pas du hasard (placebo p = 0,59)."),
    ("Autour des chocs, la presse enregistre en même temps que les sondages (Fillon 2017) ou "
     "après eux (Pécresse 2022), et l'agenda par thèmes classés ne précède rien."),
)
NLP_OUT_OF_SAMPLE = (
    "Test hors échantillon déclaré pour 2027 : pour chaque candidat officiel, ton moyen des 28 "
    "jours avant le premier tour (même prompt, même modèle) contre résultat moins moyenne des "
    "sondages des 14 derniers jours ; prédiction : un meilleur ton va avec une surperformance "
    "(β > 0). En 2017 et 2022 : β = +6, p = 0,06, n = 23, porté par 2022."
)
NLP_TONE_NOTE = (
    "Ton moyen des titres qui nomment le candidat, de −1 (hostile) à +1 (favorable), sur deux "
    "semaines glissantes ; bande : intervalle à 90 % par rééchantillonnage des titres. Masqué sous "
    "10 titres classés. Le ton est la mesure la moins fiable des trois (accord κ = 0,60 entre deux "
    "formulations du prompt)."
)
NLP_STANCE_NOTE = (
    "Posture de chaque titre vis-à-vis du candidat, une seule par titre. « Attaques subies » : "
    "le candidat est la cible d'une critique ; « annonce » comprend ses propres attaques contre "
    "un adversaire. Masqué sous 10 titres classés."
)
NLP_HEADLINES_NOTE = (
    "Titres classés hostiles (−1) et favorables (+1) de la semaine, les plus récents d'abord : "
    "le ton n'a que trois niveaux, il n'y a pas de « plus négatif » parmi eux. Un titre peut être "
    "mal classé."
)
NLP_SOV_NOTE = (
    "Part de voix : titres qui nomment le candidat, en part des titres nommant l'un des candidats "
    "suivis (un titre qui en nomme deux compte pour chacun). Attention : pages vues Wikipédia des "
    "mêmes candidats, en part de leur total, mêmes semaines."
)
NLP_AGENDA_NOTE = (
    "Part des titres classés de la semaine (tous candidats suivis) qui portent le thème ; un "
    "titre peut en porter plusieurs. Seuls les quatre thèmes où deux formulations du prompt "
    "s'accordent (κ ≥ 0,6) sont montrés ; « institutions », « affaires » et les autres ne sont "
    "pas fiables."
)
NLP_LANDMARKS_NOTE = (
    "Deux chocs de campagnes passées, classés avec le même modèle et le même prompt : ton et part "
    "de la posture « affaire » jour par jour (moyenne sur trois jours, jours de moins de 3 titres "
    "écartés). Les traits verticaux marquent le choc et la première rupture de chaque série ; "
    "rupture des sondages : premier sondage à plus de 2 points sous la moyenne des 28 jours "
    "d'avant."
)
ATTENTION_NOTE = (
    "Témoin d'actualité, n'entre pas dans la prévision : pages vues des articles Wikipédia des "
    "candidats suivis pour 2027, en part du total de la semaine. Un pic signale un événement, "
    "pas un mouvement d'opinion."
)
INA_NOTE = (
    "Contexte d'agenda, n'entre pas dans la prévision : sujets des journaux télévisés (TF1, "
    "France 2, France 3, Arte, M6) classés par rubrique par l'INA, de 2000 à 2020. Part des "
    "sujets de l'année. Pas de 2022, pas de titres de sujets, sécurité mal isolée."
)
GDELT_NOTE = (
    "N'entre pas dans la prévision. GDELT cherche dans une traduction anglaise automatique des "
    "articles et le volume surveillé s'effondre (624 000 articles par jour en 2017, 103 000 en "
    "2026) : seule la part du volume est comparable d'une année à l'autre. Trois candidats "
    "seulement, faute de débit (une requête réussie toutes les 10 à 20 minutes)."
)
RATINGS_NOTE = "Classement non reproductible d'une élection à l'autre, n'entre pas dans la prévision."
RATINGS_LABELS = {"verian": "Verian (Sofres, Kantar)"}
VARIABLES_TESTED = (
    # (id, libellé, verdict, publiée, une ligne) : docs/backtest-log.md, vague du 2026-10-20
    ("substitution", "Substitution intra-bloc (logit emboîté)", "non admise", True,
     ("Meilleure sur les hypothèses emboîtées de 2022 et 2027, moins bonne en 2017 ; log-score "
      "inchangé à 90 jours, en baisse à 180. Utilisée par la projection publiée par choix "
      "éditorial, la variante sans substitution reste calculée.")),
    ("bounce", "Rebonds de primaire et d'annonce", "non admise", False,
     ("Seules les primaires ont une forme commune (saut de 2 à 10 points, entièrement érodé au "
      "vote) ; le gain à 90 jours tient au seul cas Pécresse 2022.")),
    ("pollster_ratings", "Palmarès des instituts", "non admise", False,
     ("Précision et effets maison non reproductibles d'une élection à l'autre (r ≈ 0,1) ; gain "
      "sur une élection sur trois.")),
    ("potential", "Potentiel électoral et rejet (second tour)", "non admise", False,
     "Pas un plafond : les votes de barrage le dépassent ; CRPS inchangé à 180 et 90 jours."),
    ("certainty", "Électeurs sûrs de leur choix", "non admise", False,
     ("Lien avec le mouvement ultérieur de signe opposé en 2017 et en 2022 ; CRPS et couverture "
      "dégradés au backtest.")),
    ("attention", "Attention en ligne (Wikipédia, Google Trends)", "non retenue", False,
     ("Wikipédia précède les sondages autour des événements seulement, effet nul hors événements "
      "et aux horizons de la règle ; Trends ne réplique pas.")),
    ("press_nlp", "Presse classée (ton, posture, thèmes)", "non retenue", False,
     ("Titres classés par un modèle local (2026-10-22) : le ton a le signe attendu mais un effet "
      "indistinguable du placebo, la presse bouge avec les sondages ou après eux, l'agenda par "
      "thèmes ne tient pas. Pas essayée au backtest ; test hors échantillon déclaré pour 2027.")),
)
UNIT_LABELS = {
    "percent": "%",
    "persons": "personnes",
    "index_2025_100": "indice 100 = 2025",
    "index_mean_100": "indice, moyenne = 100",
    "balance_points": "solde, points",
    "eur_per_litre": "€ par litre",
}
FREQUENCY = {"M": "monthly", "Q": "quarterly", "W": "weekly"}
SOURCE_LABELS = {
    "insee": "Insee",
    "dares": "Dares",
    "mte": "Ministère de la Transition écologique",
    "ifop": "Ifop",
}
FIRST_ROUND_MODELS = {
    "model": ("Modèle + η", "model",
              ("Agrégation bayésienne des sondages (effets maison, plancher d'erreur), projection "
              "au vote par le modèle d'erreur η calibré sans l'élection évaluée.")),
    "mean14": ("Moyenne 14 jours + η", "baseline",
               "Moyenne simple des sondages des 14 derniers jours, même modèle d'erreur η."),
    "last_pollster": ("Dernier par institut + η", "baseline",
                      "Dernier sondage de chaque institut, moyenné, même modèle d'erreur η."),
    "model_noerror": ("Modèle sans η", "ablation",
                      ("Modèle de la phase 2 : projection par la seule marche aléatoire, sans "
                      "modèle d'erreur.")),
}
SECOND_ROUND_MODELS = {
    "model": ("2d tour : reports + sondages", "model",
              ("Second tour par reports de voix et sondages de duel (biais RN corrigé). Seuls "
              "le log-score (log P du vainqueur réel, duel réel connu) et la couverture du score "
              "du vainqueur sont remplis. La veille du second tour n'est pas représentée ici "
              "(horizon négatif) : erreur absolue 1,6 point (docs/backtest-log.md).")),
    "model_no_bias": ("2d tour sans biais RN", "ablation",
                      "Même modèle sans correction du biais d'extrême droite des sondages de duel."),
    "polls_14d": ("2d tour : sondages 14 jours", "baseline",
                  "Moyenne des sondages de duel des 14 derniers jours."),
    "transfers_only": ("2d tour : reports seuls", "baseline",
                       "Reports de voix historiques, sans sondages de duel."),
}


# --- utilitaires


def _r(x: float | None, nd: int = 1) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def _interval(row: dict, nd: int = 1, scale: float = 1.0, prefix: str = "") -> dict:
    q = [float(row[f"{prefix}{k}"]) * scale for k in QNAMES]
    q = np.maximum.accumulate(q)  # quantiles croissants malgré les arrondis flottants
    return {k: round(float(v), nd) for k, v in zip(QNAMES, q)}


def _write(out: Path, rel: str, obj) -> None:
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True,
                           check=True)
        return r.stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def clean_commit() -> str | None:
    """Commit courant si le code du back et les registres sont sans modification, sinon None."""
    head = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain", "--", "back/src", "data/registry")
    if head is None or status is None or status.strip():
        return None
    return head.strip()


def run_date_of(run: Path, manifest: dict) -> str:
    name = run.name
    try:
        return dt.date.fromisoformat(name[:10]).isoformat()
    except ValueError:
        return manifest["created_at"][:10]


# --- candidats


def candidate_entry(cid: str, reg: Registry, election: int, bloc_hint: dict[str, str]) -> dict:
    if cid.startswith("placeholder_"):
        bloc = cid.removeprefix("placeholder_")
        names = {
            "centre": ("Autre candidat du bloc central", "Autre (centre)"),
        }
        name, short = names.get(bloc, (f"Autre candidat ({bloc})", f"Autre ({bloc})"))
        return {"id": cid, "name": name, "short_name": short, "bloc": bloc, "party": None,
                "is_placeholder": True}
    c = reg.candidates[cid]
    bloc = bloc_hint.get(cid)
    if bloc is None:
        try:
            bloc = reg.bloc_of(cid, election)
        except KeyError:
            bloc = "other"
            for y in sorted(reg.elections, reverse=True):
                try:
                    bloc = reg.bloc_of(cid, y)
                    break
                except KeyError:
                    continue
    return {
        "id": cid,
        "name": c.name,
        "short_name": c.last_name,
        "bloc": bloc,
        "party": c.party_at(election),
        "is_placeholder": False,
    }


# --- conditions


def load_conditions(run: Path) -> list[dict]:
    return json.loads((run / "conditions.json").read_text())


def condition_json(c: dict) -> dict:
    return {
        "id": c["id"],
        "kind": c["kind"],
        "label": c["label"],
        "states": c["states"],
        "probability": round(float(c["probability"]), 4),
    }


def compatible(candidates: Iterable[str], states: dict[str, str], f: ScenariosFile) -> bool:
    """Règle de docs/08 : l'hypothèse contient les candidats de chaque état de la condition et
    aucun candidat d'un autre état de la même dimension."""
    present = set(candidates)
    for dim_id, state_id in states.items():
        d = f.dimension(dim_id)
        need = set(d.state(state_id).candidates)
        others = d.candidate_ids() - need
        if not need <= present or present & others:
            return False
    return True


# --- fichiers


def outcome_set(summary: pl.DataFrame, cond_id: str) -> dict:
    sub = summary.filter(pl.col("condition_id") == cond_id).sort(
        ["p_win", "p_qualify", "q50"], descending=True
    )
    cands = []
    for r in sub.iter_rows(named=True):
        cands.append(
            {
                "candidate_id": r["candidate_id"],
                "p_present": round(r["p_present"], 4),
                "p_qualify": round(r["p_qualify"], 4),
                "p_win": round(r["p_win"], 4),
                "first_round_share": _interval(r),
            }
        )
    return {"condition_id": cond_id, "candidates": cands}


def build_summary(run_manifest: dict, summary: pl.DataFrame, conditions: list[dict]) -> dict:
    return {
        "asof": run_manifest["asof"],
        "n_draws": run_manifest["n_draws"],
        "conditions": [condition_json(c) for c in conditions],
        "mixture": outcome_set(summary, "mix"),
        "by_condition": [outcome_set(summary, c["id"]) for c in conditions if c["id"] != "mix"],
    }


def build_first_round(
    run_manifest: dict,
    summary_json: dict,
    traj: pl.DataFrame,
    conditions: list[dict],
    polls_r1: pl.DataFrame,
    f: ScenariosFile,
) -> dict:
    asof = dt.date.fromisoformat(run_manifest["asof"])
    dates = (
        traj.filter((pl.col("condition_id") == "mix") & (pl.col("date") <= asof))
        .select("week", "date")
        .unique()
        .sort("week")
    )
    traj = traj.join(dates.select("week"), on="week")
    n_weeks = dates.height
    outcomes = {o["condition_id"]: o for o in [summary_json["mixture"], *summary_json["by_condition"]]}
    series = []
    for c in conditions:
        o = outcomes[c["id"]]
        kept = [x for x in o["candidates"] if x["p_present"] >= SERIES_MIN_PRESENCE]
        if c["id"] != "mix":
            kept = kept[:SERIES_MAX_CANDIDATES]
        t = traj.filter(pl.col("condition_id") == c["id"])
        for x in kept:
            tc = t.filter(pl.col("candidate_id") == x["candidate_id"]).sort("week")
            if tc.height != n_weeks:
                continue
            tracks = {}
            for k in QNAMES:
                tracks[k] = [_r(v) for v in tc[k].to_list()]
            # quantiles croissants semaine par semaine malgré l'arrondi
            for w in range(n_weeks):
                vals = [tracks[k][w] for k in QNAMES]
                if vals[0] is not None:
                    acc = np.maximum.accumulate(vals)
                    for k, v in zip(QNAMES, acc):
                        tracks[k][w] = round(float(v), 1)
            series.append(
                {
                    "condition_id": c["id"],
                    "candidate_id": x["candidate_id"],
                    **tracks,
                    "election_day": x["first_round_share"],
                }
            )
    other_conds = [c for c in conditions if c["id"] != "mix"]
    polls = []
    for (poll_id, hyp_id), g in polls_r1.group_by(["poll_id", "hypothesis_id"], maintain_order=True):
        ids = g["candidate_id"].to_list()
        r0 = g.row(0, named=True)
        polls.append(
            {
                "poll_id": poll_id,
                "hypothesis_id": hyp_id,
                "pollster": r0["pollster"],
                "fieldwork_end": r0["fieldwork_end"].isoformat(),
                "population": r0["population"] or "unknown",
                "conditions": [c["id"] for c in other_conds if compatible(ids, c["states"], f)],
                "shares": {cid: _r(s) for cid, s in zip(ids, g["share"].to_list())},
            }
        )
    return {
        "asof": run_manifest["asof"],
        "election_date": run_manifest["first_round"],
        "dates": [d.isoformat() for d in dates["date"].to_list()],
        "conditions": [condition_json(c) for c in conditions],
        "series": series,
        "polls": polls,
    }


def build_second_round(
    run_manifest: dict,
    pairs: pl.DataFrame,
    transfers: pl.DataFrame,
    draws: pl.DataFrame,
    polls_r2: pl.DataFrame,
    bloc_of: dict[str, str],
) -> dict:
    mix = pairs.filter((pl.col("condition_id") == "mix") & (pl.col("p_pair") >= PAIR_MIN_PROBABILITY))
    out_pairs, out_transfers = [], []
    orient = {}
    for r in mix.sort("p_pair", descending=True).iter_rows(named=True):
        a, b = r["candidate_a"], r["candidate_b"]
        orient[frozenset((a, b))] = (a, b)
        out_pairs.append(
            {
                "candidate_a": a,
                "candidate_b": b,
                "p_pair": round(r["p_pair"], 4),
                "p_a_wins": round(r["p_a_wins"], 4),
                "share_a": _interval(r),
            }
        )
        if r["p_pair"] < TRANSFER_MIN_PAIR:
            continue
        sub = draws.filter((pl.col("qualified_a") == a) & (pl.col("qualified_b") == b))
        tr = transfers.filter((pl.col("candidate_a") == a) & (pl.col("candidate_b") == b))
        groups = {t["group"]: t for t in tr.iter_rows(named=True)}
        cands = [c for c in bloc_of if c in sub.columns and c not in (a, b)]
        means = {c: float(sub[c].fill_null(0.0).mean()) for c in cands}
        for c in sorted(cands, key=lambda x: -means[x]):
            if means[c] < TRANSFER_MIN_SHARE:
                continue
            g = groups.get(f"bloc:{bloc_of[c]}")
            if g is None:
                continue
            out_transfers.append(
                {
                    "candidate_a": a,
                    "candidate_b": b,
                    "from": c,
                    "to_a": _interval(g, 0, 100.0, "to_a_"),
                    "to_b": _interval(g, 0, 100.0, "to_b_"),
                    "to_none": _interval(g, 0, 100.0, "none_"),
                }
            )
    polls = []
    for (poll_id, _hyp), g in polls_r2.group_by(["poll_id", "hypothesis_id"], maintain_order=True):
        if g.height != 2:
            continue
        x, y = g["candidate_id"].to_list()
        sx, sy = g["share"].to_list()
        a, b = orient.get(frozenset((x, y))) or (
            (x, y) if bloc_of.get(x) == "far_right" or sx >= sy else (y, x)
        )
        sa = sx if a == x else sy
        r0 = g.row(0, named=True)
        polls.append(
            {
                "poll_id": poll_id,
                "pollster": r0["pollster"],
                "fieldwork_end": r0["fieldwork_end"].isoformat(),
                "candidate_a": a,
                "candidate_b": b,
                "share_a": round(100 * sa / (sx + sy), 1),
            }
        )
    polls.sort(key=lambda p: p["fieldwork_end"], reverse=True)
    return {
        "asof": run_manifest["asof"],
        "election_date": run_manifest["second_round"],
        "pairs": out_pairs,
        "transfers": out_transfers,
        "polls": polls,
    }


def _source_ref(s: str, refs: dict[str, str]) -> dict:
    if s in refs:
        return {"label": s, "href": refs[s]}
    try:
        host = urlparse(s).hostname
    except ValueError:
        host = None
    return {"label": (host or s).removeprefix("www."), "href": s}


def scenario_history(f: ScenariosFile) -> list[dict]:
    """Probabilités des états à chaque version du registre de scénarios : versions commitées
    (git) puis version de travail ; une entrée par `as_of`, la plus récente l'emporte."""
    import yaml

    def probs(data: dict) -> dict[str, float]:
        return {
            f"{d['id']}.{s['id']}": float(s["probability"])
            for d in data["dimensions"]
            for s in d["states"]
        }

    points: dict[str, dict[str, float]] = {}
    rel = "data/registry/scenarios_2027.yaml"
    revs = (_git("log", "--format=%H", "--reverse", "--", rel) or "").split()
    for h in revs:
        txt = _git("show", f"{h}:{rel}")
        if not txt:
            continue
        try:
            data = yaml.safe_load(txt)
            points[str(data["as_of"])] = probs(data)
        except (yaml.YAMLError, KeyError, TypeError):
            continue
    points[f.as_of.isoformat()] = {
        f"{d.id}.{s.id}": s.probability for d in f.dimensions for s in d.states
    }
    return [{"date": k, "probabilities": v} for k, v in sorted(points.items())]


def build_scenarios(f: ScenariosFile) -> dict:
    refs = f.references
    enum = enumerate_configurations(f, small="marginal")
    dims = [d.id for d in f.dimensions]
    configs = []
    for c in list(enum)[:SCENARIO_CONFIGS]:
        states = dict(c.states)
        configs.append(
            {
                "condition_id": "cfg." + "-".join(states[d] for d in dims),
                "states": states,
                "probability": round(c.probability, 4),
                "candidates": sorted(c.candidates),
                "placeholders": list(c.placeholders),
                "small_presence": {k: round(v, 4) for k, v in c.small_presence.items()},
            }
        )
    return {
        "as_of": f.as_of.isoformat(),
        "election": f.election,
        "dimensions": [
            {
                "id": d.id,
                "question": d.question,
                "review": [str(x) for x in d.review],
                "states": [
                    {
                        "id": s.id,
                        "probability": s.probability,
                        "candidates": list(s.candidates),
                        "placeholders": list(s.placeholders),
                        "description": s.description,
                        "sources": [_source_ref(x, refs) for x in s.sources],
                        "condition_id": f"{d.id}.{s.id}",
                    }
                    for s in d.states
                ],
            }
            for d in f.dimensions
        ],
        "small_candidates": [
            {
                "candidate_id": s.candidate_id,
                "probability": s.probability,
                "bloc": s.bloc,
                "rationale": s.rationale,
                "sources": [_source_ref(x, refs) for x in s.sources],
            }
            for s in f.small_candidates
        ],
        "constraints": [
            {
                "id": c.id,
                "when": dict(c.when),
                "absent": list(c.absent),
                "impossible": c.impossible,
                "rationale": c.rationale,
            }
            for c in f.constraints
        ],
        "revisions": [
            {"date": r.date.isoformat(), "author": r.author, "summary": r.summary,
             "rationale": r.rationale}
            for r in sorted(f.revisions, key=lambda r: r.date, reverse=True)
        ],
        "configurations": configs,
        "n_configurations": len(enum),
        "impossible_mass": round(enum.impossible_mass, 4),
        "probability_history": scenario_history(f),
    }


def build_polls(
    run_manifest: dict, polls: pl.DataFrame, results: pl.DataFrame, house: pl.DataFrame,
    reg: Registry, ratings: dict | None = None,
) -> dict:
    out = []
    for r in polls.sort("fieldwork_end", "poll_id", descending=[True, False]).iter_rows(named=True):
        res = results.filter(pl.col("poll_id") == r["poll_id"])
        hyps = []
        for (hyp_id, rnd), g in res.group_by(["hypothesis_id", "round"], maintain_order=True):
            g = g.sort("share", descending=True)
            if g.height < 2:
                continue
            hyps.append(
                {
                    "hypothesis_id": hyp_id,
                    "round": int(rnd),
                    "population": g["population"][0] or "unknown",
                    "shares": [
                        {"candidate_id": c, "share": _r(s)}
                        for c, s in zip(g["candidate_id"].to_list(), g["share"].to_list())
                    ],
                }
            )
        if not hyps:
            continue
        size = r["sample_size"]
        out.append(
            {
                "poll_id": r["poll_id"],
                "pollster": r["pollster"],
                "sponsor": r["sponsor"],
                "fieldwork_start": r["fieldwork_start"].isoformat(),
                "fieldwork_end": r["fieldwork_end"].isoformat(),
                "publish_date": r["publish_date"].isoformat() if r["publish_date"] else None,
                "sample_size": int(size) if size and size > 0 else None,
                "mode": r["mode"] if r["mode"] in ("online", "phone", "mixed") else "unknown",
                "source_url": r["source_url"],
                "hypotheses": hyps,
            }
        )
    pollsters = sorted({p["pollster"] for p in out} | set(house["pollster"].to_list()))
    return {
        "asof": run_manifest["asof"],
        "pollsters": [
            {"id": p, "name": reg.pollsters[p].name if p in reg.pollsters else p} for p in pollsters
        ],
        "polls": out,
        "house_effects": [
            {
                "pollster": h["pollster"],
                "bloc": h["bloc"],
                "effect": _interval(h, 2),
                "n_polls": int(h["n_polls"]),
            }
            for h in house.sort("pollster", "bloc").iter_rows(named=True)
        ],
        "pollster_ratings": ratings,
    }


def build_correlations(study: Path) -> tuple[list[dict], str]:
    """Une ligne par (indicateur, bloc) de l'étude économie (`coef_pairs.parquet`) : effet au
    meilleur retard d'un écart type de variation mensuelle de l'indicateur, en points de part
    du bloc par mois, avec l'intervalle à 90 % de la régression (HAC) et le p du placebo,
    seul test qui corrige le choix du retard."""
    from vvs.studies.panel import INDICATORS

    pairs = pl.read_parquet(study / "coef_pairs.parquet")
    order = {k: i for i, k in enumerate(INDICATORS)}
    border = {b: i for i, b in enumerate(STUDY_BLOCS)}
    pairs = pairs.with_columns(
        pl.col("indicator").replace_strict(order, default=len(order)).alias("_i"),
        pl.col("bloc").replace_strict(border, default=len(border)).alias("_b"),
    ).sort("_i", "_b")
    out = []
    for r in pairs.iter_rows(named=True):
        sd, b, se = r["sd_d1"], r["best_beta"], r["best_se"]
        q = [(b + z * se) * sd for z in Z90.values()]
        ind = INDICATORS.get(r["indicator"])
        out.append(
            {
                "series_id": r["indicator"],
                "series_label": ind.label if ind else r["indicator"],
                "target": r["bloc"],
                "target_label": STUDY_BLOC_LABELS.get(r["bloc"], r["bloc"]),
                "lag_months": int(r["best_lag"]),
                "effect": {k: round(float(v), 2) for k, v in zip(QNAMES, q)},
                "p_placebo": round(float(r["p_placebo"]), 3),
                "n": int(r["best_n"]),
                "n_cycles": int(r["best_n_cycles"]),
            }
        )
    n = len(out)
    n_pass = sum(c["p_placebo"] < 0.05 for c in out)
    note = (
        f"Effet d'une variation mensuelle typique (un écart type) de l'indicateur sur la part du "
        f"bloc, en points par mois, au retard (0 à 6 mois) où il est le plus net, avec son "
        f"intervalle à 90 %. Ce retard est choisi après coup : seul le test placebo (séries "
        f"décalées dans le temps) corrige ce choix. {n_pass} paire{'s' if n_pass > 1 else ''} "
        f"sur {n} passe{'nt' if n_pass > 1 else ''} le placebo à 5 % ; le hasard en donnerait "
        f"environ {round(0.05 * n)}."
    )
    return out, note


def build_economy(asof: str, study: Path | None = None, curated: Path | None = None) -> dict:
    import yaml

    reg = yaml.safe_load((paths.REGISTRY / "macro_series.yaml").read_text(encoding="utf-8"))
    macro = pl.read_parquet((curated or paths.CURATED) / "macro.parquet")
    series = []
    for s in reg["series"]:
        sub = macro.filter(
            (pl.col("series_id") == s["series_id"])
            & (pl.col("date") >= ECONOMY_SINCE)
            & (pl.col("date") <= dt.date.fromisoformat(asof))
        ).sort("date")
        if sub.is_empty():
            continue
        series.append(
            {
                "id": s["series_id"],
                "label": s["label"],
                "unit": UNIT_LABELS.get(s["unit"], s["unit"]),
                "frequency": FREQUENCY[s["freq"]],
                "source": {"label": SOURCE_LABELS.get(s["source"], s["source"]), "href": s["url"]},
                "in_forecast": False,
                "points": [
                    {"date": d.isoformat(), "value": round(float(v), 3)}
                    for d, v in zip(sub["date"].to_list(), sub["value"].to_list())
                ],
            }
        )
    study = study or paths.RUNS / ECONOMY_STUDY
    correlations, note = build_correlations(study)
    return {
        "asof": asof,
        "in_forecast": False,
        "in_forecast_note": ECONOMY_NOTE,
        "series": series,
        "correlations": correlations,
        "correlations_note": note,
        "study": {
            "date": run_date_of(study, {"created_at": asof}),
            "verdict": "negative",
            "summary": ECONOMY_STUDY_SUMMARY,
            "findings": list(ECONOMY_STUDY_FINDINGS),
            "reference": "docs/05-economie.md",
        },
    }


def build_attention(asof: str, curated: Path, universe: list[str]) -> dict:
    """Part hebdomadaire des pages vues Wikipédia des candidats 2027 : pour chaque semaine
    complète (lundi à dimanche) des `ATTENTION_WEEKS` dernières, pages vues du candidat sur le
    total des membres du cycle 2027 (`attention.parquet`, `share` non vide), en %. Candidats de
    l'univers de la simulation, les plus vus sur les douze dernières semaines."""
    a = pl.read_parquet(curated / "attention.parquet").filter(
        (pl.col("source") == "wikipedia") & (pl.col("election") == 2027)
        & pl.col("share").is_not_null() & (pl.col("date") <= dt.date.fromisoformat(asof))
    )
    if a.is_empty():
        return {"in_forecast": False, "note": ATTENTION_NOTE, "source": "Wikipédia (pages vues)",
                "weeks": [], "tracks": []}
    a = a.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
    total = a.group_by("date", "week").agg(pl.col("value").sum().alias("total"))
    days = total.group_by("week").agg(pl.len().alias("n_days"), pl.col("total").sum())
    weeks = days.filter(pl.col("n_days") == 7).sort("week").tail(ATTENTION_WEEKS)
    by_cand = (
        a.filter(pl.col("candidate_id").is_in(universe))
        .group_by("candidate_id", "week").agg(pl.col("value").sum())
        .join(weeks.select("week", "total"), on="week")
        .with_columns((100 * pl.col("value") / pl.col("total")).alias("pct"))
    )
    week_list = weeks["week"].to_list()
    recent = set(week_list[-12:])
    rank = (
        by_cand.filter(pl.col("week").is_in(list(recent)))
        .group_by("candidate_id").agg(pl.col("pct").mean()).sort("pct", descending=True)
    )
    tracks = []
    for cid in rank["candidate_id"].to_list()[:ATTENTION_MAX_CANDIDATES]:
        vals = dict(by_cand.filter(pl.col("candidate_id") == cid).select("week", "pct").iter_rows())
        tracks.append({"candidate_id": cid,
                       "values": [_r(vals.get(w), 2) if w in vals else 0.0 for w in week_list]})
    return {
        "in_forecast": False,
        "note": ATTENTION_NOTE,
        "source": "Wikipédia (pages vues, API Wikimedia)",
        "weeks": [w.isoformat() for w in week_list],
        "tracks": tracks,
    }


def build_ina(curated: Path) -> dict:
    """Part de chaque rubrique INA dans les sujets de JT de chaque année électorale, et sur
    l'ensemble 2000-2020 pour référence, en %."""
    ina = pl.read_parquet(curated / "press_ina.parquet").with_columns(
        pl.col("date").dt.year().alias("year")
    )
    years = [y for y in INA_ELECTIONS if y in set(ina["year"].unique().to_list())]
    total = ina.group_by("theme").agg(pl.col("n_subjects").sum())
    total = total.with_columns((100 * pl.col("n_subjects") / pl.col("n_subjects").sum()).alias("pct"))
    by_year = ina.filter(pl.col("year").is_in(years)).group_by("year", "theme").agg(
        pl.col("n_subjects").sum()
    ).with_columns((100 * pl.col("n_subjects") / pl.col("n_subjects").sum().over("year")).alias("pct"))
    topics = []
    for r in total.sort("pct", descending=True).iter_rows(named=True):
        t = by_year.filter(pl.col("theme") == r["theme"])
        vals = dict(zip(t["year"].to_list(), t["pct"].to_list()))
        topics.append({"label": r["theme"], "shares": [_r(vals.get(y, 0.0), 1) for y in years],
                       "share_all_years": _r(r["pct"], 1)})
    span = (ina["year"].min(), ina["year"].max())
    return {
        "in_forecast": False,
        "note": INA_NOTE,
        "source": "INA, sujets des journaux télévisés (data.gouv.fr)",
        "years": years,
        "all_years_label": f"{span[0]}-{span[1]}",
        "topics": topics,
    }


def build_gdelt(asof: str, curated: Path) -> dict:
    """Part mensuelle des articles GDELT qui citent chaque candidat suivi (`volume_share`
    recalculé au mois : volume sur volume surveillé), en %, depuis 2017."""
    t = pl.read_parquet(curated / "press_timelines.parquet").filter(
        (pl.col("source") == "gdelt") & (pl.col("date") >= GDELT_SINCE)
        & (pl.col("date") <= dt.date.fromisoformat(asof))
    )
    if t.is_empty():
        return {"in_forecast": False, "note": GDELT_NOTE, "source": "GDELT", "months": [],
                "tracks": []}
    t = t.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    m = t.group_by("candidate_id", "month").agg(pl.col("volume").sum(),
                                                pl.col("total_monitored").sum())
    m = m.with_columns((100 * pl.col("volume") / pl.col("total_monitored")).alias("pct"))
    months = sorted(m["month"].unique().to_list())
    order = m.group_by("candidate_id").agg(pl.col("pct").mean()).sort("pct", descending=True)
    tracks = []
    for cid in order["candidate_id"].to_list():
        vals = dict(m.filter(pl.col("candidate_id") == cid).select("month", "pct").iter_rows())
        tracks.append({"candidate_id": cid, "values": [_r(vals.get(x), 3) for x in months]})
    return {
        "in_forecast": False,
        "note": GDELT_NOTE,
        "source": "GDELT 2.0, API DOC (timelinevolraw)",
        "months": [x.isoformat() for x in months],
        "tracks": tracks,
    }


# --- presse classée (docs/06-presse.md, runs/<PRESS_NLP_STUDY>/)


def _share_list(counts: list[int], n: int, nd: int = 3) -> list[float] | None:
    """Parts arrondies d'une partition ; None sous `NLP_MIN_TITLES` titres."""
    if n < NLP_MIN_TITLES:
        return None
    return [round(c / n, nd) for c in counts]


def nlp_labels(curated: Path, model: str) -> pl.DataFrame:
    """Étiquettes `NLP_VERSION` de `model` sur l'univers classé (titres qui nomment le candidat,
    plafonnés par candidat et par semaine : celui de press_features), une ligne par titre ×
    candidat, avec cycle, date, semaine, média, titre et lien."""
    from vvs.nlp.features import cap_sample

    u = cap_sample(pl.read_parquet(curated / "press_titles.parquet"))
    lab = (pl.read_parquet(curated / "press_labels.parquet")
           .filter((pl.col("prompt_version") == NLP_VERSION) & (pl.col("model") == model))
           .select("story_id", "candidate_id", "tone", "stance", "themes")
           .unique(["story_id", "candidate_id"], keep="first"))
    return u.join(lab, on=["story_id", "candidate_id"])


def _nlp_grid(asof: dt.date, feats: pl.DataFrame, universe: list[str]
              ) -> tuple[list[dt.date], list[str]]:
    """Semaines complètes du cycle 2027 (7 jours dans la fenêtre de collecte, finies avant
    `asof`) et candidats suivis de l'univers, du plus au moins classé sur le dernier mois."""
    f = feats.filter(pl.col("cycle") == NLP_CYCLE)
    weeks = sorted(f.filter((pl.col("days") == 7)
                            & (pl.col("week") + pl.duration(days=6) <= asof))
                   ["week"].unique().to_list())
    recent = weeks[-NLP_STANCE_WEEKS:]
    rank = (f.filter(pl.col("candidate_id").is_in(universe))
            .group_by("candidate_id")
            .agg(pl.col("n_labeled").filter(pl.col("week").is_in(recent)).sum().alias("n"),
                 pl.col("n_labeled").sum().alias("n_all"))
            .sort(["n", "n_all", "candidate_id"], descending=[True, True, False]))
    return weeks, rank["candidate_id"].to_list()


def build_nlp_tone(lab: pl.DataFrame, weeks: list[dt.date], cands: list[str]) -> dict:
    """Ton moyen sur `NLP_TONE_WEEKS` semaines glissantes (la semaine et la précédente), par
    candidat ; intervalle par bootstrap des titres (tirage multinomial des trois niveaux de
    ton, `NLP_BOOTSTRAP` tirages, graine fixe) ; masqué sous `NLP_MIN_TITLES` titres."""
    rng = np.random.default_rng(NLP_SEED)
    c = (lab.group_by("candidate_id", "week")
         .agg(*[(pl.col("tone") == v).sum().alias(f"t{v}") for v in (-1, 0, 1)]))
    counts = {(r["candidate_id"], r["week"]): (r["t-1"], r["t0"], r["t1"])
              for r in c.iter_rows(named=True)}
    tracks = []
    for cid in cands:
        q05, q50, q95, ns = [], [], [], []
        for w in weeks:
            k = np.zeros(3, dtype=np.int64)
            for j in range(NLP_TONE_WEEKS):
                k += np.array(counts.get((cid, w - dt.timedelta(weeks=j)), (0, 0, 0)))
            n = int(k.sum())
            ns.append(n)
            if n < NLP_MIN_TITLES:
                q05.append(None), q50.append(None), q95.append(None)
                continue
            d = rng.multinomial(n, k / n, size=NLP_BOOTSTRAP)
            q = np.quantile((d[:, 2] - d[:, 0]) / n, [0.05, 0.5, 0.95])
            q05.append(round(float(q[0]), 2)), q50.append(round(float(q[1]), 2))
            q95.append(round(float(q[2]), 2))
        tracks.append({"candidate_id": cid, "q05": q05, "q50": q50, "q95": q95, "n": ns})
    return {"note": NLP_TONE_NOTE, "window_weeks": NLP_TONE_WEEKS,
            "min_titles": NLP_MIN_TITLES, "weeks": [w.isoformat() for w in weeks],
            "tracks": tracks}


def build_nlp_stance(lab: pl.DataFrame, weeks: list[dt.date], cands: list[str]) -> dict:
    """Parts de chaque posture (ordre `NLP_STANCES`) sur les `NLP_STANCE_WEEKS` dernières
    semaines et semaine par semaine ; parts entre 0 et 1, masquées sous `NLP_MIN_TITLES`."""
    ids = [k for k, _ in NLP_STANCES]
    c = lab.group_by("candidate_id", "week", "stance").len()
    counts = {(r["candidate_id"], r["week"], r["stance"]): r["len"]
              for r in c.iter_rows(named=True)}
    recent = weeks[-NLP_STANCE_WEEKS:]
    last, tracks = [], []
    for cid in cands:
        per_week = [[counts.get((cid, w, s), 0) for s in ids] for w in weeks]
        tot = [sum(k) for k in per_week[-NLP_STANCE_WEEKS:]]
        agg = [sum(k[i] for k in per_week[-NLP_STANCE_WEEKS:]) for i in range(len(ids))]
        n = sum(tot)
        last.append({"candidate_id": cid, "n": n, "shares": _share_list(agg, n)})
        tracks.append({"candidate_id": cid, "n": [sum(k) for k in per_week],
                       "shares": [_share_list(k, sum(k)) for k in per_week]})
    return {
        "note": NLP_STANCE_NOTE,
        "categories": [{"id": k, "label": lab_} for k, lab_ in NLP_STANCES],
        "min_titles": NLP_MIN_TITLES,
        "last_weeks": {"start": recent[0].isoformat(),
                       "end": (recent[-1] + dt.timedelta(days=6)).isoformat(),
                       "rows": last},
        "weeks": [w.isoformat() for w in weeks],
        "tracks": tracks,
    }


def build_nlp_headlines(lab: pl.DataFrame, week: dt.date, cands: list[str]) -> dict:
    """Pour la dernière semaine complète, par candidat, `NLP_HEADLINES` titres classés −1 et +1,
    les plus récents d'abord (titres identiques dédoublonnés, ordre stable par story_id)."""
    w = lab.filter(pl.col("week") == week).with_columns(
        pl.col("title").str.strip_chars().str.to_lowercase().alias("_key"))
    rows = []
    for cid in cands:
        wc = w.filter(pl.col("candidate_id") == cid)

        def pick(tone: int, wc=wc) -> list[dict]:
            t = (wc.filter(pl.col("tone") == tone)
                 .sort(["publish_date", "story_id"], descending=[True, False])
                 .unique("_key", keep="first", maintain_order=True).head(NLP_HEADLINES))
            return [{"title": r["title"].strip(), "outlet": r["outlet"] or "",
                     "date": r["publish_date"].isoformat(), "url": r["url"],
                     "stance": r["stance"]} for r in t.iter_rows(named=True)]

        rows.append({"candidate_id": cid, "n": wc.height, "negative": pick(-1),
                     "positive": pick(1)})
    return {"note": NLP_HEADLINES_NOTE, "week": week.isoformat(),
            "week_end": (week + dt.timedelta(days=6)).isoformat(), "candidates": rows}


def build_nlp_share_of_voice(feats: pl.DataFrame, curated: Path, weeks: list[dt.date],
                             cands: list[str]) -> dict:
    """Part de voix (titres qui nomment le candidat, `mentions` de press_features) parmi les
    candidats suivis, et part des pages vues Wikipédia des mêmes candidats, mêmes semaines
    (semaines de 7 jours de pages vues, sinon null) ; parts entre 0 et 1."""
    f = feats.filter((pl.col("cycle") == NLP_CYCLE) & pl.col("candidate_id").is_in(cands)
                     & pl.col("week").is_in(weeks))
    men = {(r["candidate_id"], r["week"]): r["mentions"] for r in f.iter_rows(named=True)}
    tot = {w: sum(men.get((c, w), 0) for c in cands) for w in weeks}
    a = pl.read_parquet(curated / "attention.parquet").filter(
        (pl.col("source") == "wikipedia") & (pl.col("election") == NLP_CYCLE)
        & pl.col("candidate_id").is_in(cands)
    ).with_columns(pl.col("date").dt.truncate("1w").alias("week")).filter(
        pl.col("week").is_in(weeks))
    full = set(a.group_by("week").agg(pl.col("date").n_unique().alias("d"))
               .filter(pl.col("d") == 7)["week"].to_list())
    views = {(r["candidate_id"], r["week"]): r["value"]
             for r in a.group_by("candidate_id", "week").agg(pl.col("value").sum())
             .iter_rows(named=True)}
    vtot = {w: sum(views.get((c, w), 0.0) for c in cands) for w in weeks}

    def frac(x: float, t: float, ok: bool = True) -> float | None:
        return round(x / t, 4) if ok and t > 0 else None

    return {
        "note": NLP_SOV_NOTE,
        "weeks": [w.isoformat() for w in weeks],
        "tracks": [{"candidate_id": c,
                    "values": [frac(men.get((c, w), 0), tot[w]) for w in weeks]}
                   for c in cands],
        "attention": [{"candidate_id": c,
                       "values": [frac(views.get((c, w), 0.0), vtot[w], w in full)
                                  for w in weeks]}
                      for c in cands],
    }


def build_nlp_agenda(lab: pl.DataFrame, weeks: list[dt.date], cands: list[str]) -> dict:
    """Part hebdomadaire des titres classés (candidats suivis réunis) portant chacun des thèmes
    concordants ; multi-étiquette, entre 0 et 1, masquée sous `NLP_MIN_TITLES` titres."""
    lw = lab.filter(pl.col("candidate_id").is_in(cands) & pl.col("week").is_in(weeks))
    agg = lw.group_by("week").agg(
        pl.len().alias("n"),
        *[pl.col("themes").list.contains(k).sum().alias(k) for k, _ in NLP_AGENDA_THEMES])
    by = {r["week"]: r for r in agg.iter_rows(named=True)}
    n = [int(by[w]["n"]) if w in by else 0 for w in weeks]
    themes = []
    for k, label in NLP_AGENDA_THEMES:
        vals = [round(by[w][k] / by[w]["n"], 4) if ni >= NLP_MIN_TITLES else None
                for w, ni in zip(weeks, n)]
        themes.append({"id": k, "label": label, "values": vals})
    return {"note": NLP_AGENDA_NOTE, "min_titles": NLP_MIN_TITLES,
            "weeks": [w.isoformat() for w in weeks], "n": n, "themes": themes}


def build_nlp_landmarks(lab: pl.DataFrame, study: Path) -> dict:
    """Repères 2017 et 2022 : ton et part de la posture « affaire », jour par jour, moyennés sur
    trois jours (jours de moins de 3 titres écartés, comme l'étude), de `NLP_LANDMARK_DAYS[0]`
    jours avant le choc à `NLP_LANDMARK_DAYS[1]` après ; dates de rupture lues dans
    `shocks.csv` de l'étude."""
    shocks = {r["event"]: r for r in pl.read_csv(study / "shocks.csv", try_parse_dates=True)
              .iter_rows(named=True)}
    events = []
    for ev, label, summary in NLP_LANDMARKS:
        sh = shocks[ev]
        cid, day = sh["candidate_id"], sh["date"]
        cycle = day.year if day.month <= 6 else day.year + 1  # présidentielle au printemps
        before, after = NLP_LANDMARK_DAYS
        start, end = day - dt.timedelta(days=before), day + dt.timedelta(days=after)
        d = (lab.filter((pl.col("candidate_id") == cid) & (pl.col("cycle") == cycle))
             .group_by("publish_date")
             .agg(pl.len().alias("n"), pl.col("tone").cast(pl.Float64).mean().alias("tone"),
                  (pl.col("stance") == "scandal").mean().alias("scandal")))
        cal = pl.DataFrame({"publish_date": pl.date_range(start - dt.timedelta(days=2), end,
                                                          "1d", eager=True)})
        d = (cal.join(d, on="publish_date", how="left").sort("publish_date")
             .with_columns(pl.col("n").fill_null(0))
             .with_columns([pl.when(pl.col("n") >= 3).then(pl.col(c)).otherwise(None)
                           .rolling_mean(3, min_samples=2).alias(c)
                            for c in ("tone", "scandal")])
             .filter(pl.col("publish_date") >= start))

        def iso(x) -> str | None:
            return None if x is None else x.isoformat()

        events.append({
            "id": ev, "label": label, "candidate_id": cid, "cycle": cycle,
            "event_date": day.isoformat(), "summary": summary,
            "dates": [x.isoformat() for x in d["publish_date"]],
            "tone": [_r(x, 2) for x in d["tone"]],
            "scandal_share": [_r(x, 3) for x in d["scandal"]],
            "n": [int(x) for x in d["n"]],
            "breaks": {"polls": iso(sh["poll_break"]), "tone": iso(sh["break_tone_3d"]),
                       "scandal": iso(sh["break_scandal_share_3d"])},
        })
    return {"note": NLP_LANDMARKS_NOTE, "events": events}


def build_nlp_quality(study: Path, model: dict, lab: pl.DataFrame) -> dict:
    """Accords entre deux formulations du prompt (`agreement.json` de l'étude), modèle et
    version de prompt, titres classés par cycle."""
    ag = json.loads((study / "agreement.json").read_text())
    s = ag["sample"]
    per_cycle = lab.group_by("cycle").len().sort("cycle")
    return {
        "model": model["id"],
        "model_label": f"{Path(model.get('file', model['id'])).stem} ({model.get('engine', '?')})",
        "prompt_version": NLP_VERSION,
        "n_sample": int(s["n"]),
        "tone_kappa": round(float(s["tone_kappa"]), 3),
        "stance_kappa": round(float(s["stance_kappa"]), 3),
        "themes_f1": round(float(s["themes_f1"]), 3),
        "thresholds": {k: float(v) for k, v in ag["thresholds"].items()},
        "passes": {k: bool(v) for k, v in ag["passes"].items()},
        "labeled": [{"cycle": int(r["cycle"]), "n": int(r["len"])}
                    for r in per_cycle.iter_rows(named=True)],
        "reference": f"runs/{study.name}/report.md",
    }


def build_press(asof: str, curated: Path | None = None, universe: list[str] | None = None,
                nlp_study: Path | None = None) -> dict:
    """press.json : attention, INA, GDELT, et presse classée (`nlp_study` : dossier de l'étude,
    défaut runs/<PRESS_NLP_STUDY>, pour les accords et les dates de rupture)."""
    from vvs.nlp.prompts import load_prompts

    curated = curated or paths.CURATED
    nlp_study = nlp_study or paths.RUNS / PRESS_NLP_STUDY
    prompts = load_prompts()
    if {k for k, _ in NLP_STANCES} != set(prompts.stances):
        raise ValueError(f"postures du registre {list(prompts.stances)} ≠ NLP_STANCES")
    feats = pl.read_parquet(curated / "press_features.parquet")
    lab = nlp_labels(curated, prompts.model["id"])
    weeks, cands = _nlp_grid(dt.date.fromisoformat(asof), feats, universe or [])
    if not weeks or not cands:
        raise ValueError("presse classée : aucune semaine complète ou aucun candidat suivi")
    lab27 = lab.filter(pl.col("cycle") == NLP_CYCLE)
    return {
        "asof": asof,
        "in_forecast": False,
        "in_forecast_note": PRESS_NOTE,
        "nlp": {"in_forecast": False, "note": NLP_NOTE, "verdict": list(NLP_VERDICT),
                "out_of_sample_test": NLP_OUT_OF_SAMPLE, "reference": "docs/06-presse.md"},
        "tone": build_nlp_tone(lab27, weeks, cands),
        "stance": build_nlp_stance(lab27, weeks, cands),
        "headlines": build_nlp_headlines(lab27, weeks[-1], cands),
        "share_of_voice": build_nlp_share_of_voice(feats, curated, weeks, cands),
        "agenda_nlp": build_nlp_agenda(lab27, weeks, cands),
        "landmarks": build_nlp_landmarks(lab, nlp_study),
        "quality": build_nlp_quality(nlp_study, prompts.model, lab),
        "attention": build_attention(asof, curated, universe or []),
        "ina": build_ina(curated),
        "gdelt": build_gdelt(asof, curated),
    }


def _consecutive_pearson(values: pl.DataFrame, key: list[str], col: str) -> dict:
    """Corrélation poolée de `col` entre élections consécutives, sur les clés communes (même
    calcul que back/studies/pollster_ratings.py)."""
    from scipy import stats

    elections = sorted(values["election"].unique().to_list())
    a_all, b_all = [], []
    for a, b in itertools.pairwise(elections):
        va = values.filter(pl.col("election") == a).select(*key, pl.col(col).alias("a"))
        vb = values.filter(pl.col("election") == b).select(*key, pl.col(col).alias("b"))
        j = va.join(vb, on=key).drop_nulls()
        a_all += j["a"].to_list()
        b_all += j["b"].to_list()
    if len(a_all) < 3:
        return {"n_pairs": len(a_all), "pearson": None, "p": None}
    r, p = stats.pearsonr(a_all, b_all)
    return {"n_pairs": len(a_all), "pearson": round(float(r), 2), "p": round(float(p), 2)}


def build_pollster_ratings(study: Path, reg: Registry) -> dict:
    """Palmarès des instituts 2002-2022 (`palmares.parquet` de l'étude) et stabilité du
    classement d'une élection à la suivante."""
    pal = pl.read_parquet(study / "palmares.parquet")
    per_e = pl.read_parquet(study / "palmares_by_election.parquet")
    house = pl.read_parquet(study / "house_effects.parquet").filter(
        pl.col("bloc").is_in(MAIN_BLOCS)
    )
    rows = []
    for r in pal.iter_rows(named=True):
        lid = r["lineage"]
        name = RATINGS_LABELS.get(lid) or (reg.pollsters[lid].name if lid in reg.pollsters else lid)
        rows.append(
            {
                "pollster": lid,
                "name": name,
                "elections": [int(x) for x in str(r["elections"]).split(",") if x],
                "n_polls": int(r["n_polls"] or 0),
                "n_final": int(r["n_final"] or 0),
                "mae_candidate": _r(r["mae_cand"], 2),
                "mae_bloc": _r(r["mae_bloc"], 2),
                "relative_error": _r(r["pm_cand"], 2),
                "bias": {b: _r(r.get(b), 1) for b in MAIN_BLOCS},
                "precision": _r(r["x_raw"], 2),
                "n_better": int(r["n_better"] or 0),
                "n_rated": int(r["n_x"] or 0),
            }
        )
    stability = [
        {"id": "precision", "label": "Précision relative (sondages des 21 derniers jours)",
         **_consecutive_pearson(per_e, ["lineage"], "x")},
        {"id": "relative_error", "label": "Erreur absolue relative du dernier sondage",
         **_consecutive_pearson(per_e, ["lineage"], "pm_cand")},
        {"id": "house_effects", "label": "Effets maison de campagne (cinq blocs)",
         **_consecutive_pearson(house, ["lineage", "bloc"], "alpha")},
    ]
    return {
        "in_forecast": False,
        "note": RATINGS_NOTE,
        "elections": sorted(per_e["election"].unique().to_list()),
        "rows": rows,
        "stability": stability,
    }


def build_backtest(generated_at: str, first: Path, second: Path) -> dict:
    fr = pl.read_parquet(first / "metrics.parquet").filter(pl.col("level") == "summary")
    variant = "floor" if "floor" in fr["variant"].unique().to_list() else fr["variant"][0]
    fr = fr.filter(pl.col("variant") == variant)
    sr = pl.read_parquet(second / "metrics.parquet").filter(pl.col("horizon_days") >= 0)
    rows = []
    metric = {
        "log_score_pair": "log_score_pair",
        "brier_qualify": "brier_qual",
        "crps_first_round": "crps",
        "mae_first_round": "mae",
        "coverage_50": "cov50",
        "coverage_90": "cov90",
    }
    for (model, e, h), g in fr.group_by(["model", "election", "horizon"], maintain_order=True):
        if model not in FIRST_ROUND_MODELS:
            continue
        vals = dict(zip(g["metric"].to_list(), g["value"].to_list()))
        row = {"model_id": model, "election": int(e), "horizon_days": int(h),
               "log_score_winner": None}
        for k, m in metric.items():
            v = vals.get(m)
            row[k] = None if v is None or not np.isfinite(v) else round(float(v), 4)
        rows.append(row)
    for r in sr.iter_rows(named=True):
        if r["method"] not in SECOND_ROUND_MODELS:
            continue
        rows.append(
            {
                "model_id": f"r2_{r['method']}",
                "election": int(r["election"]),
                "horizon_days": int(r["horizon_days"]),
                "log_score_pair": None,
                "log_score_winner": _r(r["log_score"], 4),
                "brier_qualify": None,
                "crps_first_round": None,
                "mae_first_round": None,
                "coverage_50": None if r["cover50"] is None else float(r["cover50"]),
                "coverage_90": None if r["cover90"] is None else float(r["cover90"]),
            }
        )
    rows.sort(key=lambda x: (x["horizon_days"], x["election"], x["model_id"]))
    models = [
        {"id": k, "label": lab, "kind": kind, "description": desc}
        for k, (lab, kind, desc) in FIRST_ROUND_MODELS.items()
    ] + [
        {"id": f"r2_{k}", "label": lab, "kind": kind, "description": desc}
        for k, (lab, kind, desc) in SECOND_ROUND_MODELS.items()
    ]
    return {
        "generated_at": generated_at,
        "model_version": MODEL_VERSION,
        "elections": sorted({r["election"] for r in rows}),
        "horizons_days": sorted({r["horizon_days"] for r in rows}, reverse=True),
        "models": models,
        "rows": rows,
        "variables_tested": [
            {"id": i, "label": lab, "verdict": verdict, "admitted": False, "published": pub,
             "summary": text}
            for i, lab, verdict, pub, text in VARIABLES_TESTED
        ],
        "variables_tested_reference": "docs/backtest-log.md",
    }


# --- sondages intégrés


def integrated_polls(run_manifest: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(polls, résultats longs des deux tours, hypothèses de premier tour de la grille) des
    sondages intégrés au run : premier tour dans la grille de l'ajustement, second tour dans la
    fenêtre du modèle de duel (`MAX_POLL_AGE`)."""
    from vvs.model.data import load_polls
    from vvs.model.second_round import MAX_POLL_AGE

    election = run_manifest["election"]
    asof = dt.date.fromisoformat(run_manifest["asof"])
    fit_dir = Path(run_manifest["fit"]["dir"])
    observed = pl.read_parquet(fit_dir / "observed.parquet").select("poll_id", "hypothesis_id").unique()
    r1 = load_polls(election, asof, round_=1).join(observed, on=["poll_id", "hypothesis_id"])
    r1 = r1.with_columns(pl.lit(1, dtype=pl.Int8).alias("round"))
    gap = (dt.date.fromisoformat(run_manifest["second_round"])
           - dt.date.fromisoformat(run_manifest["first_round"])).days
    oldest = asof - dt.timedelta(days=MAX_POLL_AGE - run_manifest["horizon_days"] - gap)
    r2 = load_polls(election, asof, round_=2).filter(pl.col("fieldwork_end") >= oldest)
    r2 = r2.with_columns(pl.lit(2, dtype=pl.Int8).alias("round"))
    results = pl.concat([r1, r2])
    polls = pl.read_parquet(paths.CURATED / "polls.parquet").filter(
        pl.col("poll_id").is_in(results["poll_id"].unique().to_list())
    )
    return polls, results, r1


# --- export


def export_run(
    run: Path,
    out: Path,
    registry: Registry | None = None,
    scenarios: ScenariosFile | None = None,
    backtest_first: Path | None = None,
    backtest_second: Path | None = None,
    polls_data: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None = None,
    export_configs: int = EXPORT_CONFIGS,
    economy_study: Path | None = None,
    pollster_ratings: Path | None = None,
    curated: Path | None = None,
    press_nlp_study: Path | None = None,
) -> dict:
    """Écrit les fichiers du contrat JSON dans `out` ; rend un petit compte rendu.

    `economy_study`, `pollster_ratings`, `press_nlp_study` : dossiers des études (défaut
    runs/<ECONOMY_STUDY>, runs/<POLLSTER_RATINGS>, runs/<PRESS_NLP_STUDY>) ; `curated` : dossier
    des parquet explicatifs (macro, attention, presse, titres classés ; défaut data/curated).
    Injectables pour les tests."""
    run, out = Path(run), Path(out)
    reg = registry or default_registry()
    f = scenarios or load_scenarios(registry=reg)
    rm = json.loads((run / "manifest.json").read_text())
    summary = pl.read_parquet(run / "summary.parquet")
    pairs = pl.read_parquet(run / "pairs.parquet")
    transfers = pl.read_parquet(run / "transfers.parquet")
    traj = pl.read_parquet(run / "trajectories.parquet")
    house = pl.read_parquet(run / "house_effects.parquet")
    draws = pl.read_parquet(run / "draws.parquet")
    run_date = run_date_of(run, rm)
    generated_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    all_conditions = load_conditions(run)
    have = set(summary["condition_id"].unique().to_list())
    all_conditions = [c for c in all_conditions if c["id"] in have]
    cfg_conditions = [c for c in all_conditions if c["kind"] == "configuration"]
    conditions = [c for c in all_conditions if c["kind"] != "configuration"]
    conditions += cfg_conditions[:export_configs]
    fr_conditions = [c for c in all_conditions if c["kind"] != "configuration"]
    fr_conditions += cfg_conditions[:FIRST_ROUND_CONFIGS]

    polls, results, r1 = polls_data or integrated_polls(rm)
    r2 = results.filter(pl.col("round") == 2)
    bloc_hint = {u["candidate_id"]: u["bloc"] for u in rm["universe"]}

    summary_json = build_summary(rm, summary, conditions)
    first_round = build_first_round(rm, summary_json, traj, fr_conditions, r1, f)
    second_round = build_second_round(rm, pairs, transfers, draws, r2, bloc_hint)
    scenarios_json = build_scenarios(f)
    ratings = build_pollster_ratings(pollster_ratings or paths.RUNS / POLLSTER_RATINGS, reg)
    polls_json = build_polls(rm, polls, results, house, reg, ratings)
    economy = build_economy(rm["asof"], economy_study, curated)
    press = build_press(rm["asof"], curated, [u["candidate_id"] for u in rm["universe"]],
                        press_nlp_study)
    backtest = build_backtest(
        generated_at,
        backtest_first or paths.RUNS / BACKTEST_FIRST_ROUND,
        backtest_second or paths.RUNS / BACKTEST_SECOND_ROUND,
    )

    cited = set(bloc_hint)
    cited |= set(results["candidate_id"].unique().to_list())
    for p in second_round["pairs"]:
        cited |= {p["candidate_a"], p["candidate_b"]}
    cited |= {t["from"] for t in second_round["transfers"]}
    for block in (press["attention"], press["gdelt"], press["tone"]):
        cited |= {t["candidate_id"] for t in block["tracks"]}
    cited |= {e["candidate_id"] for e in press["landmarks"]["events"]}
    candidates = [candidate_entry(c, reg, rm["election"], bloc_hint) for c in sorted(cited)]

    last_poll = max(p["fieldwork_end"] for p in polls_json["polls"])
    commit = clean_commit()
    polls_hash = rm.get("polls_hash")
    fit_dir = rm.get("fit", {}).get("dir")
    if polls_hash is None and fit_dir:  # simulations antérieures à l'empreinte : l'ajustement
        fit_manifest = Path(fit_dir) / "manifest.json"
        if fit_manifest.is_file():
            polls_hash = json.loads(fit_manifest.read_text())["data"].get("rows_hash")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "generated_at": generated_at,
        "asof": rm["asof"],
        "model_version": MODEL_VERSION,
        "commit": commit,
        "last_poll_date": last_poll,
        "n_polls": len(polls_json["polls"]),
        "polls_hash": polls_hash,
        "n_draws": rm["n_draws"],
        "mock": False,
        "election": {
            "year": rm["election"],
            "first_round": rm["first_round"],
            "second_round": rm["second_round"],
        },
        "candidates": candidates,
    }

    # historique : on retire les entrées factices, on n'écrase jamais une entrée réelle
    hist_dir = out / "history"
    old_manifest = out / "manifest.json"
    if old_manifest.exists():
        try:
            was_mock = json.loads(old_manifest.read_text()).get("mock", False)
        except json.JSONDecodeError:
            was_mock = False
        if was_mock and hist_dir.exists():
            for p in hist_dir.glob("*.json"):
                p.unlink()
    entry = {
        "run_date": run_date,
        "asof": rm["asof"],
        "model_version": MODEL_VERSION,
        "last_poll_date": last_poll,
        "n_polls": manifest["n_polls"],
        "candidates": [
            {k: c[k] for k in ("candidate_id", "p_qualify", "p_win", "first_round_share")}
            for c in summary_json["mixture"]["candidates"]
        ],
    }
    hist_file = hist_dir / f"{run_date}.json"
    history_written = False
    if hist_file.exists():
        if json.loads(hist_file.read_text()) != entry:
            log.warning("%s existe déjà et n'est pas réécrit", hist_file)
    else:
        _write(out, f"history/{run_date}.json", entry)
        history_written = True

    _write(out, "manifest.json", manifest)
    _write(out, "summary.json", summary_json)
    _write(out, "first_round.json", first_round)
    _write(out, "second_round.json", second_round)
    _write(out, "scenarios.json", scenarios_json)
    _write(out, "polls.json", polls_json)
    _write(out, "economy.json", economy)
    _write(out, "press.json", press)
    _write(out, "backtest.json", backtest)
    entries = [
        json.loads(p.read_text())
        for p in sorted(hist_dir.glob("*.json"))
        if p.name != "index.json"
    ]
    _write(out, "history/index.json", {"entries": sorted(entries, key=lambda e: e["run_date"])})
    sizes = {p.name: p.stat().st_size for p in out.glob("*.json")}
    return {
        "out": str(out),
        "run_date": run_date,
        "commit": commit,
        "n_polls": manifest["n_polls"],
        "polls_hash": polls_hash,
        "n_candidates": len(candidates),
        "history_written": history_written,
        "history_entries": len(entries),
        "sizes": sizes,
    }
