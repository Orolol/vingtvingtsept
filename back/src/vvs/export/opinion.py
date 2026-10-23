"""Volets d'opinion du site (docs/08-front.md), tous marqués « n'entre pas dans la prévision ».

- `build_opinion` → `economy.opinion` : séries longues d'attitudes et de préoccupations
  (`data/curated/attitudes.parquet`, registre `data/registry/attitude_series.yaml`) groupées par
  thème (`OPINION_THEMES`), chacune avec sa formulation exacte, sa population, sa source, ses
  ruptures signalées dans le registre, et la part mensuelle du bloc lié dans les sondages
  (`poll_results_by_bloc`, mêmes règles que l'étude économie : `vvs.studies.panel`) ; le
  bloc `study` relit l'étude du 2026-10-23 (`runs/<date>-attitudes-prior/`, CSV).
- `build_rn_image` → `second_round.rn_image` : baromètre d'image du FN/RN (trois items),
  décalage commun u des reports par élection, nuage u contre « danger » (n = 3) et parts par
  duel avec et sans centrage (`runs/<date>-rn-image/`).

Valeurs exclues : tout document dont la note du registre porte « réservé aux souscripteurs »
(note CRÉDOC 2015) et les documents de `EXCLUDED_DOCUMENTS` (licence non précisée). Une série
d'un thème sans valeur publiable est listée dans `excluded` avec la raison.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from vvs import paths
from vvs.registry import Registry

ATTITUDES_STUDY = "2026-10-23-attitudes-prior"
RN_IMAGE_STUDY = "2026-10-23-rn-image"
REFERENCE = "docs/05-economie.md"

BLOC_LABELS = {
    "far_right": "Extrême droite", "right": "Droite", "centre": "Centre", "left": "Gauche",
    "far_left": "Gauche radicale", "incumbent": "Bloc du président sortant",
}
BLOC_NOTES = {
    "incumbent": ("Bloc du président sortant : droite en 2002, 2007 et 2012, gauche en 2017, "
                  "centre en 2022 et 2027."),
}
# (id, libellé, séries (series_id, bloc lié)) ; ordre d'affichage.
OPINION_THEMES = (
    ("immigration_security", "Immigration et sécurité", (
        ("eb_mii_immigration", "far_right"),
        ("eb_mii_crime", "far_right"),
        ("credoc_concern_immigration", "far_right"),
        ("credoc_concern_violence_insecurity", "far_right"),
        ("opinion_too_many_immigrants", "far_right"),
        ("opinion_not_at_home_anymore", "far_right"),
        ("opinion_more_power_to_police", "far_right"),
        ("opinion_justice_too_lenient", "far_right"),
    )),
    ("tolerance", "Tolérance envers les minorités", (
        ("cncdh_tolerance_index", "far_right"),
    )),
    ("rn_image", "Image du FN/RN", (
        ("rn_danger_democracy_yes", "far_right"),
        ("rn_ideas_agree", "far_right"),
    )),
    ("economy_concerns", "Chômage et prix", (
        ("eb_mii_unemployment", "left"),
        ("credoc_concern_unemployment", "left"),
        ("eb_mii_rising_prices", "left"),
    )),
    ("trust", "Confiance politique", (
        ("cevipof_trust_government", "incumbent"),
    )),
)
# Libellés courts des tableaux de l'étude (sinon libellé du registre).
SHORT_LABELS = {
    "eb_mii_immigration": "Immigration, problème important (Eurobaromètre)",
    "eb_mii_crime": "Insécurité, problème important (Eurobaromètre)",
    "eb_mii_unemployment": "Chômage, problème important (Eurobaromètre)",
    "eb_mii_rising_prices": "Hausse des prix, problème important (Eurobaromètre)",
    "credoc_concern_immigration": "Immigration, préoccupation (CRÉDOC)",
    "credoc_concern_violence_insecurity": "Violence et insécurité, préoccupation (CRÉDOC)",
    "credoc_concern_unemployment": "Chômage, préoccupation (CRÉDOC)",
    "opinion_too_many_immigrants": "« Trop d'immigrés » (Verian)",
    "opinion_not_at_home_anymore": "« Plus vraiment chez soi » (Verian)",
    "opinion_more_power_to_police": "« Plus de pouvoir à la police » (Verian)",
    "opinion_justice_too_lenient": "« Justice pas assez sévère » (Verian)",
    "cncdh_tolerance_index": "Indice de tolérance (CNCDH)",
    "rn_ideas_agree": "Adhésion aux idées du FN/RN (Verian)",
    "rn_danger_democracy_yes": "FN/RN danger pour la démocratie (Verian)",
    "cevipof_trust_government": "Confiance dans le gouvernement (Cevipof)",
}
EXCLUDED_DOCUMENTS = {
    "credoc_note_2024": (
        "note CRÉDOC 2024 : licence de republication non précisée dans le registre ; elle ne "
        "donnerait que 2023 et 2024, après le passage en ligne de l'enquête"),
}
SUBSCRIBERS_ONLY = re.compile(r"réservé aux souscripteurs", re.IGNORECASE)
SOURCE_LABELS = {
    "verian": "Baromètre d'image du RN (Verian, ex-Sofres, Kantar)",
    "eurobarometer": "Eurobaromètre standard (Commission européenne)",
    "cncdh": "CNCDH, rapport sur le racisme (indice de V. Tiberj)",
    "cevipof": "Baromètre de la confiance politique (OpinionWay pour le Cevipof)",
    "credoc": "CRÉDOC, Conditions de vie et aspirations",
    "drees": "Baromètre d'opinion de la DREES",
}
POPULATION_LABELS = {
    "fr_adults_18plus": "France, 18 ans et plus (quotas)",
    "fr_registered_18plus": "inscrits sur les listes électorales, 18 ans et plus",
    "fr_metro_adults_18plus": "France métropolitaine ou hexagonale, 18 ans et plus",
    "fr_adults_15plus": "France, 15 ans et plus",
}
UNIT_LABELS = {"percent": "%", "index_0_100": "indice de 0 à 100"}
FREQUENCY = {"A": "annual", "wave": "waves"}
# Premiers tours antérieurs au registre des élections (repères des séries depuis 1983).
EARLY_FIRST_ROUNDS = (dt.date(1988, 4, 24), dt.date(1995, 4, 23))

OPINION_NOTE = (
    "Ces séries d'opinion n'entrent pas dans la prévision. L'étude du 23 octobre 2026 "
    "(docs/05-economie.md) ne trouve aucune attitude ni préoccupation qui améliore hors "
    "échantillon le point de départ des blocs, et ce point de départ n'a presque pas de "
    "levier : les sondages fixent le niveau des blocs dès 180 jours du vote. Elles sont "
    "montrées pour le contexte, face à la part du bloc lié dans les sondages."
)
OPINION_STUDY_FINDINGS = (
    ("Saillances Eurobaromètre (immigration, insécurité, chômage, prix) : elles dégradent la "
     "prévision hors échantillon de l'écart entre le résultat d'un bloc et son point de départ. "
     "Les rares pentes proches du seuil (« trop d'immigrés », « plus chez soi », Verian) ont le "
     "signe contraire à l'hypothèse."),
    ("Image du FN/RN et tolérance CNCDH : bon signe parfois, mais aucun gain hors échantillon "
     "qui résiste au placebo sur cinq élections."),
    ("Dynamique : les variations entre deux vagues Eurobaromètre ne précèdent pas celles de la "
     "part du bloc (aucune paire au retard d'une vague)."),
)

RN_ITEMS = (
    ("danger", "rn_danger_democracy_yes", "Danger pour la démocratie"),
    ("capable", "rn_can_join_government", "Capable de participer à un gouvernement"),
    ("adhesion", "rn_ideas_agree", "Adhésion aux idées"),
)
RN_VARIANTS = {
    "défaut": ("default", "Défaut du modèle"),
    "persistance 2022": ("persistence_2022", "Persistance de 2022"),
    "pente 2017-2022": ("local_slope", "Pente 2017 → 2022"),
    "relation baromètre": ("relation", "Relation baromètre (n = 3)"),
}
RN_POLLS = {"avec sondages": "with_polls", "prior seul": "prior_only"}
RN_U_DEFINITION = (
    "u : décalage commun des reports de voix vers le finaliste FN/RN, en logit, par rapport à "
    "la matrice de proximité du modèle (fixe, donc comparable d'une élection à l'autre) ; u > 0 "
    "veut dire un barrage plus faible que la référence. Mesuré dans les enquêtes "
    "post-électorales (intervalle à 90 % : effet institut et échantillonnage) et par le "
    "résultat officiel agrégé."
)
RN_DUELS_NOTE = (
    "Part du candidat RN au second tour et probabilité qu'il gagne, conditionnelles au duel, "
    "calculées par l'étude (premier tour : moyenne bruitée des sondages de 14 jours) : ce ne "
    "sont pas les chiffres de la simulation publiée plus haut. La différence entre lignes est "
    "une sensibilité au centre des reports, pas une prévision alternative."
)
Z90 = 1.6449


def _r(x, nd: int = 1) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def _french(x: float, nd: int, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def _study_date(study: Path, asof: str) -> str:
    """Date du dossier d'étude (runs/<date>-…), à défaut `asof`."""
    try:
        return dt.date.fromisoformat(study.name[:10]).isoformat()
    except ValueError:
        return asof


def load_registry() -> dict:
    import yaml

    return yaml.safe_load((paths.REGISTRY / "attitude_series.yaml").read_text(encoding="utf-8"))


def excluded_documents(registry: dict) -> dict[str, str]:
    """Documents dont les valeurs ne sont pas publiées, avec la raison."""
    out = {}
    for d in registry.get("documents", []):
        if SUBSCRIBERS_ONLY.search(d.get("notes") or ""):
            out[d["id"]] = (f"{d.get('title', d['id'])} : couverture « document réservé aux "
                            "souscripteurs », droits de republication non établis")
    for k, v in EXCLUDED_DOCUMENTS.items():
        out.setdefault(k, v)
    return out


def _resolve(meta: dict, by_id: dict, field: str) -> str | None:
    """Texte d'un champ du registre ; « voir <id> ; « item » » est développé à partir de la
    série citée (question : amorce jusqu'au point d'interrogation, puis l'item)."""
    text = meta.get(field)
    if not text:
        return None
    text = " ".join(str(text).split())
    m = re.match(r"^voir ([a-z0-9_]+)\s*(?:;\s*(.*))?$", text)
    if not m or m.group(1) not in by_id:
        return text
    ref = by_id[m.group(1)].get(field)
    if not ref:
        return m.group(2) or None
    ref = " ".join(str(ref).split())
    if field == "question" and m.group(2):
        stem = ref[: ref.index("?") + 1] if "?" in ref else ref
        item = m.group(2).strip().strip("«» ").strip()
        return f"{stem} {item}"
    return ref if not m.group(2) else f"{ref} {m.group(2)}"


_LABEL = re.compile(r"libellé : ([^;]*)")


def _label_key(note: str | None) -> str | None:
    m = _LABEL.search(note or "")
    if not m:
        return None
    s = re.sub(r"^\d+\.\s*", "", m.group(1).strip()).lower()
    return " ".join(re.sub(r"[^a-zà-ÿ]+", " ", s).split())


def _point_note(note: str | None) -> str | None:
    """Note d'une valeur sans le libellé Eurobaromètre (traité comme rupture)."""
    parts = [p.strip() for p in (note or "").split(";")]
    parts = [p for p in parts if p and not p.startswith("libellé :")]
    return "; ".join(parts) or None


def series_breaks(sub: pl.DataFrame, family: str | None, registry: dict) -> list[dict]:
    """Ruptures datées : vagues annotées du registre (`waves.<famille>.<vague>.note`) où la
    série a une valeur, et changements de libellé de l'item (notes `libellé : …`)."""
    out = []
    waves = (registry.get("waves") or {}).get(family or "", {}) or {}
    by_wave = {w: d for w, d in zip(sub["wave"].to_list(), sub["date"].to_list()) if w}
    for w, info in waves.items():
        if isinstance(info, dict) and info.get("note") and str(w) in by_wave:
            out.append({"date": by_wave[str(w)].isoformat(),
                        "label": " ".join(info["note"].split())})
    prev = None
    for d, note in zip(sub["date"].to_list(), sub["notes"].to_list()):
        k = _label_key(note)
        if prev is not None and k is not None and k != prev:
            item = re.sub(r"\s*[\\/]\s*", " / ", _LABEL.search(note).group(1).strip())
            out.append({"date": d.isoformat(), "label": f"libellé de l'item : « {item} »"})
        if k is not None:
            prev = k
    return sorted(out, key=lambda b: b["date"])


def first_rounds(reg: Registry, since: dt.date, until: dt.date) -> list[str]:
    days = list(EARLY_FIRST_ROUNDS) + [el.round_(1).date for el in reg.elections.values()]
    return sorted({d.isoformat() for d in days if since <= d <= until})


def bloc_shares(curated: Path, asof: dt.date, blocs: list[str], reg: Registry) -> list[dict]:
    """Part mensuelle des blocs dans les sondages de premier tour (`vvs.studies.panel` :
    hypothèses suspectes et Macron 2027 écartés, un mois = moyenne des instituts)."""
    from vvs.studies import panel as P

    by_bloc = pl.read_parquet(curated / "poll_results_by_bloc.parquet")
    polls = pl.read_parquet(curated / "polls.parquet")
    votes = {e: el.round_(1).date for e, el in reg.elections.items()}
    monthly = P.monthly_bloc_shares(P.poll_bloc_shares(by_bloc, polls))
    votes = {e: votes.get(e, dt.date(e, 4, 30)) for e in monthly["election"].unique().to_list()}
    pan = (P.bloc_panel(monthly, votes)
           .filter(pl.col("share").is_not_null() & (pl.col("month") <= asof))
           .sort("month", "election")
           .unique(["bloc", "month"], keep="first", maintain_order=True))
    out = []
    for b in blocs:
        sub = pan.filter(pl.col("bloc") == b)
        out.append({
            "bloc": b,
            "label": BLOC_LABELS[b],
            "note": BLOC_NOTES.get(b),
            "points": [{"date": m.isoformat(), "value": round(float(s), 1), "n_pollsters": int(n)}
                       for m, s, n in zip(sub["month"].to_list(), sub["share"].to_list(),
                                          sub["n_pollsters"].to_list())],
        })
    return out


def opinion_study(study: Path, asof: str, labels: dict[str, str] | None = None) -> dict:
    """Bloc `study` : verdict, tableaux niveau (gain hors échantillon, placebo par
    permutation) et dynamique (placebo par décalage), faux positifs attendus."""
    level = pl.read_csv(study / "level_results.csv")
    dyn = pl.read_csv(study / "dynamic_results.csv")
    lev_path = study / "leverage.csv"
    labels = labels or {}
    levels = []
    for r in level.iter_rows(named=True):
        levels.append({
            "bloc": r["bloc"],
            "bloc_label": BLOC_LABELS.get(r["bloc"], r["bloc"]),
            "series_id": r["series_id"],
            "series_label": labels.get(r["series_id"], r["series_id"]),
            "spec": r["spec"],
            "n": int(r["n"]),
            "elections": [int(e) for e in str(r["elections"]).split(",") if e.strip()],
            "expected_sign": int(r["expected_sign"]),
            "slope_per_sd": _r(r["slope_per_sd"], 2),
            "gain": _r(r["gain_vs_m1"], 2),
            "p_placebo": _r(r["p_perm"], 3),
            "testable": _bool(r["testable"]),
            "holds": _bool(r["holds"]),
        })
    dynamic = []
    for r in dyn.iter_rows(named=True):
        dynamic.append({
            "bloc": r["bloc"],
            "bloc_label": BLOC_LABELS.get(r["bloc"], r["bloc"]),
            "series_id": r["series_id"],
            "series_label": labels.get(r["series_id"], r["series_id"]),
            "lag": int(r["lag"]),
            "n": int(r["n"]),
            "n_cycles": int(r["n_cycles"]),
            "effect_per_sd": _r(r["effect_per_sd"], 2),
            "p_placebo": _r(r["p_placebo"], 3),
            "testable": _bool(r["testable"]),
            "holds": _bool(r["holds"]),
        })
    k = sum(x["testable"] for x in levels)
    n_pass = sum(x["testable"] and x["p_placebo"] is not None and x["p_placebo"] < 0.05
                 for x in levels)
    n_hold = sum(x["holds"] for x in levels)
    kd = sum(x["testable"] and x["lag"] >= 1 for x in dynamic)
    nd_hold = sum(x["holds"] for x in dynamic)
    summary = (
        f"Seul point d'entrée possible pour une série annuelle ou semestrielle : le point de "
        f"départ (prior) de chaque bloc, 180 jours avant le vote. En validation croisée par "
        f"élection, {n_hold} test sur {k} tient au niveau et {nd_hold} paire sur {kd} en "
        f"dynamique. Critères fixés avant lecture."
    )
    if lev_path.exists():
        lev = pl.read_csv(lev_path)
        mx = float(lev["delta_points"].abs().max())
        summary += (
            f" Décaler le prior d'un bloc de +0,3 logit (environ +30 % relatifs) ne déplace sa "
            f"part a posteriori que de {_french(mx, 3)} point au plus : un terme d'attitude y "
            f"serait sans effet mesurable. Question close pour le premier tour.")
    fp = (
        f"{k} test{'s' if k > 1 else ''} testable{'s' if k > 1 else ''} au seuil de 5 % non "
        f"corrigé : le hasard donnerait environ "
        f"{_french(0.05 * k, 1)} faux positif{'s' if 0.05 * k >= 2 else ''} ; {n_pass} test"
        f"{'s' if n_pass > 1 else ''} passe{'nt' if n_pass > 1 else ''} le placebo, {n_hold} "
        f"tient tous les critères (signe attendu, signe stable en retirant chaque élection, gain "
        f"hors échantillon d'au moins un point)."
    )
    return {
        "date": _study_date(study, asof),
        "verdict": "negative",
        "summary": summary,
        "findings": list(OPINION_STUDY_FINDINGS),
        "reference": REFERENCE,
        "level": levels,
        "dynamic": dynamic,
        "false_positives_note": fp,
    }


def build_opinion(asof: str, reg: Registry, curated: Path | None = None,
                  study: Path | None = None, registry: dict | None = None) -> dict:
    curated = curated or paths.CURATED
    study = study or paths.RUNS / ATTITUDES_STUDY
    registry = registry or load_registry()
    day = dt.date.fromisoformat(asof)
    meta = {s["series_id"]: s for s in registry["series"]}
    docs = {d["id"]: d for d in registry.get("documents", [])}
    banned = excluded_documents(registry)
    att = pl.read_parquet(curated / "attitudes.parquet").filter(pl.col("date") <= day)

    themes, excluded, used_blocs = [], [], []
    for tid, tlabel, members in OPINION_THEMES:
        series = []
        for sid, bloc in members:
            m = meta.get(sid)
            raw = att.filter(pl.col("series_id") == sid).sort("date")
            if m is None or raw.is_empty():
                continue
            sub = raw.filter(~pl.col("document").is_in(list(banned)))
            dropped = sorted(set(raw["document"].to_list()) - set(sub["document"].to_list()))
            if sub.is_empty():
                excluded.append({"series_id": sid, "label": m["label"], "theme": tid,
                                 "n_values": raw.height,
                                 "reason": " ; ".join(banned[d] for d in dropped)})
                continue
            notes = [_point_note(n) for n in sub["notes"].to_list()]
            common = notes[0] if notes and all(n == notes[0] for n in notes) else None
            last_doc = docs.get(sub["document"][-1], {})
            src = m.get("source", sub["source"][-1])
            series.append({
                "id": sid,
                "label": m["label"],
                "question": _resolve(m, meta, "question") or m["label"],
                "answer": _resolve(m, meta, "answer"),
                "population": POPULATION_LABELS.get(m.get("population"), m.get("population")),
                "source": {"label": SOURCE_LABELS.get(src, src),
                           "href": last_doc.get("url") or "data/registry/attitude_series.yaml"},
                "frequency": FREQUENCY.get(m.get("freq"), "waves"),
                "unit": UNIT_LABELS.get(m.get("unit"), m.get("unit")),
                "bloc": bloc,
                "changes": _resolve(m, meta, "changes"),
                "missing": _resolve(m, meta, "missing"),
                "note": common,
                "breaks": series_breaks(sub, m.get("family"), registry),
                "excluded_values": [{"document": d, "n_values": raw.height - sub.height,
                                     "reason": banned[d]} for d in dropped],
                "points": [{"date": d.isoformat(), "value": round(float(v), 1),
                            "note": None if common else n}
                           for d, v, n in zip(sub["date"].to_list(), sub["value"].to_list(),
                                              notes)],
            })
            if bloc not in used_blocs:
                used_blocs.append(bloc)
        if series:
            themes.append({"id": tid, "label": tlabel, "series": series})
    starts = [dt.date.fromisoformat(s["points"][0]["date"]) for t in themes for s in t["series"]]
    return {
        "in_forecast": False,
        "note": OPINION_NOTE,
        "first_rounds": first_rounds(reg, min(starts, default=day), day),
        "themes": themes,
        "bloc_shares": bloc_shares(curated, day, used_blocs, reg),
        "excluded": excluded,
        "study": opinion_study(study, asof,
                               {k: SHORT_LABELS.get(k, v["label"]) for k, v in meta.items()}),
    }


# --------------------------------------------------------------------- image du RN, 2d tour


def _duel_label(duel: str, reg: Registry) -> str:
    names = []
    for cid in duel.split("–"):
        c = reg.candidates.get(cid)
        names.append(c.name if c is not None else cid)
    return " – ".join(names)


def build_rn_image(asof: str, reg: Registry, curated: Path | None = None,
                   study: Path | None = None, registry: dict | None = None) -> dict:
    from vvs.model import rn_image as R

    curated = curated or paths.CURATED
    study = study or paths.RUNS / RN_IMAGE_STUDY
    registry = registry or load_registry()
    day = dt.date.fromisoformat(asof)
    meta = {s["series_id"]: s for s in registry["series"]}
    docs = {d["id"]: d for d in registry.get("documents", [])}
    att = pl.read_parquet(curated / "attitudes.parquet").filter(pl.col("date") <= day)

    items = []
    for iid, sid, label in RN_ITEMS:
        sub = att.filter(pl.col("series_id") == sid).sort("date")
        m = meta.get(sid, {})
        items.append({
            "id": iid,
            "series_id": sid,
            "label": label,
            "question": _resolve(m, meta, "question") or label,
            "answer": _resolve(m, meta, "answer"),
            "breaks": series_breaks(sub, m.get("family"), registry),
            "points": [{"date": d.isoformat(), "value": round(float(v), 1)}
                       for d, v in zip(sub["date"].to_list(), sub["value"].to_list())],
        })
    m0 = meta.get(RN_ITEMS[0][1], {})
    last_doc = att.filter(pl.col("series_id") == RN_ITEMS[0][1]).sort("date")["document"]
    href = docs.get(last_doc[-1], {}).get("url") if last_doc.len() else None
    starts = [dt.date.fromisoformat(i["points"][0]["date"]) for i in items if i["points"]]

    obs = pl.read_csv(study / "u_observed.csv").sort("election")
    rel = json.loads((study / "relation.json").read_text())
    bar = pl.read_csv(study / "barometer_values.csv")
    d = rel["items"]["danger"]
    elections = [int(e) for e in obs["election"].to_list()]
    by_election = {}
    for r in obs.iter_rows(named=True):
        by_election[int(r["election"])] = {
            "election": int(r["election"]),
            "duel": _duel_label(r["duel"], reg),
            "pollsters": int(r["pollsters"]),
            "u": round(float(r["u_survey"]), 2),
            "lo90": round(float(r["lo90"]), 2),
            "hi90": round(float(r["hi90"]), 2),
            "u_result": round(float(r["u_result"]), 2),
            "rn_result": round(float(r["rn_result"]), 1),
        }
    wave = {int(r["point"]): r["wave"] for r in
            bar.filter(pl.col("item") == "danger").iter_rows(named=True)}
    points = []
    for e in elections:
        x = d["x"].get(str(e), d["x"].get(e))
        if x is None:
            continue
        o = by_election[e]
        points.append({"election": e, "x": float(x), "wave": str(wave.get(e) or "")[:10] or None,
                       "u": o["u"], "lo90": o["lo90"], "hi90": o["hi90"],
                       "u_result": o["u_result"],
                       "predicted_without": _r(rel["loeo"][str(e)]["u_hat"], 2)
                       if str(e) in rel.get("loeo", {}) else None})
    xs = [p["x"] for p in points]
    x2027 = d.get("x2027")
    lo = min(xs + ([x2027] if x2027 is not None else [])) - 3
    hi = max(xs) + 3
    lf = R.LineFit(d["wls_a"], d["wls_slope"], d["wls_x0"], np.array(d["wls_cov"]),
                   float(rel["tau"]), len(points), d["wls_rss"])
    line = []
    for x in np.linspace(lo, hi, 25):
        c, sd = lf.predict(float(x))
        line.append({"x": round(float(x), 2), "u": round(c, 3), "lo90": round(c - Z90 * sd, 3),
                     "hi90": round(c + Z90 * sd, 3)})
    h12 = rel.get("h2012")
    scatter = {
        "n": len(points),
        "x_label": "« danger pour la démocratie » (% oui, vague avant le vote)",
        "points": points,
        "line": line,
        "slope_ols": round(float(d["ols_slope"]), 3),
        "slope_ols_lo90": round(float(d["ols_lo90"]), 3),
        "slope_ols_hi90": round(float(d["ols_hi90"]), 3),
        "r2": round(float(d["r2"]), 2),
        "x_2027": _r(x2027, 1),
        "hypothetical_2012": None if not h12 or "c4_x2012" not in d else {
            "x": float(d["c4_x2012"]), "u": round(float(h12["u"]), 2),
            "lo90": round(float(h12["u"] - Z90 * h12["se"]), 2),
            "hi90": round(float(h12["u"] + Z90 * h12["se"]), 2), "n_duels": int(h12["n_duels"])},
    }

    duels_csv = pl.read_csv(study / "prior_2027_duels.csv")
    duels = []
    for (duel,), g in duels_csv.group_by(["duel"], maintain_order=True):
        rn, other = duel.split("–")
        rows = []
        for r in g.iter_rows(named=True):
            vid, vlabel = RN_VARIANTS.get(r["variant"], (r["variant"], r["variant"]))
            rows.append({
                "polls": RN_POLLS.get(r["polls"], r["polls"]),
                "variant": vid,
                "label": vlabel,
                "delta": round(float(r["delta"]), 2),
                "sigma_u": round(float(r["sigma_u"]), 2),
                "median": round(float(r["median"]), 1),
                "q05": round(float(r["q05"]), 1),
                "q95": round(float(r["q95"]), 1),
                "p_rn": round(float(r["p_rn"]), 4),
            })
        n_polls = int(g.filter(pl.col("polls") == "avec sondages")["n_polls"].max() or 0)
        duels.append({"rn_candidate": rn, "other_candidate": other, "n_polls": n_polls,
                      "rows": rows})

    c2 = by_election.get(2022)
    pred22 = rel["loeo"].get("2022", {}).get("u_hat") if rel.get("loeo") else None
    note = (
        "Le centre des reports du modèle reste son défaut. "
        f"Trois élections seulement (n = {len(points)}) : la relation entre l'image du RN et le "
        "décalage des reports a le bon signe mais tient au seul point 2002 (choc du 21 avril, "
        "duel droite – extrême droite)"
    )
    if pred22 is not None and c2 is not None:
        note += (f" ; estimée sur 2002 et 2017, elle aurait prédit u = {_french(pred22, 2, True)} "
                 f"pour 2022, observé {_french(c2['u'], 2, True)}")
    note += ". Les parts par duel avec centrage sont des sensibilités, pas des prévisions."
    return {
        "in_forecast": False,
        "note": note,
        "study_date": _study_date(study, asof),
        "reference": REFERENCE,
        "barometer": {
            "items": items,
            "source": {"label": SOURCE_LABELS["verian"],
                       "href": href or "data/registry/attitude_series.yaml"},
            "population": POPULATION_LABELS.get(m0.get("population"), m0.get("population")),
            "changes": _resolve(m0, meta, "changes"),
            "first_rounds": first_rounds(reg, min(starts, default=day), day),
        },
        "u_definition": RN_U_DEFINITION,
        "u_observed": [by_election[e] for e in elections],
        "scatter": scatter,
        "duels_note": RN_DUELS_NOTE,
        "duels": duels,
    }
