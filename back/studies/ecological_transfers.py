"""Étude : reports de voix du second tour par inférence écologique sur les communes.

    uv run python studies/ecological_transfers.py [--out runs/<date>-ecological] [--boot 200]

Entrées : data/curated/results_<année>.parquet (`vvs parse interieur --election <année>`),
data/curated/communes_socio.parquet (`vvs parse conflit-politique`), data/registry/transfers.yaml.
Méthodes dans vvs.model.ecological.

Critères et choix fixés avant lecture des résultats (2026-10-22) :

(1) Méthodes pré-enregistrées, sur les communes de métropole appariées aux indicateurs :
    (a) `pooled` : Goodman contraint, une matrice nationale, pondérée par les inscrits du
        premier tour ; intervalles à 90 % par bootstrap de communes (`--boot` tirages) ;
    (b) `income_urban` : la même par strate, quintile de revenu par habitant (`income_ratio`,
        quintiles pondérés par les inscrits) × type de commune (rural : agglomération de moins
        de 2 000 habitants ; town : de 2 000 à 100 000 ; metro : 100 000 et plus), 15 strates,
        agrégée au national par la taille de l'électorat dans chaque strate ; bootstrap des
        communes dans chaque strate.
    Électorats : finalistes et candidats d'au moins 4 % seuls, les autres regroupés par bloc,
    blancs et nuls, abstention.
(2) Validation : la méthode est validée si, pour 2017 et 2022, pour CHACUN des grands
    électorats (2017 : Mélenchon, Fillon, Hamon ; 2022 : Mélenchon, Pécresse, Zemmour, Jadot),
    l'écart à la moyenne des enquêtes (transfer_prior : toutes les sources de transfers.yaml,
    FES 2017 comprise) est inférieur à 8 points sur la part du premier finaliste parmi les
    exprimés de l'électorat, et à 10 points sur la part « ni l'un ni l'autre » (blancs, nuls,
    abstention, en % de l'électorat).
(3) Proposition pour transfers.yaml : la méthode (b), sauf si (a) est validée et (b) non.
    Si aucune n'est validée, la proposition est faite avec le meilleur modèle exploratoire
    (point 4) et signalée comme non validée.
(4) Exploratoire, déclaré comme tel : strates selon le contexte politique (part de la
    candidate ou du candidat FN/RN au premier tour parmi les exprimés de la commune), qui
    autorise des reports différents là où l'extrême droite est forte. Ensemble de candidats
    fixé : `rn_urban` (quintile RN × type de commune), `rn10` (décile RN), `rn_income`
    (quintile RN × quintile de revenu), plus (b). Règle de choix : le plus petit écart
    normalisé maximal en 2017 seulement, max(|écart répartition| / 8, |écart ni-ni| / 10) ;
    2022 sert d'échantillon de contrôle. (Les écarts 2022 de ces quatre variantes ont été vus
    pendant l'exploration ; la règle de choix ne les utilise pas.)
(5) Valeur de la décomposition sociale (point 3c de la demande) : électorats harmonisés entre
    2017 et 2022 (Mélenchon, autre gauche radicale, gauche, droite, autre extrême droite,
    autres, finalistes, blancs et nuls, abstention). Les coefficients 2017 par strate sont
    appliqués à la composition 2022 (tailles des électorats 2022 dans chaque strate, strates
    2022 définies avec les bornes 2017) et comparés aux reports 2017 bruts (même
    coefficients, composition 2017). La décomposition a de la valeur si elle fait mieux que
    les reports bruts sur les deux cibles suivantes : (T1) écart absolu moyen aux enquêtes
    2022 sur les électorats Mélenchon, droite, gauche, autre extrême droite (répartition et
    ni-ni ensemble) ; (T3) erreur quadratique moyenne, pondérée par les inscrits, de la part
    de Le Pen parmi les exprimés du second tour 2022 prédite commune par commune. (T2) écart
    à l'estimation écologique 2022 du même modèle, rapporté seulement.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # petits produits matriciels : les threads BLAS coûtent

import argparse
import datetime as dt
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from vvs import paths
from vvs.model import ecological as E
from vvs.model.second_round import (
    KAPPA_PROXIMITY,
    PROXIMITY,
    PROXIMITY_LAMBDA,
    SIGMA_U,
    SIGMA_V,
    proximity_mean,
)
from vvs.registry import BLOCS, default_registry
from vvs.registry.transfers import transfer_prior

ELECTIONS = (2002, 2007, 2012, 2017, 2022)
LEVEL = 0.90
SPLIT_TOL = 8.0
NONE_TOL = 10.0
VALIDATION = {
    2017: ["melenchon_jean_luc", "fillon_francois", "hamon_benoit"],
    2022: ["melenchon_jean_luc", "pecresse_valerie", "zemmour_eric", "jadot_yannick"],
}
PREREGISTERED = ("pooled", "income_urban")
EXPLORATORY = ("income_urban", "rn_urban", "rn10", "rn_income")
FAR_RIGHT_MAIN = {
    2002: "le_pen_jean_marie",
    2007: "le_pen_jean_marie",
    2012: "le_pen_marine",
    2017: "le_pen_marine",
    2022: "le_pen_marine",
}
# Électorats harmonisés 2017 / 2022 (test 3c)
HARMONIZED = {
    2017: {
        "macron_emmanuel": ["macron_emmanuel"],
        "le_pen_marine": ["le_pen_marine"],
        "melenchon_jean_luc": ["melenchon_jean_luc"],
        "far_left_other": ["arthaud_nathalie", "poutou_philippe"],
        "left": ["hamon_benoit"],
        "right": ["fillon_francois"],
        "far_right_other": ["dupont_aignan_nicolas"],
        "other": ["lassalle_jean", "asselineau_francois", "cheminade_jacques"],
    },
    2022: {
        "macron_emmanuel": ["macron_emmanuel"],
        "le_pen_marine": ["le_pen_marine"],
        "melenchon_jean_luc": ["melenchon_jean_luc"],
        "far_left_other": ["arthaud_nathalie", "poutou_philippe", "roussel_fabien"],
        "left": ["jadot_yannick", "hidalgo_anne"],
        "right": ["pecresse_valerie"],
        "far_right_other": ["zemmour_eric", "dupont_aignan_nicolas"],
        "other": ["lassalle_jean"],
    },
}
TEST_ELECTORATES = ["melenchon_jean_luc", "right", "left", "far_right_other"]
# Décomposition sociale : (modèle, dimension agrégée)
DECOMPOSITIONS = [
    ("income_urban", "income"),
    ("income_urban", "urban"),
    ("rn_income", "income"),
    ("rn_urban", "urban"),
    ("rn_edu", "edu"),
    ("rn_workers", "workers"),
]
DECOMP_ELECTIONS = (2017, 2022)
WORKERS = 12


# --- données


@dataclass
class Data:
    election: int
    groups: E.Groups
    frame: pl.DataFrame  # communes appariées, avec indicateurs
    X: np.ndarray
    Y: np.ndarray
    w: np.ndarray
    coverage: pl.DataFrame


def load_data(election: int, socio: pl.DataFrame, groups: E.Groups | None = None) -> Data:
    groups = groups or E.default_groups(election)
    codes = set(socio["codecommune"].unique())
    allc = E.match_communes(E.commune_counts(E.load_results(election), groups), codes)
    cov = E.coverage(allc)
    s = socio.filter(pl.col("year") == election).select(
        pl.col("codecommune").alias("geo_code"),
        "income_ratio", "higher_edu_share", "workers_employees_share", "urban_type",
    )  # fmt: skip
    f = allc.filter(pl.col("match") == "matched").join(s, on="geo_code", how="left")
    # communes sans exprimés au premier tour (part RN indéfinie) : écartées, quelques unités
    f = f.filter(
        pl.col(f"x_{E.BLANK_NULL}") + pl.col(f"x_{E.ABSTAIN}") < pl.col("registered1")
    )
    X, Y, w = E.matrices(f, groups)
    return Data(election, groups, f, X, Y, w, cov)


def rn_share(d: Data) -> np.ndarray:
    k = d.groups.ids.index(FAR_RIGHT_MAIN[d.election])
    expressed = 1.0 - d.X[:, -1] - d.X[:, -2]
    return d.X[:, k] / expressed


def _cuts(values: np.ndarray, w: np.ndarray, q: int) -> np.ndarray:
    ok = ~np.isnan(values)
    order = np.argsort(values[ok])
    cw = np.cumsum(w[ok][order])
    cw = cw / cw[-1]
    return np.array([values[ok][order][np.searchsorted(cw, k / q)] for k in range(1, q)])


def _bin(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    out = 1 + np.searchsorted(cuts, values, side="left")
    return np.where(np.isnan(values), 0, out)


def strata(d: Data, model: str, cuts: dict | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    """(étiquettes de strate, bornes, communes utilisables) pour un modèle ; `cuts` impose
    les bornes (bornes 2017 appliquées à 2022 dans le test 3c). Une commune sans l'indicateur
    nécessaire (quelques dizaines) n'est pas utilisable."""
    cuts = dict(cuts or {})
    w = d.w
    f = d.frame
    valid = np.ones(len(w), dtype=bool)

    def binned(name: str, values: np.ndarray, q: int) -> np.ndarray:
        nonlocal valid
        valid &= ~np.isnan(values)
        if name not in cuts:
            cuts[name] = _cuts(values, w, q)
        return _bin(values, cuts[name])

    def num(col: str) -> np.ndarray:
        return f[col].cast(pl.Float64).fill_null(float("nan")).to_numpy()

    urban = f["urban_type"].fill_null("unknown").to_numpy().astype(str)
    social = {"income": ("income5", "income_ratio"), "edu": ("edu5", "higher_edu_share"),
              "workers": ("workers5", "workers_employees_share")}
    if model == "pooled":
        lab = np.array(["all"] * len(w))
    elif model == "income_urban":
        valid &= urban != "unknown"
        inc = binned("income5", num("income_ratio"), 5)
        lab = np.array([f"{u}_q{b}" for u, b in zip(urban, inc)])
    elif model == "rn_urban":
        valid &= urban != "unknown"
        rn = binned("rn5", rn_share(d), 5)
        lab = np.array([f"{u}_rn{b}" for u, b in zip(urban, rn)])
    elif model == "rn10":
        rn = binned("rn10", rn_share(d), 10)
        lab = np.array([f"rn{b:02d}" for b in rn])
    elif model in ("rn_income", "rn_edu", "rn_workers"):
        name, col = social[model.split("_")[1]]
        rn = binned("rn5", rn_share(d), 5)
        sb = binned(name, num(col), 5)
        lab = np.array([f"rn{a}_q{b}" for a, b in zip(rn, sb)])
    else:
        raise KeyError(model)
    return lab, cuts, valid


def dimension_of(label: str, dim: str) -> str:
    """Valeur de la dimension sociale `dim` d'une étiquette de strate."""
    if dim == "urban":
        return label.split("_")[0]
    return label.split("_")[-1]  # q1…q5


# --- ajustements (parallèles)


def _boot_task(args):
    X, Y, w, n_boot, seed, start = args
    return E.bootstrap(X, Y, w, n_boot, np.random.default_rng(seed), start=start)


def fit_model(d: Data, model: str, n_boot: int, pool: ProcessPoolExecutor | None,
              seed: int, cuts: dict | None = None) -> tuple[E.StratifiedFit, dict]:
    lab, cuts, keep = strata(d, model, cuts)
    X, Y, w, lab = d.X[keep], d.Y[keep], d.w[keep], lab[keep]
    fit = E.stratified(X, Y, w, lab, min_communes=30)
    if n_boot and pool is not None:
        tasks = []
        for i, s in enumerate(fit.strata):
            m = lab == s
            tasks.append((X[m], Y[m], w[m], n_boot, seed + i, fit.B[s]))
        for s, res in zip(fit.strata, pool.map(_boot_task, tasks)):
            fit.boot[s] = res
    return fit, cuts


# --- résumés


def q(a: np.ndarray, p: float) -> np.ndarray:
    return np.nanquantile(a, p, axis=0)


def rows_table(election: int, model: str, groups: E.Groups, B: np.ndarray,
               boot: np.ndarray | None) -> pl.DataFrame:
    split, none = E.split_none(B)
    lo_p, hi_p = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    if boot is not None and len(boot):
        bs, bn = E.split_none(boot)
        blo, bhi = q(boot, lo_p), q(boot, hi_p)
        slo, shi, nlo, nhi = q(bs, lo_p), q(bs, hi_p), q(bn, lo_p), q(bn, hi_p)
    else:
        blo = bhi = np.full_like(B, np.nan)
        slo = shi = nlo = nhi = np.full(len(B), np.nan)
    recs = []
    for k, g in enumerate(groups.ids):
        rec = {"election": election, "model": model, "from": g,
               "members": ",".join(groups.members[g])}
        for j, o in enumerate(["fin1", "fin2", "blank_null", "abstain"]):
            rec[o] = 100 * B[k, j]
            rec[f"{o}_low"] = 100 * blo[k, j]
            rec[f"{o}_high"] = 100 * bhi[k, j]
        rec.update(split=100 * split[k], split_low=100 * slo[k], split_high=100 * shi[k],
                   none=100 * none[k], none_low=100 * nlo[k], none_high=100 * nhi[k])
        recs.append(rec)
    return pl.DataFrame(recs)


def survey_targets(election: int, cid: str, pair: tuple[str, str]) -> dict:
    out = {"survey_split": np.nan, "survey_split_min": np.nan, "survey_split_max": np.nan,
           "survey_none": np.nan, "survey_none_min": np.nan, "survey_none_max": np.nan,
           "sources_split": "", "sources_none": ""}
    try:
        ex = transfer_prior(election, cid, pair, basis="expressed")
        v = [s[pair[0]] for s in ex.per_source.values()]
        out.update(survey_split=ex.shares[pair[0]], survey_split_min=min(v),
                   survey_split_max=max(v), sources_split=",".join(ex.sources))
    except KeyError:
        pass
    try:
        el = transfer_prior(election, cid, pair, basis="electorate")
        v = [s["blank_or_abstain"] for s in el.per_source.values()]
        out.update(survey_none=el.shares["blank_or_abstain"], survey_none_min=min(v),
                   survey_none_max=max(v), sources_none=",".join(el.sources))
    except KeyError:
        pass
    return out


def group_survey(election: int, members: list[str], pair: tuple[str, str]) -> tuple[float, float]:
    """Cible d'enquête d'un électorat harmonisé : moyenne des membres documentés pondérée par
    leur part de premier tour (répartition, puis ni-ni)."""
    reg = default_registry()
    shares = {r.candidate_id: r.share for r in reg.elections[election].round_(1).results}
    s_num = s_den = n_num = n_den = 0.0
    for c in members:
        t = survey_targets(election, c, pair)
        if not math.isnan(t["survey_split"]):
            s_num += shares[c] * t["survey_split"]
            s_den += shares[c]
        if not math.isnan(t["survey_none"]):
            n_num += shares[c] * t["survey_none"]
            n_den += shares[c]
    return (s_num / s_den if s_den else np.nan, n_num / n_den if n_den else np.nan)


def validation_table(rows: pl.DataFrame, election: int, pair: tuple[str, str]) -> pl.DataFrame:
    recs = []
    for r in rows.filter(pl.col("from").is_in(VALIDATION.get(election, []))).iter_rows(named=True):
        t = survey_targets(election, r["from"], pair)
        es = r["split"] - t["survey_split"]
        en = r["none"] - t["survey_none"]
        recs.append({**{k: r[k] for k in ("election", "model", "from", "split", "split_low",
                                          "split_high", "none", "none_low", "none_high")},
                     **t, "err_split": es, "err_none": en,
                     "pass_split": abs(es) < SPLIT_TOL, "pass_none": abs(en) < NONE_TOL})
    return pl.DataFrame(recs)


def normalized_max_error(v: pl.DataFrame) -> float:
    return float(max((v["err_split"].abs() / SPLIT_TOL).max(), (v["err_none"].abs() / NONE_TOL).max()))


# --- décomposition sociale


def decomposition(fit: E.StratifiedFit, groups: E.Groups, election: int, model: str,
                  dim: str) -> pl.DataFrame:
    values = sorted({dimension_of(s, dim) for s in fit.strata})
    recs = []
    lo_p, hi_p = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    for v in values:
        sub = {s: fit.sizes[s] for s in fit.strata if dimension_of(s, dim) == v}
        B = fit.national(sub)
        split, none = E.split_none(B)
        if fit.boot:
            bs, bn = E.split_none(fit.national_boot(sub))
            slo, shi, nlo, nhi = q(bs, lo_p), q(bs, hi_p), q(bn, lo_p), q(bn, hi_p)
        else:
            slo = shi = nlo = nhi = np.full(len(B), np.nan)
        size = sum(sub.values())
        for k, g in enumerate(groups.ids):
            recs.append({"election": election, "model": model, "dimension": dim, "value": v,
                         "from": g, "electorate_registered": float(size[k]),
                         "fin1": 100 * B[k, 0], "fin2": 100 * B[k, 1],
                         "split": 100 * split[k], "split_low": 100 * slo[k],
                         "split_high": 100 * shi[k], "none": 100 * none[k],
                         "none_low": 100 * nlo[k], "none_high": 100 * nhi[k]})
    return pl.DataFrame(recs)


# --- proposition transfers.yaml


def proposal_entry(rows: pl.DataFrame, groups: E.Groups, model: str, out_dir: Path,
                   validated: bool) -> tuple[dict, list[str]]:
    reg = default_registry()
    e = reg.elections[groups.election]
    a, b = groups.finalists
    entry = {
        "id": f"ei_{groups.election}",
        "election": groups.election,
        "pollster": "ecological",
        "title": f"Inférence écologique sur les communes de métropole, second tour {groups.election} "
                 f"(Goodman contraint, strates « {model} »)",
        "url": f"{out_dir.relative_to(paths.ROOT) if out_dir.is_relative_to(paths.ROOT) else out_dir}/report.md",
        "pages": "vvs.model.ecological ; étude back/studies/ecological_transfers.py",
        "fieldwork_start": e.round_(2).date,
        "fieldwork_end": e.round_(2).date,
        "mode": None,
        "base": "first_round_electorate",
        "base_note": (
            "En % des inscrits de chaque électorat de premier tour (vote observé par commune, pas "
            "déclaré), lignes sommant à 100 avec blancs ou nuls et abstention au second tour. "
            "Communes de métropole appariées aux indicateurs de Cagé et Piketty (outre-mer et "
            "étranger exclus)."
        ),
        "columns": "split",
        "notes": (
            "Valeurs calculées, à une décimale. Inférence écologique : le vote des électorats "
            "n'est pas observé, seule la corrélation entre communes l'est (biais d'agrégation). "
            + ("Méthode validée contre les enquêtes 2017 et 2022."
               if validated else
               "Méthode NON validée au critère fixé (écarts aux enquêtes 2017-2022 au-delà de 8 "
               "points sur la répartition ou 10 sur le ni-ni pour au moins un grand électorat).")
            + " Intervalles bootstrap à 90 % en commentaire et dans transfers_ei.csv. Le "
            "producteur `ecological` doit être ajouté à OTHER_PRODUCERS de vvs.registry.transfers."
        ),
        "rows": [],
    }
    comments = []
    for r in rows.iter_rows(named=True):
        g = r["from"]
        row = {"from": g}
        if g not in E.SPECIAL and groups.members[g] != (g,):
            row["members"] = list(groups.members[g])
        vals = [round(r[k], 1) for k in ("fin1", "fin2", "blank_null", "abstain")]
        # arrondi sommant à 100 : l'écart d'arrondi va à la plus grande cellule
        vals[int(np.argmax(vals))] = round(vals[int(np.argmax(vals))] + 100 - sum(vals), 1)
        row["to"] = {a: vals[0], b: vals[1], "blank_null": vals[2], "abstain": vals[3]}
        entry["rows"].append(row)
        comments.append(
            f"{g} : IC90 {a} [{r['fin1_low']:.1f} ; {r['fin1_high']:.1f}], {b} "
            f"[{r['fin2_low']:.1f} ; {r['fin2_high']:.1f}], blancs-nuls [{r['blank_null_low']:.1f} ; "
            f"{r['blank_null_high']:.1f}], abstention [{r['abstain_low']:.1f} ; {r['abstain_high']:.1f}]"
        )
    return entry, comments


class _NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):  # pas d'ancres YAML dans la proposition
        return True


def write_proposal(entries: list[tuple[dict, list[str]]], path: Path) -> None:
    lines = [
        "# Proposition de sources `ei_<année>` pour data/registry/transfers.yaml (NE PAS fusionner",
        "# sans lire runs/…-ecological/report.md). Format identique aux autres sources ; le",
        "# producteur `ecological` est à ajouter à OTHER_PRODUCERS (vvs.registry.transfers).",
        "",
        "transfers:",
    ]
    for entry, comments in entries:
        body = yaml.dump([entry], Dumper=_NoAlias, allow_unicode=True, sort_keys=False, width=100)
        lines.append(f"  # --- {entry['id']} : intervalles bootstrap à 90 % (points d'inscrits)")
        lines.extend(f"  #   {c}" for c in comments)
        lines.extend("  " + ln for ln in body.rstrip().splitlines())
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_proposal(path: Path) -> str:
    """Charge transfers.yaml + la proposition avec le validateur du registre."""
    import vvs.registry.transfers as T

    base = yaml.safe_load(T.default_path().read_text(encoding="utf-8"))
    prop = yaml.safe_load(path.read_text(encoding="utf-8"))
    tmp = path.with_name("_check_transfers.yaml")
    tmp.write_text(yaml.safe_dump({"transfers": base["transfers"] + prop["transfers"]},
                                  allow_unicode=True, sort_keys=False), encoding="utf-8")
    saved = set(T.OTHER_PRODUCERS)
    T.OTHER_PRODUCERS.add("ecological")
    try:
        srcs = T.load_transfers(tmp)
        return f"valide : {len(srcs)} sources ({len(prop['transfers'])} proposées)"
    except T.TransferError as exc:
        return f"INVALIDE : {exc}"
    finally:
        T.OTHER_PRODUCERS.clear()
        T.OTHER_PRODUCERS.update(saved)
        tmp.unlink()


# --- matrice de proximité et σ (point 3b)


LOGIT_CLIP = 0.01  # les coins des estimations contraintes (0 ou 100 %) donneraient ±7 en logit


def logit(p: float) -> float:
    p = min(max(p, LOGIT_CLIP), 1 - LOGIT_CLIP)
    return math.log(p / (1 - p))


def bloc_rows(election: int, groups: E.Groups, B: np.ndarray, sizes: np.ndarray) -> dict:
    """Électorats agrégés par bloc (hors finalistes) : bloc -> (répartition, ni-ni, taille),
    répartition = part du premier finaliste parmi les exprimés."""
    reg = default_registry()
    e = reg.elections[election]
    bloc_of = {c.candidate_id: c.bloc for c in e.candidates}
    acc: dict[str, list] = {}
    for k, g in enumerate(groups.ids):
        if g in groups.finalists:
            continue
        if g == E.ABSTAIN:
            b = "abstain"
        elif g == E.BLANK_NULL:
            continue
        else:
            b = bloc_of[groups.members[g][0]]
        acc.setdefault(b, []).append((sizes[k], B[k]))
    out = {}
    for b, items in acc.items():
        n = sum(s for s, _ in items)
        row = sum(s * r for s, r in items) / n
        out[b] = (row[0] / (row[0] + row[1]), row[2] + row[3], float(n))
    return out


def survey_bloc_rows(election: int, groups: E.Groups) -> dict:
    reg = default_registry()
    e = reg.elections[election]
    shares = {r.candidate_id: r.share for r in e.round_(1).results}
    pair = groups.finalists
    acc: dict[str, list] = {}
    for c in e.candidates:
        if c.candidate_id in pair:
            continue
        t = survey_targets(election, c.candidate_id, pair)
        acc.setdefault(c.bloc, []).append((shares[c.candidate_id], t["survey_split"], t["survey_none"]))
    t = survey_targets(election, "abstain", pair)
    acc["abstain"] = [(1.0, t["survey_split"], t["survey_none"])]
    out = {}
    for b, items in acc.items():
        sp = [(w, s) for w, s, _ in items if not math.isnan(s)]
        nn = [(w, n) for w, _, n in items if not math.isnan(n)]
        out[b] = (
            sum(w * s for w, s in sp) / sum(w for w, _ in sp) / 100 if sp else np.nan,
            sum(w * n for w, n in nn) / sum(w for w, _ in nn) / 100 if nn else np.nan,
        )
    return out


# --- rapport


def fmt(x: float, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def md_table(df: pl.DataFrame, cols: list[str], heads: list[str] | None = None, d: int = 1) -> str:
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows(named=True):
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt(v, d) if isinstance(v, float) else ("oui" if v is True else "non" if v is False else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=paths.RUNS / f"{dt.date.today()}-ecological")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    socio = pl.read_parquet(paths.CURATED / "communes_socio.parquet")
    reg = default_registry()
    report: list[str] = []

    with ProcessPoolExecutor(WORKERS) as pool:
        data = {e: load_data(e, socio) for e in ELECTIONS}
        # couverture
        cov = pl.concat([d.coverage.with_columns(pl.lit(e).alias("election")) for e, d in data.items()])
        cov.write_csv(out / "coverage.csv")

        # (1) modèles pré-enregistrés et exploratoires, toutes élections
        fits: dict[tuple[int, str], E.StratifiedFit] = {}
        cuts_by: dict[tuple[int, str], dict] = {}
        rows = []
        models = list(dict.fromkeys([*PREREGISTERED, *EXPLORATORY]))
        for e, d in data.items():
            for i, m in enumerate(models):
                fit, cuts = fit_model(d, m, args.boot, pool, args.seed + 1000 * i + e)
                fits[(e, m)] = fit
                cuts_by[(e, m)] = cuts
                rows.append(rows_table(e, m, d.groups, fit.national(), fit.national_boot() if fit.boot else None))
                print(f"{e} {m} : {len(fit.strata)} strates", flush=True)
        rows_df = pl.concat(rows, how="diagonal")
        rows_df.write_csv(out / "transfers_ei.csv")

        # (2) validation
        val = pl.concat([
            validation_table(rows_df.filter((pl.col("election") == e) & (pl.col("model") == m)), e,
                             data[e].groups.finalists)
            for e in VALIDATION for m in models
        ])
        val.write_csv(out / "validation.csv")
        verdict = {}
        for m in models:
            v = val.filter(pl.col("model") == m)
            verdict[m] = bool(v["pass_split"].all() and v["pass_none"].all())
        score17 = {m: normalized_max_error(val.filter((pl.col("model") == m) & (pl.col("election") == 2017)))
                   for m in EXPLORATORY}
        chosen = min(score17, key=score17.get)
        if verdict["income_urban"] or not verdict["pooled"]:
            proposal_model = "income_urban" if verdict["income_urban"] else chosen
        else:
            proposal_model = "pooled"

        # (3a) proposition transfers.yaml
        entries = []
        for e, d in data.items():
            r = rows_df.filter((pl.col("election") == e) & (pl.col("model") == proposal_model))
            entries.append(proposal_entry(r, d.groups, proposal_model, out, verdict[proposal_model]))
        write_proposal(entries, out / "transfers_proposal.yaml")
        proposal_check = check_proposal(out / "transfers_proposal.yaml")

        # décomposition sociale
        dec = []
        for e in DECOMP_ELECTIONS:
            for i, (m, dim) in enumerate(DECOMPOSITIONS):
                if (e, m) not in fits:
                    fit, cuts = fit_model(data[e], m, args.boot, pool, args.seed + 50000 + 100 * i + e)
                    fits[(e, m)] = fit
                    cuts_by[(e, m)] = cuts
                dec.append(decomposition(fits[(e, m)], data[e].groups, e, m, dim))
        dec_df = pl.concat(dec)
        dec_df.write_csv(out / "decomposition.csv")

        # (3c) test 2017 -> 2022
        hd = {e: load_data(e, socio, E.custom_groups(e, HARMONIZED[e])) for e in (2017, 2022)}
        test_rows = []
        commune_rows = []
        pair22 = hd[2022].groups.finalists
        targets = {g: group_survey(2022, HARMONIZED[2022][g], pair22) for g in TEST_ELECTORATES}
        surveys17 = {g: group_survey(2017, HARMONIZED[2017][g], hd[2017].groups.finalists)
                     for g in TEST_ELECTORATES}
        for m in ("income_urban", *[x for x in (chosen, "rn_income") if x != "income_urban"]):
            f17, c17 = fit_model(hd[2017], m, 0, None, 0)
            lab22, _, keep = strata(hd[2022], m, c17)
            X22, Y22, w22, lab22 = hd[2022].X[keep], hd[2022].Y[keep], hd[2022].w[keep], lab22[keep]
            sizes22 = {s: E.electorate_sizes(X22[lab22 == s], w22[lab22 == s]) for s in f17.strata}
            f22, _ = fit_model(hd[2022], m, 0, None, 0)
            B_raw, B_rw, B_22 = f17.national(), f17.national(sizes22), f22.national()
            pooled17 = E.goodman(hd[2017].X, hd[2017].Y, hd[2017].w)
            ids = hd[2022].groups.ids
            for label, B in (("raw_2017", B_raw), ("reweighted_2017", B_rw), ("pooled_2017", pooled17),
                             ("ei_2022", B_22)):
                s, n = E.split_none(B)
                for g in TEST_ELECTORATES:
                    k = ids.index(g)
                    test_rows.append({"model": m, "prediction": label, "from": g,
                                      "split": 100 * s[k], "none": 100 * n[k],
                                      "survey_split": targets[g][0], "survey_none": targets[g][1],
                                      "ei22_split": 100 * E.split_none(B_22)[0][k],
                                      "ei22_none": 100 * E.split_none(B_22)[1][k]})
            # communes 2022
            ok = (Y22[:, 0] + Y22[:, 1]) > 0
            X22, Y22, w22, lab22 = X22[ok], Y22[ok], w22[ok], lab22[ok]
            actual = Y22[:, 1] / (Y22[:, 0] + Y22[:, 1])
            for label in ("raw_2017", "reweighted_2017", "pooled_2017"):
                if label == "reweighted_2017":
                    Yh = np.vstack([X22[i] @ f17.B[lab22[i]] for i in range(len(w22))])
                else:
                    Yh = X22 @ (B_raw if label == "raw_2017" else pooled17)
                pred = Yh[:, 1] / (Yh[:, 0] + Yh[:, 1])
                ww = w22 / w22.sum()
                nat_pred = (w22 * Yh[:, 1]).sum() / (w22 * (Yh[:, 0] + Yh[:, 1])).sum()
                nat_act = (w22 * Y22[:, 1]).sum() / (w22 * (Y22[:, 0] + Y22[:, 1])).sum()
                commune_rows.append({
                    "model": m, "prediction": label,
                    "rmse_le_pen_share": 100 * math.sqrt(float((ww * (pred - actual) ** 2).sum())),
                    "rmse_none_share": 100 * math.sqrt(float((ww * ((Yh[:, 2] + Yh[:, 3]) - (Y22[:, 2] + Y22[:, 3])) ** 2).sum())),
                    "national_le_pen_pred": 100 * nat_pred, "national_le_pen_actual": 100 * nat_act,
                    "national_none_pred": 100 * float((ww * (Yh[:, 2] + Yh[:, 3])).sum()),
                    "national_none_actual": 100 * float((ww * (Y22[:, 2] + Y22[:, 3])).sum()),
                })
        for g in TEST_ELECTORATES:  # référence : reports d'enquête 2017 tels quels (prior analogue)
            test_rows.append({"model": "surveys", "prediction": "survey_2017", "from": g,
                              "split": surveys17[g][0], "none": surveys17[g][1],
                              "survey_split": targets[g][0], "survey_none": targets[g][1],
                              "ei22_split": np.nan, "ei22_none": np.nan})
        test_df = pl.DataFrame(test_rows).with_columns(
            (pl.col("split") - pl.col("survey_split")).abs().alias("abs_err_split"),
            (pl.col("none") - pl.col("survey_none")).abs().alias("abs_err_none"),
            (pl.col("split") - pl.col("ei22_split")).abs().alias("abs_err_split_ei22"),
            (pl.col("none") - pl.col("ei22_none")).abs().alias("abs_err_none_ei22"),
        )
        test_df.write_csv(out / "transfer_test_rows.csv")
        commune_df = pl.DataFrame(commune_rows)
        commune_df.write_csv(out / "transfer_test_communes.csv")
        test_summary = (
            test_df.filter(pl.col("prediction") != "ei_2022")
            .group_by("model", "prediction")
            .agg(pl.col("abs_err_split").mean().alias("mae_split"),
                 pl.col("abs_err_none").mean().alias("mae_none"),
                 ((pl.col("abs_err_split") + pl.col("abs_err_none")) / 2).mean().alias("mae_t1"),
                 ((pl.col("abs_err_split_ei22") + pl.col("abs_err_none_ei22")) / 2).mean().alias("mae_t2"))
            .join(commune_df, on=["model", "prediction"], how="left")
            .sort("model", "prediction")
        )
        test_summary.write_csv(out / "transfer_test_summary.csv")

    # (3b) proximité : cellules observées, matrice contre enquêtes et EI (modèle retenu)
    prox_rows = []
    for e, d in data.items():
        fin = d.groups.finalists
        bi, bj = (reg.bloc_of(c, e) for c in fin)
        fit = fits[(e, proposal_model)]
        sizes = sum(fit.sizes.values())
        ei = bloc_rows(e, d.groups, fit.national(), sizes)
        sv = survey_bloc_rows(e, d.groups)
        # orientation canonique (ordre de BLOCS) : 2007 et 2012 comparables
        flip = BLOCS.index(bi) > BLOCS.index(bj)

        def orient(x: float, flip: bool = flip) -> float:
            return 100 - x if flip else x

        for b in sorted(PROXIMITY):
            key = "abstain" if b == "abstain" else b
            if key not in ei and key not in sv:
                continue
            mi, mj, mn = proximity_mean(b, bi, bj)
            prox_rows.append({
                "election": e, "duel": f"{bj}-{bi}" if flip else f"{bi}-{bj}", "electorate": b,
                "matrix_split": orient(100 * mi / (mi + mj)), "matrix_none": 100 * mn,
                "survey_split": orient(100 * sv.get(key, (np.nan, np.nan))[0]),
                "survey_none": 100 * sv.get(key, (np.nan, np.nan))[1],
                "ei_split": orient(100 * ei.get(key, (np.nan, np.nan, 0))[0]),
                "ei_none": 100 * ei.get(key, (np.nan, np.nan, 0))[1],
                "size": ei.get(key, (0, 0, 0.0))[2],
            })
    prox = pl.DataFrame(prox_rows)

    def lg(col: str) -> pl.Expr:
        return pl.col(col).map_elements(lambda x: None if math.isnan(x) else logit(x / 100), return_dtype=pl.Float64)

    prox = prox.with_columns(
        (lg("survey_split") - lg("matrix_split")).alias("dlogit_split_survey"),
        (lg("ei_split") - lg("matrix_split")).alias("dlogit_split_ei"),
        (lg("survey_none") - lg("matrix_none")).alias("dlogit_none_survey"),
        (lg("ei_none") - lg("matrix_none")).alias("dlogit_none_ei"),
        (lg("survey_split") / PROXIMITY_LAMBDA).alias("implied_gap_survey"),
        (lg("ei_split") / PROXIMITY_LAMBDA).alias("implied_gap_ei"),
    )
    prox.write_csv(out / "proximity_cells.csv")
    # décalage commun par élection (moyenne pondérée des électorats non finalistes, hors
    # finaliste de leur propre bloc) et dispersion résiduelle
    shifts = []
    for e in ELECTIONS:
        p = prox.filter((pl.col("election") == e) & (pl.col("size") > 0))
        rec = {"election": e}
        for src in ("survey", "ei"):
            for kind in ("split", "none"):
                c = f"dlogit_{kind}_{src}"
                q_ = p.filter(pl.col(c).is_not_null())
                if q_.height == 0:
                    rec[f"u_{kind}_{src}"] = rec[f"sd_{kind}_{src}"] = np.nan
                    continue
                ww = q_["size"].to_numpy()
                x = q_[c].to_numpy()
                mu = float((ww * x).sum() / ww.sum())
                rec[f"u_{kind}_{src}"] = mu
                rec[f"sd_{kind}_{src}"] = float(np.sqrt((ww * (x - mu) ** 2).sum() / ww.sum()))
        shifts.append(rec)
    shifts_df = pl.DataFrame(shifts)
    shifts_df.write_csv(out / "proximity_shifts.csv")
    # analogues : même type de duel à deux élections (2017/2022, 2007/2012), enquêtes et EI
    analog = []
    for a, b in ((2017, 2022), (2022, 2017), (2007, 2012), (2012, 2007)):
        pa = prox.filter(pl.col("election") == a)
        pb = prox.filter(pl.col("election") == b)
        j = pa.join(pb, on="electorate", suffix="_target")
        for src in ("survey", "ei"):
            for kind in ("split", "none"):
                jj = j.filter(pl.col(f"{src}_{kind}").is_not_null() & pl.col(f"{src}_{kind}_target").is_not_null()
                              & ~pl.col(f"{src}_{kind}").is_nan() & ~pl.col(f"{src}_{kind}_target").is_nan())
                if jj.height == 0:
                    continue
                d_ = [logit(t / 100) - logit(s / 100) for s, t in zip(jj[f"{src}_{kind}"], jj[f"{src}_{kind}_target"])]
                ww = jj["size_target"].to_numpy()
                ww = ww if ww.sum() > 0 else np.ones(len(d_))
                analog.append({"from": a, "to": b, "source": src, "kind": kind, "n": len(d_),
                               "u": float(np.average(d_, weights=ww)),
                               "sd": float(np.sqrt(np.average((np.array(d_) - np.average(d_, weights=ww)) ** 2, weights=ww)))})
    analog_df = pl.DataFrame(analog)
    analog_df.write_csv(out / "proximity_analogs.csv")

    # --- rapport
    report += [
        f"# Reports de voix par inférence écologique ({dt.date.today()})",
        "",
        ("Étude `back/studies/ecological_transfers.py`, méthodes `vvs.model.ecological`. Critères "
        "fixés avant lecture : voir l'en-tête du script. Bootstrap : "
        f"{args.boot} tirages par strate, intervalles à 90 %."),
        "",
        "## Couverture de l'appariement",
        "",
        md_table(cov.select("election", "match", "communes", "registered", "registered_share"),
                 ["election", "match", "communes", "registered", "registered_share"], d=4),
        "",
        "## Validation contre les enquêtes (2017, 2022)",
        "",
        ("Écarts EI − moyenne des enquêtes ; répartition = part de Macron parmi les exprimés de "
        "l'électorat ; ni-ni = blancs, nuls, abstention en % de l'électorat."),
        "",
        md_table(val.sort("model", "election"), ["model", "election", "from", "split", "split_low", "split_high",
                                                  "survey_split", "err_split", "none", "survey_none",
                                                  "err_none", "pass_split", "pass_none"]),
        "",
        "Verdict au critère : " + ", ".join(f"{m} {'validé' if v else 'non validé'}" for m, v in verdict.items()),
        "",
        "Score exploratoire 2017 (max des écarts normalisés) : "
        + ", ".join(f"{m} {s:.2f}" for m, s in score17.items()) + f" ; retenu : {chosen}.",
        f"Modèle de la proposition : {proposal_model} ; contrôle du fichier : {proposal_check}.",
        "",
        "## Test 2017 → 2022 (valeur de la décomposition)",
        "",
        md_table(test_summary, ["model", "prediction", "mae_split", "mae_none", "mae_t1", "mae_t2",
                                "rmse_le_pen_share", "national_le_pen_pred", "national_le_pen_actual"], d=2),
        "",
        md_table(test_df.sort("model", "from", "prediction"),
                 ["model", "prediction", "from", "split", "survey_split", "ei22_split", "none",
                  "survey_none", "ei22_none"]),
        "",
        "## Décomposition sociale (2017, 2022)",
        "",
        ("Part de Macron parmi les exprimés (split) et ni-ni de l'électorat, par classe de la "
        "dimension sociale (agrégat des strates de la classe). Modèles `rn_*` : strates croisant "
        "le quintile RN et la dimension sociale (exploratoires)."),
        "",
        md_table(
            dec_df.filter(pl.col("from").is_in(["fillon_francois", "pecresse_valerie", "melenchon_jean_luc",
                                                "hamon_benoit", "jadot_yannick", "zemmour_eric",
                                                "dupont_aignan_nicolas"]))
            .sort("election", "from", "model", "dimension", "value"),
            ["election", "from", "model", "dimension", "value", "split", "split_low", "split_high",
             "none", "none_low", "none_high"]),
        "",
        "## Proximité : cellules observées (matrice, enquêtes, EI du modèle retenu)",
        "",
        ("Duel `a-b` dans l'ordre des blocs (far_right, right, centre, left, far_left) : split = part "
        "du finaliste du bloc `a` parmi les exprimés de l'électorat ; implied_gap = logit(split) / λ, "
        "l'écart d'attirance a − b que la matrice devrait porter pour reproduire la cellule."),
        "",
        md_table(prox.sort("election", "electorate"),
                 ["election", "duel", "electorate", "matrix_split", "survey_split", "ei_split",
                  "matrix_none", "survey_none", "ei_none", "implied_gap_survey", "implied_gap_ei"]),
        "",
        "## Proximité : décalages communs (logit) par élection",
        "",
        md_table(shifts_df, list(shifts_df.columns), d=2),
        "",
        md_table(analog_df, list(analog_df.columns), d=2),
        "",
        (f"Constantes actuelles : λ = {PROXIMITY_LAMBDA}, κ_proximité = {KAPPA_PROXIMITY}, "
        f"σ_u = {SIGMA_U}, σ_v = {SIGMA_V}."),
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
