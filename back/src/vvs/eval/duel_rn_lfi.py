"""Duel RN contre LFI : les législatives comme analogues des reports (étude, hors prévision).

Protocole et critères : docs/results/2026-11-24-duel-rn-lfi.md (écrits avant calcul). Données :
data/curated/legislative_duels.parquet (vvs.ingest.legislative_duels).

Pour chaque circonscription d'un duel, les électorats de premier tour sont ceux du modèle de
second tour (`vvs.model.second_round`) : électorat propre de chaque finaliste, autres candidats
regroupés par bloc, et `abstain` (inscrits − exprimés du premier tour). On compare deux priors
des reports, appliqués au premier tour de chaque circonscription :

- (a) `proximity_priors` : ceux que le modèle construit aujourd'hui pour un duel `far_right`
  contre `far_left` (cas inédit : matrice `PROXIMITY`, κ = 6, σ_u = 0,60, σ_v = 0,40) ;
- (b) `analog_priors` : mêmes électorats propres, autres électorats estimés par régression de
  Goodman contrainte sur des duels d'une autre année (lignes des finalistes fixées au prior
  propre du modèle), κ et décalages communs d'un analogue d'une seule élection.

Rien ici ne modifie le modèle : les fonctions du modèle sont appelées telles quelles, avec ses
constantes par défaut.

Rejouer : `uv run --project back python -m vvs.eval.duel_rn_lfi` (écrit dans
runs/2026-11-24-duel-rn-lfi/).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import gaussian_kde

from vvs import paths
from vvs.model.ecological import solve_constrained, sufficient_stats
from vvs.model.second_round import (
    KAPPA_ANALOG,
    KAPPA_PROXIMITY,
    SIGMA_U,
    SIGMA_V,
    GroupPrior,
    build_groups,
    draw_transfers,
    duel_share,
    own_prior,
)
from vvs.registry import default_registry
from vvs.registry.transfers import default_transfers

OUT_DIR = paths.RUNS / "2026-11-24-duel-rn-lfi"
SURVEYS_FILE = paths.REGISTRY / "transfers_legislatives.yaml"

BLOC_GROUPS = ("far_left", "left", "centre", "right", "far_right", "other")
GROUPS = ("own_fr", "own_opp", *BLOC_GROUPS, "abstain")
OUTCOMES = ("fr", "opp", "none")  # RN, adversaire, ni l'un ni l'autre (abstention, blanc, nul)
MODEL_KEY = {"own_fr": "own:fr", "own_opp": "own:opp", "abstain": "abstain",
             **{b: f"bloc:{b}" for b in BLOC_GROUPS}}  # fmt: skip
TARGET_YEAR = 2027  # le prior (a) est celui que le modèle applique en 2027
MIN_WEIGHT = 0.01  # poids cumulé (part des inscrits) sous lequel un électorat n'est pas estimé
N_DRAWS = 4000
SEED = 20261124
LOG_FLOOR = 1e-6
DUEL_TYPES = ("rn_lfi", "rn_left_other", "rn_centre", "rn_right")


# --- données


def load_duels(path: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet(path or paths.CURATED / "legislative_duels.parquet")


def design(duels: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, w) : électorats du premier tour (G = len(GROUPS)) et issues du second tour
    (RN, adversaire, ni l'un ni l'autre), en parts des inscrits de chaque tour ; w = inscrits
    du premier tour."""
    r1 = duels["registered1"].to_numpy().astype(float)
    r2 = duels["registered2"].to_numpy().astype(float)
    cols = [duels["fr_votes1"], duels["opp_votes1"], *[duels[f"votes1_{b}"] for b in BLOC_GROUPS],
            duels["registered1"] - duels["expressed1"]]  # fmt: skip
    X = np.column_stack([c.to_numpy().astype(float) for c in cols]) / r1[:, None]
    fr2 = duels["fr_votes2"].to_numpy().astype(float)
    op2 = duels["opp_votes2"].to_numpy().astype(float)
    Y = np.column_stack([fr2, op2, r2 - fr2 - op2]) / r2[:, None]
    return X, Y, r1


def observed_share(duels: pl.DataFrame) -> np.ndarray:
    return duels["rn_share2"].to_numpy().astype(float)


# --- estimation écologique


def fit_transfers(
    X: np.ndarray, Y: np.ndarray, w: np.ndarray, fixed: Mapping[int, np.ndarray] | None = None
) -> np.ndarray:
    """Goodman contraint (lignes sur le simplexe), pondéré par `w` ; les lignes `fixed`
    (indice -> vecteur des issues) sont imposées et retirées de l'estimation."""
    fixed = dict(fixed or {})
    free = [g for g in range(X.shape[1]) if g not in fixed]
    Yr = Y.copy()
    for g, v in fixed.items():
        Yr = Yr - np.outer(X[:, g], np.asarray(v, dtype=float))
    G, H = sufficient_stats(X[:, free], Yr, w)
    B = np.empty((X.shape[1], Y.shape[1]))
    B[free] = solve_constrained(G, H)
    for g, v in fixed.items():
        B[g] = v
    return B


def bootstrap_transfers(
    X: np.ndarray, Y: np.ndarray, w: np.ndarray, n_boot: int, rng: np.random.Generator,
    fixed: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:  # fmt: skip
    """Tirages (n_boot, G, 3) par rééchantillonnage des circonscriptions avec remise."""
    n = len(w)
    out = np.empty((n_boot, X.shape[1], Y.shape[1]))
    for b in range(n_boot):
        m = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        keep = m > 0
        out[b] = fit_transfers(X[keep], Y[keep], (w * m)[keep], fixed)
    return out


def group_weights(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Poids cumulé de chaque électorat, en part des inscrits du premier tour."""
    return (X * w[:, None]).sum(0) / w.sum()


# --- priors (a) et (b)


@dataclass(frozen=True)
class Prior:
    """Priors des électorats dans l'ordre de GROUPS, et décalages communs."""

    name: str
    groups: tuple[GroupPrior, ...]
    sigma_u: float
    sigma_v: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def means(self) -> np.ndarray:
        m = np.array([g.mean for g in self.groups], dtype=float)
        return m / m.sum(-1, keepdims=True)


def own_vector(year: int = TARGET_YEAR) -> tuple[float, float, float]:
    """Prior propre du modèle : (vers soi, vers l'adversaire, ni l'un ni l'autre)."""
    return own_prior(year)


def fixed_own_rows(own: Sequence[float]) -> dict[int, np.ndarray]:
    """Lignes des finalistes dans l'ordre des issues (RN, adversaire, ni)."""
    return {GROUPS.index("own_fr"): np.array([own[0], own[1], own[2]]),
            GROUPS.index("own_opp"): np.array([own[1], own[0], own[2]])}  # fmt: skip


def proximity_priors(year: int = TARGET_YEAR) -> Prior:
    """(a) : les priors du modèle pour un duel far_right contre far_left (`build_groups`)."""
    groups, inedit, notes = build_groups(
        ("fr", "opp"), ("far_right", "far_left"), year, set(), own_vector(year),
        default_transfers(), default_registry(),
    )  # fmt: skip
    kind = "inedit" if inedit else "analog"
    return Prior("proximity", tuple(groups[MODEL_KEY[g]] for g in GROUPS),
                 SIGMA_U[kind], SIGMA_V[kind], tuple(notes))  # fmt: skip


def analog_priors(
    B: np.ndarray, weights: np.ndarray, name: str = "analog", year: int = TARGET_YEAR,
    min_weight: float = MIN_WEIGHT, own_from_fit: bool = False,
) -> Prior:  # fmt: skip
    """(b) : électorats non finalistes = lignes de `B` (κ d'un analogue d'une élection,
    décalages communs « analog ») ; un électorat de poids < `min_weight` garde la moyenne de
    proximité (κ = 6). `own_from_fit` : lignes des finalistes prises aussi dans `B`
    (variante, pas le critère)."""
    base = proximity_priors(year)
    kappa = KAPPA_ANALOG[1]
    out, notes = [], []
    for k, (g, p) in enumerate(zip(GROUPS, base.groups)):
        if g.startswith("own_"):
            if own_from_fit:
                out.append(replace(p, mean=tuple(float(x) for x in B[k]), source="analog_fit"))
            else:
                out.append(p)
            continue
        if weights[k] < min_weight:
            out.append(replace(p, kappa=KAPPA_PROXIMITY, source="proximity",
                               note=f"poids {weights[k]:.4f} < {min_weight} : non estimé"))  # fmt: skip
            notes.append(f"{g} : poids {100 * weights[k]:.2f} % des inscrits, moyenne de proximité")
            continue
        v = np.clip(B[k], 0.0, None)
        v = v / v.sum()
        out.append(GroupPrior(p.group, tuple(float(x) for x in v), kappa, True, "analog"))
    return Prior(name, tuple(out), SIGMA_U["analog"], SIGMA_V["analog"], tuple(notes))


# --- prédiction et scores


def predictive_draws(prior: Prior, X: np.ndarray, seed: int = SEED, n: int = N_DRAWS) -> np.ndarray:
    """Tirages (n_circ, n) de la part RN parmi les exprimés, prior seul : pour chaque
    circonscription, `n` tirages des reports appliqués à son premier tour. Même graine par
    circonscription pour toutes les variantes."""
    out = np.empty((X.shape[0], n))
    groups = list(prior.groups)
    for c in range(X.shape[0]):
        rng = np.random.default_rng([seed, c])
        t = draw_transfers(groups, prior.sigma_u, prior.sigma_v, rng, n)
        out[c] = duel_share(X[c], t)
    return out


def point_prediction(means: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Part RN prévue avec les moyennes des reports (sans tirage)."""
    return duel_share(X, np.broadcast_to(means, (X.shape[0], *means.shape)))


def log_scores(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log densité prédictive en y (noyau gaussien, largeur de Scott), plancher LOG_FLOOR."""
    out = np.empty(len(y))
    for c in range(len(y)):
        d = draws[c]
        if np.ptp(d) < 1e-9:
            out[c] = np.log(LOG_FLOOR)
            continue
        out[c] = np.log(max(float(gaussian_kde(d).evaluate([y[c]])[0]), LOG_FLOOR))
    return out


def per_circ_scores(draws: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    med = np.median(draws, axis=1)
    lo90, hi90 = np.quantile(draws, [0.05, 0.95], axis=1)
    lo50, hi50 = np.quantile(draws, [0.25, 0.75], axis=1)
    return {
        "median": med, "abs_error": np.abs(med - y), "signed_error": med - y,
        "log_score": log_scores(draws, y),
        "cov90": (lo90 <= y) & (y <= hi90), "cov50": (lo50 <= y) & (y <= hi50),
        "lo90": lo90, "hi90": hi90, "width90": hi90 - lo90,
    }  # fmt: skip


def summarize_scores(s: Mapping[str, np.ndarray], mask: np.ndarray | None = None) -> dict:
    m = np.ones(len(s["median"]), dtype=bool) if mask is None else mask
    return {
        "n": int(m.sum()),
        "mae": float(s["abs_error"][m].mean()),
        "log_score": float(s["log_score"][m].mean()),
        "cov90": float(s["cov90"][m].mean()),
        "cov50": float(s["cov50"][m].mean()),
        "bias": float(s["signed_error"][m].mean()),
        "width90": float(s["width90"][m].mean()),
    }


def verdict(a: Mapping[str, float], b: Mapping[str, float]) -> dict:
    """Critère écrit avant calcul : (b) bat (a) sur l'EAM et le score logarithmique, et la
    couverture à 90 % de (b) est dans [0,80 ; 0,97]."""
    checks = {
        "mae_b_lower": b["mae"] < a["mae"],
        "log_score_b_higher": b["log_score"] > a["log_score"],
        "cov90_b_in_range": 0.80 <= b["cov90"] <= 0.97,
    }
    return {**checks, "success": all(checks.values())}


# --- tableaux de reports estimés


def transfers_table(
    duels: pl.DataFrame, n_boot: int = 500, seed: int = SEED, fixed_own: bool = False
) -> pl.DataFrame:
    """Reports estimés par électorat, type de duel et année : part vers le RN parmi les
    exprimés de l'électorat (`to_rn_expressed`) et part « ni l'un ni l'autre » (`none`), en
    part de l'électorat ; intervalles à 90 % par bootstrap des circonscriptions."""
    rows = []
    own = own_vector()
    for (election, dtype), d in duels.filter(pl.col("duel_type").is_in(DUEL_TYPES)).group_by(
        ["election", "duel_type"], maintain_order=True
    ):
        if d.height < 10:
            continue
        X, Y, w = design(d)
        fixed = fixed_own_rows(own) if fixed_own else None
        B = fit_transfers(X, Y, w, fixed)
        boot = bootstrap_transfers(X, Y, w, n_boot, np.random.default_rng([seed, d.height]), fixed)
        wg = group_weights(X, w)
        for k, g in enumerate(GROUPS):
            def split(M: np.ndarray) -> np.ndarray:
                tot = M[..., 0] + M[..., 1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(tot > 0, M[..., 0] / tot, np.nan)

            sb, nb = split(boot[:, k]), boot[:, k, 2]
            rows.append({
                "election": election, "duel_type": dtype, "n_circ": d.height, "electorate": g,
                "weight": float(wg[k]), "to_rn": float(B[k, 0]), "to_opp": float(B[k, 1]),
                "none": float(B[k, 2]), "to_rn_expressed": float(split(B[k])),
                "to_rn_expressed_q05": float(np.nanquantile(sb, 0.05)),
                "to_rn_expressed_q95": float(np.nanquantile(sb, 0.95)),
                "none_q05": float(np.quantile(nb, 0.05)), "none_q95": float(np.quantile(nb, 0.95)),
            })  # fmt: skip
    return pl.DataFrame(rows)


# --- enquêtes publiées (diagnostic)


SURVEY_ELECTORATE = {
    # électorat de l'enquête -> électorat du modèle dans un duel RN contre gauche
    "nfp": "own_opp", "nupes": "own_opp", "rn": "own_fr", "rn_uxd": "own_fr",
    "ensemble": "centre", "lr": "right", "lr_dvd": "right", "lr_udi_dvd": "right",
    "reconquete": "far_right", "dvg_hors_nupes": "left",
}  # fmt: skip
FR_KEYS = ("rn",)
OPP_KEYS = ("nfp", "nfp_lfi", "nupes", "nupes_lfi")
NONE_KEYS = ("blank_or_abstain", "non_expressed", "abstain", "blank_null")


def load_surveys(path: Path | None = None) -> list[dict]:
    data = yaml.safe_load((path or SURVEYS_FILE).read_text(encoding="utf-8"))
    return list(data["transfers"])


def survey_means(source: dict, base: Prior) -> tuple[np.ndarray, list[str]]:
    """Moyennes des reports d'après une enquête : électorats couverts pris dans l'enquête
    (répartition RN / adversaire ; part « ni » de l'enquête si elle la donne, sinon celle de
    `base`), électorats non couverts : moyennes de `base`. Renvoie (moyennes, couverts)."""
    means = base.means().copy()
    covered = []
    for row in source["rows"]:
        g = SURVEY_ELECTORATE.get(row["from"])
        if g is None:
            continue
        to = row["to"]
        fr = next((to[k] for k in FR_KEYS if k in to), None)
        op = next((to[k] for k in OPP_KEYS if k in to), None)
        if fr is None or op is None or fr + op == 0:
            continue
        k = GROUPS.index(g)
        none_vals = [to[x] for x in NONE_KEYS if x in to and to[x] is not None]
        none = sum(none_vals) / 100.0 if none_vals else means[k, 2]
        split = fr / (fr + op)
        means[k] = [(1 - none) * split, (1 - none) * (1 - split), none]
        covered.append(g)
    return means, covered


# --- 2027, sensibilité


def first_round_2027(draws_path: Path, summary_path: Path) -> np.ndarray:
    """Électorats (GROUPS) des tirages de la projection où Le Pen et Mélenchon sont qualifiés,
    en points d'exprimés ; abstain = (1 − participation) / participation × 100."""
    d = pl.read_parquet(draws_path)
    blocs = dict(pl.read_parquet(summary_path).select("candidate_id", "bloc").unique().iter_rows())
    d = d.filter((pl.col("qualified_a") == "le_pen_marine")
                 & (pl.col("qualified_b") == "melenchon_jean_luc"))  # fmt: skip
    cands = [c for c in blocs if c in d.columns]
    X = np.zeros((d.height, len(GROUPS)))
    for c in cands:
        v = d[c].fill_null(0.0).fill_nan(0.0).to_numpy()
        if c == "le_pen_marine":
            X[:, 0] += v
        elif c == "melenchon_jean_luc":
            X[:, 1] += v
        else:
            X[:, GROUPS.index(blocs[c])] += v
    t = d["turnout"].to_numpy()
    X[:, -1] = 100.0 * (1 - t) / t
    return X


def prior_share_2027(prior: Prior, X: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Part de Le Pen, prior seul : un tirage des reports par tirage de premier tour."""
    rng = np.random.default_rng([seed, 2027])
    t = draw_transfers(list(prior.groups), prior.sigma_u, prior.sigma_v, rng, len(X))
    return duel_share(X, t)


# --- expérience


def _fmt_prior(p: Prior) -> list[dict]:
    return [{"group": g, "mean": [round(x, 4) for x in gp.mean], "kappa": gp.kappa,
             "source": gp.source} for g, gp in zip(GROUPS, p.groups)]  # fmt: skip


def run(out_dir: Path = OUT_DIR, n_boot: int = 500) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    duels = load_duels()
    train = duels.filter((pl.col("election") == "leg2022") & (pl.col("duel_type") == "rn_lfi"))
    test = duels.filter((pl.col("election") == "leg2024") & (pl.col("duel_type") == "rn_lfi"))
    own = own_vector()
    Xtr, Ytr, wtr = design(train)
    B22 = fit_transfers(Xtr, Ytr, wtr, fixed_own_rows(own))
    w22 = group_weights(Xtr, wtr)
    prior_a = proximity_priors()
    prior_b = analog_priors(B22, w22, "analog_2022")
    B22_free = fit_transfers(Xtr, Ytr, wtr)
    prior_b_free = analog_priors(B22_free, w22, "analog_2022_own_free", own_from_fit=True)

    X, _, w = design(test)
    y = observed_share(test)
    res: dict = {"date": dt.date.today().isoformat(), "seed": SEED, "n_draws": N_DRAWS,
                 "n_train_2022": train.height, "n_test_2024": test.height,
                 "own_prior_2027": list(own),
                 "priors": {p.name: {"sigma_u": p.sigma_u, "sigma_v": p.sigma_v,
                                     "groups": _fmt_prior(p), "notes": list(p.notes)}
                            for p in (prior_a, prior_b, prior_b_free)}}  # fmt: skip
    per_circ = test.select("circ_code", "dept", "opp_name", "fr_name", "rn_share2",
                           "triangular_withdrawn", "withdrawn_nuances")  # fmt: skip
    scores = {}
    for p in (prior_a, prior_b, prior_b_free):
        s = per_circ_scores(predictive_draws(p, X), y)
        scores[p.name] = s
        per_circ = per_circ.with_columns(
            pl.Series(f"{p.name}_median", s["median"]), pl.Series(f"{p.name}_lo90", s["lo90"]),
            pl.Series(f"{p.name}_hi90", s["hi90"]), pl.Series(f"{p.name}_log_score", s["log_score"]),
        )  # fmt: skip
    no_tri = ~test["triangular_withdrawn"].to_numpy()
    res["scores"] = {name: {"all": summarize_scores(s), "no_withdrawal": summarize_scores(s, no_tri)}
                     for name, s in scores.items()}  # fmt: skip
    res["verdict"] = verdict(res["scores"]["proximity"]["all"], res["scores"]["analog_2022"]["all"])

    # variante : validation croisée par département sur 2024 (pas un critère)
    depts = test["dept"].to_numpy()
    Xall, Yall, wall = design(test)
    cv_draws = np.empty((test.height, N_DRAWS))
    for d in sorted(set(depts)):
        m = depts == d
        Bd = fit_transfers(Xall[~m], Yall[~m], wall[~m], fixed_own_rows(own))
        pd_ = analog_priors(Bd, group_weights(Xall[~m], wall[~m]), "cv")
        cv_draws[m] = predictive_draws(pd_, Xall[m])[:, :]
    # mêmes graines par circonscription que les autres variantes
    cv = per_circ_scores(cv_draws, y)
    res["scores"]["analog_2024_cv_dept"] = {"all": summarize_scores(cv),
                                            "no_withdrawal": summarize_scores(cv, no_tri)}  # fmt: skip
    per_circ = per_circ.with_columns(pl.Series("cv_median", cv["median"]))
    B24 = fit_transfers(Xall, Yall, wall, fixed_own_rows(own))
    res["fit_2024_all_own_fixed"] = {g: [round(float(x), 4) for x in B24[k]] for k, g in enumerate(GROUPS)}
    res["fit_2022_own_fixed"] = {g: [round(float(x), 4) for x in B22[k]] for k, g in enumerate(GROUPS)}
    res["fit_2022_free"] = {g: [round(float(x), 4) for x in B22_free[k]] for k, g in enumerate(GROUPS)}
    res["weights_2022"] = {g: round(float(w22[k]), 4) for k, g in enumerate(GROUPS)}
    res["weights_2024"] = {g: round(float(x), 4) for g, x in zip(GROUPS, group_weights(X, w))}

    # diagnostics (pas des critères) : biais des moyennes sur 2022 (en échantillon pour
    # l'analogue) et jeu de test élargi aux candidats du quota LFI de l'accord NFP que la
    # source principale range ailleurs (NPA, POI, Picardie debout…)
    ytr = observed_share(train)
    res["in_sample_2022"] = {
        name: {"bias": float((point_prediction(p.means(), Xtr) - ytr).mean()),
               "mae": float(np.abs(point_prediction(p.means(), Xtr) - ytr).mean())}
        for name, p in (("proximity", prior_a), ("analog_2022", prior_b))
    }  # fmt: skip
    cands = pl.read_parquet(paths.CURATED / "legislative_candidates.parquet")
    quota = (
        cands.filter((pl.col("election") == "leg2024") & (pl.col("round") == 1)
                     & (pl.col("party_note") == "accord NFP : lfi"))
        .select("circ_code").unique()
    )  # fmt: skip
    extra = duels.filter((pl.col("election") == "leg2024") & (pl.col("duel_type") == "rn_left_other")
                         & pl.col("circ_code").is_in(quota["circ_code"].to_list()))  # fmt: skip
    wide = pl.concat([test, extra.select(test.columns)])
    Xq = design(wide)[0]
    yq = observed_share(wide)
    res["test_lfi_quota"] = {"n_extra": extra.height,
                             "extra": extra.select("circ_code", "opp_name").rows(),
                             **{p.name: summarize_scores(per_circ_scores(predictive_draws(p, Xq), yq))
                                for p in (prior_a, prior_b)}}  # fmt: skip

    # diagnostic : biais des moyennes (prévision ponctuelle) et des enquêtes publiées
    diag = {}
    for name, means in (("proximity", prior_a.means()), ("analog_2022", prior_b.means())):
        e = point_prediction(means, X) - y
        diag[name] = {"bias": float(e.mean()), "mae": float(np.abs(e).mean()), "covered": []}
    if SURVEYS_FILE.exists():
        for src in load_surveys():
            if src.get("election") not in ("leg2024", "leg2022"):
                continue
            if src.get("duel") not in ("rn_vs_nfp", "rn_vs_nfp_lfi", "rn_vs_nupes", "rn_vs_nupes_lfi"):
                continue
            target = test if src["election"] == "leg2024" else train
            Xs, _, _ = design(target)
            ys = observed_share(target)
            means, cov = survey_means(src, prior_a)
            if not cov:
                continue
            e = point_prediction(means, Xs) - ys
            diag[src["id"]] = {"bias": float(e.mean()), "mae": float(np.abs(e).mean()),
                               "covered": cov, "timing": src.get("timing"),
                               "applied_to": f"{src['election']} rn_lfi ({len(ys)})"}  # fmt: skip
            # même enquête, électorats non couverts pris dans l'estimation écologique de la
            # même année (après coup) : isole l'effet des lignes de l'enquête
            Xy, Yy, wy = design(target)
            fill = analog_priors(fit_transfers(Xy, Yy, wy, fixed_own_rows(own)),
                                 group_weights(Xy, wy), "fill")  # fmt: skip
            m2, _ = survey_means(src, fill)
            e2 = point_prediction(m2, Xs) - ys
            base2 = point_prediction(fill.means(), Xs) - ys
            diag[src["id"]] |= {"bias_fill_same_year": float(e2.mean()),
                                "bias_same_year_fit_alone": float(base2.mean())}  # fmt: skip
    res["survey_diagnostic"] = diag

    # sensibilité 2027
    dp = paths.RUNS / "2026-11-24-simulate-subst" / "draws.parquet"
    sp = paths.RUNS / "2026-11-24-simulate-subst" / "summary.parquet"
    if dp.exists() and sp.exists():
        X27 = first_round_2027(dp, sp)
        sens = {"n_first_round_draws": len(X27),
                "mean_first_round": {g: round(float(x), 2) for g, x in zip(GROUPS, X27.mean(0))}}
        prior_24 = analog_priors(B24, group_weights(Xall, wall), "analog_2024_all")
        for p in (prior_a, prior_b, prior_b_free, prior_24):
            s = prior_share_2027(p, X27)
            sens[p.name] = {"median": float(np.median(s)), "q05": float(np.quantile(s, 0.05)),
                            "q95": float(np.quantile(s, 0.95)), "p_le_pen": float((s > 50).mean()),
                            "point_at_means": float(point_prediction(p.means(), X27.mean(0, keepdims=True))[0])}  # fmt: skip
        res["sensitivity_2027"] = sens

    tt = transfers_table(duels, n_boot=n_boot)
    tt.write_parquet(out_dir / "transfers_estimates.parquet")
    tt_own = transfers_table(duels, n_boot=n_boot, fixed_own=True)
    tt_own.write_parquet(out_dir / "transfers_estimates_own_fixed.parquet")
    per_circ.write_parquet(out_dir / "per_circ.parquet")
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")  # fmt: skip
    return res


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    n_boot = int(argv[0]) if argv else 500
    res = run(n_boot=n_boot)
    print(json.dumps({k: res[k] for k in ("n_train_2022", "n_test_2024", "scores", "verdict")},
                     indent=2, ensure_ascii=False))  # fmt: skip


if __name__ == "__main__":
    main()
