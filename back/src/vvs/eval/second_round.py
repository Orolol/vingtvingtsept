"""Backtest du modèle de second tour (docs/07-evaluation.md) : leave-one-election-out.

Pour chaque élection de 2002 à 2022 et le duel réellement joué (orienté sur le vainqueur) :

- horizons : la veille du second tour (`r2_eve`, parts de premier tour = résultats officiels),
  30 et 90 jours avant le premier tour (`r1_30`, `r1_90`, parts de premier tour = moyenne
  simple des sondages des 14 jours précédents, bruitée par `noisy_first_round`), et 180 jours
  (`r1_180`, l'horizon de la prévision 2027 d'octobre 2026, à titre indicatif) ;
- méthodes : `model` (reports + sondages de duel), `model_no_bias` (sans correction du biais
  d'extrême droite), `transfers_only` (reports historiques sans sondages de duel, prior seul),
  `polls_14d` (moyenne des sondages de duel des 14 derniers jours, loi normale d'écart-type
  l'erreur d'horizon calibrée, sans biais) ;
- tout ce qui est calibré sur l'histoire (analogues de reports, erreur des sondages de duel,
  bruit du premier tour) exclut l'élection évaluée.

Métriques : erreur absolue de la médiane, CRPS et couverture 50 et 90 % de `vvs.eval.metrics`
(module partagé avec le backtest du premier tour) ; log-score du vainqueur ici.

`run_structural` : comparaison appariée des options structurelles du second tour (biais RN par
horizon, couplage avec l'erreur d'extrême droite du premier tour, participation du second tour
tirée à part), même graine par cas pour toutes les variantes ; voir son docstring.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from vvs.eval import metrics as mt
from vvs.model.second_round import (
    ABSTAIN,
    ErrorParams,
    SecondRoundModel,
    _analog_vector,
    _group_members,
    calibrate_errors,
    far_right_coupling,
    first_round_concentration,
    first_round_mean,
    noisy_first_round,
    observed_far_right_error,
    summarize,
)
from vvs.registry import BLOCS as BLOCS_ORDER
from vvs.registry import Registry, default_registry
from vvs.registry.transfers import default_transfers

log = logging.getLogger(__name__)

ELECTIONS = (2002, 2007, 2012, 2017, 2022)
# libellé -> jours avant le premier tour (None : veille du second tour)
HORIZONS: dict[str, int | None] = {"r2_eve": None, "r1_30": 30, "r1_90": 90, "r1_180": 180}
METHODS = ("model", "model_no_bias", "transfers_only", "polls_14d")
# méthode ajoutée par `run(..., with_potential=True)` (plafonds du potentiel électoral)
POTENTIAL_METHOD = "model_potential"
PAIRED_METHOD = "model_paired"  # modèle sans option, mêmes tirages que model_potential
RECENT_DAYS = 14


# --- métriques (CRPS et couverture : vvs.eval.metrics, partagé avec le premier tour)


def winner_log_score(draws: np.ndarray) -> tuple[float, float]:
    """(P(vainqueur gagne), log-score) ; fréquence lissée (k + ½) / (N + 1) (Jeffreys), comme
    `metrics.pair_log_score`, pour éviter −∞ quand aucun tirage ne donne le vainqueur."""
    k = int(np.sum(np.asarray(draws) > 50.0))
    n = len(draws)
    return k / n, float(np.log((k + 0.5) / (n + 1)))


def score(draws: np.ndarray, y: float) -> dict:
    """Métriques d'une loi prédictive (tirages de la part du vainqueur) contre le résultat."""
    q = np.quantile(draws, [0.05, 0.25, 0.5, 0.75, 0.95])
    p, ls = winner_log_score(draws)
    return {
        "median": float(q[2]),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "abs_error": float(mt.abs_error_median(draws, y)),
        "crps": float(mt.crps(draws, y)),
        "cover50": bool(mt.covered(draws, y, 0.5)),
        "cover90": bool(mt.covered(draws, y, 0.9)),
        "p_winner": p,
        "log_score": ls,
    }


# --- entrées


@dataclass
class Case:
    election: int
    horizon_label: str
    horizon: int  # jours avant le premier tour (négatif après)
    asof: dt.date
    pair: tuple[str, str]  # (vainqueur, perdant)
    result: float  # part du vainqueur
    first_round: dict[str, np.ndarray]
    first_round_note: str
    # composante d'extrême droite réduite du premier tour par tirage (case_eta), ou observée
    # (veille du second tour) ; None : pas de couplage possible
    shared: np.ndarray | float | None = None


def case(
    election: int,
    label: str,
    rng: np.random.Generator,
    registry: Registry,
    n_first_round: int = 2000,
) -> Case:
    e = registry.elections[election]
    d1, d2 = e.round_(1).date, e.round_(2).date
    res = sorted(((r.share, r.candidate_id) for r in e.round_(2).results), reverse=True)
    pair = (res[0][1], res[1][1])
    days = HORIZONS[label]
    if days is None:
        asof = d2 - dt.timedelta(days=1)
        fr = {r.candidate_id: np.array([r.share / 100]) for r in e.round_(1).results}
        note = "résultats officiels du premier tour"
    else:
        asof = d1 - dt.timedelta(days=days)
        mean, (start, end), n_hyp = first_round_mean(election, asof, pair, RECENT_DAYS)
        kappa = first_round_concentration(days, exclude={election})
        fr = noisy_first_round(mean, kappa, rng, n_first_round)
        note = f"{n_hyp} hypothèses du {start + dt.timedelta(days=1)} au {end}, Dirichlet κ={kappa:.0f}"
    return Case(election, label, (d1 - asof).days, asof, pair, res[0][0], fr, note)


def polls_14d(model: SecondRoundModel, c: Case, errors: ErrorParams, rng, n: int):
    """Baseline : moyenne simple des sondages du duel des 14 derniers jours, N(moyenne, σ(h)²)."""
    polls = model.pair_fit(c.pair).polls
    if polls.n == 0:
        return None, 0
    keep = np.array([(c.asof - d).days < RECENT_DAYS for d in polls.fieldwork_end])
    if not keep.any():
        return None, 0
    gap = model.pair_fit(c.pair).gap_days
    age = c.horizon + gap + RECENT_DAYS / 2
    mean = float(polls.share[keep].mean())
    return rng.normal(mean, float(errors.sd(age)), size=n), int(keep.sum())


# --- backtest


def run(
    elections: tuple[int, ...] = ELECTIONS,
    horizons: tuple[str, ...] = tuple(HORIZONS),
    n: int = 8000,
    seed: int = 0,
    registry: Registry | None = None,
    with_potential: bool = False,
) -> dict[str, pl.DataFrame]:
    """`with_potential` ajoute la méthode `model_potential` (modèle + plafonds du potentiel
    électoral, `SecondRoundModel(use_potential=True)`), désactivée par défaut.

    Rend `metrics` (une ligne par élection × horizon × méthode), `errors` (paramètres
    d'erreur par pli), `transfers` (reports a priori, a posteriori et observés à la veille du
    second tour) et `summaries` (résumé du modèle par cas)."""
    registry = registry or default_registry()
    rng = np.random.default_rng(seed)
    rows, err_rows, tr_rows, sum_rows = [], [], [], []
    potential_table = None
    if with_potential:
        from vvs.model import potential as pot

        # normalisation sans résultat électoral (couples de mesures seulement) : pas de fuite
        raw = pot.load().filter(pl.col("scale") != "attitude")
        potential_table = pot.normalize(raw)
    for election in elections:
        errors = calibrate_errors(exclude={election}, registry=registry)
        err_rows.append(
            {
                "election": election,
                "sigma0": errors.sigma0,
                "sigma1": errors.sigma1,
                "bias_far_right": errors.bias_far_right,
                "sigma_house": errors.sigma_house,
                "calibrated_on": ",".join(map(str, errors.calibrated_on)),
            }
        )
        for label in horizons:
            c = case(election, label, rng, registry)
            variants = {
                "model": SecondRoundModel(registry, errors=errors),
                "model_no_bias": SecondRoundModel(registry, errors=errors, use_bias=False),
                "transfers_only": SecondRoundModel(registry, errors=errors, use_polls=False),
            }
            if with_potential:
                variants[POTENTIAL_METHOD] = SecondRoundModel(
                    registry, errors=errors, use_potential=True, potential=potential_table
                )
                # comparaison appariée : le modèle sans option, mêmes tirages du prior que
                # `model_potential` (même graine), seuls les poids diffèrent
                variants[PAIRED_METHOD] = SecondRoundModel(registry, errors=errors)
            case_seed = [seed, election, list(HORIZONS).index(label)]
            base = {
                "election": election,
                "horizon": label,
                "horizon_days": c.horizon,
                "asof": c.asof,
                "winner": c.pair[0],
                "loser": c.pair[1],
                "result": c.result,
                "first_round": c.first_round_note,
            }
            for name, m in variants.items():
                m.fit(election, c.asof, [c.pair])
                r = np.random.default_rng(case_seed) if name in (POTENTIAL_METHOD, PAIRED_METHOD) else rng
                d = m.sample(c.pair, c.first_round, c.horizon, r, n)
                s = summarize(m.pair_fit(c.pair), d)
                rows.append(
                    base
                    | {"method": name, "n_polls": s["n_polls"], "inedit": s["inedit"], "ess": d.ess}
                    | score(d.share, c.result)
                )
                if name == POTENTIAL_METHOD:
                    sum_rows.append(
                        base | {"method": name}
                        | {k: v for k, v in s.items() if k not in ("transfers", "horizon")}
                    )
                if name == "model":
                    sum_rows.append(
                        base | {k: v for k, v in s.items() if k not in ("transfers", "horizon")}
                    )
                    if label == "r2_eve":
                        tr_rows += transfer_rows(election, c, s, registry)
            draws, k = polls_14d(variants["model"], c, errors, rng, n)
            row = base | {"method": "polls_14d", "n_polls": k, "inedit": None, "ess": None}
            if draws is not None:
                row |= score(draws, c.result)
            rows.append(row)
            log.info("%s %s fait", election, label)
    return {
        "metrics": pl.DataFrame(rows, infer_schema_length=None),
        "errors": pl.DataFrame(err_rows),
        "transfers": pl.DataFrame(tr_rows, infer_schema_length=None),
        "summaries": pl.DataFrame(sum_rows, infer_schema_length=None),
    }


def observed_transfers(
    election: int, group: str, pair: tuple[str, str], registry: Registry
) -> tuple[float | None, float | None]:
    """(part vers pair[0] parmi les exprimés, part ni-l'un-ni-l'autre) observées dans les
    enquêtes de l'élection elle-même, électorats regroupés comme dans le modèle."""
    transfers = default_transfers()
    if group.startswith("own:"):
        members = [(group[4:], 1.0)]
    elif group == ABSTAIN:
        members = _group_members(registry, election, ABSTAIN, set())
    else:
        members = _group_members(registry, election, group[5:], set(pair))
    split, none, _ = _analog_vector(election, members, pair, transfers, registry)
    return split, none


def transfer_rows(election: int, c: Case, s: dict, registry: Registry) -> list[dict]:
    out = []
    for t in s["transfers"]:
        split_obs, none_obs = observed_transfers(election, t["group"], c.pair, registry)
        prior_split = t["prior_to_i"] / max(t["prior_to_i"] + t["prior_to_j"], 1e-9)
        out.append(
            {
                "election": election,
                "winner": c.pair[0],
                "loser": c.pair[1],
                "group": t["group"],
                "weight": t["weight"],
                "source": t["source"],
                "analogs": t["analogs"],
                "prior_split": prior_split,
                "prior_none": t["prior_none"],
                "post_to_i": t["to_i_median"],
                "post_to_i_q05": t["to_i_q05"],
                "post_to_i_q95": t["to_i_q95"],
                "post_none": t["none_median"],
                "post_none_q05": t["none_q05"],
                "post_none_q95": t["none_q95"],
                "post_split": t["split_median"],
                "post_split_q05": t["split_q05"],
                "post_split_q95": t["split_q95"],
                "obs_split": split_obs,
                "obs_none": none_obs,
            }
        )
    return out


def aggregate(metrics: pl.DataFrame, common: bool = False) -> pl.DataFrame:
    """Moyennes par horizon et méthode (élections à poids égal, cas manquants exclus).

    `common` : seulement les (élection, horizon) où toutes les méthodes ont une prévision
    (la baseline `polls_14d` manque quand aucun sondage du duel n'a moins de 14 jours).
    """
    m = metrics.filter(pl.col("median").is_not_null())
    if common:
        n_methods = metrics["method"].n_unique()
        full = (
            m.group_by("election", "horizon").len().filter(pl.col("len") == n_methods)
            .select("election", "horizon")
        )
        m = m.join(full, on=["election", "horizon"])
    return (
        m
        .group_by("horizon", "method")
        .agg(
            pl.len().alias("n"),
            pl.col("abs_error").mean().alias("mae"),
            pl.col("crps").mean(),
            pl.col("cover50").cast(pl.Float64).mean(),
            pl.col("cover90").cast(pl.Float64).mean(),
            pl.col("log_score").mean(),
        )
        .sort("horizon", "method")
    )


def proximity_check(
    elections: tuple[int, ...] = ELECTIONS, registry: Registry | None = None
) -> pl.DataFrame:
    """Matrice de proximité contre reports observés, par élection et groupe d'électorat
    (recoupement de la matrice ; ne sert à aucun calibrage)."""
    from vvs.model.second_round import proximity_mean

    registry = registry or default_registry()
    rows = []
    for e in elections:
        el = registry.elections[e]
        pair = tuple(r.candidate_id for r in el.round_(2).results)
        blocs = tuple(registry.bloc_of(c, e) for c in pair)
        for group in [*(f"bloc:{b}" for b in BLOCS_ORDER), ABSTAIN]:
            split, none = observed_transfers(e, group, pair, registry)
            if split is None:
                continue
            electorate = ABSTAIN if group == ABSTAIN else group[5:]
            m = proximity_mean(electorate, *blocs)
            rows.append(
                {
                    "election": e,
                    "candidate_i": pair[0],
                    "candidate_j": pair[1],
                    "group": group,
                    "matrix_split": m[0] / (m[0] + m[1]),
                    "obs_split": split,
                    "matrix_none": m[2],
                    "obs_none": none,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


# --- options structurelles du second tour (comparaison appariée)

# variante -> (premier tour : "dirichlet" comme `run` ou "eta" pour le modèle d'erreur η, options) ;
# option `settings` : réglages du prior des reports (`SecondRoundSettings`, σ_u, σ_v, proximité)
STRUCT_VARIANTS: dict[str, tuple[str, dict]] = {
    "base": ("dirichlet", {}),
    # atténuation linéaire du biais constant (défaut de la simulation avant la courbe)
    "bias_linear": ("dirichlet", {"bias_linear": True}),
    "bias_h": ("dirichlet", {"bias_by_horizon": True}),
    "turnout2": ("dirichlet", {"second_round_turnout": True}),
    "bias_h_turnout2": ("dirichlet", {"bias_by_horizon": True, "second_round_turnout": True}),
    "eta": ("eta", {}),
    "eta_coupled": ("eta", {"couple_far_right": True}),
    "eta_all": ("eta", {"bias_by_horizon": True, "couple_far_right": True,
                        "second_round_turnout": True}),
}


def case_eta(
    election: int,
    label: str,
    rng: np.random.Generator,
    registry: Registry,
    error_model,
    sd_x: float,
    n_first_round: int = 2000,
    curated=None,
) -> Case:
    """Comme `case`, mais le premier tour avant le vote est bruité par le modèle d'erreur η
    (`ErrorModel.sample_eta_parts`, calibré hors élection, horizon = âge de la moyenne des
    14 jours) au lieu de la Dirichlet, ce qui donne la composante d'extrême droite réduite de
    chaque tirage (`Case.shared`). À la veille du second tour, premier tour officiel et
    composante observée : z = −x / `sd_x`, x = log(sondages des 14 derniers jours / résultat)
    du bloc d'extrême droite (`observed_far_right_error`)."""
    from vvs.model.simulate import perturbed_shares

    e = registry.elections[election]
    d1, d2 = e.round_(1).date, e.round_(2).date
    res = sorted(((r.share, r.candidate_id) for r in e.round_(2).results), reverse=True)
    pair = (res[0][1], res[1][1])
    days = HORIZONS[label]
    if days is None:
        asof = d2 - dt.timedelta(days=1)
        fr = {r.candidate_id: np.array([r.share / 100]) for r in e.round_(1).results}
        x = observed_far_right_error(election, curated)
        shared = None if x is None else -x / sd_x
        note = "résultats officiels du premier tour" + (
            "" if shared is None else f", erreur d'extrême droite observée z = {shared:+.2f}"
        )
    else:
        asof = d1 - dt.timedelta(days=days)
        mean, (start, end), n_hyp = first_round_mean(election, asof, pair, RECENT_DAYS)
        cands = list(mean)
        blocs = [registry.bloc_of(c, election) for c in cands]
        ref = np.tile(np.array([mean[c] for c in cands]), (n_first_round, 1))
        parts = error_model.sample_eta_parts(days + RECENT_DAYS / 2, blocs, ref, rng)
        sh = perturbed_shares(ref, parts.eta, np.ones_like(ref, dtype=bool))
        fr = {c: sh[:, k] for k, c in enumerate(cands)}
        z = parts.bloc("far_right")
        shared = z if z is not None else None
        note = f"{n_hyp} hypothèses du {start + dt.timedelta(days=1)} au {end}, η"
    return Case(election, label, (d1 - asof).days, asof, pair, res[0][0], fr, note, shared)


def run_structural(
    elections: tuple[int, ...] = ELECTIONS,
    horizons: tuple[str, ...] = tuple(HORIZONS),
    n: int = 8000,
    seed: int = 0,
    registry: Registry | None = None,
    variants: tuple[str, ...] | None = None,
    specs: dict[str, tuple[str, dict]] | None = None,
) -> dict[str, pl.DataFrame]:
    """Backtest apparié des options structurelles (`STRUCT_VARIANTS`, ou `specs` de même forme,
    dont l'option `settings` : réglages du prior des reports passés à `SecondRoundModel`).

    Pour chaque élection et horizon, toutes les variantes d'une même famille de premier tour
    partagent les mêmes tirages de premier tour (`case` ou `case_eta`, graine du cas) et la
    même graine d'échantillonnage du second tour ; seules les options diffèrent. Tout ce qui
    est calibré sur l'histoire exclut l'élection évaluée : erreurs des sondages de duel
    (constantes ou par horizon), ρ du couplage, loi de participation du second tour, modèle
    d'erreur η (biais de bloc désactivé, comme la simulation).

    Rend `metrics` (élection × horizon × variante, plus la participation du second tour
    simulée et observée), `folds` (paramètres par pli : biais constant, courbe, ρ, loi de
    participation)."""
    from vvs.model.error import ErrorModel
    from vvs.model.second_round import turnout2_model

    registry = registry or default_registry()
    specs = STRUCT_VARIANTS if specs is None else specs
    variants = tuple(specs) if variants is None else variants
    families = {specs[v][0] for v in variants}
    rows, fold_rows = [], []
    for election in elections:
        errors = calibrate_errors(exclude={election}, registry=registry)
        errors_h = calibrate_errors(exclude={election}, registry=registry, bias_by_horizon=True)
        coupling = far_right_coupling({election}, registry=registry, errors=errors)
        t2 = turnout2_model(registry, exclude={election})
        em = (
            ErrorModel.fit(elections=ELECTIONS, exclude=[election], registry=registry,
                           use_bias=False)
            if "eta" in families else None
        )
        fold_rows.append(
            {
                "election": election,
                "sigma0": errors.sigma0,
                "sigma1": errors.sigma1,
                "bias_far_right": errors.bias_far_right,
                "sigma0_h": errors_h.sigma0,
                "sigma1_h": errors_h.sigma1,
                **{f"bias_age_{int(a)}": b for a, b in errors_h.bias_curve},
                "rho": coupling["rho"],
                "rho_raw": coupling["rho_raw"],
                "rho_elections": ",".join(map(str, coupling["elections"])),
                "sd_x": coupling["sd_x"],
                "turnout2_mean": float(1 / (1 + np.exp(-t2["logit_mean"]))),
                "turnout2_logit_sd": t2["logit_sd"],
            }
        )
        r2 = registry.elections[election].round_(2)
        obs_turnout = r2.expressed / r2.registered
        for h_idx, label in enumerate(horizons):
            hseed = list(HORIZONS).index(label)
            cases = {}
            if "dirichlet" in families:
                cases["dirichlet"] = case(
                    election, label, np.random.default_rng([seed, election, hseed, 1]), registry
                )
            if "eta" in families:
                cases["eta"] = case_eta(
                    election, label, np.random.default_rng([seed, election, hseed, 2]), registry,
                    em, coupling["sd_x"],
                )
            for name in variants:
                family, opts = specs[name]
                c = cases[family]
                errs = errors_h if opts.get("bias_by_horizon") else errors
                if opts.get("bias_linear"):
                    from vvs.model.simulate import rn_bias_scale

                    errs = errors.scaled_bias(rn_bias_scale(c.horizon))
                m = SecondRoundModel(
                    registry,
                    errors=errs,
                    couple_far_right=opts.get("couple_far_right", False),
                    coupling_rho=coupling["rho"],
                    second_round_turnout=opts.get("second_round_turnout", False),
                    settings=opts.get("settings"),
                )
                m.fit(election, c.asof, [c.pair])
                shared = c.shared if opts.get("couple_far_right") else None
                d = m.sample(c.pair, c.first_round, c.horizon,
                             np.random.default_rng([seed, election, hseed, 3]), n,
                             shared_error=shared)
                s = summarize(m.pair_fit(c.pair), d)
                rows.append(
                    {
                        "election": election,
                        "horizon": label,
                        "horizon_days": c.horizon,
                        "asof": c.asof,
                        "winner": c.pair[0],
                        "loser": c.pair[1],
                        "result": c.result,
                        "first_round": c.first_round_note,
                        "method": name,
                        "family": family,
                        "n_polls": s["n_polls"],
                        "poll_bias": s["poll_bias"],
                        "coupling_kappa": s["coupling_kappa"],
                        "inedit": s["inedit"],
                        "ess": d.ess,
                        "turnout2_median": float(np.median(d.turnout)),
                        "turnout2_q05": float(np.quantile(d.turnout, 0.05)),
                        "turnout2_q95": float(np.quantile(d.turnout, 0.95)),
                        "turnout2_obs": obs_turnout,
                    }
                    | score(d.share, c.result)
                )
            log.info("%s %s fait", election, label)
    return {"metrics": pl.DataFrame(rows, infer_schema_length=None),
            "folds": pl.DataFrame(fold_rows, infer_schema_length=None)}
