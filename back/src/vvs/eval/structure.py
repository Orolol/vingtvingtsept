"""Expériences isolées de docs/13 ; aucun défaut du modèle publié n'est modifié."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, replace

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


def mix_trajectories(first, second):
    """Mélange empirique exact 1/2 ; conserve les dépendances de chaque trajectoire."""
    if set(first) != set(second) or not np.array_equal(first["candidates"], second["candidates"]):
        raise ValueError("univers différents")
    if len(first["winner"]) != len(second["winner"]):
        raise ValueError("le mélange 1/2 exige des effectifs identiques")
    return {
        k: v.copy() if k == "candidates" else np.concatenate([v, second[k]])
        for k, v in first.items()
    }


def fixed_count_scores(summary, n):
    """Même pseudo-effectif de Jeffreys pour une distribution empirique mélangée."""
    result = dict(summary)
    k = result["n_scoring_candidates"]
    result["log_score_winner"] = math.log((n * result["p_winner"] + .5) / (n + .5 * k))
    result["log_score_pair"] = math.log((n * result["p_pair"] + .5) / (n + .5 * k * (k-1)/2))
    result["score_effective_draws"] = n
    return result


def bootstrap_errors(base, audit, spec, protocol):
    """Pondérations de scrutins, pas de candidats ou d'horizons indépendants."""
    from vvs.model import error as er
    from vvs.model.error_lists import list_items
    from vvs.registry import BLOCS

    items = er.calibration_items(audit["presidentials"])
    keys = tuple(audit["list_elections"])
    if keys:
        items += list_items(weight=spec["error"]["list_weight"], keys=keys)
    groups = sorted({it.election for it in items}, key=str)
    group_index = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(protocol["seed"])
    weights = rng.dirichlet(np.ones(len(groups)), protocol["bootstrap_replicates"]) * len(groups)
    params, diagnostics = [], []
    for replicate, w in enumerate(weights):
        weighted = [replace(it, weight=it.weight*w[group_index[it.election]]) for it in items]
        x0 = np.r_[np.log([.1, .1, .001, .05, .05]), np.zeros(len(BLOCS))]
        result = minimize(
            er._neg_objective, x0,
            args=(weighted, base.params.gamma, base.params.nu, base.params.bias),
            method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise ValueError(f"bootstrap {replicate}: {result.message}")
        sb, sc, rho, ab, ac, bw = er._unpack(result.x)
        p = replace(base.params, sigma_bloc_final=math.sqrt(sb), sigma_cand_final=math.sqrt(sc),
                    rho=rho, a_bloc=ab, a_cand=ac,
                    bloc_weight=dict(zip(BLOCS, map(float, bw))))
        params.append(asdict(p))
        diagnostics.append({"replicate": replicate, "objective": float(result.fun),
                            "iterations": int(result.nit),
                            "weights": {str(g): float(w[i]) for i, g in enumerate(groups)}})
    return params, diagnostics


class ErrorMixture:
    """Un modèle d'erreur par ligne simulée, partagé entre tous ses candidats."""

    def __init__(self, parameters):
        from vvs.model.error import ErrorModel, ErrorParams

        if not parameters:
            raise ValueError("mélange de calibration vide")
        self.models = [ErrorModel(ErrorParams(**p)) for p in parameters]

    def sample_eta(self, horizon, blocs, p, rng):
        choice = rng.integers(len(self.models), size=len(p))
        out = np.empty_like(p)
        for index, model in enumerate(self.models):
            rows = np.flatnonzero(choice == index)
            if len(rows):
                out[rows] = model.sample_eta(horizon, blocs, p[rows], rng)
        return out


def overlap_covariance(records, design=.6, n_max=2500., overlap=1.):
    """Gram des jours communs : PSD par construction, plancher indépendant."""
    if not 0 <= overlap <= 1 or design <= 0 or n_max <= 0:
        raise ValueError("paramètres de covariance invalides")
    n = len(records)
    if not n:
        raise ValueError("aucun sondage")
    sampling = np.array([1/(design*float(r["sample_size"] or 1000)) for r in records])
    if not np.isfinite(sampling).all() or (sampling <= 0).any():
        raise ValueError("effectif invalide")
    cov = np.diag(sampling + 1/n_max)
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            if not (a["is_rolling"] and b["is_rolling"] and a["pollster"] == b["pollster"]):
                continue
            for r in (a, b):
                if r["fieldwork_start"] is None or r["fieldwork_end"] < r["fieldwork_start"]:
                    raise ValueError("terrain rolling absent ou inversé")
            days = max(0, (min(a["fieldwork_end"], b["fieldwork_end"])
                           - max(a["fieldwork_start"], b["fieldwork_start"])).days + 1)
            length = math.sqrt(((a["fieldwork_end"]-a["fieldwork_start"]).days+1)
                               *((b["fieldwork_end"]-b["fieldwork_start"]).days+1))
            cov[i, j] = cov[j, i] = overlap * days/length * math.sqrt(sampling[i]*sampling[j])
    return cov


def rolling_summary(polls, asof, protocol, overlap):
    """Statistique de niveau injectée dans mean14d ; pas de nouvelle loi d'erreur."""
    recent = polls.filter(pl.col("fieldwork_end") > asof-dt.timedelta(days=14))
    if recent.is_empty():
        recent = polls.filter(pl.col("fieldwork_end") == polls["fieldwork_end"].max())
    columns = ["poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size", "is_rolling"]
    means = recent.group_by(columns + ["candidate_id"]).agg(pl.col("share").mean())
    synthetic, diagnostics = [], []
    for (candidate,), data in means.group_by("candidate_id", maintain_order=True):
        data = data.sort("poll_id")
        records = data.to_dicts()
        cov = overlap_covariance(records, protocol["rolling_design"],
                                 protocol["rolling_n_max"], overlap)
        weights = np.linalg.solve(cov, np.ones(len(records)))
        weights /= weights.sum()
        share = float(weights @ data["share"].to_numpy())
        if not np.isfinite(share) or not 0 <= share <= 100:
            raise ValueError(f"moyenne GLS hors bornes : {candidate}, {share}")
        synthetic.append({"poll_id": "gls_summary", "candidate_id": candidate,
                          "fieldwork_end": asof, "share": share})
        diagnostics.append({"candidate": candidate, "polls": len(records),
                            "rolling_polls": int(data["is_rolling"].sum()),
                            "overlap_pairs": int(np.count_nonzero(np.triu(cov, 1))),
                            "min_weight": float(weights.min()),
                            "variance_factor": float(weights @ cov @ weights)})
    # Fallback inchangé pour les candidats non observés récemment : médiane de leur passé.
    past = polls.filter(~pl.col("candidate_id").is_in([r["candidate_id"] for r in synthetic]))
    past = past.select("poll_id", "candidate_id", "fieldwork_end", "share")
    return pl.concat([pl.DataFrame(synthetic), past], how="vertical_relaxed"), diagnostics


def unthinned_polls(election, asof=None, curated=None):
    """Chargeur du gel sans éclaircissement ; tous les autres filtres sont conservés."""
    from vvs.model import data

    original = data._thin_rolling
    try:
        data._thin_rolling = lambda frame: frame
        return data.load_polls(election, asof, curated)
    finally:
        data._thin_rolling = original


def nested_prob(utility, blocs, lam):
    blocs = np.asarray(blocs)
    group = list(dict.fromkeys(blocs))
    conditional, inclusive = [], []
    for b in group:
        mask = blocs == b
        scale = max(lam.get(b, 1.), .02)
        values = utility[mask]/scale
        total = logsumexp(values)
        conditional.append((mask, np.exp(values-total)))
        inclusive.append(scale*total)
    mass = np.exp(inclusive-logsumexp(inclusive))
    out = np.zeros(len(utility))
    for p, (mask, q) in zip(mass, conditional):
        out[mask] = p*q
    return out


def mixed_prediction(pair, lam, tau, axis):
    """Inverse les marges larges, puis retire les candidats ; jamais de y_red en entrée."""
    p, blocs, keep = pair["p_full"], np.array(pair["blocs"]), pair["present"]
    utility = np.empty(len(p))
    scales = np.array([max(lam.get(b, 1.), .02) for b in blocs])
    for b in set(blocs):
        mask = blocs == b
        utility[mask] = math.log(p[mask].sum()) + scales[mask]*np.log(p[mask]/p[mask].sum())
    location = np.array([axis[b] for b in blocs])

    def probabilities(u, bs, loc):
        return .5*(nested_prob(u + tau*loc, bs, lam) + nested_prob(u-tau*loc, bs, lam))

    if tau:
        for _ in range(1000):
            predicted = probabilities(utility, blocs, location)
            if np.max(np.abs(predicted-p)) < 1e-9:
                break
            utility += scales*(np.log(p)-np.log(predicted))
            utility -= utility.mean()
        else:
            raise ValueError("inversion mixed logit non convergée")
    out = np.zeros(len(p))
    out[keep] = probabilities(utility[keep], blocs[keep], location[keep])
    return out


def fit_mixture(pairs, lam, protocol):
    if not pairs:
        return 0., {"training_polls": 0, "no_training": True}
    weight = sum(p["weight"] for p in pairs)

    def objective(tau):
        total = 0.
        for pair in pairs:
            q = mixed_prediction(pair, lam, tau, protocol["bloc_axis"])
            keep = pair["present"]
            total -= pair["weight"]*float(pair["y_red"][keep] @ np.log(q[keep]))
        return total/weight + protocol["substitution_tau_penalty"]*tau*tau

    result = minimize_scalar(objective, bounds=(0, protocol["substitution_tau_max"]),
                             method="bounded", options={"xatol": 1e-4})
    if not result.success:
        raise ValueError(f"ajustement mixed logit : {result.message}")
    tau = float(result.x) if result.fun < objective(0) else 0.
    return tau, {"training_polls": len({(p["election"], p["poll_id"]) for p in pairs}),
                  "no_training": False, "objective": objective(tau)}


def panel_transitions(initial, later, states, *, linked_id_verified=False, min_linked=100,
                      prior_strength=10):
    """Transitions d'intentions, poids initiaux ; aucune reconstruction de vote passé."""
    if not linked_id_verified:
        raise ValueError("identifiant longitudinal non validé")
    if not states or len(set(states)) != len(states) or prior_strength <= 0:
        raise ValueError("états ou régularisation invalides")
    for frame in (initial, later):
        if frame.is_empty() or any(frame[c].null_count() for c in
                                  ("fieldwork_start", "fieldwork_end", "vote_round1_kind")):
            raise ValueError("vague vide ou métadonnées manquantes")
        if frame["respondent_id"].null_count() or frame["respondent_id"].n_unique() != len(frame):
            raise ValueError("identifiants absents ou dupliqués")
        if set(frame["vote_round1_kind"].drop_nulls()) != {"intention"}:
            raise ValueError("intentions requises, souvenirs de vote interdits")
        if frame["series"].n_unique() != 1 or frame["election"].n_unique() != 1:
            raise ValueError("vague ambiguë")
    if initial["series"][0] != later["series"][0] or initial["election"][0] != later["election"][0]:
        raise ValueError("séries ou élections différentes")
    if initial["fieldwork_end"].max() >= later["fieldwork_start"].min():
        raise ValueError("vagues non ordonnées")
    joined = initial.select("respondent_id", "weight", pl.col("vote_round1").alias("origin")).join(
        later.select("respondent_id", pl.col("vote_round1").alias("target")), on="respondent_id")
    joined = joined.filter(pl.col("origin").is_in(states) & pl.col("target").is_in(states))
    if len(joined) < min_linked:
        raise ValueError(f"appariements insuffisants : {len(joined)} < {min_linked}")
    weights = joined["weight"].to_numpy()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("poids invalides")
    counts = np.zeros((len(states), len(states)))
    index = {s: i for i, s in enumerate(states)}
    target = np.zeros(len(states))
    for row in joined.iter_rows(named=True):
        target[index[row["target"]]] += row["weight"]
    target /= target.sum()
    rows = []
    for state, i in index.items():
        group = joined.filter(pl.col("origin") == state)
        w = group["weight"].to_numpy()
        ess = float(w.sum()**2/(w@w)) if len(w) else 0.
        if len(w):
            for row in group.iter_rows(named=True):
                counts[i, index[row["target"]]] += row["weight"]*ess/w.sum()
        counts[i] = (counts[i] + prior_strength*target)/(ess+prior_strength)
        rows.append({"origin": state, "n": len(w), "ess": ess})
    return counts, {"linked": len(joined), "initial": len(initial),
                    "retention": len(joined)/len(initial), "rows": rows}
