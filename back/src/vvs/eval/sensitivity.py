"""Sensibilité des probabilités aux poids des scénarios, sur les tirages publiés figés."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl

from vvs.registry.scenarios import ScenariosFile, enumerate_configurations, load_scenarios


def state_key(states: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(states.items()))


def set_probability(
    scenarios: ScenariosFile, dimension: str, state: str, value: float
) -> ScenariosFile:
    if not 0 <= value <= 1:
        raise ValueError("probabilité hors de [0, 1]")
    data = copy.deepcopy(scenarios.model_dump(mode="json"))
    dim = next(d for d in data["dimensions"] if d["id"] == dimension)
    old = next(s["probability"] for s in dim["states"] if s["id"] == state)
    for s in dim["states"]:
        if s["id"] == state:
            s["probability"] = value
        elif old < 1:
            s["probability"] *= (1 - value) / (1 - old)
        else:
            s["probability"] = (1 - value) / (len(dim["states"]) - 1)
    return ScenariosFile.model_validate(data)


def variants(scenarios: ScenariosFile, settings: dict):
    yield "reference", "Poids de référence", scenarios
    for dim in scenarios.dimensions:
        for s in dim.states:
            for sign, suffix in ((-1, "minus"), (1, "plus")):
                p = np.clip(s.probability + sign * settings["probability_shift"], 0, 1)
                if p == s.probability:
                    continue
                label = f"{dim.label} : {s.label}, poids {s.probability:.0%} → {p:.0%}"
                yield (
                    f"{dim.id}.{s.id}.{suffix}",
                    label,
                    set_probability(scenarios, dim.id, s.id, float(p)),
                )
            if settings["include_conditional_states"]:
                yield (
                    f"{dim.id}.{s.id}.certain",
                    f"Condition : {dim.label}, {s.label}",
                    set_probability(scenarios, dim.id, s.id, 1.0),
                )
    if settings["include_uniform"]:
        data = scenarios.model_dump(mode="json")
        for d in data["dimensions"]:
            for s in d["states"]:
                s["probability"] = 1 / len(d["states"])
        yield "uniform", "Poids uniformes dans chaque dimension", ScenariosFile.model_validate(data)


def draw_weights(original, target, config_ids: np.ndarray) -> tuple[np.ndarray, float]:
    """p(configuration)/effectif réel ; ne dépend pas du rang du nouveau tri."""
    counts = np.bincount(config_ids, minlength=len(original))
    old_index = {state_key(c.states): i for i, c in enumerate(original)}
    weight = np.zeros(len(original))
    missing = 0.0
    for cfg in target:
        i = old_index.get(state_key(cfg.states))
        if i is None or counts[i] == 0:
            missing += cfg.probability
        else:
            weight[i] = cfg.probability / counts[i]
    w = weight[config_ids]
    if not np.isclose(w.sum() + missing, 1.0):
        raise ValueError("masse des scénarios non conservée")
    return w, float(missing)


def probability_bounds(event: np.ndarray, weights: np.ndarray, missing: float) -> dict:
    """Bornes d'identification dues au support manquant, distinctes de l'erreur Monte-Carlo."""
    lower = float(weights @ np.asarray(event, dtype=float))
    represented = float(weights.sum())
    return {
        "probability": lower / represented if represented > 0 else None,
        "lower": lower,
        "upper": min(1.0, lower + missing),
    }


def analyze(out: Path, spec: dict) -> None:
    from vvs.eval.consolidation import write_json

    scenarios = load_scenarios()
    original = enumerate_configurations(scenarios, small="marginal")
    draws = pl.read_parquet(out / "published/draws.parquet")
    manifest = json.loads((out / "published/manifest.json").read_text())
    ids = draws["config"].to_numpy()
    if len(ids) != manifest["n_draws"] or ids.max() >= len(original):
        raise ValueError("tirages incompatibles avec les scénarios")
    names = [c["candidate_id"] for c in manifest["universe"]]
    wins = draws["winner"].to_numpy()
    a, b = draws["qualified_a"].to_numpy(), draws["qualified_b"].to_numpy()
    rows, metadata = [], []
    base = {}
    for vid, label, sc in variants(scenarios, spec["sensitivity"]):
        target = enumerate_configurations(sc, small="marginal")
        weights, missing = draw_weights(original, target, ids)
        represented = float(weights.sum())
        ess = represented**2 / float(weights @ weights) if represented else 0.0
        metadata.append(
            {
                "id": vid,
                "label": label,
                "ess": ess,
                "missing_mass": missing,
                "impossible_mass": target.impossible_mass,
                "low_ess": ess < spec["sensitivity"]["minimum_ess"],
                "dimension_weights": {
                    d.id: {s.id: s.probability for s in d.states} for d in sc.dimensions
                },
            }
        )
        for c in names:
            events = {
                "win": wins == c,
                "qualify": (a == c) | (b == c),
                "present": draws[c].is_not_null().to_numpy(),
            }
            row = {
                "variant": vid,
                "label": label,
                "candidate_id": c,
                "ess": ess,
                "missing_mass": missing,
                "low_ess": ess < spec["sensitivity"]["minimum_ess"],
            }
            for metric, event in events.items():
                bound = probability_bounds(event, weights, missing)
                for key, value in bound.items():
                    suffix = "" if key == "probability" else f"_{key}"
                    row[f"p_{metric}{suffix}"] = value
            if vid == "reference":
                base[c] = row
            row["delta_win_pp"] = 100 * (row["p_win"] - base[c]["p_win"]) if represented else None
            row["delta_qualify_pp"] = (
                100 * (row["p_qualify"] - base[c]["p_qualify"]) if represented else None
            )
            rows.append(row)
    target_dir = out / "sensitivity"
    target_dir.mkdir(exist_ok=True)
    frame = pl.DataFrame(rows)
    frame.write_parquet(target_dir / "probabilities.parquet")
    frame.write_csv(target_dir / "probabilities.csv")
    write_json(target_dir / "variants.json", metadata)
    leaders = sorted(names, key=lambda c: -base[c]["p_win"])[:6]
    lines = [
        "# Sensibilité aux scénarios — 22 octobre 2026",
        "",
        "Modèle et tirages figés. Les chiffres centraux sont conditionnels à la masse",
        "représentée ; les bornes lower/upper dans le CSV incluent la masse manquante.",
        "Ce sont des variations d'hypothèses, pas des intervalles de confiance ni des effets causaux.",
        "",
        "| Variante | ESS | Masse manquante | " + " | ".join(leaders) + " |",
        "|---|---|---|" + "---|" * len(leaders),
    ]
    for meta in metadata:
        sub = frame.filter(pl.col("variant") == meta["id"])
        values = {r["candidate_id"]: r["p_win"] for r in sub.iter_rows(named=True)}
        label = meta["label"] + (" ⚠ ESS faible" if meta["low_ess"] else "")
        vals = [
            f"{100 * values[c]:.1f} %" if values[c] is not None else "non estimable"
            for c in leaders
        ]
        lines.append(
            f"| {label} | {meta['ess']:.0f} | {100 * meta['missing_mass']:.3f} % | "
            + " | ".join(vals)
            + " |"
        )
    lines += [
        "",
        "Les états certains ne sont pas des alternatives de poids plausibles : ils isolent une",
        "condition. Les petits candidats et les règles de reports restent inchangés.",
        "Le seuil de 1 000 tirages effectifs est un signal de précision, pas une garantie.",
        "",
    ]
    (target_dir / "report.md").write_text("\n".join(lines))
    print(f"Sensibilité : {len(metadata)} variantes, {len(names)} candidats", flush=True)
