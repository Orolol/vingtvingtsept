"""Second tour à premier tour fixé dans la chaîne chronologique (docs/results/2026-11-24-premier-tour-fixe.md).

Usage depuis back/ :
    uv run python studies/fixed_first_round.py \
        --reference ../runs/2026-10-22-consolidation --out ../runs/2026-11-24-premier-tour-fixe/chain

Reprend les douze cas du gel (case.json, curated/, fit/draws.npz, spécification) et rejoue
le premier tour et le second tour avec le code courant, `VVS_ROOT` pointé sur la copie figée
des registres et tables. Bras : R0 (référence, graine du gel), V0 (`fixed_first_round`, même
graine), R1 (référence, graine + 1, bruit Monte-Carlo) ; V0_k1024 (contrôle de convergence
de V0, hors critère). R0 est comparé aux tirages du gel.
Aucun fichier du gel n'est écrit. Un cas terminé (complete.json) est repris tel quel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ARMS = {  # nom : (décalage de graine, premier tour fixé, propositions par ligne)
    "R0": (0, False, None),
    "V0": (0, True, None),
    "R1": (1, False, None),
    # Contrôle de convergence ajouté après lecture des avertissements de taille effective par
    # ligne du premier calcul (pas un critère) : même variante, 1024 propositions par ligne.
    "V0_k1024": (0, True, 1024),
}
CRITERION_ARMS = ("R0", "V0", "R1")
METRICS = ("log_score_winner", "log_score_pair", "crps", "coverage90", "crps2", "coverage90_2")


def run_case(reference: Path, out: Path, name: str, spec: dict) -> None:
    import datetime as dt

    import numpy as np
    import polars as pl

    from vvs.eval import consolidation as co
    from vvs.model import second_round as sr
    from vvs.model.data import load_polls
    from vvs.model.second_round import SecondRoundModel
    from vvs.registry import default_registry

    folder, original = out / "cases" / name, reference / "cases" / name
    if (folder / "complete.json").exists():
        print(f"Reprise : {name}", flush=True)
        return
    case = json.loads((original / "case.json").read_text())
    co.check_case(original, case)
    if not co.complete_score(original):
        raise ValueError(f"cas du gel incomplet : {name}")
    folder.mkdir(parents=True, exist_ok=True)
    asof = dt.date.fromisoformat(case["asof"])
    errors, errors2, transfers, turnout, excluded, audit = co.calibration(asof, spec)
    if json.loads(json.dumps(audit, default=str)) != json.loads(
        (original / "calibration.json").read_text()
    ):
        raise ValueError(f"calibration différente du gel : {name}")
    polls = load_polls(case["election"], asof, curated=original / "curated")
    reg = default_registry()
    with np.load(original / "draws_reference.npz") as d:
        frozen = dict(d)
    rows, cells, control = [], [], {}
    with np.load(original / "fit/draws.npz") as draws:
        default_k = sr.FIXED_PROPOSALS
        for arm, (offset, fixed, proposals) in ARMS.items():
            model2 = SecondRoundModel(
                curated=original / "curated",
                transfers=transfers,
                errors=errors2,
                exclude=excluded,
                **spec["second_round"],
            )
            arm_spec = spec | {"seed": spec["seed"] + offset}
            sr.FIXED_PROPOSALS = default_k if proposals is None else proposals
            try:
                sim, detail = co.simulate_chain(
                    case, draws, errors, model2, turnout, arm_spec, polls, "reference",
                    fixed_first_round=fixed,
                )
            finally:
                sr.FIXED_PROPOSALS = default_k
            if arm == "R0":
                control = {k: bool(np.array_equal(sim[k], frozen[k])) for k in frozen}
            summary, _ = co.score_chain(sim, case["election"], reg)
            second = co.score_second_round(sim, case["election"], reg)
            key = {"election": case["election"], "horizon": case["horizon"], "arm": arm,
                   "seed": arm_spec["seed"], "fixed_first_round": fixed,
                   "proposals": (proposals or default_k) if fixed else None}
            rows.append(key | summary | second)
            cells += [key | c for c in detail]
            np.savez_compressed(folder / f"draws_{arm}.npz", **sim)
    pl.DataFrame(rows).write_csv(folder / "metrics.csv", float_precision=8)
    pl.DataFrame(cells, infer_schema_length=None).write_csv(folder / "cells.csv")
    co.write_json(folder / "complete.json", {"replay_matches_frozen_reference": control})
    print(f"Terminé : {name} ; rejeu R0 identique au gel : {all(control.values())}", flush=True)


def aggregate(out: Path, spec: dict) -> dict:
    import polars as pl

    frames = [
        pl.read_csv(out / "cases" / f"{e}-{h:03d}" / "metrics.csv")
        for e in spec["elections"]
        for h in spec["horizons"]
    ]
    m = pl.concat(frames, how="vertical_relaxed").sort("election", "horizon", "arm")
    m.write_csv(out / "metrics.csv", float_precision=8)
    wide = {arm: m.filter(pl.col("arm") == arm).sort("election", "horizon") for arm in ARMS}
    n = wide["R0"].height
    excluded = [
        f"{r['election']}-{r['horizon']}"
        for arm in ARMS
        for r in wide[arm].filter(pl.col("n_duel_draws") == 0).iter_rows(named=True)
    ]
    result: dict = {"n_cases": n, "excluded_second_round_cases": sorted(set(excluded)),
                    "overall": {}, "by_horizon": {}}

    def mean(x):
        x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(x) / len(x) if x else None

    for metric in METRICS:
        col = {arm: wide[arm][metric].to_list() for arm in ARMS}
        d_noise = [b - a for a, b in zip(col["R0"], col["R1"])]
        sigma = math.sqrt(sum(d * d for d in d_noise) / len(d_noise))
        diff = [v - r for v, r in zip(col["V0"], col["R0"])]
        result["overall"][metric] = {
            **{arm: mean(col[arm]) for arm in ARMS},
            "diff_V0_R0": mean(diff), "noise_sigma_case": sigma, "noise_se_mean": sigma / math.sqrt(n),
            "max_abs_diff_V0_R0": max(abs(d) for d in diff),
            "diff_V0_k1024_R0": mean([v - r for v, r in zip(col["V0_k1024"], col["R0"])]),
        }
        for h in spec["horizons"]:
            sel = [i for i, x in enumerate(wide["R0"]["horizon"].to_list()) if x == h]
            result["by_horizon"].setdefault(str(h), {})[metric] = {
                **{arm: mean([col[arm][i] for i in sel]) for arm in ARMS},
                "diff_V0_R0": mean([diff[i] for i in sel]),
                "noise_se_mean": sigma / math.sqrt(len(sel)),
            }
    o = result["overall"]
    lw, c2, cov = o["log_score_winner"], o["crps2"], o["coverage90_2"]
    checks = {
        "log_score_winner": lw["diff_V0_R0"] >= -2 * lw["noise_se_mean"],
        "crps2": c2["diff_V0_R0"] <= 2 * c2["noise_se_mean"],
        "coverage90_2": abs(cov["V0"] - 0.9) <= abs(cov["R0"] - 0.9) + 2 * cov["noise_se_mean"],
    }
    controls = {
        "log_score_pair_identical": o["log_score_pair"]["max_abs_diff_V0_R0"] == 0.0,
        "first_round_crps_identical": o["crps"]["max_abs_diff_V0_R0"] == 0.0,
        "replay_R0_equals_frozen": all(
            all(json.loads((out / "cases" / f"{e}-{h:03d}" / "complete.json").read_text())[
                "replay_matches_frozen_reference"].values())
            for e in spec["elections"] for h in spec["horizons"]
        ),
    }
    result |= {"checks": checks, "controls": controls, "admitted": all(checks.values())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference, out = args.reference.resolve(), args.out.resolve()
    if out == reference or out.is_relative_to(reference):
        raise ValueError("sorties et référence doivent être séparées")
    os.environ["VVS_ROOT"] = str(reference / "source")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from vvs.eval import consolidation as co
    from vvs.model.simulate import _versions

    record = co.verify(reference)
    if _versions() != record["versions"]:
        raise ValueError("versions de dépendances différentes du gel")
    spec = record["spec"]
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    sources = ["back/src/vvs/model/second_round.py", "back/src/vvs/model/simulate.py",
               "back/src/vvs/eval/consolidation.py", "back/studies/fixed_first_round.py"]
    manifest = {
        "reference": record["id"],
        "reference_commit": record["commit"],
        "code_head": head,
        "code_dirty": subprocess.check_output(
            ["git", "status", "--porcelain", "--", *sources], cwd=repo, text=True
        ).splitlines(),
        "sources_sha256": {s: co.digest(repo / s) for s in sources},
        "versions": record["versions"],
        "arms": {k: {"seed": spec["seed"] + v[0], "fixed_first_round": v[1], "proposals": v[2]}
                 for k, v in ARMS.items()},
        "criterion_arms": list(CRITERION_ARMS),
        "n_draws": spec["n_draws"],
        "command": " ".join(sys.argv),
    }
    co.write_json(out / "manifest.json", manifest)
    for e in spec["elections"]:
        for h in spec["horizons"]:
            run_case(reference, out, f"{e}-{h:03d}", spec)
    result = aggregate(out, spec)
    co.write_json(out / "results.json", result)
    print(json.dumps({k: result[k] for k in ("checks", "controls", "admitted")}, indent=2))


if __name__ == "__main__":
    main()
