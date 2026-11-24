"""Reports publiés pour l'élection en cours dans la chaîne chronologique et la projection 2027
(docs/results/2026-11-24-reports-publies.md ; protocole écrit avant calcul).

Usage depuis back/ :
    uv run python studies/current_transfers.py chain \
        --reference ../runs/2026-10-22-consolidation --out ../runs/2026-11-24-reports-publies/chain
    uv run python studies/current_transfers.py projection \
        --fit ../runs/2026-11-24-fit-2027-subst --out ../runs/2026-11-24-reports-publies/projection

`chain` reprend les douze cas du gel comme studies/fixed_first_round.py (`VVS_ROOT` pointé sur
la copie figée, code courant, tableaux lus dans data/registry/transfers_current.yaml du dépôt).
Bras : R0 (référence), V1 (tableaux, mode conjoint), V2 (tableaux + premier tour fixé), R1
(référence, graine + 1) ; hors verdict : F (premier tour fixé sans tableau) et S1 (V2 avec
κ = 60). Nombre de propositions k de V2, F et S1 : 1024, multiplié par 4 (jusqu'à 16 384) tant
qu'un cas a 1 % ou plus de lignes de second tour de taille effective < 2 sous V2 ; décidé sur
les seules tailles effectives (`k_choice.json`) avant tout score.

`projection` rejoue la simulation 2027 du 2026-11-24 (graine 0, substitution, courbe de biais)
pour R0, V1 et V2 (même k que la chaîne si `--k` n'est pas donné).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "data" / "registry" / "transfers_current.yaml"
K_LADDER = (1024, 4096, 16384)
MAX_LOW_SHARE = 0.01  # part maximale de lignes de taille effective < 2, dans chaque cas
S1_KAPPA = 60.0
# nom : (décalage de graine, premier tour fixé, tableaux, κ des tableaux)
ARMS = {
    "R0": (0, False, False, None),
    "V1": (0, False, True, None),
    "V2": (0, True, True, None),
    "R1": (1, False, False, None),
    "F": (0, True, False, None),
    "S1": (0, True, True, S1_KAPPA),
}
METRICS = ("log_score_winner", "log_score_pair", "crps", "coverage90", "crps2", "coverage90_2")
SOURCES = ["back/src/vvs/model/second_round.py", "back/src/vvs/model/simulate.py",
           "back/src/vvs/eval/consolidation.py", "back/src/vvs/registry/transfers_current.py",
           "back/studies/current_transfers.py", "data/registry/transfers_current.yaml"]


def _case_names(spec: dict) -> list[str]:
    return [f"{e}-{h:03d}" for e in spec["elections"] for h in spec["horizons"]]


def _simulate(reference: Path, name: str, spec: dict, arm: str, k: int | None):
    """(tirages, détail par cellule, notes des duels) d'un bras pour un cas du gel."""
    import datetime as dt

    import numpy as np

    from vvs.eval import consolidation as co
    from vvs.model import second_round as sr
    from vvs.model.data import load_polls
    from vvs.model.second_round import SecondRoundModel
    from vvs.registry import default_registry
    from vvs.registry.transfers_current import load_current_transfers

    original = reference / "cases" / name
    case = json.loads((original / "case.json").read_text())
    co.check_case(original, case)
    asof = dt.date.fromisoformat(case["asof"])
    errors, errors2, transfers, turnout, excluded, audit = co.calibration(asof, spec)
    if json.loads(json.dumps(audit, default=str)) != json.loads(
        (original / "calibration.json").read_text()
    ):
        raise ValueError(f"calibration différente du gel : {name}")
    polls = load_polls(case["election"], asof, curated=original / "curated")
    offset, fixed, use_tables, kappa = ARMS[arm]
    tables = None
    if use_tables:
        tables = load_current_transfers(TABLES, default_registry(), elections={case["election"]})
    model2 = SecondRoundModel(
        curated=original / "curated", transfers=transfers, errors=errors2, exclude=excluded,
        current_transfers=tables, current_kappa=kappa, **spec["second_round"],
    )
    default_k = sr.FIXED_PROPOSALS
    sr.FIXED_PROPOSALS = k if (fixed and k) else default_k
    try:
        with np.load(original / "fit/draws.npz") as draws:
            sim, detail = co.simulate_chain(
                case, draws, errors, model2, turnout, spec | {"seed": spec["seed"] + offset},
                polls, "reference", fixed_first_round=fixed,
            )
    finally:
        sr.FIXED_PROPOSALS = default_k
    notes = {f"{a}+{b}": f.notes for (a, b), f in model2.fits.items()
             if any(g.source == "current" for g in f.groups.values())}
    return case, sim, detail, notes


def check_k(args: tuple) -> dict:
    """Tailles effectives de V2 pour un cas et un k (aucun score calculé)."""
    reference, out, name, spec, k = args
    import numpy as np

    folder = out / "kcheck" / f"k{k}" / name
    if (folder / "ess.json").exists():
        return json.loads((folder / "ess.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    _, sim, detail, notes = _simulate(reference, name, spec, "V2", k)
    n = sum(d["n"] for d in detail)
    low = sum(d["n"] * d.get("ess_row_low_share", 0.0) for d in detail) / n
    res = {"case": name, "k": k, "rows": n, "low_share": low,
           "min_row_ess": min(d["ess_row_min"] for d in detail)}
    np.savez_compressed(folder / "draws_V2.npz", **sim)
    (folder / "detail.json").write_text(json.dumps({"detail": detail, "notes": notes}, indent=1))
    (folder / "ess.json").write_text(json.dumps(res, indent=1))
    return res


def run_case(args: tuple) -> str:
    reference, out, name, spec, k = args
    import numpy as np
    import polars as pl

    from vvs.eval import consolidation as co
    from vvs.registry import default_registry

    folder = out / "cases" / name
    if (folder / "complete.json").exists():
        return f"Reprise : {name}"
    folder.mkdir(parents=True, exist_ok=True)
    original = reference / "cases" / name
    reg = default_registry()
    with np.load(original / "draws_reference.npz") as d:
        frozen = dict(d)
    rows, cells, control, all_notes = [], [], {}, {}
    for arm, (offset, fixed, use_tables, kappa) in ARMS.items():
        cached = out / "kcheck" / f"k{k}" / name
        if arm == "V2" and (cached / "draws_V2.npz").exists():
            case = json.loads((original / "case.json").read_text())
            with np.load(cached / "draws_V2.npz") as d:
                sim = dict(d)
            stored = json.loads((cached / "detail.json").read_text())
            detail, notes = stored["detail"], stored["notes"]
        else:
            case, sim, detail, notes = _simulate(reference, name, spec, arm, k)
        if arm == "R0":
            control = {c: bool(np.array_equal(sim[c], frozen[c])) for c in frozen}
        summary, _ = co.score_chain(sim, case["election"], reg)
        second = co.score_second_round(sim, case["election"], reg)
        key = {"election": case["election"], "horizon": case["horizon"], "arm": arm,
               "seed": spec["seed"] + offset, "fixed_first_round": fixed, "tables": use_tables,
               "kappa": kappa, "proposals": k if fixed else None}
        rows.append(key | summary | second)
        cells += [key | c for c in detail]
        all_notes[arm] = notes
        np.savez_compressed(folder / f"draws_{arm}.npz", **sim)
    pl.DataFrame(rows).write_csv(folder / "metrics.csv", float_precision=8)
    pl.DataFrame(cells, infer_schema_length=None).write_csv(folder / "cells.csv")
    co.write_json(folder / "notes.json", all_notes)
    co.write_json(folder / "complete.json", {"replay_matches_frozen_reference": control})
    return f"Terminé : {name} ; rejeu R0 identique au gel : {all(control.values())}"


def aggregate(out: Path, spec: dict) -> dict:
    import polars as pl

    m = pl.concat(
        [pl.read_csv(out / "cases" / n / "metrics.csv") for n in _case_names(spec)],
        how="vertical_relaxed",
    ).sort("election", "horizon", "arm")
    m.write_csv(out / "metrics.csv", float_precision=8)
    wide = {arm: m.filter(pl.col("arm") == arm).sort("election", "horizon") for arm in ARMS}
    n = wide["R0"].height
    excluded = sorted({f"{r['election']}-{r['horizon']}" for arm in ARMS
                       for r in wide[arm].filter(pl.col("n_duel_draws") == 0).iter_rows(named=True)})
    result: dict = {"n_cases": n, "excluded_second_round_cases": excluded,
                    "overall": {}, "by_horizon": {}}

    def mean(x):
        x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(x) / len(x) if x else None

    for metric in METRICS:
        col = {arm: wide[arm][metric].to_list() for arm in ARMS}
        d_noise = [b - a for a, b in zip(col["R0"], col["R1"])]
        sigma = math.sqrt(sum(d * d for d in d_noise) / len(d_noise))
        entry = {arm: mean(col[arm]) for arm in ARMS}
        entry |= {"noise_sigma_case": sigma, "noise_se_mean": sigma / math.sqrt(n)}
        for arm in ("V1", "V2", "F", "S1"):
            diff = [v - r for v, r in zip(col[arm], col["R0"])]
            entry[f"diff_{arm}_R0"] = mean(diff)
            entry[f"max_abs_diff_{arm}_R0"] = max(abs(d) for d in diff)
        result["overall"][metric] = entry
        for h in spec["horizons"]:
            sel = [i for i, x in enumerate(wide["R0"]["horizon"].to_list()) if x == h]
            result["by_horizon"].setdefault(str(h), {})[metric] = {
                **{arm: mean([col[arm][i] for i in sel]) for arm in ARMS},
                **{f"diff_{arm}_R0": mean([col[arm][i] - col["R0"][i] for i in sel])
                   for arm in ("V1", "V2", "F", "S1")},
                "noise_se_mean": sigma / math.sqrt(len(sel)),
            }
    o = result["overall"]
    lw, c2, cov = o["log_score_winner"], o["crps2"], o["coverage90_2"]
    checks = {
        "log_score_winner": lw["diff_V2_R0"] >= -2 * lw["noise_se_mean"],
        "crps2": c2["diff_V2_R0"] <= 2 * c2["noise_se_mean"],
        "coverage90_2": abs(cov["V2"] - 0.9) <= abs(cov["R0"] - 0.9) + 2 * cov["noise_se_mean"],
    }
    controls = {
        f"{metric}_identical_{arm}": o[metric][f"max_abs_diff_{arm}_R0"] == 0.0
        for metric in ("log_score_pair", "crps") for arm in ("V1", "V2", "F", "S1")
    } | {
        "replay_R0_equals_frozen": all(
            all(json.loads((out / "cases" / nm / "complete.json").read_text())[
                "replay_matches_frozen_reference"].values())
            for nm in _case_names(spec)
        ),
    }
    result |= {"checks": checks, "controls": controls, "admitted": all(checks.values())}
    return result


def chain(args) -> None:
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
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {
        "reference": record["id"], "reference_commit": record["commit"], "code_head": head,
        "code_dirty": subprocess.check_output(
            ["git", "status", "--porcelain", "--", *SOURCES], cwd=REPO, text=True
        ).splitlines(),
        "sources_sha256": {s: co.digest(REPO / s) for s in SOURCES},
        "versions": record["versions"],
        "arms": {a: {"seed": spec["seed"] + v[0], "fixed_first_round": v[1], "tables": v[2],
                     "kappa": v[3]} for a, v in ARMS.items()},
        "criterion_arms": ["R0", "V2", "R1"], "n_draws": spec["n_draws"],
        "command": " ".join(sys.argv),
    }
    co.write_json(out / "manifest.json", manifest)
    names = _case_names(spec)
    with ProcessPoolExecutor(args.jobs) as pool:
        chosen, ladder = None, []
        for k in K_LADDER:
            res = list(pool.map(check_k, [(reference, out, n, spec, k) for n in names]))
            ok = all(r["low_share"] < MAX_LOW_SHARE for r in res)
            ladder.append({"k": k, "ok": ok, "cases": res})
            print(f"k = {k} : part maximale de lignes de taille effective < 2 = "
                  f"{max(r['low_share'] for r in res):.4f}", flush=True)
            if ok:
                chosen = k
                break
        co.write_json(out / "k_choice.json", {"chosen": chosen, "rule": f"< {MAX_LOW_SHARE} par cas",
                                              "ladder": ladder})
        if chosen is None:
            raise RuntimeError("aucun k de l'échelle ne satisfait la règle de taille effective")
        for msg in pool.map(run_case, [(reference, out, n, spec, chosen) for n in names]):
            print(msg, flush=True)
    result = aggregate(out, spec) | {"proposals": chosen}
    co.write_json(out / "results.json", result)
    print(json.dumps({k: result[k] for k in ("proposals", "checks", "controls", "admitted")}, indent=2))


def projection(args) -> None:
    import datetime as dt

    from vvs.model import second_round as sr
    from vvs.model.simulate import simulate
    from vvs.registry.transfers_current import load_current_transfers

    out = args.out.resolve()
    k = args.k
    if k is None and "V2" in args.arms:
        k = json.loads((args.chain / "k_choice.json").read_text())["chosen"]
    tables = load_current_transfers(TABLES, elections={2027})
    arms = {"R0": (False, None), "V1": (False, tables), "V2": (True, tables)}
    default_k = sr.FIXED_PROPOSALS
    for arm, (fixed, tab) in arms.items():
        if arm not in args.arms:
            continue
        folder = out / arm
        if (folder / "manifest.json").exists():
            print(f"Reprise : {arm}", flush=True)
            continue
        sr.FIXED_PROPOSALS = k if fixed else default_k
        try:
            res = simulate(args.fit, dt.date(2026, 11, 24), folder, seed=0, rn_bias_curve=True,
                           fixed_first_round=fixed, current_transfers=tab)
        finally:
            sr.FIXED_PROPOSALS = default_k
        print(f"{arm} : {res.manifest['duration_s']} s", flush=True)
    if not all((out / a / "draws.parquet").exists() for a in arms):
        return
    summary = projection_summary(out, list(arms))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary["win"], indent=1))


def projection_summary(out: Path, arms: list[str]) -> dict:
    """Victoire par candidat ; dix duels les plus probables sous R0 (P(duel), part médiane de a et
    intervalle à 90 %, P(a gagne)) ; reports a posteriori du duel Le Pen – Mélenchon par groupe."""
    import numpy as np
    import polars as pl

    draws = {a: pl.read_parquet(out / a / "draws.parquet") for a in arms}
    n = draws["R0"].height
    win = {}
    for a, d in draws.items():
        counts = d.group_by("winner").len().sort("len", descending=True)
        win[a] = {w: c / n for w, c in counts.iter_rows() if c / n >= 0.001}
    duels = (draws["R0"].group_by("qualified_a", "qualified_b").len()
             .sort("len", descending=True).head(10))
    top = []
    for qa, qb, cnt in duels.iter_rows():
        row = {"a": qa, "b": qb}
        for a, d in draws.items():
            s = d.filter((pl.col("qualified_a") == qa) & (pl.col("qualified_b") == qb))
            x = s["share_a_second_round"].to_numpy()
            row[a] = {"p_duel": s.height / n, "median": float(np.median(x)),
                      "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95)),
                      "p_a_wins": float(np.mean(x > 50))}
        top.append(row)
    transfers = {}
    for a in arms:
        t = pl.read_parquet(out / a / "transfers.parquet").filter(
            pl.col("candidate_a").is_in(["le_pen_marine", "melenchon_jean_luc"])
            & pl.col("candidate_b").is_in(["le_pen_marine", "melenchon_jean_luc"])
        )
        transfers[a] = t.select("candidate_a", "candidate_b", "group", "n", "weight",
                                "to_a_q50", "to_b_q50", "none_q50").to_dicts()
    return {"n_draws": n, "win": win, "top_duels": top, "le_pen_melenchon_transfers": transfers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chain")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("projection")
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chain", type=Path, default=REPO / "runs/2026-11-24-reports-publies/chain")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--arms", nargs="+", default=["R0", "V1", "V2"])
    args = parser.parse_args()
    chain(args) if args.cmd == "chain" else projection(args)


if __name__ == "__main__":
    main()
