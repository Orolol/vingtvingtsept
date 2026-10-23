"""Cinq expériences isolées ; lancer scripts/run_structure_study.sh (reprise vérifiée)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def initialize(args):
    import yaml

    root, reference, out = args.project.resolve(), args.reference.resolve(), args.out.resolve()
    if out == reference or out.is_relative_to(reference) or reference.is_relative_to(out):
        raise ValueError("sorties et référence doivent être séparées")
    source = reference / "source"
    os.environ["VVS_ROOT"] = str(source)
    sys.path.insert(0, str(source / "back/src"))
    sys.dont_write_bytecode = True
    from vvs.eval import consolidation as frozen
    from vvs.model.simulate import _versions

    record = frozen.verify(reference)
    pinned = json.loads((root / "data/registry/reference_2026-10-22.lock.json").read_text())
    for key in ("id", "commit", "spec", "versions", "files"):
        if record[key] != pinned[key]:
            raise ValueError(f"référence différente du verrou : {key}")
    if _versions() != record["versions"]:
        raise ValueError("versions différentes du gel")
    inputs = {"runner.py": Path(__file__), "implementation.py": root / "back/src/vvs/eval/structure.py",
              "substitution.py": root / "back/studies/substitution.py",
              "protocol.yaml": root / "data/registry/structure_study_2026-10-23.yaml"}
    expected = {k: digest(v) for k, v in inputs.items()}
    protocol = yaml.safe_load(inputs["protocol.yaml"].read_text())
    if protocol["reference"] != record["id"]:
        raise ValueError("référence incorrecte")
    auxiliary = {str(p.relative_to(root)): digest(p) for p in
                 [root / "data/curated/cevipof_votes.parquet",
                  root / "data/registry/cevipof_datasets.yaml"] if p.exists()}
    manifest = out / "study.json"
    if not manifest.exists():
        if out.exists() and any(out.iterdir()):
            raise ValueError("dossier non vide sans manifeste")
        (out / "source").mkdir(parents=True)
        for name, path in inputs.items():
            shutil.copy2(path, out / "source" / name)
        frozen.write_json(manifest, {"inputs": expected, "reference_files": inventory(reference),
                                     "auxiliary": auxiliary, "protocol": protocol,
                                     "created_at": str(dt.datetime.now(dt.UTC)),
                                     "versions": record["versions"]})
    saved = json.loads(manifest.read_text())
    if saved["inputs"] != expected or inventory(out / "source") != expected:
        raise ValueError("sources modifiées ; utiliser un nouveau dossier d'étude")
    if saved["auxiliary"] != auxiliary or saved["reference_files"] != inventory(reference):
        raise ValueError("données ou référence modifiées")
    implementation = module("structure_experiment", out / "source/implementation.py")
    return root, reference, out, frozen, record["spec"], protocol, implementation


def completed(folder):
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    actual = inventory(folder)
    actual.pop("complete.json")
    if actual != json.loads(marker.read_text()):
        raise ValueError(f"sorties modifiées : {folder}")
    return True


def finish(folder, frozen):
    files = inventory(folder)
    files.pop("complete.json", None)
    frozen.write_json(folder / "complete.json", files)


def case_stage(ctx, name):
    import numpy as np
    import polars as pl

    from vvs.model.data import load_polls
    from vvs.model.second_round import SecondRoundModel
    from vvs.registry import default_registry

    _, reference, out, frozen, spec, protocol, imp = ctx
    folder, original = out / "cases" / name, reference / "cases" / name
    if completed(folder):
        print(f"Reprise vérifiée : {name}", flush=True)
        return
    case = json.loads((original / "case.json").read_text())
    frozen.check_case(original, case)
    if not frozen.complete_score(original):
        raise ValueError(f"cas du gel incomplet : {name}")
    if not json.loads((original / "fit/quality.json").read_text())["acceptable"]:
        raise ValueError(f"ajustement non conforme : {name}")
    folder.mkdir(parents=True, exist_ok=True)
    asof = dt.date.fromisoformat(case["asof"])
    errors, errors2, transfers, turnout, excluded, audit = frozen.calibration(asof, spec)
    canonical = json.loads(json.dumps(audit, default=str))
    if canonical != json.loads((original / "calibration.json").read_text()):
        raise ValueError(f"calibration différente : {name}")
    polls = load_polls(case["election"], asof, curated=original / "curated")
    reg = default_registry()
    metrics, outcomes, cells = [], [], []

    def save(variant, result, details=()):
        shares = result["shares"]
        if (not np.isfinite(shares).all() or (shares < 0).any()
                or not np.allclose(shares.sum(1), 100, atol=1e-9, rtol=0)):
            raise ValueError("parts invalides")
        if not (result["qualified"] == result["winner"][:, None]).any(1).all():
            raise ValueError("vainqueur non qualifié")
        row, people = frozen.score_chain(result, case["election"], reg)
        row = imp.fixed_count_scores(row, spec["n_draws"])
        key = {"election": case["election"], "horizon": case["horizon"], "variant": variant}
        metrics.append(key | row)
        outcomes.extend(key | r for r in people)
        cells.extend(key | r for r in details)
        np.savez_compressed(folder / f"draws_{variant}.npz", **result)

    saved = {}
    for method in ("reference", "mean14d"):
        with np.load(original / f"draws_{method}.npz") as d:
            saved[method] = dict(d)
        save(method, saved[method])
    save("ensemble50", imp.mix_trajectories(saved["reference"], saved["mean14d"]))
    print(f"{name} : mélange terminé ; calibration bootstrap", flush=True)
    parameters, diagnostics = imp.bootstrap_errors(errors, audit, spec, protocol)
    frozen.write_json(folder / "bootstrap.json", {"parameters": parameters, "diagnostics": diagnostics,
                                                   "calibration": audit})

    with np.load(original / "fit/draws.npz") as draws:
        def simulate(error, frame, method):
            model2 = SecondRoundModel(curated=original / "curated", transfers=transfers,
                                      errors=errors2, exclude=excluded, **spec["second_round"])
            return frozen.simulate_chain(case, draws, error, model2, turnout, spec, frame, method)

        # Vérifie indépendamment que l'appel du runner reproduit le témoin avant variantes.
        replay, _ = simulate(errors, polls, "mean14d")
        for key, value in replay.items():
            exact = np.array_equal(value, saved["mean14d"][key])
            if not exact and (value.dtype.kind not in "fc" or not np.allclose(
                    value, saved["mean14d"][key], rtol=1e-12, atol=1e-12)):
                raise ValueError(f"rejeu différent : {name}/{key}")
        save("calibration_bootstrap", *simulate(imp.ErrorMixture(parameters), polls, "reference"))
        full = imp.unthinned_polls(case["election"], asof, curated=original / "curated")
        rolling = {}
        variants = [("gls_thinned", polls, 0.), ("rolling_overlap", full, 1.),
                    ("rolling_half", full, .5), ("rolling_independent", full, 0.)]
        for variant, frame, overlap in variants:
            summary, diag = imp.rolling_summary(frame, asof, protocol, overlap)
            save(variant, *simulate(errors, summary, "mean14d"))
            rolling[variant] = diag
    frozen.write_json(folder / "rolling.json", rolling)
    pl.DataFrame(metrics).write_csv(folder / "metrics.csv")
    pl.DataFrame(outcomes).write_csv(folder / "outcomes.csv")
    if cells:
        pl.DataFrame(cells).write_csv(folder / "cells.csv")
    finish(folder, frozen)
    print(f"Terminé : {name}", flush=True)


def substitution_stage(ctx):
    import numpy as np
    import polars as pl

    from vvs.registry import default_registry

    _, _, out, frozen, _, protocol, imp = ctx
    folder = out / "substitution"
    if completed(folder):
        print("Reprise vérifiée : substitution", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    previous = module("previous_substitution", out / "source/substitution.py")
    pairs = {e: previous.nested_pairs(e, default_registry()) for e in (2002, 2007, 2012, 2017, 2022)}
    rows, fits = [], []
    for election in protocol["elections"]:
        for fold in ("loo", "chronological"):
            train_years = [e for e in pairs if e != election and (fold == "loo" or e < election)]
            train = [p for e in train_years for p in pairs[e]]
            # Même borne numérique pour le témoin emboîté et le modèle mixte.
            lam = {b: max(.02, x) for b, x in previous.fit_lambda(train).items()}
            tau, diag = imp.fit_mixture(train, lam, protocol)
            fits.append({"election": election, "fold": fold, "training_elections": train_years,
                         "lambda": lam, "tau": tau, **diag})
            for pair in pairs[election]:
                keep = pair["present"]
                predictions = {"nested": previous.predict(pair, lam),
                               "mixed": imp.mixed_prediction(pair, lam, tau, protocol["bloc_axis"])}
                for method, q in predictions.items():
                    rows.append({"election": election, "fold": fold, "method": method,
                                 "poll_id": pair["poll_id"], "full": pair["full"],
                                 "reduced": pair["reduced"], "weight": pair["weight"],
                                 "mae": float(100*np.abs(q[keep]-pair["y_red"][keep]).mean()),
                                 "cross_entropy": float(-pair["y_red"][keep] @ np.log(q[keep]))})
            print(f"Substitution {election}/{fold}: tau={tau:.4f}, {len(pairs[election])} paires", flush=True)
    if rows:
        frame = pl.DataFrame(rows)
        frame.write_csv(folder / "pairs.csv")
        means = frame.group_by("election", "fold", "method").agg(
            pl.col("poll_id").n_unique().alias("polls"),
            *[(pl.col(c)*pl.col("weight")).sum().truediv(pl.col("weight").sum()).alias(c)
              for c in ("mae", "cross_entropy")]).sort("fold", "election", "method")
        means.write_csv(folder / "means.csv")
    frozen.write_json(folder / "fits.json", fits)
    frozen.write_json(folder / "inventory.json", {str(e): {"pairs": len(p),
                       "polls": len({r["poll_id"] for r in p})} for e, p in pairs.items()})
    finish(folder, frozen)


def panel_stage(ctx):
    import polars as pl
    import yaml

    root, _, out, frozen, _, _, _ = ctx
    folder = out / "panel"
    if completed(folder):
        print("Reprise vérifiée : panel", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    path = root / "data/curated/cevipof_votes.parquet"
    datasets = yaml.safe_load((root / "data/registry/cevipof_datasets.yaml").read_text())["datasets"]
    public = [d["key"] for d in datasets if any(f.get("role") in ("data", "data_alt") and not f.get("restricted")
                                               for f in d["files"])]
    report = {"registered_datasets": len(datasets), "public_microdata": public,
              "available": path.exists(), "status": "blocked_missing_longitudinal_intentions"}
    if path.exists():
        frame = pl.read_parquet(path)
        summary = frame.group_by("dataset", "series", "election", "wave", "stage").agg(
            pl.len().alias("respondents"), pl.col("vote_round1_kind").unique().alias("kinds"))
        report["curated_rows"] = len(frame)
        report["waves"] = summary.to_dicts()
        usable = frame.filter(pl.col("vote_round1_kind") == "intention")
        report["intention_rows"] = len(usable)
        report["intention_datasets"] = usable["dataset"].unique().to_list()
        if usable["dataset"].n_unique() >= 2:
            raise ValueError("plusieurs vagues disponibles : valider l'identifiant longitudinal avant étude")
    report["reason"] = ("Aucune paire de vagues d'intentions avec identifiant longitudinal validé. "
                         "Les souvenirs post-électoraux ne sont pas des intentions préélectorales.")
    frozen.write_json(folder / "audit.json", report)
    finish(folder, frozen)


def report_stage(ctx):
    import polars as pl

    _, reference, out, frozen, spec, _, _ = ctx
    frames, outcomes = [], []
    for election in spec["elections"]:
        for horizon in spec["horizons"]:
            folder = out / "cases" / f"{election}-{horizon:03d}"
            if not completed(folder):
                raise ValueError(f"cas incomplet : {folder}")
            frames.append(pl.read_csv(folder / "metrics.csv"))
            outcomes.append(pl.read_csv(folder / "outcomes.csv"))
    for stage in ("panel", "substitution"):
        if not completed(out / stage):
            raise ValueError(f"étape incomplète : {stage}")
    frame = pl.concat(frames, how="vertical_relaxed")
    dest = out / "review"
    dest.mkdir(exist_ok=True)
    frame.write_csv(dest / "metrics.csv")
    pl.concat(outcomes).write_csv(dest / "outcomes.csv")
    columns = ["log_score_winner", "log_score_pair", "crps", "coverage50", "coverage90", "brier_win"]
    base = frame.filter(pl.col("variant") == "reference").select(
        "election", "horizon", *[pl.col(c).alias(c+"_reference") for c in columns])
    delta = frame.join(base, on=["election", "horizon"]).with_columns(
        *[(pl.col(c)-pl.col(c+"_reference")).alias("delta_"+c) for c in columns])
    delta.write_csv(dest / "comparisons.csv")
    delta.group_by("variant", "horizon").agg(
        *[pl.col("delta_"+c).mean() for c in columns]).sort("variant", "horizon").write_csv(dest / "means.csv")
    saved = json.loads((out / "study.json").read_text())
    if saved["reference_files"] != inventory(reference):
        raise ValueError("référence modifiée")
    frozen.write_json(dest / "audit.json", {"reference_unchanged": True,
                          "reference_files": len(saved["reference_files"]), "cases": len(frames),
                          "rows": len(frame), "input_hashes": saved["inputs"]})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    ctx = initialize(args)
    if args.stage == "all":
        # Manifest créé une seule fois, puis processus isolés, trois tâches CPU au maximum.
        stages = ["panel", "substitution"] + [f"{e}-{h:03d}" for e in ctx[4]["elections"]
                                                for h in ctx[4]["horizons"]]
        def dispatch(stage):
            log = args.out / f"{stage}.log"
            with log.open("w") as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--project", str(args.project), "--reference", str(args.reference),
                                "--out", str(args.out), "--stage", stage],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(f"Étape terminée : {stage}", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(dispatch, stages))
        report_stage(ctx)
    elif args.stage == "substitution":
        substitution_stage(ctx)
    elif args.stage == "panel":
        panel_stage(ctx)
    elif args.stage == "report":
        report_stage(ctx)
    else:
        case_stage(ctx, args.stage)


if __name__ == "__main__":
    main()
