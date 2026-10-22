"""Compare les variantes de docs/11 avec les fonctions statistiques du gel, sur CPU.

Lancer scripts/run_historical_scenarios.sh. Les trois entrées de l'étude sont copiées
et hachées ; une reprise exige les mêmes fichiers et vérifie aussi toute la référence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    return {
        str(p.relative_to(folder)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def verify_inventory(folder, expected):
    actual = inventory(folder)
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f"fichiers modifiés : {folder}: {changed}")


def validate_simulation(sim, n):
    import numpy as np

    shares = sim["shares"]
    if shares.shape != (n, len(sim["candidates"])) or not np.isfinite(shares).all():
        raise ValueError("dimensions ou parts invalides")
    if (shares < 0).any() or not np.allclose(shares.sum(1), 100, atol=1e-10, rtol=0):
        raise ValueError("parts non normalisées")
    qualified = sim["qualified"]
    if qualified.shape != (n, 2) or (qualified[:, 0] == qualified[:, 1]).any():
        raise ValueError("paire de qualifiés invalide")
    if not (qualified == sim["winner"][:, None]).any(1).all():
        raise ValueError("vainqueur hors duel")
    second = sim["share_second"]
    if not (np.isfinite(second) & (second >= 0) & (second <= 100)).all():
        raise ValueError("part de second tour invalide")


def run(args):
    reference, out = args.reference.resolve(), args.out.resolve()
    if out == reference or out.is_relative_to(reference) or reference.is_relative_to(out):
        raise ValueError("le dossier d'étude doit être séparé de la référence")
    source = reference / "source"
    os.environ.update(
        VVS_ROOT=str(source),
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        POLARS_MAX_THREADS="2",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "back/src"))
    import numpy as np
    import polars as pl
    import yaml

    from vvs.eval import consolidation as frozen
    from vvs.model.data import load_polls
    from vvs.model.second_round import SecondRoundModel
    from vvs.model.simulate import _versions
    from vvs.registry import default_registry

    if not Path(frozen.__file__).resolve().is_relative_to(source):
        raise ValueError("le code statistique n'est pas celui du gel")
    record = frozen.verify(reference)
    protocol = yaml.safe_load(args.protocol.read_text())
    pin = json.loads(args.pin.read_text())
    for key in ("id", "commit", "spec", "versions", "files"):
        if record[key] != pin[key]:
            raise ValueError(f"référence différente du verrou : {key}")
    if protocol["reference"] != record["id"] or _versions() != record["versions"]:
        raise ValueError("référence ou dépendances différentes du protocole")
    spec, registry = record["spec"], default_registry()
    inputs = {
        "runner.py": Path(__file__),
        "implementation.py": args.implementation,
        "protocol.yaml": args.protocol,
        "reference.lock.json": args.pin,
    }
    input_hashes = {name: digest(p) for name, p in inputs.items()}
    manifest_file = out / "study.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["inputs"] != input_hashes:
            raise ValueError("entrées de l'étude modifiées : utiliser un autre dossier")
        verify_inventory(out / "source", input_hashes)
        verify_inventory(reference, manifest["reference_files"])
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("dossier d'étude non vide sans manifest")
        manifest = {
            "id": protocol["id"],
            "protocol": protocol,
            "inputs": input_hashes,
            "reference_files": inventory(reference),
            "reference_commit": record["commit"],
            "versions": record["versions"],
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        (out / "source").mkdir(parents=True)
        for name, path in inputs.items():
            shutil.copy2(path, out / "source" / name)
        frozen.write_json(manifest_file, manifest)
    loader = importlib.util.spec_from_file_location(
        "scenario_variant", out / "source/implementation.py"
    )
    variant_module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(variant_module)
    events = yaml.safe_load((source / protocol["events_source"]).read_text())["events"]
    all_metrics, all_outcomes, audit_cases = [], [], []
    for election in spec["elections"]:
        for horizon in spec["horizons"]:
            name = f"{election}-{horizon:03d}"
            original, folder = reference / "cases" / name, out / "cases" / name
            if not frozen.complete_score(original):
                raise ValueError(f"cas de référence incomplet : {name}")
            case = json.loads((original / "case.json").read_text())
            frozen.check_case(original, case)
            if not json.loads((original / "fit/quality.json").read_text())["acceptable"]:
                raise ValueError(f"ajustement non conforme : {name}")
            metrics = pl.read_parquet(original / "metrics.parquet").with_columns(
                pl.lit("frozen").alias("variant")
            )
            all_metrics.append(metrics)
            all_outcomes.append(
                pl.read_parquet(original / "outcomes.parquet").with_columns(
                    pl.lit("frozen").alias("variant")
                )
            )
            if (folder / "complete.json").exists():
                hashes = json.loads((folder / "complete.json").read_text())
                # complete.json lui-même n'est pas dans son inventaire.
                actual = inventory(folder)
                actual.pop("complete.json")
                if hashes != actual:
                    raise ValueError(f"sortie d'étude modifiée : {name}")
                print(f"Reprise vérifiée : {name}", flush=True)
            else:
                folder.mkdir(parents=True, exist_ok=True)
                asof = dt.date.fromisoformat(case["asof"])
                polls = load_polls(election, asof, curated=original / "curated")
                errors, errors2, transfers, turnout, excluded, calibration = frozen.calibration(
                    asof, spec
                )
                # La sérialisation reproduit celle des fichiers du gel (dates, tuples).
                canonical = json.loads(json.dumps(calibration, default=str))
                if canonical != json.loads((original / "calibration.json").read_text()):
                    raise ValueError(f"calibration différente : {name}")

                def simulate(
                    configuration,
                    method,
                    draws,
                    original=original,
                    transfers=transfers,
                    errors2=errors2,
                    excluded=excluded,
                    errors=errors,
                    turnout=turnout,
                    polls=polls,
                ):
                    model2 = SecondRoundModel(
                        curated=original / "curated",
                        transfers=transfers,
                        errors=errors2,
                        exclude=excluded,
                        **spec["second_round"],
                    )
                    result, cells = frozen.simulate_chain(
                        configuration, draws, errors, model2, turnout, spec, polls, method
                    )
                    validate_simulation(result, spec["n_draws"])
                    return result, cells

                with np.load(original / "fit/draws.npz") as draws:
                    # Contrôle indépendant de la chaîne, y compris second tour et RNG.
                    replay = {}
                    for method in ("reference", "mean14d"):
                        result, _ = simulate(case, method, draws)
                        with np.load(original / f"draws_{method}.npz") as saved:
                            differences = {}
                            for key, value in result.items():
                                if value.dtype.kind in "fc":
                                    delta = float(np.max(np.abs(value - saved[key])))
                                    if not np.allclose(value, saved[key], rtol=1e-12, atol=1e-12):
                                        raise ValueError(
                                            f"rejeu différent : {name}/{method}/{key}: {delta}"
                                        )
                                    differences[key] = delta
                                elif not np.array_equal(value, saved[key]):
                                    raise ValueError(f"rejeu différent : {name}/{method}/{key}")
                            replay[method] = differences
                    frozen.write_json(folder / "replay.json", replay)
                    for variant in protocol["variants"]:
                        configurations, rule = variant_module.reconstruct(
                            polls, election, asof, registry, protocol, events, variant
                        )
                        missing = set(variant_module.presence(configurations)) - set(
                            draws["candidates"]
                        )
                        if missing:
                            raise ValueError(
                                f"candidats absents de l'ajustement : {sorted(missing)}"
                            )
                        changed = configurations != case["configurations"]
                        new_case = case | {"configurations": configurations, "scenario_rule": rule}
                        dest = folder / variant
                        frozen.write_json(
                            dest / "case.json",
                            new_case
                            | {
                                "reused_reference": not changed,
                                "presence": variant_module.presence(configurations),
                                "reference_presence": variant_module.presence(
                                    case["configurations"]
                                ),
                            },
                        )
                        metrics_rows, outcomes, cells_rows = [], [], []
                        for method in ("reference", "mean14d"):
                            if changed:
                                result, cells = simulate(new_case, method, draws)
                                np.savez_compressed(dest / f"draws_{method}.npz", **result)
                            else:
                                shutil.copy2(
                                    original / f"draws_{method}.npz", dest / f"draws_{method}.npz"
                                )
                                with np.load(dest / f"draws_{method}.npz") as saved:
                                    result = dict(saved)
                                validate_simulation(result, spec["n_draws"])
                                cells = (
                                    pl.read_parquet(original / "cells.parquet")
                                    .filter(pl.col("method") == method)
                                    .drop("election", "horizon", "method")
                                    .to_dicts()
                                )
                            summary, rows = frozen.score_chain(result, election, registry)
                            summary["unforecast_share"] = float(summary["unforecast_share"])
                            key = {
                                "election": election,
                                "horizon": horizon,
                                "method": method,
                                "variant": variant,
                            }
                            metrics_rows.append(key | summary | {"fit_acceptable": True})
                            outcomes.extend(key | r for r in rows)
                            cells_rows.extend(key | r for r in cells)
                        pl.DataFrame(metrics_rows).write_parquet(dest / "metrics.parquet")
                        pl.DataFrame(outcomes).write_parquet(dest / "outcomes.parquet")
                        pl.DataFrame(cells_rows).write_parquet(dest / "cells.parquet")
                        print(
                            f"Terminé : {name}/{variant} ({len(configurations)} configurations)",
                            flush=True,
                        )
                frozen.write_json(folder / "complete.json", inventory(folder))
            for variant in protocol["variants"]:
                dest = folder / variant
                all_metrics.append(pl.read_parquet(dest / "metrics.parquet"))
                all_outcomes.append(pl.read_parquet(dest / "outcomes.parquet"))
                audit_cases.append(
                    json.loads((dest / "case.json").read_text())
                    | {
                        "variant": variant,
                        "outputs": json.loads((folder / "complete.json").read_text()),
                        "replay": json.loads((folder / "replay.json").read_text()),
                    }
                )
    # La référence entière, pas seulement ses 165 entrées, doit être restée intacte.
    verify_inventory(reference, manifest["reference_files"])
    metrics = pl.concat(all_metrics, how="diagonal_relaxed")
    metrics.write_parquet(out / "metrics.parquet")
    metrics.write_csv(out / "metrics.csv", float_precision=10)
    pl.concat(all_outcomes, how="diagonal_relaxed").write_csv(
        out / "outcomes.csv", float_precision=10
    )
    frozen.write_json(
        out / "audit.json", {"study": manifest, "cases": audit_cases, "reference_unchanged": True}
    )
    print(f"Étude complète : {out} ; référence intacte", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "out", "protocol", "implementation", "pin"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
