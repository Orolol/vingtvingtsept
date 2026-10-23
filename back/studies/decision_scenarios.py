"""Scénarios historiques par décision de candidature (lot 2 de docs/12), sur le gel, CPU.

Protocole : docstring de vvs.eval.decision_scenarios et
data/registry/historical_decisions_2026-10-23.yaml. Lancer scripts/run_decision_scenarios.sh.

Trois étapes, chacune reprenable :

- `build` (code courant) : lit le registre des candidatures (lot 1), les noms des tirages
  figés et les sondages du gel ; écrit les configurations et leur audit dans
  <out>/decisions/<cas>/<variante>.json. Aucune simulation, aucun score.
- `run` (code du gel, comme studies/historical_scenarios.py) : vérifie la référence et le
  verrou, rejoue les douze cas de référence avec la fonction de simulation de la variante
  (identité au bit près attendue), simule les variantes, score avec `score_chain` du gel ;
  écrit <out>/cases/<cas>/<variante>/ au format de runs/2026-10-22-historical-scenarios.
- `report` (code courant) : diagnostics par étape (`stage_diagnostics.compare_study`),
  erreurs Monte-Carlo par lots, critères déclarés, tableaux et report.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def check_separate(reference: Path, out: Path) -> None:
    if out == reference or out.is_relative_to(reference) or reference.is_relative_to(out):
        raise ValueError("le dossier d'étude doit être séparé de la référence")
    for frozen in ("2026-10-22-consolidation", "2026-10-22-historical-scenarios"):
        if out.name == frozen:
            raise ValueError(f"dossier réservé : {frozen}")


# --------------------------------------------------------------------------- build


def build(args) -> None:
    import numpy as np
    import polars as pl
    import yaml

    from vvs import paths
    from vvs.eval import decision_scenarios as ds
    from vvs.eval.configurations import list_publication
    from vvs.eval.consolidation import verify
    from vvs.registry import default_registry
    from vvs.registry.candidacies import load_candidacies

    reference, out = args.reference.resolve(), args.out.resolve()
    check_separate(reference, out)
    record = verify(reference)
    pin = json.loads(args.pin.read_text())
    for key in ("id", "commit", "spec", "files"):
        if record[key] != pin[key]:
            raise ValueError(f"référence différente du verrou : {key}")
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol["reference"] != record["id"]:
        raise ValueError("référence différente du protocole")
    spec, registry = record["spec"], default_registry()
    for election in spec["elections"]:
        caps, total = ds.bloc_caps(registry, election)
        if caps | {"total": total} != protocol["caps"]["values"][election]:
            raise ValueError(f"plafonds différents du protocole : {election}")
    frozen_curated = reference / "source/data/curated"
    polls_all = pl.read_parquet(frozen_curated / "polls.parquet")
    results_all = pl.read_parquet(frozen_curated / "poll_results.parquet")
    manifest = out / "decisions/manifest.json"
    inputs = {
        "protocol.yaml": digest(args.protocol),
        "implementation.py": digest(Path(ds.__file__)),
        "builder.py": digest(Path(__file__)),
        **{
            f"candidacies_{e}.yaml": digest(paths.REGISTRY / f"candidacies_{e}.yaml")
            for e in spec["elections"]
        },
    }
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["inputs"] != inputs:
            raise ValueError("entrées de la construction modifiées : utiliser un autre dossier")
        print("Construction déjà faite, entrées identiques", flush=True)
        return
    summary = []
    for election in spec["elections"]:
        cr = load_candidacies(election)
        previous_election = max(
            y for y, e in registry.elections.items() if y < election and e.candidates
        )
        previous = ds.poll_medians(polls_all, results_all, previous_election)
        official = sorted(c.candidate_id for c in registry.elections[election].candidates)
        for horizon in spec["horizons"]:
            name = f"{election}-{horizon:03d}"
            case = json.loads((reference / "cases" / name / "case.json").read_text())
            asof = dt.date.fromisoformat(case["asof"])
            if asof >= list_publication(election, registry):
                states = cr.state_at(asof)
                listed = sorted(p for p, s in states.items() if s.state == "official")
                if listed != official:
                    raise ValueError(f"liste officielle du registre différente : {name}")
                for variant in ds.variants(protocol):
                    write_json(
                        out / "decisions" / name / f"{variant}.json",
                        {"election": election, "horizon": horizon, "asof": str(asof),
                         "source": "official_list", "candidates": official},
                    )
                summary.append({"case": name, "variant": "*", "source": "official_list"})
                continue
            curated = reference / "cases" / name / "curated"
            current = ds.poll_medians(
                pl.read_parquet(curated / "polls.parquet"),
                pl.read_parquet(curated / "poll_results.parquet"),
                election,
            )
            with np.load(reference / "cases" / name / "fit/draws.npz") as raw:
                draws = {k: raw[k] for k in raw.files}
            for variant in ds.variants(protocol):
                settings = ds.Settings.from_protocol(protocol, variant)
                start = time.perf_counter()
                decision = ds.build_case(
                    cr, registry, election, horizon, asof, settings, draws, current, previous,
                    spec["n_draws"], spec["seed"],
                )
                decision["variant"] = variant
                decision["build_seconds"] = time.perf_counter() - start
                write_json(out / "decisions" / name / f"{variant}.json", decision)
                a = decision["audit"]
                summary.append(
                    {"case": name, "variant": variant, "source": "decisions"}
                    | {k: a[k] for k in ("n_signatures", "n_top", "top_coverage", "n_strata",
                                         "n_sampled_signatures", "n_configurations",
                                         "pruned_mass_bound",
                                         "max_presence_shift_from_sampling",
                                         "expected_candidates")}
                    | {"seconds": round(decision["build_seconds"], 1)}
                )
                print(f"Construit : {name}/{variant} {summary[-1]}", flush=True)
    write_json(manifest, {"inputs": inputs, "summary": summary,
                          "created_at": dt.datetime.now(dt.UTC).isoformat()})


# --------------------------------------------------------------------------- run


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
    rows = np.arange(n)
    if (shares[rows, qualified[:, 0]] <= 0).any() or (shares[rows, qualified[:, 1]] <= 0).any():
        raise ValueError("qualifié absent")
    second = sim["share_second"]
    if not (np.isfinite(second) & (second >= 0) & (second <= 100)).all():
        raise ValueError("part de second tour invalide")


def run(args) -> None:
    reference, out = args.reference.resolve(), args.out.resolve()
    check_separate(reference, out)
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
    decisions_dir = out / "decisions"
    if not (decisions_dir / "manifest.json").exists():
        raise ValueError("lancer d'abord `build`")
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
        verify_inventory(decisions_dir, manifest["decisions"])
        verify_inventory(reference, manifest["reference_files"])
    else:
        manifest = {
            "id": protocol["id"],
            "protocol": protocol,
            "inputs": input_hashes,
            "decisions": inventory(decisions_dir),
            "reference_files": inventory(reference),
            "reference_commit": record["commit"],
            "versions": record["versions"],
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        (out / "source").mkdir(parents=True, exist_ok=True)
        for name, path in inputs.items():
            shutil.copy2(path, out / "source" / name)
        write_json(manifest_file, manifest)
    loader = importlib.util.spec_from_file_location(
        "decision_variant", out / "source/implementation.py"
    )
    impl = importlib.util.module_from_spec(loader)
    sys.modules["decision_variant"] = impl  # requis par dataclasses (annotations différées)
    loader.loader.exec_module(impl)
    variant_names = impl.variants(protocol)
    selected = set(args.cases.split(",")) if args.cases else None
    for election in spec["elections"]:
        for horizon in spec["horizons"]:
            name = f"{election}-{horizon:03d}"
            if selected is not None and name not in selected:
                continue
            original, folder = reference / "cases" / name, out / "cases" / name
            if not frozen.complete_score(original):
                raise ValueError(f"cas de référence incomplet : {name}")
            case = json.loads((original / "case.json").read_text())
            frozen.check_case(original, case)
            if not json.loads((original / "fit/quality.json").read_text())["acceptable"]:
                raise ValueError(f"ajustement non conforme : {name}")
            if (folder / "complete.json").exists():
                hashes = json.loads((folder / "complete.json").read_text())
                actual = inventory(folder)
                actual.pop("complete.json")
                if hashes != actual:
                    raise ValueError(f"sortie d'étude modifiée : {name}")
                print(f"Reprise vérifiée : {name}", flush=True)
                continue
            if folder.exists():
                shutil.rmtree(folder)  # cas interrompu : on le refait entièrement
            folder.mkdir(parents=True)
            asof = dt.date.fromisoformat(case["asof"])
            polls = load_polls(election, asof, curated=original / "curated")
            errors, errors2, transfers, turnout, excluded, calibration = frozen.calibration(
                asof, spec
            )
            canonical = json.loads(json.dumps(calibration, default=str))
            if canonical != json.loads((original / "calibration.json").read_text()):
                raise ValueError(f"calibration différente : {name}")
            older = dict(polls.group_by("candidate_id").agg(pl.col("share").median()).iter_rows())

            def model2(original=original, transfers=transfers, errors2=errors2,
                       excluded=excluded):
                return SecondRoundModel(
                    curated=original / "curated",
                    transfers=transfers,
                    errors=errors2,
                    exclude=excluded,
                    **spec["second_round"],
                )

            with np.load(original / "fit/draws.npz") as draws:
                # Rejeu : la fonction de la variante, sans prior, doit reproduire le gel.
                replay = {}
                for method in ("reference", "mean14d"):
                    result, _ = impl.simulate_chain(
                        case, draws, errors, model2(), turnout, spec, polls, method
                    )
                    with np.load(original / f"draws_{method}.npz") as saved:
                        differences = {}
                        for key, value in result.items():
                            if value.dtype.kind in "fc":
                                delta = float(np.max(np.abs(value - saved[key])))
                                if not np.allclose(value, saved[key], rtol=1e-12, atol=1e-12):
                                    raise ValueError(f"rejeu différent : {name}/{method}/{key}")
                                differences[key] = delta
                            elif not np.array_equal(value, saved[key]):
                                raise ValueError(f"rejeu différent : {name}/{method}/{key}")
                        replay[method] = differences
                write_json(folder / "replay.json", replay)
                for variant in variant_names:
                    decision = json.loads((decisions_dir / name / f"{variant}.json").read_text())
                    dest = folder / variant
                    dest.mkdir(parents=True)
                    metrics_rows, outcomes, cells_rows = [], [], []
                    expanded_ref = None
                    if decision["source"] == "official_list":
                        configurations = [{"candidates": decision["candidates"], "probability": 1.0}]
                        if configurations != case["configurations"]:
                            raise ValueError(f"liste officielle différente du gel : {name}")
                    for method in ("reference", "mean14d"):
                        if decision["source"] == "official_list":
                            shutil.copy2(original / f"draws_{method}.npz",
                                         dest / f"draws_{method}.npz")
                            with np.load(dest / f"draws_{method}.npz") as saved:
                                final = dict(saved)
                            cells = (
                                pl.read_parquet(original / "cells.parquet")
                                .filter(pl.col("method") == method)
                                .drop("election", "horizon", "method")
                                .to_dicts()
                            )
                            expanded = configurations
                        else:
                            settings = impl.Settings(**decision["settings"])
                            sim_configs, priors = impl.simulation_configurations(decision, variant)
                            impl.augment_registry(registry, election, priors)
                            for row in decision["persons"]:
                                if row["type"] is None:
                                    continue
                                m14 = decision["types"][row["type"]]["centre_mean14d"]
                                if abs(older.get(row["person"], 1.0) - m14) > 1e-9:
                                    raise ValueError(f"centre mean14d incohérent : {row['person']}")
                            new_case = case | {"configurations": sim_configs}
                            result, cells = impl.simulate_chain(
                                new_case, draws, errors, model2(), turnout, spec, polls,
                                method, priors, settings,
                            )
                            rng_names = np.random.default_rng([spec["seed"], 2])
                            final, expanded = impl.assign_names(
                                decision, variant, result, sim_configs, rng_names
                            )
                            if expanded_ref is not None and expanded != expanded_ref:
                                raise ValueError("configurations développées différentes")
                            expanded_ref = expanded
                            np.savez_compressed(dest / f"draws_{method}.npz", **final)
                        validate_simulation(final, spec["n_draws"])
                        summary, rows = frozen.score_chain(final, election, registry)
                        summary["unforecast_share"] = float(summary["unforecast_share"])
                        key = {"election": election, "horizon": horizon, "method": method,
                               "variant": variant}
                        metrics_rows.append(key | summary | {"fit_acceptable": True})
                        outcomes.extend(key | r for r in rows)
                        cells_rows.extend(key | r for r in cells)
                    write_json(
                        dest / "case.json",
                        case
                        | {
                            "configurations": expanded,
                            "scenario_rule": {"source": decision["source"], "variant": variant},
                            "reused_reference": decision["source"] == "official_list",
                            "presence": impl.presence(expanded),
                            "reference_presence": impl.presence(case["configurations"]),
                            "n_signatures_simulated": len(decision.get("configurations", [])),
                        },
                    )
                    pl.DataFrame(metrics_rows).write_parquet(dest / "metrics.parquet")
                    pl.DataFrame(outcomes).write_parquet(dest / "outcomes.parquet")
                    pl.DataFrame(cells_rows, infer_schema_length=None).write_parquet(
                        dest / "cells.parquet"
                    )
                    print(f"Terminé : {name}/{variant} ({len(expanded)} configurations)",
                          flush=True)
            write_json(folder / "complete.json", inventory(folder))
    verify_inventory(reference, manifest["reference_files"])
    verify_inventory(decisions_dir, manifest["decisions"])
    if selected is None:
        print(f"Étude complète : {out} ; référence intacte", flush=True)
    else:
        print(f"Cas terminés : {sorted(selected)} ; référence intacte", flush=True)


# --------------------------------------------------------------------------- report

# Colonnes d'écarts publiées (signe : +1 si plus haut = mieux).
STAGE = [
    ("log_score_winner", 1, "log victoire"),
    ("log_score_pair", 1, "log paire"),
    ("crps_global", -1, "CRPS"),
    ("coverage90_global", 1, "couverture 90"),
    ("brier_presence_sum", -1, "Brier présence"),
    ("log_presence_winner", 1, "log P(W présent)"),
    ("log_presence_pair", 1, "log P(paire présente)"),
    ("log_qualify_given_presence_winner", 1, "log P(W qualifié | présent)"),
    ("log_pair_given_presence", 1, "log P(paire | présence)"),
    ("log_win_given_qualify_winner", 1, "log P(W gagne | qualifié)"),
]
DOWNSTREAM = [
    ("log_pair_given_presence", 1),
    ("brier_qualify_given_presence", -1),
    ("crps_conditional", -1),
    ("coverage90_conditional", 1),
]
BATCHES = 20


def batch_errors(folder, method, outcome, conditional, seed):
    """Erreurs Monte-Carlo par lots des métriques aval sans formule binomiale simple."""
    import numpy as np

    from vvs.eval import metrics as mt
    from vvs.eval import stage_diagnostics as sd

    sim, configs = sd.load_case_draws(folder, method)
    present = sd.presence_matrix(sim, configs)
    names = [str(c) for c in sim["candidates"]]
    index = {c: j for j, c in enumerate(names)}
    shares = np.asarray(sim["shares"], dtype=float)
    qual = np.sort(np.array(names)[np.asarray(sim["qualified"])], axis=1)
    n = shares.shape[0]
    finals = set(outcome.finalists)
    rng = np.random.default_rng(seed)
    values = {"brier_qualify_given_presence": [], "crps_conditional": [],
              "coverage90_conditional": []}
    for rows in np.array_split(rng.permutation(n), BATCHES):
        brier, crps, cover = [], [], []
        for c in sorted(conditional):
            mask = present[rows, index[c]]
            if not mask.any():
                continue
            r = rows[mask]
            q = (qual[r] == c).any(axis=1).mean()
            brier.append((q - (c in finals)) ** 2)
            x = shares[r, index[c]]
            crps.append(float(mt.crps(x, outcome.shares[c])))
            cover.append(float(mt.covered(x, outcome.shares[c], 0.9)))
        values["brier_qualify_given_presence"].append(np.mean(brier) if brier else np.nan)
        values["crps_conditional"].append(np.mean(crps) if crps else np.nan)
        values["coverage90_conditional"].append(np.mean(cover) if cover else np.nan)
    return {k: float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())) for k, v in values.items()}


def event_se(summary, event):
    from vvs.eval import stage_diagnostics as sd

    n = summary[f"n_{event}"]
    p = summary[f"p_{event}"]
    if not n:
        return None
    return sd.mc_se(round(p * n), n)


def fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        return f"{x:+.{digits}f}".replace("+0.000", "0").replace("-0.000", "0") if digits else f"{x:.0f}"
    return str(x)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(v) if not isinstance(v, str) else v for v in r) + " |" for r in rows]
    return "\n".join(out)


def report(args) -> None:
    import polars as pl

    from vvs.eval import stage_diagnostics as sd
    from vvs.eval.consolidation import complete_score, verify
    from vvs.registry import default_registry

    reference, out = args.reference.resolve(), args.out.resolve()
    registry = default_registry()
    spec = verify(reference)["spec"]
    study = json.loads((out / "study.json").read_text())
    protocol = study["protocol"]
    main = protocol["main_variant"]
    names = [n for _, _, n in sd.case_names(spec)]
    for name in names:
        if not (out / "cases" / name / "complete.json").exists():
            raise ValueError(f"cas incomplet : {name}")
        if not complete_score(reference / "cases" / name):
            raise ValueError(f"référence incomplète : {name}")
    verify_inventory(reference, study["reference_files"])
    verify_inventory(out / "decisions", study["decisions"])

    # 1. Scores globaux du gel et de la variante, diagnostics par étape à univers commun.
    frozen_metrics = pl.concat(
        [pl.read_parquet(reference / "cases" / n / "metrics.parquet") for n in names],
        how="diagonal_relaxed",
    ).with_columns(pl.lit("reference_frozen").alias("variant"))
    variant_metrics = pl.concat(
        [pl.read_parquet(p) for n in names for p in sorted((out / "cases" / n).glob("*/metrics.parquet"))],
        how="diagonal_relaxed",
    )
    metrics = pl.concat([frozen_metrics, variant_metrics], how="diagonal_relaxed")
    metrics.write_csv(out / "metrics.csv", float_precision=10)
    summaries, deltas = sd.compare_study(out, reference, registry)
    summaries.write_csv(out / "stage_summaries.csv", float_precision=10)

    # 2. Erreurs Monte-Carlo (événements binaires, puis lots pour les métriques aval).
    se_rows = []
    for election, horizon, name in sd.case_names(spec):
        outcome = sd.election_outcome(election, registry)
        for method in sd.METHODS:
            ref_case = sd.load_case_draws(reference / "cases" / name, method)
            ref_support = sd.support(*ref_case)
            for variant in sorted(p.name for p in (out / "cases" / name).iterdir() if p.is_dir()):
                folder = out / "cases" / name / variant
                var_support = sd.support(*sd.load_case_draws(folder, method))
                conditional = set(outcome.shares) & ref_support & var_support
                pair = summaries.filter(
                    (pl.col("election") == election) & (pl.col("horizon") == horizon)
                    & (pl.col("method") == method)
                )
                s_var = pair.filter(pl.col("variant") == variant).row(0, named=True)
                s_ref = pair.filter((pl.col("variant") == "reference_frozen")
                                    & (pl.col("against") == variant)).row(0, named=True)
                row = {"election": election, "horizon": horizon, "method": method,
                       "variant": variant}
                for side, s, fold in (("variant", s_var, folder),
                                      ("reference", s_ref, reference / "cases" / name)):
                    row[f"se_log_pair_given_presence_{side}"] = event_se(s, "pair_given_presence")
                    row[f"se_log_win_given_pair_{side}"] = event_se(s, "win_given_pair")
                    k = s["p_presence_winner"] * s["n_draws"]
                    p = s["phantom_qualify_given_presence_winner"]
                    row[f"se_phantom_qualify_{side}"] = (
                        math.sqrt(p * (1 - p) / k) if p is not None and k else None
                    )
                    for key, value in batch_errors(fold, method, outcome, conditional,
                                                   spec["seed"]).items():
                        row[f"se_{key}_{side}"] = value
                se_rows.append(row)
    errors = pl.DataFrame(se_rows, infer_schema_length=None)
    errors.write_csv(out / "mc_errors.csv", float_precision=10)
    for column, _ in DOWNSTREAM + [("log_win_given_pair", 1), ("phantom_qualify", 1)]:
        a, b = f"se_{column}_variant", f"se_{column}_reference"
        errors = errors.with_columns(
            (pl.col(a).cast(float) ** 2 + pl.col(b).cast(float) ** 2).sqrt().alias(f"se_delta_{column}")
        )
    phantom = summaries.select(
        "election", "horizon", "method", "variant", "against",
        "phantom_qualify_given_presence_winner", "phantom_share_given_presence_winner",
    )
    ref_ph = phantom.filter(pl.col("variant") == "reference_frozen").select(
        "election", "horizon", "method", pl.col("against").alias("variant"),
        pl.col("phantom_qualify_given_presence_winner").alias("ref_phantom_qualify"),
        pl.col("phantom_share_given_presence_winner").alias("ref_phantom_share"),
    )
    var_ph = phantom.filter(pl.col("variant") != "reference_frozen").select(
        "election", "horizon", "method", "variant",
        pl.col("phantom_qualify_given_presence_winner").alias("var_phantom_qualify"),
        pl.col("phantom_share_given_presence_winner").alias("var_phantom_share"),
    )
    keys = ["election", "horizon", "method", "variant"]
    deltas = (
        deltas.join(errors, on=keys, how="left")
        .join(ref_ph, on=keys, how="left")
        .join(var_ph, on=keys, how="left")
        .with_columns(
            (pl.col("var_phantom_qualify") - pl.col("ref_phantom_qualify")).alias(
                "delta_phantom_qualify"
            )
        )
    )
    deltas.write_csv(out / "stage_deltas.csv", float_precision=10)

    # 3. Critères déclarés (agrégateur bayésien, variante principale).
    d = deltas.filter((pl.col("method") == "reference") & (pl.col("variant") == main))
    criteria = {}
    long_h = (180, 90)
    presence = {}
    for h in long_h:
        sub = d.filter(pl.col("horizon") == h)
        presence[h] = {c: int(sub[f"better_{c}"].sum()) for c in
                       ("log_presence_winner", "log_presence_pair", "brier_presence_sum")}
    criteria["1_presence"] = {
        "detail": presence,
        "pass": all(v >= 2 for h in long_h for v in presence[h].values()),
    }
    down = {}
    for h in long_h:
        sub = d.filter(pl.col("horizon") == h)
        down[h] = {}
        for column, sign in DOWNSTREAM:
            vals = sub[f"delta_{column}"].cast(float).to_list()
            ses = sub[f"se_delta_{column}"].cast(float).to_list()
            ok = [(v, e) for v, e in zip(vals, ses) if v is not None and e is not None
                  and not math.isnan(v) and not math.isnan(e)]
            if not ok:
                down[h][column] = {"mean": None, "se": None, "degraded": None}
                continue
            mean = sum(v for v, _ in ok) / len(ok)
            se = math.sqrt(sum(e**2 for _, e in ok)) / len(ok)
            down[h][column] = {"mean": mean, "se": se, "n": len(ok),
                               "degraded": bool(sign * mean < -2 * se)}
    criteria["2_downstream"] = {
        "detail": down,
        "pass": not any(v["degraded"] for h in long_h for v in down[h].values()),
    }
    wp = []
    for r in d.iter_rows(named=True):
        v, e = r["delta_log_win_given_pair"], r["se_delta_log_win_given_pair"]
        if v is not None and e is not None:
            wp.append({"case": f"{r['election']}-{r['horizon']}", "delta": v, "se": e,
                       "ok": abs(v) <= 2 * e})
    criteria["3_win_given_pair"] = {"detail": wp, "pass": all(x["ok"] for x in wp)}
    ph = []
    for r in d.filter(pl.col("horizon").is_in(long_h)).iter_rows(named=True):
        v, e = r["delta_phantom_qualify"], r["se_delta_phantom_qualify"]
        ph.append({"case": f"{r['election']}-{r['horizon']}", "delta": v, "se": e,
                   "ok": v is None or e is None or v <= 2 * e})
    criteria["4_phantoms"] = {"detail": ph, "pass": all(x["ok"] for x in ph)}
    late = deltas.filter(pl.col("horizon").is_in([30, 7]))
    delta_cols = [c for c in late.columns if c.startswith("delta_") and c != "delta_phantom_qualify"]
    late_max = max(
        (abs(x) for c in delta_cols for x in late[c].cast(float).to_list() if x is not None),
        default=0.0,
    )
    gm = metrics.filter(pl.col("horizon").is_in([30, 7]))
    late_global = 0.0
    for col in ("log_score_winner", "log_score_pair", "crps", "coverage90", "brier_win",
                "brier_qualify"):
        piv = gm.pivot(on="variant", index=["election", "horizon", "method"], values=col)
        for v in [c for c in piv.columns if c not in ("election", "horizon", "method",
                                                      "reference_frozen")]:
            late_global = max(late_global, float((piv[v] - piv["reference_frozen"]).abs().max()))
    criteria["5_late_zero"] = {"max_abs_stage_delta": late_max,
                               "max_abs_global_delta": late_global,
                               "pass": late_max == 0 and late_global == 0}
    glob = {}
    for h in long_h:
        sub = d.filter(pl.col("horizon") == h)
        glob[h] = {
            "mean_delta_log_pair": float(sub["delta_log_score_pair"].mean()),
            "mean_delta_crps": float(sub["delta_crps_global"].mean()),
            "mean_delta_coverage90": float(sub["delta_coverage90_global"].mean()),
            "n_better_log_pair": int(sub["better_log_score_pair"].sum()),
            "n_better_crps": int(sub["better_crps_global"].sum()),
        }
        g = glob[h]
        g["pass"] = (g["mean_delta_log_pair"] > 0 and g["mean_delta_crps"] < 0
                     and g["n_better_log_pair"] >= 2 and g["n_better_crps"] >= 2
                     and g["mean_delta_coverage90"] >= 0)
    criteria["6_global_docs07"] = {"detail": glob, "pass": all(glob[h]["pass"] for h in long_h)
                                   and criteria["5_late_zero"]["pass"]}
    criteria["promotion"] = all(v["pass"] for k, v in criteria.items() if k[0].isdigit())
    write_json(out / "criteria.json", criteria)

    # 4. Présence par candidat et fantômes (variante et référence, agrégateur bayésien).
    presence_rows = []
    for election, horizon, name in sd.case_names(spec):
        outcome = sd.election_outcome(election, registry)
        ref_case = json.loads((reference / "cases" / name / "case.json").read_text())
        ref_presence = {}
        for cfg in ref_case["configurations"]:
            for c in cfg["candidates"]:
                ref_presence[c] = ref_presence.get(c, 0.0) + cfg["probability"]
        for variant in sorted(p.name for p in (out / "cases" / name).iterdir() if p.is_dir()):
            case = json.loads((out / "cases" / name / variant / "case.json").read_text())
            var_out = pl.read_parquet(out / "cases" / name / variant / "outcomes.parquet")
            ref_out = pl.read_parquet(reference / "cases" / name / "outcomes.parquet")
            people = sorted(set(case["presence"]) | set(ref_presence) | set(outcome.shares))
            for method in sd.METHODS:
                vo = {r["candidate_id"]: r for r in var_out.filter(pl.col("method") == method).iter_rows(named=True)}
                ro = {r["candidate_id"]: r for r in ref_out.filter(pl.col("method") == method).iter_rows(named=True)}
                for c in people:
                    presence_rows.append({
                        "election": election, "horizon": horizon, "variant": variant,
                        "method": method, "candidate_id": c,
                        "official": c in outcome.shares, "actual_share": outcome.shares.get(c),
                        "presence_reference": ref_presence.get(c, 0.0),
                        "presence_variant": case["presence"].get(c, 0.0),
                        "p_qualify_reference": ro.get(c, {}).get("p_qualify", 0.0),
                        "p_qualify_variant": vo.get(c, {}).get("p_qualify", 0.0),
                        "p_win_reference": ro.get(c, {}).get("p_win", 0.0),
                        "p_win_variant": vo.get(c, {}).get("p_win", 0.0),
                    })
    presence_df = pl.DataFrame(presence_rows, infer_schema_length=None)
    presence_df.write_csv(out / "presence.csv", float_precision=8)

    # 5. Audit de construction.
    audit_rows, person_rows = [], []
    for name in names:
        for f in sorted((out / "decisions" / name).glob("*.json")):
            dec = json.loads(f.read_text())
            if dec["source"] != "decisions":
                audit_rows.append({"case": name, "variant": f.stem, "source": "official_list"})
                continue
            a = dec["audit"]
            outcome = sd.election_outcome(dec["election"], registry)
            exact = {r["person"]: r["presence_exact"] for r in dec["persons"] if r["active"]}
            phantoms = {p: v for p, v in exact.items() if p not in outcome.shares}
            audit_rows.append({
                "case": name, "variant": f.stem, "source": "decisions",
                "n_active": sum(r["active"] for r in dec["persons"]),
                "n_without_support": len(a["without_estimated_support"]),
                "n_processes_pending": len(a["pools"]),
                "n_signatures": a["n_signatures"], "n_top": a["n_top"],
                "top_coverage": a["top_coverage"], "n_strata": a["n_strata"],
                "n_simulated": a["n_configurations"],
                "max_presence_shift_from_sampling": a["max_presence_shift_from_sampling"],
                "pruned_mass_bound": a["pruned_mass_bound"],
                "log10_constraint_probability": a["log10_constraint_probability"],
                "expected_candidates": a["expected_candidates"],
                "expected_phantoms": sum(phantoms.values()),
                "expected_officials": sum(v for p, v in exact.items() if p in outcome.shares),
                **{f"mass_{k}": v["mass_exact"] for k, v in a["mass_by_state"].items()},
                **{f"n_{k}": v["n"] for k, v in a["mass_by_state"].items()},
            })
            for r in dec["persons"]:
                person_rows.append({"case": name, "variant": f.stem,
                                    "final_official": r["person"] in outcome.shares} | {
                    k: v for k, v in r.items()})
    audit_df = pl.DataFrame(audit_rows, infer_schema_length=None)
    audit_df.write_csv(out / "audit_cases.csv", float_precision=8)
    pl.DataFrame(person_rows, infer_schema_length=None).write_csv(
        out / "audit_persons.csv", float_precision=8
    )

    # 6. Rapport.
    write_markdown(out, spec, protocol, metrics, deltas, criteria, presence_df, audit_df,
                   args.previous)
    print(f"Rapport écrit : {out / 'report.md'}", flush=True)


def write_markdown(out, spec, protocol, metrics, deltas, criteria, presence, audit, previous):
    import polars as pl

    main = protocol["main_variant"]
    lines = [
        "# Scénarios historiques par décision de candidature (lot 2) — bilan",
        "",
        (f"Étude `{protocol['id']}`, référence `{protocol['reference']}` (gel a89d304). "
         "Protocole déclaré avant calcul : `data/registry/historical_decisions_2026-10-23.yaml` "
         "et docstring de `back/src/vvs/eval/decision_scenarios.py` (amendement 1 avant toute "
         "simulation). Douze cas, 40 000 tirages, graine du gel ; écarts variante − référence "
         "figée, univers communs de `stage_diagnostics.compare_study`. Log-scores et couverture : "
         "plus haut = mieux ; CRPS et Brier : plus bas = mieux. Couverture en points."),
        "",
    ]
    cols = STAGE
    for method, title in (("reference", "Agrégateur bayésien"), ("mean14d", "Témoin mean14d")):
        d = deltas.filter((pl.col("method") == method) & (pl.col("variant") == main)).sort(
            "election", pl.col("horizon"), descending=[False, True])
        lines += [f"## {title} : écarts par élection et horizon (variante `{main}`)", ""]
        rows = []
        for r in d.iter_rows(named=True):
            vals = []
            for c, _, _ in cols:
                v = r[f"delta_{c}"]
                vals.append(None if v is None else (100 * v if c == "coverage90_global" else v))
            rows.append([str(r["election"]), str(r["horizon"])] + vals)
        lines += [md_table(["Élection", "Horizon"] + [lab for _, _, lab in cols], rows), ""]
        means = d.group_by("horizon").agg(
            *[pl.col(f"delta_{c}").cast(float).mean() for c, _, _ in cols]
        ).sort("horizon", descending=True)
        rows = [[str(r["horizon"])] + [
            None if r[f"delta_{c}"] is None else (100 * r[f"delta_{c}"] if c == "coverage90_global" else r[f"delta_{c}"])
            for c, _, _ in cols] for r in means.iter_rows(named=True)]
        lines += ["Moyennes à poids égal par élection :", "",
                  md_table(["Horizon"] + [lab for _, _, lab in cols], rows), ""]
    # Voix hors support.
    m = metrics.filter(pl.col("method") == "reference")
    piv = m.pivot(on="variant", index=["election", "horizon"], values="unforecast_share").sort(
        "election", pl.col("horizon"), descending=[False, True])
    lines += ["## Voix finales hors support (agrégateur bayésien)", "",
              md_table(["Élection", "Horizon", "Référence", "Variante"],
                       [[str(r["election"]), str(r["horizon"]),
                         f"{r['reference_frozen']:.2f} %", f"{r[main]:.2f} %"]
                        for r in piv.iter_rows(named=True)]), ""]
    # Critères.
    c = criteria
    lines += ["## Verdict selon les critères déclarés", ""]
    pres = c["1_presence"]["detail"]
    lines.append(
        f"1. Présence (≥ 2 élections sur 3 meilleures à 180 et 90 j) : "
        f"{'satisfait' if c['1_presence']['pass'] else 'non satisfait'} — " + "; ".join(
            f"{h} j : W {v['log_presence_winner']}/3, paire {v['log_presence_pair']}/3, "
            f"Brier {v['brier_presence_sum']}/3" for h, v in pres.items()))
    down = c["2_downstream"]["detail"]
    parts = []
    for h, v in down.items():
        for col, x in v.items():
            if x["mean"] is not None:
                parts.append(f"{h} j {col} {x['mean']:+.4f} ± {x['se']:.4f}"
                             + (" (dégradé)" if x["degraded"] else ""))
    lines.append(f"2. Aval sans dégradation au-delà de 2 erreurs MC : "
                 f"{'satisfait' if c['2_downstream']['pass'] else 'non satisfait'} — " + "; ".join(parts))
    bad = [x for x in c["3_win_given_pair"]["detail"] if not x["ok"]]
    lines.append(f"3. P(W | paire réelle) inchangée : "
                 f"{'satisfait' if c['3_win_given_pair']['pass'] else 'non satisfait'}"
                 + ("" if not bad else " — " + "; ".join(f"{x['case']} {x['delta']:+.4f} ± {x['se']:.4f}" for x in bad)))
    bad = [x for x in c["4_phantoms"]["detail"] if not x["ok"]]
    lines.append(f"4. Probabilité qu'un fantôme se qualifie non croissante : "
                 f"{'satisfait' if c['4_phantoms']['pass'] else 'non satisfait'}"
                 + ("" if not bad else " — " + "; ".join(f"{x['case']} {x['delta']:+.3f} ± {x['se']:.3f}" for x in bad)))
    lines.append(f"5. Écarts nuls à 30 et 7 j : {'satisfait' if c['5_late_zero']['pass'] else 'non satisfait'} "
                 f"(écart absolu maximal {c['5_late_zero']['max_abs_stage_delta']:.3g} par étape, "
                 f"{c['5_late_zero']['max_abs_global_delta']:.3g} au score global)")
    g = c["6_global_docs07"]["detail"]
    lines.append("6. Critère global de docs/07 : "
                 f"{'satisfait' if c['6_global_docs07']['pass'] else 'non satisfait'} — " + "; ".join(
                     f"{h} j : Δ log paire {v['mean_delta_log_pair']:+.4f} ({v['n_better_log_pair']}/3), "
                     f"Δ CRPS {v['mean_delta_crps']:+.4f} ({v['n_better_crps']}/3), "
                     f"Δ couverture {100 * v['mean_delta_coverage90']:+.2f} pt" for h, v in g.items()))
    lines += ["", f"**Promotion : {'oui' if c['promotion'] else 'non'}.**", ""]
    # Sensibilité.
    lines += ["## Sensibilité annoncée (agrégateur bayésien, moyennes par horizon)", ""]
    sens_cols = ["log_score_winner", "log_score_pair", "crps_global", "log_presence_winner",
                 "log_presence_pair", "brier_presence_sum"]
    d = deltas.filter((pl.col("method") == "reference") & pl.col("horizon").is_in([180, 90]))
    means = d.group_by("variant", "horizon").agg(
        *[pl.col(f"delta_{x}").cast(float).mean() for x in sens_cols]).sort(
        "variant", pl.col("horizon"), descending=[False, True])
    lines += [md_table(["Variante", "Horizon"] + sens_cols,
                       [[r["variant"], str(r["horizon"])] + [r[f"delta_{x}"] for x in sens_cols]
                        for r in means.iter_rows(named=True)]), ""]
    # 2017 à 180 jours.
    p = presence.filter((pl.col("election") == 2017) & (pl.col("horizon") == 180)
                        & (pl.col("variant") == main) & (pl.col("method") == "reference"))
    focus = ["macron_emmanuel", "le_pen_marine", "fillon_francois", "melenchon_jean_luc",
             "hamon_benoit", "juppe_alain", "sarkozy_nicolas", "hollande_francois",
             "valls_manuel", "bayrou_francois", "asselineau_francois", "lassalle_jean"]
    rows = []
    for r in p.filter(pl.col("candidate_id").is_in(focus)).iter_rows(named=True):
        rows.append([r["candidate_id"], f"{r['presence_reference']:.3f}", f"{r['presence_variant']:.3f}",
                     f"{r['p_qualify_reference']:.3f}", f"{r['p_qualify_variant']:.3f}",
                     f"{r['p_win_reference']:.3f}", f"{r['p_win_variant']:.3f}"])
    lines += ["## 2017 à 180 jours : présence, qualification, victoire", "",
              md_table(["Candidat", "Présence réf.", "Présence var.", "Qualif. réf.",
                        "Qualif. var.", "Victoire réf.", "Victoire var."], rows), ""]
    ph = deltas.filter((pl.col("election") == 2017) & (pl.col("horizon") == 180)
                       & (pl.col("method") == "reference") & (pl.col("variant") == main)).row(0, named=True)
    au = audit.filter((pl.col("case") == "2017-180") & (pl.col("variant") == main)).row(0, named=True)
    lines += [
        (f"Fantômes (W présent) : part moyenne {ph['ref_phantom_share']:.2f} → "
         f"{ph['var_phantom_share']:.2f} points ; probabilité qu'un fantôme se qualifie "
         f"{ph['ref_phantom_qualify']:.3f} → {ph['var_phantom_qualify']:.3f}. Nombre attendu de "
         f"candidats {au['expected_candidates']:.2f}, dont fantômes {au['expected_phantoms']:.2f}."),
        "",
    ]
    # Audit.
    lines += ["## Audit de construction (variante principale)", ""]
    a = audit.filter((pl.col("variant") == main) & (pl.col("source") == "decisions")).sort("case")
    lines += [md_table(
        ["Cas", "Actifs", "Sans soutien estimé", "Processus en attente", "Signatures",
         "Couverture top", "Simulées", "Candidats attendus", "Fantômes attendus",
         "log10 P(plafonds)"],
        [[r["case"], str(r["n_active"]), str(r["n_without_support"]), str(r["n_processes_pending"]),
          str(r["n_signatures"]), f"{r['top_coverage']:.3f}", str(r["n_simulated"]),
          f"{r['expected_candidates']:.2f}", f"{r['expected_phantoms']:.2f}",
          f"{r['log10_constraint_probability']:.1f}"] for r in a.iter_rows(named=True)]), ""]
    # Comparaison avec docs/11.
    if previous is not None and (previous / "variants/deltas.csv").exists():
        prev = pl.read_csv(previous / "variants/deltas.csv").filter(
            (pl.col("method") == "reference") & (pl.col("variant") == "memory_events")
            & pl.col("horizon").is_in([180, 90]))
        cur = deltas.filter((pl.col("method") == "reference") & (pl.col("variant") == main)
                            & pl.col("horizon").is_in([180, 90]))
        comp_cols = ["log_score_winner", "log_score_pair", "crps_global", "log_presence_winner",
                     "log_presence_pair", "brier_presence_sum", "log_pair_given_presence"]
        rows = []
        for label, frame in (("docs/11 memory_events", prev), (f"lot 2 {main}", cur)):
            mm = frame.group_by("horizon").agg(
                *[pl.col(f"delta_{x}").cast(float).mean() for x in comp_cols]).sort(
                "horizon", descending=True)
            rows += [[label, str(r["horizon"])] + [r[f"delta_{x}"] for x in comp_cols]
                     for r in mm.iter_rows(named=True)]
        lines += ["## Comparaison avec docs/11 (mémoire + événements, agrégateur bayésien)", "",
                  md_table(["Variante", "Horizon"] + comp_cols, rows), ""]
    (out / "report_tables.md").write_text("\n".join(lines) + "\n")
    if not (out / "report.md").exists():
        (out / "report.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["build", "run", "report"])
    for name in ("reference", "out", "protocol", "implementation", "pin", "previous"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--cases", help="run : cas à traiter (ex. 2017-180,2022-180)")
    args = parser.parse_args()
    if args.step == "build":
        build(args)
    elif args.step == "run":
        run(args)
    else:
        report(args)
