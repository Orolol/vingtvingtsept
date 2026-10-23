"""Contrôles d'intégrité lancés sur chaque PR (.github/workflows/ci.yml, docs/13).

- verrou de la référence : `data/registry/reference_2026-10-22.lock.json` reprend la
  configuration versionnée et les empreintes des sources au commit épinglé (historique Git
  complet requis) ; le gel local `runs/2026-10-22-consolidation/` est contrôlé s'il existe ;
- manifest du site : export réel, version de schéma, dates du scrutin du registre, entrée
  d'historique du jour présente et indexée ;
- archives : `history/<date>.json` du site et `data/prospective/<date>/<nom>/` présents dans
  la base ne sont jamais modifiés ; empreintes des archives prospectives.

Entrée : `uv run vvs check-integrity [--base <réf. Git>]`, code de sortie 1 au moindre écart.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from vvs import paths
from vvs.export import prospective

LOCK = Path("data/registry/reference_2026-10-22.lock.json")
SITE_DATA = Path("front/static/data")
HISTORY = SITE_DATA / "history"
POLLS_HASH_RE = re.compile(r"^[0-9a-f]{16}$")


def _git_bytes(root: Path, *args: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    return r.stdout if r.returncode == 0 else None


def check_lock(root: Path, lock: Path = LOCK) -> list[str]:
    problems = []
    pinned = json.loads((root / lock).read_text())
    spec_file = root / str(lock).replace(".lock.json", ".yaml")
    spec = yaml.safe_load(spec_file.read_text())
    if pinned.get("spec") != spec:
        problems.append(f"{lock} : configuration différente de {spec_file.name}")
    if pinned.get("id") != spec.get("id"):
        problems.append(f"{lock} : identifiant différent de la configuration")
    commit = pinned.get("commit", "")
    if _git_bytes(root, "cat-file", "-e", f"{commit}^{{commit}}") is None:
        return [*problems, f"{lock} : commit {commit} absent (historique Git complet requis)"]
    for rel, sha in sorted(pinned["files"].items()):
        if not rel.startswith("source/"):
            continue  # tirages publiés : hors Git, contrôlés dans le gel local
        blob = _git_bytes(root, "show", f"{commit}:{rel.removeprefix('source/')}")
        if blob is None:
            problems.append(f"{lock} : {rel} absent du commit {commit[:7]}")
        elif hashlib.sha256(blob).hexdigest() != sha:
            problems.append(f"{lock} : {rel} diffère du commit {commit[:7]}")
    frozen = root / "runs" / "2026-10-22-consolidation" / "reference.json"
    if frozen.is_file():
        record = json.loads(frozen.read_text())
        for key in ("id", "commit", "spec", "versions", "files"):
            if record.get(key) != pinned.get(key):
                problems.append(f"gel local différent du verrou : {key}")
        bad = [rel for rel, sha in record["files"].items()
               if not (frozen.parent / rel).is_file()
               or prospective.digest(frozen.parent / rel) != sha]
        if bad:
            problems.append(f"gel local modifié ou incomplet : {bad[:5]}")
    return problems


def check_site_manifest(root: Path) -> list[str]:
    from vvs.export.export import SCHEMA_VERSION
    from vvs.registry import default_registry

    problems = []
    m = json.loads((root / SITE_DATA / "manifest.json").read_text())
    if m.get("mock"):
        problems.append("manifest.json : jeu factice publié")
    if m.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest.json : schema_version {m.get('schema_version')}")
    if not POLLS_HASH_RE.match(str(m.get("polls_hash"))):
        problems.append("manifest.json : polls_hash absent ou mal formé")
    if not isinstance(m.get("n_draws"), int) or m["n_draws"] < 10_000:
        problems.append("manifest.json : moins de 10 000 tirages")
    if m.get("asof", "") > m.get("run_date", ""):
        problems.append("manifest.json : asof postérieur à run_date")
    election = default_registry().elections[m["election"]["year"]]
    for key, number in (("first_round", 1), ("second_round", 2)):
        if m["election"][key] != election.round_(number).date.isoformat():
            problems.append(f"manifest.json : {key} différent de elections.yaml")
    entry = root / HISTORY / f"{m['run_date']}.json"
    if not entry.is_file():
        problems.append(f"historique : {entry.name} absent")
    index = json.loads((root / HISTORY / "index.json").read_text())
    if m["run_date"] not in {e["run_date"] for e in index["entries"]}:
        problems.append("historique : run du manifest absent de index.json")
    return problems


def rewritten_history(base: str, root: Path) -> list[str]:
    """Entrées `history/<date>.json` présentes dans `base` puis modifiées ou supprimées."""
    out = _git_bytes(root, "diff", "--name-status", "--no-renames", base, "--", str(HISTORY))
    if out is None:
        raise ValueError(f"référence Git introuvable : {base}")
    return sorted(
        path for status, path in (line.split("\t", 1) for line in out.decode().splitlines())
        if status[0] != "A" and not path.endswith("/index.json")
    )


def check_all(root: Path | None = None, base: str | None = None) -> list[str]:
    root = Path(root or paths.ROOT)
    problems = check_lock(root) + check_site_manifest(root)
    for folder in prospective.archives(root / prospective.PROSPECTIVE):
        rep = prospective.verify_archive(folder, root)
        problems += [f"{folder.relative_to(root)} : {p}" for p in rep["problems"]]
    if base:
        problems += [f"archive prospective réécrite : {p}"
                     for p in prospective.rewritten_archives(base, root)]
        problems += [f"historique du site réécrit : {p}" for p in rewritten_history(base, root)]
    return problems
