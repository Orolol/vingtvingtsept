"""Archives prospectives 2027 (docs/13-validation-prospective.md).

À chaque date du calendrier d'archivage, la projection publiée (`front/static/data/`) est
copiée telle quelle dans `data/prospective/<date>/<nom>/` avec les empreintes du run de
simulation (tirages compris), de l'ajustement, du code, des données et des hypothèses.
Une archive existante n'est jamais réécrite : une seconde tentative avec la même prévision
ne fait rien, une prévision différente échoue explicitement (ou est ignorée avec
`skip_existing`, mode du run quotidien). Les tirages restent hors Git : le run archivé est
copié dans `runs/prospective/<date>/<nom>/` (un second run du même jour réécrit son dossier
d'origine) et `verify_archive` recalcule ses empreintes tant que cette copie est présente.

Entrées : `uv run vvs archive-forecast [--asof] [--name] [--run]` et
`uv run vvs archive-verify [--against <réf. Git>]`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from vvs import paths

PROSPECTIVE = Path("data/prospective")
KEPT_RUNS = Path("runs/prospective")  # copies des runs archivés (tirages), hors Git
FORMAT = "vvs-prospective-1"
STABLE = "stable"  # version de comparaison stable ; les variantes portent leur propre nom
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
# Premier archivage : la projection gelée comme référence (reference-2026-10-22).
INITIAL = dt.date(2026, 10, 22)
MONTHLY_UNTIL = dt.date(2027, 2, 1)  # premier de chaque mois jusqu'à cette date incluse
WEEKLY_FROM = dt.date(2027, 2, 1)  # puis chaque semaine, même jour que le premier tour
HORIZONS = (180, 90, 30, 7)  # horizons du backtest chronologique (docs/10)
GRACE_DAYS = 2  # rattrapage d'un run manqué, jamais au-delà de la date suivante
# Fichiers du site copiés tels quels (contrat de docs/08-front.md).
SITE_FILES = ("manifest.json", "summary.json", "second_round.json", "scenarios.json")
# Fichiers dont l'identité définit « la même prévision » (le manifest porte generated_at).
IDENTITY_FILES = ("summary.json", "second_round.json", "scenarios.json")
SCENARIOS = "data/registry/scenarios_2027.yaml"
# Tables curated lues par l'ajustement et le modèle d'erreur (mêmes que le gel de docs/10).
CURATED_TABLES = (
    "polls.parquet",
    "poll_results.parquet",
    "poll_errors.parquet",
    "jw_polls.parquet",
    "poll_errors_lists.parquet",
    "polls_lists.parquet",
    "poll_results_lists.parquet",
    "results_lists_by_bloc.parquet",
)
CODE_PATHS = ("back/src/vvs", "back/pyproject.toml", "back/uv.lock")


class ArchiveError(ValueError):
    """Archivage refusé : date hors calendrier, entrée incohérente ou réécriture."""


@dataclass(frozen=True)
class Slot:
    date: dt.date
    kinds: tuple[str, ...]  # initial, monthly, horizon, weekly, eve
    label: str
    deadline: dt.date  # jour du scrutin visé : archive créée strictement avant
    grace_days: int  # délai de création (et d'asof, sauf pour le gel initial)
    asof_days: int  # écart admis entre la date et l'asof de l'export archivé

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"], d["deadline"] = self.date.isoformat(), self.deadline.isoformat()
        d["kinds"] = list(self.kinds)
        return d


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digests(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def code_digest(root: Path) -> str:
    """Empreinte du code exécutable (sources, schémas, verrou uv), même sans commit propre."""
    h = hashlib.sha256()
    for rel in CODE_PATHS:
        p = root / rel
        files = tree_digests(p).items() if p.is_dir() else [("", digest(p))]
        for name, sha in sorted(files):
            if p.is_dir() and not name.endswith((".py", ".json")):
                continue
            h.update(f"{rel}/{name}:{sha}\n".encode())
    return h.hexdigest()


# --- calendrier


def schedule(first_round: dt.date, second_round: dt.date, start: dt.date = INITIAL) -> list[Slot]:
    """Dates d'archivage fixes : gel initial, premier de chaque mois jusqu'en février 2027,
    horizons du backtest (J-180, J-90, J-30, J-7), chaque semaine ensuite (même jour que le
    premier tour), veille de chaque tour. Les doublons fusionnent leurs motifs."""
    found: dict[dt.date, list[str]] = {}
    labels: dict[dt.date, list[str]] = {}

    def add(day: dt.date, kind: str, label: str) -> None:
        if day >= start and day < second_round:
            found.setdefault(day, []).append(kind)
            labels.setdefault(day, []).append(label)

    add(start, "initial", "projection gelée (reference-2026-10-22)")
    m = dt.date(start.year, start.month, 1)
    while m <= MONTHLY_UNTIL:
        if m > start:
            add(m, "monthly", "premier du mois")
        m = dt.date(m.year + m.month // 12, m.month % 12 + 1, 1)
    for h in HORIZONS:
        add(first_round - dt.timedelta(days=h), "horizon", f"J-{h} (horizon du backtest)")
    k = 1
    while (day := first_round - dt.timedelta(days=7 * k)) >= WEEKLY_FROM:
        add(day, "weekly", "hebdomadaire")
        k += 1
    add(first_round - dt.timedelta(days=1), "eve", "veille du premier tour")
    add(second_round - dt.timedelta(days=1), "eve", "veille du second tour")
    days = sorted(found)
    slots = []
    for i, day in enumerate(days):
        deadline = first_round if day < first_round else second_round
        nxt = days[i + 1] if i + 1 < len(days) else deadline
        grace = min(GRACE_DAYS, (nxt - day).days - 1, (deadline - day).days - 1)
        grace = max(grace, 0)
        # le gel initial n'accepte que la projection du jour même, pas un export ultérieur
        asof_days = 0 if found[day] == ["initial"] else grace
        slots.append(Slot(day, tuple(dict.fromkeys(found[day])), " ; ".join(labels[day]),
                          deadline, grace, asof_days))
    return slots


def slot_for(asof: dt.date, slots: list[Slot]) -> Slot | None:
    for s in slots:
        if s.date <= asof <= s.date + dt.timedelta(days=s.asof_days):
            return s
    return None


# --- archivage


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def identity(folder: Path, record: dict) -> dict:
    """Ce qui fait qu'une prévision archivée est la même : date, données, tirages, pages."""
    return {
        "asof": record["asof"],
        "polls_hash": record["data"]["polls_hash"],
        "draws": record["run"]["files"].get("draws.parquet"),
        **{name: digest(folder / name) for name in IDENTITY_FILES},
    }


def build_record(data: Path, run: Path, root: Path, name: str, slot: Slot,
                 now: dt.datetime) -> dict:
    manifest = json.loads((data / "manifest.json").read_text())
    rm = json.loads((run / "manifest.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    if manifest.get("mock"):
        raise ArchiveError("export factice (manifest.mock) : rien à archiver")
    asof = manifest["asof"]
    if rm.get("asof") != asof or summary.get("asof") != asof:
        raise ArchiveError(f"run {run} du {rm.get('asof')} et export du {asof} : run différent")
    if rm.get("n_draws") != manifest.get("n_draws"):
        raise ArchiveError("nombre de tirages différent entre le run et l'export")
    if rm.get("polls_hash") and rm["polls_hash"] != manifest.get("polls_hash"):
        raise ArchiveError("empreinte des sondages différente entre le run et l'export")
    if not (run / "draws.parquet").is_file():
        raise ArchiveError(f"tirages absents : {run / 'draws.parquet'}")
    scen = root / SCENARIOS
    scen_sha = digest(scen)
    run_scen = (rm.get("scenarios") or {}).get("sha256")
    if run_scen and not scen_sha.startswith(run_scen):
        raise ArchiveError("scenarios_2027.yaml a changé depuis le run : hypothèses différentes")
    fit_dir = Path(rm.get("fit", {}).get("dir") or "")
    fit = None
    if fit_dir.name and (fit_dir / "manifest.json").is_file():
        fit = {"dir": _rel(fit_dir, root), "files": tree_digests(fit_dir)}
    status = _git(root, "status", "--porcelain", "--", *CODE_PATHS, "data/registry")
    head = _git(root, "rev-parse", "HEAD")
    curated = root / "data" / "curated"
    srm = rm.get("second_round_model", {})
    return {
        "format": FORMAT,
        "name": name,
        "track": "comparison" if name == STABLE else "experimental",
        "slot": slot.to_dict(),
        "asof": asof,
        "created_at": now.isoformat(timespec="seconds"),
        "created_on": now.astimezone().date().isoformat(),
        "election": manifest.get("election"),
        "model_version": manifest.get("model_version"),
        "settings": {
            "share_rule": rm.get("settings", {}).get("share_rule"),
            "bias_mode": srm.get("bias_mode"),
            "seed": rm.get("seed"),
            "n_draws": rm.get("n_draws"),
        },
        "code": {
            "export_commit": manifest.get("commit"),
            "simulate_commit": rm.get("commit"),
            "fit_commit": rm.get("fit", {}).get("commit"),
            "git_head": head.strip() if head else None,
            "uncommitted": sorted(line[3:] for line in (status or "").splitlines()),
            "code_sha256": code_digest(root),
            "versions": rm.get("versions"),
        },
        "data": {
            "polls_hash": manifest.get("polls_hash"),
            "n_polls": manifest.get("n_polls"),
            "last_poll_date": manifest.get("last_poll_date"),
            "curated": {t: digest(curated / t) for t in CURATED_TABLES if (curated / t).is_file()},
        },
        "hypotheses": {
            "file": SCENARIOS,
            "sha256": scen_sha,
            "as_of": (rm.get("scenarios") or {}).get("as_of"),
        },
        "run": {"dir": _rel(run, root), "files": tree_digests(run)},
        "fit": fit,
    }


def archive_forecast(
    data: Path | None = None,
    run: Path | None = None,
    dest: Path | None = None,
    name: str = STABLE,
    asof: dt.date | None = None,
    today: dt.date | None = None,
    skip_existing: bool = False,
    scheduled_only: bool = False,
    root: Path | None = None,
) -> dict:
    """Archive la projection publiée à une date du calendrier ; rend un compte rendu.

    `status` : `written`, `unchanged` (même prévision déjà archivée), `kept` (archive
    différente conservée, `skip_existing`), `not_scheduled` (`scheduled_only`). Toute autre
    situation lève `ArchiveError` : date hors calendrier ou trop tardive, run incohérent,
    tentative de réécriture."""
    root = Path(root or paths.ROOT)
    data = Path(data or root / "front" / "static" / "data")
    dest = Path(dest or root / PROSPECTIVE)
    if not NAME_RE.match(name):
        raise ArchiveError(f"nom d'archive invalide : {name!r} (minuscules, chiffres, tirets)")
    manifest = json.loads((data / "manifest.json").read_text())
    published = dt.date.fromisoformat(manifest["asof"])
    if asof is not None and asof != published:
        raise ArchiveError(f"l'export publié est du {published}, pas du {asof}")
    election = manifest["election"]
    slots = schedule(dt.date.fromisoformat(election["first_round"]),
                     dt.date.fromisoformat(election["second_round"]))
    slot = slot_for(published, slots)
    if slot is None:
        if scheduled_only:
            return {"status": "not_scheduled", "asof": published.isoformat()}
        nxt = next((s.date.isoformat() for s in slots if s.date > published), "aucune")
        raise ArchiveError(f"{published} n'est pas une date d'archivage (suivante : {nxt})")
    today = today or dt.date.today()
    if today > slot.date + dt.timedelta(days=slot.grace_days) or today >= slot.deadline:
        raise ArchiveError(f"trop tard pour archiver le {slot.date} (aujourd'hui {today})")
    run = Path(run or root / "runs" / f"{published.isoformat()}-simulate-subst")
    target = dest / slot.date.isoformat() / name
    report = {"slot": slot.date.isoformat(), "name": name, "path": str(target),
              "asof": published.isoformat()}
    now = dt.datetime.now(dt.UTC)
    record = build_record(data, run, root, name, slot, now)
    if target.exists():
        old = json.loads((target / "archive.json").read_text())
        staged = {"asof": record["asof"], "data": record["data"], "run": record["run"]}
        new_id = identity(data, staged)
        if identity(target, old) == new_id:
            return {**report, "status": "unchanged"}
        if skip_existing:
            return {**report, "status": "kept"}
        raise ArchiveError(f"{target} existe et diffère de l'export : jamais réécrit")
    record["run"] = keep_run(run, root / KEPT_RUNS / slot.date.isoformat() / name, root,
                             record["run"]["files"])
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=target.parent))
    try:
        for f in SITE_FILES:
            shutil.copy2(data / f, tmp / f)
        shutil.copy2(run / "manifest.json", tmp / "run_manifest.json")
        shutil.copy2(root / SCENARIOS, tmp / Path(SCENARIOS).name)
        record["files"] = tree_digests(tmp)
        (tmp / "archive.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.chmod(tmp, 0o755)
        tmp.rename(target)  # échoue si une archive concurrente est apparue entre-temps
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return {**report, "status": "written", "n_files": len(record["files"]) + 1}


def keep_run(run: Path, keep: Path, root: Path, files: dict[str, str]) -> dict:
    """Copie du run hors de portée des runs suivants (un second run du même jour réécrit
    `runs/<date>-simulate-subst`) ; la copie est contrôlée contre les empreintes du run."""
    if not keep.exists():
        keep.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{keep.name}-", dir=keep.parent))
        try:
            shutil.copytree(run, tmp, dirs_exist_ok=True)
            tmp.rename(keep)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    if tree_digests(keep) != files:
        raise ArchiveError(f"copie du run {keep} différente du run {run}")
    return {"dir": _rel(keep, root), "source_dir": _rel(run, root), "files": files}


# --- contrôle


def verify_archive(folder: Path, root: Path | None = None) -> dict:
    """Empreintes de l'archive, puis du run et de l'ajustement s'ils sont encore dans runs/."""
    root = Path(root or paths.ROOT)
    problems: list[str] = []
    record = json.loads((folder / "archive.json").read_text())
    if record.get("format") != FORMAT:
        problems.append(f"format inconnu : {record.get('format')}")
    if folder.name != record.get("name") or folder.parent.name != record["slot"]["date"]:
        problems.append("dossier différent du nom ou de la date enregistrés")
    present = sorted(str(p.relative_to(folder)) for p in folder.rglob("*")
                     if p.is_file() and p.name != "archive.json")
    if present != sorted(record["files"]):
        problems.append(f"fichiers ajoutés ou retirés : {sorted(set(present) ^ set(record['files']))}")
    for rel, sha in record["files"].items():
        if (folder / rel).is_file() and digest(folder / rel) != sha:
            problems.append(f"fichier modifié : {rel}")
    slot_date = dt.date.fromisoformat(record["slot"]["date"])
    created = dt.date.fromisoformat(record["created_on"])
    if created > slot_date + dt.timedelta(days=record["slot"]["grace_days"]):
        problems.append(f"archive créée le {created}, après le délai de la date {slot_date}")
    checked = {}
    for key in ("run", "fit"):
        block = record.get(key)
        if not block:
            continue
        d = root / block["dir"]
        if not d.is_dir():
            checked[key] = "absent"
            continue
        got = tree_digests(d)
        diff = sorted(k for k in block["files"] if got.get(k) != block["files"][k])
        if diff:
            problems.append(f"{key} {block['dir']} modifié : {diff}")
        checked[key] = "ok" if not diff else "modifié"
    return {"path": str(folder), "problems": problems, "checked": checked}


def archives(dest: Path) -> list[Path]:
    return sorted(p.parent for p in Path(dest).glob("*/*/archive.json"))


def rewritten_archives(base: str, root: Path | None = None) -> list[str]:
    """Chemins d'archives présentes dans `base` et modifiés depuis (arbre de travail compris).

    Une nouvelle archive est permise ; tout ajout, modification ou suppression dans un
    dossier `<date>/<nom>/` qui existait déjà dans `base` est une réécriture."""
    root = Path(root or paths.ROOT)
    listed = _git(root, "ls-tree", "-r", "--name-only", base, "--", str(PROSPECTIVE))
    if listed is None:
        raise ArchiveError(f"référence Git introuvable : {base} (historique complet requis)")
    existing = {"/".join(p.split("/")[:4]) for p in listed.splitlines() if p.count("/") >= 4}
    changed = (_git(root, "diff", "--name-only", "--no-renames", base, "--", str(PROSPECTIVE))
               or "").splitlines()
    changed += (_git(root, "ls-files", "--others", "--exclude-standard", "--",
                     str(PROSPECTIVE)) or "").splitlines()
    return sorted({p for p in changed if "/".join(p.split("/")[:4]) in existing})
