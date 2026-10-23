"""Archives prospectives 2027, contrôles d'intégrité de la CI et archive durable des runs
(docs/13-validation-prospective.md)."""

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess

import pytest
import yaml
from typer.testing import CliRunner

from vvs import paths
from vvs.export import integrity
from vvs.export.prospective import (
    ArchiveError,
    archive_forecast,
    rewritten_archives,
    schedule,
    slot_for,
    verify_archive,
)

D1, D2 = dt.date(2027, 4, 18), dt.date(2027, 5, 2)
ASOF = "2026-11-01"
SCRIPT = paths.ROOT / "scripts" / "archive_runs.sh"


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _project(root, asof=ASOF, p_win=0.7):
    """Dépôt minimal : export du site, run de simulation, ajustement, registre, code."""
    (root / "data/registry").mkdir(parents=True)
    scen = root / "data/registry/scenarios_2027.yaml"
    scen.write_text("election: 2027\n")
    scen_sha = hashlib.sha256(scen.read_bytes()).hexdigest()[:16]
    (root / "data/curated").mkdir(parents=True)
    (root / "data/curated/polls.parquet").write_bytes(b"polls")
    (root / "back/src/vvs").mkdir(parents=True)
    (root / "back/src/vvs/m.py").write_text("x = 1\n")
    (root / "back/pyproject.toml").write_text("[project]\n")
    (root / "back/uv.lock").write_text("lock\n")
    data = root / "front/static/data"
    election = {"year": 2027, "first_round": str(D1), "second_round": str(D2)}
    _write_json(data / "manifest.json", {
        "asof": asof, "run_date": asof, "generated_at": f"{asof}T06:00:00Z", "mock": False,
        "n_draws": 10_000, "polls_hash": "0123456789abcdef", "n_polls": 3, "commit": "abc1234",
        "model_version": "0.4.1", "election": election})
    _write_json(data / "summary.json", {"asof": asof, "mixture": {"p_win": p_win}})
    _write_json(data / "second_round.json", {"pairs": []})
    _write_json(data / "scenarios.json", {"dimensions": []})
    fit = root / "runs" / f"{asof}-fit-2027-subst"
    _write_json(fit / "manifest.json", {"asof": asof})
    (fit / "draws.npz").write_bytes(b"theta")
    run = root / "runs" / f"{asof}-simulate-subst"
    _write_json(run / "manifest.json", {
        "asof": asof, "n_draws": 10_000, "polls_hash": "0123456789abcdef", "commit": "abc1234",
        "seed": 0, "fit": {"dir": str(fit), "commit": "abc1234"},
        "scenarios": {"as_of": "2026-10-20", "sha256": scen_sha},
        "second_round_model": {"bias_mode": "curve"}, "settings": {"share_rule": "nested"}})
    (run / "draws.parquet").write_bytes(b"draws-" + str(p_win).encode())
    return root


def _archive(root, today=ASOF, **kw):
    return archive_forecast(root=root, today=dt.date.fromisoformat(today), **kw)


def test_schedule_fixed_dates():
    slots = schedule(D1, D2)
    days = [s.date for s in slots]
    assert days[:6] == [dt.date(2026, 10, 22), dt.date(2026, 11, 1), dt.date(2026, 12, 1),
                        dt.date(2027, 1, 1), dt.date(2027, 1, 18), dt.date(2027, 2, 1)]
    weekly = [s.date for s in slots if "weekly" in s.kinds]
    assert weekly[0] == dt.date(2027, 2, 7) and weekly[-1] == dt.date(2027, 4, 11)
    assert all(d.weekday() == D1.weekday() for d in weekly)
    by_day = {s.date: s for s in slots}
    assert by_day[dt.date(2027, 4, 11)].kinds == ("horizon", "weekly")  # J-7 fusionné
    assert "horizon" in by_day[dt.date(2027, 3, 19)].kinds  # J-30
    assert [s.date for s in slots if "eve" in s.kinds] == [D1 - dt.timedelta(1),
                                                           D2 - dt.timedelta(1)]
    for s, nxt in zip(slots, [*days[1:], D2], strict=True):
        assert s.date + dt.timedelta(days=s.grace_days) < min(nxt, s.deadline)
    assert by_day[D1 - dt.timedelta(1)].grace_days == 0
    assert slot_for(dt.date(2026, 11, 3), slots).date == dt.date(2026, 11, 1)
    assert slot_for(dt.date(2026, 11, 4), slots) is None
    # gel initial : seulement la projection du 22 octobre, archivable jusqu'au 24
    assert by_day[dt.date(2026, 10, 22)].grace_days == 2
    assert slot_for(dt.date(2026, 10, 22), slots).date == dt.date(2026, 10, 22)
    assert slot_for(dt.date(2026, 10, 23), slots) is None


def test_archive_written_verified_and_never_rewritten(tmp_path):
    root = _project(tmp_path)
    rep = _archive(root)
    assert rep["status"] == "written"
    target = root / "data/prospective" / ASOF / "stable"
    names = sorted(p.name for p in target.iterdir())
    assert names == ["archive.json", "manifest.json", "run_manifest.json", "scenarios.json",
                     "scenarios_2027.yaml", "second_round.json", "summary.json"]
    rec = json.loads((target / "archive.json").read_text())
    assert rec["track"] == "comparison" and rec["slot"]["kinds"] == ["monthly"]
    assert rec["run"]["dir"] == f"runs/prospective/{ASOF}/stable"
    assert rec["run"]["source_dir"] == f"runs/{ASOF}-simulate-subst"
    assert set(rec["run"]["files"]) == {"draws.parquet", "manifest.json"}
    assert rec["data"]["polls_hash"] == "0123456789abcdef" and rec["code"]["code_sha256"]
    assert rec["fit"]["files"]["draws.npz"]
    assert not list(target.parent.glob(".*"))  # pas de dossier temporaire résiduel
    ok = verify_archive(target, root)
    assert ok["problems"] == [] and ok["checked"] == {"run": "ok", "fit": "ok"}
    before = {p.name: p.read_bytes() for p in target.iterdir()}

    # même prévision réexportée (generated_at change) : rien à faire
    m = json.loads((root / "front/static/data/manifest.json").read_text())
    m["generated_at"] = f"{ASOF}T19:00:00Z"
    _write_json(root / "front/static/data/manifest.json", m)
    assert _archive(root, today="2026-11-02")["status"] == "unchanged"
    # prévision différente le même jour : échec explicite, ou archive gardée en mode quotidien
    _write_json(root / "front/static/data/summary.json", {"asof": ASOF, "mixture": {"p_win": 0.6}})
    with pytest.raises(ArchiveError, match="jamais réécrit"):
        _archive(root)
    assert _archive(root, skip_existing=True)["status"] == "kept"
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before

    # le run du jour réécrit par un second run ne touche pas la copie archivée
    (root / f"runs/{ASOF}-simulate-subst/draws.parquet").write_bytes(b"autre")
    assert verify_archive(target, root)["problems"] == []
    # altérations détectées
    (root / f"runs/prospective/{ASOF}/stable/draws.parquet").write_bytes(b"x")
    assert any("run" in p for p in verify_archive(target, root)["problems"])
    shutil.rmtree(root / "runs")
    assert verify_archive(target, root) == {"path": str(target), "problems": [],
                                            "checked": {"run": "absent", "fit": "absent"}}
    (target / "summary.json").write_text("{}")
    (target / "extra.json").write_text("{}")
    problems = verify_archive(target, root)["problems"]
    assert any("modifié : summary.json" in p for p in problems)
    assert any("extra.json" in p for p in problems)


@pytest.mark.parametrize(
    ("asof", "today", "change", "message"),
    [
        ("2026-11-05", "2026-11-05", None, "pas une date d'archivage"),
        (ASOF, "2026-11-04", None, "trop tard"),
        (ASOF, ASOF, "mock", "factice"),
        (ASOF, ASOF, "run_asof", "run différent"),
        (ASOF, ASOF, "scenarios", "scenarios_2027.yaml a changé"),
        (ASOF, ASOF, "name", "nom d'archive invalide"),
    ],
)
def test_archive_refusals(tmp_path, asof, today, change, message):
    root = _project(tmp_path, asof=asof)
    kw = {}
    if change == "mock":
        m = json.loads((root / "front/static/data/manifest.json").read_text())
        _write_json(root / "front/static/data/manifest.json", {**m, "mock": True})
    elif change == "run_asof":
        run = root / f"runs/{asof}-simulate-subst/manifest.json"
        _write_json(run, {**json.loads(run.read_text()), "asof": "2026-10-31"})
    elif change == "scenarios":
        (root / "data/registry/scenarios_2027.yaml").write_text("election: 2027\nnew: 1\n")
    elif change == "name":
        kw["name"] = "Stable/../x"
    with pytest.raises(ArchiveError, match=message):
        _archive(root, today=today, **kw)
    assert not (root / "data/prospective").exists() or not any(
        (root / "data/prospective").rglob("archive.json"))


def test_scheduled_only_and_experimental_name(tmp_path):
    root = _project(tmp_path, asof="2026-11-05")
    assert _archive(root, today="2026-11-05", scheduled_only=True)["status"] == "not_scheduled"
    root2 = _project(tmp_path / "b")
    rep = _archive(root2, name="candidacies-v1")
    rec = json.loads((root2 / "data/prospective" / ASOF / "candidacies-v1/archive.json")
                     .read_text())
    assert rep["status"] == "written" and rec["track"] == "experimental"
    with pytest.raises(ArchiveError, match="pas du"):
        _archive(root2, asof=dt.date(2026, 11, 2))


def test_cli_exit_codes(tmp_path, monkeypatch):
    from vvs.cli import app

    root = _project(tmp_path)
    _archive(root)
    _write_json(root / "front/static/data/summary.json", {"asof": ASOF, "mixture": {"p_win": 0.1}})
    monkeypatch.setattr(paths, "ROOT", root)
    runner = CliRunner()
    run = str(root / f"runs/{ASOF}-simulate-subst")
    r = runner.invoke(app, ["archive-forecast", "--run", run])
    assert r.exit_code == 1 and "jamais réécrit" in r.output
    r = runner.invoke(app, ["archive-forecast", "--run", run, "--skip-existing",
                            "--scheduled-only"])
    assert r.exit_code == 0 and "conservée" in r.output
    r = runner.invoke(app, ["archive-verify"])
    assert r.exit_code == 0 and "1 archive(s), 0 écart(s)" in r.output


@pytest.mark.skipif(shutil.which("git") is None, reason="git absent")
def test_rewrites_detected_against_base(tmp_path):
    root = _project(tmp_path)
    _archive(root)
    hist = root / "front/static/data/history"
    _write_json(hist / "2026-10-22.json", {"run_date": "2026-10-22"})
    _write_json(hist / "index.json", {"entries": []})
    _git(root, "init", "-q")
    _git(root, "add", "data/prospective", "front/static/data")
    _git(root, "commit", "-q", "-m", "base")
    assert rewritten_archives("HEAD", root) == []
    assert integrity.rewritten_history("HEAD", root) == []
    # nouvelle archive et index régénéré : permis
    (root / "data/prospective/2026-12-01/stable").mkdir(parents=True)
    (root / "data/prospective/2026-12-01/stable/archive.json").write_text("{}")
    _write_json(hist / "index.json", {"entries": [1]})
    assert rewritten_archives("HEAD", root) == []
    assert integrity.rewritten_history("HEAD", root) == []
    # modification, ajout dans une archive existante, suppression d'une entrée d'historique
    target = f"data/prospective/{ASOF}/stable"
    (root / target / "summary.json").write_text("{}")
    (root / target / "notes.txt").write_text("ajout")
    (hist / "2026-10-22.json").unlink()
    assert rewritten_archives("HEAD", root) == [f"{target}/notes.txt", f"{target}/summary.json"]
    assert integrity.rewritten_history("HEAD", root) == [
        "front/static/data/history/2026-10-22.json"]
    with pytest.raises(ArchiveError, match="introuvable"):
        rewritten_archives("pas-une-ref", root)


@pytest.mark.skipif(shutil.which("git") is None, reason="git absent")
def test_reference_lock_check(tmp_path):
    root = tmp_path
    spec = {"id": "reference-x", "seed": 1}
    (root / "data/registry").mkdir(parents=True)
    (root / "data/registry/reference_x.yaml").write_text(yaml.safe_dump(spec))
    (root / "back").mkdir()
    (root / "back/uv.lock").write_text("lock\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "gel")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
    sha = hashlib.sha256(b"lock\n").hexdigest()
    lock = {"id": "reference-x", "commit": commit.strip(), "spec": spec,
            "files": {"source/back/uv.lock": sha, "published/draws.parquet": "0" * 64}}
    path = root / "data/registry/reference_x.lock.json"
    _write_json(path, lock)
    assert integrity.check_lock(root, path.relative_to(root)) == []
    _write_json(path, {**lock, "files": {"source/back/uv.lock": "1" * 64}})
    assert "diffère du commit" in integrity.check_lock(root, path.relative_to(root))[0]
    _write_json(path, {**lock, "spec": {**spec, "seed": 2}})
    assert "configuration différente" in integrity.check_lock(root, path.relative_to(root))[0]
    _write_json(path, {**lock, "commit": "0" * 40})
    assert "historique Git complet" in integrity.check_lock(root, path.relative_to(root))[-1]


# --- scripts/archive_runs.sh


def _runs(tmp_path):
    runs = tmp_path / "runs"
    for name, files in {"2026-10-22-consolidation": {"reference.json": "{}", "source/a.py": "a"},
                        "2026-10-22-simulate-subst": {"draws.parquet": "d"},
                        "2026-10-22-economy": {"x": "y"}}.items():
        for rel, text in files.items():
            (runs / name / rel).parent.mkdir(parents=True, exist_ok=True)
            (runs / name / rel).write_text(text)
    return runs


def _script(*args, runs, env=None):
    e = {**os.environ, "VVS_RUNS_DIR": str(runs), **(env or {})}
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], env=e, capture_output=True,
                          text=True, check=False)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash absent")
@pytest.mark.parametrize("compression", ["auto", "gzip"])
def test_archive_runs_create_verify_restore(tmp_path, compression):
    runs = _runs(tmp_path)
    env = {"VVS_ARCHIVE_COMPRESSION": compression}
    assert _script(runs=runs).returncode == 2  # destination obligatoire
    inside = _script(paths.ROOT / "runs" / "jamais", runs=runs)
    assert inside.returncode == 1 and "dans le dépôt" in inside.stderr
    r = _script(tmp_path / "dest", runs=runs, env=env)
    assert r.returncode == 0, r.stderr
    (snap,) = (tmp_path / "dest").iterdir()
    index = (snap / "index.tsv").read_text().splitlines()
    assert [line.split("\t")[0] for line in index[1:]] == [
        "2026-10-22-consolidation", "2026-10-22-simulate-subst"]  # pas les études
    if compression == "gzip" or shutil.which("zstd") is None:
        assert all(line.split("\t")[2] == "gzip" for line in index[1:])
    sums = (snap / "SHA256SUMS").read_text()
    assert "index.tsv" in sums and "meta.txt" in sums and "2026-10-22-consolidation.files" in sums
    restore = tmp_path / "restore"
    v = _script("--verify", snap, restore, runs=runs)
    assert v.returncode == 0, v.stderr
    assert (restore / "2026-10-22-consolidation/source/a.py").read_text() == "a"
    assert "identique à l'instantané" in v.stdout
    # run local modifié : signalé sans échec ; archive altérée : échec
    (runs / "2026-10-22-simulate-subst/draws.parquet").write_text("autre")
    v = _script("--verify", snap, runs=runs)
    assert v.returncode == 0 and "modifié depuis l'instantané" in v.stdout
    arc = next(snap.glob("2026-10-22-simulate-subst.tar.*"))
    arc.write_bytes(arc.read_bytes() + b"\0")
    assert _script("--verify", snap, runs=runs).returncode == 1
