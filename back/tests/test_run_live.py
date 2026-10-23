"""Chaîne live : scripts/run_live.sh (--dry-run), résumé de run, réutilisation de l'ajustement."""

import datetime as dt
import json
import os
import shutil
import subprocess

import pytest

from vvs import paths
from vvs.cli import find_reusable_fit, reusable_fit
from vvs.export.live import live_summary

SCRIPT = paths.ROOT / "scripts" / "run_live.sh"


def _dry_run(tmp_path, **env):
    # faux uv et npm : s'ils sont appelés, le marqueur apparaît
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    for tool in ("uv", "npm"):
        f = fake / tool
        f.write_text(f"#!/bin/sh\ntouch {tmp_path}/called-{tool}\n")
        f.chmod(0o755)
    e = {"PATH": f"{fake}:{os.environ['PATH']}", "HOME": str(tmp_path), **env}
    return subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=e, capture_output=True,
                          text=True, check=False)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash absent")
def test_dry_run_lists_commands_without_running_them(tmp_path):
    logs = paths.RUNS / "logs"
    before = set(logs.iterdir()) if logs.exists() else set()
    r = _dry_run(tmp_path, VVS_ASOF="2027-01-05", VVS_SUBSTITUTION="0", VVS_DRAWS="20000")
    assert r.returncode == 0, r.stderr
    cmds = [line.split("$ ", 1)[1] for line in r.stdout.splitlines() if "$ " in line]
    run_dir = paths.RUNS / "2027-01-05-simulate"
    assert cmds[:8] == [
        "uv run vvs ingest commission --since 2026-01-01",
        "uv run vvs ingest wikipedia --election 2027",
        "uv run vvs ingest mieuxvoter",
        "uv run vvs parse commission",
        "uv run vvs parse wikipedia --election 2027",
        "uv run vvs parse mieuxvoter",
        "uv run vvs curate",
        f"uv run vvs simulate --asof 2027-01-05 --rn-bias-curve --draws 20000 --out {run_dir}",
    ]
    assert cmds[9] == f"uv run vvs export --run {run_dir}"
    assert cmds[10:12] == ["npm run validate-data", "npm run build"]
    assert cmds[-1].startswith("uv run vvs live-summary --previous")
    assert not list(tmp_path.glob("called-*"))
    assert (set(logs.iterdir()) if logs.exists() else set()) == before  # aucun journal écrit


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash absent")
def test_dry_run_substitution_default_and_bad_values(tmp_path):
    r = _dry_run(tmp_path, VVS_ASOF="2026-10-20")
    assert r.returncode == 0, r.stderr
    sim = [line for line in r.stdout.splitlines() if "vvs simulate" in line]
    assert sim and "--substitution" in sim[0] and "2026-10-20-simulate-subst" in sim[0]
    # archive prospective en dernier, après le résumé, sans réécriture possible
    last = r.stdout.splitlines()[-2].split("$ ", 1)[1]
    assert last.startswith("uv run vvs archive-forecast --asof 2026-10-20 --run ")
    assert last.endswith("--name stable --scheduled-only --skip-existing")
    off = _dry_run(tmp_path, VVS_ASOF="2026-10-20", VVS_ARCHIVE="0")
    assert "archive-forecast" not in off.stdout and "sautée" in off.stdout
    assert _dry_run(tmp_path, VVS_ARCHIVE="peut-être").returncode == 2
    assert _dry_run(tmp_path, VVS_SUBSTITUTION="peut-être").returncode == 2
    assert _dry_run(tmp_path, VVS_ASOF="20/10/2026").returncode == 2


def _export(d, polls, mix, run_date="2026-10-20", mock=False):
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "run_date": run_date, "asof": run_date, "generated_at": f"{run_date}T06:00:00Z",
        "n_polls": len(polls), "last_poll_date": "2026-10-16", "polls_hash": "abc", "mock": mock,
        "candidates": [{"id": c, "name": c.split("_")[0].title()} for c in mix]}))
    (d / "polls.json").write_text(json.dumps({"polls": [
        {"poll_id": p, "pollster": "ifop", "fieldwork_end": "2026-10-16"} for p in polls]}))
    (d / "summary.json").write_text(json.dumps({"mixture": {"candidates": [
        {"candidate_id": c, "p_win": w, "p_qualify": q} for c, (w, q) in mix.items()]}}))
    return d


def test_live_summary(tmp_path):
    prev = _export(tmp_path / "prev", ["a", "b"], {"le_pen_marine": (0.70, 0.80),
                                                   "philippe_edouard": (0.30, 0.50)})
    cur = _export(tmp_path / "cur", ["a", "b", "c"], {"le_pen_marine": (0.65, 0.80),
                                                      "philippe_edouard": (0.35, 0.50),
                                                      "attal_gabriel": (0.0, 0.002)})
    txt = live_summary(prev, cur)
    assert "nouveaux sondages intégrés : 1" in txt and " c" in txt
    lines = {line.split()[0]: line for line in txt.splitlines() if line.startswith("  ")}
    assert "-5.0 pt" in lines["Le"] and "+5.0 pt" in lines["Philippe"]
    assert "Attal" not in txt  # qualification sous 0,5 %
    mock = _export(tmp_path / "mock", ["a"], {"le_pen_marine": (0.5, 0.9)}, mock=True)
    assert "aucune comparaison" in live_summary(mock, cur)
    assert "aucune comparaison" in live_summary(None, cur)


def _fit(root, name, asof, rows_hash, substitution=True, draws=True):
    d = root / name
    d.mkdir()
    if draws:
        (d / "draws.npz").write_bytes(b"")
    (d / "manifest.json").write_text(json.dumps({
        "election": 2027, "asof": asof, "data": {"rows_hash": rows_hash},
        "parameters": {"weeks": 78, "step_days": 7, "priors": {"substitution": substitution}}}))
    return d


def test_fit_reuse_needs_same_polls_week_and_variant(tmp_path):
    d = _fit(tmp_path, "2026-10-20-fit-2027-subst", "2026-10-20", "h1")
    same_week = dt.date(2026, 10, 22)  # même semaine de grille avant le vote du 18 avril
    assert reusable_fit(d, 2027, same_week, "h1", True)
    assert not reusable_fit(d, 2027, same_week, "h2", True)  # nouveaux sondages
    assert not reusable_fit(d, 2027, same_week, "h1", False)  # autre variante
    assert not reusable_fit(d, 2027, dt.date(2026, 10, 30), "h1", True)  # semaine suivante
    _fit(tmp_path, "2026-10-21-fit-2027-subst", "2026-10-21", "h1", draws=False)
    assert find_reusable_fit(tmp_path, 2027, same_week, "h1", True) == d
    assert find_reusable_fit(tmp_path, 2027, same_week, "h1", False) is None
