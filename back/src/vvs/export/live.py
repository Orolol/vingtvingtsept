"""Résumé d'un run live (scripts/run_live.sh) : ce qui a changé entre deux exports du site.

`live_summary(previous, current)` compare deux dossiers au format du contrat JSON
(front/static/data) : la copie prise avant l'export et l'export du jour. Il rend un texte
court : sondages nouvellement intégrés, probabilités du mélange de scénarios et écart avec
l'export précédent, en points de probabilité.
"""

from __future__ import annotations

import json
from pathlib import Path

MIN_PROBABILITY = 0.005  # candidats affichés : qualification d'au moins 0,5 %


def _read(d: Path | None, name: str) -> dict | None:
    if d is None:
        return None
    p = Path(d) / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def _delta(p: float, q: float | None) -> str:
    if q is None:
        return "   nouveau"
    d = 100 * (p - q)
    return "         =" if abs(d) < 0.05 else f"{d:+7.1f} pt"


def live_summary(previous: Path | None, current: Path) -> str:
    cur_m = _read(current, "manifest.json")
    cur_s = _read(current, "summary.json")
    cur_p = _read(current, "polls.json")
    if cur_m is None or cur_s is None or cur_p is None:
        raise FileNotFoundError(f"export incomplet dans {current}")
    prev_m = _read(previous, "manifest.json")
    if prev_m is not None and prev_m.get("mock"):
        prev_m = None  # données factices : pas de comparaison
    prev_s = _read(previous, "summary.json") if prev_m else None
    prev_p = _read(previous, "polls.json") if prev_m else None

    names = {c["id"]: c["name"] for c in cur_m["candidates"]}
    lines = [
        (f"run {cur_m['run_date']}, prévision au {cur_m['asof']} : {cur_m['n_polls']} sondages "
         f"intégrés, dernier terrain le {cur_m['last_poll_date']}, empreinte "
         f"{cur_m.get('polls_hash') or '?'}"),
    ]
    if prev_m is None:
        lines.append("pas d'export réel précédent : aucune comparaison")
    else:
        lines.append(
            f"export précédent : run {prev_m['run_date']} (généré {prev_m['generated_at']}), "
            f"{prev_m['n_polls']} sondages, empreinte {prev_m.get('polls_hash') or '?'}"
        )
        before = {p["poll_id"] for p in prev_p["polls"]} if prev_p else set()
        new = [p for p in cur_p["polls"] if p["poll_id"] not in before]
        gone = before - {p["poll_id"] for p in cur_p["polls"]}
        lines.append(f"nouveaux sondages intégrés : {len(new)}")
        for p in sorted(new, key=lambda p: p["fieldwork_end"]):
            lines.append(f"  {p['fieldwork_end']}  {p['pollster']:<12s} {p['poll_id']}")
        if gone:
            lines.append(f"sondages sortis de la fenêtre ou de la curation : {len(gone)}")

    prev_mix = {}
    if prev_s is not None:
        prev_mix = {c["candidate_id"]: c for c in prev_s["mixture"]["candidates"]}
    lines.append("mélange de scénarios (victoire, qualification ; écart à l'export précédent) :")
    for c in cur_s["mixture"]["candidates"]:
        if c["p_qualify"] < MIN_PROBABILITY:
            continue
        old = prev_mix.get(c["candidate_id"])
        lines.append(
            f"  {names.get(c['candidate_id'], c['candidate_id'])[:30]:<30s} "
            f"victoire {_pct(c['p_win'])} {_delta(c['p_win'], old and old['p_win'])}   "
            f"qualif. {_pct(c['p_qualify'])} {_delta(c['p_qualify'], old and old['p_qualify'])}"
        )
    return "\n".join(lines)
