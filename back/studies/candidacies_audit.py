"""Audit du registre historique des candidatures aux douze dates du backtest (docs/12, lot 1).

Pour chaque élection (2012, 2017, 2022) et chaque horizon (180, 90, 30, 7 jours), à la date
`asof` du backtest (`vvs.eval.backtest.asof_for`) :

- état de chaque personne déjà tracée (`state_at`, événements publiés avant `asof`) ;
- personnes possibles ou déclarées (états actifs) comparées aux candidats officiels finaux :
  couverts, manqués, en trop ;
- comparaison avec les configurations sondées dans les 30 jours précédents (sondages terminés
  dans (asof − 30, asof], fenêtre de `known_configuration`) : sondé sans trace dans le
  registre, sondé mais exclu ou retiré dans le registre, actif dans le registre mais non sondé ;
- part des voix finales couverte par le registre, par les sondages récents et par la
  configuration de la référence (`known_configuration`) : complément des « voix finales hors
  support » de docs/10 et docs/11 ;
- primaires, retraits, retours et engagements appliqués à la date.

Le résultat électoral ne sert qu'à mesurer la couverture, jamais à corriger le registre.

    uv run python studies/candidacies_audit.py --out ../runs/2026-10-23-candidacies
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter
from pathlib import Path

import polars as pl

from vvs.eval.backtest import asof_for
from vvs.eval.configurations import WINDOW_DAYS, known_configuration
from vvs.model.data import load_polls
from vvs.registry import default_registry
from vvs.registry.candidacies import ACTIVE_STATES, load_candidacies

ELECTIONS = (2012, 2017, 2022)
HORIZONS = (180, 90, 30, 7)
STATE_ORDER = ["official", "designated", "declared", "possible", "undetermined", "withdrawn",
               "excluded"]


def final_shares(registry, election: int) -> dict[str, float]:
    return {r.candidate_id: r.share for r in registry.elections[election].round_(1).results}


def polled_recent(polls: pl.DataFrame, asof: dt.date) -> set[str]:
    recent = polls.filter(
        (pl.col("fieldwork_end") <= asof)
        & (pl.col("fieldwork_end") > asof - dt.timedelta(days=WINDOW_DAYS))
    )
    return set(recent["candidate_id"].unique().to_list())


def audit_case(election: int, horizon: int, registry, cr, polls) -> tuple[dict, list[dict]]:
    asof = asof_for(election, horizon)
    states = cr.state_at(asof)
    shares = final_shares(registry, election)
    official = set(shares)
    active = {p for p, s in states.items() if s.state in ACTIVE_STATES}
    inactive = {p for p, s in states.items() if s.state not in ACTIVE_STATES}
    polled = polled_recent(polls, asof)
    reference = set(known_configuration(election, asof, registry, polls).candidates)

    def cover(group):
        return sum(shares.get(c, 0.0) for c in group)

    rows = []
    for p in sorted(set(cr.persons) | polled):
        s = states.get(p)
        person = cr.persons.get(p)
        rows.append({
            "election": election, "horizon": horizon, "asof": str(asof), "person": p,
            "in_registry": person is not None,
            "state": s.state if s else ("untraced" if person else "absent"),
            "since": str(s.since) if s else "", "reason": s.reason if s else "",
            "last_event": s.trail[-1] if s else "", "returned": bool(s and s.returned),
            "final_status": person.final_status if person else "",
            "bloc": person.bloc if person else "",
            "polled_30d": p in polled, "in_reference_config": p in reference,
            "final_share": shares.get(p, ""),
        })
    processes = []
    for proc in cr.processes.values():
        rounds_done = [r for r in proc.rounds if r.published < asof]
        entries = [e for e in cr.events
                   if e.type == "primary_entry" and e.process == proc.id and e.published < asof]
        if not entries and not rounds_done:
            continue
        status = ("terminé" if len(rounds_done) == len(proc.rounds)
                  else f"{len(rounds_done)}/{len(proc.rounds)} tours publiés")
        processes.append(f"{proc.id} ({status}) : " + ", ".join(
            f"{x} {states[x].state if x in states else 'sans trace'}" for x in proc.participants))
    summary = {
        "election": election, "horizon": horizon, "asof": asof,
        "n_traced": len(states), "n_active": len(active),
        "by_state": Counter(s.state for s in states.values()),
        "covered": sorted(official & active), "missed": sorted(official - active),
        "extra": sorted(active - official),
        "share_registry": cover(official & active),
        "share_declared": cover({p for p in official & active
                                 if states[p].state in ("declared", "designated", "official")}),
        "share_polled": cover(official & polled),
        "share_reference": cover(official & reference),
        "polled_untraced": sorted(polled - set(states)),
        "polled_inactive": sorted(polled & inactive),
        "active_not_polled": sorted(active - polled),
        "returned": sorted(p for p, s in states.items() if s.returned),
        "processes": processes,
        "n_polls": polls.filter(
            (pl.col("fieldwork_end") <= asof)
            & (pl.col("fieldwork_end") > asof - dt.timedelta(days=WINDOW_DAYS))
        )["poll_id"].n_unique(),
        "states": states,
    }
    return summary, rows


def fmt(names, states=None):
    if not names:
        return "—"
    if states is None:
        return ", ".join(names)
    return ", ".join(f"{n} ({states[n].state})" for n in names)


def report(summaries: list[dict], crs: dict) -> str:
    lines = [
        "# Audit du registre des candidatures aux douze dates du backtest",
        "",
        f"Produit par `back/studies/candidacies_audit.py` le {dt.date.today()}. États calculés",
        "par `state_at` à partir des seuls événements publiés avant la date ; « actifs » =",
        "possible, déclaré, désigné, indéterminé ou officiel. Couverture = part des voix finales",
        "du premier tour (exprimés) portée par les candidats officiels présents dans l'ensemble.",
        "Sondages : terminés dans les 30 jours avant la date (polls.parquet). Référence : la",
        "configuration de `known_configuration` (gel `reference-2026-10-22`), avant la",
        "correspondance des « tenants lieu » : son complément diffère donc des « voix finales hors",
        "support » publiées dans docs/10. « Dont déclarés » : candidats officiels déjà déclarés,",
        "désignés ou inscrits à la date (les personnes seulement envisagées sont exclues).",
        "",
        "## Synthèse",
        "",
        ("| Scrutin | Horizon | Date | Tracés | Actifs | Officiels couverts | Manqués | "
        "Voix couvertes : registre | dont déclarés | sondages 30 j | référence |"),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        n_off = len(s["covered"]) + len(s["missed"])
        lines.append(
            f"| {s['election']} | {s['horizon']} | {s['asof']} | {s['n_traced']} | {s['n_active']} "
            f"| {len(s['covered'])}/{n_off} | {fmt(s['missed'])} | {s['share_registry']:.2f} % "
            f"| {s['share_declared']:.2f} % "
            f"| {s['share_polled']:.2f} % | {s['share_reference']:.2f} % |"
        )
    lines += ["", "## Registre et sources", ""]
    for e, cr in crs.items():
        explicit = [x for x in cr.events if not x.derived]
        basis = Counter(x.published_basis for x in explicit)
        types = Counter(x.type for x in explicit)
        final = Counter(p.final_status for p in cr.persons.values())
        lines.append(
            f"- {e} : {len(cr.persons)} personnes ({final['official']} officielles, "
            f"{final['declared_absent']} déclarées absentes, {final['never_declared']} jamais "
            f"déclarées), {len(cr.processes)} processus, {len(explicit)} événements explicites "
            f"et {len(cr.events) - len(explicit)} déduits des tours ; dates de publication : "
            f"{basis['source']} par la source, {basis['wikipedia_revision']} par une révision "
            f"Wikipédia (borne par excès), {basis['imputed']} imputées. Types : "
            + ", ".join(f"{k} {v}" for k, v in sorted(types.items())) + "."
        )
    for s in summaries:
        st = s["states"]
        lines += [
            "",
            f"## {s['election']} à {s['horizon']} jours ({s['asof']})",
            "",
            "États : " + ", ".join(f"{k} {s['by_state'][k]}" for k in STATE_ORDER
                                    if s["by_state"][k]) + f" ; {s['n_polls']} sondages sur 30 j.",
            "",
            f"- Officiels couverts : {fmt(s['covered'], st)}",
            f"- Officiels manqués : {fmt(s['missed'], st)}",
            f"- Actifs non officiels ({len(s['extra'])}) : {fmt(s['extra'], st)}",
            f"- Sondés sans trace dans le registre : {fmt(s['polled_untraced'])}",
            f"- Sondés mais exclus ou retirés dans le registre : {fmt(s['polled_inactive'], st)}",
            (f"- Actifs non sondés ({len(s['active_not_polled'])}) : "
            f"{fmt(s['active_not_polled'], st)}"),
            (f"- Retours (déclaration ou nouvelle mention après exclusion ou retrait) : "
            f"{fmt(s['returned'], st)}"),
            "- Processus : " + ("; ".join(s["processes"]) or "—"),
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    registry = default_registry()
    summaries, rows, crs = [], [], {}
    for election in ELECTIONS:
        cr = crs[election] = load_candidacies(election)
        polls = load_polls(election)
        for horizon in HORIZONS:
            summary, case_rows = audit_case(election, horizon, registry, cr, polls)
            summaries.append(summary)
            rows += case_rows
    with (args.out / "audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "audit.md").write_text(report(summaries, crs))
    for s in summaries:
        print(s["election"], s["horizon"], s["asof"], f"actifs {s['n_active']}",
              f"couverts {len(s['covered'])}", f"manqués {s['missed']}",
              f"voix {s['share_registry']:.2f} / sondés {s['share_polled']:.2f} "
              f"/ réf {s['share_reference']:.2f}")


if __name__ == "__main__":
    main()
