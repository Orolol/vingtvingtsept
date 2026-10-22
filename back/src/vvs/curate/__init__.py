"""Curation des sondages : fusion des sources, règles explicites, tables dérivées.

Commande : `uv run vvs curate` (après les `vvs parse …`). Lit dans data/curated les tables
par source (`polls_wikipedia_<AAAA>`, `polls_commission`, `polls_mieuxvoter` et leurs
`poll_results_*`), les résultats officiels `results_<AAAA>.parquet`, le registre, et écrit :

| Fichier | Contenu |
|---|---|
| `polls.parquet` | un sondage fusionné par ligne, `sources`, `is_rolling` (schéma `CURATED_POLLS_SCHEMA`) |
| `poll_results.parquet` | intentions de vote fusionnées, `source` de la valeur, `suspect`, `suspect_reason` ; une hypothèse publiée sur deux bases y figure deux fois, une par `population` (`rules.one_base` pour n'en garder qu'une) |
| `poll_sources.parquet` | sondage de chaque source -> `canonical_poll_id`, règle d'appariement |
| `curation_log.jsonl` | un conflit entre sources (ou une règle appliquée) par ligne |
| `poll_results_by_bloc.parquet` | parts par bloc et par hypothèse |
| `poll_errors.parquet` | erreurs historiques des sondages (2002-2022), par candidat et par bloc |

Modules : `sources` (lecture, libellés rolling), `rules` (toutes les règles et leurs
seuils, divergences connues), `merge` (appariement, précédence, journal), `derived` (tables
dérivées). Le détail des règles est dans le docstring de `rules` ; docs/03-donnees.md en
donne les principes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from vvs import paths

OUTPUTS = {
    "polls": "polls.parquet",
    "poll_results": "poll_results.parquet",
    "poll_sources": "poll_sources.parquet",
    "log": "curation_log.jsonl",
    "by_bloc": "poll_results_by_bloc.parquet",
    "errors": "poll_errors.parquet",
}


@dataclass
class CurationReport:
    polls: pl.DataFrame
    poll_results: pl.DataFrame
    poll_sources: pl.DataFrame
    log: list[dict]
    stats: Counter
    mv_unreliable: set[str]
    known_divergences: dict[str, bool]
    errors: pl.DataFrame
    by_bloc: pl.DataFrame
    written: list[Path] = field(default_factory=list)

    def summary(self, verbose: bool = False) -> str:
        p, r = self.polls, self.poll_results
        lines = [f"{p.height} sondages fusionnés, {r.height} lignes d'intentions"]
        lines.append("par élection (sondages ; dont Commission / Wikipédia / MieuxVoter) :")
        for row in (
            p.group_by("election")
            .agg(
                pl.len().alias("n"),
                pl.col("sources").list.contains("commission").sum().alias("c"),
                pl.col("sources").list.contains("wikipedia").sum().alias("w"),
                pl.col("sources").list.contains("mieuxvoter").sum().alias("m"),
                pl.col("is_rolling").sum().alias("rolling"),
            )
            .sort("election")
            .iter_rows(named=True)
        ):
            lines.append(
                f"  {row['election']} : {row['n']:4d} ; {row['c']:3d} / {row['w']:4d} / "
                f"{row['m']:3d} ; rolling {row['rolling']}"
            )
        combos = p.group_by("sources").len().sort("len", descending=True)
        lines.append("combinaisons de sources : " + ", ".join(
            f"{'+'.join(s)} {n}" for s, n in combos.iter_rows()
        ))
        hyp = r.unique(["poll_id", "round", "hypothesis_id", "population"])
        second = hyp.height - r.unique(["poll_id", "round", "hypothesis_id"]).height
        lines.append(
            f"hypothèses publiées sur deux bases (règle second_base) : {second}"
        )
        sus = hyp.filter("suspect")
        reasons = Counter(x for v in sus["suspect_reason"] for x in v.split("; "))
        lines.append(
            f"hypothèses : {hyp.height}, suspectes {sus.height} ("
            + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ")"
        )
        by_type = Counter((e["rule"], e["field"]) for e in self.log)
        lines.append(f"journal : {len(self.log)} entrées")
        for (rule, fld), n in sorted(by_type.items()):
            lines.append(f"  {rule:26s} {fld:16s} {n}")
        lines.append(
            "population MieuxVoter écartée pour : " + (", ".join(sorted(self.mv_unreliable)) or "aucun")
        )
        lines.append("divergences connues : " + ", ".join(
            f"{k} {'résolue' if v else 'NON TROUVÉE'}" for k, v in self.known_divergences.items()
        ))
        if verbose:
            for k, v in sorted(self.stats.items()):
                lines.append(f"  {k} : {v}")
        lines.append(f"poll_errors : {self.errors.height} lignes ; by_bloc : {self.by_bloc.height} lignes")
        for path in self.written:
            lines.append(f"écrit : {path}")
        return "\n".join(lines)


def run(curated: Path | None = None, write: bool = True) -> CurationReport:
    from vvs.curate import derived, rules
    from vvs.curate.merge import merge
    from vvs.curate.sources import load_sources
    from vvs.registry import default_registry

    curated = curated or paths.CURATED
    reg = default_registry()
    sources = load_sources(curated)
    res = merge(sources, rules.official_races(reg), set(reg.candidates))
    known = rules.check_known_divergences(res.log)

    by_bloc = derived.results_by_bloc(res.polls, res.results, reg.bloc_of)
    races = rules.official_races(reg)
    official = []
    round_dates = {}
    for year, race in sorted(races.items()):
        path = curated / f"results_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} absent : lancer `vvs parse interieur --election {year}`"
            )
        official.append(derived.official_shares(pl.read_parquet(path)))
        round_dates[(year, 1)] = race.round1
        if race.round2 is not None:
            round_dates[(year, 2)] = race.round2
    errors = derived.poll_errors(
        res.polls, res.results, pl.concat(official), round_dates, reg.bloc_of
    )
    report = CurationReport(
        res.polls, res.results, res.poll_sources, res.log, res.stats, res.mv_unreliable,
        known, errors, by_bloc,
    )
    if write:
        out = {k: curated / v for k, v in OUTPUTS.items()}
        res.polls.write_parquet(out["polls"])
        res.results.write_parquet(out["poll_results"])
        res.poll_sources.write_parquet(out["poll_sources"])
        with out["log"].open("w", encoding="utf-8") as f:
            for e in res.log:
                f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
        by_bloc.write_parquet(out["by_bloc"])
        errors.write_parquet(out["errors"])
        report.written = list(out.values())
    return report
