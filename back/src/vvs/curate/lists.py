"""Scrutins de liste et législatives : registre des listes, résultats par bloc, erreurs.

Étend la calibration de l'erreur des sondages (docs/04-modele.md, section 2 : « on complète
la calibration avec les législatives et européennes pour les composantes de bloc ») aux
européennes 2014, 2019, 2024 et aux législatives 2022, 2024.

Entrées : `data/registry/lists.yaml` (voir son en-tête), `polls_lists.parquet` et
`poll_results_lists.parquet` (`vvs.ingest.wikipedia.scrutins`), `results_lists_<scrutin>
.parquet` (`vvs.ingest.interieur`). Sorties :

| Fichier | Contenu |
|---|---|
| `results_lists_by_bloc.parquet` | résultats nationaux par unité de résultat (`level` = `list`) et par bloc (`level` = `bloc`) |
| `poll_errors_lists.parquet` | colonnes de `poll_errors` plus `election_type` (`euro`, `leg`) ; `election` en texte |

Règles propres à ces scrutins, en plus de celles de `vvs.curate.derived` (réutilisées telles
quelles : choix de l'hypothèse, moyennes, éclaircissement des rollings) :

- un libellé du ministère absent de `lists.yaml` fait échouer le rattachement ;
- les composantes (`group`) s'ajoutent à leur unité de résultat, et la colonne « Autres »
  (id réservé `other`) à l'unité résiduelle du scrutin, avant le choix de l'hypothèse ;
- dans un sondage, on ne garde que les hypothèses qui utilisent le moins de composantes : une
  hypothèse « gauche unie » l'emporte sur la même hypothèse aux partis séparés ;
- une unité dont le sondage ne teste qu'une partie des composantes (`complete` faux, voir
  `unit_poll_results`) compte au niveau bloc mais pas au niveau liste : les erreurs par liste
  viennent d'un second passage de `derived.poll_errors` sur les unités complètes ;
- hypothèse suspecte : somme des parts exprimées hors de `rules.SHARE_SUM_BOUNDS` ;
- rollings : `rules.rolling_by_regularity` (vagues chevauchantes d'un même institut) ;
- seul le premier tour compte ; aux législatives, l'unité est la coalition ou la nuance
  (sondages nationaux d'intentions par coalition), pas le candidat.
"""

from __future__ import annotations

import datetime as dt
import functools
from collections import Counter
from pathlib import Path

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vvs import paths, schemas
from vvs.curate import derived, rules
from vvs.registry.loader import normalize_label
from vvs.registry.models import ID_RE, Bloc

LISTS_PATH = paths.REGISTRY / "lists.yaml"

ELECTION_TYPES = ("euro", "leg")
VOTES_TOLERANCE = 10

POLL_ERRORS_LISTS_SCHEMA = pl.Schema(
    {**schemas.POLL_ERRORS_SCHEMA, "election": pl.Utf8, "election_type": pl.Utf8}
)

RESULTS_LISTS_BY_BLOC_SCHEMA = pl.Schema(
    {
        "election": pl.Utf8,
        "election_type": pl.Utf8,
        "round": pl.Int8,
        "level": pl.Utf8,  # list (unité de résultat) ou bloc
        "unit": pl.Utf8,
        "bloc": pl.Utf8,
        "votes": pl.Int64,
        "expressed": pl.Int64,
        "share": pl.Float64,
        "labels": pl.List(pl.Utf8),  # libellés du ministère agrégés
    }
)


def election_type(key: str) -> str:
    kind = key.rstrip("0123456789")
    if kind not in ELECTION_TYPES:
        raise ValueError(f"scrutin {key!r} : type inconnu ({kind!r})")
    return kind


# ---------------------------------------------------------------------------
# Registre lists.yaml
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListAliases(_Model):
    wikipedia: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)


class ListEntry(_Model):
    label: str
    head: str | None = None
    parties: list[str] = Field(min_length=1)
    bloc: Bloc | None = None
    note: str | None = None
    group: str | None = None
    residual: bool = False
    aliases: ListAliases = Field(default_factory=ListAliases)

    @model_validator(mode="after")
    def _shape(self) -> ListEntry:
        if self.group and self.aliases.results:
            raise ValueError("une composante (group) ne porte pas d'alias de résultats")
        if self.residual and self.group:
            raise ValueError("l'unité résiduelle n'a pas de group")
        return self

    @property
    def is_unit(self) -> bool:
        return bool(self.aliases.results) or self.residual


class ListsFile(_Model):
    elections: dict[str, dict[str, ListEntry]]
    pollsters: dict[str, str | dict[str, str]] = Field(default_factory=dict)
    ambiguous: dict[str, list[str]] = Field(default_factory=dict)


class ListRegistry:
    """Listes validées, avec leur bloc résolu (parties.yaml)."""

    def __init__(self, data: ListsFile, blocs: dict[tuple[str, str], str]):
        self.elections = data.elections
        self.pollsters = data.pollsters
        self.ambiguous = data.ambiguous
        self._blocs = blocs

    def for_election(self, key: str) -> dict[str, ListEntry]:
        try:
            return self.elections[key]
        except KeyError:
            raise ValueError(f"lists.yaml : scrutin {key!r} absent") from None

    def bloc_of(self, key: str, list_id: str) -> str:
        try:
            return self._blocs[(key, list_id)]
        except KeyError:
            raise KeyError(f"lists.yaml {key} : liste inconnue {list_id!r}") from None

    def residual(self, key: str) -> str:
        return next(lid for lid, e in self.for_election(key).items() if e.residual)

    def unit_of(self, key: str, list_id: str) -> str:
        """Unité de résultat d'un id de sondage : son groupe, l'unité résiduelle pour
        `other`, lui-même sinon (unité, ou liste d'une hypothèse qui n'a pas eu lieu)."""
        if list_id == "other":
            return self.residual(key)
        entry = self.for_election(key).get(list_id)
        if entry is None:
            raise KeyError(f"lists.yaml {key} : liste inconnue {list_id!r}")
        return entry.group or list_id

    def units(self, key: str) -> list[str]:
        return [lid for lid, e in self.for_election(key).items() if e.is_unit]

    def result_aliases(self, key: str) -> dict[str, str]:
        """libellé du ministère normalisé -> unité."""
        out: dict[str, str] = {}
        for lid, e in self.for_election(key).items():
            for label in e.aliases.results:
                out[normalize_label(label)] = lid
        return out


def _validate(data: ListsFile, parties: dict) -> tuple[list[str], dict[tuple[str, str], str]]:
    errors: list[str] = []
    blocs: dict[tuple[str, str], str] = {}
    seen_ids: dict[str, str] = {}
    for key, entries in data.elections.items():
        try:
            election_type(key)
        except ValueError as e:
            errors.append(str(e))
        residuals = [lid for lid, e in entries.items() if e.residual]
        if len(residuals) != 1:
            errors.append(f"{key} : il faut exactement une unité résiduelle, trouvé {residuals}")
        res_labels: dict[str, str] = {}
        for lid, e in entries.items():
            where = f"{key}.{lid}"
            if not ID_RE.match(lid):
                errors.append(f"{where} : id mal formé")
            if lid in seen_ids:
                errors.append(f"{where} : id déjà utilisé par {seen_ids[lid]}")
            seen_ids[lid] = key
            unknown = [p for p in e.parties if p not in parties]
            if unknown:
                errors.append(f"{where} : partis inconnus {unknown}")
                continue
            party_blocs = {parties[p].bloc for p in e.parties}
            if e.bloc is None:
                if len(party_blocs) != 1:
                    errors.append(
                        f"{where} : partis de blocs différents {sorted(party_blocs)}, "
                        "donner `bloc` (dominant) et une `note`"
                    )
                    continue
                bloc = next(iter(party_blocs))
            else:
                bloc = e.bloc
                if party_blocs != {bloc} and not e.note:
                    errors.append(f"{where} : bloc {bloc} différent des partis {sorted(party_blocs)} sans note")
            blocs[(key, lid)] = bloc
            if e.group is not None:
                g = entries.get(e.group)
                if g is None or not g.is_unit:
                    errors.append(f"{where} : group {e.group!r} n'est pas une unité de résultat du scrutin")
            for label in e.aliases.results:
                k = normalize_label(label)
                if k in res_labels:
                    errors.append(f"{where} : libellé de résultat {label!r} déjà rattaché à {res_labels[k]}")
                res_labels[k] = lid
    return errors, blocs


@functools.cache
def load_lists(path: Path = LISTS_PATH) -> ListRegistry:
    """Charge et valide lists.yaml contre parties.yaml ; lève à la moindre incohérence."""
    from vvs.registry import default_registry

    data = ListsFile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    errors, blocs = _validate(data, default_registry().parties)
    if errors:
        raise ValueError("lists.yaml incohérent :\n- " + "\n- ".join(errors))
    return ListRegistry(data, blocs)


# ---------------------------------------------------------------------------
# Résultats : rattachement aux unités et aux blocs
# ---------------------------------------------------------------------------


class UnknownListLabelError(ValueError):
    """Libellé du ministère absent des alias `results` de lists.yaml."""


def results_by_unit(results: pl.DataFrame, reg: ListRegistry, key: str) -> pl.DataFrame:
    """Résultats nationaux d'un scrutin agrégés par unité de résultat (level = list)."""
    nat = results.filter((pl.col("geo_level") == "national") & (pl.col("election") == key))
    if nat.is_empty():
        raise ValueError(f"{key} : pas de résultats nationaux")
    aliases = reg.result_aliases(key)
    labels = nat["label"].unique().to_list()
    unknown = sorted(lab for lab in labels if normalize_label(lab) not in aliases)
    if unknown:
        raise UnknownListLabelError(
            f"{key} : libellés du ministère absents de lists.yaml (aliases.results) : {unknown}"
        )
    unused = sorted(set(aliases) - {normalize_label(lab) for lab in labels})
    if unused:
        raise UnknownListLabelError(
            f"{key} : alias de résultats sans libellé du ministère (coquille, ou liste YAML "
            f"coupée à une virgule ?) : {unused}"
        )
    mapped = nat.with_columns(
        pl.col("label").map_elements(lambda s: aliases[normalize_label(s)], return_dtype=pl.Utf8).alias("unit")
    )
    out = (
        mapped.group_by("round", "unit")
        .agg(
            pl.col("votes").sum(),
            pl.col("expressed").first(),
            pl.col("label").sort().alias("labels"),
        )
        .with_columns(
            pl.lit(key).alias("election"),
            pl.lit(election_type(key)).alias("election_type"),
            pl.lit("list").alias("level"),
            pl.col("unit").map_elements(lambda u: reg.bloc_of(key, u), return_dtype=pl.Utf8).alias("bloc"),
            (100 * pl.col("votes") / pl.col("expressed")).alias("share"),
        )
    )
    return out.select(list(RESULTS_LISTS_BY_BLOC_SCHEMA)).cast(dict(RESULTS_LISTS_BY_BLOC_SCHEMA))


def results_lists_by_bloc(
    results: dict[str, pl.DataFrame], reg: ListRegistry | None = None
) -> pl.DataFrame:
    """Unités (level = list) et blocs (level = bloc) de chaque scrutin."""
    reg = reg or load_lists()
    frames = []
    for key, df in sorted(results.items()):
        units = results_by_unit(df, reg, key)
        blocs = (
            units.group_by("election", "election_type", "round", "bloc")
            .agg(pl.col("votes").sum(), pl.col("expressed").first(),
                 pl.col("labels").list.explode().sort().alias("labels"))
            .with_columns(
                pl.lit("bloc").alias("level"),
                pl.col("bloc").alias("unit"),
                (100 * pl.col("votes") / pl.col("expressed")).alias("share"),
            )
            .select(list(RESULTS_LISTS_BY_BLOC_SCHEMA))
            .cast(dict(RESULTS_LISTS_BY_BLOC_SCHEMA))
        )
        frames += [units, blocs]
    out = pl.concat(frames).sort("election", "round", "level", "unit")
    sums = out.group_by("election", "round", "level").agg(pl.col("votes").sum(), pl.col("expressed").first())
    # le fichier du ministère des législatives 2024 a 1 à 2 voix d'écart entre la somme des
    # nuances et les exprimés : tolérance de 10 voix, pas davantage
    bad = sums.filter((pl.col("votes") - pl.col("expressed")).abs() > VOTES_TOLERANCE)
    if bad.height:
        raise ValueError(f"somme des voix != exprimés : {bad.rows()}")
    return out


def load_results(keys, curated: Path | None = None) -> dict[str, pl.DataFrame]:
    curated = curated or paths.CURATED
    return {k: pl.read_parquet(curated / f"results_lists_{k}.parquet") for k in keys}


# ---------------------------------------------------------------------------
# Sondages : passage aux unités de résultat
# ---------------------------------------------------------------------------


def unit_poll_results(
    polls: pl.DataFrame, results: pl.DataFrame, reg: ListRegistry, key: str, stats: Counter | None = None
) -> pl.DataFrame:
    """Intentions de premier tour en unités de résultat, une hypothèse par configuration.

    Rend (poll_id, round, hypothesis_id, candidate_id, share, suspect, complete) :
    `candidate_id` est l'unité (ou la liste d'une hypothèse qui n'a pas eu lieu),
    `hypothesis_id` recalculé ; `complete` est faux pour une unité dont l'hypothèse teste
    certaines composantes mais pas toutes celles que d'autres sondages du scrutin testent
    (Nous Citoyens, non testé par Ifop en 2014, manque à l'unité divers droite). L'unité
    résiduelle est toujours complète : ce qui manque est dans « Autres »."""
    stats = stats if stats is not None else Counter()
    ids = polls.filter(pl.col("election") == key)["poll_id"]
    res = results.filter(
        pl.col("poll_id").is_in(ids.implode())
        & (pl.col("round") == 1)
        & ~pl.col("candidate_id").is_in(derived.NON_EXPRESSED)
    )
    entries = reg.for_election(key)
    hyps: dict[str, dict[str, dict[str, float]]] = {}
    for pid, hid, cid, share in res.select("poll_id", "hypothesis_id", "candidate_id", "share").iter_rows():
        hyps.setdefault(pid, {}).setdefault(hid, {})[cid] = share
    # composantes de chaque unité effectivement sondées dans ce scrutin
    tested: dict[str, set[str]] = {}
    for cid in res["candidate_id"].unique():
        if cid != "other" and entries[cid].group is not None:
            tested.setdefault(entries[cid].group, set()).add(cid)
    rows = []
    lo, hi = rules.SHARE_SUM_BOUNDS
    for pid, by_hid in sorted(hyps.items()):

        def n_components(shares: dict[str, float]) -> int:
            return sum(1 for c in shares if c != "other" and entries[c].group is not None)

        least = min(n_components(s) for s in by_hid.values())
        seen: set[str] = set()
        for hid, shares in sorted(by_hid.items()):
            if n_components(shares) > least:
                stats["hypothèses écartées (composantes séparées)"] += 1
                continue
            units: dict[str, float] = {}
            for cid, s in shares.items():
                u = reg.unit_of(key, cid)
                units[u] = units.get(u, 0.0) + s
            new_hid = schemas.hypothesis_id(list(units))
            if new_hid in seen:
                stats["hypothèses identiques une fois regroupées"] += 1
                continue
            seen.add(new_hid)
            total = sum(units.values())
            suspect = not lo <= total <= hi
            stats["hypothèses suspectes (somme)"] += suspect
            for u, s in units.items():
                present = {c for c in shares if c in tested.get(u, ())}
                complete = entries[u].residual if u in entries else True
                complete = complete or not present or present == tested[u]
                stats["unités incomplètes (composante non testée)"] += not complete
                rows.append((pid, 1, new_hid, u, s, suspect, complete))
    return pl.DataFrame(
        rows,
        schema={"poll_id": pl.Utf8, "round": pl.Int8, "hypothesis_id": pl.Utf8,
                "candidate_id": pl.Utf8, "share": pl.Float64, "suspect": pl.Boolean,
                "complete": pl.Boolean},
        orient="row",
    )


def polls_with_rolling(polls: pl.DataFrame) -> pl.DataFrame:
    """Ajoute `is_rolling` (vagues chevauchantes d'un même institut, `rules`)."""
    rolling = rules.rolling_by_regularity(polls)
    return polls.with_columns(pl.col("poll_id").is_in(list(rolling)).alias("is_rolling"))


# ---------------------------------------------------------------------------
# Erreurs
# ---------------------------------------------------------------------------


def vote_date(key: str) -> dt.date:
    from vvs.ingest.wikipedia.scrutins import scrutin

    return scrutin(key).vote


def poll_errors_lists(
    polls: pl.DataFrame,
    poll_results: pl.DataFrame,
    by_bloc: pl.DataFrame,
    reg: ListRegistry | None = None,
    keys=None,
    horizons: tuple[int, ...] = derived.HORIZONS,
    stats: Counter | None = None,
) -> pl.DataFrame:
    """`derived.poll_errors` appliqué à chaque scrutin, en unités de résultat.

    `by_bloc` : sortie de `results_lists_by_bloc` (niveau list, premier tour)."""
    reg = reg or load_lists()
    keys = sorted(keys or polls["election"].unique().to_list())
    polls = polls_with_rolling(polls)
    out = []
    for code, key in enumerate(keys, start=1):
        # derived.poll_errors exige une élection entière : code local, remplacé à la sortie
        p = polls.filter(pl.col("election") == key).with_columns(pl.lit(code).alias("election"))
        r = unit_poll_results(polls, poll_results, reg, key, stats)
        off = by_bloc.filter(
            (pl.col("election") == key) & (pl.col("level") == "list") & (pl.col("round") == 1)
        ).select(pl.lit(code).alias("election"), "round", pl.col("unit").alias("candidate_id"), "share")
        def bloc_fn(cid: str, _election: int, key: str = key) -> str:
            return reg.bloc_of(key, cid)

        days = {(code, 1): vote_date(key)}
        # blocs : toutes les unités (une composante non testée est dans « Autres » de ce
        # sondage, on ne peut pas mieux faire) ; listes : unités complètes seulement
        by_unit = derived.poll_errors(p, r, off, days, bloc_fn, horizons=horizons)
        complete = derived.poll_errors(p, r.filter("complete"), off, days, bloc_fn, horizons=horizons)
        err = pl.concat([
            by_unit.filter(pl.col("level") == "bloc"),
            complete.filter(pl.col("level") == "candidate"),
        ])
        out.append(
            err.with_columns(pl.lit(key).alias("election"), pl.lit(election_type(key)).alias("election_type"))
        )
    df = pl.concat(out).select(list(POLL_ERRORS_LISTS_SCHEMA)).cast(dict(POLL_ERRORS_LISTS_SCHEMA))
    schemas.validate_poll_errors(df.with_columns(pl.lit(0).alias("election")))
    return df


def run(curated: Path | None = None, write: bool = True) -> dict[str, pl.DataFrame]:
    """Lit polls_lists / poll_results_lists / results_lists_*, écrit les deux tables."""
    curated = curated or paths.CURATED
    reg = load_lists()
    polls = pl.read_parquet(curated / "polls_lists.parquet")
    poll_results = pl.read_parquet(curated / "poll_results_lists.parquet")
    keys = sorted(reg.elections)
    by_bloc = results_lists_by_bloc(load_results(keys, curated), reg)
    stats: Counter = Counter()
    errors = poll_errors_lists(polls, poll_results, by_bloc, reg, keys, stats=stats)
    if write:
        by_bloc.write_parquet(curated / "results_lists_by_bloc.parquet")
        errors.write_parquet(curated / "poll_errors_lists.parquet")
    return {"by_bloc": by_bloc, "errors": errors, "stats": stats}
