"""Résultats électoraux officiels du ministère de l'Intérieur (data.gouv.fr).

Deux étapes, pilotées par data/registry/interieur_sources.yaml :

- `ingest(election)` télécharge les fichiers listés dans data/raw/results/<election>/ ; un
  fichier présent n'est jamais réécrit, son empreinte sha256 est vérifiée ;
- `parse(election)` produit data/curated/results_<election>.parquet (présidentielles,
  schéma `RESULTS_SCHEMA`) ou data/curated/results_lists_<election>.parquet (européennes et
  législatives, niveau national, schéma `RESULTS_LISTS_SCHEMA`).

Présidentielles : niveaux `national` (geo_code `FE`), `department` (code INSEE, `ZZ` pour
l'étranger) et `commune` (voir `geo`). Le niveau département vient du fichier de rôle
`department` s'il existe (2002, 2007 : les fichiers par bureau n'y couvrent pas tout
l'outre-mer ni l'étranger), sinon de l'agrégation des communes. Le national est la somme des
départements ; si un fichier de rôle `national` existe (2022), la somme doit l'égaler
exactement.

Les chiffres sont ceux du ministère, qui diffèrent parfois de quelques centaines de voix de
la proclamation du Conseil constitutionnel (annulations de bureaux) : voir les tests.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from vvs import paths
from vvs.ingest.interieur import formats, geo
from vvs.ingest.interieur.candidates import candidate_id
from vvs.schemas import RESULTS_SCHEMA, validate_results

SOURCES_FILE = paths.REGISTRY / "interieur_sources.yaml"
NATIONAL_CODE = "FE"
PRESIDENTIAL = "presidential"
LIST_KINDS = ("european", "legislative")
USER_AGENT = "vingtvingtsept/0.0.1"

RESULTS_LISTS_SCHEMA = pl.Schema(
    {
        "election": pl.Utf8,
        "round": pl.Int8,
        "geo_level": pl.Utf8,
        "geo_code": pl.Utf8,
        "label": pl.Utf8,
        "nuance": pl.Utf8,
        "votes": pl.Int64,
        "registered": pl.Int64,
        "voters": pl.Int64,
        "expressed": pl.Int64,
    }
)


class ConsistencyError(ValueError):
    pass


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    format: str
    role: str
    rounds: tuple[int, ...]
    sha256: str | None
    dataset: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ElectionSources:
    key: str
    kind: str
    year: int
    files: tuple[SourceFile, ...]
    # Écart connu, interne au fichier source, entre somme des voix et exprimés, par tour :
    # sum(votes) - expressed. Par défaut 0 ; toute autre valeur fait échouer le parsing.
    vote_sum_gaps: dict[int, int] = field(default_factory=dict)


def load_sources(path: Path | None = None) -> dict[str, ElectionSources]:
    data = yaml.safe_load((path or SOURCES_FILE).read_text(encoding="utf-8"))
    out = {}
    for key, e in data["elections"].items():
        key = str(key)
        files = tuple(
            SourceFile(
                name=f["name"],
                url=f["url"],
                format=f["format"],
                role=f["role"],
                rounds=tuple(f["rounds"]),
                sha256=f.get("sha256"),
                dataset=f.get("dataset", ""),
                notes=f.get("notes", ""),
            )
            for f in e["files"]
        )
        if e["kind"] != PRESIDENTIAL and e["kind"] not in LIST_KINDS:
            raise ValueError(f"{key}: kind inconnu {e['kind']!r}")
        gaps = {int(k): int(v) for k, v in (e.get("vote_sum_gaps") or {}).items()}
        out[key] = ElectionSources(
            key=key, kind=e["kind"], year=int(e["year"]), files=files, vote_sum_gaps=gaps
        )
    return out


def get_election(
    election: str, sources: dict[str, ElectionSources] | None = None
) -> ElectionSources:
    sources = sources if sources is not None else load_sources()
    if election not in sources:
        raise KeyError(f"élection {election!r} absente de {SOURCES_FILE.name} ({sorted(sources)})")
    return sources[election]


def raw_dir(election: str) -> Path:
    return paths.RAW / "results" / election


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_hash(path: Path, f: SourceFile) -> str:
    digest = sha256_of(path)
    if f.sha256 and digest != f.sha256:
        raise ConsistencyError(
            f"{path}: sha256 {digest} ≠ {f.sha256} attendu (fichier tronqué ou modifié ?)"
        )
    return digest


# --- Ingestion ---------------------------------------------------------------------------


def _download(url: str, dest: Path, retries: int = 4) -> None:
    import httpx

    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=120,
                headers={"User-Agent": USER_AGENT},
            ) as r:  # fmt: skip
                r.raise_for_status()
                expected = r.headers.get("content-length")
                with part.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            if expected is not None and part.stat().st_size != int(expected):
                raise OSError(f"taille {part.stat().st_size} ≠ content-length {expected}")
            part.rename(dest)
            return
        except (httpx.HTTPError, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"échec du téléchargement de {url} : {last}")


def ingest(
    election: str, sources: dict[str, ElectionSources] | None = None
) -> list[tuple[str, str]]:
    """Télécharge les fichiers de l'élection. Renvoie [(nom, statut)] avec statut
    `downloaded` ou `present`. Les téléchargements sont faits en série (des téléchargements
    parallèles depuis static.data.gouv.fr ont donné des fichiers tronqués)."""
    e = get_election(election, sources)
    d = raw_dir(election)
    d.mkdir(parents=True, exist_ok=True)
    report = []
    for f in e.files:
        dest = d / f.name
        if dest.exists():
            _check_hash(dest, f)
            report.append((f.name, "present"))
            continue
        _download(f.url, dest)
        try:
            digest = _check_hash(dest, f)
        except ConsistencyError:
            dest.unlink()  # fichier tout juste téléchargé et invalide : on ne le garde pas
            raise
        meta = {
            "url": f.url,
            "dataset": f.dataset,
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        dest.with_name(dest.name + ".source.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report.append((f.name, "downloaded"))
    return report


# --- Parsing des présidentielles ----------------------------------------------------------


def _read_units(e: ElectionSources, role: str, base: Path) -> pl.DataFrame | None:
    files = [f for f in e.files if f.role == role]
    if not files:
        return None
    parts = []
    for f in files:
        path = base / f.name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} absent : lancer `vvs ingest interieur --election {e.key}`"
            )
        _check_hash(path, f)
        reader = formats.PRESIDENTIAL_READERS.get(f.format)
        if reader is None:
            raise ValueError(f"{f.name}: format {f.format!r} non géré pour une présidentielle")
        parts.append(reader(path, f.rounds))
    return pl.concat(parts)


def _attach_candidates(units: pl.DataFrame) -> pl.DataFrame:
    mapping = pl.DataFrame(
        [(lab, candidate_id(lab)) for lab in units["label"].unique().sort().to_list()],
        schema=["label", "candidate_id"],
        orient="row",
    )
    return units.join(mapping, on="label", how="left")


def _unit_totals(units: pl.DataFrame) -> pl.DataFrame:
    """Une ligne par (tour, unité), en vérifiant que les totaux répétés sont cohérents."""
    totals = units.group_by("round", "unit").agg(
        pl.col("dept_src").first(),
        pl.col("commune_src").first(),
        *(pl.col(c).first() for c in ("registered", "voters", "expressed")),
        *(pl.col(c).n_unique().alias(f"_n_{c}") for c in ("registered", "voters", "expressed")),
    )
    bad = totals.filter(pl.any_horizontal(pl.col("^_n_.*$") > 1))
    if bad.height:
        raise ConsistencyError(f"totaux d'unité incohérents : {bad.head(5)}")
    return totals.drop(pl.col("^_n_.*$"))


def _aggregate(units: pl.DataFrame, geo_col: str, level: str) -> pl.DataFrame:
    """Agrège la table unités (avec `candidate_id` et la colonne `geo_col`) au niveau voulu."""
    totals = (
        _unit_totals(units)
        .join(units.select("round", "unit", geo_col).unique(), on=["round", "unit"])
        .group_by("round", geo_col)
        .agg(pl.col("registered").sum(), pl.col("voters").sum(), pl.col("expressed").sum())
    )
    votes = units.group_by("round", geo_col, "candidate_id").agg(pl.col("votes").sum())
    return votes.join(totals, on=["round", geo_col]).select(
        pl.col("round"),
        pl.lit(level).alias("geo_level"),
        pl.col(geo_col).alias("geo_code"),
        "candidate_id",
        "votes",
        "registered",
        "voters",
        "expressed",
    )


def _with_geo(units: pl.DataFrame) -> pl.DataFrame:
    pairs = units.select("dept_src", "commune_src").unique()
    rows = []
    for d, c in pairs.iter_rows():
        rows.append((d, c, geo.department(d, c), geo.commune(d, c) if c is not None else None))
    mapping = pl.DataFrame(
        rows,
        schema={
            "dept_src": pl.Utf8,
            "commune_src": pl.Utf8,
            "department": pl.Utf8,
            "commune": pl.Utf8,
        },
        orient="row",
    )
    return units.join(mapping, on=["dept_src", "commune_src"], how="left", nulls_equal=True)


def _check_sums(df: pl.DataFrame, what: str) -> None:
    """La somme des voix des candidats doit égaler les exprimés de chaque unité géographique."""
    s = df.group_by("round", "geo_level", "geo_code").agg(
        pl.col("votes").sum().alias("sum_votes"), pl.col("expressed").first()
    )
    bad = s.filter(pl.col("sum_votes") != pl.col("expressed"))
    if bad.height:
        raise ConsistencyError(
            f"{what}: somme des voix ≠ exprimés pour {bad.height} unités : {bad.head(5)}"
        )


def parse_presidential(election: str, base: Path | None = None,
                       sources: dict[str, ElectionSources] | None = None) -> pl.DataFrame:  # fmt: skip
    e = get_election(election, sources)
    if e.kind != PRESIDENTIAL:
        raise ValueError(f"{election} n'est pas une présidentielle")
    base = base or raw_dir(election)
    communes = _read_units(e, "commune", base)
    departments = _read_units(e, "department", base)
    national_src = _read_units(e, "national", base)

    parts = []
    if communes is not None:
        communes = _with_geo(_attach_candidates(communes))
        parts.append(_aggregate(communes, "commune", "commune"))
    if departments is not None:
        departments = _with_geo(_attach_candidates(departments))
        dept = _aggregate(departments, "department", "department")
    elif communes is not None:
        dept = _aggregate(communes, "department", "department")
    else:
        raise ValueError(f"{election}: ni fichier département ni fichier commune")
    parts.append(dept)

    national = (
        dept.group_by("round", "candidate_id")
        .agg(pl.col("votes").sum())
        .join(
            dept.select("round", "geo_code", "registered", "voters", "expressed")
            .unique()
            .group_by("round")
            .agg(pl.col("registered").sum(), pl.col("voters").sum(), pl.col("expressed").sum()),
            on="round",
        )
        .with_columns(
            pl.lit("national").alias("geo_level"), pl.lit(NATIONAL_CODE).alias("geo_code")
        )
    )
    parts.append(national.select(dept.columns))

    if national_src is not None:
        ref = _attach_candidates(national_src)
        ref = ref.with_columns(pl.lit(NATIONAL_CODE).alias("geo"))
        ref = _aggregate(ref, "geo", "national")
        cols = ["round", "candidate_id", "votes", "registered", "voters", "expressed"]
        mine = national.select(cols).sort("round", "candidate_id")
        theirs = ref.select(cols).sort("round", "candidate_id")
        if not mine.equals(theirs):
            diff = mine.join(theirs, on=["round", "candidate_id"], how="full", suffix="_ref")
            raise ConsistencyError(
                f"{election}: national agrégé ≠ fichier France entière :\n{diff}"
            )

    out = pl.concat(parts, how="vertical_relaxed").with_columns(pl.lit(e.year).alias("election"))
    _check_sums(out, election)
    out = validate_results(out).sort("round", "geo_level", "geo_code", "candidate_id")
    return out


# --- Parsing des scrutins de listes ------------------------------------------------------


def parse_lists(election: str, base: Path | None = None,
                sources: dict[str, ElectionSources] | None = None) -> pl.DataFrame:  # fmt: skip
    """Européennes et législatives : niveau national, une ligne par liste ou nuance."""
    e = get_election(election, sources)
    if e.kind not in LIST_KINDS:
        raise ValueError(f"{election} n'est pas un scrutin de listes ou législatif")
    base = base or raw_dir(election)
    parts = []
    for f in e.files:
        path = base / f.name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} absent : lancer `vvs ingest interieur --election {election}`"
            )
        _check_hash(path, f)
        reader = formats.LIST_READERS.get(f.format)
        if reader is None:
            raise ValueError(f"{f.name}: format {f.format!r} non géré pour {e.kind}")
        parts.append(reader(path, f.rounds))
    units = pl.concat(parts)
    # Une clé doit garder le même libellé et la même nuance partout
    amb = (
        units.group_by("round", "key")
        .agg(
            pl.col("label").n_unique().alias("n_label"),
            pl.col("nuance").n_unique().alias("n_nuance"),
        )
        .filter((pl.col("n_label") > 1) | (pl.col("n_nuance") > 1))
    )
    if amb.height:
        raise ConsistencyError(f"{election}: libellés ambigus pour une même liste : {amb}")
    totals = (
        units.select("round", "unit", "registered", "voters", "expressed")
        .unique()
        .group_by("round")
        .agg(pl.col("registered").sum(), pl.col("voters").sum(), pl.col("expressed").sum())
    )
    if (
        units.select("round", "unit").unique().height
        != units.select("round", "unit", "registered", "voters", "expressed").unique().height
    ):
        raise ConsistencyError(f"{election}: totaux d'unité incohérents")
    out = (
        units.group_by("round", "key")
        .agg(pl.col("label").first(), pl.col("nuance").first(), pl.col("votes").sum())
        .join(totals, on="round")
        .with_columns(
            pl.lit(election).alias("election"),
            pl.lit("national").alias("geo_level"),
            pl.lit(NATIONAL_CODE).alias("geo_code"),
        )
    )
    gaps = out.group_by("round").agg(
        (pl.col("votes").sum() - pl.col("expressed").first()).alias("gap")
    )
    bad = [
        (r, g, e.vote_sum_gaps.get(r, 0))
        for r, g in gaps.iter_rows()
        if g != e.vote_sum_gaps.get(r, 0)
    ]
    if bad:
        raise ConsistencyError(
            f"{election}: somme des voix - exprimés (tour, constaté, attendu) : {bad}"
        )
    return (
        out.select(list(RESULTS_LISTS_SCHEMA.keys()))
        .cast(dict(RESULTS_LISTS_SCHEMA))
        .sort("round", "votes", descending=[False, True])
    )


# --- Point d'entrée ----------------------------------------------------------------------


def parse(election: str) -> tuple[Path, pl.DataFrame]:
    """Parse une élection et écrit le parquet curated. Renvoie (chemin, table)."""
    e = get_election(election)
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    if e.kind == PRESIDENTIAL:
        df = parse_presidential(election)
        out = paths.CURATED / f"results_{election}.parquet"
    else:
        df = parse_lists(election)
        out = paths.CURATED / f"results_lists_{election}.parquet"
    df.write_parquet(out)
    return out, df


__all__ = [
    "RESULTS_LISTS_SCHEMA",
    "RESULTS_SCHEMA",
    "ConsistencyError",
    "ingest",
    "load_sources",
    "parse",
    "parse_lists",
    "parse_presidential",
]
