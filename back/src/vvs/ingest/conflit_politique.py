"""Indicateurs socio-économiques communaux de Cagé et Piketty (2023).

Deux étapes, pilotées par data/registry/conflit_politique.yaml :

- `ingest()` télécharge les archives listées dans data/raw/conflit_politique/ avec un
  `.meta.json` (URL, heure UTC, taille, sha256, licence, citation) ; un fichier présent n'est
  jamais réécrit, son empreinte est vérifiée contre le registre ;
- `parse()` lit les membres CSV des archives et écrit data/curated/communes_socio.parquet.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from vvs import paths

REGISTRY_FILE = paths.REGISTRY / "conflit_politique.yaml"
USER_AGENT = "vingtvingtsept/0.0.1 (+https://github.com/Orolol)"


class ConsistencyError(ValueError):
    pass


@dataclass(frozen=True)
class SourceFile:
    key: str
    name: str
    url: str
    members: tuple[str, ...]
    sha256: str | None
    bytes: int | None


@dataclass(frozen=True)
class Registry:
    title: str
    url: str
    licence: str
    citation: str
    years: tuple[int, ...]
    files: tuple[SourceFile, ...]

    def file(self, key: str) -> SourceFile:
        for f in self.files:
            if f.key == key:
                return f
        raise KeyError(f"archive {key!r} absente de {REGISTRY_FILE.name}")


def load_registry(path: Path | None = None) -> Registry:
    data = yaml.safe_load((path or REGISTRY_FILE).read_text(encoding="utf-8"))
    src = data["source"]
    files = tuple(
        SourceFile(
            key=f["key"],
            name=f["name"],
            url=f["url"],
            members=tuple(f["members"]),
            sha256=f.get("sha256"),
            bytes=f.get("bytes"),
        )
        for f in data["files"]
    )
    keys = [f.key for f in files]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{REGISTRY_FILE.name} : clé d'archive en double")
    return Registry(
        title=src["title"],
        url=src["url"],
        licence=" ".join(src["licence"].split()),
        citation=" ".join(src["citation"].split()),
        years=tuple(int(y) for y in data["years"]),
        files=files,
    )


def raw_dir() -> Path:
    return paths.RAW / "conflit_politique"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: Path, f: SourceFile) -> str:
    digest = sha256_of(path)
    if f.sha256 and digest != f.sha256:
        raise ConsistencyError(f"{path}: sha256 {digest} ≠ {f.sha256} attendu")
    if f.bytes and path.stat().st_size != f.bytes:
        raise ConsistencyError(f"{path}: {path.stat().st_size} octets ≠ {f.bytes} attendus")
    return digest


def _download(url: str, dest: Path, retries: int = 4) -> None:
    import httpx

    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=300,
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


def ingest(keys: list[str] | None = None, registry: Registry | None = None) -> list[tuple[str, str, str]]:
    """Télécharge les archives (toutes, ou `keys`). Renvoie [(nom, statut, sha256)] avec
    statut `downloaded` ou `present`."""
    reg = registry or load_registry()
    d = raw_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for f in reg.files:
        if keys and f.key not in keys:
            continue
        dest = d / f.name
        if dest.exists():
            out.append((f.name, "present", _check(dest, f)))
            continue
        _download(f.url, dest)
        try:
            digest = _check(dest, f)
        except ConsistencyError:
            dest.unlink()  # tout juste téléchargé et invalide : on ne le garde pas
            raise
        meta = {
            "url": f.url,
            "page": reg.url,
            "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "bytes": dest.stat().st_size,
            "sha256": digest,
            "members": list(f.members),
            "licence": reg.licence,
            "citation": reg.citation,
        }
        dest.with_name(dest.name + ".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        out.append((f.name, "downloaded", digest))
    return out


# --- Extraction ----------------------------------------------------------------------------

# Variables d'origine retenues par archive (radical sans l'année ; le fichier porte une colonne
# `<radical><AAAA>` par année). Définitions d'après les do-files des auteurs (codes
# informatiques du livre, annexe B) ; les `per*` sont des rangs (percentiles 0-1) des
# communes, pondérés par la population, calculés par les auteurs.
ORIGINAL_VARIABLES: dict[str, tuple[str, ...]] = {
    # revratio : revenu moyen par habitant / moyenne France entière ; revmoy : revenu moyen
    # par habitant (euros 2020) ; revmoyadu : par adulte de 20 ans et plus ; perrev : rang
    "revenus": ("revratio", "revmoy", "revmoyadu", "perrev"),
    # psup, pbac : parts de diplômés du supérieur, du bac (population de 25 ans et plus) ;
    # persup : rang
    "diplomes": ("psup", "pbac", "persup"),
    # ppropri : part des ménages propriétaires de leur logement ; perpropri : rang
    "proprietaires": ("ppropri", "perpropri"),
    # parts de la population active (25-54 ans) : agriculteurs, indépendants, cadres,
    # professions intermédiaires, employés, ouvriers, ouvriers + employés, chômeurs
    "csp": ("pagri", "pindp", "pcadr", "ppint", "pempl", "pouvr", "pouem", "pchom", "percadr", "perouem"),
    # age : âge moyen ; prop60p, prop014 : parts des 60 ans et plus, des moins de 15 ans
    "age": ("age", "prop60p", "prop014", "perage"),
    # pop : population de la commune ; popagglo : population de son agglomération (unité
    # urbaine 2020, la commune seule hors agglomération) ; percommu, peragglo : rangs
    "agglo": ("pop", "popagglo", "percommu", "peragglo"),
}
# Colonnes fixes (sans année) gardées depuis l'archive agglo
AGGLO_STATIC = ("dep", "nomcommune", "codeagglo", "nomagglo", "multicommune")
ID = "codecommune"

# Indicateurs retenus (noms anglais) -> variable d'origine
RETAINED: dict[str, str] = {
    "income_ratio": "revratio",
    "income_mean": "revmoy",
    "income_rank": "perrev",
    "higher_edu_share": "psup",
    "owner_share": "ppropri",
    "workers_employees_share": "pouem",
    "workers_share": "pouvr",
    "employees_share": "pempl",
    "managers_share": "pcadr",
    "mean_age": "age",
    "share_60plus": "prop60p",
    "population": "pop",
    "agglo_population": "popagglo",
}
# Type de commune selon la population de son agglomération (seuils de l'INSEE pour l'unité
# urbaine, 2 000 habitants ; 100 000 pour les grandes agglomérations, comme Cagé et Piketty)
URBAN_THRESHOLDS = (2_000, 100_000)
URBAN_TYPES = ("rural", "town", "metro")

# Arrondissements de Paris, Lyon et Marseille : nos résultats portent sur la commune entière,
# qui a sa propre ligne dans les fichiers des auteurs ; les arrondissements sont écartés.
PLM_ARRONDISSEMENTS = re.compile(r"^(751(0[1-9]|1[0-9]|20)|132(0[1-9]|1[0-6])|6938[1-9])$")


def _read_member(path: Path, member: str, stems: tuple[str, ...], years: tuple[int, ...],
                 static: tuple[str, ...] = ()) -> pl.DataFrame:
    """Colonnes `<radical><année>` d'un CSV large de l'archive, passées en format long
    (codecommune, year, radicaux) ; les colonnes absentes du fichier font échouer."""
    with zipfile.ZipFile(path) as z, z.open(member) as fh:
        header = fh.readline().decode("utf-8").strip().split(",")
        wanted = [ID, *static, *(f"{s}{y}" for s in stems for y in years)]
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ConsistencyError(f"{path.name}/{member} : colonnes absentes {missing[:10]}")
        content = z.read(member)
    df = pl.read_csv(io.BytesIO(content), columns=wanted, infer_schema=False)
    df = df.with_columns(pl.col(ID).str.strip_chars())
    if df[ID].is_duplicated().any():
        raise ConsistencyError(f"{member} : codecommune en double")
    frames = []
    for y in years:
        frames.append(
            df.select(
                ID, *static,
                pl.lit(y, dtype=pl.Int16).alias("year"),
                *(pl.col(f"{s}{y}").cast(pl.Float64, strict=True).alias(s) for s in stems),
            )
        )
    return pl.concat(frames)


def urban_type(agglo_population: pl.Expr) -> pl.Expr:
    lo, hi = URBAN_THRESHOLDS
    return (
        pl.when(agglo_population.is_null()).then(None)
        .when(agglo_population < lo).then(pl.lit(URBAN_TYPES[0]))
        .when(agglo_population < hi).then(pl.lit(URBAN_TYPES[1]))
        .otherwise(pl.lit(URBAN_TYPES[2]))
    )  # fmt: skip


def _city_of(code: pl.Expr) -> pl.Expr:
    """Code de la commune entière d'un arrondissement de Paris, Marseille ou Lyon."""
    return (
        pl.when(code.str.starts_with("751")).then(pl.lit("75056"))
        .when(code.str.starts_with("132")).then(pl.lit("13055"))
        .otherwise(pl.lit("69123"))
    )  # fmt: skip


def fill_plm(df: pl.DataFrame, variables: list[str]) -> pl.DataFrame:
    """Complète la ligne commune entière de Paris, Marseille et Lyon par la moyenne de ses
    arrondissements pondérée par leur population (`pop`), variable par variable et année par
    année, là où les auteurs ne fournissent que les arrondissements (Paris dans les fichiers
    du recensement), puis écarte les arrondissements."""
    arr = df.filter(pl.col(ID).str.contains(PLM_ARRONDISSEMENTS.pattern))
    if arr.height:
        w = pl.col("pop")
        means = (
            arr.with_columns(_city_of(pl.col(ID)).alias("_city"))
            .group_by("_city", "year")
            .agg(
                *(
                    ((pl.col(v) * w).sum() / w.filter(pl.col(v).is_not_null()).sum()).alias(f"_{v}")
                    for v in variables
                )
            )
        )
        df = df.join(means, left_on=[ID, "year"], right_on=["_city", "year"], how="left")
        df = df.with_columns(
            *(pl.coalesce(pl.col(v), pl.col(f"_{v}")).alias(v) for v in variables)
        ).drop([f"_{v}" for v in variables])
    return df.filter(~pl.col(ID).str.contains(PLM_ARRONDISSEMENTS.pattern))


def build(registry: Registry | None = None) -> pl.DataFrame:
    """Table longue (codecommune, year) : variables d'origine et indicateurs retenus.

    Géographie de référence : celle du fichier des populations des auteurs (archive agglo,
    base INSEE des populations historiques en géographie récente, environ 34 800 communes de
    métropole, sans l'outre-mer). Les autres fichiers contiennent aussi d'anciennes communes,
    fusionnées depuis, dont les indicateurs récents sont vides ou figés : elles sont écartées.
    """
    reg = registry or load_registry()
    d = raw_dir()
    parts: dict[str, pl.DataFrame] = {}
    for f in reg.files:
        path = d / f.name
        if not path.exists():
            raise FileNotFoundError(f"{path} absent : lancer `vvs ingest conflit-politique`")
        _check(path, f)
        static = AGGLO_STATIC if f.key == "agglo" else ()
        parts[f.key] = _read_member(path, f.members[0], ORIGINAL_VARIABLES[f.key], reg.years, static)
    out = parts.pop("agglo")
    for part in parts.values():
        out = out.join(part, on=[ID, "year"], how="left")
    variables = [s for k, stems in ORIGINAL_VARIABLES.items() if k != "agglo" for s in stems]
    out = fill_plm(out, variables)
    out = out.with_columns(
        *(pl.col(src).alias(dst) for dst, src in RETAINED.items()),
    ).with_columns(urban_type(pl.col("agglo_population")).alias("urban_type"))
    return out.sort([ID, "year"])


def curated_path() -> Path:
    return paths.CURATED / "communes_socio.parquet"


def parse(registry: Registry | None = None) -> tuple[Path, pl.DataFrame]:
    df = build(registry)
    out = curated_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out, df
