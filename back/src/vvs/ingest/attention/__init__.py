"""Signaux d'attention en ligne : pages vues Wikipédia et Google Trends.

Proxys d'émergence des candidats, étudiés dans `back/studies/attention.py` ; rien n'entre
dans la prévision sans gain au backtest (AGENTS.md, docs/07-evaluation.md).

Registre : `data/registry/attention_queries.yaml` (article Wikipédia FR, sujet Trends de
chaque candidat, composition des cycles, pivot Trends). Cache brut :
`data/raw/attention/<source>/`, écrit une fois et jamais réécrit (`RawCache`), avec un
journal `requests.jsonl` par source.

Sortie `parse_all()` -> `data/curated/attention.parquet` (`ATTENTION_SCHEMA`), une ligne par
(source, candidate_id, date) :

- `source` : `wikipedia` (quotidien, `freq` = `D`) ou `trends` (hebdomadaire, `freq` = `W`,
  `date` = dimanche qui ouvre la semaine Google) ;
- `value` : pages vues (agent `user`, tous accès, somme de l'article et de ses anciens
  titres, plateaux de trafic automatique remplacés, voir `wikipedia_pageviews`) ou indice
  Trends chaîné sur le pivot (voir `google_trends`) ;
- `raw_value` : la valeur avant correction des plateaux ; `suspect` : jour corrigé ;
- `election` : cycle de la date, c'est-à-dire la première élection dont le premier tour est
  postérieur ou égal à la date (2027 après le 10 avril 2022) ;
- `share` : part du candidat dans le total des membres du cycle (`cycles` du registre) ce
  jour-là ou cette semaine-là, vide si le candidat n'est pas membre du cycle de la date.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import yaml

from vvs import paths

REGISTRY_FILE = "attention_queries.yaml"
OUTPUT_NAME = "attention.parquet"
SOURCES = ("wikipedia", "trends")

ATTENTION_SCHEMA = pl.Schema(
    {
        "source": pl.Utf8,
        "candidate_id": pl.Utf8,
        "date": pl.Date,
        "freq": pl.Utf8,
        "election": pl.Int32,
        "value": pl.Float64,
        "raw_value": pl.Float64,
        "suspect": pl.Boolean,
        "share": pl.Float64,
    }
)


# ---------------------------------------------------------------------------- registre


@dataclass(frozen=True)
class AttentionQuery:
    candidate_id: str
    name: str
    wikipedia: str | None
    extra_titles: tuple[str, ...] = ()
    trends_topic: str | None = None
    trends_keyword: str | None = None

    @property
    def trends_term(self) -> str | None:
        """Terme envoyé à Trends : le sujet (mid) s'il existe, sinon le mot-clé."""
        return self.trends_topic or self.trends_keyword


@dataclass(frozen=True)
class AttentionRegistry:
    candidates: dict[str, AttentionQuery]
    cycles: dict[int, tuple[str, ...]]
    first_rounds: dict[int, date]
    trends_pivot: str
    trends_start: date
    wikipedia_start: date
    notes: dict = field(default_factory=dict)

    def cycle_of(self, d: date) -> int | None:
        """Première élection dont le premier tour est >= d (None après la dernière)."""
        for e in sorted(self.first_rounds):
            if d <= self.first_rounds[e]:
                return e
        return None

    def members(self, election: int) -> tuple[str, ...]:
        return self.cycles.get(election, ())

    def cycle_frame(self) -> pl.DataFrame:
        """(election, candidate_id) des membres de chaque cycle."""
        rows = [(e, c) for e, cs in self.cycles.items() for c in cs]
        return pl.DataFrame(rows, schema={"election": pl.Int32, "candidate_id": pl.Utf8},
                            orient="row")


def load_registry(path: Path | None = None, registry=None) -> AttentionRegistry:
    """Lit et vérifie le registre contre candidates.yaml et elections.yaml (`registry`,
    défaut : registre du projet), d'où viennent aussi les dates de premier tour."""
    path = path or paths.REGISTRY / REGISTRY_FILE
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if registry is None:
        from vvs.registry import default_registry

        registry = default_registry()
    known_ids = set(registry.candidates)
    errors = []
    cands = {}
    for cid, v in raw["candidates"].items():
        if cid not in known_ids:
            errors.append(f"{cid} absent de candidates.yaml")
        wp = v.get("wikipedia") or {}
        tr = v.get("trends") or {}
        cands[cid] = AttentionQuery(
            candidate_id=cid,
            name=v["name"],
            wikipedia=wp.get("title"),
            extra_titles=tuple(wp.get("extra_titles") or ()),
            trends_topic=tr.get("topic"),
            trends_keyword=tr.get("keyword"),
        )
    cycles = {}
    for e, ids in raw["cycles"].items():
        missing = [c for c in ids if c not in cands]
        if missing:
            errors.append(f"cycle {e} : {missing} absents de `candidates`")
        if len(set(ids)) != len(ids):
            errors.append(f"cycle {e} : doublons")
        cycles[int(e)] = tuple(ids)
    pivot = raw["trends"]["pivot"]
    if pivot not in cands or cands[pivot].trends_term is None:
        errors.append(f"pivot Trends {pivot} sans sujet ni mot-clé")
    if errors:
        raise ValueError("attention_queries.yaml : " + " ; ".join(errors))
    first_rounds = {e: el.round_(1).date for e, el in registry.elections.items()}
    return AttentionRegistry(
        candidates=cands,
        cycles=cycles,
        first_rounds=first_rounds,
        trends_pivot=pivot,
        trends_start=date.fromisoformat(str(raw["trends"]["start"])),
        wikipedia_start=date.fromisoformat(str(raw["wikipedia"]["start"])),
        notes=raw.get("notes") or {},
    )


# ---------------------------------------------------------------------------- cache brut


def attention_dir(source: str) -> Path:
    return paths.RAW / "attention" / source


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def params_key(params: dict) -> str:
    canon = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def meta_path(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


class RawCache:
    """Réponses brutes d'une source sous `root` : octets reçus tels quels, plus un
    `.meta.json` (requête, statut HTTP, heure UTC, taille, sha256). Jamais réécrites."""

    def __init__(self, source: str, root: Path | None = None):
        self.source = source
        self.root = root or attention_dir(source)

    def get(self, path: Path) -> tuple[bytes, dict] | None:
        if not path.exists():
            return None
        content = path.read_bytes()
        meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
        digest = hashlib.sha256(content).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError(f"{path}: sha256 {digest} différent de la méta {meta['sha256']}")
        return content, meta

    def put(self, path: Path, content: bytes, request: dict, status: int = 200) -> dict:
        if path.exists():
            raise FileExistsError(f"{path} existe déjà : le cache brut n'est jamais réécrit")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "request": request,
            "status": status,
            "retrieved_at": now_utc().isoformat(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(content)
        tmp.rename(path)
        meta_path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        return meta

    def log(self, **event) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        event = {"at": now_utc().isoformat(), **event}
        with (self.root / "requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def files(self, sub: str, pattern: str = "*.json") -> list[Path]:
        folder = self.root / sub
        if not folder.exists():
            return []
        return sorted(p for p in folder.rglob(pattern) if not p.name.endswith(".meta.json"))


# ---------------------------------------------------------------------------- curation


def with_cycle_share(df: pl.DataFrame, reg: AttentionRegistry) -> pl.DataFrame:
    """Ajoute `election` (cycle de la date) et `share` (part parmi les membres du cycle
    présents dans la source ce jour-là ; vide hors cycle). `df` : source, candidate_id,
    date, freq, value, et le cas échéant raw_value, suspect."""
    if "raw_value" not in df.columns:
        df = df.with_columns(pl.col("value").alias("raw_value"))
    if "suspect" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("suspect"))
    dates = df.select("date").unique()
    dates = dates.with_columns(
        pl.col("date").map_elements(reg.cycle_of, return_dtype=pl.Int32).alias("election")
    )
    out = df.join(dates, on="date", how="left")
    members = reg.cycle_frame().with_columns(pl.lit(True).alias("_member"))
    out = out.join(members, on=["election", "candidate_id"], how="left")
    total = (
        out.filter(pl.col("_member").fill_null(False))
        .group_by("source", "date")
        .agg(pl.col("value").sum().alias("_total"))
    )
    out = out.join(total, on=["source", "date"], how="left").with_columns(
        pl.when(pl.col("_member").fill_null(False) & (pl.col("_total") > 0))
        .then(pl.col("value") / pl.col("_total"))
        .otherwise(None)
        .alias("share")
    )
    return out.select(ATTENTION_SCHEMA.names()).cast(ATTENTION_SCHEMA).sort(
        "source", "candidate_id", "date"
    )


@dataclass
class ParseReport:
    rows: dict[str, int] = field(default_factory=dict)
    candidates: dict[str, int] = field(default_factory=dict)
    span: dict[str, tuple[date, date]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    path: Path | None = None

    def summary(self) -> str:
        lines = []
        for s, n in self.rows.items():
            d0, d1 = self.span.get(s, (None, None))
            lines.append(f"{s} : {n} lignes, {self.candidates.get(s, 0)} candidats, {d0} -> {d1}")
        lines += [f"  {n}" for n in self.notes]
        if self.path:
            lines.append(f"écrit : {self.path}")
        return "\n".join(lines)


def parse_all(sources: list[str] | None = None, reg: AttentionRegistry | None = None,
              write: bool = True) -> tuple[pl.DataFrame, ParseReport]:
    """Cache brut -> attention.parquet. Une source sans cache est ignorée (note) ; une
    source non demandée garde ses lignes du parquet existant."""
    from vvs.ingest.attention import google_trends, wikipedia_pageviews

    reg = reg or load_registry()
    sources = sources or list(SOURCES)
    rep = ParseReport()
    out_path = paths.CURATED / OUTPUT_NAME
    parts = []
    if out_path.exists():
        old = pl.read_parquet(out_path)
        parts.append(old.filter(~pl.col("source").is_in(sources)))
    for s in sources:
        if s == "wikipedia":
            daily, notes = wikipedia_pageviews.parse_cached(reg)
        elif s == "trends":
            daily, notes = google_trends.parse_cached(reg)
        else:
            raise ValueError(f"source inconnue : {s}")
        rep.notes += notes
        if daily.height == 0:
            rep.notes.append(f"{s} : aucun fichier brut, lignes existantes conservées")
            if out_path.exists():
                parts.append(old.filter(pl.col("source") == s))
            continue
        parts.append(with_cycle_share(daily, reg))
    df = (pl.concat(parts, how="diagonal_relaxed").select(ATTENTION_SCHEMA.names())
          .cast(ATTENTION_SCHEMA) if parts else pl.DataFrame(schema=ATTENTION_SCHEMA))
    for s, g in df.group_by("source"):
        rep.rows[s[0]] = g.height
        rep.candidates[s[0]] = g["candidate_id"].n_unique()
        rep.span[s[0]] = (g["date"].min(), g["date"].max())
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        rep.path = out_path
    return df, rep
