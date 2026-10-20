"""Séries économiques et de popularité (docs/05-economie.md), pilotées par
data/registry/macro_series.yaml.

- `ingest(series_ids)` télécharge chaque série dans data/raw/macro/<raw_dir>/<AAAA-MM-JJ>.<ext>
  (INSEE : SDMX-ML ; DARES : CSV ; ministère de la Transition écologique : xlsx ; Ifop : liste
  JSON + PDF) ; rien n'est réécrit ;
- `parse(series_ids)` lit le dernier fichier brut de chaque série et écrit
  data/curated/macro.parquet (`MACRO_SCHEMA`), une ligne par série et par période depuis
  `START`. Les fréquences d'origine sont conservées (trimestriel reste trimestriel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

from vvs import paths
from vvs.ingest.macro import dares, fuel, ifop, insee, raw
from vvs.ingest.macro.registry import Series, load

START = date(2000, 1, 1)
OUTPUT_NAME = "macro.parquet"

MACRO_SCHEMA = pl.Schema(
    {
        "series_id": pl.Utf8,
        "date": pl.Date,
        "value": pl.Float64,
        "freq": pl.Utf8,
        "source": pl.Utf8,
        "retrieved_at": pl.Datetime("us"),
    }
)

__all__ = ["MACRO_SCHEMA", "OUTPUT_NAME", "START", "ingest", "parse", "validate_macro"]


def ingest(series_ids: list[str] | None = None, log=print) -> list[tuple[str, str]]:
    """Télécharge les séries demandées (toutes par défaut). Renvoie [(series_id, statut)]."""
    reg = load()
    selected = reg.select(series_ids)
    statuses: list[tuple[str, str]] = []
    done_dirs: dict[str, str] = {}
    with raw.client() as http:
        for s in selected:
            if s.raw_dir in done_dirs:  # les deux séries Ifop partagent la collection
                statuses.append((s.series_id, done_dirs[s.raw_dir]))
                continue
            if s.source == "insee":
                written = insee.ingest(http, s)
            elif s.source == "dares":
                written = dares.ingest(http, s)
            elif s.source == "mte":
                written = fuel.ingest(http, s, START)
            elif s.source == "ifop":
                written, failures = ifop.ingest(http, reg.officeholders, log=log)
                if failures:
                    log(f"Ifop : {len(failures)} publication(s) en échec, hors de la liste du jour")
            else:  # pragma: no cover - le registre valide déjà la source
                raise ValueError(s.source)
            status = (
                f"écrit {written.path.relative_to(paths.ROOT)}"
                if written
                else "déjà présent aujourd'hui"
            )
            done_dirs[s.raw_dir] = status
            statuses.append((s.series_id, status))
            log(f"{s.series_id:48s} {status}")
    return statuses


@dataclass
class ParseReport:
    rows: dict[str, tuple[int, date, date]] = field(default_factory=dict)  # n, première, dernière
    gaps: dict[str, list[date]] = field(default_factory=dict)
    ifop: ifop.IfopParse | None = None

    def summary(self) -> str:
        lines = []
        for sid, (n, first, last) in self.rows.items():
            gaps = self.gaps.get(sid, [])
            gap_txt = f", {len(gaps)} période(s) manquante(s)" if gaps else ""
            lines.append(f"{sid:48s} {n:5d} lignes {first} -> {last}{gap_txt}")
        if self.ifop is not None:
            r = self.ifop.report
            lines.append(
                f"Ifop : {len(self.ifop.publications)} PDF, {len(self.ifop.conflicts)} désaccord(s) "
                f"entre publications, {len(r.rows_out_of_tenure)} ligne(s) hors mandat rejetée(s), "
                f"{r.rows_without_person} ligne(s) sans titulaire identifié, "
                f"{len(r.rows_unreadable)} ligne(s) de mois illisible(s), "
                f"{len(r.headlines_unreadable)} chiffre(s) du mois illisible(s)"
            )
        return "\n".join(lines)


def parse(
    series_ids: list[str] | None = None, write: bool = True
) -> tuple[pl.DataFrame, ParseReport]:
    reg = load()
    selected = reg.select(series_ids)
    report = ParseReport()
    frames = []
    ifop_series = [s for s in selected if s.source == "ifop"]
    if ifop_series:
        report.ifop = ifop.parse_collection(reg.officeholders)
        frames.append(_ifop_frame(ifop_series, report.ifop))
    for s in selected:
        if s.source != "ifop":
            frames.append(_series_frame(s))
    df = pl.concat(frames) if frames else pl.DataFrame(schema=MACRO_SCHEMA)
    df = df.filter(pl.col("date") >= START).sort("series_id", "date")
    validate_macro(df)
    for s in selected:
        sub = df.filter(pl.col("series_id") == s.series_id)
        if sub.height == 0:
            raise ValueError(f"{s.series_id}: aucune ligne après {START}")
        report.rows[s.series_id] = (sub.height, sub["date"].min(), sub["date"].max())
        report.gaps[s.series_id] = missing_periods(sub["date"].to_list(), s.freq)
    if write:
        if series_ids:  # mise à jour partielle : on remplace seulement les séries demandées
            out = paths.CURATED / OUTPUT_NAME
            if out.exists():
                old = pl.read_parquet(out).filter(
                    ~pl.col("series_id").is_in([s.series_id for s in selected])
                )
                df = pl.concat([old, df]).sort("series_id", "date")
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(paths.CURATED / OUTPUT_NAME)
    return df, report


def _series_frame(s: Series) -> pl.DataFrame:
    if s.source == "insee":
        f = raw.latest_raw(s.raw_dir, "xml")
        freq, obs = insee.parse(f.path.read_bytes(), s.source_id)
        if freq != s.freq:
            raise ValueError(f"{s.series_id}: fréquence INSEE {freq}, registre {s.freq}")
    elif s.source == "dares":
        f = raw.latest_raw(s.raw_dir, "csv")
        obs = dares.parse(f.path.read_bytes(), s.params["champ"])
    elif s.source == "mte":
        f = raw.latest_raw(s.raw_dir, "xlsx")
        obs = fuel.parse(f.path.read_bytes(), s.source_id, s.params["sheet"])
    else:  # pragma: no cover
        raise ValueError(s.source)
    return _frame(s, [d for d, _ in obs], [v for _, v in obs], [f.retrieved_at] * len(obs))


def _ifop_frame(selected: list[Series], parsed: ifop.IfopParse) -> pl.DataFrame:
    frames = []
    for s in selected:
        obs = [o for o in parsed.observations if o.role == s.source_id]
        frames.append(
            _frame(
                s,
                [o.month for o in obs],
                [o.satisfied for o in obs],
                [parsed.retrieved_at[o.publication] for o in obs],
            )
        )
    return pl.concat(frames)


def _frame(
    s: Series, dates: list[date], values: list[float], retrieved: list[datetime]
) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame(
        {
            "series_id": [s.series_id] * n,
            "date": dates,
            "value": values,
            "freq": [s.freq] * n,
            "source": [s.source] * n,
            "retrieved_at": retrieved,
        },
        schema=MACRO_SCHEMA,
    )


def validate_macro(df: pl.DataFrame) -> None:
    if df.schema != MACRO_SCHEMA:
        raise ValueError(f"schéma macro inattendu : {df.schema}")
    if df.select(pl.all().is_null().any()).row(0) != (False,) * len(MACRO_SCHEMA):
        raise ValueError("valeurs nulles dans macro")
    if df.select(pl.struct("series_id", "date").is_duplicated().any()).item():
        raise ValueError("(series_id, date) en double dans macro")
    bad = df.filter(~pl.col("freq").is_in(["M", "Q", "W"]))
    if bad.height:
        raise ValueError(f"fréquences hors M/Q/W : {bad['freq'].unique().to_list()}")
    starts = df.filter(
        ((pl.col("freq") == "M") & (pl.col("date").dt.day() != 1))
        | (
            (pl.col("freq") == "Q")
            & ((pl.col("date").dt.day() != 1) | ((pl.col("date").dt.month() - 1) % 3 != 0))
        )
        | ((pl.col("freq") == "W") & (pl.col("date").dt.weekday() != 1))
    )
    if starts.height:
        raise ValueError(f"dates qui ne sont pas un début de période : {starts.head(5)}")


def missing_periods(dates: list[date], freq: str) -> list[date]:
    """Périodes absentes entre la première et la dernière date (trous internes)."""
    if not dates:
        return []
    have = set(dates)
    out = []
    d = min(dates)
    last = max(dates)
    while d <= last:
        if d not in have:
            out.append(d)
        d = _next_period(d, freq)
    return out


def _next_period(d: date, freq: str) -> date:
    from datetime import timedelta

    if freq == "W":
        return d + timedelta(days=7)
    step = 3 if freq == "Q" else 1
    m = d.month - 1 + step
    return date(d.year + m // 12, m % 12 + 1, 1)
