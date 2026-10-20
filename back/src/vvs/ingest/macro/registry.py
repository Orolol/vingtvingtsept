"""Chargement de data/registry/macro_series.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from vvs import paths

SOURCES = ("insee", "dares", "mte", "ifop")
FREQS = ("M", "Q", "W")
ROLES = ("president", "prime_minister")


@dataclass(frozen=True)
class Series:
    series_id: str
    label: str
    source: str
    source_id: str
    url: str
    freq: str
    unit: str
    since: str
    raw_dir: str
    params: dict = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class Officeholder:
    person_id: str
    name: str
    role: str
    start: date
    end: date | None

    def covers_month(self, month: date) -> bool:
        """Vrai si le mois (premier jour) chevauche le mandat."""
        first = self.start.replace(day=1)
        return month >= first and (self.end is None or month <= self.end)


@dataclass(frozen=True)
class MacroRegistry:
    series: tuple[Series, ...]
    officeholders: tuple[Officeholder, ...]

    def get(self, series_id: str) -> Series:
        for s in self.series:
            if s.series_id == series_id:
                return s
        known = ", ".join(s.series_id for s in self.series)
        raise KeyError(f"série inconnue : {series_id} (connues : {known})")

    def select(self, series_ids: list[str] | None) -> list[Series]:
        if not series_ids:
            return list(self.series)
        return [self.get(s) for s in series_ids]


def registry_file() -> Path:
    return paths.REGISTRY / "macro_series.yaml"


def load(path: Path | None = None) -> MacroRegistry:
    raw = yaml.safe_load((path or registry_file()).read_text(encoding="utf-8"))
    series = []
    for item in raw["series"]:
        s = Series(
            series_id=item["series_id"],
            label=item["label"],
            source=item["source"],
            source_id=str(item["source_id"]),
            url=item["url"],
            freq=item["freq"],
            unit=item["unit"],
            since=str(item["since"]),
            raw_dir=item.get("raw_dir") or item["series_id"],
            params=item.get("params") or {},
            notes=item.get("notes") or "",
        )
        if s.source not in SOURCES:
            raise ValueError(f"{s.series_id}: source {s.source!r} hors {SOURCES}")
        if s.freq not in FREQS:
            raise ValueError(f"{s.series_id}: freq {s.freq!r} hors {FREQS}")
        series.append(s)
    ids = [s.series_id for s in series]
    if len(ids) != len(set(ids)):
        raise ValueError("series_id en double dans macro_series.yaml")
    holders = []
    for item in raw.get("officeholders") or []:
        h = Officeholder(
            person_id=item["person_id"],
            name=item["name"],
            role=item["role"],
            start=_as_date(item["start"]),
            end=_as_date(item["end"]) if item.get("end") else None,
        )
        if h.role not in ROLES:
            raise ValueError(f"{h.person_id}: rôle {h.role!r} hors {ROLES}")
        holders.append(h)
    return MacroRegistry(series=tuple(series), officeholders=tuple(holders))


def _as_date(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))
