"""Panel de titres (data/registry/outlets.yaml) et requêtes de presse
(data/registry/press_queries.yaml), avec la traduction des requêtes dans la syntaxe de
GDELT et de Media Cloud.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from vvs import paths
from vvs.registry.loader import normalize_label

KINDS = ("daily", "weekly", "online", "tv_news", "radio", "agency", "regional_daily")


class PressRegistryError(ValueError):
    """Incohérence dans outlets.yaml ou press_queries.yaml."""


# --- titres


@dataclass(frozen=True)
class Outlet:
    outlet_id: str
    name: str
    kind: str
    domains: tuple[str, ...]
    panel: bool
    since: int | None = None
    mediacloud_in_collection: bool | None = None
    url_patterns: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class Outlets:
    outlets: dict[str, Outlet]

    def panel(self) -> list[Outlet]:
        return [o for o in self.outlets.values() if o.panel]

    def panel_domains(self) -> list[str]:
        return [d for o in self.panel() for d in o.domains]

    def match(self, url_or_domain: str) -> str | None:
        """outlet_id d'une URL ou d'un domaine, ou None hors registre.

        Les motifs d'URL (relais d'agence) passent d'abord ; sinon le domaine enregistré le
        plus long dont l'hôte est un suffixe l'emporte (rmc.bfmtv.com avant bfmtv.com).
        """
        host, path = _split(url_or_domain)
        if not host:
            return None
        full = host + path
        for o in self.outlets.values():
            for p in o.url_patterns:
                if full.startswith(p):
                    return o.outlet_id
        best: tuple[int, str] | None = None
        for o in self.outlets.values():
            for d in o.domains:
                if (host == d or host.endswith("." + d)) and (best is None or len(d) > best[0]):
                    best = (len(d), o.outlet_id)
        return best[1] if best else None


def _split(url_or_domain: str) -> tuple[str, str]:
    s = url_or_domain.strip().lower()
    if "://" not in s:
        s = "http://" + s
    parts = urlsplit(s)
    host = (parts.hostname or "").removeprefix("www.")
    return host, parts.path or "/"


def load_outlets(path: Path | None = None) -> Outlets:
    path = path or paths.REGISTRY / "outlets.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["outlets"]
    outlets: dict[str, Outlet] = {}
    seen: dict[str, str] = {}
    errors: list[str] = []
    for oid, d in raw.items():
        if d.get("kind") not in KINDS:
            errors.append(f"{oid}: kind {d.get('kind')!r} hors {KINDS}")
        domains = tuple(d.get("domains") or ())
        if not domains:
            errors.append(f"{oid}: aucun domaine")
        for dom in domains:
            if dom != dom.lower() or "/" in dom or dom.startswith("www."):
                errors.append(f"{oid}: domaine mal formé {dom!r}")
            if dom in seen:
                errors.append(f"{oid}: domaine {dom} déjà attribué à {seen[dom]}")
            seen[dom] = oid
        outlets[oid] = Outlet(
            outlet_id=oid,
            name=d["name"],
            kind=d["kind"],
            domains=domains,
            panel=bool(d["panel"]),
            since=d.get("since"),
            mediacloud_in_collection=d.get("mediacloud_in_collection"),
            url_patterns=tuple(d.get("url_patterns") or ()),
            notes=(d.get("notes") or "").strip(),
        )
    if errors:
        raise PressRegistryError("outlets.yaml :\n  " + "\n  ".join(errors))
    return Outlets(outlets)


# --- requêtes


@dataclass(frozen=True)
class PressQuery:
    """Requête d'un candidat (`candidate_id` renseigné) ou d'un thème (`topic`)."""

    query_id: str
    terms: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    candidate_id: str | None = None
    topic: str | None = None
    gdelt_themes: tuple[str, ...] = ()
    gdelt_terms: tuple[str, ...] = ()  # remplace `terms` pour GDELT si non vide


@dataclass(frozen=True)
class Theme:
    topic: str
    keywords: tuple[str, ...]
    gdelt_themes: tuple[str, ...]
    ina_fit: str
    ina_rubrics: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class PressQueries:
    gdelt_filters: str
    gdelt_start: date
    mediacloud_collection: int
    candidates: dict[str, PressQuery]
    themes: dict[str, Theme]
    rejected: dict[str, dict[str, str]] = field(default_factory=dict)

    def candidate(self, candidate_id: str) -> PressQuery:
        if candidate_id not in self.candidates:
            raise KeyError(f"{candidate_id} absent de press_queries.yaml")
        return self.candidates[candidate_id]

    def theme_query(self, topic: str) -> PressQuery:
        t = self.themes[topic]
        return PressQuery(
            query_id=f"topic_{topic}", terms=t.keywords, topic=topic, gdelt_themes=t.gdelt_themes
        )

    def all_queries(self) -> dict[str, PressQuery]:
        out = dict(self.candidates)
        for topic in self.themes:
            q = self.theme_query(topic)
            out[q.query_id] = q
        return out


INA_FITS = ("direct", "partial", "none")


@functools.cache
def _known_candidates() -> frozenset[str]:
    raw = yaml.safe_load((paths.REGISTRY / "candidates.yaml").read_text(encoding="utf-8"))
    return frozenset(raw["candidates"])


def load_queries(path: Path | None = None) -> PressQueries:
    path = path or paths.REGISTRY / "press_queries.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw["defaults"]
    errors: list[str] = []
    candidates: dict[str, PressQuery] = {}
    rejected: dict[str, dict[str, str]] = {}
    known = _known_candidates()
    for cid, d in raw["candidates"].items():
        if cid not in known:
            errors.append(f"{cid}: absent de candidates.yaml")
        aliases = tuple(d.get("aliases") or ())
        if not aliases:
            errors.append(f"{cid}: aucun alias")
        rej = {str(k): str(v) for k, v in (d.get("rejected") or {}).items()}
        bad = {normalize_label(a) for a in aliases} & {normalize_label(r) for r in rej}
        if bad:
            errors.append(f"{cid}: alias rejeté repris dans aliases : {sorted(bad)}")
        candidates[cid] = PressQuery(
            query_id=cid,
            terms=aliases,
            gdelt_terms=tuple(d.get("gdelt_aliases") or ()),
            exclude=tuple(d.get("exclude") or ()),
            candidate_id=cid,
        )
        rejected[cid] = rej
    themes: dict[str, Theme] = {}
    for topic, d in raw["themes"].items():
        ina = d.get("ina_rubrics") or {}
        if ina.get("fit") not in INA_FITS:
            errors.append(f"thème {topic}: ina_rubrics.fit {ina.get('fit')!r} hors {INA_FITS}")
        themes[topic] = Theme(
            topic=topic,
            keywords=tuple(str(k) for k in d["keywords"]),
            gdelt_themes=tuple(d.get("gdelt_themes") or ()),
            ina_fit=ina.get("fit"),
            ina_rubrics=tuple(ina.get("rubrics") or ()),
            notes=(d.get("notes") or "").strip(),
        )
    if errors:
        raise PressRegistryError("press_queries.yaml :\n  " + "\n  ".join(errors))
    return PressQueries(
        gdelt_filters=defaults["gdelt_filters"],
        gdelt_start=date.fromisoformat(str(defaults["gdelt_start"])),
        mediacloud_collection=int(defaults["mediacloud_collection"]),
        candidates=candidates,
        themes=themes,
        rejected=rejected,
    )


def _term(t: str) -> str:
    return f'"{t}"' if (" " in t or "'" in t or "-" in t or "." in t) else t


def _or_group(terms: list[str]) -> str:
    # GDELT refuse les parenthèses autour d'un terme unique
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def gdelt_query(q: PressQuery, filters: str) -> str:
    """Requête DOC 2.0 : `("A B" OR C) -"D" <filtres>` (thèmes GKG en OU avec les mots)."""
    parts = [_term(t) for t in (q.gdelt_terms or q.terms)] + [f"theme:{t}" for t in q.gdelt_themes]
    out = [_or_group(parts)]
    out += [f"-{_term(e)}" for e in q.exclude]
    if filters:
        out.append(filters)
    return " ".join(out)


def mediacloud_query(q: PressQuery) -> str:
    """Requête Media Cloud (syntaxe Lucene) : `("A B" OR C) AND NOT ("D")`."""
    terms = [_term(t) for t in q.terms]
    s = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    if q.exclude:
        s += " AND NOT (" + " OR ".join(_term(e) for e in q.exclude) + ")"
    return s
