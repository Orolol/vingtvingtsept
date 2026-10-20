"""Sondages Wikipédia des scrutins de liste et des législatives (européennes, législatives).

Usage :

    from vvs.ingest.wikipedia import scrutins
    html = scrutins.fetch_page("euro2019"); scrutins.store_raw("euro2019", html)
    polls, poll_results, report = scrutins.parse_with_report("euro2019", html)

Mêmes principes que les pages présidentielles (`engine.py`), avec trois différences :

- les entités testées sont des listes (européennes) ou des nuances et coalitions
  (législatives), désignées par un `list_id` stable de `data/registry/lists.yaml`
  (`rn_2019`, `nupes_2022`…). Les libellés de colonnes et de cellules sont résolus par les
  alias `wikipedia` de ce fichier ; un libellé inconnu fait échouer le parsing ;
- `election` est une chaîne (`euro2019`, `leg2024`) : les tables de sortie suivent les
  schémas `polls` et `poll_results` à cette colonne près (`POLLS_LISTS_SCHEMA`) ;
- seul le premier tour est lu : les sondages de second tour des législatives (projections
  de duels par circonscription) sont ignorés, ainsi que les projections en sièges.

Le HTML brut est stocké comme pour les présidentielles, dans
`data/raw/wikipedia/<scrutin>/<horodatage UTC>.html`, avec l'en-tête `<!-- vvs:wikipedia
election=<scrutin> … -->`. Les adaptateurs par page sont dans `euro2014.py`, `euro2019.py`,
`euro2024.py`, `leg2022.py`, `leg2024.py`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import httpx
import polars as pl
import yaml

from vvs import paths, schemas
from vvs.ingest.wikipedia.common import (
    ALIASES_PATH,
    API_URL,
    RAW_DIR,
    USER_AGENT,
    Aliases,
    RawMeta,
    UnknownLabelError,
    _check_id,
    normalize_label,
    strip_party,
)
from vvs.ingest.wikipedia.engine import Column, PageAdapter, ParseReport, TableContext

_HEADER_RE = re.compile(r"^<!-- vvs:wikipedia (?P<body>.*?) -->\n", re.DOTALL)

# Schéma de `polls_lists.parquet` : celui de `polls`, `election` en texte.
POLLS_LISTS_SCHEMA = pl.Schema({**schemas.POLLS_SCHEMA, "election": pl.Utf8})


@dataclass(frozen=True)
class Scrutin:
    key: str  # euro2019, leg2024
    kind: str  # euro ou leg
    title: str  # titre exact de la page Wikipédia FR
    vote: date  # jour du vote (premier tour des législatives)
    earliest: date  # aucun sondage de la page n'est antérieur


SCRUTINS: dict[str, Scrutin] = {
    s.key: s
    for s in (
        # Pas de page « Liste de sondages sur les élections européennes de AAAA en France »
        # (vérifié le 2026-10-20) : en 2019 et 2024 la série complète est dans la section
        # France des pages européennes « Sondages sur les élections européennes de AAAA »
        # (l'article de l'élection n'a que les derniers sondages) ; en 2014, dans l'article
        # de l'élection (section « Sondages > Au niveau national »).
        Scrutin("euro2014", "euro", "Élections européennes de 2014 en France",
                date(2014, 5, 25), date(2012, 6, 18)),
        Scrutin("euro2019", "euro", "Sondages sur les élections européennes de 2019",
                date(2019, 5, 26), date(2017, 6, 19)),
        Scrutin("euro2024", "euro", "Sondages sur les élections européennes de 2024",
                date(2024, 6, 9), date(2022, 6, 20)),
        Scrutin("leg2022", "leg", "Liste de sondages sur les élections législatives françaises de 2022",
                date(2022, 6, 12), date(2021, 1, 1)),
        Scrutin("leg2024", "leg", "Liste de sondages sur les élections législatives françaises de 2024",
                date(2024, 6, 30), date(2022, 6, 20)),
    )
}  # fmt: skip


def scrutin(key: str) -> Scrutin:
    try:
        return SCRUTINS[key]
    except KeyError:
        raise ValueError(f"scrutin inconnu {key!r} (disponibles : {sorted(SCRUTINS)})") from None


# ---------------------------------------------------------------------------
# Téléchargement et stockage brut
# ---------------------------------------------------------------------------


def fetch_page(key: str, client: httpx.Client | None = None, retries: int = 6) -> str:
    """Rendu HTML courant de la page du scrutin, avec son en-tête vvs.

    L'API renvoie 429 quand d'autres clients de la machine l'interrogent : on attend
    `Retry-After` (ou 30 s, doublé à chaque essai) avant de réessayer."""
    s = scrutin(key)
    params = {
        "action": "parse",
        "page": s.title,
        "prop": "text|revid",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0)
    try:
        wait = 30.0
        for attempt in range(retries + 1):
            resp = client.get(API_URL, params=params)
            if resp.status_code != 429 or attempt == retries:
                break
            time.sleep(float(resp.headers.get("Retry-After") or wait))
            wait *= 2
        resp.raise_for_status()
        data = resp.json()
    finally:
        if own:
            client.close()
    if "error" in data:
        raise RuntimeError(f"API MediaWiki : {data['error']}")
    parsed = data["parse"]
    fetched_at = datetime.now(UTC).replace(microsecond=0)
    header = (
        f"<!-- vvs:wikipedia election={key} revid={parsed['revid']} "
        f'fetched_at={fetched_at.isoformat()} title="{parsed["title"]}" -->\n'
    )
    return header + parsed["text"]


def read_meta(html: str, key: str) -> RawMeta:
    """En-tête vvs d'un HTML brut de scrutin (`election` = clé texte)."""
    m = _HEADER_RE.match(html)
    if not m:
        raise ValueError("HTML sans en-tête `<!-- vvs:wikipedia … -->` (voir fetch_page)")
    fields = {k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', m.group("body"))}
    if fields["election"] != key:
        raise ValueError(f"HTML du scrutin {fields['election']}, attendu {key}")
    fetched = datetime.fromisoformat(fields["fetched_at"]).astimezone(UTC).replace(tzinfo=None)
    revid = int(fields["revid"]) if fields.get("revid", "").isdigit() else None
    return RawMeta(election=key, title=fields["title"], revid=revid, fetched_at=fetched)  # type: ignore[arg-type]


def store_raw(key: str, html: str) -> Path:
    """Écrit le HTML brut dans data/raw/wikipedia/<scrutin>/ ; refuse d'écraser."""
    meta = read_meta(html, key)
    out_dir = RAW_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{meta.fetched_at:%Y-%m-%dT%H%M}.html"
    if path.exists():
        raise FileExistsError(f"{path} existe déjà (data/raw est immuable)")
    path.write_text(html, encoding="utf-8")
    return path


def latest_raw(key: str) -> Path:
    """Dernier HTML brut de la page du scrutin (`Scrutin.title`). Un fichier d'une autre
    page (l'article de l'élection, téléchargé avant de trouver la série complète) est
    ignoré : data/raw est en ajout seul, on ne le supprime pas."""
    title = scrutin(key).title
    files = [
        f for f in sorted((RAW_DIR / key).glob("*.html"))
        if read_meta(f.read_text(encoding="utf-8")[:2000], key).title == title
    ]
    if not files:
        raise FileNotFoundError(f"aucun HTML brut de {title!r} dans {RAW_DIR / key} (voir fetch_page)")
    return files[-1]


# ---------------------------------------------------------------------------
# Alias : listes (lists.yaml) et instituts (aliases_wikipedia.yaml + lists.yaml)
# ---------------------------------------------------------------------------


class ListLabels(Aliases):
    """Alias d'un scrutin ; un libellé inconnu renvoie vers lists.yaml."""

    def candidate(self, label: str) -> str:
        try:
            return super().candidate(label)
        except UnknownLabelError:
            key = normalize_label(strip_party(label))
            raise UnknownLabelError(
                f"libellé de liste inconnu pour {self.election} : {label!r} (clé {key!r}). "
                f"Ajoute-le aux alias wikipedia de data/registry/lists.yaml"
            ) from None


@lru_cache(maxsize=16)
def load_aliases(key: str, lists_path: Path | None = None) -> Aliases:
    """Alias d'un scrutin : `list_id` <- libellés Wikipédia (et le `list_id` lui-même).

    Les instituts viennent de la section `pollsters` d'`aliases_wikipedia.yaml`, complétée
    par celle de `lists.yaml` (libellés propres à ces pages)."""
    from vvs.curate.lists import LISTS_PATH, load_lists

    reg = load_lists(lists_path or LISTS_PATH)
    cands: dict[str, str] = {}
    for lid, entry in reg.for_election(key).items():
        for label in (lid, *entry.aliases.wikipedia):
            k = normalize_label(strip_party(label))
            if k in cands and cands[k] != lid:
                raise ValueError(f"lists.yaml {key} : alias {label!r} pour {cands[k]} et {lid}")
            cands[k] = lid
    raw = yaml.safe_load(ALIASES_PATH.read_text(encoding="utf-8"))
    pollsters: dict[str, tuple[str, str | None]] = {}
    for label, val in {**(raw.get("pollsters") or {}), **reg.pollsters}.items():
        pid, sponsor = (val, None) if isinstance(val, str) else (val["pollster"], val.get("sponsor"))
        pollsters[normalize_label(str(label))] = (_check_id(pid, f"pollsters:{label}"), sponsor)
    ambiguous = frozenset(normalize_label(strip_party(x)) for x in reg.ambiguous.get(key, []))
    return ListLabels(election=key, candidates=cands, pollsters=pollsters, ambiguous=ambiguous)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adaptateur commun aux pages de scrutins
# ---------------------------------------------------------------------------


class ListPageAdapter(PageAdapter):
    """Pages de scrutins : un seul tour lu, tableaux choisis par motif de contexte.

    - `round1_tables` : motif (regex sur le contexte normalisé, du titre de section à la
      légende) qu'un tableau doit vérifier pour être lu ; les autres sont ignorés et
      journalisés avec leur contexte. `skip_tables` s'applique avant ;
    - une cellule fusionnée sur plusieurs colonnes de listes différentes vaut pour leur
      groupe commun de `lists.yaml` (« RN » et « LR (Ciotti) » -> `rn_allies_2024`), ou, à
      défaut, pour l'en-tête commun dont elle couvre toutes les sous-colonnes (« NFP » sur
      « LFI », « PCF », « EELV », « PS » -> `nfp_2024`) ; sinon l'hypothèse est écartée (règle
      du moteur : cellule sur une partie des partis d'une coalition, ou sur deux coalitions) ;
    - `absent_texts` : marqueurs « non testé » propres à une page (`N.t.`) ;
    - colonne d'échantillon coupée par un trait d'union (`Échan- tillon`).
    """

    election: str  # type: ignore[assignment]
    polls_schema = POLLS_LISTS_SCHEMA
    round1_tables: str = r"sondage|intentions? de vote"
    skip_tables: ClassVar[dict[str, str]] = {
        **PageAdapter.skip_tables,
        r"second tour|\b2nd tour\b": "second tour (ignoré pour les scrutins de liste)",
        r"sieges?|projections?": "projections en sièges",
    }
    results_row_pattern = r"resultat|\d+ ?(er|e|nd) tour\b|elections? de \d{4}"
    absent_texts: tuple[str, ...] = ()

    def __init__(self) -> None:
        s = scrutin(self.election)
        self.round1_date = s.vote
        self.round2_date = s.vote
        self.earliest = s.earliest
        self._groups: dict[str, str] | None = None

    def table_round(self, ctx: TableContext) -> int | str:
        text = normalize_label(" ".join(ctx.texts_specific_first()[::-1]))
        for pat, reason in self.skip_tables.items():
            if re.search(pat, text):
                return reason
        if re.search(self.round1_tables, text):
            return 1
        return f"tableau hors sondages nationaux : {ctx.label()[:60]!r}"

    def classify_column(self, label: str, aliases: Aliases) -> Column:
        key = normalize_label(strip_party(label)).replace(" ", "")
        if key.startswith(("echantillon", "panel")):
            return Column("sample", None, label)
        return super().classify_column(label, aliases)

    def groups(self) -> dict[str, str]:
        if self._groups is None:
            from vvs.curate.lists import load_lists

            entries = load_lists().for_election(self.election)
            self._groups = {lid: e.group for lid, e in entries.items() if e.group}
        return self._groups

    def row_shares(self, row, columns, aliases, dropped):
        groups = self.groups()
        cols = list(columns)
        row = [
            replace(c, text="—") if c is not None and c.text.strip().lower() in self.absent_texts else c
            for c in row
        ]
        done: set[int] = set()
        for c, cell in enumerate(row):
            if cell is None or cell.uid in done or cols[c].kind != "candidate":
                continue
            done.add(cell.uid)
            span = [k for k in range(len(cols)) if row[k] is cell]
            heads = {cols[k].candidate_id for k in span}
            if len(heads) < 2 or None in heads or any(cols[k].kind != "candidate" for k in span):
                continue
            common = {groups.get(h, h) for h in heads}
            target = common.pop() if len(common) == 1 else self._span_header(span, cols, aliases)
            if target is not None:
                for k in span:
                    cols[k] = Column("candidate", target, cols[k].label)
        return super().row_shares(row, cols, aliases, dropped)

    @staticmethod
    def _span_header(span: list[int], cols: list[Column], aliases: Aliases) -> str | None:
        """Liste de l'en-tête commun d'une cellule fusionnée qui couvre exactement toutes les
        sous-colonnes de cet en-tête (« NFP » au-dessus de « LFI », « PCF », « EELV »,
        « PS ») ; None sinon (cellule sur une partie des sous-colonnes, ou sur deux en-têtes)."""
        words = [cols[k].label.split() for k in span]
        n = 0
        while all(len(w) > n for w in words) and len({w[n] for w in words}) == 1:
            n += 1
        for m in range(n, 0, -1):
            prefix = words[0][:m]
            cover = {k for k, c in enumerate(cols) if c.kind == "candidate" and c.label.split()[:m] == prefix}
            if cover != set(span):
                continue
            try:
                return aliases.candidate(" ".join(prefix))
            except UnknownLabelError:
                continue
        return None

    @classmethod
    def validate_polls(cls, df: pl.DataFrame) -> pl.DataFrame:
        df = df.select(list(POLLS_LISTS_SCHEMA.keys())).cast(dict(POLLS_LISTS_SCHEMA))
        schemas.validate_polls(df.with_columns(pl.lit(0).alias("election")))
        return df


def adapters() -> dict[str, ListPageAdapter]:
    from vvs.ingest.wikipedia import euro2014, euro2019, euro2024, leg2022, leg2024

    return {
        "euro2014": euro2014.AdapterEuro2014(),
        "euro2019": euro2019.AdapterEuro2019(),
        "euro2024": euro2024.AdapterEuro2024(),
        "leg2022": leg2022.AdapterLeg2022(),
        "leg2024": leg2024.AdapterLeg2024(),
    }


def parse_with_report(
    key: str, html: str, aliases: Aliases | None = None
) -> tuple[pl.DataFrame, pl.DataFrame, ParseReport]:
    adapter = adapters()[key]
    return adapter.parse(html, read_meta(html, key), aliases or load_aliases(key))


def parse_all(keys=tuple(SCRUTINS)) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, ParseReport]]:
    """Concatène les dernières pages brutes de chaque scrutin (polls_lists, poll_results_lists)."""
    polls, results, reports = [], [], {}
    for k in keys:
        p, r, rep = parse_with_report(k, latest_raw(k).read_text(encoding="utf-8"))
        polls.append(p)
        results.append(r)
        reports[k] = rep
    return pl.concat(polls), pl.concat(results), reports


def write_curated(polls: pl.DataFrame, results: pl.DataFrame, curated: Path | None = None) -> list[Path]:
    curated = curated or paths.CURATED
    out = [curated / "polls_lists.parquet", curated / "poll_results_lists.parquet"]
    polls.write_parquet(out[0])
    results.write_parquet(out[1])
    return out
