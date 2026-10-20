"""Moteur générique de lecture des tableaux de sondages Wikipédia.

Les pages « Liste de sondages sur l'élection présidentielle française de AAAA » partagent
une forme de tableau : une colonne institut (`Sondeur`), une colonne de dates, une colonne
d'échantillon, d'éventuelles colonnes réservées (abstention, indécis, vote blanc), une
colonne par candidat, et parfois une colonne `Autres`. Un sondage occupe une ou plusieurs
lignes (une par hypothèse de candidats), les cellules communes étant fusionnées
(`rowspan`). Les variations propres à chaque page passent par une sous-classe de
`PageAdapter` (voir `e2017.py`, `e2022.py`, `e2027.py`).

Règles appliquées à toutes les pages :

- la grille est reconstruite en développant `rowspan`/`colspan` ; une cellule fusionnée sur
  plusieurs hypothèses vaut pour chacune ;
- une cellule de candidat peut porter un nom (`14 % Bertrand`, `9 Hollande ( PS )`) : le nom
  l'emporte sur l'en-tête de colonne ; sans nom, c'est l'en-tête qui désigne le candidat ;
  une colonne générique (`Candidat LR`) sans nom dans la cellule rend l'hypothèse
  inexploitable : elle est ignorée et journalisée ;
- `Autres` : chaque couple (part, nom) devient une ligne ; une part sans nom va dans l'id
  réservé `other` ;
- abstention, indécis et vote blanc sont des ids réservés, sur la base `all_respondents` ;
  les parts des candidats sont sur la base `expressed` ;
- `<x %` vaut x/2 (voir `common.parse_share`), `—` signifie « non testé » (absent, pas 0) ;
- une note de bas de page qui signale une candidature générique (« sans mention
  nominative », « aucun nom de candidat ») écarte l'hypothèse (ou le sondage si la note est
  sur l'institut) ; une note « Sondage commandé par X » renseigne `sponsor` ;
- les lignes d'évènement (cellule unique sur toute la largeur), les lignes `Résultats` et
  les lignes vides sont sautées sans être comptées comme des sondages ;
- une hypothèse dont les parts exprimées (candidats et `other`) somment hors de
  `PageAdapter.share_sum_bounds` (95 à 105 par défaut) est écartée et journalisée : c'est le
  signe d'une coquille (`38 %` pour `28 %`) ou d'un candidat testé mais perdu par la page ;
  la bande est plus large que celle de la curation (97 à 103, sondage marqué `suspect`), qui
  reste seule juge des écarts modérés ;
- un libellé de candidat ou d'institut inconnu lève `UnknownLabelError` : tout le parsing
  échoue, on complète `data/registry/aliases_wikipedia.yaml`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import ClassVar

import polars as pl
from bs4 import BeautifulSoup, Tag

from vvs import schemas
from vvs.ingest.wikipedia.common import (
    Aliases,
    AmbiguousLabelError,
    DateBounds,
    RawMeta,
    bounds_from_context,
    clean_text,
    normalize_label,
    parse_sample,
    parse_share,
    parse_share_tokens,
    resolve_range,
    strip_party,
)

# ---------------------------------------------------------------------------
# Grille
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """Cellule HTML d'origine ; partagée par toutes les cases de la grille qu'elle couvre."""

    uid: int
    text: str
    notes: tuple[str, ...]  # textes des notes de bas de page appelées dans la cellule
    times: tuple[date, ...]  # dates complètes des balises <time datetime="AAAA-MM-JJ">
    rowspan: int
    colspan: int


def _span(tag: Tag, attr: str) -> int:
    try:
        return max(1, int(str(tag.get(attr, "1")).strip().rstrip(";")))
    except ValueError:
        return 1


def _make_cell(tag: Tag, uid: int, notes: dict[str, str]) -> Cell:
    note_texts = []
    for sup in tag.select("sup.reference"):
        a = sup.find("a", href=True)
        if a and a["href"].startswith("#"):
            note_texts.append(notes.get(a["href"][1:], ""))
        sup.decompose()
    for junk in tag.select("sup.plainlinks, .noprint, [style*='display:none']"):
        junk.decompose()
    times = []
    for t in tag.find_all("time"):
        dt = str(t.get("datetime", ""))
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dt):
            times.append(date.fromisoformat(dt))
    return Cell(
        uid=uid,
        text=clean_text(tag.get_text(" ")),
        notes=tuple(note_texts),
        times=tuple(times),
        rowspan=_span(tag, "rowspan"),
        colspan=_span(tag, "colspan"),
    )


def _place_carry(
    c: int,
    carry: dict[int, tuple[Cell, int]],
    row: dict[int, Cell],
    new_carry: dict[int, tuple[Cell, int]],
) -> None:
    """Prolonge en colonne `c` une cellule fusionnée verticalement depuis une ligne au-dessus."""
    cell, left = carry[c]
    row[c] = cell
    if left > 1:
        new_carry[c] = (cell, left - 1)


def expand_table(table: Tag, notes: dict[str, str]) -> list[list[Cell | None]]:
    """Grille rectangulaire du tableau, `rowspan`/`colspan` développés."""
    rows: list[dict[int, Cell]] = []
    carry: dict[int, tuple[Cell, int]] = {}
    uid = 0
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # tableau imbriqué
        row: dict[int, Cell] = {}
        new_carry: dict[int, tuple[Cell, int]] = {}
        col = 0
        for tag in tr.find_all(["td", "th"], recursive=False):
            while col in carry:
                _place_carry(col, carry, row, new_carry)
                col += 1
            cell = _make_cell(tag, uid, notes)
            uid += 1
            for k in range(cell.colspan):
                row[col + k] = cell
                if cell.rowspan > 1:
                    new_carry[col + k] = (cell, cell.rowspan - 1)
            col += cell.colspan
        for c in sorted(carry):
            if c not in row:
                _place_carry(c, carry, row, new_carry)
        carry = new_carry
        rows.append(row)
    width = max((max(r) + 1 for r in rows if r), default=0)
    return [[r.get(c) for c in range(width)] for r in rows]


# ---------------------------------------------------------------------------
# Contexte des tableaux
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableContext:
    index: int  # rang parmi les tableaux `wikitable` de la page
    headings: tuple[str, ...]  # titres h2 > h3 > h4 > h5 englobants
    navhead: str | None  # titre du bloc repliable (div.NavFrame > div.NavHead)
    caption: str | None

    def texts_specific_first(self) -> list[str]:
        out = [self.caption or "", self.navhead or ""]
        out += list(reversed(self.headings))
        return [t for t in out if t]

    def label(self) -> str:
        return " > ".join(self.texts_specific_first()[::-1])


def iter_tables(soup: BeautifulSoup):
    """Tableaux `wikitable` dans l'ordre du document, avec leur contexte."""
    path: dict[int, str] = {}
    index = 0
    for el in soup.find_all(["h2", "h3", "h4", "h5", "table"]):
        if el.name != "table":
            level = int(el.name[1])
            path = {k: v for k, v in path.items() if k < level}
            path[level] = clean_text(el.get_text(" "))
            continue
        if "wikitable" not in (el.get("class") or []):
            continue
        nav = el.find_parent("div", class_="NavFrame")
        navhead = nav.find("div", class_="NavHead") if nav else None
        cap = el.find("caption")
        yield (
            el,
            TableContext(
                index=index,
                headings=tuple(path[k] for k in sorted(path)),
                navhead=clean_text(navhead.get_text(" ")) if navhead else None,
                caption=clean_text(cap.get_text(" ")) if cap else None,
            ),
        )
        index += 1


def page_notes(soup: BeautifulSoup) -> dict[str, str]:
    """Texte des notes de bas de page, par id d'ancre (`cite_note-EXG-10`)."""
    out = {}
    for li in soup.select("ol.references > li[id]"):
        for back in li.select(".mw-cite-backlink"):
            back.decompose()
        out[li["id"]] = clean_text(li.get_text(" "))
    return out


# ---------------------------------------------------------------------------
# Lignes extraites et journal
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """Une hypothèse d'un sondage : une ligne de tableau après résolution."""

    pollster: str
    sponsor: str | None
    start: date
    end: date
    sample_size: int | None
    round: int
    shares: dict[str, tuple[float, str]]  # candidate_id -> (share, share_base)
    where: str  # « t12 l5 » : tableau, ligne de la grille


@dataclass
class Skip:
    table: int
    row: int | None
    label: str
    reason: str

    def __str__(self) -> str:
        loc = f"t{self.table}" + (f" l{self.row}" if self.row is not None else "")
        return f"{loc} [{self.label}] {self.reason}"


@dataclass
class ParseReport:
    election: int
    skipped_rows: list[Skip] = field(default_factory=list)
    skipped_tables: list[Skip] = field(default_factory=list)
    dropped_values: list[Skip] = field(default_factory=list)  # valeurs réservées illisibles
    counters: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        lines = [f"Wikipédia {self.election} :"]
        for k, v in sorted(self.counters.items()):
            lines.append(f"  {k}: {v}")
        lines.append(f"  tableaux ignorés: {len(self.skipped_tables)}")
        for s in self.skipped_tables:
            lines.append(f"    - {s}")
        reasons = Counter(re.sub(r"\s*:.*", "", s.reason) for s in self.skipped_rows)
        lines.append(f"  valeurs réservées illisibles écartées: {len(self.dropped_values)}")
        for s in self.dropped_values:
            lines.append(f"    - {s}")
        lines.append(f"  hypothèses ignorées: {len(self.skipped_rows)}")
        for r, n in reasons.most_common():
            lines.append(f"    - {r}: {n}")
        return "\n".join(lines)


class RowSkip(Exception):
    """Hypothèse (ligne) à écarter, avec son motif."""


# ---------------------------------------------------------------------------
# Adaptateur de page
# ---------------------------------------------------------------------------

_FIXED_PREFIXES = (
    ("sondeur", "pollster"),
    ("institut", "pollster"),
    ("date", "dates"),
    ("echantillon", "sample"),
    ("abstention", "abstention"),
    ("indecis", "undecided"),
    ("vote blanc", "blank"),
    ("autre", "others"),
    ("candidat", "generic"),
)


_ID_KINDS = ("pollster", "dates", "sample")


@dataclass(frozen=True)
class Column:
    kind: str  # pollster, dates, sample, abstention, undecided, blank, others, candidate
    candidate_id: str | None = None  # None pour une colonne générique (`Candidat LR`)
    label: str = ""


class PageAdapter:
    """Lecture d'une page. Les sous-classes fixent l'élection et leurs particularités."""

    election: int
    round1_date: date
    round2_date: date
    earliest: date  # aucun sondage de la page n'est antérieur
    population: str = "unknown"
    # Tableaux ignorés : motif (regex sur le contexte normalisé) -> raison.
    skip_tables: ClassVar[dict[str, str]] = {
        r"reports? de voix": "reports de voix (matrice de transferts, pas des intentions)",
        r"criteres|sous echantillon|categorie socio|religion|\bage\b": "sous-échantillons",
        r"estimations?": "estimations soir d'élection",
    }
    generic_note_patterns: tuple[str, ...] = (
        r"sans mention nominative",
        r"aucun nom de candidat",
    )
    sponsor_note_pattern: str = r"^sondage commande par (?P<sponsor>[^,.;(]+)"
    # Lignes de résultats officiels insérées dans les tableaux (« Résultats », « 1er tour »).
    results_row_pattern: str = r"resultat|\d+ ?(er|e|nd) tour\b"
    # Bornes de la somme des parts exprimées d'une hypothèse ; None désactive le contrôle.
    share_sum_bounds: tuple[float, float] | None = (95.0, 105.0)
    # Schéma et validation de la table `polls` produite (les scrutins de liste ont une
    # `election` texte, voir `scrutins.py`).
    polls_schema: ClassVar[pl.Schema] = schemas.POLLS_SCHEMA

    @classmethod
    def validate_polls(cls, df: pl.DataFrame) -> pl.DataFrame:
        return schemas.validate_polls(df)

    # --- classement des tableaux -------------------------------------------

    def table_round(self, ctx: TableContext) -> int | None | str:
        """1 ou 2, ou une chaîne (motif) pour ignorer le tableau."""
        text = normalize_label(" ".join(ctx.texts_specific_first()))
        for pat, reason in self.skip_tables.items():
            if re.search(pat, text):
                return reason
        top = normalize_label(ctx.headings[0]) if ctx.headings else ""
        if "premier tour" in top:
            return 1
        if "second tour" in top:
            return 2
        return f"section hors premier/second tour : {ctx.headings[:1]}"

    def fallback_bounds(self, round_: int) -> DateBounds:
        upper = self.round1_date if round_ == 1 else self.round2_date
        return DateBounds(lower=self.earliest, upper=upper)

    def table_bounds(self, ctx: TableContext, round_: int) -> DateBounds:
        return bounds_from_context(ctx.texts_specific_first(), self.fallback_bounds(round_))

    # --- colonnes ------------------------------------------------------------

    def classify_column(self, label: str, aliases: Aliases) -> Column:
        key = normalize_label(strip_party(label))
        for prefix, kind in _FIXED_PREFIXES:
            if key.startswith(prefix):
                if kind == "generic":
                    return Column("candidate", None, label)
                return Column(kind, None, label)
        if not key:
            return Column("candidate", None, label)
        return Column("candidate", aliases.candidate(label), label)

    def pollster_label(self, text: str) -> str:
        """Libellé d'institut nettoyé (guillemets d'un lien mort, espaces)."""
        return clean_text(re.sub(r"[«»\"]", " ", text))

    # --- lecture d'une page --------------------------------------------------

    def parse(
        self, html: str, meta: RawMeta, aliases: Aliases
    ) -> tuple[pl.DataFrame, pl.DataFrame, ParseReport]:
        soup = BeautifulSoup(html, "lxml")
        notes = page_notes(soup)
        report = ParseReport(self.election)
        hyps: list[Hypothesis] = []
        for table, ctx in iter_tables(soup):
            round_ = self.table_round(ctx)
            if isinstance(round_, str):
                report.skipped_tables.append(Skip(ctx.index, None, ctx.label()[:90], round_))
                continue
            hyps += self.parse_table(table, ctx, round_, notes, aliases, report)
        polls, results = build_frames(self, hyps, meta, report)
        return polls, results, report

    def parse_table(
        self,
        table: Tag,
        ctx: TableContext,
        round_: int,
        notes: dict[str, str],
        aliases: Aliases,
        report: ParseReport,
    ) -> list[Hypothesis]:
        grid = expand_table(table, notes)
        if not grid or grid[0][0] is None:
            raise ValueError(f"tableau {ctx.index} ({ctx.label()}) : grille vide")
        first = grid[0][0]
        if normalize_label(first.text) not in ("sondeur", "institut"):
            raise ValueError(f"tableau {ctx.index} ({ctx.label()}) : 1re colonne {first.text!r}")
        n_head = first.rowspan
        columns = []
        for c in range(len(grid[0])):
            seen, texts = set(), []
            for r in range(n_head):
                cell = grid[r][c]
                if cell is not None and cell.uid not in seen and cell.text:
                    seen.add(cell.uid)
                    texts.append(cell.text)
            columns.append(self.classify_column(" ".join(texts), aliases))
        kinds = [col.kind for col in columns]
        for needed in ("pollster", "dates"):
            if needed not in kinds:
                raise ValueError(f"tableau {ctx.index} ({ctx.label()}) : colonne {needed} absente")
        bounds = self.table_bounds(ctx, round_)
        prev_end: date | None = None
        date_cache: dict[int, tuple[date, date] | str] = {}
        out = []
        for r in range(n_head, len(grid)):
            row = grid[r]
            where = f"t{ctx.index} l{r}"
            p_cell = row[kinds.index("pollster")]
            uids = {c.uid for c in row if c is not None}
            if p_cell is None or not p_cell.text:
                report.counters["lignes vides"] += 1
                continue
            if len(uids) == 1 or p_cell.colspan > 3:
                report.counters["lignes d'évènement"] += 1
                continue
            if re.match(self.results_row_pattern, normalize_label(p_cell.text)):
                report.counters["lignes de résultats"] += 1
                continue
            if all(row[c] is None for c, k in enumerate(kinds) if k not in _ID_KINDS):
                # ligne qui ne porte que institut/date/échantillon, hypothèses en dessous
                report.counters["lignes d'en-tête de sondage"] += 1
                continue
            label = self.pollster_label(p_cell.text)
            pollster, sponsor = aliases.pollster(label)
            try:
                sponsor = self._sponsor(p_cell, sponsor)
                self._check_generic(p_cell, "sondage")
                d_cell = row[kinds.index("dates")]
                if d_cell is None:
                    raise RowSkip("cellule de date absente")
                if d_cell.uid not in date_cache:
                    try:
                        start, end = resolve_range(
                            d_cell.text,
                            bounds,
                            prev_end,
                            exact_end=d_cell.times[-1] if d_cell.times else None,
                            exact_start=d_cell.times[0] if len(d_cell.times) == 2 else None,
                        )
                        self._check_date(end, round_)
                        date_cache[d_cell.uid] = (start, end)
                        prev_end = end
                    except ValueError as e:
                        date_cache[d_cell.uid] = f"date : {e}"
                dates = date_cache[d_cell.uid]
                if isinstance(dates, str):
                    raise RowSkip(dates)
                s_cell = row[kinds.index("sample")] if "sample" in kinds else None
                sample = parse_sample(s_cell.text) if s_cell else None
                dropped: list[str] = []
                shares = self.row_shares(row, columns, aliases, dropped)
                for msg in dropped:
                    report.dropped_values.append(Skip(ctx.index, r, label, msg))
            except RowSkip as e:
                report.skipped_rows.append(Skip(ctx.index, r, label, str(e)))
                continue
            out.append(
                Hypothesis(pollster, sponsor, dates[0], dates[1], sample, round_, shares, where)
            )
        return out

    def _sponsor(self, cell: Cell, default: str | None) -> str | None:
        for note in cell.notes:
            m = re.search(self.sponsor_note_pattern, normalize_label(note))
            if m:
                # on reprend le texte original de la note pour garder accents et casse
                raw = re.sub(r"^\s*Sondage commandé par\s*", "", note, flags=re.IGNORECASE)
                raw = re.sub(r"^(le|la|les|l')\s*", "", clean_text(re.split(r"[,.;(]", raw)[0]))
                return raw or default
        return default

    def _check_generic(self, cell: Cell, what: str) -> None:
        for note in cell.notes:
            n = normalize_label(note)
            if any(re.search(p, n) for p in self.generic_note_patterns):
                raise RowSkip(f"candidature générique ({what}) : {note[:80]}")

    def _check_date(self, end: date, round_: int) -> None:
        upper = self.round1_date if round_ == 1 else self.round2_date
        if not self.earliest <= end <= upper:
            raise ValueError(f"fin de terrain {end} hors de [{self.earliest}, {upper}]")

    def row_shares(
        self,
        row: list[Cell | None],
        columns: list[Column],
        aliases: Aliases,
        dropped: list[str],
    ) -> dict[str, tuple[float, str]]:
        """Parts d'une ligne : candidate_id -> (part, base).

        Une valeur illisible dans une colonne réservée (abstention « 20-24 % ») est écartée
        et consignée dans `dropped` sans perdre l'hypothèse ; dans une colonne de candidat
        elle écarte l'hypothèse.
        """
        shares: dict[str, tuple[float, str]] = {}

        def resolve(name: str) -> str:
            try:
                return aliases.candidate(name)
            except AmbiguousLabelError as e:
                raise RowSkip(f"libellé ambigu : {name!r}") from e

        def add(cid: str, value: float, base: str) -> None:
            if cid in shares:
                if cid == "other":
                    shares[cid] = (shares[cid][0] + value, base)
                    return
                raise RowSkip(f"candidat en double : {cid}")
            shares[cid] = (value, base)

        done: set[int] = set()
        for c, col in enumerate(columns):
            cell = row[c]
            if col.kind in _ID_KINDS or cell is None or cell.uid in done:
                continue
            done.add(cell.uid)
            span = [columns[k] for k in range(len(columns)) if row[k] is cell]
            kinds = {s.kind for s in span}
            if len(kinds) > 1:
                raise RowSkip(f"cellule à cheval sur des colonnes {sorted(kinds)} : {cell.text!r}")
            if col.kind in ("abstention", "undecided", "blank"):
                try:
                    v = parse_share(cell.text)
                except ValueError as e:
                    dropped.append(f"{col.kind} : {e}")
                    continue
                if v is not None:
                    add(col.kind, v, "all_respondents")
                continue
            try:
                tokens = parse_share_tokens(cell.text)
            except ValueError as e:
                raise RowSkip(f"cellule illisible : {e}") from None
            if not tokens:
                continue
            self._check_generic(cell, "cellule")
            if col.kind == "others":
                for v, name in tokens:
                    add(resolve(name) if name else "other", v, "expressed")
                continue
            heads = {s.candidate_id for s in span}
            for v, name in tokens:
                if name:
                    cid = resolve(name)
                elif len(heads) == 1 and None not in heads:
                    cid = next(iter(heads))
                else:
                    labels = "/".join(sorted({s.label for s in span}))
                    raise RowSkip(f"candidat non nommé sous un en-tête générique : {labels!r}")
                add(cid, v, "expressed")
        if not any(base == "expressed" for _, base in shares.values()):
            raise RowSkip("aucune intention de vote")
        if self.share_sum_bounds is not None:
            lo, hi = self.share_sum_bounds
            total = sum(v for v, base in shares.values() if base == "expressed")
            if not lo <= total <= hi:
                raise RowSkip(f"somme des intentions hors [{lo:g}, {hi:g}] : {total:g}")
        return shares


# ---------------------------------------------------------------------------
# Assemblage des tables
# ---------------------------------------------------------------------------


def _hyp_keys(hyps: list[Hypothesis]) -> set[tuple[int, str]]:
    return {(h.round, schemas.hypothesis_id(list(h.shares))) for h in hyps}


def group_polls(hyps: list[Hypothesis], report: ParseReport) -> list[dict]:
    """Regroupe les hypothèses en sondages (unités de collecte).

    1. Les lignes de même (institut, début, fin, échantillon) forment un sous-sondage : ses
       hypothèses de premier et de second tour, dispersées dans plusieurs tableaux, se
       retrouvent ensemble.
    2. Les sous-sondages de même (institut, début, fin) mais d'échantillons différents sont
       fusionnés quand leurs hypothèses ne se recouvrent pas et que leurs commanditaires
       connus ne divergent pas : c'est le cas courant d'un second tour posé à un
       sous-échantillon, ou d'une coquille d'effectif (1 090 / 1 093). `sample_size` garde
       alors le plus grand effectif. Deux sous-sondages qui testent la même hypothèse (un
       rolling et une vague Cevipof le même jour) restent distincts.
    """
    subs: dict[tuple, dict] = {}
    for h in hyps:
        key = (h.pollster, h.start, h.end, h.sample_size)
        sp = subs.setdefault(key, {"sponsor": h.sponsor, "hyps": [], "sample": h.sample_size})
        sp["sponsor"] = sp["sponsor"] or h.sponsor
        sp["hyps"].append(h)
    by_fieldwork: dict[tuple, list[dict]] = {}
    for (pollster, start, end, _), sp in subs.items():
        by_fieldwork.setdefault((pollster, start, end), []).append(sp)
    polls = []
    for (pollster, start, end), sps in by_fieldwork.items():
        sps.sort(key=lambda sp: -(sp["sample"] or 0))
        clusters: list[dict] = []
        for sp in sps:
            for cl in clusters:
                compatible = not (
                    cl["sponsor"] and sp["sponsor"] and cl["sponsor"] != sp["sponsor"]
                )
                if compatible and not (_hyp_keys(cl["hyps"]) & _hyp_keys(sp["hyps"])):
                    cl["hyps"] += sp["hyps"]
                    cl["sponsor"] = cl["sponsor"] or sp["sponsor"]
                    report.counters["sous-échantillons fusionnés"] += 1
                    break
            else:
                clusters.append(dict(sp, hyps=list(sp["hyps"])))
        for cl in clusters:
            polls.append(dict(cl, pollster=pollster, start=start, end=end))
    return polls


def build_frames(
    adapter: PageAdapter, hyps: list[Hypothesis], meta: RawMeta, report: ParseReport
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Produit (polls, poll_results) validés à partir des hypothèses (voir `group_polls`).

    Le numéro de séquence départage les sondages d'un même institut finissant le même jour,
    dans l'ordre (début, échantillon, commanditaire), pour rester stable quand la page
    s'enrichit.
    """
    election = adapter.election
    by_day: dict[tuple, list[dict]] = {}
    for p in group_polls(hyps, report):
        by_day.setdefault((p["pollster"], p["end"]), []).append(p)
    poll_rows, result_rows = [], []
    for (pollster, end), group in sorted(by_day.items()):
        group.sort(key=lambda p: (p["start"], p["sample"] or -1, p["sponsor"] or ""))
        for seq, p in enumerate(group, start=1):
            poll_id = f"{election}-{pollster}-{end:%Y%m%d}-{seq}"
            poll_rows.append(
                {
                    "poll_id": poll_id,
                    "election": election,
                    "pollster": pollster,
                    "sponsor": p["sponsor"],
                    "fieldwork_start": p["start"],
                    "fieldwork_end": end,
                    "publish_date": None,
                    "sample_size": p["sample"],
                    "mode": "unknown",
                    "source": "wikipedia",
                    "source_url": meta.source_url,
                    "ingested_at": meta.fetched_at,
                }
            )
            seen: dict[tuple, dict] = {}
            for h in p["hyps"]:
                hid = schemas.hypothesis_id(list(h.shares))
                k = (h.round, hid)
                if k in seen:
                    if seen[k] == h.shares:
                        report.counters["hypothèses répétées à l'identique"] += 1
                    else:
                        report.skipped_rows.append(
                            Skip(
                                -1,
                                None,
                                poll_id,
                                f"hypothèse en double aux parts différentes"
                                f" : tour {h.round}, {h.where}",
                            )
                        )
                    continue
                seen[k] = h.shares
                for cid, (share, base) in h.shares.items():
                    result_rows.append(
                        {
                            "poll_id": poll_id,
                            "round": h.round,
                            "hypothesis_id": hid,
                            "population": adapter.population,
                            "candidate_id": cid,
                            "share": share,
                            "share_base": base,
                            "raw_share": None,
                        }
                    )
    polls_df = pl.DataFrame(poll_rows, schema=adapter.polls_schema)
    results_df = pl.DataFrame(result_rows, schema=schemas.POLL_RESULTS_SCHEMA)
    polls_df = adapter.validate_polls(polls_df.sort("fieldwork_end", "poll_id"))
    results_df = schemas.validate_poll_results(results_df)
    report.counters["sondages"] = polls_df.height
    report.counters["lignes poll_results"] = results_df.height
    return polls_df, results_df
