"""Fusion des sondages des différentes sources en sondages uniques (voir `rules` pour les règles).

Entrée : une liste de `SourcePoll` par source (`sources.load_sources` les lit dans
data/curated). Sortie (`MergeResult`) : les tables `polls`, `poll_results`, `poll_sources` au
format de `vvs.schemas` et le journal des conflits.

Les sources sont traitées dans l'ordre de `rules.PRECEDENCE` : un sondage de la Commission
fonde un sondage fusionné, Wikipédia puis MieuxVoter s'y rattachent ou en fondent un nouveau.
Une valeur déjà posée par une source prioritaire n'est jamais remplacée ; une valeur absente
(commanditaire inconnu, mode `unknown`, population `unknown`, résultat brut) est complétée.
Une hypothèse publiée sur une seconde base par la même source (règle `second_base` de
`rules`) suit son sondage dans `alt_hyps` : elle n'est ni comparée ni fusionnée avec les autres
sources, mais reçoit les corrections de libellés de l'hypothèse principale et sort dans
`poll_results` avec sa population.

Chaque désaccord donne une entrée de journal :

    {"poll_id": …, "key": {election, pollster, fieldwork_start, fieldwork_end, sponsor},
     "field": "share", "round": 1, "hypothesis_id": "…", "candidate_id": "…",
     "values": {"commission": 48.0, "mieuxvoter": 52.0}, "retained": 48.0,
     "rule": "precedence"}
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

from vvs import schemas
from vvs.curate import rules

HypKey = tuple[int, str]

COMPARED_FIELDS = ("sponsor", "fieldwork_start", "fieldwork_end", "sample_size", "publish_date", "mode")
META_FIELDS = (*COMPARED_FIELDS, "source_url", "ingested_at")


@dataclass
class Row:
    share: float
    share_base: str
    raw_share: float | None
    source: str
    alts: dict[str, float] = field(default_factory=dict)  # valeurs divergentes par source


@dataclass
class Hyp:
    round: int
    hypothesis_id: str
    population: str
    rows: dict[str, Row]
    source: str
    population_source: str | None = None
    seen: set[str] = field(default_factory=set)  # sources qui donnent cette hypothèse

    def __post_init__(self) -> None:
        if self.population_source is None and self.population != "unknown":
            self.population_source = self.source
        self.seen.add(self.source)

    @property
    def key(self) -> HypKey:
        return (self.round, self.hypothesis_id)

    def shares(self) -> dict[str, float]:
        return {c: r.share for c, r in self.rows.items()}

    def copy(self) -> Hyp:
        return Hyp(
            self.round, self.hypothesis_id, self.population,
            {c: Row(r.share, r.share_base, r.raw_share, r.source, dict(r.alts))
             for c, r in self.rows.items()},
            self.source, self.population_source, set(self.seen),
        )


@dataclass
class SourcePoll:
    source: str
    poll_id: str
    election: int
    pollster: str
    meta: dict
    hyps: dict[HypKey, Hyp] = field(default_factory=dict)
    rolling_label: bool = False
    dropped_population: frozenset[str] = frozenset()
    alt_hyps: dict[HypKey, Hyp] = field(default_factory=dict)  # seconde base (`second_base`)

    @property
    def start(self) -> date:
        return self.meta["fieldwork_start"]

    @property
    def end(self) -> date:
        return self.meta["fieldwork_end"]


@dataclass(eq=False)
class Canon:
    election: int
    pollster: str
    meta: dict
    meta_source: dict
    members: list[tuple[str, str, str]] = field(default_factory=list)  # source, poll_id, règle
    hyps: dict[HypKey, Hyp] = field(default_factory=dict)
    rolling_label: bool = False
    poll_id: str | None = None
    alt_hyps: dict[HypKey, Hyp] = field(default_factory=dict)  # seconde base (`second_base`)

    @property
    def start(self) -> date:
        return self.meta["fieldwork_start"]

    @property
    def end(self) -> date:
        return self.meta["fieldwork_end"]

    def member_sources(self) -> set[str]:
        return {s for s, _, _ in self.members}

    def member_ids(self) -> set[tuple[str, str]]:
        return {(s, p) for s, p, _ in self.members}

    @property
    def top_source(self) -> str:
        return min(self.member_sources(), key=rules.PRECEDENCE.index)


@dataclass
class MergeResult:
    polls: pl.DataFrame
    results: pl.DataFrame
    poll_sources: pl.DataFrame
    log: list[dict]
    stats: Counter
    mv_unreliable: set[str]


def _missing(field_name: str, value) -> bool:
    return value is None or (field_name == "mode" and value == "unknown")


def _higher(a: str, b: str) -> bool:
    """La source `a` est strictement prioritaire sur `b`."""
    return rules.PRECEDENCE.index(a) < rules.PRECEDENCE.index(b)


def _jsonable(v):
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


class Merger:
    def __init__(
        self,
        polls_by_source: dict[str, list[SourcePoll]],
        races: dict[int, rules.OfficialRace] | None = None,
        known_candidates: set[str] | None = None,
        mv_unreliable: set[str] | None = None,
    ):
        unknown = set(polls_by_source) - set(rules.PRECEDENCE)
        if unknown:
            raise ValueError(f"sources sans précédence : {sorted(unknown)}")
        self.polls_by_source = polls_by_source
        self.races = races or {}
        self.known_candidates = known_candidates
        self.forced_mv_unreliable = mv_unreliable
        self.canons: list[Canon] = []
        self._groups: dict[tuple[int, str], list[Canon]] = {}
        self._log: list[tuple[Canon, dict]] = []
        self.stats: Counter = Counter()

    # --- journal ---------------------------------------------------------------

    def log(self, canon: Canon, field_name: str, values: dict, retained, rule: str, **extra) -> None:
        entry = {"field": field_name, **extra}
        entry["values"] = {k: _jsonable(v) for k, v in values.items()}
        entry["retained"] = _jsonable(retained)
        entry["rule"] = rule
        self._log.append((canon, entry))
        self.stats[f"log:{rule}:{field_name}"] += 1

    # --- appariement -----------------------------------------------------------

    def _group(self, sp: SourcePoll) -> list[Canon]:
        return self._groups.setdefault((sp.election, sp.pollster), [])

    def _new_canon(self, sp: SourcePoll, rule: str = "new") -> Canon:
        c = Canon(
            sp.election, sp.pollster,
            {f: sp.meta.get(f) for f in META_FIELDS},
            {f: sp.source for f in META_FIELDS},
            members=[(sp.source, sp.poll_id, rule)],
            hyps={k: h.copy() for k, h in sp.hyps.items()},
            rolling_label=sp.rolling_label,
        )
        self._attach_alt(c, sp)
        self.canons.append(c)
        self._group(sp).append(c)
        return c

    @staticmethod
    def _close(c: Canon, sp: SourcePoll) -> bool:
        tol = rules.DATE_TOLERANCE_DAYS
        return abs((c.start - sp.start).days) <= tol and abs((c.end - sp.end).days) <= tol

    @staticmethod
    def _rank(c: Canon, sp: SourcePoll) -> tuple:
        """Meilleur candidat d'abord : hypothèses communes, commanditaire compatible, même
        effectif, dates les plus proches."""
        return (
            -len(set(c.hyps) & set(sp.hyps)),
            not rules.sponsors_compatible(c.meta["sponsor"], sp.meta.get("sponsor")),
            c.meta["sample_size"] != sp.meta.get("sample_size"),
            abs((c.start - sp.start).days) + abs((c.end - sp.end).days),
        )

    def _date_candidates(self, sp: SourcePoll) -> list[Canon]:
        return [
            c for c in self._group(sp)
            if sp.source not in c.member_sources() and self._close(c, sp)
        ]

    def add(self, sp: SourcePoll) -> Canon:
        cands = self._date_candidates(sp)
        if cands:
            primary = min(cands, key=lambda c: self._rank(c, sp))
            targets: dict[int, list[Hyp]] = {}
            for key, h in sp.hyps.items():
                dest = next((c for c in cands if key in c.hyps), None)
                if dest is None:
                    dest = next(
                        (c for c in cands if self._swap_target(c, h) is not None), primary
                    )
                targets.setdefault(id(dest), []).append(h)
            for c in cands:
                if c is not primary and id(c) not in targets:
                    continue
                c.members.append((sp.source, sp.poll_id, "date_match"))
                c.rolling_label |= sp.rolling_label
                if c is primary:
                    self._merge_meta(c, sp, "precedence")
                for h in targets.get(id(c), []):
                    self._merge_hyp(c, h, sp)
                    self._attach_alt(c, sp, [h.key])
            if len(targets) > 1:
                self.stats["source poll split over several polls"] += 1
            self._log_dropped_population(primary, sp)
            return primary
        same = self._same_results(sp)
        if same is not None:
            c = same
            if sp.source not in c.member_sources() and set(c.hyps) == set(sp.hyps):
                c.members.append((sp.source, sp.poll_id, "same_results"))
                c.rolling_label |= sp.rolling_label
                self._merge_meta(c, sp, "same_results")
                for h in sp.hyps.values():
                    self._merge_hyp(c, h, sp)
                self._attach_alt(c, sp)
                self._log_dropped_population(c, sp)
                return c
            n = self._split_wave(c, sp)
            self._log_dropped_population(n, sp)
            return n
        c = self._new_canon(sp)
        self._log_dropped_population(c, sp)
        return c

    def _same_results(self, sp: SourcePoll) -> Canon | None:
        if not sp.hyps:
            return None
        found = []
        for c in self._group(sp):
            if abs((c.end - sp.end).days) > rules.SAME_RESULTS_WINDOW_DAYS:
                continue
            # seules comptent les hypothèses d'une source prioritaire : deux sondages d'une
            # même source ne sont jamais rapprochés (l'ingestion les a voulus distincts)
            if all(k in c.hyps and _higher(c.hyps[k].source, sp.source)
                   and rules.same_shares(c.hyps[k].shares(), h.shares())
                   for k, h in sp.hyps.items()):
                found.append(c)
        if not found:
            return None
        return min(found, key=lambda c: abs((c.end - sp.end).days))

    def _split_wave(self, c: Canon, sp: SourcePoll) -> Canon:
        """Notice à deux vagues : les hypothèses de `sp` quittent `c` pour un sondage à part,
        daté et dimensionné comme `sp`, avec les autres métadonnées (commanditaire, mode…) de
        `c`."""
        moved = {k: c.hyps.pop(k) for k in sp.hyps}
        moved_alt = {k: c.alt_hyps.pop(k) for k in sp.hyps if k in c.alt_hyps}
        meta = dict(c.meta)
        meta_source = dict(c.meta_source)
        for f in ("fieldwork_start", "fieldwork_end", "sample_size"):
            meta[f] = sp.meta.get(f)
            meta_source[f] = sp.source
        value_sources = {h.source for h in moved.values()}
        members = [(s, p, "multi_wave_split") for s, p, _ in c.members if s in value_sources]
        members.append((sp.source, sp.poll_id, "multi_wave_split"))
        n = Canon(c.election, c.pollster, meta, meta_source, members, moved, sp.rolling_label,
                  alt_hyps=moved_alt)
        self.canons.append(n)
        self._group(sp).append(n)
        self.log(
            n, "hypotheses",
            {
                c.top_source: f"{c.start.isoformat()}/{c.end.isoformat()}",
                sp.source: f"{sp.start.isoformat()}/{sp.end.isoformat()}",
            },
            f"{len(moved)} hypothèses en sondage à part",
            "multi_wave_split",
            hypothesis_ids=sorted(f"{r}:{h}" for r, h in moved),
        )
        for h in sp.hyps.values():
            self._merge_hyp(n, h, sp)
        self._attach_alt(n, sp)
        return n

    def _attach_alt(self, c: Canon, sp: SourcePoll, keys=None) -> None:
        """Règle `second_base` : la seconde base d'une hypothèse suit l'hypothèse principale ;
        elle n'est reprise que si le sondage fusionné n'en a pas déjà une."""
        for key in sp.alt_hyps if keys is None else keys:
            h = sp.alt_hyps.get(key)
            if h is None:
                continue
            if key in c.hyps and key not in c.alt_hyps:
                c.alt_hyps[key] = h.copy()
                self.stats[f"second base from {h.source}"] += 1
            else:
                self.stats[f"second base dropped from {h.source}"] += 1

    # --- valeurs -----------------------------------------------------------------

    def _merge_meta(self, c: Canon, sp: SourcePoll, rule: str) -> None:
        for f in COMPARED_FIELDS:
            hi, lo = c.meta.get(f), sp.meta.get(f)
            if _missing(f, lo):
                continue
            if _missing(f, hi):
                c.meta[f], c.meta_source[f] = lo, sp.source
                self.stats[f"filled:{f}"] += 1
                continue
            if f == "sponsor":
                differ = not rules.sponsors_compatible(hi, lo)
            else:
                differ = hi != lo
            if differ:
                date_rule = rule if f in ("fieldwork_start", "fieldwork_end") else "precedence"
                self.log(c, f, {c.meta_source[f]: hi, sp.source: lo}, hi, date_rule)

    def _swap_target(self, c: Canon, h: Hyp) -> tuple[Hyp, str, str] | None:
        found = []
        for (rnd, _), H in c.hyps.items():
            if rnd != h.round or not _higher(H.source, h.source):
                continue
            sw = rules.label_swap(H.shares(), h.shares())
            if sw is not None:
                found.append((H, *sw))
        return found[0] if len(found) == 1 else None

    def _sibling_with(self, c: Canon, h: Hyp) -> Canon | None:
        """Vague sœur de `c` (sondage fusionné qui partage un sondage source avec lui) où une
        source prioritaire donne l'hypothèse `h`."""
        ids = c.member_ids()
        for o in self._groups.get((c.election, c.pollster), []):
            if (o is not c and h.key in o.hyps and ids & o.member_ids()
                    and _higher(o.hyps[h.key].source, h.source)):
                return o
        return None

    def _merge_hyp(self, c: Canon, h: Hyp, sp: SourcePoll) -> None:
        if h.key in c.hyps:
            self._compare(c, c.hyps[h.key], h, "precedence")
            return
        swap = self._swap_target(c, h)
        if swap is not None:
            H, ch, cl = swap
            self.log(
                c, "candidate_id", {H.source: ch, h.source: cl}, ch, "label_swap",
                round=H.round, hypothesis_id=H.hypothesis_id,
                share=h.rows[cl].share,
            )
            renamed = h.copy()
            renamed.rows[ch] = renamed.rows.pop(cl)
            self._compare(c, H, renamed, "precedence")
            return
        sib = self._sibling_with(c, h)
        if sib is not None:
            self._compare(sib, sib.hyps[h.key], h, "sibling_wave")
            self.stats["hypotheses attached to sibling wave"] += 1
            return
        c.hyps[h.key] = h.copy()
        self.stats[f"hypotheses added from {h.source}"] += 1

    def _compare(self, c: Canon, H: Hyp, h: Hyp, rule: str) -> None:
        H.seen.add(h.source)
        for cid, r in h.rows.items():
            R = H.rows.get(cid)
            if R is None:
                H.rows[cid] = Row(r.share, r.share_base, r.raw_share, r.source)
                self.stats["rows filled"] += 1
                continue
            if abs(R.share - r.share) > rules.CONFLICT_TOLERANCE:
                R.alts.setdefault(r.source, r.share)
                self.log(
                    c, "share", {R.source: R.share, r.source: r.share}, R.share, rule,
                    round=H.round, hypothesis_id=H.hypothesis_id, candidate_id=cid,
                )
            if R.raw_share is None and r.raw_share is not None:
                R.raw_share = r.raw_share
                self.stats["filled:raw_share"] += 1
        if h.population == "unknown":
            return
        if H.population == "unknown":
            H.population, H.population_source = h.population, h.population_source or h.source
            self.stats[f"filled:population from {h.source}"] += 1
        elif H.population != h.population:
            self.log(
                c, "population",
                {H.population_source or H.source: H.population, h.source: h.population},
                H.population, "precedence" if rule != "sibling_wave" else rule,
                round=H.round, hypothesis_id=H.hypothesis_id,
            )

    # --- population MieuxVoter ---------------------------------------------------

    def _mieuxvoter_unreliable(self, mv_polls: list[SourcePoll]) -> set[str]:
        """Instituts dont la population MieuxVoter contredit la Commission sur plus de la
        moitié des sondages communs comparables (au moins `MV_UNRELIABLE_MIN_POLLS`)."""
        counts: dict[str, list[int]] = {}
        for sp in mv_polls:
            cands = [c for c in self._date_candidates(sp) if "commission" in c.member_sources()]
            if not cands:
                continue
            c = min(cands, key=lambda c: self._rank(c, sp))
            pairs = [
                (c.hyps[k].population, h.population)
                for k, h in sp.hyps.items()
                if k in c.hyps and c.hyps[k].population_source == "commission"
                and h.population != "unknown"
            ]
            if not pairs:
                continue
            n = counts.setdefault(sp.pollster, [0, 0])
            n[0] += 1
            n[1] += sum(a != b for a, b in pairs) * 2 > len(pairs)
        return {
            p for p, (n, d) in counts.items()
            if n >= rules.MV_UNRELIABLE_MIN_POLLS and d * 2 > n
        }

    def _log_dropped_population(self, c: Canon, sp: SourcePoll) -> None:
        if sp.dropped_population:
            self.log(
                c, "population", {sp.source: ", ".join(sorted(sp.dropped_population))},
                "unknown", "mv_population_unreliable",
            )

    # --- exécution ---------------------------------------------------------------

    def run(self) -> MergeResult:
        mv_unreliable: set[str] = set()
        for source in rules.PRECEDENCE:
            polls = sorted(
                self.polls_by_source.get(source, []),
                key=lambda p: (p.election, p.pollster, p.end, p.start, p.poll_id),
            )
            if source == "mieuxvoter":
                mv_unreliable = (
                    self.forced_mv_unreliable
                    if self.forced_mv_unreliable is not None
                    else self._mieuxvoter_unreliable(polls)
                )
                for sp in polls:
                    if sp.pollster in mv_unreliable:
                        dropped = {h.population for h in sp.hyps.values()} - {"unknown"}
                        for h in sp.hyps.values():
                            h.population, h.population_source = "unknown", None
                        sp.dropped_population = frozenset(dropped)
            for sp in polls:
                self.add(sp)
        self._resolve_swaps()
        self._infer_population()
        self._assign_ids()
        polls_df, results_df, sources_df = self._frames()
        log = []
        for c, entry in self._log:
            log.append({
                "poll_id": c.poll_id,
                "key": {
                    "election": c.election,
                    "pollster": c.pollster,
                    "fieldwork_start": c.start.isoformat(),
                    "fieldwork_end": c.end.isoformat(),
                    "sponsor": c.meta["sponsor"],
                },
                **entry,
            })
        log.sort(key=lambda e: (e["key"]["election"], e["poll_id"], e["field"], e["rule"]))
        return MergeResult(polls_df, results_df, sources_df, log, self.stats, mv_unreliable)

    def _resolve_swaps(self) -> None:
        """Règle `swap_consistency` (voir `rules.prefer_swapped`)."""
        for c in self.canons:
            for H in c.hyps.values():
                top = H.shares()
                for s in sorted(H.seen - {H.source}, key=rules.PRECEDENCE.index):
                    alt = {cid: r.alts.get(s, r.share) for cid, r in H.rows.items()}
                    pair = rules.swapped_pair(top, alt)
                    if pair is None:
                        continue
                    others = [
                        o.shares() for o in c.hyps.values() if o is not H and o.round == H.round
                    ]
                    if not rules.prefer_swapped(top, alt, pair, others):
                        continue
                    for cid in pair:
                        R = H.rows[cid]
                        values = {R.source: R.share, **R.alts}
                        self._log = [
                            (lc, e) for lc, e in self._log
                            if not (lc is c and e["field"] == "share"
                                    and e.get("hypothesis_id") == H.hypothesis_id
                                    and e.get("round") == H.round and e.get("candidate_id") == cid)
                        ]
                        self.log(
                            c, "share", values, alt[cid], "swap_consistency",
                            round=H.round, hypothesis_id=H.hypothesis_id, candidate_id=cid,
                        )
                        R.alts = {k: v for k, v in R.alts.items() if k != s}
                        R.alts[R.source] = R.share
                        R.share, R.source = alt[cid], s
                    self._swap_second_base(c, H, pair)
                    break

    def _swap_second_base(self, c: Canon, H: Hyp, pair: tuple[str, str]) -> None:
        """Libellés inversés sur une page de notice : la seconde base, lue sur le même tableau,
        porte la même inversion ; ses deux lignes sont échangées et l'échange journalisé."""
        A = c.alt_hyps.get(H.key)
        a, b = pair
        # la valeur écartée de l'hypothèse principale venait de la source de la seconde base
        if A is None or a not in A.rows or b not in A.rows or A.source not in H.rows[a].alts:
            return
        A.rows[a], A.rows[b] = A.rows[b], A.rows[a]
        self.log(
            c, "share", {A.source: {a: A.rows[b].share, b: A.rows[a].share}},
            {a: A.rows[a].share, b: A.rows[b].share}, "swap_consistency",
            round=A.round, hypothesis_id=A.hypothesis_id, population=A.population,
        )

    def _infer_population(self) -> None:
        for c in self.canons:
            pops = {
                h.population for h in c.hyps.values()
                if h.population_source == "commission" and h.population != "unknown"
            }
            if len(pops) != 1:
                continue
            (pop,) = pops
            for h in c.hyps.values():
                if h.population == "unknown":
                    h.population, h.population_source = pop, "commission"
                    self.stats["population inferred from commission poll"] += 1

    def _assign_ids(self) -> None:
        by_day: dict[tuple, list[Canon]] = {}
        for c in self.canons:
            by_day.setdefault((c.election, c.pollster, c.end), []).append(c)
        for (election, pollster, end), group in by_day.items():
            group.sort(key=lambda c: (c.start, c.meta["sample_size"] or -1, c.meta["sponsor"] or ""))
            for seq, c in enumerate(group, start=1):
                c.poll_id = f"{election}-{pollster}-{end:%Y%m%d}-{seq}"

    def _frames(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        polls, results, sources = [], [], []
        regular = rules.rolling_by_regularity(pl.DataFrame(
            [(c.poll_id, c.election, c.pollster, c.start, c.end, c.meta["sample_size"])
             for c in self.canons],
            schema=["poll_id", "election", "pollster", "fieldwork_start", "fieldwork_end",
                    "sample_size"],
            orient="row",
        ), {c.poll_id for c in self.canons if c.rolling_label}) if self.canons else set()
        known = self.known_candidates
        for c in self.canons:
            srcs = sorted(c.member_sources(), key=rules.PRECEDENCE.index)
            rolling = c.rolling_label or c.poll_id in regular
            self.stats["rolling (label)" if c.rolling_label else
                       "rolling (regularity only)" if rolling else "not rolling"] += 1
            polls.append({
                "poll_id": c.poll_id, "election": c.election, "pollster": c.pollster,
                **{f: c.meta.get(f) for f in META_FIELDS},
                "mode": c.meta.get("mode") or "unknown",
                "source": srcs[0], "sources": srcs, "is_rolling": rolling,
            })
            for s, pid, rule in c.members:
                sources.append({
                    "source": s, "source_poll_id": pid, "canonical_poll_id": c.poll_id,
                    "match_rule": rule,
                })
            race = self.races.get(c.election)
            hyps = [*c.hyps.values(), *c.alt_hyps.values()]
            for h in sorted(hyps, key=lambda h: (h.key, rules.population_rank(h.population))):
                reasons = rules.suspect_reasons(
                    h.round, {cid: (r.share, r.share_base) for cid, r in h.rows.items()},
                    c.start, c.end, race,
                    known if known is not None else rules.named(h.shares()),
                )
                for cid, r in h.rows.items():
                    results.append({
                        "poll_id": c.poll_id, "round": h.round, "hypothesis_id": h.hypothesis_id,
                        "population": h.population, "candidate_id": cid, "share": r.share,
                        "share_base": r.share_base, "raw_share": r.raw_share, "source": r.source,
                        "suspect": bool(reasons),
                        "suspect_reason": "; ".join(reasons) if reasons else None,
                    })
        polls_df = pl.DataFrame(polls, schema=schemas.CURATED_POLLS_SCHEMA)
        results_df = pl.DataFrame(results, schema=schemas.CURATED_POLL_RESULTS_SCHEMA)
        sources_df = pl.DataFrame(sources, schema=schemas.POLL_SOURCES_SCHEMA)
        polls_df = schemas.validate_curated_polls(polls_df.sort("election", "fieldwork_end", "poll_id"))
        results_df = schemas.validate_curated_poll_results(
            results_df.sort("poll_id", "round", "hypothesis_id", "population", "candidate_id")
        )
        sources_df = schemas.validate_poll_sources(
            sources_df.unique(keep="first", maintain_order=True).sort("canonical_poll_id", "source")
        )
        return polls_df, results_df, sources_df


def merge(
    polls_by_source: dict[str, list[SourcePoll]],
    races: dict[int, rules.OfficialRace] | None = None,
    known_candidates: set[str] | None = None,
    mv_unreliable: set[str] | None = None,
) -> MergeResult:
    return Merger(polls_by_source, races, known_candidates, mv_unreliable).run()
