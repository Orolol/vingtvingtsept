"""data/raw/commission -> data/curated/polls_commission.parquet, poll_results_commission.parquet.

Une ligne de `polls` par notice d'intention de vote présidentielle (question « si le premier
tour avait lieu dimanche prochain… » présente), même quand ses tableaux n'ont pas pu être
lus : la notice fournit au moins les métadonnées. Une ligne de `poll_results` par candidat,
hypothèse et tour quand les tableaux ont été extraits (voir `notice` pour les règles).
"""

from __future__ import annotations

import calendar
import concurrent.futures as cf
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from vvs import paths, schemas
from vvs.ingest.commission import notice as nt
from vvs.ingest.commission.aliases import norm_label, pollster_id
from vvs.ingest.commission.listing import RAW_DIR, load_index, month_start

POLLS_PATH = paths.CURATED / "polls_commission.parquet"
RESULTS_PATH = paths.CURATED / "poll_results_commission.parquet"
ELECTION = 2027


@dataclass
class NoticeMeta:
    pollster: str
    sponsor: str | None
    fieldwork_start: date | None
    fieldwork_end: date | None
    publish_date: date | None
    sample_size: int | None
    mode: str
    population: str


def _month_window(month_label: str) -> tuple[date, date] | None:
    start = month_start(month_label)
    if start is None:
        return None
    last = date(start.year, start.month, calendar.monthrange(start.year, start.month)[1])
    # Le terrain précède le dépôt ; quelques notices sont déposées des mois après (rectificatifs).
    return start - timedelta(days=120), last


def notice_metadata(text: nt.NoticeText, entry: dict) -> NoticeMeta:
    head = "\n".join(text.pages[:6])
    cols = "\n".join(text.columns)
    window = _month_window(entry["month"])
    fieldwork = (
        nt.find_fieldwork(head, require_cue=True, window=window)
        or nt.find_fieldwork(cols, require_cue=True, window=window)
        or nt.find_fieldwork(head, window=window)
    )
    year = fieldwork[1].year if fieldwork else int(entry["month"][-4:])
    publish = nt.find_publish_date(head, year)
    if publish is None and entry.get("date"):
        publish = date.fromisoformat(entry["date"])
    return NoticeMeta(
        pollster=pollster_id(entry["title"]),
        sponsor=nt.find_sponsor(text.pages) or entry.get("sponsor"),
        fieldwork_start=fieldwork[0] if fieldwork else None,
        fieldwork_end=fieldwork[1] if fieldwork else None,
        publish_date=publish,
        sample_size=nt.find_sample_size(head) or nt.find_sample_size(cols),
        mode=nt.find_mode(head + "\n" + cols),
        population=nt.find_population(head),
    )


@dataclass
class NoticeParse:
    file_id: int
    entry: dict
    is_iv: bool = False
    meta: NoticeMeta | None = None
    hypotheses: list[nt.Hypothesis] = field(default_factory=list)
    outcomes: list[nt.BlockOutcome] = field(default_factory=list)
    error: str | None = None
    unknown_label: bool = False


def parse_notice(path: Path, entry: dict, text: nt.NoticeText | None = None) -> NoticeParse:
    """Parse une notice ; les erreurs sont capturées dans le résultat, jamais avalées."""
    res = NoticeParse(file_id=entry["file_id"], entry=entry)
    text = text or nt.extract_text(path)
    full = "\n".join(text.pages)
    res.is_iv = nt.is_vote_intention(full)
    if not res.is_iv:
        return res
    try:
        res.meta = notice_metadata(text, entry)
    except KeyError as exc:  # institut inconnu
        res.error = str(exc)
        return res
    try:
        blocks = nt.find_blocks(text.pages) + nt.find_transposed_blocks(text.pages)
        for page_no, page in enumerate(text.pages, start=1):
            halves = nt.split_two_column_page(page)
            if halves:
                for b in nt.find_blocks(list(halves)):
                    b.page = page_no
                    blocks.append(b)
        found = nt.classify_blocks(blocks, res.meta.pollster, res.meta.population)
    except nt.UnknownLabelError as exc:
        res.error, res.unknown_label = str(exc), True
        return res
    res.hypotheses, res.outcomes = found.hypotheses, found.outcomes
    return res


def _parse_one(args: tuple[str, dict]) -> NoticeParse:
    path, entry = args
    return parse_notice(Path(path), entry)


@dataclass
class ParseReport:
    notices: list[NoticeParse]
    polls: pl.DataFrame
    results: pl.DataFrame

    def summary(self, verbose: bool = False) -> str:
        iv = [n for n in self.notices if n.is_iv]
        failed = [n for n in iv if n.error]
        no_fw = [n for n in iv if n.meta and n.meta.fieldwork_end is None]
        with_res = [n for n in iv if n.hypotheses and n.file_id not in EXCLUDED_NOTICES]
        with_raw = [
            n for n in with_res if any(r is not None for h in n.hypotheses for *_, r in h.results)
        ]
        tables = self.results.select("poll_id", "round", "hypothesis_id").unique().height
        lines = [
            f"{len(self.notices)} notices lues, {len(iv)} avec intentions de vote présidentielle",
            f"{self.polls.height} sondages écrits dans {POLLS_PATH.name}",
            (
                f"{len(with_res)} notices avec tableaux extraits ({len(with_raw)} avec résultats"
                f" bruts) : {tables} tableaux (sondage x tour x hypothèse),"
                f" {self.results.height} lignes dans {RESULTS_PATH.name}"
            ),
        ]
        for n in failed:
            lines.append(f"  ÉCHEC {n.file_id} {n.entry['title']} : {n.error}")
        for n in no_fw:
            lines.append(f"  SANS DATES DE TERRAIN (exclu) {n.file_id} {n.entry['title']}")
        for n in iv:
            if n.file_id in EXCLUDED_NOTICES:
                lines.append(f"  exclu {n.file_id} : {EXCLUDED_NOTICES[n.file_id]}")
            elif not n.error and not n.hypotheses:
                rejected = Counter(o.status for o in n.outcomes if o.status != "not_candidates")
                detail = ", ".join(f"{k} x{v}" for k, v in sorted(rejected.items())) or "aucun"
                lines.append(
                    f"  sans tableau lisible {n.file_id} {n.entry['title']} (blocs écartés : {detail})"
                )
        if verbose:
            for n in with_res:
                lines.append(
                    f"  {n.file_id} {n.meta.pollster} {n.meta.fieldwork_end} "
                    f"{len(n.hypotheses)} hyp. ({n.entry['title']})"
                )
        return "\n".join(lines)

    @property
    def has_unknown_labels(self) -> bool:
        return any(n.unknown_label for n in self.notices)


# Notices écartées par règle explicite : intentions de vote mesurées sur une sous-population,
# non comparables aux sondages en population générale.
EXCLUDED_NOTICES: dict[int, str] = {
    2143: "sous-population : électeurs LGBT (Ifop pour Têtu, mars 2026)",
}


def _same_poll_key(m: NoticeMeta) -> tuple:
    """Deux notices d'un même institut, mêmes dates de terrain et même commanditaire décrivent
    la même collecte (notice complémentaire, ex. Verian 18-20/11/2025 déposée deux fois).

    L'échantillon n'entre pas dans la clé : sa lecture varie d'une notice à l'autre quand la
    mise en page coupe la phrase (938 inscrits / 1 000 répondants pour ce même Verian)."""
    sponsor = norm_label(m.sponsor) if m.sponsor else None
    return (m.pollster, m.fieldwork_start, m.fieldwork_end, sponsor)


def build_tables(parsed: list[NoticeParse]) -> tuple[pl.DataFrame, pl.DataFrame]:
    polls_rows, result_rows = [], []
    kept = [
        n
        for n in parsed
        if n.is_iv
        and n.meta
        and n.meta.fieldwork_end
        and not n.error
        and n.file_id not in EXCLUDED_NOTICES
    ]
    kept.sort(key=lambda n: n.file_id)
    seq: dict[tuple[str, date], int] = {}
    poll_of: dict[tuple, str] = {}
    seen_hyp: set[tuple] = set()
    for n in kept:
        m = n.meta
        same = _same_poll_key(m)
        poll_id = poll_of.get(same)
        if poll_id is None:
            key = (m.pollster, m.fieldwork_end)
            seq[key] = seq.get(key, 0) + 1
            poll_id = f"{ELECTION}-{m.pollster}-{m.fieldwork_end:%Y%m%d}-{seq[key]}"
            poll_of[same] = poll_id
            polls_rows.append(
                {
                    "poll_id": poll_id,
                    "election": ELECTION,
                    "pollster": m.pollster,
                    "sponsor": m.sponsor,
                    "fieldwork_start": m.fieldwork_start,
                    "fieldwork_end": m.fieldwork_end,
                    "publish_date": m.publish_date,
                    "sample_size": m.sample_size,
                    "mode": m.mode,
                    "source": "commission",
                    "source_url": n.entry["url"],
                    "ingested_at": datetime.fromisoformat(n.entry["ingested_at"]),
                }
            )
        for h in n.hypotheses:
            hyp = schemas.hypothesis_id([c for c, _, _ in h.results])
            if (poll_id, h.round, hyp) in seen_hyp:
                continue  # hypothèse déjà donnée par une notice précédente du même sondage
            seen_hyp.add((poll_id, h.round, hyp))
            for cid, share, raw in h.results:
                result_rows.append(
                    {
                        "poll_id": poll_id,
                        "round": h.round,
                        "hypothesis_id": hyp,
                        "population": h.population,
                        "candidate_id": cid,
                        "share": share,
                        "share_base": "expressed",
                        "raw_share": raw,
                    }
                )
    polls = schemas.validate_polls(pl.DataFrame(polls_rows, schema=schemas.POLLS_SCHEMA))
    results = schemas.validate_poll_results(
        pl.DataFrame(result_rows, schema=schemas.POLL_RESULTS_SCHEMA)
    )
    return polls, results


def run(raw_dir: Path = RAW_DIR, workers: int = 8, write: bool = True) -> ParseReport:
    index = load_index(raw_dir / "index.jsonl")
    jobs = []
    for file_id, entry in sorted(index.items()):
        path = raw_dir / entry["file"]
        if path.suffix != ".pdf":
            raise ValueError(f"format de notice non géré : {path.name}")
        jobs.append((str(path), entry))
    with cf.ProcessPoolExecutor(workers) as ex:
        parsed = list(ex.map(_parse_one, jobs))
    polls, results = build_tables(parsed)
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        polls.write_parquet(POLLS_PATH)
        results.write_parquet(RESULTS_PATH)
    return ParseReport(parsed, polls, results)


def dump_debug(report: ParseReport) -> str:
    """Détail par notice et par bloc, pour le diagnostic du parseur."""
    out = []
    for n in report.notices:
        if not n.is_iv:
            continue
        out.append(json.dumps({"file_id": n.file_id, "title": n.entry["title"], "error": n.error}))
        for o in n.outcomes:
            if o.status != "not_candidates":
                out.append(f"   p{o.page} {o.status} {o.detail}")
    return "\n".join(out)
