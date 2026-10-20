"""Notices de la Commission -> data/curated/poll_certainty.parquet.

Part des électeurs de chaque candidat « sûrs de leur choix » (choix définitif, ne changeront
pas d'avis), par sondage, pour les présidentielles 2017, 2022 et 2027. Lecture des tableaux :
`vvs.ingest.commission.certainty` ; métadonnées (institut, dates de terrain) : celles du
parseur des intentions de vote (`commission.parse.notice_metadata`).

Une ligne par (sondage, candidat), plus une ligne `candidate_id = "_all"` quand la notice
donne la part de l'ensemble des électeurs. Colonnes :

- `poll_id` : id de `polls.parquet` quand la notice s'y aligne (même institut, fin de terrain
  à un jour près, début à trois jours près, appariement un à un), sinon
  `<élection>-<institut>-<fin>-cert<file_id>` ;
- `election`, `candidate_id`, `certain_share` (0-100), `source` (« commission », ou
  « ifop_report » pour les rapports quotidiens du rolling Ifop 2022, voir plus bas),
  `source_url` (page de la notice ou du rapport) ;
- `pollster`, `fieldwork_start`, `fieldwork_end`, `fmt` (format du tableau, voir
  `commission.certainty`), `page`, `file_id` (négatif pour un rapport Ifop : −AAAAMMJJ),
  `aligned`.

Bases : selon l'institut, inscrits certains d'aller voter (Ipsos, BVA) ou inscrits (Toluna
Harris Interactive) ayant exprimé une intention de vote ; la part n'est pas comparable en
niveau d'un institut à l'autre, d'où l'usage centré par sondage dans l'étude et le modèle.

Sondages à plusieurs hypothèses : la première table de la notice (hypothèse principale, ou
seule ventilée). Notices rectificatives d'un même sondage : la première déposée.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from pdfminer.psparser import PSException

from vvs import paths
from vvs.ingest.commission import certainty as ce
from vvs.ingest.commission import notice as nt
from vvs.ingest.commission.aliases import CANDIDATES
from vvs.ingest.commission.listing import RAW_DIR, archive_dir, load_index
from vvs.ingest.commission.parse import EXCLUDED_NOTICES as PARSE_EXCLUDED
from vvs.ingest.commission.parse import notice_metadata
from vvs.registry import default_registry

log = logging.getLogger(__name__)

OUT_PATH = paths.CURATED / "poll_certainty.parquet"
ELECTIONS = (2017, 2022, 2027)
SCHEMA = pl.Schema(
    {
        "poll_id": pl.Utf8,
        "election": pl.Int32,
        "candidate_id": pl.Utf8,
        "certain_share": pl.Float64,
        "source": pl.Utf8,
        "source_url": pl.Utf8,
        "pollster": pl.Utf8,
        "fieldwork_start": pl.Date,
        "fieldwork_end": pl.Date,
        "fmt": pl.Utf8,
        "page": pl.Int32,
        "file_id": pl.Int32,
        "aligned": pl.Boolean,
    }
)
# Notices écartées par règle explicite (sous-populations), en plus de celles du parseur 2027
# (`commission.parse.EXCLUDED_NOTICES`).
EXCLUDED_NOTICES: dict[tuple[int, int], str] = {
    (2017, 429): "sous-population : Français de l'étranger (BVA pour lepetitjournal.com)",
}
# Fenêtre de terrain retenue : de 18 mois avant le premier tour à la veille du vote.
WINDOW_DAYS = 548


def raw_dir_for(election: int, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir if election == 2027 else archive_dir(election, raw_dir)


def candidate_labels() -> dict[str, list[str]]:
    """{id: libellés} : registre (nom complet, prénom + nom) et alias de la Commission."""
    reg = default_registry()
    out = {cid: [c.name, f"{c.first_name} {c.last_name}"] for cid, c in reg.candidates.items()}
    for cid, labels in CANDIDATES.items():
        out.setdefault(cid, []).extend(labels)
    return out


# Abréviations des titres de 2016-2022 absentes des alias de la Commission (`aliases`), qui
# servent aussi au parseur 2027 et qu'on ne modifie pas pour cela : « HI » (Harris
# Interactive), « OW » (OpinionWay), coquilles vues dans les titres.
_ARCHIVE_POLLSTERS = {
    "hi": "Harris Interactive",
    "phi": "Harris Interactive",
    "ow": "OpinionWay",
    "op": "OpinionWay",
    "ipsoso": "Ipsos",
    "iposos": "Ipsos",
    "cluster": "Cluster17",
    "presitrack": "OpinionWay",
}


def entry_with_pollster(entry: dict) -> dict:
    """Entrée d'index dont le titre nomme un institut reconnu par `aliases` : on préfixe le
    nom complet quand le titre n'a qu'une abréviation d'archive."""
    from vvs.ingest.commission.aliases import find_pollster_in_title, norm_label

    if find_pollster_in_title(entry["title"]):
        return entry
    for tok in norm_label(entry["title"]).split():
        if tok in _ARCHIVE_POLLSTERS:
            return entry | {"title": f"{_ARCHIVE_POLLSTERS[tok]} {entry['title']}"}
    return entry


@dataclass
class NoticeCertainty:
    file_id: int
    election: int
    entry: dict
    pollster: str | None = None
    fieldwork_start: dt.date | None = None
    fieldwork_end: dt.date | None = None
    rows: dict = field(default_factory=dict)  # candidate_id -> CertaintyRow
    total: float | None = None
    cue_pages: int = 0
    error: str | None = None


def read_notice(path: Path, entry: dict, election: int, labels: dict) -> NoticeCertainty:
    """Tableaux de sûreté du choix et métadonnées d'une notice ; erreurs capturées."""
    entry = entry_with_pollster(entry)
    res = NoticeCertainty(entry["file_id"], election, entry)
    try:
        text = nt.extract_text(path, column_pages=2)
    except (OSError, ValueError, PSException) as exc:  # PDF illisible : signalé, pas avalé
        res.error = f"pdf : {exc}"
        return res
    res.cue_pages = sum(ce.is_certainty_page(p) for p in text.pages)
    if not res.cue_pages:
        return res
    tables = ce.find_certainty_tables(text.pages, ce.candidate_lookup(labels))
    res.rows, res.total = ce.merge_tables(tables)
    if not res.rows and res.total is None:
        return res
    try:
        meta = notice_metadata(text, entry)
    except KeyError as exc:
        res.error = f"institut : {exc}"
        return res
    res.pollster = meta.pollster
    res.fieldwork_start, res.fieldwork_end = meta.fieldwork_start, meta.fieldwork_end
    if meta.fieldwork_end is None:
        res.error = "dates de terrain introuvables"
    return res


def _read_one(args) -> NoticeCertainty:
    path, entry, election, labels = args
    return read_notice(Path(path), entry, election, labels)


def match_keys(polls: pl.DataFrame, election: int, pollster: str, start, end) -> list[tuple]:
    """Sondages de `polls.parquet` compatibles avec (institut, terrain) : [(écart de fin,
    écart de début, poll_id)], fin à un jour près, début à trois jours près."""
    sub = polls.filter((pl.col("election") == election) & (pl.col("pollster") == pollster))
    out = []
    for r in sub.iter_rows(named=True):
        d_end = abs((r["fieldwork_end"] - end).days)
        d_start = abs((r["fieldwork_start"] - start).days) if start and r["fieldwork_start"] else 0
        if d_end <= 1 and d_start <= 3:
            out.append((d_end, d_start, r["poll_id"]))
    return out


def align_all(notices: list[NoticeCertainty], polls: pl.DataFrame) -> dict[int, str]:
    """Appariement un à un notice -> poll_id, des écarts les plus faibles aux plus forts (les
    vagues quotidiennes d'un rolling se chevauchent : deux vagues ne prennent pas le même
    sondage)."""
    cands = []
    for n in notices:
        for d_end, d_start, pid in match_keys(
            polls, n.election, n.pollster, n.fieldwork_start, n.fieldwork_end
        ):
            cands.append((d_end, d_start, n.file_id, pid))
    out: dict[int, str] = {}
    taken: set[str] = set()
    for _, _, fid, pid in sorted(cands):
        if fid not in out and pid not in taken:
            out[fid] = pid
            taken.add(pid)
    return out


def build_table(notices: list[NoticeCertainty], polls: pl.DataFrame) -> pl.DataFrame:
    reg = default_registry()
    kept = []
    seen: set[tuple] = set()
    for n in sorted(notices, key=lambda n: n.file_id):
        if n.error or n.fieldwork_end is None or (not n.rows and n.total is None):
            continue
        if (n.election, n.file_id) in EXCLUDED_NOTICES or (
            n.election == 2027 and n.file_id in PARSE_EXCLUDED
        ):
            continue
        vote = reg.elections[n.election].round_(1).date if n.election in reg.elections else None
        if vote is not None and not (
            vote - dt.timedelta(days=WINDOW_DAYS) <= n.fieldwork_end < vote
        ):
            continue
        key = (n.pollster, n.fieldwork_start, n.fieldwork_end)
        if key in seen:
            continue  # notice complémentaire ou rectificative d'un sondage déjà lu
        seen.add(key)
        kept.append(n)
    matched = align_all(kept, polls)
    rows = []
    for n in kept:
        pid = matched.get(n.file_id)
        aligned = pid is not None
        if pid is None:
            pid = f"{n.election}-{n.pollster}-{n.fieldwork_end:%Y%m%d}-cert{n.file_id}"
        base = {
            "poll_id": pid,
            "election": n.election,
            "source": "commission" if n.file_id > 0 else "ifop_report",
            "source_url": n.entry["url"],
            "pollster": n.pollster,
            "fieldwork_start": n.fieldwork_start,
            "fieldwork_end": n.fieldwork_end,
            "file_id": n.file_id,
            "aligned": aligned,
        }
        for cid, r in n.rows.items():
            rows.append(
                base
                | {"candidate_id": cid, "certain_share": r.certain_share, "fmt": r.fmt,
                   "page": r.page}
            )
        if n.total is not None:
            rows.append(
                base
                | {"candidate_id": ce.TOTAL_ID, "certain_share": n.total, "fmt": "total",
                   "page": None}
            )
    return pl.DataFrame(rows, schema=SCHEMA)


@dataclass
class CertaintyReport:
    notices: list[NoticeCertainty]
    table: pl.DataFrame

    def coverage(self) -> pl.DataFrame:
        """Couverture par élection et institut : notices lues, notices posant la question,
        sondages avec ventilation par candidat, lignes."""
        info = pl.DataFrame(
            [
                {
                    "election": n.election,
                    "pollster": n.pollster or _title_pollster(n.entry["title"]),
                    "asks": n.cue_pages > 0,
                }
                for n in self.notices
            ],
            schema={"election": pl.Int32, "pollster": pl.Utf8, "asks": pl.Boolean},
        )
        asked = info.group_by("election", "pollster").agg(
            pl.len().alias("notices"), pl.col("asks").sum().alias("notices_asking")
        )
        t = self.table.filter(pl.col("candidate_id") != ce.TOTAL_ID)
        by = t.group_by("election", "pollster").agg(
            pl.col("poll_id").n_unique().alias("polls_by_candidate"),
            pl.len().alias("rows"),
            pl.col("fieldwork_end").min().alias("first"),
            pl.col("fieldwork_end").max().alias("last"),
        )
        tot = (
            self.table.filter(pl.col("candidate_id") == ce.TOTAL_ID)
            .group_by("election", "pollster")
            .agg(pl.len().alias("polls_total_only_or_both"))
        )
        return (
            asked.join(by, on=["election", "pollster"], how="left")
            .join(tot, on=["election", "pollster"], how="left")
            .sort("election", "pollster")
        )


def _title_pollster(title: str) -> str | None:
    from vvs.ingest.commission.aliases import find_pollster_in_title

    found = find_pollster_in_title(title)
    return found[0] if found else None


def run(
    elections=ELECTIONS, raw_dir: Path = RAW_DIR, workers: int = 8, write: bool = True
) -> CertaintyReport:
    labels = candidate_labels()
    jobs = []
    for e in elections:
        d = raw_dir_for(e, raw_dir)
        for _, entry in sorted(load_index(d / "index.jsonl").items()):
            path = d / entry["file"]
            if path.suffix == ".pdf":
                jobs.append((str(path), entry, e, labels))
    ifop_jobs = []
    if 2022 in elections:
        names = ifop_names(2022)
        d = raw_dir.parent / "institutes" / "ifop" / "2022"
        for entry in load_index_any(d / "index.jsonl"):
            ifop_jobs.append((str(d / entry["file"]), entry, names))
    with cf.ProcessPoolExecutor(workers) as ex:
        notices = list(ex.map(_read_one, jobs, chunksize=4))
        notices += list(ex.map(_read_ifop_one, ifop_jobs, chunksize=4))
    polls = pl.read_parquet(paths.CURATED / "polls.parquet")
    table = build_table(notices, polls)
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        table.write_parquet(OUT_PATH)
    return CertaintyReport(notices, table)


def load(path: Path = OUT_PATH) -> pl.DataFrame:
    return pl.read_parquet(path)


def dump_report(report: CertaintyReport, path: Path) -> None:
    """Détail par notice (questions posées sans ventilation lisible), pour le diagnostic."""
    with path.open("w") as f:
        for n in report.notices:
            if n.cue_pages:
                f.write(
                    json.dumps(
                        {
                            "file_id": n.file_id,
                            "election": n.election,
                            "title": n.entry["title"],
                            "pollster": n.pollster,
                            "fieldwork_end": str(n.fieldwork_end),
                            "n_rows": len(n.rows),
                            "total": n.total,
                            "error": n.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ------------------------------------------------- rapports Ifop (rolling 2022, hors notices)
#
# Les notices du rolling Ifop-Fiducial 2022 ne déposent que la part des sûrs dans l'ensemble ;
# le rapport publié chaque jour sur le site de l'institut la ventile par candidat (page « La
# sûreté du choix au 1er tour de l'élection présidentielle », prénom / « NN % » / NOM sur trois
# lignes). URL régulière : /wp-content/uploads/<AAAA>/<MM>/ROL22_<AAAA.MM.JJ>.pdf, redirigée
# vers ifopgroup.com. Le serveur n'envoie pas son certificat intermédiaire (GlobalSign GCC R46
# OV TLS CA 2025) : on l'ajoute au magasin de certifi après l'avoir lu à l'adresse donnée par le
# certificat (champ AIA) ; la vérification TLS reste complète (l'intermédiaire doit être signé
# par une racine de confiance). Pour 2017, les rapports Ifop sont archivés sous des numéros
# d'étude non prévisibles : non collectés.

IFOP_DIR = paths.RAW / "institutes" / "ifop"
IFOP_URL = "https://www.ifop.com/wp-content/uploads/{d:%Y}/{d:%m}/ROL22_{d:%Y.%m.%d}.pdf"
IFOP_ROLLING_2022 = (dt.date(2022, 1, 10), dt.date(2022, 4, 8))
GLOBALSIGN_AIA = "http://secure.globalsign.com/cacert/gsgccr46ovtlsca2025.crt"


def _tls_bundle(raw_dir: Path = IFOP_DIR) -> Path:
    import ssl

    import certifi
    import httpx

    path = raw_dir / "ca_bundle.pem"
    if not path.exists():
        der = httpx.get(GLOBALSIGN_AIA, timeout=30).content
        pem = ssl.DER_cert_to_PEM_cert(der)
        raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(Path(certifi.where()).read_text() + pem)
    return path


def ingest_ifop_rolling(
    start: dt.date = IFOP_ROLLING_2022[0],
    end: dt.date = IFOP_ROLLING_2022[1],
    raw_dir: Path = IFOP_DIR,
    delay_s: float = 2.0,
) -> dict:
    """Télécharge les rapports quotidiens du rolling Ifop 2022 (idempotent, index en ajout)."""
    import time

    import httpx

    from vvs.ingest.commission.listing import USER_AGENT

    out_dir = raw_dir / "2022"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    done = {r["file"] for r in load_index_any(index_path)}
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True,
                          timeout=60.0, verify=str(_tls_bundle(raw_dir)))
    stats = {"tried": 0, "downloaded": 0, "missing": 0}
    d = start
    with index_path.open("a") as out:
        while d <= end:
            name = f"ROL22_{d:%Y.%m.%d}.pdf"
            if name not in done:
                stats["tried"] += 1
                r = client.get(IFOP_URL.format(d=d))
                time.sleep(delay_s)
                if r.status_code == 200 and "pdf" in r.headers.get("content-type", ""):
                    (out_dir / name).write_bytes(r.content)
                    out.write(json.dumps({"file": name, "publish_date": d.isoformat(),
                                          "url": str(r.url),
                                          "ingested_at": dt.datetime.now(dt.UTC)
                                          .replace(tzinfo=None).isoformat()}) + "\n")
                    stats["downloaded"] += 1
                else:
                    stats["missing"] += 1
            d += dt.timedelta(days=1)
    return stats


def load_index_any(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def parse_ifop_rolling_page(page: str, names: dict[str, tuple[str, str]]) -> dict[str, float]:
    """Page « La sûreté du choix au 1er tour » d'un rapport du rolling Ifop. Deux mises en
    page : prénom seul sur une ligne puis « NN % » sur la suivante (mars 2022), ou
    « Prénom NN % » sur une ligne, parfois en fin de ligne de légende (janvier 2022) ; le nom
    en capitales termine l'une des deux lignes suivantes. `names` : {id: (prénom, nom)}."""
    from unidecode import unidecode

    from vvs.ingest.commission.aliases import norm_label

    head = norm_label(" ".join(page.splitlines()[:2]))
    if "surete du choix au 1er tour" not in head or "electorat" in head:
        return {}
    lines = page.splitlines()
    low = [re.sub(r"[^a-z0-9%]+", " ", unidecode(x).lower()).strip() for x in lines]
    out: dict[str, float] = {}
    for cid, (first, last) in names.items():
        f, ln = norm_label(first), norm_label(last)
        for k in range(len(lines) - 1):
            m = re.search(rf"(?:^|\s){re.escape(f)}\s+(\d{{1,3}})\s?%", low[k])
            j = k + 1
            if not m and norm_label(lines[k]) == f:
                m, j = re.match(r"^\s*(\d{1,3})\s?%", lines[k + 1]), k + 2
            if not m:
                continue
            if any(norm_label(x).endswith(ln) for x in lines[j : j + 2]):
                out.setdefault(cid, float(m.group(1)))
                break
    return out


def read_ifop_report(path: Path, entry: dict, names: dict) -> NoticeCertainty:
    pub = dt.date.fromisoformat(entry["publish_date"])
    res = NoticeCertainty(-int(f"{pub:%Y%m%d}"), 2022, {"url": entry["url"], "title": path.name})
    res.pollster = "ifop"
    try:
        text = nt.extract_text(path, column_pages=0)
    except (OSError, ValueError, PSException) as exc:
        res.error = f"pdf : {exc}"
        return res
    res.cue_pages = sum(ce.is_certainty_page(p) for p in text.pages)
    for k, page in enumerate(text.pages, start=1):
        vals = parse_ifop_rolling_page(page, names)
        if len(vals) >= 2:
            res.rows = {c: ce.CertaintyRow(c, v, "ifop_report", k, (v,)) for c, v in vals.items()}
            break
    fw = nt.find_fieldwork("\n".join(text.pages[-6:]), require_cue=True,
                           window=(pub - dt.timedelta(days=10), pub))
    res.fieldwork_start, res.fieldwork_end = fw if fw else (pub - dt.timedelta(days=3),
                                                            pub - dt.timedelta(days=1))
    return res


def _read_ifop_one(args) -> NoticeCertainty:
    path, entry, names = args
    return read_ifop_report(Path(path), entry, names)


def ifop_names(election: int = 2022) -> dict[str, tuple[str, str]]:
    reg = default_registry()
    e = reg.elections[election]
    return {c.candidate_id: (reg.candidates[c.candidate_id].first_name,
                             reg.candidates[c.candidate_id].last_name) for c in e.candidates}
