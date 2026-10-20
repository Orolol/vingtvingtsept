"""Ingestion du dépôt MieuxVoter/presidentielle2027, contrôle croisé des sondages 2027.

Source
------

<https://github.com/MieuxVoter/presidentielle2027>, licence MIT, successeur de fait de
NSPPolls (même logique : une ligne par sondage × hypothèse, chaque ligne traçable jusqu'à la
notice PDF de la Commission des sondages). Le dépôt est tenu à la main par des bénévoles,
assistés d'une extraction LLM des notices relue en pull request ; il accuse en général une
dizaine de jours de retard sur la Commission.

Fichiers utilisés (à la racine du dépôt, branche `main`) :

- `presidentielle2027.json`, produit par `csv_to_json.py` depuis `presidentielle2027.csv` :

      {
        "source": {...},                 # nom, dépôt, consignes de citation
        "usage_guidelines": [...],       # recommandations d'usage (texte)
        "hypotheses": [                  # catalogue des hypothèses de premier tour
          {"id": "H1", "candidates": ["Nathalie Arthaud", ...], "comment": "..."}
        ],
        "polls": [                       # une entrée par sondage × hypothèse × tour
          {
            "poll_id": "20240707_0708_hi_A",   # <début>_<fin MMJJ>_<institut>_<lettre>
            "institut": "Harris Interactive",
            "commanditaire": "M6, RTL",        # texte libre, médias séparés par « , »
            "debut_enquete": "2024-07-07", "fin_enquete": "2024-07-08",
            "echantillon": 2496,               # échantillon total
            "population": "représentatif de la population française âgée de 18 ans et plus",
            "hypothese": "H2",                 # id du catalogue (« H2_1 »… au second tour,
                                               # vide pour certaines entrées)
            "tour": "1er Tour" | "2nd Tour",
            "rolling": "", "media": "True",
            "filename": "9888-pres-harris-interactive-challenges-m6-rtl-le-8-juillet.pdf",
            "candidats": [
              {"candidate_id": "NA", "candidat": "Nathalie Arthaud",
               "complete_name": "Nathalie Arthaud", "name": "Nathalie", "surname": "Arthaud",
               "parti": "Lutte Ouvrière", "annonce_candidature": "2025-12-08",
               "retrait_candidature": "", "second_round": "",
               "intentions": 1, "erreur_sup": 0.41, "erreur_inf": -0.41}
            ]
          }
        ],
        "contribution": {...}
      }

  Il n'y a ni niveau « région » ni regroupement par sondage : la hiérarchie sondage →
  tour → hypothèse → candidat est à plat, une entrée par (sondage, tour, hypothèse). Les
  intentions sont en pourcentage des exprimés (sommes de 98 à 102) ; ni abstention, ni
  blancs, ni indécis. `erreur_sup`/`erreur_inf` sont des marges calculées par le dépôt
  (`compute_confidence_intervals.py`), pas des données de la notice : on ne les garde pas.

- `polls.csv` : métadonnées des mêmes entrées (même `poll_id`), avec en plus ce que le JSON
  ne reprend pas : jusqu'à trois sous-échantillons (`sous_echantillon<i>`,
  `sous_population<i>`), par exemple « Inscrits sur les listes électorales » puis
  « certains d'aller voter ». C'est la seule information de population de la source.

Champs réellement disponibles (relevé du 20 octobre 2026)
---------------------------------------------------------

- institut, commanditaire, dates de terrain, échantillon total : toujours ;
- population interrogée : le champ `population` du JSON décrit l'échantillon total
  (« représentatif de la population française… ») et ne distingue pas inscrits et certains
  d'aller voter. La distinction vient des sous-échantillons de `polls.csv`, déclaratifs et
  inégalement renseignés (Elabe s'arrête à « Inscrits » alors que ses intentions portent sur
  les certains d'aller voter selon la Commission) ;
- mode de collecte : absent (`mode` = `unknown`, comme pour Wikipédia) ;
- date de publication : absente (`publish_date` nul) ;
- notice de la Commission : nom du PDF (`filename`) seulement, pas d'URL. On le résout en
  URL de notice par l'index de l'ingestion Commission (`data/raw/commission/index.jsonl`,
  champ `file_url`) ; à défaut (index absent ou notice antérieure à sa couverture),
  `source_url` pointe sur le JSON au commit téléchargé.

Stockage brut
-------------

`vvs ingest mieuxvoter` lit le SHA du dernier commit de `main` (API GitHub), puis
télécharge à ce commit le JSON et `polls.csv`, de sorte que les deux soient cohérents :

    data/raw/mieuxvoter/<AAAA-MM-JJ>.json        # presidentielle2027.json, octets inchangés
    data/raw/mieuxvoter/<AAAA-MM-JJ>.polls.csv   # polls.csv, octets inchangés
    data/raw/mieuxvoter/<AAAA-MM-JJ>.meta.json   # dépôt, SHA, URLs, horodatage UTC

Un jour = un instantané : si le JSON du jour existe, rien n'est retéléchargé ni réécrit.

Curation
--------

`vvs parse mieuxvoter` lit le dernier instantané et écrit `polls_mieuxvoter.parquet` et
`poll_results_mieuxvoter.parquet` (schémas de `vvs.schemas`, `source` = `mieuxvoter`) :

- un sondage = (institut, début, fin, notice) ; `poll_id` =
  `2027-{pollster}-{fieldwork_end:%Y%m%d}-{seq}`, `seq` dans l'ordre (début, échantillon,
  commanditaire, notice) comme pour Wikipédia ;
- `hypothesis_id` recalculé depuis les candidats (`schemas.hypothesis_id`), pas repris du
  catalogue `H…` du dépôt ;
- libellés de candidats et d'instituts résolus par `data/registry/aliases_mieuxvoter.yaml` ;
  un libellé inconnu lève `UnknownLabelError` ;
- `population` : `certain_voters` si un sous-échantillon déclaré filtre les certains d'aller
  voter (« certains d'aller voter », « comptant aller voter », « votants décidés »),
  `registered` s'il ne mentionne que les inscrits, `unknown` sinon — même règle que
  l'ingestion Commission ;
- exclusions explicites (`EXCLUDED_NOTICES`) : sondages sur une sous-population.

`compare_with_wikipedia()` rapproche cette source des parquets Wikipédia 2027.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import httpx
import polars as pl
import yaml

from vvs import paths, schemas
from vvs.ingest.wikipedia.common import UnknownLabelError, normalize_label
from vvs.registry.models import candidate_id_for

REPO = "MieuxVoter/presidentielle2027"
BRANCH = "main"
COMMIT_API_URL = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
RAW_FILE_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
BLOB_URL = "https://github.com/{repo}/blob/{sha}/{path}"
JSON_PATH = "presidentielle2027.json"
POLLS_CSV_PATH = "polls.csv"
USER_AGENT = "vingtvingtsept/0.1 (French 2027 presidential forecast; https://github.com/Orolol)"

ELECTION = 2027
RAW_DIR = paths.RAW / "mieuxvoter"
ALIASES_PATH = paths.REGISTRY / "aliases_mieuxvoter.yaml"
COMMISSION_INDEX_PATH = paths.RAW / "commission" / "index.jsonl"
POLLS_OUTPUT = "polls_mieuxvoter.parquet"
RESULTS_OUTPUT = "poll_results_mieuxvoter.parquet"

ROUNDS = {"1er Tour": 1, "2nd Tour": 2}

# Notices écartées par règle explicite (clé : `filename` du dépôt) : intentions de vote
# mesurées sur une sous-population, non comparables aux sondages en population générale.
EXCLUDED_NOTICES: dict[str, str] = {
    "10141-pres-vote-lgbt-ifop-tetu-9-mars.pdf": "sous-population : électeurs LGBT+ (Ifop pour Têtu)",
}

_CERTAIN_RE = re.compile(r"certain(e|s|es)? d aller voter|comptant aller voter|votants decides")
_REGISTERED_RE = re.compile(r"\binscrit")
_DATE_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


# ---------------------------------------------------------------------------
# Téléchargement et stockage brut
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """Instantané brut : JSON, polls.csv et métadonnées d'un même commit."""

    json_text: str
    polls_csv_text: str
    meta: dict

    @property
    def sha(self) -> str:
        return self.meta["sha"]

    @property
    def fetched_at(self) -> datetime:
        """Horodatage UTC naïf (comme `ingested_at` des autres sources)."""
        return datetime.fromisoformat(self.meta["fetched_at"]).astimezone(UTC).replace(tzinfo=None)

    @property
    def json_url(self) -> str:
        return BLOB_URL.format(repo=self.meta["repository"], sha=self.sha, path=JSON_PATH)


def _snapshot_paths(day: date, raw_dir: Path) -> tuple[Path, Path, Path]:
    stem = raw_dir / day.isoformat()
    return stem.with_suffix(".json"), Path(f"{stem}.polls.csv"), Path(f"{stem}.meta.json")


def fetch_snapshot(client: httpx.Client | None = None) -> Snapshot:
    """Télécharge le JSON et polls.csv au dernier commit de `main`."""
    own = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True
    )
    try:
        resp = client.get(COMMIT_API_URL, headers={"Accept": "application/vnd.github+json"})
        resp.raise_for_status()
        commit = resp.json()
        sha = commit["sha"]
        files = {}
        for key, path in (("json", JSON_PATH), ("polls_csv", POLLS_CSV_PATH)):
            url = RAW_FILE_URL.format(repo=REPO, sha=sha, path=path)
            r = client.get(url)
            r.raise_for_status()
            files[key] = (url, r.content.decode("utf-8"))
    finally:
        if own:
            client.close()
    json.loads(files["json"][1])  # un JSON tronqué ou une page d'erreur échoue ici
    meta = {
        "repository": REPO,
        "branch": BRANCH,
        "sha": sha,
        "commit_date": commit["commit"]["committer"]["date"],
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "files": {k: url for k, (url, _) in files.items()},
    }
    return Snapshot(files["json"][1], files["polls_csv"][1], meta)


def store_snapshot(snap: Snapshot, raw_dir: Path = RAW_DIR) -> tuple[Path, bool]:
    """Écrit l'instantané sous la date UTC de téléchargement ; (chemin du JSON, écrit ?).

    `data/raw/` est en ajout seul : si le JSON du jour existe, on ne touche à rien. Le JSON
    est écrit en dernier et sert de marqueur d'instantané complet ; des fichiers compagnons
    sans JSON signalent une ingestion interrompue et lèvent plutôt que d'être écrasés.
    """
    json_path, csv_path, meta_path = _snapshot_paths(snap.fetched_at.date(), raw_dir)
    if json_path.exists():
        return json_path, False
    for p in (csv_path, meta_path):
        if p.exists():
            raise FileExistsError(f"{p} existe sans {json_path.name} : ingestion interrompue ?")
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(snap.meta, ensure_ascii=False, indent=2) + "\n", "utf-8")
    csv_path.write_text(snap.polls_csv_text, encoding="utf-8")
    json_path.write_text(snap.json_text, encoding="utf-8")
    return json_path, True


def ingest(raw_dir: Path = RAW_DIR) -> tuple[Path, bool]:
    """`vvs ingest mieuxvoter` : un instantané par jour, jamais réécrit."""
    existing, _, _ = _snapshot_paths(datetime.now(UTC).date(), raw_dir)
    if existing.exists():
        return existing, False
    return store_snapshot(fetch_snapshot(), raw_dir)


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Snapshot:
    """Dernier instantané complet (ordre des dates)."""
    files = sorted(p for p in raw_dir.glob("*.json") if _DATE_FILE_RE.match(p.name))
    if not files:
        raise FileNotFoundError(f"aucun instantané dans {raw_dir} : lancer `vvs ingest mieuxvoter`")
    json_path, csv_path, meta_path = _snapshot_paths(date.fromisoformat(files[-1].stem), raw_dir)
    return Snapshot(
        json_path.read_text(encoding="utf-8"),
        csv_path.read_text(encoding="utf-8"),
        json.loads(meta_path.read_text(encoding="utf-8")),
    )


# ---------------------------------------------------------------------------
# Alias et résolutions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Aliases:
    candidates: dict[str, str]
    pollsters: dict[str, str]
    path: Path = ALIASES_PATH

    def _where(self) -> str:
        try:
            return str(self.path.relative_to(paths.ROOT))
        except ValueError:
            return str(self.path)

    def candidate(self, label: str) -> str:
        try:
            return self.candidates[normalize_label(label)]
        except KeyError:
            raise UnknownLabelError(
                f"libellé de candidat inconnu : {label!r} ; ajoute-le à {self._where()}"
            ) from None

    def pollster(self, label: str) -> str:
        try:
            return self.pollsters[normalize_label(label)]
        except KeyError:
            raise UnknownLabelError(
                f"libellé d'institut inconnu : {label!r} ; ajoute-le à {self._where()}"
            ) from None


@lru_cache(maxsize=4)
def load_aliases(path: Path = ALIASES_PATH) -> Aliases:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for section in ("candidates", "pollsters"):
        table: dict[str, str] = {}
        for label, cid in (raw.get(section) or {}).items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", str(cid)):
                raise ValueError(f"{path.name}:{section}:{label} : id invalide {cid!r}")
            key = normalize_label(str(label))
            if key in table and table[key] != cid:
                raise ValueError(f"{path.name}:{section} : clé {key!r} ambiguë")
            table[key] = cid
        out[section] = table
    return Aliases(out["candidates"], out["pollsters"], path)


def population_of(sub_populations: list[str]) -> str:
    """Population des intentions d'après les sous-échantillons déclarés dans polls.csv."""
    text = normalize_label(" ".join(sub_populations))
    if _CERTAIN_RE.search(text):
        return "certain_voters"
    if _REGISTERED_RE.search(text):
        return "registered"
    return "unknown"


def load_notice_urls(path: Path = COMMISSION_INDEX_PATH) -> dict[str, str]:
    """Nom de PDF de notice -> URL de la notice (index de l'ingestion Commission).

    Index absent : dictionnaire vide (les `source_url` retombent sur le dépôt).
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("file_url"):
                out[e["file_url"].rsplit("/", 1)[-1]] = e["url"]
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class ParseReport:
    sha: str
    excluded: list[str] = field(default_factory=list)
    counters: Counter = field(default_factory=Counter)
    population: Counter = field(default_factory=Counter)  # entrées par population

    def summary(self) -> str:
        lines = [f"MieuxVoter (commit {self.sha[:10]}) :"]
        for k, v in sorted(self.counters.items()):
            lines.append(f"  {k}: {v}")
        lines.append(
            "  population des hypothèses: "
            + ", ".join(f"{k} {v}" for k, v in sorted(self.population.items()))
        )
        lines.append(f"  entrées exclues: {len(self.excluded)}")
        lines.extend(f"    - {x}" for x in self.excluded)
        return "\n".join(lines)


def _sub_populations(polls_csv_text: str) -> dict[str, list[str]]:
    out = {}
    for row in csv.DictReader(io.StringIO(polls_csv_text)):
        out[row["poll_id"]] = [
            row[f"sous_population{i}"] for i in (1, 2, 3) if row.get(f"sous_population{i}")
        ]
    return out


def parse(
    snap: Snapshot,
    aliases: Aliases | None = None,
    notice_urls: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, ParseReport]:
    """(polls, poll_results, rapport) d'un instantané, validés par `vvs.schemas`."""
    aliases = aliases or load_aliases()
    notice_urls = load_notice_urls() if notice_urls is None else notice_urls
    report = ParseReport(snap.sha)
    subpops = _sub_populations(snap.polls_csv_text)
    entries = json.loads(snap.json_text)["polls"]

    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        if e["filename"] in EXCLUDED_NOTICES:
            report.excluded.append(f"{e['poll_id']} : {EXCLUDED_NOTICES[e['filename']]}")
            continue
        pollster = aliases.pollster(e["institut"])
        start = date.fromisoformat(e["debut_enquete"])
        end = date.fromisoformat(e["fin_enquete"])
        groups.setdefault((pollster, start, end, e["filename"]), []).append(e)
    report.counters["entrées sondage × hypothèse"] = len(entries)

    by_day: dict[tuple, list[tuple]] = {}
    for key, group in groups.items():
        by_day.setdefault((key[0], key[2]), []).append(key)

    poll_rows, result_rows = [], []
    for (pollster, end), keys in sorted(by_day.items()):

        def order(k: tuple) -> tuple:
            first = groups[k][0]
            return (k[1], first["echantillon"] or -1, first["commanditaire"] or "", k[3])

        for seq, key in enumerate(sorted(keys, key=order), start=1):
            _, start, _, filename = key
            group = groups[key]
            poll_id = f"{ELECTION}-{pollster}-{end:%Y%m%d}-{seq}"
            meta_fields = {(g["echantillon"], g["commanditaire"]) for g in group}
            if len(meta_fields) > 1:
                raise ValueError(f"{poll_id} : échantillon ou commanditaire variable {meta_fields}")
            sample, sponsor = meta_fields.pop()
            url = notice_urls.get(filename)
            report.counters["source_url notice Commission" if url else "source_url dépôt"] += 1
            poll_rows.append(
                {
                    "poll_id": poll_id,
                    "election": ELECTION,
                    "pollster": pollster,
                    "sponsor": sponsor or None,
                    "fieldwork_start": start,
                    "fieldwork_end": end,
                    "publish_date": None,
                    "sample_size": sample,
                    "mode": "unknown",
                    "source": "mieuxvoter",
                    "source_url": url or snap.json_url,
                    "ingested_at": snap.fetched_at,
                }
            )
            seen: dict[tuple, dict[str, float]] = {}
            for e in group:
                rnd = ROUNDS[e["tour"]]
                if e["poll_id"] not in subpops:
                    raise ValueError(f"{e['poll_id']} absent de polls.csv")
                population = population_of(subpops[e["poll_id"]])
                shares: dict[str, float] = {}
                for c in e["candidats"]:
                    cid = aliases.candidate(c["candidat"])
                    expected = candidate_id_for(c["surname"], c["name"])
                    if cid != expected:
                        raise ValueError(
                            f"{e['poll_id']} : {c['candidat']!r} -> {cid} dans les alias, mais "
                            f"nom/prénom du JSON -> {expected}"
                        )
                    if c["intentions"] is None:
                        raise ValueError(f"{e['poll_id']} : intention manquante pour {cid}")
                    if cid in shares:
                        raise ValueError(f"{e['poll_id']} : {cid} en double")
                    shares[cid] = float(c["intentions"])
                hid = schemas.hypothesis_id(list(shares))
                if (rnd, hid) in seen:
                    if seen[(rnd, hid)] != shares:
                        raise ValueError(f"{poll_id} : hypothèse {hid} répétée, parts différentes")
                    report.counters["hypothèses répétées à l'identique"] += 1
                    continue
                seen[(rnd, hid)] = shares
                report.population[population] += 1
                for cid, share in shares.items():
                    result_rows.append(
                        {
                            "poll_id": poll_id,
                            "round": rnd,
                            "hypothesis_id": hid,
                            "population": population,
                            "candidate_id": cid,
                            "share": share,
                            "share_base": "expressed",
                            "raw_share": None,
                        }
                    )
    polls_df = pl.DataFrame(poll_rows, schema=schemas.POLLS_SCHEMA)
    results_df = pl.DataFrame(result_rows, schema=schemas.POLL_RESULTS_SCHEMA)
    polls_df = schemas.validate_polls(polls_df.sort("fieldwork_end", "poll_id"))
    results_df = schemas.validate_poll_results(results_df)
    report.counters["sondages"] = polls_df.height
    report.counters["lignes poll_results"] = results_df.height
    for rnd in (1, 2):
        sub = results_df.filter(pl.col("round") == rnd)
        report.counters[f"hypothèses tour {rnd}"] = sub.select(
            "poll_id", "hypothesis_id"
        ).n_unique()
    return polls_df, results_df, report


def run(raw_dir: Path = RAW_DIR, out_dir: Path = paths.CURATED) -> tuple[Path, Path, ParseReport]:
    """`vvs parse mieuxvoter` : dernier instantané -> deux parquets."""
    polls, results, report = parse(latest_snapshot(raw_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    polls_path, results_path = out_dir / POLLS_OUTPUT, out_dir / RESULTS_OUTPUT
    polls.write_parquet(polls_path)
    results.write_parquet(results_path)
    return polls_path, results_path, report


# ---------------------------------------------------------------------------
# Contrôle croisé avec Wikipédia
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """Rapprochement MieuxVoter / Wikipédia (voir `compare_with_wikipedia`)."""

    matches: pl.DataFrame  # couples de sondages appariés
    only_mieuxvoter: pl.DataFrame
    only_wikipedia: pl.DataFrame  # colonne `in_window` : terrain dans la période MieuxVoter
    hypotheses: pl.DataFrame  # par couple et tour : hypothèses communes / d'un seul côté
    share_gaps: pl.DataFrame  # parts différant de plus de `tolerance` sur une hypothèse commune
    probable_matches: pl.DataFrame  # non appariés aux dates trop éloignées, mêmes résultats
    tolerance: float

    def summary(self, verbose: bool = False) -> str:
        h = self.hypotheses
        lines = [
            "Contrôle croisé MieuxVoter / Wikipédia 2027 :",
            (
                f"  sondages appariés: {self.matches.height}"
                f" (dates décalées d'un jour : {self.matches.filter(pl.col('date_shift') > 0).height})"
            ),
            f"  seulement MieuxVoter: {self.only_mieuxvoter.height}",
            f"  dont même sondage probable aux dates divergentes: {self.probable_matches.height}",
            (
                f"  seulement Wikipédia: {self.only_wikipedia.height}"
                f" (dont {self.only_wikipedia['in_window'].sum()} dans la période couverte"
                " par MieuxVoter)"
            ),
            (
                f"  hypothèses communes: {h['common'].sum()} ; seulement MieuxVoter:"
                f" {h['only_mieuxvoter'].sum()} ; seulement Wikipédia: {h['only_wikipedia'].sum()}"
            ),
            (
                f"  parts différant de plus de {self.tolerance} point: {self.share_gaps.height}"
                f" (sur {self.share_gaps.select('poll_id_mv', 'round', 'hypothesis_id').n_unique()}"
                " hypothèses)"
            ),
        ]
        if verbose:
            lines.append("  seulement MieuxVoter :")
            for r in self.only_mieuxvoter.iter_rows(named=True):
                lines.append(f"    {r['poll_id']} {r['fieldwork_start']}→{r['fieldwork_end']}")
            lines.append("  même sondage probable (hypothèses identiques, dates divergentes) :")
            for r in self.probable_matches.iter_rows(named=True):
                lines.append(
                    f"    MV {r['poll_id_mv']} {r['fieldwork_start']}→{r['fieldwork_end']}"
                    f" n={r['sample_size']} / Wikipédia {r['poll_id_wiki']}"
                    f" {r['fieldwork_start_wiki']}→{r['fieldwork_end_wiki']} n={r['sample_size_wiki']}"
                    f" : {r['identical_hypotheses']} hypothèses identiques"
                )
            lines.append("  seulement Wikipédia :")
            for r in self.only_wikipedia.iter_rows(named=True):
                tag = "" if r["in_window"] else " (hors période MieuxVoter)"
                lines.append(f"    {r['poll_id']} {r['fieldwork_start']}→{r['fieldwork_end']}{tag}")
            lines.append("  hypothèses d'un seul côté (sondages appariés) :")
            for r in h.filter(
                (pl.col("only_mieuxvoter") > 0) | (pl.col("only_wikipedia") > 0)
            ).iter_rows(named=True):
                lines.append(
                    f"    {r['poll_id_mv']} / {r['poll_id_wiki']} tour {r['round']} :"
                    f" MV seul {r['only_mieuxvoter']}, Wikipédia seul {r['only_wikipedia']}"
                )
            lines.append("  écarts de parts :")
            for r in self.share_gaps.iter_rows(named=True):
                lines.append(
                    f"    {r['poll_id_mv']} / {r['poll_id_wiki']} tour {r['round']}"
                    f" {r['candidate_id']}: MV {r['share_mv']} / Wikipédia {r['share_wiki']}"
                )
        return "\n".join(lines)


def _match_polls(mv: pl.DataFrame, wiki: pl.DataFrame, max_shift: int) -> pl.DataFrame:
    """Appariement un à un sur (pollster, début, fin) à `max_shift` jours près.

    Glouton par décalage total croissant, puis par écart d'échantillon : un sondage n'est
    apparié qu'une fois de chaque côté.
    """
    cols = ["poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size"]
    cand = (
        mv.select(cols)
        .join(wiki.select(cols), on="pollster", suffix="_wiki")
        .with_columns(
            ds=(pl.col("fieldwork_start") - pl.col("fieldwork_start_wiki")).dt.total_days().abs(),
            de=(pl.col("fieldwork_end") - pl.col("fieldwork_end_wiki")).dt.total_days().abs(),
            dn=(pl.col("sample_size") - pl.col("sample_size_wiki")).abs().fill_null(10**9),
        )
        .filter((pl.col("ds") <= max_shift) & (pl.col("de") <= max_shift))
        .with_columns(date_shift=pl.col("ds") + pl.col("de"))
        .sort("date_shift", "dn", "poll_id", "poll_id_wiki")
    )
    used_mv, used_wiki, rows = set(), set(), []
    for r in cand.iter_rows(named=True):
        if r["poll_id"] in used_mv or r["poll_id_wiki"] in used_wiki:
            continue
        used_mv.add(r["poll_id"])
        used_wiki.add(r["poll_id_wiki"])
        rows.append(r)
    schema = dict(cand.schema)
    return (
        pl.DataFrame(rows, schema=schema)
        .rename({"poll_id": "poll_id_mv"})
        .select(
            "poll_id_mv",
            "poll_id_wiki",
            "pollster",
            "fieldwork_start",
            "fieldwork_end",
            "fieldwork_start_wiki",
            "fieldwork_end_wiki",
            "sample_size",
            "sample_size_wiki",
            "date_shift",
        )
        .sort("fieldwork_end", "poll_id_mv")
    )


def _identical_hypotheses(
    mv_r: pl.DataFrame, wiki_r: pl.DataFrame, tolerance: float
) -> pl.DataFrame:
    """Par couple (poll_id_mv, poll_id_wiki) : nombre d'hypothèses aux parts toutes égales
    à `tolerance` près (mêmes candidats, même tour)."""
    hk = ["poll_id_mv", "poll_id_wiki", "round", "hypothesis_id"]
    return (
        mv_r.join(wiki_r, on=[*hk, "candidate_id"])
        .group_by(hk)
        .agg(ok=((pl.col("share_mv") - pl.col("share_wiki")).abs() <= tolerance).all())
        .filter(pl.col("ok"))
        .group_by("poll_id_mv", "poll_id_wiki")
        .agg(identical_hypotheses=pl.len())
    )


def compare_with_wikipedia(
    mv_polls: pl.DataFrame | None = None,
    mv_results: pl.DataFrame | None = None,
    wiki_polls: pl.DataFrame | None = None,
    wiki_results: pl.DataFrame | None = None,
    tolerance: float = 0.5,
    max_shift_days: int = 1,
    probable_window_days: int = 7,
) -> Comparison:
    """Rapproche MieuxVoter et Wikipédia 2027 sondage par sondage.

    Appariement sur (pollster, fieldwork_start, fieldwork_end) à `max_shift_days` près ;
    sur les couples, comparaison des hypothèses communes (même tour, même `hypothesis_id`)
    candidat par candidat, en base exprimés (abstention et indécis de Wikipédia écartés).
    Les sondages restés seuls de chaque côté sont ensuite confrontés deux à deux (même
    institut, fins de terrain à `probable_window_days` près) : un couple qui partage au moins
    une hypothèse aux parts identiques est signalé comme même sondage probable, aux dates
    divergentes (`probable_matches`), sans être fusionné.
    Par défaut, lit les parquets curated des deux sources.
    """
    cur = paths.CURATED
    mv_polls = pl.read_parquet(cur / POLLS_OUTPUT) if mv_polls is None else mv_polls
    mv_results = pl.read_parquet(cur / RESULTS_OUTPUT) if mv_results is None else mv_results
    if wiki_polls is None:
        wiki_polls = pl.read_parquet(cur / f"polls_wikipedia_{ELECTION}.parquet")
    if wiki_results is None:
        wiki_results = pl.read_parquet(cur / f"poll_results_wikipedia_{ELECTION}.parquet")

    matches = _match_polls(mv_polls, wiki_polls, max_shift_days)
    only_mv = mv_polls.filter(~pl.col("poll_id").is_in(matches["poll_id_mv"].implode()))
    lo = mv_polls["fieldwork_start"].min() - timedelta(days=max_shift_days)
    hi = mv_polls["fieldwork_end"].max() + timedelta(days=max_shift_days)
    only_wiki = wiki_polls.filter(
        ~pl.col("poll_id").is_in(matches["poll_id_wiki"].implode())
    ).with_columns(in_window=pl.col("fieldwork_end").is_between(lo, hi))

    keep = ["poll_id", "round", "hypothesis_id", "candidate_id", "share"]
    mv_all = (
        mv_results.filter(~pl.col("candidate_id").is_in(schemas.RESERVED_CANDIDATE_IDS))
        .select(keep)
        .rename({"poll_id": "poll_id_mv", "share": "share_mv"})
    )
    wiki_all = (
        wiki_results.filter(~pl.col("candidate_id").is_in(schemas.RESERVED_CANDIDATE_IDS))
        .select(keep)
        .rename({"poll_id": "poll_id_wiki", "share": "share_wiki"})
    )
    pair = matches.select("poll_id_mv", "poll_id_wiki")
    mv_r = mv_all.join(pair, on="poll_id_mv")
    wiki_r = wiki_all.join(pair, on="poll_id_wiki")

    cols = ["poll_id", "pollster", "fieldwork_start", "fieldwork_end", "sample_size"]
    loose = (
        only_mv.select(cols)
        .join(only_wiki.select(cols), on="pollster", suffix="_wiki")
        .filter(
            (pl.col("fieldwork_end") - pl.col("fieldwork_end_wiki")).dt.total_days().abs()
            <= probable_window_days
        )
        .rename({"poll_id": "poll_id_mv"})
    )
    lp = loose.select("poll_id_mv", "poll_id_wiki")
    probable = loose.join(
        _identical_hypotheses(
            mv_all.join(lp, on="poll_id_mv"), wiki_all.join(lp, on="poll_id_wiki"), tolerance
        ),
        on=["poll_id_mv", "poll_id_wiki"],
    ).sort("fieldwork_end", "poll_id_mv")
    hk = ["poll_id_mv", "poll_id_wiki", "round", "hypothesis_id"]
    hyp = (
        mv_r.select(hk)
        .unique()
        .with_columns(mv=True)
        .join(wiki_r.select(hk).unique().with_columns(wiki=True), on=hk, how="full", coalesce=True)
        .group_by("poll_id_mv", "poll_id_wiki", "round")
        .agg(
            common=(pl.col("mv") & pl.col("wiki")).fill_null(False).sum(),
            only_mieuxvoter=(pl.col("mv").fill_null(False) & pl.col("wiki").is_null()).sum(),
            only_wikipedia=(pl.col("wiki").fill_null(False) & pl.col("mv").is_null()).sum(),
        )
        .sort("poll_id_mv", "round")
    )
    gaps = (
        mv_r.join(wiki_r, on=[*hk, "candidate_id"], how="inner")
        .with_columns(diff=(pl.col("share_mv") - pl.col("share_wiki")).round(3))
        .filter(pl.col("diff").abs() > tolerance)
        .select(*hk, "candidate_id", "share_mv", "share_wiki", "diff")
        .sort("poll_id_mv", "round", "hypothesis_id", "candidate_id")
    )
    return Comparison(matches, only_mv, only_wiki, hyp, gaps, probable, tolerance)
