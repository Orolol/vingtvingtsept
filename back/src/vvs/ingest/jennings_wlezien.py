"""Base de Jennings et Wlezien (2018) sur les erreurs des sondages préélectoraux.

Source décrite dans data/registry/jennings_wlezien_source.yaml.

- `ingest()` télécharge les fichiers listés dans data/raw/jennings_wlezien/ ; un fichier
  présent n'est jamais réécrit, son empreinte sha256 est vérifiée.
- `parse()` produit data/curated/jw_polls.parquet.
- `error_summary()` résume l'erreur sondage − résultat par horizon : c'est l'entrée du
  modèle d'erreur (docs/04-modele.md, section 2).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import yaml

from vvs import paths

SOURCE_FILE = paths.REGISTRY / "jennings_wlezien_source.yaml"
RAW_DIRNAME = "jennings_wlezien"
OUTPUT_NAME = "jw_polls.parquet"
USER_AGENT = "vingtvingtsept/0.0.1"


class ConsistencyError(ValueError):
    pass


@dataclass(frozen=True)
class SourceFile:
    name: str
    role: str
    url: str
    sha256: str | None
    file_doi: str = ""


def load_source(path: Path | None = None) -> tuple[dict, tuple[SourceFile, ...]]:
    data = yaml.safe_load((path or SOURCE_FILE).read_text(encoding="utf-8"))
    files = tuple(
        SourceFile(
            name=f["name"],
            role=f["role"],
            url=f["url"],
            sha256=f.get("sha256"),
            file_doi=f.get("file_doi", ""),
        )
        for f in data["files"]
    )
    return data["dataset"], files


def raw_dir() -> Path:
    return paths.RAW / RAW_DIRNAME


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_hash(path: Path, f: SourceFile) -> str:
    digest = sha256_of(path)
    if f.sha256 and digest != f.sha256:
        raise ConsistencyError(
            f"{path}: sha256 {digest} ≠ {f.sha256} attendu (fichier tronqué ou modifié ?)"
        )
    return digest


# --- Ingestion ---------------------------------------------------------------------------


def _download(url: str, dest: Path, retries: int = 4) -> None:
    import httpx

    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            # Dataverse répond par une redirection 303 vers une URL S3 signée.
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=300,
                headers={"User-Agent": USER_AGENT},
            ) as r:  # fmt: skip
                r.raise_for_status()
                expected = r.headers.get("content-length")
                with part.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            if expected is not None and part.stat().st_size != int(expected):
                raise OSError(f"taille {part.stat().st_size} ≠ content-length {expected}")
            part.rename(dest)
            return
        except (httpx.HTTPError, OSError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"échec du téléchargement de {url} : {last}")


def ingest(files: tuple[SourceFile, ...] | None = None) -> list[tuple[str, str, str]]:
    """Télécharge les fichiers de la source. Renvoie [(nom, statut, sha256)] avec statut
    `downloaded` ou `present`."""
    if files is None:
        _, files = load_source()
    d = raw_dir()
    d.mkdir(parents=True, exist_ok=True)
    report = []
    for f in files:
        dest = d / f.name
        if dest.exists():
            report.append((f.name, "present", _check_hash(dest, f)))
            continue
        _download(f.url, dest)
        try:
            digest = _check_hash(dest, f)
        except ConsistencyError:
            dest.unlink()  # fichier tout juste téléchargé et invalide : on ne le garde pas
            raise
        meta = {
            "url": f.url,
            "file_doi": f.file_doi,
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        dest.with_name(dest.name + ".source.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report.append((f.name, "downloaded", digest))
    return report


# --- Libellés du codebook ------------------------------------------------------------------
#
# Le fichier Stata ne contient que des identifiants numériques (`countryid`, `partyid`) :
# les noms de partis et de candidats ne sont que dans le codebook PDF, par pays pour les
# législatives et par pays-année(-tour) pour les présidentielles. On en extrait le texte
# (pdfplumber) puis on le lit ligne à ligne. Le libellé conservé est le texte du codebook
# après l'identifiant (« UNR De Gaulle », « Labour Party »), sans les tirets de remplissage.

LEGISLATIVE_HEADER = "Party ID – Legislative Elections"
PRESIDENTIAL_HEADER = "Candidate ID – Presidential Elections"
# Le codebook du fichier secondaire POLLS1517 commence par ce titre : fin des tables.
CODEBOOK_END = "Election Polling Errors across Time and Space"
_SKIP_LINES = (
    "Country ID Party Name",
    "Country: Election Year ID Party Candidate Name",
)
_PRES_HEADING = re.compile(
    r"^(?P<country>[A-Z][A-Za-z .]+?)(?::| –) (?P<year>\d{4})(?:/(?P<year2>\d{2}))?"
    r"(?: \(Round (?P<round>\d)\))?$"
)
_ID_LINE = re.compile(r"^(?P<id>\d{1,2})(?: (?P<rest>.*))?$")

# Noms de pays : orthographe du fichier Stata -> nom retenu (et celui du codebook).
COUNTRY_FIXES = {"Phillipines": "Philippines"}

COUNTRY_ISO2 = {
    "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Belgium": "BE", "Brazil": "BR",
    "Bulgaria": "BG", "Canada": "CA", "Chile": "CL", "Colombia": "CO", "Croatia": "HR",
    "Cyprus": "CY", "Czech Republic": "CZ", "Denmark": "DK", "Ecuador": "EC", "Finland": "FI",
    "France": "FR", "Germany": "DE", "Greece": "GR", "Hungary": "HU", "Iceland": "IS",
    "Ireland": "IE", "Italy": "IT", "Japan": "JP", "Malta": "MT", "Mexico": "MX",
    "Netherlands": "NL", "New Zealand": "NZ", "Norway": "NO", "Paraguay": "PY", "Peru": "PE",
    "Philippines": "PH", "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Serbia": "RS",
    "Slovakia": "SK", "Slovenia": "SI", "South Korea": "KR", "Spain": "ES", "Sweden": "SE",
    "Switzerland": "CH", "Turkey": "TR", "United Kingdom": "GB", "United States": "US",
    "Venezuela": "VE",
}  # fmt: skip

CODEBOOK_SCHEMA = pl.Schema(
    {
        "country": pl.Utf8,
        "election_type": pl.Utf8,
        "cycle_year": pl.Int32,  # null pour les législatives (liste valable pour toutes)
        "round": pl.Int8,  # null : liste valable pour tous les tours de l'année
        "party_id": pl.Int16,
        "label": pl.Utf8,
    }
)


def codebook_text(pdf_path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _clean_label(text: str) -> str | None:
    tokens = [t for t in text.split() if t != "-"]
    return " ".join(tokens) or None


def parse_codebook(text: str) -> pl.DataFrame:
    """Tables d'identifiants du codebook -> une ligne par (pays, type, année, tour, id).

    Règles explicites :
    - une ligne qui ne commence ni par un identifiant ni par un en-tête prolonge le libellé
      précédent (noms longs coupés par la mise en page) ;
    - « Australia (two-party preference) » reprend les ids 1 et 2 de « Australia » : ignoré ;
    - « Croatia: 2009/10 » couvre les deux années ;
    - un en-tête présidentiel répété au premier tour (« Cyprus: 2013 (Round 1) » deux fois)
      désigne le second tour à sa deuxième occurrence.
    """
    countries = set(COUNTRY_ISO2)
    rows: list[dict] = []
    section: str | None = None
    current: dict | None = None  # en-tête courant
    seen: dict[tuple, int] = {}
    last: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == LEGISLATIVE_HEADER:
            section, current, last = "legislative", None, None
            continue
        if line == PRESIDENTIAL_HEADER:
            section, current, last = "presidential", None, None
            continue
        if section is None or line in _SKIP_LINES:
            continue
        if section == "presidential" and line == CODEBOOK_END:
            break
        if section == "legislative" and (line in countries or "(two-party preference)" in line):
            ignored = line not in countries
            current = None if ignored else {"country": line, "years": [None], "round": None}
            last = None
            continue
        if section == "presidential" and (m := _PRES_HEADING.match(line)):
            year = int(m["year"])
            years = [year] if m["year2"] is None else [year, year + 1]
            rnd = int(m["round"]) if m["round"] else None
            key = (m["country"], year, rnd)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 2 and rnd == 1:
                rnd = 2
            elif seen[key] > 1:
                raise ValueError(f"codebook : en-tête répété {line!r}")
            current = {"country": m["country"], "years": years, "round": rnd}
            last = None
            continue
        if current is None:
            continue
        if m := _ID_LINE.match(line):
            last = {"party_id": int(m["id"]), "text": m["rest"] or ""}
            for y in current["years"]:
                rows.append(
                    {"country": current["country"], "election_type": section,
                     "cycle_year": y, "round": current["round"], "_ref": last}
                )  # fmt: skip
        elif last is not None:
            last["text"] += " " + line
    out = [
        {**{k: v for k, v in r.items() if k != "_ref"},
         "party_id": r["_ref"]["party_id"], "label": _clean_label(r["_ref"]["text"])}
        for r in rows
    ]  # fmt: skip
    df = pl.DataFrame(out, schema=CODEBOOK_SCHEMA, orient="row")
    dup = df.filter(df.select(["country", "election_type", "cycle_year", "round", "party_id"])
                    .is_duplicated())  # fmt: skip
    if dup.height:
        raise ValueError(f"codebook : identifiants en double\n{dup}")
    return df


# --- France : rattachement au registre ----------------------------------------------------
#
# Libellé du codebook -> candidate_id du registre (data/registry/candidates.yaml), ou None
# pour un candidat absent du registre (présidentielles 1965-1995). Un libellé français
# absent de cette table fait échouer le parsing : le codebook contient des erreurs de
# recopie (« LO Sarkozy », « MDC Hollande » en 2012 et 2017, sans données associées), et
# un id nouveau doit être vérifié à la main. Les parts au résultat sont en outre comparées
# à elections.yaml pour les élections que le registre couvre (`_check_france`).
FRANCE_ALIASES: dict[str, str | None] = {
    # 1965
    "UNR De Gaulle": None,
    "FGDS Mitterand": None,  # sic : Mitterrand
    "IFR Tixier-Vignancour": None,
    "MRP Lecanuet": None,
    # 1969
    "UDR G. Pompidou": None,
    "SFIO Defferre": None,
    "DC Poher": None,
    "PCF Duclos": None,
    "LC Krivine": None,
    "IND Ducatel": None,
    "PSU Rocard": None,
    # 1974
    "UDR Chaban-Delmas": None,
    "PS Mitterand": None,
    "IR V. G. d'Estaing": None,
    "Independent Royer": None,
    # 1981
    "RPR Chirac": "chirac_jacques",
    "UDF V. G. d'Estaing": None,
    "PCF Marchais": None,
    "PSU Bouchardeau": None,
    # 1988
    "FN Le Pen": "le_pen_jean_marie",  # 1988 à 2007
    "UDF Barre": None,
    "PCF Lajoine": None,
    "EELV Waechter": None,
    "LO Laguiller": "laguiller_arlette",
    "PSU/LCR Juquin": None,
    # 1995
    "PS Jospin": "jospin_lionel",
    "UDF Balladur": None,
    "PCF Hue": "hue_robert",
    "EELV Voynet": "voynet_dominique",
    # 2002
    "UMP Chirac": "chirac_jacques",
    "UDF Bayrou": "bayrou_francois",  # 2002 à 2012
    "FG Hue": "hue_robert",
    "EELV Mamère": "mamere_noel",
    "MDC Chevènement": "chevenement_jean_pierre",
    # 2007
    "UMP Sarkozy": "sarkozy_nicolas",
    "PS Royal": "royal_segolene",
    # 2012
    "PS Hollande": "hollande_francois",
    "FN M Le Pen": "le_pen_marine",
    "FG Mélenchon": "melenchon_jean_luc",
    "EELV Joly": "joly_eva",
    # 2017
    "UMP Fillon": "fillon_francois",
    "PS Hamon": "hamon_benoit",
    "En Marche! Macron": "macron_emmanuel",
}

# Élections où `polldate` est décalé et `daysbeforeED` juste : la date de sondage est alors
# recalculée (election_date - daysbeforeED). Finlande 2011 : la grille quotidienne du fichier
# s'arrête le 2011-03-17 (daysbeforeED = 0) alors que l'élection a eu lieu le 2011-04-17 ;
# les dates sont donc décalées d'un mois. Ailleurs, les dates font foi et days_before est
# recalculé (Hongrie 2010 : la grille s'arrête bien le jour du vote, 2010-04-11, mais
# daysbeforeED y vaut 12). Clé : code ISO et date de l'élection.
POLL_DATES_FROM_DAYS = {"FI-2011-04-17": "grille décalée d'un mois"}

# Écart admis entre la part au résultat de la base (arrondie au dixième en 2017) et celle
# du registre (Conseil constitutionnel, au centième).
RESULT_TOLERANCE = 0.15


class UnknownLabelError(ValueError):
    pass


# --- Curation ------------------------------------------------------------------------------

RAW_COLUMNS = (
    "country", "election", "system", "rule", "round", "elecdate", "polldate", "daysbeforeED",
    "poll_", "vote_", "gov_", "inc_", "sample", "npolls", "pollcycle", "partyid", "turnout",
    "enpp", "espv", "regime",
)  # fmt: skip

JW_POLLS_SCHEMA = pl.Schema(
    {
        "election_id": pl.Utf8,
        "country": pl.Utf8,
        "country_iso2": pl.Utf8,
        "election_type": pl.Utf8,  # presidential, legislative
        "system": pl.Utf8,  # parliamentary, presidential, semi-presidential
        "electoral_rule": pl.Utf8,  # rule d'origine : Majority, PR, SMDP…
        "round": pl.Int8,
        "election_date": pl.Date,
        "cycle_year": pl.Int32,  # année du premier tour
        "party_id": pl.Int16,
        "label": pl.Utf8,
        "candidate_id": pl.Utf8,  # France seulement, candidats du registre
        "poll_date": pl.Date,
        "days_before": pl.Int32,  # election_date - poll_date, recalculé
        "days_before_jw": pl.Int32,  # daysbeforeED d'origine
        "poll_share": pl.Float64,  # moyenne des sondages du jour (poll_), en points
        "result_share": pl.Float64,  # vote_, en points
        "n_polls": pl.Int16,
        "sample_size": pl.Int32,  # somme des échantillons du jour
        "sample_size_is_default": pl.Boolean,  # sample = 1000 x npolls : taille imputée
        "poll_total": pl.Float64,  # somme des poll_share des partis listés ce jour-là
        "result_total": pl.Float64,  # somme des result_share des mêmes partis
        "gov_party": pl.Boolean,
        "incumbent_party": pl.Boolean,
        "turnout": pl.Float64,
        "enpp": pl.Float64,
        "candidate_centric": pl.Boolean,  # espv == "Candidate"
        "regime": pl.Utf8,  # old (démocratie avant 1976), new
        "poll_cycle": pl.Int32,  # jours avant l'élection du premier sondage du cycle
    }
)


@dataclass
class ParseReport:
    rows: int = 0
    elections: int = 0
    countries: int = 0
    first_date: str = ""
    last_date: str = ""
    by_type: dict[str, int] | None = None
    unlabelled: list[tuple[str, str, int, int]] | None = None  # hors France
    days_mismatch: dict[str, int] | None = None  # election_id -> écart jw - recalculé
    bad_flags: int = 0
    poll_dates_shifted: list[str] | None = None
    france: list[tuple[str, int, int, str]] | None = None  # (election_id, round, n, ids)

    def summary(self, verbose: bool = False) -> str:
        lines = [
            (
                f"{self.rows} lignes sondage-parti, {self.elections} élections (tours), "
                f"{self.countries} pays, du {self.first_date} au {self.last_date}"
            ),
            "par type : " + ", ".join(f"{k} {v}" for k, v in sorted((self.by_type or {}).items())),
            f"dates de sondage recalculées depuis daysbeforeED : {self.poll_dates_shifted}",
            (
                "daysbeforeED incohérent avec les dates (days_before recalculé, écart jw − "
                f"dates) : {self.days_mismatch}"
            ),
            f"gov_/inc_ hors {{0, 1}} mis à null : {self.bad_flags} lignes",
            (
                "triplets (pays, élection, parti) sans libellé au codebook, hors France : "
                f"{len(self.unlabelled or [])}"
            ),
            "France :",
        ]
        lines += [f"  {e} tour {r} : {n} lignes, {ids}" for e, r, n, ids in self.france or []]
        if verbose:
            lines += [f"  sans libellé : {u}" for u in self.unlabelled or []]
        return "\n".join(lines)


def read_raw(path: Path) -> pl.DataFrame:
    """Lignes du fichier Stata qui portent un sondage (poll_ non nul).

    Le fichier est une grille quotidienne parti x jour (1,5 million de lignes) ; seules
    ~88 000 portent une valeur de sondage.
    """
    import pandas as pd

    chunks = []
    with pd.read_stata(
        path, columns=list(RAW_COLUMNS), convert_categoricals=False, chunksize=200_000
    ) as reader:  # fmt: skip
        for chunk in reader:
            chunks.append(chunk[chunk["poll_"].notna()])
    df = pd.concat(chunks, ignore_index=True)
    for col in ("elecdate", "polldate"):
        df[col] = df[col].astype("datetime64[s]").dt.date
    return pl.from_pandas(df)


def _cycle_year(df: pl.DataFrame) -> pl.Series:
    """Année du premier tour : le second tour de Chili 2009 ou de Croatie 2009 tombe en
    janvier suivant, le codebook le range sous l'année du premier tour."""
    firsts = (
        df.filter(pl.col("round") == 1)
        .select(["country", "election_type", "election_date"])
        .unique()
        .sort("election_date")
    )
    seconds = (
        df.filter(pl.col("round") == 2)
        .select(["country", "election_type", "election_date"])
        .unique()
        .sort("election_date")
    )
    matched = seconds.join_asof(
        firsts.rename({"election_date": "first_date"}),
        left_on="election_date", right_on="first_date", by=["country", "election_type"],
        strategy="backward", tolerance=timedelta(days=120), check_sortedness=False,
    )  # fmt: skip
    year = df.join(
        matched.select(["country", "election_type", "election_date", "first_date"]),
        on=["country", "election_type", "election_date"],
        how="left",
    ).select(pl.coalesce(pl.col("first_date"), pl.col("election_date")).dt.year().cast(pl.Int32))
    return year.to_series()


def curate(
    raw: pl.DataFrame, codebook: pl.DataFrame, registry=None
) -> tuple[pl.DataFrame, ParseReport]:
    """Lignes brutes (`read_raw`) + codebook (`parse_codebook`) -> table jw_polls."""
    report = ParseReport()
    numeric = [
        c
        for c in RAW_COLUMNS
        if c
        not in ("country", "election", "system", "rule", "elecdate", "polldate", "espv", "regime")
    ]
    df = raw.with_columns(pl.col(numeric).cast(pl.Float64))
    df = df.filter(pl.col("poll_").is_not_null()).with_columns(
        pl.col("country").replace(COUNTRY_FIXES),
        pl.col("election").str.to_lowercase().alias("election_type"),
        pl.col("round").cast(pl.Int8),
        pl.col("elecdate").cast(pl.Date).alias("election_date"),
        pl.col("polldate").cast(pl.Date).alias("poll_date"),
        pl.col("partyid").cast(pl.Int16).alias("party_id"),
    )
    unknown = set(df["country"].unique()) - set(COUNTRY_ISO2)
    if unknown:
        raise ValueError(f"pays sans code ISO : {sorted(unknown)}")
    if df.filter(pl.col("round").is_null() | ~pl.col("round").is_in([1, 2])).height:
        raise ValueError("round hors {1, 2}")

    df = df.with_columns(
        pl.col("country").replace_strict(COUNTRY_ISO2).alias("country_iso2"),
        pl.col("daysbeforeED").cast(pl.Int32).alias("days_before_jw"),
    )
    shifted = pl.concat_str(
        "country_iso2", pl.lit("-"), pl.col("election_date").dt.strftime("%Y-%m-%d")
    ).is_in(list(POLL_DATES_FROM_DAYS))
    df = df.with_columns(
        pl.when(shifted)
        .then(pl.col("election_date") - pl.duration(days=pl.col("days_before_jw")))
        .otherwise(pl.col("poll_date"))
        .alias("poll_date")
    ).with_columns(
        (pl.col("election_date") - pl.col("poll_date")).dt.total_days().cast(pl.Int32)
        .alias("days_before"),
    )  # fmt: skip
    if df.filter(pl.col("days_before") < 0).height:
        raise ValueError("sondage daté après l'élection")
    # Identifiant : pays-date ; suffixe -L pour les législatives (le même jour que la
    # présidentielle aux États-Unis).
    df = df.with_columns(
        (
            pl.col("country_iso2") + "-" + pl.col("election_date").dt.strftime("%Y-%m-%d")
            + pl.when(pl.col("election_type") == "legislative").then(pl.lit("-L"))
            .otherwise(pl.lit(""))
        ).alias("election_id"),
    )  # fmt: skip
    df = df.with_columns(_cycle_year(df).alias("cycle_year"))

    # Libellés : présidentielles par (pays, année, tour) puis (pays, année) ; législatives
    # par pays.
    pres = codebook.filter(pl.col("election_type") == "presidential")
    by_round = pres.filter(pl.col("round").is_not_null()).select(
        "country", "cycle_year", "round", "party_id", pl.col("label").alias("_l_round")
    )
    by_year = pres.filter(pl.col("round").is_null()).select(
        "country", "cycle_year", "party_id", pl.col("label").alias("_l_year")
    )
    leg = codebook.filter(pl.col("election_type") == "legislative").select(
        "country", "party_id", pl.col("label").alias("_l_leg")
    )
    df = (
        df.join(by_round, on=["country", "cycle_year", "round", "party_id"], how="left")
        .join(by_year, on=["country", "cycle_year", "party_id"], how="left")
        .join(leg, on=["country", "party_id"], how="left")
        .with_columns(
            pl.when(pl.col("election_type") == "presidential")
            .then(pl.coalesce("_l_round", "_l_year"))
            .otherwise(pl.col("_l_leg"))
            .alias("label")
        )
    )

    # France : alias obligatoires.
    fr = df.filter(pl.col("country") == "France")
    bad = (
        fr.filter(pl.col("label").is_null() | ~pl.col("label").is_in(list(FRANCE_ALIASES)))
        .select(["election_id", "round", "party_id", "label"])
        .unique()
        .sort(["election_id", "party_id"])
    )
    if bad.height:
        raise UnknownLabelError(
            "libellés français absents de FRANCE_ALIASES :\n"
            + "\n".join(str(r) for r in bad.iter_rows())
        )
    df = df.with_columns(
        pl.when(pl.col("country") == "France")
        .then(pl.col("label").replace_strict(FRANCE_ALIASES, default=None, return_dtype=pl.Utf8))
        .otherwise(None)
        .alias("candidate_id")
    )

    # Indicateurs : la base contient gov_ = 7 (Finlande 2011, un parti), mis à null.
    bad_flags = pl.col("gov_").is_not_null() & ~pl.col("gov_").is_in([0.0, 1.0]) | (
        pl.col("inc_").is_not_null() & ~pl.col("inc_").is_in([0.0, 1.0])
    )
    report.bad_flags = df.filter(bad_flags).height

    def flag(c: str) -> pl.Expr:
        return pl.when(pl.col(c).is_in([0.0, 1.0])).then(pl.col(c) == 1).otherwise(None)

    day = ["election_id", "poll_date"]
    df = df.with_columns(
        pl.col("poll_").alias("poll_share"),
        pl.col("vote_").cast(pl.Float64).alias("result_share"),
        pl.col("npolls").cast(pl.Int16).alias("n_polls"),
        pl.col("sample").round(0).cast(pl.Int32).alias("sample_size"),
        (pl.col("sample") == 1000 * pl.col("npolls")).alias("sample_size_is_default"),
        pl.col("poll_").sum().over(day).alias("poll_total"),
        pl.col("vote_").cast(pl.Float64).sum().over(day).alias("result_total"),
        flag("gov_").alias("gov_party"),
        flag("inc_").alias("incumbent_party"),
        pl.col("rule").alias("electoral_rule"),
        pl.col("system").str.to_lowercase(),
        (pl.col("espv") == "Candidate").alias("candidate_centric"),
        pl.col("regime").str.to_lowercase(),
        pl.col("pollcycle").cast(pl.Int32).alias("poll_cycle"),
    )
    out = (
        df.select(list(JW_POLLS_SCHEMA))
        .cast(dict(JW_POLLS_SCHEMA))
        .sort(["country", "election_date", "election_type", "poll_date", "party_id"])
    )
    if out.select(["election_id", "party_id", "poll_date"]).is_duplicated().any():
        raise ValueError("clé (election_id, party_id, poll_date) non unique")
    if (
        out.group_by(["election_id", "party_id"])
        .agg(pl.col("result_share").n_unique() > 1)["result_share"]
        .any()
    ):
        raise ValueError("result_share varie au sein d'une élection-parti")
    if registry is not None:
        _check_france(out, registry)

    els = out.select(["election_id", "election_type"]).unique()
    mism = (
        out.filter(pl.col("days_before") != pl.col("days_before_jw"))
        .group_by("election_id")
        .agg((pl.col("days_before_jw") - pl.col("days_before")).first())
        .sort("election_id")
    )
    unl = (
        out.filter(pl.col("label").is_null())
        .select(["country", "election_id", "round", "party_id"])
        .unique()
        .sort(["election_id", "party_id"])
    )
    fr_sum = (
        out.filter(pl.col("country") == "France")
        .group_by(["election_id", "round"])
        .agg(
            pl.len(),
            pl.concat_str(
                pl.col("party_id").cast(pl.Utf8),
                pl.lit("="),
                pl.coalesce("candidate_id", "label"),
            )
            .unique()
            .sort()
            .str.join(", "),
        )
        .sort("election_id")
    )
    report.rows = out.height
    report.elections = els.height
    report.countries = out["country"].n_unique()
    report.first_date = str(out["poll_date"].min())
    report.last_date = str(out["election_date"].max())
    report.by_type = dict(els.group_by("election_type").len().iter_rows())
    report.days_mismatch = dict(mism.iter_rows())
    report.poll_dates_shifted = sorted(POLL_DATES_FROM_DAYS)
    report.unlabelled = list(unl.iter_rows())
    report.france = list(fr_sum.iter_rows())
    return out, report


def _check_france(df: pl.DataFrame, registry) -> None:
    """Date et part au résultat des candidats français contre elections.yaml."""
    errors = []
    fr = (
        df.filter(pl.col("candidate_id").is_not_null())
        .select(["election_id", "election_date", "round", "candidate_id", "result_share"])
        .unique()
    )
    for eid, edate, rnd, cid, share in fr.iter_rows():
        if cid not in registry.candidates:
            errors.append(f"{eid}: {cid} absent de candidates.yaml")
            continue
        e = registry.elections.get(edate.year)
        if e is None:
            continue
        r = e.round_(rnd)
        if r.date != edate:
            errors.append(f"{eid}: date du tour {rnd} {edate} ≠ registre {r.date}")
        ref = next((x.share for x in r.results if x.candidate_id == cid), None)
        if ref is None:
            errors.append(f"{eid}: {cid} sans résultat au tour {rnd} dans elections.yaml")
        elif abs(ref - share) > RESULT_TOLERANCE:
            errors.append(f"{eid}: {cid} {share:.2f} ≠ registre {ref:.2f}")
    if errors:
        raise ConsistencyError("France contre le registre :\n" + "\n".join(errors))


def main_file(files: tuple[SourceFile, ...], role: str) -> Path:
    f = next(f for f in files if f.role == role)
    path = raw_dir() / f.name
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `vvs ingest jennings-wlezien`")
    _check_hash(path, f)
    return path


def parse() -> tuple[Path, pl.DataFrame, ParseReport]:
    from vvs.registry import default_registry

    _, files = load_source()
    raw = read_raw(main_file(files, "main"))
    codebook = parse_codebook(codebook_text(main_file(files, "codebook")))
    df, report = curate(raw, codebook, default_registry())
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    out = paths.CURATED / OUTPUT_NAME
    df.write_parquet(out)
    return out, df, report


# --- Erreur par horizon -------------------------------------------------------------------

HORIZONS = (180, 90, 30, 7)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def load_polls(path: Path | None = None) -> pl.DataFrame:
    path = path or paths.CURATED / OUTPUT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} absent : lancer `vvs parse jennings-wlezien`")
    return pl.read_parquet(path)


def _filter(
    df: pl.DataFrame,
    country: str | None,
    election_type: str | None,
    round: int | None,
    exclude: tuple[str, ...] | list[str],
) -> pl.DataFrame:
    if country is not None:
        df = df.filter((pl.col("country") == country) | (pl.col("country_iso2") == country))
    if election_type is not None:
        df = df.filter(pl.col("election_type") == election_type)
    if round is not None:
        df = df.filter(pl.col("round") == round)
    if exclude:
        df = df.filter(~pl.col("election_id").is_in(list(exclude)))
    return df


def horizon_errors(
    df: pl.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    max_staleness: int = 30,
    normalize: bool = False,
) -> pl.DataFrame:
    """Erreur du dernier sondage avant chaque horizon, par élection et par parti.

    Pour l'horizon h, on prend pour chaque (élection, parti) la dernière ligne dont
    `days_before` est dans [h, h + max_staleness] : un sondage plus ancien ne dit rien de
    l'état de la campagne à h jours, l'élection-parti est alors absente à cet horizon.
    `error` = poll_share - result_share, en points.

    `normalize=True` ramène les parts du jour au même périmètre que le résultat :
    poll_share x result_total / poll_total, où les deux totaux portent sur les partis
    listés ce jour-là. Cela neutralise les indécis et les « autres » que les sondages
    anciens laissent dans le total (France 1965 : 60 % seulement répartis), mais force
    aussi à zéro l'erreur moyenne des partis listés.
    """
    share = (
        pl.col("poll_share") * pl.col("result_total") / pl.col("poll_total")
        if normalize
        else pl.col("poll_share")
    )
    keep = [
        "election_id", "country", "country_iso2", "election_type", "round", "election_date",
        "party_id", "label", "candidate_id", "poll_date", "days_before", "n_polls",
    ]  # fmt: skip
    parts = []
    for h in horizons:
        sub = (
            df.filter(pl.col("days_before").is_between(h, h + max_staleness))
            .sort("days_before")
            .group_by(["election_id", "party_id"], maintain_order=True)
            .first()
        )
        parts.append(
            sub.select(
                pl.lit(h, dtype=pl.Int32).alias("horizon"),
                *keep,
                share.alias("poll_share"),
                "result_share",
            ).with_columns((pl.col("poll_share") - pl.col("result_share")).alias("error"))
        )
    return pl.concat(parts).sort(
        ["horizon", "election_id", "party_id"], descending=[True, False, False]
    )


def error_summary(
    country: str | None = None,
    election_type: str | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    *,
    round: int | None = None,
    max_staleness: int = 30,
    normalize: bool = False,
    exclude: tuple[str, ...] | list[str] = (),
    df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Distribution de l'erreur sondage - résultat par horizon (voir `horizon_errors`).

    Une ligne par horizon : nombre d'élections et d'élections-partis, moyenne, écart type,
    erreur absolue moyenne et quantiles de l'erreur ; `sd_election_mean` est l'écart type,
    entre élections, de l'erreur moyenne de l'élection (composante commune à tous les
    partis d'une élection), `sd_within` celui de l'erreur centrée sur cette moyenne.

    `country` : nom ou code ISO ; `exclude` : election_id à écarter (leave-one-election-out,
    voir docs/07-evaluation.md).
    """
    df = _filter(df if df is not None else load_polls(), country, election_type, round, exclude)
    err = horizon_errors(df, horizons, max_staleness, normalize).with_columns(
        pl.col("error").mean().over(["horizon", "election_id"]).alias("election_mean")
    )
    per_election = err.group_by(["horizon", "election_id"]).agg(pl.col("election_mean").first())
    common = per_election.group_by("horizon").agg(
        pl.col("election_mean").std().alias("sd_election_mean")
    )
    out = (
        err.group_by("horizon")
        .agg(
            pl.col("election_id").n_unique().alias("n_elections"),
            pl.len().alias("n_obs"),
            pl.col("error").mean().alias("mean"),
            pl.col("error").std().alias("sd"),
            pl.col("error").abs().mean().alias("mae"),
            *[
                pl.col("error").quantile(q, interpolation="linear").alias(f"q{round_q(q)}")
                for q in QUANTILES
            ],
            (pl.col("error") - pl.col("election_mean")).std().alias("sd_within"),
        )
        .join(common, on="horizon", how="left")
    )
    order = pl.DataFrame({"horizon": list(horizons)}, schema={"horizon": pl.Int32})
    return order.join(out, on="horizon", how="left")


def round_q(q: float) -> str:
    return f"{round(q * 100):02d}"
