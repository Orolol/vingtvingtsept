"""INA, classement thématique des sujets des journaux télévisés du soir, 2000-2020.

Jeu data.gouv.fr « Classement thématique des sujets de journaux télévisés (janvier 2000 -
décembre 2020) », id `57fe1446c751df182f79df72`, producteur Institut national de
l'audiovisuel, licence ouverte (`fr-lo`), clos au 31 décembre 2020. Une seule ressource :
`ina-barometre-jt-tv-donnees-quotidiennes-2000-2020-nbre-sujets-durees-202410.csv`
(9 857 547 octets, mise à jour du 15 octobre 2024, sha1 annoncé par data.gouv
`52104bd1da01e861eaa84b7b555633d302cab234`).

Format relevé le 20 octobre 2026 : CSV sans en-tête, séparateur `;`, latin-1, fins de ligne
CRLF, 268 424 lignes de 6 champs :

    02/01/2000;France 3;;Catastrophes;2;235

1. jour (JJ/MM/AAAA) ; 2. chaîne (TF1, France 2, France 3, Arte, M6) ; 3. champ toujours
vide (édition ? le jeu ne couvre que le JT du soir) ; 4. rubrique, une parmi `RUBRICS` ;
5. nombre de sujets de la rubrique ce jour-là sur cette chaîne ; 6. durée cumulée en
secondes. Une ligne par (jour, chaîne, rubrique) ayant au moins un sujet : une rubrique
absente un jour vaut zéro sujet. Ni titre de sujet, ni sous-thème, ni mention de personne.

Chaque sujet est classé dans une seule rubrique, par algorithme, à partir des mots-clés du
thésaurus de l'INA (méthodologie InaStat).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl

from vvs import paths
from vvs.ingest.press.cache import now_utc, press_dir

DATASET_ID = "57fe1446c751df182f79df72"
DATASET_API = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/"
USER_AGENT = "vingtvingtsept/0.0.1 (+https://github.com/Orolol)"
OUTPUT_NAME = "press_ina.parquet"

CHANNELS = ("TF1", "France 2", "France 3", "Arte", "M6")
RUBRICS = (
    "Catastrophes",
    "Culture-loisirs",
    "Economie",
    "Education",
    "Environnement",
    "Faits divers",
    "Histoire-hommages",
    "International",
    "Justice",
    "Politique France",
    "Santé",
    "Sciences et techniques",
    "Société",
    "Sport",
)

INA_SCHEMA = pl.Schema(
    {
        "date": pl.Date,
        "channel": pl.Utf8,
        "theme": pl.Utf8,
        "n_subjects": pl.Int32,
        "duration_s": pl.Int32,
    }
)


def ina_dir() -> Path:
    return press_dir("ina")


@dataclass(frozen=True)
class Resource:
    url: str
    title: str
    last_modified: str
    checksum: dict | None


def find_resource(http: httpx.Client) -> Resource:
    """La ressource CSV du jeu, via l'API data.gouv (échoue s'il n'y en a pas exactement une)."""
    r = http.get(DATASET_API)
    r.raise_for_status()
    csvs = [x for x in r.json()["resources"] if (x.get("format") or "").lower() == "csv"]
    if len(csvs) != 1:
        raise ValueError(f"{len(csvs)} ressources CSV dans le jeu INA, une attendue")
    x = csvs[0]
    return Resource(x["url"], x["title"], x.get("last_modified", ""), x.get("checksum"))


def ingest(http: httpx.Client | None = None) -> tuple[Path, str]:
    """Télécharge la ressource dans data/raw/press/ina/<nom>.csv ; ne réécrit jamais.

    Renvoie (chemin, statut). Le sha1 annoncé par data.gouv est vérifié avant écriture.
    """
    own = http is None
    http = http or httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(300.0, connect=30.0),
        follow_redirects=True,
    )
    try:
        res = find_resource(http)
        path = ina_dir() / res.url.rsplit("/", 1)[-1]
        if path.exists():
            return path, "déjà présent (non réécrit)"
        r = http.get(res.url)
        r.raise_for_status()
        content = r.content
        if res.checksum and res.checksum.get("type") == "sha1":
            got = hashlib.sha1(content).hexdigest()
            if got != res.checksum["value"]:
                raise ValueError(f"sha1 {got} différent de data.gouv {res.checksum['value']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(content)
        tmp.rename(path)
        meta = {
            "dataset": DATASET_API,
            "url": res.url,
            "title": res.title,
            "last_modified": res.last_modified,
            "retrieved_at": now_utc().isoformat(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "sha1_datagouv": (res.checksum or {}).get("value"),
        }
        path.with_name(path.stem + ".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path, "écrit"
    finally:
        if own:
            http.close()


def latest_raw(folder: Path | None = None) -> Path:
    folder = folder or ina_dir()
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        raise FileNotFoundError(f"aucun CSV dans {folder} : lancer `vvs ingest press --source ina`")
    path = files[-1]
    meta = json.loads(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta["sha256"]:
        raise ValueError(f"{path}: sha256 {digest} différent de la méta {meta['sha256']}")
    return path


def parse_bytes(content: bytes) -> pl.DataFrame:
    """CSV INA brut -> table `INA_SCHEMA` ; échoue sur tout écart au format relevé."""
    text = content.decode("latin-1").replace("\r\n", "\n")
    df = pl.read_csv(
        io.StringIO(text),
        separator=";",
        has_header=False,
        new_columns=["day", "channel", "edition", "theme", "n_subjects", "duration_s"],
        schema_overrides={
            "day": pl.Utf8,
            "edition": pl.Utf8,
            "n_subjects": pl.Int32,
            "duration_s": pl.Int32,
        },
    )
    if df["edition"].drop_nulls().len():
        raise ValueError(
            f"colonne 3 non vide : {df['edition'].drop_nulls().unique().to_list()[:5]}"
        )
    df = df.with_columns(pl.col("day").str.strptime(pl.Date, "%d/%m/%Y").alias("date")).select(
        list(INA_SCHEMA)
    )
    validate_ina(df)
    return df.sort("date", "channel", "theme")


def validate_ina(df: pl.DataFrame) -> None:
    if df.schema != INA_SCHEMA:
        raise ValueError(f"schéma INA inattendu : {df.schema}")
    if df.select(pl.all().is_null().any()).row(0) != (False,) * len(INA_SCHEMA):
        raise ValueError("valeurs nulles dans le CSV INA")
    unknown = set(df["channel"].unique()) - set(CHANNELS)
    if unknown:
        raise ValueError(f"chaînes inconnues : {sorted(unknown)}")
    unknown = set(df["theme"].unique()) - set(RUBRICS)
    if unknown:
        raise ValueError(f"rubriques inconnues : {sorted(unknown)}")
    if df.select(pl.struct("date", "channel", "theme").is_duplicated().any()).item():
        raise ValueError("(date, chaîne, rubrique) en double dans le CSV INA")
    if df.filter((pl.col("n_subjects") <= 0) | (pl.col("duration_s") < 0)).height:
        raise ValueError("nombre de sujets nul ou durée négative dans le CSV INA")


def parse(raw: Path | None = None, write: bool = True) -> tuple[Path, pl.DataFrame]:
    raw = raw or latest_raw()
    df = parse_bytes(raw.read_bytes())
    out = paths.CURATED / OUTPUT_NAME
    if write:
        paths.CURATED.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
    return out, df


def theme_shares(df: pl.DataFrame, every: str = "1w") -> pl.DataFrame:
    """Part de la durée de JT par rubrique et par période (toutes chaînes confondues).

    Une rubrique absente d'une période vaut 0. Sert de premier `topic_<thème>` via la
    correspondance `ina_rubrics` de press_queries.yaml.
    """
    agg = (
        df.sort("date")
        .group_by_dynamic("date", every=every, group_by="theme")
        .agg(pl.col("duration_s").sum())
    )
    full = agg.select("date").unique().join(pl.DataFrame({"theme": list(RUBRICS)}), how="cross")
    agg = full.join(agg, on=["date", "theme"], how="left").fill_null(0)
    return agg.with_columns(
        (pl.col("duration_s") / pl.col("duration_s").sum().over("date")).alias("share")
    ).sort("date", "theme")
