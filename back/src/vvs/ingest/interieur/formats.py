"""Lecteurs des fichiers bruts du ministère, un par format (clé `format` du registre).

Présidentielles : chaque lecteur renvoie une table « unités » au format long, une ligne par
unité de dépouillement (bureau, commune ou département selon le fichier) et par candidat :

    round, dept_src, commune_src, unit, registered, voters, expressed, label, votes

`dept_src` et `commune_src` sont les codes du ministère (convertis en INSEE plus tard par
`geo`), `label` est le libellé normalisé « NOM PRENOM » (voir `candidates`). Les totaux
d'unité sont répétés sur chaque ligne candidat.

Listes et législatives : chaque lecteur renvoie une table par unité et par liste ou nuance :

    round, unit, registered, voters, expressed, key, label, nuance, votes

où `key` identifie la liste (numéro de panneau, nuance…) pour l'agrégation nationale.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Sequence
from pathlib import Path

import polars as pl

from vvs.ingest.interieur import xls, xlsx
from vvs.ingest.interieur.candidates import normalize_label
from vvs.ingest.interieur.geo import normalize_dept_src

UNITS_SCHEMA = pl.Schema(
    {
        "round": pl.Int8,
        "dept_src": pl.Utf8,
        "commune_src": pl.Utf8,
        "unit": pl.Utf8,
        "registered": pl.Int64,
        "voters": pl.Int64,
        "expressed": pl.Int64,
        "label": pl.Utf8,
        "votes": pl.Int64,
    }
)

LIST_UNITS_SCHEMA = pl.Schema(
    {
        "round": pl.Int8,
        "unit": pl.Utf8,
        "registered": pl.Int64,
        "voters": pl.Int64,
        "expressed": pl.Int64,
        "key": pl.Utf8,
        "label": pl.Utf8,
        "nuance": pl.Utf8,
        "votes": pl.Int64,
    }
)


class FormatError(ValueError):
    pass


def to_int(value: object) -> int:
    """Entier depuis une cellule : '1 234', '1\xa0234', 1234.0 ; échoue sinon."""
    if isinstance(value, bool) or value is None:
        raise FormatError(f"entier attendu, reçu {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise FormatError(f"entier attendu, reçu {value!r}")
        return int(value)
    s = str(value).replace("\xa0", "").replace(" ", "").replace(" ", "").strip()
    if not s.lstrip("-").isdigit():
        raise FormatError(f"entier attendu, reçu {value!r}")
    return int(s)


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _groups(row: Sequence[object], start: int, size: int) -> Iterator[Sequence[object]]:
    """Blocs répétés de `size` colonnes à partir de `start`, en s'arrêtant au premier bloc
    vide (les lignes sont parfois complétées par des « ; » finaux)."""
    for i in range(start, len(row), size):
        chunk = row[i : i + size]
        if all(_blank(v) for v in chunk):
            break
        if len(chunk) < size:
            raise FormatError(f"bloc incomplet en colonne {i} : {list(chunk)!r}")
        yield chunk


def _index(header: Sequence[object], name: str) -> int:
    try:
        return list(header).index(name)
    except ValueError:
        raise FormatError(f"colonne {name!r} absente de l'en-tête {list(header)[:25]!r}") from None


def _read_text(path: Path, encoding: str) -> str:
    return path.read_bytes().decode(encoding)


# --- Présidentielles ---------------------------------------------------------------------

_BVOT_LONG_COLUMNS = [
    "round", "dept_src", "commune_src", "commune_name", "station", "registered", "voters",
    "expressed", "ballot_no", "last_name", "first_name", "code", "votes",
]  # fmt: skip
# 2012 : deux champs de plus après le nom de commune, non documentés ; d'après les valeurs,
# circonscription législative puis canton.
_BVOT_LONG_COLUMNS_15 = (
    _BVOT_LONG_COLUMNS[:4] + ["circonscription", "canton"] + _BVOT_LONG_COLUMNS[4:]
)
_INT_COLUMNS = ("registered", "voters", "expressed", "votes")


def read_bvot_long(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Format long 2002-2012 : une ligne par bureau et par candidat, sans en-tête, avec un
    éventuel bloc de commentaires « -- » (latin-1, « ; »). 13 champs en 2002-2007, 15 en
    2012. Certains nombres sont précédés d'espaces (« ␣␣1003 » en 2002)."""
    text = _read_text(path, "latin-1")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("--")]
    widths = {ln.count(";") + 1 for ln in lines}
    if len(widths) != 1 or (width := widths.pop()) not in (13, 15):
        raise FormatError(f"{path.name}: nombre de champs inattendu {sorted(widths)}")
    columns = _BVOT_LONG_COLUMNS if width == 13 else _BVOT_LONG_COLUMNS_15
    df = pl.read_csv(
        "\n".join(lines).encode("utf-8"),
        separator=";",
        has_header=False,
        new_columns=columns,
        schema_overrides={c: pl.Utf8 for c in columns},
        quote_char=None,
    )
    df = df.with_columns(
        pl.col("round").str.strip_chars().cast(pl.Int8),
        pl.col("dept_src").str.strip_chars(),
        pl.col("commune_src").str.strip_chars(),
        *(pl.col(c).str.strip_chars().cast(pl.Int64) for c in _INT_COLUMNS),
    )
    bad_rounds = set(df["round"].unique().to_list()) - set(rounds)
    if bad_rounds:
        raise FormatError(f"{path.name}: tours inattendus {sorted(bad_rounds)}")
    labels = {
        (ln, fn): normalize_label(ln, fn)
        for ln, fn in df.select("last_name", "first_name").unique().iter_rows()
    }
    label_df = pl.DataFrame(
        [(ln, fn, lab) for (ln, fn), lab in labels.items()],
        schema=["last_name", "first_name", "label"],
        orient="row",
    )
    df = df.join(label_df, on=["last_name", "first_name"], how="left")
    return df.select(
        "round",
        "dept_src",
        "commune_src",
        pl.concat_str(["dept_src", "commune_src", "station"], separator="|").alias("unit"),
        "registered",
        "voters",
        "expressed",
        "label",
        "votes",
    ).cast(dict(UNITS_SCHEMA))


def read_wide_txt(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Format large 2017-2022 (bureaux, communes ou France entière) : une ligne par unité,
    colonnes fixes puis blocs de 7 par candidat (N°Panneau, Sexe, Nom, Prénom, Voix, % ins,
    % exp) ; l'en-tête ne nomme que le premier bloc."""
    if len(rounds) != 1:
        raise FormatError(f"{path.name}: un seul tour attendu par fichier large, reçu {rounds}")
    rows = list(csv.reader(io.StringIO(_read_text(path, "latin-1")), delimiter=";"))
    header, body = rows[0], [r for r in rows[1:] if r]
    start = _index(header, "N°Panneau")
    i_name, i_first, i_votes = (
        _index(header, "Nom") - start,
        _index(header, "Prénom") - start,
        _index(header, "Voix") - start,
    )
    size = 7
    i_reg, i_vot, i_exp = (
        _index(header, "Inscrits"),
        _index(header, "Votants"),
        _index(header, "Exprimés"),
    )
    i_dept = header.index("Code du département") if "Code du département" in header else None
    i_com = header.index("Code de la commune") if "Code de la commune" in header else None
    i_bv = header.index("Code du b.vote") if "Code du b.vote" in header else None
    i_level = header.index("Code du niveau") if "Code du niveau" in header else None
    out = []
    for row in body:
        dept = row[i_dept].strip() if i_dept is not None else None
        com = row[i_com].strip() if i_com is not None else None
        unit = "|".join(row[i].strip() for i in (i_dept, i_com, i_bv, i_level) if i is not None)
        reg, vot, exp = to_int(row[i_reg]), to_int(row[i_vot]), to_int(row[i_exp])
        for g in _groups(row, start, size):
            label = normalize_label(g[i_name], g[i_first])
            out.append((rounds[0], dept, com, unit, reg, vot, exp, label, to_int(g[i_votes])))
    return pl.DataFrame(out, schema=UNITS_SCHEMA, orient="row")


def read_xls_dept(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Classeurs 2002-2007 : feuilles « Départements T1/T2 », une ligne par département,
    blocs de 6 par candidat (Sexe, Nom, Prénom, Voix, % ins, % exp), ligne total sans code."""
    book = xls.read_xls(path)
    out = []
    for rnd in rounds:
        sheet = f"Départements T{rnd}"
        if sheet not in book:
            raise FormatError(f"{path.name}: feuille {sheet!r} absente ({list(book)})")
        rows = book[sheet]
        h = next((i for i, r in enumerate(rows) if r and r[0] == "Code du département"), None)
        if h is None:
            raise FormatError(f"{path.name}/{sheet}: en-tête introuvable")
        header = rows[h]
        start = _index(header, "Sexe")
        i_reg, i_vot, i_exp = (
            _index(header, "Inscrits"),
            _index(header, "Votants"),
            _index(header, "Exprimés"),
        )
        for row in rows[h + 1 :]:
            if _blank(row[0]):
                continue  # ligne total ou vide
            dept = normalize_dept_src(row[0])
            reg, vot, exp = to_int(row[i_reg]), to_int(row[i_vot]), to_int(row[i_exp])
            for g in _groups(row, start, 6):
                label = normalize_label(str(g[1]), str(g[2]))
                out.append((rnd, dept, None, dept, reg, vot, exp, label, to_int(g[3])))
    return pl.DataFrame(out, schema=UNITS_SCHEMA, orient="row")


# --- Européennes et législatives --------------------------------------------------------


def _list_rows(
    header: Sequence[object],
    body: Sequence[Sequence[object]],
    *,
    rnd: int,
    unit_col: str,
    first_group_col: str,
    size: int,
    key: int,
    label: int,
    nuance: int | None,
    votes: int,
    registered: str = "Inscrits",
    voters: str = "Votants",
    expressed: str = "Exprimés",
) -> list[tuple]:
    start = _index(header, first_group_col)
    i_unit = _index(header, unit_col)
    i_reg, i_vot, i_exp = (
        _index(header, registered),
        _index(header, voters),
        _index(header, expressed),
    )
    out = []
    for row in body:
        if _blank(row[i_unit]):
            continue
        unit = normalize_dept_src(row[i_unit])
        reg, vot, exp = to_int(row[i_reg]), to_int(row[i_vot]), to_int(row[i_exp])
        for g in _groups(row, start, size):
            k = g[key]
            k = str(to_int(k)) if isinstance(k, float) else str(k).strip()
            nu = None if nuance is None or _blank(g[nuance]) else str(g[nuance]).strip()
            out.append((rnd, unit, reg, vot, exp, k, str(g[label]).strip(), nu, to_int(g[votes])))
    return out


def read_euro2014_xlsx(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Européennes 2014 : feuille « Départements », blocs de 7 (N°Liste, Nuance, Libellé
    abrégé, Tête de liste, Voix, % ins, % exp). Clé d'agrégation : nuance + libellé abrégé
    (les listes diffèrent d'une circonscription à l'autre)."""
    rows = xlsx.read_xlsx(path, sheets=["Départements"])["Départements"]
    h = next(i for i, r in enumerate(rows) if "Code du département" in r)
    header = rows[h]
    out = _list_rows(
        header, rows[h + 1 :], rnd=rounds[0], unit_col="Code du département",
        first_group_col="N°Liste", size=7, key=1, label=2, nuance=1, votes=4,
    )  # fmt: skip
    df = pl.DataFrame(out, schema=LIST_UNITS_SCHEMA, orient="row")
    return df.with_columns(pl.concat_str(["nuance", "label"], separator="|").alias("key"))


def read_euro_dept_txt(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Européennes 2019 : une ligne par département, blocs de 7 (N°Liste, Libellé abrégé,
    Libellé étendu, Tête de liste, Voix, % ins, % exp), sans nuance."""
    rows = list(csv.reader(io.StringIO(_read_text(path, "latin-1")), delimiter=";"))
    out = _list_rows(
        rows[0], [r for r in rows[1:] if r], rnd=rounds[0], unit_col="Code du département",
        first_group_col="N°Liste", size=7, key=0, label=2, nuance=None, votes=4,
    )  # fmt: skip
    return pl.DataFrame(out, schema=LIST_UNITS_SCHEMA, orient="row")


def read_euro_dept_csv(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Européennes 2024 : CSV UTF-8 entre guillemets, blocs de 8 (panneau, nuance, libellé
    abrégé, libellé, voix, % ins, % exp, sièges) ; en-têtes numérotés (« Voix 1 »…)."""
    rows = list(csv.reader(io.StringIO(_read_text(path, "utf-8-sig")), delimiter=";"))
    out = _list_rows(
        rows[0], [r for r in rows[1:] if r], rnd=rounds[0], unit_col="Code département",
        first_group_col="Numéro de panneau 1", size=8, key=0, label=3, nuance=1, votes=4,
    )  # fmt: skip
    return pl.DataFrame(out, schema=LIST_UNITS_SCHEMA, orient="row")


def read_leg_fe_txt(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Législatives 2022, France entière : blocs de 5 (nuance, voix, % ins, % exp, sièges)."""
    rows = list(csv.reader(io.StringIO(_read_text(path, "latin-1")), delimiter=";"))
    out = _list_rows(
        rows[0], [r for r in rows[1:] if r], rnd=rounds[0], unit_col="Code du niveau",
        first_group_col="Code Nuance", size=5, key=0, label=0, nuance=0, votes=1,
    )  # fmt: skip
    return pl.DataFrame(out, schema=LIST_UNITS_SCHEMA, orient="row")


def read_leg_fe_xlsx(path: Path, rounds: Sequence[int]) -> pl.DataFrame:
    """Législatives 2024, France entière : blocs de 4 (nuance, voix, % ins, % exp)."""
    book = xlsx.read_xlsx(path)
    if len(book) != 1:
        raise FormatError(f"{path.name}: une seule feuille attendue, reçu {list(book)}")
    rows = next(iter(book.values()))
    out = _list_rows(
        rows[0], rows[1:], rnd=rounds[0], unit_col="Code localisation",
        first_group_col="Nuance candidat 1", size=4, key=0, label=0, nuance=0, votes=1,
    )  # fmt: skip
    return pl.DataFrame(out, schema=LIST_UNITS_SCHEMA, orient="row")


PRESIDENTIAL_READERS = {
    "bvot_long": read_bvot_long,
    "wide_txt": read_wide_txt,
    "xls_dept": read_xls_dept,
}

LIST_READERS = {
    "euro2014_xlsx": read_euro2014_xlsx,
    "euro_dept_txt": read_euro_dept_txt,
    "euro_dept_csv": read_euro_dept_csv,
    "leg_fe_txt": read_leg_fe_txt,
    "leg_fe_xlsx": read_leg_fe_xlsx,
}
