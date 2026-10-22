"""Microdonnées du Cevipof -> data/curated/cevipof_*.parquet.

Recodage piloté par data/registry/aliases_cevipof.yaml :
- `labels` : libellé de modalité normalisé -> code (candidate_id, bloc pour les duels de
  législatives, ou code réservé) ; `null` = question non posée au répondant ;
- `datasets.<key>` : variables à lire (identifiant, poids, vote, intentions de duel,
  certitude, participation, anticipations, sociodémographie).

Une modalité absente de la table fait échouer le parsing (`UnknownModality`) : on complète
la table, on ne devine pas.

Sorties (une ligne par répondant et par vague) :
- `cevipof_votes.parquet` : election, dataset, series, wave, wave_order, stage,
  fieldwork_start/end, respondent_id, weight, vote_round1 (+ `_kind` intention/recall),
  certainty_round1, vote_round2 (+ `_kind`), turnout_intent (0-10), expected_winner,
  sex, age, age_group, education ;
- `cevipof_duels.parquet` : une ligne par répondant et par hypothèse de duel posée
  (election, dataset, wave, respondent_id, weight, office, option_a, option_b, choice,
  certainty) ;
- `cevipof_expectations.parquet` : anticipations sur échelle 0-10 (probabilité qu'un
  candidat gagne un duel, accède au second tour), une ligne par répondant et par question.

Codes réservés : `abstention`, `blank` (blanc ou nul), `undecided` (ne sait pas, pas
encore décidé), `refused` (refus de répondre) ; valeur nulle = non posée ou non concerné.
"""

from __future__ import annotations

import functools
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from unidecode import unidecode

from vvs import paths
from vvs.ingest.cevipof.codebook import load_codebook
from vvs.ingest.cevipof.ingest import local_path
from vvs.ingest.cevipof.registry import Dataset, Inventory, default_inventory

ALIASES_FILE = paths.REGISTRY / "aliases_cevipof.yaml"
RESERVED = {"abstention", "blank", "undecided", "refused"}
TURNOUT_VALUES = {"voted", "abstention", "refused", None}
VOTES_OUT = "cevipof_votes.parquet"
DUELS_OUT = "cevipof_duels.parquet"
EXPECT_OUT = "cevipof_expectations.parquet"


class UnknownModality(ValueError):
    """Modalité absente de aliases_cevipof.yaml ; le message les liste toutes."""


class ConfigError(ValueError):
    pass


def normalize(label: str) -> str:
    """Clé de comparaison d'un libellé : ASCII minuscule, ponctuation et parenthèses en
    espaces, espaces fusionnés (« (A voté blanc) » == « a vote blanc »)."""
    s = unidecode(str(label)).lower()
    s = re.sub(r"[()\[\].,;:!?'’`\"/\\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- configuration


@dataclass(frozen=True)
class VoteSpec:
    var: str
    kind: str  # intention | recall
    turnout_var: str | None = None


@dataclass(frozen=True)
class DuelSpec:
    var: str
    office: str  # president | legislative
    options: tuple[str, ...]  # codes des options proposées (candidate_id ou bloc)
    certainty_var: str | None = None


@dataclass(frozen=True)
class ExpectSpec:
    var: str
    question: str  # win_duel | qualify | win_preferred
    target: str | None  # candidate_id visé (None : candidat souhaité du répondant)
    opponent: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    reader: str  # stata | spss | csv_zip | sas_zip
    member: str | None
    sep: str
    respondent_id: str | None  # None : pas encore relevé (codebook muet)
    weight: str | None
    vote_round1: VoteSpec | None
    certainty_round1: str | None
    vote_round2: VoteSpec | None
    turnout_intent: str | None
    expected_winner: str | None
    duels: tuple[DuelSpec, ...]
    expectations: tuple[ExpectSpec, ...]
    socio: dict[str, str]
    labels: dict[str, str | None]  # surcharges propres au jeu
    cluster: str | None = None  # unité primaire de sondage (bootstrap par grappes)

    def variables(self) -> list[str]:
        out = [self.respondent_id] if self.respondent_id else []
        if self.weight:
            out.append(self.weight)
        for v in (self.vote_round1, self.vote_round2):
            if v:
                out.append(v.var)
                if v.turnout_var:
                    out.append(v.turnout_var)
        for x in (self.certainty_round1, self.turnout_intent, self.expected_winner):
            if x:
                out.append(x)
        for d in self.duels:
            out.append(d.var)
            if d.certainty_var:
                out.append(d.certainty_var)
        out += [e.var for e in self.expectations]
        out += list(self.socio.values())
        if self.cluster:
            out.append(self.cluster)
        return list(dict.fromkeys(out))

    def categorical(self) -> list[str]:
        """Variables dont les modalités passent par la table de libellés."""
        out = []
        for v in (self.vote_round1, self.vote_round2):
            if v:
                out.append(v.var)
                if v.turnout_var:
                    out.append(v.turnout_var)
        if self.certainty_round1:
            out.append(self.certainty_round1)
        if self.expected_winner:
            out.append(self.expected_winner)
        for d in self.duels:
            out.append(d.var)
            if d.certainty_var:
                out.append(d.certainty_var)
        return out


@dataclass(frozen=True)
class Aliases:
    labels: dict[str, str | None]  # normalisé -> code
    turnout: dict[str, str | None]
    certainty: dict[str, str | None]
    datasets: dict[str, DatasetSpec]

    def code(self, label: str, ds: DatasetSpec, table: str = "labels") -> tuple[bool, str | None]:
        key = normalize(label)
        if key in ds.labels:
            return True, ds.labels[key]
        t = getattr(self, table)
        if key in t:
            return True, t[key]
        return False, None


def _vote(d: dict | None) -> VoteSpec | None:
    if not d:
        return None
    if d.get("kind") not in ("intention", "recall"):
        raise ConfigError(f"vote : kind {d.get('kind')!r} (intention ou recall)")
    return VoteSpec(var=d["var"], kind=d["kind"], turnout_var=d.get("turnout"))


def _norm_table(t: dict | None) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for k, v in (t or {}).items():
        nk = normalize(k)
        if nk in out and out[nk] != v:
            raise ConfigError(f"libellé {k!r} : deux codes ({out[nk]}, {v})")
        out[nk] = v
    return out


def load_aliases(path: Path | None = None) -> Aliases:
    data = yaml.safe_load((path or ALIASES_FILE).read_text(encoding="utf-8"))
    labels: dict[str, str | None] = {}
    for section in ("candidates", "parties", "special"):
        for k, v in _norm_table(data.get(section)).items():
            if k in labels and labels[k] != v:
                raise ConfigError(f"libellé {k!r} : deux codes ({labels[k]}, {v})")
            labels[k] = v
    turnout = _norm_table(data.get("turnout"))
    bad = set(turnout.values()) - TURNOUT_VALUES
    if bad:
        raise ConfigError(
            f"turnout : valeurs {sorted(map(str, bad))} (voted, abstention, refused, null)"
        )
    certainty = _norm_table(data.get("certainty"))
    specs = {}
    for key, d in (data.get("datasets") or {}).items():
        reader = d.get("reader", "stata")
        if reader not in ("stata", "spss", "csv_zip", "sas_zip"):
            raise ConfigError(f"{key} : reader {reader!r}")
        duels = tuple(
            DuelSpec(
                var=x["var"],
                office=x.get("office", "president"),
                options=tuple(x["options"]),
                certainty_var=x.get("certainty"),
            )
            for x in d.get("duels", [])
        )
        expects = tuple(
            ExpectSpec(
                var=x["var"],
                question=x["question"],
                target=x.get("target"),
                opponent=x.get("opponent"),
            )
            for x in d.get("expectations", [])
        )
        specs[key] = DatasetSpec(
            key=key,
            reader=reader,
            member=d.get("member"),
            sep=d.get("sep", ","),
            respondent_id=d.get("respondent_id"),
            weight=d.get("weight"),
            vote_round1=_vote(d.get("vote_round1")),
            certainty_round1=d.get("certainty_round1"),
            vote_round2=_vote(d.get("vote_round2")),
            turnout_intent=d.get("turnout_intent"),
            expected_winner=d.get("expected_winner"),
            duels=duels,
            expectations=expects,
            socio=dict(d.get("socio") or {}),
            labels=_norm_table(d.get("labels")),
            cluster=d.get("cluster"),
        )
    return Aliases(labels=labels, turnout=turnout, certainty=certainty, datasets=specs)


@functools.cache
def default_aliases() -> Aliases:
    return load_aliases()


# --- lecture des fichiers


def _codebook_labels(ds: Dataset, variables: list[str]) -> dict[str, dict[str, str]]:
    cb = {v.name: v for v in load_codebook(ds)}
    out = {}
    for name in variables:
        v = cb.get(name)
        if v is None:
            raise ConfigError(f"{ds.key} : variable {name} absente du codebook")
        out[name] = {c.value: c.label for c in v.categories if c.label}
    return out


def _code_str(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or x is pd.NA:
        return None
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()


def read_raw(ds: Dataset, spec: DatasetSpec, path: Path | None = None) -> pd.DataFrame:
    """Colonnes de `spec.variables()` ; variables catégorielles en libellés (str)."""
    if path is None:
        f = ds.data_file
        if spec.reader in ("csv_zip", "sas_zip"):
            f = next((x for x in ds.files if x.fetch and x.role == "data_alt"), None)
        if f is None:
            raise ConfigError(f"{ds.key} : pas de fichier de données au registre")
        path = local_path(ds, f)
    if not path.exists():
        raise FileNotFoundError(path)
    wanted = spec.variables()
    cat = spec.categorical()
    if spec.reader == "stata":
        df = pd.read_stata(path, columns=wanted, convert_categoricals=False)
        try:
            # libellés de valeur du fichier ; une valeur sans libellé reste son code
            labelled = pd.read_stata(path, columns=cat, convert_categoricals=True)
        except ValueError as exc:  # libellés non uniques
            raise ConfigError(f"{ds.key} : libellés Stata inexploitables ({exc})") from exc
        for v in cat:
            df[v] = labelled[v].astype(object).where(labelled[v].notna(), None)
        maps = {}
    elif spec.reader == "spss":
        try:
            import pyreadstat
        except ImportError as exc:  # pragma: no cover - dépend du groupe installé
            raise ConfigError(
                f"{ds.key} : fichier SPSS ; installer le groupe : uv sync --group cevipof"
            ) from exc
        df, meta = pyreadstat.read_sav(str(path), usecols=wanted)
        maps = {
            v: {_code_str(k): s for k, s in meta.variable_value_labels.get(v, {}).items()}
            for v in cat
        }
    else:
        with zipfile.ZipFile(path) as z:
            member = spec.member or z.namelist()[0]
            raw = z.read(member)
        if raw[:2] == b"PK":  # archive dans l'archive (FES 2017)
            with zipfile.ZipFile(io.BytesIO(raw)) as z2:
                raw = z2.read(z2.namelist()[0])
        if spec.reader == "csv_zip":
            df = pd.read_csv(
                io.BytesIO(raw), sep=spec.sep, encoding="utf-8-sig", usecols=wanted, dtype=str
            )
        else:
            df = pd.read_sas(io.BytesIO(raw), format="sas7bdat", encoding="utf-8")[wanted]
        maps = _codebook_labels(ds, cat)
    missing = [v for v in wanted if v not in df.columns]
    if missing:
        raise ConfigError(f"{ds.key} : variables absentes du fichier {missing}")
    for v in cat:
        m = maps.get(v, {})
        df[v] = [None if (c := _code_str(x)) is None else m.get(c, c) for x in df[v]]
    return df


def codebook_coverage(ds: Dataset, spec: DatasetSpec, aliases: Aliases) -> list[str]:
    """Problèmes de configuration visibles dès le codebook public : variable absente,
    modalité sans code. Vide = le jeu se parsera dès que les microdonnées seront là."""
    cb = {v.name: v for v in load_codebook(ds)}
    if not cb:
        return [f"{ds.key} : pas de codebook public"]
    problems = [f"{ds.key} : {v} absente du codebook" for v in spec.variables() if v not in cb]
    tables = {}
    for v in (spec.vote_round1, spec.vote_round2):
        if v and v.turnout_var:
            tables[v.turnout_var] = "turnout"
    if spec.certainty_round1:
        tables[spec.certainty_round1] = "certainty"
    for d in spec.duels:
        if d.certainty_var:
            tables[d.certainty_var] = "certainty"
    for var in spec.categorical():
        if var not in cb:
            continue
        for c in cb[var].categories:
            if not c.label:
                continue
            ok, _ = aliases.code(c.label, spec, tables.get(var, "labels"))
            if not ok:
                problems.append(f"{ds.key} : {var} = {c.value} {c.label!r} sans code")
    return problems


# --- recodage


@dataclass
class Recoder:
    aliases: Aliases
    spec: DatasetSpec
    unknown: dict[tuple[str, str], int] = field(default_factory=dict)

    def _map(self, var: str, values, table: str = "labels"):
        out = []
        for x in values:
            if x is None:
                out.append(None)
                continue
            ok, code = self.aliases.code(x, self.spec, table)
            if not ok:
                self.unknown[(var, str(x))] = self.unknown.get((var, str(x)), 0) + 1
            out.append(code)
        return out

    def vote(self, df: pd.DataFrame, v: VoteSpec) -> list[str | None]:
        choice = self._map(v.var, df[v.var])
        if v.turnout_var is None:
            return choice
        turnout = self._map(v.turnout_var, df[v.turnout_var], "turnout")
        out = []
        for t, c in zip(turnout, choice, strict=True):
            if t in ("abstention", "refused"):
                out.append(t)
            elif t == "voted":
                out.append(c if c is not None else "refused")
            else:
                out.append(c)
        return out

    def check(self, valid: set[str], var: str, codes) -> None:
        for c in codes:
            if c is not None and c not in valid and c not in RESERVED:
                self.unknown[(var, f"code {c} hors registre")] = 1


def _num(s: pd.Series) -> pd.Series:
    """Numérique ; les CSV du CDSP écrivent la virgule décimale sans zéro (« ,9997 »)."""
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def recode(
    ds: Dataset,
    spec: DatasetSpec,
    raw: pd.DataFrame,
    aliases: Aliases,
    valid_codes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(votes, duels, anticipations) ; lève `UnknownModality` sur toute modalité inconnue."""
    if spec.respondent_id is None:
        raise ConfigError(f"{ds.key} : identifiant de répondant à relever dans le fichier")
    rc = Recoder(aliases, spec)
    n = len(raw)
    base = pd.DataFrame(
        {
            "election": [ds.election] * n,
            "dataset": ds.key,
            "series": ds.series,
            "wave": ds.wave,
            "wave_order": ds.wave_order,
            "stage": ds.stage,
            "fieldwork_start": ds.fieldwork_start,
            "fieldwork_end": ds.fieldwork_end,
            "respondent_id": raw[spec.respondent_id].map(_code_str),
            "weight": _num(raw[spec.weight]) if spec.weight else 1.0,
            "cluster": raw[spec.cluster].map(_code_str) if spec.cluster else None,
        }
    )
    votes = base.copy()
    for col, v in (("vote_round1", spec.vote_round1), ("vote_round2", spec.vote_round2)):
        votes[col] = rc.vote(raw, v) if v else None
        votes[col + "_kind"] = v.kind if v else None
        if v:
            rc.check(valid_codes, v.var, votes[col])
    votes["certainty_round1"] = (
        rc._map(spec.certainty_round1, raw[spec.certainty_round1], "certainty")
        if spec.certainty_round1
        else None
    )
    votes["turnout_intent"] = (
        _num(raw[spec.turnout_intent]).where(lambda s: s.between(0, 10))
        if spec.turnout_intent
        else None
    )
    if spec.expected_winner:
        votes["expected_winner"] = rc._map(spec.expected_winner, raw[spec.expected_winner])
        rc.check(valid_codes, spec.expected_winner, votes["expected_winner"])
    else:
        votes["expected_winner"] = None
    for col in ("sex", "age", "age_group", "education"):
        var = spec.socio.get(col)
        votes[col] = raw[var].map(_code_str) if var else None
    duel_rows = []
    for d in spec.duels:
        choice = rc._map(d.var, raw[d.var])
        for c in choice:
            if c is not None and c not in d.options and c not in RESERVED:
                rc.unknown[(d.var, f"code {c} hors options {d.options}")] = 1
        cert = (
            rc._map(d.certainty_var, raw[d.certainty_var], "certainty")
            if d.certainty_var
            else [None] * n
        )
        t = base.drop(
            columns=["series", "wave_order", "fieldwork_start", "fieldwork_end", "cluster"]
        ).copy()
        t["office"] = d.office
        t["variable"] = d.var
        t["option_a"], t["option_b"] = d.options[0], d.options[1]
        t["option_c"] = d.options[2] if len(d.options) > 2 else None
        t["choice"] = choice
        t["certainty"] = cert
        duel_rows.append(t[t["choice"].notna()])
    exp_rows = []
    for e in spec.expectations:
        val = _num(raw[e.var]).where(lambda s: s.between(0, 10))
        t = base.drop(
            columns=["series", "wave_order", "fieldwork_start", "fieldwork_end", "cluster"]
        ).copy()
        t["variable"], t["question"], t["target"], t["opponent"] = (
            e.var,
            e.question,
            e.target,
            e.opponent,
        )
        t["value"] = val
        exp_rows.append(t[t["value"].notna()])
    if rc.unknown:
        lines = [
            f"  {var} : {lab!r} ({k} réponses)" for (var, lab), k in sorted(rc.unknown.items())
        ]
        raise UnknownModality(
            f"{ds.key} : modalités absentes de aliases_cevipof.yaml :\n" + "\n".join(lines)
        )
    duels = pd.concat(duel_rows, ignore_index=True) if duel_rows else pd.DataFrame()
    expects = pd.concat(exp_rows, ignore_index=True) if exp_rows else pd.DataFrame()
    return votes, duels, expects


# --- exécution


@dataclass
class ParseReport:
    rows: list[tuple[str, str, int]] = field(default_factory=list)  # key, statut, lignes
    outputs: list[Path] = field(default_factory=list)

    def summary(self, verbose: bool = False) -> str:
        done = [r for r in self.rows if r[1] == "parsed"]
        lines = [f"{len(done)} jeux lus, {sum(r[2] for r in done)} lignes de votes"]
        for key, status, k in self.rows:
            if verbose or status != "parsed":
                lines.append(f"  {status:14s} {key} {k or ''}".rstrip())
            else:
                lines.append(f"  parsed         {key} {k}")
        lines += [f"écrit : {p}" for p in self.outputs]
        return "\n".join(lines)


def valid_codes_for(election: int | None) -> set[str]:
    from vvs.registry import default_registry

    reg = default_registry()
    codes = {"far_left", "left", "centre", "right", "far_right", "other"}
    if election in reg.elections:
        codes |= {c.candidate_id for c in reg.elections[election].candidates}
    return codes


def run(
    inventory: Inventory | None = None, aliases: Aliases | None = None, out_dir: Path | None = None
) -> ParseReport:
    inv = inventory or default_inventory()
    al = aliases or default_aliases()
    out_dir = out_dir or paths.CURATED
    report = ParseReport()
    votes, duels, expects = [], [], []
    for key, spec in al.datasets.items():
        ds = inv.get(key)
        try:
            raw = read_raw(ds, spec)
        except FileNotFoundError:
            report.rows.append((key, "absent", 0))
            continue
        v, d, e = recode(ds, spec, raw, al, valid_codes_for(ds.election))
        votes.append(v)
        duels.append(d)
        expects.append(e)
        report.rows.append((key, "parsed", len(v)))
    for name, frames in ((VOTES_OUT, votes), (DUELS_OUT, duels), (EXPECT_OUT, expects)):
        frames = [f for f in frames if len(f)]
        if not frames:
            continue
        df = pl.from_pandas(pd.concat(frames, ignore_index=True))
        out = out_dir / name
        df.write_parquet(out)
        report.outputs.append(out)
    return report
