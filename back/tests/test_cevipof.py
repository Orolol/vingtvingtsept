"""Enquêtes du Cevipof (vvs.ingest.cevipof) : inventaire, codebooks DDI, recodage, reports.

Fixtures jouets dans tests/fixtures/cevipof/ :
- ddi122_toy.xml : codebook au format de l'export Dataverse (DDI 1.2.2, espace de noms ICPSR),
  variables du FES 2017 (participation, vote aux deux tours, poids w5) et une anticipation ;
- ddi25_toy.xml : codebook au format CDSP (DDI 2.5), un duel du PEF 2002 ;
- toy_avant_import.zip : CSV au format du fichier public FES 2017 (point-virgule, BOM,
  virgule décimale sans zéro, zip dans le zip), huit répondants ;
- aliases_toy.yaml : table d'alias et configuration du jeu jouet.

Aucun accès réseau : le client Dataverse est remplacé quand il le faut.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from vvs.ingest.cevipof import codebook as cb
from vvs.ingest.cevipof import dataverse as dv
from vvs.ingest.cevipof import ingest as cv
from vvs.ingest.cevipof import parse as P
from vvs.ingest.cevipof import transfers as T
from vvs.ingest.cevipof.registry import DataFile, Dataset, Inventory, load_inventory

FIXTURES = Path(__file__).parent / "fixtures" / "cevipof"


# --- inventaire


def test_registry_file_is_valid():
    inv = load_inventory()
    keys = {d.key for d in inv.datasets}
    assert {"fes2017", "enef2017_v13", "enef2022_v10", "pef2002_v1"} <= keys
    # toute demande d'accès porte sur un fichier `data` restreint
    for d in inv.datasets:
        if d.request:
            assert d.data_file is not None and d.data_file.restricted, d.key


def _dataset(**kw) -> dict:
    base = {
        "key": "toy_v1",
        "pid": "doi:10.21410/7E4/ABCDEF",
        "url": "https://doi.org/10.21410/7E4/ABCDEF",
        "title": "Jouet",
        "series": "toy",
        "election": 2017,
        "version": "1.0",
        "license": "CC BY-SA 4.0",
        "access": "restricted_request",
        "files": [
            {
                "id": 1,
                "name": "x-1.tab",
                "format": "application/x-stata-14",
                "size": 10,
                "restricted": True,
                "role": "data",
            }
        ],
    }
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://doi.org/10.21410/7E4/ZZZZZZ"},  # url ≠ DOI
        {"pid": "doi:10.1234/xyz"},
        {"key": "Toy-V1"},
        {"request": True, "files": []},
        {"fieldwork_start": "2017-05-02", "fieldwork_end": "2017-05-01"},
    ],
)
def test_registry_rejects_inconsistent_dataset(change):
    with pytest.raises(ValueError):
        Inventory.model_validate(
            {"retrieved": "2026-10-22", "server": "x", "datasets": [_dataset(**change)]}
        )


# --- codebooks DDI


@pytest.mark.parametrize("name", ["ddi122_toy.xml", "ddi25_toy.xml"])
def test_parse_ddi_both_variants(name):
    vs = {v.name: v for v in cb.parse_ddi(FIXTURES / name)}
    if name == "ddi122_toy.xml":
        assert vs["A1"].label == "ID" and vs["A1"].categories == ()
        v2 = vs["V2"]
        assert v2.question == "Who did you vote for ?"
        assert [(c.value, c.label, c.freq) for c in v2.categories][:2] == [
            ("2", "Melenchon", 2),
            ("4", "Macron", 2),
        ]
    else:
        v = vs["xq24"]
        assert v.categories[-1].freq is None  # catégorie sans effectif
        assert "Chirac" in v.question
        assert set(cb.find_questions([v])) >= {"duel_intention"}


# --- recodage


def test_normalize_labels():
    assert P.normalize("(A voté blanc)") == P.normalize("a vote blanc") == "a vote blanc"
    assert P.normalize("Nicolas Dupont- Aignan") == P.normalize("nicolas dupont aignan")
    assert P.normalize("Je n'irais pas voter") == "je n irais pas voter"


def test_real_aliases_load_and_cover_reserved_codes():
    al = P.load_aliases()
    codes = set(al.labels.values()) - {None}
    assert {"abstention", "blank", "refused", "undecided"} <= codes
    assert al.labels[P.normalize("Le Pen")] == "le_pen_marine"
    assert al.labels[P.normalize("Jean-Marie LE PEN")] == "le_pen_jean_marie"
    # refuser l'urne du vote simulé (PEF 2007) est un refus, pas un vote blanc
    assert al.labels[P.normalize("Refuse de voter dans l'urne")] == "refused"


def test_aliases_conflict_is_an_error(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(
        yaml.safe_dump({"candidates": {"Macron": "macron_emmanuel", "MACRON": "fillon_francois"}})
    )
    with pytest.raises(P.ConfigError):
        P.load_aliases(p)


@pytest.fixture()
def toy_env(monkeypatch):
    """Jeu jouet : registre d'un seul jeu, codebook DDI 1.2.2 jouet, CSV zippé jouet."""
    al = P.load_aliases(FIXTURES / "aliases_toy.yaml")
    ds = Dataset.model_validate(
        _dataset(
            key="fes2017",
            series="fes2017",
            files=[
                {
                    "id": 3792,
                    "name": "toy_avant_import.zip",
                    "format": "application/zip",
                    "size": 1,
                    "restricted": False,
                    "role": "data_alt",
                    "fetch": True,
                }
            ],
        )
    )
    monkeypatch.setattr(P, "load_codebook", lambda d: cb.parse_ddi(FIXTURES / "ddi122_toy.xml"))
    return al, ds


def test_read_and_recode_toy_csv(toy_env):
    al, ds = toy_env
    spec = al.datasets["fes2017"]
    raw = P.read_raw(ds, spec, FIXTURES / "toy_avant_import.zip")
    votes, duels, exp = P.recode(ds, spec, raw, al, P.valid_codes_for(2017))
    assert votes["weight"].tolist() == pytest.approx([1.2, 0.8, 1, 1, 1.5, 1, 0.5, 1])
    assert votes["vote_round1"].tolist() == [
        "macron_emmanuel",
        "macron_emmanuel",
        "melenchon_jean_luc",
        "melenchon_jean_luc",
        "le_pen_marine",
        "abstention",
        "abstention",
        "refused",
    ]
    # répondant 8 : a voté au second tour mais choix « non concerné » -> refus
    assert votes["vote_round2"].tolist()[-3:] == ["abstention", "macron_emmanuel", "refused"]
    assert votes["cluster"].tolist()[:2] == ["p1", "p1"]
    assert duels.empty and exp.empty


def test_unknown_modality_fails_explicitly(toy_env):
    al, ds = toy_env
    spec = al.datasets["fes2017"]
    raw = P.read_raw(ds, spec, FIXTURES / "toy_avant_import.zip")
    raw.loc[0, "V2"] = "Lassalle"  # absent de la table jouet
    with pytest.raises(P.UnknownModality, match="V2 : 'Lassalle'"):
        P.recode(ds, spec, raw, al, P.valid_codes_for(2017))


def test_duel_choice_outside_options_fails(toy_env):
    al, ds = toy_env
    spec = P.DatasetSpec(
        key="fes2017",
        reader="csv_zip",
        member=None,
        sep=";",
        respondent_id="A1",
        weight=None,
        vote_round1=None,
        certainty_round1=None,
        vote_round2=None,
        turnout_intent=None,
        expected_winner=None,
        duels=(
            P.DuelSpec(
                var="V4", office="president", options=("macron_emmanuel", "fillon_francois")
            ),
        ),
        expectations=(),
        socio={},
        labels={},
    )
    raw = pd.DataFrame({"A1": ["1", "2"], "V4": ["Macron", "Le Pen"]})
    with pytest.raises(P.UnknownModality, match="hors options"):
        P.recode(ds, spec, raw, al, P.valid_codes_for(2017))


def test_codebook_coverage_reports_missing_labels(toy_env):
    al, ds = toy_env
    assert P.codebook_coverage(ds, al.datasets["fes2017"], al) == []
    partial = P.Aliases(
        labels={k: v for k, v in al.labels.items() if k != "le pen"},
        turnout=al.turnout,
        certainty=al.certainty,
        datasets=al.datasets,
    )
    problems = P.codebook_coverage(ds, al.datasets["fes2017"], partial)
    assert any("'Le Pen' sans code" in p for p in problems)


# --- matrices de reports


def _votes(rows, dataset="fes2017", series="fes2017", wave_order=None, kinds=("recall", "recall")):
    df = pd.DataFrame(
        rows, columns=["respondent_id", "vote_round1", "vote_round2", "weight", "cluster"]
    )
    df["election"] = 2017
    df["dataset"], df["series"], df["wave_order"] = dataset, series, wave_order
    df["vote_round1_kind"], df["vote_round2_kind"] = kinds
    return df


def test_transfer_matrix_shares_exclusions_and_bootstrap():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(400):
        r1 = ["melenchon_jean_luc", "fillon_francois", "abstention"][i % 3]
        r2 = rng.choice(
            ["macron_emmanuel", "le_pen_marine", "blank", "abstention", "refused"],
            p=[0.5, 0.15, 0.15, 0.15, 0.05],
        )
        rows.append((str(i), r1, r2, 1.0 + (i % 2), f"c{i % 20}"))
    votes = pl.from_pandas(_votes(rows))
    (src,) = T.pair_sources(votes, 2017)
    m = T.transfer_matrix(src, ("macron_emmanuel", "le_pen_marine"), bootstrap=200)
    assert np.allclose(m.shares.sum(axis=1), 100)
    assert (m.excluded_share > 0).all()  # les refus du second tour sont rapportés
    assert ((m.low <= m.shares + 1e-9) & (m.shares <= m.high + 1e-9)).all().all()
    est, lo, hi = m.interval(T.expressed_split(m.finalists))
    assert ((lo <= est) & (est <= hi)).all()
    # poids : la ligne Mélenchon est une moyenne pondérée, pas une moyenne simple
    d = src.data[(src.data["r1"] == "melenchon_jean_luc") & ~src.data["r2"].isin(T.EXCLUDED)]
    w = d.groupby("r2")["weight"].sum()
    assert m.shares.loc["melenchon_jean_luc", "macron_emmanuel"] == pytest.approx(
        100 * w["macron_emmanuel"] / w.sum()
    )


def test_transfer_matrix_rejects_unexpected_second_round_vote():
    votes = pl.from_pandas(_votes([("1", "hamon_benoit", "fillon_francois", 1.0, None)]))
    (src,) = T.pair_sources(votes, 2017)
    with pytest.raises(ValueError, match="inattendus"):
        T.transfer_matrix(src, ("macron_emmanuel", "le_pen_marine"))


def test_panel_waves_are_joined_by_respondent():
    w14 = _votes(
        [("a", "hamon_benoit", None, 1.0, None), ("b", "fillon_francois", None, 1.0, None)],
        dataset="enef2017_v14",
        series="enef2017",
        wave_order=14,
        kinds=("recall", "intention"),
    )
    w15 = _votes(
        [
            ("a", None, "macron_emmanuel", 2.0, None),
            ("b", None, "blank", 1.0, None),
            ("c", None, "abstention", 1.0, None),
        ],
        dataset="enef2017_v15",
        series="enef2017",
        wave_order=15,
        kinds=(None, "recall"),
    )
    votes = pl.from_pandas(pd.concat([w14, w15], ignore_index=True))
    (src,) = T.pair_sources(votes, 2017)
    assert src.label == "enef2017_v14 -> enef2017_v15"
    assert sorted(zip(src.data["r1"], src.data["r2"], src.data["weight"], strict=True)) == [
        ("fillon_francois", "blank", 1.0),
        ("hamon_benoit", "macron_emmanuel", 2.0),
    ]


def test_proposal_format_is_accepted_by_transfers_registry(tmp_path):
    """Une source `cevipof_<année>` au format de la proposition passe la validation."""
    from vvs.registry.transfers import load_transfers

    src = {
        "id": "cevipof_2017",
        "election": 2017,
        "pollster": "cevipof",
        "title": "t",
        "url": "https://doi.org/10.21410/7E4/2XE05I",
        "base": "first_round_electorate",
        "base_note": "n",
        "columns": "split",
        "rows": [
            {
                "from": "melenchon_jean_luc",
                "to": {
                    "macron_emmanuel": 47.8,
                    "le_pen_marine": 10.1,
                    "blank_null": 25.1,
                    "abstain": 17.0,
                },
            },
            {
                "from": "small_far_left",
                "members": ["arthaud_nathalie", "poutou_philippe"],
                "to": {
                    "macron_emmanuel": 37.0,
                    "le_pen_marine": 8.9,
                    "blank_null": 38.5,
                    "abstain": 15.6,
                },
            },
            {
                "from": "abstain",
                "to": {
                    "macron_emmanuel": 11.9,
                    "le_pen_marine": 5.2,
                    "blank_null": 2.9,
                    "abstain": 80.0,
                },
            },
        ],
    }
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump({"transfers": [src]}, allow_unicode=True))
    (s,) = load_transfers(p)
    assert s.rows[1].members == ["arthaud_nathalie", "poutou_philippe"]


# --- téléchargement


def test_local_name_uses_original_extension():
    f = DataFile(
        id=1,
        name=" fr_cdsp_ddi_fes2017.tab",
        format="application/x-stata-14",
        size=1,
        restricted=True,
        role="data",
    )
    assert cv.local_name(f) == "fr_cdsp_ddi_fes2017.dta"
    g = DataFile(
        id=2,
        name="x.tab",
        format="application/x-spss-sav",
        size=1,
        restricted=True,
        role="data_alt",
    )
    assert cv.local_name(g) == "x.sav"


def test_ingest_never_rewrites_and_checks_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(cv.paths, "RAW", tmp_path)
    ds = Dataset.model_validate(
        _dataset(
            files=[
                {
                    "id": 9,
                    "name": "Doc.pdf",
                    "format": "application/pdf",
                    "size": 3,
                    "restricted": False,
                    "role": "documentation",
                    "fetch": True,
                }
            ]
        )
    )
    inv = Inventory(retrieved="2026-10-22", server="x", datasets=[ds])
    calls = []

    def fake_download(file_id, dest, original=True):
        calls.append(file_id)
        dest.write_bytes(b"abc")
        return dv.Download(3, "application/pdf")

    monkeypatch.setattr(cv.dv, "download", fake_download)
    r1 = cv.ingest(inv, keys=["toy_v1"])
    assert [r[2] for r in r1.rows] == ["downloaded"]
    meta = json.loads((tmp_path / "cevipof" / "toy_v1" / "Doc.pdf.source.json").read_text())
    assert meta["sha256"] == cv.sha256_of(tmp_path / "cevipof" / "toy_v1" / "Doc.pdf")
    r2 = cv.ingest(inv, keys=["toy_v1"])
    assert [r[2] for r in r2.rows] == ["present"] and calls == [9]
    (tmp_path / "cevipof" / "toy_v1" / "Doc.pdf").write_bytes(b"abd")  # fichier altéré
    with pytest.raises(cv.ConsistencyError):
        cv.ingest(inv, keys=["toy_v1"])


def test_restricted_file_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(cv.paths, "RAW", tmp_path)
    inv = Inventory(
        retrieved="2026-10-22", server="x", datasets=[Dataset.model_validate(_dataset())]
    )

    def denied(file_id, dest, original=True):
        raise dv.AccessDenied("refusé")

    monkeypatch.setattr(cv.dv, "download", denied)
    rep = cv.ingest(inv, keys=["toy_v1"])
    assert [r[2] for r in rep.rows] == ["restricted"]
    assert not any((tmp_path / "cevipof" / "toy_v1").iterdir())


def test_token_is_never_sent_without_env(monkeypatch):
    monkeypatch.delenv(dv.TOKEN_ENV, raising=False)
    assert "X-Dataverse-key" not in dv._headers()
    with pytest.raises(dv.DataverseError):
        dv.request_access(1)
