"""Inférence écologique des reports (vvs.model.ecological) et indicateurs communaux de Cagé et
Piketty (vvs.ingest.conflit_politique).

Fixtures : tests/fixtures/conflit_politique/, extrait figé des six archives (10 communes, années
2017 et 2022, colonnes retenues seulement) avec son registre. Aucun accès réseau.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from vvs.ingest import conflit_politique as C
from vvs.model import ecological as E

FIXTURES = Path(__file__).parent / "fixtures" / "conflit_politique"


# --- Goodman contraint


def simulate(B: np.ndarray, n: int, rng: np.random.Generator, size: int = 2000) -> tuple:
    """Communes de `size` inscrits ; compositions de premier tour tirées d'une Dirichlet,
    second tour tiré électeur par électeur selon B (multinomiale par électorat)."""
    K, J = B.shape
    X = rng.dirichlet(np.full(K, 2.0), size=n)
    counts = np.array([rng.multinomial(size, x) for x in X])
    Y = np.zeros((n, J))
    for k in range(K):
        Y += np.array([rng.multinomial(c, B[k]) for c in counts[:, k]])
    w = np.full(n, float(size))
    return counts / size, Y / size, w


TRUE_B = np.array([
    [0.90, 0.02, 0.03, 0.05],
    [0.05, 0.85, 0.04, 0.06],
    [0.55, 0.15, 0.12, 0.18],
    [0.10, 0.05, 0.00, 0.85],  # cellule nulle : solution au bord du simplexe
])


def test_goodman_recovers_known_transfers():
    rng = np.random.default_rng(1)
    X, Y, w = simulate(TRUE_B, 3000, rng)
    B = E.goodman(X, Y, w)
    assert np.abs(B - TRUE_B).max() < 0.02
    np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-9)
    assert (B >= 0).all()


def test_constraints_bind_when_unconstrained_fit_leaves_simplex():
    # bruit fort sur peu de communes : les moindres carrés libres sortent de [0, 1]
    rng = np.random.default_rng(2)
    X, Y, w = simulate(TRUE_B, 15, rng, size=25)
    G, H = E.sufficient_stats(X, Y, w)
    free = np.linalg.solve(G, H)
    assert (free < 0).any() or (free > 1).any()
    B = E.solve_constrained(G, H)
    np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-9)
    assert (B >= 0).all() and (B <= 1).all()
    # optimalité : aucune matrice admissible voisine ne fait mieux
    obj = lambda M: np.sum(M * (G @ M)) - 2 * np.sum(M * H)
    for _ in range(200):
        P = E.project_simplex_rows(B + rng.normal(0, 0.01, B.shape))
        assert obj(P) >= obj(B) - 1e-10


def test_project_simplex_rows():
    P = E.project_simplex_rows(np.array([[0.5, 0.5, 0.5], [2.0, -1.0, 0.0], [0.2, 0.3, 0.5]]))
    np.testing.assert_allclose(P.sum(1), 1.0)
    assert (P >= 0).all()
    np.testing.assert_allclose(P[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(P[2], [0.2, 0.3, 0.5])


def test_stratified_recovers_each_stratum_and_aggregates_by_size():
    rng = np.random.default_rng(3)
    B2 = TRUE_B.copy()
    B2[2] = [0.20, 0.50, 0.12, 0.18]  # le troisième électorat se comporte autrement en strate b
    Xa, Ya, wa = simulate(TRUE_B, 1500, rng)
    Xb, Yb, wb = simulate(B2, 1500, rng)
    X, Y, w = np.vstack([Xa, Xb]), np.vstack([Ya, Yb]), np.concatenate([wa, wb])
    labels = ["a"] * 1500 + ["b"] * 1500
    fit = E.stratified(X, Y, w, labels)
    assert np.abs(fit.B["a"] - TRUE_B).max() < 0.03
    assert np.abs(fit.B["b"] - B2).max() < 0.03
    nat = fit.national()
    na, nb = fit.sizes["a"][2], fit.sizes["b"][2]
    np.testing.assert_allclose(nat[2], (na * fit.B["a"][2] + nb * fit.B["b"][2]) / (na + nb))
    np.testing.assert_allclose(nat.sum(1), 1.0, atol=1e-9)
    # composition imposée : seule la strate b compte
    only_b = fit.national({"b": fit.sizes["b"]})
    np.testing.assert_allclose(only_b, fit.B["b"])


def test_split_none():
    s, n = E.split_none(np.array([[0.6, 0.2, 0.05, 0.15], [0.0, 0.0, 0.5, 0.5]]))
    assert s[0] == pytest.approx(0.75) and n[0] == pytest.approx(0.2)
    assert np.isnan(s[1]) and n[1] == pytest.approx(1.0)


def test_weighted_quantile_bins():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert E.weighted_quantile_bins(v, w, 2).tolist() == [1, 1, 2, 2, 0]
    # poids : la première commune porte la moitié du poids, elle forme la première classe
    assert E.weighted_quantile_bins(v[:4], np.array([3.0, 1.0, 1.0, 1.0]), 2).tolist() == [1, 2, 2, 2]


# --- données communales et appariement


def toy_results() -> pl.DataFrame:
    rows = []

    def add(rnd, code, reg, voters, expressed, votes):
        for cid, v in votes.items():
            rows.append({"election": 2017, "round": rnd, "geo_level": "commune", "geo_code": code,
                         "candidate_id": cid, "votes": v, "registered": reg, "voters": voters,
                         "expressed": expressed})

    add(1, "01001", 100, 80, 78, {"macron_emmanuel": 30, "le_pen_marine": 28, "hamon_benoit": 20})
    add(2, "01001", 101, 75, 70, {"macron_emmanuel": 45, "le_pen_marine": 25})
    add(1, "75056", 1000, 800, 790, {"macron_emmanuel": 500, "le_pen_marine": 90, "hamon_benoit": 200})
    add(2, "75056", 1000, 750, 720, {"macron_emmanuel": 600, "le_pen_marine": 120})
    return pl.DataFrame(rows)


def test_commune_counts_groups_and_special_electorates():
    g = E.Groups(2017, ("macron_emmanuel", "le_pen_marine"), {
        "macron_emmanuel": ("macron_emmanuel",), "le_pen_marine": ("le_pen_marine",),
        "left": ("hamon_benoit",), E.BLANK_NULL: (), E.ABSTAIN: (),
    })
    df = E.commune_counts(toy_results(), g)
    r = df.filter(pl.col("geo_code") == "01001").row(0, named=True)
    assert r["x_left"] == 20 and r["x_blank_null"] == 2 and r["x_abstain"] == 20
    assert r["y_blank_null"] == 5 and r["y_abstain"] == 26 and r["registered2"] == 101
    X, Y, w = E.matrices(df, g)
    np.testing.assert_allclose(X.sum(1), 1.0)
    np.testing.assert_allclose(Y.sum(1), 1.0)
    assert w.tolist() == [100.0, 1000.0]


def test_commune_counts_rejects_ungrouped_candidate():
    g = E.Groups(2017, ("macron_emmanuel", "le_pen_marine"), {
        "macron_emmanuel": ("macron_emmanuel",), "le_pen_marine": ("le_pen_marine",),
        E.BLANK_NULL: (), E.ABSTAIN: (),
    })
    with pytest.raises(ValueError, match="hamon_benoit"):
        E.commune_counts(toy_results(), g)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("01001", "matched"),
        ("75056", "matched"),
        ("75108", "plm_arrondissement"),
        ("13201", "plm_arrondissement"),
        ("69381", "plm_arrondissement"),
        ("97101", "overseas_or_abroad"),
        ("98714", "overseas_or_abroad"),
        ("ZZ079", "overseas_or_abroad"),
        ("50173", "merged_or_unknown"),  # Cherbourg-Octeville, dans Cherbourg-en-Cotentin depuis 2016
    ],
)
def test_match_status(code, status):
    assert E.match_status(code, {"01001", "75056", "50129"}) == status


# --- indicateurs communaux (extrait figé)


@pytest.fixture
def fixture_registry(monkeypatch):
    monkeypatch.setattr(C, "raw_dir", lambda: FIXTURES)
    return C.load_registry(FIXTURES / "registry.yaml")


def test_build_from_frozen_extract(fixture_registry):
    df = C.build(fixture_registry)
    assert set(df["year"].unique()) == {2017, 2022}
    codes = set(df["codecommune"].unique())
    # géographie de référence : le fichier des populations ; arrondissements PLM écartés,
    # ancienne commune (50173, absente du fichier des populations) écartée
    assert codes == {"01001", "01004", "75056", "69123", "13055", "50129", "59350"}
    assert df.select(pl.len().over("codecommune", "year").max()).item() == 1
    for col in C.RETAINED:
        assert col in df.columns
    r = df.filter((pl.col("codecommune") == "01001") & (pl.col("year") == 2017)).row(0, named=True)
    assert r["urban_type"] == "rural" and r["population"] == pytest.approx(r["pop"])
    assert 0 < r["higher_edu_share"] < 1 and 0 < r["owner_share"] < 1


def test_paris_filled_from_arrondissements(fixture_registry):
    """Paris n'a pas de ligne entière dans le fichier des CSP : moyenne des arrondissements de
    l'extrait (1er et 2e) pondérée par leur population."""
    import io
    import zipfile

    def raw(key, col):
        f = fixture_registry.file(key)
        with zipfile.ZipFile(FIXTURES / f.name) as z:
            d = pl.read_csv(io.BytesIO(z.read(f.members[0])), infer_schema=False)
        return {c: float(v) for c, v in d.select("codecommune", col).iter_rows() if v is not None}

    pcadr, pop = raw("csp", "pcadr2017"), raw("agglo", "pop2017")
    assert "75056" not in pcadr
    expected = (pcadr["75101"] * pop["75101"] + pcadr["75102"] * pop["75102"]) / (pop["75101"] + pop["75102"])
    df = C.build(fixture_registry)
    got = df.filter((pl.col("codecommune") == "75056") & (pl.col("year") == 2017))["managers_share"].item()
    assert got == pytest.approx(expected)
    assert df.filter(pl.col("codecommune") == "75056")["urban_type"].to_list() == ["metro", "metro"]


def test_urban_type_thresholds():
    s = pl.Series([None, 0.0, 1999.0, 2000.0, 99_999.0, 100_000.0])
    got = pl.select(C.urban_type(pl.lit(s))).to_series().to_list()
    assert got == [None, "rural", "rural", "town", "town", "metro"]


def test_checksum_mismatch_fails(fixture_registry, monkeypatch):
    bad = [f if f.key != "csp" else C.SourceFile(f.key, f.name, f.url, f.members, "0" * 64, f.bytes)
           for f in fixture_registry.files]
    reg = C.Registry(**{**fixture_registry.__dict__, "files": tuple(bad)})
    with pytest.raises(C.ConsistencyError, match="sha256"):
        C.build(reg)


def test_real_registry_is_consistent():
    reg = C.load_registry()
    assert {f.key for f in reg.files} == set(C.ORIGINAL_VARIABLES)
    assert all(f.sha256 and len(f.sha256) == 64 for f in reg.files)
    assert sum(f.bytes for f in reg.files) < 2e9
    assert set(C.RETAINED.values()) <= {s for v in C.ORIGINAL_VARIABLES.values() for s in v}
