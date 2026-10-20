"""Palmarès des instituts : notes sur données jouets, rétrécissement, successions,
leave-one-election-out, réduction au modèle actuel quand l'option est désactivée."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from vvs.model import first_round as fr
from vvs.model import pollster_ratings as pr

BLOCS = ["far_right", "right", "left"]


def toy_tables(
    elections=(1, 2, 3, 4),
    noise_sd: dict[str, float] | None = None,
    house: dict[tuple[str, str], float] | None = None,
    n_polls: dict[str, int] | None = None,
    seed: int = 0,
    election_noise: float = 0.0,
) -> dict[int, tuple[pl.DataFrame, pl.DataFrame]]:
    """Tables (erreurs finales, effets maison) synthétiques : chaque institut a une dispersion
    propre de ses erreurs standardisées (`noise_sd`) et un effet maison persistant par bloc
    (`house`), plus un écart propre à chaque élection (`election_noise`)."""
    rng = np.random.default_rng(seed)
    noise_sd = noise_sd or {"a": 0.6, "b": 1.6, "c": 1.0, "d": 1.0}
    house = house or {}
    n_polls = n_polls or {}
    out = {}
    for e in elections:
        err_rows, house_rows = [], []
        for p, sd in noise_sd.items():
            for k in range(4):
                for c in range(10):
                    err_rows.append({"election": e, "pollster": p, "poll_id": f"{e}-{p}-{k}",
                                     "candidate_id": f"c{c}", "z": float(rng.normal(0, sd))})
            for b in BLOCS:
                a = house.get((p, b), 0.0) + rng.normal(0, election_noise)
                house_rows.append({"election": e, "pollster": p, "bloc": b, "alpha": float(a),
                                   "se": 0.01, "n_polls": n_polls.get(p, 20)})
        h = pl.DataFrame(house_rows)
        # double centrage, comme les estimations réelles
        h = h.with_columns(pl.col("alpha") - pl.col("alpha").mean().over("bloc"))
        h = h.with_columns(pl.col("alpha") - pl.col("alpha").mean().over("pollster"))
        out[e] = (pl.DataFrame(err_rows), h)
    return out


def test_precision_factor_ranks_pollsters_and_is_bounded():
    r = pr.PollsterRatings.fit((1, 2, 3, 4), exclude=(), tables=toy_tables())
    q = dict(zip(["a", "b", "c", "inconnu"], r.precision_factor(["a", "b", "c", "inconnu"])))
    assert q["a"] > 1.2 and q["b"] < 0.8  # précis pèse plus, imprécis moins
    assert all(pr.PRECISION_BOUNDS[0] <= v <= pr.PRECISION_BOUNDS[1] for v in q.values())
    assert q["inconnu"] == 1.0
    mu, sd = r.house_prior(["inconnu"], BLOCS)
    assert np.all(mu == 0) and np.all(sd > 0)


def test_persistent_house_effect_is_learned_and_noise_is_shrunk_to_zero():
    persistent = {("a", "left"): 0.10, ("a", "right"): -0.10}
    r = pr.PollsterRatings.fit(
        (1, 2, 3, 4), exclude=(), tables=toy_tables(house=persistent, election_noise=0.02)
    )
    mu, _sd = r.house_prior(["a", "b"], BLOCS)
    assert mu[0, BLOCS.index("left")] > 0.04 and mu[0, BLOCS.index("right")] < -0.04
    # effets sans persistance : rétrécis vers 0, pas recopiés
    ten = {f"p{i}": 1.0 for i in range(10)}
    noise_only = pr.PollsterRatings.fit(
        (1, 2, 3, 4), exclude=(), tables=toy_tables(noise_sd=ten, election_noise=0.05, seed=3)
    )
    assert noise_only.house_sm < noise_only.house_se / 2
    raw = noise_only.house["raw_mean"].abs().mean()
    assert noise_only.house["mu"].abs().mean() < 0.5 * raw


def test_shrinkage_grows_with_fewer_elections_and_polls():
    y = np.array([0.08])
    few, _ = pr.shrunk_mean(y, np.array([0.05**2]), sm2=0.03**2, se2=0.03**2)
    many, _ = pr.shrunk_mean(np.repeat(y, 4), np.repeat(0.01**2, 4), sm2=0.03**2, se2=0.03**2)
    assert 0 < few < many < 0.08


def test_successions_pool_renames_but_not_absorptions():
    assert pr.lineage("sofres") == pr.lineage("kantar") == pr.lineage("verian") == "verian"
    assert pr.lineage("bva") == "bva" and pr.lineage("ipsos") == "ipsos"
    t = toy_tables(elections=(1, 2), noise_sd={"sofres": 0.6, "ifop": 1.0, "bva": 1.0},
                   house={("sofres", "left"): 0.1})
    t2 = toy_tables(elections=(3,), noise_sd={"kantar": 0.6, "ifop": 1.0, "ipsos": 1.0},
                    house={("kantar", "left"): 0.1}, seed=1)
    r = pr.PollsterRatings.fit((1, 2, 3), exclude=(), tables=t | t2)
    row = r.house.filter((pl.col("lineage") == "verian") & (pl.col("bloc") == "left"))
    assert row["n_elections"].item() == 3
    # un sondage « verian » de 2027 hérite de l'historique Sofres puis Kantar
    q = r.precision_factor(["verian", "kantar", "ipsos"])
    assert q[0] == q[1] and q[0] > 1
    assert "bva" in r.precision["lineage"].to_list()  # BVA garde son propre historique


def test_leave_one_out_ignores_the_excluded_election():
    t = toy_tables(elections=(1, 2, 3))
    absurd = toy_tables(elections=(4,), noise_sd={"a": 50.0, "b": 0.01, "c": 1.0, "d": 1.0},
                        house={("a", "left"): 3.0})
    full = pr.PollsterRatings.fit((1, 2, 3, 4), exclude=(4,), tables=t | absurd)
    ref = pr.PollsterRatings.fit((1, 2, 3), exclude=(), tables=t)
    assert full.excluded == (4,) and full.elections == (1, 2, 3)
    # égalité aux arrondis près (les agrégations parallèles de polars ne sont pas bit à bit)
    assert_frame_equal(full.precision, ref.precision, rtol=1e-6)
    assert_frame_equal(full.house, ref.house, rtol=1e-6)
    with pytest.raises(TypeError):  # l'exclusion est obligatoire
        pr.PollsterRatings.fit((1, 2, 3), tables=t)


def test_real_ratings_exclude_the_evaluated_election():
    """Instituts présents seulement en 2022 : absents des notes calibrées sans 2022."""
    r = pr.PollsterRatings.fit(exclude=(2022,))
    assert 2022 not in r.elections
    lineages = set(r.house["lineage"].to_list()) | set(r.precision["lineage"].to_list())
    assert not {"cluster17", "atlasintel", "yougov"} & lineages
    q = r.precision_factor(["cluster17", "ifop"])
    assert q[0] == 1.0 and 0.5 <= q[1] <= 1.5


# --- branchement dans le modèle


def test_option_disabled_by_default_reduces_to_current_model():
    assert fr.Priors().pollster_ratings is False
    assert pr.ratings_inputs(object(), fr.Priors()) == {}
    assert pr.ratings_inputs(object(), None) == {}


@pytest.mark.parametrize("n_max", [None, 2500.0])
def test_rated_concentration_with_unit_factor_matches_concentration(n_max):
    f, n = jnp.asarray(0.7), jnp.asarray([600.0, 1000.0, 3000.0])
    base = fr.concentration(f, n, None if n_max is None else jnp.asarray(n_max))
    rated = pr.rated_concentration(f, n, None if n_max is None else jnp.asarray(n_max),
                                   jnp.ones(3))
    assert np.asarray(rated) == pytest.approx(np.asarray(base), rel=1e-5)


def test_rated_concentration_scales_effective_size():
    """q multiplie la précision de l'erreur excédentaire : 1/n_eff = 1/n + (1/n_eff° − 1/n)/q."""
    f, n, n_max, q = 0.6, 1000.0, 2500.0, 1.3
    kappa = float(pr.rated_concentration(jnp.asarray(f), jnp.asarray(n), jnp.asarray(n_max),
                                         jnp.asarray(q)))
    n_eff = 1 / (1 / n + (1 / (f * n) + 1 / n_max - 1 / n) / q)
    p = 0.2
    var_share = p * (1 - p) / n * (n + kappa) / (1 + kappa)
    assert var_share == pytest.approx(p * (1 - p) / n_eff, rel=1e-4)
    # lisse et valide même pour un facteur de design proche de 1 et q au maximum
    k = pr.rated_concentration(jnp.asarray(0.95), jnp.asarray([300.0, 1000.0]),
                               jnp.asarray(9000.0), jnp.asarray(1.5))
    assert np.all(np.isfinite(np.asarray(k))) and np.all(np.asarray(k) > 1)


def test_rated_house_bloc_with_flat_prior_matches_current_house_effects():
    class Inp:
        rating_mu = jnp.zeros((4, 3))
        rating_sd = jnp.full((4, 3), 0.12)

    z = jnp.asarray(np.random.default_rng(0).normal(size=(4, 3)))
    current = 0.12 * fr._center(fr._center(z, 0), 1)
    assert np.asarray(pr.rated_house_bloc(z, Inp)) == pytest.approx(np.asarray(current), rel=1e-5)
    # une moyenne apprise est recentrée sur les instituts présents et sur les blocs
    Inp.rating_mu = jnp.asarray(np.random.default_rng(1).normal(size=(4, 3)))
    hb = np.asarray(pr.rated_house_bloc(jnp.zeros((4, 3)), Inp))
    assert hb.sum(0) == pytest.approx(np.zeros(3), abs=1e-6)
    assert hb.sum(1) == pytest.approx(np.zeros(4), abs=1e-6)
