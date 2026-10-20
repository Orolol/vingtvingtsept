import numpy as np
import polars as pl
import pytest
from toy_run import ASOF_WEEK, FIT_CANDIDATES, run_toy

from vvs.model.simulate import (
    QNAMES,
    allocate,
    iia_shares,
    insert_prior_shares,
    rn_bias_scale,
)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return run_toy(tmp_path_factory.mktemp("toy"), n=3000)


def test_allocate_is_proportional_and_exact():
    k = allocate(np.array([0.5, 0.3, 0.2 - 1e-6, 1e-6]), 1000)
    assert list(k) == [500, 300, 200, 0]
    k = allocate(np.array([1, 1, 1]), 1000)
    assert k.sum() == 1000 and k.max() - k.min() == 1


@pytest.mark.parametrize(("h", "scale"), [(0, 1.0), (30, 1.0), (105, 0.5), (180, 0.0), (400, 0.0)])
def test_rn_bias_scale(h, scale):
    assert rn_bias_scale(h) == pytest.approx(scale)


def test_iia_shares_restricted_to_present():
    x = np.array([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    present = np.array([[True, True, False], [True, True, True]])
    s = iia_shares(x, present, ["a", "b", "c"])
    np.testing.assert_allclose(s.sum(-1), 1.0)
    assert s[0, 2] == 0
    # IIA : le rapport des deux premiers ne dépend pas de la présence du troisième
    np.testing.assert_allclose(s[0, 0] / s[0, 1], s[1, 0] / s[1, 1])


def test_insert_prior_shares_keeps_ratios():
    from vvs.model.simulate import Universe

    uni = Universe(["a", "b", "p"], ["left", "right", "other"], ["posterior", "posterior", "prior"],
                   [0, 1, None], [None, None, 4.0], [None, None, "test"])
    n = 20_000
    theta = np.tile(np.log([0.6, 0.4, np.nan]), (n, 1))
    present = np.ones((n, 3), dtype=bool)
    present[: n // 2, 2] = False
    out = insert_prior_shares(theta, present, uni, np.random.default_rng(0))
    s = iia_shares(np.where(present, out, 0.0), present, uni.blocs)
    on = present[:, 2]
    assert np.median(100 * s[on, 2]) == pytest.approx(4.0, rel=0.05)
    np.testing.assert_allclose(s[:, 0] / s[:, 1], 1.5)
    assert np.all(s[~on, 2] == 0)


def test_draws_are_coherent(toy):
    d = toy.draws
    cands = [u["candidate_id"] for u in toy.manifest["universe"]]
    shares = d.select(cands).fill_null(0.0).to_numpy()
    np.testing.assert_allclose(shares.sum(1), 100.0, atol=1e-6)
    top2 = np.argsort(-shares, axis=1)[:, :2]
    ids = np.array(cands)
    qa, qb = d["qualified_a"].to_numpy(), d["qualified_b"].to_numpy()
    assert all(sorted(t) == sorted(q) for t, q in zip(ids[top2].tolist(), zip(qa, qb)))
    w = d["winner"].to_numpy()
    assert np.all((w == qa) | (w == qb))
    assert np.all((d["share_a_second_round"].to_numpy() > 50) == (w == qa))
    assert d["turnout"].is_between(0.4, 0.95).all()
    # un placeholder n'est présent que dans l'état qui le porte
    assert d.filter(pl.col("placeholder_centre").is_not_null())["config"].n_unique() > 0


def test_summary_invariants(toy):
    s = toy.summary
    agg = s.group_by("condition_id").agg(pl.col("p_win").sum(), pl.col("p_qualify").sum())
    np.testing.assert_allclose(agg["p_win"].to_numpy(), 1.0, atol=1e-9)
    np.testing.assert_allclose(agg["p_qualify"].to_numpy(), 2.0, atol=1e-9)
    assert (s["p_win"] <= s["p_qualify"] + 1e-12).all()
    assert (s["p_qualify"] <= s["p_present"] + 1e-12).all()
    q = s.select(QNAMES).to_numpy()
    assert np.all(np.diff(q, axis=1) >= -1e-9)
    # présence certaine des candidats nommés dans leur état
    row = s.filter((pl.col("condition_id") == "rn.bardella") & (pl.col("candidate_id") == "bardella_jordan"))
    assert row["p_present"][0] == 1.0
    assert s.filter((pl.col("condition_id") == "rn.bardella") & (pl.col("candidate_id") == "le_pen_marine")).is_empty()


@pytest.mark.parametrize("dim", ["rn", "centre", "lr"])
def test_mixture_is_weighted_sum_of_states(toy, dim):
    s = toy.summary
    conds = [c for c in toy.conditions if c.kind == "dimension_state" and dim in c.states]
    n_mix = s.filter(pl.col("condition_id") == "mix")["n_draws"][0]
    for metric in ("p_present", "p_qualify", "p_win"):
        mix = dict(s.filter(pl.col("condition_id") == "mix").select("candidate_id", metric).iter_rows())
        total: dict[str, float] = {}
        for c in conds:
            sub = s.filter(pl.col("condition_id") == c.id)
            w = sub["n_draws"][0] / n_mix
            # les fréquences empiriques suivent les probabilités des états (allocation)
            assert w == pytest.approx(c.probability, abs=2e-3)
            for cid, v in sub.select("candidate_id", metric).iter_rows():
                total[cid] = total.get(cid, 0.0) + w * v
        for cid, v in mix.items():
            assert total.get(cid, 0.0) == pytest.approx(v, abs=1e-9)


def test_pairs_and_transfers(toy):
    mix = toy.pairs.filter(pl.col("condition_id") == "mix")
    assert mix["p_pair"].sum() == pytest.approx(1.0)
    assert mix["p_a_wins"].is_between(0, 1).all()
    # le finaliste d'extrême droite est toujours candidat a
    far = {c for c, (b, *_) in FIT_CANDIDATES.items() if b == "far_right"}
    for a, b in mix.select("candidate_a", "candidate_b").iter_rows():
        assert not (b in far and a not in far)
    pairs = set(mix.select("candidate_a", "candidate_b").iter_rows())
    assert set(toy.transfers.select("candidate_a", "candidate_b").iter_rows()) <= pairs
    for k in ("to_a", "to_b", "none"):
        assert toy.transfers[f"{k}_q50"].is_between(0, 1).all()
    assert toy.manifest["second_round_model"]["rn_bias_scale"] == 0.0


def test_trajectories_null_before_first_poll(toy):
    t = toy.trajectories.filter(pl.col("condition_id") == "mix")
    assert t["week"].max() == ASOF_WEEK
    bard = t.filter(pl.col("candidate_id") == "bardella_jordan").sort("week")
    assert bard["q50"][:2].is_null().all() and bard["q50"][2:].is_not_null().all()
    # candidats sans θ propre : pas de trajectoire
    assert t.filter(pl.col("candidate_id") == "placeholder_centre")["q50"].is_null().all()
    q = t.drop_nulls("q50").select(QNAMES).to_numpy()
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_nested_shares_reduces_to_iia_at_lambda_one():
    import numpy as np

    from vvs.model.simulate import iia_shares, nested_shares

    rng = np.random.default_rng(1)
    n, u, b = 50, 6, 3
    cand_bloc = np.array([0, 0, 1, 1, 2, 2])
    beta = rng.normal(size=(n, b))
    delta = rng.normal(size=(n, u))
    theta = beta[:, cand_bloc] + delta
    present = rng.random((n, u)) > 0.3
    present[:, 0] = True
    ctx = {"beta": beta, "lam": np.ones((n, b)), "cand_bloc": cand_bloc}
    np.testing.assert_allclose(nested_shares(theta, present, [], ctx), iia_shares(theta, present, []), atol=1e-12)


def test_nested_shares_keeps_bloc_share_at_lambda_zero():
    """À λ = 0, retirer un candidat laisse la part de son bloc inchangée."""
    import numpy as np

    from vvs.model.simulate import nested_shares

    cand_bloc = np.array([0, 0, 1, 1])
    beta = np.array([[0.3, -0.3]])
    delta = np.array([[0.5, -0.5, 0.2, -0.2]])
    theta = beta[:, cand_bloc] + delta
    ctx = {"beta": beta, "lam": np.zeros((1, 2)), "cand_bloc": cand_bloc}
    full = nested_shares(theta, np.ones((1, 4), bool), [], ctx)
    reduced = nested_shares(theta, np.array([[True, False, True, True]]), [], ctx)
    assert reduced[0, 1] == 0
    np.testing.assert_allclose(reduced[0, :2].sum(), full[0, :2].sum(), atol=1e-12)
    assert np.isclose(reduced.sum(), 1.0)
