"""Potentiel électoral : parseurs sur extraits figés, vérification des relevés, normalisation
par institut (vvs.model.potential) et plafonds du second tour (option use_potential)."""

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from vvs.ingest.potential import (
    PotentialRow,
    _pair_numbers,
    _segments,
    _squash,
    parse_zero_ten_lines,
    verify_row,
)
from vvs.model import potential as P
from vvs.model.second_round import ceiling_log_weight, sample_pair

FIXTURES = Path(__file__).parent / "fixtures" / "potential"


# --- parseurs


def test_zero_ten_lines_frozen_extract():
    lines = (FIXTURES / "harris_zero_ten_2026-09-24.txt").read_text(encoding="utf-8").splitlines()
    rows, warnings = parse_zero_ten_lines(lines)
    by = {r.label: r for r in rows}
    assert set(by) == {
        "Marine Le Pen", "Édouard Philippe", "Bruno Retailleau", "Raphaël Glucksmann", "Gabriel Attal"
    }
    # note moyenne renvoyée à la ligne suivante (Le Pen) : la ligne reste lue
    lp = by["Marine Le Pen"]
    assert (lp.potential, lp.reject, lp.firm) == (39.0, 50.0, 29.0)
    assert (by["Gabriel Attal"].potential, by["Gabriel Attal"].reject) == (14.0, 66.0)
    assert warnings == []
    rows, warnings = parse_zero_ten_lines(["Marine Le Pen 50 11 10 19 4,4"])
    assert rows == [] and "somme 90" in warnings[0]


def _word(text, x0, top, w=None):
    w = w or 6.0 * len(text)
    return {"text": text, "x0": x0, "x1": x0 + w, "top": top, "bottom": top + 12}


def test_binary_layout_pairs_numbers_with_names():
    # trois colonnes « nom … barre … valeur » ; le second nombre est 2 pt plus bas que son nom
    words = [
        _word("Marine", 100, 113), _word("Le", 140, 113), _word("Pen", 155, 113), _word("40", 266, 113),
        _word("Jean-Luc", 373, 113), _word("Mélenchon", 425, 113), _word("15", 538, 115),
        _word("François", 400, 290), _word("Baroin", 455, 290), _word("13", 538, 296),
    ]
    segs, nums = _segments(words)
    pairs = dict(_pair_numbers(segs, nums))
    assert pairs == {"Marine Le Pen": 40, "Jean-Luc Mélenchon": 15, "François Baroin": 13}


# --- vérification des relevés


TEXT = _squash(
    "Marine Le Pen 60,1% 8,4% 31,5% 59,2% 8,6% 32,2%\n"
    "Jordan Bardella 61,1% 8,1% 30,8% 60,1% 7,9% 32,0%\n"
    "Marine Le Pen\nVous êtes sûr(e) de voter pour lui/elle 9% 7%\n"
    "Vous voterez peut-être pour lui/elle 28% 26%\n"
    "un potentiel de vote de 56%, dont 34% de Français"
)


@pytest.mark.parametrize(
    "row, ok",
    [
        (PotentialRow(candidate="x", potential=32.2, reject=59.2,
                      quote="Marine Le Pen 60,1% 8,4% 31,5% 59,2% 8,6% 32,2%"), True),
        # « 32,0 » publié, 32 relevé
        (PotentialRow(candidate="x", potential=32, reject=60.1,
                      quote="Jordan Bardella 61,1% 8,1% 30,8% 60,1% 7,9% 32,0%"), True),
        # chiffre absent de l'extrait
        (PotentialRow(candidate="x", potential=33, quote="Marine Le Pen 60,1% 8,4% 31,5%"), False),
        # extrait absent du brut
        (PotentialRow(candidate="x", potential=31.5, quote="Marine Le Pen 31,5% 99%"), False),
        # somme de modalités : on cherche les modalités, pas le total
        (PotentialRow(candidate="x", potential=33, potential_parts=[7, 26],
                      quote="Marine Le Pen Vous êtes sûr(e) de voter pour lui/elle 9% 7% "
                            "Vous voterez peut-être pour lui/elle 28% 26%"), True),
        # extraits multiples séparés par « ; »
        (PotentialRow(candidate="x", potential=56, firm=34,
                      quote="Vous voterez peut-être pour lui/elle 28% ; un potentiel de vote de 56%, dont 34%"), True),
    ],
)
def test_verify_row(row, ok):
    assert verify_row(row, TEXT) is ok


def test_potential_parts_must_sum():
    with pytest.raises(ValueError):
        PotentialRow(candidate="x", potential=34, potential_parts=[7, 26])


# --- normalisation par institut


def _obs(measure, scale, cand, day, p, election=2027, reject=None):
    return {
        "election": election, "candidate_id": cand, "measure": measure, "scale": scale,
        "electorate": "all", "date": dt.date(2026, 1, 1) + dt.timedelta(days=day),
        "potential_share": p, "reject_share": reject,
    }


def _toy() -> pl.DataFrame:
    """Référence `five_level` ; `zero_ten` de deux instituts : A décalé de −0,5 logit, B de
    −0,5 − 0,3 ; B ne recoupe que A ; `four_level` sans aucun recoupement."""
    rows = []
    base = {"c1": 45.0, "c2": 30.0, "c3": 20.0, "c4": 38.0, "c5": 12.0}
    for k, (c, p) in enumerate(base.items()):
        lp = P.logit(p)
        rows.append(_obs("kantar_five_level", "five_level", c, 0, p))
        rows.append(_obs("ipsos_zero_ten", "zero_ten", c, 5, float(P.expit(lp - 0.5))))
        rows.append(_obs("harris_zero_ten", "zero_ten", c, 60, float(P.expit(lp - 0.8))))
        rows.append(_obs("ipsos_zero_ten", "zero_ten", c, 62, float(P.expit(lp - 0.5))))
    rows.append(_obs("elabe_four_level", "four_level", "c1", 200, 50.0))
    return pl.DataFrame(rows)


def test_offsets_recover_scale_and_house_effects():
    off = P.fit_offsets(_toy(), "potential_share")
    assert off.by_measure["kantar_five_level"] == pytest.approx(0.0, abs=1e-3)
    assert off.by_measure["ipsos_zero_ten"] == pytest.approx(-0.5, abs=0.05)
    assert off.by_measure["harris_zero_ten"] == pytest.approx(-0.8, abs=0.05)
    # sans recoupement : décalage de son échelle (ici inconnue, donc 0)
    assert off.by_measure["elabe_four_level"] == pytest.approx(0.0, abs=1e-6)
    assert off.n_pairs["elabe_four_level"] == 0


def test_normalize_maps_measures_to_reference():
    df = _toy()
    n = P.normalize(df.with_columns(pl.lit(None, pl.Float64).alias("reject_share")))
    ref = n.filter(pl.col("measure") == "kantar_five_level").sort("candidate_id")
    for m in ("ipsos_zero_ten", "harris_zero_ten"):
        x = n.filter(pl.col("measure") == m).group_by("candidate_id").agg(pl.col("potential_norm").mean())
        x = x.sort("candidate_id")
        np.testing.assert_allclose(x["potential_norm"], ref["potential_share"], atol=1.0)
    assert n["reject_norm"].null_count() == n.height


def test_summary_at_uses_window_before_asof():
    df = P.normalize(_toy())
    s = P.summary_at(df, 2027, "c1", dt.date(2026, 1, 3))
    assert s.n == 1 and s.potential == pytest.approx(45.0) and s.reject is None
    assert P.summary_at(df, 2027, "c1", dt.date(2025, 12, 1)) is None


# --- plafonds du second tour


def _toy_fit(ceilings=(None, None)):
    from vvs.model.second_round import ABSTAIN, DuelPolls, ErrorParams, GroupPrior, PairFit

    groups = {
        "own:a": GroupPrior("own:a", (0.95, 0.02, 0.03), 200.0, False, "own"),
        "own:b": GroupPrior("own:b", (0.02, 0.95, 0.03), 200.0, False, "own"),
        "bloc:left": GroupPrior("bloc:left", (0.5, 0.2, 0.3), 6.0, True, "proximity"),
        ABSTAIN: GroupPrior(ABSTAIN, (0.2, 0.2, 0.6), 6.0, True, "proximity"),
    }
    return PairFit(
        election=2027, asof=dt.date(2027, 4, 1), pair=("a", "b"), blocs=("far_right", "left"),
        groups=groups, sigma_u=0.6, sigma_v=0.4, polls=DuelPolls.empty(),
        errors=ErrorParams(sigma0=2.0, sigma1=1.0), gap_days=14, abstain_weight=0.3,
        bloc_of=lambda c: {"a": "far_right", "b": "left", "l": "left"}[c],
        ceilings=ceilings, ceiling_tau=0.66,
    )


FIRST = {"a": np.array([0.35]), "b": np.array([0.35]), "l": np.array([0.30])}


def test_ceiling_pulls_share_below_converted_ceiling():
    free = sample_pair(_toy_fit(), FIRST, 10, np.random.default_rng(0), n=4000)
    capped = sample_pair(_toy_fit((33.0, None)), FIRST, 10, np.random.default_rng(0), n=4000)
    cap = 33.0 / 0.66  # 50 % des exprimés
    assert np.median(free.share) > cap
    assert np.median(capped.share) < np.median(free.share)
    assert np.mean(capped.share > cap + 5) < 0.5 * np.mean(free.share > cap + 5)
    # plafond de j : symétrique, pousse i vers le haut
    capped_j = sample_pair(_toy_fit((None, 25.0)), FIRST, 10, np.random.default_rng(0), n=4000)
    assert np.median(capped_j.share) > np.median(free.share)


def test_option_off_is_current_behaviour():
    # sans plafond, aucun poids ajouté : mêmes tirages, à la graine près
    assert ceiling_log_weight((None, None), np.array([40.0, 60.0]), 0.66) == 0.0
    a = sample_pair(_toy_fit(), FIRST, 10, np.random.default_rng(5), n=2000)
    b = sample_pair(_toy_fit((None, None)), FIRST, 10, np.random.default_rng(5), n=2000)
    np.testing.assert_array_equal(a.share, b.share)
    assert _toy_fit((30.0, 50.0)).swapped().ceilings == (50.0, 30.0)


def test_model_default_has_no_potential():
    from vvs.model.second_round import SecondRoundModel

    m = SecondRoundModel()
    assert m.use_potential is False and m.potential is None
