"""Certitude du choix : extraction des notices, alignement, mise à l'échelle, option désactivée.

Deux pages de notices figées dans tests/fixtures/certainty/ (extraites avec pdfseparate) :
Ipsos pour France Info / Le Parisien, 5 avril 2022, p. 17 (format complément) ; BVA pour RTL
et Orange, décembre 2021, p. 10 (format « électorat potentiel », valeurs renvoyées sur la ligne
au-dessus du nom par la mise en page).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from vvs.ingest import certainty as ic
from vvs.ingest.commission import certainty as ce
from vvs.ingest.commission.notice import extract_text
from vvs.model import certainty as mc
from vvs.model import error as er
from vvs.model import first_round as fr

FIXTURES = Path(__file__).parent / "fixtures" / "certainty"

LABELS = {
    "melenchon_jean_luc": ["Jean-Luc Mélenchon"],
    "hidalgo_anne": ["Anne Hidalgo"],
    "jadot_yannick": ["Yannick Jadot"],
    "macron_emmanuel": ["Emmanuel Macron"],
    "pecresse_valerie": ["Valérie Pécresse"],
    "zemmour_eric": ["Éric Zemmour"],
    "le_pen_marine": ["Marine Le Pen"],
    "roussel_fabien": ["Fabien Roussel"],
    "montebourg_arnaud": ["Arnaud Montebourg"],
    "poutou_philippe": ["Philippe Poutou"],
    "fillon_francois": ["François Fillon"],
    "hamon_benoit": ["Benoît Hamon"],
}
LOOKUP = ce.candidate_lookup(LABELS)


def _rows(path: Path) -> dict[str, float]:
    pages = extract_text(path, column_pages=0).pages
    rows, _ = ce.merge_tables(ce.find_certainty_tables(pages, LOOKUP))
    return {c: r.certain_share for c, r in rows.items()}


# --- extraction sur notices figées


def test_ipsos_complement_table():
    assert _rows(FIXTURES / "ipsos_1486_p17.pdf") == {
        "melenchon_jean_luc": 76.0,
        "jadot_yannick": 60.0,
        "macron_emmanuel": 84.0,
        "pecresse_valerie": 63.0,
        "le_pen_marine": 83.0,
        "zemmour_eric": 70.0,
    }


def test_bva_potential_table_with_shifted_values():
    # Roussel, Montebourg, Poutou : « * » (effectif trop faible), pas de valeur ;
    # Macron et Hidalgo : valeurs sur la ligne au-dessus du nom ; fourchettes ignorées.
    assert _rows(FIXTURES / "bva_1187_p10.pdf") == {
        "melenchon_jean_luc": 74.0,
        "hidalgo_anne": 51.0,
        "jadot_yannick": 48.0,
        "macron_emmanuel": 73.0,
        "pecresse_valerie": 60.0,
        "zemmour_eric": 65.0,
        "le_pen_marine": 74.0,
    }


# --- formats sur texte


def _page(text: str) -> ce.CertaintyTable | None:
    return ce.parse_certainty_page(text, 1, LOOKUP)


def test_direct_share_table_and_total():
    t = _page(
        "La sûreté de choix au 1er tour de l'élection présidentielle\n"
        "Pour chaque candidat, part des électeurs se disant sûrs de leur choix :\n"
        "Ensemble\n65 +5\nJean-Luc Mélenchon\n66 +5\nBenoît Hamon\n56 +8\n"
        "Emmanuel Macron\n63 +8\nFrançois Fillon\n80 +5\nMarine Le Pen\n78 +2\n"
    )
    assert t.fmt == "direct" and t.total == 65
    assert {r.candidate_id: r.certain_share for r in t.rows} == {
        "melenchon_jean_luc": 66, "hamon_benoit": 56, "macron_emmanuel": 63,
        "fillon_francois": 80, "le_pen_marine": 78,
    }


def test_reversed_legend_and_second_round_page():
    t = _page(
        "Sûreté du choix\nPEUT ENCORE CHANGER IL EST DÉFINITIF\n"
        "Emmanuel Macron 20 80\nMarine Le Pen 15 85\nJean-Luc Mélenchon 30 70\n"
    )
    assert [r.certain_share for r in t.rows] == [80, 85, 70]
    assert _page(
        "Sûreté du choix au second tour\nIL EST DÉFINITIF IL PEUT ENCORE CHANGER\n"
        "Emmanuel Macron 80 20\nMarine Le Pen 85 15\n"
    ) is None


def test_vote_intention_table_is_not_certainty():
    # page qui pose la question mais dont le tableau est une intention de vote (somme 100)
    t = _page(
        "Etes-vous sûr(e) de votre choix ou pourriez-vous changer d'avis ?\n"
        "Emmanuel Macron 27\nMarine Le Pen 22\nJean-Luc Mélenchon 15\nÉric Zemmour 12\n"
        "Valérie Pécresse 11\nYannick Jadot 8\nAnne Hidalgo 5\n"
        "Vous êtes sûr(e) de votre choix 76,6 75,8\n"
    )
    assert t.fmt == "total" and t.rows == [] and t.total == 75.8


# --- alignement sur polls.parquet


def _notice(fid, start, end, pollster="ipsos"):
    n = ic.NoticeCertainty(fid, 2022, {"url": f"u{fid}", "title": "t"})
    n.pollster, n.fieldwork_start, n.fieldwork_end = pollster, start, end
    n.rows = {"macron_emmanuel": ce.CertaintyRow("macron_emmanuel", 80.0, "complement", 1)}
    return n


def test_alignment_is_one_to_one_and_falls_back_to_own_id():
    d = dt.date
    polls = pl.DataFrame(
        {
            "poll_id": ["2022-ipsos-20220316-1", "2022-ipsos-20220201-1"],
            "election": [2022, 2022],
            "pollster": ["ipsos", "ipsos"],
            "fieldwork_start": [d(2022, 3, 13), d(2022, 1, 28)],
            "fieldwork_end": [d(2022, 3, 16), d(2022, 2, 1)],
        }
    )
    # deux vagues quotidiennes de rolling compatibles avec le même sondage : la plus proche
    # le prend, l'autre garde un id propre ; une notice sans sondage garde un id propre
    notices = [
        _notice(1, d(2022, 3, 12), d(2022, 3, 15)),
        _notice(2, d(2022, 3, 13), d(2022, 3, 16)),
        _notice(3, d(2022, 3, 1), d(2022, 3, 3)),
        _notice(4, d(2022, 3, 13), d(2022, 3, 16)),  # rectificative de la notice 2
    ]
    t = ic.build_table(notices, polls)
    ids = dict(zip(t["file_id"].to_list(), t["poll_id"].to_list()))
    assert ids == {
        1: "2022-ipsos-20220315-cert1",
        2: "2022-ipsos-20220316-1",
        3: "2022-ipsos-20220303-cert3",
    }
    assert t.schema == ic.SCHEMA
    assert t.filter(pl.col("file_id") == 2)["aligned"].item()


# --- mesure et mise à l'échelle


def test_two_way_effects_remove_pollster_levels():
    # deux instituts sur des bases différentes (+10 points pour le second)
    b = {"a": 10.0, "b": 0.0, "c": -10.0}
    rows = [
        {"poll_id": p, "candidate_id": c, "certain_share": 70 + v + off}
        for p, off in (("p1", 0.0), ("p2", 10.0), ("p3", 10.0))
        for c, v in b.items()
        if not (p == "p3" and c == "c")
    ]
    est = mc.two_way_effects(pl.DataFrame(rows))
    assert est == pytest.approx(b, abs=1e-6)


def test_effects_window_and_scale():
    d = dt.date
    df = pl.DataFrame(
        {
            "poll_id": ["p1", "p1", "p2", "p2"],
            "election": [2022] * 4,
            "candidate_id": ["a", "b", "a", "b"],
            "certain_share": [80.0, 60.0, 50.0, 90.0],
            "fieldwork_end": [d(2022, 3, 1), d(2022, 3, 1), d(2022, 4, 5), d(2022, 4, 5)],
        }
    )
    x = mc.effects(2022, d(2022, 3, 10), df)  # p2 postérieur à asof : ignoré
    assert x == pytest.approx({"a": 0.10, "b": -0.10})
    s = mc.scales_for(["a", "b", "z"], x, kappa=2.0)
    assert s == pytest.approx([np.exp(-0.2), np.exp(0.2), 1.0])
    assert mc.scale(10.0, 5.0) == pytest.approx(mc.SCALE_BOUNDS[0])
    assert mc.scales_for(["a", "b"], x, kappa=0.0) == pytest.approx([1.0, 1.0])


def test_kappa_fit_recovers_sign_on_toy_cells():
    rng = np.random.default_rng(0)
    cells = []
    for k in range(40):
        x = rng.uniform(-0.15, 0.15, 8)
        x -= x.mean()
        vb, vc = np.full(8, 0.002), np.full(8, 0.02)
        sd = np.sqrt(vb + vc * np.exp(-2 * 4.0 * x))
        cells.append(mc.Cell(2099, k, [str(i) for i in range(8)], x, rng.normal(0, sd), vb, vc))
    kappa, gain, _ = mc.fit_kappa_cells(cells)
    assert 1.5 < kappa < 7 and gain > 0


# --- réduction au comportement actuel


def _toy_inputs(priors):
    from test_model_first_round import synthetic

    data, _ = synthetic(n_hyp=30, T=6)
    return data, fr.ModelInputs.from_data(data, priors, centered=False)


def _sites(inp, seed=3):
    rng = np.random.default_rng(seed)
    B, C, T, P = inp.B, inp.C, inp.T, inp.P
    return {
        "sigma_beta": 0.05, "sigma_delta": 0.07, "sigma_house_bloc": 0.1,
        "sigma_house_cand": 0.03,
        "z_beta0": jnp.asarray(rng.normal(size=B)),
        "z_beta": jnp.asarray(rng.normal(size=(T - 1, B))),
        "z_delta0": jnp.asarray(rng.normal(size=C)),
        "z_delta": jnp.asarray(rng.normal(size=(T - 1, C))),
        "z_house_bloc": jnp.asarray(rng.normal(size=(P, B))),
        "z_house_cand": jnp.asarray(rng.normal(size=(P, C))),
    }


def test_first_round_option_off_or_neutral_reduces_to_current(monkeypatch):
    assert fr.Priors().certainty is False
    data, base = _toy_inputs(fr.Priors(n_max=None))
    assert base.delta_scale is None
    s = _sites(base)
    ref = np.asarray(fr.latent(s, base)["theta"])
    # option active mais échelle neutre (aucune mesure de certitude) : mêmes latents
    monkeypatch.setattr(mc, "effects", lambda *a, **k: {})
    _, on = _toy_inputs(fr.Priors(n_max=None, certainty=True, certainty_kappa=3.0))
    assert np.asarray(on.delta_scale) == pytest.approx(np.ones(len(data.candidates)))
    assert np.asarray(fr.latent(s, on)["theta"]) == pytest.approx(ref, abs=1e-6)
    # échelle non neutre : la marche du candidat le moins sûr bouge davantage
    monkeypatch.setattr(mc, "effects", lambda *a, **k: {"a_fr": -0.2, "b_fr": 0.2})
    _, sc = _toy_inputs(fr.Priors(n_max=None, certainty=True, certainty_kappa=3.0))
    d0 = np.asarray(fr.latent(s, base)["delta"])
    d1 = np.asarray(fr.latent(s, sc)["delta"])
    assert np.abs(np.diff(d1[:, 0])).sum() > np.abs(np.diff(d0[:, 0])).sum()


def _eparams():
    return er.ErrorParams(
        sigma_bloc_final=0.06, sigma_cand_final=0.08, rho=1e-3, a_bloc=0.05, a_cand=0.04,
        gamma=0.5, nu=8.0, final_age=7.0,
    )


def test_error_model_scaler_off_or_neutral_is_identical():
    blocs = ["far_right", "right", "centre", "left"]
    theta = np.log(np.array([[0.3, 0.2, 0.3, 0.2]]))
    cands = ["a", "b", "c", "d"]
    m = er.ErrorModel(_eparams())
    assert m.cand_scale is None
    ref = m.sample(theta, 90, cands, blocs, np.random.default_rng(5), n=2000)
    m.cand_scale = lambda cs: np.ones(len(cs))
    same = m.sample(theta, 90, cands, blocs, np.random.default_rng(5), n=2000)
    assert same == pytest.approx(ref)
    m.cand_scale = lambda cs: np.array([2.0, 1.0, 1.0, 1.0])
    wide = m.sample(theta, 90, cands, blocs, np.random.default_rng(5), n=20_000)
    assert wide[:, 0].std() > 1.3 * m.__class__(_eparams()).sample(
        theta, 90, cands, blocs, np.random.default_rng(5), n=20_000
    )[:, 0].std()


def test_certainty_variant_registered():
    from vvs.eval.backtest import VARIANTS

    assert VARIANTS["certainty"].certainty is True
    assert VARIANTS["floor"].certainty is False


def test_ifop_rolling_report_layouts():
    names = {
        "melenchon_jean_luc": ("Jean-Luc", "Mélenchon"),
        "pecresse_valerie": ("Valérie", "Pécresse"),
        "le_pen_marine": ("Marine", "Le Pen"),
        "zemmour_eric": ("Éric", "Zemmour"),
    }
    head = "La sûreté du choix au 1er tour de l’élection présidentielle\nSûreté\n"
    # mars 2022 : prénom seul, valeur sur la ligne suivante, nom repoussé par la légende
    march = head + (
        "Jean-Luc\n79 %\nMÉLENCHON\nValérie\n65 %\nddeess éélleecctteeuurrss\nPÉCRESSE\n"
        "Marine\n82 %\nddee lleeuurr cchhooiixx LE PEN\n"
    )
    assert ic.parse_ifop_rolling_page(march, names) == {
        "melenchon_jean_luc": 79.0, "pecresse_valerie": 65.0, "le_pen_marine": 82.0,
    }
    # janvier 2022 : « Prénom NN % » sur une ligne, parfois en fin de légende
    january = head + (
        "Jean-Luc 69 %\nMÉLENCHON\nMarine 82 %\nLE PEN\n"
        "PPeeuuvveenntt cchhaannggeerr :: % (−2) Éric 70 %\nZEMMOUR\n"
    )
    assert ic.parse_ifop_rolling_page(january, names) == {
        "melenchon_jean_luc": 69.0, "le_pen_marine": 82.0, "zemmour_eric": 70.0,
    }
    # page d'évolution par électorat : ignorée
    assert ic.parse_ifop_rolling_page(head.replace("présidentielle", "présidentielle "
                                      "(évolution par électorat)") + "Marine 82 %\nLE PEN\n",
                                      names) == {}
