"""Registre des listes (lists.yaml), rattachement des résultats, erreurs des scrutins de liste."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from vvs.curate import lists

TOY_YAML = """
elections:
  leg2024:
    rn_allies_2024:
      label: RN et alliés
      parties: [rn, udr]
      aliases:
        results: [RN, UXD]
    rn_2024:
      label: RN
      parties: [rn]
      group: rn_allies_2024
      aliases:
        wikipedia: [RN]
    uxd_2024:
      label: Ciotti
      parties: [udr]
      group: rn_allies_2024
      aliases:
        wikipedia: [Ciotti]
    nfp_2024:
      label: NFP
      parties: [lfi, ps]
      bloc: far_left
      note: alliance, bloc dominant
      aliases:
        wikipedia: [NFP]
        results: [UG]
    lfi_2024:
      label: LFI seule
      parties: [lfi]
      aliases:
        wikipedia: [LFI]
    ens_2024:
      label: Ensemble
      parties: [renaissance]
      aliases:
        results: [ENS]
    divers_2024:
      label: Divers
      parties: [div]
      residual: true
      aliases:
        results: [DIV, REG]
"""


@pytest.fixture()
def toy(tmp_path):
    path = tmp_path / "lists.yaml"
    path.write_text(TOY_YAML, encoding="utf-8")
    lists.load_lists.cache_clear()
    yield lists.load_lists(path)
    lists.load_lists.cache_clear()


def test_real_registry_loads_with_one_residual_per_election():
    reg = lists.load_lists()
    assert set(reg.elections) == {"euro2014", "euro2019", "euro2024", "leg2022", "leg2024"}
    for key in reg.elections:
        assert reg.bloc_of(key, reg.residual(key)) == "other"
    assert reg.bloc_of("leg2024", "rn_allies_2024") == "far_right"
    assert reg.bloc_of("leg2022", "nupes_2022") == "far_left"
    assert reg.unit_of("leg2024", "uxd_2024") == "rn_allies_2024"
    assert reg.unit_of("euro2019", "other") == "divers_euro2019"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (("      bloc: far_left\n      note: alliance, bloc dominant\n", ""), "blocs différents"),
        (("parties: [rn, udr]", "parties: [rn, inconnu]"), "partis inconnus"),
        (("group: rn_allies_2024\n      aliases:\n        wikipedia: [RN]",
          "group: lfi_2024\n      aliases:\n        wikipedia: [RN]"), "n'est pas une unité"),
        (("results: [ENS]", "results: [ENS, UG]"), "déjà rattaché"),
        (("      residual: true\n", ""), "exactement une unité résiduelle"),
    ],
)
def test_registry_validation_errors(tmp_path, patch, message):
    old, new = patch
    assert old in TOY_YAML
    path = tmp_path / "lists.yaml"
    path.write_text(TOY_YAML.replace(old, new, 1), encoding="utf-8")
    lists.load_lists.cache_clear()
    with pytest.raises(ValueError, match=message):
        lists.load_lists(path)
    lists.load_lists.cache_clear()


def _results(labels_votes: dict[str, int], key="leg2024") -> pl.DataFrame:
    expressed = sum(labels_votes.values())
    return pl.DataFrame(
        {
            "election": key, "round": 1, "geo_level": "national", "geo_code": "FE",
            "label": list(labels_votes), "nuance": list(labels_votes),
            "votes": list(labels_votes.values()), "registered": 2 * expressed,
            "voters": expressed, "expressed": expressed,
        }
    ).with_columns(pl.col("round").cast(pl.Int8))


def test_results_join_aggregates_aliases_and_blocs(toy):
    res = _results({"RN": 300, "UXD": 40, "UG": 280, "ENS": 200, "DIV": 100, "REG": 80})
    out = lists.results_lists_by_bloc({"leg2024": res}, toy)
    units = dict(out.filter(pl.col("level") == "list").select("unit", "share").iter_rows())
    assert units == pytest.approx({"rn_allies_2024": 34.0, "nfp_2024": 28.0, "ens_2024": 20.0, "divers_2024": 18.0})
    blocs = dict(out.filter(pl.col("level") == "bloc").select("unit", "share").iter_rows())
    assert blocs == pytest.approx({"far_right": 34.0, "far_left": 28.0, "centre": 20.0, "other": 18.0})


def test_results_join_fails_on_unknown_or_unused_label(toy):
    with pytest.raises(lists.UnknownListLabelError, match="DVD"):
        lists.results_by_unit(_results({"RN": 1, "UXD": 1, "UG": 1, "ENS": 1, "DIV": 1, "REG": 1, "DVD": 1}), toy, "leg2024")
    with pytest.raises(lists.UnknownListLabelError, match="sans libellé"):
        lists.results_by_unit(_results({"RN": 1, "UXD": 1, "UG": 1, "ENS": 1, "DIV": 1}), toy, "leg2024")


def _polls(rows) -> tuple[pl.DataFrame, pl.DataFrame]:
    """rows : (poll_id, pollster, fin, {hid: {id: part}})."""
    polls, results = [], []
    for pid, pollster, end, hyps in rows:
        polls.append({"poll_id": pid, "election": "leg2024", "pollster": pollster,
                      "fieldwork_start": end - dt.timedelta(days=2), "fieldwork_end": end, "sample_size": 1000})
        for hid, shares in hyps.items():
            results += [{"poll_id": pid, "round": 1, "hypothesis_id": hid, "candidate_id": c, "share": s}
                        for c, s in shares.items()]
    return pl.DataFrame(polls), pl.DataFrame(results).with_columns(pl.col("round").cast(pl.Int8))


VOTE = dt.date(2024, 6, 30)


def test_unit_poll_results_groups_other_and_prefers_union(toy):
    polls, results = _polls([
        ("p1", "ifop", VOTE - dt.timedelta(days=3), {
            "union": {"rn_2024": 31, "uxd_2024": 4, "nfp_2024": 28, "ens_2024": 21, "other": 16},
            "split": {"rn_2024": 31, "uxd_2024": 4, "lfi_2024": 12, "ens_2024": 21, "other": 32},
        }),
        ("p2", "ipsos", VOTE - dt.timedelta(days=3), {
            "h": {"rn_allies_2024": 35, "nfp_2024": 29, "ens_2024": 20, "other": 16},
        }),
    ])
    out = lists.unit_poll_results(polls, results, toy, "leg2024")
    p1 = dict(out.filter(pl.col("poll_id") == "p1").select("candidate_id", "share").iter_rows())
    # l'hypothèse aux composantes RN / Ciotti est gardée (même nombre), regroupée ; `other`
    # va à l'unité résiduelle ; les deux hypothèses restent (LFI seule n'est pas une composante)
    assert out.filter(pl.col("poll_id") == "p1")["hypothesis_id"].n_unique() == 2
    union = out.filter((pl.col("poll_id") == "p1") & pl.col("hypothesis_id").str.contains("nfp"))
    assert dict(union.select("candidate_id", "share").iter_rows()) == {
        "rn_allies_2024": 35, "nfp_2024": 28, "ens_2024": 21, "divers_2024": 16}
    assert "lfi_2024" in p1
    assert out["complete"].all()


def test_component_not_tested_is_incomplete(toy):
    polls, results = _polls([
        ("p1", "ifop", VOTE - dt.timedelta(days=3), {"h": {"rn_2024": 33, "uxd_2024": 4, "nfp_2024": 28, "ens_2024": 21, "other": 14}}),
        ("p2", "elabe", VOTE - dt.timedelta(days=3), {"h": {"rn_2024": 33, "nfp_2024": 28, "ens_2024": 21, "other": 18}}),
    ])
    out = lists.unit_poll_results(polls, results, toy, "leg2024")
    flags = dict(out.filter(pl.col("candidate_id") == "rn_allies_2024").select("poll_id", "complete").iter_rows())
    assert flags == {"p1": True, "p2": False}


def test_poll_errors_toy_case(toy, monkeypatch):
    monkeypatch.setattr(lists, "vote_date", lambda key: VOTE)
    polls, results = _polls([
        ("p1", "ifop", VOTE - dt.timedelta(days=3), {"h": {"rn_2024": 33, "uxd_2024": 4, "nfp_2024": 27, "ens_2024": 21, "other": 15}}),
        ("p2", "ipsos", VOTE - dt.timedelta(days=5), {"h": {"rn_allies_2024": 35, "nfp_2024": 29, "ens_2024": 19, "other": 17}}),
        # composante RN seule (Ciotti non testé) : bloc compté, liste écartée
        ("p3", "elabe", VOTE - dt.timedelta(days=4), {"h": {"rn_2024": 36, "nfp_2024": 28, "ens_2024": 20, "other": 16}}),
        ("p0", "ifop", VOTE - dt.timedelta(days=40), {"h": {"rn_allies_2024": 30, "nfp_2024": 25, "ens_2024": 25, "other": 20}}),
    ])
    res = _results({"RN": 300, "UXD": 40, "UG": 280, "ENS": 200, "DIV": 100, "REG": 80})
    by_bloc = lists.results_lists_by_bloc({"leg2024": res}, toy)
    err = lists.poll_errors_lists(polls, results, by_bloc, toy, keys=["leg2024"], horizons=(30,))
    assert err.schema == lists.POLL_ERRORS_LISTS_SCHEMA
    assert (err["election"] == "leg2024").all() and (err["election_type"] == "leg").all()
    m = err.filter(pl.col("method") == "mean_14d")
    rn_list = m.filter((pl.col("level") == "candidate") & (pl.col("unit") == "rn_allies_2024")).row(0, named=True)
    assert rn_list["n_polls"] == 2 and rn_list["poll_share"] == pytest.approx(36) and rn_list["error"] == pytest.approx(2)
    fr = m.filter((pl.col("level") == "bloc") & (pl.col("unit") == "far_right")).row(0, named=True)
    assert fr["n_polls"] == 3 and fr["poll_share"] == pytest.approx(36)
    other = m.filter((pl.col("level") == "candidate") & (pl.col("unit") == "divers_2024")).row(0, named=True)
    assert other["poll_share"] == pytest.approx(16) and other["result_share"] == pytest.approx(18)
    # horizon 30 : fenêtre ]vote − 60, vote − 30], seul p0
    h = err.filter((pl.col("method") == "horizon") & (pl.col("level") == "bloc") & (pl.col("unit") == "centre"))
    assert h["n_polls"].to_list() == [1] and h["error"].item() == pytest.approx(5)
    last = err.filter((pl.col("method") == "last_by_pollster") & (pl.col("pollster") == "ifop"))
    assert set(last["poll_id"]) == {"p1"}
