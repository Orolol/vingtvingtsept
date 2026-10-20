import copy

import pytest
import yaml

from vvs.registry.transfers import TransferError, load_transfers, transfer_prior

PAIR_2017 = ("macron_emmanuel", "le_pen_marine")


@pytest.fixture(scope="module")
def real():
    return load_transfers()


def _source(**over) -> dict:
    s = {
        "id": "ipsos_2017",
        "election": 2017,
        "pollster": "ipsos",
        "title": "t",
        "url": "https://example.org/a.pdf",
        "base": "first_round_electorate",
        "base_note": "-",
        "columns": "split",
        "rows": [
            {
                "from": "melenchon_jean_luc",
                "to": {"macron_emmanuel": 52, "le_pen_marine": 7, "blank_null": 17, "abstain": 24},
            },
            {"from": "fillon_francois", "to": {"macron_emmanuel": 48, "le_pen_marine": 20}},
        ],
    }
    s.update(over)
    return s


def _combined() -> dict:
    return _source(
        id="ifop_2017",
        pollster="ifop",
        columns="combined",
        rows=[
            {
                "from": "melenchon_jean_luc",
                "to": {"macron_emmanuel": 54, "le_pen_marine": 14, "blank_or_abstain": 32},
            }
        ],
    )


def _expressed() -> dict:
    return _source(
        id="harris_2017_expressed",
        pollster="harris",
        columns="expressed",
        base="second_round_expressed",
        rows=[{"from": "melenchon_jean_luc", "to": {"macron_emmanuel": 80, "le_pen_marine": 20}}],
    )


def _load(tmp_path, *sources):
    p = tmp_path / "transfers.yaml"
    p.write_text(yaml.safe_dump({"transfers": list(sources)}, allow_unicode=True), encoding="utf-8")
    return load_transfers(p)


def test_real_file_covers_every_second_round(real):
    years = {s.election for s in real}
    assert years == {2002, 2007, 2012, 2017, 2022}
    for s in real:
        assert s.url.startswith("http")
        # chaque source a au moins une ligne complète
        assert s.rows


@pytest.mark.parametrize(
    "election, from_candidate, pair",
    [
        (2017, "melenchon_jean_luc", PAIR_2017),
        (2022, "zemmour_eric", ("le_pen_marine", "macron_emmanuel")),
    ],
)
def test_real_prior_is_a_distribution(real, election, from_candidate, pair):
    prior = transfer_prior(election, from_candidate, pair, transfers=real)
    assert list(prior.shares)[:2] == list(pair)
    assert sum(prior.shares.values()) == pytest.approx(100, abs=2)
    assert len(prior.sources) >= 2


def test_prior_averages_and_collapses_columns(tmp_path):
    sources = _load(tmp_path, _source(), _combined(), _expressed())
    prior = transfer_prior(2017, "melenchon_jean_luc", PAIR_2017, transfers=sources)
    # la source « exprimés » n'entre pas dans la base électorat
    assert prior.sources == ("ipsos_2017", "ifop_2017")
    assert prior.shares == pytest.approx(
        {"macron_emmanuel": 53, "le_pen_marine": 10.5, "blank_or_abstain": 36.5}
    )
    split = transfer_prior(2017, "melenchon_jean_luc", PAIR_2017, basis="split", transfers=sources)
    assert split.sources == ("ipsos_2017",)
    assert split.shares["abstain"] == 24
    expressed = transfer_prior(
        2017, "melenchon_jean_luc", PAIR_2017, basis="expressed", transfers=sources
    )
    assert len(expressed.sources) == 3
    assert expressed.shares["macron_emmanuel"] == pytest.approx(
        (100 * 52 / 59 + 100 * 54 / 68 + 80) / 3
    )


def test_prior_ignores_incomplete_rows_and_checks_pair(tmp_path):
    sources = _load(tmp_path, _source())
    with pytest.raises(KeyError, match="aucune source"):
        transfer_prior(2017, "fillon_francois", PAIR_2017, transfers=sources)
    with pytest.raises(KeyError, match="paire"):
        transfer_prior(
            2017, "melenchon_jean_luc", ("macron_emmanuel", "fillon_francois"), transfers=sources
        )


def _bad(path, value):
    def apply(s):
        target = s
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
        return s

    return apply


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_bad(["rows", 0, "to", "abstain"], 30), "somme"),
        (_bad(["rows", 1, "to", "blank_or_abstain"], 10), "colonnes inattendues"),
        (_bad(["rows", 0, "from"], "jadot_yannick"), "ni candidat"),
        (_bad(["pollster"], "inconnu"), "institut"),
        (_bad(["election"], 2027), "second tour"),
        (_bad(["base"], "second_round_voters"), "incompatible"),
        (_bad(["rows", 1, "from"], "melenchon_jean_luc"), "double"),
        (_bad(["rows", 1, "to"], {"macron_emmanuel": 80, "le_pen_marine": 30}), "dépasse"),
    ],
)
def test_validation_errors(tmp_path, mutate, message):
    with pytest.raises(TransferError, match=message):
        _load(tmp_path, mutate(copy.deepcopy(_source())))


def test_grouped_electorate_needs_known_members(tmp_path):
    row = {
        "from": "far_left",
        "members": ["poutou_philippe", "arthaud_nathalie"],
        "to": {"macron_emmanuel": 60, "le_pen_marine": 10, "blank_null": 10, "abstain": 20},
    }
    s = _source(rows=[row])
    assert _load(tmp_path, s)[0].rows[0].members == ["poutou_philippe", "arthaud_nathalie"]
    s["rows"][0]["members"] = ["laguiller_arlette"]
    with pytest.raises(TransferError, match="membre"):
        _load(tmp_path, s)


def test_group_without_members_needs_a_note(tmp_path):
    row = {
        "from": "left_excl_hamon",
        "to": {"macron_emmanuel": 60, "le_pen_marine": 10, "blank_null": 10, "abstain": 20},
    }
    with pytest.raises(TransferError, match="group_note"):
        _load(tmp_path, _source(rows=[row]))
    row["group_note"] = "sous-total de la source, composition non détaillée"
    assert _load(tmp_path, _source(rows=[row]))[0].rows[0].group_note


def test_other_base_excluded_unless_requested(real):
    pair = ("hollande_francois", "sarkozy_nicolas")
    default = transfer_prior(2012, "bayrou_francois", pair, transfers=real)
    assert all(s.base != "other" for s in real if s.id in default.sources)
    wide = transfer_prior(2012, "bayrou_francois", pair, transfers=real, include_other=True)
    assert set(default.sources) < set(wide.sources)
