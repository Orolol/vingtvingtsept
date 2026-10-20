import shutil

import pytest
import yaml

from vvs import paths
from vvs.registry import BLOCS, RegistryError, candidate_id_for, load_registry


@pytest.fixture(scope="module")
def reg():
    return load_registry()


def test_registry_loads_with_all_elections(reg):
    assert set(reg.blocs) == set(BLOCS)
    assert sorted(reg.elections) == [2002, 2007, 2012, 2017, 2022, 2027]
    for year in (2002, 2007, 2012, 2017, 2022):
        e = reg.elections[year]
        assert [r.round for r in e.rounds] == [1, 2]
        assert all(r.results and r.date_status == "official" for r in e.rounds)
        assert all("conseil-constitutionnel.fr" in r.source_url for r in e.rounds)
    e2027 = reg.elections[2027]
    assert e2027.candidates == []
    assert all(not r.results and r.date_status == "announced" for r in e2027.rounds)


@pytest.mark.parametrize(
    "year, round_, candidate_id, votes, share",
    [
        (2022, 2, "macron_emmanuel", 18_768_639, 58.55),
        (2002, 2, "chirac_jacques", 25_537_956, 82.21),
        (2017, 1, "macron_emmanuel", 8_656_346, 24.01),
    ],
)
def test_official_results(reg, year, round_, candidate_id, votes, share):
    r = next(
        x for x in reg.elections[year].round_(round_).results if x.candidate_id == candidate_id
    )
    assert (r.votes, r.share) == (votes, share)


def test_official_turnout_2022_second_round(reg):
    r = reg.elections[2022].round_(2)
    assert (r.registered, r.voters, r.expressed) == (48_752_339, 35_096_478, 32_057_325)


@pytest.mark.parametrize(
    "candidate_id, election, bloc",
    [
        ("bayrou_francois", 2002, "right"),
        ("bayrou_francois", 2007, "centre"),
        ("melenchon_jean_luc", 2012, "far_left"),
        ("jadot_yannick", 2022, "left"),
        ("zemmour_eric", 2022, "far_right"),
        ("hollande_francois", 2027, "left"),  # cas limite blocs.yaml
        ("bardella_jordan", 2027, "far_right"),  # bloc par défaut du parti
        ("philippe_edouard", 2027, "centre"),
    ],
)
def test_bloc_of(reg, candidate_id, election, bloc):
    assert reg.bloc_of(candidate_id, election) == bloc


def test_bloc_of_unknown_candidate(reg):
    with pytest.raises(KeyError):
        reg.bloc_of("inconnu_x", 2027)


@pytest.mark.parametrize(
    "label, pollster",
    [
        ("IFOP-FIDUCIAL", "ifop"),
        ("Toluna Harris Interactive", "harris"),
        ("Ipsos Sopra Steria", "ipsos"),
        ("Ipsos bva", "ipsos"),
        ("BVA", "bva"),
        ("Kantar Sofres", "kantar"),
        ("TNS Sofres", "sofres"),
        ("Cluster 17", "cluster17"),
        ("Élabe", "elabe"),
    ],
)
def test_pollster_of(reg, label, pollster):
    assert reg.pollster_of(label) == pollster


def test_pollster_of_unknown_label(reg):
    with pytest.raises(KeyError):
        reg.pollster_of("Institut imaginaire")


def test_candidate_id_convention():
    assert candidate_id_for("Le Pen", "Jean-Marie") == "le_pen_jean_marie"
    assert candidate_id_for("Mélenchon", "Jean-Luc") == "melenchon_jean_luc"


def _round(data, year, number):
    return next(r for r in data["elections"][year]["rounds"] if r["round"] == number)


def _set_bloc(data, year, cid, bloc):
    candidacy = next(c for c in data["elections"][year]["candidates"] if c["candidate_id"] == cid)
    candidacy["bloc"] = bloc


@pytest.mark.parametrize(
    "filename, mutate, message",
    [
        (
            "candidates.yaml",
            lambda d: d["candidates"].pop("arthaud_nathalie"),
            "arthaud_nathalie absent de candidates.yaml",
        ),
        (
            "elections.yaml",
            lambda d: d["elections"][2022]["candidates"][0].update(party="parti_fantome"),
            "parti inconnu",
        ),
        (
            "elections.yaml",
            lambda d: d["elections"][2022]["candidates"][0].update(party="ps"),
            "pas actif",
        ),
        (
            "elections.yaml",
            lambda d: _round(d, 2022, 2)["results"][0].update(share=60.0),
            "incohérent avec voix",
        ),
        (
            "elections.yaml",
            lambda d: _round(d, 2022, 2)["results"][0].update(votes=1),
            "somme des voix",
        ),
        (
            "elections.yaml",
            lambda d: _set_bloc(d, 2002, "bayrou_francois", "centre"),
            "le cas limite dit right",
        ),
        (
            "pollsters.yaml",
            lambda d: d["pollsters"]["elabe"]["aliases"].append("IFOP"),
            "ambigu",
        ),
        (
            "candidates.yaml",
            lambda d: d["candidates"]["arthaud_nathalie"].update(last_name="Arthaux"),
            "nom_prenom",
        ),
    ],
)
def test_cross_validation_rejects(tmp_path, filename, mutate, message):
    shutil.copytree(paths.REGISTRY, tmp_path, dirs_exist_ok=True)
    path = tmp_path / filename
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises((RegistryError, ValueError), match=message):
        load_registry(tmp_path)
