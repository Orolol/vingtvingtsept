from datetime import date

import pytest

from vvs.ingest.wikipedia.common import (
    DateBounds,
    UnknownLabelError,
    bounds_from_context,
    load_aliases,
    parse_share,
    parse_share_tokens,
    resolve_range,
)

UPPER = DateBounds(lower=date(2017, 5, 8), upper=date(2022, 4, 10))


@pytest.mark.parametrize(
    ("text", "prev", "expected"),
    [
        ("7-8 avril", None, (date(2022, 4, 7), date(2022, 4, 8))),
        ("31 mars - 1 er avril", None, (date(2022, 3, 31), date(2022, 4, 1))),
        ("29 janv - 1 er fév.", None, (date(2022, 1, 29), date(2022, 2, 1))),
        ("28 décembre - 3 janvier", date(2022, 1, 10), (date(2021, 12, 28), date(2022, 1, 3))),
        ("15 décembre", date(2022, 1, 5), (date(2021, 12, 15), date(2021, 12, 15))),
        ("14 et 15 décembre 2015", None, (date(2015, 12, 14), date(2015, 12, 15))),
        ("1 er au 6 avril 2017", None, (date(2017, 4, 1), date(2017, 4, 6))),
    ],
)
def test_resolve_range(text, prev, expected):
    assert resolve_range(text, UPPER, prev) == expected


def test_month_and_year_from_context():
    bounds = bounds_from_context(["Mars 2017", "Sondages de 2017"], UPPER)
    assert bounds.default_month == 3
    assert resolve_range("du 17 au 19", bounds, None) == (date(2017, 3, 17), date(2017, 3, 19))
    sem = bounds_from_context(["Second semestre 2021"], UPPER)
    assert (sem.lower, sem.upper) == (date(2021, 7, 1), date(2021, 12, 31))
    # « l'élection présidentielle 2017 » ne borne pas la période
    assert bounds_from_context(["hypothèses de l'élection présidentielle 2017"], UPPER) == UPPER


def test_ambiguous_year_is_rejected():
    with pytest.raises(ValueError, match="ambiguë"):
        resolve_range("5 mai", UPPER, date(2021, 12, 1))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("12,5 %", 12.5), ("<0,5 %", 0.25), ("< 1", 0.5), ("—", None), ("38 % (1 006)", 38)],
)
def test_parse_share(text, expected):
    assert parse_share(text) == expected


def test_parse_share_rejects_garbage():
    with pytest.raises(ValueError):
        parse_share("20-24 %")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14 % Bertrand", [(14, "Bertrand")]),
        ("9 Hollande ( PS )", [(9, "Hollande")]),
        ("4 Lisnard 2 Ruffin 1 Villepin", [(4, "Lisnard"), (2, "Ruffin"), (1, "Villepin")]),
        ("<1 Bouamrane", [(0.5, "Bouamrane")]),
        ("Alliot-Marie : 1 % Yade : 1 %", [(1, "Alliot-Marie"), (1, "Yade")]),
        ("12,5 %", [(12.5, None)]),
        ("–", []),
    ],
)
def test_parse_share_tokens(text, expected):
    assert parse_share_tokens(text) == expected


def test_aliases_are_scoped_and_strict():
    aliases = load_aliases(2022)
    assert aliases.candidate("Jean-Luc Mélenchon ( LFI )") == "melenchon_jean_luc"
    assert aliases.candidate("Le Pen") == "le_pen_marine"
    assert aliases.pollster("Ifop (rolling)") == ("ifop", None)
    assert aliases.pollster("Ifop-Fiducial") == ("ifop", "Fiducial")
    with pytest.raises(UnknownLabelError):
        aliases.candidate("Inconnu Total")
    # « Le Pen » seul n'est pas dans la section commune : Jean-Marie avant 2012, Marine après
    assert load_aliases(2007).candidate("Le Pen") == "le_pen_jean_marie"
    assert load_aliases(2012).candidate("Le Pen") == "le_pen_marine"
    with pytest.raises(UnknownLabelError):
        load_aliases(1995).candidate("Le Pen")
