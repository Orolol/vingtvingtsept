"""Curation : fusion des sources, précédence, journal, rolling, suspects, tables dérivées.

Petits jeux de données construits à la main ; chaque test reproduit un cas réel (cité) ou
une règle de docs/03-donnees.md.
"""

from datetime import date, datetime

import polars as pl
import pytest

from vvs import schemas
from vvs.curate import derived, rules
from vvs.curate.merge import merge
from vvs.curate.sources import to_source_polls

D = date.fromisoformat
FIRST = ["arthaud_nathalie", "melenchon_jean_luc", "glucksmann_raphael", "philippe_edouard",
         "retailleau_bruno", "le_pen_marine", "zemmour_eric"]
BASE = [1.0, 13.0, 12.0, 20.0, 9.0, 35.0, 10.0]


def poll(pid, source, start, end, pollster="ifop", sponsor=None, n=1000, mode="unknown", election=2027):
    return {
        "poll_id": pid, "election": election, "pollster": pollster, "sponsor": sponsor,
        "fieldwork_start": D(start), "fieldwork_end": D(end), "publish_date": None,
        "sample_size": n, "mode": mode, "source": source, "source_url": f"https://x/{pid}",
        "ingested_at": datetime(2026, 10, 20),
    }


def hyp(pid, shares: dict[str, float], round_=1, population="unknown", hid=None):
    hid = hid or schemas.hypothesis_id(list(shares))
    return [
        {"poll_id": pid, "round": round_, "hypothesis_id": hid, "population": population,
         "candidate_id": c, "share": s,
         "share_base": "all_respondents" if c in ("abstention", "undecided") else "expressed",
         "raw_share": None}
        for c, s in shares.items()
    ]


def first(**override) -> dict[str, float]:
    shares = dict(zip(FIRST, BASE))
    for k, v in override.items():
        shares.pop(k, None) if v is None else shares.__setitem__(k, v)
    return shares


def src(source, polls, rows, rolling=()):
    p = pl.DataFrame(polls, schema=schemas.POLLS_SCHEMA)
    r = pl.DataFrame(rows, schema=schemas.POLL_RESULTS_SCHEMA)
    return to_source_polls(source, p, r, set(rolling))


def run(**by_source):
    return merge({k: v for k, v in by_source.items()})


def shares_of(res, poll_id, hid=None, round_=1):
    r = res.results.filter((pl.col("poll_id") == poll_id) & (pl.col("round") == round_))
    if hid is not None:
        r = r.filter(pl.col("hypothesis_id") == hid)
    return dict(r.select("candidate_id", "share").iter_rows())


# ---------------------------------------------------------------- appariement


def test_dedup_one_day_tolerance_and_sources():
    c = src("commission", [poll("c1", "commission", "2026-03-25", "2026-03-27", sponsor="BFMTV",
                                n=1504, mode="online")], hyp("c1", first(), population="certain_voters"))
    w = src("wikipedia", [poll("w1", "wikipedia", "2026-03-26", "2026-03-27", n=1400),
                          poll("w2", "wikipedia", "2026-03-20", "2026-03-22")],
            hyp("w1", first()) + hyp("w2", first(le_pen_marine=30.0, zemmour_eric=15.0)))
    res = run(commission=c, wikipedia=w)
    assert res.polls.height == 2
    merged = res.polls.filter(pl.col("sources").list.len() == 2).row(0, named=True)
    assert merged["sources"] == ["commission", "wikipedia"] and merged["source"] == "commission"
    assert (merged["fieldwork_start"], merged["sample_size"], merged["mode"]) == (D("2026-03-25"), 1504, "online")
    assert merged["poll_id"] == "2027-ifop-20260327-1"
    mapping = dict(res.poll_sources.select("source_poll_id", "canonical_poll_id").iter_rows())
    assert mapping["c1"] == mapping["w1"] == "2027-ifop-20260327-1"
    # population de Wikipédia complétée par la Commission (même hypothèse)
    assert res.results.filter(pl.col("poll_id") == merged["poll_id"])["population"].unique().to_list() == ["certain_voters"]
    # écarts de métadonnées journalisés, Commission retenue
    log = {(e["field"], e["rule"]): e for e in res.log}
    assert log[("fieldwork_start", "precedence")]["retained"] == "2026-03-25"
    assert log[("sample_size", "precedence")]["values"] == {"commission": 1504, "wikipedia": 1400}


def test_sponsor_breaks_ties_but_never_splits_same_dates():
    c = src("commission", [poll("c1", "commission", "2026-10-13", "2026-10-14", sponsor="CNews", n=1043),
                           poll("c2", "commission", "2026-10-13", "2026-10-14", n=1001)],
            hyp("c1", first()) + hyp("c2", first(zemmour_eric=None, arthaud_nathalie=None,
                                                 le_pen_marine=46.0)))
    # Wikipédia a fusionné les deux notices en un sondage (7 hypothèses -> 2 ici)
    w = src("wikipedia", [poll("w1", "wikipedia", "2026-10-13", "2026-10-14", n=1043)],
            hyp("w1", first()) + hyp("w1", first(zemmour_eric=None, arthaud_nathalie=None,
                                                 le_pen_marine=46.0)))
    m = src("mieuxvoter", [poll("m1", "mieuxvoter", "2026-10-13", "2026-10-14", sponsor="Facebook", n=1043)],
            hyp("m1", first()))
    res = run(commission=c, wikipedia=w, mieuxvoter=m)
    assert res.polls.height == 2
    assert res.poll_sources.filter(pl.col("source_poll_id") == "w1").height == 2
    sponsor = [e for e in res.log if e["field"] == "sponsor"]
    assert len(sponsor) == 1 and sponsor[0]["retained"] == "CNews"
    assert not [e for e in res.log if e["field"] == "share"]


def test_same_source_polls_never_merged():
    w = src("wikipedia", [poll("w1", "wikipedia", "2022-03-01", "2022-03-02"),
                          poll("w2", "wikipedia", "2022-03-10", "2022-03-11")],
            hyp("w1", first()) + hyp("w2", first()))
    assert run(wikipedia=w).polls.height == 2


# ---------------------------------------------------------------- précédence et journal


def test_precedence_and_log_entry():
    c = src("commission", [poll("c1", "commission", "2025-09-24", "2025-09-25")],
            hyp("c1", first(), population="registered"))
    w = src("wikipedia", [poll("w1", "wikipedia", "2025-09-24", "2025-09-25")],
            hyp("w1", first(arthaud_nathalie=0.25)))
    m = src("mieuxvoter", [poll("m1", "mieuxvoter", "2025-09-24", "2025-09-25")],
            hyp("m1", first(arthaud_nathalie=0.5), population="certain_voters"))
    res = merge({"commission": c, "wikipedia": w, "mieuxvoter": m}, mv_unreliable=set())
    assert shares_of(res, "2027-ifop-20250925-1")["arthaud_nathalie"] == 1.0
    entries = [e for e in res.log if e["field"] == "share"]
    assert {tuple(e["values"].items()) for e in entries} == {
        (("commission", 1.0), ("wikipedia", 0.25)), (("commission", 1.0), ("mieuxvoter", 0.5))
    }
    e = entries[0]
    assert e["retained"] == 1.0 and e["rule"] == "precedence" and e["candidate_id"] == "arthaud_nathalie"
    assert e["key"] == {"election": 2027, "pollster": "ifop", "fieldwork_start": "2025-09-24",
                        "fieldwork_end": "2025-09-25", "sponsor": None}
    pop = [x for x in res.log if x["field"] == "population"]
    assert pop[0]["values"] == {"commission": "registered", "mieuxvoter": "certain_voters"}
    assert pop[0]["retained"] == "registered"


def test_label_swap_elabe_march_2026():
    """Elabe 25-27 mars 2026 : Wikipédia met 8,5 sous Glucksmann, la notice sous Hollande."""
    good = first(glucksmann_raphael=None, hollande_francois=8.5, philippe_edouard=23.5)
    shifted = first(philippe_edouard=23.5, glucksmann_raphael=8.5)
    c = src("commission", [poll("c1", "commission", "2026-03-25", "2026-03-27", pollster="elabe")],
            hyp("c1", good))
    w = src("wikipedia", [poll("w1", "wikipedia", "2026-03-25", "2026-03-27", pollster="elabe")],
            hyp("w1", shifted) + hyp("w1", {"attal_gabriel": 49.0, "le_pen_marine": 51.0}, round_=2))
    res = run(commission=c, wikipedia=w)
    pid = "2027-elabe-20260327-1"
    assert "glucksmann_raphael" not in shares_of(res, pid)
    assert shares_of(res, pid)["hollande_francois"] == 8.5
    assert shares_of(res, pid, round_=2) == {"attal_gabriel": 49.0, "le_pen_marine": 51.0}
    (swap,) = [e for e in res.log if e["rule"] == "label_swap"]
    assert swap["values"] == {"commission": "hollande_francois", "wikipedia": "glucksmann_raphael"}
    assert rules.check_known_divergences(res.log)["elabe_2026_03_column_shift"]


def test_same_results_wrong_dates_ifop_march_2025():
    c = src("commission", [poll("c1", "commission", "2025-03-26", "2025-03-27", sponsor="Le JDD")],
            hyp("c1", first()))
    w = src("wikipedia", [poll("w1", "wikipedia", "2025-03-21", "2025-03-26")], hyp("w1", first()))
    res = run(commission=c, wikipedia=w)
    assert res.polls.height == 1
    assert res.poll_sources.filter(pl.col("source") == "wikipedia")["match_rule"].to_list() == ["same_results"]
    dates = {e["field"]: e for e in res.log if e["rule"] == "same_results"}
    assert dates["fieldwork_start"]["values"] == {"commission": "2025-03-26", "wikipedia": "2025-03-21"}
    assert rules.check_known_divergences(res.log)["ifop_2025_03_dates"]


def test_two_waves_in_one_notice_and_inverted_duel():
    """Ifop pour Hexagone : une notice (11-18 avril 2025) porte aussi la vague du 29-30 avril ;
    MieuxVoter rattache à la première vague le duel Philippe–Le Pen, inversé."""
    wave1 = first()
    wave2 = first(retailleau_bruno=None, attal_gabriel=9.0)
    duel = {"philippe_edouard": 52.0, "le_pen_marine": 48.0}
    inverted = {"philippe_edouard": 48.0, "le_pen_marine": 52.0}
    c = src("commission", [poll("c1", "commission", "2025-04-11", "2025-04-18", sponsor="Hexagone", n=10000)],
            hyp("c1", wave1) + hyp("c1", wave2) + hyp("c1", duel, round_=2))
    w = src("wikipedia", [poll("w1", "wikipedia", "2025-04-11", "2025-04-18", n=9128),
                          poll("w2", "wikipedia", "2025-04-29", "2025-04-30", n=1838)],
            hyp("w1", wave1) + hyp("w2", wave2) + hyp("w2", duel, round_=2))
    m = src("mieuxvoter", [poll("m1", "mieuxvoter", "2025-04-11", "2025-04-18", n=10000),
                           poll("m2", "mieuxvoter", "2025-04-29", "2025-04-30", n=2000)],
            hyp("m1", wave1) + hyp("m1", inverted, round_=2) + hyp("m2", wave2) + hyp("m2", duel, round_=2))
    res = merge({"commission": c, "wikipedia": w, "mieuxvoter": m}, mv_unreliable=set())
    assert res.polls["poll_id"].to_list() == ["2027-ifop-20250418-1", "2027-ifop-20250430-1"]
    second = res.polls.row(1, named=True)
    assert (second["fieldwork_start"], second["sample_size"], second["sponsor"]) == (D("2025-04-29"), 1838, "Hexagone")
    assert second["sources"] == ["commission", "wikipedia", "mieuxvoter"]
    assert res.results.filter(pl.col("poll_id") == "2027-ifop-20250418-1")["round"].unique().to_list() == [1]
    assert shares_of(res, "2027-ifop-20250430-1", round_=2) == duel
    known = rules.check_known_divergences(res.log)
    assert known["ifop_2025_04_two_waves"] and known["ifop_2025_04_duel_inverted"]


def test_swap_consistency_overrides_notice_typo():
    """OpinionWay sept. 2026 : une page de la notice inverse Le Pen (34) et Zemmour (4)."""
    good = first(le_pen_marine=34.0, zemmour_eric=4.0, philippe_edouard=26.0)
    typo = first(le_pen_marine=4.0, zemmour_eric=34.0, philippe_edouard=26.0)
    other = first(le_pen_marine=33.0, zemmour_eric=5.0, philippe_edouard=None, attal_gabriel=26.0)
    c = src("commission", [poll("c1", "commission", "2026-09-08", "2026-09-09", pollster="opinionway")],
            hyp("c1", typo) + hyp("c1", other))
    w = src("wikipedia", [poll("w1", "wikipedia", "2026-09-08", "2026-09-09", pollster="opinionway")],
            hyp("w1", good) + hyp("w1", other))
    res = run(commission=c, wikipedia=w)
    hid = schemas.hypothesis_id(list(good))
    got = shares_of(res, "2027-opinionway-20260909-1", hid)
    assert (got["le_pen_marine"], got["zemmour_eric"]) == (34.0, 4.0)
    rows = res.results.filter(pl.col("hypothesis_id") == hid)
    assert dict(rows.select("candidate_id", "source").iter_rows())["le_pen_marine"] == "wikipedia"
    assert {e["rule"] for e in res.log if e["field"] == "share"} == {"swap_consistency"}
    # sans autre hypothèse pour juger, la précédence s'applique
    c2 = src("commission", [poll("c1", "commission", "2026-09-08", "2026-09-09", pollster="opinionway")], hyp("c1", typo))
    w2 = src("wikipedia", [poll("w1", "wikipedia", "2026-09-08", "2026-09-09", pollster="opinionway")], hyp("w1", good))
    res2 = run(commission=c2, wikipedia=w2)
    assert shares_of(res2, "2027-opinionway-20260909-1")["le_pen_marine"] == 4.0


# ---------------------------------------------------------------- population


def test_population_inferred_from_commission_poll_and_mv_unreliable():
    pops = [("2026-02-26", "2026-02-27"), ("2026-05-26", "2026-05-28"), ("2026-06-22", "2026-06-24")]
    c_polls, c_rows, m_polls, m_rows = [], [], [], []
    for i, (s, e) in enumerate(pops):
        c_polls.append(poll(f"c{i}", "commission", s, e))
        c_rows += hyp(f"c{i}", first(), population="registered")
        m_polls.append(poll(f"m{i}", "mieuxvoter", s, e))
        m_rows += hyp(f"m{i}", first(), population="certain_voters")
        m_rows += hyp(f"m{i}", {"philippe_edouard": 50.0, "le_pen_marine": 50.0}, 2, "certain_voters")
    # sondage MieuxVoter seul, même institut
    m_polls.append(poll("m9", "mieuxvoter", "2024-12-06", "2024-12-09"))
    m_rows += hyp("m9", first(), population="certain_voters")
    res = run(commission=src("commission", c_polls, c_rows), mieuxvoter=src("mieuxvoter", m_polls, m_rows))
    assert res.mv_unreliable == {"ifop"}
    r2 = res.results.filter(pl.col("round") == 2)
    assert r2["population"].unique().to_list() == ["registered"]  # hérité de la Commission
    alone = res.results.filter(pl.col("poll_id") == "2027-ifop-20241209-1")
    assert alone["population"].unique().to_list() == ["unknown"]
    assert sum(e["rule"] == "mv_population_unreliable" for e in res.log) == 4


# ---------------------------------------------------------------- seconde base


def _ow(pid, source):
    return [poll(pid, source, "2026-10-13", "2026-10-14", pollster="opinionway")]


def test_second_base_kept_and_one_base_selects_certain_voters():
    """OpinionWay 2026 : chaque tableau publié sur les certains d'aller voter (colonne
    publiée, recopiée par Wikipédia) et sur les inscrits, sans un petit candidat."""
    cert = first(le_pen_marine=35.0, zemmour_eric=9.5, bouamrane_karim=0.5)
    reg = first(le_pen_marine=32.0, zemmour_eric=13.0)
    hid = schemas.hypothesis_id(list(cert))
    c = src("commission", _ow("c1", "commission"),
            hyp("c1", cert, population="certain_voters")
            + hyp("c1", reg, population="registered", hid=hid))
    w = src("wikipedia", _ow("w1", "wikipedia"), hyp("w1", cert))
    res = run(commission=c, wikipedia=w)
    r = res.results.filter(pl.col("poll_id") == "2027-opinionway-20261014-1")
    assert r["hypothesis_id"].unique().to_list() == [hid]
    assert sorted(r["population"].unique().to_list()) == ["certain_voters", "registered"]
    # Wikipédia comparé au publié seulement : aucun conflit
    assert not [e for e in res.log if e["field"] in ("share", "population")]
    kept = rules.one_base(res.results)
    assert kept["population"].unique().to_list() == ["certain_voters"]
    assert dict(kept.select("candidate_id", "share").iter_rows())["le_pen_marine"] == 35.0
    # une hypothèse sur une seule base garde la sienne, même si une autre hypothèse du même
    # sondage est sur une autre base (populations MieuxVoter, YouGov septembre 2026)
    mixed = pl.DataFrame(
        hyp("y", first(), population="registered")
        + hyp("y", first(zemmour_eric=None, le_pen_marine=45.0), population="certain_voters"),
        schema=schemas.POLL_RESULTS_SCHEMA,
    )
    assert rules.one_base(mixed).height == mixed.height


def test_second_base_follows_swap_consistency():
    """La page inversée (Le Pen 4, Zemmour 34) l'est sur les deux bases du même tableau."""
    typo = first(le_pen_marine=4.0, zemmour_eric=34.0, philippe_edouard=26.0)
    typo_reg = first(le_pen_marine=5.0, zemmour_eric=31.0, philippe_edouard=28.0)
    good = first(le_pen_marine=34.0, zemmour_eric=4.0, philippe_edouard=26.0)
    other = first(le_pen_marine=33.0, zemmour_eric=5.0, philippe_edouard=None, attal_gabriel=26.0)
    c = src("commission", _ow("c1", "commission"),
            hyp("c1", typo, population="certain_voters")
            + hyp("c1", typo_reg, population="registered")
            + hyp("c1", other, population="certain_voters"))
    w = src("wikipedia", _ow("w1", "wikipedia"), hyp("w1", good) + hyp("w1", other))
    res = run(commission=c, wikipedia=w)
    hid = schemas.hypothesis_id(list(good))
    r = res.results.filter(pl.col("hypothesis_id") == hid)
    got = {(p, c): s for p, c, s in r.select("population", "candidate_id", "share").iter_rows()}
    assert (got[("certain_voters", "le_pen_marine")], got[("certain_voters", "zemmour_eric")]) == (34.0, 4.0)
    assert (got[("registered", "le_pen_marine")], got[("registered", "zemmour_eric")]) == (31.0, 5.0)
    swaps = [e for e in res.log if e["rule"] == "swap_consistency"]
    assert sum(e.get("population") == "registered" for e in swaps) == 1


# ---------------------------------------------------------------- rolling


def _rolling_frame(rows):
    return pl.DataFrame(
        [(pid, 2022, p, D(s), D(e), n) for pid, p, s, e, n in rows],
        schema=["poll_id", "election", "pollster", "fieldwork_start", "fieldwork_end", "sample_size"],
        orient="row",
    )


def test_rolling_by_regularity():
    df = _rolling_frame([
        # Ifop-Fiducial 2017 : vagues glissantes publiées tous les 2-4 jours
        ("r1", "ifop", "2022-02-02", "2022-02-06", 1433),
        ("r2", "ifop", "2022-02-06", "2022-02-09", 1407),
        ("r3", "ifop", "2022-02-09", "2022-02-13", 1392),
        # baromètre ordinaire : pas de chevauchement
        ("o1", "ifop", "2022-02-20", "2022-02-21", 1000),
        # deux vagues qui se chevauchent ne suffisent pas
        ("p1", "elabe", "2022-02-01", "2022-02-02", 1000),
        ("p2", "elabe", "2022-02-02", "2022-02-03", 1000),
        # Ipsos-Cevipof (13 000) chevauchant un rolling Ipsos : effectifs trop différents
        ("i1", "ipsos", "2022-03-10", "2022-03-13", 1650),
        ("i2", "ipsos", "2022-03-10", "2022-03-14", 8790),
        ("i3", "ipsos", "2022-03-13", "2022-03-16", 1651),
        ("i4", "ipsos", "2022-03-16", "2022-03-19", 1723),
    ])
    assert rules.rolling_by_regularity(df) == {"r1", "r2", "r3", "i1", "i3", "i4"}
    # une vague ordinaire ne se relie pas à des vagues étiquetées rolling
    df2 = _rolling_frame([
        ("a", "ifop", "2022-03-14", "2022-03-17", 1506),
        ("x", "ifop", "2022-03-17", "2022-03-18", 1359),
        ("b", "ifop", "2022-03-17", "2022-03-21", 1508),
    ])
    assert rules.rolling_by_regularity(df2, {"a", "b"}) == set()


def test_is_rolling_from_label():
    assert rules.is_rolling_label("10259 Pres Presitrack V2 OPINIONWAY Les Echos 10 septembre")
    assert rules.is_rolling_label("Ifop (rolling)") and not rules.is_rolling_label("Ifop-Fiducial")
    w = src("wikipedia", [poll("w1", "wikipedia", "2022-03-01", "2022-03-03", pollster="opinionway")],
            hyp("w1", first()), rolling={"w1"})
    assert run(wikipedia=w).polls["is_rolling"].to_list() == [True]


# ---------------------------------------------------------------- suspects


RACE = rules.OfficialRace(D("2022-04-10"), D("2022-04-24"), frozenset(FIRST), frozenset({"le_pen_marine", "philippe_edouard"}))


def _reasons(shares, start, round_=1, base="expressed"):
    return rules.suspect_reasons(
        round_, {c: (s, base) for c, s in shares.items()}, D(start), D(start), RACE,
        set(FIRST) | {"hollande_francois", "jadot_yannick"},
    )


@pytest.mark.parametrize(
    "shares, start, round_, expected",
    [
        (first(), "2022-04-05", 1, []),
        (first(le_pen_marine=40.0), "2022-01-05", 1, ["sum_out_of_range"]),
        (first(arthaud_nathalie=None, le_pen_marine=36.0), "2022-04-05", 1, []),  # un seul absent
        (first(arthaud_nathalie=None, zemmour_eric=None, le_pen_marine=46.0), "2022-04-05", 1,
         ["incomplete_hypothesis"]),
        (first(arthaud_nathalie=None, zemmour_eric=None, le_pen_marine=46.0), "2022-01-05", 1, []),
        ({"melenchon_jean_luc": 30.0, "philippe_edouard": 30.0, "le_pen_marine": 40.0}, "2022-01-05", 1,
         ["incomplete_hypothesis"]),
        (first(arthaud_nathalie=None, jadot_yannick=1.0), "2022-04-05", 1, ["undeclared_candidate"]),
        ({"philippe_edouard": 55.0, "le_pen_marine": 45.0}, "2022-04-15", 2, []),
        ({"hollande_francois": 55.0, "le_pen_marine": 45.0}, "2022-04-15", 2, ["undeclared_candidate"]),
        ({"hollande_francois": 55.0, "le_pen_marine": 45.0}, "2022-03-15", 2, []),
        ({"unknown_person": 55.0, "le_pen_marine": 45.0}, "2021-03-15", 2, ["undeclared_candidate"]),
    ],
)
def test_suspect_reasons(shares, start, round_, expected):
    assert _reasons(shares, start, round_) == expected


def test_suspect_flag_in_results():
    w = src("wikipedia", [poll("w1", "wikipedia", "2026-03-01", "2026-03-02")],
            hyp("w1", first(le_pen_marine=40.0)) + hyp("w1", first(arthaud_nathalie=None, le_pen_marine=36.0)))
    res = run(wikipedia=w)
    flags = res.results.unique(["hypothesis_id"]).select("suspect", "suspect_reason").rows()
    assert sorted(flags, key=str) == sorted([(True, "sum_out_of_range"), (False, None)], key=str)


# ---------------------------------------------------------------- tables dérivées


BLOCS = {"arthaud_nathalie": "far_left", "melenchon_jean_luc": "far_left", "glucksmann_raphael": "left",
         "philippe_edouard": "centre", "retailleau_bruno": "right", "le_pen_marine": "far_right",
         "zemmour_eric": "far_right", "hollande_francois": "left"}


def bloc_of(cid, election):
    return BLOCS[cid]


def _toy():
    """Élection jouet 2022 (tour 1 le 10 avril) ; résultat officiel = `first()` avec Le Pen à 37
    et Zemmour à 8."""
    w_polls = [
        poll("a1", "wikipedia", "2022-03-01", "2022-03-02", pollster="ifop", election=2022),
        poll("a2", "wikipedia", "2022-04-04", "2022-04-05", pollster="ifop", election=2022),
        poll("b1", "wikipedia", "2022-04-01", "2022-04-03", pollster="elabe", election=2022),
        # rolling : deux vagues qui se chevauchent -> une seule compte dans les moyennes
        poll("r1", "wikipedia", "2022-04-02", "2022-04-05", pollster="opinionway", election=2022),
        poll("r2", "wikipedia", "2022-04-03", "2022-04-06", pollster="opinionway", election=2022),
        # sondage ancien sans la bonne configuration (Hollande au lieu de Glucksmann)
        poll("h1", "wikipedia", "2021-10-10", "2021-10-11", pollster="ifop", election=2022),
    ]
    rows = (
        hyp("a1", first(le_pen_marine=33.0, zemmour_eric=12.0))
        + hyp("a2", first())
        + hyp("a2", first(zemmour_eric=None, le_pen_marine=45.0))  # hypothèse sans Zemmour
        + hyp("b1", first(le_pen_marine=36.0, zemmour_eric=9.0))
        + hyp("r1", first(le_pen_marine=34.0, zemmour_eric=11.0))
        + hyp("r2", first(le_pen_marine=38.0, zemmour_eric=7.0))
        + hyp("h1", first(glucksmann_raphael=None, hollande_francois=12.0))
    )
    res = run(wikipedia=src("wikipedia", w_polls, rows, rolling={"r1", "r2"}))
    official = pl.DataFrame(
        [(2022, 1, c, s) for c, s in first(le_pen_marine=37.0, zemmour_eric=8.0).items()],
        schema={"election": pl.Int32, "round": pl.Int8, "candidate_id": pl.Utf8, "share": pl.Float64},
        orient="row",
    )
    return res, official


def test_results_by_bloc():
    res, _ = _toy()
    bb = derived.results_by_bloc(res.polls, res.results, bloc_of)
    a2 = bb.filter((pl.col("poll_id") == "2022-ifop-20220405-1")
                   & (pl.col("hypothesis_id") == schemas.hypothesis_id(FIRST)))
    got = dict(a2.select("bloc", "share").iter_rows())
    assert got == {"far_left": 14.0, "left": 12.0, "centre": 20.0, "right": 9.0, "far_right": 45.0}
    assert dict(a2.select("bloc", "n_candidates").iter_rows())["far_right"] == 2


def test_poll_errors_toy_election():
    res, official = _toy()
    err = derived.poll_errors(res.polls, res.results, official, {(2022, 1): D("2022-04-10")}, bloc_of,
                              horizons=(7, 180))

    def get(method, unit, horizon=None, pollster=None):
        f = (pl.col("method") == method) & (pl.col("unit") == unit)
        if horizon is not None:
            f &= pl.col("horizon_days") == horizon
        if pollster is not None:
            f &= pl.col("pollster") == pollster
        return err.filter(f).row(0, named=True)

    # dernier sondage de chaque institut, configuration exacte préférée (a2 et non a1)
    last = get("last_by_pollster", "le_pen_marine", pollster="ifop")
    assert (last["poll_id"], last["poll_share"], last["error"]) == ("2022-ifop-20220405-1", 35.0, -2.0)
    assert get("last_by_pollster", "far_right", pollster="ifop")["error"] == 0.0
    assert get("last_by_pollster", "le_pen_marine", pollster="opinionway")["poll_share"] == 38.0
    # 14 derniers jours : a2 (35), b1 (36), une seule vague rolling (r2, 38)
    m = get("mean_14d", "le_pen_marine")
    assert m["n_polls"] == 3 and m["poll_share"] == pytest.approx((35 + 36 + 38) / 3)
    assert m["exact_config"] and m["config_distance"] == 0
    # horizon 7 : fenêtre ]2 mars, 3 avril] -> b1 seul (a1 finit le 2 mars, exclu)
    h7 = get("horizon", "zemmour_eric", horizon=7)
    assert (h7["n_polls"], h7["poll_share"], h7["error"]) == (1, 9.0, 1.0)
    # horizon 180 : seule l'hypothèse la plus proche (Hollande), signalée
    h180 = get("horizon", "left", horizon=180)
    assert (h180["exact_config"], h180["config_distance"], h180["poll_share"]) == (False, 2, 12.0)
    assert err.filter((pl.col("horizon_days") == 180) & (pl.col("unit") == "glucksmann_raphael")).is_empty()
