"""Duels des législatives : lecture des fichiers du ministère, partis des candidats d'union,
types de duel (vvs.ingest.legislative_duels)."""

import polars as pl
import pytest

from vvs.ingest import legislative_duels as L

T1_2024 = (
    "Code département;Libellé département;Code circonscription législative;Libellé;Inscrits;"
    "Votants;% Votants;Abstentions;% Abstentions;Exprimés;% Exp/ins;% Exp/vot;Blancs;% B/i;"
    "% B/v;Nuls;% N/i;% N/v;Numéro de panneau 1;Nuance candidat 1;Nom candidat 1;"
    "Prénom candidat 1;Sexe candidat 1;Voix 1;% Voix/inscrits 1;% Voix/exprimés 1;Elu 1\n"
    "1;Ain;101;1ère circonscription;1000;700;70%;300;30%;680;68%;97%;15;1,5%;2%;5;0,5%;1%;"
    "1;RN;DUPONT;Jean;MASCULIN;300;30%;44%;;2;UG;MARTIN;Anne;FEMININ;200;20%;29%;;"
    "3;ENS;DURAND;Paul;MASCULIN;130;13%;19%;;4;LR;PETIT;Luc;MASCULIN;50;5%;7%;\n"
)
T2_2024 = (
    '"Code département";"Libellé";"Code circ";"Libellé circ";Inscrits;Votants;"%";Abst;"%";'
    'Exprimés;"%";"%";Blancs;"%";"%";Nuls;"%";"%";"P1";"N1";"Nom1";"Pr1";"S1";"V1";"%";"%";"E1"\n'
    '"01";Ain;"0101";"1ère circonscription";1002;690;"69%";312;"31%";650;"65%";"94%";30;"3%";'
    '"4%";10;"1%";"1%";1;RN;DUPONT;Jean;MASCULIN;340;"34%";"52%";élu;2;UG;MARTIN;Anne;FEMININ;'
    '310;"31%";"48%";;;;;;;;;;\n'
)


def test_parse_2024_files_and_build_duel():
    r = pl.concat([
        L.parse_results_text(T1_2024, "leg2024_circ", "leg2024", 1),
        L.parse_results_text(T2_2024, "leg2024_circ", "leg2024", 2),
    ])  # fmt: skip
    # le code « 1 » du T1 et « 01 » du T2 désignent la même circonscription
    assert set(r["circ_code"]) == {"01-01"}
    parties = pl.DataFrame({"election": ["leg2024"], "circ_code": ["01-01"], "panel": [2],
                            "party": ["lfi"], "party_source": ["test"], "party_check": ["agree"],
                            "party_note": [""]})  # fmt: skip
    _, duels = L.build_tables(r, parties)
    d = duels.row(0, named=True)
    assert d["duel_type"] == "rn_lfi"
    assert d["fr_votes1"] == 300 and d["opp_votes1"] == 200
    assert d["votes1_centre"] == 130 and d["votes1_right"] == 50
    assert d["rn_share2"] == pytest.approx(100 * 340 / 650)
    # ENS (130 ≥ 12,5 % de 1000 inscrits) qualifié mais absent du second tour : désistement
    assert d["triangular_withdrawn"] and d["withdrawn_nuances"] == "ENS"


def test_vote_sum_gap_is_bounded():
    bad = T1_2024.replace(";680;68%", ";600;68%")
    with pytest.raises(L.SourceError):
        L.parse_results_text(bad, "leg2024_circ", "leg2024", 1)


@pytest.mark.parametrize(
    "text, code",
    [
        ("LFI", "lfi"), ("LFI - PEPS", "lfi"), ("PS", "ps"), ("PCF", "pcf"), ("LÉ", "ecologists"),
        ("EÉLV", "ecologists"), ("G.s", "generation_s"), ("GRS", "other_left"), ("", "unknown"),
    ],
)  # fmt: skip
def test_party_code(text, code):
    assert L.party_code(text) == code


def test_department_page_row():
    page = """{| class="wikitable"
|-
| {{Infobox Parti politique français/couleurs|PS}} |
| align="left" |Sébastien Guéraud
| align="left" |{{nowrap|[[Parti socialiste (France)|PS]] ([[Nouvelle Union populaire écologique et sociale|NUPES]])}}
| align="center" |{{Abréviation discrète|NUP|Nouvelle Union Populaire Ecologique et Sociale}}
| {{formatnum:9982}}
| 23,87
|-
| {{Infobox Parti politique français/couleurs|LFI}} |
| align=left | Anne Martin
| align=left | [[La France insoumise|LFI]] <small>([[Nouveau Front populaire|NFP]])</small>
| align=center | {{Abrd|UG|Union de la gauche}}
| '''{{formatnum:14188}}'''
| 23,45
|}"""
    rows = L.parse_department_page(page)
    assert [(r["nuance"], L.party_code(r["party_text"]), r["votes1"]) for r in rows] == [
        ("NUP", "ps", 9982), ("UG", "lfi", 14188)]


@pytest.mark.parametrize(
    "election, fr, opp, party, expected",
    [
        ("leg2024", "UXD", "UG", "lfi", "rn_lfi"),
        ("leg2024", "RN", "UG", "pcf", "rn_left_other"),
        ("leg2022", "RN", "NUP", "unknown", "rn_left_unknown"),
        ("leg2024", "RN", "UDI", None, "rn_centre"),
        ("leg2022", "RN", "UDI", None, "rn_right"),
        ("leg2024", None, "UG", "lfi", "no_far_right"),
    ],
)  # fmt: skip
def test_duel_type(election, fr, opp, party, expected):
    assert L.duel_type(election, fr, opp, party) == expected


def test_allocation_fixes():
    text = "Circonscription,Etiquette\n49-01,PS\n49-01,FI\n64-01,PS\n99-01,PE\n"
    a = L._read_allocation(text)
    assert a["49-01"] == "ps" and a["49-02"] == "lfi"  # doublon 49-01 = 49-02
    assert a["64-01"] == "lfi"  # investiture finale FI
    assert a["ZZ-01"] == "ecologists"
