"""Libellés de candidats du ministère -> candidate_id du registre.

Le libellé source est « NOM Prénom » tel que les fichiers l'écrivent, normalisé par
`normalize_label` (ASCII, majuscules, espaces simples) : « MÉLENCHON Jean-Luc » et
« MELENCHON JEAN-LUC » donnent tous deux « MELENCHON JEAN-LUC ». Un libellé absent de la
table fait échouer le parsing : on ajoute la ligne ici, on ne devine pas.
"""

from __future__ import annotations

import unicodedata

CANDIDATE_IDS: dict[str, str] = {
    # 2002
    "MEGRET BRUNO": "megret_bruno",
    "LEPAGE CORINNE": "lepage_corinne",
    "GLUCKSTEIN DANIEL": "gluckstein_daniel",
    "BAYROU FRANCOIS": "bayrou_francois",
    "CHIRAC JACQUES": "chirac_jacques",
    "LE PEN JEAN-MARIE": "le_pen_jean_marie",
    "TAUBIRA CHRISTIANE": "taubira_christiane",
    "SAINT-JOSSE JEAN": "saint_josse_jean",
    "MAMERE NOEL": "mamere_noel",
    "JOSPIN LIONEL": "jospin_lionel",
    "BOUTIN CHRISTINE": "boutin_christine",
    "HUE ROBERT": "hue_robert",
    "CHEVENEMENT JEAN-PIERRE": "chevenement_jean_pierre",
    "MADELIN ALAIN": "madelin_alain",
    "LAGUILLER ARLETTE": "laguiller_arlette",
    "BESANCENOT OLIVIER": "besancenot_olivier",
    # 2007
    "BUFFET MARIE-GEORGE": "buffet_marie_george",
    "SCHIVARDI GERARD": "schivardi_gerard",
    "BOVE JOSE": "bove_jose",
    "VOYNET DOMINIQUE": "voynet_dominique",
    "DE VILLIERS PHILIPPE": "de_villiers_philippe",
    "ROYAL SEGOLENE": "royal_segolene",
    "NIHOUS FREDERIC": "nihous_frederic",
    "SARKOZY NICOLAS": "sarkozy_nicolas",
    # 2012
    "JOLY EVA": "joly_eva",
    "LE PEN MARINE": "le_pen_marine",
    "MELENCHON JEAN-LUC": "melenchon_jean_luc",
    "POUTOU PHILIPPE": "poutou_philippe",
    "ARTHAUD NATHALIE": "arthaud_nathalie",
    "CHEMINADE JACQUES": "cheminade_jacques",
    "DUPONT-AIGNAN NICOLAS": "dupont_aignan_nicolas",
    "HOLLANDE FRANCOIS": "hollande_francois",
    # 2017
    "MACRON EMMANUEL": "macron_emmanuel",
    "HAMON BENOIT": "hamon_benoit",
    "LASSALLE JEAN": "lassalle_jean",
    "ASSELINEAU FRANCOIS": "asselineau_francois",
    "FILLON FRANCOIS": "fillon_francois",
    # 2022
    "ROUSSEL FABIEN": "roussel_fabien",
    "ZEMMOUR ERIC": "zemmour_eric",
    "HIDALGO ANNE": "hidalgo_anne",
    "JADOT YANNICK": "jadot_yannick",
    "PECRESSE VALERIE": "pecresse_valerie",
}


class UnknownCandidateError(ValueError):
    pass


def normalize_label(last_name: str, first_name: str) -> str:
    raw = f"{last_name} {first_name}"
    ascii_ = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_.upper().split())


def candidate_id(label: str) -> str:
    try:
        return CANDIDATE_IDS[label]
    except KeyError:
        raise UnknownCandidateError(
            f"libellé candidat inconnu : {label!r} (ajouter la ligne dans "
            "vvs/ingest/interieur/candidates.py)"
        ) from None
