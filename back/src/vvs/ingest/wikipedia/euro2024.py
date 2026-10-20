"""Adaptateur des sondages des européennes 2024, page « Sondages sur les élections européennes de 2024 ».

Comme en 2019, la série française complète est dans la page européenne, section « Sondages >
Par pays > France » (révision 236003076, téléchargée le 2026-10-20) :

- t32 « après le dépôt des listes » (`RES` y porte le score d'Alliance rurale, 2,35 % au
  résultat ; `LP` = L'Europe ça suffit, Philippot ; `UDMF` = Free Palestine ; `ND` = Changer
  l'Europe ; `PRG - RPS` = Europe territoires écologie ; `EAC`, `EP` écologistes) et t33
  « avant le dépôt » (colonnes `AR` et `RES` séparées, `TEM - AC` et `DLF` sans liste
  déposée) ;
- avant le dépôt, Ifop, Ipsos, Elabe et Cluster17 fusionnent LO et NPA : unité extrême gauche
  `exg_eu2024` (lists.yaml) ;
- noms de têtes de liste en cellule : `Hayer` vaut la liste Besoin d'Europe ; `Denormandie`,
  `Beaune`, `Le Maire`, `Séjourné`, `Breton` une liste de la majorité hypothétique
  (`ensemble_alt_eu2024`), pour que l'hypothèse Hayer d'un même sondage reste distincte ;
- t34 « Sondages testant une liste d'union de la gauche » ignoré (NUPES jamais déposée).

Hypothèses écartées sur la révision figée (9, toutes de 2023 ou de février 2024) :

- 5 au libellé ambigu (lists.yaml, `ambiguous`) : « Royal » et « Toussaint » têtes d'une liste
  d'union sur plusieurs colonnes, « Aubry » tantôt LFI tantôt union ;
- Ipsos juin 2023 : EELV et PS-PP dans une même cellule ;
- 3 doublons de configuration (même sondage, mêmes listes, parts différentes) : hypothèses
  de têtes de liste de la majorité (Ifop octobre 2023, Ipsos novembre 2023, Ifop février
  2024) ; la première est gardée.
"""

from __future__ import annotations

from typing import ClassVar

from vvs.ingest.wikipedia.scrutins import ListPageAdapter


class AdapterEuro2024(ListPageAdapter):
    election = "euro2024"
    round1_tables = r"^sondages par pays france"
    absent_texts = ()
    skip_tables: ClassVar[dict[str, str]] = {
        **ListPageAdapter.skip_tables,
        r"liste d union de la gauche": (
            "hypothèses d'une liste d'union de la gauche (NUPES), jamais déposée ; têtes de "
            "liste nommées dans les cellules"
        ),
    }
