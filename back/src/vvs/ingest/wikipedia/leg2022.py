"""Adaptateur de la page « Liste de sondages sur les élections législatives françaises de 2022 ».

Révision 239948712 (téléchargée le 2026-10-20). Un seul tableau lu : t0 « Sondages à
l'échelle nationale > Premier tour » ; les projections en sièges, les sondages par
circonscription ou par région et les tableaux sociologiques sont ignorés.

- entités : nuances et coalitions (`LO - NPA`, `NUPES` sur quatre sous-colonnes PCF, LFI,
  EELV, PS, `DVG - FGR`, `ECO`, `DVC`, `Ensemble`, `UDC ( LR - UDI )`, `DVD`, `UPF ( DLF -
  LP )`, `RN`, `REC`, `PA`, `REG`, `Autres`, `DIV`) ; une cellule NUPES sur les quatre
  sous-colonnes vaut la coalition (en-tête commun) ; Ipsos, Cluster17, Harris et Elabe
  fusionnent souvent UDC et DVD (unité `udc_dvd_2022`) ;
- « Autres » porte parfois un nom (`1 % DSV`, `1,5 % DSV - EXD`, `0,5 % LMR`) ;
- dates sans année ; premier sondage national le 9 mai 2022 (après l'accord NUPES) : aucun
  sondage aux horizons 90 et 180 jours ;
- lignes d'évènement (formation des coalitions, nomination d'Élisabeth Borne) sautées.

Hypothèses écartées sur la révision figée (8, fin avril et début mai 2022, avant ou pendant la
formation des coalitions) : cellule couvrant NUPES et DVG, PS et DVG, ou Ensemble et UDC
(hypothèses de blocs de Harris et Cluster17).
"""

from __future__ import annotations

from vvs.ingest.wikipedia.scrutins import ListPageAdapter


class AdapterLeg2022(ListPageAdapter):
    election = "leg2022"
    round1_tables = r"^sondages a l echelle nationale premier tour$"
    absent_texts = ()
