"""Adaptateur des sondages des européennes 2019, page « Sondages sur les élections européennes de 2019 ».

Pas de page « Liste de sondages sur les élections européennes de 2019 en France » ; l'article
de l'élection n'a que les derniers sondages (son HTML, téléchargé d'abord, reste dans
data/raw, ignoré par `latest_raw`). Page européenne, section « À l'échelle nationale >
France » (révision 234700270, téléchargée le 2026-10-20) :

- t24 « Sondages réalisés après le dépôt des listes » (une colonne par liste déposée, sigles :
  `PP - PS ND - PRG` = Envie d'Europe, `GÉ MEI - MDP` = Urgence écologie, `LREM - MoDem Agir -
  MR` = Renaissance, `LR LC` = Union de la droite et du centre, `LP` = Les Patriotes, `SIEL
  PI` = La Ligne claire, `PP` = Parti pirate, `AJ` = Alliance jaune, `NA`, `AE`, `PACE`,
  `PRC`, `UDMF`) et t25 « avant le dépôt » (colonnes `NPA`, `Agir`, `RES` pour Lassalle et
  `GJ` pour une liste gilets jaunes générique, sans liste déposée correspondante) ;
- en-tête `Échan- tillon` coupé (lu comme l'échantillon) ; dates avec année ; « <0,5 % »
  vaut 0,25 ; commanditaire entre parenthèses dans la cellule institut (alias de lists.yaml) ;
- tableaux des autres pays et projections en sièges ignorés.

Hypothèses écartées sur la révision figée (18, toutes avant le dépôt des listes) :

- 12 où LO et NPA partagent une cellule (Elabe, Ipsos pour LREM, Harris, Ifop) : liste
  commune jamais déposée, le NPA n'ayant pas présenté de liste ;
- 3 où Agir et UDI partagent une cellule (Ipsos, Ifop-Fiducial, Ifop, été 2018) ;
- 3 où plusieurs listes de gauche partagent une cellule (Ifop pour Place publique, Viavoice
  pour EELV) : hypothèses de liste d'union.
"""

from __future__ import annotations

from vvs.ingest.wikipedia.scrutins import ListPageAdapter


class AdapterEuro2019(ListPageAdapter):
    election = "euro2019"
    round1_tables = r"^a l echelle nationale france sondages en des voix"
    absent_texts = ()
