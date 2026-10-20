"""Adaptateur des sondages des européennes 2014, article « Élections européennes de 2014 en France ».

Pas de page « Liste de sondages » : la série nationale est la section « Sondages > Au niveau
national » de l'article (révision 240479484, téléchargée le 2026-10-20).

- un seul tableau lu (t2), légendé par des lignes d'évènement « mai 2014 », « janvier 2014 -
  avril 2014 », « 2013 » qui donnent l'année ; dates sans année (`21 - 23 mai`), résolues en
  remontant depuis le vote ; pas d'échantillon ;
- colonnes par parti (LO, NPA, FG, EELV, PS, PRG, ND, Cap21, MoDem, UDI, NC, UMP, PCD, DLR,
  FN, Autres) ; les listes communes sont des cellules fusionnées : PS | PRG (« Choisir notre
  Europe »), MoDem | UDI (« Les Européens »), parfois LO | NPA en 2013 ; MoDem et UDI
  séparés en 2013 sont des composantes de `udi_modem_2014` (lists.yaml) ;
- `N.t.` (non testé) vaut absent ; `NC` est Nous Citoyens (Denis Payre), pas « non connu » ;
- lignes ignorées : « Résultat final », « Élections de 2009 » (rappel des résultats) ;
- tableaux ignorés : sondages par circonscription (t3-t5), prévisions en sièges, résultats.
- les résultats du ministère ne sont que par nuance (LFN, LUMP, LUG…) : DLR, Nous Citoyens et
  le PCD forment l'unité divers droite, Nouvelle Donne et Cap21 l'unité résiduelle (lists.yaml).

Aucune hypothèse écartée sur la révision figée (54 sondages, dont 27 vagues du rolling Ifop
détectées par la curation).
"""

from __future__ import annotations

from vvs.ingest.wikipedia.scrutins import ListPageAdapter


class AdapterEuro2014(ListPageAdapter):
    election = "euro2014"
    round1_tables = r"^sondages au niveau national$"
    absent_texts = ("n.t.",)
