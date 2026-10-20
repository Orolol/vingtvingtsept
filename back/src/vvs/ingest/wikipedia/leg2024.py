"""Adaptateur de la page « Liste de sondages sur les élections législatives françaises de 2024 ».

Révision 240550959 (téléchargée le 2026-10-20). Campagne de trois semaines (dissolution le
9 juin, vote le 30). Un seul tableau lu : t0 « Sondages à l'échelle nationale > Premier tour
> Sondages nationaux » ; les sondages par circonscription, les projections de second tour et
en sièges sont ignorés.

- en-têtes à deux niveaux : `NFP` (LFI, PCF, EELV, PS), `ENS` (RE, MoDem, HOR, PRV, UDI),
  `RN et alliés` (LR (Ciotti), RN) ; une cellule sur toutes les sous-colonnes vaut
  l'en-tête (`nfp_2024`, `ensemble_2024`) ; RN et Ciotti fusionnés (Ifop, Elabe, OpinionWay,
  Odoxa) valent `rn_allies_2024`, DVD et LR fusionnés (Ipsos, Elabe, Ifop) `lr_dvd_2024` ;
- dates avec année ; sondages hypothétiques de 2022-2023 (configuration de 2022) en bas.

Hypothèses écartées sur la révision figée (19) :

- 18 où LR et LR (Ciotti) partagent une cellule : tous les sondages d'avant la scission de LR
  (10-14 juin 2024) et les sondages hypothétiques de 2022-2023 ; ces électorats vont ensuite
  dans deux unités différentes (droite et RN et alliés), on ne peut pas les répartir ;
- Ifop 10-11 juin : PCF, EELV et PS dans une cellule, LFI à part (listes séparées).

Il reste 23 sondages du 17 au 28 juin : seul l'horizon 7 jours (et la moyenne des 14
derniers jours) existe.
"""

from __future__ import annotations

from vvs.ingest.wikipedia.scrutins import ListPageAdapter


class AdapterLeg2024(ListPageAdapter):
    election = "leg2024"
    round1_tables = r"^sondages a l echelle nationale premier tour sondages nationaux$"
    absent_texts = ()
