"""Adaptateur de la page « Liste de sondages sur l'élection présidentielle française de 2027 ».

Page vivante : la description vaut pour la révision 240522402 (téléchargée le 2026-10-20).

- premier tour : un tableau par période, dont l'année vient du titre de section
  (« Octobre-décembre 2026 ») ou du bloc repliable (« Sondages réalisés en 2025 ») ; dates
  sans année (`14-16 octobre`) ;
- en-têtes `Glucksmann ( PP )` ou `Candidat EPR` sur deux sous-colonnes (`colspan=2`) :
  une cellule qui couvre les deux est le candidat de l'en-tête ; quand la cellule est
  scindée, la sous-cellule nommée (`4 Hollande ( PS )`) est ce candidat et l'autre le
  candidat de l'en-tête ;
- colonnes génériques `Candidat RN`, `Candidat EPR`, `Candidat LR`, `Candidat PS / PP` :
  le nom est dans la cellule (`33 Bardella`) ;
- colonne `Autres` à plusieurs candidats (`4 Lisnard 2 Ruffin 1 Villepin`) ;
- notes « Sondage commandé par X » sur l'institut : `sponsor` (Picardie Debout, Sébastien
  Bonnal, Parti du vote blanc) ;
- section « Autres » : t6 « Sondages comptant le vote blanc comme un vote exprimé »
  (ignoré, base non standard ; l'hypothèse standard du même sondage est dans le tableau
  principal) et t7 « Sondages impliquant Emmanuel Macron, qui ne peut plus se présenter »
  (gardé : hypothèses réelles, filtrées plus tard par les scénarios) ;
- second tour : un tableau par duel (`Hypothèse Attal – Le Pen`…), dates complètes ; les
  duels contre Bardella sont dans des blocs repliables.

Lignes écartées sur la révision figée :

- Harris 24-25 avril 2025 (3 hypothèses) : note « Aucun nom de candidat n'est mentionné
  dans ce sondage » sur l'institut ;
- OpinionWay 10-11 juin 2026 (3 hypothèses) : `34 Bardella / Le Pen`, candidat RN non
  tranché (libellé ambigu).

Les dates des tours sont provisoires : elles ne servent qu'à borner la résolution des
années.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

from vvs.ingest.wikipedia.engine import PageAdapter


class Adapter2027(PageAdapter):
    election = 2027
    round1_date = date(2027, 4, 18)  # annoncé en Conseil des ministres, décret non vu
    round2_date = date(2027, 5, 2)  # idem
    earliest = date(2022, 4, 25)
    skip_tables: ClassVar[dict[str, str]] = {
        **PageAdapter.skip_tables,
        r"vote blanc comme un vote exprime": (
            "vote blanc compté comme exprimé : base non standard, l'hypothèse standard du "
            "même sondage figure dans le tableau principal"
        ),
    }
