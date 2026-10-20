"""Adaptateur de la page « Liste de sondages sur l'élection présidentielle française de 2007 ».

Structure de la page (révision 233109259, téléchargée le 2026-10-20) :

- premier tour : un tableau par mois de 2007 (titres « 2007 » > « Avril »… « Janvier »,
  t0–t3), puis « 2006 » (t4) et « 2005 » (t5) ; la date est un seul jour sans année
  (`20 avril`, `1er mars`) : `fieldwork_start` = `fieldwork_end`, sans qu'on sache s'il
  s'agit de la fin de terrain ou de la publication ;
- en-têtes en nom complet avec parti (`Ségolène Royal ( PS )`), pas d'échantillon ni de
  colonnes réservées ; Hulot apparaît en janvier 2007 (t3), Fabius, Lang et Hollande en
  2005 (t5) ;
- Ifop 20-21 janvier 2005 (Paris Match) : trois hypothèses de premier tour (un candidat
  PS chacune) sur trois lignes non fusionnées, et trois duels contre Sarkozy dans les
  tableaux t10–t12 : un seul sondage ;
- second tour : t6–t7 Sarkozy – Royal par année (dates complètes `4 mai 2007`), t8–t12
  « Autres hypothèses » (Bayrou – Sarkozy, Bayrou – Royal, Fabius, Hollande, Lang contre
  Sarkozy) avec des plages `18 avril - 19 avril 2007` ; les notes de ces lignes sont des
  références de presse, sans effet ;
- les dates du premier et du second tour coïncident d'un tableau à l'autre : 95 sondages
  réunissent leurs deux tours ;
- `1.5 %` (point décimal) lu comme 1,5 ; `<0,5 %` vaut 0,25 (Nihous, Sofres 18 janvier) ;
- « Le Pen » est Jean-Marie (lien /wiki/Jean-Marie_Le_Pen, alias de la section 2007) ;
  `Sofres` → `sofres`, `LH2` → `lh2`.

Lignes écartées sur la révision figée (2 hypothèses de premier tour) :

- Sofres 24 mars 2007 : Sarkozy à `38 %`, somme 110 (coquille probable pour 28 %, non
  corrigée) : hors des bornes `share_sum_bounds` ;
- CSA 21 février 2007 : deux lignes de même institut et même jour aux parts différentes
  (29/28 contre 29/29 pour Royal/Sarkozy) ; rien ne dit s'il s'agit de deux sondages ou
  d'une erreur : la première est gardée, la seconde journalisée (« hypothèse en double »).

Ifop 18 novembre 2006 somme à 96 : gardé, la curation le marquera `suspect`.
"""

from __future__ import annotations

from datetime import date

from vvs.ingest.wikipedia.engine import PageAdapter


class Adapter2007(PageAdapter):
    election = 2007
    round1_date = date(2007, 4, 22)
    round2_date = date(2007, 5, 6)
    earliest = date(2002, 5, 6)
