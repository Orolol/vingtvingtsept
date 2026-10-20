"""Adaptateur de la page « Liste de sondages sur l'élection présidentielle française de 2002 ».

Structure de la page (révision 236399441, téléchargée le 2026-10-20) :

- premier tour : un tableau par mois de 2002 (titres « 2002 » > « Avril »… « Janvier »,
  t0–t3), puis un tableau par année (« 2001 », « 2000 », « 1999 », t4–t6) ; dates sans
  année (`17 et 18 avril`, `du 13 au 15 avril`, `du 28 février au 2 mars`), l'année vient
  du titre de section ;
- en-têtes de candidats en nom complet avec parti (`Arlette Laguiller ( LO )`) sur trois
  lignes (portrait, nom, bandeau de couleur) ; la liste des colonnes change d'un tableau à
  l'autre (Pasqua jusqu'en mars 2002, Gluckstein et Saint-Josse selon les mois, Voynet et
  de Villiers dans l'unique sondage de 1999) ;
- pas de colonne d'échantillon ni de colonnes réservées (abstention, indécis) :
  `sample_size` est nul, `population` vaut `unknown` ; une ligne = un sondage = une
  hypothèse ; `-` = candidat non testé (absent, pas 0) ;
- second tour : t7 Chirac – Le Pen entre les deux tours (dates complètes, dont les
  sondages du soir du 21 avril), t8–t9 Chirac – Jospin, hypothèse abandonnée testée par
  TNS Sofres seule (`17- 18 avril`, année tirée du titre « 2002 » ou « 2001 ») ;
- libellés d'institut : `Sofres` (premier tour) et `TNS Sofres` (second tour) vont tous deux
  sous `sofres`, de sorte que 21 sondages Sofres réunissent leurs deux tours ;
  `Louis Harris` va sous `lh2` (ex-Louis Harris France, distinct de Harris Interactive) ;
- « Le Pen » est Jean-Marie (lien /wiki/Jean-Marie_Le_Pen, alias de la section 2002) ;
- lignes `Résultats officiels` (premier et second tour) : sautées.

Lignes écartées sur la révision figée : aucune. Deux hypothèses Ifop (21-22 février et
28 février - 1er mars 2002) somment à 96 et 96,5 : gardées, la curation les marquera
`suspect`.
"""

from __future__ import annotations

from datetime import date

from vvs.ingest.wikipedia.engine import PageAdapter


class Adapter2002(PageAdapter):
    election = 2002
    round1_date = date(2002, 4, 21)
    round2_date = date(2002, 5, 5)
    earliest = date(1995, 5, 8)
