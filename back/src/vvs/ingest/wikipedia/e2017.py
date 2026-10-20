"""Adaptateur de la page « Liste de sondages sur l'élection présidentielle française de 2017 ».

Structure de la page (révision 240308996, téléchargée le 2026-10-20) :

- le second tour vient AVANT le premier tour sur la page ;
- second tour : t0 sondages entre les deux tours (colonnes `Abstention` et
  `Indécis (échantillon)`), t1 reports de voix (ignoré), t2 Macron – Le Pen avant le premier
  tour, t3–t7 hypothèses abandonnées et t8–t18 hypothèses d'avant les primaires (2015-2016)
  dans des blocs repliables ; ces tableaux ont une colonne `Abstention, blanc ou nul` ;
- premier tour : t19 après la publication de la liste officielle (dates `21 avr.` sans
  année), puis un tableau par mois (légende « Mars 2017 », « Décembre 2016 »…, t20–t33) où
  les dates n'ont souvent ni année ni mois (`du 17 au 19`, `le 20`) : le mois et l'année
  viennent de la légende ; t34–t37 un tableau par année (« Sondages de 2015 »…, dates
  complètes) ;
- en-têtes de candidats en nom complet avec parti : `Nathalie Arthaud ( LO )` ;
- colonne `Autres candidats` au format nom puis valeur (`Alliot-Marie : 1 %`) ;
- hypothèses Juppé en remplacement de Fillon : `20 % Juppé` dans la colonne Fillon ;
- cellules fusionnées verticalement à l'intérieur de la zone des candidats (une même valeur
  pour plusieurs hypothèses), et lignes qui ne portent que institut/date/échantillon, les
  hypothèses étant sur les lignes suivantes (5 lignes) ;
- libellés d'institut avec partenaire : `Ifop-Fiducial`, `Cevipof Ipsos-Sopra Steria`,
  `Kantar Sofres - OnePoint`, `Scan Research - Le Terrain` (sponsor tiré de la table
  d'alias) ; un lien mort rend `« Ipsos » (Archive.org • Wikiwix • Que faire ?)`, nettoyé ;
- lignes de résultats officiels `Résultats` et `1 er tour` (dans un tableau de second
  tour) : sautées.

Choix et limites :

- `Abstention, blanc ou nul` est rangé sous l'id réservé `abstention` (base
  `all_respondents`) : approximation, la page ne distingue pas les trois ;
- `Indécis` et `Indécis (échantillon)` vont dans `undecided` (base `all_respondents`) ;
- valeur réservée illisible écartée sans perdre l'hypothèse : BVA, terrain terminé le
  2 mai 2017, abstention `20-24 %` (fourchette) ;
- tableaux ignorés : reports de voix (t1) et sous-échantillons régionaux, étrangers et
  sociologiques (t38–t46) ;
- aucune hypothèse écartée sur la révision figée.
"""

from __future__ import annotations

from datetime import date

from vvs.ingest.wikipedia.engine import PageAdapter


class Adapter2017(PageAdapter):
    election = 2017
    round1_date = date(2017, 4, 23)
    round2_date = date(2017, 5, 7)
    earliest = date(2012, 5, 7)
