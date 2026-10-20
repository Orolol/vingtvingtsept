"""Adaptateur de la page « Liste de sondages sur l'élection présidentielle française de 2012 ».

Structure de la page (révision 238741176, téléchargée le 2026-10-20) :

- premier tour : titres « Premier tour » > « Année 2012 » > « Avril 2012 »… (t0–t3),
  « Année 2011 » > « Quatrième trimestre 2011 »… (t4–t7), « Année 2010 » (t8),
  « Années 2007-2009 » (t9, dates complètes) ; ailleurs un seul jour sans année
  (`17 avril`) : `fieldwork_start` = `fieldwork_end` ; pas d'échantillon ni de colonnes
  réservées ;
- colonnes génériques `Candidat PS`, `Candidat NPA`, `Candidat EELV`, `Candidat FN` dont la
  cellule porte le nom testé (`24 % Aubry`, `<1 % Poutou`, `11 % M. Le Pen`) ; en-têtes
  nominatifs annotés « Sauf précision » dont la cellule peut nommer un autre candidat
  (`1 % Arthaud` sous Laguiller, `20 % Aubry` sous Royal, `5 % Duflot` sous Voynet) ;
- hypothèses multiples d'un même sondage (Aubry, Hollande, Royal, Strauss-Kahn) sur des
  lignes à institut et date fusionnés ;
- « Candidat FN » de 2007-2009 : `J-M. Le Pen` (lien /wiki/Jean-Marie_Le_Pen) et
  `M. Le Pen` (lien /wiki/Marine_Le_Pen, Ifop 30 octobre 2009) ; « Le Pen » seul serait
  Marine (alias de la section 2012) ;
- second tour : t10–t11 Hollande – Sarkozy par année, t12–t22 hypothèses abandonnées
  (Aubry, Strauss-Kahn, Royal, Fillon, Delanoë, Bayrou, Marine Le Pen) ; deux duels
  répétés à l'identique sur deux lignes (LH2 7 mai 2011, Harris 5 juin 2011), comptés
  une fois ; 68 sondages réunissent leurs deux tours ;
- libellés d'institut : coquille `Ispos` (alias vers `ipsos`), `Harris` (Harris
  Interactive), `LH2`, `Sofres`.

Tableau ignoré : t6 « Second trimestre 2011 ». La colonne « Jean-Pierre Chevènement » y
porte des scores de 5 à 9 % qui sont ceux de Nicolas Hulot : la révision 79265234 (30 mai
2012) donne, pour Ifop 20-21 avril 2011, « Strauss-Kahn 30, Sarkozy 22,5, Le Pen 19,
Hulot 8,5 » là où la page actuelle met 8,5 sous Chevènement ; Ipsos 18-20 juin 2011
« Aubry 30… Hulot 7 » devient Chevènement 7 ; Ifop 21-23 juin 2011 testait Hulot seul
(6,5) puis Joly seule (6,5), la page actuelle donne « Joly 1,5 » dans les deux lignes.
D'autres valeurs diffèrent (Le Pen 20,5 contre 19 chez Ifop) et le tableau a perdu
plusieurs sondages de la période (CSA 26 avril, LH2 6-7 mai…). On ne réattribue pas la
colonne : le tableau entier est écarté (25 hypothèses).

Lignes écartées sur la révision figée : Harris 16 décembre 2011, somme 94,75 (Chevènement,
Lepage, Morin, Villepin et Boutin à `—`) : hors de `share_sum_bounds`. Six hypothèses de
novembre 2011 à mars 2012 somment entre 96 et 97 : gardées, la curation les marquera
`suspect`.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

from vvs.ingest.wikipedia.engine import PageAdapter


class Adapter2012(PageAdapter):
    election = 2012
    round1_date = date(2012, 4, 22)
    round2_date = date(2012, 5, 6)
    earliest = date(2007, 5, 7)
    skip_tables: ClassVar[dict[str, str]] = {
        **PageAdapter.skip_tables,
        r"second trimestre 2011": (
            "tableau corrompu : la colonne « Jean-Pierre Chevènement » porte les scores de "
            "Nicolas Hulot (5 à 9 %), et les hypothèses Joly/Hulot ne correspondent plus à la "
            "révision 79265234 de mai 2012"
        ),
    }
