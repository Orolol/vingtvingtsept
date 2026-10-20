"""Adaptateur de la page « Liste de sondages sur l'élection présidentielle française de 2022 ».

Structure de la page (révision 240448433, téléchargée le 2026-10-20) :

- premier tour : t0 après la publication de la liste officielle (dates `7-8 avril` sans
  année, résolues près du 10 avril 2022), puis des tableaux par période dans des blocs
  repliables, légendés « Mars 2022 », « Décembre 2021 », « Octobre-novembre 2021 »… (t1–t8),
  et « Sondages effectués de 2017 à 2020 » (t9–t10, dates complètes) ;
- en-têtes de candidats en nom seul avec parti : `Arthaud ( LO )` ; colonnes génériques
  `Candidat LR`, `Candidat LR / SL / DVD`, `Candidat PS / DVG` dont la cellule porte le nom
  testé (`14 % Bertrand`) ; colonne `Autres` (`2 % Thouy`) ;
- note « candidature simultanée de Xavier Bertrand et de Valérie Pécresse » : la cellule
  porte les deux noms (`11 % Bertrand 8 % Pécresse`), les deux sont gardés ;
- instituts « (rolling) » (Ifop, Ipsos, OpinionWay) : gardés comme sondages de l'institut ;
  le schéma n'a pas de drapeau rolling, le recouvrement des terrains est à traiter en
  curation ; coquille `Harris Interractive` résolue par alias ;
- second tour : t11 après le premier tour, t12 Macron – Le Pen avant le premier tour,
  t14–t24 hypothèses abandonnées (dates complètes) ; t13 reports de voix (ignoré) ;
- t25–t27 intentions par critères sociologiques (ignorés).

Lignes écartées sur la révision figée (33 hypothèses de premier tour) :

- 32 hypothèses Ipsos et Ifop de 2021 dont une cellule porte la note « candidature
  d'extrême gauche » ou « candidature écologiste » « sans mention nominative » : la part ne
  désigne aucun candidat (la cellule `EXG` couvre d'ailleurs les colonnes Arthaud et
  Poutou) ;
- Ifop 26-27 octobre 2021 : `12,5 %` dans la colonne `Candidat LR` sans nom.
"""

from __future__ import annotations

from datetime import date

from vvs.ingest.wikipedia.engine import PageAdapter


class Adapter2022(PageAdapter):
    election = 2022
    round1_date = date(2022, 4, 10)
    round2_date = date(2022, 4, 24)
    earliest = date(2017, 5, 8)
