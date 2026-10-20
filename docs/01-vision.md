# 01 — Vision

## Objectif

Produire et publier une projection probabiliste de l'élection présidentielle française
de 2027 (premier tour le 18 avril 2027, second tour le 2 mai, dates fixées en Conseil des
ministres le 1er juillet 2026 ; décret de convocation attendu avant le 7 février 2027), mise à jour à chaque nouveau sondage, avec :

- la probabilité de qualification au second tour de chaque candidat ;
- la probabilité de victoire finale ;
- la distribution des scores de premier tour et des duels de second tour ;
- la décomposition par scénario de candidats (qui se présente ?) ;
- un volet explicatif : ce que disent l'économie et la presse de la dynamique.

Le modèle est inspiré de FiveThirtyEight mais adapté aux particularités françaises :
deux tours, dix à douze candidats, configuration de candidats incertaine jusqu'en mars,
sondages nationaux par quotas sans détail géographique.

## Ce qu'on peut honnêtement prédire

- **Dernière semaine** : les sondages français sont bons sur les gros candidats (erreur
  typique de 1 à 3 points). Le modèle apporte surtout la gestion des cas serrés (2e contre 3e
  et 4e en 2017 et 2022) et des reports de second tour.
- **À trois mois** : la structure des blocs est fiable, l'ordre entre candidats proches ne
  l'est pas. Les sondages de second tour hypothétiques ont une erreur historique de 5 à
  10 points.
- **À six mois et plus** : on mesure la taille des blocs et l'incertitude sur les
  configurations. Octobre 2016 ne voyait ni Fillon ni Macron ; octobre 2021 voyait Zemmour
  à 17 %. Le modèle doit afficher cette incertitude, pas la masquer.

La valeur du projet est de quantifier proprement l'incertitude et de la voir se resserrer,
pas de produire un point estimé précoce.

## Principes

1. **Les sondages sont la colonne vertébrale.** Tout le reste (prior structurel, économie,
   presse) module la trajectoire ou l'incertitude, et seulement s'il a prouvé son apport.
2. **Calibration empirique.** L'incertitude vient de l'erreur observée des sondages sur
   2002 à 2024, par bloc et corrélée, gonflée selon l'horizon. Pas de marge d'erreur
   théorique d'échantillonnage.
3. **Scénarios explicites.** L'incertitude sur les candidats est une couche à part, avec
   des probabilités versionnées et justifiées, et des prévisions conditionnelles lisibles.
4. **Backtest avant croyance.** Toute variable passe le protocole de `07-evaluation.md`.
5. **Reproductible et publiable.** Un run = une commande, des sorties JSON datées, un site
   statique régénéré.

## Hors périmètre pour l'instant

- Législatives (modèle de sièges par circonscription, règles de second tour, désistements).
  Prévu en seconde étape, les données de résultats collectées serviront.
- Modélisation infranationale par MRP : les sondages publiés n'ont pas de détail
  géographique exploitable ; on y reviendra si une source le permet.
- Marchés de prédiction comme entrée du modèle : on pourra les afficher en comparaison.
