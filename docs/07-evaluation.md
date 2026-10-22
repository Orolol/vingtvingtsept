# 07 — Évaluation et backtest

## Principe

L'objectif est de juger les prévisions avec l'information disponible à la date `asof`.
Deux protocoles sont désormais distingués : les études historiques en validation croisée
par élection, et la consolidation chronologique décrite dans
[10-consolidation.md](10-consolidation.md).

Les premières études excluent l'élection évaluée de la calibration, mais peuvent utiliser
des élections ultérieures, filtrent surtout sur la fin du terrain et emploient parfois des
correspondances de candidats connues après le scrutin. Leurs résultats restent valables
comme comparaisons rétrospectives ; ils ne constituent pas une reproduction aveugle de
l'information disponible à l'époque.

La consolidation retire aussi les élections futures de **toutes** les calibrations actives,
applique un filtre de publication (date manquante = fin du terrain + 2 jours), reconstruit
les configurations sans substitution de candidats après coup, puis simule le vainqueur
sans imposer les finalistes observés. Les classifications et bases rétrospectives restent
des limites explicites. Elle ne change pas la règle d'admission ci-dessous après lecture
des scores, ni le statut des options déjà publiées.

## Protocole des études rétrospectives

- **Élections** : 2012, 2017, 2022 en priorité (séries de sondages denses) ; 2007 et 2002
  si la couverture Wikipédia le permet, avec un poids réduit dans la synthèse.
- **Horizons** : 180, 90, 30, 7 jours avant le premier tour, et la veille du second tour pour
  le modèle de reports.
- **Scénarios** : les études historiques comportent des évaluations conditionnelles aux
  candidats finalement présents et des correspondances de candidats après primaire.
  Elles sont à distinguer du rejeu sans correspondance ex post de la consolidation.
- **Validation croisée** : leave-one-election-out pour tout paramètre estimé sur l'histoire
  (modèle d'erreur, facteur de design, biais de bloc, coefficients économiques et presse).

## Métriques

| Métrique | Sur quoi |
|---|---|
| Log-score | paire de qualifiés, vainqueur final |
| Brier | qualification de chaque candidat, victoire |
| CRPS et erreur absolue | parts de premier tour par candidat, parts de second tour |
| Couverture | fraction des résultats dans les intervalles à 50 % et 90 % |
| Score par bloc | erreurs agrégées par bloc, pour voir où ça casse |

Les métriques sont calculées par élection et par horizon, puis moyennées avec une
pondération explicite. On publie le tableau complet, pas seulement la moyenne.

## Règle d'admission d'une variable

Une variable (économique, presse, ou autre) entre dans la prévision si, et seulement si :

1. elle améliore le log-score moyen sur les paires de qualifiés et le CRPS des parts aux
   horizons où elle agit par construction, en leave-one-election-out : 180 et 90 jours pour
   une variable de niveau ou de dynamique longue (économie, presse, attention), 30 et 7 jours
   pour un effet de fin de campagne (vote utile, rebond récent), et elle ne dégrade aucun
   des autres horizons ;
2. le gain est de même signe sur au moins deux des trois élections de référence ;
3. elle ne dégrade pas la couverture des intervalles ;
4. le mécanisme est plausible et documenté dans son fichier de spec.

Sinon, elle reste dans le volet explicatif du site, et le résultat négatif est consigné dans
le document de la source.

## Comparaisons de référence

- Moyenne simple des sondages des 14 derniers jours (baseline naïve).
- Dernier sondage de chaque institut, moyenné.
- Le modèle sans prior structurel, sans effets maison, pour mesurer l'apport de chaque brique.

## Sorties

`runs/<date>/backtest/` : tableau des métriques par élection, horizon et version ;
trajectoires des probabilités rejouées ; figures de calibration. Un résumé est copié dans
`docs/backtest-log.md` à chaque changement de modèle, avec le commit.

La consolidation écrit séparément `runs/2026-10-22-consolidation/` : copie figée des
entrées, manifests et empreintes, douze ajustements, tirages du premier tour jusqu'à la
victoire, scores par candidat et par scrutin, diagnostics d'inférence et de repondération,
audit des sources de calibration et analyses de sensibilité. Le témoin `mean14d` partage
les scénarios et le second tour avec la référence : il teste l'apport de l'agrégateur.
Les moyennes donnent un poids égal aux trois élections ; les quatre horizons d'une même
élection ne sont pas quatre observations indépendantes.

L'[étude de reconstruction historique](11-scenarios-historiques.md) réutilise ce gel et
ses ajustements pour comparer deux autres lois de scénarios, avec les mêmes scores et
les deux agrégateurs. Le protocole a été déclaré avant ses calculs ; le défaut initial
ayant été observé sur les mêmes élections, cette étude reste exploratoire. Aucun score
ni fichier de la référence n'est remplacé. Les deux variantes ne sont pas admises.
