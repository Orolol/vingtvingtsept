# 07 — Évaluation et backtest

## Principe

Toute version du modèle, et toute variable candidate, est jugée sur des prévisions faites
« comme à l'époque » : à la date `asof`, seuls les sondages et données publiés avant cette
date sont visibles, et le modèle d'erreur est calibré sans l'élection évaluée.

## Protocole

- **Élections** : 2012, 2017, 2022 en priorité (séries de sondages denses) ; 2007 et 2002
  si la couverture Wikipédia le permet, avec un poids réduit dans la synthèse.
- **Horizons** : 180, 90, 30, 7 jours avant le premier tour, et la veille du second tour pour
  le modèle de reports.
- **Scénarios** : pour le passé, on rejoue avec les configurations connues à chaque date
  (par exemple avant et après la primaire de la droite en novembre 2016), pour ne pas donner
  au modèle une information qu'il n'avait pas.
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
   horizons 180 et 90 jours, en leave-one-election-out ;
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
