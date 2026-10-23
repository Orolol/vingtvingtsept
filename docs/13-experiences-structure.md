# 13 — Cinq expériences de structure

Protocole déclaré le 23 octobre 2026, avant calcul, sur la branche
`codex/prediction-structure`. Étude exploratoire : les mêmes trois élections ont déjà
servi à proposer ces pistes. Aucun défaut publié ni fichier de la référence figée ne change.
Le verrou numérique est `data/registry/structure_study_2026-10-23.yaml`.

## Comparaison et règles communes

Référence : gel `reference-2026-10-22`, douze dates 2012/2017/2022 × 180/90/30/7 jours.
Les scénarios, ajustements bayésiens, résultats et fonctions de score du gel sont repris.
Les calibrations ne voient que les scrutins terminés avant la date évaluée. Empreintes
des entrées et sorties, reprise par étape, CPU uniquement. Les données supplémentaires
sont copiées dans le worktree et identifiées dans le manifeste.

Log-score de la paire et du vainqueur, CRPS, Brier et couvertures sont tous publiés,
par élection et horizon. Les diagnostics sur sondages ne remplacent jamais le score
électoral. Aucun composant n'est admis automatiquement : la règle à deux niveaux de
`07-evaluation.md` reste nécessaire. Les douze dates ne font pas douze élections.

## 1. Mélange des prévisions

Mélange exact à poids 1/2 des distributions empiriques complètes de la référence et
de `mean14d`. Les 80 000 trajectoires sont conservées, sans moyenner leurs parts ligne
à ligne. Pour comparer le log-score lissé, le pseudo-effectif de Jeffreys reste celui
du gel (40 000), pas 80 000. Aucun poids appris sur les trois élections.

## 2. Incertitude de calibration

Bootstrap bayésien de 32 pondérations Dirichlet des **scrutins entiers** antérieurs.
Les horizons et candidats d'un scrutin partagent son poids ; son poids total initial
et le poids 0,5 des scrutins de liste sont conservés en espérance. Réestimation des
cinq paramètres d'échelle et des poids de bloc avec la pénalisation du gel. Les biais,
la forme internationale gamma et les degrés de liberté restent ceux de la calibration
de référence : cette approximation ne représente pas toute l'incertitude des paramètres.
Une pondération est tirée par trajectoire simulée, commune à tous ses candidats.
Tout échec d'optimisation est explicite. Aucune inflation réglée après lecture des scores.

## 3. Dépendance des sondages continus

Comparaison de trois moyennes sur 14 jours : GLS sur vagues éclaircies, GLS conservant
toutes les vagues, GLS avec recouvrement de terrain. Pour chaque institut en rolling,
la corrélation d'échantillonnage supposée est la fraction normalisée de jours communs
(renouvellement uniforme). Plancher indépendant 1/2500 et facteur de design 0,6.
Ce sont des hypothèses de sensibilité, pas des taux de répondants communs observés.
Recouvrement principal 1 ; sensibilités 0 et 0,5, sans sélection du meilleur.
Toutes les autres étapes de la chaîne et le traitement des candidats anciens sont
identiques. Pas de remplacement de l'agrégateur bayésien par cette étude limitée.
Horizons d'action : 30 et 7 jours ; les horizons longs restent publiés comme contrôles.

## 4. Substitution entre blocs

Petit mélange de deux logits emboîtés, préférences sur un axe de blocs fixé à l'avance
(−2, −1, 0, 1, 2 ; other = 0), de même poids et décalages opposés ±tau. Les paramètres
lambda intra-bloc sont appris sur les seules élections d'apprentissage ; un seul tau
supplémentaire, entre 0 et 2, pénalité 0,01 tau² sur la perte moyenne par sondage.
À tau=0, le modèle redonne exactement la substitution intra-bloc.
Les lambda sont bornés à 0,02 pour l'inversion numérique, dans le témoin comme dans
la variante ; cette convention est fixée avant calcul.

Pour une paire d'hypothèses emboîtées du même sondage, inverser les parts de l'hypothèse
large pour ses utilités, puis prédire la réduite. Les observations réduites du test
n'entrent pas dans l'inversion. Poids égal par sondage, réparti entre ses paires.
Validation leave-one-election-out et, séparément, apprentissage sur les élections
antérieures. Ni les paires ni les candidats d'un sondage ne sont répartis entre plis.
Scores : entropie croisée et erreur absolue, par élection ; tau, convergence et effectifs.
Un gain sur deux élections évaluables au moins, dans les deux protocoles et sur les deux
scores, est requis avant de développer un réajustement complet. Sinon le pilote s'arrête
avec résultat négatif ou données insuffisantes. Les marges agrégées n'identifient pas
des reports individuels ; cette restriction est conservée dans la conclusion.

## 5. Transitions des indécis

Audit des microdonnées déjà accessibles : au moins deux vagues d'une même série,
identifiant de panel explicitement validé, intentions réellement mesurées avant le
vote et au moins 100 personnes appariées. Refus et questions non posées ne sont pas
recodés en indécision. Les souvenirs de vote ne deviennent pas des intentions passées.
Dates de disponibilité nécessaires à une admission chronologique.

Si les données existent : transitions pondérées (poids de la vague initiale), effectif
de Kish, régularisation Dirichlet de force 10 vers la distribution d'arrivée globale,
validation sur une vague ultérieure ou une autre élection, sans découper les répondants
d'une même transition. Si elles manquent : audit chiffré, contrat d'entrée, estimateur
testé sur données synthétiques ; aucun coefficient appris ni score électoral inventé.
La collecte externe et les demandes d'accès existantes ne sont pas renouvelées.

## Sources méthodologiques

- [Gneiting et Ranjan : combinaison de distributions](https://arxiv.org/abs/1106.1638).
- [Train : modèles de choix mixtes](https://eml.berkeley.edu/choice2/ch6.pdf).
- [Bon, Ballard et Baffour : indécis](https://arxiv.org/abs/1703.09430).

Les résultats et les limites d'exécution seront ajoutés après les calculs.
