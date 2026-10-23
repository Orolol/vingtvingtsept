# 14 — Cinq expériences de structure

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

## Résultats

**Aucune admission, aucun changement de prévision publiée.** Les cinq pistes ont été
traitées : trois familles comparées dans la chaîne complète, un pilote de substitution
arrêté au seuil annoncé, un estimateur de transitions prêt mais sans panel exploitable.
Le [bilan versionné](results/structure-2026-10-23/README.md) contient les scores par cas,
les comparaisons avec chaque témoin, les paramètres et les diagnostics.

Écarts moyens à la référence, poids égal aux trois élections. Un log-score plus haut
est préférable, un CRPS plus bas est préférable. La couverture à 90 % reste inchangée
pour le mélange et la calibration rééchantillonnée.

| Variante | Horizon | Δ log victoire | Δ log paire | Δ CRPS |
|---|---:|---:|---:|---:|
| Mélange 50/50 | 180 | −0,0756 | −0,0635 | +0,0053 |
| Mélange 50/50 | 90 | +0,0041 | +0,0061 | −0,0113 |
| Mélange 50/50 | 30 | −0,0040 | +0,0003 | +0,0220 |
| Mélange 50/50 | 7 | +0,0165 | +0,0410 | +0,0315 |
| Calibration rééchantillonnée | 180 | +0,0044 | +0,0103 | +0,0042 |
| Calibration rééchantillonnée | 90 | −0,0007 | −0,0004 | −0,0038 |
| Calibration rééchantillonnée | 30 | −0,0058 | −0,0132 | +0,0024 |
| Calibration rééchantillonnée | 7 | −0,0008 | −0,0061 | +0,0003 |

Le mélange peut améliorer un événement tout en dégradant les parts ; il ne domine
pas les deux composants. Pas de recherche a posteriori d'un poids dépendant de l'horizon.
Le bootstrap de calibration a un effet faible et partagé. Il conserve les biais, gamma,
nu et les pénalisations du gel : ce résultat négatif ne réfute pas tout modèle bayésien
de calibration. Les petits écarts sont à lire avec le bruit Monte-Carlo d'une seule graine.

### Sondages continus

Contre l'agrégateur de référence, la variante avec recouvrement gagne en log-score de
paire à J−7 (+0,0880), mais dégrade le CRPS (+0,0994). Cette comparaison change aussi
l'agrégateur : elle ne mesure pas seule l'apport du recouvrement.

Contre le **même GLS sur vagues éclaircies**, les effets sont beaucoup plus petits :

| Horizon | Δ log paire | Δ CRPS |
|---|---:|---:|
| 180 | 0 | 0 |
| 90 | −0,0023 | +0,0071 |
| 30 | +0,0057 | +0,0006 |
| 7 | −0,0007 | +0,0172 |

Contre le GLS gardant toutes les vagues comme indépendantes, le recouvrement réduit
le CRPS à J−7 de 0,0075 point mais perd en log-score de paire à J−30 (−0,0172).
Les sensibilités 0, 0,5 et 1 sont toutes conservées. Pas de gain admissible.

Le gel ne contient aucune vague marquée rolling en 2012, ni à J−180 dans les trois
élections. Le pilote ne corrige pas ces métadonnées historiques : sa portée se limite
aux vagues reconnues en 2017 et 2022. La covariance suppose un renouvellement uniforme
des répondants, non vérifié dans les notices ; ce n'est pas une reconstruction certaine.
Le GLS autorise de petits poids négatifs (minimum observé −0,00043) ; les parts sont
contrôlées et aucune troncature silencieuse n'a été appliquée.

### Substitution entre blocs

58 sondages fournissent des paires emboîtées : 29 en 2017, 29 en 2022 ; aucun en
2002/2007/2012 dans les entrées figées. Sur tous les plis, l'amplitude supplémentaire
tau retenue est **zéro**. L'erreur absolue moyenne est donc identique au témoin :
0,654 point en 2017 et 0,754 en 2022 en leave-one-election-out ; 0,660 et 0,754 en
chronologique. Le premier pli chronologique ne dispose d'aucune paire d'apprentissage,
ce qui est explicitement enregistré, sans emprunt aux élections futures.

Le critère de passage annoncé échoue : **pas de réajustement NUTS ni de backtest électoral
pour cette variante**. Le résultat porte sur ce petit mélange symétrique et son axe fixé,
pas sur tous les modèles de préférences possibles. Ces données agrégées restent trop
limitées pour identifier des reports individuels.

### Indécis et abstentionnistes

L'inventaire local répertorie 80 jeux Cevipof, dont deux possèdent une alternative de
microdonnées publique : FES 2017 et FES ELIPSS 2022 vague 3. Cette dernière est une
vague isolée dont le terrain chevauche le premier tour ; elle ne fournit pas à elle
seule des transitions chronologiques. L'état des autorisations n'a pas été interrogé
à nouveau sur l'entrepôt : cet audit décrit les données déjà disponibles localement.

La table curatée contient **1 830 répondants, tous en FES 2017 post-second tour**,
avec souvenirs de vote ; **zéro ligne d'intention** et aucune paire de vagues utilisable.
L'estimateur de transitions est implémenté et testé (appariement explicitement validé,
poids, effectif de Kish, régularisation, refus exclus), mais aucun coefficient réel ni
score électoral n'est produit. L'étape suivante requiert des vagues d'intentions
appariables et leurs dates de disponibilité ; aucun accès ni donnée n'est inventé.

## Exécution et validation

- Douze cas, huit distributions par cas : 96 lignes de scores ; 384 optimisations
  de calibration convergées. Les distributions du mélange contiennent 80 000 trajectoires,
  celles des autres variantes 40 000 ; le pseudo-effectif de score reste 40 000 partout.
- Les 409 fichiers de référence sont inchangés. Le témoin `mean14d` est rejoué avant
  chaque variante et comparé à ses tirages figés (tolérance numérique 10⁻¹²).
- Les résultats portent sur le niveau chronologique : les variantes échouent déjà à
  ce niveau nécessaire. Aucun gain rétrospectif supplémentaire n'est recherché pour
  contourner cet échec ; aucune option n'a satisfait la règle d'admission à deux niveaux.
- Tests après intégration des ajouts récents de main : 771 réussis, un test ignoré
  (dépendance front absente du worktree), quatre
  avertissements préexistants. Contrôles Ruff, syntaxe shell et reprise vérifiée.
- Deux défauts de préparation ont été corrigés : appel à un vérificateur absent du
  code ancien, puis types entier/décimal à l'assemblage des CSV. L'inventaire a également
  été corrigé pour inclure les formats alternatifs publics. Les calculs finaux ont été
  refaits dans `runs/2026-10-23-structure-v2`, avec protocole statistique inchangé.

Reproduction : `bash scripts/run_structure_study.sh`, puis la commande du bilan versionné.
Le script s'appuie sur un gel déjà présent ; `VVS_STRUCTURE_REFERENCE` permet d'en indiquer
un autre emplacement. Ses empreintes et versions doivent correspondre au verrou. Il
refuse une reprise si les sources ou les résultats ont changé, et ne relance pas les
étapes terminées. Les tirages complets restent hors Git dans `runs/`.
