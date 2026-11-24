# Duel Le Pen – Mélenchon : d'où vient l'écart entre les sondages et la projection

Diagnostic du 2026-11-24 (sous-agent Opus, lecture seule), sur la projection publiée le même
jour (`runs/2026-11-24-simulate-subst/`, commit `4c07d85`, code du second tour identique à
`ce1fcd7`). Aucune variante ci-dessous n'est passée au backtest : ce sont des décompositions et
des sensibilités, pas des prévisions.

## Question

Les sondages du duel donnent Le Pen à 67 % en moyenne, la projection publie une médiane de
63,1 % [56,8 – 69,2]. Le biais RN appliqué à cet horizon ne vaut que 1,3 point.

## Rejouer

```bash
cd back && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" uv run vvs simulate --asof 2026-11-24 --fit ../runs/2026-11-24-fit-2027-subst --out ../runs/2026-11-24-duel-diagnostic/rerun --seed 0 --substitution --rn-bias-curve
cd back && JAX_PLATFORMS=cpu uv run python ../runs/2026-11-24-duel-diagnostic/duel_diag.py
```

Puis `duel_diag_extra.py` et `duel_diag_combo.py` (graine 20261124, 22 400 tirages du duel,
5 tirages de sortie par tirage de premier tour). Le rejeu de référence donne 63,05
[56,84 – 69,21] contre 63,08 [56,85 – 69,23] publié. Sorties : `results.json`,
`results_extra.json`, `results_combo.json` (dossier `runs/`, hors git).

## Décomposition (médianes, part de Le Pen en points d'exprimés)

| Étape | Valeur | Δ |
|---|---|---|
| Moyenne simple des 33 lignes de sondage | 67,22 | |
| 26 sondages retenus (2024 exclus par âge, OpinionWay réduit à une base) | 67,28 | +0,06 |
| Moyenne GLS : la covariance d'horizon privilégie les sondages récents | 66,16 | −1,12 |
| Biais RN de la courbe (1,29 point) | 64,87 | −1,29 |
| Prior des reports (médiane 53,97, écart-type 9,3), premier tour fixé | 60,64 | −4,23 |
| Rééchantillonnage du premier tour par les sondages de duel | 63,05 | +2,41 |
| Bruit Monte-Carlo | 63,08 | +0,03 |

Contrôle gaussien : la vraisemblance des sondages (écart-type 4,17) pèse 0,83, le prior (9,31)
0,17 ; postérieur attendu 63,0, écart-type 3,8, conforme à l'observé.

L'effet de sélection (le duel n'a lieu que si Mélenchon est haut) est faible : −0,35 point.
Aucun scénario de candidats ne porte l'écart (médiane de 62,4 à 63,6 sur 45 conditions).

## Ce que valent les 26 sondages

La variance de la moyenne est de 17,41 (écart-type 4,17), dont 16,92 d'erreur d'horizon commune
(σ0 = 1,35, σ1 = 1,65 par racine de mois, 164 jours). Vingt-six sondages ne valent donc guère
mieux qu'un seul. Taille effective des poids GLS : 4,5 sondages ; le Harris du 19 novembre pèse
0,38 à lui seul, les cinq sondages de novembre 0,89. L'échantillonnage d'importance ne dégénère
pas (taille effective médiane 0,34 par tirage candidat).

## Reports a posteriori contre reports publiés

Seul Harris publie le détail des reports (notices de la Commission des sondages
`data/raw/commission/2321.pdf`, `2351.pdf`, `2381.pdf`, `2417.pdf`, quatre vagues d'octobre et
de novembre 2026, base « certains d'aller voter ayant exprimé une intention »).

| Électorat | Modèle a posteriori (LP / JLM / ni) | Harris, moyenne des vagues |
|---|---|---|
| Glucksmann (bloc gauche) | 1 / 76 / 18 | 11 / 27 / 62 |
| Philippe, Attal (bloc centre) | 27 / 24 / 42 | 30 / 10 / 60 et 31 / 15 / 54 |
| Retailleau, Lisnard (bloc droite) | 66 / 3 / 27 | 63 / 3 / 34 et 65 / 8 / 27 |
| Riner (autres) | 45 / 18 / 30 | 33 / 20 / 47 (une vague) |
| Mélenchon | 2 / 94 / 3 | 3 / 92 / 5 |
| Le Pen | 94 / 2 / 3 | 97,5 / 1 / 1,5 |
| Abstentionnistes | 16 / 7 / 73 | non mesuré |

Deux valeurs du modèle ne tiennent pas : la gauche vers Mélenchon (76 contre 27) et le centre
vers Mélenchon (24 contre 10 à 15). Les reports a posteriori restent à peu près le prior : le
décalage commun `u` ne peut pas corriger un groupe dont le prior est extrême, et l'ajustement se
fait sur les abstentionnistes et sur le premier tour.

## Anomalies

1. **Bug de cohérence.** `sample_pair` rééchantillonne les tirages de premier tour avec les
   poids des sondages de duel, et `simulate` recolle le résultat sur des lignes dont le premier
   tour n'a pas bougé. Ici le second tour est calculé avec Le Pen à 36,5 % au premier tour au
   lieu de 30,8 % (Mélenchon 18,1 au lieu de 22,3), soit +2,4 points. Correctif et mesure :
   `docs/results/2026-11-24-premier-tour-fixe.md`.
2. **Matrice de proximité mal centrée** pour un duel extrême droite contre gauche radicale
   (ligne gauche : attirance 7 pour la gauche radicale ; ligne centre : 2,5 contre 2). Avec les
   reports Harris, le prior seul passe de 54,0 à 59,9.
3. **Présentation.** L'export affiche des reports de bloc sous des noms de candidats (Philippe
   et Attal identiques, Lisnard et Retailleau aussi) ; ce sont presque des hypothèses du prior.
4. **Covariance d'horizon en marche aléatoire** entre sondages (−1,1 point ici) et effet maison
   au plancher de 0,5 alors que YouGov (63) et Elabe (68,5) diffèrent de 5,5 points la même
   semaine. Choix de modèle, à tester.
5. **σ1 fait toute la largeur de l'intervalle** (avec σ1 = 0 : 4,5 points de large). Calibré sur
   peu de points, dont l'erreur de 9,4 points de 2012 à 180 jours.
6. Mineurs : fidélité des électeurs de Le Pen à 94 % contre 97–98 % chez Harris ; report des
   abstentionnistes sans aucune donnée de contrôle ; `sample_size` est l'échantillon total, pas
   la base du second tour (effet négligeable).

Contrôles sans anomalie : unités, base des exprimés du second tour, signe du biais, absence de
doublon d'hypothèse.

## Sensibilités (une chose changée à la fois, référence 63,08)

| Variante | Médiane |
|---|---|
| Sans biais RN | 64,09 |
| Biais constant 2,97 | 61,69 |
| Moyenne des sondages sans pondération de récence | 64,14 |
| Premier tour fixé (correctif du bug) | 60,64 [51,9 – 68,0] |
| Prior recentré sur les reports Harris | 64,19 |
| Prior Harris + premier tour fixé | 63,10 [54,8 – 70,2] |
| Vraisemblance seule | 64,87 [58,0 – 71,7] |
| Prior seul | 53,9 [38,3 – 68,7] |

## Non vérifié

Flux aléatoire du second tour non reproduit ligne à ligne (loi concordante à 0,03 point) ;
sensibilité de σ1 à l'exclusion de 2012 non testée ; rattachement des électorats Harris aux
blocs du modèle fait à la main.
