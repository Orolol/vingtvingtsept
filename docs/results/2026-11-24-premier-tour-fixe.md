# Second tour à premier tour fixé (`fixed_first_round`) — 24 novembre 2026

## Protocole et critères (écrits le 2026-11-24, avant tout calcul)

**Défaut corrigé.** `sample_pair` rééchantillonne conjointement, dans une cellule
(configuration, paire), les tirages de premier tour et les reports, pondérés par les
sondages de duel. Les appelants affectent ensuite les `n` parts de second tour obtenues
aux `n` lignes de la cellule, dont le premier tour n'a pas été rééchantillonné : la loi
jointe premier tour / second tour d'une ligne est incohérente (diagnostic :
`runs/2026-11-24-duel-diagnostic/`). Le mode `fixed_first_round` garde le premier tour de
chaque ligne et ne tire que les reports : k = 64 tirages du prior des reports par ligne,
pondérés par la même log-vraisemblance (sondages de duel, plafonds, couplage, participation),
un seul retenu. C'est un correctif de cohérence, pas une variable nouvelle.

**Bras.** Les douze cas de la référence figée (`runs/2026-10-22-consolidation`, 2012, 2017,
2022 × 180, 90, 30, 7 jours) : mêmes `case.json`, entrées `curated/`, ajustements
`fit/draws.npz`, spécification `reference.json` (40 000 tirages, graine 20261022, second
tour `bias_by_horizon`), calibrations recalculées et comparées à `calibration.json` du gel.
Agrégateur bayésien (méthode `reference`) seulement. Code : arbre de travail courant, avec
`VVS_ROOT` pointé sur la copie figée des registres et tables.

- **R0** : référence, `fixed_first_round=False`, graine 20261022. Contrôle de protocole :
  ses tirages doivent reproduire `draws_reference.npz` du gel ; sinon l'écart est documenté
  et la comparaison se fait contre R0 recalculé (même code pour tous les bras).
- **V0** : variante `fixed_first_round=True`, graine 20261022, tout le reste identique.
- **R1** : référence, graine 20261023 : mesure du bruit Monte-Carlo.

La chaîne tire tout le premier tour avant le second : R0 et V0 partagent donc exactement
leurs tirages de premier tour et leurs paires de qualifiés.

**Métriques**, par cas, puis moyenne par horizon (poids égal aux trois élections) et
moyenne des douze cas :

- `log_score_winner` (lissage de Jeffreys du gel) ;
- `log_score_pair` : contrôle, doit être identique entre R0 et V0 ; de même le CRPS et la
  couverture du premier tour ;
- `crps2` et `coverage90_2` : CRPS et couverture de l'intervalle central à 90 % de la part
  de second tour du duel réellement disputé, sur les tirages où ce duel est qualifié
  (diagnostic conditionnel au duel réalisé, séparé du score de victoire, docs/10). Un cas
  où ce duel n'a aucun tirage dans un bras est exclu de ces deux moyennes pour tous les
  bras, et signalé.

**Bruit Monte-Carlo.** Pour une métrique M : d_c = M(R1, c) − M(R0, c) sur les douze cas,
σ̂ = √(moyenne des d_c²) (écart-type de la différence entre deux runs indépendants d'un
cas), SE = σ̂ / √12 pour la moyenne des douze cas (σ̂ / √3 pour une moyenne par horizon).
R1 change aussi le premier tour alors que V0 le partage avec R0 : cette barre est donc
large pour la différence V0 − R0. Elle est conservée telle quelle.

**Verdict (barre « ne dégrade pas »).** La variante est admise comme correctif si, sur la
moyenne des douze cas :

1. moyenne(V0 − R0) de `log_score_winner` ≥ −2 SE ;
2. moyenne(V0 − R0) de `crps2` ≤ +2 SE ;
3. |couverture90_2(V0) − 0,90| ≤ |couverture90_2(R0) − 0,90| + 2 SE (SE de la couverture
   calculé comme ci-dessus).

Sinon le résultat est rapporté tel quel, l'option reste désactivée et la note explique
pourquoi. Les moyennes par horizon sont publiées sans entrer dans le verdict. Quel que soit
le verdict, aucun défaut ne change dans ce travail : l'option est livrée désactivée ;
l'activer dans la projection publiée est une décision séparée.

**Annexe (sensibilité, pas un critère).** Projection du 2026-11-24 rejouée avec
`vvs simulate --asof 2026-11-24 --fit ../runs/2026-11-24-fit-2027-subst --seed 0
--substitution --rn-bias-curve`, avec et sans `--fixed-first-round`, sorties sous
`runs/2026-11-24-premier-tour-fixe/` : probabilités de victoire par candidat et part
médiane des dix duels les plus probables.

## Résultats (calculés après l'écriture des critères ci-dessus)

Sorties : `runs/2026-11-24-premier-tour-fixe/` (hors Git). Chaîne : `chain/` (manifeste avec
empreintes des sources, `metrics.csv`, `results.json`, tirages et cellules par cas). Durée
réelle : huit minutes pour les douze cas et les quatre bras, sur un seul processus, dont
l'essentiel pour le bras à 1024 propositions ; aucune réduction des tirages.

**Contrôles.** Le rejeu R0 reproduit à l'identique les tirages `draws_reference.npz` du gel
dans les douze cas (toutes les colonnes) : le code courant, refactorisé, ne change rien au
mode conjoint. Entre R0 et V0, `log_score_pair`, CRPS et couverture du premier tour sont
identiques (écart maximal 0). Les calibrations recalculées sont égales à celles du gel.

**Bras de contrôle ajouté après coup.** Le premier calcul de V0 a émis des avertissements
de taille effective par ligne (jusqu'à 14 % des tirages de 2012-180 avec moins de deux
propositions effectives). J'ai donc ajouté, **après lecture de ces avertissements et des
premiers scores**, un bras V0_k1024 (même variante, 1024 propositions par ligne au lieu de
64) pour mesurer l'erreur d'approximation de k = 64. Il n'entre pas dans le verdict, qui
porte sur V0 comme déclaré. Les bras R0, V0 et R1 ont été recalculés à l'identique dans
le même run, puis tout le run a été refait une dernière fois avec les sources finales
(changements de docstrings seulement) : `metrics.csv` identique octet pour octet.

### Tableau principal (moyennes à poids égal par élection)

| Horizon | Log victoire R0 / V0 / R1 | Δ V0−R0 (2 SE) | CRPS 2ᵉ tour R0 / V0 / R1 | Δ V0−R0 (2 SE) | Couv. 90 % 2ᵉ tour R0 / V0 |
|---|---|---|---|---|---|
| 180 | −0,938 / −0,953 / −0,937 | −0,0149 (0,0059) | 3,302 / 3,095 / 3,294 | −0,207 (0,021) | 0,67 / 1,00 |
| 90 | −0,518 / −0,534 / −0,523 | −0,0159 (0,0059) | 2,407 / 2,272 / 2,400 | −0,135 (0,021) | 0,67 / 1,00 |
| 30 | −0,116 / −0,124 / −0,118 | −0,0074 (0,0059) | 2,555 / 2,559 / 2,554 | +0,004 (0,021) | 0,67 / 0,67 |
| 7 | −0,106 / −0,107 / −0,107 | −0,0015 (0,0059) | 1,870 / 1,793 / 1,856 | −0,077 (0,021) | 0,67 / 0,67 |
| **12 cas** | −0,420 / −0,430 / −0,421 | **−0,0099** (0,0029) | 2,533 / 2,430 / 2,526 | **−0,104** (0,011) | 0,667 / 0,833 |

Bruit Monte-Carlo (R1 − R0) : σ̂ par cas 0,0051 pour le log-score du vainqueur, 0,018 pour
le CRPS du second tour, 0 pour la couverture (aucun cas ne change de statut entre graines).
Contrôle V0_k1024, moyennes des douze cas : Δ log victoire −0,0050, Δ CRPS 2ᵉ tour −0,026,
couverture 0,667.

### Par cas

Part de second tour du finaliste réel premier par ordre alphabétique (Hollande en 2012,
Le Pen en 2017 et 2022), sur les tirages où le duel réel est qualifié.

| Cas | P(vainqueur) R0 / V0 / V0 k=1024 | Log victoire Δ V0−R0 | CRPS 2ᵉ tour R0 / V0 / k=1024 | Médiane 2ᵉ tour R0 / V0 / k=1024 (réel) | Couv. 90 % R0 / V0 / k=1024 |
|---|---|---|---|---|---|
| 2012-7 | 96,3 / 95,9 / 96,1 % | −0,0045 | 2,93 / 2,63 / 2,69 | 55,6 / 55,3 / 55,4 (51,64) | 0 / 0 / 0 |
| 2012-30 | 89,2 / 87,4 / 88,6 % | −0,0207 | 3,93 / 3,08 / 3,26 | 56,6 / 56,0 / 56,1 (51,64) | 0 / 0 / 0 |
| 2012-90 | 84,0 / 82,1 / 83,2 % | −0,0228 | 4,76 / 3,38 / 3,74 | 57,5 / 56,5 / 56,7 (51,64) | 0 / 1 / 0 |
| 2012-180 | 93,4 / 90,6 / 92,2 % | −0,0306 | 6,24 / 4,27 / 4,74 | 59,0 / 57,9 / 58,0 (51,64) | 0 / 1 / 0 |
| 2017-7 | 76,3 / 76,2 / 76,2 % | −0,0001 | 2,18 / 2,22 / 2,23 | 37,2 / 37,3 / 37,3 (33,90) | 1 / 1 / 1 |
| 2017-30 | 81,8 / 81,7 / 81,7 % | −0,0014 | 2,20 / 2,46 / 2,45 | 37,4 / 37,7 / 37,7 (33,90) | 1 / 1 / 1 |
| 2017-90 | 32,1 / 31,4 / 31,5 % | −0,0242 | 1,68 / 2,14 / 2,10 | 36,7 / 37,4 / 37,4 (33,90) | 1 / 1 / 1 |
| 2017-180 | 8,0 / 7,9 / 7,9 % | −0,0097 | 2,85 / 3,44 / 3,56 | 38,5 / 39,5 / 39,7 (33,90) | 1 / 1 / 1 |
| 2022-7 | 99,2 / 99,2 / 99,2 % | +0,0002 | 0,51 / 0,53 / 0,51 | 41,7 / 41,2 / 41,3 (41,45) | 1 / 1 / 1 |
| 2022-30 | 96,7 / 96,7 / 96,7 % | +0,0001 | 1,54 / 2,13 / 2,03 | 38,9 / 38,1 / 38,2 (41,45) | 1 / 1 / 1 |
| 2022-90 | 78,3 / 78,2 / 78,4 % | −0,0007 | 0,78 / 1,29 / 1,25 | 40,6 / 39,3 / 39,4 (41,45) | 1 / 1 / 1 |
| 2022-180 | 80,3 / 80,0 / 80,2 % | −0,0044 | 0,82 / 1,58 / 1,51 | 41,2 / 38,9 / 39,0 (41,45) | 1 / 1 / 1 |

### Verdict : non admis

1. Log-score du vainqueur : Δ = −0,0099 < −2 SE = −0,0029. **Échec.** La dégradation vient
   surtout de 2012 (quatre horizons), un peu de 2017 ; 2022 est neutre.
2. CRPS du second tour : Δ = −0,104 ≤ +0,011. Satisfait, mais le gain vient entièrement de
   2012 ; les huit cas de 2017 et 2022 se dégradent (de +0,02 à +0,76).
3. Couverture à 90 % du second tour : 0,833 contre 0,667, plus proche de 90 %. Satisfait.

Le critère 1 échoue : **l'option reste désactivée**. Le contrôle à 1024 propositions
confirme le sens de la dégradation du log-score (−0,0050, encore au-delà de 2 SE), mais il
montre aussi que les deux autres gains de V0 sont en bonne partie un artefact de k = 64 :
gain de CRPS réduit à −0,026, et couverture revenue à 0,667 (les cas 2012-90 et 2012-180
ne sont couverts qu'à k = 64). Les médianes bougent peu entre 64 et 1024 propositions ; ce
sont les queues qui changent. Une ligne dont la taille effective est proche de 1 retient
la proposition la plus compatible parmi 64 tirages du prior, encore loin de la loi visée :
la loi du second tour est alors élargie et tirée vers le prior des reports.

**Mécanisme.** En mode conjoint, la vraisemblance des sondages de duel est satisfaite en
partie en choisissant, dans la cellule, des premiers tours qui la rendent probable. À
premier tour fixé, seuls les reports peuvent bouger, et leur prior (concentrations de
Dirichlet, décalages σ_u et σ_v) retient davantage : le second tour suit moins les sondages
de duel. En 2012, les sondages surestimaient Hollande : suivre moins les sondages rapproche
la part du résultat (CRPS meilleur) mais réduit la probabilité du vainqueur réel. En 2017
et 2022, cela éloigne la part du résultat. Trois élections ne permettent pas de dire
lequel des deux comportements est le bon en général ; elles ne montrent pas de gain.

**Ce que le résultat ne dit pas.** Le défaut de cohérence ligne à ligne est réel et
documenté (docstring de `simulate.py`) ; ce test montre seulement que sa correction par ce
mode, à k = 64, ne passe pas la barre déclarée. Deux prolongements possibles, chacun à
déclarer avant calcul : une proposition adaptée par ligne (raffiner les lignes de faible
taille effective, ou décaler la proposition de `u`) pour obtenir la loi visée sans
dégénérescence ; un examen du prior des reports, puisque c'est lui qui fixe désormais la
réaction aux sondages de duel.

## Annexe : projection du 2026-11-24 (sensibilité, pas un critère)

Le rejeu de référence (`sim-ref`) reproduit à l'identique les tirages publiés
(`runs/2026-11-24-simulate-subst`). La variante partage exactement ses tirages de premier
tour. 2,8 % des tirages de la variante ont une taille effective par ligne inférieure à 2
(avertissements dans `sim.log`, médiane par ligne 27 sur 64 ; 0,2 % à 1024 propositions) ; le contrôle à 1024
propositions (`sim-k1024`, script `sim_k1024.py`) est donné à droite.

| Candidat | Référence | Premier tour fixé (k = 64) | Écart (points) | Contrôle k = 1024 |
|---|---|---|---|---|
| Marine Le Pen | 73,4 % | 68,1 % | −5,3 | 68,9 % |
| Jordan Bardella | 9,2 % | 8,6 % | −0,6 | 8,6 % |
| Édouard Philippe | 6,0 % | 8,8 % | +2,9 | 8,8 % |
| David Lisnard | 5,6 % | 5,6 % | 0,0 | 5,6 % |
| Raphaël Glucksmann | 2,5 % | 3,2 % | +0,7 | 3,2 % |
| Jean-Luc Mélenchon | 1,4 % | 3,5 % | +2,1 | 2,5 % |
| Jean Lassalle | 0,6 % | 0,6 % | 0,0 | 0,5 % |
| Xavier Bertrand | 0,5 % | 0,5 % | 0,0 | 0,5 % |

| Duel (part de a) | P(duel) | Sondages | 1ᵉʳ tour moyen a / b | Médiane réf. | Médiane fixé | Écart | P(a gagne) réf. / fixé | k = 1024 : médiane ; P(a gagne) |
|---|---|---|---|---|---|---|---|---|
| Le Pen – Mélenchon | 56,0 % | 28 | 30,8 / 22,3 | 63,1 | 60,7 | −2,4 | 100,0 / 97,1 % | 60,9 ; 98,6 % |
| Le Pen – Philippe | 10,0 % | 30 | 31,5 / 19,1 | 52,2 | 49,2 | −3,0 | 72,4 / 44,0 % | 49,3 ; 44,3 % |
| Le Pen – Glucksmann | 8,2 % | 15 | 32,2 / 18,7 | 55,2 | 54,4 | −0,8 | 91,5 / 83,2 % | 54,3 ; 83,2 % |
| Bardella – Mélenchon | 7,5 % | 4 | 30,5 / 22,3 | 63,8 | 60,2 | −3,5 | 99,7 / 95,2 % | 60,3 ; 95,7 % |
| Le Pen – Lisnard | 4,9 % | 0 | 31,8 / 18,0 | 41,2 | 41,0 | −0,2 | 19,3 / 19,3 % | 41,3 ; 19,2 % |
| Mélenchon – Philippe | 2,0 % | 3 | 26,4 / 21,7 | 39,1 | 40,5 | +1,4 | 0,5 / 2,8 % | 40,3 ; 2,8 % |
| Mélenchon – Glucksmann | 1,9 % | 0 | 26,3 / 21,9 | 47,4 | 48,0 | +0,6 | 38,5 / 42,0 % | 47,1 ; 39,0 % |
| Bardella – Philippe | 1,3 % | 8 | 31,3 / 18,8 | 48,8 | 46,8 | −2,0 | 39,8 / 28,9 % | 46,4 ; 29,5 % |
| Bardella – Glucksmann | 1,1 % | 2 | 31,9 / 19,1 | 54,4 | 53,7 | −0,7 | 78,5 / 71,2 % | 53,4 ; 70,3 % |
| Mélenchon – Lisnard | 1,1 % | 0 | 26,5 / 20,5 | 44,3 | 44,8 | +0,5 | 26,9 / 29,0 % | 44,1 ; 27,6 % |

Les duels sans sondage ne bougent qu'au bruit Monte-Carlo près (≤ 0,6 point de médiane),
comme attendu. Les duels sondés perdent 1 à 3,5 points pour le finaliste d'extrême droite :
le premier tour réellement tiré pour ces lignes est moins favorable au RN que celui que le
mode conjoint retenait. L'écart de −2,4 points sur Le Pen – Mélenchon reproduit le
diagnostic du jour. Le pivot Le Pen – Philippe (72 % → 44 %) explique l'essentiel du
transfert de probabilité de victoire de Le Pen vers Philippe.

## Rejouer

Depuis `back/` :

```sh
uv run python studies/fixed_first_round.py --reference ../runs/2026-10-22-consolidation \
    --out ../runs/2026-11-24-premier-tour-fixe/chain
uv run vvs simulate --asof 2026-11-24 --fit ../runs/2026-11-24-fit-2027-subst --seed 0 \
    --substitution --rn-bias-curve --out ../runs/2026-11-24-premier-tour-fixe/sim-ref
uv run vvs simulate --asof 2026-11-24 --fit ../runs/2026-11-24-fit-2027-subst --seed 0 \
    --substitution --rn-bias-curve --fixed-first-round --out ../runs/2026-11-24-premier-tour-fixe/sim
uv run python ../runs/2026-11-24-premier-tour-fixe/sim_k1024.py
```

Code exécuté : HEAD `03178d5` plus les modifications non committées de ce travail ; les
empreintes SHA-256 des sources sont dans `chain/manifest.json`. Gel : `reference-2026-10-22`
(commit `a89d304`), versions de dépendances identiques à celles du gel.
