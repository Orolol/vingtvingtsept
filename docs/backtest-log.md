# Journal de backtest

Une entrée par changement de modèle, avec le commit. Protocole : `07-evaluation.md`.
Sorties détaillées dans `runs/<date>-backtest/` (premier tour) et
`runs/<date>-second-round/` (second tour).

## 2026-10-20 — phase 3 : premier modèle d'erreur, backtest 2002-2022

Commits `452e382` (premier tour) et `65bf3f5` (second tour). 48 ajustements NUTS du modèle
de premier tour (cinq élections, quatre horizons, 500 + 500 itérations, r_hat des parts
≤ 1,006), 2002 et 2007 pondérées à 0,5.

### Modèle d'erreur retenu (premier tour)

Sur l'échelle logit, η_c = λ·(u_b + v_c) avec une échelle λ de Student partagée par tous
les candidats (ν = 16, choisi par validation croisée interne), u_b par bloc et v_c par
candidat, variances de la forme σ²_final + a·(h/90)^γ, γ ≈ 0,37 estimé sur
Jennings–Wlezien hors France 1965-1969 et hors élection évaluée. Estimation par maximum de
vraisemblance pénalisé en leave-one-election-out, sur `poll_errors` (erreur finale) et les
mouvements entre l'agrégat à l'horizon et l'agrégat final. Le logit s'impose parce que
l'erreur finale en points croît avec la taille du candidat alors qu'en log de la part elle
est homogène (0,10 à 0,32).

Échelle en points (moyenne quadratique sur les candidats) : 1,6 à 2,8 à 7 jours, 2,9 à 4,5 à
30, 3,5 à 5,3 à 90, 3,9 à 5,9 à 180 selon l'élection laissée. C'est l'échelle France de
Jennings–Wlezien (2,6 / 3,2 / 4,4 / 5,9), nettement sous la moyenne des présidentielles
tous pays (4,6 / 6,7 / 7,4 / 8,9). Corrélation intra-bloc 0,2 à 7 jours, 0,35 à 0,42
au-delà.

**Biais de bloc** : admis deux fois sur cinq par la règle de stabilité, et les deux fois il
dégrade l'élection laissée. Désactivé.

### Résultats premier tour (moyennes pondérées, 180 / 90 / 30 / 7 jours)

| Métrique | modèle + η | phase 2 sans η | moyenne 14 j + η | dernier par institut + η |
|---|---|---|---|---|
| Log-score de la paire réelle | **−1,30 / −1,04 / −0,80 / −0,64** | −2,64 / −1,15 / −1,32 / −1,28 | −1,46 / −1,08 / −0,85 / −0,69 | −1,34 / −1,13 / −0,81 / −0,69 |
| CRPS par candidat (points) | **2,43** / 2,04 / 1,46 / **0,90** | 2,74 / 2,18 / 1,52 / 0,91 | 2,50 / **1,99** / 1,50 / 1,00 | 2,48 / 2,04 / **1,45** / 0,95 |
| Couverture à 90 % | 91 / 80 / 91 / 89 % | 53 / 51 / 48 / 48 % | 86 / 78 / 91 / 94 % | 84 / 84 / 92 / 94 % |
| Couverture à 50 % | 42 / 52 / 52 / 53 % | 21 / 29 / 29 / 32 % | 44 / 52 / 53 / 53 % | 51 / 49 / 51 / 55 % |

Lecture :

- **L'apport principal de la phase est η.** Sans lui, la paire réelle reçoit une
  probabilité nulle en 2002 à 30 et 7 jours et en 2017 à 180 jours, et la couverture à 90 %
  tombe à 50 %.
- **Le modèle bat la moyenne naïve** en log-score à tous les horizons et en CRPS à 180 et
  7 jours. Il ne la bat pas à 90 jours en CRPS, fait jeu égal avec le dernier sondage par
  institut à 30 jours, et perd en 2002 à 7 jours (trop sûr de Jospin). Avec cinq
  élections, aucun écart n'est significatif.
- **Où ça casse** : Mélenchon 2022 (15,2 % [11,5–19,7] prévu à 7 jours pour 21,95, hors
  intervalle à tous les horizons), Pécresse 2022 (9,6 pour 4,78), Bayrou 2007 et Macron
  2017 à 180 jours. 2022 est trop étroite (couverture 90 % observée de 58 à 75 %), 2012 et
  2017 trop larges.
- **Limite structurelle visible** : la configuration « union compatible » d'avant la liste
  officielle contient plus de candidats qu'aucun sondage (16 en 2022 à 90 jours) et dilue
  tout le monde par indépendance des alternatives. La substitution intra-bloc est la
  priorité de la phase 4.
- **Plancher d'erreur non due à l'échantillonnage** (1/n_eff = 1/(f·n) + 1/n_max) : rend les
  paramètres plausibles (facteur de design 0,7 à 0,8 au lieu de 0,3 à 0,5, n_max 1 300 à
  2 400) sans effet mesurable au backtest. Les priors du facteur de design et des effets
  maison ne changent rien non plus : les données dominent.

### Résultats second tour (leave-one-election-out, part du vainqueur)

Modèle par reports de voix : Dirichlet par électorat centré sur les reports historiques
analogues (même paire de blocs) ou sur une matrice de proximité pour les duels inédits,
décalages communs pour le front républicain, sondages de duel avec biais RN et erreur
d'horizon calibrés, échantillonnage d'importance.

| Horizon | méthode | erreur abs. | CRPS | couv. 50 / 90 % |
|---|---|---|---|---|
| veille du 2d tour (n = 5) | **modèle** | **1,58** | **1,22** | 0,40 / 0,80 |
| | modèle sans biais RN | 2,89 | 2,26 | 0,20 / 0,40 |
| | sondages de duel 14 j | 2,96 | 2,29 | 0,20 / 0,40 |
| | reports historiques seuls | 5,96 | 4,50 | 0,40 / 0,80 |
| 30 j avant le 1er tour (n = 4) | modèle | 2,11 | 1,44 | 0,25 / 0,75 |
| | sondages 14 j | 2,10 | 1,53 | 0,50 / 0,75 |
| 90 j (n = 4) | modèle | 2,47 | 1,65 | 0,50 / 0,75 |
| | sondages 14 j | 2,57 | 1,79 | 0,75 / 0,75 |
| 180 j (n = 3) | modèle | 4,17 | 3,03 | 0,67 / 0,67 |
| | modèle sans biais RN | 3,49 | 2,85 | 0,67 / 0,67 |

Lecture : à la veille du second tour, le gain vient presque entièrement de la correction du
biais RN (+3,2 en 2002, +4,1 en 2017, +4,5 en 2022 sur le score de l'extrême droite dans
les sondages de duel à 7 jours ; biais rétréci à 3,0). Avant le premier tour, le gain sur la
moyenne des sondages est nul à 30 jours, faible à 90, et la correction dégrade à 180 jours.
Le biais RN est donc une variable à n'appliquer qu'aux horizons courts tant qu'il n'est
pas réévalué par horizon. Les reports seuls sont mauvais en niveau (2002 : 64,5 contre
82,2 ; 2022 : 63,1 contre 58,5). Les reports par électorat ne sont qu'indicatifs : les
sondages ne contraignent qu'une combinaison linéaire.

### Décisions

1. Garder η, biais de bloc désactivé ; le plancher d'erreur devient le défaut.
2. Phase 4 : substitution intra-bloc dans le modèle de premier tour, puis mélange de
   scénarios à la place de la configuration déterministe, et rejouer ce backtest.
3. Second tour : coupler l'erreur de duel et les tirages du premier tour par une composante
   d'extrême droite partagée ; appliquer le biais RN par horizon ; tirer la participation.
4. Vérifier les dates de publication des listes officielles (actuellement de mémoire dans
   `eval/configurations.py`) et les reporter dans `elections.yaml`.
5. Ligne de référence pour l'admission d'une variable : « modèle + η ».

## 2026-10-20 — vague de variables candidates : cinq tests, aucune admission

Commits `75f6767` (substitution), `a13bcc8` (rebonds), `0dabddc` (palmarès), `0ee5955`
(potentiel), `f86ca06` (certitude), `a3c2175` (attention), `8de3d45` (options partagées).
Référence : « modèle + η », variante `floor`. Chaque variable est une option désactivée par
défaut dans le code ; les études et comparaisons sont dans `runs/2026-10-20-*`.

| Variable | Étude | Backtest contre floor (log-score 180 / 90 j) | Verdict |
|---|---|---|---|
| Substitution intra-bloc (logit emboîté) | gain net sur les hypothèses emboîtées 2022 (0,94 → 0,83) et 2027 (1,27 → 1,01), perte 2017 (Bayrou) | −1,33 → −1,37 / −0,98 → −0,98 ; 2022 à 90 j s'améliore (CRPS −0,26), 2017 à 90 j se dégrade avec absorption | **non admise** ; publiée sur le site à la demande de Gaétan, le run IIA reste disponible |
| Rebonds de primaire et d'annonce | forme commune seulement pour les primaires : saut de 2 à 10 points entièrement érodé au vote (7 cas sur 7) ; aucune forme pour déclarations, retraits, chocs | −1,326 → −1,343 / −0,978 → −0,821 ; le gain à 90 j tient à Pécresse 2022 | **non admise** |
| Palmarès des instituts | précision et effets maison non reproductibles d'une élection à l'autre (r ≈ 0,1) ; seul signal persistant : sous-estimation de la gauche radicale par OpinionWay et Ifop | −1,326 → −1,311 / −0,978 → −0,983 ; gain sur une élection sur trois | **non admise** |
| Potentiel électoral et rejet (second tour) | pas un plafond général : les votes de barrage le dépassent ; seul le finaliste RN reste sous son potentiel (n = 2) ; reports via potentiel moins bons que la matrice | CRPS identique à 180 et 90 j, 1,225 → 1,214 à la veille du second tour | **non admise** |
| Sûrs de leur choix | corrélation avec le mouvement ultérieur de signe opposé entre 2017 (−0,43) et 2022 (+0,20) ; aucune ventilation par candidat publiée pour 2027 | +0,03 / 0,00 en log-score, CRPS dégradé, couverture −5 pts | **non admise** |
| Attention en ligne (Wikipédia, Trends) | Wikipédia précède les sondages de 1 à 4 semaines (β = 0,085, placebo p = 0,01) mais l'effet disparaît hors événements et à retard d'une semaine ; Granger inverse aussi significatif ; Trends ne réplique pas | non testée au backtest : effet nul aux horizons de la règle | **non retenue** |

Enseignement : avec cinq élections, la ligne « modèle + η » est difficile à battre, et la
plupart des signaux plausibles sont soit déjà dans les sondages, soit instables d'une
élection à l'autre. Les volets explicatifs du site peuvent montrer ces signaux avec la
mention « n'entre pas dans la prévision ». Pistes restantes jugées les plus utiles :
tirer la participation du second tour à part (implicite et mal calibrée : 78,6 % contre
66 % en 2017), coupler l'erreur de duel et le premier tour par une composante d'extrême droite
partagée, et réévaluer le biais RN du second tour par horizon.

## 2026-10-20 — second tour, trois corrections de structure

Évaluation appariée sur cinq élections et trois graines
(`runs/2026-10-20-second-round-v2/report.md`).

| Correction | Résultat | Décision |
|---|---|---|
| Biais RN par horizon (courbe isotone rétrécie : 2,97 à la veille, 2,93 à 22 j, 1,64 à 45 j, 1,36 à 105 j, 1,26 à 195 j) | contre l'atténuation linéaire provisoire : CRPS −0,23 à 180 j (stable sur 3 graines, 2 élections sur 3), −0,01 à 90 j ; contre le biais constant : +0,11 à 30 j ; log-score −0,001 à −0,004 (2007 et 2012, par réestimation de σ) ; couverture 90 % +0,13 à 90 j | non admise strictement ; **retenue pour la publication** à 180 j parce qu'elle remplace un jugement jamais évalué et fait mieux à l'horizon actuel ; à réévaluer avant 30 jours |
| Couplage erreur premier tour / duel (ρ sur far_right) | ρ non identifiable : 0,70 sans 2002, −0,36 sans 2017 ; l'historique contredit un biais commun (2002 : FN sous-estimé de 5 au premier tour, surestimé de 3 aux duels) ; CRPS +0,01 | désactivé |
| Participation du second tour tirée | erreur de participation 7,8 → 6,9 points avant le premier tour mais 4,0 → 6,7 à la veille ; CRPS +0,10 à 180 j | désactivée |

Effet sur 2027 (mélange, substitution) : P(Le Pen) 70,1 → 66,1 %, Philippe 13,6 → 16,4 %,
duel Le Pen–Philippe 68 → 57 % pour Le Pen, Le Pen–Glucksmann 85 → 76 %.

## 2026-10-20 — calibration de η étendue aux scrutins de liste : admise (première admission)

Sondages des européennes 2014, 2019, 2024 et des législatives 2022, 2024 (298 sondages,
`data/curated/polls_lists.parquet`), rattachés aux blocs par `data/registry/lists.yaml`.
L'erreur finale de η (σ_Bf, σ_Cf, ρ, poids de bloc) est calibrée sur les présidentielles plus
ces cinq scrutins pondérés 0,5 ; les mouvements restent calibrés sur les présidentielles (la
version incluant les mouvements des scrutins de liste fait tomber la couverture à 90 % de
92 à 81 % à 180 jours : rejetée). Détail dans `runs/2026-10-20-backtest-lists/comparison.md`.

| Horizon | log-score floor → lists (moy. 2012-2022) | CRPS | couverture 50 / 90 % candidats |
|---|---|---|---|
| 180 j | −1,326 → −1,325 | 2,648 → 2,637 | 35/92 → 39/92 |
| 90 j | −0,978 → −0,909 | 2,119 → 2,074 | 55/79 → 48/85 |
| 30 j | −0,505 → −0,435 | 1,532 → 1,494 | 55/92 → 55/92 |
| 7 j | −0,375 → −0,344 | 0,904 → 0,875 | 58/92 → 64/92 |

Verdict : **admise, de justesse**, et passée en défaut (`ERROR_WITH_LISTS = True`,
`vvs simulate --no-lists` pour revenir). Réserves consignées : égalité à 180 jours (recul de
0,007 en moyenne pondérée), 2002 perd à tous les horizons, couverture des blocs à 50 %
dégradée (un bloc sur 17 par horizon) ; sous une lecture case par case du critère de
couverture, elle ne serait pas admise. Paramètres : ν passe de 16 à 8, σ_Bf monte, σ_Cf
baisse (0,180 → 0,128), le biais de bloc disparaît des plis.

Effet sur 2027 (v2 → v3) : faible, η un peu plus étroit à 180 jours. Le Pen 66,1 → 67,1 % de
victoire, Philippe 16,4 → 16,1 %, duel Le Pen–Philippe 57,0 → 55,8 % pour Le Pen.

**Biais d'extrême droite depuis 2017** (surestimée en fin de campagne dans les six scrutins
depuis 2017, μ ≈ −0,055 en logit) : testé comme option, **rejeté** (log-score −0,03 à −0,06
à tous les horizons, perte sur les trois élections ; en 2017 la médiane s'éloigne du
résultat).

## 2026-10-20 — deuxième vague de pistes : volatilité, vote utile, presse

| Piste | Résultat | Verdict |
|---|---|---|
| Erreur adaptée à la volatilité réalisée | κ = 0 en leave-one-election-out ; seul 2022 serait mieux couvert, au prix du log-score ; signal réel seulement par candidat (β = 0,28) | non admise |
| Vote utile de fin de campagne | consolidation nette à gauche seulement (5/5, p = 0,001), absente ailleurs ; la dérive surcorrige à 7 jours (Jospin 2002 projeté à 29 %), perd en validation croisée | non admise ; la règle de docs/07 juge désormais aux horizons d'action |
| Presse par comptages (part de voix, ton, agenda, affaires) | aucun des six tests ne passe ; mesure les mêmes événements que l'attention Wikipédia avec moins de signal | non testée au backtest |

Pistes restantes notées : volatilité par candidat pour la composante v_c de η ; consolidation
limitée à la gauche comme information qualitative sur le site.

## 2026-10-22 — troisième vague : volatilité par candidat, troupeau, anticipations, abstention différentielle, Cevipof

| Piste | Étude | Backtest | Verdict |
|---|---|---|---|
| Volatilité par candidat sur v_c | β = 0,28 reproduit (p = 0,0005), Jennings–Wlezien 0,19 sur 40 pays | κ = 0 en leave-one-election-out dans 4 plis sur 5 : la loi actuelle (taille, mouvement) capte déjà le signal ; seule 2002 bouge, en pire | non admise |
| Troupeau des instituts | resserrement réel en fin de campagne (7 élections sur 8, p = 0,012) ; mais dispersion et erreur vont ensemble (ρ = +0,50), pas en sens inverse | κ négatif ; `lists_herding` perd à 180 j, `lists_herding_final` échoue au critère 2 et éloigne la couverture | non admise ; dispersion 2027 très élevée (D = 3,5 contre 2,0) par divergence des configurations testées |
| Anticipations « qui va gagner ? » (110 mesures 2007-2027) | le plus cité est le vainqueur dans tous les cas appariés ; part normalisée très sous-confiante (41 à 61 % en moyenne par horizon ; maximum 70 % en 2007, 63 % depuis 2012) | ne bat le modèle sur aucune élection de référence à 90 ou 180 j ; conversion part → probabilité non estimable | non intégrée ; indicateur d'opinion sur le site (Le Pen 54 % chez Cluster17 en juillet 2026, modèle 67 %) |
| Abstention différentielle (certains − inscrits, OpinionWay seul) | écart stable par candidat (r 0,9 entre moitiés de campagne) ; lien avec l'erreur finale seulement en 2022 et de signe négatif (électorats fragiles finissant au-dessus des sondages) ; validation croisée 2022 → 2017 échoue | non testé au backtest | non intégrée ; base publiée par chaque institut conservée |
| Cevipof (microdonnées) | 80 jeux inventoriés, tous restreints ; 47 demandes déposées ; FES 2017 publique : matrice de reports complète (abstentionnistes et blancs compris), écart de 10 points avec les instituts sur l'électorat Fillon | source `cevipof_2017` ajoutée au prior des reports ; duels par électorat et anticipations ENEF à venir avec l'accès | en attente du CDSP |

Défauts de curation relevés par l'étude d'abstention, corrigés le 2026-10-22 : les sondages
Ifop et Harris 2027 étaient marqués `registered` alors que leurs tableaux portent sur les
certains d'aller voter ; la seconde base d'OpinionWay était perdue par la déduplication sur
(tour, candidats). Règles de correction et de sélection d'une base : `03-donnees.md`.

## 2026-10-22 — reports de voix par inférence écologique (communes, Cagé-Piketty) : non validée

Goodman contraint pondéré par les inscrits et version par strates revenu × type de commune
sur les cinq seconds tours (`runs/2026-10-22-ecological/`). Critère : moins de 8 points
d'écart avec les enquêtes post-électorales sur la répartition entre finalistes. Échec :
biais d'agrégation (Fillon 2017 +16, Mélenchon 2022 +29 vers Macron). Une variante par
contexte RN passe pour 6 électorats sur 7 mais est fragile (2002 absurde). La décomposition
sociale appliquée à la composition 2022 ne fait pas mieux que les reports 2017 bruts : le
passage 2017 → 2022 est un décalage commun (érosion du front républicain de 0,6 à 0,8 en
logit), pas un effet de composition. Sources `ei_<année>` non fusionnées.

Propositions issues de l'étude, à évaluer au protocole du second tour : σ_u analogue 0,30 →
0,5, σ_u inédit 0,6 → 0,8, σ_v 0,25 → 0,30 et 0,40 → 0,45 ; ligne `far_right` de PROXIMITY :
centre 2,5 → 6,5, droite 6 → 7, « ni l'un ni l'autre » (0,10 ; 0,55).

## 2026-10-22 — Consolidation chronologique de la chaîne complète

Référence statistique figée au commit `a89d304`, bilan conservé sous le tag
`reference-2026-10-22`. Protocole annoncé avant le calcul :
[10-consolidation.md](10-consolidation.md). 2012, 2017, 2022 × 180, 90, 30, 7 jours ;
40 000 tirages par cas et par modèle. Toutes les calibrations actives excluent le scrutin
évalué et les scrutins futurs. Publication manquante = fin du terrain + 2 jours ; scénarios
historiques sans correspondance de candidats après coup. Le vainqueur est simulé après
les candidatures, le premier tour et la qualification, sans imposer le duel observé.

Le témoin `mean14d` partage la loi d'erreur, les scénarios et le second tour avec la
référence ; cette comparaison mesure l'apport de l'agrégateur.

| Horizon | Δ log-score victoire | Δ log-score paire | Δ CRPS | Lecture |
|---|---|---|---|---|
| 180 j | +0,175 | +0,155 | −0,020 | gain surtout porté par 2017 ; couverture 90 % à 66,5 % |
| 90 j | −0,008 | −0,011 | +0,020 | petite dégradation sur les trois métriques |
| 30 j | +0,008 | −0,001 | −0,048 | résultat partagé selon la métrique |
| 7 j | −0,032 | −0,079 | −0,068 | parts mieux prédites au CRPS, événements moins bien scorés |

Différences référence moins témoin ; positif préférable pour les log-scores, négatif pour
le CRPS. **Pas de domination uniforme, pas de nouvelle admission.** En 2017 à 180 jours,
les candidats absents des scénarios représentent 28,5 % des voix finales ; cette erreur
reste comptée. La méthode et les classifications ont été élaborées rétrospectivement,
et les dates de publication historiques sont toutes imputées dans les cas retenus.

Douze ajustements conformes : R-hat maximal 1,0079, ESS minimale des parts 3 073, aucune
divergence. Reprise vérifiée sans recalcul ; correction limitée à la promotion de type
entier/décimal lors de l'assemblage des métriques, sans modification du gel ou des cas.
719 tests passent. Scores complets, audit et figures :
[jeu de résultats versionné](results/consolidation-2026-10-22/README.md).

Sensibilité 2027 : 47 variantes nommées sur les mêmes tirages publiés. Sous les variations
locales de ±10 points des poids, victoire Le Pen 62,1–78,0 %, Bardella 1,4–15,9 %, ensemble
RN environ 77,6–79,7 %. Les poids uniformes laissent 14,286 % de masse sans tirage et une
ESS d'environ 649 : les bornes sont publiées, le chiffre central reste conditionnel au
support représenté. Ces étendues sont des sensibilités aux hypothèses, pas des intervalles
de confiance ; certaines perturbations binaires sont équivalentes.

## 2026-10-22 — second tour : σ élargis et ligne far_right, non admis

Évaluation appariée (5 élections, 3 graines, `runs/2026-10-22-second-round-prior/`).
σ_u analogue 0,30 → 0,5 et σ_u inédit 0,6 → 0,8 : CRPS meilleur à tous les horizons (2017 et
2022 gagnent, 2012 perd) mais log-score en baisse avant le premier tour (8 couples graine ×
horizon sur 9), couverture strictement inchangée : la loi a posteriori est tenue par les
sondages de duel, l'intervalle à 90 % ne s'élargit que de 0,1 à 0,7 point. Ligne `far_right`
de PROXIMITY : n'agit qu'en 2002 (électorat Mégret) et s'appliquerait en 2027 à tout
l'électorat RN dès que le RN est absent du second tour, cas que les cellules d'origine ne
renseignent pas (Philippe–Glucksmann passerait de 48 à 66 % pour Philippe). Combinaisons :
non admises. Les réglages du second tour sont désormais injectables
(`SecondRoundSettings`), défauts inchangés. Effet sur 2027 des variantes : P(Le Pen) 70,0 →
70,0 à 70,4 %.
