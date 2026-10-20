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
tirer la participation du second tour à part (l'écart 78 % contre 66 % signalé compare un
premier tour simulé à un second tour observé), coupler l'erreur de duel et le premier tour par une composante d'extrême droite
partagée, et réévaluer le biais RN du second tour par horizon.
