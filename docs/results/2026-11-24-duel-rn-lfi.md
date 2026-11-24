# Duel RN contre LFI : les législatives 2022 et 2024 comme analogues des reports

Étude du 2026-11-24 (sous-agent Opus). Rien de ce qui suit n'entre dans la prévision publiée :
aucun défaut, aucun fichier de `back/src/vvs/model/`, ni `transfers.yaml` ne sont modifiés.

## Critères, écrits avant tout calcul

Cette section a été écrite et enregistrée avant le téléchargement des résultats par
circonscription et avant tout calcul. Elle n'est pas retouchée après lecture des résultats ;
toute précision ultérieure figure dans une section distincte, datée.

**Question.** Pour un duel extrême droite contre gauche radicale, le modèle tire ses reports de
la matrice de proximité `PROXIMITY` (duel « inédit »). Des reports estimés sur les duels RN
contre LFI des législatives 2022 prévoient-ils mieux, circonscription par circonscription, les
seconds tours RN contre LFI de 2024 ?

**Jeu de test.** Les seconds tours des législatives 2024 qui opposent exactement deux candidats,
l'un de nuance RN ou UXD (alliés d'Éric Ciotti), l'autre de nuance UG (ou FI) dont le parti est
LFI selon la source d'appartenance documentée plus bas. Les duels issus d'une triangulaire
après retrait du troisième qualifié restent dans le jeu (signalés ; résultats aussi donnés
sans eux, à titre de diagnostic).

**Entrées de la prévision.** Pour chaque circonscription, les voix de premier tour de chaque
candidat, les inscrits et les exprimés du premier tour ; rien du second tour.

**Électorats.** La structure du modèle (`vvs.model.second_round.build_groups`) : électorat
propre de chaque finaliste (`own:`), les autres candidats regroupés par bloc du modèle d'après
leur nuance (EXG, COM, FI → `far_left` ; SOC, VEC, ECO, DVG, RDG → `left` ; ENS, HOR, DVC et
UDI en 2024 → `centre` ; LR, DVD et UDI en 2022 → `right` ; RN, UXD, REC, DSV, EXD, DXD →
`far_right` ; DIV, REG et le reste → `other`), et `abstain` = inscrits − exprimés du premier
tour (abstention, blancs et nuls). Part prédite du RN parmi les exprimés :
Σ w t_RN / Σ w (t_RN + t_LFI) (`duel_share`).

**(a) Matrice de proximité.** Les priors que le modèle construit aujourd'hui pour un duel
`far_right` contre `far_left` en 2027 (`build_groups` avec les réglages par défaut,
`own_prior(2027)` pour les finalistes) : moyennes `PROXIMITY`, κ = `KAPPA_PROXIMITY` (6),
décalages communs σ_u = 0,60 et σ_v = 0,40 (cas inédit).

**(b) Prior analogue 2022.** Mêmes électorats propres (moyenne `own_prior(2027)`, κ =
`KAPPA_OWN`). Pour chaque autre électorat, moyenne = reports estimés par régression de Goodman
contrainte (`vvs.model.ecological.solve_constrained`, lignes sur le simplexe, pondération par
les inscrits) sur les duels RN contre LFI des législatives **2022 seulement**, issues (RN, LFI,
ni l'un ni l'autre) en part des inscrits de chaque tour, les lignes des deux finalistes étant
fixées à `own_prior(2027)` pour rester cohérentes avec la prévision. Concentration et
décalages communs : ceux du modèle pour un analogue d'une seule élection, κ = `KAPPA_ANALOG[1]`
(20), σ_u = 0,30, σ_v = 0,25. Un électorat dont le poids cumulé dans les duels 2022 est
inférieur à 1 % des inscrits n'est pas estimé : il garde la moyenne de proximité et κ = 6
(signalé).

**Distribution prédictive.** Pour chaque circonscription, 4 000 tirages des reports
(`draw_transfers`, graine fixe, la même pour (a) et (b)) appliqués à son premier tour.

**Métriques.** Sur les circonscriptions du jeu de test, à poids égal :
- erreur absolue moyenne (EAM) de la médiane prédictive par rapport à la part RN observée ;
- score logarithmique moyen : log de la densité prédictive en la part observée (points
  d'exprimés), densité estimée par noyau gaussien (`scipy.stats.gaussian_kde`, largeur de
  Scott) sur les 4 000 tirages, plancher 1e-6 ; plus haut = meilleur ;
- couverture : part des circonscriptions dont la part observée tombe dans l'intervalle
  central à 90 % (quantiles 5 % et 95 % des tirages).

**Succès.** (b) bat (a) sur l'EAM (plus basse) **et** sur le score logarithmique moyen (plus
haut), **et** la couverture à 90 % de (b) est comprise entre 80 % et 97 %. Sinon, résultat
négatif, rapporté tel quel.

**Variante, pas un critère : validation croisée par département sur 2024.** Même prédiction
(b), mais reports estimés sur les duels RN contre LFI de 2024 des autres départements (chaque
département prédit sans ses propres circonscriptions). Elle mesure ce que vaudrait un analogue
« de la même année », borne haute de ce qu'on peut attendre d'un analogue législatif.

**Diagnostic annexe, pas un critère.**
- Biais moyen signé (prévu − observé, part RN) des enquêtes de reports publiées avant le second
  tour de 2024 (`data/registry/transfers_legislatives.yaml`), appliquées aux premiers tours des
  duels RN contre LFI de 2024 ; électorats non couverts par l'enquête : moyenne de proximité.
  À comparer au biais RN de 2,97 points que le modèle applique aux duels RN contre centre.
- Ce que donnerait le prior (b) pour Le Pen – Mélenchon 2027 aux premiers tours actuels de la
  projection du 2026-11-24, prior seul (sans sondage de duel), comparé au prior (a), en simple
  sensibilité.
- Reports estimés par électorat (centre, droite, gauche non LFI, participation) pour chaque type
  de duel et chaque année : RN contre LFI, RN contre gauche non LFI, RN contre Ensemble ou LR,
  avec intervalles par bootstrap des circonscriptions.

**Réserve principale annoncée d'avance : la transposabilité législatives → présidentielle**
(participation plus faible et différentielle, effet du candidat local et du sortant, front
républicain organisé par désistements, offre de premier tour réduite par les accords NUPES/NFP).
Elle est discutée plus bas quel que soit le verdict.

---

## Précisions ajoutées après l'écriture des critères (avant le calcul des scores)

Ajoutées le 2026-11-24 au moment de construire les données, avant de lancer l'évaluation ;
elles ne changent ni les métriques ni le seuil.

- **Source d'appartenance partisane** (`data/registry/legislative_duels_sources.yaml`) : en
  2024, colonne « Parti et coalition » des pages Wikipédia départementales (révisions figées),
  appariée au candidat du ministère par les voix de premier tour ; en 2022, tableau
  « Répartition par circonscription » de l'article Wikipédia de la NUPES. Un candidat est
  « LFI » si cette source dit LFI. Les candidats du quota LFI de l'accord NFP qui ne sont pas
  étiquetés LFI (Poutou, NPA ; Legavre, POI ; Ruffin, Picardie debout ; Arnault, NPA ;
  Prandt) restent donc hors du jeu de test ; un diagnostic les y remet.
- Électorat `other` et `far_left` (autres que le finaliste) : poids sous 1 % des inscrits
  dans les duels 2022, donc non estimés (moyenne de proximité), comme prévu par la règle.

## Données

`vvs.ingest.legislative_duels` (sources : `data/registry/legislative_duels_sources.yaml`) :

- résultats par circonscription des deux tours, ministère de l'Intérieur via data.gouv.fr
  (4 fichiers, 0,9 Mo, empreintes sha256 vérifiées), dans `data/raw/legislative_duels/` ;
- 1 section et 208 pages Wikipédia figées par révision, plus le CSV non officiel de la
  répartition NFP (contrôle) ;
- tables propres : `data/curated/legislative_candidates.parquet` (12 541 lignes candidat × tour) et `data/curated/legislative_duels.parquet` (une ligne par second tour à deux
  candidats, type de duel, électorats du premier tour par bloc, retrait d'un qualifié).

Deux écarts internes au fichier 2024 entre somme des voix et exprimés (−2 à +1 voix dans cinq
circonscriptions) sont gardés (`vote_sum_gap`), comme au niveau national.

**Couverture du parti des candidats d'union** (premier tour) : 2022, 517 sur 543 (26
inconnus : outre-mer et circonscriptions sans accord) ; 2024, 520 sur 544 (24 inconnus, dont
les 11 des Français de l'étranger et 3 de La Réunion). **Tous les duels RN contre union de la
gauche ont un parti.** Contrôle par une seconde source :

| | concorde | diffère | contrôle absent |
|---|---|---|---|
| 2022 (tableau NUPES contre pages départementales) | 490 | 29 | 24 |
| 2024 (pages départementales contre accord NFP) | 473 | 44 | 27 |

Dans les duels RN contre gauche, les désaccords 2024 portent sur des candidats du quota LFI
d'autres partis (voir plus haut), des écologistes de Génération.s, deux circonscriptions dont
l'accord NFP a changé après le fichier de presse (64-05, 66-04 : candidats PS) et quelques
échanges PS / écologistes à Marseille ; en 2022 un seul désaccord (02-03, Jean-Louis Bricout,
PS selon la NUPES, sans étiquette selon la page départementale). Les pages départementales
2022 ont des formats hétérogènes : le contrôle 2022 manque pour une partie des candidats.

**Duels retenus** (seconds tours à deux candidats) :

| | RN contre LFI | RN contre gauche non LFI | RN contre centre | RN contre droite |
|---|---|---|---|---|
| 2022 | 33 | 26 | 108 | 31 |
| 2024 | 47 (dont 23 après retrait d'un qualifié) | 100 (59) | 132 | 58 |

Part RN moyenne dans les duels RN contre LFI : 51,9 % en 2022, 48,2 % en 2024.

## Résultats de l'expérience

Commit `38015b3`, graine 20261124, 4 000 tirages par circonscription. Sorties :
`runs/2026-11-24-duel-rn-lfi/results.json`, `per_circ.parquet`.

**Jeu de test : 47 duels RN contre LFI de 2024.**

| Prior | EAM (pts) | Score log moyen | Couverture 90 % | Couverture 50 % | Biais (prévu − observé) | Largeur 90 % |
|---|---|---|---|---|---|---|
| (a) matrice de proximité | **4,99** | **−3,21** | 91,5 % | 59,6 % | +4,82 | 22,8 |
| (b) analogue 2022 (critère) | 6,08 | −4,03 | 53,2 % | 0 % | +6,08 | 10,8 |
| (b') analogue 2022, fidélités estimées aussi | 6,13 | −4,23 | 44,7 % | 0 % | +6,13 | 9,5 |
| variante : analogue 2024, validation croisée par département | 2,05 | −2,43 | 95,7 % | 61,7 % | +0,39 | 11,3 |

Sans les 23 duels issus d'un retrait : (a) EAM 5,16, couverture 83 % ; (b) EAM 5,49,
couverture 67 % ; même conclusion. En élargissant le jeu de test aux 7 duels que l'accord NFP
attribuait à LFI sans que la source principale dise LFI (Poutou, Prandt, Ruffin, Arnault,
Legavre, plus deux candidats finalement investis PS, 64-05 et 66-04 ; 54 duels) : (a) EAM 5,02, score −3,21, couverture 92,6 % ; (b) EAM 6,22, score
−4,09, couverture 50 %.

## Verdict : résultat négatif

(b) ne bat (a) ni sur l'EAM (6,08 contre 4,99) ni sur le score logarithmique (−4,03 contre
−3,21), et sa couverture à 90 % (53 %) est hors de [80 %, 97 %]. **Le prior analogue construit
sur les législatives 2022 ne prévoit pas mieux les duels RN contre LFI de 2024 que la matrice
de proximité ; il fait moins bien.** Rien ne justifie de remplacer la matrice par cet analogue.

Lecture :

- Les deux priors surestiment le RN en 2024, (a) de 4,8 points, (b) de 6,1 ; (b) surestime
  le RN dans les 47 circonscriptions. En 2022, les mêmes moyennes sont presque sans biais
  (−0,4 point pour la matrice et pour l'analogue, en échantillon pour ce dernier). Ce qui a
  changé entre 2022 et 2024, c'est le comportement des électorats non finalistes : l'électorat
  du centre est estimé à 70 % pour le RN parmi ses exprimés en 2022 [IC 90 % : 59–83], à 24 %
  en 2024 [16–32] ; celui de la droite à 73 % [64–82] puis 45 % [29–57].
- L'analogue d'une seule élection, traité par le modèle avec κ = 20 et σ_u = 0,30, est bien
  trop confiant au regard de cette variation d'une élection à l'autre : intervalle deux fois
  plus étroit que la matrice, centré au mauvais endroit.
- Un analogue de la même année (validation croisée par département) prévoit bien (EAM 2,1,
  couverture 96 %) : les reports sont assez homogènes entre circonscriptions d'une même
  élection, mais pas stables d'une élection à l'autre. C'est la variance inter-élections, et
  non la variance locale, qui domine ; elle n'est mesurable qu'avec plusieurs analogues.

## Reports estimés par électorat et type de duel

Régression de Goodman contrainte (`vvs.model.ecological.solve_constrained`, lignes sur le
simplexe, pondération par les inscrits), toutes lignes libres, sur les circonscriptions de
chaque type de duel ; intervalles à 90 % par bootstrap des circonscriptions (500 tirages).
Électorats du premier tour en part des inscrits ; « ni l'un ni l'autre » = abstention, blanc ou
nul au second tour. « Adversaire » = finaliste opposé au RN (LFI, gauche non LFI, centre ou
droite selon le type). Fichiers : `runs/2026-11-24-duel-rn-lfi/transfers_estimates.parquet`
(et `_own_fixed` avec fidélités fixées au prior du modèle, chiffres très proches).

Les lignes « propres » des finalistes (fidélités de 100 %) et celles de poids inférieur à 1 %
ne sont pas interprétables : la régression les sature aux bornes. La ligne « abstention »
mesure l'effet de la participation : la mobilisation entre les tours d'électeurs absents au
premier tour est faible (de 3 à 8 % des abstentionnistes vont à un finaliste) et sa
répartition entre RN et adversaire n'est pas identifiée (intervalles de 0 à 100 %).


**2022, rn_lfi** (33 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 11.7 | 100 | 0 | 0 [0–0] | 100 [96–100] |
| adversaire (propre) | 11.7 | 0 | 100 | 0 [0–6] | 0 [0–0] |
| gauche radicale (autres) (poids < 1 %, non interprétable) | 0.5 | 0 | 0 | 100 [42–100] | 0 [0–3] |
| gauche (autres) | 3.0 | 24 | 40 | 35 [26–44] | 38 [27–49] |
| centre | 10.5 | 30 | 13 | 57 [51–63] | 70 [59–83] |
| droite | 5.0 | 45 | 17 | 38 [29–46] | 73 [64–82] |
| abstention+blancs T1 | 54.6 | 0 | 6 | 94 [92–95] | 0 [0–10] |

**2022, rn_centre** (108 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 13.1 | 100 | 0 | 0 [0–0] | 100 [100–100] |
| adversaire (propre) | 12.3 | 0 | 92 | 8 [4–12] | 0 [0–0] |
| gauche radicale (autres) | 7.3 | 30 | 35 | 35 [31–40] | 46 [37–51] |
| gauche (autres) | 4.7 | 29 | 39 | 33 [29–37] | 42 [35–47] |
| centre (poids < 1 %, non interprétable) | 0.2 | 19 | 54 | 27 [8–43] | 26 [0–46] |
| droite | 5.4 | 31 | 56 | 13 [9–17] | 36 [32–40] |
| abstention+blancs T1 | 53.5 | 1 | 3 | 96 [95–97] | 16 [0–53] |

**2022, rn_right** (31 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 11.9 | 100 | 0 | 0 [0–0] | 100 [99–100] |
| adversaire (propre) | 14.1 | 0 | 85 | 15 [10–20] | 0 [0–0] |
| gauche radicale (autres) | 6.5 | 31 | 36 | 33 [20–54] | 46 [19–59] |
| gauche (autres) | 3.7 | 30 | 42 | 28 [15–49] | 42 [16–54] |
| centre | 6.8 | 10 | 67 | 22 [0–37] | 13 [0–27] |
| droite (poids < 1 %, non interprétable) | 0.6 | 35 | 55 | 10 [0–31] | 39 [4–84] |
| abstention+blancs T1 | 53.8 | 1 | 6 | 93 [90–96] | 19 [0–51] |

**2022, rn_left_other** (26 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 13.6 | 100 | 0 | 0 [0–8] | 100 [99–100] |
| adversaire (propre) | 14.5 | 0 | 92 | 8 [0–15] | 0 [0–0] |
| gauche radicale (autres) (poids < 1 %, non interprétable) | 0.7 | 0 | 100 | 0 [0–69] | 0 [0–100] |
| gauche (autres) | 2.1 | 30 | 70 | 0 [0–29] | 30 [0–64] |
| centre | 8.9 | 24 | 39 | 38 [25–46] | 38 [24–51] |
| droite | 3.7 | 40 | 24 | 36 [28–49] | 63 [41–83] |
| abstention+blancs T1 | 54.0 | 0 | 3 | 97 [94–99] | 0 [0–50] |

**2024, rn_right** (58 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 26.3 | 98 | 2 | 0 [0–4] | 98 [95–100] |
| adversaire (propre) | 21.9 | 0 | 95 | 5 [2–9] | 0 [0–0] |
| gauche radicale (autres) | 7.1 | 0 | 86 | 14 [9–20] | 0 [0–4] |
| gauche (autres) | 7.2 | 0 | 84 | 16 [11–21] | 0 [0–3] |
| centre | 3.3 | 7 | 84 | 9 [5–14] | 8 [3–14] |
| droite (poids < 1 %, non interprétable) | 0.1 | 32 | 68 | 0 [0–9] | 32 [0–100] |
| abstention+blancs T1 | 33.4 | 7 | 0 | 93 [88–94] | 100 [50–100] |

**2024, rn_centre** (132 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 25.1 | 100 | 0 | 0 [0–0] | 100 [100–100] |
| adversaire (propre) | 21.0 | 0 | 92 | 8 [1–14] | 0 [0–0] |
| gauche radicale (autres) | 9.0 | 1 | 86 | 13 [8–18] | 1 [0–3] |
| gauche (autres) | 7.1 | 0 | 87 | 13 [8–18] | 0 [0–2] |
| centre (poids < 1 %, non interprétable) | 0.5 | 12 | 65 | 24 [11–38] | 15 [8–23] |
| droite | 3.5 | 17 | 71 | 13 [6–18] | 19 [16–23] |
| abstention+blancs T1 | 32.5 | 5 | 3 | 92 [89–95] | 59 [39–92] |

**2024, rn_lfi** (47 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 24.5 | 99 | 0 | 1 [0–5] | 100 [100–100] |
| adversaire (propre) | 19.9 | 0 | 100 | 0 [0–4] | 0 [0–0] |
| gauche radicale (autres) (poids < 1 %, non interprétable) | 0.8 | 0 | 75 | 25 [0–79] | 0 [0–47] |
| gauche (autres) | 1.1 | 10 | 62 | 28 [15–37] | 14 [0–35] |
| centre | 13.0 | 15 | 48 | 37 [29–43] | 24 [16–32] |
| droite | 4.1 | 26 | 33 | 41 [30–50] | 45 [29–57] |
| abstention+blancs T1 | 35.3 | 3 | 3 | 93 [89–99] | 49 [0–78] |

**2024, rn_left_other** (100 circ.)

| Électorat T1 | poids (% inscrits) | vers RN | vers adversaire | ni l'un ni l'autre [IC 90 %] | part RN parmi ses exprimés [IC 90 %] |
|---|---|---|---|---|---|
| RN (propre) | 24.9 | 100 | 0 | 0 [0–0] | 100 [100–100] |
| adversaire (propre) | 21.5 | 0 | 100 | 0 [0–0] | 0 [0–0] |
| gauche radicale (autres) (poids < 1 %, non interprétable) | 0.8 | 0 | 100 | 0 [0–0] | 0 [0–16] |
| gauche (autres) (poids < 1 %, non interprétable) | 0.7 | 0 | 77 | 23 [13–47] | 0 [0–24] |
| centre | 13.2 | 18 | 54 | 27 [24–30] | 25 [22–28] |
| droite | 4.1 | 26 | 48 | 25 [21–30] | 35 [29–42] |
| abstention+blancs T1 | 33.8 | 1 | 3 | 96 [95–98] | 29 [0–73] |

Ce qu'il faut retenir, avec la prudence due à l'inférence écologique :

- **Duels RN contre LFI de 2024** : l'électorat Ensemble se répartit 15 % RN, 48 % LFI, 37 %
  ni l'un ni l'autre (24 % pour le RN parmi ceux qui choisissent) ; l'électorat LR-DVD 26 %,
  33 %, 41 % (45 % pour le RN parmi ceux qui choisissent).
- **Face à un candidat de gauche non LFI en 2024**, le centre se répartit 18 / 54 / 27 et la
  droite 26 / 48 / 25 : la part RN parmi les exprimés est voisine (25 % et 35 %), mais
  l'abstention est plus forte face à LFI (37 % contre 27 % pour le centre). La pénalité LFI
  passe surtout par l'abstention, pas par un vote RN.
- **En 2022**, les estimations sont très différentes : le centre aurait voté RN à 70 % parmi
  ses exprimés face à LFI (38 % face à la gauche non LFI). Le sondage Harris du jour du vote
  2022 (duels NUPES contre RN, électeurs Ensemble : 34 NUPES, 18 RN, 48 non-exprimés) dit
  l'inverse : 35 % pour le RN parmi les exprimés. Avec 33 circonscriptions, un biais
  d'agrégation peut suffire à expliquer l'écart ; l'estimation 2022 RN contre LFI est la plus
  fragile du tableau.
- **Front républicain 2024** : dans les duels RN contre centre ou droite, les électorats de
  gauche votent à 0-3 % pour le RN ; en 2022, de 29 à 31 % des électorats de gauche allaient
  au RN face à Ensemble (Harris 2022 : 24 % RN, 31 % Ensemble, 45 % non-exprimés).

## Diagnostic annexe : les enquêtes de reports publiées

Fichier `data/registry/transfers_legislatives.yaml` (19 tableaux de 11 documents). Les
chiffres utilisés ci-dessous ont été relus dans les PDF (diapositives 13 à 15 d'Elabe, p. 6
d'Ifop, diapositive 14 de Harris) ; les autres ont été lus par le sous-agent de recherche.

Enquêtes appliquées aux premiers tours des duels RN contre LFI (prévision ponctuelle avec les
moyennes des reports, électorats non couverts : matrice de proximité ; deuxième colonne :
électorats non couverts pris dans l'estimation écologique de la même année, ce qui isole
l'effet des lignes de l'enquête) :

| Enquête | Moment | Électorats couverts | Biais (pts RN) | Biais, autres électorats estimés |
|---|---|---|---|---|
| Elabe, 3-4 juillet 2024, duel NFP-DVG contre RN | entre les tours | NFP, Ensemble, RN | +5,4 | +3,8 |
| Ifop LCI, 1er-2 juillet 2024, souhait de victoire, candidat LFI contre RN | entre les tours | NFP, Ensemble, LR-DVD, RN (répartition seule) | +2,4 | +2,0 |
| Harris, 19 juin 2022, duels NUPES contre RN (appliqué aux duels RN contre LFI 2022) | jour du vote | NUPES, DVG, Ensemble, LR, RN, Reconquête | −2,9 | −8,1 |
| Référence : matrice de proximité seule, 2024 | | | +4,7 | |
| Référence : estimation écologique 2024 seule (en échantillon) | | | +0,5 | |

Les deux enquêtes d'avant le second tour de 2024 surestiment le RN dans les duels RN contre
LFI de **2 à 4 points** une fois isolées (2 à 5,4 points avec la matrice pour le reste). C'est
le même sens et le même ordre de grandeur que le biais de 2,97 points que le modèle applique
aux sondages des duels RN contre centre. Deux réserves : ce sont des reports appliqués à des
premiers tours de circonscription, pas des intentions de vote de duel, et Elabe sous-estime
surtout la fidélité NFP (86 %, 11 % d'abstention) par rapport au RN (92 %, 7 %). Le sondage
Harris du jour du vote 2022 penche dans l'autre sens, mais c'est une déclaration après coup
et il contredit l'estimation écologique 2022 (voir plus haut).

Aucune matrice Ipsos de 2024, aucune matrice Ifop, Harris, Cluster17, Odoxa ou OpinionWay
de 2024, ni aucun sondage hypothétique Le Pen contre Mélenchon de 2017 ou 2022 avec reports
détaillés n'a été trouvé. Les sondages présidentiels 2026 trouvés (Toluna-Harris, mai, juillet
et août ; Elabe, mars, juillet et août) sont au registre ; ils ne servent pas au test.

## Sensibilité 2027 : Le Pen contre Mélenchon, prior seul

Tirages de la projection publiée le 2026-11-24 où Le Pen et Mélenchon sont qualifiés (22 400
tirages ; premiers tours moyens : Le Pen 30,8, Mélenchon 22,4, gauche 11,1, centre 10,9,
droite 10,9, autres extrême droite 4,9 ; abstention 33,3 pour 100 exprimés). Un tirage des
reports par tirage de premier tour, sans sondage de duel :

| Prior | Médiane Le Pen | Intervalle 90 % | P(Le Pen > 50 %) |
|---|---|---|---|
| (a) matrice de proximité (prior actuel du modèle) | 54,0 | 38,4 – 68,9 | 66 % |
| (b) analogue législatives 2022 | 57,4 | 42,6 – 70,6 | 81 % |
| analogue législatives 2024 (toutes les circonscriptions, en échantillon) | 50,4 | 36,5 – 65,1 | 52 % |

La médiane de (a) retrouve celle du diagnostic du même jour (53,97). Le choix de l'analogue
déplace le prior de 7 points selon l'année : c'est l'ordre de grandeur de l'incertitude qu'un
analogue législatif apporterait, pas une information. Avec le poids actuel des sondages de duel
dans la postérieure (0,83 selon le diagnostic du duel), un déplacement du prior de 3 points
déplacerait la projection d'environ 0,5 point.

## Transposabilité législatives → présidentielle : la réserve principale

- **Participation** : 46-47 % aux législatives 2022, 67 % en 2024, contre 72-80 % aux
  seconds tours présidentiels. L'électorat qui choisit entre les tours n'est pas le même ; la
  ligne « abstention » des tableaux, faible et mal identifiée, ne se transpose pas.
- **Candidat local** : notoriété du sortant, implantation, profil du candidat LFI (les plus
  clivants médiatiquement sont surreprésentés parmi les duels perdus ou gagnés de peu). Rien
  de tel au second tour présidentiel, où l'adversaire est Mélenchon lui-même, dont l'image
  auprès des électeurs du centre et de la droite est plus négative que celle d'un candidat
  NFP moyen (Ifop LCI 2024 : 66 % des électeurs Ensemble souhaitent la victoire d'un candidat
  LFI contre le RN, 77 % pour un PS ou écologiste ; Elabe juillet 2026 : 24 % des électeurs
  Ensemble de 2024 votent Mélenchon contre Le Pen, en exprimés).
- **Front républicain par désistement** : en 2024, la moitié des duels RN contre LFI
  viennent d'un retrait du troisième qualifié, avec consigne explicite. Ce mécanisme
  n'existe pas à la présidentielle (deux qualifiés d'office) ; seuls les appels à voter
  subsistent.
- **Contexte** : 2022 (après la réélection de Macron, RN loin du pouvoir) et 2024 (après la
  dissolution, majorité RN possible) donnent des reports opposés. Une présidentielle où le RN
  peut gagner ressemble plus à 2024, mais rien ne permet de le mesurer avec deux points.
- **Inférence écologique** : on n'observe que des corrélations entre circonscriptions ; un
  comportement d'électorat lié à la composition sociale de la circonscription biaise les
  estimations sans que le bootstrap le voie. Les estimations 2022 RN contre LFI (33
  circonscriptions) et celles des électorats de moins de 1 % des inscrits ne sont pas fiables.
- **Offre réduite** : les accords NUPES et NFP suppriment presque toute la gauche non
  finaliste ; les électorats de gauche autres que le finaliste pèsent 1 à 3 % des inscrits
  dans les duels RN contre LFI, alors que Glucksmann et Tondelier en 2027 en pèseraient bien
  plus. Leur report n'est pas estimé par ces duels.

Conclusion pratique : garder la matrice de proximité pour le duel Le Pen – Mélenchon. Si l'on
veut un jour utiliser les législatives, il faut au moins deux élections analogues et une
dispersion inter-élections estimée (κ bien plus faible que 20), et le passer par les deux
niveaux de `docs/07-evaluation.md`.

## Rejouer

```bash
cd ~/workspace/vingtvingtsept
uv run --project back python -m vvs.ingest.legislative_duels ingest   # téléchargements (révisions Wikipédia figées)
uv run --project back python -m vvs.ingest.legislative_duels parse    # tables data/curated/legislative_*.parquet
uv run --project back python -m vvs.eval.duel_rn_lfi 500              # expérience (≈ 35 s, CPU), 500 tirages de bootstrap
uv run --project back pytest back/tests/test_legislative_duels.py back/tests/test_duel_rn_lfi.py
```

Graine 20261124 (`SEED`), 4 000 tirages par circonscription (`N_DRAWS`), commit `38015b3`
(code du modèle), projection 2027 lue dans `runs/2026-11-24-simulate-subst/`.
`legislative_duels.parquet` : sha256 `475c39d6…3878`.

## Fichiers

- `back/src/vvs/ingest/legislative_duels.py`, `data/registry/legislative_duels_sources.yaml`
- `back/src/vvs/eval/duel_rn_lfi.py`
- `data/registry/transfers_legislatives.yaml` (enquêtes publiées ; non lu par le modèle)
- `back/tests/test_legislative_duels.py`, `back/tests/test_duel_rn_lfi.py`
- `data/curated/legislative_candidates.parquet`, `data/curated/legislative_duels.parquet`
