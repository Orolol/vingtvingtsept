# Reports publiés pour l'élection en cours comme prior du second tour — 24 novembre 2026

## Protocole et critères (écrits le 2026-11-24, avant tout calcul de score)

Cette section est écrite après l'inventaire des sources (ci-dessous) et avant tout calcul de
score. Elle n'est pas retouchée après lecture des résultats ; toute précision ultérieure figure
dans une section distincte, datée.

**Hypothèse.** Les sondages de duel hypothétique publiés avant le premier tour donnent souvent
un tableau « reports de voix » : la répartition de chaque électorat de premier tour (intention
de vote) entre les deux finalistes et « ni l'un ni l'autre ». Le modèle ne lit pas ces tableaux :
il centre les reports sur des analogues historiques ou, pour un duel inédit, sur la matrice
`PROXIMITY`. Centrer le prior des reports d'un duel sur les tableaux publiés pour ce duel, avant
la date de prévision, combiné au premier tour fixé (`fixed_first_round`), devrait donner un second
tour cohérent et au moins aussi bon que la référence.

**Bras.** Chaîne chronologique de `docs/10-consolidation.md` : les douze cas du gel
`runs/2026-10-22-consolidation` (2012, 2017, 2022 × 180, 90, 30, 7 jours), mêmes `case.json`,
`curated/`, `fit/draws.npz`, spécification `reference.json` (40 000 tirages, graine 20261022,
second tour `bias_by_horizon`), calibrations recalculées et comparées au gel, agrégateur bayésien
seulement ; branchement comme `back/studies/fixed_first_round.py` (`VVS_ROOT` sur la copie figée,
code courant).

- **R0** : référence (mode conjoint, sans tableau), graine 20261022 ; doit reproduire
  `draws_reference.npz` du gel.
- **V1** : tableaux publiés, mode conjoint actuel, graine 20261022.
- **V2** : tableaux publiés + `fixed_first_round`, graine 20261022, k propositions par ligne :
  k = 1024, multiplié par 4 (jusqu'à 16 384) tant qu'un des douze cas a 1 % ou plus de ses lignes
  de second tour avec une taille effective par ligne inférieure à 2. Cette règle est appliquée
  par le script sur les seules tailles effectives, avant l'écriture des scores.
- **R1** : référence, graine 20261023 : bruit Monte-Carlo.
- Hors verdict : **F**, premier tour fixé sans tableau, même k que V2 (sépare l'effet du mode de
  celui des tableaux au même k) ; **S1**, sensibilité de V2 avec κ = 60 au lieu de 20.

**Source « élection en cours ».** Registre séparé `data/registry/transfers_current.yaml`, chargé
seulement si l'option est activée (`SecondRoundModel(current_transfers=…)`, désactivé par défaut ;
`transfers.yaml` n'est pas touché). Une entrée = un tableau d'un institut pour un duel d'une
élection, avec URL, page, dates de terrain, date de publication si lue, échantillon, base ;
chiffres recopiés tels quels de la source primaire, aucune cellule déduite par différence.

**Paramètres, fixés avant calcul.**

1. *Disponibilité et fenêtre.* Un tableau est disponible à sa date de publication lue dans la
   source, sinon à la fin du terrain + 2 jours (règle du gel). Pour un duel {a, b} à la date
   `asof` : tableaux de la même élection et de la même paire, disponibles au plus tard à `asof`,
   terrain terminé dans les 60 jours précédant `asof`. Aucun autre vieillissement.
2. *Agrégation des vagues.* Par institut, seul le tableau le plus récent (fin de terrain, puis
   disponibilité) ; poids égal entre instituts. Raison : un institut à vagues fréquentes (rolling)
   ne doit pas dominer, et la vague la plus récente d'un institut résume les précédentes.
3. *Électorats.* Chaque ligne d'un candidat non finaliste est rattachée au groupe du modèle
   `bloc:<bloc du candidat à cette élection>` (registre). Les lignes des finalistes (fidélité),
   « Ensemble » et « abstention, vote blanc » sont ignorées : la fidélité (`own_prior`, mesurée
   sur des reports réalisés) et le groupe des abstentionnistes gardent le prior actuel. Un groupe
   sans ligne garde le prior actuel (analogue ou matrice). Plusieurs candidats d'un même bloc dans
   une hypothèse : moyenne pondérée par leur part moyenne dans les sondages de premier tour
   terminés dans les 30 jours avant `asof` (toutes hypothèses ; poids égaux sans sondage).
   Plusieurs hypothèses de premier tour dans un tableau : moyenne à poids égal entre hypothèses.
4. *« Ne se prononcent pas ».* La colonne « abstention / blanc / nul / ne se prononce pas » de la
   source est prise telle quelle comme part « ni l'un ni l'autre » ; ligne renormalisée à 1,
   « - » = 0. Pas d'harmonisation de base (inscrits, certains d'aller voter, probabilité 8 à 10) :
   la différence de base fait partie de l'erreur de ces tableaux. Les tableaux publiés seulement
   en % des exprimés (sans part « ni l'un ni l'autre ») ne sont pas retenus.
5. *Concentration et erreur propre aux tableaux.* Moyenne du groupe = moyenne des tableaux ;
   Dirichlet de concentration κ = 20 (`KAPPA_CURRENT`), décalages communs u et v conservés avec
   σ_u et σ_v inchangés (selon que le duel a un analogue ou non). Justification : d'une vague à
   l'autre du même institut, une ligne bouge d'un écart-type de 5 à 6 points (Ifop 2012, électorat
   Bayrou vers Hollande de 29 à 48 %) ; on double cet ordre de grandeur pour l'écart, non mesurable
   ici sans calibrer sur les élections évaluées, entre intention déclarée et vote (κ = 20 donne un
   écart-type d'environ 11 points sur une part de 0,4). C'est le poids d'une élection analogue
   (`KAPPA_ANALOG[1]`). Valeur fixée par jugement, pas estimée.

**Métriques**, par cas, moyennes par horizon (poids égal aux trois élections) et moyenne des
douze cas, comme dans `docs/results/2026-11-24-premier-tour-fixe.md` : `log_score_winner`
(lissage de Jeffreys du gel), `crps2` et `coverage90_2` (part de second tour du duel réellement
disputé, sur les tirages où ce duel est qualifié ; un cas sans tirage de ce duel dans un bras est
exclu de ces moyennes pour tous les bras). Contrôles : `log_score_pair`, CRPS et couverture du
premier tour identiques entre tous les bras de même graine.

**Bruit.** d_c = M(R1, c) − M(R0, c), σ̂ = √(moyenne des d_c²), SE = σ̂ / √12 pour la moyenne des
douze cas (σ̂ / √3 par horizon).

**Verdict (V2 seulement).** Admis si, sur la moyenne des douze cas :

1. moyenne(V2 − R0) de `log_score_winner` ≥ −2 SE ;
2. moyenne(V2 − R0) de `crps2` ≤ +2 SE ;
3. |couverture90_2(V2) − 0,90| ≤ |couverture90_2(R0) − 0,90| + 2 SE.

Sinon : résultat négatif, défauts inchangés. V1 est rapporté pour séparer l'effet des tableaux de
celui du premier tour fixé ; F et S1 sont descriptifs. Quel que soit le verdict, aucun défaut ne
change dans ce travail : l'option est livrée désactivée.

**Réserves connues d'avance.**

- *Biais de collecte.* Seuls les tableaux du duel finalement disputé ont été recherchés pour 2012,
  2017 et 2022 ; les autres duels testés à l'époque (Fillon – Le Pen, Macron – Pécresse…) gardent
  le prior actuel. L'option n'agit donc que sur le duel réalisé. Ce n'est pas une fuite de
  résultat (aucune donnée postérieure à `asof`), mais un choix de collecte informé par l'issue.
- *Double emploi.* Le tableau et la part agrégée du duel viennent des mêmes enquêtes : centrer le
  prior sur les tableaux réutilise en partie l'information que la vraisemblance des sondages de
  duel porte déjà. κ = 20 et les décalages communs laissent le prior large.
- *Cas sans tableau.* 2012 et 2017 à 180 jours (et une partie des électorats ailleurs) n'ont
  aucun tableau : V1 y est identique à R0.
- Trois élections, douze cas non indépendants.

**Sensibilité 2027 (pas un critère).** Projection du 2026-11-24 rejouée (`--fit
../runs/2026-11-24-fit-2027-subst --seed 0 --substitution --rn-bias-curve`) pour R0, V1, V2 :
probabilité de victoire par candidat ; pour les dix duels les plus probables, P(duel), part
médiane et intervalle à 90 %, P(a gagne) ; reports a posteriori par électorat du duel Le Pen –
Mélenchon comparés aux tableaux Harris. Sorties : `runs/2026-11-24-reports-publies/`.

## Inventaire des tableaux (avant calcul)

Recherche du 2026-11-24 (trois sous-agents de recherche web, puis relecture par l'agent principal
de chaque tableau retenu dans le texte du PDF, `pdftotext -layout`). PDF conservés sous
`runs/2026-11-24-reports-publies/sources/` (hors Git). Aucune source avant premier tour dans
`transfers.yaml` (le seul tableau d'avant second tour, `ifop_2012_soir_t1`, date du soir du
premier tour) ; aucune notice de la Commission des sondages antérieure à 2025 dans
`data/raw/commission/`.

Disponibilité = publication lue, sinon fin de terrain + 2 jours. « Utilisé » : tableau retenu par
la règle 2 (le plus récent de chaque institut, terrain dans les 60 jours) pour le cas.

| Élection, horizon (`asof`) | Tableaux disponibles (fin de terrain) | Utilisés |
|---|---|---|
| 2012, 180 j (2011-10-25) | aucun | — |
| 2012, 90 j (2012-01-23) | Ifop JDD 15 déc. ; Ifop JDD 6 janv. ; Ifop baromètre v1 13 janv. | Ifop v1 |
| 2012, 30 j (2012-03-23) | Ifop baromètres v3 12 fév., v4 26 fév., v5 12 mars ; Ifop JDD 17 mars | Ifop JDD 17 mars |
| 2012, 7 j (2012-04-15) | + Ifop v6 25 mars, v7 7 avril (publié le 10) | Ifop v7 |
| 2017, 180 j (2016-10-25) | aucun (duel Macron – Le Pen non testé avant fin novembre 2016) | — |
| 2017, 90 j (2017-01-23) | Ifop 3 déc. 2016 ; Ipsos France 2 20 janv. (électorats de gauche seulement) | les deux |
| 2017, 30 j (2017-03-24) | Ifop 4 mars, rolling 10 mars ; Harris 8 mars, 22 mars ; Ipsos Cevipof v12 15 mars | Ifop 10 mars, Ipsos 15 mars, Harris 22 mars |
| 2017, 7 j (2017-04-16) | + Ipsos 27 mars, v12bis 2 avril, 9 avril, 13 avril ; Ifop 31 mars, 6 avril ; Harris 5 avril, 6 avril | Ifop 6 avril, Ipsos 13 avril, Harris 6 avril |
| 2022, 180 j (2021-10-12) | Elabe 6 oct. (trois hypothèses LR) ; Harris v16 4 oct. | les deux |
| 2022, 90 j (2022-01-10) | Harris v25 6 déc. ; Elabe 20 déc. ; OpinionWay 22 déc. | les trois |
| 2022, 30 j (2022-03-11) | Ipsos v5 7 fév. ; Harris v32 7 fév. ; Elabe 22 fév. ; Ipsos v6 3 mars | Ipsos v6, Elabe 22 fév., Harris v32 |
| 2022, 7 j (2022-04-03) | + Elabe 21 mars, 1er avril ; Ipsos v8 24 mars ; Ifop rolling 29 mars, 1er avril | Elabe 1er avril, Ifop 1er avril, Ipsos v8, Harris v32 |
| 2027 (2026-11-24) | Harris notices 2321 (8 oct.), 2351 (21 oct.), 2381 (5 nov.), 2417 (19 nov.) : Le Pen – Mélenchon ; aussi Le Pen – Philippe et Le Pen – Glucksmann | Harris 19 nov. pour Le Pen – Mélenchon ; dernière vague de chaque autre duel |

Le seuil fixé avant l'expérience (au moins deux élections à au moins deux horizons) est atteint :
2012 et 2017 à trois horizons, 2022 aux quatre. Sources détaillées, URL, pages et chiffres :
`data/registry/transfers_current.yaml`.

Cherché sans succès : CSA, TNS Sofres, BVA, LH2, OpinionWay 2012 ; Ifop rolling 2017 du 17 mars
au 14 avril ; Elabe 2017 (détail par vote 2012 seulement) ; Kantar, BVA, Odoxa 2017 et 2022 ;
Harris 2022 vagues 26-28 et 33-36 (PDF introuvables) ; Ifop 2022 avant le 28 mars (pas de
croisement par intention 2022) ; Ipsos 2022 vague 4 (publiée après le 10 janvier).

## Résultats (calculés après l'écriture des sections précédentes)

Sorties : `runs/2026-11-24-reports-publies/chain/` (hors Git : `manifest.json`, `k_choice.json`,
`metrics.csv`, `results.json`, tirages, cellules et notes par cas). Six bras × douze cas,
40 000 tirages, six processus en parallèle, environ 1 h 15 de calcul.

**Contrôles.** Le rejeu R0 reproduit à l'identique `draws_reference.npz` du gel dans les douze
cas. `log_score_pair`, CRPS et couverture du premier tour sont identiques entre R0 et chacun des
bras V1, V2, F, S1 (écart maximal 0). Les calibrations recalculées sont égales à celles du gel.
Sans tableau (2012 et 2017 à 180 jours), V1 est identique à R0, comme prévu.

**Choix de k (règle déclarée, sur les seules tailles effectives).** Part maximale, sur les douze
cas, des lignes de second tour de taille effective < 2 sous V2 : 2,9 % à k = 1024, 1,6 % à
k = 4096, 0,9 % à k = 16 384 (toujours 2012-180, cas sans tableau). **k = 16 384** pour V2, F et S1.

**Précision de provenance.** Pendant le calcul, deux sources ont changé après l'écriture du
manifeste : une mise en forme de la note de duel dans `second_round.py` (même texte produit) et
l'ajout des options de `projection` dans le script d'étude (non utilisées par la chaîne). Les
empreintes du manifeste sont donc celles du lancement. Trois cas (2012-090, 2017-007, 2022-030)
rejoués ensuite sous V1 avec les sources finales donnent des tirages identiques octet pour octet.

### Tableau principal (moyennes à poids égal par élection ; 2 SE entre parenthèses)

| Horizon | Log victoire R0 / V1 / V2 / R1 | Δ V1 − R0 | Δ V2 − R0 (2 SE) | CRPS 2ᵉ tour R0 / V1 / V2 / R1 | Δ V1 − R0 | Δ V2 − R0 (2 SE) |
|---|---|---|---|---|---|---|
| 180 | −0,938 / −0,943 / −0,957 / −0,937 | −0,0043 | −0,0180 (0,0059) | 3,302 / 3,423 / 3,250 / 3,294 | +0,121 | −0,052 (0,021) |
| 90 | −0,518 / −0,518 / −0,528 / −0,523 | +0,0004 | −0,0103 (0,0059) | 2,407 / 2,636 / 2,786 / 2,400 | +0,229 | +0,380 (0,021) |
| 30 | −0,116 / −0,116 / −0,117 / −0,118 | +0,0001 | −0,0009 (0,0059) | 2,555 / 2,365 / 2,299 / 2,554 | −0,190 | −0,256 (0,021) |
| 7 | −0,106 / −0,106 / −0,106 / −0,107 | −0,0001 | −0,0008 (0,0059) | 1,870 / 2,262 / 2,326 / 1,856 | +0,392 | +0,457 (0,021) |
| **12 cas** | −0,420 / −0,421 / −0,427 / −0,421 | −0,0010 | **−0,0075** (0,0029) | 2,533 / 2,671 / 2,665 / 2,526 | +0,138 | **+0,132** (0,011) |

Couverture à 90 % de la part du second tour : 0,667 dans tous les bras (2012 jamais couvert,
2017 et 2022 toujours). Bruit (R1 − R0) : σ̂ par cas 0,0051 (log victoire), 0,018 (CRPS du
second tour), 0 (couverture).

Hors verdict, moyennes des douze cas : **F** (premier tour fixé sans tableau, k = 16 384) Δ log
victoire −0,0043, Δ CRPS 2ᵉ tour −0,014 ; **S1** (V2 avec κ = 60) −0,0079 et +0,159.

### Par cas

Part de second tour du finaliste premier par ordre alphabétique (Hollande en 2012, Le Pen en 2017
et 2022), tirages où le duel réel est qualifié.

| Cas | P(vainqueur) R0 / V1 / V2 / F | CRPS 2ᵉ tour R0 / V1 / V2 / F | Médiane R0 / V1 / V2 / F (réel) |
|---|---|---|---|
| 2012-180 | 93,4 / 93,4 / 92,8 / 92,8 % | 6,24 / 6,24 / 4,88 / 4,88 | 59,0 / 59,0 / 58,0 / 58,0 (51,64) |
| 2012-90 | 84,0 / 84,0 / 83,9 / 83,6 % | 4,76 / 5,28 / 5,13 / 3,82 | 57,5 / 58,0 / 58,0 / 56,7 |
| 2012-30 | 89,2 / 89,2 / 89,1 / 88,9 % | 3,93 / 4,22 / 3,85 / 3,30 | 56,6 / 56,9 / 56,7 / 56,1 |
| 2012-7 | 96,3 / 96,3 / 96,2 / 96,2 % | 2,93 / 3,14 / 3,07 / 2,69 | 55,6 / 55,8 / 55,8 / 55,4 |
| 2017-180 | 8,0 / 8,0 / 7,9 / 7,9 % | 2,85 / 2,85 / 3,51 / 3,51 | 38,5 / 38,5 / 39,6 / 39,6 (33,90) |
| 2017-90 | 32,1 / 32,0 / 31,5 / 31,6 % | 1,68 / 1,81 / 2,22 / 2,11 | 36,7 / 37,0 / 37,5 / 37,4 |
| 2017-30 | 81,8 / 81,8 / 81,7 / 81,8 % | 2,20 / 2,17 / 2,42 / 2,44 | 37,4 / 37,3 / 37,6 / 37,7 |
| 2017-7 | 76,3 / 76,3 / 76,2 / 76,2 % | 2,18 / 2,37 / 2,44 / 2,23 | 37,2 / 37,5 / 37,6 / 37,3 |
| 2022-180 | 80,3 / 79,3 / 77,4 / 79,9 % | 0,82 / 1,18 / 1,36 / 1,49 | 41,2 / 43,3 / 43,5 / 39,1 (41,45) |
| 2022-90 | 78,3 / 78,6 / 77,6 / 78,1 % | 0,78 / 0,82 / 1,01 / 1,21 | 40,6 / 42,5 / 42,9 / 39,5 |
| 2022-30 | 96,7 / 96,7 / 96,7 / 96,7 % | 1,54 / 0,70 / 0,62 / 2,04 | 38,9 / 40,5 / 41,2 / 38,2 |
| 2022-7 | 99,2 / 99,1 / 99,1 / 99,2 % | 0,51 / 1,28 / 1,47 / 0,51 | 41,7 / 43,5 / 43,8 / 41,3 |

### Verdict : non admis (résultat négatif)

1. Log-score du vainqueur : Δ(V2 − R0) = −0,0075 < −2 SE = −0,0029. **Échec.**
2. CRPS de la part du second tour : Δ = +0,132 > +2 SE = +0,011. **Échec.**
3. Couverture à 90 % : 0,667 contre 0,667. Satisfait.

Deux critères sur trois échouent : **l'option reste désactivée, aucun défaut ne change.**

**Lecture.** V1 sépare l'effet des tableaux : à mode conjoint inchangé, ils ne touchent presque
pas le log-score du vainqueur (−0,0010, dans le bruit) mais dégradent le CRPS du second tour
(+0,138). F sépare l'effet du premier tour fixé : −0,0043 en log-score (déjà sous la barre, comme
dans l'étude du premier tour fixé) et −0,014 en CRPS. V2 cumule à peu près les deux.

Les tableaux d'avant premier tour déplacent la part dans le sens où les sondages de duel se sont
trompés, sauf à 30 jours (2022 nettement, 2017 à peine) : ils poussent Hollande vers le haut en
2012 (l'électorat Le Pen y donne 20 à 33 % à Hollande, l'électorat Bayrou 29 à 48 %), et Le Pen
vers le haut en 2017 et en 2022 (électorat Mélenchon 2022 : 10 à 35 % vers Le Pen selon les
tableaux, autour de 20 % le plus souvent ; électorat Dupont-Aignan 2017 : 36 et 42 %). Le centre des tableaux ressemble aux sondages de duel, dont il
partage l'échantillon et les biais : il n'apporte pas l'information indépendante que l'hypothèse
supposait, et resserre le prior autour de la même erreur. Avec trois élections, on ne peut pas
dire si un κ plus faible ou une correction de biais propre aux tableaux changerait le verdict ;
S1 (κ = 60) fait légèrement pire que V2.

**Ce que le résultat ne dit pas.** Il ne dit pas que la matrice `PROXIMITY` est juste pour Le Pen –
Mélenchon (aucun duel extrême droite contre gauche radicale dans la chaîne) : il dit que, sur les
trois duels historiques, centrer les reports sur les tableaux publiés de l'élection en cours ne
prévoit pas mieux le second tour que les analogues et la matrice. Le biais de collecte (seul le
duel réalisé a des tableaux) n'a pas suffi à produire un gain.

## Sensibilité 2027 (pas un critère)

Projection du 2026-11-24 rejouée (`runs/2026-11-24-reports-publies/projection/`, graine 0,
substitution, courbe de biais, 40 000 tirages). R0 reproduit à l'identique les tirages publiés
(`runs/2026-11-24-simulate-subst/draws.parquet`). V2 à k = 16 384 : 0,015 % des lignes de taille
effective < 2. Tableaux Harris retenus : vague du 18-19 novembre pour Le Pen – Mélenchon, du 3-5
novembre pour Le Pen – Philippe et Le Pen – Glucksmann ; Bardella n'a pas de tableau dans la
fenêtre.

| Candidat | R0 | V1 | V2 |
|---|---|---|---|
| Marine Le Pen | 73,4 % | 73,9 % | 71,4 % |
| Jordan Bardella | 9,2 % | 9,3 % | 8,7 % |
| Édouard Philippe | 6,0 % | 5,4 % | 7,3 % |
| David Lisnard | 5,6 % | 5,6 % | 5,5 % |
| Raphaël Glucksmann | 2,5 % | 2,3 % | 2,6 % |
| Jean-Luc Mélenchon | 1,4 % | 1,4 % | 2,0 % |

Dix duels les plus probables (P(duel) identique dans les trois bras, premier tour commun) ; part
médiane de a [intervalle à 90 %] ; P(a gagne) :

| Duel (a – b) | P(duel) | R0 | V1 | V2 |
|---|---|---|---|---|
| Le Pen – Mélenchon | 56,0 % | 63,1 [56,8–69,2] ; 100,0 % | 63,9 [57,7–70,1] ; 100,0 % | 62,6 [54,8–69,7] ; 99,3 % |
| Le Pen – Philippe | 10,0 % | 52,2 [45,9–58,7] ; 72,4 % | 52,8 [46,7–59,2] ; 76,7 % | 51,1 [43,4–59,2] ; 59,2 % |
| Le Pen – Glucksmann | 8,2 % | 55,2 [49,1–61,5] ; 91,5 % | 56,1 [49,4–62,6] ; 93,1 % | 55,9 [48,7–62,9] ; 90,9 % |
| Bardella – Mélenchon | 7,5 % | 63,8 [55,7–71,8] ; 99,7 % | 63,6 [55,4–71,6] ; 99,6 % | 60,5 [51,3–69,6] ; 97,0 % |
| Le Pen – Lisnard | 4,9 % | 41,2 [26,8–57,6] ; 19,3 % | 41,2 [26,6–58,4] ; 19,0 % | 40,9 [26,8–58,3] ; 19,9 % |
| Mélenchon – Philippe | 2,0 % | 39,1 [33,0–45,3] ; 0,5 % | 39,1 [32,6–45,5] ; 0,2 % | 40,1 [33,2–48,2] ; 3,1 % |
| Mélenchon – Glucksmann | 1,9 % | 47,4 [31,6–63,7] ; 38,5 % | 46,7 [30,6–63,0] ; 37,6 % | 47,5 [31,7–63,7] ; 40,3 % |
| Bardella – Philippe | 1,3 % | 48,8 [41,7–57,2] ; 39,8 % | 49,1 [41,0–57,2] ; 45,5 % | 46,4 [37,5–57,0] ; 29,9 % |
| Bardella – Glucksmann | 1,1 % | 54,4 [46,3–64,4] ; 78,5 % | 55,6 [46,8–64,4] ; 83,6 % | 53,8 [43,5–63,3] ; 72,3 % |
| Mélenchon – Lisnard | 1,1 % | 44,3 [30,7–59,6] ; 26,9 % | 44,8 [30,2–63,6] ; 28,8 % | 44,6 [31,6–61,7] ; 27,6 % |

Reports a posteriori du duel Le Pen – Mélenchon (médianes par tirage, % de l'électorat : Le Pen /
Mélenchon / ni l'un ni l'autre), comparés aux tableaux Harris (vague du 18-19 novembre, utilisée ;
moyenne des quatre vagues d'octobre-novembre entre parenthèses quand l'électorat y figure) :

| Électorat (groupe du modèle) | R0 | V1 | V2 | Harris |
|---|---|---|---|---|
| Glucksmann (`bloc:left`) | 1 / 76 / 18 | 8 / 25 / 64 | 10 / 24 / 63 | 9 / 28 / 63 (11 / 27 / 62) |
| Philippe, Attal (`bloc:centre`) | 27 / 24 / 42 | 27 / 15 / 55 | 30 / 14 / 54 | Philippe 23 / 17 / 60 (30 / 10 / 60) ; Attal 34 / 21 / 45 (31 / 15 / 54) |
| Retailleau, Lisnard (`bloc:right`) | 66 / 3 / 27 | 64 / 4 / 29 | 66 / 4 / 28 | Retailleau 60 / 4 / 36 (63 / 3 / 34) ; Lisnard 65 / 8 / 27 |
| Riner (`bloc:other`) | 45 / 18 / 30 | 33 / 17 / 47 | 36 / 16 / 46 | 33 / 20 / 47 |
| Abstentionnistes | 16 / 7 / 73 | 14 / 8 / 74 | 19 / 6 / 69 | non mesuré |

Les tableaux corrigent les deux reports que le diagnostic du jour jugeait intenables (gauche et
centre vers Mélenchon) sans que la part médiane du duel bouge beaucoup : +0,8 point en mode
conjoint, −0,5 point avec le premier tour fixé (60,7 à 60,9 avec le premier tour fixé seul,
étude du même jour). Le duel Le Pen – Philippe est le plus sensible (P(Le Pen gagne) de 72 % à 59 % sous
V2). Ces chiffres ne sont pas une prévision admise : l'option est rejetée par la chaîne.

## Rejouer

Depuis `back/` (CPU, `JAX_PLATFORMS=cpu`) :

```sh
uv run python studies/current_transfers.py chain --reference ../runs/2026-10-22-consolidation \
    --out ../runs/2026-11-24-reports-publies/chain --jobs 6
uv run python studies/current_transfers.py projection --fit ../runs/2026-11-24-fit-2027-subst \
    --out ../runs/2026-11-24-reports-publies/projection
uv run pytest tests/test_transfers_current.py
```

Code : HEAD `7bc1752` plus les modifications non committées de ce travail (empreintes dans
`chain/manifest.json`, voir la précision de provenance). Gel : `reference-2026-10-22` (commit
`a89d304`), dépendances identiques à celles du gel. Graines 20261022 (R0, V1, V2, F, S1) et
20261023 (R1) ; projection : graine 0.

## Fichiers

- `data/registry/transfers_current.yaml` (47 tableaux : 9 de 2012, 15 de 2017, 14 de 2022, 9 de
  2027), `back/src/vvs/registry/transfers_current.py` (chargement, validation, disponibilité)
- `back/src/vvs/model/second_round.py` : `current_tables`, `current_group_means`,
  `first_round_poll_shares`, `KAPPA_CURRENT`, options `current_transfers` / `current_kappa` de
  `SecondRoundModel` (désactivées) ; `back/src/vvs/model/simulate.py` : passage de l'option
- `back/studies/current_transfers.py`, `back/tests/test_transfers_current.py`

## Non vérifié

- Dates de publication non lues pour plusieurs tableaux (règle terrain + 2 jours) ; la date du
  10 avril 2012 (Ifop vague 7) vient de la page Ifop, pas du PDF.
- Effectifs de quelques baromètres Ifop 2012 relevés par le sous-agent sans relecture ; ils
  n'entrent pas dans le calcul.
- Les tableaux en graphique (Ifop 2012, rolling Ifop 2017 et 2022, Elabe) ont été relus dans le
  texte extrait du PDF, pas sur l'image ; l'ordre des colonnes des rollings Ifop est confirmé par
  la note de lecture (2017) ou la note de bas de page (2022).
- Recherche non exhaustive (CSA, Sofres, BVA, Kantar, Odoxa, rollings Ifop 2017 introuvables) ;
  les tableaux des duels non réalisés n'ont pas été cherchés.
