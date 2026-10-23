# vingtvingtsept

Prévision probabiliste de l'élection présidentielle française de 2027, dans l'esprit de
FiveThirtyEight : agrégation bayésienne des sondages depuis 2002, erreur calibrée sur
l'histoire, arbre de scénarios de candidats, second tour par reports de voix, simulation
Monte-Carlo. Site : https://orolol.github.io/vingtvingtsept/

Dates de travail : premier tour le 18 avril 2027, second tour le 2 mai, à confirmer par le
décret de convocation. La projection est mise à jour à chaque
nouveau sondage.

## Ce que fait le modèle

1. **Sondages agrégés** : soutien latent par candidat, à deux niveaux (bloc politique et
   écart propre), marches aléatoires hebdomadaires, effets maison par institut, effectif
   effectif avec plancher d'erreur, inférence NUTS (numpyro). Option de substitution
   intra-bloc (logit emboîté) pour les hypothèses de candidats différentes.
2. **Erreur calibrée** : l'incertitude ne vient pas d'une marge théorique mais des erreurs
   observées aux présidentielles 2002-2022, complétées par les européennes 2014-2024 et les
   législatives 2022-2024, corrélées par bloc, croissantes avec l'horizon (forme estimée sur
   la base internationale de Jennings et Wlezien).
3. **Scénarios de candidats** : probabilités jugées, datées et sourcées, sur six dimensions
   (candidat RN, bloc central, gauche, LR, LFI, Reconquête), mélangées dans la simulation.
4. **Second tour** : reports de voix par électorat, centrés sur dix-neuf sources
   historiques (instituts et panel Cevipof), corrigés par les sondages de duel avec un biais
   RN estimé par horizon.
5. **Simulation** : 40 000 tirages, probabilités de qualification et de victoire, scores
   avec intervalles, par scénario et au mélange.

Sous le protocole chronologique strict de la [consolidation](docs/10-consolidation.md),
l'agrégateur bayésien ne domine pas la moyenne simple des sondages des 14 derniers jours :
son gain net de log-score de victoire à 180 jours (+0,175) est porté par 2017, et il perd
légèrement à 90 jours (−0,008) et à 7 jours (−0,032). À 180 jours, la couverture de
l'intervalle à 90 % n'est que de 66 %. La valeur du modèle tient à la calibration de
l'erreur, aux scénarios et au second tour, que ce témoin partage, pas à l'agrégation seule ;
ces composants n'ont pas encore été évalués séparément dans la chaîne chronologique.

Les études utilisent un backtest rétrospectif laissant l'élection évaluée de côté
(2002-2022, horizons 180, 90, 30 et 7 jours) et une règle d'admission écrite : voir
[le protocole](docs/07-evaluation.md) et [le journal](docs/backtest-log.md).
La [consolidation du 22 octobre](docs/10-consolidation.md) fige une référence et ajoute
une évaluation chronologique de toute la chaîne, de l'offre de candidats à la victoire,
sur 2012, 2017 et 2022. Les dates de publication manquantes y sont explicitement imputées.
Version de référence : tag `reference-2026-10-22`, avec résultats et audit versionnés.
Une [étude des scénarios historiques incomplets](docs/11-scenarios-historiques.md)
teste une mémoire des hypothèses et un filtre par événements datés : elle réduit les
candidats hors support en 2017, mais n'améliore pas globalement les scores. Les deux
variantes restent expérimentales ; la référence et la projection publiée sont inchangées.

Une [vague de cinq expériences de structure](docs/13-experiences-structure.md) compare
un mélange 50/50, l'incertitude de calibration et les sondages continus dans la même
chaîne : aucun gain admissible. Le pilote de substitution entre blocs n'améliore pas
son témoin ; les transitions des indécis restent bloquées par l'absence de vagues
d'intentions appariables dans les microdonnées disponibles.

## Pistes étudiées

Règle du projet : aucune source n'entre dans la prévision sans gain au backtest, en
laissant l'élection évaluée de côté. Quinze signaux ont été testés ; un seul a été admis.
Les autres restent des options désactivées dans le code et, pour plusieurs, des volets
explicatifs du site marqués « n'entre pas dans la prévision ».

| # | Piste | Résultat |
|---|---|---|
| 1 | Économie (chômage, prix, confiance, popularité) | aucun effet distinguable du hasard sur la dynamique des blocs |
| 2 | Presse par comptages (part de voix, agenda, affaires) | aucun test ne passe ; mesure les mêmes événements que les sondages |
| 3 | Presse classée par un modèle local (ton, posture, thèmes de 58 000 titres) | négatif ; ton des 28 derniers jours déclaré test hors échantillon pour 2027 |
| 4 | Attention en ligne (pages vues Wikipédia, Google Trends) | précède les sondages de 1 à 4 semaines comme rattrapage, nul aux horizons utiles |
| 5 | Palmarès des instituts | précision et biais non reproductibles d'une élection à l'autre |
| 6 | Sûrs de leur choix | signe opposé entre 2017 et 2022 |
| 7 | Potentiel électoral et rejet | pas un plafond général ; seul le RN reste sous le sien |
| 8 | Rebonds de primaire et d'annonce | forme réelle pour les primaires (saut puis érosion complète) mais dérive non admise |
| 9 | Vote utile de fin de campagne | réel à gauche seulement, surcorrige ailleurs |
| 10 | Volatilité de la campagne comme échelle d'erreur | ne prédit pas l'erreur de projection en France |
| 11 | Volatilité par candidat | signal réel mais déjà capté par la loi d'erreur |
| 12 | Troupeau des instituts | resserrement réel, mais le désaccord accompagne les erreurs au lieu de les masquer |
| 13 | Anticipations « qui va gagner ? » (110 mesures) | le plus cité gagne toujours, mais trop sous-confiant pour servir de probabilité |
| 14 | Abstention différentielle (certains − inscrits) | stable, mais lien avec l'erreur limité à 2022 et de signe inverse |
| 15 | Reports de voix par inférence écologique sur les communes (données Cagé-Piketty) | biais d'agrégation ; la sociologie des communes ne prédit pas l'érosion du front républicain |
| ✓ | **Calibration de l'erreur finale sur dix scrutins** (présidentielles + européennes + législatives) | **admise** au protocole rétrospectif : log-score et CRPS meilleurs à 90, 30 et 7 jours ; confirmation chronologique à faire |

Ces verdicts relèvent du protocole rétrospectif. Depuis la consolidation, toute admission
doit en plus être confirmée sous le protocole chronologique ([règle à deux
niveaux](docs/07-evaluation.md)) : la calibration admise, la substitution et la courbe de
biais publiées n'ont pas encore passé cette confirmation. Sous ce même protocole,
l'agrégateur lui-même ne bat pas la moyenne des sondages sur 14 jours (gain à 180 jours porté
par 2017, légères pertes à 90 et 7 jours ; couverture de 66 % à 180 jours pour un intervalle
à 90 %) : voir [docs/10](docs/10-consolidation.md).

Corrections de structure évaluées au même protocole : substitution intra-bloc et biais RN
par horizon (publiées sans admission stricte, choix documenté), couplage des erreurs premier
tour et duel, participation du second tour tirée à part, biais d'extrême droite depuis 2017
(rejetés). En cours : microdonnées Cevipof (demandes d'accès déposées).

## Données

Sondages Wikipédia 2002-2027 et scrutins de liste, notices de la Commission des sondages
(2017-2027, résultats bruts quand publiés), MieuxVoter, résultats officiels du ministère de
l'Intérieur jusqu'à la commune, Conseil constitutionnel, Jennings–Wlezien (45 pays), INSEE,
DARES, Ifop (popularité), Media Cloud, GDELT, INA, Wikimedia, Google Trends, Cevipof
(Dataverse de Sciences Po). Détail et schémas dans `docs/03-donnees.md`.

## Organisation

- `docs/` : spécifications, journal de backtest, notes de veille (lire `docs/README.md`).
- `back/` : Python 3.12, `uv` ; paquet `vvs` (ingestion, curation, modèle, évaluation,
  export), études dans `back/studies/`, tests `uv run pytest`.
- `front/` : SvelteKit statique, données JSON exportées par le back, déployé par GitHub
  Pages depuis `front/static/data`.
- `scripts/run_live.sh` : run quotidien (ingestion, curation, simulation, export, build,
  commit et push), voir `scripts/README.md`.
- `scripts/run_consolidation.sh` : gel, douze backtests chronologiques, témoin moyen sur
  14 jours et 47 variantes de poids des scénarios, avec reprise vérifiée sur CPU.
- `scripts/run_historical_scenarios.sh` : comparaison isolée des reconstructions
  historiques, avec les fonctions statistiques et les ajustements du gel existant.
- `data/registry/` : candidats, partis, blocs, élections, instituts, scénarios, reports de
  voix ; `data/curated/` : tables parquet.

## Démarrer

```bash
cd back && uv sync --all-groups && uv run pytest -q
cd ../front && npm ci && npm run dev
```

La prévision complète : `scripts/run_live.sh` (environ quatre minutes avec réajustement).
Inférence locale (classification de titres) : llama.cpp avec Qwen3.8-27B, voir
`docs/06-presse.md`.

## Licence et citation

Code sous licence MIT. Les données tierces restent sous leurs licences respectives (voir les
registres). Les reports de voix 2017 du Cevipof proviennent de la French Electoral Study
2017 (doi:10.21410/7E4/2XE05I).
