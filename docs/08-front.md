# 08 — Front

## Choix

SvelteKit avec `adapter-static`, déployé sur GitHub Pages (ou tout hébergeur statique).
Aucun serveur : le site lit des JSON produits par le back et publiés avec lui. Objectif :
un site partageable, rapide, lisible sur mobile, qui montre l'incertitude plutôt que de la
cacher.

Choix faits le 2026-10-20 : SvelteKit 3 (Svelte 5, Vite 8, adapter-static 4, TypeScript 6),
Tailwind 4, graphiques en d3 direct (`d3-scale`, `d3-shape`, `d3-array`) plutôt que
LayerChart, dont les dépendances `@layerstack/*` sont encore en préversion et dont le
rythme de versions est trop rapide pour nos quatre graphiques sur mesure. Chemin de base
par la variable `BASE_PATH`, tout prérendu, `404.html` de repli. Données factices
déterministes produites par `front/scripts/mock-data.js` et signalées par un bandeau tant
que `manifest.mock` vaut vrai.

## Pages

| Route | Contenu |
|---|---|
| `/` | les probabilités du jour (qualification, victoire), la date du dernier sondage intégré, l'évolution des probabilités, un résumé des scénarios |
| `/premier-tour` | courbes de soutien latent par candidat avec intervalles, sondages en points, filtre par configuration |
| `/second-tour` | matrice des duels : probabilité de chaque paire, score attendu, intervalle ; reports de voix estimés |
| `/scenarios` | configurations de candidats, leurs probabilités, l'historique des révisions, la prévision conditionnelle à chacune |
| `/sondages` | table de tous les sondages intégrés, effets maison estimés par institut |
| `/economie` | séries économiques et popularité, corrélations, ce qui entre ou non dans la prévision |
| `/presse` | part de voix, ton, agenda de la semaine, même mention |
| `/methode` | méthodologie en langage clair, résultats du backtest, limites |

## Contrat JSON (`front/static/data/`)

Produit par `vvs export`, versionné avec le site. Un fichier par page plus un manifeste :

- `manifest.json` : `run_date`, `asof`, `model_version`, `last_poll_date`, `n_polls`.
- `summary.json` : par candidat, `p_qualify`, `p_win`, quantiles de la part de premier tour ;
  et par configuration.
- `first_round.json` : trajectoires hebdomadaires (médiane, intervalles 50 et 90 %) par
  candidat et configuration ; sondages avec leur part observée.
- `second_round.json` : par paire, `p_pair`, médiane et intervalles du score, reports.
- `scenarios.json` : contenu de `scenarios_2027.yaml` avec historique.
- `polls.json`, `economy.json`, `press.json`, `backtest.json`.
- `history/<date>.json` : résumé de chaque run, jamais réécrit, et `history/index.json`
  (concaténation triée, parce qu'un site statique ne peut pas lister un dossier).

Les schémas sont décrits en JSON Schema dans `back/src/vvs/export/schemas/` et validés à
l'export ; le front les type via génération TypeScript (`npm run types`) et les valide avec
ajv plus des invariants hors schéma (`npm run validate-data`) : quantiles croissants, somme
des `p_win` égale à 1 et des `p_qualify` à 2, `p_win ≤ p_qualify ≤ p_present`, références
entre fichiers. Conventions : parts en points d'exprimés, probabilités entre 0 et 1, tout
intervalle vaut `{q05, q25, q50, q75, q95}` ; une **condition** (`mix`, `<dimension>.<état>`,
ou `cfg.<états>`) dit sur quoi porte une prévision.

Ce que `vvs export` devra fournir en plus de ce que le modèle produit :
- des libellés courts par dimension et par état (« RN : Bardella »), à ajouter dans
  `scenarios_2027.yaml` (champ `label`) ;
- le bloc des candidats 2027 via `bloc_of` ;
- la compatibilité d'un sondage avec une condition : l'hypothèse contient les candidats de
  l'état et aucun candidat d'un autre état de la même dimension.

Poids des pages : les JSON sont inlinés au prérendu (environ 600 Ko pour le premier tour
avec les données factices). Si l'export réel grossit, découper `summary.json` ou charger
côté client.

## Règles d'affichage

- Toujours un intervalle avec une médiane, jamais un point seul.
- Les probabilités de qualification et de victoire s'affichent avec le mélange de
  scénarios en premier, le détail conditionnel en second.
- Chaque page qui montre une source non sondagière dit en une phrase si elle entre dans la
  prévision.
- La date du dernier sondage intégré est visible sur chaque page.
