# 08 — Front

## Choix

SvelteKit avec `adapter-static`, déployé sur GitHub Pages (ou tout hébergeur statique).
Aucun serveur : le site lit des JSON produits par le back et publiés avec lui. Objectif :
un site partageable, rapide, lisible sur mobile, qui montre l'incertitude plutôt que de la
cacher.

Librairies pressenties : LayerChart ou d3 pour les graphiques, Tailwind pour la mise en
page. À confirmer au démarrage de la phase 5.

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
- `history/<date>.json` : résumé de chaque run, pour l'évolution des probabilités.

Les schémas sont décrits en JSON Schema dans `back/src/vvs/export/schemas/` et validés à
l'export ; le front les type via génération TypeScript.

## Règles d'affichage

- Toujours un intervalle avec une médiane, jamais un point seul.
- Les probabilités de qualification et de victoire s'affichent avec le mélange de
  scénarios en premier, le détail conditionnel en second.
- Chaque page qui montre une source non sondagière dit en une phrase si elle entre dans la
  prévision.
- La date du dernier sondage intégré est visible sur chaque page.
