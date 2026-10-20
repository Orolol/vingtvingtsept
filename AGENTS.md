# AGENTS.md — vingtvingtsept

Modèle de prévision de l'élection présidentielle française de 2027, inspiré de l'approche
de Nate Silver : agrégation bayésienne des sondages, calibration historique de l'erreur,
simulation Monte-Carlo, plus des sources non sondagières (économie, presse) admises
seulement si elles améliorent le backtest.

## Langue et conventions

- Documentation, commits, issues : **français**. Identifiants de code, noms de colonnes et
  de fichiers de données : **anglais** (`share`, `fieldwork_end`, `pollster`).
- Back : Python 3.12, `uv` (`uv sync`, `uv run …`), paquet `vvs` dans `back/`.
- Front : SvelteKit statique dans `front/`, Node 22 via `mise`.
- Pas de notebooks versionnés : les études exploratoires sont des scripts dans
  `back/studies/` qui écrivent leurs figures dans `runs/`.

## Index documentaire

Lis `docs/README.md` avant de modifier quoi que ce soit : il pointe la vision, l'architecture,
les schémas de données, le modèle, les sources économiques et presse, le protocole
d'évaluation, le front et la feuille de route.

## Règles du projet

- **Aucune source n'entre dans la prévision sans gain au backtest.** Une variable
  (économique, presse, autre) vit d'abord dans une étude et dans le tableau de bord explicatif.
  Elle n'entre dans le modèle de prévision que si elle améliore le score de log-vraisemblance
  à horizon fixe en validation croisée par élection (voir `docs/07-evaluation.md`).
- **Jamais de calibration sur l'élection évaluée.** Le modèle d'erreur est estimé en
  laissant de côté l'élection backtestée.
- **Résultat négatif = résultat.** On ne retouche ni métrique ni protocole pour faire passer
  une variable. On documente l'échec dans `docs/` et on continue.
- **Données brutes immuables.** `data/raw/` ne se modifie qu'en ajoutant ; les corrections
  passent par des règles explicites dans le code de curation, jamais à la main.
- **Scénarios de candidats versionnés.** Les probabilités de configuration de candidats sont
  des hypothèses explicites dans `data/registry/scenarios_2027.yaml`, datées et justifiées,
  jamais enfouies dans le code.
- Commits aux étapes naturelles, sans push sans demande. Les règles globales de
  `~/AGENTS.md` s'appliquent (GPU partagé, tests ciblés, pas de `git add -A`).
