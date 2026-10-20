# 09 — Feuille de route

| Phase | Contenu | Livrable | État |
|---|---|---|---|
| 0 | Dossier, git, specs | `docs/` | fait le 2026-10-20 |
| 1 | Données : Wikipédia 2002 à 2027 (978 sondages), notices de la Commission (2027), MieuxVoter, registre candidats/blocs/élections/instituts, résultats du ministère de l'Intérieur (présidentielles, européennes, législatives), séries macro et popularité, base Jennings–Wlezien, scénarios 2027, reports de voix 2002-2022 | `data/curated/*.parquet`, `data/registry/*.yaml`, 283 tests | ingestions faites le 2026-10-20 ; curation en cours |
| 2 | Agrégateur premier tour sur 2022 (sans scénarios), effets maison, figures | courbes 2022 cohérentes avec le résultat | fait le 2026-10-20 |
| 3 | Modèle d'erreur et backtest 2002-2022 aux quatre horizons ; baselines ; second tour par reports évalué | `docs/backtest-log.md` | fait le 2026-10-20 |
| 4 | Scénarios 2027, substitution intra-bloc, second tour par reports, simulation ; première projection 2027 | `runs/<date>/summary.json` | à faire |
| 5 | Front SvelteKit, export JSON, publication | site en ligne | squelette et contrat JSON faits ; export et publication à faire |
| 6 | Étude économie, admission ou non, section du site | `runs/<date>/economy/`, mise à jour de `05` | étude faite le 2026-10-20, résultat négatif ; section du site à faire |
| 7 | Ingestion presse (Media Cloud, GDELT), classification, étude, admission ou non, section du site | `runs/<date>/press/`, mise à jour de `06` | à faire |
| 8 | Live : automatisation des runs, mise à jour à chaque sondage, historique des probabilités | tâche planifiée | à faire |
| 9 | Législatives (seconde étape, hors périmètre actuel) | nouvelle spec | plus tard |

Les phases 6 et 7 sont indépendantes l'une de l'autre et de la 5 ; elles peuvent démarrer
dès que la 3 fournit un backtest de référence.

## Questions ouvertes

- Page Wikipédia 2012 : le tableau « second trimestre 2011 » est corrompu (colonne
  Chevènement contenant les scores de Hulot, sondages perdus). Écarté pour l'instant ; la
  révision 79265234 de mai 2012 est saine et pourrait être parsée à la place pour ce tableau.
- Villepin 2027 : rangé en `other` à titre provisoire, à trancher avant la phase 4.
- Jennings–Wlezien s'arrête en 2017 : 2022 doit venir de nos propres tables d'erreurs.

- Frontières des blocs (écologistes, Mélenchon avant 2012) : à trancher en phase 1.
- Dates du scrutin 2027 (18 avril et 2 mai) : à confirmer à la publication du décret.
- Successeur de NSPPolls : `MieuxVoter/presidentielle2027` (même format, MIT) ; à ingérer en
  contrôle croisé de la Commission et de Wikipédia.
- Base Jennings et Wlezien (erreurs de sondages, France depuis 1965) : à ingérer pour la
  calibration du modèle d'erreur.
- Comparateurs : modèle France 2022 de *The Economist*, modèle bayésien de Martin Devaux.
- « Surestimation du RN en 2024 » dans `04-modele.md` : affirmation à vérifier sur les
  données, aucune étude primaire trouvée.

Voir `notes/veille-2026-10-20.md` pour l'état de la course, Timestral, les archives de presse,
les reports de voix historiques et la littérature.
