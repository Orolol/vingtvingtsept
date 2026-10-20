# 09 — Feuille de route

| Phase | Contenu | Livrable | État |
|---|---|---|---|
| 0 | Dossier, git, specs | `docs/` | fait le 2026-10-20 |
| 1 | Données : scraper Wikipédia 2002 à 2027, scraper des notices de la Commission, registre candidats/blocs, résultats du ministère de l'Intérieur, curation et tests | `data/curated/*.parquet`, tests verts | à faire |
| 2 | Agrégateur premier tour sur 2022 (sans scénarios), effets maison, figures | courbes 2022 cohérentes avec le résultat | à faire |
| 3 | Modèle d'erreur et backtest 2012, 2017, 2022 aux quatre horizons ; baselines | `docs/backtest-log.md` | à faire |
| 4 | Scénarios 2027, substitution intra-bloc, second tour par reports, simulation ; première projection 2027 | `runs/<date>/summary.json` | à faire |
| 5 | Front SvelteKit, export JSON, publication | site en ligne | à faire |
| 6 | Étude économie, admission ou non, section du site | `runs/<date>/economy/`, mise à jour de `05` | à faire |
| 7 | Ingestion presse (Media Cloud, GDELT), classification, étude, admission ou non, section du site | `runs/<date>/press/`, mise à jour de `06` | à faire |
| 8 | Live : automatisation des runs, mise à jour à chaque sondage, historique des probabilités | tâche planifiée | à faire |
| 9 | Législatives (seconde étape, hors périmètre actuel) | nouvelle spec | plus tard |

Les phases 6 et 7 sont indépendantes l'une de l'autre et de la 5 ; elles peuvent démarrer
dès que la 3 fournit un backtest de référence.

## Questions ouvertes

- Frontières des blocs (écologistes, Mélenchon avant 2012) : à trancher en phase 1.
- Identifiant de la collection « France national » de Media Cloud : à retrouver en phase 7.
- Version 2027 de NSPPolls : vérifier si une communauté la maintient ; sinon, la Commission
  suffit.
- État actuel de la course 2027 (candidatures, décision en appel concernant Marine Le Pen) :
  à établir avant d'écrire `scenarios_2027.yaml`.
- Dates exactes du scrutin 2027 : à confirmer dès publication du décret de convocation.
