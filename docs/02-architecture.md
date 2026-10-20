# 02 — Architecture

## Vue d'ensemble

```
sources externes ──► ingestion ──► data/raw (immuable)
                                      │
                                      ▼
                               curation ──► data/curated (parquet, schémas de 03-donnees.md)
                                      │
                   ┌──────────────────┼───────────────────┐
                   ▼                  ▼                   ▼
             études (studies/)   modèle (model/)     backtest (eval/)
                   │                  │                   │
                   └──────────► runs/<date>/ ◄────────────┘
                                      │
                                      ▼
                           export JSON ──► front/static/data/ ──► site statique
```

Le back est un pipeline batch. Il n'y a pas de serveur : le front lit des JSON générés par
le dernier run et publiés avec le site.

## Arborescence

```
vingtvingtsept/
  AGENTS.md, README.md, .gitignore
  docs/                      specs (ce dossier)
  data/
    registry/                YAML versionnés : candidats, partis, blocs, scénarios, instituts
    raw/                     téléchargements bruts, immuables (gros fichiers ignorés par git)
    curated/                 parquet produits par la curation, versionnés s'ils sont petits
  back/
    pyproject.toml           projet uv, Python 3.12
    src/vvs/
      ingest/                un module par source (wikipedia, commission, nsppolls,
                             interieur, insee, dares, mediacloud, gdelt)
      curate/                normalisation vers les schémas, règles de correction explicites
      registry/              chargement et validation des YAML
      model/                 modèle d'agrégation, erreur, scénarios, second tour, simulation
      eval/                  backtest et métriques
      export/                production des JSON pour le front
      cli.py                 point d'entrée `uv run vvs …`
    studies/                 scripts d'étude (économie, presse), sorties dans runs/
    tests/                   tests rapides ; tests lents marqués `slow`
  front/                     SvelteKit, adapter-static
  runs/                      sorties datées, ignorées par git
```

## Commandes cibles

```
uv run vvs ingest polls --election 2027     # télécharge et stocke en raw
uv run vvs curate                           # raw → curated
uv run vvs fit --election 2027 --asof 2026-10-20
uv run vvs backtest --elections 2012,2017,2022 --horizons 180,90,30,7
uv run vvs export --run runs/2026-10-20
```

Chaque run écrit un `manifest.json` (date, commit, versions des registres, hash des données
curated, paramètres) pour être rejouable.

## Choix techniques

- **Données** : parquet via polars ; duckdb pour les requêtes ad hoc. Pas de base serveur.
- **Modèle** : numpyro (JAX) pour l'inférence NUTS, CPU suffisant pour les volumes en jeu.
  Le code du modèle est isolé dans `vvs/model/` pour pouvoir changer de moteur.
- **Validation de schémas** : pydantic pour les registres YAML, pandera ou contrôles polars
  pour les parquet.
- **Automatisation live** : une tâche planifiée (cron local ou GitHub Actions) relance
  ingestion, fit, export et commit du site. À définir en phase 8.
- **Front** : SvelteKit + adapter-static, déployé sur GitHub Pages. Détails dans
  `08-front.md`.

## Flux d'un sondage, de la publication au site

1. Notice publiée sur commission-des-sondages.fr, tableau Wikipédia mis à jour.
2. `ingest` récupère les deux, stocke les pages brutes avec horodatage.
3. `curate` parse, rattache chaque ligne à un candidat du registre, résout les doublons entre
   sources (clé : institut, dates de terrain, commanditaire), signale les écarts.
4. `fit` recalcule les postérieurs et les simulations.
5. `export` écrit les JSON, le front est reconstruit et publié.
