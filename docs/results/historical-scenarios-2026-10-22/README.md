# Reconstruction des scénarios historiques — 22 octobre 2026

Protocole et interprétation : [11-scenarios-historiques.md](../../11-scenarios-historiques.md).

Comparaison exploratoire, référence statistique `a89d304` intacte. Variante principale : `memory_events`.

- `protocol.yaml` : paramètres déclarés avant calcul.
- `metrics.csv` : 72 scores (12 cas × 2 agrégateurs × 3 versions de scénarios).
- `comparisons.csv` : écarts à la référence, par cas ; `means.csv` et `delta_means.csv` : moyennes à poids égal par élection.
- `outcomes.csv` : probabilités de présence, qualification et victoire ; zéro de présence pour les candidats hors support.
- `diagnostics.csv` : configurations sans tirage et qualité du second tour.
- `audit.json` : empreintes, scénarios, événements appliqués, masse exclue, contrôles des rejeux.
- `comparison.png` : log-score de paire et CRPS, par élection et horizon long.

Les écarts sont variante moins référence : positif préférable pour les log-scores, négatif pour le CRPS et les Brier. Les quatre dates d'une même élection ne sont pas indépendantes. Les tirages complets restent dans `runs/2026-10-22-historical-scenarios/`.

Reproduction depuis la racine : `scripts/run_historical_scenarios.sh`, puis depuis `back/` :

```sh
uv run python studies/historical_scenarios_report.py --run ../runs/2026-10-22-historical-scenarios --reference ../runs/2026-10-22-consolidation
```
