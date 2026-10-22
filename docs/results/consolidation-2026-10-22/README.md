# Bilan de la référence du 22 octobre 2026

Protocole, lecture des scores et limites : [10-consolidation.md](../../10-consolidation.md).
Ces fichiers conservent les résultats compacts du gel statistique `a89d304`.

| Fichier | Contenu |
|---|---|
| `audit.json` | Manifest du gel et ses 165 empreintes ; scénarios, disponibilité des sondages, calibrations et diagnostics de chaque cas ; empreintes des sorties et du code de posttraitement |
| `metrics.csv` | 24 lignes : trois élections × quatre horizons × référence ou moyenne sur 14 jours |
| `outcomes.csv` | Probabilités de victoire et de qualification par candidat, support des scénarios et résultats observés |
| `sensitivity.csv` | Probabilités, bornes de masse manquante, écarts à la référence et tailles effectives pour 47 variantes × 33 candidats |
| `variants.json` | Libellés et poids explicites de chaque variante, masse exclue par les contraintes et masse non représentée |
| `backtest.png` | Log-score du vainqueur par élection et horizon |
| `sensitivity.png` | Étendues de probabilité de victoire sous les perturbations locales des poids |

Les CSV arrondissent à huit décimales ; les parquets et tirages complets restent dans
`runs/2026-10-22-consolidation/`. Les chemins de l'audit sont relatifs à ce dossier ou au
sous-dossier du cas concerné. Les métriques plus hautes sont meilleures pour les log-scores ;
les métriques plus basses sont meilleures pour le CRPS et les Brier. Les probabilités sont
brutes, les log-scores utilisent le lissage déclaré dans le protocole.

Régénérer depuis `back/`, une fois les douze cas terminés :

```sh
uv run python studies/consolidation_report.py --run ../runs/2026-10-22-consolidation
```

Le script vérifie l'intégrité des cas et la cohérence des tirages, puis écrit `review/`
dans le dossier du run. Le contenu de `review/` est copié ici pour le versionner.
Les données brutes, le gel de calcul et les tirages ne sont pas dupliqués dans cette synthèse.
