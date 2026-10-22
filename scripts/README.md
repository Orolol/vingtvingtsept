# scripts/

## `run_historical_scenarios.sh` : reconstruction historique expérimentale

```sh
scripts/run_historical_scenarios.sh
```

Requiert le gel complet `runs/2026-10-22-consolidation/`. Compare une mémoire des listes
sondées sur un an et sa variante filtrée par événements datés, avec le code statistique,
les ajustements et les métriques du gel. Protocole et résultat négatif dans
[`docs/11-scenarios-historiques.md`](../docs/11-scenarios-historiques.md).

Écrit seulement dans `runs/2026-10-22-historical-scenarios/` (premier argument pour
changer ce dossier). Les 409 fichiers de référence, les dépendances et les sources
de l'étude sont contrôlés par empreintes. Les douze cas terminés sont repris après
vérification ; les cas incomplets sont rejoués. Un verrou refuse les lancements
concurrents sur le même dossier, avec le code de sortie 75. Calcul CPU uniquement,
sans réajustement NUTS, export live ni publication.

Depuis `back/`, le bilan vérifié et ses figures s'exportent dans `review/` :

```sh
uv run python studies/historical_scenarios_report.py --run ../runs/2026-10-22-historical-scenarios --reference ../runs/2026-10-22-consolidation
```

## `run_consolidation.sh` : référence et évaluation chronologique

Depuis la racine du dépôt, avec l'environnement Python déjà installé
(`cd back && uv sync --all-groups`) :

```sh
scripts/run_consolidation.sh
```

Le script fige la référence définie dans `data/registry/reference_2026-10-22.yaml`, puis
enchaîne les douze cas (2012, 2017, 2022 × 180, 90, 30, 7 jours), les scores de victoire
et de qualification, et les 47 variantes de scénarios 2027. Il utilise exclusivement le
CPU, avec deux calculs simultanés de quatre cœurs par défaut. `VVS_CONSOLIDATION_JOBS=1`
permet de limiter la concurrence ; un premier argument change le dossier de sortie.

La même commande reprend après interruption : elle vérifie les empreintes du gel et des
étapes terminées, conserve les ajustements complets, et ne relance que les cas incomplets.
Un fichier altéré provoque un échec explicite. Les diagnostics insuffisants restent dans
les résultats et sont signalés ; ils ne provoquent pas un changement automatique du
protocole ou des seuils. Aucun export live, commit ou push n'est exécuté.
Un verrou placé à côté du dossier de sortie refuse deux exécutions simultanées sur le
même gel (code de sortie 75).

Sorties dans `runs/2026-10-22-consolidation/` :

- `reference.json`, `source/`, `published/` : commit, paramètres, versions, empreintes,
  copie des sources et données, tirages de la projection publiée ;
- `cases/<année>-<horizon>/run.log` : journal de chaque cas ; `fit/quality.json` :
  diagnostics ; `calibration.json` et `case.json` : traçabilité des entrées ;
- `metrics.parquet`, `means.parquet`, `report.md` : résultats complets et synthèse ;
- `sensitivity/probabilities.csv`, `variants.json`, `report.md` : sensibilités, poids,
  tailles effectives et bornes dues aux configurations non représentées.

Le dossier `runs/` est ignoré par Git : conserver le gel et les tirages avec les résultats.
Le fichier `data/registry/reference_2026-10-22.lock.json` épingle le commit, les versions et
les empreintes des entrées. Recréer ce gel dans un autre dossier reprend donc `a89d304`,
même si le dépôt a évolué ; des entrées différentes font échouer le gel. Une référence
sans verrou archive initialement le HEAD courant. Pour comparer une nouvelle version,
utiliser un nouvel identifiant et un nouveau protocole, sans écraser cette référence.
Méthode et limites : [10-consolidation.md](../docs/10-consolidation.md).

## `run_live.sh` : un run live de la prévision

Enchaîne, depuis la racine du dépôt :

| Étape | Dossier | Commande |
|---|---|---|
| ingestion | `back/` | `uv run vvs ingest commission --since <1er janvier de l'année précédente>` |
| | | `uv run vvs ingest wikipedia --election 2027` |
| | | `uv run vvs ingest mieuxvoter` |
| extraction | | `uv run vvs parse commission`, `parse wikipedia --election 2027`, `parse mieuxvoter` |
| curation | | `uv run vvs curate` |
| simulation | | `uv run vvs simulate --asof <aujourd'hui> --substitution --out runs/<aujourd'hui>-simulate-subst` |
| export | | `uv run vvs export --run runs/<aujourd'hui>-simulate-subst` (après copie de l'export précédent) |
| site | `front/` | `npm run validate-data`, `npm run build` |
| résumé | `back/` | `uv run vvs live-summary --previous <copie> --current front/static/data` |

```sh
scripts/run_live.sh            # run complet
scripts/run_live.sh --dry-run  # affiche les commandes sans rien exécuter ni écrire
```

Le script est en bash (pas fish) ; depuis fish, on le lance tel quel.

- **Arrêt au premier échec**, avec le code de la commande fautive (par exemple 1 si
  `vvs parse commission` rencontre un libellé inconnu, ou si `vvs curate` ne retrouve pas une
  divergence connue). Un run déjà en cours bloque le suivant (verrou `runs/logs/live.lock`,
  code 75).
- **Journal horodaté** : `runs/logs/live-<AAAA-MM-JJTHHMMSS>.log` (sortie complète de chaque
  étape, durée de chaque étape et du run).
- **Résumé final** : sondages nouvellement intégrés depuis l'export précédent, probabilités
  de victoire et de qualification du mélange de scénarios, écart en points avec l'export
  précédent, empreinte des sondages.
- **Idempotent** : les ingestions n'ajoutent que ce qui manque (`data/raw/` en ajout seul ;
  Wikipédia garde un instantané horodaté par run) ; `vvs simulate` ne refait l'ajustement
  du premier tour (environ trois minutes) que si l'empreinte des sondages 2027 a changé ou si
  la date de prévision change de semaine de grille, sinon il réutilise le dernier ajustement
  compatible (`runs/<date>-fit-2027-subst`) ; l'export n'écrase jamais
  `front/static/data/history/<date>.json` : un second run le même jour met à jour les pages
  mais garde la première entrée d'historique du jour. Une modification du code du modèle ou
  des registres n'est pas détectée par l'empreinte : relancer alors `vvs simulate … --refit`.

Variables d'environnement :

| Variable | Défaut | Rôle |
|---|---|---|
| `VVS_SUBSTITUTION` | `1` | `1` : substitution intra-bloc (projection publiée actuelle) ; `0` : modèle IIA, run `runs/<date>-simulate` |
| `VVS_ASOF` | aujourd'hui | date de prévision `AAAA-MM-JJ` |
| `VVS_DRAWS` | celui de `vvs simulate` (40 000) | tirages Monte-Carlo |
| `VVS_ENV_FILE` | `~/.config/vingtvingtsept/env` | fichier d'environnement lu s'il existe |

## Planification (à installer par Gaétan)

Rien n'est installé par le dépôt. Deux options, un run par jour à 7 h 30 (les notices de la
Commission et Wikipédia sont mises à jour en journée ; un second run le soir est possible).

**cron utilisateur** (`crontab -e`) :

```cron
30 7 * * * /home/orosius/workspace/vingtvingtsept/scripts/run_live.sh >/dev/null 2>&1
```

La sortie va déjà dans `runs/logs/` ; le script complète lui-même le `PATH` (uv dans
`~/.local/bin`, Node via `mise where node`).

**systemd utilisateur** : `~/.config/systemd/user/vvs-live.service`

```ini
[Unit]
Description=vingtvingtsept : run live de la prévision

[Service]
Type=oneshot
ExecStart=/home/orosius/workspace/vingtvingtsept/scripts/run_live.sh
EnvironmentFile=-%h/.config/vingtvingtsept/env
Nice=10
```

et `~/.config/systemd/user/vvs-live.timer`

```ini
[Unit]
Description=vingtvingtsept : run live quotidien

[Timer]
OnCalendar=*-*-* 07:30
Persistent=true

[Install]
WantedBy=timers.target
```

puis `systemctl --user daemon-reload` et `systemctl --user enable --now vvs-live.timer`
(`loginctl enable-linger orosius` pour qu'il tourne sans session ouverte). `Persistent=true`
rattrape un run manqué si la machine était éteinte.

## Secrets : `MEDIACLOUD_API_KEY`

La clé Media Cloud est lue dans l'environnement par `vvs ingest press --source mediacloud`
(`back/src/vvs/ingest/press/mediacloud.py`) et n'est jamais écrite : ni dans `data/raw/`, ni
dans les journaux, ni dans les manifests. Le run live par défaut n'interroge pas Media Cloud ;
la clé ne sert qu'aux ingestions presse lancées à la main.

Pour la fournir à cron ou systemd sans la mettre dans le dépôt, la placer dans
`~/.config/vingtvingtsept/env` (hors dépôt, `chmod 600`) :

```sh
MEDIACLOUD_API_KEY=…
```

`run_live.sh` lit ce fichier s'il existe (sans l'afficher ni le journaliser, et pas en
`--dry-run`) ; le service systemd ci-dessus le charge aussi par `EnvironmentFile`.

## Publication

Le back ne tourne pas sur GitHub : le workflow `.github/workflows/pages.yml` construit le site
à partir des données **commitées** dans `front/static/data/` (il refuse le jeu factice, valide
contre le contrat JSON, vérifie les types, construit avec le chemin de base de
`actions/configure-pages`, publie). Le site en ligne est donc celui du **dernier export
commité et poussé**. Après un run live :

```sh
git add front/static/data data/curated
git commit -m "Run live du <date>"
git push
```

Le dépôt distant et le site Pages sont en place (URL dans le README).

## Publication automatique

À la fin d'un run, le script commite `front/static/data` et `data/curated` (seulement si quelque
chose a changé), puis pousse sur `origin/main`, ce qui déclenche le workflow GitHub Pages. Il ne
publie que depuis la branche `main` et seulement si un remote `origin` existe. `VVS_PUBLISH=0`
désactive cette étape (le site construit reste dans `front/build`). Le push passe par la clé SSH
de la machine : pour cron, elle doit être utilisable sans agent interactif.
