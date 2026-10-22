# scripts/

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

Le dépôt distant et l'activation de Pages (Settings → Pages → source « GitHub Actions ») sont
à créer.

## Publication automatique

À la fin d'un run, le script commite `front/static/data` et `data/curated` (seulement si quelque
chose a changé), puis pousse sur `origin/main`, ce qui déclenche le workflow GitHub Pages. Il ne
publie que depuis la branche `main` et seulement si un remote `origin` existe. `VVS_PUBLISH=0`
désactive cette étape (le site construit reste dans `front/build`). Le push passe par la clé SSH
de la machine : pour cron, elle doit être utilisable sans agent interactif.
