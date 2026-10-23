# 13 — Validation prospective 2027, contrôle des PR et archive durable

Statut : protocole fixé le 23 octobre 2026, avant toute issue de 2027. Il met en œuvre le
lot 4 de [12-prochaines-etapes.md](12-prochaines-etapes.md) : archiver les probabilités
publiées à des dates fixes, contrôler automatiquement les PR et conserver hors de Git les
tirages et le gel nécessaires à une reprise. Les règles de score ci-dessous sont déclarées
maintenant ; elles ne seront ni choisies ni retouchées après lecture des résultats.

## 1. Pourquoi un protocole prospectif

Les élections 2012, 2017 et 2022 ont servi à concevoir, comparer et rejeter de nombreuses
variantes ([10](10-consolidation.md), [11](11-scenarios-historiques.md)). Elles ne forment
plus un test indépendant : réutiliser un jeu de test pour choisir des hypothèses conduit à
s'y surajuster ([Dwork et al., 2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/bad5f33780c42f2588878a9d07405083-Paper.pdf)).
2027 est la seule élection que la conception actuelle n'a pas vue. Pour qu'elle serve de
test, il faut prouver ce qui était publié, avec quelles données, quel code et quelles
hypothèses, **avant** chaque événement.

### Une seule issue par étape

2027 fournit une seule réalisation de chaque étape de la chaîne : une liste officielle (la
présence), une paire de qualifiés, un vainqueur, un vecteur de parts du premier tour et un
score du second tour. Les dix-neuf dates d'archivage sont dix-neuf prévisions **corrélées**
de ces mêmes issues, pas dix-neuf tests. Elles montrent comment la prévision converge,
pas si ses probabilités sont calibrées : une probabilité de victoire de 70 % ne se valide
ni ne se réfute sur un seul scrutin.

Les candidatures, primaires et retraits de 2026-2027 donnent des diagnostics
intermédiaires (la probabilité de présence d'un candidat avant sa déclaration ou son
retrait). Ils restent liés à la même campagne et souvent entre eux : une primaire décide
de plusieurs présences à la fois. Ils ne transforment pas une présidentielle en plusieurs
élections indépendantes. Après le vote, 2027 s'ajoute comme quatrième élection au tableau
chronologique de [10](10-consolidation.md), à poids égal avec les trois autres, et chaque
étape reçoit une observation de plus. Aucune conclusion de calibration fine n'en sera tirée.

## 2. Calendrier d'archivage

Dates fixes, calculées par `vvs.export.prospective.schedule` à partir des dates du scrutin
du manifest publié (registre `elections.yaml` : 18 avril et 2 mai 2027, décret encore à
confirmer). Les motifs d'une même date sont fusionnés.

| Date | Jour | Motif | Délai de rattrapage |
|---|---|---|---|
| 2026-10-22 | jeudi | projection gelée (`reference-2026-10-22`), J-178 | 2 j pour la créer, export du 22 octobre seulement |
| 2026-11-01 | dimanche | premier du mois | 2 j |
| 2026-12-01 | mardi | premier du mois | 2 j |
| 2027-01-01 | vendredi | premier du mois | 2 j |
| 2027-01-18 | lundi | J-90, horizon du backtest | 2 j |
| 2027-02-01 | lundi | premier du mois | 2 j |
| 2027-02-07 à 2027-04-04 | dimanches | hebdomadaire (J-70 à J-14), neuf dates | 2 j |
| 2027-03-19 | vendredi | J-30, horizon du backtest | 1 j |
| 2027-04-11 | dimanche | J-7, horizon du backtest et hebdomadaire | 2 j |
| 2027-04-17 | samedi | veille du premier tour | 0 j |
| 2027-05-01 | samedi | veille du second tour | 0 j |

Soit dix-neuf dates : le gel initial, le premier de chaque mois jusqu'en février 2027, puis
chaque semaine le même jour que le premier tour, les horizons J-90, J-30 et J-7 du
backtest chronologique (J-180, le 20 octobre 2026, précède le gel) et la veille de chaque
tour. Les horizons communs rendent ces archives directement comparables aux lignes de
[10](10-consolidation.md).

Règles :

- l'archive d'une date reçoit le premier export dont l'`asof` tombe entre cette date et la
  fin de son délai de rattrapage (run manqué, machine éteinte). Le délai ne chevauche
  jamais la date suivante ni le jour du vote ; il est nul la veille d'un tour. Le gel
  initial n'accepte que l'export du 22 octobre, archivable jusqu'au 24 octobre 2026 ;
- elle doit être créée au plus tard à la fin de ce délai et strictement avant le tour visé ;
  la date de création est enregistrée et contrôlée par `archive-verify` ;
- si les dates officielles changent (décret), les dates hebdomadaires suivent le nouveau
  premier tour ; les archives déjà créées gardent leur date et leurs motifs enregistrés ;
- une date sans export dans son délai reste vide. Ce trou est publié, jamais comblé après
  coup.

## 3. Contenu d'une archive

`data/prospective/<date>/<nom>/`, versionné (environ 245 Ko par archive, 5 Mo pour la
campagne) :

| Fichier | Contenu |
|---|---|
| `manifest.json`, `summary.json`, `second_round.json`, `scenarios.json` | pages du site publiées, copiées octet pour octet |
| `run_manifest.json` | manifest complet du run de simulation (paramètres, modèle d'erreur, second tour, commits, versions) |
| `scenarios_2027.yaml` | hypothèses de candidature utilisées par le run |
| `archive.json` | métadonnées et empreintes SHA-256, détaillées ci-dessous |

`archive.json` porte : le nom et la piste (`comparison` pour `stable`, `experimental`
sinon) ; la date, ses motifs et son délai ; l'`asof` ; l'heure de création ; le code
(commit de l'export, de la simulation et de l'ajustement, HEAD Git, fichiers non committés
du back et des registres, empreinte globale des sources Python, schémas et verrou uv,
versions des bibliothèques) ; les données (`polls_hash`, nombre de sondages, dernier
sondage, empreintes des huit tables curated lues par l'ajustement et le modèle d'erreur) ;
les hypothèses (empreinte complète de `scenarios_2027.yaml`, qui doit prolonger
l'empreinte courte du run) ; les empreintes de chaque fichier du run (dont
`draws.parquet`) et de l'ajustement (dont `draws.npz`) ; enfin celles des fichiers de
l'archive.

Les tirages ne sont pas versionnés. Le run archivé est copié dans
`runs/prospective/<date>/<nom>/` (6,7 Mo), parce qu'un second run du même jour réécrit
`runs/<date>-simulate-subst`. L'ajustement reste dans son dossier
`runs/<date>-fit-2027-subst`, qui n'est réécrit que par un `--refit` explicite à la même
date. Ces copies entrent dans l'archive durable (§ 8).

L'archivage échoue explicitement si l'export est factice, si le run n'a pas la même date,
le même nombre de tirages ou la même empreinte de sondages que l'export, si le registre de
scénarios a changé depuis le run, ou si la date n'est pas au calendrier.

## 4. Commandes

Depuis `back/` :

```sh
uv run vvs archive-forecast [--asof AAAA-MM-JJ] [--name stable] [--run runs/<date>-simulate-subst]
uv run vvs archive-verify [data/prospective/<date>/<nom>] [--against <réf. Git>]
uv run vvs check-integrity [--base <réf. Git>]
```

- `archive-forecast` archive l'export de `front/static/data/`. `--asof` vérifie seulement
  que l'export publié est bien celui de cette date. Une archive existante n'est **jamais
  réécrite** : même prévision (même `asof`, même empreinte des sondages et des tirages,
  mêmes pages hors `generated_at`) → rien à faire, code 0 ; prévision différente → échec,
  code 1. `--skip-existing` garde l'archive existante avec le code 0, et
  `--scheduled-only` ne fait rien hors calendrier : ce sont les deux options du run
  quotidien. L'écriture passe par un dossier temporaire renommé à la fin : une archive
  est complète ou absente.
- `archive-verify` recalcule les empreintes de chaque archive (fichiers modifiés, ajoutés
  ou retirés, date de création), puis celles du run et de l'ajustement archivés s'ils sont
  encore présents dans `runs/` ; un run absent est signalé sans erreur. `--against`
  refuse en plus toute modification d'une archive qui existe dans la référence Git donnée.
- `check-integrity` regroupe les contrôles de la CI (§ 7).

Dans `scripts/run_live.sh`, l'archivage suit l'export, la validation, le build et le
résumé : `vvs archive-forecast --asof <date> --run <run> --name stable --scheduled-only
--skip-existing`. Avec `VVS_PUBLISH=1`, `data/prospective/` est ajouté au commit des
données ; le message signale l'archive. `VVS_ARCHIVE=0` saute l'étape. Une variante
(`VVS_SUBSTITUTION=0` ou `VVS_RN_BIAS_CURVE=0`) n'archive rien sans `VVS_ARCHIVE_NAME`
explicite : elle ne peut pas prendre le nom `stable` par défaut.

## 5. Version stable et versions expérimentales

Deux archives nommées peuvent coexister à une même date :

- **`stable`**, la version de comparaison : les choix de la référence
  `reference-2026-10-22` (substitution, erreur finale élargie, biais du second tour par
  horizon). Tant que la projection publiée garde ces choix, `run_live.sh` la produit.
  Si une variante est un jour publiée, `stable` doit continuer d'être produite avec le code
  du tag `reference-2026-10-22` sur les données du jour (worktree au tag, même commande
  avec `VVS_ARCHIVE_NAME=stable` et sans publication du site), et la variante publiée
  prend son propre nom ;
- **une archive par variante expérimentale** (`candidacies-v1`, par exemple), nommée avant
  sa première date et jamais renommée. Elle suit le même calendrier, les mêmes règles de
  délai et le même score. Une variante créée après une date n'a pas d'archive pour cette
  date : on ne reconstruit pas une prévision passée.

La comparaison finale oppose les archives de même date. Une variante ne remplace pas
`stable` dans le tableau 2027 parce qu'elle a mieux scoré : elle reste une ligne à part.
Le témoin `mean14d` de [10](10-consolidation.md) peut être archivé de la même façon si son
calcul est branché sur les données 2027 avant la première date visée.

## 6. Ce qui sera scoré après le vote

Pour chaque archive et chaque date, avec les tirages de `runs/prospective/<date>/<nom>/`
contrôlés par leurs empreintes, et les définitions de [10](10-consolidation.md) et de
`vvs.eval.stage_diagnostics` (lot 3), sans changement :

| Étape | Issue | Scores |
|---|---|---|
| Présence | liste officielle du Conseil constitutionnel | Brier de présence (somme et moyenne sur l'univers officiels ∪ support), P(vainqueur présent), P(paire présente) |
| Qualification | les deux qualifiés | log-score de la paire réelle, Brier de qualification ; conditionnels : P(qualifié \| présent), P(paire \| deux présents) |
| Victoire | le vainqueur | log-score du vainqueur, Brier de victoire (somme multicatégorie) ; conditionnel : P(vainqueur \| qualifié) |
| Parts | résultats officiels du premier tour en % des exprimés | CRPS et couverture à 50 et 90 % sur tous les officiels (zéro pour un absent des tirages) ; version conditionnelle à la présence |
| Duel | score du second tour | P(vainqueur \| paire réelle qualifiée), CRPS de la part du second tour sur les tirages du duel réalisé |

Conventions reprises telles quelles : fréquences brutes publiées ; lissage de Jeffreys
séparé pour les log-scores, sur l'univers de score (union des candidats prévus et
officiels) ; moyennes sur un univers fixé avant lecture quand on compare `stable` et une
variante ; aucun report de voix ex post. Les candidats génériques (`placeholder_<bloc>`)
ne sont jamais rattachés après coup au candidat finalement investi : leur masse compte
comme celle d'un candidat absent de la liste officielle. L'archive de la veille du second
tour est scorée sur le second tour seulement. À défaut d'un mode entre les deux tours, elle
contient une projection qui ignore les résultats du premier tour ; seul son diagnostic
conditionnel au duel réalisé est alors lu.

Les diagnostics intermédiaires (déclarations, primaires, retraits, liste officielle)
seront lus avec les mêmes scores de présence, à chaque date archivée, sans changer de
règle après lecture.

### Silence électoral

La loi n° 77-808 du 19 juillet 1977 interdit la publication, la diffusion et le commentaire
de tout sondage la veille et le jour de chaque tour ; depuis 2016, sa définition inclut
les simulations de vote réalisées à partir de sondages. Pousser sur le dépôt public l'archive de la veille,
ou mettre à jour le site ces jours-là, peut tomber sous cette interdiction. Décision à
prendre par Gaétan avant avril 2027 : lancer le run de la veille avec `VVS_PUBLISH=0`
(archive créée localement, puis commitée et poussée après la clôture du scrutin), et
suspendre le run quotidien le samedi et le dimanche des deux tours. Un commit local ne
prouve pas son antériorité à un tiers : pour ces deux dates, publier avant le silence
l'empreinte d'`archive.json` sur un support horodaté indépendant (par exemple
OpenTimestamps) donnerait cette preuve sans publier la prévision.

## 7. Contrôle automatisé des PR

`.github/workflows/ci.yml` tourne sur chaque `pull_request`, sur chaque push sur `main`
et à la demande. Deux tâches parallèles, CPU uniquement :

| Tâche | Étapes |
|---|---|
| `back` | checkout avec l'historique complet ; uv et Python 3.12 (cache sur `back/uv.lock`) ; `uv sync --locked --all-groups` ; `uv run pytest -q -m "not slow" -n auto` avec `JAX_PLATFORMS=cpu` ; `uv run ruff check` ; `uv run vvs check-integrity --base <base>` |
| `front` | Node 22 (cache npm) ; `npm ci` ; `npm run validate-data` ; `npm run check` |

`check-integrity` vérifie :

- le verrou `data/registry/reference_2026-10-22.lock.json` : configuration identique à
  `reference_2026-10-22.yaml`, commit `a89d304` présent, et chacune des 161 empreintes
  `source/…` égale au fichier de ce commit (les 4 fichiers `published/…` sont hors Git) ;
  le gel local `runs/2026-10-22-consolidation/` est contrôlé en plus quand il existe ;
- `front/static/data/manifest.json` : export réel, version de schéma, `polls_hash` bien
  formé, au moins 10 000 tirages, `asof` au plus `run_date`, dates du scrutin identiques au
  registre, entrée `history/<run_date>.json` présente et indexée (le schéma complet est
  vérifié par `npm run validate-data`) ;
- les archives prospectives : empreintes de chacune, et, contre la base (base de la PR,
  ou commit précédent pour un push), aucun fichier ajouté, modifié ou supprimé dans un
  dossier `data/prospective/<date>/<nom>/` qui existait déjà ; une nouvelle archive est
  permise ;
- l'historique du site : aucune entrée `history/<date>.json` de la base modifiée ou
  supprimée (`index.json` est régénéré à chaque export).

Durées mesurées le 23 octobre sur un clone propre (16 cœurs, cache uv chaud) :
synchronisation et tests rapides en 59 s avec `-n auto` (738 tests avant ce lot) ; avec
ce lot, 753 tests en 50 s sur quatre processus ; `ruff` et `check-integrity` en quelques
secondes ; front en 5 s avec le cache npm. Sur un runner
GitHub à quatre cœurs, l'estimation est de 4 à 6 minutes pour `back` et de 1 à 2 minutes
pour `front`, sous la cible de 10 minutes ; `timeout-minutes` arrête une tâche bloquée à
15 et 10 minutes. Le workflow n'a pas encore tourné sur GitHub (`act` n'est pas installé) :
YAML et scripts ont été contrôlés localement, puis chaque commande a été lancée sur un
clone propre.

Pour la mise en route, la règle de protection de `main` peut exiger les deux tâches.
Elle relève des réglages du dépôt et reste à activer par Gaétan.

## 8. Archive durable du gel et des tirages

Le dossier `runs/` est ignoré par Git. Le bilan compact de `docs/results/` ne suffit pas à
reprendre les calculs : le gel contient les sources figées, les tables utilisées, les
ajustements des douze cas et les tirages publiés. Sans lui, la référence ne se rejoue pas
exactement.

### Contenu conservé

Par défaut, `scripts/archive_runs.sh` prend :

- `runs/2026-10-22-consolidation/` complet : `reference.json`, `source/` (code `a89d304`,
  verrou uv, registres, tables curated), `published/` (tirages et manifests de la
  projection du 22 octobre), `cases/` (ajustements, tirages et journaux des douze cas),
  métriques, sensibilités et `review/` ;
- `runs/*-fit-2027*` : ajustements du premier tour 2027 (`draws.npz`, manifest) ;
- `runs/*-simulate*` : simulations 2027 (`draws.parquet`, résumés, manifest) ;
- `runs/prospective/` dès qu'il existe : copies des runs archivés au § 3.

Les études (`runs/2026-10-22-historical-scenarios/`, 183 Mo, et les autres dossiers) ne
sont pas prises par défaut ; elles s'ajoutent en arguments.

### Tailles mesurées le 23 octobre 2026

| Dossier | Fichiers | Taille | Archive zstd |
|---|---|---|---|
| `2026-10-22-consolidation` | 451 | 116,8 Mo | 111,6 Mo |
| `2026-10-20-fit-2027` | 10 | 40,1 Mo | 40,1 Mo |
| `2026-10-20-fit-2027-subst` | 9 | 52,1 Mo | 52,1 Mo |
| `2026-10-22-fit-2027-subst` | 9 | 52,1 Mo | 52,1 Mo |
| `2026-10-20-simulate` | 8 | 6,6 Mo | 6,5 Mo |
| `2026-10-20-simulate-subst` | 8 | 6,9 Mo | 6,8 Mo |
| `2026-10-20-simulate-subst-v2` | 8 | 6,9 Mo | 6,8 Mo |
| `2026-10-20-simulate-subst-v3` | 8 | 6,9 Mo | 6,9 Mo |
| `2026-10-22-simulate-subst` | 8 | 6,9 Mo | 6,8 Mo |
| **Total** | 519 | 295,3 Mo | 289,7 Mo (instantané : 277 Mio) |

Les parquet et `npz` sont déjà compressés : zstd ne gagne presque rien, mais l'archive
reste un seul fichier contrôlable par dossier. Pendant la campagne, prévoir environ
50 Mo par nouvel ajustement et 7 Mo par simulation ou copie archivée, soit de l'ordre de
1 à 2 Go jusqu'en mai 2027.

### Commande

```sh
scripts/archive_runs.sh <destination> [dossier de runs/ …]
scripts/archive_runs.sh --verify <instantané> [dossier de restauration]
```

Sans destination, le script affiche son aide et s'arrête (code 2) ; il refuse une
destination située dans le dépôt. Il écrit `<destination>/<AAAA-MM-JJTHHMMSS>/` : une
archive `<nom>.tar.zst` par dossier (`tar.gz` si `zstd` manque), la liste
`<nom>.files.sha256` des empreintes de chaque fichier, `index.tsv` (fichiers, tailles,
compression, empreinte de l'archive), `meta.txt` (date, machine, HEAD du dépôt, empreinte du
verrou de la référence) et `SHA256SUMS` sur l'ensemble. Il écrit d'abord un dossier
`.partial` renommé à la fin et n'écrase jamais un instantané. Aucune copie réseau.

Le contrôle `--verify` exécute `sha256sum -c SHA256SUMS`, extrait chaque archive (dans un
dossier temporaire supprimé ensuite, ou dans le dossier donné), compte les fichiers,
compare chacun à son empreinte, puis signale sans échec les dossiers de `runs/` modifiés
depuis l'instantané.

### Instantané du 23 octobre 2026 et contrôle de restauration

Destination retenue par Gaétan : **locale**, `/home/orosius/vingtvingtsept-archives/`,
hors du dépôt mais sur le même disque. Elle protège contre une suppression ou une
réécriture de `runs/` (nettoyage, reprise, run du même jour), pas contre la perte du disque
ou de la machine.

- Instantané : `/home/orosius/vingtvingtsept-archives/2026-10-23T102626/`, 20 fichiers,
  277 Mio, `SHA256SUMS` d'empreinte
  `00f2be61bf1582a5941ecbe5272bc14c6c3a5faa9ceda4e54a335e78a398ee9e`.
- Restauration complète dans un dossier temporaire : `SHA256SUMS` conforme, neuf dossiers
  et 519 fichiers restaurés avec leurs empreintes, tous identiques aux dossiers de `runs/`.
- Le gel restauré passe `vvs.eval.consolidation.verify` : 165 fichiers figés conformes,
  et son `reference.json` est identique au verrou versionné (identifiant, commit
  `a89d304`, configuration, versions, empreintes).

**À faire par Gaétan** : copier l'archive sur un support externe, puis la contrôler sur ce
support (une ligne chacune, valables sous fish) :

```sh
rsync -a --checksum /home/orosius/vingtvingtsept-archives/ /chemin/du/support/vingtvingtsept-archives/
scripts/archive_runs.sh --verify /chemin/du/support/vingtvingtsept-archives/2026-10-23T102626
```

Relancer `scripts/archive_runs.sh /home/orosius/vingtvingtsept-archives` après chaque
nouvel ajustement publié ou chaque date d'archivage prospective, puis le même `rsync`. Les
instantanés successifs se recopient en entier ; les supprimer reste une décision manuelle.

### Options de destination externe

Coûts en ordre de grandeur (octobre 2026), à vérifier au moment du choix.

| Option | Coût | Fragilité | Remarque |
|---|---|---|---|
| Disque externe USB | 50 à 80 € pour 1 To, aucun abonnement | panne ou perte du disque, oubli de la copie ; à rebrancher pour chaque copie | simple et hors ligne ; contrôle `--verify` tous les six mois |
| NAS domestique (RAID 1) | 300 à 500 € à l'achat, électricité | même lieu que le poste (vol, incendie) ; le RAID n'est pas une sauvegarde contre une suppression | `rsync` automatisable par cron |
| Git LFS sur GitHub | quota gratuit limité (de l'ordre du Gio), puis facturation au volume | quotas, réécritures d'historique compliquées, dépendance au dépôt | lie les tirages aux commits, mais alourdit chaque clone qui récupère LFS |
| Stockage objet (Backblaze B2, S3, Scaleway Glacier) | quelques centimes par Go et par mois : moins de 1 € par mois pour 2 Go | identifiants à protéger, frais de sortie, dépend d'un compte et d'un moyen de paiement | verrou d'objet (immutabilité) possible ; chiffrer avant envoi si besoin |
| Pièce jointe de release GitHub | gratuit, 2 Gio par fichier | public si le dépôt l'est ; suppression possible par erreur | adapté à un instantané figé comme le gel, peu à la campagne |

Recommandation : un disque externe tout de suite pour le gel (coût faible, aucun compte),
puis une seconde copie hors du domicile avant le début des archives hebdomadaires, de
préférence un stockage objet avec verrou d'objet, dont le coût reste inférieur à 1 € par
mois. Git LFS n'est pas recommandé : la campagne approche ou dépasse le quota gratuit, et les
tirages n'ont pas besoin de suivre chaque commit.
