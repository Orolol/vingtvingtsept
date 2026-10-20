# 03 — Données

## Sources

### Sondages

| Source | Couverture | Usage | Notes |
|---|---|---|---|
| Wikipédia FR, pages « Liste de sondages sur l'élection présidentielle française de AAAA » | 2002, 2007, 2012, 2017, 2022, 2027 | série historique principale | tableaux hétérogènes d'une page à l'autre : un adaptateur par élection |
| Commission des sondages (notices) | 2016 à 2027, exhaustif par obligation légale | série live 2027, méthodes, population, résultats bruts quand publiés | PDF texte sans gabarit commun ; listing par thème `bytag/14/2027-Presidentielle` plus recherche « Pres » |
| MieuxVoter/presidentielle2027 (GitHub, JSON + CSV, MIT) | 2025 à 2027, environ dix jours de retard | contrôle croisé de couverture et de dates ; sous-échantillons inscrits / certains d'aller voter | successeur de fait de NSPPolls, qui s'arrête en mai 2022 |
| Sondages de second tour | mêmes pages Wikipédia | reports de voix | hypothèses de duels, série creuse |

Règle : Wikipédia est la source d'enregistrement pour le passé, la Commission pour 2027, et
toute divergence entre sources sur un même sondage est journalisée dans la curation.

Ce que les notices contiennent réellement (relevé d'octobre 2026) : Cluster17, OpinionWay
(depuis mars 2026) et YouGov publient brut, redressé et publié pour chaque hypothèse ; Ifop
seulement pour sa grande enquête de mai 2025 ; Elabe, Toluna Harris, Ipsos BVA et Verian
le publié seulement ; Odoxa aucun résultat. Les parts sont quasi identiques entre Wikipédia
et MieuxVoter (aucun écart au-delà de 0,5 point sur 286 hypothèses), signe que les deux
recopient les notices : le contrôle croisé vaut pour la couverture et les dates, pas pour
les valeurs. Divergences connues, à traiter par règles de curation et non à la main :
Elabe 25-27 mars 2026 (colonne décalée sur Wikipédia, Hollande et non Glucksmann à 8,5),
Ifop Hexagone avril 2025 (duel Philippe–Le Pen inversé chez MieuxVoter), Ifop mars 2025
(dates fausses sur Wikipédia), deux OpinionWay 2026 et un Cluster17 juillet 2026 absents de
Wikipédia.

Points de vigilance connus sur Wikipédia :

- les tableaux changent de forme entre pages et parfois au milieu d'une page (colonnes
  par candidat, hypothèses imbriquées, notes de bas de tableau) ;
- la population est tantôt « inscrits », tantôt « certains d'aller voter », rarement
  explicite ;
- les dates de terrain sont des plages, parfois sans année dans la cellule ;
- les hypothèses de premier tour avec des candidats différents sont des lignes distinctes
  ou des sous-tableaux ;
- les anciens sondages (2002, 2007) sont moins nombreux et moins documentés.

On stocke le HTML brut horodaté dans `data/raw/wikipedia/`, et on parse à partir de là, de
sorte que le parseur puisse être corrigé sans retélécharger.

### Résultats électoraux

| Source | Contenu |
|---|---|
| Ministère de l'Intérieur via data.gouv.fr | présidentielles 2002 à 2022 par commune et bureau de vote ; législatives 2022, 2024 ; européennes 2014, 2019, 2024 |

Usage : vérité terrain du backtest (national), prior structurel des blocs (national et
par département), base future des législatives. On conserve les fichiers bruts hors git
(`data/raw/results/`) et on versionne les agrégats nationaux et départementaux.

### Registre candidats, partis, blocs

Fichiers YAML dans `data/registry/`, validés par pydantic :

- `candidates.yaml` : une entrée par personne (id stable, nom, partis successifs).
- `elections.yaml` : par élection, la liste des candidats officiels avec parti et bloc, les
  dates des tours, les résultats nationaux officiels.
- `blocs.yaml` : définition des blocs et de leurs frontières, avec la justification.
- `pollsters.yaml` : instituts, alias rencontrés dans les sources, mode (téléphone, en ligne,
  mixte), date d'entrée.
- `scenarios_2027.yaml` : configurations de candidats et leurs probabilités, datées et
  justifiées (voir `04-modele.md`).

Blocs retenus pour commencer, à discuter avant la phase 2 :

| Bloc | Contenu |
|---|---|
| `far_right` | FN/RN, Zemmour, Debout la France |
| `right` | UMP/LR, Fillon, UDI quand hors majorité |
| `centre` | Bayrou 2007-2012, Macron et majorité présidentielle, Horizons |
| `left` | PS, Hollande, Jospin, écologistes, Hamon |
| `far_left` | Mélenchon, PCF, NPA, LO |
| `other` | candidats non rattachés (Lassalle, Cheminade, chasseurs...) |

La frontière écologistes / gauche radicale et la position de Mélenchon avant 2012 sont les
cas limites à trancher explicitement dans `blocs.yaml`.

## Schémas

Format parquet, colonnes en anglais, une table par entité.

### `polls`

Une ligne par sondage (unité de collecte).

| Colonne | Type | Description |
|---|---|---|
| `poll_id` | str | clé stable : `{election}-{pollster}-{fieldwork_end}-{seq}` |
| `election` | int | année de l'élection |
| `pollster` | str | id du registre `pollsters.yaml` |
| `sponsor` | str, nullable | commanditaire |
| `fieldwork_start`, `fieldwork_end` | date | terrain |
| `publish_date` | date, nullable | publication |
| `sample_size` | int, nullable | échantillon total |
| `mode` | enum | `online`, `phone`, `mixed`, `unknown` |
| `source` | enum | `wikipedia`, `commission`, `nsppolls` |
| `source_url` | str | page ou notice |
| `ingested_at` | datetime | horodatage de l'ingestion |

### `poll_results`

Une ligne par intention de vote.

| Colonne | Type | Description |
|---|---|---|
| `poll_id` | str | clé vers `polls` |
| `round` | int | 1 ou 2 |
| `hypothesis_id` | str | identifie la configuration de candidats testée (hash trié des ids candidats) |
| `population` | enum | `registered`, `certain_voters`, `unknown` |
| `candidate_id` | str | clé vers le registre ; `abstention`, `blank`, `undecided` sont des ids réservés |
| `share` | float | pourcentage, base exprimés sauf mention |
| `share_base` | enum | `expressed`, `all_respondents`, `unknown` |
| `raw_share` | float, nullable | résultat brut avant redressement, si la notice le donne |

### `results`

Une ligne par candidat, par élection, par tour, par niveau géographique.

| Colonne | Type |
|---|---|
| `election` | int |
| `round` | int |
| `geo_level` | enum `national`, `department`, `commune`, `polling_station` |
| `geo_code` | str : `FE` au national ; code INSEE sur 2 ou 3 caractères au département, `ZZ` pour les Français de l'étranger ; code INSEE sur 5 caractères à la commune, `ZW001`/`ZZ001`… pour Wallis-et-Futuna et les postes à l'étranger qui n'en ont pas ; Saint-Martin et Saint-Barthélemy séparés en 978 et 977 |
| `candidate_id` | str |
| `votes` | int |
| `registered`, `voters`, `expressed` | int |

Les fichiers du ministère sont lus au bureau de vote puis agrégés à la commune ; le niveau
bureau n'est pas conservé. Paris, Lyon et Marseille sont des communes entières. En 2002 et
2007, les fichiers par bureau ne couvrent ni l'étranger ni le Pacifique : le niveau
département vient des classeurs récapitulatifs et le niveau commune est partiel. Le fichier
2002 précède les rectifications du Conseil constitutionnel (écarts de quelques dizaines à
quelques centaines de voix, figés dans les tests). Les parquets `results_<election>.parquet`
(1,5 Mo chacun à cause des communes) ne sont pas versionnés ; ils se régénèrent depuis les
sources dont l'empreinte sha256 est dans `data/registry/interieur_sources.yaml`.

### `results_lists`

Scrutins de liste et législatives, au national seulement pour l'instant, libellés
conservés tels quels ; le rattachement aux blocs se fait dans la curation.

| Colonne | Type |
|---|---|
| `election` | str : `euro2014`, `euro2019`, `euro2024`, `leg2022`, `leg2024` |
| `round`, `geo_level`, `geo_code` | comme `results` |
| `label` | str, libellé de liste ou de nuance tel que publié |
| `nuance` | str, nullable (vide pour les européennes 2019, le fichier n'en a pas) |
| `votes`, `registered`, `voters`, `expressed` | int |

### Tables dérivées

- `poll_results_by_bloc` : agrégation des parts par bloc et hypothèse, utilisée par le
  prior structurel et les études.
- `poll_errors` : pour chaque élection, écart entre le dernier sondage de chaque institut
  (et l'agrégat) et le résultat, par candidat et par bloc. Entrée du modèle d'erreur.

## Ce que l'ingestion Wikipédia a révélé (2017, 2022, 2027)

- Les pages n'indiquent ni la population interrogée ni le mode : `population` et `mode`
  valent `unknown` partout. Les notices de la Commission les fournissent ; la curation
  complétera par jointure quand le sondage est présent dans les deux sources.
- Les candidats sont en base exprimés ; `abstention` et `undecided` sont stockés en base
  `all_respondents`. La colonne « Abstention, blanc ou nul » de 2017 est rangée sous
  `abstention`, approximation assumée.
- « <0,5 % » est stocké comme 0,25 ; « — » signifie non testé, pas zéro.
- Un même institut avec les mêmes dates de terrain et des échantillons différents (second
  tour posé à un sous-échantillon, coquille sur l'effectif) est fusionné en un seul sondage
  si les hypothèses ne se recouvrent pas et que les commanditaires ne se contredisent pas ;
  `sample_size` garde le plus grand. Un rolling et une vague Cevipof du même jour restent
  distincts.
- Les rollings (« Ifop (rolling) ») sont des sondages ordinaires de l'institut. **À faire en
  curation** : une colonne `is_rolling` booléenne, parce que les vagues d'un rolling ne sont
  pas indépendantes et le modèle doit en tenir compte (effectif effectif réduit).
- Les hypothèses avec un candidat non nommé (« candidat LR », « Bardella / Le Pen »,
  « un candidat écologiste ») sont ignorées et journalisées, jamais devinées.
- Les tableaux de reports de voix et de sous-échantillons sociologiques sont sautés ; les
  reports sont une source à part pour le modèle de second tour (`04-modele.md`).
- Les sondages 2027 testant Emmanuel Macron (2022-2023) sont conservés ; le registre de
  scénarios les écartera.

## Curation : règles explicites

- Déduplication entre sources sur (`pollster`, `fieldwork_start`, `fieldwork_end`,
  `sponsor`), tolérance d'un jour sur les dates ; en cas de conflit sur une valeur, la
  Commission l'emporte sur Wikipédia, qui l'emporte sur NSPPolls, et l'écart est journalisé.
- Un sondage dont les parts d'une hypothèse ne somment pas entre 97 et 103 est marqué
  `suspect` et exclu par défaut, jamais corrigé à la main.
- Rattachement des libellés de candidats au registre par table d'alias versionnée ; un
  libellé inconnu fait échouer la curation, on ne devine pas.
- Les candidats absents d'une hypothèse ne sont pas des zéros : ils sont absents. Le modèle
  travaille sur l'ensemble des candidats de l'hypothèse.

## Tests

- Parseurs Wikipédia : un fichier HTML figé par élection dans `tests/fixtures/`, avec le
  nombre de sondages attendu et quelques lignes vérifiées à la main.
- Curation : invariants (sommes, unicité des clés, alias résolus, dates ordonnées).
- Registre : chaque candidat des `results` nationaux existe dans `elections.yaml`.
