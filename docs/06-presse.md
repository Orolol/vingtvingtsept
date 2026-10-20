# 06 — Presse et NLP

Objectif : construire, par semaine et par candidat ou bloc, des variables issues des titres
de presse (volume, ton, agenda), les étudier contre la dynamique des sondages, puis les
admettre ou non dans la prévision selon `07-evaluation.md`.

## Sources d'archives

Critère principal : profondeur d'archive, pour backtester 2012, 2017 et 2022.

| Source | Profondeur | Contenu | Accès |
|---|---|---|---|
| Media Cloud | 2008 en théorie, mais les collections par pays datent de 2018 : couverture 2012 et 2017 à mesurer avant d'y compter | titres, dates, URL ; collection « France - National » id `34412146`, 330 sources | gratuit, clé après inscription sur search.mediacloud.org, 4 000 requêtes par semaine, client Python `mediacloud` |
| GDELT | API DOC depuis le 1er janvier 2017 (`sourcelang:french`, `sourcecountry:FR`, mode liste limité à trois mois par requête, donc interrogation jour par jour) ; fichiers GKG bruts avec titres depuis le 22 septembre 2019 seulement, français dans le flux `translation` | titres, ton, thèmes, volume | gratuit, sans clé, erreurs 429 fréquentes : client avec cache et reprise |
| NewsCatcher | 2019 | titres et résumés | sur devis, 10 000 appels offerts |
| WorldNewsAPI | 2022 pour la France | titres, environ 1 560 sources françaises | payant |
| Europresse, Factiva | antérieur à 2000 | texte intégral | pas d'API individuelle ; export manuel en bibliothèque, pour un contrôle ponctuel |
| Wayback Machine sur flux RSS (Le Monde, Le Figaro, AFP via sites) | variable depuis 2008 | titres | gratuit, scraping, couverture irrégulière |
| INA, sujets des journaux télévisés (data.gouv) | 2000 à 2020 | thème et durée de chaque sujet de JT | gratuit ; seule série d'agenda homogène avant 2012 |

Décision : **Media Cloud en colonne vertébrale** (titres historiques), **GDELT en
complément** (ton, thèmes, volume depuis 2015), NewsCatcher seulement si des résumés
deviennent nécessaires sur la phase live. Les résumés ne sont pas indispensables : les
variables prévues se calculent sur les titres, et un résumé peut être généré localement
(Qwen local) à partir du titre et du chapô.

Stockage : `data/raw/press/<source>/<année>/` en fichiers bruts (hors git), puis
`data/curated/press_items.parquet` : `item_id`, `source`, `outlet`, `published_at`,
`title`, `url`, `lang`, et colonnes de ton et thèmes natives quand la source les fournit.

Panel de titres : on fixe une liste de titres de presse nationale (quotidiens, hebdos,
chaînes d'info, AFP via relais) stable sur la période, pour ne pas confondre une hausse de
volume avec un changement de couverture de la source. Liste dans
`data/registry/outlets.yaml`.

## Variables dérivées (`press_features.parquet`, une ligne par semaine et par entité)

| Variable | Définition |
|---|---|
| `mentions` | nombre de titres mentionnant le candidat (alias du registre) |
| `share_of_voice` | `mentions` / total des mentions de candidats |
| `tone` | moyenne du sentiment des titres mentionnant le candidat, échelle −1 à 1 |
| `shock` | indicateur de pic de volume négatif (z-score sur 8 semaines) |
| `topic_<thème>` | part des titres politiques portant sur chaque thème : immigration, sécurité, pouvoir d'achat, chômage, santé, écologie, Europe, international, institutions, affaires |
| `issue_fit_<bloc>` | saillance pondérée des thèmes « possédés » par le bloc (théorie de la propriété des enjeux) |

## NLP

- **Détection des mentions** : alias du registre, désambiguïsation simple (prénom + nom,
  fonction). Évaluation sur un échantillon annoté à la main de 300 titres.
- **Ton** : classification zéro-shot par le modèle local (Qwen3.6-27B) avec un prompt fixé
  et versionné, ou CamemBERT affiné sur un jeu annoté si le volume impose la vitesse. Le
  modèle local tourne sur le GPU partagé : respecter la règle globale de vérification avant
  usage, traiter par lots, sauvegarder les sorties en parquet pour ne jamais reclassifier.
- **Thèmes** : même approche, étiquetage multi-label dans la liste fermée ci-dessus.
- **Qualité** : accord avec les annotations manuelles rapporté dans `runs/<date>/press/`,
  seuil d'acceptation documenté avant de lancer la classification de masse.

## Protocole d'étude (`back/studies/press.py`)

1. **Part de voix contre émergence** : la part de voix précède-t-elle la montée dans les
   sondages (Macron fin 2016, Zemmour été 2021) ? Corrélations croisées avec retards, test de
   Granger dans les deux sens, car la presse suit aussi les sondages.
2. **Chocs** : les pics négatifs (affaire Fillon, janvier 2017) sont-ils suivis d'un
   décrochage, et à quel délai ?
3. **Agenda** : la saillance des thèmes explique-t-elle les variations de part des blocs,
   seule et avec les séries économiques (`05`) ?
4. **Robustesse** : par horizon, par élection laissée de côté, par sous-ensemble de titres.

## Intégration dans la prévision, si admission

Dérive `μ_c(τ)` ou `μ_b(τ)` dans la projection, avec les mêmes conditions que pour
l'économie. La part de voix est le candidat le plus plausible pour un effet à court terme ;
l'agenda pour un effet de bloc. Résultat négatif consigné ici.

## Phase live

Ingestion quotidienne depuis Media Cloud et GDELT, classification incrémentale, section
« Presse » du site : part de voix, ton, agenda de la semaine, avec la mention de ce qui entre
ou non dans la prévision.

## Résultats du 2026-10-20

- **Ingestion** (`runs/2026-10-20-press/`) : INA 2000-2020 complet (268 424 lignes,
  cinq chaînes, 14 rubriques ; pas 2022, pas de titres de sujets, sécurité mal isolée).
  GDELT : l'API cherche dans la traduction anglaise automatique des articles (« Marine Le
  Pen » en français donne 3 articles en janvier 2017), d'où des alias adaptés
  (`gdelt_aliases`), et le volume total surveillé s'effondre (624 000 articles par jour
  en 2017, 103 000 en 2026) : seul `volume_share` est comparable entre années. Limite de
  débit réelle : une réussite toutes les 10 à 20 minutes. Media Cloud : client prêt, clé
  fournie, couverture 2012-2022 en cours de mesure (limite d'une requête toutes les 35 à
  70 s).
- **Attention en ligne** (`runs/2026-10-20-attention/report.md`) : pages vues Wikipédia (97
  candidats depuis 2015) et Google Trends (90 candidats depuis 2004, chaînage sur pivot,
  imprécis après 2014). Résultat : l'attention Wikipédia précède les sondages de une à
  quatre semaines comme rattrapage de mesure autour des événements (0,35 point par écart
  type), effet nul hors événements, Granger inverse significatif, Trends ne réplique pas.
  **Non retenue** pour la prévision ; utilisable comme témoin d'actualité sur le site.

### Couverture Media Cloud mesurée le 2026-10-20 (`data/curated/press/mediacloud_coverage_2012_2017_2022.csv`)

| Année | Titres dans la collection France - National |
|---|---|
| 2012 | 32 816 |
| 2017 | 1 204 574 |
| 2022 | 1 779 821 |

2012 est inutilisable (quelques dizaines de titres par média sur l'année). 2017 et 2022 sont
exploitables au niveau de la collection, mais la composition par média change fortement
d'une année à l'autre (BFMTV 1 597 titres en 2017 contre 140 704 en 2022 ; RTL 33 939 en
2017 contre 53 en 2022) : toute part de voix doit être normalisée par le total de la
collection, et une comparaison média par média n'a pas de sens. Limite de débit observée :
une requête toutes les 35 à 70 secondes. Conséquence : Media Cloud sert au backtest 2017 et
2022 et au live 2027 par comptages (`count-over-time`), pas par extraction massive de
titres ; INA couvre 2002 à 2017 pour l'agenda télévisé.
