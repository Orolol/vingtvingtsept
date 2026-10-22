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

### Étude par comptages du 2026-10-20 (`runs/2026-10-20-press-study/report.md`)

Media Cloud, 180 requêtes `count-over-time` (part de voix de 18 candidats par cycle, 10
thèmes, co-mentions « affaires » comme proxy de couverture négative), GDELT ton pour quatre
candidats, INA pour 2002-2017 ; critères fixés avant lecture, placebos, 2017, 2022, 2027.

**Négatif sur les six tests.** Part de voix → sondages à 4 semaines : β = −0,009 (placebo
p = 0,96) ; ton GDELT et part d'affaires : rien ; pics de volume : mouvement moyen 0,82
contre 0,77 au hasard ; agenda immigration et sécurité → extrême droite : mauvais signe dans
chaque cycle, réplication INA nulle ; médiation chômage × couverture : signe opposé ;
erreur finale : signes opposés entre 2017 et 2022. Dans une régression jointe, l'attention
Wikipédia garde son effet (+0,16) et la presse prend un signe négatif (−0,17) : la presse
mesure les mêmes événements avec moins de signal, et ce qui précède un rattrapage des
sondages est l'intérêt du public au-delà de la couverture. Lecture faite après coup, signe
inversé en 2027.

Descriptif : l'affaire Fillon n'est pas un pic de volume mais un changement de composition
(part de sa couverture mentionnant des affaires de 2 à 22 % la semaine du Canard enchaîné),
simultané à la chute dans les sondages. Pécresse 2022 perd cinq points après le Zénith sans
aucun signal de presse.

Décision : **aucune variable de presse au backtest**. Media Cloud reste pour la section
« Presse » du site (part de voix et agenda, environ 300 requêtes par semaine), marquée
« n'entre pas dans la prévision ». La classification NLP des titres n'a plus d'objet
prédictif ; elle reste possible pour l'affichage.

### Classification des titres par modèle local (2026-10-22, `runs/2026-10-22-press-nlp/report.md`)

178 000 articles Media Cloud (2017, 2022, 2027), 58 000 titres classés par Qwen3.8-27B via
llama.cpp (ton −1/0/+1, posture, thèmes ; grammaire contrainte, 9,5 titres/s, prompts
versionnés dans `data/registry/nlp_prompts.yaml`). Accord entre deux formulations : posture
fiable (κ 0,74), ton à la limite (κ 0,60), thèmes non fiables (F1 0,55, à cause
d'« institutions »). Échantillon `sample_300.csv` à relire à la main.

**Résultat : négatif pour la prévision.** Le ton a le signe attendu dans les trois cycles
mais un effet indistinguable du placebo ; autour des chocs, la presse enregistre en même
temps que les sondages (Fillon) ou après (Pécresse) ; l'agenda par thèmes classés ne tient
pas. Seule piste : le ton des 28 derniers jours contre l'erreur finale des sondages (β +6,
p = 0,06, n = 23, porté par 2022), déclaré **test hors échantillon pour 2027** : pour chaque
candidat officiel, ton moyen des 28 jours avant le premier tour (même prompt, même modèle)
contre résultat moins moyenne des sondages des 14 derniers jours, prédiction β > 0.

Apport réel du NLP : descriptif. Il sépare une couverture abondante et favorable d'une
couverture abondante et hostile, ce que les comptages confondent. Le site l'affiche avec la
mention « n'entre pas dans la prévision » : ton hebdomadaire, posture, titres extrêmes, part
de voix, agenda limité aux thèmes concordants. Coût live : environ six minutes de GPU par
semaine.

Incident du 2026-10-22 : un premier essai avec vLLM (NVFP4, mémoire GPU 0,82) a gelé la
machine pendant la capture des graphes CUDA. Règle du projet : inférence locale par
llama.cpp ; vLLM seulement avec workers de compilation limités, swap de 32 Go, mémoire GPU
réduite et accord explicite.
