# 08 — Front

## Choix

SvelteKit avec `adapter-static`, déployé sur GitHub Pages (ou tout hébergeur statique).
Aucun serveur : le site lit des JSON produits par le back et publiés avec lui. Objectif :
un site partageable, rapide, lisible sur mobile, qui montre l'incertitude plutôt que de la
cacher.

Choix faits le 2026-10-20 : SvelteKit 3 (Svelte 5, Vite 8, adapter-static 4, TypeScript 6),
Tailwind 4, graphiques en d3 direct (`d3-scale`, `d3-shape`, `d3-array`) plutôt que
LayerChart, dont les dépendances `@layerstack/*` sont encore en préversion et dont le
rythme de versions est trop rapide pour nos quatre graphiques sur mesure. Chemin de base
par la variable `BASE_PATH`, tout prérendu, `404.html` de repli. Données factices
déterministes produites par `front/scripts/mock-data.js` et signalées par un bandeau tant
que `manifest.mock` vaut vrai.

## Pages

| Route | Contenu |
|---|---|
| `/` | les probabilités du jour (qualification, victoire), la date du dernier sondage intégré, l'évolution des probabilités, un résumé des scénarios ; « Selon vous, qui sera élu ? » face à nos probabilités (hors prévision) |
| `/premier-tour` | courbes de soutien latent par candidat avec intervalles, sondages en points, filtre par configuration |
| `/second-tour` | matrice des duels : probabilité de chaque paire, score attendu, intervalle ; reports de voix estimés |
| `/scenarios` | configurations de candidats, leurs probabilités, l'historique des révisions, la prévision conditionnelle à chacune |
| `/sondages` | table de tous les sondages intégrés, dispersion des instituts et électorats fragiles (hors prévision), effets maison estimés par institut, palmarès |
| `/economie` | séries économiques et popularité, corrélations, ce qui entre ou non dans la prévision |
| `/presse` | presse classée (ton, posture, titres extrêmes, part de voix, agenda, repères 2017 et 2022, qualité), attention Wikipédia, rubriques INA, GDELT, même mention |
| `/methode` | méthodologie en langage clair, résultats du backtest, variables testées par vague, limites |

## Contrat JSON (`front/static/data/`)

Produit par `vvs export`, versionné avec le site. Un fichier par page plus un manifeste :

- `manifest.json` : `run_date`, `asof`, `model_version`, `last_poll_date`, `n_polls`.
- `summary.json` : par candidat, `p_qualify`, `p_win`, quantiles de la part de premier tour ;
  et par configuration.
- `first_round.json` : trajectoires hebdomadaires (médiane, intervalles 50 et 90 %) par
  candidat et configuration ; sondages avec leur part observée.
- `second_round.json` : par paire, `p_pair`, médiane et intervalles du score, reports.
- `scenarios.json` : contenu de `scenarios_2027.yaml` avec historique.
- `polls.json`, `economy.json`, `press.json`, `backtest.json`.
- `history/<date>.json` : résumé de chaque run, jamais réécrit, et `history/index.json`
  (concaténation triée, parce qu'un site statique ne peut pas lister un dossier).

Volets explicatifs (ajoutés le 2026-10-20, phase 8) ; chacun porte `in_forecast: false` et sa
phrase de statut, affichée sur la page :

- `manifest.polls_hash` : empreinte des sondages de premier tour vus par l'ajustement
  (`data.rows_hash`), recopiée par `vvs simulate` ; l'ajustement n'est refait que si elle change.
- `economy.correlations` : une ligne par paire indicateur × bloc de l'étude
  (`runs/<date>-economy/coef_pairs.parquet`) : effet d'un écart type de variation mensuelle de
  l'indicateur, en points de part du bloc par mois, au meilleur retard (`lag_months`), intervalle
  à 90 % (`effect`) et `p_placebo` ; `correlations_note` donne la lecture et le nombre de paires
  qui passent le placebo ; `economy.study` : verdict (`negative`), résumé et constats de
  `05-economie.md`.
- `press.attention` : part hebdomadaire des pages vues Wikipédia des candidats de l'univers 2027
  (`attention.parquet`, semaines complètes, 78 dernières, douze candidats les plus vus sur douze
  semaines) ; `press.ina` : part des rubriques INA dans les sujets de JT de 2002, 2007, 2012 et
  2017, et sur 2000-2020 ; `press.gdelt` : part mensuelle des articles GDELT citant chaque
  candidat suivi depuis 2017 (volume sur volume surveillé).
- Presse classée (ajoutée le 2026-10-22, `runs/2026-10-22-press-nlp/`, prompt `title-v3`,
  titres plafonnés de `press_features`) : `press.nlp` porte `in_forecast: false`, la note, le
  verdict de l'étude en deux phrases et le test hors échantillon déclaré pour 2027. Parts entre
  0 et 1, candidats suivis de l'univers 2027 classés par titres du dernier mois, semaines
  complètes (lundi) finies avant `asof`. `press.tone` : ton moyen sur deux semaines glissantes,
  quantiles bootstrap `q05`, `q50`, `q95` (tirage multinomial des trois niveaux, 1 000 tirages,
  graine fixe) et `n` titres, null sous 10 titres ; `press.stance` : parts des sept postures
  (`categories`) sur les quatre dernières semaines (`last_weeks`) et par semaine, null sous 10
  titres ; `press.headlines` : dernière semaine, par candidat, trois titres classés −1 et trois
  +1 (titre, média, date, url, posture), les plus récents d'abord ; `press.share_of_voice` :
  part des titres nommant chaque candidat (`mentions`) et, mêmes semaines et candidats, part de
  leurs pages vues Wikipédia (`attention`) ; `press.agenda_nlp` : part hebdomadaire des titres
  portant immigration, pouvoir d'achat, international, sécurité (seuls thèmes concordants),
  null sous 10 titres ; `press.landmarks` : Fillon 2017 et Pécresse 2022, ton et part « affaire »
  jour par jour (moyenne trois jours) de 28 jours avant à 42 jours après le choc, dates de
  rupture lues dans `shocks.csv` ; `press.quality` : κ et F1 d'`agreement.json`, modèle, version
  de prompt, titres classés par cycle. Les anciens champs vides `weeks` et `agenda` sont retirés.
  Invariants vérifiés par `validate-data` : ton dans [−1, 1] et quantiles croissants, parts de
  postures et de voix de somme 1 à 0,01 près, valeur nulle si et seulement si moins de
  `min_titles` titres, titres extrêmes datés dans la semaine, dates de rupture dans la fenêtre.
- `polls.pollster_ratings` : palmarès 2002-2022 par lignée (`runs/<date>-pollster-ratings/`) :
  erreur du dernier sondage, écart à l'institut moyen, biais par bloc, précision, et stabilité
  du classement entre élections consécutives (corrélations poolées), avec la phrase « classement
  non reproductible d'une élection à l'autre, n'entre pas dans la prévision ».
- `backtest.variables_tested` : les six variables de la vague du 2026-10-20, la presse classée
  et les cinq pistes de la troisième vague du 2026-10-22 (volatilité par candidat, troupeau,
  anticipations, abstention différentielle, Cevipof « en attente d'accès ») ; une ligne chacune,
  `published` vrai pour la substitution, utilisée par la projection publiée, et `wave` (date de
  la vague) qui les groupe sur `/methode`.
- Indicateurs d'opinion de la troisième vague (ajoutés le 2026-10-22), tous `in_forecast: false`
  avec leur phrase de statut :
  - `summary.expectations` (accueil) : « Selon vous, qui sera élu ? »
    (`poll_expectations.parquet`). `measures` : dernière mesure 2027 terminée avant `asof` par
    institut, forme de question (`winner` : une réponse, part brute et `share_named`, part parmi
    les répondants qui nomment un candidat ; `chances` : « peut-il gagner ? » posée candidat par
    candidat) et objet (`candidate` ou `party`, Toluna Harris Interactive par parti) ; le site
    les met en regard de `p_win` du mélange. `history` : mesures `winner` vérifiées de 2007 à
    2022 d'avant le premier tour, part normalisée du futur vainqueur et candidat le plus cité,
    en petit graphique. `landmarks` : deux repères calculés sur cette série à l'export (le plus
    cité élu dans 38 mesures sur 39, seule exception l'épisode Fillon ; part du futur
    vainqueur jamais au-dessus de 70 %, 63 % depuis 2012).
  - `polls.dispersion` (`/sondages`) : dispersion relative D des instituts
    (`vvs.model.herding.presidential_dispersion`, un sondage par institut sur 30 jours,
    configuration connue à la date), un point par semaine sur 26 semaines jusqu'à `asof`, null
    sous quatre instituts ; `reference` : D_ref des présidentielles 2002-2022 au même nombre de
    jours du vote (`herding.reference`, interpolé, constant au-delà de 180 jours faute de
    campagnes mesurables) ; `past_same_horizon` : D de chaque campagne passée à la même distance.
    Phrase imposée : « les instituts divergent nettement, surtout parce qu'ils testent des
    configurations différentes ; historiquement, le désaccord accompagne les grosses erreurs, il
    ne les masque pas ».
  - `polls.turnout_gap` (`/sondages`) : électorats fragiles, écart certains d'aller voter −
    inscrits par candidat en 2027 (`turnout_gap.parquet`, OpinionWay, premier tour) : moyenne
    par sondage puis entre sondages, erreur-type entre sondages (null sous deux sondages), triés
    du plus fragile au moins fragile, avec le rappel de 2017 et 2022.
  Invariants de `validate-data` : mesures et sondages datés au plus `asof`, candidat ou parti
  (pas les deux), part normalisée pour `winner` seulement et au moins égale à la part brute,
  `top_is_winner` cohérent ; séries de dispersion alignées sur `weeks`, dernière semaine =
  `asof` ; écarts triés, cohérents avec les deux parts (à l'arrondi près), erreur-type nulle si
  et seulement si moins de deux sondages ; candidats présents dans le manifeste.

Les schémas sont décrits en JSON Schema dans `back/src/vvs/export/schemas/` et validés à
l'export ; le front les type via génération TypeScript (`npm run types`) et les valide avec
ajv plus des invariants hors schéma (`npm run validate-data`) : quantiles croissants, somme
des `p_win` égale à 1 et des `p_qualify` à 2, `p_win ≤ p_qualify ≤ p_present`, références
entre fichiers. Conventions : parts en points d'exprimés, probabilités entre 0 et 1, tout
intervalle vaut `{q05, q25, q50, q75, q95}` ; une **condition** (`mix`, `<dimension>.<état>`,
ou `cfg.<états>`) dit sur quoi porte une prévision.

Ce que `vvs export` devra fournir en plus de ce que le modèle produit :
- des libellés courts par dimension et par état (« RN : Bardella »), à ajouter dans
  `scenarios_2027.yaml` (champ `label`) ;
- le bloc des candidats 2027 via `bloc_of` ;
- la compatibilité d'un sondage avec une condition : l'hypothèse contient les candidats de
  l'état et aucun candidat d'un autre état de la même dimension.

Poids des pages : les JSON sont inlinés au prérendu (environ 600 Ko pour le premier tour
avec les données factices). Si l'export réel grossit, découper `summary.json` ou charger
côté client.

## Règles d'affichage

- Toujours un intervalle avec une médiane, jamais un point seul.
- Les probabilités de qualification et de victoire s'affichent avec le mélange de
  scénarios en premier, le détail conditionnel en second.
- Chaque page qui montre une source non sondagière dit en une phrase si elle entre dans la
  prévision.
- La date du dernier sondage intégré est visible sur chaque page.
