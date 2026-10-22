# 11 — Reconstruction des scénarios historiques

Protocole déclaré le 22 octobre 2026 **avant calcul des variantes**. La référence
`reference-2026-10-22`, ses sources et ses résultats restent inchangés. Paramètres de
l'étude : `data/registry/historical_scenarios_2026-10-22.yaml`.

## Hypothèse et variantes

La fenêtre récente de 30 jours peut éliminer un candidat déjà observé. On teste une
mémoire de **365 jours**, avec décroissance exponentielle de **demi-vie 60 jours**.
Ces deux durées sont des choix de protocole communs aux trois élections, sans recherche
de paramètres sur les scores. Un sondage reçoit le poids `2 ** (-âge / 60)` ; ce poids
est partagé également entre ses ensembles distincts de candidats. Les configurations
identiques sont regroupées et les poids normalisés. Les hypothèses restent des listes
complètes effectivement sondées : aucune combinaison de candidats n'est inventée.

- `memory` : cette mémoire seule, ablation du filtre événementiel.
- `memory_events` : variante principale, même mémoire puis exclusion des configurations
  incompatibles avec les événements déjà connus. Un retrait exclut le candidat ; une
  désignation exclut les autres candidats du même parti, selon l'affiliation à l'année
  de l'événement dans le registre. On élimine la configuration entière et renormalise ;
  on ne remplace jamais un candidat par le vainqueur d'une primaire.

Les événements viennent exclusivement du **registre déjà figé**, pour l'élection
concernée, à partir du lendemain de leur date. Seuls `withdrawal` et `primary` agissent.
Les candidats et événements futurs ne peuvent pas enrichir le support. Les déclarations
ne créent pas de candidat sans sondage. La liste officielle remplace les scénarios
exactement à la même date que dans la référence. Un ensemble admissible vide est une
erreur explicite ; aucune reprise silencieuse de la référence.

Le filtre par parti est une approximation de désignation, pas une liste exhaustive de
participants aux primaires. Le registre des événements, constitué pour une autre étude,
est incomplet ; les affiliations n'ont qu'une résolution annuelle et sont rétrospectives.
Il peut donc rester des hypothèses périmées ou en disparaître trop. L'audit conservera les
événements appliqués, les candidats exclus et la masse supprimée. Les sources datées du
registre seront conservées avec l'audit ; aucune nouvelle chronologie n'est déduite des
résultats présidentiels.

## Comparaison fixée

Mêmes douze cas (2012, 2017, 2022 × 180, 90, 30, 7 jours), disponibilité des données,
calibrations chronologiques, priors, 40 000 simulations, graine et métriques que dans
[la consolidation](10-consolidation.md). Les variantes s'appliquent à l'agrégateur
bayésien **et** au témoin `mean14d`. Les ajustements bayésiens figés sont réutilisés
uniquement si tous les candidats demandés figurent déjà dans leurs tirages ; sinon
l'étude s'arrête. Leur estimation ne dépend pas des poids des scénarios. Les fonctions
de simulation et de score sont celles du gel `a89d304`, importées depuis sa copie.

Un rejeu des scénarios originaux vérifiera les résultats numériques avant comparaison.
Les cas dont les configurations restent identiques réutiliseront les sorties validées.
Les scénarios reconstruits, probabilités de présence, scores détaillés, diagnostics et
empreintes seront écrits dans un dossier d'étude distinct. Une reprise vérifiera les
entrées et sorties ; aucun fichier du gel ne sera réécrit.

Critères de lecture inchangés : log-score de la paire et CRPS à 180 et 90 jours, signe
par élection, couverture, absence de dégradation aux autres horizons ; log-score et
Brier de victoire également publiés. Aucune variante ne sera promue automatiquement.
On publiera aussi les voix finales hors support et les probabilités de présence :
ajouter un candidat avec une probabilité infime ne suffit pas à bien le prévoir.

Cette comparaison est exploratoire après constat du défaut sur les mêmes élections,
et non un nouveau test indépendant. Trois élections ne permettent pas de conclure à
une amélioration générale. La même graine ne garantit pas des tirages appariés quand
les configurations changent ; les petits écarts restent soumis au bruit Monte-Carlo.
Un candidat jamais sondé dans les données disponibles demeure absent, notamment Hamon
à 180 jours de 2017. Une mémoire plus longue ne corrige pas cette limite.

## Résultats

À renseigner après exécution, sans modification des règles ci-dessus.
