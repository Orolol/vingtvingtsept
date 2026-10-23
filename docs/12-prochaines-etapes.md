# 12 — Suite proposée après la consolidation

Statut : feuille de route proposée le 22 octobre 2026. Les hypothèses, paramètres et
critères des prochaines expériences seront déclarés avant leurs calculs. La référence
`reference-2026-10-22` conserve son code, ses données et ses résultats.

L'[étude des scénarios historiques](11-scenarios-historiques.md) donne une direction
précise : élargir les listes sondées réduit les omissions, mais leurs fréquences ne
fournissent pas des probabilités de candidature satisfaisantes. En 2017 à 180 jours,
les voix finales hors support passent de 28,50 % à 6,36 %, tandis que le log-score de
victoire se dégrade. Le prochain chantier prioritaire porte donc sur la représentation
des candidatures et de leur chronologie.

## 1. Construire un registre historique des candidatures

Livrable proposé : un registre sourcé pour 2012, 2017 et 2022, interrogeable à une date
donnée, accompagné de son audit aux douze dates du backtest.

Pour chaque événement, distinguer la date à laquelle il se produit de la date à laquelle
l'information devient publique. Conserver le candidat, le scrutin, le type d'événement,
le processus de désignation concerné, la source, sa date de publication et la date de
collecte. Toute date imputée doit rester identifiable. Inclure les déclarations,
participations et éliminations aux primaires, désignations, renoncements et retraits.

Le registre doit couvrir aussi les candidatures finalement absentes et les personnes
déjà envisagées publiquement mais jamais sondées. La collecte doit suivre des sources
et des périodes définies à l'avance ; la liste des candidats finalement présents ne
doit pas servir de liste de collecte. Les données historiques restent reconstruites
aujourd'hui, même avec des sources contemporaines correctement datées.

Les décisions d'une primaire s'appliquent à ses participants et aux engagements documentés.
Une affiliation commune ne suffit pas à exclure tous les autres candidats du parti.
L'absence de sondage récent ne vaut pas retrait, et une déclaration formelle n'est pas
nécessaire pour qu'une candidature soit publiquement envisagée.

Critères de fin du lot :

- chaque transition appliquée possède une source et une date de disponibilité ;
- l'état à une date ne change pas quand on ajoute des événements futurs au registre ;
- aucun candidat n'apparaît avant la première trace publique retenue ;
- les primaires, retraits et éventuels retours sont audités aux douze dates ;
- les omissions et désaccords entre sources sont publiés, sans correction fondée sur le
  résultat électoral.

Ce lot peut être livré et relu avant tout ajustement d'une nouvelle loi de scénarios.

## 2. Construire les scénarios par décision de candidature

Livrable proposé : une variante expérimentale qui assemble des choix de candidats et
des candidatures indépendantes selon des contraintes explicites. Les axes suivent les
processus de désignation documentés ; un bloc politique peut contenir plusieurs
candidatures simultanées.

Le cas de régression est concret : retirer Hollande doit laisser une possibilité de
présence à Asselineau et Lassalle, même si leur seule liste sondée commune contient
Hollande. Il faut donc pouvoir recomposer une configuration sans réutiliser toute une
ancienne liste, tout en respectant les incompatibilités établies par les sources.

Séparer deux incertitudes : la probabilité d'être candidat et le soutien électoral
conditionnel à cette candidature. Les sondages présidentiels renseignent surtout la
seconde. Les poids de candidature devront avoir leur propre justification : état daté,
éventuels sondages de primaire disponibles à cette date, ou hypothèses explicites dont
on mesure la sensibilité. Une forte fréquence de test par les instituts ne doit pas
devenir automatiquement une forte probabilité de candidature.

Commencer par les personnes déjà nommées dans les sources disponibles. Un candidat
connu mais non sondé nécessite un prior de soutien déclaré, avec une forte incertitude,
et une évaluation séparée de son apport. Il ne reçoit jamais sa popularité future ni
les voix du candidat finalement investi. Une éventuelle catégorie d'entrant encore
inconnu demandera une autre étude : elle ne permet pas d'attribuer après coup une
probabilité nominative à un vainqueur que le modèle ne connaissait pas.

Les probabilités et contraintes seront inspectées avant lecture des scores. Les durées
365/60 jours de l'étude précédente ne deviennent pas des paramètres à optimiser sur
les mêmes trois élections.

## 3. Localiser les erreurs avant de promouvoir une variante

Conserver le score global de victoire et le même backtest chronologique. Ajouter des
diagnostics séparés pour la présence, la qualification conditionnelle à la présence,
puis la victoire conditionnelle à la qualification. Ces diagnostics permettent de
distinguer une omission de candidature d'une mauvaise estimation du soutien ou des reports.

Un score de Brier sur la présence peut compléter les log-scores et le CRPS. Fixer un
univers de candidats commun aux variantes pour chaque comparaison supplémentaire,
afin que l'ajout de noms ne fasse pas baisser artificiellement une moyenne. Les règles
de score propres encouragent la déclaration de probabilités sincères ; leur choix doit
précéder la comparaison. [Gneiting et Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

Prévoir des ablations limitées, annoncées avant calcul : calendrier seul, recomposition
des scénarios, puis traitement des candidats connus mais non sondés. Chaque étape
doit exposer son effet propre. Répéter les graines prévues au protocole pour distinguer
les petits écarts du bruit Monte-Carlo, en gardant les élections comme unités de
validation. La référence et ses tableaux publiés ne sont pas réécrits.

Le principal critère reste un gain qui ne repose pas sur un seul scrutin, sans perte
de couverture ni dégradation aux autres horizons, conformément à
[la règle d'admission](07-evaluation.md). Une amélioration de la présence seule ne
suffit pas à promouvoir la chaîne complète.

## 4. Préparer une validation prospective et une reprise durable

Les nombreux essais sur 2012, 2017 et 2022 ne fournissent plus un test indépendant de
la conception. Réutiliser un jeu de test pour choisir de nouvelles hypothèses peut
conduire à s'y surajuster. [Dwork et al., 2015](https://proceedings.neurips.cc/paper_files/paper/2015/file/bad5f33780c42f2588878a9d07405083-Paper.pdf).

Pour 2027, proposer un protocole prospectif à horizons fixes : archiver les probabilités,
les données disponibles, le code et les hypothèses avant les événements. Conserver une
version de comparaison stable en parallèle des versions expérimentales. L'évaluation
des candidatures et des primaires donnera des diagnostics intermédiaires ; elle ne
transformera pas une seule présidentielle en plusieurs élections indépendantes.

Une amélioration opérationnelle immédiate serait de conserver une archive vérifiable
du gel et des tirages complets hors du poste de travail, avec leurs empreintes et leurs
conditions d'accès. Les bilans compacts de Git ne suffisent pas à reprendre les calculs :
les dossiers `runs/` contiennent encore des entrées nécessaires. Ajouter ensuite un
contrôle automatisé des tests CPU et de l'intégrité des manifests sur les PR.

## Ordre recommandé

| Priorité | Lot | Dépendance et décision de passage |
|---|---|---|
| 1 | Registre historique daté et sourcé | Audit des douze dates ; publication des lacunes restantes |
| 2 | Scénarios par décision de candidature | Registre relu ; poids et contraintes déclarés avant scores |
| 3 | Diagnostics et ablations de la chaîne | Comparaison chronologique inchangée ; promotion uniquement selon les critères annoncés |
| En parallèle | Archive durable et protocole prospectif 2027 | Peut commencer avec la référence actuelle ; aucune nouvelle variable requise |

Premier lot recommandé : le registre historique et son audit. Il doit notamment
permettre d'expliquer, pour chaque date, pourquoi un candidat est possible, exclu ou
encore indéterminé. Ce résultat sera utile même si la prochaine variante ne gagne pas
au backtest.
