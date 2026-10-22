# 10 — Référence figée et consolidation chronologique

Protocole fixé le 22 octobre 2026 avant lecture des nouveaux scores. Configuration :
`data/registry/reference_2026-10-22.yaml`. Aucune option ne sera sélectionnée sur ces scores.

## Référence

La référence conserve les choix de la projection publiée du 22 octobre : substitution,
calibration de l'erreur finale élargie, biais du second tour par horizon ; autres variantes
désactivées. C'est une référence de comparaison, pas une nouvelle admission statistique.
La substitution et la courbe de biais gardent leur statut de choix publiés sans admission
stricte. Les paramètres des priors et de l'inférence sont explicites.

Un gel conserve le code Python, le verrou uv, les registres, les tables utilisées, les
tirages de la projection publiée et leurs empreintes SHA-256. Les calculs s'exécutent sur
cette copie avec VVS_ROOT et PYTHONPATH, indépendamment du répertoire de travail courant.
Une reprise vérifie les empreintes et ne rejoue que les étapes absentes.

## Évaluation chronologique

Élections : 2012, 2017, 2022 ; horizons : 180, 90, 30, 7 jours. 2002 et 2007 fournissent
l'amorçage : deux présidentielles antérieures sont requises pour estimer la participation.
Tous les paramètres appris (erreur, forme internationale, reports, participation) ne voient
que les scrutins terminés avant la date de prévision. Les scrutins de liste postérieurs sont
exclus, ainsi que les enquêtes de reports dont le terrain est postérieur à cette date.

Un sondage entre après sa publication ; si cette date manque, disponibilité = fin du
terrain + 2 jours. L'audit rapporte les effectifs sous des délais 0, 2, 4 et 7 jours.
Les dates manquantes sont imputées, pas présentées comme retrouvées. Les bases historiques
et les classifications politiques restent reconstruites aujourd'hui : ce protocole teste
une calibration chronologique, pas une sélection du modèle historiquement aveugle, ni les
versions exactes des bases disponibles à chaque époque.

Avant la liste officielle, les scénarios sont les configurations effectivement testées
dans les 30 derniers jours disponibles (élargissement jusqu'à 180 jours en cas de vide).
Chaque sondage pèse un, partagé entre ses hypothèses ; les configurations identiques sont
regroupées. Ces fréquences sont une approximation de l'offre de candidats, pas des
probabilités éditoriales reconstituées. Après publication, la liste officielle est utilisée.
Les noms des vainqueurs de primaires et des finalistes ne sont jamais injectés après coup.
Les candidats absents des scénarios gardent une probabilité nulle, même s'ils ont gagné.

La simulation tire le scénario, le premier tour et son erreur, les deux qualifiés, puis le
second tour propre à chaque duel. Les sondages de duel sont conditionnels : ils ne modifient
pas le poids préalable des duels. Scores : log-score du vainqueur et de la paire, Brier de
victoire (somme multicatégorie), Brier de qualification, CRPS et couverture du premier tour.
Le CRPS inclut tous les candidats officiels, avec zéro pour ceux non prévus. Le Brier inclut
aussi les candidats envisagés mais finalement absents. Aucun report ex post des voix.

Les fréquences brutes et leurs zéros sont publiés. Pour le log-score, lissage de Jeffreys
sur l'univers de score (union des candidats prévus et officiels), explicitement séparé de
la probabilité publiée. Une moyenne par horizon donne le même poids aux trois élections.
Le diagnostic du second tour sur le duel réalisé reste séparé du score de victoire global.

Le témoin `mean14d` remplace seulement l'agrégateur bayésien par la moyenne des sondages
disponibles des 14 derniers jours (une mesure par sondage et candidat). Si la fenêtre est
vide, il prend la dernière date disponible ; un candidat absent de cette fenêtre prend
sa médiane passée, ou 1 % en l'absence de toute observation. Il conserve les mêmes scénarios,
l'erreur électorale et le modèle du second tour. La comparaison isole donc l'apport de
l'agrégation, sans constituer une ablation de chaque composant.

## Sensibilité aux scénarios 2027

Même modèle et mêmes tirages ; seuls changent les poids des six dimensions : ±10 points
pour un état (autres états renormalisés), états certains, puis poids uniformes. Les
contraintes sont réappliquées avant normalisation. Les petits candidats conservent leur loi.
Les configurations sont appariées par leurs états, jamais par leur rang après tri.

Les tirages sont repondérés par configuration, avec correction des effectifs entiers du
tirage stratifié. On rapporte qualification, victoire, présence, écarts à la référence,
taille effective et masse des configurations sans tirage. Cette masse donne des bornes
explicites sur les probabilités ; elle n'est jamais supprimée silencieusement. Une faible
taille effective est signalée. Les états certains sont des conditions, pas des prévisions
de candidature ni des effets causaux d'un retrait.

## Résultats

À renseigner après l'exécution ; les échecs de convergence et les configurations non
représentées seront conservés dans le bilan.
