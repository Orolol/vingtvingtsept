# Documentation — index

Specs du projet, à lire dans l'ordre la première fois. Chaque document est la référence de
son sujet : quand le code et la doc divergent, on corrige l'un ou l'autre, pas les deux en
silence.

| Fichier | Sujet |
|---|---|
| [01-vision.md](01-vision.md) | Objectif, ce qu'on peut raisonnablement prédire, principes |
| [02-architecture.md](02-architecture.md) | Découpage back / front / données, arborescence, flux |
| [03-donnees.md](03-donnees.md) | Sources et schémas : sondages, résultats, registre candidats/blocs |
| [04-modele.md](04-modele.md) | Modèle statistique : agrégation, erreur, scénarios, second tour, simulation |
| [05-economie.md](05-economie.md) | Indicateurs économiques et politiques : étude puis intégration conditionnelle |
| [06-presse.md](06-presse.md) | Titres de presse : sources, NLP, variables dérivées |
| [07-evaluation.md](07-evaluation.md) | Backtest, métriques, règle d'admission d'une variable |
| [08-front.md](08-front.md) | Site statique SvelteKit, pages, contrat JSON |
| [09-roadmap.md](09-roadmap.md) | Phases et état d'avancement |
| [10-consolidation.md](10-consolidation.md) | Référence figée, chaîne chronologique jusqu'à la victoire, sensibilité aux scénarios |
| [11-scenarios-historiques.md](11-scenarios-historiques.md) | Reconstruction avec mémoire et événements datés : comparaison à la référence, variantes non admises |
| [12-prochaines-etapes.md](12-prochaines-etapes.md) | Suite proposée : registre daté, scénarios par décision de candidature, diagnostics et validation prospective |
| [13-experiences-structure.md](13-experiences-structure.md) | Mélange, incertitude de calibration, sondages continus, substitution entre blocs et transitions des indécis : protocoles et résultats |

Décisions prises le 2026-10-20 (brainstorm initial) :

- Présidentielle seule pour commencer ; les législatives viendront après, sur la même base.
- Python pour le back, SvelteKit statique pour le front, site partageable.
- Sondages historiques depuis Wikipédia (listes 2002 à 2027), notices de la Commission des
  sondages pour 2027, NSPPolls en contrôle croisé pour 2017 et 2022.
- Presse : Media Cloud en colonne vertébrale, GDELT en complément. Économie : INSEE, DARES,
  cote de popularité Ifop.
