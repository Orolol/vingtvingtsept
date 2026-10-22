# 09 — Feuille de route

| Phase | Contenu | Livrable | État |
|---|---|---|---|
| 0 | Dossier, git, specs | `docs/` | fait le 2026-10-20 |
| 1 | Données : Wikipédia 2002 à 2027, Commission, MieuxVoter, registres, résultats officiels, macro, Jennings–Wlezien, reports historiques | `data/curated/*.parquet`, `data/registry/*.yaml` | ingestions et curation implémentées ; populations Ifop/Harris et seconde base OpinionWay corrigées le 2026-10-22 ; limites restantes dans `03-donnees.md` |
| 2 | Agrégateur premier tour sur 2022 (sans scénarios), effets maison, figures | courbes 2022 cohérentes avec le résultat | fait le 2026-10-20 |
| 3 | Modèle d'erreur et backtest 2002-2022 aux quatre horizons ; baselines ; second tour par reports évalué | `docs/backtest-log.md` | fait le 2026-10-20 |
| 4 | Scénarios 2027, substitution intra-bloc, second tour par reports, simulation ; première projection 2027 | `runs/2026-10-20-simulate-subst/`, site local | fait le 2026-10-20 |
| 5 | Front SvelteKit, export JSON, publication | site en ligne | fait ; URL dans le README |
| 6 | Étude économie, admission ou non, section du site | `runs/<date>/economy/`, mise à jour de `05` | étude faite le 2026-10-20, résultat négatif ; section du site faite (séries, corrélations, note) |
| 7 | Ingestion presse, attention en ligne, classification de titres, étude, section du site | études et volet explicatif | fait ; presse par comptages, classification et attention non admises ; ton des 28 derniers jours déclaré hors échantillon pour 2027 |
| 8 | Live : automatisation des runs, mise à jour à chaque sondage, historique des probabilités | `scripts/run_live.sh`, `.github/workflows/pages.yml` | chaîne et publication implémentées ; installation de la planification locale à vérifier sur la machine |
| 9 | Législatives (seconde étape, hors périmètre actuel) | nouvelle spec | plus tard |
| 10 | Référence figée, backtest chronologique de la chaîne jusqu'à la victoire, sensibilité aux scénarios | `scripts/run_consolidation.sh`, `10-consolidation.md` | fait le 2026-10-22 : douze cas, diagnostics conformes, 47 variantes ; tag `reference-2026-10-22`, résultats et limites dans `10-consolidation.md` |
| 11 | Reconstruction des scénarios historiques incomplets | `scripts/run_historical_scenarios.sh`, `11-scenarios-historiques.md` | fait le 2026-10-22 : mémoire annuelle et filtre événementiel testés sur le même gel ; support élargi, variantes non admises |

Les phases 6 et 7 sont indépendantes l'une de l'autre et de la 5 ; elles peuvent démarrer
dès que la 3 fournit un backtest de référence.

## Vague de variables testées le 2026-10-20

Substitution intra-bloc, rebonds, palmarès des instituts, potentiel électoral, sûrs de leur
choix, attention en ligne : aucune admise (voir `backtest-log.md`). La substitution est
néanmoins activée dans la projection publiée, avec ce statut explicite ; les autres
restent désactivées et servent, le cas échéant, au volet explicatif.

## Questions ouvertes
- Scénarios historiques : la mémoire des listes sondées ne suffit pas (étude `11`).
  Construire des états de candidature datés et des règles de compatibilité indépendantes
  des listes complètes ; couvrir aussi les candidats jamais sondés et les candidatures
  finalement absentes, avant une nouvelle comparaison annoncée.
- Curation : maintenir les audits de population et de doublons ; corrections Ifop, Harris
  et OpinionWay terminées (voir `03-donnees.md`). Retrouver les dates de publication
  historiques pour remplacer le délai imputé du backtest chronologique.
- Cevipof : 47 demandes d'accès déposées le 2026-10-22 ; dès l'accès, `vvs ingest cevipof`
  puis `vvs parse cevipof` et l'étude des duels par électorat (ENEF 2017 V13, ENEF 2022).
- Participation du second tour : elle était implicite, déduite des reports, et mal calibrée
  (78,6 % à la veille de 2017 contre 66 % observés ; ma note précédente qui parlait d'une
  confusion de tours était fausse). Option `second_round_turnout` codée, non admise au
  backtest (CRPS +0,10 à 180 j) : désactivée.
- Biais RN du second tour : courbe par horizon estimée et **retenue pour la publication**
  (remplace l'atténuation linéaire, CRPS −0,23 à 180 j) ; à réévaluer avant 30 jours du vote.
- Couplage premier tour / duel : codé, ρ non identifiable (signe changeant par pli), désactivé.

- Page Wikipédia 2012 : le tableau « second trimestre 2011 » est corrompu (colonne
  Chevènement contenant les scores de Hulot, sondages perdus). Écarté pour l'instant ; la
  révision 79265234 de mai 2012 est saine et pourrait être parsée à la place pour ce tableau.
- Villepin 2027 : classement en `other` à réexaminer dans une étude de sensibilité aux blocs.
- Jennings–Wlezien s'arrête en 2017 : 2022 doit venir de nos propres tables d'erreurs.

- Frontières des blocs : mesurer la sensibilité aux classifications historiques.
- Dates du scrutin 2027 (18 avril et 2 mai) : à confirmer à la publication du décret.
- MieuxVoter et Jennings–Wlezien sont ingérés ; préserver leurs limites de couverture
  dans les audits chronologiques.
- Comparateurs : modèle France 2022 de *The Economist*, modèle bayésien de Martin Devaux.
- « Surestimation du RN en 2024 » dans `04-modele.md` : affirmation à vérifier sur les
  données, aucune étude primaire trouvée.

Voir `notes/veille-2026-10-20.md` pour l'état de la course, Timestral, les archives de presse,
les reports de voix historiques et la littérature.
