# 09 — Feuille de route

| Phase | Contenu | Livrable | État |
|---|---|---|---|
| 0 | Dossier, git, specs | `docs/` | fait le 2026-10-20 |
| 1 | Données : Wikipédia 2002 à 2027 (978 sondages), notices de la Commission (2027), MieuxVoter, registre candidats/blocs/élections/instituts, résultats du ministère de l'Intérieur (présidentielles, européennes, législatives), séries macro et popularité, base Jennings–Wlezien, scénarios 2027, reports de voix 2002-2022 | `data/curated/*.parquet`, `data/registry/*.yaml`, 283 tests | ingestions faites le 2026-10-20 ; curation en cours |
| 2 | Agrégateur premier tour sur 2022 (sans scénarios), effets maison, figures | courbes 2022 cohérentes avec le résultat | fait le 2026-10-20 |
| 3 | Modèle d'erreur et backtest 2002-2022 aux quatre horizons ; baselines ; second tour par reports évalué | `docs/backtest-log.md` | fait le 2026-10-20 |
| 4 | Scénarios 2027, substitution intra-bloc, second tour par reports, simulation ; première projection 2027 | `runs/2026-10-20-simulate-subst/`, site local | fait le 2026-10-20 |
| 5 | Front SvelteKit, export JSON, publication | site en ligne | squelette, contrat JSON et export faits ; workflow Pages prêt ; publication à faire (dépôt distant) |
| 6 | Étude économie, admission ou non, section du site | `runs/<date>/economy/`, mise à jour de `05` | étude faite le 2026-10-20, résultat négatif ; section du site faite (séries, corrélations, note) |
| 7 | Ingestion presse (Media Cloud, GDELT, INA), attention en ligne, classification, étude, admission ou non, section du site | `runs/2026-10-20-press/`, `runs/2026-10-20-attention/` | ingestion et étude d'attention faites (non retenue) ; section du site faite (attention Wikipédia, rubriques INA, GDELT) ; classification des titres à faire |
| 8 | Live : automatisation des runs, mise à jour à chaque sondage, historique des probabilités | `scripts/run_live.sh`, `.github/workflows/pages.yml`, tâche planifiée | chaîne live, réutilisation de l'ajustement par empreinte des sondages, résumé de run et workflow Pages faits le 2026-10-20 ; volets explicatifs du site (économie, presse, palmarès, variables non admises) faits ; planification à installer (`scripts/README.md`), dépôt distant et publication à faire |
| 9 | Législatives (seconde étape, hors périmètre actuel) | nouvelle spec | plus tard |

Les phases 6 et 7 sont indépendantes l'une de l'autre et de la 5 ; elles peuvent démarrer
dès que la 3 fournit un backtest de référence.

## Vague de variables testées le 2026-10-20

Substitution intra-bloc, rebonds, palmarès des instituts, potentiel électoral, sûrs de leur
choix, attention en ligne : aucune admise (voir `backtest-log.md`). Toutes restent des
options désactivées dans le code et des volets explicatifs possibles du site.

## Questions ouvertes
- Curation 2027 : population Ifop et Harris à corriger (certains d'aller voter), seconde base
  OpinionWay à conserver (déduplication sur tour et candidats trop large).
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
- Villepin 2027 : rangé en `other` à titre provisoire, à trancher avant la phase 4.
- Jennings–Wlezien s'arrête en 2017 : 2022 doit venir de nos propres tables d'erreurs.

- Frontières des blocs (écologistes, Mélenchon avant 2012) : à trancher en phase 1.
- Dates du scrutin 2027 (18 avril et 2 mai) : à confirmer à la publication du décret.
- Successeur de NSPPolls : `MieuxVoter/presidentielle2027` (même format, MIT) ; à ingérer en
  contrôle croisé de la Commission et de Wikipédia.
- Base Jennings et Wlezien (erreurs de sondages, France depuis 1965) : à ingérer pour la
  calibration du modèle d'erreur.
- Comparateurs : modèle France 2022 de *The Economist*, modèle bayésien de Martin Devaux.
- « Surestimation du RN en 2024 » dans `04-modele.md` : affirmation à vérifier sur les
  données, aucune étude primaire trouvée.

Voir `notes/veille-2026-10-20.md` pour l'état de la course, Timestral, les archives de presse,
les reports de voix historiques et la littérature.
