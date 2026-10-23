# Registres

YAML versionnés, validés par `vvs.registry` (pydantic, `load_registry()`). Conventions :

- `candidate_id` : `nom_prenom` en minuscules ASCII sans accents, tirets remplacés par `_`
  (`le_pen_marine`, `melenchon_jean_luc`, `macron_emmanuel`, `dupont_aignan_nicolas`).
  La particule fait partie du nom (`de_villiers_philippe`, `de_villepin_dominique`) ; le
  loader vérifie que l'id vaut `candidate_id_for(last_name, first_name)`.
- `pollster` : id court en minuscules (`ifop`, `elabe`, `harris`, `opinionway`, `ipsos`,
  `odoxa`, `cluster17`, `verian`, `bva`, `csa`, `lh2`, `sofres`, `kantar`, `viavoice`).
- `bloc` : `far_right`, `right`, `centre`, `left`, `far_left`, `other`
  (voir `docs/03-donnees.md`).

Fichiers :

| Fichier | Contenu |
|---|---|
| `candidates.yaml` | personnes, partis successifs avec années |
| `parties.yaml` | partis et nuances, bloc par défaut, successeurs (UMP → LR, FN → RN…) |
| `blocs.yaml` | définition des six blocs et cas limites tranchés, avec justification |
| `elections.yaml` | dates des tours, candidats officiels (parti, bloc), résultats nationaux du Conseil constitutionnel |
| `pollsters.yaml` | instituts, alias de libellés, mode de collecte |
| `scenarios_2027.yaml` | dimensions de candidatures 2027, petits candidats, contraintes, révisions datées (`vvs.registry.scenarios`) |
| `historical_scenarios_2026-10-22.yaml` | protocole expérimental de reconstruction historique, séparé de la référence figée ; voir `docs/11-scenarios-historiques.md` |
| `transfers.yaml` | matrices de reports de voix des seconds tours 2002-2022 par source (`vvs.registry.transfers`) |
| `outlets.yaml` | panel de titres de presse nationale, domaines, présence dans la collection Media Cloud (`vvs.ingest.press.outlets`) |
| `press_queries.yaml` | requêtes presse par candidat (alias, exclusions, formes rejetées) et par thème (mots-clés, thèmes GKG, rubriques INA) |
| `attitude_series.yaml` | séries longues d'attitudes et d'opinion (image du RN, confiance politique Cevipof, ILT CNCDH, Eurobaromètre, CRÉDOC, DREES) : documents, formulations, extractions (`vvs.ingest.attitudes`) |

Validations croisées (`vvs/registry/loader.py`) : candidats et partis d'`elections.yaml`
présents dans les autres fichiers, parti actif l'année de l'élection, cas limites de
`blocs.yaml` appliqués dans `elections.yaml`, somme des voix = exprimés, pourcentages
cohérents avec voix / exprimés et sommant à 100 à 0,2 près, alias d'instituts non ambigus.
