# Registres

YAML versionnés, validés par `vvs.registry` (pydantic). Conventions :

- `candidate_id` : `nom_prenom` en minuscules ASCII sans accents, tirets remplacés par `_`
  (`le_pen_marine`, `melenchon_jean_luc`, `macron_emmanuel`, `dupont_aignan_nicolas`).
- `pollster` : id court en minuscules (`ifop`, `elabe`, `harris`, `opinionway`, `ipsos`,
  `odoxa`, `cluster17`, `verian`, `bva`, `csa`, `lh2`, `sofres`, `kantar`, `viavoice`).
- `bloc` : `far_right`, `right`, `centre`, `left`, `far_left`, `other`
  (voir `docs/03-donnees.md`).
