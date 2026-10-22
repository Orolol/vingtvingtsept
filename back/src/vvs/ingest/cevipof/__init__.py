"""Enquêtes électorales du Cevipof (data.sciencespo.fr, Dataverse du CDSP).

Panels et enquêtes post-électorales : Panel électoral français 2002 et 2007, French Electoral
Study et enquête post-électorale 2012, Enquête électorale française (ENEF) 2017, 2022 et
2024. Inventaire versionné dans data/registry/cevipof_datasets.yaml ; recodage des modalités
vers nos candidate_id dans data/registry/aliases_cevipof.yaml.

- `dataverse` : client de l'API native (jeton `DATAVERSE_API_TOKEN`, jamais écrit ni affiché) ;
- `registry` : chargement et validation de l'inventaire ;
- `ingest` : téléchargement immuable vers data/raw/cevipof/<dataset>/ (sha256), demandes
  d'accès aux fichiers restreints ;
- `codebook` : lecture des codebooks DDI publics (variables, libellés, modalités) ;
- `parse` : microdonnées -> data/curated/cevipof_votes.parquet ;
- `transfers` : matrices de reports pondérées et intervalles bootstrap.
"""
