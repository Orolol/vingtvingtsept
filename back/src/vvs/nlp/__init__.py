"""Classification des titres de presse par un modèle local (docs/06-presse.md, « NLP »).

- `prompts` : prompts versionnés (data/registry/nlp_prompts.yaml), schéma JSON de sortie,
  validation stricte des réponses ;
- `client` : client de l'API OpenAI-compatible d'un serveur local (llama.cpp
  `llama-server`), requêtes concurrentes, température 0, sortie contrainte par schéma ;
- `server` : commande de lancement de llama-server et garde-fous GPU ;
- `labels` : cache parquet des étiquettes, clé (story_id, candidate_id, prompt_version,
  model), jamais de reclassification ;
- `features` : échantillon plafonné par candidat et semaine, variables hebdomadaires
  (`press_features.parquet`).
"""
