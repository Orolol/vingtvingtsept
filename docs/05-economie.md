# 05 — Économie et popularité

Question posée : des facteurs économiques ont-ils joué, dans le passé, sur l'évolution des
intentions de vote ? Par exemple, un chômage qui monte a-t-il profité à la droite ou à
l'extrême droite ? Et peut-on s'en servir pour prévoir ?

## Ce que dit la littérature

- Evans et Ivaldi (2008, *French Politics*) modélisent le vote FN de 1984 à 2007 avec
  chômage, criminalité et immigration ; le modèle tient en 2012. Effet de niveau sur le bloc,
  estimé sur des élections, pas sur la dynamique mensuelle.
- Nadeau, Lewis-Beck et Bélanger : modèles politico-économiques français où la popularité
  du sortant et la conjoncture prédisent le vote pour le camp au pouvoir.
- Conclusion attendue : des effets modestes et lents, au niveau des blocs, qui ne capturent
  pas la volatilité des candidats. On teste quand même, proprement, et on publie le résultat
  quel qu'il soit.

## Séries candidates

| Série | Source | Fréquence | Depuis |
|---|---|---|---|
| Demandeurs d'emploi catégorie A | DARES | mensuel | 1996 |
| Taux de chômage BIT | INSEE | trimestriel | 1975 |
| Inflation IPC, et IPC énergie / alimentation | INSEE | mensuel | 1990 |
| Confiance des ménages, et sous-questions niveau de vie futur, chômage anticipé | INSEE | mensuel | 1987 |
| Prix du carburant à la pompe | ministère de la Transition écologique | hebdomadaire | 2007 |
| Climat des affaires | INSEE | mensuel | 1976 |
| Cote de popularité du président et du Premier ministre | Ifop, et autres baromètres | mensuel | 1958 |
| Préoccupations des Français (enquêtes « premier problème ») | Ipsos, Elabe, Kantar | irrégulier | variable |

Ingestion : API INSEE (BDM), fichiers DARES, séries Ifop reconstituées depuis leurs
publications et Wikipédia. Stockées en `data/curated/macro.parquet` (une ligne par
série et par mois).

## Protocole d'étude (`back/studies/economy.py`)

1. **Panel mensuel 2002 à 2026** : part de chaque bloc dans l'agrégat des sondages (sortie
   du modèle `04`, ou moyenne glissante pondérée à défaut), contre les séries macro, en
   variations et avec retards de 0 à 6 mois.
2. **Spécification** : régression en panel par bloc des variations de part sur les
   variations des indicateurs, effets fixes par élection, erreurs standard robustes. Test de
   Granger dans les deux sens, parce que la causalité inverse (les sondages font la une et
   pèsent sur la confiance) est plausible.
3. **Hors campagne** : la série de sondages est creuse avant 2017 hors des six mois
   précédant le vote. On documente la couverture et on teste la robustesse en se limitant aux
   campagnes.
4. **Niveau contre dynamique** : on sépare l'effet sur le niveau d'un bloc en début de
   campagne (prior structurel, estimable sur 6 élections) de l'effet sur la trajectoire
   pendant la campagne (estimable sur le panel mensuel).
5. **Médiation par la presse** : on refait l'analyse en ajoutant la saillance médiatique des
   thèmes économiques (`06`). Hypothèse : le chômage joue quand on en parle.

## Intégration dans la prévision, si admission

- Niveau : terme dans le prior structurel de `β_b` au début de la campagne.
- Dynamique : dérive `μ_b(τ) = γ_b · x_t` dans la projection, où `x_t` sont les variations
  récentes des indicateurs. Prévoir les indicateurs eux-mêmes jusqu'au vote est une autre
  incertitude : on la traite avec **Timestral** (Mistral, modèles de fondation de séries
  temporelles sortis le 15 octobre 2026, 5M à 209M de paramètres, prévisions probabilistes
  à 21 quantiles, covariables passées et futures, CPU suffisant, poids sur Hugging Face
  sous licence MIT modifiée), choix de Gaétan. Les quantiles de Timestral sur chaque série
  deviennent la distribution de `x_t` jusqu'au vote. Évaluation d'usage détaillée dans
  `docs/notes/veille-2026-10-20.md`.
- Admission selon `07-evaluation.md` : gain de log-score aux horizons 180 et 90 jours, en
  laissant chaque élection de côté. Si le gain est nul ou instable, la variable reste dans le
  tableau de bord explicatif du site et le résultat négatif est consigné ici.

## Livrables

- `runs/<date>/economy/` : tableaux de coefficients, figures par bloc, note de résultats.
- Section « Économie » du site : séries et corrélations, avec la mention claire de ce qui
  entre ou non dans la prévision.
