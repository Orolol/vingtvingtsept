# 04 — Modèle

Vue d'ensemble : un modèle d'agrégation des sondages par élection, un modèle d'erreur
calibré sur les élections passées, une couche de scénarios de candidats pour 2027, un
modèle de second tour par reports, et une simulation qui assemble le tout.

Notation : élection `e`, candidat `c`, bloc `b(c)`, institut `p`, temps `t` en jours avant
le premier tour (`t = 0` le jour du vote), hypothèse de configuration `h` (sous-ensemble de
candidats), horizon `τ` (jours entre la date de prévision et le vote).

## 1. Agrégation des sondages (premier tour)

### Soutien latent

Chaque candidat a un soutien latent sur l'échelle logit multinomiale, décomposé en une
composante de bloc et un écart propre :

```
θ_{c,t} = β_{b(c),t} + δ_{c,t}
β_{b,t} = β_{b,t+1} + ε^β_{b,t},   ε^β ~ N(0, σ²_β)        (marche aléatoire du bloc)
δ_{c,t} = δ_{c,t+1} + ε^δ_{c,t},   ε^δ ~ N(0, σ²_δ)        (marche aléatoire du candidat)
```

Pas de temps : la semaine, avec interpolation si nécessaire. `σ_β` et `σ_δ` sont estimés,
avec un prior qui autorise plus de volatilité loin de l'élection (voir modèle d'erreur).

Pourquoi deux niveaux : les hypothèses de candidats rares ont peu de sondages, mais le bloc
est mesuré par toutes. Le candidat emprunte sa dynamique au bloc, et n'en dévie qu'avec
des données propres.

### Observation d'un sondage

Pour un sondage `i` de l'institut `p`, à la date `t_i`, testant l'hypothèse `h_i`, on observe
le vecteur des parts `y_i` sur les candidats de `h_i` :

```
π_{i,c} = softmax_{c ∈ h_i}( θ_{c,t_i} + α_{p,c} + γ_{pop(i)} )
y_i · n_i ~ Multinomial(n_i, π_i)       avec n_i = taille d'échantillon effective
```

- `α_{p,c}` : effet maison de l'institut sur le candidat, prior hiérarchique centré sur 0,
  somme contrainte à 0 par candidat. On commence par un effet maison par (institut, bloc),
  plus un terme candidat plus faible.
- `γ` : effet de la population interrogée (inscrits vs certains d'aller voter).
- `n_i` : effectif réduit par un facteur de design (prior autour de 0,5 à 0,7) pour refléter
  les quotas et le redressement. Estimé.
- Le softmax restreint à `h_i` suppose l'indépendance des alternatives non pertinentes :
  retirer un candidat redistribue ses voix proportionnellement. C'est faux (les voix de
  Zemmour ne vont pas à Mélenchon). **Correction prévue en phase 4** : un terme de
  substitution par bloc, où le soutien d'un bloc se redistribue d'abord dans le bloc.
  On mesure sur les sondages testant plusieurs hypothèses le même jour.

### Prior structurel au début de la série

À la première date observée, `β_{b}` a pour prior la part du bloc aux dernières élections
nationales (présidentielle précédente, européennes et législatives intermédiaires, pondérées
par récence), avec une variance large. Ce prior pèse peu dès que les sondages s'accumulent ;
c'est l'équivalent des « fundamentals » de Silver, avec un rôle volontairement modeste.

## 2. Modèle d'erreur et projection au jour du vote

À la date de prévision `t = τ`, le postérieur donne `θ_{c,τ}`. Le vote se passe en `t = 0`.
On modélise :

```
θ_{c,0} = θ_{c,τ} + μ_{c}(τ) + η_c(τ)
```

- `μ(τ)` : dérive moyenne, par défaut nulle ; c'est ici que les covariables économiques ou
  presse entreraient (`05`, `06`), après admission par le backtest.
- `η(τ)` : erreur de projection, somme de deux composantes estimées sur les élections
  passées (leave-one-election-out) :
  1. **mouvement entre τ et 0** : variance de la variation du soutien agrégé entre l'horizon
     τ et le dernier sondage, par bloc, croissant en τ ;
  2. **erreur sondage final contre résultat** : variance et corrélation des écarts du
     dernier agrégat au résultat, par bloc. Les écarts sont corrélés dans un bloc (tous les
     candidats de gauche sous-estimés ensemble en 2022) et entre blocs voisins.

La matrice de covariance de `η` est donc structurée par bloc, et l'on autorise un biais
moyen par bloc s'il est stable historiquement (sous-estimation du FN avant 2017,
surestimation du RN en 2024), avec un prior qui le ramène vers 0 : ces biais ne sont pas
stables, et c'est bien le problème.

Avec cinq présidentielles, on complète la calibration avec les législatives et européennes
pour les composantes de bloc, en documentant ce choix.

## 3. Scénarios de candidats (2027)

Tant que les candidatures ne sont pas officielles (parrainages validés début mars 2027), la
prévision est un mélange :

```
P(événement) = Σ_k P(config_k) · P(événement | config_k)
```

- `scenarios_2027.yaml` liste les configurations plausibles avec une probabilité, une date
  de révision et une justification en une phrase. Exemple de dimensions : candidat RN
  (Bardella ou Le Pen), gauche unie ou éclatée, candidat du bloc central, candidat LR, présence
  de petits candidats structurants.
- Les probabilités sont des jugements, versionnés, visibles sur le site. Pas de prétention
  à les estimer statistiquement ; l'historique des révisions est publié.
- `P(événement | config_k)` vient du modèle d'agrégation restreint aux sondages testant
  `config_k` ou des configurations proches, via le niveau bloc.
- Quand une dimension est tranchée (décision de justice, déclaration, primaire), les
  configurations incompatibles passent à 0 et le reste se renormalise.

## 4. Second tour

Pour une paire de qualifiés `(i, j)` tirée en simulation :

```
part_i = Σ_c π_c · T_{c→i}(i, j)
```

- `π_c` : parts de premier tour simulées.
- `T_{c→·}(i, j)` : vecteur de reports de l'électorat du candidat `c` vers `i`, `j` ou
  l'abstention/blanc, spécifique au duel.
- Prior de `T` : Dirichlet centré sur les reports observés dans les enquêtes post-électorales
  (Ipsos, Cevipof, Ifop) des seconds tours passés et sur la proximité de bloc, avec une
  incertitude large sur les cas inédits (duel sans candidat RN, duel RN contre gauche
  radicale).
- Mise à jour par les sondages de second tour hypothétiques, observés avec le même modèle
  d'effets maison, et un effet horizon spécifique : l'erreur historique de ces sondages
  loin du vote est bien plus forte qu'au premier tour.
- La « prime au front républicain » n'est pas un paramètre à part ; elle émerge des
  reports et de leur érosion observée (2002, 2017, 2022).

## 5. Simulation

Pour chaque tirage (10 000 au minimum) :

1. tirer une configuration `k` ;
2. tirer `θ_{·,τ}` dans le postérieur conditionnel ;
3. tirer `η(τ)` et calculer les parts de premier tour `π`, puis l'abstention et les blancs
   (modèle simple : part d'exprimés tirée autour des précédents scrutins) ;
4. déterminer les deux qualifiés ;
5. tirer `T` et calculer le second tour ;
6. enregistrer : parts, qualifiés, vainqueur, configuration.

Sorties agrégées : probabilités de qualification et de victoire (globales et par
configuration), quantiles des parts, matrice des duels (probabilité et score attendu pour
chaque paire), trajectoire des probabilités au fil des runs.

## 6. Ce qui est volontairement simple au début

- Pas d'effet de mode d'enquête au départ ; on l'ajoute si les effets maison le réclament.
- Pas de modélisation des indécis : on travaille en base exprimés. Un terme « sûrs de leur
  choix » pourrait moduler la variance plus tard.
- Une seule échelle temporelle (semaine) et un seul jeu de variances pour toute la
  campagne ; à raffiner si le backtest montre une volatilité différente avant et après la
  clôture des candidatures.
- Dérive `μ` nulle : les covariables arrivent par les études de `05` et `06`.
- Le modèle état-espace est écrit à la main en numpyro. Une alternative à tester en
  phase 3, en comparaison et non en remplacement : prévoir la trajectoire des agrégats par
  bloc avec Timestral (séries temporelles de fondation, covariables admises), évaluée au
  même backtest. Avec cinq élections, l'intérêt attendu est surtout la prolongation des
  séries exogènes (`05`), pas la prévision du vote lui-même.
