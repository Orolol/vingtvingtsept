/* Fichier généré par scripts/gen-types.js depuis back/src/vvs/export/schemas/.
 * Ne pas modifier à la main : npm run types. */

/**
 * Date calendaire ISO 8601 (AAAA-MM-JJ).
 */
export type IsoDate = string;
/**
 * Horodatage ISO 8601 avec fuseau (UTC recommandé).
 */
export type IsoDateTime = string;
/**
 * Clé du registre data/registry/candidates.yaml, ou `placeholder_<bloc>` pour un candidat non identifié connu seulement par son bloc.
 */
export type CandidateId = string;
/**
 * Bloc politique (data/registry/blocs.yaml).
 */
export type Bloc = 'far_right' | 'right' | 'centre' | 'left' | 'far_left' | 'other';
/**
 * Identifiant d'une condition (voir Condition) : `mix` pour le mélange de scénarios.
 */
export type ConditionId = string;
export type Probability = number;
/**
 * Un quantile par semaine, aligné sur `dates`, en points d'exprimés.
 */
export type QuantileTrack = (number | null)[];
/**
 * Clé du registre data/registry/pollsters.yaml.
 */
export type PollsterId = string;
/**
 * Base de l'échantillon publié : inscrits, ou certains d'aller voter.
 */
export type Population = 'registered' | 'certain_voters' | 'unknown';
/**
 * Part en points de pourcentage des suffrages exprimés.
 */
export type Share = number;

/**
 * Fichiers de static/data/ par clé de chargement (voir src/lib/data.ts).
 */
export interface DataFiles {
	manifest: Manifest;
	summary: Summary;
	first_round: FirstRound;
	second_round: SecondRound;
	scenarios: Scenarios;
	polls: Polls;
	economy: Economy;
	press: Press;
	backtest: Backtest;
	history: HistoryIndex;
	history_entry: HistoryEntry;
}
/**
 * Manifeste du run exporté : chargé par toutes les pages (pied de page, date du dernier sondage, registre des candidats affichés).
 */
export interface Manifest {
	/**
	 * Version du contrat JSON ; change à toute rupture de forme.
	 */
	schema_version: '1';
	run_date: IsoDate;
	generated_at: IsoDateTime;
	asof: IsoDate;
	model_version: string;
	/**
	 * Commit git du back ayant produit le run (null si arbre modifié ou inconnu).
	 */
	commit: string | null;
	last_poll_date: IsoDate;
	/**
	 * Nombre de sondages intégrés (premier et second tour).
	 */
	n_polls: number;
	/**
	 * Nombre de tirages Monte-Carlo.
	 */
	n_draws: number;
	/**
	 * Vrai si les données sont factices (squelette du site) : le site affiche alors un bandeau.
	 */
	mock: boolean;
	election: {
		year: number;
		first_round: IsoDate;
		second_round: IsoDate;
	};
	/**
	 * Tous les candidats cités par les autres fichiers (y compris placeholders).
	 */
	candidates: Candidate[];
}
/**
 * Candidat 2027 tel qu'affiché par le site.
 */
export interface Candidate {
	id: CandidateId;
	/**
	 * Nom complet (« Marine Le Pen »).
	 */
	name: string;
	/**
	 * Nom court pour les graphiques (« Le Pen »).
	 */
	short_name: string;
	bloc: Bloc;
	/**
	 * Id du parti actuel (parties.yaml), null si sans parti ou inconnu.
	 */
	party: string | null;
	/**
	 * Vrai pour un candidat non identifié (placeholder de bloc).
	 */
	is_placeholder: boolean;
}
/**
 * Probabilités du jour : mélange de scénarios d'abord (`mixture`), puis détail conditionnel (`by_condition`).
 */
export interface Summary {
	asof: IsoDate;
	n_draws: number;
	/**
	 * Conditions décrites dans ce fichier ; la première est le mélange (`mix`).
	 *
	 * @minItems 1
	 */
	conditions: [Condition, ...Condition[]];
	mixture: OutcomeSet;
	/**
	 * Un résultat par condition autre que `mix` : chaque état de dimension et les configurations les plus probables.
	 */
	by_condition: OutcomeSet[];
}
/**
 * Ce sur quoi une prévision est conditionnée. `mixture` : mélange de toutes les configurations pondérées par leur probabilité (id `mix`, states vide, probability 1). `dimension_state` : un état d'une dimension, les autres dimensions restant mélangées (states à une clé). `configuration` : une configuration complète (un état par dimension ; petits candidats à leur probabilité de présence).
 */
export interface Condition {
	id: ConditionId;
	kind: 'mixture' | 'dimension_state' | 'configuration';
	/**
	 * Libellé court affichable (« Bardella candidat du RN »).
	 */
	label: string;
	states: StateSelection;
	probability: Probability;
}
/**
 * États choisis par dimension de scénario : clé = id de dimension, valeur = id d'état (scenarios_2027.yaml).
 */
export interface StateSelection {
	[k: string]: string | undefined;
}
export interface OutcomeSet {
	condition_id: ConditionId;
	/**
	 * Candidats avec p_present > 0 sous la condition, triés par p_win puis p_qualify décroissants. Somme des p_win = 1, somme des p_qualify = 2.
	 */
	candidates: CandidateOutcome[];
}
export interface CandidateOutcome {
	candidate_id: CandidateId;
	p_present: Probability;
	p_qualify: Probability;
	p_win: Probability;
	first_round_share: Interval;
}
/**
 * Quantiles d'une distribution : médiane (q50), intervalle à 50 % (q25 à q75) et à 90 % (q05 à q95). Les quantiles sont croissants. Unité donnée par le champ qui le porte.
 */
export interface Interval {
	q05: number;
	q25: number;
	q50: number;
	q75: number;
	q95: number;
}
/**
 * Trajectoires hebdomadaires du soutien latent de premier tour par condition et candidat, projection au jour du vote, sondages observés.
 */
export interface FirstRound {
	asof: IsoDate;
	election_date: IsoDate;
	/**
	 * Axe temporel commun : un lundi par semaine, croissant, dernier ≤ asof.
	 *
	 * @minItems 1
	 */
	dates: [IsoDate, ...IsoDate[]];
	/**
	 * Conditions disponibles au filtre ; la première est `mix`.
	 *
	 * @minItems 1
	 */
	conditions: [Condition, ...Condition[]];
	series: TrajectorySeries[];
	polls: FirstRoundPoll[];
}
/**
 * Trajectoire d'un candidat sous une condition. Les tableaux q05 à q95 ont la longueur de `dates` (null avant le premier sondage qui teste le candidat).
 */
export interface TrajectorySeries {
	condition_id: ConditionId;
	candidate_id: CandidateId;
	q05: QuantileTrack;
	q25: QuantileTrack;
	q50: QuantileTrack;
	q75: QuantileTrack;
	q95: QuantileTrack;
	election_day: Interval;
}
/**
 * Une hypothèse de premier tour d'un sondage (un sondage peut en tester plusieurs).
 */
export interface FirstRoundPoll {
	poll_id: string;
	hypothesis_id: string;
	pollster: PollsterId;
	fieldwork_end: IsoDate;
	population: Population;
	/**
	 * Conditions (hors `mix`) compatibles avec la liste de candidats testée ; tout sondage est affiché sous `mix`.
	 */
	conditions: ConditionId[];
	/**
	 * Part observée par candidat (points d'exprimés), telle que publiée.
	 */
	shares: {
		[k: string]: Share | undefined;
	};
}
/**
 * Matrice des duels de second tour (mélange de scénarios), reports de voix estimés et sondages de second tour hypothétiques.
 */
export interface SecondRound {
	asof: IsoDate;
	election_date: IsoDate;
	/**
	 * Paires avec p_pair ≥ seuil d'export, triées par p_pair décroissant ; une paire non ordonnée n'apparaît qu'une fois.
	 */
	pairs: Duel[];
	/**
	 * Reports pour les paires principales et les électorats de premier tour significatifs.
	 */
	transfers: Transfer[];
	polls: SecondRoundPoll[];
}
/**
 * Un duel possible. Le score du second nommé (b) est 100 − score de a : ses quantiles se déduisent par symétrie (q05_b = 100 − q95_a).
 */
export interface Duel {
	candidate_a: CandidateId;
	candidate_b: CandidateId;
	p_pair: Probability;
	p_a_wins: Probability;
	share_a: Interval;
}
/**
 * Report estimé de l'électorat de premier tour de `from` dans le duel (a, b), en points de cet électorat ; to_a + to_b + to_none ≈ 100 en médiane.
 */
export interface Transfer {
	candidate_a: CandidateId;
	candidate_b: CandidateId;
	from: CandidateId;
	to_a: Interval;
	to_b: Interval;
	to_none: Interval;
}
export interface SecondRoundPoll {
	poll_id: string;
	pollster: PollsterId;
	fieldwork_end: IsoDate;
	candidate_a: CandidateId;
	candidate_b: CandidateId;
	share_a: Share;
}
/**
 * Contenu de data/registry/scenarios_2027.yaml (sources résolues), configurations les plus probables et historique des probabilités. La prévision conditionnelle à chaque condition est dans summary.json (`by_condition`).
 */
export interface Scenarios {
	as_of: IsoDate;
	election: number;
	dimensions: ScenarioDimension[];
	small_candidates: SmallCandidate[];
	constraints: ScenarioConstraint[];
	/**
	 * Révisions du registre, de la plus récente à la plus ancienne.
	 */
	revisions: ScenarioRevision[];
	/**
	 * Configurations les plus probables, par probabilité décroissante.
	 */
	configurations: ScenarioConfiguration[];
	/**
	 * Nombre total de configurations possibles après contraintes.
	 */
	n_configurations: number;
	impossible_mass: Probability;
	/**
	 * Probabilités des états à chaque révision, par date croissante.
	 */
	probability_history: ScenarioProbabilityPoint[];
}
export interface ScenarioDimension {
	id: string;
	question: string;
	/**
	 * Échéances de révision (texte libre).
	 */
	review: string[];
	/**
	 * Probabilités sommant à 1.
	 *
	 * @minItems 1
	 */
	states: [ScenarioState, ...ScenarioState[]];
}
export interface ScenarioState {
	id: string;
	probability: Probability;
	candidates: CandidateId[];
	/**
	 * Blocs d'un candidat non identifié (candidate_id `placeholder_<bloc>`).
	 */
	placeholders: Bloc[];
	description: string;
	sources: SourceRef[];
	condition_id: ConditionId;
}
/**
 * Source citée : libellé court et lien (URL, ou chemin dans le dépôt pour une note interne).
 */
export interface SourceRef {
	label: string;
	href: string;
}
export interface SmallCandidate {
	candidate_id: CandidateId;
	probability: Probability;
	bloc: Bloc;
	rationale: string;
	sources: SourceRef[];
}
export interface ScenarioConstraint {
	id: string;
	when: StateSelection;
	absent: CandidateId[];
	impossible: boolean;
	rationale: string;
}
export interface ScenarioRevision {
	date: IsoDate;
	author: string;
	summary: string;
	rationale: string;
}
/**
 * Configuration de dimensions (petits candidats à leur probabilité de présence conditionnelle).
 */
export interface ScenarioConfiguration {
	condition_id: ConditionId;
	states: StateSelection;
	probability: Probability;
	candidates: CandidateId[];
	placeholders: Bloc[];
	/**
	 * Probabilité de présence de chaque petit candidat dans cette configuration.
	 */
	small_presence: {
		[k: string]: Probability | undefined;
	};
}
/**
 * Probabilités des états à une révision : clé `<dimension>.<état>`.
 */
export interface ScenarioProbabilityPoint {
	date: IsoDate;
	probabilities: {
		[k: string]: Probability | undefined;
	};
}
/**
 * Tous les sondages intégrés au run (premier et second tour) et effets maison estimés par institut.
 */
export interface Polls {
	asof: IsoDate;
	pollsters: Pollster[];
	/**
	 * Par fin de terrain décroissante.
	 */
	polls: Poll[];
	house_effects: HouseEffect[];
}
export interface Pollster {
	id: PollsterId;
	name: string;
}
export interface Poll {
	/**
	 * Clé de la table curated `polls`.
	 */
	poll_id: string;
	pollster: PollsterId;
	/**
	 * Commanditaire(s).
	 */
	sponsor: string | null;
	fieldwork_start: IsoDate;
	fieldwork_end: IsoDate;
	publish_date: IsoDate | null;
	sample_size: number | null;
	mode: 'online' | 'phone' | 'mixed' | 'unknown';
	/**
	 * Notice de la Commission des sondages, ou page source.
	 */
	source_url: string | null;
	/**
	 * @minItems 1
	 */
	hypotheses: [PollHypothesis, ...PollHypothesis[]];
}
/**
 * Une hypothèse testée : liste de candidats (premier tour) ou duel (second tour), une population.
 */
export interface PollHypothesis {
	hypothesis_id: string;
	round: 1 | 2;
	population: Population;
	/**
	 * Parts publiées en points d'exprimés, dans l'ordre décroissant.
	 *
	 * @minItems 2
	 */
	shares: [
		{
			candidate_id: CandidateId;
			share: Share;
		},
		{
			candidate_id: CandidateId;
			share: Share;
		},
		...{
			candidate_id: CandidateId;
			share: Share;
		}[]
	];
}
/**
 * Effet maison estimé de l'institut sur un bloc, en points d'exprimés (positif = l'institut surestime le bloc par rapport au consensus).
 */
export interface HouseEffect {
	pollster: PollsterId;
	bloc: Bloc;
	effect: Interval;
	n_polls: number;
}
/**
 * Volet explicatif économie et popularité (docs/05-economie.md). N'entre dans la prévision que si `in_forecast` est vrai, après admission au backtest.
 */
export interface Economy {
	asof: IsoDate;
	/**
	 * Vrai si au moins une variable économique est admise dans la prévision.
	 */
	in_forecast: boolean;
	/**
	 * Phrase affichée en tête de page.
	 */
	in_forecast_note: string;
	series: EconomySeries[];
	correlations: EconomyCorrelation[];
}
export interface EconomySeries {
	/**
	 * Id de macro_series.yaml.
	 */
	id: string;
	label: string;
	/**
	 * Unité affichée (« % », « points », « indice 100 = 2015 »).
	 */
	unit: string;
	frequency: 'monthly' | 'quarterly' | 'weekly';
	source: SourceRef;
	/**
	 * Vrai si cette série entre dans la prévision.
	 */
	in_forecast: boolean;
	points: {
		date: IsoDate;
		value: number;
	}[];
}
/**
 * Résultat d'étude : corrélation d'une série avec une cible (part d'un bloc, popularité) à un horizon donné.
 */
export interface EconomyCorrelation {
	series_id: string;
	target: string;
	horizon_days: number;
	estimate: Interval;
	n: number;
}
/**
 * Volet explicatif presse (docs/06-presse.md) : part de voix, ton, agenda. N'entre dans la prévision que si `in_forecast` est vrai.
 */
export interface Press {
	asof: IsoDate;
	in_forecast: boolean;
	in_forecast_note: string;
	weeks: IsoDate[];
	/**
	 * Part des mentions de candidats, en % des mentions de la semaine.
	 */
	share_of_voice: PressCandidateTrack[];
	/**
	 * Ton moyen des articles qui citent le candidat, de −1 (négatif) à +1 (positif).
	 */
	tone: PressCandidateTrack[];
	agenda: PressAgendaWeek[];
}
/**
 * Une valeur par semaine, alignée sur `weeks`.
 */
export interface PressCandidateTrack {
	candidate_id: CandidateId;
	values: (number | null)[];
}
export interface PressAgendaWeek {
	week: IsoDate;
	topics: {
		label: string;
		share: Share;
	}[];
}
/**
 * Résultats du backtest (docs/07-evaluation.md) : métriques par modèle, élection et horizon, tableau complet.
 */
export interface Backtest {
	generated_at: IsoDateTime;
	model_version: string;
	elections: number[];
	horizons_days: number[];
	models: BacktestModel[];
	rows: BacktestRow[];
}
export interface BacktestModel {
	id: string;
	label: string;
	kind: 'model' | 'baseline' | 'ablation';
	description: string;
}
/**
 * Métriques d'un modèle pour une élection et un horizon. Null si non calculée.
 */
export interface BacktestRow {
	model_id: string;
	election: number;
	horizon_days: number;
	/**
	 * Log-score de la paire de qualifiés (plus haut = mieux).
	 */
	log_score_pair: number | null;
	/**
	 * Log-score du vainqueur final.
	 */
	log_score_winner: number | null;
	/**
	 * Brier moyen sur la qualification de chaque candidat (plus bas = mieux).
	 */
	brier_qualify: number | null;
	/**
	 * CRPS moyen des parts de premier tour, en points.
	 */
	crps_first_round: number | null;
	/**
	 * Erreur absolue moyenne des médianes de premier tour, en points.
	 */
	mae_first_round: number | null;
	/**
	 * Part des résultats dans l'intervalle à 50 %.
	 */
	coverage_50: Probability | null;
	/**
	 * Part des résultats dans l'intervalle à 90 %.
	 */
	coverage_90: Probability | null;
}
/**
 * history/index.json : concaténation des history/<date>.json par run_date croissante, pour que le site trace l'évolution sans lister le dossier (impossible sur un hébergement statique).
 */
export interface HistoryIndex {
	entries: HistoryEntry[];
}
/**
 * Résumé d'un run, écrit dans history/<run_date>.json : probabilités mélangées par candidat. Un fichier n'est jamais réécrit une fois publié.
 */
export interface HistoryEntry {
	run_date: IsoDate;
	asof: IsoDate;
	model_version: string;
	last_poll_date: IsoDate;
	n_polls: number;
	candidates: HistoryCandidate[];
}
export interface HistoryCandidate {
	candidate_id: CandidateId;
	p_qualify: Probability;
	p_win: Probability;
	first_round_share: Interval;
}
