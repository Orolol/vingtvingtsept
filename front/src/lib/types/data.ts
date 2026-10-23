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
 * Clé du registre data/registry/pollsters.yaml.
 */
export type PollsterId = string;
/**
 * Un quantile par semaine, aligné sur `dates`, en points d'exprimés.
 */
export type QuantileTrack = (number | null)[];
/**
 * Base de l'échantillon publié : inscrits, ou certains d'aller voter.
 */
export type Population = 'registered' | 'certain_voters' | 'unknown';
/**
 * Part en points de pourcentage des suffrages exprimés.
 */
export type Share = number;
/**
 * Ton de −1 (hostile) à +1 (favorable) ; null si masqué.
 */
export type PressToneValue = number | null;
/**
 * Part entre 0 et 1.
 */
export type PressFraction = number;

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
	 * Empreinte des sondages de premier tour vus par l'ajustement (data.rows_hash du manifest de l'ajustement) : l'ajustement n'est refait que si elle change.
	 */
	polls_hash: string | null;
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
	expectations: Expectations;
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
 * « Selon vous, qui sera élu ? » : indicateur d'opinion, hors prévision (runs/2026-10-22-expectations). Données : data/curated/poll_expectations.parquet.
 */
export interface Expectations {
	in_forecast: boolean;
	note: string;
	/**
	 * Dernières mesures 2027 terminées avant `asof` : candidats (`winner`, puis `chances`), puis partis.
	 */
	measures: ExpectationMeasure[];
	/**
	 * Repères calculés sur la série historique (mesures `winner` vérifiées, avant le premier tour).
	 */
	landmarks: {
		id: string;
		text: string;
	}[];
	/**
	 * Série 2007-2022 : une entrée par élection, points du plus ancien au plus récent.
	 */
	history: {
		election: number;
		winner: CandidateId;
		points: ExpectationPoint[];
	}[];
}
/**
 * Dernière mesure d'un institut pour une forme de question et un objet (candidat ou parti).
 */
export interface ExpectationMeasure {
	pollster: PollsterId;
	pollster_name: string;
	fieldwork_end: IsoDate;
	/**
	 * `winner` : une seule réponse parmi les candidats ; `chances` : posée candidat par candidat, les parts ne somment pas à 100.
	 */
	question_type: 'winner' | 'chances';
	kind: 'candidate' | 'party';
	/**
	 * Forme de la question (libellé générique, pas la formulation exacte de l'institut).
	 */
	question: string;
	/**
	 * Base des parts affichées.
	 */
	basis: string;
	sample_size: number | null;
	source_url: string | null;
	/**
	 * @minItems 1
	 */
	items: [ExpectationItem, ...ExpectationItem[]];
}
export interface ExpectationItem {
	/**
	 * Candidat nommé ; null si la question porte sur un parti.
	 */
	candidate_id: CandidateId | null;
	/**
	 * Parti (Toluna Harris Interactive 2026 : « un candidat du RN… »).
	 */
	party: string | null;
	/**
	 * Libellé du parti, null pour un candidat.
	 */
	label: string | null;
	/**
	 * Part publiée, en % des répondants.
	 */
	share: number;
	/**
	 * Question `winner` : part parmi les répondants qui nomment un candidat ; null pour `chances`.
	 */
	share_named: number | null;
}
export interface ExpectationPoint {
	pollster: PollsterId;
	fieldwork_end: IsoDate;
	/**
	 * Jours avant le premier tour.
	 */
	days_before: number;
	/**
	 * Part normalisée du futur vainqueur ; null s'il n'est pas proposé.
	 */
	winner_share_named: number | null;
	top_candidate_id: CandidateId;
	top_is_winner: boolean;
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
	pollster_ratings: PollsterRatings;
	dispersion: PollsterDispersion;
	turnout_gap: TurnoutGap;
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
 * Palmarès des instituts et stabilité du classement d'une élection à la suivante. Volet explicatif : le palmarès a été testé et n'est pas admis (docs/backtest-log.md).
 */
export interface PollsterRatings {
	in_forecast: boolean;
	note: string;
	elections: number[];
	rows: PollsterRating[];
	/**
	 * Corrélation poolée entre élections consécutives.
	 */
	stability: {
		id: string;
		label: string;
		n_pairs: number;
		pearson: number | null;
		p: Probability | null;
	}[];
}
/**
 * Ligne du palmarès 2002-2022 (runs/<date>-pollster-ratings/palmares.parquet), par lignée d'instituts.
 */
export interface PollsterRating {
	pollster: PollsterId;
	name: string;
	elections: number[];
	/**
	 * Sondages de premier tour des campagnes.
	 */
	n_polls: number;
	/**
	 * Sondages des 21 derniers jours.
	 */
	n_final: number;
	/**
	 * Erreur absolue moyenne par candidat du dernier sondage, en points.
	 */
	mae_candidate: number | null;
	/**
	 * Erreur absolue moyenne par bloc du dernier sondage, en points.
	 */
	mae_bloc: number | null;
	/**
	 * Écart d'erreur à l'institut moyen de l'élection, en points (négatif = meilleur).
	 */
	relative_error: number | null;
	/**
	 * Biais signé moyen par bloc du dernier sondage (sondage − résultat), en points.
	 */
	bias: {
		far_right: number | null;
		right: number | null;
		centre: number | null;
		left: number | null;
		far_left: number | null;
	};
	/**
	 * Précision relative x, moyenne sur les élections (négatif = plus précis).
	 */
	precision: number | null;
	/**
	 * Élections où l'institut bat l'institut moyen.
	 */
	n_better: number;
	/**
	 * Élections notées.
	 */
	n_rated: number;
}
/**
 * Dispersion relative D des instituts (vvs.model.herding) : observée / attendue sous tirage aléatoire, un sondage par institut sur `window_days` jours. Volet explicatif, hors prévision (runs/2026-10-22-herding).
 */
export interface PollsterDispersion {
	in_forecast: boolean;
	note: string;
	reading: string;
	window_days: number;
	/**
	 * Dates de calcul, une par semaine ; la dernière vaut asof.
	 */
	weeks: IsoDate[];
	/**
	 * D 2027 à chaque date ; null sous quatre instituts.
	 */
	D: (number | null)[];
	/**
	 * D_ref : moyenne géométrique de D aux présidentielles 2002-2022 au même nombre de jours du vote.
	 */
	reference: (number | null)[];
	n_pollsters: number[];
	current: {
		asof: IsoDate;
		horizon_days: number;
		D: number | null;
		reference: number | null;
		pollsters: PollsterId[];
	};
	/**
	 * D de chaque présidentielle passée au même nombre de jours du vote (null : moins de quatre instituts).
	 */
	past_same_horizon: {
		election: number;
		D: number | null;
	}[];
}
/**
 * Électorats fragiles : écart certains d'aller voter − inscrits par candidat en 2027, premier tour, sondages publiant les deux bases (data/curated/turnout_gap.parquet). Volet explicatif, hors prévision (runs/2026-10-22-turnout-gap).
 */
export interface TurnoutGap {
	in_forecast: boolean;
	note: string;
	/**
	 * Ce que sont devenus ces électorats en 2017 et en 2022.
	 */
	history: string;
	pollsters: string[];
	n_polls: number;
	first_poll: IsoDate | null;
	last_poll: IsoDate | null;
	/**
	 * Triées par écart croissant (les plus fragiles d'abord).
	 */
	rows: {
		candidate_id: CandidateId;
		/**
		 * Part chez les certains − part chez les inscrits, en points d'exprimés ; moyenne des sondages.
		 */
		gap: number;
		/**
		 * Erreur-type (écart type entre sondages / √n) ; null sous deux sondages.
		 */
		se: number | null;
		share_certain: Share;
		share_registered: Share;
		n_polls: number;
		last_poll: IsoDate;
	}[];
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
	/**
	 * Lecture du tableau des corrélations, avec le nombre de paires qui passent le placebo.
	 */
	correlations_note: string;
	study: EconomyStudy;
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
 * Résultat de l'étude économie (runs/<date>-economy/coef_pairs.parquet) pour une paire indicateur × bloc : effet d'une variation mensuelle d'un écart type de l'indicateur sur la part du bloc, au retard où il est le plus net.
 */
export interface EconomyCorrelation {
	/**
	 * Id de macro_series.yaml.
	 */
	series_id: string;
	/**
	 * Libellé court de l'indicateur.
	 */
	series_label: string;
	/**
	 * Bloc dont la part est expliquée ; `incumbent` = bloc du président sortant.
	 */
	target: 'far_right' | 'right' | 'centre' | 'incumbent' | 'left' | 'far_left';
	target_label: string;
	/**
	 * Retard retenu (0 à 6 mois), choisi après coup.
	 */
	lag_months: number;
	effect: Interval;
	p_placebo: Probability;
	/**
	 * Variations mensuelles utilisées.
	 */
	n: number;
	/**
	 * Cycles électoraux représentés.
	 */
	n_cycles: number;
}
/**
 * Résultat de l'étude économie (docs/05-economie.md).
 */
export interface EconomyStudy {
	date: IsoDate;
	verdict: 'negative' | 'positive' | 'inconclusive';
	summary: string;
	findings: string[];
	/**
	 * Document du dépôt qui détaille l'étude.
	 */
	reference: string;
}
/**
 * Volet explicatif presse (docs/06-presse.md) : titres classés par un modèle local (ton, posture, titres extrêmes, part de voix, agenda, repères 2017 et 2022, qualité), attention Wikipédia, rubriques des JT (INA), volume GDELT. N'entre dans la prévision que si `in_forecast` est vrai. Parts des titres classés entre 0 et 1.
 */
export interface Press {
	asof: IsoDate;
	in_forecast: boolean;
	in_forecast_note: string;
	nlp: PressNlpStatus;
	tone: PressTone;
	stance: PressStance;
	headlines: PressHeadlines;
	share_of_voice: PressShareOfVoice;
	agenda_nlp: PressAgendaNlp;
	landmarks: PressLandmarks;
	quality: PressNlpQuality;
	attention: PressAttention;
	ina: PressIna;
	gdelt: PressGdelt;
}
/**
 * Statut de la presse classée : n'entre pas dans la prévision, verdict de l'étude (deux phrases) et test hors échantillon déclaré pour 2027.
 */
export interface PressNlpStatus {
	in_forecast: boolean;
	note: string;
	/**
	 * @minItems 1
	 */
	verdict: [string, ...string[]];
	out_of_sample_test: string;
	reference: string;
}
/**
 * Ton moyen des titres classés qui nomment le candidat, sur `window_weeks` semaines glissantes finissant à la semaine indiquée ; quantiles bootstrap q05, q50, q95 ; null quand `n` < `min_titles`.
 */
export interface PressTone {
	note: string;
	window_weeks: number;
	min_titles: number;
	/**
	 * Lundi de chaque semaine complète.
	 */
	weeks: IsoDate[];
	tracks: PressToneTrack[];
}
export interface PressToneTrack {
	candidate_id: CandidateId;
	q05: PressToneValue[];
	q50: PressToneValue[];
	q95: PressToneValue[];
	/**
	 * Titres classés dans la fenêtre.
	 */
	n: number[];
}
/**
 * Parts de chaque posture (ordre de `categories`) par candidat, sur les dernières semaines (`last_weeks`) et semaine par semaine ; une liste de parts qui somme à 1, ou null sous `min_titles` titres.
 */
export interface PressStance {
	note: string;
	categories: PressStanceCategory[];
	min_titles: number;
	last_weeks: PressStanceLastWeeks;
	weeks: IsoDate[];
	tracks: PressStanceTrack[];
}
export interface PressStanceCategory {
	id: string;
	label: string;
}
export interface PressStanceLastWeeks {
	start: IsoDate;
	end: IsoDate;
	rows: PressStanceRow[];
}
export interface PressStanceRow {
	candidate_id: CandidateId;
	n: number;
	shares: PressFraction[] | null;
}
export interface PressStanceTrack {
	candidate_id: CandidateId;
	n: number[];
	/**
	 * Une entrée par semaine de `weeks`.
	 */
	shares: (PressFraction[] | null)[];
}
/**
 * Dernière semaine complète : par candidat, titres classés hostiles (−1) et favorables (+1), les plus récents d'abord.
 */
export interface PressHeadlines {
	note: string;
	week: IsoDate;
	week_end: IsoDate;
	candidates: PressHeadlinesCandidate[];
}
export interface PressHeadlinesCandidate {
	candidate_id: CandidateId;
	n: number;
	negative: PressHeadline[];
	positive: PressHeadline[];
}
export interface PressHeadline {
	title: string;
	outlet: string;
	date: IsoDate;
	url: string;
	stance: string;
}
/**
 * Part des titres nommant chaque candidat parmi les candidats suivis, et part des pages vues Wikipédia des mêmes candidats, mêmes semaines (chaque semaine somme à 1).
 */
export interface PressShareOfVoice {
	note: string;
	weeks: IsoDate[];
	tracks: PressFractionTrack[];
	attention: PressFractionTrack[];
}
/**
 * Une part (0 à 1) par semaine, alignée sur `weeks`.
 */
export interface PressFractionTrack {
	candidate_id: CandidateId;
	values: (number | null)[];
}
/**
 * Part hebdomadaire des titres classés portant chacun des thèmes concordants (multi-étiquette : pas de somme à 1) ; null sous `min_titles` titres.
 */
export interface PressAgendaNlp {
	note: string;
	min_titles: number;
	weeks: IsoDate[];
	n: number[];
	themes: PressAgendaTheme[];
}
export interface PressAgendaTheme {
	id: string;
	label: string;
	values: (number | null)[];
}
/**
 * Repères de campagnes passées : ton et part de la posture « affaire » jour par jour (moyenne sur trois jours) autour d'un choc, et dates de première rupture.
 */
export interface PressLandmarks {
	note: string;
	events: PressLandmark[];
}
export interface PressLandmark {
	id: string;
	label: string;
	candidate_id: CandidateId;
	cycle: number;
	event_date: IsoDate;
	summary: string;
	dates: IsoDate[];
	tone: PressToneValue[];
	scandal_share: (number | null)[];
	/**
	 * Titres classés du jour.
	 */
	n: number[];
	breaks: PressLandmarkBreaks;
}
export interface PressLandmarkBreaks {
	polls: IsoDate | null;
	tone: IsoDate | null;
	scandal: IsoDate | null;
}
/**
 * Accord entre deux formulations du prompt sur un échantillon de titres (κ de Cohen pour le ton et la posture, F1 pour les thèmes), modèle et version de prompt.
 */
export interface PressNlpQuality {
	model: string;
	model_label: string;
	prompt_version: string;
	n_sample: number;
	tone_kappa: number;
	stance_kappa: number;
	themes_f1: PressFraction;
	thresholds: {
		tone_kappa: number;
		stance_kappa: number;
		themes_f1: number;
	};
	passes: {
		tone_kappa: boolean;
		stance_kappa: boolean;
		themes_f1: boolean;
	};
	labeled: PressNlpLabeled[];
	reference: string;
}
export interface PressNlpLabeled {
	cycle: number;
	n: number;
}
/**
 * Témoin d'actualité : part hebdomadaire des pages vues Wikipédia de chaque candidat parmi les candidats suivis pour 2027 (data/curated/attention.parquet), en %.
 */
export interface PressAttention {
	in_forecast: boolean;
	note: string;
	source: string;
	/**
	 * Lundi de chaque semaine complète.
	 */
	weeks: IsoDate[];
	/**
	 * Une valeur par semaine (%), alignée sur `weeks`.
	 */
	tracks: PressCandidateTrack[];
}
/**
 * Une valeur par période, alignée sur la liste de dates du bloc qui la porte (`weeks` ou `months`).
 */
export interface PressCandidateTrack {
	candidate_id: CandidateId;
	values: (number | null)[];
}
/**
 * Rubriques des sujets de JT (INA, data/curated/press_ina.parquet) : part de chaque rubrique dans les sujets de chaque année électorale, en %.
 */
export interface PressIna {
	in_forecast: boolean;
	note: string;
	source: string;
	years: number[];
	/**
	 * Période de référence de `share_all_years` (« 2000-2020 »).
	 */
	all_years_label: string;
	topics: {
		label: string;
		/**
		 * Une part (%) par année de `years`.
		 */
		shares: Share[];
		share_all_years: Share;
	}[];
}
/**
 * Part mensuelle des articles surveillés par GDELT qui citent le candidat (data/curated/press_timelines.parquet, volume sur volume surveillé), en %.
 */
export interface PressGdelt {
	in_forecast: boolean;
	note: string;
	source: string;
	/**
	 * Premier jour de chaque mois.
	 */
	months: IsoDate[];
	/**
	 * Une valeur par mois (%), alignée sur `months`.
	 */
	tracks: PressCandidateTrack[];
}
/**
 * Résultats du backtest (docs/07-evaluation.md) : backtest rétrospectif (métriques par modèle, élection et horizon, tableau complet) et backtest chronologique de la consolidation (docs/10-consolidation.md).
 */
export interface Backtest {
	generated_at: IsoDateTime;
	model_version: string;
	elections: number[];
	horizons_days: number[];
	models: BacktestModel[];
	rows: BacktestRow[];
	variables_tested: TestedVariable[];
	/**
	 * Journal qui détaille les tests.
	 */
	variables_tested_reference: string;
	chronological: ChronologicalBacktest;
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
 * Variable candidate testée contre la référence du backtest (docs/backtest-log.md).
 */
export interface TestedVariable {
	id: string;
	label: string;
	/**
	 * « non admise » (échec au backtest), « non retenue » (écartée dès l'étude) ou « en attente d'accès » (données demandées).
	 */
	verdict: string;
	admitted: boolean;
	/**
	 * Vrai si la projection publiée l'utilise malgré tout (choix éditorial).
	 */
	published: boolean;
	/**
	 * Une ligne.
	 */
	summary: string;
	wave: IsoDate;
}
/**
 * Backtest chronologique de la référence figée (docs/10-consolidation.md) : calibrations sans élection future, délai de publication, scénarios sans correspondance après coup, score jusqu'au vainqueur. Référence contre témoin mean14d.
 */
export interface ChronologicalBacktest {
	/**
	 * Identifiant (et tag Git) de la référence.
	 */
	reference_id: string;
	reference_asof: IsoDate;
	/**
	 * Commit du code statistique figé.
	 */
	source_commit: string;
	/**
	 * Document du protocole, relatif à la racine du dépôt.
	 */
	protocol: string;
	/**
	 * Bilan versionné, relatif à la racine du dépôt.
	 */
	results: string;
	/**
	 * Tirages par modèle et par cas.
	 */
	n_draws: number;
	elections: number[];
	horizons_days: number[];
	methods: {
		id: 'reference' | 'mean14d';
		label: string;
		description: string;
	}[];
	by_horizon: {
		horizon_days: number;
		reference: ChronologicalScores;
		mean14d: ChronologicalScores;
	}[];
	by_election: ChronologicalRow[];
	/**
	 * Lecture du bilan, sans enjoliver.
	 */
	reading: string;
	limitations: {
		id: string;
		/**
		 * Une phrase.
		 */
		text: string;
	}[];
}
/**
 * Moyennes d'un modèle à un horizon, même poids pour chaque élection.
 */
export interface ChronologicalScores {
	/**
	 * Log-score du vainqueur (lissage de Jeffreys, plus haut = mieux).
	 */
	log_score_winner: number;
	/**
	 * Log-score de la paire de qualifiés.
	 */
	log_score_pair: number;
	/**
	 * CRPS des parts de premier tour, en points (plus bas = mieux).
	 */
	crps_first_round: number;
	coverage_90: Probability;
}
/**
 * Un modèle, une élection, un horizon (metrics.csv de la consolidation).
 */
export interface ChronologicalRow {
	election: number;
	horizon_days: number;
	method: 'reference' | 'mean14d';
	p_winner: Probability;
	p_pair: Probability;
	log_score_winner: number;
	log_score_pair: number;
	crps_first_round: number;
	coverage_90: Probability;
	/**
	 * Voix finales (en points) des candidats absents des scénarios.
	 */
	unforecast_share: number;
	/**
	 * Noms des candidats officiels absents des scénarios.
	 */
	missing_candidates: string[];
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
