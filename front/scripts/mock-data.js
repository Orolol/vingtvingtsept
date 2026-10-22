// Génère des données FACTICES mais cohérentes dans static/data/, conformes au contrat JSON,
// pour développer le site avant que `vvs export` existe. Le manifeste porte `mock: true` et le
// site affiche un bandeau. Usage : npm run mock-data (régénère puis valide).
//
// Ce qui est réel : le contenu des scénarios (data/registry/scenarios_2027.yaml), les noms des
// candidats et instituts (registres). Tout le reste (trajectoires, sondages, probabilités,
// reports, backtest, séries économiques) est simulé par un petit modèle jouet : poids latents
// par candidat, marche aléatoire hebdomadaire, substitution intra-bloc, reports par bloc,
// Monte-Carlo. Il n'a aucune valeur prédictive.
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parse } from 'yaml';
import { dataDir, frontDir } from './schemas.js';

const registryDir = path.resolve(frontDir, '../data/registry');
const readYaml = async (name) => parse(await readFile(path.join(registryDir, name), 'utf8'));

// --- aléa reproductible
function mulberry32(seed) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
const rand = mulberry32(2027);
function normal() {
	let u = 0;
	while (u === 0) u = rand();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
const pick = (xs) => xs[Math.floor(rand() * xs.length)];

// --- dates
const DAY = 86_400_000;
const toDate = (s) => new Date(`${s}T00:00:00Z`);
const iso = (d) => d.toISOString().slice(0, 10);
const addDays = (s, n) => iso(new Date(toDate(s).getTime() + n * DAY));
const daysBetween = (a, b) => Math.round((toDate(b) - toDate(a)) / DAY);

const RUN_DATE = '2026-10-20';
const ASOF = '2026-10-20';
const FIRST_ROUND = '2027-04-18';
const SECOND_ROUND = '2027-05-02';
const FIRST_WEEK = '2025-03-24';
const LAST_POLL = '2026-10-16';

const weeks = [];
for (let d = FIRST_WEEK; d <= ASOF; d = addDays(d, 7)) weeks.push(d);
const weekIndex = (date) =>
	Math.max(0, Math.min(weeks.length - 1, Math.floor(daysBetween(FIRST_WEEK, date) / 7)));

// --- registres
const scenariosYaml = await readYaml('scenarios_2027.yaml');
const candidatesYaml = (await readYaml('candidates.yaml')).candidates;
const pollstersYaml = (await readYaml('pollsters.yaml')).pollsters;

// Bloc des candidats des dimensions (les petits candidats ont le leur dans le YAML).
const dimensionBloc = {
	le_pen_marine: 'far_right',
	bardella_jordan: 'far_right',
	zemmour_eric: 'far_right',
	knafo_sarah: 'far_right',
	philippe_edouard: 'centre',
	attal_gabriel: 'centre',
	retailleau_bruno: 'right',
	glucksmann_raphael: 'left',
	tondelier_marine: 'left',
	melenchon_jean_luc: 'far_left',
	bompard_manuel: 'far_left',
	placeholder_centre: 'centre'
};

// Poids latents à la date de prévision (en points, « tout le monde présent »), et première
// semaine où le candidat est testé.
const weight = {
	le_pen_marine: 33,
	bardella_jordan: 31,
	melenchon_jean_luc: 15.5,
	bompard_manuel: 7,
	philippe_edouard: 15,
	attal_gabriel: 9,
	placeholder_centre: 6,
	glucksmann_raphael: 11,
	tondelier_marine: 3.5,
	retailleau_bruno: 8,
	zemmour_eric: 4.5,
	knafo_sarah: 2,
	dupont_aignan_nicolas: 2,
	philippot_florian: 1,
	de_villiers_philippe: 1.5,
	bertrand_xavier: 3,
	lisnard_david: 2.5,
	platret_gilles: 0.5,
	becht_olivier: 0.8,
	ruffin_francois: 3,
	cazeneuve_bernard: 2.5,
	hollande_francois: 5,
	bouamrane_karim: 0.7,
	batho_delphine: 0.8,
	roussel_fabien: 2.5,
	arthaud_nathalie: 1,
	kazib_anasse: 0.5,
	labib_selma: 0.4,
	poutou_philippe: 1,
	lassalle_jean: 2,
	asselineau_francois: 1,
	de_villepin_dominique: 2.5,
	polony_natacha: 1.2
};
const firstTested = {
	polony_natacha: '2026-10-12',
	platret_gilles: '2026-10-05',
	becht_olivier: '2026-09-21',
	kazib_anasse: '2026-06-01',
	labib_selma: '2026-06-15',
	bouamrane_karim: '2026-06-08',
	cazeneuve_bernard: '2026-07-13',
	bompard_manuel: '2025-09-01',
	knafo_sarah: '2025-06-02',
	tondelier_marine: '2025-05-05',
	philippot_florian: '2026-05-11',
	de_villiers_philippe: '2026-01-05',
	placeholder_centre: '9999-12-31' // jamais testé : pas de trajectoire, seulement la projection
};
// Candidats qui se remplacent dans une même dimension : l'absence de l'un n'est pas une masse
// à redistribuer quand l'autre est présent.
const alternates = [
	['le_pen_marine', 'bardella_jordan'],
	['melenchon_jean_luc', 'bompard_manuel'],
	['zemmour_eric', 'knafo_sarah']
];
const alternateOf = new Map(
	alternates.flatMap(([a, b]) => [
		[a, b],
		[b, a]
	])
);

const bloc = { ...dimensionBloc };
for (const s of scenariosYaml.small_candidates) bloc[s.candidate_id] = s.bloc;
const roster = Object.keys(weight);
for (const id of roster) if (!bloc[id]) throw new Error(`bloc inconnu pour ${id}`);

// --- trajectoires latentes (médianes) : marche aléatoire en log ancrée à la dernière semaine
const latent = {};
for (const id of roster) {
	const n = weeks.length;
	const x = new Array(n).fill(0);
	for (let t = n - 2; t >= 0; t--) x[t] = x[t + 1] + 0.035 * normal();
	// tendances narratives légères : Glucksmann monte après la primaire, Attal recule
	if (id === 'glucksmann_raphael')
		for (let t = 0; t < n; t++) x[t] -= 0.25 * Math.max(0, Math.min(1, (n - 1 - t) / 8));
	if (id === 'attal_gabriel') for (let t = 0; t < n; t++) x[t] += 0.2 * ((n - 1 - t) / n);
	const smooth = x.map((_, t) => (x[Math.max(0, t - 1)] + x[t] + x[Math.min(n - 1, t + 1)]) / 3);
	latent[id] = smooth.map((v) => weight[id] * Math.exp(v - smooth[n - 1]));
}
const testedAt = (id, t) => (firstTested[id] ?? FIRST_WEEK) <= weeks[t];

// --- configurations (dimensions seulement, petits candidats en présence marginale)
const dims = scenariosYaml.dimensions;
const smalls = scenariosYaml.small_candidates;
const constraints = scenariosYaml.constraints;
const placeholderId = (b) => `placeholder_${b}`;

function enumerateConfigurations() {
	const out = [];
	let impossible = 0;
	const walk = (i, states, p) => {
		if (i === dims.length) {
			const active = constraints.filter((c) =>
				Object.entries(c.when).every(([d, s]) => states[d] === s)
			);
			const chosen = dims.map((d) => d.states.find((s) => s.id === states[d.id]));
			const named = chosen.flatMap((s) => s.candidates ?? []);
			const absent = new Set(active.flatMap((c) => c.absent ?? []));
			if (active.some((c) => c.impossible) || named.some((id) => absent.has(id))) {
				impossible += p;
				return;
			}
			const placeholders = chosen.flatMap((s) => s.placeholders ?? []);
			const smallPresence = Object.fromEntries(
				smalls
					.filter((s) => !absent.has(s.candidate_id))
					.map((s) => [s.candidate_id, s.probability])
			);
			out.push({ states: { ...states }, p, named, placeholders, smallPresence });
			return;
		}
		for (const s of dims[i].states)
			walk(i + 1, { ...states, [dims[i].id]: s.id }, p * s.probability);
	};
	walk(0, {}, 1);
	const total = out.reduce((a, c) => a + c.p, 0);
	for (const c of out) c.probability = c.p / total;
	out.sort((a, b) => b.probability - a.probability);
	return { configurations: out, impossible };
}
const { configurations, impossible: impossibleMass } = enumerateConfigurations();

const dimShort = {
	rn: 'RN',
	centre: 'Bloc central',
	left: 'Gauche hors LFI',
	lr: 'LR',
	lfi: 'LFI',
	reconquete: 'Reconquête'
};
const stateShort = {
	'rn.le_pen': 'Le Pen',
	'rn.bardella': 'Bardella',
	'centre.philippe_only': 'Philippe seul',
	'centre.philippe_attal': 'Philippe et Attal',
	'centre.attal_only': 'Attal seul',
	'centre.other': 'autre candidat',
	'left.consolidated': 'Glucksmann seul',
	'left.fragmented': 'Glucksmann et Tondelier',
	'lr.retailleau': 'Retailleau',
	'lr.rally': 'ralliement au centre',
	'lfi.melenchon': 'Mélenchon',
	'lfi.substitute': 'remplaçant',
	'reconquete.zemmour': 'Zemmour',
	'reconquete.knafo': 'Knafo',
	'reconquete.none': 'pas de candidat'
};
const configId = (states) => `cfg.${dims.map((d) => states[d.id]).join('-')}`;
const configLabel = (states) => dims.map((d) => stateShort[`${d.id}.${states[d.id]}`]).join(' · ');

const mixCondition = {
	id: 'mix',
	kind: 'mixture',
	label: 'Mélange des scénarios',
	states: {},
	probability: 1
};
const dimConditions = dims.flatMap((d) =>
	d.states.map((s) => ({
		id: `${d.id}.${s.id}`,
		kind: 'dimension_state',
		label: `${dimShort[d.id]} : ${stateShort[`${d.id}.${s.id}`]}`,
		states: { [d.id]: s.id },
		probability: round(
			configurations.filter((c) => c.states[d.id] === s.id).reduce((a, c) => a + c.probability, 0),
			4
		)
	}))
);
const TOP_CONFIGS = 8;
const configConditions = configurations.slice(0, TOP_CONFIGS).map((c) => ({
	id: configId(c.states),
	kind: 'configuration',
	label: configLabel(c.states),
	states: c.states,
	probability: round(c.probability, 4)
}));
const conditionConfigs = (cond) =>
	configurations.filter((c) => Object.entries(cond.states).every(([d, s]) => c.states[d] === s));

function sampleConfig(pool, total) {
	let u = rand() * total;
	for (const c of pool) {
		u -= c.probability;
		if (u <= 0) return c;
	}
	return pool[pool.length - 1];
}
function samplePresent(config) {
	const present = new Set(config.named);
	for (const b of config.placeholders) present.add(placeholderId(b));
	for (const [id, p] of Object.entries(config.smallPresence)) if (rand() < p) present.add(id);
	return present;
}

// --- parts de premier tour d'un ensemble de candidats présents
const IN_BLOC = 0.6; // part de la masse d'un absent qui reste dans son bloc
function shares(present, level, blocNoise, candNoise) {
	const w = {};
	for (const id of present)
		w[id] = level(id) * Math.exp((blocNoise?.[bloc[id]] ?? 0) + (candNoise?.[id] ?? 0));
	const blocMass = {};
	for (const id of present) blocMass[bloc[id]] = (blocMass[bloc[id]] ?? 0) + w[id];
	const extra = {};
	for (const id of roster) {
		if (present.has(id) || id.startsWith('placeholder_')) continue;
		const alt = alternateOf.get(id);
		if (alt && present.has(alt)) continue;
		const m = blocMass[bloc[id]];
		if (m) extra[bloc[id]] = (extra[bloc[id]] ?? 0) + IN_BLOC * level(id);
	}
	let total = 0;
	for (const id of present) {
		const b = bloc[id];
		w[id] += ((extra[b] ?? 0) * w[id]) / blocMass[b];
		total += w[id];
	}
	const out = {};
	for (const id of present) out[id] = (100 * w[id]) / total;
	return out;
}

// --- second tour : reports par bloc
const affinity = {
	far_right: { far_right: 1.0, right: 0.35, centre: 0.12, left: 0.05, far_left: 0.08, none: 0.25 },
	right: { far_right: 0.35, right: 1.0, centre: 0.55, left: 0.08, far_left: 0.03, none: 0.25 },
	centre: { far_right: 0.08, right: 0.6, centre: 1.0, left: 0.45, far_left: 0.1, none: 0.25 },
	left: { far_right: 0.05, right: 0.15, centre: 0.55, left: 1.0, far_left: 0.6, none: 0.3 },
	far_left: { far_right: 0.2, right: 0.04, centre: 0.25, left: 0.6, far_left: 1.0, none: 0.5 },
	other: { far_right: 0.3, right: 0.2, centre: 0.2, left: 0.15, far_left: 0.15, none: 0.5 }
};
function transferRates(from, a, b, noise) {
	if (from === a) return [1, 0, 0];
	if (from === b) return [0, 1, 0];
	const row = affinity[bloc[from]];
	const n = noise ? () => Math.exp(0.3 * normal()) : () => 1;
	const ta = row[bloc[a]] * n();
	const tb = row[bloc[b]] * n();
	const tn = row.none * n();
	const s = ta + tb + tn;
	return [ta / s, tb / s, tn / s];
}
function duel(first, a, b, noise, sd = 0) {
	let va = 0;
	let vb = 0;
	for (const [id, s] of Object.entries(first)) {
		const [ra, rb] = transferRates(id, a, b, noise);
		va += s * ra;
		vb += s * rb;
	}
	const logit = Math.log(va / vb) + sd * normal();
	return 100 / (1 + Math.exp(-logit));
}

// --- utilitaires statistiques
function round(x, d = 1) {
	const k = 10 ** d;
	return Math.round(x * k) / k;
}
function quantiles(xs, d = 1) {
	const s = Float64Array.from(xs).sort();
	const q = (p) => {
		const i = p * (s.length - 1);
		const lo = Math.floor(i);
		const hi = Math.ceil(i);
		return s[lo] + (s[hi] - s[lo]) * (i - lo);
	};
	return {
		q05: round(q(0.05), d),
		q25: round(q(0.25), d),
		q50: round(q(0.5), d),
		q75: round(q(0.75), d),
		q95: round(q(0.95), d)
	};
}
const blocs = Object.keys(affinity);
const horizonSd = (tau) => ({
	bloc: 0.04 + 0.12 * Math.sqrt(Math.max(tau, 0) / 180),
	cand: 0.05 + 0.15 * Math.sqrt(Math.max(tau, 0) / 180),
	duel: 0.03 + 0.1 * Math.sqrt(Math.max(tau, 0) / 180)
});

// Simulation complète à une semaine donnée, sous une condition.
function simulate({ t, tau, pool, nDraws, withPairs = false }) {
	const sd = horizonSd(tau);
	const total = pool.reduce((a, c) => a + c.probability, 0);
	const level = (id) => latent[id][t];
	const present = {};
	const sharesOf = {};
	const qualify = {};
	const win = {};
	const pairs = new Map();
	for (let i = 0; i < nDraws; i++) {
		const config = sampleConfig(pool, total);
		const set = samplePresent(config);
		const bn = Object.fromEntries(blocs.map((b) => [b, sd.bloc * normal()]));
		const cn = Object.fromEntries([...set].map((id) => [id, sd.cand * normal()]));
		const first = shares(set, level, bn, cn);
		for (const [id, s] of Object.entries(first)) {
			present[id] = (present[id] ?? 0) + 1;
			(sharesOf[id] ??= []).push(s);
		}
		const [a, b] = Object.entries(first)
			.sort((x, y) => y[1] - x[1])
			.slice(0, 2)
			.map(([id]) => id)
			.sort((x, y) => weight[y] - weight[x]);
		qualify[a] = (qualify[a] ?? 0) + 1;
		qualify[b] = (qualify[b] ?? 0) + 1;
		const shareA = duel(first, a, b, true, sd.duel);
		const winner = shareA >= 50 ? a : b;
		win[winner] = (win[winner] ?? 0) + 1;
		if (withPairs) {
			const key = `${a}|${b}`;
			if (!pairs.has(key)) pairs.set(key, { a, b, n: 0, shareA: [], aWins: 0, transfers: {} });
			const p = pairs.get(key);
			p.n++;
			p.shareA.push(shareA);
			if (winner === a) p.aWins++;
			for (const [id, s] of Object.entries(first)) {
				if (id === a || id === b || s < 1.5) continue;
				const r = transferRates(id, a, b, true);
				const tr = (p.transfers[id] ??= { a: [], b: [], n: [] });
				tr.a.push(100 * r[0]);
				tr.b.push(100 * r[1]);
				tr.n.push(100 * r[2]);
			}
		}
	}
	const candidates = Object.keys(present)
		.map((id) => ({
			candidate_id: id,
			p_present: round(present[id] / nDraws, 4),
			p_qualify: round((qualify[id] ?? 0) / nDraws, 4),
			p_win: round((win[id] ?? 0) / nDraws, 4),
			first_round_share: quantiles(sharesOf[id])
		}))
		.sort(
			(x, y) =>
				y.p_win - x.p_win ||
				y.p_qualify - x.p_qualify ||
				y.first_round_share.q50 - x.first_round_share.q50
		);
	return { candidates, pairs, nDraws };
}

// Trajectoire latente (pas de bruit d'horizon) d'une condition : quantiles hebdomadaires.
function trajectories(cond, candidateIds, nDraws = 600) {
	const pool = conditionConfigs(cond);
	const total = pool.reduce((a, c) => a + c.probability, 0);
	const tracks = Object.fromEntries(
		candidateIds.map((id) => [id, { q05: [], q25: [], q50: [], q75: [], q95: [] }])
	);
	for (let t = 0; t < weeks.length; t++) {
		const samples = {};
		const level = (id) => latent[id][t];
		for (let i = 0; i < nDraws; i++) {
			const set = samplePresent(sampleConfig(pool, total));
			const bn = Object.fromEntries(blocs.map((b) => [b, 0.03 * normal()]));
			const cn = Object.fromEntries(
				[...set].map((id) => [id, (testedAt(id, t) ? 0.05 : 0.2) * normal()])
			);
			const s = shares(set, level, bn, cn);
			for (const [id, v] of Object.entries(s)) (samples[id] ??= []).push(v);
		}
		for (const id of candidateIds) {
			const q = samples[id] && testedAt(id, t) ? quantiles(samples[id]) : null;
			for (const k of ['q05', 'q25', 'q50', 'q75', 'q95']) tracks[id][k].push(q ? q[k] : null);
		}
	}
	// Lissage sur cinq semaines (même noyau pour chaque quantile : l'ordre est conservé).
	for (const tr of Object.values(tracks))
		for (const k of ['q05', 'q25', 'q50', 'q75', 'q95']) {
			const v = tr[k];
			tr[k] = v.map((x, t) => {
				if (x === null) return null;
				const win = v.slice(Math.max(0, t - 2), t + 3).filter((y) => y !== null);
				return round(win.reduce((a, b) => a + b, 0) / win.length, 1);
			});
		}
	return tracks;
}

// --- sondages factices
const pollsterIds = [
	'ifop',
	'harris',
	'opinionway',
	'elabe',
	'cluster17',
	'odoxa',
	'ipsos',
	'yougov',
	'verian'
];
const sponsors = {
	ifop: ['Le JDD', 'Sud Radio', 'Le Figaro'],
	harris: ['RTL', 'Challenges'],
	opinionway: ['CNews, Europe 1, JDD'],
	elabe: ['BFMTV, La Tribune Dimanche'],
	cluster17: ['Le Point'],
	odoxa: ['Public Sénat', 'Mascaret'],
	ipsos: ['Le Monde, Cevipof, Fondation Jean-Jaurès'],
	yougov: [null],
	verian: ['Le Figaro Magazine']
};
const house = Object.fromEntries(
	pollsterIds.map((p) => [p, Object.fromEntries(blocs.map((b) => [b, 0.06 * normal()]))])
);
const commonSmalls = [
	'roussel_fabien',
	'arthaud_nathalie',
	'dupont_aignan_nicolas',
	'lassalle_jean',
	'asselineau_francois',
	'poutou_philippe'
];
const occasionalSmalls = [
	'ruffin_francois',
	'hollande_francois',
	'de_villepin_dominique',
	'lisnard_david',
	'cazeneuve_bernard',
	'bertrand_xavier',
	'philippot_florian',
	'polony_natacha',
	'becht_olivier',
	'batho_delphine',
	'bouamrane_karim'
];

function pollHypothesis(t) {
	const rn = rand() < 0.85 ? 'le_pen_marine' : 'bardella_jordan';
	const centre = pick([
		['philippe_edouard', 'attal_gabriel'],
		['philippe_edouard', 'attal_gabriel'],
		['philippe_edouard'],
		['attal_gabriel']
	]);
	const left = rand() < 0.6 ? ['glucksmann_raphael', 'tondelier_marine'] : ['glucksmann_raphael'];
	const set = new Set([rn, ...centre, ...left, 'melenchon_jean_luc', 'zemmour_eric']);
	if (rand() < 0.85) set.add('retailleau_bruno');
	for (const id of commonSmalls) if (rand() < 0.75) set.add(id);
	for (const id of occasionalSmalls) if (rand() < 0.15) set.add(id);
	return [...set].filter((id) => testedAt(id, t));
}

function roundHalf(x) {
	return Math.round(x * 2) / 2;
}

const polls = [];
const pollDates = [];
for (let d = '2025-03-27'; d <= LAST_POLL;) {
	pollDates.push(d);
	const gap =
		d < '2026-01-01'
			? 14 + Math.floor(rand() * 18)
			: d < '2026-09-01'
				? 7 + Math.floor(rand() * 10)
				: 3 + Math.floor(rand() * 5);
	const next = addDays(d, gap);
	if (next > LAST_POLL && d !== LAST_POLL) {
		pollDates.push(LAST_POLL);
		break;
	}
	d = next;
}
const seq = {};
for (const end of pollDates) {
	const pollster = pick(pollsterIds);
	const t = weekIndex(end);
	const n = 50 * Math.round((900 + rand() * 1200) / 50);
	const key = `${pollster}-${end.replaceAll('-', '')}`;
	seq[key] = (seq[key] ?? 0) + 1;
	const population = rand() < 0.65 ? 'registered' : 'certain_voters';
	const hypotheses = [];
	const nh = 1 + Math.floor(rand() * 3);
	const seen = new Set();
	for (let h = 0; h < nh; h++) {
		const ids = pollHypothesis(t);
		const hid = [...ids].sort().join('+');
		if (seen.has(hid)) continue;
		seen.add(hid);
		const truth = shares(new Set(ids), (id) => latent[id][t], house[pollster], null);
		const neff = n * 0.6;
		const raw = Object.fromEntries(
			ids.map((id) => {
				const p = truth[id] / 100;
				return [id, Math.max(0, truth[id] + 100 * Math.sqrt((p * (1 - p)) / neff) * normal())];
			})
		);
		const sum = Object.values(raw).reduce((a, b) => a + b, 0);
		const sharesList = ids
			.map((id) => ({ candidate_id: id, share: roundHalf((100 * raw[id]) / sum) }))
			.sort((x, y) => y.share - x.share);
		hypotheses.push({ hypothesis_id: hid, round: 1, population, shares: sharesList });
	}
	// duels hypothétiques
	if (rand() < 0.45) {
		const ref = shares(new Set(pollHypothesis(t)), (id) => latent[id][t], house[pollster], null);
		const rn = ref.bardella_jordan ? 'bardella_jordan' : 'le_pen_marine';
		const opponents = [
			'philippe_edouard',
			'melenchon_jean_luc',
			'attal_gabriel',
			'glucksmann_raphael',
			'retailleau_bruno'
		].filter((id) => ref[id] && rand() < 0.6);
		for (const opp of opponents) {
			const sa = roundHalf(duel(ref, rn, opp, false) + 1.5 * normal());
			hypotheses.push({
				hypothesis_id: [rn, opp].sort().join('+'),
				round: 2,
				population,
				shares: [
					{ candidate_id: rn, share: sa },
					{ candidate_id: opp, share: round(100 - sa, 1) }
				].sort((x, y) => y.share - x.share)
			});
		}
	}
	const start = addDays(end, -(2 + Math.floor(rand() * 3)));
	polls.push({
		poll_id: `2027-${key}-${seq[key]}`,
		pollster,
		sponsor: pick(sponsors[pollster]),
		fieldwork_start: start,
		fieldwork_end: end,
		publish_date: addDays(end, 1 + Math.floor(rand() * 2)),
		sample_size: n,
		mode: 'online',
		source_url: 'https://www.commission-des-sondages.fr/notices/',
		hypotheses
	});
}
polls.sort((a, b) => (a.fieldwork_end < b.fieldwork_end ? 1 : -1));

// --- candidats affichés
function currentParty(id) {
	const parties = candidatesYaml[id]?.parties ?? [];
	const open = parties.filter((p) => p.to === null);
	return (open.at(-1) ?? parties.at(-1))?.party ?? null;
}
const manifestCandidates = roster.map((id) => {
	if (id.startsWith('placeholder_')) {
		return {
			id,
			name: 'Autre candidat du bloc central',
			short_name: 'Autre (centre)',
			bloc: bloc[id],
			party: null,
			is_placeholder: true
		};
	}
	const c = candidatesYaml[id];
	if (!c) throw new Error(`candidat absent de candidates.yaml : ${id}`);
	return {
		id,
		name: c.name,
		short_name: c.last_name,
		bloc: bloc[id],
		party: currentParty(id),
		is_placeholder: false
	};
});

// --- summary.json
const lastT = weeks.length - 1;
const tauNow = daysBetween(ASOF, FIRST_ROUND);
const N_DRAWS = 10_000;
const mix = simulate({
	t: lastT,
	tau: tauNow,
	pool: configurations,
	nDraws: N_DRAWS,
	withPairs: true
});
const byCondition = [...dimConditions, ...configConditions].map((cond) => ({
	condition_id: cond.id,
	candidates: simulate({ t: lastT, tau: tauNow, pool: conditionConfigs(cond), nDraws: 4000 })
		.candidates
}));
// anticipations « qui sera élu ? » factices : les trois favoris du mélange, part normalisée
// sur les répondants qui nomment un candidat (20 % ne savent pas)
const favourites = mix.candidates.slice(0, 3).map((c) => c.candidate_id);
const expectationItems = (shares, named) =>
	favourites.map((id, i) => ({
		candidate_id: id,
		party: null,
		label: null,
		share: shares[i],
		share_named: named ? round((100 * shares[i]) / 80) : null
	}));
const expectations = {
	in_forecast: false,
	note: 'Données factices : indicateur d’opinion, hors prévision.',
	measures: [
		{
			pollster: 'cluster17',
			pollster_name: 'Cluster17',
			fieldwork_end: addDays(ASOF, -88),
			question_type: 'winner',
			kind: 'candidate',
			question: 'Selon vous, qui sera élu ? (une seule réponse)',
			basis: '% des répondants qui nomment un candidat',
			sample_size: 1500,
			source_url: null,
			items: expectationItems([44, 20, 16], true)
		},
		{
			pollster: 'harris',
			pollster_name: 'Toluna Harris Interactive',
			fieldwork_end: addDays(ASOF, -200),
			question_type: 'chances',
			kind: 'party',
			question: 'Peut-il remporter l’élection ? (posée candidat par candidat)',
			basis: '% de oui, chaque candidat séparément',
			sample_size: 1300,
			source_url: null,
			items: [
				{
					candidate_id: null,
					party: 'rn',
					label: 'Un candidat du RN',
					share: 60,
					share_named: null
				}
			]
		}
	],
	landmarks: [{ id: 'mock', text: 'Repère factice.' }],
	history: [
		{
			election: 2022,
			winner: favourites[0],
			points: [90, 60, 30, 7].map((d, i) => ({
				pollster: 'ifop',
				fieldwork_end: addDays('2022-04-10', -d),
				days_before: d,
				winner_share_named: 48 + 4 * i,
				top_candidate_id: favourites[0],
				top_is_winner: true
			}))
		}
	]
};
const summary = {
	asof: ASOF,
	n_draws: N_DRAWS,
	conditions: [mixCondition, ...dimConditions, ...configConditions],
	mixture: { condition_id: 'mix', candidates: mix.candidates },
	by_condition: byCondition,
	expectations
};

// --- first_round.json
const frConditions = [
	mixCondition,
	...dimConditions.filter((c) =>
		['rn.bardella', 'centre.philippe_attal', 'left.consolidated', 'lr.rally'].includes(c.id)
	),
	...configConditions.slice(0, 3)
];
const series = [];
for (const cond of frConditions) {
	const outcome =
		cond.id === 'mix' ? summary.mixture : byCondition.find((o) => o.condition_id === cond.id);
	// Taille du fichier : candidats présents au moins une fois sur dix ; sous une condition
	// autre que `mix`, les dix premiers dans l'ordre du résumé.
	const kept =
		cond.id === 'mix'
			? outcome.candidates.filter((c) => c.p_present >= 0.1)
			: outcome.candidates.filter((c) => c.p_present >= 0.1).slice(0, 10);
	const tracks = trajectories(
		cond,
		kept.map((c) => c.candidate_id)
	);
	for (const c of kept) {
		series.push({
			condition_id: cond.id,
			candidate_id: c.candidate_id,
			...tracks[c.candidate_id],
			election_day: c.first_round_share
		});
	}
}
function compatible(ids, cond) {
	const set = new Set(ids);
	return Object.entries(cond.states).every(([dimId, stateId]) => {
		const dim = dims.find((d) => d.id === dimId);
		const chosen = dim.states.find((s) => s.id === stateId);
		const need = new Set(chosen.candidates ?? []);
		const others = dim.states.flatMap((s) => s.candidates ?? []).filter((id) => !need.has(id));
		return [...need].every((id) => set.has(id)) && others.every((id) => !set.has(id));
	});
}
const firstRound = {
	asof: ASOF,
	election_date: FIRST_ROUND,
	dates: weeks,
	conditions: frConditions,
	series,
	polls: polls.flatMap((p) =>
		p.hypotheses
			.filter((h) => h.round === 1)
			.map((h) => {
				const ids = h.shares.map((s) => s.candidate_id);
				return {
					poll_id: p.poll_id,
					hypothesis_id: h.hypothesis_id,
					pollster: p.pollster,
					fieldwork_end: p.fieldwork_end,
					population: h.population,
					conditions: frConditions
						.filter((c) => c.id !== 'mix' && compatible(ids, c))
						.map((c) => c.id),
					shares: Object.fromEntries(h.shares.map((s) => [s.candidate_id, s.share]))
				};
			})
	)
};

// --- second_round.json
const pairList = [...mix.pairs.values()]
	.map((p) => ({ ...p, p_pair: p.n / N_DRAWS }))
	.filter((p) => p.p_pair >= 0.002)
	.sort((x, y) => y.p_pair - x.p_pair);
const secondRound = {
	asof: ASOF,
	election_date: SECOND_ROUND,
	pairs: pairList.map((p) => ({
		candidate_a: p.a,
		candidate_b: p.b,
		p_pair: round(p.p_pair, 4),
		p_a_wins: round(p.aWins / p.n, 4),
		share_a: quantiles(p.shareA)
	})),
	transfers: pairList
		.filter((p) => p.p_pair >= 0.03)
		.flatMap((p) =>
			Object.entries(p.transfers)
				.filter(([, tr]) => tr.a.length >= 50)
				.map(([from, tr]) => ({
					candidate_a: p.a,
					candidate_b: p.b,
					from,
					to_a: quantiles(tr.a, 0),
					to_b: quantiles(tr.b, 0),
					to_none: quantiles(tr.n, 0)
				}))
		),
	polls: polls.flatMap((p) =>
		p.hypotheses
			.filter((h) => h.round === 2)
			.map((h) => {
				const [x, y] = h.shares;
				const [a, b] = weight[x.candidate_id] >= weight[y.candidate_id] ? [x, y] : [y, x];
				return {
					poll_id: p.poll_id,
					pollster: p.pollster,
					fieldwork_end: p.fieldwork_end,
					candidate_a: a.candidate_id,
					candidate_b: b.candidate_id,
					share_a: a.share
				};
			})
	)
};

// --- scenarios.json (contenu réel du registre)
function sourceRef(s) {
	const refs = scenariosYaml.references ?? {};
	if (refs[s]) return { label: s, href: refs[s] };
	try {
		return { label: new URL(s).hostname.replace(/^www\./, ''), href: s };
	} catch {
		return { label: s, href: s };
	}
}
const scenarios = {
	as_of: String(scenariosYaml.as_of),
	election: scenariosYaml.election,
	dimensions: dims.map((d) => ({
		id: d.id,
		question: d.question,
		review: (d.review ?? []).map(String),
		states: d.states.map((s) => ({
			id: s.id,
			probability: s.probability,
			candidates: s.candidates ?? [],
			placeholders: s.placeholders ?? [],
			description: s.description,
			sources: (s.sources ?? []).map(sourceRef),
			condition_id: `${d.id}.${s.id}`
		}))
	})),
	small_candidates: smalls.map((s) => ({
		candidate_id: s.candidate_id,
		probability: s.probability,
		bloc: s.bloc,
		rationale: s.rationale,
		sources: (s.sources ?? []).map(sourceRef)
	})),
	constraints: constraints.map((c) => ({
		id: c.id,
		when: c.when,
		absent: c.absent ?? [],
		impossible: c.impossible ?? false,
		rationale: c.rationale
	})),
	revisions: [...scenariosYaml.revisions]
		.map((r) => ({
			date: String(r.date),
			author: r.author,
			summary: r.summary,
			rationale: r.rationale
		}))
		.sort((a, b) => (a.date < b.date ? 1 : -1)),
	configurations: configurations.slice(0, 12).map((c) => ({
		condition_id: configId(c.states),
		states: c.states,
		probability: round(c.probability, 4),
		candidates: c.named,
		placeholders: c.placeholders,
		small_presence: c.smallPresence
	})),
	n_configurations: configurations.length,
	impossible_mass: round(impossibleMass, 4),
	probability_history: [...scenariosYaml.revisions]
		.map((r) => String(r.date))
		.sort()
		.map((date) => ({
			date,
			probabilities: Object.fromEntries(
				dims.flatMap((d) => d.states.map((s) => [`${d.id}.${s.id}`, s.probability]))
			)
		}))
};

// --- polls.json
const pollCount = Object.fromEntries(
	pollsterIds.map((p) => [p, polls.filter((x) => x.pollster === p).length])
);
const blocShare = { far_right: 40, right: 12, centre: 25, left: 15, far_left: 20, other: 5 };
const pollsJson = {
	asof: ASOF,
	pollsters: pollsterIds.map((id) => ({ id, name: pollstersYaml[id]?.name ?? id })),
	polls,
	house_effects: pollsterIds.flatMap((p) =>
		blocs
			.filter((b) => b !== 'other')
			.map((b) => {
				const mid = blocShare[b] * (Math.exp(house[p][b]) - 1);
				const sd = 1.2 / Math.sqrt(1 + pollCount[p]);
				const z = { q05: -1.645, q25: -0.674, q50: 0, q75: 0.674, q95: 1.645 };
				return {
					pollster: p,
					bloc: b,
					effect: Object.fromEntries(
						Object.entries(z).map(([k, v]) => [k, round(mid + v * sd, 1)])
					),
					n_polls: pollCount[p]
				};
			})
	),
	pollster_ratings: {
		in_forecast: false,
		note: "Classement non reproductible d'une élection à l'autre, n'entre pas dans la prévision.",
		elections: [2012, 2017, 2022],
		rows: pollsterIds.map((p) => ({
			pollster: p,
			name: pollstersYaml[p]?.name ?? p,
			elections: [2017, 2022],
			n_polls: 20,
			n_final: 5,
			mae_candidate: round(1 + 0.2 * normal(), 2),
			mae_bloc: round(2 + 0.3 * normal(), 2),
			relative_error: round(0.2 * normal(), 2),
			bias: { far_right: 0, right: 0, centre: 0, left: 0, far_left: 0 },
			precision: round(0.2 * normal(), 2),
			n_better: 1,
			n_rated: 2
		})),
		stability: [{ id: 'precision', label: 'Précision relative', n_pairs: 24, pearson: 0.1, p: 0.6 }]
	},
	dispersion: (() => {
		const dates = Array.from({ length: 12 }, (_, i) => addDays(ASOF, -7 * (11 - i)));
		const D = dates.map((_, i) => (i % 4 === 0 ? null : round(2.5 + rand() * 2.5, 2)));
		return {
			in_forecast: false,
			note: 'Données factices.',
			reading: 'Données factices.',
			window_days: 30,
			weeks: dates,
			D,
			reference: dates.map(() => 2),
			n_pollsters: D.map((d) => (d === null ? 2 : 5)),
			current: {
				asof: ASOF,
				horizon_days: daysBetween(ASOF, FIRST_ROUND),
				D: D.at(-1),
				reference: 2,
				pollsters: pollsterIds.slice(0, 5)
			},
			past_same_horizon: [
				{ election: 2017, D: null },
				{ election: 2022, D: 2.4 }
			]
		};
	})(),
	turnout_gap: {
		in_forecast: false,
		note: 'Données factices.',
		history: 'Données factices.',
		pollsters: ['OpinionWay'],
		n_polls: 6,
		first_poll: addDays(ASOF, -150),
		last_poll: addDays(ASOF, -6),
		rows: favourites
			.map((id, i) => ({ id, gap: round(-0.8 + 0.8 * i, 2) }))
			.map(({ id, gap }) => ({
				candidate_id: id,
				gap,
				se: 0.2,
				share_certain: round(20 + gap),
				share_registered: 20,
				n_polls: 6,
				last_poll: addDays(ASOF, -6)
			}))
	}
};

// --- history/
const historyRuns = [];
for (let d = '2026-06-01'; d <= '2026-10-12'; d = addDays(d, 7)) historyRuns.push(d);
historyRuns.push(RUN_DATE);
const history = historyRuns.map((run) => {
	const pollsBefore = polls.filter((p) => p.fieldwork_end <= run);
	const sim =
		run === RUN_DATE
			? mix
			: simulate({
					t: weekIndex(run),
					tau: daysBetween(run, FIRST_ROUND),
					pool: configurations,
					nDraws: 3000
				});
	return {
		run_date: run,
		asof: run,
		model_version: '0.0.0-mock',
		last_poll_date: pollsBefore.reduce((m, p) => (p.fieldwork_end > m ? p.fieldwork_end : m), ''),
		n_polls: pollsBefore.length,
		candidates: sim.candidates.map(({ candidate_id, p_qualify, p_win, first_round_share }) => ({
			candidate_id,
			p_qualify,
			p_win,
			first_round_share
		}))
	};
});

// --- economy.json, press.json, backtest.json
function monthlySeries(start, n, f) {
	const pts = [];
	for (let i = 0; i < n; i++) {
		const d = toDate(start);
		d.setUTCMonth(d.getUTCMonth() + i);
		pts.push({ date: iso(d), value: round(f(i), 1) });
	}
	return pts;
}
let walk = 0;
const economy = {
	asof: ASOF,
	in_forecast: false,
	in_forecast_note:
		"Ces séries n'entrent pas dans la prévision : aucune n'a encore passé le backtest (docs/07-evaluation.md). Elles sont montrées à titre explicatif.",
	series: [
		{
			id: 'unemployment_rate_ilo',
			label: 'Taux de chômage au sens du BIT (CVS)',
			unit: '%',
			frequency: 'quarterly',
			source: {
				label: 'Insee',
				href: 'https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/001688527'
			},
			in_forecast: false,
			points: [0, 3, 6, 9, 12, 15, 18].map((m, i) => {
				const d = toDate('2025-01-01');
				d.setUTCMonth(d.getUTCMonth() + m);
				return { date: iso(d), value: round(7.3 + 0.08 * i + 0.1 * normal(), 1) };
			})
		},
		{
			id: 'consumer_confidence',
			label: 'Confiance des ménages (indicateur synthétique, CVS)',
			unit: 'indice, moyenne = 100',
			frequency: 'monthly',
			source: { label: 'Insee', href: 'https://www.insee.fr/fr/statistiques/serie/001759970' },
			in_forecast: false,
			points: monthlySeries('2025-01-01', 21, () => 89 + (walk += 0.8 * normal()))
		},
		{
			id: 'approval_president',
			label: 'Satisfaits du président de la République (Ifop)',
			unit: '%',
			frequency: 'monthly',
			source: { label: 'Ifop', href: 'https://www.ifop.com/' },
			in_forecast: false,
			points: monthlySeries('2025-01-01', 21, (i) => 24 - 0.15 * i + 1.5 * normal())
		}
	],
	correlations: ['far_right', 'right', 'centre'].map((b, i) => ({
		series_id: 'unemployment_rate_ilo',
		series_label: 'Chômage BIT',
		target: b,
		target_label: { far_right: 'Extrême droite', right: 'Droite', centre: 'Centre' }[b],
		lag_months: i + 1,
		effect: { q05: -0.6, q25: -0.3, q50: 0, q75: 0.3, q95: 0.6 },
		p_placebo: 0.5,
		n: 80,
		n_cycles: 6
	})),
	correlations_note: 'Données factices.',
	study: {
		date: RUN_DATE,
		verdict: 'negative',
		summary: 'Données factices : aucune étude publiée.',
		findings: [],
		reference: 'docs/05-economie.md'
	}
};
const mockWeeks = Array.from({ length: 12 }, (_, i) => addDays('2026-07-27', 7 * i));
const mockMonths = Array.from({ length: 6 }, (_, i) => `2026-0${4 + i}-01`);
// presse classée factice : six candidats, douze semaines, masquage sous 10 titres
const nlpCands = roster.slice(0, 6);
const stanceIds = ['announcement', 'programme', 'support', 'attack', 'scandal', 'poll', 'other'];
const stanceLabels = [
	'Annonce',
	'Programme',
	'Soutien',
	'Attaques subies',
	'Affaires',
	'Sondages',
	'Autre'
];
const mockN = (ci) => mockWeeks.map(() => Math.max(0, Math.round(80 / (ci + 1) + 10 * normal())));
const nByCand = nlpCands.map((_, ci) => mockN(ci));
function partition(k) {
	const w = Array.from({ length: k }, () => rand() + 0.05);
	const t = w.reduce((a, v) => a + v, 0);
	const out = w.map((v) => round(v / t, 3));
	out[k - 1] = round(1 - out.slice(0, -1).reduce((a, v) => a + v, 0), 3);
	return out;
}
const normalise = (xs) => {
	const t = xs.reduce((a, v) => a + v, 0);
	return xs.map((v) => round(v / t, 4));
};
const sovRaw = mockWeeks.map(() => normalise(nlpCands.map(() => rand() + 0.1)));
const attRaw = mockWeeks.map(() => normalise(nlpCands.map(() => rand() + 0.1)));
const lastMockWeek = mockWeeks.at(-1);
const landmarkDates = (day) => Array.from({ length: 71 }, (_, i) => addDays(day, i - 28));
const mockLandmark = (id, cid, cycle, day) => {
	const dates = landmarkDates(day);
	return {
		id,
		label: `Repère factice ${cycle}`,
		candidate_id: cid,
		cycle,
		event_date: day,
		summary: 'Données factices.',
		dates,
		tone: dates.map((d) => round(Math.max(-1, Math.min(1, d < day ? -0.1 : -0.6)), 2)),
		scandal_share: dates.map((d) => (d < day ? 0.02 : 0.6)),
		n: dates.map(() => 30),
		breaks: { polls: addDays(day, 2), tone: addDays(day, 1), scandal: day }
	};
};
const headline = (cid, k, sign) => ({
	title: `Titre factice ${sign > 0 ? 'favorable' : 'hostile'} ${k + 1}`,
	outlet: 'exemple.fr',
	date: addDays(lastMockWeek, 6 - k),
	url: `https://example.org/${cid}/${sign > 0 ? 'pos' : 'neg'}/${k + 1}`,
	stance: sign > 0 ? 'support' : 'attack'
});
const press = {
	asof: ASOF,
	in_forecast: false,
	in_forecast_note: "La presse n'entre pas dans la prévision (données factices).",
	nlp: {
		in_forecast: false,
		note: 'Données factices.',
		verdict: ['Données factices : aucune étude publiée.'],
		out_of_sample_test: 'Données factices.',
		reference: 'docs/06-presse.md'
	},
	tone: {
		note: 'Données factices.',
		window_weeks: 2,
		min_titles: 10,
		weeks: mockWeeks,
		tracks: nlpCands.map((cid, ci) => {
			const n = nByCand[ci];
			const mid = n.map((k) => (k < 10 ? null : round(0.3 * normal() * 0.5, 2)));
			const half = n.map((k) => (k < 10 ? null : round(1 / Math.sqrt(k), 2)));
			return {
				candidate_id: cid,
				q05: mid.map((m, i) => (m === null ? null : round(Math.max(-1, m - half[i]), 2))),
				q50: mid,
				q95: mid.map((m, i) => (m === null ? null : round(Math.min(1, m + half[i]), 2))),
				n
			};
		})
	},
	stance: {
		note: 'Données factices.',
		categories: stanceIds.map((id, i) => ({ id, label: stanceLabels[i] })),
		min_titles: 10,
		last_weeks: {
			start: mockWeeks.at(-4),
			end: addDays(lastMockWeek, 6),
			rows: nlpCands.map((cid, ci) => {
				const n = nByCand[ci].slice(-4).reduce((a, v) => a + v, 0);
				return { candidate_id: cid, n, shares: n < 10 ? null : partition(stanceIds.length) };
			})
		},
		weeks: mockWeeks,
		tracks: nlpCands.map((cid, ci) => ({
			candidate_id: cid,
			n: nByCand[ci],
			shares: nByCand[ci].map((k) => (k < 10 ? null : partition(stanceIds.length)))
		}))
	},
	headlines: {
		note: 'Données factices.',
		week: lastMockWeek,
		week_end: addDays(lastMockWeek, 6),
		candidates: nlpCands.map((cid, ci) => ({
			candidate_id: cid,
			n: nByCand[ci].at(-1),
			negative: [0, 1, 2].map((k) => headline(cid, k, -1)),
			positive: ci % 2 ? [] : [0, 1].map((k) => headline(cid, k, 1))
		}))
	},
	share_of_voice: {
		note: 'Données factices.',
		weeks: mockWeeks,
		tracks: nlpCands.map((cid, ci) => ({ candidate_id: cid, values: sovRaw.map((w) => w[ci]) })),
		attention: nlpCands.map((cid, ci) => ({ candidate_id: cid, values: attRaw.map((w) => w[ci]) }))
	},
	agenda_nlp: {
		note: 'Données factices.',
		min_titles: 10,
		weeks: mockWeeks,
		n: mockWeeks.map(() => 400),
		themes: [
			['immigration', 'Immigration'],
			['purchasing_power', "Pouvoir d'achat"],
			['international', 'International'],
			['security', 'Sécurité']
		].map(([id, label]) => ({
			id,
			label,
			values: mockWeeks.map(() => round(Math.max(0, 0.04 + 0.02 * normal()), 4))
		}))
	},
	landmarks: {
		note: 'Données factices.',
		events: [
			mockLandmark('mock_2017', roster[0], 2017, '2017-01-25'),
			mockLandmark('mock_2022', roster[1], 2022, '2022-02-13')
		]
	},
	quality: {
		model: 'factice',
		model_label: 'modèle factice',
		prompt_version: 'mock',
		n_sample: 300,
		tone_kappa: 0.6,
		stance_kappa: 0.74,
		themes_f1: 0.55,
		thresholds: { tone_kappa: 0.6, stance_kappa: 0.5, themes_f1: 0.6 },
		passes: { tone_kappa: true, stance_kappa: true, themes_f1: false },
		labeled: [{ cycle: 2027, n: 1000 }],
		reference: 'runs/mock/report.md'
	},
	attention: {
		in_forecast: false,
		note: 'Données factices.',
		source: 'Wikipédia (pages vues)',
		weeks: mockWeeks,
		tracks: roster.slice(0, 5).map((id) => ({
			candidate_id: id,
			values: mockWeeks.map(() => round(Math.max(0, 10 + 4 * normal()), 2))
		}))
	},
	ina: {
		in_forecast: false,
		note: 'Données factices.',
		source: 'INA',
		years: [2002, 2007, 2012, 2017],
		all_years_label: '2000-2020',
		topics: ['Société', 'International', 'Politique France'].map((label) => ({
			label,
			shares: [2002, 2007, 2012, 2017].map(() => round(15 + 3 * normal(), 1)),
			share_all_years: 15
		}))
	},
	gdelt: {
		in_forecast: false,
		note: 'Données factices.',
		source: 'GDELT',
		months: mockMonths,
		tracks: roster.slice(0, 2).map((id) => ({
			candidate_id: id,
			values: mockMonths.map(() => round(Math.max(0, 0.02 + 0.005 * normal()), 3))
		}))
	}
};
const backtestModels = [
	{
		id: 'model',
		label: 'Modèle complet',
		kind: 'model',
		description:
			'Agrégation bayésienne, effets maison, modèle d’erreur calibré hors élection évaluée.'
	},
	{
		id: 'poll_avg_14d',
		label: 'Moyenne 14 jours',
		kind: 'baseline',
		description: 'Moyenne simple des sondages des 14 derniers jours.'
	},
	{
		id: 'last_by_pollster',
		label: 'Dernier par institut',
		kind: 'baseline',
		description: 'Dernier sondage de chaque institut, moyenné.'
	},
	{
		id: 'no_house_effects',
		label: 'Sans effets maison',
		kind: 'ablation',
		description: 'Modèle sans effets maison ni prior structurel.'
	}
];
const backtest = {
	generated_at: `${RUN_DATE}T06:00:00Z`,
	model_version: '0.0.0-mock',
	elections: [2012, 2017, 2022],
	horizons_days: [180, 90, 30, 7],
	models: backtestModels,
	variables_tested: [
		{
			id: 'mock',
			label: 'Variable factice',
			verdict: 'non admise',
			admitted: false,
			published: false,
			summary: 'Données factices.',
			wave: RUN_DATE
		}
	],
	variables_tested_reference: 'docs/backtest-log.md',
	rows: backtestModels.flatMap((m, mi) =>
		[2012, 2017, 2022].flatMap((e) =>
			[180, 90, 30, 7].map((h) => {
				const k = Math.sqrt(h / 180);
				const pen = 1 + 0.12 * mi;
				return {
					model_id: m.id,
					election: e,
					horizon_days: h,
					log_score_pair: round(-(0.4 + 1.6 * k) * pen + 0.1 * normal(), 3),
					log_score_winner: round(-(0.2 + 0.6 * k) * pen + 0.05 * normal(), 3),
					brier_qualify: round((0.02 + 0.08 * k) * pen, 3),
					crps_first_round: round((0.6 + 1.6 * k) * pen, 2),
					mae_first_round: round((1 + 2.4 * k) * pen, 2),
					coverage_50: m.kind === 'baseline' ? null : round(Math.min(1, 0.5 + 0.08 * normal()), 2),
					coverage_90: m.kind === 'baseline' ? null : round(Math.min(1, 0.88 + 0.04 * normal()), 2)
				};
			})
		)
	)
};

// --- manifest.json
const manifest = {
	schema_version: '1',
	run_date: RUN_DATE,
	generated_at: `${RUN_DATE}T06:00:00Z`,
	asof: ASOF,
	model_version: '0.0.0-mock',
	commit: null,
	last_poll_date: polls[0].fieldwork_end,
	n_polls: polls.length,
	polls_hash: null,
	n_draws: N_DRAWS,
	mock: true,
	election: { year: 2027, first_round: FIRST_ROUND, second_round: SECOND_ROUND },
	candidates: manifestCandidates
};

// --- écriture
await rm(path.join(dataDir, 'history'), { recursive: true, force: true });
await mkdir(path.join(dataDir, 'history'), { recursive: true });
const write = (rel, obj) => writeFile(path.join(dataDir, rel), JSON.stringify(obj) + '\n');
await write('manifest.json', manifest);
await write('summary.json', summary);
await write('first_round.json', firstRound);
await write('second_round.json', secondRound);
await write('scenarios.json', scenarios);
await write('polls.json', pollsJson);
await write('economy.json', economy);
await write('press.json', press);
await write('backtest.json', backtest);
for (const h of history) await write(`history/${h.run_date}.json`, h);
await write('history/index.json', { entries: history });
console.log(
	`données factices écrites : ${polls.length} sondages, ${configurations.length} configurations, ${series.length} séries, ${history.length} runs d'historique`
);
