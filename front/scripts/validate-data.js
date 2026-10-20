// Valide static/data/ contre le contrat JSON (back/src/vvs/export/schemas/), puis vérifie les
// invariants que JSON Schema n'exprime pas (longueurs alignées, quantiles croissants, sommes
// de probabilités, références entre fichiers). Usage : npm run validate-data [-- <dossier>]
import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { dataDir as defaultDataDir, dataFiles, historyEntrySchema, schemasDir } from './schemas.js';

const dataDir = path.resolve(process.argv[2] ?? defaultDataDir);
const errors = [];
const fail = (where, msg) => errors.push(`${where} : ${msg}`);
const readJson = async (p) => JSON.parse(await readFile(p, 'utf8'));

// --- schémas
const ajv = new Ajv2020({ allErrors: true, strict: true });
addFormats(ajv);
const schemaIds = {};
for (const name of (await readdir(schemasDir)).filter((f) => f.endsWith('.json'))) {
	const schema = await readJson(path.join(schemasDir, name));
	ajv.addSchema(schema);
	schemaIds[name] = schema.$id;
}

function validate(rel, schemaName, data) {
	const check = ajv.getSchema(schemaIds[schemaName]);
	if (!check) throw new Error(`schéma introuvable : ${schemaName}`);
	if (!check(data)) {
		for (const e of check.errors.slice(0, 20)) fail(rel, `${e.instancePath || '/'} ${e.message}`);
	}
}

const data = {};
for (const [key, { file, schema }] of Object.entries(dataFiles)) {
	try {
		data[key] = await readJson(path.join(dataDir, file));
	} catch (e) {
		fail(file, `lecture impossible (${e.message})`);
		continue;
	}
	validate(file, schema, data[key]);
}

const historyFiles = (await readdir(path.join(dataDir, 'history')).catch(() => []))
	.filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f))
	.sort();
const historyEntries = [];
for (const f of historyFiles) {
	const entry = await readJson(path.join(dataDir, 'history', f));
	validate(`history/${f}`, historyEntrySchema, entry);
	if (entry.run_date !== f.replace('.json', '')) fail(`history/${f}`, 'run_date ≠ nom du fichier');
	historyEntries.push(entry);
}

if (errors.length) report();

// --- invariants
const EPS = 1e-6;
const near = (a, b, tol) => Math.abs(a - b) <= tol;

function checkIntervals(where, node) {
	if (Array.isArray(node)) return node.forEach((x, i) => checkIntervals(`${where}[${i}]`, x));
	if (node === null || typeof node !== 'object') return;
	if ('q05' in node && typeof node.q05 === 'number') {
		const q = [node.q05, node.q25, node.q50, node.q75, node.q95];
		if (q.some((v, i) => i > 0 && v < q[i - 1] - EPS)) fail(where, `quantiles non croissants ${q}`);
		return;
	}
	for (const [k, v] of Object.entries(node)) checkIntervals(`${where}.${k}`, v);
}
for (const [key, value] of Object.entries(data)) checkIntervals(key, value);

const { manifest, summary, first_round: fr, second_round: sr, scenarios, polls, history } = data;
const candidateIds = new Set(manifest.candidates.map((c) => c.id));
const knownCandidate = (where, id) => {
	if (!candidateIds.has(id)) fail(where, `candidat absent du manifeste : ${id}`);
};

// manifeste et sondages
if (manifest.last_poll_date > manifest.asof) fail('manifest', 'last_poll_date > asof');
const lastPoll = polls.polls.reduce((m, p) => (p.fieldwork_end > m ? p.fieldwork_end : m), '');
if (lastPoll !== manifest.last_poll_date)
	fail(
		'manifest',
		`last_poll_date ${manifest.last_poll_date} ≠ dernier sondage de polls.json ${lastPoll}`
	);
if (manifest.n_polls !== polls.polls.length)
	fail('manifest', `n_polls ${manifest.n_polls} ≠ ${polls.polls.length} sondages dans polls.json`);
for (const f of ['summary', 'first_round', 'second_round', 'polls'])
	if (data[f].asof !== manifest.asof) fail(f, `asof ${data[f].asof} ≠ manifeste ${manifest.asof}`);
const pollsterIds = new Set(polls.pollsters.map((p) => p.id));
for (const p of polls.polls) {
	if (!pollsterIds.has(p.pollster)) fail(`polls/${p.poll_id}`, `institut inconnu ${p.pollster}`);
	if (p.fieldwork_start > p.fieldwork_end) fail(`polls/${p.poll_id}`, 'terrain inversé');
	for (const h of p.hypotheses) {
		h.shares.forEach((s) => knownCandidate(`polls/${p.poll_id}`, s.candidate_id));
		const total = h.shares.reduce((a, s) => a + s.share, 0);
		// Les parts publiées sont arrondies (au point ou au demi-point) : tolérance de 2 points.
		if (total > 102 + EPS) fail(`polls/${p.poll_id}/${h.hypothesis_id}`, `somme ${total} > 102`);
	}
}

// conditions et résultats
function checkConditions(where, conditions) {
	if (conditions[0].id !== 'mix' || conditions[0].kind !== 'mixture')
		fail(where, 'la première condition doit être `mix` (mixture)');
	const ids = new Set();
	for (const c of conditions) {
		if (ids.has(c.id)) fail(where, `condition en double ${c.id}`);
		ids.add(c.id);
	}
	return ids;
}
const summaryConditions = checkConditions('summary', summary.conditions);
for (const set of [summary.mixture, ...summary.by_condition]) {
	const where = `summary/${set.condition_id}`;
	if (!summaryConditions.has(set.condition_id)) fail(where, 'condition non déclarée');
	const pw = set.candidates.reduce((a, c) => a + c.p_win, 0);
	const pq = set.candidates.reduce((a, c) => a + c.p_qualify, 0);
	if (!near(pw, 1, 0.01)) fail(where, `somme des p_win = ${pw.toFixed(4)}`);
	if (!near(pq, 2, 0.02)) fail(where, `somme des p_qualify = ${pq.toFixed(4)}`);
	for (const c of set.candidates) {
		knownCandidate(where, c.candidate_id);
		if (c.p_win > c.p_qualify + EPS || c.p_qualify > c.p_present + EPS)
			fail(where, `${c.candidate_id} : il faut p_win ≤ p_qualify ≤ p_present`);
	}
}
if (summary.mixture.condition_id !== 'mix') fail('summary', 'mixture.condition_id ≠ mix');

// premier tour
const frConditions = checkConditions('first_round', fr.conditions);
for (const c of fr.conditions)
	if (!summaryConditions.has(c.id))
		fail('first_round', `condition ${c.id} absente de summary.json`);
fr.dates.forEach((d, i) => {
	if (i > 0 && d <= fr.dates[i - 1]) fail('first_round', 'dates non strictement croissantes');
});
if (fr.dates.at(-1) > fr.asof) fail('first_round', 'dernière date > asof');
const seen = new Set();
for (const s of fr.series) {
	const where = `first_round/${s.condition_id}/${s.candidate_id}`;
	if (!frConditions.has(s.condition_id)) fail(where, 'condition non déclarée');
	knownCandidate(where, s.candidate_id);
	if (seen.has(`${s.condition_id}|${s.candidate_id}`)) fail(where, 'série en double');
	seen.add(`${s.condition_id}|${s.candidate_id}`);
	const tracks = ['q05', 'q25', 'q50', 'q75', 'q95'].map((k) => s[k]);
	if (tracks.some((t) => t.length !== fr.dates.length)) fail(where, 'longueur ≠ dates');
	for (let i = 0; i < fr.dates.length; i++) {
		const q = tracks.map((t) => t[i]);
		if (q.some((v) => v === null) && q.some((v) => v !== null))
			fail(where, `semaine ${i} partiellement nulle`);
		else if (q[0] !== null && q.some((v, j) => j > 0 && v < q[j - 1] - EPS))
			fail(where, `semaine ${i} : quantiles non croissants`);
	}
}
for (const p of fr.polls) {
	for (const c of p.conditions)
		if (!frConditions.has(c)) fail(`first_round/polls/${p.poll_id}`, `condition inconnue ${c}`);
	Object.keys(p.shares).forEach((id) => knownCandidate(`first_round/polls/${p.poll_id}`, id));
}

// second tour
const pairKeys = new Set();
let pairMass = 0;
for (const p of sr.pairs) {
	const where = `second_round/${p.candidate_a}-${p.candidate_b}`;
	knownCandidate(where, p.candidate_a);
	knownCandidate(where, p.candidate_b);
	if (p.candidate_a === p.candidate_b) fail(where, 'duel d’un candidat avec lui-même');
	const key = [p.candidate_a, p.candidate_b].sort().join('|');
	if (pairKeys.has(key)) fail(where, 'paire en double');
	pairKeys.add(key);
	pairMass += p.p_pair;
}
if (pairMass > 1 + 0.01) fail('second_round', `somme des p_pair = ${pairMass.toFixed(4)} > 1`);
for (const t of sr.transfers) {
	const key = [t.candidate_a, t.candidate_b].sort().join('|');
	if (!pairKeys.has(key)) fail('second_round/transfers', `paire absente ${key}`);
}

// scénarios
for (const d of scenarios.dimensions) {
	const total = d.states.reduce((a, s) => a + s.probability, 0);
	if (!near(total, 1, 1e-6)) fail(`scenarios/${d.id}`, `somme des probabilités = ${total}`);
	for (const s of d.states) {
		s.candidates.forEach((id) => knownCandidate(`scenarios/${d.id}.${s.id}`, id));
		if (!summaryConditions.has(s.condition_id))
			fail(`scenarios/${d.id}.${s.id}`, `condition ${s.condition_id} absente de summary.json`);
	}
}
scenarios.configurations.forEach((c, i, all) => {
	if (i > 0 && c.probability > all[i - 1].probability + EPS)
		fail('scenarios/configurations', 'non triées par probabilité décroissante');
});
const configMass = scenarios.configurations.reduce((a, c) => a + c.probability, 0);
if (configMass > 1 + EPS) fail('scenarios/configurations', `masse ${configMass} > 1`);

// volets explicatifs : séries alignées sur leurs dates, candidats connus, ids uniques
const { economy, press, backtest } = data;
const seriesIds = new Set(economy.series.map((x) => x.id));
for (const c of economy.correlations)
	if (!seriesIds.has(c.series_id))
		fail('economy/correlations', `série absente de economy.series : ${c.series_id}`);
function checkTracks(where, dates, tracks) {
	dates.forEach((d, i) => {
		if (i > 0 && d <= dates[i - 1]) fail(where, 'dates non strictement croissantes');
	});
	if (dates.at(-1) > manifest.asof) fail(where, 'dernière date > asof');
	const ids = new Set();
	for (const t of tracks) {
		knownCandidate(where, t.candidate_id);
		if (ids.has(t.candidate_id)) fail(where, `candidat en double ${t.candidate_id}`);
		ids.add(t.candidate_id);
		if (t.values.length !== dates.length) fail(`${where}/${t.candidate_id}`, 'longueur ≠ dates');
	}
}
checkTracks('press/attention', press.attention.weeks, press.attention.tracks);
checkTracks('press/gdelt', press.gdelt.months, press.gdelt.tracks);
checkTracks('press', press.weeks, [...press.share_of_voice, ...press.tone]);
for (const t of press.ina.topics)
	if (t.shares.length !== press.ina.years.length) fail(`press/ina/${t.label}`, 'longueur ≠ years');
const unique = (where, ids) => {
	const seen = new Set();
	for (const id of ids) {
		if (seen.has(id)) fail(where, `id en double ${id}`);
		seen.add(id);
	}
};
unique(
	'polls/pollster_ratings',
	polls.pollster_ratings.rows.map((r) => r.pollster)
);
unique(
	'backtest/variables_tested',
	backtest.variables_tested.map((v) => v.id)
);
if (
	backtest.variables_tested.some((v) => v.admitted) ||
	[economy, press, polls.pollster_ratings, press.attention, press.ina, press.gdelt].some(
		(x) => x.in_forecast
	)
)
	console.warn(
		'attention : une variable explicative est marquée admise ; vérifier docs/backtest-log.md'
	);

// historique
const indexDates = history.entries.map((e) => e.run_date);
if (indexDates.join() !== historyEntries.map((e) => e.run_date).join())
	fail('history/index.json', 'entrées ≠ fichiers history/<date>.json (ou ordre non croissant)');
history.entries.forEach((e, i) => {
	if (JSON.stringify(e) !== JSON.stringify(historyEntries[i] ?? null))
		fail('history/index.json', `entrée ${e.run_date} différente du fichier`);
});
if (indexDates.at(-1) !== manifest.run_date)
	fail(
		'history/index.json',
		`dernière entrée ${indexDates.at(-1)} ≠ run_date ${manifest.run_date}`
	);

report();

function report() {
	if (errors.length) {
		console.error(
			`${errors.length} erreur(s) dans ${path.relative(process.cwd(), dataDir) || '.'} :`
		);
		for (const e of errors) console.error(`  - ${e}`);
		process.exit(1);
	}
	console.log(
		`static/data valide : ${Object.keys(dataFiles).length} fichiers, ${historyFiles.length} entrées d'historique`
	);
	process.exit(0);
}
