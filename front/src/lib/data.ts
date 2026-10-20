// Chargement commun des JSON publiés sous static/data/ (contrat : back/src/vvs/export/schemas/).
// À appeler depuis une fonction `load` avec le `fetch` fourni par SvelteKit : au prérendu, les
// réponses sont lues sur disque et sérialisées dans la page ; en navigation, elles sont
// récupérées sous le chemin de base (`asset`).
import { error } from '@sveltejs/kit';
import { asset } from '$app/paths';
import type { AssetPath } from '$app/types';
import type { DataFiles, HistoryEntry } from '#lib/types/data.js';

export type DataKey = Exclude<keyof DataFiles, 'history_entry'>;

const files: Record<DataKey, string> = {
	manifest: 'data/manifest.json',
	summary: 'data/summary.json',
	first_round: 'data/first_round.json',
	second_round: 'data/second_round.json',
	scenarios: 'data/scenarios.json',
	polls: 'data/polls.json',
	economy: 'data/economy.json',
	press: 'data/press.json',
	backtest: 'data/backtest.json',
	history: 'data/history/index.json'
};

type Fetch = typeof globalThis.fetch;

async function getJson<T>(fetch: Fetch, file: string): Promise<T> {
	const url = asset(file as AssetPath);
	let res: Response;
	try {
		res = await fetch(url);
	} catch (e) {
		error(503, `Données inaccessibles (${file}) : ${(e as Error).message}`);
	}
	if (!res.ok)
		error(res.status === 404 ? 404 : 502, `Données introuvables : ${file} (HTTP ${res.status})`);
	return (await res.json()) as T;
}

/** Charge un fichier de données par sa clé. */
export function loadData<K extends DataKey>(fetch: Fetch, key: K): Promise<DataFiles[K]> {
	return getJson<DataFiles[K]>(fetch, files[key]);
}

/** Charge plusieurs fichiers en parallèle : `loadAll(fetch, ['summary', 'history'])`. */
export async function loadAll<const K extends readonly DataKey[]>(
	fetch: Fetch,
	keys: K
): Promise<{ [I in K[number]]: DataFiles[I] }> {
	const values = await Promise.all(keys.map((k) => loadData(fetch, k)));
	return Object.fromEntries(keys.map((k, i) => [k, values[i]])) as {
		[I in K[number]]: DataFiles[I];
	};
}

/** Charge le résumé archivé d'un run (history/<date>.json). */
export function loadHistoryEntry(fetch: Fetch, runDate: string): Promise<HistoryEntry> {
	if (!/^\d{4}-\d{2}-\d{2}$/.test(runDate)) error(400, `Date de run invalide : ${runDate}`);
	return getJson<HistoryEntry>(fetch, `data/history/${runDate}.json`);
}
