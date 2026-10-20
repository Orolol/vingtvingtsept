import { loadAll } from '#lib/data.js';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ fetch }) =>
	loadAll(fetch, ['summary', 'history', 'scenarios', 'second_round']);
