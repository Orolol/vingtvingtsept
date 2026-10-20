import { loadAll } from '#lib/data.js';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ fetch }) => loadAll(fetch, ['second_round', 'summary', 'polls']);
