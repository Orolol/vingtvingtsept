import { loadAll } from '#lib/data.js';
import type { PageLoad } from './$types';

export const load: PageLoad = ({ fetch }) => loadAll(fetch, ['first_round', 'summary']);
