import { loadData } from '#lib/data.js';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => ({ polls: await loadData(fetch, 'polls') });
