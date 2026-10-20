import { loadData } from '#lib/data.js';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ fetch }) => ({
	backtest: await loadData(fetch, 'backtest')
});
