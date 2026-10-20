import { loadData } from '#lib/data.js';
import type { LayoutLoad } from './$types';

// Site entièrement statique : toutes les pages sont prérendues au build.
export const prerender = true;

export const load: LayoutLoad = async ({ fetch }) => ({
	manifest: await loadData(fetch, 'manifest')
});
