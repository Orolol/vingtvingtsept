import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// Chemin de base du site : vide en local, `/<dépôt>` sur GitHub Pages (BASE_PATH fourni par le
// workflow). Doit commencer par `/` et ne pas finir par `/`.
const base = process.env.BASE_PATH ?? '';
if (base !== '' && (!base.startsWith('/') || base.endsWith('/'))) {
	throw new Error(`BASE_PATH invalide : « ${base} » (attendu : vide ou /chemin sans / final)`);
}

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Mode runes forcé pour le projet, sauf pour les librairies.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// Site entièrement prérendu ; 404.html sert les URL inconnues sur GitHub Pages.
			adapter: adapter({ fallback: '404.html', strict: true }),
			paths: { base: base as '' | `/${string}` },
			prerender: { handleHttpError: 'fail', handleMissingId: 'fail' }
		})
	]
});
