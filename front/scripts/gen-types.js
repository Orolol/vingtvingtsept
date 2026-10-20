// Génère src/lib/types/data.ts depuis les JSON Schema du contrat (back/src/vvs/export/schemas/).
// Un schéma racine synthétique référence tous les fichiers : chaque type partagé (Interval,
// Candidate…) n'est déclaré qu'une fois. Usage : npm run types
import { compile } from 'json-schema-to-typescript';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { dataFiles, frontDir, historyEntrySchema, schemasDir } from './schemas.js';

const properties = Object.fromEntries(
	Object.entries(dataFiles).map(([key, { schema }]) => [key, { $ref: schema }])
);
properties.history_entry = { $ref: historyEntrySchema };

const root = {
	$schema: 'https://json-schema.org/draft/2020-12/schema',
	title: 'DataFiles',
	description: 'Fichiers de static/data/ par clé de chargement (voir src/lib/data.ts).',
	type: 'object',
	additionalProperties: false,
	required: Object.keys(properties),
	properties
};

// Un `$ref` accompagné d'une `description` ferait générer une copie numérotée du type
// (Interval1, IsoDate2…) : on retire ces annotations sœurs avant compilation. Elles restent
// dans les schémas, source de vérité de la documentation.
function stripRefSiblings(node) {
	if (Array.isArray(node)) return node.map(stripRefSiblings);
	if (node === null || typeof node !== 'object') return node;
	if ('$ref' in node) return { $ref: node.$ref };
	return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, stripRefSiblings(v)]));
}

const fileResolver = {
	order: 1,
	canRead: /\.json$/i,
	async read(file) {
		const url = file.url.startsWith('file:') ? new URL(file.url) : file.url;
		return JSON.stringify(stripRefSiblings(JSON.parse(await readFile(url, 'utf8'))));
	}
};

const banner = `/* Fichier généré par scripts/gen-types.js depuis back/src/vvs/export/schemas/.
 * Ne pas modifier à la main : npm run types. */`;

const ts = await compile(root, 'DataFiles', {
	cwd: schemasDir,
	$refOptions: { resolve: { file: fileResolver } },
	bannerComment: banner,
	additionalProperties: false,
	unreachableDefinitions: false,
	strictIndexSignatures: true,
	style: { useTabs: true, singleQuote: true, printWidth: 100 }
});

const outDir = path.join(frontDir, 'src/lib/types');
await mkdir(outDir, { recursive: true });
await writeFile(path.join(outDir, 'data.ts'), ts);
console.log(`types écrits dans ${path.relative(frontDir, outDir)}/data.ts`);
