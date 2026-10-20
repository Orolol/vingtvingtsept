// Liste des schémas du contrat JSON (back/src/vvs/export/schemas/) et du fichier de données
// correspondant sous static/data/. Partagé par gen-types.js et validate-data.js.
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
export const frontDir = path.resolve(here, '..');
export const schemasDir = path.resolve(frontDir, '../back/src/vvs/export/schemas');
export const dataDir = path.resolve(frontDir, 'static/data');

/** Fichier de données → schéma. `history/<date>.json` est traité à part (history_entry). */
export const dataFiles = {
	manifest: { file: 'manifest.json', schema: 'manifest.json' },
	summary: { file: 'summary.json', schema: 'summary.json' },
	first_round: { file: 'first_round.json', schema: 'first_round.json' },
	second_round: { file: 'second_round.json', schema: 'second_round.json' },
	scenarios: { file: 'scenarios.json', schema: 'scenarios.json' },
	polls: { file: 'polls.json', schema: 'polls.json' },
	economy: { file: 'economy.json', schema: 'economy.json' },
	press: { file: 'press.json', schema: 'press.json' },
	backtest: { file: 'backtest.json', schema: 'backtest.json' },
	history: { file: 'history/index.json', schema: 'history_index.json' }
};

export const historyEntrySchema = 'history_entry.json';
