// Registre des candidats affichés (manifest.json) et couleurs. La couleur suit l'entité
// (le candidat ou la candidature qu'il remplace), jamais son rang. Huit teintes validées en
// clair et en sombre ; les autres candidats sont en gris et toujours étiquetés.
import type { Bloc, Candidate, Manifest } from '#lib/types/data.js';

type Pair = readonly [light: string, dark: string];

const BLUE: Pair = ['#2a78d6', '#3987e5'];
const RED: Pair = ['#e34948', '#e66767'];
const AQUA: Pair = ['#1baf7a', '#199e70'];
const MAGENTA: Pair = ['#e87ba4', '#d55181'];
const YELLOW: Pair = ['#eda100', '#c98500'];
const VIOLET: Pair = ['#6250d6', '#9085e9'];
const GREEN: Pair = ['#008300', '#008300'];
const ORANGE: Pair = ['#eb6834', '#d95926'];
const GREY: Pair = ['#898781', '#898781'];

const byCandidate: Record<string, Pair> = {
	le_pen_marine: BLUE,
	bardella_jordan: BLUE,
	melenchon_jean_luc: RED,
	bompard_manuel: RED,
	philippe_edouard: AQUA,
	glucksmann_raphael: MAGENTA,
	attal_gabriel: YELLOW,
	retailleau_bruno: VIOLET,
	tondelier_marine: GREEN,
	zemmour_eric: ORANGE,
	knafo_sarah: ORANGE
};

export const BLOC_LABELS: Record<Bloc, string> = {
	far_right: 'Extrême droite',
	right: 'Droite',
	centre: 'Centre',
	left: 'Gauche',
	far_left: 'Gauche radicale',
	other: 'Autres'
};

/** Couleur CSS du candidat, adaptée au thème (`light-dark`). */
export function candidateColor(id: string): string {
	const [light, dark] = byCandidate[id] ?? GREY;
	return `light-dark(${light}, ${dark})`;
}

export function hasOwnColor(id: string): boolean {
	return id in byCandidate;
}

export interface CandidateRegistry {
	get(id: string): Candidate | undefined;
	name(id: string): string;
	short(id: string): string;
	color(id: string): string;
}

export function candidateRegistry(manifest: Manifest): CandidateRegistry {
	const map = new Map(manifest.candidates.map((c) => [c.id, c]));
	return {
		get: (id) => map.get(id),
		name: (id) => map.get(id)?.name ?? id,
		short: (id) => map.get(id)?.short_name ?? id,
		color: candidateColor
	};
}
