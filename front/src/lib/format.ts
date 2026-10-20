// Mise en forme française des nombres, probabilités, intervalles et dates.
import type { Interval } from '#lib/types/data.js';

const NNBSP = ' ';
const num1 = new Intl.NumberFormat('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const num0 = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 });
const signed1 = new Intl.NumberFormat('fr-FR', {
	minimumFractionDigits: 1,
	maximumFractionDigits: 1,
	signDisplay: 'exceptZero'
});

/** Probabilité (0–1) en pourcentage arrondi, bornée pour ne jamais afficher 0 % ou 100 % à tort. */
export function prob(p: number): string {
	if (p <= 0) return `0${NNBSP}%`;
	if (p < 0.005) return `<${NNBSP}1${NNBSP}%`;
	if (p >= 1) return `100${NNBSP}%`;
	if (p > 0.995) return `>${NNBSP}99${NNBSP}%`;
	return `${num0.format(p * 100)}${NNBSP}%`;
}

/** Part en points d'exprimés (0–100). */
export function share(x: number): string {
	return `${num1.format(x)}${NNBSP}%`;
}

/** Nombre décimal sans unité. */
export function decimal(x: number, digits = 1): string {
	return new Intl.NumberFormat('fr-FR', {
		minimumFractionDigits: digits,
		maximumFractionDigits: digits
	}).format(x);
}

/** Écart signé en points (« +1,2 »). */
export function points(x: number): string {
	return signed1.format(x);
}

/** Médiane et intervalle à 90 % : « 28,5 % (20,5 à 38,3) ». */
export function interval(q: Interval, fmt: (x: number) => string = share): string {
	return `${fmt(q.q50)} (${num1.format(q.q05)} à ${num1.format(q.q95)})`;
}

export function integer(n: number): string {
	return num0.format(n);
}

const dateLong = new Intl.DateTimeFormat('fr-FR', {
	day: 'numeric',
	month: 'long',
	year: 'numeric',
	timeZone: 'UTC'
});
const dateShort = new Intl.DateTimeFormat('fr-FR', {
	day: 'numeric',
	month: 'short',
	timeZone: 'UTC'
});
const dateNumeric = new Intl.DateTimeFormat('fr-FR', { dateStyle: 'short', timeZone: 'UTC' });

export const parseDate = (iso: string): Date => new Date(`${iso.slice(0, 10)}T00:00:00Z`);

/** « 16 octobre 2026 » */
export function date(iso: string): string {
	return dateLong.format(parseDate(iso));
}
/** « 16 oct. » */
export function dateShortFmt(iso: string | Date): string {
	return dateShort.format(typeof iso === 'string' ? parseDate(iso) : iso);
}
/** « 16/10/2026 » */
export function dateNum(iso: string): string {
	return dateNumeric.format(parseDate(iso));
}

/** Nombre de jours de `from` à `to` (dates ISO). */
export function daysBetween(from: string, to: string): number {
	return Math.round((parseDate(to).getTime() - parseDate(from).getTime()) / 86_400_000);
}
