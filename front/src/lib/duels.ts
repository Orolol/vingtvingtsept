// Lecture orientée des duels de second tour : second_round.json ne stocke chaque paire qu'une
// fois (a, b) ; le score de b se déduit par symétrie (100 − score de a).
import type { Duel, Interval } from '#lib/types/data.js';

export interface OrientedDuel {
	first: string;
	second: string;
	p_pair: number;
	/** Probabilité de victoire de `first`, conditionnelle au duel. */
	p_first_wins: number;
	/** Score de `first`, en points d'exprimés. */
	share_first: Interval;
}

export function mirror(q: Interval): Interval {
	return {
		q05: 100 - q.q95,
		q25: 100 - q.q75,
		q50: 100 - q.q50,
		q75: 100 - q.q25,
		q95: 100 - q.q05
	};
}

export function orient(d: Duel, first: string): OrientedDuel {
	if (d.candidate_a === first)
		return {
			first,
			second: d.candidate_b,
			p_pair: d.p_pair,
			p_first_wins: d.p_a_wins,
			share_first: d.share_a
		};
	return {
		first,
		second: d.candidate_a,
		p_pair: d.p_pair,
		p_first_wins: 1 - d.p_a_wins,
		share_first: mirror(d.share_a)
	};
}

export function pairKey(a: string, b: string): string {
	return a < b ? `${a}|${b}` : `${b}|${a}`;
}

export function indexDuels(pairs: Duel[]): Map<string, Duel> {
	return new Map(pairs.map((d) => [pairKey(d.candidate_a, d.candidate_b), d]));
}
