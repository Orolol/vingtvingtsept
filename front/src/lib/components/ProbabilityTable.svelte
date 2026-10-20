<!-- Probabilités par candidat : qualification, victoire, part de premier tour (médiane et
     intervalles). `reference` (facultatif) affiche l'écart à une autre condition (le mélange). -->
<script lang="ts">
	import type { CandidateOutcome } from '#lib/types/data.js';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { prob } from '#lib/format.js';
	import CandidateName from './CandidateName.svelte';
	import IntervalStrip from './IntervalStrip.svelte';
	import ProbBar from './ProbBar.svelte';

	let {
		outcomes,
		candidates,
		limit = 10,
		showPresence = true,
		reference,
		caption
	}: {
		outcomes: CandidateOutcome[];
		candidates: CandidateRegistry;
		limit?: number;
		showPresence?: boolean;
		reference?: CandidateOutcome[];
		caption?: string;
	} = $props();

	let expanded = $state(false);
	const rows = $derived(expanded ? outcomes : outcomes.slice(0, limit));
	const maxShare = $derived(Math.max(10, ...outcomes.map((o) => o.first_round_share.q95)) * 1.05);
	const refById = $derived(new Map((reference ?? []).map((o) => [o.candidate_id, o])));

	function delta(o: CandidateOutcome, key: 'p_win' | 'p_qualify'): string | null {
		const r = refById.get(o.candidate_id);
		if (!r) return null;
		const d = Math.round((o[key] - r[key]) * 100);
		if (d === 0) return '=';
		return `${d > 0 ? '+' : '−'}${Math.abs(d)}`;
	}
</script>

<div class="overflow-x-auto rounded-md border border-line bg-surface">
	<table class="w-full text-sm">
		{#if caption}<caption class="sr-only">{caption}</caption>{/if}
		<thead class="border-b border-line text-left text-xs text-muted">
			<tr>
				<th scope="col" class="px-3 py-2 font-medium">Candidat</th>
				{#if showPresence}<th scope="col" class="px-3 py-2 text-right font-medium">Présence</th
					>{/if}
				<th scope="col" class="px-3 py-2 font-medium">Qualification</th>
				<th scope="col" class="px-3 py-2 font-medium">Victoire</th>
				<th scope="col" class="px-3 py-2 font-medium">
					Premier tour <span class="font-normal">(médiane, 50 % et 90 %)</span>
				</th>
			</tr>
		</thead>
		<tbody>
			{#each rows as o (o.candidate_id)}
				{@const color = candidates.color(o.candidate_id)}
				<tr class="border-b border-line last:border-0">
					<th scope="row" class="px-3 py-2 text-left font-normal whitespace-nowrap">
						<CandidateName id={o.candidate_id} name={candidates.name(o.candidate_id)} />
					</th>
					{#if showPresence}
						<td class="tabular px-3 py-2 text-right text-ink-2">{prob(o.p_present)}</td>
					{/if}
					<td class="px-3 py-2 whitespace-nowrap">
						<ProbBar p={o.p_qualify} {color} />
						{#if reference}<span class="tabular ml-1 text-xs text-muted"
								>{delta(o, 'p_qualify')}</span
							>{/if}
					</td>
					<td class="px-3 py-2 whitespace-nowrap">
						<ProbBar p={o.p_win} {color} />
						{#if reference}<span class="tabular ml-1 text-xs text-muted">{delta(o, 'p_win')}</span
							>{/if}
					</td>
					<td class="px-3 py-2">
						<IntervalStrip q={o.first_round_share} max={maxShare} {color} />
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
	{#if outcomes.length > limit}
		<button
			type="button"
			class="w-full border-t border-line px-3 py-2 text-left text-sm text-accent hover:bg-page"
			onclick={() => (expanded = !expanded)}
			aria-expanded={expanded}
		>
			{expanded ? 'Réduire' : `Afficher les ${outcomes.length} candidats`}
		</button>
	{/if}
</div>
{#if reference}
	<p class="mt-1 text-xs text-muted">
		Écarts en points de probabilité par rapport au mélange des scénarios.
	</p>
{/if}
