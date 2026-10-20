<script lang="ts">
	import { candidateRegistry } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ConditionSelect from '#lib/components/ConditionSelect.svelte';
	import TrajectoryChart, { type PollPoint } from '#lib/components/TrajectoryChart.svelte';
	import ProbabilityTable from '#lib/components/ProbabilityTable.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const fr = $derived(data.first_round);
	const candidates = $derived(candidateRegistry(data.manifest));

	let conditionId = $state('mix');
	const condition = $derived(fr.conditions.find((c) => c.id === conditionId) ?? fr.conditions[0]);
	const outcome = $derived(
		condition.id === 'mix'
			? data.summary.mixture
			: data.summary.by_condition.find((o) => o.condition_id === condition.id)
	);
	// Ordre des candidats : celui du résumé (victoire, puis qualification), sinon part projetée.
	const rank = $derived(new Map(outcome?.candidates.map((c, i) => [c.candidate_id, i]) ?? []));
	const available = $derived(
		fr.series
			.filter((s) => s.condition_id === condition.id)
			.sort(
				(a, b) =>
					(rank.get(a.candidate_id) ?? Infinity) - (rank.get(b.candidate_id) ?? Infinity) ||
					b.election_day.q50 - a.election_day.q50
			)
	);

	// Sélection de candidats : null = les six premiers de la condition.
	let picked = $state<string[] | null>(null);
	const shown = $derived.by(() => {
		const ids = available.map((s) => s.candidate_id);
		const kept = picked?.filter((id) => ids.includes(id)) ?? [];
		return kept.length ? kept : ids.slice(0, 6);
	});
	function toggle(id: string) {
		const cur = shown;
		picked = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id];
	}

	const series = $derived(available.filter((s) => shown.includes(s.candidate_id)));
	const compatiblePolls = $derived(
		condition.id === 'mix' ? fr.polls : fr.polls.filter((p) => p.conditions.includes(condition.id))
	);
	const points = $derived<PollPoint[]>(
		compatiblePolls.flatMap((p) =>
			shown
				.filter((id) => p.shares[id] !== undefined)
				.map((id) => ({
					date: p.fieldwork_end,
					candidate_id: id,
					share: p.shares[id] as number,
					pollster: p.pollster
				}))
		)
	);
</script>

<PageTitle title="Premier tour">
	<p>
		Soutien estimé de chaque candidat semaine après semaine, avec ses intervalles, les sondages
		publiés, et la part projetée au jour du vote ({fmt.date(fr.election_date)}). L'intervalle
		s'élargit vers le vote : six mois avant, l'ordre des candidats proches n'est pas connu.
	</p>
</PageTitle>

<div class="mb-4 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
	<div class="min-w-0">
		<ConditionSelect conditions={fr.conditions} bind:value={conditionId} />
	</div>
	<p class="text-sm text-ink-2">
		{#if condition.kind === 'mixture'}
			Toutes configurations, pondérées par leur probabilité · {compatiblePolls.length} hypothèses de sondage
		{:else}
			Probabilité du scénario : <strong>{fmt.prob(condition.probability)}</strong> · {compatiblePolls.length}
			hypothèses de sondage compatibles
		{/if}
	</p>
</div>

<fieldset class="mb-4">
	<legend class="mb-1 text-sm text-ink-2">Candidats affichés</legend>
	<div class="flex flex-wrap gap-1.5">
		{#each available as s (s.candidate_id)}
			{@const on = shown.includes(s.candidate_id)}
			<button
				type="button"
				aria-pressed={on}
				onclick={() => toggle(s.candidate_id)}
				class={[
					'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs',
					on ? 'border-ink bg-surface text-ink' : 'border-line text-muted'
				]}
			>
				<span
					class="inline-block size-2 rounded-full"
					style:background={on ? candidates.color(s.candidate_id) : 'transparent'}
					style:border="1px solid {candidates.color(s.candidate_id)}"
				></span>
				{candidates.short(s.candidate_id)}
			</button>
		{/each}
	</div>
</fieldset>

<section class="mb-10 rounded-md border border-line bg-surface p-2 sm:p-4">
	<TrajectoryChart
		dates={fr.dates}
		{series}
		polls={points}
		asof={fr.asof}
		electionDate={fr.election_date}
		{candidates}
	/>
</section>

<section aria-labelledby="titre-table">
	<h2 id="titre-table" class="mb-3 text-lg font-semibold">Probabilités sous ce scénario</h2>
	{#if outcome}
		<ProbabilityTable
			outcomes={outcome.candidates}
			{candidates}
			reference={condition.id === 'mix' ? undefined : data.summary.mixture.candidates}
			caption="Probabilités par candidat sous le scénario choisi"
		/>
	{:else}
		<p class="text-sm text-muted">Pas de résultat exporté pour ce scénario.</p>
	{/if}
</section>
