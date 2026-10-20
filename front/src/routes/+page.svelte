<script lang="ts">
	import { resolve } from '$app/paths';
	import { candidateRegistry } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import { orient } from '#lib/duels.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ProbabilityTable from '#lib/components/ProbabilityTable.svelte';
	import HistoryChart from '#lib/components/HistoryChart.svelte';
	import CandidateName from '#lib/components/CandidateName.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const candidates = $derived(candidateRegistry(data.manifest));
	const mixture = $derived(data.summary.mixture.candidates);
	const leader = $derived(mixture[0]);

	let metric = $state<'p_win' | 'p_qualify'>('p_win');
	const historyIds = $derived(
		[...mixture]
			.sort((a, b) => b[metric] - a[metric])
			.slice(0, 6)
			.map((c) => c.candidate_id)
	);
	const topDuels = $derived(data.second_round.pairs.slice(0, 4));
</script>

<PageTitle title="Présidentielle 2027 : la prévision">
	<p>
		Probabilités de qualification et de victoire au {fmt.date(data.manifest.election.first_round)},
		tirées de {fmt.integer(data.summary.n_draws)} simulations qui mélangent les scénarios de candidatures.
		Le premier tour est dans {fmt.daysBetween(
			data.manifest.asof,
			data.manifest.election.first_round
		)} jours : l'incertitude reste large et c'est elle qu'on montre.
	</p>
</PageTitle>

<section aria-labelledby="titre-proba" class="mb-10">
	<div class="mb-3 flex flex-wrap items-baseline justify-between gap-2">
		<h2 id="titre-proba" class="text-lg font-semibold">Probabilités du jour</h2>
		<p class="text-sm text-ink-2">
			Prévision au {fmt.date(data.summary.asof)}, dernier sondage du {fmt.date(
				data.manifest.last_poll_date
			)}
		</p>
	</div>
	{#if leader}
		<p class="mb-4 max-w-3xl">
			En mélangeant les scénarios, <strong>{candidates.name(leader.candidate_id)}</strong> l'emporte
			dans {fmt.prob(leader.p_win)} des simulations et se qualifie dans {fmt.prob(
				leader.p_qualify
			)}.
		</p>
	{/if}
	<ProbabilityTable
		outcomes={mixture}
		{candidates}
		limit={8}
		caption="Probabilités par candidat, mélange des scénarios"
	/>
	<p class="mt-2 text-xs text-muted">
		Présence : probabilité d'être candidat au premier tour selon les scénarios. Qualification et
		victoire incluent le risque de ne pas se présenter. La part de premier tour est conditionnelle à
		la présence. Détail par scénario : <a href={resolve('/scenarios')} class="text-accent"
			>scénarios</a
		>.
	</p>
</section>

<section aria-labelledby="titre-evolution" class="mb-10">
	<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
		<h2 id="titre-evolution" class="text-lg font-semibold">Évolution des probabilités</h2>
		<div
			role="radiogroup"
			aria-label="Probabilité affichée"
			class="flex rounded-md border border-line text-sm"
		>
			{#each [['p_win', 'Victoire'], ['p_qualify', 'Qualification']] as const as [key, label] (key)}
				<button
					type="button"
					role="radio"
					aria-checked={metric === key}
					class={['px-3 py-1', metric === key ? 'bg-ink text-page' : 'text-ink-2']}
					onclick={() => (metric = key)}>{label}</button
				>
			{/each}
		</div>
	</div>
	<HistoryChart entries={data.history.entries} {candidates} {metric} ids={historyIds} />
	<p class="mt-1 text-xs text-muted">
		Un point par run publié ({data.history.entries.length} runs depuis le {fmt.date(
			data.history.entries[0]?.run_date ?? data.manifest.run_date
		)}).
	</p>
</section>

<div class="grid gap-8 md:grid-cols-2">
	<section aria-labelledby="titre-duels">
		<h2 id="titre-duels" class="mb-3 text-lg font-semibold">Duels les plus probables</h2>
		<ul class="divide-y divide-line rounded-md border border-line bg-surface">
			{#each topDuels as d (d.candidate_a + d.candidate_b)}
				{@const o = orient(d, d.candidate_a)}
				<li class="px-3 py-2 text-sm">
					<div class="flex flex-wrap items-baseline justify-between gap-2">
						<span>
							<CandidateName id={o.first} name={candidates.short(o.first)} /> –
							<CandidateName id={o.second} name={candidates.short(o.second)} />
						</span>
						<span class="tabular font-medium">{fmt.prob(d.p_pair)}</span>
					</div>
					<p class="mt-0.5 text-xs text-ink-2">
						Si ce duel a lieu : {candidates.short(o.first)}
						{fmt.interval(o.share_first)}, victoire dans {fmt.prob(o.p_first_wins)} des cas.
					</p>
				</li>
			{/each}
		</ul>
		<p class="mt-2 text-sm">
			<a href={resolve('/second-tour')} class="text-accent">Tous les duels</a>
		</p>
	</section>

	<section aria-labelledby="titre-scenarios">
		<h2 id="titre-scenarios" class="mb-3 text-lg font-semibold">Scénarios de candidatures</h2>
		<ul class="divide-y divide-line rounded-md border border-line bg-surface">
			{#each data.scenarios.dimensions as dim (dim.id)}
				{@const top = [...dim.states].sort((a, b) => b.probability - a.probability)[0]}
				<li class="px-3 py-2 text-sm">
					<p class="text-ink-2">{dim.question}</p>
					<p class="mt-0.5">
						<span class="tabular font-medium">{fmt.prob(top.probability)}</span>
						{top.candidates.map((id) => candidates.short(id)).join(' + ') || 'aucun candidat nommé'}
						{#if dim.states.length > 1}
							<span class="text-muted"
								>· {dim.states.length - 1} autre{dim.states.length > 2 ? 's' : ''} état{dim.states
									.length > 2
									? 's'
									: ''}</span
							>
						{/if}
					</p>
				</li>
			{/each}
		</ul>
		<p class="mt-2 text-sm">
			<a href={resolve('/scenarios')} class="text-accent"
				>Probabilités, justifications et prévisions conditionnelles</a
			>
		</p>
	</section>
</div>
