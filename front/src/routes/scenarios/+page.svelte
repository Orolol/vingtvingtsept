<script lang="ts">
	import { candidateRegistry, BLOC_LABELS } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ConditionSelect from '#lib/components/ConditionSelect.svelte';
	import ProbabilityTable from '#lib/components/ProbabilityTable.svelte';
	import CandidateName from '#lib/components/CandidateName.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const sc = $derived(data.scenarios);
	const candidates = $derived(candidateRegistry(data.manifest));
	const conditionLabel = $derived(new Map(data.summary.conditions.map((c) => [c.id, c.label])));

	let conditionId = $state('mix');
	const outcome = $derived(
		conditionId === 'mix'
			? data.summary.mixture
			: data.summary.by_condition.find((o) => o.condition_id === conditionId)
	);
	function showCondition(id: string) {
		conditionId = id;
		document.getElementById('prevision-conditionnelle')?.scrollIntoView({ behavior: 'smooth' });
	}
	const stateNames = (ids: string[], placeholders: string[]) =>
		[
			...ids.map((id) => candidates.short(id)),
			...placeholders.map(
				(b) =>
					`un candidat ${BLOC_LABELS[b as keyof typeof BLOC_LABELS].toLowerCase()} non identifié`
			)
		].join(' + ') || 'aucun candidat';
</script>

<PageTitle title="Scénarios de candidatures">
	<p>
		Tant que les candidatures ne sont pas validées (parrainages clos le 12 mars 2027), la prévision
		mélange plusieurs configurations. Leurs probabilités sont des <strong
			>jugements datés et sourcés</strong
		>, pas des estimations statistiques ; chaque révision est publiée. Registre au {fmt.date(
			sc.as_of
		)}.
	</p>
</PageTitle>

<section aria-labelledby="titre-dimensions" class="mb-10">
	<h2 id="titre-dimensions" class="mb-3 text-lg font-semibold">Les incertitudes</h2>
	<div class="grid gap-4 lg:grid-cols-2">
		{#each sc.dimensions as dim (dim.id)}
			<article class="rounded-md border border-line bg-surface p-4">
				<h3 class="font-medium">{dim.question}</h3>
				{#if dim.review.length}
					<p class="mt-0.5 text-xs text-muted">À revoir : {dim.review.join(' ; ')}</p>
				{/if}
				<ul class="mt-3 space-y-3">
					{#each dim.states as s (s.id)}
						<li>
							<div class="flex items-center gap-3">
								<span class="tabular w-12 shrink-0 text-right font-semibold"
									>{fmt.prob(s.probability)}</span
								>
								<span
									class="relative h-2 flex-1 overflow-hidden rounded-full bg-grid"
									aria-hidden="true"
								>
									<span
										class="absolute inset-y-0 left-0 rounded-full bg-accent"
										style:width="{s.probability * 100}%"
									></span>
								</span>
							</div>
							<p class="mt-1 text-sm font-medium">
								{s.candidates.length || s.placeholders.length
									? stateNames(s.candidates, s.placeholders)
									: (conditionLabel.get(s.condition_id)?.split(' : ').pop() ?? s.id)}
							</p>
							<details class="mt-0.5 text-sm text-ink-2">
								<summary class="cursor-pointer text-xs text-accent"
									>Justification et sources</summary
								>
								<p class="mt-1">{s.description}</p>
								{#if s.sources.length}
									<ul class="mt-1 flex flex-wrap gap-x-3 text-xs">
										{#each s.sources as src (src.href)}
											<li>
												{#if src.href.startsWith('http')}
													<a
														href={src.href}
														class="text-accent"
														rel="external noopener"
														target="_blank">{src.label}</a
													>
												{:else}
													<span class="text-muted">{src.label} ({src.href})</span>
												{/if}
											</li>
										{/each}
									</ul>
								{/if}
							</details>
							<button
								type="button"
								class="mt-1 text-xs text-accent hover:underline"
								onclick={() => showCondition(s.condition_id)}
							>
								Prévision si cet état se réalise
							</button>
						</li>
					{/each}
				</ul>
			</article>
		{/each}
	</div>
</section>

<section aria-labelledby="titre-config" class="mb-10">
	<h2 id="titre-config" class="mb-1 text-lg font-semibold">Configurations les plus probables</h2>
	<p class="mb-3 text-sm text-ink-2">
		{sc.n_configurations} configurations possibles après contraintes ({fmt.prob(sc.impossible_mass)} de
		la masse initiale exclue par les contraintes, puis renormalisée). Petits candidats à leur probabilité
		de présence.
	</p>
	<ol class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
		{#each sc.configurations as c, i (c.condition_id)}
			<li class="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-3 py-2">
				<span class="tabular w-6 text-muted">{i + 1}.</span>
				<span class="tabular w-12 font-semibold">{fmt.prob(c.probability)}</span>
				<span class="min-w-0 flex-1">{stateNames(c.candidates, c.placeholders)}</span>
				{#if conditionLabel.has(c.condition_id)}
					<button
						type="button"
						class="text-xs text-accent hover:underline"
						onclick={() => showCondition(c.condition_id)}
					>
						Prévision
					</button>
				{/if}
			</li>
		{/each}
	</ol>
</section>

<section id="prevision-conditionnelle" aria-labelledby="titre-cond" class="mb-10 scroll-mt-4">
	<h2 id="titre-cond" class="mb-3 text-lg font-semibold">Prévision conditionnelle</h2>
	<div class="mb-3">
		<ConditionSelect
			conditions={data.summary.conditions}
			bind:value={conditionId}
			id="condition-scenarios"
		/>
	</div>
	{#if outcome}
		<ProbabilityTable
			outcomes={outcome.candidates}
			{candidates}
			reference={conditionId === 'mix' ? undefined : data.summary.mixture.candidates}
			caption="Probabilités sous la condition choisie"
		/>
	{/if}
</section>

<section aria-labelledby="titre-petits" class="mb-10">
	<h2 id="titre-petits" class="mb-3 text-lg font-semibold">Petits candidats</h2>
	<div class="overflow-x-auto rounded-md border border-line bg-surface">
		<table class="w-full text-sm">
			<caption class="sr-only">Probabilité de présence des petits candidats</caption>
			<thead class="border-b border-line text-left text-xs text-muted">
				<tr>
					<th scope="col" class="px-3 py-2 font-medium">Candidat</th>
					<th scope="col" class="px-3 py-2 font-medium">Bloc</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Présence</th>
					<th scope="col" class="px-3 py-2 font-medium">Pourquoi</th>
				</tr>
			</thead>
			<tbody>
				{#each [...sc.small_candidates].sort((a, b) => b.probability - a.probability) as s (s.candidate_id)}
					<tr class="border-b border-line align-top last:border-0">
						<th scope="row" class="px-3 py-2 text-left font-normal whitespace-nowrap">
							<CandidateName id={s.candidate_id} name={candidates.name(s.candidate_id)} />
						</th>
						<td class="px-3 py-2 whitespace-nowrap text-ink-2">{BLOC_LABELS[s.bloc]}</td>
						<td class="tabular px-3 py-2 text-right">{fmt.prob(s.probability)}</td>
						<td class="min-w-64 px-3 py-2 text-ink-2">{s.rationale}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	{#if sc.constraints.length}
		<h3 class="mt-6 mb-2 font-medium">Contraintes</h3>
		<ul class="list-disc space-y-1 pl-5 text-sm text-ink-2">
			{#each sc.constraints as c (c.id)}
				<li>
					Si {Object.entries(c.when)
						.map(([d, s]) => `${d} = ${s}`)
						.join(' et ')} :
					{c.impossible
						? 'combinaison impossible'
						: `absents : ${c.absent.map((id) => candidates.short(id)).join(', ')}`}. {c.rationale}
				</li>
			{/each}
		</ul>
	{/if}
</section>

<section aria-labelledby="titre-revisions">
	<h2 id="titre-revisions" class="mb-3 text-lg font-semibold">Historique des révisions</h2>
	{#if sc.probability_history.length < 2}
		<p class="mb-3 text-sm text-muted">
			Une seule version publiée pour l'instant : l'évolution des probabilités s'affichera à la
			prochaine révision.
		</p>
	{:else}
		<div class="mb-4 overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only">Probabilité de chaque état à chaque révision</caption>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-3 py-2 font-medium">État</th>
						{#each sc.probability_history as h (h.date)}
							<th scope="col" class="px-3 py-2 text-right font-medium">{fmt.dateNum(h.date)}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each Object.keys(sc.probability_history[sc.probability_history.length - 1].probabilities) as key (key)}
						<tr class="border-b border-line last:border-0">
							<th scope="row" class="px-3 py-1.5 text-left font-normal"
								>{conditionLabel.get(key) ?? key}</th
							>
							{#each sc.probability_history as h (h.date)}
								<td class="tabular px-3 py-1.5 text-right">
									{h.probabilities[key] === undefined ? '—' : fmt.prob(h.probabilities[key])}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
	<ol class="space-y-4">
		{#each sc.revisions as r (r.date + r.author)}
			<li class="rounded-md border border-line bg-surface p-4 text-sm">
				<p class="font-medium">
					{fmt.date(r.date)} <span class="font-normal text-muted">· {r.author}</span>
				</p>
				<p class="mt-1">{r.summary}</p>
				<p class="mt-1 text-ink-2">{r.rationale}</p>
			</li>
		{/each}
	</ol>
</section>
