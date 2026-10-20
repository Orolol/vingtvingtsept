<script lang="ts">
	import { candidateRegistry } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import { indexDuels, orient, pairKey } from '#lib/duels.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import DuelMatrix from '#lib/components/DuelMatrix.svelte';
	import CandidateName from '#lib/components/CandidateName.svelte';
	import IntervalStrip from '#lib/components/IntervalStrip.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const sr = $derived(data.second_round);
	const candidates = $derived(candidateRegistry(data.manifest));
	const pollsterName = $derived(new Map(data.polls.pollsters.map((p) => [p.id, p.name])));

	// Candidats de la matrice : ceux qui se qualifient le plus souvent.
	const ids = $derived.by(() => {
		const inPairs = new Set(sr.pairs.flatMap((p) => [p.candidate_a, p.candidate_b]));
		return data.summary.mixture.candidates
			.filter((c) => inPairs.has(c.candidate_id) && c.p_qualify >= 0.01)
			.sort((a, b) => b.p_qualify - a.p_qualify)
			.slice(0, 7)
			.map((c) => c.candidate_id);
	});
	const index = $derived(indexDuels(sr.pairs));
	let selected = $state<string | null>(null);
	const current = $derived((selected && index.get(selected)) || sr.pairs[0]);
	const duel = $derived(current ? orient(current, current.candidate_a) : null);
	const transfers = $derived(
		current
			? sr.transfers
					.filter(
						(t) =>
							pairKey(t.candidate_a, t.candidate_b) ===
							pairKey(current.candidate_a, current.candidate_b)
					)
					.map((t) =>
						t.candidate_a === current.candidate_a
							? t
							: {
									...t,
									candidate_a: t.candidate_b,
									candidate_b: t.candidate_a,
									to_a: t.to_b,
									to_b: t.to_a
								}
					)
			: []
	);
	const pairPolls = $derived(
		current
			? sr.polls
					.filter(
						(p) =>
							pairKey(p.candidate_a, p.candidate_b) ===
							pairKey(current.candidate_a, current.candidate_b)
					)
					.map((p) => ({
						...p,
						share: p.candidate_a === current.candidate_a ? p.share_a : 100 - p.share_a
					}))
			: []
	);
	const coveredMass = $derived(sr.pairs.reduce((a, p) => a + p.p_pair, 0));
</script>

<PageTitle title="Second tour">
	<p>
		Chaque duel possible avec sa probabilité d'avoir lieu et le score attendu s'il a lieu. Les
		scores viennent des reports de voix estimés de chaque électorat de premier tour ; les sondages
		de second tour hypothétiques, très incertains loin du vote, servent à les corriger.
	</p>
</PageTitle>

<section aria-labelledby="titre-matrice" class="mb-10">
	<h2 id="titre-matrice" class="mb-3 text-lg font-semibold">Matrice des duels</h2>
	<div class="rounded-md border border-line bg-surface p-2 sm:p-4">
		<DuelMatrix pairs={sr.pairs} {ids} {candidates} bind:selected />
	</div>
	<p class="mt-1 text-xs text-muted">
		Les {sr.pairs.length} duels exportés couvrent {fmt.prob(coveredMass)} des simulations. Cliquer une
		case pour le détail.
	</p>
</section>

{#if current && duel}
	<section aria-labelledby="titre-duel" class="mb-10">
		<h2 id="titre-duel" class="mb-3 text-lg font-semibold">
			{candidates.name(duel.first)} contre {candidates.name(duel.second)}
		</h2>
		<div class="grid gap-4 md:grid-cols-2">
			<div class="rounded-md border border-line bg-surface p-4 text-sm">
				<p>
					Probabilité que ce duel ait lieu : <strong class="tabular">{fmt.prob(duel.p_pair)}</strong
					>
				</p>
				<dl class="mt-3 space-y-2">
					<div>
						<dt><CandidateName id={duel.first} name={candidates.name(duel.first)} /></dt>
						<dd class="mt-1 flex flex-wrap items-center gap-x-3">
							<IntervalStrip q={duel.share_first} max={100} color={candidates.color(duel.first)} />
							<span class="text-ink-2">victoire : {fmt.prob(duel.p_first_wins)}</span>
						</dd>
					</div>
					{#if current}
						{@const other = orient(current, duel.second)}
						<div>
							<dt><CandidateName id={other.first} name={candidates.name(other.first)} /></dt>
							<dd class="mt-1 flex flex-wrap items-center gap-x-3">
								<IntervalStrip
									q={other.share_first}
									max={100}
									color={candidates.color(other.first)}
								/>
								<span class="text-ink-2">victoire : {fmt.prob(other.p_first_wins)}</span>
							</dd>
						</div>
					{/if}
				</dl>
				<p class="mt-3 text-xs text-muted">Scores conditionnels au duel, en % des exprimés.</p>
			</div>

			<div class="rounded-md border border-line bg-surface p-4 text-sm">
				<h3 class="mb-2 font-medium">Sondages de ce duel</h3>
				{#if pairPolls.length}
					<ul class="space-y-1">
						{#each pairPolls as p (p.poll_id)}
							<li class="flex justify-between gap-2">
								<span
									>{pollsterName.get(p.pollster) ?? p.pollster}, {fmt.dateNum(
										p.fieldwork_end
									)}</span
								>
								<span class="tabular">{candidates.short(duel.first)} {fmt.share(p.share)}</span>
							</li>
						{/each}
					</ul>
				{:else}
					<p class="text-muted">Aucun sondage n'a testé ce duel.</p>
				{/if}
			</div>
		</div>

		<h3 class="mt-6 mb-2 font-medium">Reports de voix estimés</h3>
		{#if transfers.length}
			<div class="overflow-x-auto rounded-md border border-line bg-surface">
				<table class="w-full text-sm">
					<caption class="sr-only"
						>Reports de voix de chaque électorat de premier tour dans ce duel</caption
					>
					<thead class="border-b border-line text-left text-xs text-muted">
						<tr>
							<th scope="col" class="px-3 py-2 font-medium">Électorat de</th>
							<th scope="col" class="px-3 py-2 font-medium">vers {candidates.short(duel.first)}</th>
							<th scope="col" class="px-3 py-2 font-medium">vers {candidates.short(duel.second)}</th
							>
							<th scope="col" class="px-3 py-2 font-medium">abstention, blanc, nul</th>
						</tr>
					</thead>
					<tbody>
						{#each transfers as t (t.from)}
							<tr class="border-b border-line last:border-0">
								<th scope="row" class="px-3 py-2 text-left font-normal whitespace-nowrap">
									<CandidateName id={t.from} name={candidates.name(t.from)} />
								</th>
								<td class="tabular px-3 py-2 whitespace-nowrap">{fmt.interval(t.to_a)}</td>
								<td class="tabular px-3 py-2 whitespace-nowrap">{fmt.interval(t.to_b)}</td>
								<td class="tabular px-3 py-2 whitespace-nowrap">{fmt.interval(t.to_none)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="mt-1 text-xs text-muted">
				En % de l'électorat de premier tour du candidat : médiane (intervalle à 90 %).
			</p>
		{:else}
			<p class="text-sm text-muted">Reports non exportés pour ce duel (probabilité trop faible).</p>
		{/if}
	</section>
{/if}

<section aria-labelledby="titre-liste">
	<h2 id="titre-liste" class="mb-3 text-lg font-semibold">Tous les duels</h2>
	<ul class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
		{#each sr.pairs as d (pairKey(d.candidate_a, d.candidate_b))}
			{@const o = orient(d, d.candidate_a)}
			<li class="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 px-3 py-2">
				<button
					type="button"
					class="text-left hover:underline"
					onclick={() => (selected = pairKey(d.candidate_a, d.candidate_b))}
				>
					{candidates.short(o.first)} – {candidates.short(o.second)}
				</button>
				<span class="tabular text-ink-2">
					{fmt.prob(d.p_pair)} · {candidates.short(o.first)}
					{fmt.interval(o.share_first)}
				</span>
			</li>
		{/each}
	</ul>
</section>
