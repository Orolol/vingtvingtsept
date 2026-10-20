<script lang="ts">
	import { candidateRegistry } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import CandidateName from '#lib/components/CandidateName.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const press = $derived(data.press);
	const candidates = $derived(candidateRegistry(data.manifest));
	const last = $derived(press.weeks.length - 1);
	const latestShare = $derived(
		press.share_of_voice
			.map((t) => ({ id: t.candidate_id, v: t.values[last] ?? null }))
			.filter((r): r is { id: string; v: number } => r.v !== null)
			.sort((a, b) => b.v - a.v)
	);
	const toneOf = $derived(new Map(press.tone.map((t) => [t.candidate_id, t.values[last] ?? null])));
	const agenda = $derived(press.agenda[press.agenda.length - 1]);
</script>

<PageTitle title="Presse">
	<p>
		Part de voix des candidats dans la presse, ton des articles et agenda de la semaine. Volet
		explicatif : la presse n'entre dans la prévision que si elle améliore le backtest.
	</p>
</PageTitle>

<ForecastNote inForecast={press.in_forecast} note={press.in_forecast_note} />

{#if press.weeks.length === 0}
	<p class="text-sm text-muted">
		Aucune donnée de presse publiée pour l'instant (ingestion Media Cloud et GDELT prévue en phase
		7).
	</p>
{:else}
	<div class="grid gap-8 md:grid-cols-2">
		<section aria-labelledby="titre-voix">
			<h2 id="titre-voix" class="mb-3 text-lg font-semibold">
				Part de voix, semaine du {fmt.date(press.weeks[last])}
			</h2>
			<ul class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
				{#each latestShare as r (r.id)}
					{@const tone = toneOf.get(r.id)}
					<li class="flex items-center justify-between gap-3 px-3 py-2">
						<CandidateName id={r.id} name={candidates.name(r.id)} />
						<span class="tabular text-ink-2">
							{fmt.share(r.v)}{#if tone !== null && tone !== undefined}
								· ton {fmt.points(tone)}{/if}
						</span>
					</li>
				{/each}
			</ul>
		</section>
		{#if agenda}
			<section aria-labelledby="titre-agenda">
				<h2 id="titre-agenda" class="mb-3 text-lg font-semibold">Agenda de la semaine</h2>
				<ul class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
					{#each agenda.topics as t (t.label)}
						<li class="flex justify-between gap-3 px-3 py-2">
							<span>{t.label}</span><span class="tabular text-ink-2">{fmt.share(t.share)}</span>
						</li>
					{/each}
				</ul>
			</section>
		{/if}
	</div>
{/if}
