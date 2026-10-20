<script lang="ts">
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import SeriesChart from '#lib/components/SeriesChart.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const eco = $derived(data.economy);
	const FREQ = { monthly: 'mensuelle', quarterly: 'trimestrielle', weekly: 'hebdomadaire' };
</script>

<PageTitle title="Économie et popularité">
	<p>
		Ce que disent la conjoncture et la popularité de l'exécutif de la dynamique de la campagne. Une
		série n'entre dans la prévision que si elle améliore le backtest (docs/07-evaluation.md).
	</p>
</PageTitle>

<ForecastNote inForecast={eco.in_forecast} note={eco.in_forecast_note} />

{#if eco.series.length}
	<div class="mb-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
		{#each eco.series as s (s.id)}
			<article class="rounded-md border border-line bg-surface p-4">
				<h2 class="text-sm font-medium">{s.label}</h2>
				<p class="mb-2 text-xs text-muted">
					{s.unit}, série {FREQ[s.frequency]} ·
					<a href={s.source.href} class="text-accent" rel="external noopener" target="_blank"
						>{s.source.label}</a
					>
					· {s.in_forecast ? 'entre dans la prévision' : "n'entre pas dans la prévision"}
				</p>
				<SeriesChart points={s.points} unit={s.unit} label={s.label} />
			</article>
		{/each}
	</div>
{:else}
	<p class="mb-10 text-sm text-muted">Aucune série publiée pour l'instant.</p>
{/if}

<section aria-labelledby="titre-correlations">
	<h2 id="titre-correlations" class="mb-3 text-lg font-semibold">Études</h2>
	{#if eco.correlations.length}
		<div class="overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only"
					>Corrélations des séries économiques avec les intentions de vote</caption
				>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-3 py-2 font-medium">Série</th>
						<th scope="col" class="px-3 py-2 font-medium">Cible</th>
						<th scope="col" class="px-3 py-2 text-right font-medium">Horizon</th>
						<th scope="col" class="px-3 py-2 font-medium">Estimation (90 %)</th>
						<th scope="col" class="px-3 py-2 text-right font-medium">n</th>
					</tr>
				</thead>
				<tbody>
					{#each eco.correlations as c (c.series_id + c.target + c.horizon_days)}
						<tr class="border-b border-line last:border-0">
							<td class="px-3 py-2">{c.series_id}</td>
							<td class="px-3 py-2">{c.target}</td>
							<td class="tabular px-3 py-2 text-right">{c.horizon_days} j</td>
							<td class="tabular px-3 py-2">{fmt.interval(c.estimate, (x) => fmt.decimal(x, 2))}</td
							>
							<td class="tabular px-3 py-2 text-right">{c.n}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<p class="text-sm text-muted">
			L'étude économique (phase 6 de la feuille de route) n'a pas encore de résultat publié.
		</p>
	{/if}
</section>
