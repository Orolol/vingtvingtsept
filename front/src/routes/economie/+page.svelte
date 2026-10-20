<script lang="ts">
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import SeriesChart from '#lib/components/SeriesChart.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const eco = $derived(data.economy);
	const FREQ = { monthly: 'mensuelle', quarterly: 'trimestrielle', weekly: 'hebdomadaire' };
	const VERDICT = {
		negative: 'Résultat négatif',
		positive: 'Résultat positif',
		inconclusive: 'Non concluant'
	};

	// Tableau indicateur × bloc, dans l'ordre de l'export.
	const indicators = $derived([
		...new Map(eco.correlations.map((c) => [c.series_id, c.series_label])).entries()
	]);
	const targets = $derived([
		...new Map(eco.correlations.map((c) => [c.target, c.target_label])).entries()
	]);
	const cell = $derived(new Map(eco.correlations.map((c) => [`${c.series_id}|${c.target}`, c])));
	const maxEffect = $derived(Math.max(0.1, ...eco.correlations.map((c) => Math.abs(c.effect.q50))));
	const POS = 'light-dark(#2a78d6, #3987e5)';
	const NEG = 'light-dark(#eb6834, #d95926)';
	const tint = (v: number) =>
		`color-mix(in oklab, ${v >= 0 ? POS : NEG} ${Math.round((50 * Math.abs(v)) / maxEffect)}%, transparent)`;
	const pval = (p: number) => (p < 0.001 ? '< 0,001' : fmt.decimal(p, p < 0.1 ? 3 : 2));
</script>

<PageTitle title="Économie et popularité">
	<p>
		Ce que disent la conjoncture et la popularité de l'exécutif de la dynamique de la campagne. Une
		série n'entre dans la prévision que si elle améliore le backtest (docs/07-evaluation.md).
	</p>
</PageTitle>

<ForecastNote inForecast={eco.in_forecast} note={eco.in_forecast_note} />

<section aria-labelledby="titre-etude" class="mb-10 max-w-3xl">
	<h2 id="titre-etude" class="mb-2 text-lg font-semibold">
		L'étude du {fmt.date(eco.study.date)} : {VERDICT[eco.study.verdict].toLowerCase()}
	</h2>
	<p class="mb-3 leading-relaxed">{eco.study.summary}</p>
	{#if eco.study.findings.length}
		<ul class="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-2">
			{#each eco.study.findings as f (f)}<li>{f}</li>{/each}
		</ul>
	{/if}
	<p class="mt-2 text-xs text-muted">Détail : {eco.study.reference} dans le dépôt.</p>
</section>

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
	<h2 id="titre-correlations" class="mb-1 text-lg font-semibold">
		Corrélations : indicateur et intentions de vote par bloc
	</h2>
	{#if eco.correlations.length}
		<p class="mb-3 max-w-3xl text-sm text-ink-2">{eco.correlations_note}</p>
		<div class="overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only"
					>Effet d'une variation typique de chaque indicateur sur la part de chaque bloc, en points
					par mois</caption
				>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="sticky left-0 bg-surface px-3 py-2 font-medium">Indicateur</th>
						{#each targets as [id, label] (id)}
							<th scope="col" class="px-2 py-2 text-right font-medium">{label}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each indicators as [sid, slabel] (sid)}
						<tr class="border-b border-line last:border-0">
							<th
								scope="row"
								class="sticky left-0 bg-surface px-3 py-1.5 text-left font-normal whitespace-nowrap"
								>{slabel}</th
							>
							{#each targets as [tid] (tid)}
								{@const c = cell.get(`${sid}|${tid}`)}
								<td class="tabular px-1 py-1 text-right">
									{#if c}
										<span
											class={[
												'inline-block min-w-14 rounded px-1.5 py-0.5',
												c.p_placebo < 0.05 && 'font-semibold ring-1 ring-ink/50'
											]}
											style:background={tint(c.effect.q50)}
											title="{fmt.points(c.effect.q50)} pt/mois ({fmt.decimal(
												c.effect.q05,
												2
											)} à {fmt.decimal(
												c.effect.q95,
												2
											)}), retard {c.lag_months} mois, p placebo {pval(c.p_placebo)}, n = {c.n}"
											>{fmt.decimal(c.effect.q50, 2)}{#if c.p_placebo < 0.05}<span class="sr-only">
													(passe le placebo)</span
												>{/if}</span
										>
									{:else}<span class="text-muted">—</span>{/if}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="mt-1 max-w-3xl text-xs text-muted">
			Bleu : le bloc monte quand l'indicateur monte ; orange : il baisse. Cases encadrées : paires
			qui passent le placebo à 5 %. Survoler une case pour l'intervalle à 90 %, le retard et le p du
			placebo. « Bloc du président sortant » : droite en 2002, 2007 et 2012, gauche en 2017, centre
			en 2022 et 2027.
		</p>
	{:else}
		<p class="text-sm text-muted">Pas de résultat d'étude publié.</p>
	{/if}
</section>
