<script lang="ts">
	import { candidateRegistry } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import CandidateName from '#lib/components/CandidateName.svelte';
	import TrackChart from '#lib/components/TrackChart.svelte';
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

	// Attention : les cinq candidats les plus vus sur douze semaines par défaut (ordre de l'export),
	// les autres à cocher.
	const att = $derived(press.attention);
	let picked = $state<string[] | null>(null);
	const shown = $derived(picked ?? att.tracks.slice(0, 5).map((t) => t.candidate_id));
	function toggle(id: string) {
		picked = shown.includes(id) ? shown.filter((x) => x !== id) : [...shown, id];
	}
	const attLast = $derived(att.weeks.length - 1);
	const attRanking = $derived(
		att.tracks
			.map((t) => ({ id: t.candidate_id, v: t.values[attLast] ?? null }))
			.filter((r): r is { id: string; v: number } => r.v !== null)
			.sort((a, b) => b.v - a.v)
	);
	const pct = (v: number) => `${fmt.decimal(v, v < 10 ? 1 : 0)} %`;
	const pctFine = (v: number) => `${fmt.decimal(v, 2)} %`;

	const ina = $derived(press.ina);
	const inaMax = $derived(
		Math.max(1, ...ina.topics.flatMap((t) => [...t.shares, t.share_all_years]))
	);
</script>

<PageTitle title="Presse et attention">
	<p>
		Ce dont on parle et qui l'on cherche : attention en ligne pour les candidats de 2027, agenda des
		journaux télévisés aux présidentielles passées, volume d'articles dans la presse mondiale. Volet
		explicatif : rien ici n'entre dans la prévision tant qu'un gain au backtest ne l'a pas justifié.
	</p>
</PageTitle>

<ForecastNote inForecast={press.in_forecast} note={press.in_forecast_note} />

<section aria-labelledby="titre-attention" class="mb-12">
	<h2 id="titre-attention" class="mb-1 text-lg font-semibold">
		Attention en ligne, témoin d'actualité
	</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{att.note}</p>
	{#if att.weeks.length}
		<div class="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="Candidats affichés">
			{#each att.tracks as t (t.candidate_id)}
				{@const on = shown.includes(t.candidate_id)}
				<button
					type="button"
					aria-pressed={on}
					onclick={() => toggle(t.candidate_id)}
					class={[
						'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs',
						on ? 'border-ink/40 bg-raised text-ink' : 'border-line text-muted'
					]}
				>
					<span
						class="inline-block size-2 rounded-full"
						style:background={on ? candidates.color(t.candidate_id) : 'transparent'}
						style:border="1px solid {candidates.color(t.candidate_id)}"
					></span>
					{candidates.short(t.candidate_id)}
				</button>
			{/each}
		</div>
		<div class="rounded-md border border-line bg-surface p-3">
			<TrackChart
				dates={att.weeks}
				tracks={att.tracks.filter((t) => shown.includes(t.candidate_id))}
				{candidates}
				format={pct}
				label="Part hebdomadaire des pages vues Wikipédia par candidat"
			/>
		</div>
		<p class="mt-1 text-xs text-muted">
			Source : {att.source}. Semaines complètes, du lundi au dimanche ; dernière semaine : celle du
			{fmt.date(att.weeks[attLast])}.
		</p>
		<details class="mt-3 max-w-md text-sm">
			<summary class="cursor-pointer text-ink-2">
				Classement de la semaine du {fmt.dateShortFmt(att.weeks[attLast])}
			</summary>
			<ol class="mt-2 divide-y divide-line rounded-md border border-line bg-surface">
				{#each attRanking as r (r.id)}
					<li class="flex items-center justify-between gap-3 px-3 py-1.5">
						<CandidateName id={r.id} name={candidates.name(r.id)} />
						<span class="tabular text-ink-2">{pct(r.v)}</span>
					</li>
				{/each}
			</ol>
		</details>
	{:else}
		<p class="text-sm text-muted">Pas de pages vues publiées pour l'instant.</p>
	{/if}
</section>

<section aria-labelledby="titre-ina" class="mb-12">
	<h2 id="titre-ina" class="mb-1 text-lg font-semibold">
		Agenda des journaux télévisés aux présidentielles passées
	</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{ina.note}</p>
	{#if ina.topics.length}
		<div class="overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only"
					>Part des rubriques dans les sujets de JT, par année de présidentielle (%)</caption
				>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-3 py-2 font-medium">Rubrique</th>
						{#each ina.years as y (y)}
							<th scope="col" class="px-3 py-2 text-right font-medium">{y}</th>
						{/each}
						<th scope="col" class="px-3 py-2 text-right font-medium">{ina.all_years_label}</th>
					</tr>
				</thead>
				<tbody>
					{#each ina.topics as t (t.label)}
						<tr class="border-b border-line last:border-0">
							<th scope="row" class="px-3 py-1.5 text-left font-normal whitespace-nowrap"
								>{t.label}</th
							>
							{#each t.shares as v, i (ina.years[i])}
								<td class="tabular px-3 py-1.5 text-right">
									<span
										class="inline-block rounded px-1"
										style:background="color-mix(in oklab, var(--color-accent) {Math.round(
											(45 * v) / inaMax
										)}%, transparent)">{fmt.decimal(v)}</span
									>
								</td>
							{/each}
							<td class="tabular px-3 py-1.5 text-right text-ink-2"
								>{fmt.decimal(t.share_all_years)}</td
							>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="mt-1 text-xs text-muted">
			Source : {ina.source}. Part des sujets de l'année civile, en %. Plus la case est foncée, plus
			la rubrique pèse.
		</p>
	{/if}
</section>

<section aria-labelledby="titre-gdelt" class="mb-12">
	<h2 id="titre-gdelt" class="mb-1 text-lg font-semibold">Volume d'articles dans GDELT</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{press.gdelt.note}</p>
	{#if press.gdelt.months.length}
		<div class="rounded-md border border-line bg-surface p-3">
			<TrackChart
				dates={press.gdelt.months}
				tracks={press.gdelt.tracks}
				{candidates}
				format={pctFine}
				period="month"
				label="Part mensuelle des articles GDELT citant chaque candidat"
			/>
		</div>
		<p class="mt-1 text-xs text-muted">
			Source : {press.gdelt.source}. Part des articles surveillés du mois qui citent le candidat, en
			%.
		</p>
	{:else}
		<p class="text-sm text-muted">Pas de série GDELT publiée pour l'instant.</p>
	{/if}
</section>

<section aria-labelledby="titre-voix">
	<h2 id="titre-voix" class="mb-3 text-lg font-semibold">Part de voix, ton, agenda</h2>
	{#if press.weeks.length === 0}
		<p class="max-w-3xl text-sm text-muted">
			Pas encore publiés : il faut d'abord classer les titres de presse (Media Cloud) par candidat
			et par thème.
		</p>
	{:else}
		<div class="grid gap-8 md:grid-cols-2">
			<div>
				<h3 class="mb-3 font-medium">Semaine du {fmt.date(press.weeks[last])}</h3>
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
			</div>
			{#if agenda}
				<div>
					<h3 class="mb-3 font-medium">Agenda de la semaine</h3>
					<ul class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
						{#each agenda.topics as t (t.label)}
							<li class="flex justify-between gap-3 px-3 py-2">
								<span>{t.label}</span><span class="tabular text-ink-2">{fmt.share(t.share)}</span>
							</li>
						{/each}
					</ul>
				</div>
			{/if}
		</div>
	{/if}
</section>
