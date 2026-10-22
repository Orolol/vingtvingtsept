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
	const nlp = $derived(press.nlp);

	const toneFmt = (v: number) =>
		new Intl.NumberFormat('fr-FR', {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2,
			signDisplay: 'exceptZero'
		}).format(v);
	const frac = (v: number) => `${fmt.decimal(100 * v, v < 0.1 ? 1 : 0)} %`;
	const fracFine = (v: number) => `${fmt.decimal(100 * v, 1)} %`;

	// Sélection de candidats par pastilles : `null` = sélection par défaut.
	function toggler(get: () => string[], set: (v: string[]) => void) {
		return (id: string) => {
			const cur = get();
			set(cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]);
		};
	}

	// Sélection par défaut : les premiers de la liste, une couleur chacun (Le Pen et Bardella
	// partagent la couleur du RN : un seul des deux).
	function distinctColors(ids: string[], k: number): string[] {
		const out: string[] = [];
		const used: string[] = [];
		for (const id of ids) {
			const c = candidates.color(id);
			if (used.includes(c) && !c.includes('#898781')) continue;
			used.push(c);
			out.push(id);
			if (out.length === k) break;
		}
		return out;
	}

	// Ton : les quatre candidats les plus couverts sur le dernier mois (ordre de l'export).
	const tone = $derived(press.tone);
	const toneLast = $derived(tone.weeks.length - 1);
	let tonePicked = $state<string[] | null>(null);
	const toneShown = $derived(
		tonePicked ??
			distinctColors(
				tone.tracks.map((t) => t.candidate_id),
				4
			)
	);
	const toggleTone = toggler(
		() => toneShown,
		(v) => (tonePicked = v)
	);
	const toneTracks = $derived(
		tone.tracks
			.filter((t) => toneShown.includes(t.candidate_id))
			.map((t) => ({ candidate_id: t.candidate_id, values: t.q50, lo: t.q05, hi: t.q95 }))
	);
	const toneLatest = $derived(
		tone.tracks
			.map((t) => ({
				id: t.candidate_id,
				n: t.n[toneLast],
				q05: t.q05[toneLast],
				q50: t.q50[toneLast],
				q95: t.q95[toneLast]
			}))
			.sort((a, b) => (b.q50 ?? -9) - (a.q50 ?? -9))
	);
	const toneWeekStart = $derived(
		tone.weeks[Math.max(0, toneLast - tone.window_weeks + 1)] ?? tone.weeks[toneLast]
	);

	// Posture : couleurs fixes par posture (palette validée en clair et en sombre, « autre » en
	// gris) ; l'identité est aussi portée par la légende et le tableau.
	const stance = $derived(press.stance);
	const STANCE_COLORS: Record<string, string> = {
		announcement: 'light-dark(#2a78d6, #3987e5)',
		programme: 'light-dark(#eda100, #c98500)',
		support: 'light-dark(#008300, #008300)',
		attack: 'light-dark(#6250d6, #9085e9)',
		scandal: 'light-dark(#e34948, #e66767)',
		poll: 'light-dark(#1baf7a, #199e70)',
		other: 'light-dark(#898781, #898781)'
	};
	const stanceColor = (id: string) => STANCE_COLORS[id] ?? 'var(--color-muted)';
	const stanceLabel = $derived(new Map(stance.categories.map((c) => [c.id, c.label])));

	// Titres de la semaine : un candidat à la fois.
	const headlines = $derived(press.headlines);
	let headlinePick = $state<string | null>(null);
	const headlineShown = $derived(
		headlines.candidates.find((c) => c.candidate_id === headlinePick) ?? headlines.candidates[0]
	);

	// Part de voix et attention : mêmes candidats, mêmes semaines, même échelle.
	const sov = $derived(press.share_of_voice);
	const sovLast = $derived(sov.weeks.length - 1);
	let sovPicked = $state<string[] | null>(null);
	const sovShown = $derived(
		sovPicked ??
			distinctColors(
				[...sov.tracks]
					.sort((a, b) => (b.values[sovLast] ?? 0) - (a.values[sovLast] ?? 0))
					.map((t) => t.candidate_id),
				5
			)
	);
	const toggleSov = toggler(
		() => sovShown,
		(v) => (sovPicked = v)
	);
	const sovMax = $derived(
		Math.max(
			0.01,
			...[...sov.tracks, ...sov.attention]
				.filter((t) => sovShown.includes(t.candidate_id))
				.flatMap((t) => t.values.filter((v): v is number => v !== null))
		)
	);
	const sovDomain = $derived([0, Math.ceil(sovMax * 20) / 20] as [number, number]);

	// Agenda : petits multiples à échelle commune.
	const agendaNlp = $derived(press.agenda_nlp);
	const agendaMax = $derived(
		Math.max(
			0.01,
			...agendaNlp.themes.flatMap((t) => t.values.filter((v): v is number => v !== null))
		)
	);
	const agendaDomain = $derived([0, Math.ceil(agendaMax * 50) / 50] as [number, number]);
	const ACCENT = 'var(--color-accent)';

	const landmarks = $derived(press.landmarks);
	const quality = $derived(press.quality);
	const kappaFmt = (v: number) => fmt.decimal(v, 2);

	// Attention : les cinq candidats les plus vus sur douze semaines par défaut (ordre de l'export),
	// les autres à cocher.
	const att = $derived(press.attention);
	let picked = $state<string[] | null>(null);
	const shown = $derived(picked ?? att.tracks.slice(0, 5).map((t) => t.candidate_id));
	const toggle = toggler(
		() => shown,
		(v) => (picked = v)
	);
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
		Ce dont on parle, sur quel ton, et qui l'on cherche : titres de presse classés par un modèle de
		langue local, attention en ligne pour les candidats de 2027, agenda des journaux télévisés aux
		présidentielles passées, volume d'articles dans la presse mondiale. Volet explicatif : rien ici
		n'entre dans la prévision tant qu'un gain au backtest ne l'a pas justifié.
	</p>
</PageTitle>

<ForecastNote inForecast={press.in_forecast} note={press.in_forecast_note} />

{#snippet pills(ids: string[], on: string[], toggle: (id: string) => void, label: string)}
	<div class="mb-3 flex flex-wrap gap-1.5" role="group" aria-label={label}>
		{#each ids as id (id)}
			{@const active = on.includes(id)}
			<button
				type="button"
				aria-pressed={active}
				onclick={() => toggle(id)}
				class={[
					'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs',
					active ? 'border-ink/40 bg-raised text-ink' : 'border-line text-muted'
				]}
			>
				<span
					class="inline-block size-2 rounded-full"
					style:background={active ? candidates.color(id) : 'transparent'}
					style:border="1px solid {candidates.color(id)}"
				></span>
				{candidates.short(id)}
			</button>
		{/each}
	</div>
{/snippet}

<aside
	aria-labelledby="titre-nlp-statut"
	class="mb-10 max-w-3xl rounded-md border border-l-4 border-line border-l-accent bg-surface px-4 py-3 text-sm"
	data-testid="nlp-status"
>
	<h2 id="titre-nlp-statut" class="mb-1 font-semibold">
		Presse classée : {nlp.in_forecast ? 'entre dans la prévision' : "n'entre pas dans la prévision"}
	</h2>
	<p class="text-ink-2">{nlp.note}</p>
	<p class="mt-2"><strong>Verdict de l'étude.</strong> {nlp.verdict.join(' ')}</p>
	<p class="mt-2 text-ink-2">{nlp.out_of_sample_test}</p>
</aside>

<section aria-labelledby="titre-ton" class="mb-12">
	<h2 id="titre-ton" class="mb-1 text-lg font-semibold">Ton de la presse par candidat</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{tone.note}</p>
	{#if tone.weeks.length}
		{@render pills(
			tone.tracks.map((t) => t.candidate_id),
			toneShown,
			toggleTone,
			'Candidats affichés (ton)'
		)}
		<div class="rounded-md border border-line bg-surface p-3">
			<TrackChart
				dates={tone.weeks}
				tracks={toneTracks}
				{candidates}
				format={toneFmt}
				label="Ton moyen des titres par candidat, deux semaines glissantes, de −1 à +1"
			/>
		</div>
		<p class="mt-1 text-xs text-muted">
			Au-dessus de zéro : titres plutôt favorables ; en dessous : plutôt hostiles. Semaine du
			{fmt.dateShortFmt(tone.weeks[0])} à celle du {fmt.date(tone.weeks[toneLast])}.
		</p>
		<details class="mt-3 max-w-xl text-sm">
			<summary class="cursor-pointer text-ink-2">
				Ton des deux semaines du {fmt.dateShortFmt(toneWeekStart)} au {fmt.dateShortFmt(
					headlines.week_end
				)}
			</summary>
			<table class="mt-2 w-full rounded-md border border-line bg-surface">
				<caption class="sr-only">Ton par candidat, médiane et intervalle à 90 %</caption>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-3 py-1.5 font-medium">Candidat</th>
						<th scope="col" class="px-3 py-1.5 text-right font-medium">Ton (90 %)</th>
						<th scope="col" class="px-3 py-1.5 text-right font-medium">Titres</th>
					</tr>
				</thead>
				<tbody>
					{#each toneLatest as r (r.id)}
						<tr class="border-b border-line last:border-0">
							<th scope="row" class="px-3 py-1.5 text-left font-normal">
								<CandidateName id={r.id} name={candidates.name(r.id)} />
							</th>
							<td class="tabular px-3 py-1.5 text-right whitespace-nowrap">
								{#if r.q50 !== null && r.q05 !== null && r.q95 !== null}
									{toneFmt(r.q50)}
									<span class="text-muted">({toneFmt(r.q05)} à {toneFmt(r.q95)})</span>
								{:else}
									<span class="text-muted">moins de {tone.min_titles} titres</span>
								{/if}
							</td>
							<td class="tabular px-3 py-1.5 text-right text-ink-2">{r.n}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</details>
	{:else}
		<p class="text-sm text-muted">Pas de titres classés publiés pour l'instant.</p>
	{/if}
</section>

<section aria-labelledby="titre-posture" class="mb-12">
	<h2 id="titre-posture" class="mb-1 text-lg font-semibold">Posture des titres, dernier mois</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{stance.note}</p>
	<ul class="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="Légende">
		{#each stance.categories as c (c.id)}
			<li class="inline-flex items-center gap-1.5">
				<span class="inline-block size-2.5 rounded-sm" style:background={stanceColor(c.id)}
				></span>{c.label}
			</li>
		{/each}
	</ul>
	<ul class="space-y-2 rounded-md border border-line bg-surface p-3 text-sm">
		{#each stance.last_weeks.rows as r (r.candidate_id)}
			<li class="grid grid-cols-[6.5rem_1fr] items-center gap-x-3 sm:grid-cols-[9rem_1fr_4rem]">
				<span class="truncate">
					<CandidateName id={r.candidate_id} name={candidates.short(r.candidate_id)} />
				</span>
				{#if r.shares}
					{@const shares = r.shares}
					<div
						class="flex h-4 w-full overflow-hidden rounded"
						role="img"
						aria-label="{candidates.name(r.candidate_id)} : {stance.categories
							.map((c, i) => `${c.label} ${frac(shares[i])}`)
							.join(', ')}"
					>
						{#each stance.categories as c, i (c.id)}
							{#if shares[i] > 0}
								<span
									class="h-full border-r-2 border-surface last:border-r-0"
									style:width="{100 * shares[i]}%"
									style:background={stanceColor(c.id)}
									title="{c.label} : {frac(shares[i])}"
								></span>
							{/if}
						{/each}
					</div>
				{:else}
					<span class="text-xs text-muted">moins de {stance.min_titles} titres</span>
				{/if}
				<span class="tabular col-start-2 text-xs text-muted sm:col-start-3 sm:text-right"
					>{fmt.integer(r.n)} titres</span
				>
			</li>
		{/each}
	</ul>
	<p class="mt-1 text-xs text-muted">
		Titres classés du {fmt.date(stance.last_weeks.start)} au {fmt.date(stance.last_weeks.end)}.
	</p>
	<details class="mt-3 text-sm">
		<summary class="cursor-pointer text-ink-2">Tableau des postures (%)</summary>
		<div class="mt-2 overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-xs">
				<caption class="sr-only">Part de chaque posture par candidat, dernier mois (%)</caption>
				<thead class="border-b border-line text-left text-muted">
					<tr>
						<th scope="col" class="px-2 py-1.5 font-medium">Candidat</th>
						{#each stance.categories as c (c.id)}
							<th scope="col" class="px-2 py-1.5 text-right font-medium">{c.label}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each stance.last_weeks.rows as r (r.candidate_id)}
						<tr class="border-b border-line last:border-0">
							<th scope="row" class="px-2 py-1 text-left font-normal whitespace-nowrap"
								>{candidates.short(r.candidate_id)}</th
							>
							{#each stance.categories as c, i (c.id)}
								<td class="tabular px-2 py-1 text-right"
									>{r.shares ? fmt.decimal(100 * r.shares[i], 0) : '–'}</td
								>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</details>
</section>

<section aria-labelledby="titre-titres" class="mb-12">
	<h2 id="titre-titres" class="mb-1 text-lg font-semibold">
		Titres hostiles et favorables de la semaine
	</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">
		Semaine du {fmt.date(headlines.week)} au {fmt.date(headlines.week_end)}. {headlines.note}
	</p>
	<div class="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="Candidat affiché (titres)">
		{#each headlines.candidates as c (c.candidate_id)}
			{@const active = c.candidate_id === headlineShown?.candidate_id}
			<button
				type="button"
				aria-pressed={active}
				onclick={() => (headlinePick = c.candidate_id)}
				class={[
					'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs',
					active ? 'border-ink/40 bg-raised text-ink' : 'border-line text-muted'
				]}
			>
				<span
					class="inline-block size-2 rounded-full"
					style:background={active ? candidates.color(c.candidate_id) : 'transparent'}
					style:border="1px solid {candidates.color(c.candidate_id)}"
				></span>
				{candidates.short(c.candidate_id)}
			</button>
		{/each}
	</div>
	{#if headlineShown}
		<p class="mb-2 text-sm">
			<CandidateName
				id={headlineShown.candidate_id}
				name={candidates.name(headlineShown.candidate_id)}
			/> : {fmt.integer(headlineShown.n)} titres classés cette semaine.
		</p>
		<div class="grid gap-4 md:grid-cols-2">
			{#each [{ key: 'negative', title: 'Hostiles (−1)', items: headlineShown.negative }, { key: 'positive', title: 'Favorables (+1)', items: headlineShown.positive }] as col (col.key)}
				<div>
					<h3 class="mb-2 text-sm font-medium">{col.title}</h3>
					{#if col.items.length}
						<ul class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
							{#each col.items as h (h.url)}
								<li class="px-3 py-2">
									<a
										href={h.url}
										rel="external noopener noreferrer"
										target="_blank"
										class="break-words text-ink underline decoration-line hover:decoration-ink"
										>{h.title}</a
									>
									<p class="mt-0.5 text-xs text-muted">
										{h.outlet} · {fmt.dateShortFmt(h.date)} · {stanceLabel.get(h.stance) ??
											h.stance}
									</p>
								</li>
							{/each}
						</ul>
					{:else}
						<p class="text-sm text-muted">Aucun titre classé ainsi cette semaine.</p>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</section>

<section aria-labelledby="titre-voix" class="mb-12">
	<h2 id="titre-voix" class="mb-1 text-lg font-semibold">Part de voix et attention en ligne</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{sov.note}</p>
	{@render pills(
		sov.tracks.map((t) => t.candidate_id),
		sovShown,
		toggleSov,
		'Candidats affichés (part de voix)'
	)}
	<div class="grid gap-4 md:grid-cols-2">
		<div class="min-w-0 rounded-md border border-line bg-surface p-3">
			<h3 class="mb-2 text-sm font-medium">Part des titres qui nomment le candidat</h3>
			<TrackChart
				dates={sov.weeks}
				tracks={sov.tracks.filter((t) => sovShown.includes(t.candidate_id))}
				{candidates}
				format={frac}
				yDomain={sovDomain}
				height={220}
				label="Part de voix hebdomadaire dans les titres de presse"
			/>
		</div>
		<div class="min-w-0 rounded-md border border-line bg-surface p-3">
			<h3 class="mb-2 text-sm font-medium">Part des pages vues Wikipédia</h3>
			<TrackChart
				dates={sov.weeks}
				tracks={sov.attention.filter((t) => sovShown.includes(t.candidate_id))}
				{candidates}
				format={frac}
				yDomain={sovDomain}
				height={220}
				label="Part hebdomadaire des pages vues Wikipédia, mêmes candidats"
			/>
		</div>
	</div>
	<p class="mt-1 text-xs text-muted">
		Parts parmi les {sov.tracks.length} candidats suivis, même échelle sur les deux graphiques.
	</p>
</section>

<section aria-labelledby="titre-agenda" class="mb-12">
	<h2 id="titre-agenda" class="mb-1 text-lg font-semibold">Agenda des titres, quatre thèmes</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{agendaNlp.note}</p>
	<div class="grid gap-3 sm:grid-cols-2">
		{#each agendaNlp.themes as th (th.id)}
			<div class="min-w-0 rounded-md border border-line bg-surface p-3">
				<h3 class="mb-1 text-sm font-medium">{th.label}</h3>
				<TrackChart
					dates={agendaNlp.weeks}
					tracks={[{ candidate_id: th.id, values: th.values, label: th.label, color: ACCENT }]}
					{candidates}
					format={fracFine}
					yDomain={agendaDomain}
					height={150}
					showLabels={false}
					label="Part hebdomadaire des titres classés portant le thème {th.label}"
				/>
			</div>
		{/each}
	</div>
	<p class="mt-1 text-xs text-muted">
		Part des titres classés de la semaine, même échelle pour les quatre thèmes.
	</p>
</section>

<section aria-labelledby="titre-reperes" class="mb-12">
	<h2 id="titre-reperes" class="mb-1 text-lg font-semibold">Repères : deux chocs passés</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">{landmarks.note}</p>
	<div class="grid gap-4 lg:grid-cols-2">
		{#each landmarks.events as ev (ev.id)}
			<article class="min-w-0 rounded-md border border-line bg-surface p-3">
				<h3 class="font-medium">{ev.label}</h3>
				<p class="mt-1 mb-3 text-sm text-ink-2">{ev.summary}</p>
				{#each [{ key: 'tone', title: 'Ton (moyenne sur trois jours)', values: ev.tone, f: toneFmt, br: ev.breaks.tone, brLabel: 'ton' }, { key: 'scandal', title: 'Part de la posture « affaire »', values: ev.scandal_share, f: frac, br: ev.breaks.scandal, brLabel: 'affaire' }] as panel (panel.key)}
					<h4 class="mt-2 text-xs font-medium text-ink-2">{panel.title}</h4>
					<TrackChart
						dates={ev.dates}
						tracks={[
							{
								candidate_id: ev.candidate_id,
								values: panel.values,
								label: panel.key === 'tone' ? 'ton' : 'affaire',
								color: ACCENT
							}
						]}
						{candidates}
						format={panel.f}
						period="day"
						height={170}
						showLabels={false}
						markers={[
							{ date: ev.event_date, label: 'choc', kind: 'event' as const },
							...(ev.breaks.polls
								? [{ date: ev.breaks.polls, label: 'sondages', kind: 'polls' as const }]
								: []),
							...(panel.br
								? [{ date: panel.br, label: `rupture ${panel.brLabel}`, kind: 'series' as const }]
								: [])
						]}
						label="{ev.label} : {panel.title}, jour par jour"
					/>
				{/each}
				<p class="mt-2 text-xs text-muted">
					Choc le {fmt.dateShortFmt(ev.event_date)}{#if ev.breaks.polls}
						· rupture des sondages le {fmt.dateShortFmt(ev.breaks.polls)}{/if}{#if ev.breaks.tone}
						· du ton le {fmt.dateShortFmt(ev.breaks.tone)}{/if}{#if ev.breaks.scandal}
						· de la posture « affaire » le {fmt.dateShortFmt(ev.breaks.scandal)}{/if}.
				</p>
			</article>
		{/each}
	</div>
</section>

<section aria-labelledby="titre-qualite" class="mb-12 max-w-3xl">
	<h2 id="titre-qualite" class="mb-1 text-lg font-semibold">Qualité de la classification</h2>
	<p class="mb-3 text-sm text-ink-2">
		Modèle {quality.model_label}, prompt {quality.prompt_version}, température nulle. Accord entre
		deux formulations du prompt sur {quality.n_sample} titres tirés au hasard : κ de Cohen pour le ton
		et la posture (1 = accord parfait, 0 = hasard), F1 pour les thèmes.
	</p>
	<table class="w-full rounded-md border border-line bg-surface text-sm">
		<caption class="sr-only">Accord entre deux formulations du prompt</caption>
		<thead class="border-b border-line text-left text-xs text-muted">
			<tr>
				<th scope="col" class="px-3 py-1.5 font-medium">Mesure</th>
				<th scope="col" class="px-3 py-1.5 text-right font-medium">Accord</th>
				<th scope="col" class="px-3 py-1.5 text-right font-medium">Seuil</th>
				<th scope="col" class="px-3 py-1.5 text-right font-medium">Fiable</th>
			</tr>
		</thead>
		<tbody>
			{#each [{ k: 'tone_kappa', label: 'Ton (κ)', v: quality.tone_kappa }, { k: 'stance_kappa', label: 'Posture (κ)', v: quality.stance_kappa }, { k: 'themes_f1', label: 'Thèmes (F1)', v: quality.themes_f1 }] as row (row.k)}
				{@const key = row.k as keyof typeof quality.passes}
				<tr class="border-b border-line last:border-0">
					<th scope="row" class="px-3 py-1.5 text-left font-normal">{row.label}</th>
					<td class="tabular px-3 py-1.5 text-right">{fmt.decimal(row.v, 3)}</td>
					<td class="tabular px-3 py-1.5 text-right text-ink-2"
						>{kappaFmt(quality.thresholds[key])}</td
					>
					<td class="px-3 py-1.5 text-right">{quality.passes[key] ? 'oui' : 'non'}</td>
				</tr>
			{/each}
		</tbody>
	</table>
	<p class="mt-1 text-xs text-muted">
		Titres classés : {quality.labeled.map((l) => `${fmt.integer(l.n)} en ${l.cycle}`).join(', ')}.
		Ton à la limite du seuil, thèmes non fiables hors des quatre montrés plus haut. Détail : {quality.reference}
		dans le dépôt.
	</p>
</section>

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
