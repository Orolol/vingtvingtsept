<script lang="ts">
	import { candidateRegistry, BLOC_LABELS } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import TrackChart from '#lib/components/TrackChart.svelte';
	import type { Bloc } from '#lib/types/data.js';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const candidates = $derived(candidateRegistry(data.manifest));
	const pollsterName = $derived(new Map(data.polls.pollsters.map((p) => [p.id, p.name])));

	const POPULATION = { registered: 'inscrits', certain_voters: 'certains de voter', unknown: '—' };

	interface Row {
		key: string;
		poll_id: string;
		date: string;
		pollster: string;
		sponsor: string;
		sample: number | null;
		population: string;
		n: number;
		shares: Record<string, number>;
	}
	const rows1 = $derived<Row[]>(
		data.polls.polls.flatMap((p) =>
			p.hypotheses
				.filter((h) => h.round === 1)
				.map((h) => ({
					key: `${p.poll_id}/${h.hypothesis_id}/${h.population}`,
					poll_id: p.poll_id,
					date: p.fieldwork_end,
					pollster: pollsterName.get(p.pollster) ?? p.pollster,
					sponsor: p.sponsor ?? '—',
					sample: p.sample_size,
					population: POPULATION[h.population],
					n: h.shares.length,
					shares: Object.fromEntries(h.shares.map((s) => [s.candidate_id, s.share]))
				}))
		)
	);
	const rows2 = $derived(
		data.polls.polls.flatMap((p) =>
			p.hypotheses
				.filter((h) => h.round === 2)
				.map((h) => ({
					key: `${p.poll_id}/${h.hypothesis_id}/${h.population}`,
					date: p.fieldwork_end,
					pollster: pollsterName.get(p.pollster) ?? p.pollster,
					a: h.shares[0],
					b: h.shares[1]
				}))
		)
	);

	// Colonnes de candidats : les dix plus hauts en moyenne, parmi ceux qui pèsent au moins 2 %.
	const columns = $derived.by(() => {
		const stats: Record<string, { n: number; sum: number }> = {};
		for (const r of rows1)
			for (const [id, s] of Object.entries(r.shares)) {
				const st = (stats[id] ??= { n: 0, sum: 0 });
				st.n++;
				st.sum += s;
			}
		return Object.entries(stats)
			.filter(([, s]) => s.sum / s.n >= 2)
			.sort((a, b) => b[1].sum / b[1].n - a[1].sum / a[1].n)
			.slice(0, 10)
			.map(([id]) => id);
	});

	let pollster = $state('');
	type SortKey = 'date' | 'pollster' | 'sample' | string;
	let sortKey = $state<SortKey>('date');
	let sortDir = $state<1 | -1>(-1);
	function sortBy(key: SortKey) {
		if (sortKey === key) sortDir = sortDir === 1 ? -1 : 1;
		else {
			sortKey = key;
			sortDir = key === 'pollster' ? 1 : -1;
		}
	}
	const value = (r: Row, key: SortKey): string | number | null =>
		key === 'date'
			? r.date
			: key === 'pollster'
				? r.pollster
				: key === 'sample'
					? r.sample
					: (r.shares[key] ?? null);
	const sorted = $derived(
		rows1
			.filter((r) => !pollster || r.pollster === pollster)
			.toSorted((a, b) => {
				const va = value(a, sortKey);
				const vb = value(b, sortKey);
				if (va === vb) return b.date.localeCompare(a.date);
				if (va === null) return 1;
				if (vb === null) return -1;
				return (va < vb ? -1 : 1) * sortDir;
			})
	);
	const ariaSort = (key: SortKey) =>
		sortKey === key ? (sortDir === 1 ? 'ascending' : 'descending') : 'none';
	const pollsterNames = $derived([...new Set(rows1.map((r) => r.pollster))].sort());

	const blocs = $derived([...new Set(data.polls.house_effects.map((h) => h.bloc))] as Bloc[]);
	const effectsByPollster = $derived(
		data.polls.pollsters
			.map((p) => ({
				id: p.id,
				name: p.name,
				effects: new Map(
					data.polls.house_effects.filter((h) => h.pollster === p.id).map((h) => [h.bloc, h])
				),
				n: data.polls.house_effects.find((h) => h.pollster === p.id)?.n_polls ?? 0
			}))
			.filter((p) => p.effects.size)
	);

	const ratings = $derived(data.polls.pollster_ratings);
	const RATING_BLOCS = ['far_right', 'right', 'centre', 'left', 'far_left'] as const;
	const num = (x: number | null, d = 2) => (x === null ? '—' : fmt.decimal(x, d));
	const signed = (x: number | null) => (x === null ? '—' : fmt.points(x));

	// Dispersion des instituts et électorats fragiles : volets explicatifs, hors prévision.
	const disp = $derived(data.polls.dispersion);
	const dispTracks = $derived([
		{
			candidate_id: 'd_2027',
			label: '2027',
			color: 'var(--color-accent)',
			values: disp.D
		},
		{
			candidate_id: 'd_ref',
			label: 'Référence',
			color: 'var(--color-muted)',
			values: disp.reference
		}
	]);
	const tg = $derived(data.polls.turnout_gap);
	const gapMax = $derived(Math.max(1, ...tg.rows.map((r) => Math.abs(r.gap) + (r.se ?? 0))));
</script>

<PageTitle title="Sondages">
	<p>
		Tous les sondages intégrés au modèle ({fmt.integer(data.polls.polls.length)} au {fmt.date(
			data.polls.asof
		)}), une ligne par hypothèse testée. Parts publiées en % des exprimés. Cliquer un en-tête pour
		trier.
	</p>
</PageTitle>

<section aria-labelledby="titre-t1" class="mb-10">
	<div class="mb-3 flex flex-wrap items-end justify-between gap-3">
		<h2 id="titre-t1" class="text-lg font-semibold">Premier tour</h2>
		<label class="text-sm text-ink-2">
			Institut
			<select
				bind:value={pollster}
				class="ml-1 rounded-md border border-line bg-surface px-2 py-1 text-sm"
			>
				<option value="">tous</option>
				{#each pollsterNames as name (name)}<option value={name}>{name}</option>{/each}
			</select>
		</label>
	</div>
	<div class="max-h-[70vh] overflow-auto rounded-md border border-line bg-surface">
		<table class="w-full text-sm">
			<caption class="sr-only">Sondages de premier tour, triables</caption>
			<thead
				class="sticky top-0 z-10 bg-surface text-left text-xs text-muted shadow-[0_1px_0_var(--color-line)]"
			>
				<tr>
					{#snippet head(key: SortKey, label: string, cls = '')}
						<th
							scope="col"
							aria-sort={ariaSort(key)}
							class={['px-2 py-2 font-medium whitespace-nowrap', cls]}
						>
							<button
								type="button"
								class="inline-flex items-center gap-1 hover:text-ink"
								onclick={() => sortBy(key)}
							>
								{label}
								<span aria-hidden="true" class="w-2"
									>{sortKey === key ? (sortDir === 1 ? '▲' : '▼') : ''}</span
								>
							</button>
						</th>
					{/snippet}
					{@render head('date', 'Fin de terrain', 'sticky left-0 bg-surface')}
					{@render head('pollster', 'Institut')}
					<th scope="col" class="px-2 py-2 font-medium">Commanditaire</th>
					{@render head('sample', 'Échantillon', 'text-right')}
					<th scope="col" class="px-2 py-2 font-medium">Base</th>
					{#each columns as id (id)}
						{@render head(id, candidates.short(id), 'text-right')}
					{/each}
					<th scope="col" class="px-2 py-2 text-right font-medium">Testés</th>
				</tr>
			</thead>
			<tbody>
				{#each sorted as r (r.key)}
					<tr class="border-t border-line">
						<td class="tabular sticky left-0 bg-surface px-2 py-1.5 whitespace-nowrap"
							>{fmt.dateNum(r.date)}</td
						>
						<td class="px-2 py-1.5 whitespace-nowrap">{r.pollster}</td>
						<td class="max-w-40 truncate px-2 py-1.5 text-ink-2" title={r.sponsor}>{r.sponsor}</td>
						<td class="tabular px-2 py-1.5 text-right"
							>{r.sample === null ? '—' : fmt.integer(r.sample)}</td
						>
						<td class="px-2 py-1.5 whitespace-nowrap text-ink-2">{r.population}</td>
						{#each columns as id (id)}
							<td class="tabular px-2 py-1.5 text-right">
								{#if r.shares[id] === undefined}<span class="text-muted">—</span
									>{:else}{fmt.decimal(r.shares[id])}{/if}
							</td>
						{/each}
						<td class="tabular px-2 py-1.5 text-right text-muted">{r.n}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	<p class="mt-1 text-xs text-muted">
		« — » : candidat non testé dans cette hypothèse. « Testés » : nombre de candidats de
		l'hypothèse.
	</p>
</section>

<section aria-labelledby="titre-t2" class="mb-10">
	<h2 id="titre-t2" class="mb-3 text-lg font-semibold">Second tour (duels hypothétiques)</h2>
	{#if rows2.length}
		<div class="max-h-96 overflow-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only">Sondages de second tour</caption>
				<thead class="sticky top-0 bg-surface text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-2 py-2 font-medium">Fin de terrain</th>
						<th scope="col" class="px-2 py-2 font-medium">Institut</th>
						<th scope="col" class="px-2 py-2 font-medium">Duel</th>
					</tr>
				</thead>
				<tbody>
					{#each rows2 as r (r.key)}
						<tr class="border-t border-line">
							<td class="tabular px-2 py-1.5 whitespace-nowrap">{fmt.dateNum(r.date)}</td>
							<td class="px-2 py-1.5 whitespace-nowrap">{r.pollster}</td>
							<td class="tabular px-2 py-1.5">
								{candidates.short(r.a.candidate_id)}
								{fmt.decimal(r.a.share)} – {fmt.decimal(r.b.share)}
								{candidates.short(r.b.candidate_id)}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<p class="text-sm text-muted">Aucun sondage de second tour intégré.</p>
	{/if}
</section>

<section aria-labelledby="titre-dispersion" class="mb-10">
	<h2 id="titre-dispersion" class="mb-3 text-lg font-semibold">Dispersion des instituts</h2>
	<ForecastNote inForecast={disp.in_forecast} note={disp.note} />
	{#if disp.current.D !== null}
		<p class="mb-3 max-w-3xl">
			Au {fmt.date(disp.current.asof)}, à {disp.current.horizon_days} jours du vote, les sondages des
			{disp.current.pollsters.length} instituts des {disp.window_days} derniers jours s'écartent
			<strong class="tabular">{fmt.decimal(disp.current.D)}</strong> fois plus que le seul tirage
			aléatoire ne le voudrait, contre
			<strong class="tabular"
				>{disp.current.reference === null ? '—' : fmt.decimal(disp.current.reference)}</strong
			> en moyenne aux présidentielles passées au même moment.
		</p>
	{:else}
		<p class="mb-3 max-w-3xl text-sm text-muted">
			Moins de quatre instituts sur les {disp.window_days} derniers jours : dispersion non calculée.
		</p>
	{/if}
	<TrackChart
		dates={disp.weeks}
		tracks={dispTracks}
		{candidates}
		format={(v) => fmt.decimal(v, 1)}
		label="Dispersion relative des instituts par semaine en 2027, et référence des présidentielles passées"
		height={220}
	/>
	<p class="mt-1 max-w-3xl text-xs text-muted">
		{disp.reading} Ligne interrompue : moins de quatre instituts dans la fenêtre.
		{#if disp.past_same_horizon.length}
			Même distance du vote :
			{disp.past_same_horizon
				.map((p) => `${p.election} ${p.D === null ? 'non calculable' : fmt.decimal(p.D)}`)
				.join(', ')}.
		{/if}
	</p>
</section>

<section aria-labelledby="titre-fragiles" class="mb-10">
	<h2 id="titre-fragiles" class="mb-3 text-lg font-semibold">Électorats fragiles</h2>
	<ForecastNote inForecast={tg.in_forecast} note={tg.note} />
	<p class="mb-3 max-w-3xl text-sm text-ink-2">
		{tg.pollsters.join(', ')}, {tg.n_polls} sondages{tg.first_poll && tg.last_poll
			? ` du ${fmt.date(tg.first_poll)} au ${fmt.date(tg.last_poll)}`
			: ''}, premier tour. Écart moyen en points d'exprimés et erreur-type entre sondages.
		{tg.history}
	</p>
	{#if tg.rows.length}
		<div class="overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only"
					>Écart entre certains d'aller voter et inscrits, par candidat, en 2027</caption
				>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-3 py-2 font-medium">Candidat</th>
						<th scope="col" class="px-3 py-2 text-right font-medium">Écart (pts)</th>
						<th scope="col" class="px-3 py-2 font-medium"><span class="sr-only">Graphique</span></th
						>
						<th scope="col" class="px-3 py-2 text-right font-medium">Certains</th>
						<th scope="col" class="px-3 py-2 text-right font-medium">Inscrits</th>
						<th scope="col" class="px-3 py-2 text-right font-medium">Sondages</th>
					</tr>
				</thead>
				<tbody>
					{#each tg.rows as r (r.candidate_id)}
						{@const half = 50 / gapMax}
						<tr class="border-b border-line last:border-0">
							<th scope="row" class="px-3 py-1.5 text-left font-normal whitespace-nowrap"
								>{candidates.short(r.candidate_id)}</th
							>
							<td class="tabular px-3 py-1.5 text-right whitespace-nowrap">
								{fmt.points(r.gap)}
								<span class="text-xs text-muted"
									>{r.se === null ? '(1 sondage)' : `± ${fmt.decimal(r.se, 1)}`}</span
								>
							</td>
							<td class="w-40 min-w-28 px-3 py-1.5" aria-hidden="true">
								<svg viewBox="0 0 100 10" preserveAspectRatio="none" class="block h-3 w-full">
									<line x1="50" x2="50" y1="0" y2="10" class="stroke-axis" stroke-width="0.6" />
									<rect
										x={r.gap < 0 ? 50 + r.gap * half : 50}
										width={Math.abs(r.gap) * half}
										y="2"
										height="6"
										class={r.gap < 0 ? 'fill-warn-ink' : 'fill-accent'}
									/>
									{#if r.se !== null}
										<line
											x1={50 + (r.gap - r.se) * half}
											x2={50 + (r.gap + r.se) * half}
											y1="5"
											y2="5"
											class="stroke-ink"
											stroke-width="0.8"
										/>
									{/if}
								</svg>
							</td>
							<td class="tabular px-3 py-1.5 text-right">{fmt.decimal(r.share_certain)}</td>
							<td class="tabular px-3 py-1.5 text-right">{fmt.decimal(r.share_registered)}</td>
							<td class="tabular px-3 py-1.5 text-right text-muted">{r.n_polls}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="mt-1 text-xs text-muted">
			Barre vers la gauche : électorat moins sûr d'aller voter que la moyenne ; trait : ± une
			erreur-type. Les parts publiées sont arrondies au point, ce qui pèse sur les petits candidats.
		</p>
	{/if}
</section>

<section aria-labelledby="titre-effets">
	<h2 id="titre-effets" class="mb-1 text-lg font-semibold">Effets maison estimés</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">
		Écart moyen de chaque institut au consensus sur chaque bloc, en points d'exprimés : positif,
		l'institut mesure le bloc plus haut que les autres. Médiane et intervalle à 90 % ; un institut
		qui publie peu a un intervalle large.
	</p>
	<div class="overflow-x-auto rounded-md border border-line bg-surface">
		<table class="w-full text-sm">
			<caption class="sr-only">Effets maison par institut et par bloc</caption>
			<thead class="border-b border-line text-left text-xs text-muted">
				<tr>
					<th scope="col" class="px-3 py-2 font-medium">Institut</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Sondages</th>
					{#each blocs as b (b)}<th scope="col" class="px-3 py-2 font-medium">{BLOC_LABELS[b]}</th
						>{/each}
				</tr>
			</thead>
			<tbody>
				{#each effectsByPollster as p (p.id)}
					<tr class="border-b border-line last:border-0">
						<th scope="row" class="px-3 py-2 text-left font-normal whitespace-nowrap">{p.name}</th>
						<td class="tabular px-3 py-2 text-right">{p.n}</td>
						{#each blocs as b (b)}
							{@const e = p.effects.get(b)}
							<td class="tabular px-3 py-2 whitespace-nowrap">
								{#if e}
									<span class={Math.abs(e.effect.q50) >= 1 ? 'font-medium' : ''}
										>{fmt.points(e.effect.q50)}</span
									>
									<span class="text-xs text-muted"
										>({fmt.points(e.effect.q05)} à {fmt.points(e.effect.q95)})</span
									>
								{:else}—{/if}
							</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</section>

<section aria-labelledby="titre-palmares" class="mt-10">
	<h2 id="titre-palmares" class="mb-1 text-lg font-semibold">
		Palmarès des instituts, {ratings.elections[0]}-{ratings.elections.at(-1)}
	</h2>
	<p
		class="mb-3 max-w-3xl rounded-md border border-line bg-surface px-4 py-3 text-sm"
		data-testid="forecast-note"
	>
		<strong>{ratings.note}</strong>
	</p>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">
		Erreur du dernier sondage de chaque institut avant le premier tour, par lignée (Sofres, Kantar
		et Verian ensemble). « Écart » : erreur par candidat moins celle de l'institut moyen de la même
		élection (négatif = plus juste). Biais : sondage moins résultat, en points, par bloc. Précision
		: dispersion des sondages des trois dernières semaines autour de l'institut moyen (négatif =
		plus précis). « Élections » : nombre de présidentielles couvertes ; un institut qui n'en a
		couvert qu'une n'a qu'un point de comparaison.
	</p>
	<div class="overflow-x-auto rounded-md border border-line bg-surface">
		<table class="w-full text-sm">
			<caption class="sr-only"
				>Palmarès des instituts de sondage aux présidentielles passées</caption
			>
			<thead class="border-b border-line text-left text-xs text-muted">
				<tr>
					<th scope="col" class="sticky left-0 bg-surface px-3 py-2 font-medium">Institut</th>
					<th scope="col" class="px-2 py-2 text-right font-medium">Élections</th>
					<th scope="col" class="px-2 py-2 text-right font-medium">Erreur cand. (pts)</th>
					<th scope="col" class="px-2 py-2 text-right font-medium">Écart</th>
					{#each RATING_BLOCS as b (b)}
						<th scope="col" class="px-2 py-2 text-right font-medium">Biais {BLOC_LABELS[b]}</th>
					{/each}
					<th scope="col" class="px-2 py-2 text-right font-medium">Précision</th>
					<th scope="col" class="px-2 py-2 text-right font-medium">Mieux que la moyenne</th>
				</tr>
			</thead>
			<tbody>
				{#each ratings.rows as r (r.pollster)}
					<tr class="border-b border-line last:border-0">
						<th
							scope="row"
							class="sticky left-0 bg-surface px-3 py-1.5 text-left font-normal whitespace-nowrap"
							>{r.name}</th
						>
						<td class="tabular px-2 py-1.5 text-right" title={r.elections.join(', ')}
							>{r.elections.length}</td
						>
						<td class="tabular px-2 py-1.5 text-right">{num(r.mae_candidate)}</td>
						<td class="tabular px-2 py-1.5 text-right">{signed(r.relative_error)}</td>
						{#each RATING_BLOCS as b (b)}
							<td class="tabular px-2 py-1.5 text-right">{signed(r.bias[b])}</td>
						{/each}
						<td class="tabular px-2 py-1.5 text-right">{signed(r.precision)}</td>
						<td class="tabular px-2 py-1.5 text-right">{r.n_better} sur {r.n_rated}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	<h3 class="mt-5 mb-2 font-medium">Le classement tient-il d'une élection à l'autre ?</h3>
	<ul class="max-w-3xl space-y-1 text-sm text-ink-2">
		{#each ratings.stability as st (st.id)}
			<li>
				{st.label} : corrélation
				<span class="tabular font-medium text-ink">{num(st.pearson)}</span>
				entre élections consécutives ({st.n_pairs} paires{#if st.p !== null}, p = {num(st.p)}{/if}).
			</li>
		{/each}
	</ul>
	<p class="mt-2 max-w-3xl text-sm text-ink-2">
		Une corrélation proche de zéro veut dire qu'être bien classé à une élection ne dit presque rien
		de la suivante : les écarts entre instituts sont petits devant l'erreur que tous commettent
		ensemble, déjà comptée dans le modèle d'erreur.
	</p>
</section>
