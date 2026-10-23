<script lang="ts">
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import SeriesChart from '#lib/components/SeriesChart.svelte';
	import OpinionChart from '#lib/components/OpinionChart.svelte';
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

	// Volet opinion : chaque série face à la part mensuelle de son bloc lié.
	const op = $derived(eco.opinion);
	const BLOC_COLOR: Record<string, string> = {
		far_right: 'light-dark(#2a78d6, #3987e5)',
		left: 'light-dark(#e87ba4, #d55181)',
		incumbent: 'light-dark(#eda100, #c98500)'
	};
	const blocTrack = $derived(
		new Map(
			op.bloc_shares.map((b) => [
				b.bloc,
				{ label: b.label, color: BLOC_COLOR[b.bloc] ?? 'var(--color-ink-2)', points: b.points }
			])
		)
	);
	const blocNotes = $derived(op.bloc_shares.filter((b) => b.note));
	const OP_FREQ = { annual: 'annuelle', waves: 'par vagues' };
	const SPEC = { level: 'niveau', change: 'variation' };
	const signed = (v: number | null, d = 2) =>
		v === null ? '—' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${fmt.decimal(Math.abs(v), d)}`;
	const levelVerdict = (r: { testable: boolean; holds: boolean }) =>
		!r.testable ? 'non testable' : r.holds ? 'tient' : 'non';
	const dynVerdict = (r: { testable: boolean; holds: boolean; lag: number }) =>
		r.lag === 0 ? 'contemporain' : !r.testable ? 'non testable' : r.holds ? 'tient' : 'non';
</script>

<PageTitle title="Économie et opinion">
	<p>
		Ce que disent la conjoncture, la popularité de l'exécutif et les grandes séries d'opinion
		(immigration, sécurité, image du RN, confiance) de la dynamique de la campagne. Une série
		n'entre dans la prévision que si elle améliore le backtest (docs/07-evaluation.md).
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

<section aria-labelledby="titre-opinion" class="mt-14 border-t border-line pt-8">
	<h2 id="titre-opinion" class="mb-1 text-xl font-semibold">
		Opinion : attitudes et préoccupations
	</h2>
	<p class="mb-4 max-w-3xl text-sm text-ink-2">
		Séries longues d'enquêtes, chacune face à la part, dans les sondages de premier tour, du bloc
		qu'elle est censée faire bouger. Traits pointillés : premiers tours de l'élection présidentielle
		; triangles : ruptures signalées par la source (changement de mode, d'item) ; points creux :
		vague annotée.
	</p>
	<ForecastNote inForecast={op.in_forecast} note={op.note} />

	<section aria-labelledby="titre-etude-opinion" class="mb-10 max-w-4xl">
		<h3 id="titre-etude-opinion" class="mb-2 text-lg font-semibold">
			L'étude du {fmt.date(op.study.date)} : {VERDICT[op.study.verdict].toLowerCase()}
		</h3>
		<p class="mb-3 leading-relaxed">{op.study.summary}</p>
		{#if op.study.findings.length}
			<ul class="mb-3 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-ink-2">
				{#each op.study.findings as f (f)}<li>{f}</li>{/each}
			</ul>
		{/if}
		<p class="mb-3 text-sm text-ink-2">{op.study.false_positives_note}</p>
		<div class="overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="px-3 pt-2 text-left text-xs text-muted">
					Niveau : écart entre le résultat d'un bloc et son point de départ, expliqué par l'attitude
					180 jours avant le vote, en laissant chaque élection de côté. Gain hors échantillon :
					erreur sans l'attitude moins erreur avec (points ; positif = l'attitude aide).
				</caption>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="sticky left-0 bg-surface px-3 py-2 font-medium">Série</th>
						<th scope="col" class="px-2 py-2 font-medium">Bloc</th>
						<th scope="col" class="px-2 py-2 font-medium">Mesure</th>
						<th scope="col" class="px-2 py-2 text-right font-medium">Élections</th>
						<th scope="col" class="px-2 py-2 text-right font-medium">Pente / é.-t.</th>
						<th scope="col" class="px-2 py-2 text-right font-medium">Gain hors éch.</th>
						<th scope="col" class="px-2 py-2 text-right font-medium">p placebo</th>
						<th scope="col" class="px-3 py-2 font-medium">Tient</th>
					</tr>
				</thead>
				<tbody>
					{#each op.study.level as r (r.bloc + r.series_id + r.spec)}
						<tr class={['border-b border-line last:border-0', !r.testable && 'text-muted']}>
							<th
								scope="row"
								class="sticky left-0 max-w-56 truncate bg-surface px-3 py-1.5 text-left font-normal"
								title={r.series_label}>{r.series_label}</th
							>
							<td class="px-2 py-1.5 whitespace-nowrap">{r.bloc_label}</td>
							<td class="px-2 py-1.5">{SPEC[r.spec]}</td>
							<td class="tabular px-2 py-1.5 text-right">{r.n}</td>
							<td class="tabular px-2 py-1.5 text-right"
								>{signed(
									r.slope_per_sd
								)}{#if r.slope_per_sd !== null && Math.sign(r.slope_per_sd) !== r.expected_sign}<span
										class="text-muted"
										title="signe contraire à l'hypothèse">*</span
									>{/if}</td
							>
							<td class="tabular px-2 py-1.5 text-right">{signed(r.gain)}</td>
							<td class="tabular px-2 py-1.5 text-right"
								>{r.p_placebo === null ? '—' : pval(r.p_placebo)}</td
							>
							<td class="px-3 py-1.5 whitespace-nowrap">{levelVerdict(r)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="mt-1 text-xs text-muted">
			* signe contraire à l'hypothèse. Pente en points de part du bloc par écart type de l'attitude.
			Testable : au moins quatre élections. Placebo : toutes les permutations des valeurs entre
			élections.
		</p>
		{#if op.study.dynamic.length}
			<div class="mt-4 overflow-x-auto rounded-md border border-line bg-surface">
				<table class="w-full text-sm">
					<caption class="px-3 pt-2 text-left text-xs text-muted">
						Dynamique : variation de la part du bloc entre deux vagues Eurobaromètre et variation de
						la préoccupation (retard 0 : même période, sans valeur prédictive ; retard 1 : vague
						précédente).
					</caption>
					<thead class="border-b border-line text-left text-xs text-muted">
						<tr>
							<th scope="col" class="sticky left-0 bg-surface px-3 py-2 font-medium">Série</th>
							<th scope="col" class="px-2 py-2 font-medium">Bloc</th>
							<th scope="col" class="px-2 py-2 text-right font-medium">Retard</th>
							<th scope="col" class="px-2 py-2 text-right font-medium">Variations</th>
							<th scope="col" class="px-2 py-2 text-right font-medium">Effet / é.-t.</th>
							<th scope="col" class="px-2 py-2 text-right font-medium">p placebo</th>
							<th scope="col" class="px-3 py-2 font-medium">Tient</th>
						</tr>
					</thead>
					<tbody>
						{#each op.study.dynamic as r (r.bloc + r.series_id + r.lag)}
							<tr class={['border-b border-line last:border-0', r.lag === 0 && 'text-muted']}>
								<th
									scope="row"
									class="sticky left-0 max-w-56 truncate bg-surface px-3 py-1.5 text-left font-normal"
									title={r.series_label}>{r.series_label}</th
								>
								<td class="px-2 py-1.5 whitespace-nowrap">{r.bloc_label}</td>
								<td class="tabular px-2 py-1.5 text-right">{r.lag}</td>
								<td class="tabular px-2 py-1.5 text-right">{r.n}</td>
								<td class="tabular px-2 py-1.5 text-right">{signed(r.effect_per_sd)}</td>
								<td class="tabular px-2 py-1.5 text-right"
									>{r.p_placebo === null ? '—' : pval(r.p_placebo)}</td
								>
								<td class="px-3 py-1.5 whitespace-nowrap">{dynVerdict(r)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
		<p class="mt-2 text-xs text-muted">Détail : {op.study.reference} dans le dépôt.</p>
	</section>

	{#each op.themes as theme (theme.id)}
		<section aria-labelledby="theme-{theme.id}" class="mb-10">
			<h3 id="theme-{theme.id}" class="mb-3 text-lg font-semibold">{theme.label}</h3>
			<div class="grid gap-4 md:grid-cols-2">
				{#each theme.series as s (s.id)}
					{@const track = blocTrack.get(s.bloc)}
					<article class="min-w-0 rounded-md border border-line bg-surface p-4">
						<h4 class="text-sm font-medium">{s.label}</h4>
						<p class="mt-1 text-xs leading-relaxed text-ink-2">
							« {s.question} »{#if s.answer}<span> — réponse comptée : {s.answer}</span>{/if}
						</p>
						<p class="mt-1 mb-2 text-xs text-muted">
							{s.population} · série {OP_FREQ[s.frequency]}, {s.unit} ·
							<a href={s.source.href} class="text-accent" rel="external noopener" target="_blank"
								>{s.source.label}</a
							>
						</p>
						<OpinionChart
							series={[{ id: s.id, label: s.label, color: 'var(--color-ink)', points: s.points }]}
							label="{s.label}, et part du bloc lié dans les sondages"
							unit={s.unit}
							bloc={track ?? null}
							markers={op.first_rounds}
							breaks={s.breaks}
							end={eco.asof}
						/>
						{#if s.breaks.length}
							<ul class="mt-2 space-y-0.5 text-xs text-warn-ink">
								{#each s.breaks as b (b.date + b.label)}
									<li>▼ {fmt.date(b.date)} : {b.label}</li>
								{/each}
							</ul>
						{/if}
						{#if s.changes || s.missing || s.note || s.excluded_values.length}
							<details class="mt-2 text-xs text-ink-2">
								<summary class="cursor-pointer text-muted">Méthode, ruptures et lacunes</summary>
								<div class="mt-1 space-y-1 leading-relaxed">
									{#if s.changes}<p><strong>Changements :</strong> {s.changes}</p>{/if}
									{#if s.note}<p><strong>Lecture :</strong> {s.note}</p>{/if}
									{#if s.missing}<p><strong>Manque :</strong> {s.missing}</p>{/if}
									{#each s.excluded_values as x (x.document)}
										<p><strong>Valeurs retirées :</strong> {x.reason}.</p>
									{/each}
								</div>
							</details>
						{/if}
					</article>
				{/each}
			</div>
		</section>
	{/each}

	{#if blocNotes.length}
		<p class="mb-4 max-w-3xl text-xs text-muted">
			{#each blocNotes as b (b.bloc)}<span>{b.note} </span>{/each}Part du bloc : moyenne mensuelle
			des instituts (sondages de premier tour, hypothèses avec Emmanuel Macron écartées pour 2027) ;
			un mois sans sondage reste vide.
		</p>
	{/if}

	{#if op.excluded.length}
		<section aria-labelledby="titre-exclues" class="max-w-3xl">
			<h3 id="titre-exclues" class="mb-2 text-base font-semibold">Séries non publiées</h3>
			<ul class="list-disc space-y-1 pl-5 text-sm text-ink-2">
				{#each op.excluded as x (x.series_id)}
					<li>{x.label} ({x.n_values} valeurs) : {x.reason}.</li>
				{/each}
			</ul>
		</section>
	{/if}
</section>
