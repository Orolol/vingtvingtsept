<script lang="ts">
	import { candidateRegistry } from '#lib/candidates.js';
	import * as fmt from '#lib/format.js';
	import { indexDuels, orient, pairKey } from '#lib/duels.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import DuelMatrix from '#lib/components/DuelMatrix.svelte';
	import CandidateName from '#lib/components/CandidateName.svelte';
	import IntervalStrip from '#lib/components/IntervalStrip.svelte';
	import ForecastNote from '#lib/components/ForecastNote.svelte';
	import OpinionChart from '#lib/components/OpinionChart.svelte';
	import UScatter from '#lib/components/UScatter.svelte';
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

	// Normalisation du RN : baromètre, u par élection, nuage, sensibilité par duel.
	const rn = $derived(sr.rn_image);
	const ITEM_COLOR: Record<string, string> = {
		danger: 'light-dark(#e34948, #e66767)',
		capable: 'light-dark(#1baf7a, #199e70)',
		adhesion: 'light-dark(#6250d6, #9085e9)'
	};
	const barometerSeries = $derived(
		rn.barometer.items.map((it) => ({
			id: it.id,
			label: it.label,
			color: ITEM_COLOR[it.id] ?? 'var(--color-ink)',
			points: it.points
		}))
	);
	const barometerBreaks = $derived([
		...new Map(rn.barometer.items.flatMap((it) => it.breaks).map((b) => [b.date, b])).values()
	]);
	let pollsMode = $state<'with_polls' | 'prior_only'>('with_polls');
	const hasPriorOnly = $derived(rn.duels.some((d) => d.rows.some((r) => r.polls === 'prior_only')));
	const signedU = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${fmt.decimal(Math.abs(v), 2)}`;
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

<section aria-labelledby="titre-rn" class="mt-14 border-t border-line pt-8">
	<h2 id="titre-rn" class="mb-1 text-xl font-semibold">Normalisation du RN et front républicain</h2>
	<p class="mb-4 max-w-3xl text-sm text-ink-2">
		Le RN fait-il moins peur, et les électeurs des autres candidats font-ils moins barrage au second
		tour ? L'étude du {fmt.date(rn.study_date)} met en regard le baromètre d'image du FN/RN et les reports
		de voix observés aux trois seconds tours avec un finaliste FN/RN.
	</p>
	<ForecastNote inForecast={rn.in_forecast} note={rn.note} />

	<div class="mb-8 rounded-md border border-line bg-surface p-4">
		<h3 class="text-base font-medium">Baromètre d'image du FN/RN</h3>
		<p class="mt-1 mb-2 text-xs text-muted">
			{rn.barometer.population} ·
			<a href={rn.barometer.source.href} class="text-accent" rel="external noopener" target="_blank"
				>{rn.barometer.source.label}</a
			> · traits pointillés : premiers tours
		</p>
		<OpinionChart
			series={barometerSeries}
			label="Baromètre d'image du FN/RN : danger pour la démocratie, capacité à gouverner, adhésion aux idées"
			markers={rn.barometer.first_rounds}
			breaks={barometerBreaks}
			end={sr.asof}
			height={220}
		/>
		{#each barometerBreaks as b (b.date)}
			<p class="mt-2 text-xs text-warn-ink">▼ {fmt.date(b.date)} : {b.label}</p>
		{/each}
		<details class="mt-2 text-xs text-ink-2">
			<summary class="cursor-pointer text-muted">Questions posées et changements</summary>
			<div class="mt-1 space-y-1 leading-relaxed">
				{#each rn.barometer.items as it (it.id)}
					<p>
						<strong>{it.label} :</strong> « {it.question} »{#if it.answer}<span>
								— réponse comptée : {it.answer}</span
							>{/if}
					</p>
				{/each}
				{#if rn.barometer.changes}<p><strong>Changements :</strong> {rn.barometer.changes}</p>{/if}
			</div>
		</details>
	</div>

	<div class="mb-8 grid gap-4 lg:grid-cols-2">
		<div class="min-w-0 rounded-md border border-line bg-surface p-4">
			<h3 class="mb-2 text-base font-medium">Décalage des reports par élection</h3>
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<caption class="sr-only">Décalage commun u des reports vers le finaliste FN/RN</caption>
					<thead class="border-b border-line text-left text-xs text-muted">
						<tr>
							<th scope="col" class="py-2 pr-2 font-medium">Élection</th>
							<th scope="col" class="px-2 py-2 text-right font-medium">u enquêtes (90 %)</th>
							<th scope="col" class="px-2 py-2 text-right font-medium">u résultat</th>
							<th scope="col" class="py-2 pl-2 text-right font-medium">Score FN/RN</th>
						</tr>
					</thead>
					<tbody>
						{#each rn.u_observed as u (u.election)}
							<tr class="border-b border-line last:border-0">
								<th scope="row" class="py-1.5 pr-2 text-left font-normal">
									{u.election} <span class="block text-xs text-muted">{u.duel}</span>
								</th>
								<td class="tabular px-2 py-1.5 text-right whitespace-nowrap"
									>{signedU(u.u)}
									<span class="text-muted">({signedU(u.lo90)} à {signedU(u.hi90)})</span></td
								>
								<td class="tabular px-2 py-1.5 text-right">{signedU(u.u_result)}</td>
								<td class="tabular py-1.5 pl-2 text-right">{fmt.share(u.rn_result)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="mt-2 text-xs leading-relaxed text-muted">{rn.u_definition}</p>
		</div>
		<div class="min-w-0 rounded-md border border-line bg-surface p-4">
			<h3 class="mb-2 text-base font-medium">u contre « danger pour la démocratie »</h3>
			<UScatter scatter={rn.scatter} />
			<p class="mt-2 text-xs leading-relaxed text-muted">
				Pente MCO {signedU(rn.scatter.slope_ols)} logit par point (intervalle à 90 % {signedU(
					rn.scatter.slope_ols_lo90
				)} à {signedU(rn.scatter.slope_ols_hi90)}), R² = {fmt.decimal(rn.scatter.r2, 2)} sur {rn
					.scatter.n} points. Chaque élection prédite par la droite des deux autres :
				{#each rn.scatter.points as p, i (p.election)}{#if p.predicted_without !== null}{i > 0
							? ' ; '
							: ''}{p.election} prédit {signedU(p.predicted_without)}, observé {signedU(
							p.u
						)}{/if}{/each}.
			</p>
		</div>
	</div>

	<div>
		<h3 class="mb-1 text-base font-medium">Sensibilité des duels au centre des reports</h3>
		<p class="mb-3 max-w-3xl text-xs text-muted">{rn.duels_note}</p>
		{#if hasPriorOnly}
			<div role="radiogroup" aria-label="Données utilisées" class="mb-2 flex gap-2 text-sm">
				{#each [['with_polls', 'Avec les sondages de duel'], ['prior_only', 'Reports seuls, sans sondage']] as [id, lbl] (id)}
					<button
						type="button"
						role="radio"
						aria-checked={pollsMode === id}
						class={[
							'rounded-md border px-3 py-1',
							pollsMode === id
								? 'border-accent bg-raised font-medium text-ink'
								: 'border-line text-ink-2 hover:text-ink'
						]}
						onclick={() => (pollsMode = id as 'with_polls' | 'prior_only')}>{lbl}</button
					>
				{/each}
			</div>
		{/if}
		<div class="overflow-x-auto rounded-md border border-line bg-surface">
			<table class="w-full text-sm">
				<caption class="sr-only"
					>Part du candidat RN et probabilité de victoire selon le centrage des reports</caption
				>
				<thead class="border-b border-line text-left text-xs text-muted">
					<tr>
						<th scope="col" class="px-3 py-2 font-medium">Centrage des reports</th>
						<th scope="col" class="px-2 py-2 text-right font-medium">δ (logit)</th>
						<th scope="col" class="px-2 py-2 text-right font-medium">Part RN (90 %)</th>
						<th scope="col" class="px-3 py-2 text-right font-medium">P(RN gagne)</th>
					</tr>
				</thead>
				{#each rn.duels as d (d.rn_candidate + d.other_candidate)}
					<tbody class="border-b border-line last:border-0">
						<tr class="bg-raised">
							<th scope="rowgroup" colspan="4" class="px-3 py-1.5 text-left font-medium">
								{candidates.name(d.rn_candidate)} – {candidates.name(d.other_candidate)}
								<span class="font-normal text-muted"
									>· {pollsMode === 'with_polls'
										? `${d.n_polls} sondage${d.n_polls > 1 ? 's' : ''} de ce duel`
										: 'sans sondage de duel'}</span
								>
							</th>
						</tr>
						{#each d.rows.filter((r) => r.polls === pollsMode) as r (r.variant)}
							<tr class={[r.variant === 'default' && 'font-medium']}>
								<th scope="row" class="px-3 py-1.5 text-left font-normal">{r.label}</th>
								<td class="tabular px-2 py-1.5 text-right">{signedU(r.delta)}</td>
								<td class="tabular px-2 py-1.5 text-right whitespace-nowrap"
									>{fmt.share(r.median)}
									<span class="text-muted">({fmt.decimal(r.q05)} à {fmt.decimal(r.q95)})</span></td
								>
								<td class="tabular px-3 py-1.5 text-right">{fmt.prob(r.p_rn)}</td>
							</tr>
						{/each}
					</tbody>
				{/each}
			</table>
		</div>
		<p class="mt-1 max-w-3xl text-xs text-muted">
			δ : décalage ajouté au centre actuel des reports (logit, vers le RN). Ligne en gras : défaut
			du modèle. Détail : {rn.reference} dans le dépôt.
		</p>
	</div>
</section>
