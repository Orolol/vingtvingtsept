<!-- Anticipations « qui sera élu ? » des campagnes passées : part du futur vainqueur parmi les
     répondants qui nomment un candidat, en fonction du nombre de jours avant le premier tour
     (le vote à droite). Un cercle creux marque une mesure où le plus cité n'a pas été élu. En
     option, la mesure 2027 la plus récente (`current`), en losange. -->
<script lang="ts">
	import { scaleLinear } from 'd3-scale';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { decimal, dateNum } from '#lib/format.js';
	import type { Expectations } from '#lib/types/data.js';

	let {
		history,
		candidates,
		current = null,
		label,
		height = 220
	}: {
		history: Expectations['history'];
		candidates: CandidateRegistry;
		current?: { days_before: number; share: number; candidate_id: string } | null;
		label: string;
		height?: number;
	} = $props();

	// Une teinte par élection, lisible en clair et en sombre.
	const COLORS = [
		'light-dark(#6250d6, #9085e9)',
		'light-dark(#1baf7a, #199e70)',
		'light-dark(#eb6834, #d95926)',
		'light-dark(#2a78d6, #3987e5)',
		'light-dark(#e87ba4, #d55181)'
	];
	const colorOf = (i: number) => COLORS[i % COLORS.length];

	let width = $state(640);
	const m = { top: 10, right: 16, bottom: 30, left: 36 };
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);
	const maxDays = $derived(
		Math.max(
			30,
			current?.days_before ?? 0,
			...history.flatMap((e) => e.points.map((p) => p.days_before))
		)
	);
	const x = $derived(
		scaleLinear()
			.domain([Math.ceil(maxDays / 30) * 30, 0])
			.range([0, innerW])
	);
	const y = $derived(scaleLinear().domain([0, 100]).range([innerH, 0]));
	const xTicks = $derived(x.ticks(Math.max(2, Math.floor(innerW / 80))));
</script>

<div class="w-full" bind:clientWidth={width}>
	<svg {width} {height} role="img" aria-label={label} class="block">
		<g transform="translate({m.left},{m.top})">
			{#each [0, 25, 50, 75, 100] as t (t)}
				<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
				<text x="-6" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[11px]"
					>{t}</text
				>
			{/each}
			{#each xTicks as t (t)}
				<text x={x(t)} y={innerH + 18} text-anchor="middle" class="tabular fill-muted text-[11px]"
					>{t === 0 ? 'vote' : `${t} j`}</text
				>
			{/each}
			<line x1="0" x2={innerW} y1={innerH} y2={innerH} class="stroke-axis" />
			{#each history as e, i (e.election)}
				{#each e.points.filter((p) => p.winner_share_named !== null) as p (p.pollster + p.fieldwork_end)}
					<circle
						cx={x(p.days_before)}
						cy={y(p.winner_share_named as number)}
						r="3.5"
						style:fill={p.top_is_winner ? colorOf(i) : 'none'}
						style:stroke={colorOf(i)}
						stroke-width="1.5"
					>
						<title
							>{e.election}, {p.pollster}, {dateNum(p.fieldwork_end)} : {candidates.short(e.winner)}
							{decimal(p.winner_share_named as number, 0)} %{p.top_is_winner
								? ''
								: ` (le plus cité : ${candidates.short(p.top_candidate_id)})`}</title
						>
					</circle>
				{/each}
			{/each}
			{#if current}
				{@const cx = x(current.days_before)}
				{@const cy = y(current.share)}
				<path
					d="M {cx} {cy - 6} L {cx + 6} {cy} L {cx} {cy + 6} L {cx - 6} {cy} Z"
					class="fill-ink"
				/>
				<text
					x={cx + (cx > innerW - 90 ? -9 : 9)}
					y={cy}
					dy="0.32em"
					text-anchor={cx > innerW - 90 ? 'end' : 'start'}
					class="fill-ink text-[11px]"
					>2027 : {candidates.short(current.candidate_id)} {decimal(current.share, 0)} %</text
				>
			{/if}
		</g>
	</svg>
	<ul class="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
		{#each history as e, i (e.election)}
			<li>
				<span style:color={colorOf(i)}>●</span>
				{e.election} ({candidates.short(e.winner)}, {e.points.length} mesure{e.points.length > 1
					? 's'
					: ''})
			</li>
		{/each}
		<li><span class="text-muted">○</span> le plus cité n'a pas été élu</li>
	</ul>
</div>
