<!-- Séries par candidat sur une grille de dates (attention Wikipédia, volume GDELT) : une ligne
     par candidat, étiquetée à droite, valeurs au survol. Les valeurs nulles coupent la ligne. -->
<script lang="ts">
	import { scaleLinear, scaleUtc } from 'd3-scale';
	import { line } from 'd3-shape';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { dateShortFmt, parseDate } from '#lib/format.js';

	let {
		dates,
		tracks,
		candidates,
		format,
		label,
		period = 'week',
		height = 260
	}: {
		dates: string[];
		tracks: { candidate_id: string; values: (number | null)[] }[];
		candidates: CandidateRegistry;
		format: (v: number) => string;
		label: string;
		period?: 'week' | 'month';
		height?: number;
	} = $props();

	let width = $state(720);
	const narrow = $derived(width < 520);
	const m = $derived({ top: 10, right: narrow ? 74 : 100, bottom: 26, left: 40 });
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);

	const times = $derived(dates.map(parseDate));
	const x = $derived(
		scaleUtc()
			.domain([times[0], times[times.length - 1]])
			.range([0, innerW])
	);
	const yMax = $derived(
		Math.max(1e-9, ...tracks.flatMap((t) => t.values.filter((v): v is number => v !== null)))
	);
	const y = $derived(scaleLinear().domain([0, yMax]).nice().range([innerH, 0]));
	const path = $derived(
		line<number | null>()
			.defined((v) => v !== null)
			.x((_, i) => x(times[i]))
			.y((v) => y(v as number))
	);
	const lastValue = (t: { values: (number | null)[] }) =>
		t.values.findLast((v) => v !== null) ?? null;

	const labels = $derived.by(() => {
		const items = tracks
			.map((t) => ({ id: t.candidate_id, v: lastValue(t) }))
			.filter((it): it is { id: string; v: number } => it.v !== null)
			.map((it) => ({ id: it.id, y: y(it.v) }))
			.sort((a, b) => a.y - b.y);
		for (let i = 1; i < items.length; i++)
			if (items[i].y - items[i - 1].y < 13) items[i].y = items[i - 1].y + 13;
		const overflow = items.length ? items[items.length - 1].y - innerH : 0;
		if (overflow > 0) for (const it of items) it.y -= overflow;
		return items;
	});

	let hover = $state<number | null>(null);
	function onMove(e: PointerEvent) {
		const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
		const t = x.invert(e.clientX - rect.left).getTime();
		let best = 0;
		for (let i = 1; i < times.length; i++)
			if (Math.abs(times[i].getTime() - t) < Math.abs(times[best].getTime() - t)) best = i;
		hover = best;
	}
	const hoverRows = $derived(
		hover === null
			? []
			: tracks
					.map((t) => ({ id: t.candidate_id, v: t.values[hover as number] }))
					.filter((r): r is { id: string; v: number } => r.v !== null && r.v !== undefined)
					.sort((a, b) => b.v - a.v)
	);
	const tickFmt = $derived(
		new Intl.DateTimeFormat('fr-FR', {
			month: 'short',
			year: period === 'month' || dates.length > 60 ? '2-digit' : undefined,
			timeZone: 'UTC'
		})
	);
	const monthYear = new Intl.DateTimeFormat('fr-FR', {
		month: 'long',
		year: 'numeric',
		timeZone: 'UTC'
	});
</script>

<div class="relative w-full" bind:clientWidth={width}>
	{#if dates.length < 2 || tracks.length === 0}
		<p class="text-sm text-muted">Pas assez de données.</p>
	{:else}
		<svg {width} {height} role="img" aria-label={label} class="block touch-pan-y select-none">
			<g transform="translate({m.left},{m.top})">
				{#each y.ticks(4) as t (t)}
					<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
					<text x="-6" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[11px]"
						>{format(t)}</text
					>
				{/each}
				{#each x.ticks(Math.max(2, Math.floor(innerW / 90))) as t (t.getTime())}
					<text x={x(t)} y={innerH + 18} text-anchor="middle" class="fill-muted text-[11px]"
						>{tickFmt.format(t)}</text
					>
				{/each}
				<line x1="0" x2={innerW} y1={innerH} y2={innerH} class="stroke-axis" />
				{#each tracks as t (t.candidate_id)}
					<path
						d={path(t.values)}
						fill="none"
						style:stroke={candidates.color(t.candidate_id)}
						stroke-width="1.75"
						stroke-linejoin="round"
					/>
				{/each}
				{#each labels as l (l.id)}
					<text x={innerW + 8} y={l.y} dy="0.32em" class="fill-ink text-[11px]">
						<tspan style:fill={candidates.color(l.id)}>●</tspan>
						{candidates.short(l.id)}
					</text>
				{/each}
				{#if hover !== null}
					<line
						x1={x(times[hover])}
						x2={x(times[hover])}
						y1="0"
						y2={innerH}
						class="stroke-ink"
						opacity="0.35"
					/>
					{#each hoverRows as r (r.id)}
						<circle
							cx={x(times[hover])}
							cy={y(r.v)}
							r="3.5"
							style:fill={candidates.color(r.id)}
							class="stroke-surface"
							stroke-width="1.5"
						/>
					{/each}
				{/if}
				<rect
					width={innerW}
					height={innerH}
					fill="transparent"
					role="presentation"
					onpointermove={onMove}
					onpointerdown={onMove}
					onpointerleave={() => (hover = null)}
				/>
			</g>
		</svg>
		{#if hover !== null}
			<div
				class="pointer-events-none absolute top-1 z-10 rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-sm"
				style:left="{Math.max(0, Math.min(m.left + x(times[hover]) + 12, width - 200))}px"
			>
				<p class="mb-1 font-medium">
					{period === 'month'
						? monthYear.format(times[hover])
						: `Semaine du ${dateShortFmt(dates[hover])}`}
				</p>
				<ul class="space-y-0.5">
					{#each hoverRows as r (r.id)}
						<li class="flex items-center gap-2 whitespace-nowrap">
							<span
								class="inline-block size-2 rounded-full"
								style:background={candidates.color(r.id)}
							></span>
							<span class="flex-1">{candidates.short(r.id)}</span>
							<span class="tabular text-ink-2">{format(r.v)}</span>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	{/if}
</div>
