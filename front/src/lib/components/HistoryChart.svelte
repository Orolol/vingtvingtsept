<!-- Évolution des probabilités (mélange de scénarios) au fil des runs publiés. -->
<script lang="ts">
	import { scaleLinear, scaleUtc } from 'd3-scale';
	import { line } from 'd3-shape';
	import type { HistoryEntry } from '#lib/types/data.js';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { dateShortFmt, parseDate, prob } from '#lib/format.js';

	let {
		entries,
		candidates,
		metric,
		ids,
		height = 280
	}: {
		entries: HistoryEntry[];
		candidates: CandidateRegistry;
		metric: 'p_win' | 'p_qualify';
		ids: string[];
		height?: number;
	} = $props();

	let width = $state(720);
	const narrow = $derived(width < 520);
	const m = $derived({ top: 10, right: narrow ? 70 : 96, bottom: 26, left: 40 });
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);

	const times = $derived(entries.map((e) => parseDate(e.run_date)));
	const x = $derived(
		scaleUtc()
			.domain([times[0], times[times.length - 1]])
			.range([0, innerW])
	);
	const valueOf = (e: HistoryEntry, id: string) =>
		e.candidates.find((c) => c.candidate_id === id)?.[metric] ?? 0;
	const yMax = $derived(
		Math.min(
			1,
			Math.ceil(Math.max(0.1, ...entries.flatMap((e) => ids.map((id) => valueOf(e, id)))) * 10) / 10
		)
	);
	const y = $derived(scaleLinear().domain([0, yMax]).range([innerH, 0]));
	const path = $derived(
		line<[Date, number]>()
			.x((d) => x(d[0]))
			.y((d) => y(d[1]))
	);
	const pointsOf = (id: string): [Date, number][] =>
		entries.map((e, i) => [times[i], valueOf(e, id)]);

	const labels = $derived.by(() => {
		const last = entries[entries.length - 1];
		const items = ids.map((id) => ({ id, y: y(valueOf(last, id)) })).sort((a, b) => a.y - b.y);
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
			: ids
					.map((id) => ({ id, v: valueOf(entries[hover as number], id) }))
					.sort((a, b) => b.v - a.v)
	);
	const monthFmt = new Intl.DateTimeFormat('fr-FR', { month: 'short', timeZone: 'UTC' });
</script>

<div class="relative w-full" bind:clientWidth={width}>
	{#if entries.length < 2}
		<p class="text-sm text-muted">L'évolution s'affichera à partir du deuxième run publié.</p>
	{:else}
		<svg
			{width}
			{height}
			role="img"
			aria-label="Évolution des probabilités de {metric === 'p_win'
				? 'victoire'
				: 'qualification'} par candidat"
			class="block touch-pan-y select-none"
		>
			<g transform="translate({m.left},{m.top})">
				{#each y.ticks(4) as t (t)}
					<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
					<text x="-6" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[11px]"
						>{Math.round(t * 100)} %</text
					>
				{/each}
				{#each x.ticks(Math.max(2, Math.floor(innerW / 90))) as t (t.getTime())}
					<text x={x(t)} y={innerH + 18} text-anchor="middle" class="fill-muted text-[11px]"
						>{monthFmt.format(t)}</text
					>
				{/each}
				<line x1="0" x2={innerW} y1={innerH} y2={innerH} class="stroke-axis" />
				{#each ids as id (id)}
					<path
						d={path(pointsOf(id))}
						fill="none"
						style:stroke={candidates.color(id)}
						stroke-width="2"
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
				style:left="{Math.min(m.left + x(times[hover]) + 12, width - 190)}px"
			>
				<p class="mb-1 font-medium">Run du {dateShortFmt(entries[hover].run_date)}</p>
				<ul class="space-y-0.5">
					{#each hoverRows as r (r.id)}
						<li class="flex items-center gap-2 whitespace-nowrap">
							<span
								class="inline-block size-2 rounded-full"
								style:background={candidates.color(r.id)}
							></span>
							<span class="flex-1">{candidates.short(r.id)}</span>
							<span class="tabular text-ink-2">{prob(r.v)}</span>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	{/if}
</div>
