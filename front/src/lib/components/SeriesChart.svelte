<!-- Série observée (économie, popularité) : une ligne et ses points, dernière valeur étiquetée. -->
<script lang="ts">
	import { scaleLinear, scaleUtc } from 'd3-scale';
	import { line } from 'd3-shape';
	import { extent } from 'd3-array';
	import { dateShortFmt, decimal, parseDate } from '#lib/format.js';

	let {
		points,
		unit,
		label,
		height = 160
	}: {
		points: { date: string; value: number }[];
		unit: string;
		label: string;
		height?: number;
	} = $props();

	let width = $state(320);
	const m = { top: 10, right: 48, bottom: 22, left: 36 };
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);
	const data = $derived(points.map((p) => ({ t: parseDate(p.date), v: p.value })));
	const x = $derived(
		scaleUtc()
			.domain(extent(data, (d) => d.t) as [Date, Date])
			.range([0, innerW])
	);
	const y = $derived(
		scaleLinear()
			.domain(extent(data, (d) => d.v) as [number, number])
			.nice()
			.range([innerH, 0])
	);
	const path = $derived(
		line<{ t: Date; v: number }>()
			.x((d) => x(d.t))
			.y((d) => y(d.v))
	);
	const last = $derived(data[data.length - 1]);
</script>

<div class="w-full" bind:clientWidth={width}>
	{#if data.length < 2}
		<p class="text-sm text-muted">Pas assez de points.</p>
	{:else}
		<svg {width} {height} role="img" aria-label={label} class="block">
			<g transform="translate({m.left},{m.top})">
				{#each y.ticks(3) as t (t)}
					<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
					<text x="-6" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[10px]"
						>{decimal(t, 0)}</text
					>
				{/each}
				<text x="0" y={innerH + 16} class="fill-muted text-[10px]"
					>{dateShortFmt(points[0].date)}</text
				>
				<text x={innerW} y={innerH + 16} text-anchor="end" class="fill-muted text-[10px]"
					>{dateShortFmt(points[points.length - 1].date)}</text
				>
				<path d={path(data)} fill="none" class="stroke-accent" stroke-width="2" />
				{#each data as d (d.t.getTime())}
					<circle cx={x(d.t)} cy={y(d.v)} r="2" class="fill-accent" />
				{/each}
				<text x={x(last.t) + 6} y={y(last.v)} dy="0.32em" class="tabular fill-ink text-[11px]"
					>{decimal(last.v)} {unit === '%' ? '%' : ''}</text
				>
			</g>
		</svg>
	{/if}
</div>
