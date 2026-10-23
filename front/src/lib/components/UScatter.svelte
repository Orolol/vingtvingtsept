<!-- Nuage u (décalage des reports vers le RN) contre « danger pour la démocratie » : un point
     par élection avec son intervalle à 90 %, la croix du résultat agrégé, la droite pondérée et
     sa bande, le point 2012 hypothétique en creux, la dernière vague avant 2027 en trait
     vertical, et l'effectif (n = 3) écrit sur la figure. -->
<script lang="ts">
	import { scaleLinear } from 'd3-scale';
	import { area, line } from 'd3-shape';
	import { decimal, points as signed } from '#lib/format.js';
	import type { RnImage } from '#lib/types/data.js';

	let { scatter, height = 300 }: { scatter: RnImage['scatter']; height?: number } = $props();

	let width = $state(420);
	const m = { top: 16, right: 16, bottom: 40, left: 44 };
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);
	const h12 = $derived(scatter.hypothetical_2012);

	const xs = $derived([
		...scatter.line.map((l) => l.x),
		...scatter.points.map((p) => p.x),
		...(scatter.x_2027 !== null ? [scatter.x_2027] : [])
	]);
	const ys = $derived([
		...scatter.points.flatMap((p) => [p.lo90, p.hi90, p.u_result]),
		...scatter.line.flatMap((l) => [l.lo90, l.hi90]),
		...(h12 ? [h12.lo90, h12.hi90] : []),
		0
	]);
	const x = $derived(
		scaleLinear()
			.domain([Math.min(...xs), Math.max(...xs)])
			.range([0, innerW])
	);
	const y = $derived(
		scaleLinear()
			.domain([Math.min(...ys), Math.max(...ys)])
			.nice()
			.range([innerH, 0])
	);
	type L = RnImage['scatter']['line'][number];
	const band = $derived(
		area<L>()
			.x((l) => x(l.x))
			.y0((l) => y(l.lo90))
			.y1((l) => y(l.hi90))(scatter.line) ?? ''
	);
	const mid = $derived(
		line<L>()
			.x((l) => x(l.x))
			.y((l) => y(l.u))(scatter.line) ?? ''
	);
	const SERIES = 'light-dark(#2a78d6, #3987e5)';
	const RESULT = 'light-dark(#e34948, #e66767)';
</script>

<div class="w-full" bind:clientWidth={width}>
	<svg
		{width}
		{height}
		role="img"
		aria-label="u contre la part de « danger pour la démocratie », {scatter.n} élections"
		class="block"
	>
		<g transform="translate({m.left},{m.top})">
			{#each y.ticks(5) as t (t)}
				<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
				<text x="-6" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[10px]"
					>{decimal(t, 1)}</text
				>
			{/each}
			{#each x.ticks(Math.max(3, Math.floor(innerW / 60))) as t (t)}
				<text x={x(t)} y={innerH + 15} text-anchor="middle" class="tabular fill-muted text-[10px]"
					>{decimal(t, 0)} %</text
				>
			{/each}
			<line x1="0" x2={innerW} y1={y(0)} y2={y(0)} class="stroke-axis" stroke-dasharray="2 3" />
			<line x1="0" x2={innerW} y1={innerH} y2={innerH} class="stroke-axis" />
			<path d={band} class="fill-muted" opacity="0.1" />
			<path d={mid} fill="none" class="stroke-muted" stroke-width="1.2" />
			{#if scatter.x_2027 !== null}
				<line
					x1={x(scatter.x_2027)}
					x2={x(scatter.x_2027)}
					y1="0"
					y2={innerH}
					class="stroke-ink-2"
					stroke-dasharray="4 3"
				/>
				<text x={x(scatter.x_2027) + 4} y="8" class="fill-ink-2 text-[10px]"
					>2027 : {decimal(scatter.x_2027, 0)} %</text
				>
			{/if}
			{#if h12}
				<line
					x1={x(h12.x)}
					x2={x(h12.x)}
					y1={y(h12.lo90)}
					y2={y(h12.hi90)}
					class="stroke-ink-2"
					stroke-width="1"
				/>
				<rect
					x={x(h12.x) - 4}
					y={y(h12.u) - 4}
					width="8"
					height="8"
					class="fill-surface stroke-ink-2"
					stroke-width="1.2"
				>
					<title
						>2012 hypothétique ({h12.n_duels} sondages de duel de 2011) : u = {signed(h12.u)}</title
					>
				</rect>
				<text x={x(h12.x) + 7} y={y(h12.u)} dy="0.32em" class="fill-ink-2 text-[10px]"
					>2012 hyp.</text
				>
			{/if}
			{#each scatter.points as p (p.election)}
				<line
					x1={x(p.x)}
					x2={x(p.x)}
					y1={y(p.lo90)}
					y2={y(p.hi90)}
					style:stroke={SERIES}
					stroke-width="1.5"
				/>
				<circle cx={x(p.x)} cy={y(p.u)} r="4.5" style:fill={SERIES}>
					<title
						>{p.election} : danger {decimal(p.x, 0)} %, u = {signed(p.u)} ({signed(p.lo90)} à {signed(
							p.hi90
						)})</title
					>
				</circle>
				<path
					d="M{x(p.x) - 4},{y(p.u_result) - 4} L{x(p.x) + 4},{y(p.u_result) + 4} M{x(p.x) - 4},{y(
						p.u_result
					) + 4} L{x(p.x) + 4},{y(p.u_result) - 4}"
					style:stroke={RESULT}
					stroke-width="1.5"
				/>
				<text
					x={x(p.x) + (x(p.x) > innerW - 40 ? -8 : 8)}
					y={y(p.u) - 8}
					text-anchor={x(p.x) > innerW - 40 ? 'end' : 'start'}
					class="fill-ink text-[11px] font-medium">{p.election}</text
				>
			{/each}
			<text x={innerW - 4} y="18" text-anchor="end" class="fill-ink text-[15px] font-semibold"
				>n = {scatter.n}</text
			>
			<text x={innerW / 2} y={innerH + 32} text-anchor="middle" class="fill-muted text-[10px]"
				>{scatter.x_label}</text
			>
			<text
				transform="translate({-34},{innerH / 2}) rotate(-90)"
				text-anchor="middle"
				class="fill-muted text-[10px]">u (logit, vers le RN)</text
			>
		</g>
	</svg>
	<ul class="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-ink-2">
		<li>
			<span class="inline-block size-2 rounded-full" style:background={SERIES}></span> u des enquêtes,
			90 %
		</li>
		<li><span style:color={RESULT}>×</span> u du résultat officiel</li>
		<li><span class="inline-block h-2 w-3 bg-muted/30"></span> droite pondérée, bande à 90 %</li>
		{#if h12}<li>□ 2012 hypothétique (hors droite)</li>{/if}
	</ul>
</div>
