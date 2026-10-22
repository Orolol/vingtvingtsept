<!-- Séries par candidat sur une grille de dates (attention Wikipédia, volume GDELT, ton de la
     presse) : une ligne par candidat, étiquetée à droite, valeurs au survol. Les valeurs nulles
     coupent la ligne. En option : bande d'incertitude par série (`lo`, `hi`), libellé et couleur
     propres (série qui n'est pas un candidat), domaine vertical imposé (petits multiples), traits
     verticaux datés (`markers`). L'axe vertical inclut toujours zéro. -->
<script lang="ts">
	import { scaleLinear, scaleUtc } from 'd3-scale';
	import { area, line } from 'd3-shape';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { date as dateLong, dateShortFmt, parseDate } from '#lib/format.js';

	type Track = {
		candidate_id: string;
		values: (number | null)[];
		lo?: (number | null)[];
		hi?: (number | null)[];
		label?: string;
		color?: string;
	};
	type Marker = { date: string; label: string; kind?: 'event' | 'polls' | 'series' };

	let {
		dates,
		tracks,
		candidates,
		format,
		label,
		period = 'week',
		height = 260,
		yDomain = null,
		markers = [],
		showLabels = true
	}: {
		dates: string[];
		tracks: Track[];
		candidates: CandidateRegistry;
		format: (v: number) => string;
		label: string;
		period?: 'week' | 'month' | 'day';
		height?: number;
		yDomain?: [number, number] | null;
		markers?: Marker[];
		showLabels?: boolean;
	} = $props();

	const colorOf = (t: Track) => t.color ?? candidates.color(t.candidate_id);
	const nameOf = (t: Track) => t.label ?? candidates.short(t.candidate_id);
	const byId = $derived(new Map(tracks.map((t) => [t.candidate_id, t])));

	let width = $state(720);
	const narrow = $derived(width < 520);
	const m = $derived({
		top: 10 + 13 * markers.length,
		right: showLabels ? (narrow ? 84 : 100) : 12,
		bottom: 26,
		left: 40
	});
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);

	const times = $derived(dates.map(parseDate));
	const x = $derived(
		scaleUtc()
			.domain([times[0], times[times.length - 1]])
			.range([0, innerW])
	);
	const finite = (xs: (number | null)[] | undefined) =>
		(xs ?? []).filter((v): v is number => v !== null);
	const all = $derived(
		tracks.flatMap((t) => [...finite(t.values), ...finite(t.lo), ...finite(t.hi)])
	);
	const y = $derived(
		yDomain
			? scaleLinear().domain(yDomain).range([innerH, 0])
			: scaleLinear()
					.domain([Math.min(0, ...all), Math.max(1e-9, 0, ...all)])
					.nice()
					.range([innerH, 0])
	);
	const indices = $derived(dates.map((_, i) => i));
	function bandPath(t: Track): string {
		const { lo, hi } = t;
		if (!lo || !hi) return '';
		return (
			area<number>()
				.defined((i) => lo[i] != null && hi[i] != null)
				.x((i) => x(times[i]))
				.y0((i) => y(lo[i] as number))
				.y1((i) => y(hi[i] as number))(indices) ?? ''
		);
	}
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
			.map((it) => ({ id: it.id, y: y(it.v), t: byId.get(it.id) as Track }))
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
					.map((t) => ({
						id: t.candidate_id,
						t,
						v: t.values[hover as number],
						lo: t.lo?.[hover as number] ?? null,
						hi: t.hi?.[hover as number] ?? null
					}))
					.filter(
						(r): r is { id: string; t: Track; v: number; lo: number | null; hi: number | null } =>
							r.v !== null && r.v !== undefined
					)
					.sort((a, b) => b.v - a.v)
	);
	const tickFmt = $derived(
		new Intl.DateTimeFormat('fr-FR', {
			day: period === 'day' ? 'numeric' : undefined,
			month: 'short',
			year: period === 'month' || (period === 'week' && dates.length > 60) ? '2-digit' : undefined,
			timeZone: 'UTC'
		})
	);
	const markerLines = $derived(
		markers
			.filter((mk) => mk.date >= dates[0] && mk.date <= dates[dates.length - 1])
			.map((mk, i) => ({ ...mk, x: x(parseDate(mk.date)), row: i }))
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
				{#if y.domain()[0] < 0}
					<line x1="0" x2={innerW} y1={y(0)} y2={y(0)} class="stroke-axis" />
				{/if}
				{#each markerLines as mk (mk.date + mk.label)}
					<line
						x1={mk.x}
						x2={mk.x}
						y1={-13 * (markerLines.length - mk.row) + 4}
						y2={innerH}
						class={mk.kind === 'polls' ? 'stroke-ink' : 'stroke-muted'}
						stroke-width={mk.kind === 'polls' ? 1.5 : 1}
						stroke-dasharray={mk.kind === 'event' ? '2 3' : mk.kind === 'series' ? '5 3' : null}
					/>
					<text
						x={mk.x + (mk.x > innerW - 70 ? -4 : 4)}
						y={-13 * (markerLines.length - mk.row) + 4}
						dy="0.32em"
						text-anchor={mk.x > innerW - 70 ? 'end' : 'start'}
						class="fill-ink-2 text-[11px]">{mk.label}</text
					>
				{/each}
				{#each tracks as t (t.candidate_id)}
					{#if t.lo && t.hi}
						<path d={bandPath(t)} style:fill={colorOf(t)} opacity="0.14" />
					{/if}
				{/each}
				{#each tracks as t (t.candidate_id)}
					<path
						d={path(t.values)}
						fill="none"
						style:stroke={colorOf(t)}
						stroke-width="1.75"
						stroke-linejoin="round"
					/>
				{/each}
				{#if showLabels}
					{#each labels as l (l.id)}
						<text x={innerW + 8} y={l.y} dy="0.32em" class="fill-ink text-[11px]">
							<tspan style:fill={colorOf(l.t)}>●</tspan>
							{nameOf(l.t)}
						</text>
					{/each}
				{/if}
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
							style:fill={colorOf(r.t)}
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
				style:left="{Math.max(0, Math.min(m.left + x(times[hover]) + 12, width - 240))}px"
			>
				<p class="mb-1 font-medium">
					{period === 'month'
						? monthYear.format(times[hover])
						: period === 'day'
							? dateLong(dates[hover])
							: `Semaine du ${dateShortFmt(dates[hover])}`}
				</p>
				<ul class="space-y-0.5">
					{#each hoverRows as r (r.id)}
						<li class="flex items-center gap-2 whitespace-nowrap">
							<span class="inline-block size-2 rounded-full" style:background={colorOf(r.t)}></span>
							<span class="flex-1">{nameOf(r.t)}</span>
							<span class="tabular text-ink-2"
								>{format(r.v)}{#if r.lo !== null && r.hi !== null}
									<span class="text-muted"> ({format(r.lo)} à {format(r.hi)})</span>{/if}</span
							>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	{/if}
</div>
