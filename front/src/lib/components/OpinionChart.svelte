<!-- Séries d'opinion par vagues (dates irrégulières) : une ou plusieurs lignes à points, et en
     option, sous elles et sur le même axe du temps, la part mensuelle d'un bloc dans les
     sondages (panneau séparé : unités différentes, pas de double axe). Traits pointillés aux
     premiers tours (`markers`), triangles aux ruptures signalées (`breaks`). Valeurs au survol. -->
<script lang="ts">
	import { scaleLinear, scaleUtc } from 'd3-scale';
	import { line } from 'd3-shape';
	import { extent } from 'd3-array';
	import { decimal, parseDate } from '#lib/format.js';

	type Point = { date: string; value: number; note?: string | null };
	type Series = { id: string; label: string; color: string; points: Point[] };
	type BlocTrack = { label: string; color: string; points: { date: string; value: number }[] };

	let {
		series,
		label,
		unit = '%',
		bloc = null,
		markers = [],
		breaks = [],
		start = null,
		end = null,
		height = 150,
		blocHeight = 80
	}: {
		series: Series[];
		label: string;
		unit?: string;
		bloc?: BlocTrack | null;
		markers?: string[];
		breaks?: { date: string; label: string }[];
		start?: string | null;
		end?: string | null;
		height?: number;
		blocHeight?: number;
	} = $props();

	let width = $state(360);
	const m = { top: 12, right: 14, bottom: 22, left: 34 };
	const gap = 28;
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const topH = $derived(height - m.top);
	const botH = $derived(bloc ? blocHeight : 0);
	const totalH = $derived(m.top + topH + (bloc ? gap + botH : 0) + m.bottom);

	const fmtV = (v: number) => (unit === '%' ? `${decimal(v, 0)} %` : decimal(v, 0));
	const all = $derived(series.flatMap((s) => s.points));
	const x0 = $derived(
		start ? parseDate(start) : (extent(all, (p) => parseDate(p.date))[0] ?? new Date())
	);
	const x1 = $derived(
		end ? parseDate(end) : (extent(all, (p) => parseDate(p.date))[1] ?? new Date())
	);
	const x = $derived(scaleUtc().domain([x0, x1]).range([0, innerW]));
	const y = $derived(
		scaleLinear()
			.domain(extent(all, (p) => p.value) as [number, number])
			.nice(4)
			.range([topH, 0])
	);
	const blocPts = $derived(
		bloc ? bloc.points.filter((p) => parseDate(p.date) >= x0 && parseDate(p.date) <= x1) : []
	);
	const yb = $derived(
		scaleLinear()
			.domain(blocPts.length ? (extent(blocPts, (p) => p.value) as [number, number]) : [0, 1])
			.nice(3)
			.range([botH, 0])
	);
	const path = $derived(
		line<Point>()
			.x((p) => x(parseDate(p.date)))
			.y((p) => y(p.value))
	);
	// Part du bloc : segments coupés quand deux mois observés sont séparés de plus de 70 jours.
	const blocSegments = $derived.by(() => {
		const segs: { date: string; value: number }[][] = [];
		let cur: { date: string; value: number }[] = [];
		for (const p of blocPts) {
			const prev = cur.at(-1);
			if (prev && parseDate(p.date).getTime() - parseDate(prev.date).getTime() > 70 * 86_400_000) {
				segs.push(cur);
				cur = [];
			}
			cur.push(p);
		}
		if (cur.length) segs.push(cur);
		return segs;
	});
	const blocPath = $derived(
		line<{ date: string; value: number }>()
			.x((p) => x(parseDate(p.date)))
			.y((p) => yb(p.value))
	);
	const inRange = (d: string) => parseDate(d) >= x0 && parseDate(d) <= x1;
	const yearFmt = new Intl.DateTimeFormat('fr-FR', { year: 'numeric', timeZone: 'UTC' });
	const monthYear = new Intl.DateTimeFormat('fr-FR', {
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC'
	});

	let hover = $state<{ t: number } | null>(null);
	function onMove(e: PointerEvent) {
		const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
		hover = { t: x.invert(e.clientX - rect.left).getTime() };
	}
	const nearest = <P extends { date: string }>(pts: P[], t: number, maxDays: number) => {
		let best: P | null = null;
		let bd = Infinity;
		for (const p of pts) {
			const d = Math.abs(parseDate(p.date).getTime() - t);
			if (d < bd) [best, bd] = [p, d];
		}
		return bd <= maxDays * 86_400_000 ? best : null;
	};
	const hoverInfo = $derived.by(() => {
		if (!hover) return null;
		const anchor = nearest(all, hover.t, 100_000);
		if (!anchor) return null;
		const t = parseDate(anchor.date).getTime();
		const rows = series
			.map((s) => ({ s, p: nearest(s.points, t, 200) }))
			.filter((r): r is { s: Series; p: Point } => r.p !== null);
		const b = bloc ? nearest(blocPts, t, 45) : null;
		return { t, anchor, rows, b };
	});
</script>

<div class="relative w-full" bind:clientWidth={width}>
	{#if all.length < 2}
		<p class="text-sm text-muted">Pas assez de points.</p>
	{:else}
		{#if series.length > 1}
			<ul class="mb-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-ink-2">
				{#each series as s (s.id)}
					<li class="flex items-center gap-1.5">
						<span class="inline-block h-0.5 w-3" style:background={s.color}></span>{s.label}
					</li>
				{/each}
			</ul>
		{/if}
		<svg
			{width}
			height={totalH}
			role="img"
			aria-label={label}
			class="block touch-pan-y select-none"
		>
			<g transform="translate({m.left},{m.top})">
				{#each y.ticks(3) as t (t)}
					<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
					<text x="-5" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[10px]"
						>{decimal(t, 0)}</text
					>
				{/each}
				{#each markers.filter(inRange) as d (d)}
					<line
						x1={x(parseDate(d))}
						x2={x(parseDate(d))}
						y1="0"
						y2={topH + (bloc ? gap + botH : 0)}
						class="stroke-muted"
						stroke-dasharray="2 3"
						opacity="0.7"
					/>
				{/each}
				{#each breaks.filter((b) => inRange(b.date)) as b (b.date + b.label)}
					{@const bx = x(parseDate(b.date))}
					<path d="M{bx - 4},-9 L{bx + 4},-9 L{bx},-2 Z" class="fill-warn-ink">
						<title>Rupture : {b.label}</title>
					</path>
					<line x1={bx} x2={bx} y1="-2" y2={topH} class="stroke-warn-ink" opacity="0.5" />
				{/each}
				{#each series as s (s.id)}
					<path
						d={path(s.points)}
						fill="none"
						style:stroke={s.color}
						stroke-width="1.6"
						stroke-linejoin="round"
					/>
					{#each s.points as p (p.date)}
						<circle
							cx={x(parseDate(p.date))}
							cy={y(p.value)}
							r={p.note ? 2.6 : 1.9}
							style:fill={p.note ? 'var(--color-surface)' : s.color}
							style:stroke={s.color}
							stroke-width={p.note ? 1.2 : 0}
						/>
					{/each}
				{/each}
				{#if bloc}
					<g transform="translate(0,{topH + gap})">
						<text x="0" y="-11" class="fill-ink-2 text-[10px]"
							>{bloc.label} dans les sondages (% des exprimés)</text
						>
						{#each yb.ticks(2) as t (t)}
							<line x1="0" x2={innerW} y1={yb(t)} y2={yb(t)} class="stroke-grid" />
							<text
								x="-5"
								y={yb(t)}
								dy="0.32em"
								text-anchor="end"
								class="tabular fill-muted text-[10px]">{decimal(t, 0)}</text
							>
						{/each}
						{#if blocPts.length === 0}
							<text x={innerW / 2} y={botH / 2} text-anchor="middle" class="fill-muted text-[10px]"
								>pas de sondage sur la période</text
							>
						{/if}
						{#each blocSegments as seg, i (i)}
							<path
								d={blocPath(seg)}
								fill="none"
								style:stroke={bloc.color}
								stroke-width="1.2"
								opacity="0.85"
							/>
						{/each}
					</g>
				{/if}
				<line
					x1="0"
					x2={innerW}
					y1={topH + (bloc ? gap + botH : 0)}
					y2={topH + (bloc ? gap + botH : 0)}
					class="stroke-axis"
				/>
				{#each x.ticks(Math.max(2, Math.floor(innerW / 70))) as t (t.getTime())}
					<text
						x={x(t)}
						y={topH + (bloc ? gap + botH : 0) + 15}
						text-anchor="middle"
						class="fill-muted text-[10px]">{yearFmt.format(t)}</text
					>
				{/each}
				{#if hoverInfo}
					<line
						x1={x(new Date(hoverInfo.t))}
						x2={x(new Date(hoverInfo.t))}
						y1="0"
						y2={topH + (bloc ? gap + botH : 0)}
						class="stroke-ink"
						opacity="0.3"
					/>
					{#each hoverInfo.rows as r (r.s.id)}
						<circle
							cx={x(parseDate(r.p.date))}
							cy={y(r.p.value)}
							r="3.5"
							style:fill={r.s.color}
							class="stroke-surface"
							stroke-width="1.5"
						/>
					{/each}
				{/if}
				<rect
					width={innerW}
					height={topH + (bloc ? gap + botH : 0)}
					fill="transparent"
					role="presentation"
					onpointermove={onMove}
					onpointerdown={onMove}
					onpointerleave={() => (hover = null)}
				/>
			</g>
		</svg>
		{#if hoverInfo}
			<div
				class="pointer-events-none absolute top-6 z-10 max-w-64 rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-sm"
				style:left="{Math.max(0, Math.min(m.left + x(new Date(hoverInfo.t)) + 12, width - 256))}px"
			>
				<p class="mb-1 font-medium">{monthYear.format(new Date(hoverInfo.t))}</p>
				<ul class="space-y-0.5">
					{#each hoverInfo.rows as r (r.s.id)}
						<li>
							<span class="inline-block size-2 rounded-full" style:background={r.s.color}></span>
							{#if series.length > 1}{r.s.label} :{/if}
							<span class="tabular">{fmtV(r.p.value)}</span>
							{#if r.p.date !== hoverInfo.anchor.date}
								<span class="text-muted">({monthYear.format(parseDate(r.p.date))})</span>
							{/if}
							{#if r.p.note}<span class="block text-muted">{r.p.note}</span>{/if}
						</li>
					{/each}
					{#if hoverInfo.b}
						<li class="text-ink-2">
							{bloc?.label} : <span class="tabular">{decimal(hoverInfo.b.value)} %</span>
							<span class="text-muted">({monthYear.format(parseDate(hoverInfo.b.date))})</span>
						</li>
					{/if}
				</ul>
			</div>
		{/if}
	{/if}
</div>
