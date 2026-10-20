<!-- Soutien latent de premier tour : intervalle à 90 % (bande claire), à 50 % (bande foncée),
     médiane (trait), sondages (points), et projection au jour du vote (barre à droite). -->
<script lang="ts" module>
	export interface PollPoint {
		date: string;
		candidate_id: string;
		share: number;
		pollster: string;
	}
</script>

<script lang="ts">
	import { scaleLinear, scaleUtc } from 'd3-scale';
	import { area, curveMonotoneX, line } from 'd3-shape';
	import type { Interval, TrajectorySeries } from '#lib/types/data.js';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { dateShortFmt, interval as fmtInterval, parseDate } from '#lib/format.js';

	let {
		dates,
		series,
		polls,
		asof,
		electionDate,
		candidates,
		height = 380
	}: {
		dates: string[];
		series: TrajectorySeries[];
		polls: PollPoint[];
		asof: string;
		electionDate: string;
		candidates: CandidateRegistry;
		height?: number;
	} = $props();

	let width = $state(720);
	const narrow = $derived(width < 520);
	const m = $derived({ top: 12, right: narrow ? 82 : 100, bottom: 28, left: 34 });
	const innerW = $derived(Math.max(10, width - m.left - m.right));
	const innerH = $derived(height - m.top - m.bottom);

	const times = $derived(dates.map((d) => parseDate(d)));
	const x = $derived(
		scaleUtc()
			.domain([times[0], parseDate(electionDate)])
			.range([0, innerW])
	);
	const yMax = $derived(
		Math.max(
			10,
			...series.flatMap((s) => [
				...s.q95.filter((v): v is number => v !== null),
				s.election_day.q95
			]),
			...polls.map((p) => p.share)
		)
	);
	const y = $derived(
		scaleLinear()
			.domain([0, Math.ceil(yMax / 5) * 5])
			.range([innerH, 0])
	);

	type Row = { t: Date; v: (number | null)[] };
	const rowsOf = (s: TrajectorySeries): Row[] =>
		times.map((t, i) => ({ t, v: [s.q05[i], s.q25[i], s.q50[i], s.q75[i], s.q95[i]] }));
	const band = (lo: number, hi: number) =>
		area<Row>()
			.defined((r) => r.v[lo] !== null)
			.x((r) => x(r.t))
			.y0((r) => y(r.v[lo] as number))
			.y1((r) => y(r.v[hi] as number))
			.curve(curveMonotoneX);
	const median = $derived(
		line<Row>()
			.defined((r) => r.v[2] !== null)
			.x((r) => x(r.t))
			.y((r) => y(r.v[2] as number))
			.curve(curveMonotoneX)
	);
	const band90 = $derived(band(0, 4));
	const band50 = $derived(band(1, 3));

	const xElection = $derived(x(parseDate(electionDate)));
	const xAsof = $derived(x(parseDate(asof)));
	const yTicks = $derived(y.ticks(5));
	const xTicks = $derived(x.ticks(Math.max(2, Math.floor(innerW / 90))));
	const monthFmt = new Intl.DateTimeFormat('fr-FR', { month: 'short', timeZone: 'UTC' });
	const tickLabel = (d: Date, i: number) =>
		d.getUTCMonth() === 0 || i === 0
			? `${monthFmt.format(d)} ${d.getUTCFullYear()}`
			: monthFmt.format(d);

	// Étiquettes directes à droite (médiane du jour du vote), écartées pour ne pas se chevaucher.
	const labels = $derived.by(() => {
		const items = series
			.map((s) => ({ id: s.candidate_id, y: y(s.election_day.q50) }))
			.sort((a, b) => a.y - b.y);
		const gap = 13;
		for (let i = 1; i < items.length; i++)
			if (items[i].y - items[i - 1].y < gap) items[i].y = items[i - 1].y + gap;
		const overflow = items.length ? items[items.length - 1].y - innerH : 0;
		if (overflow > 0) for (const it of items) it.y -= overflow;
		return items;
	});
	const offset = (i: number) => (i - (series.length - 1) / 2) * 4;

	// Survol : semaine la plus proche, ou le jour du vote à droite de la dernière semaine.
	let hover = $state<number | null>(null);
	function onMove(e: PointerEvent) {
		const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
		const px = e.clientX - rect.left;
		const lastX = x(times[times.length - 1]);
		if (px > (lastX + xElection) / 2) {
			hover = -1;
			return;
		}
		const t = x.invert(px).getTime();
		let best = 0;
		for (let i = 1; i < times.length; i++)
			if (Math.abs(times[i].getTime() - t) < Math.abs(times[best].getTime() - t)) best = i;
		hover = best;
	}
	const hoverRows = $derived.by(() => {
		if (hover === null) return [];
		const h = hover;
		return series
			.map((s) => {
				const q: Interval | null =
					h === -1
						? s.election_day
						: s.q50[h] === null
							? null
							: {
									q05: s.q05[h] as number,
									q25: s.q25[h] as number,
									q50: s.q50[h] as number,
									q75: s.q75[h] as number,
									q95: s.q95[h] as number
								};
				return { id: s.candidate_id, q };
			})
			.filter((r): r is { id: string; q: Interval } => r.q !== null)
			.sort((a, b) => b.q.q50 - a.q.q50);
	});
	const hoverX = $derived(hover === null ? 0 : hover === -1 ? xElection : x(times[hover]));
	const hoverTitle = $derived(
		hover === null
			? ''
			: hover === -1
				? `Jour du vote (${dateShortFmt(electionDate)}), projection`
				: `Semaine du ${dateShortFmt(dates[hover])}`
	);
</script>

<div class="relative w-full" bind:clientWidth={width}>
	<svg
		{width}
		{height}
		role="img"
		aria-label="Soutien de premier tour par candidat, avec intervalles à 50 et 90 %, sondages et projection au jour du vote"
		class="block touch-pan-y select-none"
	>
		<g transform="translate({m.left},{m.top})">
			<!-- futur : de la date de prévision au vote -->
			<rect
				x={xAsof}
				y="0"
				width={Math.max(0, xElection - xAsof)}
				height={innerH}
				class="fill-page"
			/>
			{#each yTicks as t (t)}
				<line x1="0" x2={innerW} y1={y(t)} y2={y(t)} class="stroke-grid" />
				<text x="-6" y={y(t)} dy="0.32em" text-anchor="end" class="tabular fill-muted text-[11px]"
					>{t}</text
				>
			{/each}
			<line x1="0" x2={innerW} y1={innerH} y2={innerH} class="stroke-axis" />
			{#each xTicks as t, i (t.getTime())}
				<text x={x(t)} y={innerH + 18} text-anchor="middle" class="fill-muted text-[11px]"
					>{tickLabel(t, i)}</text
				>
			{/each}
			<line x1={xAsof} x2={xAsof} y1="0" y2={innerH} class="stroke-axis" stroke-dasharray="3 3" />
			<line x1={xElection} x2={xElection} y1="0" y2={innerH} class="stroke-axis" />
			<text x={xElection} y="-2" text-anchor="end" class="fill-muted text-[11px]">1er tour</text>

			{#each series as s (s.candidate_id)}
				{@const rows = rowsOf(s)}
				{@const color = candidates.color(s.candidate_id)}
				<path d={band90(rows)} style:fill={color} opacity="0.12" />
				<path d={band50(rows)} style:fill={color} opacity="0.22" />
				<path d={median(rows)} fill="none" style:stroke={color} stroke-width="2" />
			{/each}

			{#each polls as p, i (i)}
				<circle
					cx={x(parseDate(p.date))}
					cy={y(p.share)}
					r="2.5"
					style:fill={candidates.color(p.candidate_id)}
					class="stroke-surface"
					stroke-width="1"
					opacity="0.6"
				/>
			{/each}

			{#each series as s, i (s.candidate_id)}
				{@const q = s.election_day}
				{@const color = candidates.color(s.candidate_id)}
				{@const cx = xElection + offset(i)}
				<line
					x1={cx}
					x2={cx}
					y1={y(q.q05)}
					y2={y(q.q95)}
					style:stroke={color}
					stroke-width="1.5"
					opacity="0.6"
				/>
				<line
					x1={cx}
					x2={cx}
					y1={y(q.q25)}
					y2={y(q.q75)}
					style:stroke={color}
					stroke-width="4"
					stroke-linecap="round"
				/>
				<circle
					{cx}
					cy={y(q.q50)}
					r="3"
					style:fill={color}
					class="stroke-surface"
					stroke-width="1.5"
				/>
			{/each}

			{#each labels as l (l.id)}
				<text x={innerW + 10} y={l.y} dy="0.32em" class="fill-ink text-[11px]">
					<tspan style:fill={candidates.color(l.id)}>●</tspan>
					{candidates.short(l.id)}
				</text>
			{/each}

			{#if hover !== null}
				<line x1={hoverX} x2={hoverX} y1="0" y2={innerH} class="stroke-ink" opacity="0.35" />
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

	{#if hover !== null && hoverRows.length}
		<div
			class="pointer-events-none absolute top-2 z-10 max-w-64 rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-sm"
			style:left="{Math.min(m.left + hoverX + 12, width - 250)}px"
		>
			<p class="mb-1 font-medium">{hoverTitle}</p>
			<ul class="space-y-0.5">
				{#each hoverRows as r (r.id)}
					<li class="flex items-center gap-2 whitespace-nowrap">
						<span class="inline-block size-2 rounded-full" style:background={candidates.color(r.id)}
						></span>
						<span class="min-w-0 flex-1 truncate">{candidates.short(r.id)}</span>
						<span class="tabular text-ink-2">{fmtInterval(r.q)}</span>
					</li>
				{/each}
			</ul>
		</div>
	{/if}
</div>
<p class="mt-1 text-xs text-muted">
	Bandes : intervalles à 50 % (foncé) et 90 % (clair) autour de la médiane. Points : sondages
	publiés. Barres à droite : part projetée au {dateShortFmt(electionDate)}, médiane et intervalles.
	Zone grisée : de la date de prévision au vote. Axe vertical : % des suffrages exprimés.
</p>
