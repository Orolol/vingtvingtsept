<!-- Intervalle à 90 % (trait fin), à 50 % (trait épais) et médiane (repère) sur une échelle
     horizontale commune. Toujours accompagné de son texte (médiane et bornes). -->
<script lang="ts">
	import type { Interval } from '#lib/types/data.js';
	import { interval as fmtInterval } from '#lib/format.js';

	let {
		q,
		max,
		color,
		showText = true,
		width = 120
	}: { q: Interval; max: number; color: string; showText?: boolean; width?: number } = $props();

	const H = 14;
	const x = (v: number) => Math.max(0, Math.min(width, (v / max) * width));
</script>

<span class="inline-flex items-center gap-2">
	<svg
		{width}
		height={H}
		viewBox="0 0 {width} {H}"
		aria-hidden="true"
		class="shrink-0 overflow-visible"
	>
		<line x1="0" x2={width} y1={H / 2} y2={H / 2} class="stroke-grid" stroke-width="1" />
		<line
			x1={x(q.q05)}
			x2={x(q.q95)}
			y1={H / 2}
			y2={H / 2}
			style:stroke={color}
			stroke-width="2"
			stroke-linecap="round"
			opacity="0.45"
		/>
		<line
			x1={x(q.q25)}
			x2={x(q.q75)}
			y1={H / 2}
			y2={H / 2}
			style:stroke={color}
			stroke-width="6"
			stroke-linecap="round"
			opacity="0.75"
		/>
		<line
			x1={x(q.q50)}
			x2={x(q.q50)}
			y1="1"
			y2={H - 1}
			class="stroke-ink"
			stroke-width="2"
			stroke-linecap="round"
		/>
	</svg>
	{#if showText}
		<span class="tabular text-sm whitespace-nowrap">{fmtInterval(q)}</span>
	{/if}
</span>
