<script lang="ts">
	import { prob } from '#lib/format.js';

	let { p, color, width = 72 }: { p: number; color: string; width?: number } = $props();
</script>

<span class="inline-flex items-center gap-2">
	<span
		aria-hidden="true"
		class="relative hidden h-2 shrink-0 overflow-hidden rounded-full bg-grid sm:inline-block"
		style:width="{width}px"
	>
		<span
			class="absolute inset-y-0 left-0 rounded-full"
			style:width="{p * 100}%"
			style:background={color}
		></span>
	</span>
	<span class="tabular w-12 text-right">{prob(p)}</span>
</span>
