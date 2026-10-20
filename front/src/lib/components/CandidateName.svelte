<script lang="ts">
	import { candidateColor } from '#lib/candidates.js';

	let { id, name, detail }: { id: string; name: string; detail?: string | null } = $props();
</script>

<span class="inline-flex items-center gap-2">
	<span
		aria-hidden="true"
		class="inline-block size-2.5 shrink-0 rounded-full"
		style:background={candidateColor(id)}
	></span>
	<span>{name}</span>
	{#if detail}<span class="text-xs text-muted">{detail}</span>{/if}
</span>
