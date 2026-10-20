<script lang="ts">
	import type { Snippet } from 'svelte';

	let { title, children }: { title: string; children?: Snippet } = $props();
</script>

<svelte:head>
	<title>{title} · vingtvingtsept</title>
</svelte:head>

<div class="mb-6 max-w-3xl">
	<h1 class="text-2xl font-semibold tracking-tight sm:text-3xl">{title}</h1>
	{#if children}
		<div class="mt-2 text-ink-2">{@render children()}</div>
	{/if}
</div>
