<!-- Choix d'une condition (mélange, état d'une dimension, configuration complète). -->
<script lang="ts">
	import type { Condition } from '#lib/types/data.js';
	import { prob } from '#lib/format.js';

	let {
		conditions,
		value = $bindable(),
		id = 'condition'
	}: { conditions: Condition[]; value: string; id?: string } = $props();

	const groups = $derived(
		(
			[
				['mixture', 'Mélange'],
				['dimension_state', 'Une hypothèse, le reste mélangé'],
				['configuration', 'Configurations complètes']
			] as const
		)
			.map(([kind, label]) => ({ kind, label, items: conditions.filter((c) => c.kind === kind) }))
			.filter((g) => g.items.length)
	);
</script>

<label for={id} class="block text-sm text-ink-2">Scénario</label>
<select
	{id}
	bind:value
	class="mt-1 w-full max-w-full rounded-md border border-line bg-surface px-2 py-1.5 text-sm sm:w-auto"
>
	{#each groups as g (g.kind)}
		<optgroup label={g.label}>
			{#each g.items as c (c.id)}
				<option value={c.id}
					>{c.label}{c.kind === 'mixture' ? '' : ` (${prob(c.probability)})`}</option
				>
			{/each}
		</optgroup>
	{/each}
</select>
