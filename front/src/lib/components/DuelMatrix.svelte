<!-- Matrice des duels : la case (ligne, colonne) donne la probabilité que ce duel ait lieu
     (couleur) et le score du candidat en ligne, médiane et intervalle à 90 %. -->
<script lang="ts">
	import type { Duel } from '#lib/types/data.js';
	import type { CandidateRegistry } from '#lib/candidates.js';
	import { indexDuels, orient, pairKey } from '#lib/duels.js';
	import { decimal, prob } from '#lib/format.js';

	let {
		pairs,
		ids,
		candidates,
		selected = $bindable(null)
	}: {
		pairs: Duel[];
		ids: string[];
		candidates: CandidateRegistry;
		selected?: string | null;
	} = $props();

	const index = $derived(indexDuels(pairs));
	const maxP = $derived(Math.max(0.01, ...pairs.map((p) => p.p_pair)));
	// Intensité : racine pour que les paires rares restent visibles.
	const mix = (p: number) => Math.round(8 + 77 * Math.sqrt(p / maxP));
</script>

<div class="overflow-x-auto">
	<table class="border-separate border-spacing-0.5 text-xs">
		<caption class="sr-only">
			Probabilité de chaque duel de second tour et score du candidat en ligne
		</caption>
		<thead>
			<tr>
				<th scope="col" class="p-1 text-left font-normal text-muted">contre →</th>
				{#each ids as col (col)}
					<th scope="col" class="min-w-20 p-1 text-center font-medium">{candidates.short(col)}</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each ids as row (row)}
				<tr>
					<th scope="row" class="p-1 pr-2 text-left font-medium whitespace-nowrap">
						<span
							class="mr-1 inline-block size-2 rounded-full"
							style:background={candidates.color(row)}
						></span>
						{candidates.short(row)}
					</th>
					{#each ids as col (col)}
						{@const d = row === col ? undefined : index.get(pairKey(row, col))}
						{#if row === col}
							<td class="rounded bg-grid/40 p-1"></td>
						{:else if !d}
							<td class="rounded border border-line p-1 text-center text-muted">—</td>
						{:else}
							{@const o = orient(d, row)}
							{@const pct = mix(d.p_pair)}
							{@const isSel = selected === pairKey(row, col)}
							<td class="p-0">
								<button
									type="button"
									class={[
										'w-full rounded px-1.5 py-1 text-center leading-tight',
										isSel && 'ring-2 ring-ink',
										pct >= 55 ? 'text-white' : 'text-ink'
									]}
									style:background="color-mix(in oklab, var(--color-accent) {pct}%,
									var(--color-surface))"
									aria-pressed={isSel}
									onclick={() => (selected = pairKey(row, col))}
									title="{candidates.name(row)} contre {candidates.name(col)}"
								>
									<span class="tabular block font-semibold">{prob(d.p_pair)}</span>
									<span class="tabular block opacity-90">
										{decimal(o.share_first.q50)}
										<span class="whitespace-nowrap"
											>({decimal(o.share_first.q05, 0)}–{decimal(o.share_first.q95, 0)})</span
										>
									</span>
								</button>
							</td>
						{/if}
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>
<p class="mt-1 text-xs text-muted">
	Haut de case : probabilité que le duel ait lieu (couleur d'autant plus soutenue). Bas : score du
	candidat en ligne si le duel a lieu, médiane et intervalle à 90 %, en % des exprimés.
</p>
