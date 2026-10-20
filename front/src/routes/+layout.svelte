<script lang="ts">
	import './layout.css';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import * as fmt from '#lib/format.js';
	import type { LayoutProps } from './$types';

	let { data, children }: LayoutProps = $props();
	const manifest = $derived(data.manifest);

	const nav = [
		{ route: '/', label: 'Accueil' },
		{ route: '/premier-tour', label: 'Premier tour' },
		{ route: '/second-tour', label: 'Second tour' },
		{ route: '/scenarios', label: 'Scénarios' },
		{ route: '/sondages', label: 'Sondages' },
		{ route: '/economie', label: 'Économie' },
		{ route: '/presse', label: 'Presse' },
		{ route: '/methode', label: 'Méthode' }
	] as const;

	const daysToVote = $derived(fmt.daysBetween(manifest.asof, manifest.election.first_round));
</script>

<svelte:head>
	<title>Présidentielle 2027 · vingtvingtsept</title>
	<meta
		name="description"
		content="Prévision probabiliste de l'élection présidentielle française de 2027, mise à jour à chaque sondage."
	/>
</svelte:head>

<a
	href="#contenu"
	class="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-surface focus:px-3 focus:py-2"
	>Aller au contenu</a
>

<header class="border-b border-line bg-surface">
	<div class="mx-auto max-w-6xl px-4 pt-4">
		<div class="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
			<a href={resolve('/')} class="text-lg font-semibold tracking-tight text-ink no-underline">
				vingtvingtsept <span class="font-normal text-ink-2">· présidentielle 2027</span>
			</a>
			<p class="text-sm text-ink-2" data-testid="last-poll">
				Dernier sondage intégré : <strong class="text-ink"
					>{fmt.date(manifest.last_poll_date)}</strong
				>
				<span class="text-muted">· {fmt.integer(manifest.n_polls)} sondages · J−{daysToVote}</span>
			</p>
		</div>
		<nav aria-label="Sections" class="-mx-4 mt-2 overflow-x-auto px-4">
			<ul class="flex gap-1 text-sm whitespace-nowrap">
				{#each nav as item (item.route)}
					{@const active = page.route.id === item.route}
					<li>
						<a
							href={resolve(item.route)}
							aria-current={active ? 'page' : undefined}
							class={[
								'inline-block border-b-2 px-2 py-2 no-underline',
								active
									? 'border-accent font-medium text-ink'
									: 'border-transparent text-ink-2 hover:text-ink'
							]}>{item.label}</a
						>
					</li>
				{/each}
			</ul>
		</nav>
	</div>
</header>

{#if manifest.mock}
	<div role="note" class="bg-warn-bg text-warn-ink">
		<p class="mx-auto max-w-6xl px-4 py-2 text-sm">
			<strong>Données factices.</strong> Ce site est un squelette : les chiffres affichés sont simulés
			pour la mise au point et ne constituent pas une prévision.
		</p>
	</div>
{/if}

<main id="contenu" class="mx-auto max-w-6xl px-4 py-6 sm:py-8">
	{@render children()}
</main>

<footer class="border-t border-line bg-surface">
	<div class="mx-auto flex max-w-6xl flex-wrap gap-x-6 gap-y-1 px-4 py-4 text-xs text-muted">
		<span>Run du {fmt.date(manifest.run_date)} (prévision au {fmt.date(manifest.asof)})</span>
		<span>Modèle {manifest.model_version}</span>
		<span>Commit {manifest.commit ? manifest.commit.slice(0, 7) : '—'}</span>
		<span>{fmt.integer(manifest.n_draws)} tirages</span>
		<span>Contrat de données v{manifest.schema_version}</span>
	</div>
</footer>
