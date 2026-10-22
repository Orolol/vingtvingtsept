<script lang="ts">
	import * as fmt from '#lib/format.js';
	import PageTitle from '#lib/components/PageTitle.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const bt = $derived(data.backtest);
	let horizon = $state(90);
	const modelLabel = $derived(new Map(bt.models.map((m) => [m.id, m.label])));
	const rows = $derived(
		bt.rows
			.filter((r) => r.horizon_days === horizon)
			.toSorted(
				(a, b) =>
					a.election - b.election ||
					bt.models.findIndex((m) => m.id === a.model_id) -
						bt.models.findIndex((m) => m.id === b.model_id)
			)
	);
	const num = (x: number | null, d = 2) => (x === null ? '—' : fmt.decimal(x, d));
	const cov = (x: number | null) => (x === null ? '—' : fmt.prob(x));
	const waves = $derived(
		[...new Set(bt.variables_tested.map((v) => v.wave))]
			.sort()
			.map((w) => ({ wave: w, items: bt.variables_tested.filter((v) => v.wave === w) }))
	);
</script>

<PageTitle title="Méthode">
	<p>
		Comment la prévision est construite, comment on vérifie qu'elle tient, et ce qu'elle ne sait pas
		faire. Le détail technique est dans la documentation du dépôt (docs/04-modele.md,
		docs/07-evaluation.md).
	</p>
</PageTitle>

<div class="max-w-3xl space-y-8 leading-relaxed">
	<section>
		<h2 class="mb-2 text-lg font-semibold">1. Les sondages, agrégés</h2>
		<p>
			Chaque candidat a un soutien « latent » qui évolue semaine après semaine. Les sondages en sont
			des mesures bruitées : chaque institut a ses biais propres (effets maison), qu'on estime en
			comparant les instituts entre eux, et un échantillon par quotas vaut moins qu'un tirage
			aléatoire de même taille. Un candidat rarement testé emprunte la dynamique de son bloc
			politique, mesurée par tous les sondages.
		</p>
	</section>
	<section>
		<h2 class="mb-2 text-lg font-semibold">2. L'erreur, mesurée sur le passé</h2>
		<p>
			L'incertitude ne vient pas d'une marge d'erreur théorique mais des erreurs réellement
			observées aux présidentielles de 2002 à 2022 : combien l'opinion a bougé entre une date donnée
			et le vote, et de combien les derniers sondages se sont trompés, bloc par bloc. Ces erreurs
			sont corrélées : quand la gauche est sous-estimée, tous ses candidats le sont ensemble. Plus
			le vote est loin, plus l'intervalle est large.
		</p>
	</section>
	<section>
		<h2 class="mb-2 text-lg font-semibold">3. Qui sera candidat ?</h2>
		<p>
			Les candidatures ne sont connues qu'en mars. On décrit les incertitudes (qui porte le RN,
			combien de candidats au centre, la gauche unie ou non…) avec des probabilités jugées, datées
			et sourcées, publiées sur la page <em>Scénarios</em> avec leur historique. La prévision mélange
			les configurations selon ces probabilités, et la prévision conditionnelle à chacune est publiée
			à côté.
		</p>
	</section>
	<section>
		<h2 class="mb-2 text-lg font-semibold">4. Le second tour</h2>
		<p>
			Pour chaque duel, le score vient des reports de voix de chaque électorat de premier tour,
			estimés sur les enquêtes post-électorales des seconds tours passés et corrigés par les
			sondages de second tour hypothétiques, dont l'erreur loin du vote est grande.
		</p>
	</section>
	<section>
		<h2 class="mb-2 text-lg font-semibold">5. La simulation</h2>
		<p>
			On simule l'élection {fmt.integer(data.manifest.n_draws)} fois : configuration de candidats, parts
			de premier tour, qualifiés, reports, vainqueur. Les probabilités affichées sont les fréquences de
			ces simulations ; les scores sont donnés par leur médiane et leurs intervalles à 50 et 90 %.
		</p>
	</section>
	<section>
		<h2 class="mb-2 text-lg font-semibold">Économie et presse</h2>
		<p>
			Une variable non sondagière n'entre dans la prévision que si elle améliore le score du
			backtest à 180 et 90 jours, sur au moins deux des trois élections de référence, sans dégrader
			la couverture des intervalles. Sinon elle reste dans le volet explicatif, et l'échec est
			documenté.
		</p>
	</section>
</div>

<section aria-labelledby="titre-backtest" class="mt-10">
	<h2 id="titre-backtest" class="mb-1 text-lg font-semibold">Backtest</h2>
	<p class="mb-3 max-w-3xl text-sm text-ink-2">
		Prévisions refaites « comme à l'époque » pour {bt.elections.join(', ')}, à plusieurs horizons,
		avec un modèle d'erreur calibré sans l'élection évaluée. Log-score : plus haut est meilleur ;
		Brier, CRPS et erreur absolue : plus bas est meilleur ; couverture : doit approcher 50 % et 90
		%. Version {bt.model_version}, calculé le
		{fmt.date(bt.generated_at)}.
	</p>
	<div
		role="radiogroup"
		aria-label="Horizon"
		class="mb-3 inline-flex rounded-md border border-line text-sm"
	>
		{#each bt.horizons_days as h (h)}
			<button
				type="button"
				role="radio"
				aria-checked={horizon === h}
				class={['px-3 py-1', horizon === h ? 'bg-ink text-page' : 'text-ink-2']}
				onclick={() => (horizon = h)}>{h} j</button
			>
		{/each}
	</div>
	<div class="overflow-x-auto rounded-md border border-line bg-surface">
		<table class="w-full text-sm">
			<caption class="sr-only">Métriques du backtest à {horizon} jours du premier tour</caption>
			<thead class="border-b border-line text-left text-xs text-muted">
				<tr>
					<th scope="col" class="px-3 py-2 font-medium">Élection</th>
					<th scope="col" class="px-3 py-2 font-medium">Modèle</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Log-score paire</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Log-score vainqueur</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Brier qualif.</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">CRPS (pts)</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Erreur abs. (pts)</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Couv. 50 %</th>
					<th scope="col" class="px-3 py-2 text-right font-medium">Couv. 90 %</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as r (r.model_id + r.election)}
					<tr class="border-b border-line last:border-0">
						<td class="tabular px-3 py-1.5">{r.election}</td>
						<td class="px-3 py-1.5 whitespace-nowrap">{modelLabel.get(r.model_id) ?? r.model_id}</td
						>
						<td class="tabular px-3 py-1.5 text-right">{num(r.log_score_pair)}</td>
						<td class="tabular px-3 py-1.5 text-right">{num(r.log_score_winner)}</td>
						<td class="tabular px-3 py-1.5 text-right">{num(r.brier_qualify, 3)}</td>
						<td class="tabular px-3 py-1.5 text-right">{num(r.crps_first_round)}</td>
						<td class="tabular px-3 py-1.5 text-right">{num(r.mae_first_round)}</td>
						<td class="tabular px-3 py-1.5 text-right">{cov(r.coverage_50)}</td>
						<td class="tabular px-3 py-1.5 text-right">{cov(r.coverage_90)}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	<ul class="mt-3 space-y-1 text-xs text-muted">
		{#each bt.models as m (m.id)}
			<li><strong class="text-ink-2">{m.label}</strong> : {m.description}</li>
		{/each}
	</ul>
</section>

<section aria-labelledby="titre-variables" class="mt-10 max-w-3xl leading-relaxed">
	<h2 id="titre-variables" class="mb-1 text-lg font-semibold">Variables testées et non admises</h2>
	<p class="mb-3 text-sm text-ink-2">
		Chaque piste a d'abord été étudiée, puis, si l'étude le justifiait, essayée au backtest contre
		le modèle de référence. Aucune n'a passé la règle d'admission (une attend encore ses données) ;
		elles restent dans le code, désactivées, et certaines sont montrées dans les volets explicatifs
		du site. Détail : {bt.variables_tested_reference} dans le dépôt.
	</p>
	{#each waves as w (w.wave)}
		<h3 class="mt-4 mb-2 text-sm font-medium text-ink-2">Vague du {fmt.date(w.wave)}</h3>
		<ul class="divide-y divide-line rounded-md border border-line bg-surface text-sm">
			{#each w.items as v (v.id)}
				<li class="px-3 py-2.5">
					<div class="flex flex-wrap items-baseline gap-x-2 gap-y-1">
						<span class="font-medium">{v.label}</span>
						<span class="rounded-full border border-line px-2 text-xs text-muted">{v.verdict}</span>
						{#if v.published}
							<span class="rounded-full bg-warn-bg px-2 text-xs text-warn-ink"
								>utilisée par la projection publiée</span
							>
						{/if}
					</div>
					<p class="mt-0.5 text-ink-2">{v.summary}</p>
				</li>
			{/each}
		</ul>
	{/each}
</section>

<section class="mt-10 max-w-3xl leading-relaxed">
	<h2 class="mb-2 text-lg font-semibold">Limites</h2>
	<ul class="list-disc space-y-1 pl-5">
		<li>
			Cinq présidentielles seulement pour calibrer l'erreur : les queues de distribution sont mal
			connues.
		</li>
		<li>
			Les probabilités de scénarios sont des jugements, pas des mesures ; une décision de justice ou
			un retrait peut les faire bouger d'un coup.
		</li>
		<li>
			Six mois avant le vote, le modèle mesure surtout la taille des blocs : l'ordre de candidats
			proches n'est pas prévisible.
		</li>
		<li>Les sondages publiés n'ont pas de détail géographique : pas de carte par département.</li>
		<li>
			Les reports de second tour s'appuient sur peu de précédents pour certains duels (sans candidat
			RN, RN contre gauche radicale) : leurs intervalles sont larges à dessein.
		</li>
	</ul>
</section>
