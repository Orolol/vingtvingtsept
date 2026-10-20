<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
</script>

<h1 class="text-2xl font-semibold">
	{page.status === 404 ? 'Page introuvable' : 'Erreur'}
</h1>
<p class="mt-2 text-ink-2">{page.error?.message}</p>
<p class="mt-4"><a href={resolve('/')} class="text-accent">Retour à l'accueil</a></p>
