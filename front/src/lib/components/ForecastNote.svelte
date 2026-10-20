<!-- Phrase obligatoire sur toute page qui montre une source non sondagière (docs/08-front.md). -->
<script lang="ts">
	let { inForecast, note }: { inForecast: boolean; note: string } = $props();
</script>

<p
	class="mb-6 max-w-3xl rounded-md border border-line bg-surface px-4 py-3 text-sm"
	data-testid="forecast-note"
>
	<strong>{inForecast ? 'Entre dans la prévision.' : "N'entre pas dans la prévision."}</strong>
	{note}
</p>
