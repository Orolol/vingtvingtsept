#!/usr/bin/env bash
# Archive durable des runs à conserver (gel, ajustements et simulations 2027), hors du dépôt.
# Voir docs/13-validation-prospective.md § « Archive durable du gel ».
#
# Usage :
#   scripts/archive_runs.sh <destination> [dossier de runs/ …]
#   scripts/archive_runs.sh --verify <instantané> [dossier de restauration]
#
# Création : écrit <destination>/<AAAA-MM-JJTHHMMSS>/ avec, par dossier de runs/, une archive
# <nom>.tar.zst (tar.gz si zstd manque) et la liste <nom>.files.sha256 des empreintes de
# chaque fichier ; plus index.tsv (tailles, nombre de fichiers, compression), meta.txt
# (commit, verrou de la référence) et SHA256SUMS sur l'ensemble. Sans liste, archive
# runs/2026-10-22-consolidation, runs/*-fit-2027*, runs/*-simulate* et runs/prospective.
# Jamais d'écrasement : un instantané existant fait échouer le script. Refuse de s'exécuter
# sans destination ou avec une destination dans le dépôt ; crée la destination au besoin.
# Aucune copie réseau : pour un support externe, copier ensuite l'instantané (rsync).
#
# Contrôle (--verify) : sha256sum -c SHA256SUMS, extraction de chaque archive dans le dossier
# de restauration (temporaire et supprimé ensuite par défaut), comparaison fichier par
# fichier avec les empreintes enregistrées, et signalement des runs/ locaux modifiés depuis.
#
# Variables : VVS_RUNS_DIR (défaut <dépôt>/runs) ; VVS_ARCHIVE_COMPRESSION=gzip force gzip.
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() {
	echo "archive_runs : $*" >&2
	exit 1
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${VVS_RUNS_DIR:-$ROOT/runs}"

if [ $# -eq 0 ]; then
	usage >&2
	exit 2
fi
case "$1" in
-h | --help)
	usage
	exit 0
	;;
esac

# --- contrôle et restauration
if [ "$1" = "--verify" ]; then
	[ $# -ge 2 ] || die "--verify attend le dossier d'un instantané"
	SNAP="$(cd "$2" && pwd)"
	[ -f "$SNAP/SHA256SUMS" ] && [ -f "$SNAP/index.tsv" ] || die "pas un instantané : $SNAP"
	if [ $# -ge 3 ]; then
		RESTORE="$3"
		mkdir -p "$RESTORE"
		[ -z "$(ls -A "$RESTORE")" ] || die "dossier de restauration non vide : $RESTORE"
		RESTORE="$(cd "$RESTORE" && pwd)"
		KEEP=1
	else
		RESTORE="$(mktemp -d)"
		KEEP=0
		trap 'rm -rf "$RESTORE"' EXIT
	fi
	(cd "$SNAP" && sha256sum -c --quiet SHA256SUMS) || die "SHA256SUMS : archive altérée"
	echo "SHA256SUMS : $(wc -l <"$SNAP/SHA256SUMS") fichiers conformes"
	FAIL=0
	tail -n +2 "$SNAP/index.tsv" | while IFS=$'\t' read -r name archive compression n_files _rest; do
		case "$compression" in
		zstd) zstd -dcq "$SNAP/$archive" | tar -xf - -C "$RESTORE" ;;
		gzip) gzip -dc "$SNAP/$archive" | tar -xf - -C "$RESTORE" ;;
		*) die "compression inconnue : $compression" ;;
		esac
		got="$(find "$RESTORE/$name" -type f | wc -l)"
		[ "$got" -eq "$n_files" ] || die "$name : $got fichiers restaurés, $n_files attendus"
		(cd "$RESTORE" && sha256sum -c --quiet "$SNAP/$name.files.sha256") ||
			die "$name : empreintes restaurées différentes"
		echo "restauré : $name ($n_files fichiers, empreintes conformes)"
		if [ -d "$RUNS/$name" ]; then
			if (cd "$RUNS" && sha256sum -c --quiet "$SNAP/$name.files.sha256" >/dev/null 2>&1) &&
				[ "$(find "$RUNS/$name" -type f | wc -l)" -eq "$n_files" ]; then
				echo "  runs/$name : identique à l'instantané"
			else
				echo "  runs/$name : modifié depuis l'instantané (signalé, pas une erreur)"
			fi
		fi
	done || FAIL=1
	[ "$FAIL" = 0 ] || exit 1
	if [ "$KEEP" = 1 ]; then echo "restauration conservée dans $RESTORE"; fi
	echo "contrôle de restauration réussi : $SNAP"
	exit 0
fi

# --- création
DEST="$1"
shift
case "$DEST" in
-*) die "option inconnue : $DEST" ;;
esac
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
case "$DEST/" in
"$ROOT"/*) die "destination dans le dépôt refusée : $DEST" ;;
esac
[ -d "$RUNS" ] || die "dossier de runs absent : $RUNS"

if [ $# -gt 0 ]; then
	NAMES=("$@")
else
	NAMES=()
	for d in "$RUNS"/2026-10-22-consolidation "$RUNS"/*-fit-2027* "$RUNS"/*-simulate* "$RUNS"/prospective; do
		if [ -d "$d" ]; then NAMES+=("$(basename "$d")"); fi
	done
fi
[ ${#NAMES[@]} -gt 0 ] || die "aucun dossier à archiver dans $RUNS"
for name in "${NAMES[@]}"; do
	case "$name" in
	*/* | . | .. | "") die "nom de dossier invalide : $name (un dossier direct de runs/)" ;;
	esac
	[ -d "$RUNS/$name" ] || die "dossier absent : $RUNS/$name"
done

if [ "${VVS_ARCHIVE_COMPRESSION:-}" != gzip ] && command -v zstd >/dev/null 2>&1; then
	COMPRESSION=zstd EXT=tar.zst
else
	COMPRESSION=gzip EXT=tar.gz
fi

STAMP="$(date +%Y-%m-%dT%H%M%S)"
SNAP="$DEST/$STAMP"
[ ! -e "$SNAP" ] || die "instantané déjà présent : $SNAP"
PARTIAL="$SNAP.partial"
[ ! -e "$PARTIAL" ] || die "instantané incomplet à examiner puis supprimer : $PARTIAL"
mkdir "$PARTIAL"

printf 'name\tarchive\tcompression\tn_files\tsource_bytes\tarchive_bytes\tarchive_sha256\n' \
	>"$PARTIAL/index.tsv"
for name in "${NAMES[@]}"; do
	(cd "$RUNS" && find "$name" -type f -print0 | sort -z | xargs -0 sha256sum) \
		>"$PARTIAL/$name.files.sha256"
	n_files="$(wc -l <"$PARTIAL/$name.files.sha256")"
	src_bytes="$(du -sb "$RUNS/$name" | cut -f1)"
	if [ "$COMPRESSION" = zstd ]; then
		tar -C "$RUNS" -cf - "$name" | zstd -q -T0 -o "$PARTIAL/$name.$EXT"
	else
		tar -C "$RUNS" -cf - "$name" | gzip -6 >"$PARTIAL/$name.$EXT"
	fi
	arc_bytes="$(stat -c %s "$PARTIAL/$name.$EXT")"
	arc_sha="$(sha256sum "$PARTIAL/$name.$EXT" | cut -d' ' -f1)"
	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$name.$EXT" "$COMPRESSION" "$n_files" \
		"$src_bytes" "$arc_bytes" "$arc_sha" >>"$PARTIAL/index.tsv"
	echo "archivé : $name ($n_files fichiers, $src_bytes -> $arc_bytes octets)"
done

{
	echo "created_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "host: $(hostname)"
	echo "repo_head: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo inconnu)"
	echo "repo_describe: $(git -C "$ROOT" describe --tags --always 2>/dev/null || echo inconnu)"
	lock="$ROOT/data/registry/reference_2026-10-22.lock.json"
	if [ -f "$lock" ]; then
		echo "reference_lock_sha256: $(sha256sum "$lock" | cut -d' ' -f1)"
	fi
	echo "compression: $COMPRESSION"
	echo "restore: scripts/archive_runs.sh --verify <ce dossier> [dossier de restauration]"
} >"$PARTIAL/meta.txt"
(cd "$PARTIAL" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | sort |
	xargs sha256sum >SHA256SUMS)
mv "$PARTIAL" "$SNAP"
echo "instantané : $SNAP ($(du -sh "$SNAP" | cut -f1)) ; contrôle : $0 --verify $SNAP"
