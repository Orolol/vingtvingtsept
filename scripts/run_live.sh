#!/usr/bin/env bash
# Run live de la prévision (phase 8, docs/09-roadmap.md) : ingestion des sondages 2027,
# curation, simulation, export JSON et construction du site. Voir scripts/README.md.
#
# Usage : scripts/run_live.sh [--dry-run]
#
#   --dry-run   affiche les commandes sans rien exécuter ni écrire
#
# Variables d'environnement :
#   VVS_SUBSTITUTION=1|0   substitution intra-bloc (1 par défaut : choix publié actuel)
#   VVS_ASOF=AAAA-MM-JJ    date de prévision (défaut : aujourd'hui)
#   VVS_DRAWS=N            tirages Monte-Carlo (défaut : celui de `vvs simulate`)
#   VVS_ENV_FILE=chemin    fichier d'environnement lu s'il existe (défaut
#                          ~/.config/vingtvingtsept/env) ; jamais écrit ni journalisé
#
# Idempotent : les ingestions n'ajoutent que ce qui manque (data/raw est en ajout seul),
# l'ajustement n'est refait que si l'empreinte des sondages change, et l'export ne réécrit
# jamais history/<date>.json. Arrêt au premier échec ; journal dans runs/logs/.
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

DRY_RUN=0
while [ $# -gt 0 ]; do
	case "$1" in
	--dry-run) DRY_RUN=1 ;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "option inconnue : $1" >&2
		usage >&2
		exit 2
		;;
	esac
	shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACK="$ROOT/back"
FRONT="$ROOT/front"
DATA="$FRONT/static/data"
LOG_DIR="$ROOT/runs/logs"

ENV_FILE="${VVS_ENV_FILE:-$HOME/.config/vingtvingtsept/env}"
if [ "$DRY_RUN" = 0 ] && [ -f "$ENV_FILE" ]; then
	# MEDIACLOUD_API_KEY et autres secrets : lus ici, jamais affichés
	set -a
	# shellcheck disable=SC1090
	. "$ENV_FILE"
	set +a
fi

ASOF="${VVS_ASOF:-$(date +%F)}"
case "$ASOF" in
[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;;
*)
	echo "VVS_ASOF invalide : $ASOF (attendu AAAA-MM-JJ)" >&2
	exit 2
	;;
esac
SINCE="$((${ASOF%%-*} - 1))-01-01"
case "${VVS_SUBSTITUTION:-1}" in
1 | true | yes) SUBST_FLAG="--substitution" SUFFIX="-subst" SUBST_LABEL="oui" ;;
0 | false | no) SUBST_FLAG="" SUFFIX="" SUBST_LABEL="non" ;;
*)
	echo "VVS_SUBSTITUTION invalide : ${VVS_SUBSTITUTION} (1 ou 0)" >&2
	exit 2
	;;
esac
RUN_DIR="$ROOT/runs/${ASOF}-simulate${SUFFIX}"
DRAWS_FLAG=""
if [ -n "${VVS_DRAWS:-}" ]; then DRAWS_FLAG="--draws ${VVS_DRAWS}"; fi

# cron et systemd partent d'un PATH minimal : uv dans ~/.local/bin, Node via mise
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v npm >/dev/null 2>&1 && command -v mise >/dev/null 2>&1; then
	NODE_DIR="$(cd "$FRONT" && mise where node 2>/dev/null || true)"
	if [ -n "$NODE_DIR" ]; then export PATH="$NODE_DIR/bin:$PATH"; fi
fi

ts() { date +%H:%M:%S; }

# step <dossier> <commande…> : affiche, exécute (sauf --dry-run), chronomètre ; s'arrête au
# premier échec avec le code de la commande.
step() {
	local dir="$1"
	shift
	local where="${dir#"$ROOT"}"
	where="${where#/}"
	echo "[$(ts)] (${where:-.}) \$ $*"
	if [ "$DRY_RUN" = 1 ]; then return 0; fi
	local t0 rc
	t0=$(date +%s)
	set +e
	(cd "$dir" && "$@")
	rc=$?
	set -e
	if [ "$rc" -ne 0 ]; then
		echo "[$(ts)] ÉCHEC (code $rc) : $* ; arrêt du run" >&2
		exit "$rc"
	fi
	echo "[$(ts)] ok en $(($(date +%s) - t0)) s"
}

if [ "$DRY_RUN" = 0 ]; then
	mkdir -p "$LOG_DIR"
	LOG="$LOG_DIR/live-$(date +%Y-%m-%dT%H%M%S).log"
	exec 9>"$LOG_DIR/live.lock"
	if ! flock -n 9; then
		echo "un run live est déjà en cours (verrou $LOG_DIR/live.lock)" >&2
		exit 75
	fi
	exec > >(tee -a "$LOG") 2>&1
	PREV="$(mktemp -d)"
	trap 'rm -rf "$PREV"' EXIT
else
	LOG="$LOG_DIR/live-<horodatage>.log"
	PREV="<copie temporaire de front/static/data>"
fi

START=$(date +%s)
echo "[$(ts)] run live du $ASOF (substitution : $SUBST_LABEL) ; journal $LOG"
[ "$DRY_RUN" = 1 ] && echo "[$(ts)] --dry-run : rien n'est exécuté"

step "$BACK" uv run vvs ingest commission --since "$SINCE"
step "$BACK" uv run vvs ingest wikipedia --election 2027
step "$BACK" uv run vvs ingest mieuxvoter
step "$BACK" uv run vvs parse commission
step "$BACK" uv run vvs parse wikipedia --election 2027
step "$BACK" uv run vvs parse mieuxvoter
step "$BACK" uv run vvs curate
# shellcheck disable=SC2086
step "$BACK" uv run vvs simulate --asof "$ASOF" $SUBST_FLAG $DRAWS_FLAG --out "$RUN_DIR"
step "$ROOT" cp -a "$DATA/." "$PREV/"
step "$BACK" uv run vvs export --run "$RUN_DIR"
step "$FRONT" npm run validate-data
step "$FRONT" npm run build
step "$BACK" uv run vvs live-summary --previous "$PREV" --current "$DATA"

if [ "$DRY_RUN" = 1 ]; then
	echo "[$(ts)] fin du --dry-run"
	exit 0
fi
echo "[$(ts)] run terminé en $(($(date +%s) - START)) s ; site construit dans front/build"
echo "[$(ts)] à publier : commit de front/static/data (et data/curated), puis push"
