#!/usr/bin/env bash
# Reprend les seules étapes restantes, vérifie leurs sources et leurs sorties.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$(realpath -m "${1:-$project_root/runs/2026-10-23-structure-v2}")"
mkdir -p "$(dirname "$output_dir")"
exec 9>"$output_dir.lock"
flock -n 9 || exit 75
cd "$project_root/back"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vvs-mpl-structure}"
uv run --no-sync python studies/structure.py \
  --project "$project_root" \
  --reference "${VVS_STRUCTURE_REFERENCE:-$project_root/runs/2026-10-22-consolidation}" \
  --out "$output_dir" --jobs "${VVS_STRUCTURE_JOBS:-3}"
