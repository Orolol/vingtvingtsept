#!/usr/bin/env bash
# Reprise vérifiée des étapes restantes ; aucun export live ni publication.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-$project_root/runs/2026-10-22-consolidation}"
output_dir="$(realpath -m "$output_dir")"
jobs="${VVS_CONSOLIDATION_JOBS:-2}"
cd "$project_root/back"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=2
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vvs-mpl-consolidation}"
uv run --no-sync python -m vvs.eval.consolidation freeze --out "$output_dir"
uv run --no-sync python -m vvs.eval.consolidation run --out "$output_dir" --jobs "$jobs"
