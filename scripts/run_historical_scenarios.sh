#!/usr/bin/env bash
# Étude isolée, référence existante requise ; reprise des seuls cas restants.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$(realpath -m "${1:-$project_root/runs/2026-10-22-historical-scenarios}")"
mkdir -p "$(dirname "$output_dir")"
exec 9>"$output_dir.lock"
if ! flock -n 9; then
  printf '%s\n' "Une étude utilise déjà $output_dir" >&2
  exit 75
fi
cd "$project_root/back"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vvs-mpl-historical-scenarios}"
uv run --no-sync python studies/historical_scenarios.py \
  --reference "$project_root/runs/2026-10-22-consolidation" \
  --out "$output_dir" \
  --protocol "$project_root/data/registry/historical_scenarios_2026-10-22.yaml" \
  --implementation "$project_root/back/src/vvs/eval/historical_scenarios.py" \
  --pin "$project_root/data/registry/reference_2026-10-22.lock.json"
