#!/usr/bin/env bash
# Lot 2 de docs/12 : scénarios historiques par décision de candidature, sur le gel existant.
# Étapes reprenables : construction (code courant), simulation (code du gel), bilan.
# JOBS=n lance les cas en parallèle (défaut 1) ; un seul GPU n'est pas requis (CPU seul).
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$(realpath -m "${1:-$project_root/runs/2026-10-23-decision-scenarios}")"
jobs="${JOBS:-1}"
mkdir -p "$(dirname "$output_dir")"
exec 9>"$output_dir.lock"
if ! flock -n 9; then
  printf '%s\n' "Une étude utilise déjà $output_dir" >&2
  exit 75
fi
cd "$project_root/back"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vvs-mpl-decision-scenarios}"
common=(
  --reference "$project_root/runs/2026-10-22-consolidation"
  --out "$output_dir"
  --protocol "$project_root/data/registry/historical_decisions_2026-10-23.yaml"
  --pin "$project_root/data/registry/reference_2026-10-22.lock.json"
)
uv run --no-sync python studies/decision_scenarios.py build "${common[@]}"
run=(uv run --no-sync python studies/decision_scenarios.py run "${common[@]}"
     --implementation "$project_root/back/src/vvs/eval/decision_scenarios.py")
# Premier cas seul : crée le manifest de l'étude avant tout lancement parallèle.
"${run[@]}" --cases 2012-007
cases=(2012-180 2012-090 2012-030 2017-180 2017-090 2017-030 2017-007
       2022-180 2022-090 2022-030 2022-007)
printf '%s\n' "${cases[@]}" | xargs -P "$jobs" -I{} "${run[@]}" --cases {}
"${run[@]}"
uv run --no-sync python studies/decision_scenarios.py report "${common[@]}" \
  --previous "$project_root/runs/2026-10-23-stage-diagnostics"
