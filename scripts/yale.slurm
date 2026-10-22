#!/bin/bash
# Validated requests and signals are supplied by the HPC controller.
#SBATCH --job-name=selfplay-opsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${SLURM_JOB_ID:?Run only inside a Slurm allocation}"
: "${SELFPLAY_PROJECT:?Missing immutable project checkout}"
: "${SELFPLAY_OPERATION:?Missing operation}"
cd "$SELFPLAY_PROJECT"
test "$(git rev-parse HEAD)" = "$(basename "$SELFPLAY_PROJECT")"
test -z "$(git status --porcelain --untracked-files=all)"
if [[ -n "${SELFPLAY_SECRETS_FILE:-}" ]]; then
  test -f "$SELFPLAY_SECRETS_FILE"
  test "$(stat -c %a "$SELFPLAY_SECRETS_FILE")" = 600
  test "$(stat -c %u "$SELFPLAY_SECRETS_FILE")" = "$(id -u)"
  set -a
  source "$SELFPLAY_SECRETS_FILE"
  set +a
fi
exec python3 scripts/hpc_job.py
