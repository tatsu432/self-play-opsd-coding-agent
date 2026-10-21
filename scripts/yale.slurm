#!/bin/bash
# Set account/partition/GPU type/count/memory/time with sbatch flags; no site-wide assumptions.
#SBATCH --job-name=selfplay-opsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=selfplay-%j.out
set -euo pipefail
: "${SELFPLAY_PROJECT:?Set project checkout path on allocated project scratch}"
: "${SELFPLAY_TRAINER_SIF:?Set path to the pinned trainer SIF}"
cd "$SELFPLAY_PROJECT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-$SELFPLAY_PROJECT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
# Forward pre-timeout signals to the launcher/process group. Python handlers only
# request a stop; the training loop performs checkpointing at a safe boundary.
child=""
forward() { if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi; }
trap forward USR1 TERM
"${SELFPLAY_LAUNCH_PYTHON:-$SELFPLAY_PROJECT/.venv/bin/python}" scripts/launch.py --environment yale \
  --hardware "${SELFPLAY_HARDWARE:-dual_gpu_80gb}" \
  --settings "${SELFPLAY_SETTINGS:-configs/yale.yaml}" \
  --method "${SELFPLAY_METHOD:-B}" --image "$SELFPLAY_TRAINER_SIF" "$@" &
child=$!
status=0
while kill -0 "$child" 2>/dev/null; do
  set +e
  wait "$child"
  status=$?
  set -e
done
exit "$status"
