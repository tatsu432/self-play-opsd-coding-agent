#!/bin/bash
# Resources are supplied by the validated HPC controller, never inferred here.
#SBATCH --job-name=selfplay-opsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@300
set -euo pipefail

: "${SLURM_JOB_ID:?Run only inside a Slurm allocation}"
: "${SELFPLAY_PROJECT:?Missing immutable project checkout}"
: "${SELFPLAY_TRAINER_SIF:?Missing trainer image}"
: "${SELFPLAY_TRAINER_SHA256:?Missing trainer image digest}"
: "${SELFPLAY_DATA:?Missing prepared data}"
: "${SELFPLAY_RUNS:?Missing durable run directory}"
: "${SELFPLAY_OPERATION:?Missing operation}"
: "${SELFPLAY_METHOD:?Missing method}"
: "${SELFPLAY_HARDWARE:?Missing hardware profile}"
: "${SELFPLAY_GPU_COUNT:?Missing GPU count}"
: "${SELFPLAY_SETTINGS:?Missing settings}"

cd "$SELFPLAY_PROJECT"
test "$(git rev-parse HEAD)" = "$(basename "$SELFPLAY_PROJECT")"
actual_image_sha256="$(sha256sum "$SELFPLAY_TRAINER_SIF" | cut -d ' ' -f1)"
test "$actual_image_sha256" = "$SELFPLAY_TRAINER_SHA256"

mkdir -p "$SELFPLAY_RUNS" "${SELFPLAY_CACHE:-$SELFPLAY_PROJECT/.cache/huggingface}"
export APPTAINERENV_SELFPLAY_DATA_ROOT=/data/prepared
export APPTAINERENV_SELFPLAY_OUTPUT_ROOT=/runs
export APPTAINERENV_HF_HOME=/cache/huggingface
export APPTAINERENV_PYTHONPATH=/workspace/src
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_SELFPLAY_ALLOCATED_GPU_COUNT="$SELFPLAY_GPU_COUNT"
export APPTAINERENV_SELFPLAY_IMAGE_DIGEST="$SELFPLAY_TRAINER_SHA256"
export APPTAINERENV_SELFPLAY_STOP_FILE="/runs/$SELFPLAY_STUDY/$SELFPLAY_METHOD-seed$SELFPLAY_SEED/state/stop_requested"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

if [[ -n "${SELFPLAY_SECRETS_FILE:-}" ]]; then
  test -f "$SELFPLAY_SECRETS_FILE"
  test "$(stat -c %a "$SELFPLAY_SECRETS_FILE")" = 600
  # File is a user-owned trusted environment file, never in Git or logs.
  set -a
  source "$SELFPLAY_SECRETS_FILE"
  set +a
  if [[ -n "${SELFPLAY_WORKER_TOKEN:-}" ]]; then
    export APPTAINERENV_SELFPLAY_WORKER_TOKEN="$SELFPLAY_WORKER_TOKEN"
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
  fi
fi

args=(torchrun --standalone --nnodes=1 "--nproc-per-node=$SELFPLAY_GPU_COUNT" -m selfplay_opsd.cli "$SELFPLAY_OPERATION" --settings "/workspace/$SELFPLAY_SETTINGS")
case "$SELFPLAY_OPERATION" in
  train)
    args+=(--config "$SELFPLAY_METHOD" --hardware "$SELFPLAY_HARDWARE")
    if [[ -n "${SELFPLAY_RESUME:-}" ]]; then args+=(--resume "$SELFPLAY_RESUME"); fi
    ;;
  pilot)
    args+=(--config "$SELFPLAY_METHOD" --hardware "$SELFPLAY_HARDWARE" --iterations "${SELFPLAY_ITERATIONS:-10}")
    if [[ -n "${SELFPLAY_RESUME:-}" ]]; then args+=(--resume "$SELFPLAY_RESUME"); fi
    ;;
  backend-check)
    args+=(--hardware "$SELFPLAY_HARDWARE" --output "$SELFPLAY_OUTPUT")
    if [[ -n "${SELFPLAY_REFERENCE:-}" ]]; then args+=(--reference "$SELFPLAY_REFERENCE"); fi
    ;;
  evaluate)
    args+=(--config "$SELFPLAY_METHOD" --hardware "$SELFPLAY_HARDWARE" --split "$SELFPLAY_SPLIT" --endpoint "$SELFPLAY_ENDPOINT" --protocol frozen)
    if [[ "${SELFPLAY_BASE:-0}" = 1 ]]; then args+=(--base); else args+=(--checkpoint "$SELFPLAY_CHECKPOINT"); fi
    ;;
  diagnose)
    args+=(--config "$SELFPLAY_METHOD" --hardware "$SELFPLAY_HARDWARE" --checkpoint "$SELFPLAY_CHECKPOINT" --output "$SELFPLAY_OUTPUT")
    ;;
  *) printf 'Unsupported operation: %s\n' "$SELFPLAY_OPERATION" >&2; exit 2 ;;
esac

stop_file="$SELFPLAY_RUNS/$SELFPLAY_STUDY/$SELFPLAY_METHOD-seed$SELFPLAY_SEED/state/stop_requested"
request_stop() { mkdir -p "$(dirname "$stop_file")"; touch "$stop_file"; }
trap request_stop USR1 TERM

apptainer exec --nv --containall --cleanenv \
  --bind "$SELFPLAY_PROJECT:/workspace:ro" \
  --bind "$SELFPLAY_DATA:/data/prepared:ro" \
  --bind "$SELFPLAY_RUNS:/runs" \
  --bind "${SELFPLAY_CACHE:-$SELFPLAY_PROJECT/.cache/huggingface}:/cache/huggingface" \
  --pwd /workspace "$SELFPLAY_TRAINER_SIF" "${args[@]}" &
child=$!
status=0
while true; do
  set +e
  wait "$child"
  status=$?
  set -e
  # A Slurm signal interrupts Bash's wait; keep the batch shell alive while
  # the child observes the shared stop request and checkpoints safely.
  if kill -0 "$child" 2>/dev/null; then continue; fi
  break
done
exit "$status"
