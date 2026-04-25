#!/usr/bin/env bash
# submit_one.sh — Submit a single RRWP retrieval config to BWUNICLUSTER 3.0 (uc3).
#
# One sbatch per call. Driver scripts (run_*_grid.sh) loop over a grid and call
# this once per config. Per-config inputs come in via env vars.
#
# Required env vars (with defaults):
#   DATASET=zinc
#   HV_DIM=512
#   DEPTH=3
#   K_VALUES="6,12"
#   NUM_BINS=8
#   BEAM_SIZE=32
#   N_SAMPLES=5000
#   SEED=42
#   OUTPUT_DIR=$GHDC_PUBLIC_HOME/experiments/results/cluster/rrwp_retrieval
#   SKIP_GRAPH_DECODE=0   # set to 1 to pass --skip_graph_decode
#
# Cluster scheduling overrides (optional):
#   GPUS=1, CPUS_PER_TASK=16, NODES=1, NTASKS=1
#   TIME=24:00:00, MEM=64G            # override defaults for all partitions
#   ONLY_PARTITIONS=gpu_h100,gpu_a100_il   # comma-separated filter
#   DRY_RUN=1                          # print sbatch invocations, don't submit

set -euo pipefail

# --- DRY-RUN normalization ---
DRY_RUN_RAW="${DRY_RUN:-0}"
DRY_RUN="$(printf '%s' "$DRY_RUN_RAW" | tr -d '\r[:space:]')"
DRY_RUN="${DRY_RUN:-0}"

# -----------------------------
# Config
# -----------------------------
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"

DATASET="${DATASET:-zinc}"
HV_DIM="${HV_DIM:-512}"
DEPTH="${DEPTH:-3}"
K_VALUES="${K_VALUES:-6,12}"
NUM_BINS="${NUM_BINS:-8}"
BEAM_SIZE="${BEAM_SIZE:-32}"
N_SAMPLES="${N_SAMPLES:-5000}"
SEED="${SEED:-42}"
SKIP_GRAPH_DECODE="${SKIP_GRAPH_DECODE:-0}"

ONLY_PARTITIONS="${ONLY_PARTITIONS:-}"

# Paths — derive project root from this script's location so the script works
# regardless of caller's cwd.
SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${GHDC_PUBLIC_HOME:-$(cd "$SUBMIT_DIR/../../.." && pwd)}}"
EXPERIMENTS_PATH="${PROJECT_DIR}/experiments/rrwp_retrieval"
SCRIPT="${EXPERIMENTS_PATH}/run_rrwp_retrieval.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/experiments/results/cluster/rrwp_retrieval}"

# uc3-only — all jobs use these settings
MODULE_LOAD="module load devel/cuda"
UV_EXTRA="cuda128"

# Job name
EXP_NAME="rrwp ${DATASET} dim=${HV_DIM} d=${DEPTH} k=${K_VALUES} b=${NUM_BINS}"

# -----------------------------
# Build python args
# -----------------------------
PY_ARGS=(
  "$SCRIPT"
  --dataset "$DATASET"
  --hv_dim "$HV_DIM"
  --depth "$DEPTH"
  --k_values "$K_VALUES"
  --num_bins "$NUM_BINS"
  --beam_size "$BEAM_SIZE"
  --n_samples "$N_SAMPLES"
  --seed "$SEED"
  --output_dir "$OUTPUT_DIR"
)
if [[ "$SKIP_GRAPH_DECODE" == "1" ]]; then
  PY_ARGS+=(--skip_graph_decode)
fi
QUOTED_ARGS="$(printf '%q ' "${PY_ARGS[@]}")"

# -----------------------------
# uc3 partition tuples (partition|time|mem)
# -----------------------------
TUPLES=$'gpu_h100|4:00:00|64G\ngpu_a100_il|4:00:00|64G\ngpu_h100_il|4:00:00|64G'
TUPLES="$(printf '%s' "$TUPLES" | tr -d '\r')"

# -----------------------------
# Helpers
# -----------------------------
contains_in_filter() {
  local part="$1"
  [[ -z "$ONLY_PARTITIONS" ]] && return 0
  IFS=',' read -r -a arr <<<"$ONLY_PARTITIONS"
  for p in "${arr[@]}"; do [[ "$p" == "$part" ]] && return 0; done
  return 1
}

submit_one() {
  local partition="$1" time="$2" mem="$3"
  # Allow caller-level TIME/MEM overrides
  time="${TIME:-$time}"
  mem="${MEM:-$mem}"

  local cmd=( sbatch
    --job-name="$EXP_NAME"
    --partition="$partition"
    --time="$time"
    --gres="gpu:${GPUS}"
    --nodes="$NODES"
    --ntasks="$NTASKS"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$mem"
    --output="${OUTPUT_DIR}/slurm-%j.out"
    --wrap="$(
      cat <<WRAP
set -euo pipefail
${MODULE_LOAD}
echo 'Node:' \$(hostname)
echo 'CUDA visible devices:'; nvidia-smi || true
echo 'Running: ${SCRIPT}'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
mkdir -p '${OUTPUT_DIR}'
cd '${PROJECT_DIR}'
uv run --frozen --extra ${UV_EXTRA} python ${QUOTED_ARGS}
WRAP
    )"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN] '; printf '%q ' "${cmd[@]}"; printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

# -----------------------------
# Main
# -----------------------------
echo "Cluster : uc3 (BWUNICLUSTER 3.0)"
echo "Project : ${PROJECT_DIR}"
echo "Script  : ${SCRIPT}"
echo "Output  : ${OUTPUT_DIR}"
echo "Exp     : ${EXP_NAME}"
echo "DryRun  : ${DRY_RUN}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: Script not found: $SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

while IFS='|' read -r PARTITION P_TIME P_MEM; do
  [[ -z "${PARTITION:-}" ]] && continue
  if contains_in_filter "$PARTITION"; then
    echo "Submitting -> partition=${PARTITION} time=${TIME:-$P_TIME} mem=${MEM:-$P_MEM} cpus=${CPUS_PER_TASK}"
    submit_one "$PARTITION" "$P_TIME" "$P_MEM"
  else
    echo "Skipping (filtered) -> ${PARTITION}"
  fi
done <<< "$TUPLES"
