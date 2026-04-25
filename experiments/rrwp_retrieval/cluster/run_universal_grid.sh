#!/usr/bin/env bash
# run_universal_grid.sh — submit the universal sweep (QM9 + PubChem variants) to uc3.
# Mirrors the defaults in experiments/rrwp_retrieval/sweep_universal.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_one.sh"
PROJECT_DIR="${GHDC_PUBLIC_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

export ONLY_PARTITIONS="${ONLY_PARTITIONS:-gpu_h100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/experiments/results/cluster/rrwp_retrieval/universal}"

# ── Grid (mirrors sweep_universal.py defaults) ──────────────────────
DATASETS=(qm9 pubchem16 pubchem32 pubchem64)
DIMS=(1024)
DEPTHS=(3)
K_VALUES=(
  "4,10,14,18"
  "4,12,18,24"
  "6,8,10,12"
  "6,10,14,18"
  "6,12,18,24"
)
BINS=(8)
BEAM_SIZES=(32)
N_SAMPLES="${N_SAMPLES:-5000}"
SEED="${SEED:-42}"

N=0
for ds in "${DATASETS[@]}"; do
  for dim in "${DIMS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
      for k in "${K_VALUES[@]}"; do
        for b in "${BINS[@]}"; do
          for bs in "${BEAM_SIZES[@]}"; do
            DATASET="$ds" \
            HV_DIM="$dim" \
            DEPTH="$depth" \
            K_VALUES="$k" \
            NUM_BINS="$b" \
            BEAM_SIZE="$bs" \
            N_SAMPLES="$N_SAMPLES" \
            SEED="$SEED" \
              bash "$SUBMIT"
            N=$((N+1))
          done
        done
      done
    done
  done
done

echo
echo "Grid: $N configs (DATASETS=${DATASETS[*]}, partition=${ONLY_PARTITIONS})"
