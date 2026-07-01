#!/bin/bash
# 用法: bash scripts/train/submit_train.sh dflash
#       bash scripts/train/submit_train.sh dspark
#       bash scripts/train/submit_train.sh eagle3

set -euo pipefail

ALGO="${1:-}"
if [[ -z "${ALGO}" ]]; then
    echo "Usage: bash scripts/train/submit_train.sh {dflash|dspark|eagle3}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config_slurm.sh
source "${SCRIPT_DIR}/config_slurm.sh"

JOB_SCRIPT="${SCRIPT_DIR}/${ALGO}_slurm.sh"
if [[ ! -f "${JOB_SCRIPT}" ]]; then
    echo "Error: job script not found: ${JOB_SCRIPT}"
    exit 1
fi

mkdir -p "${DEEPSPEC_LOG_DIR}"

echo "DEEPSPEC_ROOT=${DEEPSPEC_ROOT}"
echo "LOG_DIR=${DEEPSPEC_LOG_DIR}"
echo "JOB_SCRIPT=${JOB_SCRIPT}"

sbatch \
    --chdir="${DEEPSPEC_ROOT}" \
    --output="${DEEPSPEC_LOG_DIR}/%j-${ALGO}.out" \
    --error="${DEEPSPEC_LOG_DIR}/%j-${ALGO}.err" \
    "${JOB_SCRIPT}"
