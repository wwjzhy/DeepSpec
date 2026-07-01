#!/bin/bash
#SBATCH --job-name=sft_classifier
#SBATCH --output=logs/slurm/%j-drafter.out
#SBATCH --error=logs/slurm/%j-drafter.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G

set -euo pipefail

SCRIPT_DIR="/lustre/projects/polyullm/wenjun/speculative/DeepSpec/scripts/train"
WORK_DIR="/lustre/projects/polyullm/wenjun/speculative/DeepSpec"

# ========================================================
# 按集群实际情况修改以下变量
# ========================================================
CONTAINER_NAME="deepspec_sft"
CONTAINER_IMAGE="/lustre/projects/polyullm/container/verl+cu126+0503.sqsh"

# conda 环境：容器内 activate 用的路径（需通过 mount 映射进容器）
CONDA_SH="/lustre/projects/polyullm/wenjun/envs/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="Dspec"

# ========================================================

mkdir -p "${WORK_DIR}/logs/slurm"

echo "WORK_DIR=${WORK_DIR}"
echo "CONTAINER_IMAGE=${CONTAINER_IMAGE}"

SCRIPTS="
set -euo pipefail

nvidia-smi

source '${CONDA_SH}'
conda activate '${CONDA_ENV}'

export HOME='/lustre/projects/polyullm/wenjun'
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=\${MASTER_PORT:-29500}
export RANK=\${RANK:-0}
export WORLD_SIZE=\${WORLD_SIZE:-1}


target_cache_dir=\"\${HOME}/.cache/deepspec/qwen3_4b_target_cache\"

cd '${WORK_DIR}'

python train.py \
    --config config/eagle3/eagle3_qwen3_4b.py \
    --opts \"data.target_cache_path=\${target_cache_dir}\"
"

srun --nodes=1 --ntasks=1 \
    --container-name="${CONTAINER_NAME}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/work/projects/polyullm:/work/projects/polyullm,/lustre/projects/polyullm:/lustre/projects/polyullm \
    --container-remap-root \
    --container-writable \
    bash -c "${SCRIPTS}"
