#!/bin/bash -l

#SBATCH -A Berzelius-2026-31
#SBATCH -C fat
#SBATCH --gpus=2
#SBATCH -t 20:00:00
#SBATCH -J fusionCNN
#SBATCH -o /home/x_rushe/berzelius-2024-156/users/x_rushe/super-resolution/Unet/logs/%x.%j.out

echo "${SLURM_JOB_NAME}"

unset PYTHONPATH

PROJECT_ROOT="/home/x_rushe/berzelius-2024-156/users/x_rushe/super-resolution"
UNET_DIR="${PROJECT_ROOT}/Unet"
ENV_NAME="sr_env"

mkdir -p "${UNET_DIR}/logs"

echo "========== GPU INFO =========="
nvidia-smi
echo "=============================="

cd "${PROJECT_ROOT}"

module load Mambaforge
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_NAME}"

echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "done env"
which python
python --version
python -c "import sys; print(sys.executable)"

# =========================================================
# paths: change these to your actual files
# =========================================================

# ---------- axis 0 ----------
AXIS0_PT="${PROJECT_ROOT}/Unet/resultsUnet/trial0/my_unet.pt"
AXIS0_PARAM="${PROJECT_ROOT}/Unet/resultsUnet/trial0/params.txt"
AXIS0_INPUTS="${PROJECT_ROOT}/Unet/contents_ano_0/inputs1_s3.npy"
AXIS0_OUTPUTS="${PROJECT_ROOT}/Unet/contents_ano_0/outputs1_s3.npy"
AXIS0_CONTEXTS="${PROJECT_ROOT}/Unet/contents_ano_0/contexts1_s3.npy"
AXIS0_RANGES="${PROJECT_ROOT}/Unet/contents_ano_0/ranges1_s3.npy"

# ---------- axis 1 ----------
AXIS1_PT="${PROJECT_ROOT}/Unet/resultsUnet/trial1/my_unet.pt"
AXIS1_PARAM="${PROJECT_ROOT}/Unet/resultsUnet/trial1/params.txt"
AXIS1_INPUTS="${PROJECT_ROOT}/Unet/contents_ano_1/inputs1_s3.npy"
AXIS1_OUTPUTS="${PROJECT_ROOT}/Unet/contents_ano_1/outputs1_s3.npy"
AXIS1_CONTEXTS="${PROJECT_ROOT}/Unet/contents_ano_1/contexts1_s3.npy"
AXIS1_RANGES="${PROJECT_ROOT}/Unet/contents_ano_1/ranges1_s3.npy"

# ---------- axis 2 ----------
AXIS2_PT="${PROJECT_ROOT}/Unet/resultsUnet/trial2/my_unet.pt"
AXIS2_PARAM="${PROJECT_ROOT}/Unet/resultsUnet/trial2/params.txt"
AXIS2_INPUTS="${PROJECT_ROOT}/Unet/contents_ano_2/inputs1_s3.npy"
AXIS2_OUTPUTS="${PROJECT_ROOT}/Unet/contents_ano_2/outputs1_s3.npy"
AXIS2_CONTEXTS="${PROJECT_ROOT}/Unet/contents_ano_2/contexts1_s3.npy"
AXIS2_RANGES="${PROJECT_ROOT}/Unet/contents_ano_2/ranges1_s3.npy"

OUT_DIR="${PROJECT_ROOT}/Unet/fusion_results"

# optional reference nii dir
REF_NIFTI_DIR=""
REF_NIFTI_PATTERN="{subject}.nii.gz"

mkdir -p "${OUT_DIR}"

cd "${UNET_DIR}"

python fusion_CNN_unet_1.py \
  --axis0_pt "${AXIS0_PT}" \
  --axis0_param "${AXIS0_PARAM}" \
  --axis0_inputs "${AXIS0_INPUTS}" \
  --axis0_outputs "${AXIS0_OUTPUTS}" \
  --axis0_contexts "${AXIS0_CONTEXTS}" \
  --axis0_ranges "${AXIS0_RANGES}" \
  --axis1_pt "${AXIS1_PT}" \
  --axis1_param "${AXIS1_PARAM}" \
  --axis1_inputs "${AXIS1_INPUTS}" \
  --axis1_outputs "${AXIS1_OUTPUTS}" \
  --axis1_contexts "${AXIS1_CONTEXTS}" \
  --axis1_ranges "${AXIS1_RANGES}" \
  --axis2_pt "${AXIS2_PT}" \
  --axis2_param "${AXIS2_PARAM}" \
  --axis2_inputs "${AXIS2_INPUTS}" \
  --axis2_outputs "${AXIS2_OUTPUTS}" \
  --axis2_contexts "${AXIS2_CONTEXTS}" \
  --axis2_ranges "${AXIS2_RANGES}" \
  --out_dir "${OUT_DIR}" \
  --canonical_axis 0 \
  --infer_batch_size 16 \
  --num_workers 4 \
  --fusion_epochs 20 \
  --fusion_lr 1e-4 \
  --fusion_batch_size 1 \
  --patch_size 48 48 48 \
  --patches_per_subject 8 \
  --fusion_use_perceptual_loss \
  --fusion_perceptual_weight 0.001 \
  --device cuda

echo "done"
