#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -J gan_eval_3d
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

echo "Activating UNet CUDA environment..."

unset PYTHONPATH
export PATH="/sw/apps/conda/latest/rackham_stage/bin:$PATH"
export CONDA_ENVS_PATH=/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/envs

cd /proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026

nvidia-smi || true

source envs/venv310-cuda/bin/activate

# set your project cwd (adjust if you store scripts elsewhere)
cd /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet

# Fail fast
set -euo pipefail

# -----------------------------
# Paths
# -----------------------------
BIG_PATH="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/"

PY_SCRIPT="${BIG_PATH}Unet/evaluation.py"

INFO_CSV="${BIG_PATH}patient_info.csv"

# evaluate what
# MODEL_NAME="average_UnetModel"
MODEL_NAME="average_UnetModel"

GEN_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}" # path of average .nifti to compare/evaluate#####


# Axis used for no_corrupt evaluation cropping (independent from input data source)
EVAL_AXIS=1


# where is ground truth 7T
TRUE_DIR="${BIG_PATH}Data_norm/7T" # ground truth 7T

# result csv path
METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}.csv" # result csv#####

# -----------------------------
# Knobs
# -----------------------------

# name of GEN
# GEN_PATTERN="{i}_normalized.nii.gz"      # pattern for 3T baseline
GEN_PATTERN="{i}_t1_generated__average.nii.gz" # pattern for average#####
# GEN_PATTERN="{i}_t1_generated__axis0_on_UnetModel0.nii.gz"

# name of TRUE
TRUE_PATTERN="{i}_normalized.nii.gz"     # pattern for 7T Ground Truth 





IDX_PREFIX="7T_015_BioFINDER_"
IDX_COL="7T_idx"
USAGE_COL="usage"
USAGE_VALUE="test"

START_IDX=126
END_IDX=304

# Missing list (string form is ok)
MISSING_LIST="[203,204,223,248,274,288,300,184,192]"

DISABLE_NO_CORRUPT=0

CLIP_PERCENTILE=99.0

ENABLE_PERCEPTUAL=1
USE_GPU_PERCEPTUAL=0
PRINT_RATIO=0

# -----------------------------
# Run
# -----------------------------
CMD=(python "$PY_SCRIPT"
  --big-path "$BIG_PATH"
  --info-csv "$INFO_CSV"
  --gen-dir "$GEN_DIR"
  --true-dir "$TRUE_DIR"
  --gen-pattern "$GEN_PATTERN"
  --true-pattern "$TRUE_PATTERN"
  --metrics-csv "$METRICS_CSV"
  --idx-prefix "$IDX_PREFIX"
  --idx-col "$IDX_COL"
  --usage-col "$USAGE_COL"
  --usage-value "$USAGE_VALUE"
  --start-idx "$START_IDX"
  --end-idx "$END_IDX"
  --missing "$MISSING_LIST"
  --eval-axis "$EVAL_AXIS"
  --clip-percentile "$CLIP_PERCENTILE"
)



if [[ "$DISABLE_NO_CORRUPT" -eq 1 ]]; then
  CMD+=(--disable-no-corrupt)
fi

if [[ "$ENABLE_PERCEPTUAL" -eq 1 ]]; then
  CMD+=(--enable-perceptual)
fi

if [[ "$USE_GPU_PERCEPTUAL" -eq 1 ]]; then
  CMD+=(--use-gpu-perceptual)
fi

if [[ "$PRINT_RATIO" -eq 1 ]]; then
  CMD+=(--print-ratio)
fi

echo "Executing:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "done"
