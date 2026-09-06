#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -J unet0_on_0
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

echo "Activating UNet CUDA environment..."

unset PYTHONPATH
export PATH="/sw/apps/conda/latest/rackham_stage/bin:$PATH"
export CONDA_ENVS_PATH=/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/envs

cd /proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026

nvidia-smi || true

source envs/venv310-cuda/bin/activate

cd /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet

set -euo pipefail

# -----------------------------
# Paths
# -----------------------------
BIG_PATH="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/"

PY_SCRIPT="${BIG_PATH}Unet/evaluation_2d.py"
INFO_CSV="${BIG_PATH}patient_info.csv"

# -----------------------------
# Parameter groups
# -----------------------------

# change here
EVALUATION_AXIS=1   # 0 / 1 / 2
MODEL=ave             # 0 / 1 / 2 / ave

# -----------------------------
# Evaluation-axis group
# -----------------------------
if [[ "$EVALUATION_AXIS" == "0" ]]; then
  SLICE_AXIS=0
  EVAL_AXIS=0

elif [[ "$EVALUATION_AXIS" == "1" ]]; then
  SLICE_AXIS=1
  EVAL_AXIS=1

elif [[ "$EVALUATION_AXIS" == "2" ]]; then
  SLICE_AXIS=2
  EVAL_AXIS=2

else
  echo "Error: invalid EVALUATION_AXIS=$EVALUATION_AXIS"
  exit 1
fi

# -----------------------------
# Model group
# -----------------------------
if [[ "$MODEL" == "0" ]]; then
  MODEL_NAME="axis0_on_UnetModel0"
  GEN_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_evaluated_2d_sliceaxis_${SLICE_AXIS}.csv"
  GEN_PATTERN="{i}_t1_generated__axis0_on_UnetModel0.nii.gz"

elif [[ "$MODEL" == "1" ]]; then
  MODEL_NAME="axis1_on_UnetModel1"
  GEN_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_evaluated_2d_sliceaxis_${SLICE_AXIS}.csv"
  GEN_PATTERN="{i}_t1_generated__axis1_on_UnetModel1.nii.gz"

elif [[ "$MODEL" == "2" ]]; then  
  MODEL_NAME="axis2_on_UnetModel2"
  GEN_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_evaluated_2d_sliceaxis_${SLICE_AXIS}.csv"
  GEN_PATTERN="{i}_t1_generated__axis2_on_UnetModel2.nii.gz"

elif [[ "$MODEL" == "ave" ]]; then
  MODEL_NAME="average_UnetModel"
  GEN_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_evaluated_2d_sliceaxis_${SLICE_AXIS}.csv"
  GEN_PATTERN="{i}_t1_generated__average.nii.gz"

else
  echo "Error: invalid MODEL=$MODEL"
  exit 1
fi

TRUE_DIR="${BIG_PATH}Data_norm/7T"

# -----------------------------
# Output
# -----------------------------
# already set in model group:
# METRICS_CSV=...

# -----------------------------
# Knobs
# -----------------------------
TRUE_PATTERN="{i}_normalized.nii.gz"

IDX_PREFIX="7T_015_BioFINDER_"
IDX_COL="7T_idx"
USAGE_COL="usage"
USAGE_VALUE="test"

START_IDX=126
END_IDX=304

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
  --slice-axis "$SLICE_AXIS"
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
