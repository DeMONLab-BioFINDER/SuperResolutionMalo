#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -J dice_3T
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

BIG_PATH="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/"

PY_SCRIPT="${BIG_PATH}Unet/evaluation_DICE.py"
INFO_CSV="${BIG_PATH}patient_info.csv"

# change here
MODEL=3T     # 0 / 1 / 2 / ave / cnn / 3T

if [[ "$MODEL" == "0" ]]; then
  MODEL_NAME="axis0_on_UnetModel0"
  PRED_SEG_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_synthseg"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_dice.csv"

elif [[ "$MODEL" == "1" ]]; then
  MODEL_NAME="axis1_on_UnetModel1"
  PRED_SEG_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_synthseg"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_dice.csv"

elif [[ "$MODEL" == "2" ]]; then
  MODEL_NAME="axis2_on_UnetModel2"
  PRED_SEG_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_synthseg"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_dice.csv"

elif [[ "$MODEL" == "ave" ]]; then
  MODEL_NAME="average_UnetModel"
  PRED_SEG_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_synthseg"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_dice.csv"

elif [[ "$MODEL" == "cnn" ]]; then
  MODEL_NAME="fusionCNN"
  PRED_SEG_DIR="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_synthseg"
  METRICS_CSV="${BIG_PATH}to_seg/test12_full/${MODEL_NAME}_dice.csv"

elif [[ "$MODEL" == "3T" ]]; then
  MODEL_NAME="3T_baseline_data"
  PRED_SEG_DIR="${BIG_PATH}to_seg/${MODEL_NAME}_synthseg"
  METRICS_CSV="${BIG_PATH}to_seg/3T_baseline/${MODEL_NAME}_dice.csv"

else
  echo "Error: invalid MODEL=$MODEL"
  exit 1
fi

TRUE_SEG_DIR="${BIG_PATH}Data_norm/7T_synthseg"

IDX_PREFIX="7T_015_BioFINDER_"
IDX_COL="7T_idx"
USAGE_COL="usage"
USAGE_VALUE="test"

START_IDX=126
END_IDX=304
MISSING_LIST="[203,204,223,248,274,288,300,184,192]"

CMD=(python "$PY_SCRIPT"
  --big-path "$BIG_PATH"
  --info-csv "$INFO_CSV"
  --pred-seg-dir "$PRED_SEG_DIR"
  --true-seg-dir "$TRUE_SEG_DIR"
  --metrics-csv "$METRICS_CSV"
  --idx-prefix "$IDX_PREFIX"
  --idx-col "$IDX_COL"
  --usage-col "$USAGE_COL"
  --usage-value "$USAGE_VALUE"
  --start-idx "$START_IDX"
  --end-idx "$END_IDX"
  --missing "$MISSING_LIST"
)

echo "Executing:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "done"
