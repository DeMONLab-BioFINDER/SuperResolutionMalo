#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -J inferer_test13
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

# Activate environment 
#echo "Activating environment..."
#source /proj/nobackup/sens2023026/data/ants_set_up/venv-ants/bin/activate
# Verify environment
#echo "Current Python path:"; which python
#python --version

# Activate correct environment (same as the working UNet script)
echo "Activating UNet CUDA environment..."

unset PYTHONPATH
export PATH="/sw/apps/conda/latest/rackham_stage/bin:$PATH"
export CONDA_ENVS_PATH=/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/envs


cd /proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026

nvidia-smi || true

source envs/venv310-cuda/bin/activate

cd /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet


# Fail fast
set -euo pipefail

# -----------------------------
# Paths
# -----------------------------
#BIG_PATH="/proj/backup/ns2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/"
#BIG_PATH="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/"
BIG_PATH="/proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/"
PARENT_3T_NORM="${BIG_PATH}Data_norm/3T"
PARENT_7T_NORM="${BIG_PATH}Data_norm/7T"


# If your script is elsewhere, change this:
PY_SCRIPT="${BIG_PATH}Unet/inferer2.py"   #

#PATH_MODEL="${BIG_PATH}Unet/results/trial1/my_unet.pt" # change trial here
#PATH_SETTINGS="${BIG_PATH}Unet/results/trial1/params.txt" # change trial here
#PATH_MODEL="${BIG_PATH}Unet/results/my_unet_no_diag.pt" # change trial here
#PATH_SETTINGS="${BIG_PATH}Unet/results/params_no_diag.txt" # change trial here
PATH_MODEL="${BIG_PATH}Unet/results/unet_test12/trial2/my_unet.pt" # change trial here
PATH_SETTINGS="${BIG_PATH}Unet/results/unet_test12/trial2/params.txt" # change trial here
 

# -----------------------------
# Knobs
# -----------------------------
AXIS=2                    # 0 / 1 / 2 
MODEL_DIM=2               # 0 / 1 / 2 
TAG="axis${AXIS}_on_UnetModel${MODEL_DIM}"

START_IDX=126
END_IDX=304
#END_IDX=131
QUIET_SLICES=1            # 
USE_GPU_PERCEPTUAL=0      # 

# Output names (auto-tagged by axis)
# OUT_DIR="${BIG_PATH}to_seg/UNet__axis${AXIS}"
# METRICS_CSV="${BIG_PATH}to_seg/metricsUnet_axis${AXIS}.csv"

OUT_DIR="${BIG_PATH}to_seg/test14/${TAG}"
METRICS_CSV="${BIG_PATH}to_seg/test14/metrics_${TAG}.csv"


echo "Running UNet infer + eval"
echo "AXIS=${AXIS}"
echo "OUT_DIR=${OUT_DIR}"
echo "METRICS_CSV=${METRICS_CSV}"

# -----------------------------
# Run
# -----------------------------
CMD=(python "$PY_SCRIPT"
  --axis "$AXIS"
  --model-dim "$MODEL_DIM"
  --big-path "$BIG_PATH"
  --parent-3t-norm "$PARENT_3T_NORM" 
  --parent-7t-norm "$PARENT_7T_NORM" 
  --path-model "$PATH_MODEL"
  --path-settings "$PATH_SETTINGS"
  --output-dir "$OUT_DIR"
  --metrics-csv "$METRICS_CSV"
  --start-idx "$START_IDX"
  --end-idx "$END_IDX"

)

if [[ "$QUIET_SLICES" -eq 1 ]]; then
  CMD+=(--quiet-slices)
fi

if [[ "$USE_GPU_PERCEPTUAL" -eq 1 ]]; then
  CMD+=(--use-gpu-perceptual)
fi

echo "Executing:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "done"



