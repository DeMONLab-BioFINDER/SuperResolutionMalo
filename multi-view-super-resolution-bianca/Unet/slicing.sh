#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -n 3
#SBATCH -t 01:10:00
#SBATCH -J ano3dim_strict
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

echo "Activating ANTs environment..."
source /proj/nobackup/sens2023026/data/ants_set_up/venv-ants/bin/activate

echo "Current Python path:"; which python
python --version

set -euo pipefail

# -----------------------------
# Paths
# -----------------------------
BIG_PATH="/proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/"
SAVE_PATH="${BIG_PATH}Unet/contents_2"   # change if needed
PARENT_7T="${BIG_PATH}T1_7T_processed/"
PARENT_3T="${BIG_PATH}jake__20240405_172444___t1___brain___fs/processed_reg/"
PATIENT_CSV="patient_info.csv"

# -----------------------------
# Knobs
# -----------------------------
AXIS=2                    # 0 / 1 / 2
IN_SLICES=3
USAGE=train
TRAIN_SIZE=138
SEED=2025

# Original-style index selection
PERM_RANGE=179
IDX_OFFSET=126
SKIP="203 204 223 248 274 288 300 184 192"

# Original fixed borders
BORDER_BOT="50 61 26"
BORDER_TOP="337 302 247"

# Intensity processing
NORM_METHOD=min_max       # min_max / std_mean
CLIP_LO=1
CLIP_HI=99
DTYPE=float16             # float16 / float32

# Save / run control
RUN_TAG=slurm
BATCH_SUBJECTS=10
SAVE_INDEX=1

echo "Executing ano_3dim_param_strict.py..."

python slicing.py \
  --big-path "$BIG_PATH" \
  --parent-3t "$PARENT_3T" \
  --parent-7t "$PARENT_7T" \
  --save-path "$SAVE_PATH" \
  --patient-csv "$PATIENT_CSV" \
  --axis $AXIS \
  --n-in-slices $IN_SLICES \
  --normalization-method $NORM_METHOD \
  --clip-low $CLIP_LO \
  --clip-high $CLIP_HI \
  --usage $USAGE \
  --train-size $TRAIN_SIZE \
  --seed $SEED \
  --perm-range $PERM_RANGE \
  --idx-offset $IDX_OFFSET \
  --skip $SKIP \
  --border-bot $BORDER_BOT \
  --border-top $BORDER_TOP \
  --dtype $DTYPE \
  --run-tag "$RUN_TAG" \
  --batch-subjects $BATCH_SUBJECTS \
  --save-index $SAVE_INDEX

echo "done"
