#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -J metricSummary
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out
#SBATCH --mem=16G

echo "Activating environment..."
source /proj/nobackup/sens2023026/data/ants_set_up/venv-ants/bin/activate

MAIN_DIR="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full"
BASELINE_DIR="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/3T_baseline"
OUTPUT_DIR="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full/evaluation_results"

PY_SCRIPT="summary1.py"

MODEL_2D_PATTERN="${MAIN_DIR}/{model}_evaluated_2d_sliceaxis_{axis}.csv"
MODEL_3D_PATTERN="${MAIN_DIR}/metrics_{model}.csv"
MODEL_LPIPS_PATTERN="${MAIN_DIR}/{model}_lpips_brain_2d_sliceaxis_{axis}.csv"
MODEL_DICE_PATTERN="${MAIN_DIR}/{model}_dice.csv"


BASELINE_2D_PATTERN="${BASELINE_DIR}/3T_evaluated_2d_sliceaxis_{axis}.csv"
BASELINE_3D_CSV="${BASELINE_DIR}/metrics_3T_model.csv"
BASELINE_LPIPS_PATTERN="${BASELINE_DIR}/3T_baseline_data_lpips_brain_2d_sliceaxis_{axis}.csv"
BASELINE_DICE_CSV="${BASELINE_DIR}/3T_baseline_data_dice.csv"

python "$PY_SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  --model-2d-pattern "$MODEL_2D_PATTERN" \
  --model-3d-pattern "$MODEL_3D_PATTERN" \
  --model-lpips-pattern "$MODEL_LPIPS_PATTERN" \
  --model-dice-pattern "$MODEL_DICE_PATTERN" \
  --baseline-dice-csv "$BASELINE_DICE_CSV" \
  --baseline-2d-pattern "$BASELINE_2D_PATTERN" \
  --baseline-3d-csv "$BASELINE_3D_CSV" \
  --baseline-lpips-pattern "$BASELINE_LPIPS_PATTERN"

echo "metric summary finished"
