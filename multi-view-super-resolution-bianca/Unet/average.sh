#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -J avg_unet
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

# Activate specific environment
echo "Activating ANTs environment..."
source /proj/nobackup/sens2023026/data/ants_set_up/venv-ants/bin/activate

# Verify environment
echo "Current Python path:"; which python
python --version

# Safety settings
set -euo pipefail

# Paths (edit only these)
BIG_PATH="/proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/"
TO_SEG="${BIG_PATH}to_seg/"



DIR_AXIS0="${TO_SEG}unet/axis0_on_UnetModel0/"
DIR_AXIS1="${TO_SEG}unet/axis1_on_UnetModel1/"
DIR_AXIS2="${TO_SEG}unet/axis2_on_UnetModel2/"
OUT_DIR="${TO_SEG}unet/average_UnetModel/"



# Run
echo "Running average_batch.py..."
python average.py \
  --dir-axis0 "$DIR_AXIS0" \
  --dir-axis1 "$DIR_AXIS1" \
  --dir-axis2 "$DIR_AXIS2" \
  --out-dir "$OUT_DIR" \
  --strict-affine

echo "done"



