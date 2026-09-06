#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -J mergeNpy
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out
#SBATCH --mem=64G

# Activate Python environment
echo "Activating environment..."
source /proj/nobackup/sens2023026/data/ants_set_up/venv-ants/bin/activate

# Paths
INPUT_DIR="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/contents_2" #change
OUTPUT_DIR="/home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/contents_2_merged" #change

# Run
echo "Merging batches from $INPUT_DIR ..."
python merge_batches.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"

echo "merge_baches.py end"
