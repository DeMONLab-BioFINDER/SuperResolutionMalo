#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 2
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -J synthseg_batch_3T
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

module load freesurfer/8.1.0-1
source /sw/apps/freesurfer/8.1.0-1/bianca/SetUpFreeSurfer.sh

cd /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg

echo "which:"
which mri_synthseg

echo "help:"
mri_synthseg --help

mri_synthseg --i 3T_baseline_data --o 3T_baseline_data_synthseg --robust
