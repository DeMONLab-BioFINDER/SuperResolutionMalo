#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 2
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 08:00:00
#SBATCH -J synthseg_batch
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

module load freesurfer/8.1.0-1
source /sw/apps/freesurfer/8.1.0-1/bianca/SetUpFreeSurfer.sh

cd /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Data_norm

echo "which:"
which mri_synthseg

echo "help:"
mri_synthseg --help

mri_synthseg --i 7T_selected --o 7T_synthseg --robust
