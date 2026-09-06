#!/bin/bash -l
#SBATCH -N 1 --exclusive
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH -t 24:00:00
#SBATCH -J GANGPU

module load Anaconda/2023.09-0-hpc1-bdist
echo ${SLURM_JOB_NAME}
cd /proj/berzelius-2024-156/users/x_rushe/super-resolution/Unet
nvidia-smi
conda activate sr_env
echo "done env"
# cd ../Unet/
python3 LoadingUnet_no_diag_0.py
echo "done"

