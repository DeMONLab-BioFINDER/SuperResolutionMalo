#!/bin/bash -l

#SBATCH -A sens2023026
#SBATCH -p core
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -t 12:00:00
#SBATCH -J fusion_infer_eval
#SBATCH -o /home/rzshen/Desktop/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet/logs/%x.%j.out

echo "Activating environment..."

unset PYTHONPATH
export PATH="/sw/apps/conda/latest/rackham_stage/bin:$PATH"
export CONDA_ENVS_PATH=/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/envs

cd /proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026
nvidia-smi || true
source envs/venv310-cuda/bin/activate

cd /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/Unet

set -euo pipefail

python inferer_CNN.py \
  --big-path /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution \
  --axis0-dir /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full/axis0_on_UnetModel0 \
  --axis1-dir /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full/axis1_on_UnetModel1 \
  --axis2-dir /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full/axis2_on_UnetModel2 \
  --axis0-pattern "{patient}_t1_generated__axis0_on_UnetModel0.nii.gz" \
  --axis1-pattern "{patient}_t1_generated__axis1_on_UnetModel1.nii.gz" \
  --axis2-pattern "{patient}_t1_generated__axis2_on_UnetModel2.nii.gz" \
  --fusion-ckpt /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/fusion_results/unet_fusion/fusion/fusion_last.pt \
  --output-dir /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full/fusionCNN \
  --metrics-csv /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/to_seg/test12_full/metrics_fusionCNN.csv \
  --patient-csv /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/patient_info2.csv \
  --parent-7t-norm /proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/super-resolution/Data_norm/7T \
  --eval-axis 0 \
  --use-gpu-perceptual



