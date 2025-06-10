# UNetSuperResolution
Super resolution U-Net that were used to go from 3T to 7T brain MRI. 

# Settings 
Change your paths and parameters in "params.json"

# conda env 
To create the venv do :

conda create --name SR_env python=3.10.8

conda activate SR_env

pip install numpy matplotlib einops nibabel lpips monai torch scikit-image pandas gdown 


# Preprocessing
A preprocessing pipeline is included as an exemple, you need a venv with ANTs and freesurfer v7.3.0 or later.
To use it, uncomment it in processing_pipeline.bash. Some debugging might be needed, as it is just an exemple. 
It includes, skull stripping, bias field correction and non linear registration.

# Training 
Download "https://github.com/Project-MONAI/GenerativeModels/tree/main/" and put the folder "generative" in the folder func/
If you your computer does not have access to the internet, you must download the models "medicalnet_resnet10_23datasets","medicalnet_resnet50_23datasets" and "radimagenet_resnet50" yourself.
Create a folder results/trialX/images and indicate which number X is in func/training/LoadingModel.py

# FYI
The functions in func/WarvitoCodes are a modified version of code found on https://huggingface.co/spaces/Warvito/diffusion_brain/tree/main

