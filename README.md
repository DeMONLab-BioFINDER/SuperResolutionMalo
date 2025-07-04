# UNetSuperResolution
Super resolution U-Net that were used to go from 3T to 7T brain MRI. 

# Settings 
Change your paths and parameters in "params.json"
Change the models parameters at the beginning of the file func/training/LoadingModel.py

# conda env 
To create the venv do :

conda create --name SR_env python=3.10.8

conda activate SR_env

pip install numpy matplotlib einops nibabel lpips monai torch scikit-image pandas gdown 


# Data
Create a data/<Name of your dataset> folder, with the subfolders data/<Name of your dataset>/3T and data/<Name of your dataset>/7T


# Preprocessing
A preprocessing pipeline is included as an exemple, you need a venv with ANTs and freesurfer v7.3.0 or later.

To use it, uncomment it in processing_pipeline.bash. Some debugging might be needed, as it is just an exemple. 

It includes, skull stripping, bias field correction and non linear registration.

# Training 
Download "https://github.com/Project-MONAI/GenerativeModels/tree/main/" and put the folder "generative" in the folder func/

If you your computer does not have access to the internet, you must download the models "medicalnet_resnet10_23datasets","medicalnet_resnet50_23datasets" and "radimagenet_resnet50" yourself.

Create a folder results/trialX/images and indicate which number X is in func/training/LoadingModel.py

First, run "source scirpts/processing_pipeline.sh" to process the data

Then run "source scirpts/lauching_training.sh" to do the training

You can the infere using "source scirpts/inference_pipeline.sh", if you have unprocessed 3T images that do not have an associated 7T image, uncomment the script and include the path of the reference affine registration and associated reference image.

# FYI
The functions in func/WarvitoCodes are a modified version of code found on https://huggingface.co/spaces/Warvito/diffusion_brain/tree/main

The code for the WGAN-GP comes from https://github.com/eriklindernoren/Keras-GAN

