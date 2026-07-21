# UNetSuperResolution
Super resolution U-Net that were used to go from 3T to 7T brain MRI. 
Two exemple models are included, a UNet and a UNet GAN conditionned on the age and sex of the participants : my_unet_no_diag.pt and my_unetGanNoDiag.pt.

# Settings 
Change your paths and parameters in "params.json". We included an exemple along with a second one "params_inference.json", which must be renamed to "params.json" to be used.
Change the model parameters at the beginning of the file func/training/LoadingModel.py
The parameters are :
"test_size" : n_test (number of files to infere the model on) 
"train_size" : n_train (number of files used for training)
n_test+n_train should be equal to the total number of input 3T images

"slice_dim": d (0,1 or 2. Default 1, dimension along which the 2D slicing occurs)
"n_neighboors": 2k+1, where k is an integer Default 3, number of input slices. 
"d1":256,"d2":256,"d3":256  (input dimensions, should be larger than the size of your largest brain)
"path_data": "data/DATASETNAME/" (path to your dataset, see Data section below)
"path_patient_info": "data/DATASETNAME/participants.csv" (path to your csv, see Data section below)
"path_inference_model":"models/my_unetGanNoDiag.pt" (path to your inference model file)
"path_inference_model_params":"models/paramsGanNoDiag.txt" (path to your inference model parameters file)
 "infere_mode" (used iff "infere_mode" = "True")
 "batch_size_inference" (int, size of the inference batch, reduce if cuda out of memory)


# conda env 
To create the venv do :

conda create --name SR_env python=3.10.8

conda activate SR_env

pip install numpy antspyx matplotlib einops nibabel lpips monai torch scikit-image pandas gdown 


# Data
Create a data/DATASETNAME folder, with the subfolders data/DATASETNAME/raw/3T and data/DATASETNAME/raw/7T (not required for inference). Matching images should have the same name and be in a .nii.gz format.
You all need a csv file with "ID" "Age"	"Sex" as keys
If your images are already processed, then use "process" instead of "raw"

# Preprocessing
A preprocessing pipeline is included as an exemple, you need a venv with ANTs and freesurfer v7.3.0 or later.

To use it, use processing_pipeline.bash and processing_subpipeline.bash. Some debugging might be needed. 

It includes, skull stripping, bias field correction and registration (non-linear for training, to a template for inference).

# Training  and inference
Download "https://github.com/Project-MONAI/GenerativeModels/tree/main/" and put the folder "generative" in the folder func/

First, run "source scirpts/processing_pipeline.sh" to process the data

For training: 

	- If you your computer does not have access to the internet, you must download the models "medicalnet_resnet10_23datasets","medicalnet_resnet50_23datasets" and "radimagenet_resnet50" yourself.

	- Create a folder results/trialX/images and indicate which number X is in func/training/LoadingModel.py in the variable n_trial

	- Then run "source scirpts/lauching_training.sh" to do the training


You can then infere using "source scirpts/inference_pipeline.sh". You need to indicate the path to the model and its parameters in params.json, in "path_inference_model" and "path_inference_model_params". Default : our UNet GAN.
The included .pt files are large files and require careful downloading.
The model was only trained on one 3T scanner and is not meant to work on a wide spectrum of images.
Infered results will be put under data/DATASETNAME/processed/infered

# Acknoledgements
The functions in func/WarvitoCodes are a modified version of code found on https://huggingface.co/spaces/Warvito/diffusion_brain/tree/main

The code for the WGAN-GP comes from https://github.com/eriklindernoren/Keras-GAN

