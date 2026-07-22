# UNetSuperResolution

A super-resolution U-Net used to go from 3T to 7T brain MRI, from our paper [Converting T1-weighted MRI from 3T to 7T quality using deep learning](https://arxiv.org/abs/2507.13782)

Two example models are included along their corresponding parameter file:
- `models/my_unet_no_diag.pt` — a plain U-Net (associated with params_no_diag.txt)
- `models/my_unetGanNoDiag.pt` — a U-Net trained using generative adversarial network (associated with paramsGanNoDiag.txt)

We also include our 3T and 7T templates, built on 30 subjects and used in the inference pipeline:
- `models/template_7T.nii.gz` — 7T template
- `models/template_3T.nii.gz` — 3T template
- `models/template_3T_reged.nii.gz` — 3T template registered to the 7T template using an Affine transform
- `models/template_3T_to_7T.mat` — Affine transform used to register the 3T template to the 7T template

They are used in func/processing/registration.py and func/inference/register_to_7T_template.py


## Settings

Change your paths and parameters in `params.json`. Two examples are included:
- `params.json` — training example
- `params_inference.json` — inference example, must be renamed to `params.json` to be used

Model parameters are set at the top of `func/training/LoadingModel.py`:

| Parameter | Description |
|---|---|
| `test_size` | Number of files to run inference on (`n_test`) |
| `train_size` | Number of files used for training (`n_train`). `n_test + n_train` should equal the total number of input 3T images |
| `slice_dim` | Dimension along which 2D slicing occurs (0, 1, or 2). Default: 1 |
| `n_neighboors` | Number of input slices, `2k+1` where `k` is an integer. Default: 3 |
| `d1`, `d2`, `d3` | Input dimensions, e.g. 256, 256, 256 — should be larger than your largest 7T brain |
| `path_data` | Path to your dataset, e.g. `data/DATASETNAME/` (see [Data](#data)) |
| `path_patient_info` | Path to your CSV, e.g. `data/DATASETNAME/participants.csv` (see [Data](#data)) |
| `path_inference_model` | Path to your inference model file, e.g. `models/my_unetGanNoDiag.pt` |
| `path_inference_model_params` | Path to your inference model parameters file, e.g. `models/paramsGanNoDiag.txt` |
| `infere_mode` | Whether to run in inference mode |
| `batch_size_inference` | Inference batch size (int) — reduce if you get a CUDA out-of-memory error |

## Conda environment

```bash
conda create --name SR_env python=3.10.8
conda activate SR_env
pip install numpy antspyx matplotlib einops nibabel lpips monai torch scikit-image pandas gdown
```

## Data

Create a `data/DATASETNAME/` folder with subfolders:
- `data/DATASETNAME/raw/3T`
- `data/DATASETNAME/raw/7T` (not required for inference)

Matching images should share the same filename and be in `.nii.gz` format.

You'll also need a CSV file with the columns `ID`, `Age`, `Sex`

If your images are already processed, use `process` instead of `raw` in the folder path.

## Preprocessing

An example preprocessing pipeline is included. You'll need a virtual environment with **ANTs** and **FreeSurfer v7.3.0** or later.

Run it via `script/processing_pipeline.sh` and `script/processing_subpipeline.sh`. Some debugging may be needed depending on your setup.

The pipeline includes: skull stripping, bias field correction, a second skull stripping and registration (non-linear for training; to a 7T template for inference).

## Training and inference

1. Download the [generative](https://github.com/Project-MONAI/GenerativeModels/tree/main/) folder from Project-MONAI/GenerativeModels and place it in `func/`.
2. Run `source script/processing_pipeline.sh` to process the data.

**Training:**
- If your computer doesn't have internet access, manually download the models `medicalnet_resnet10_23datasets`, `medicalnet_resnet50_23datasets`, and `radimagenet_resnet50`.
- Run `source script/lauching_training.sh $i` to start training
- Results will be saved into the folder `results/trial$i/`

**Inference:**
- Run `source script/inference_pipeline.sh`. Set the model and parameter file paths in `params.json` under `path_inference_model` and `path_inference_model_params` (default: our U-Net GAN).
- Note: the included `.pt` files are large and require careful downloading.
- The model was trained on a single 3T scanner and is not intended to generalize across a wide range of scanners/images.

Inference results are saved to `data/DATASETNAME/processed/infered`.

## Acknowledgements

- Functions in `func/WarvitoCodes` are a modified version of code from [Warvito/diffusion_brain](https://huggingface.co/spaces/Warvito/diffusion_brain/tree/main).
- The WGAN-GP code comes from [eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN).