# Super Resolution

This repository contains the code used for the multi-view super-resolution pipeline.

The project is divided into three anatomical axes:

- `0`: coronal
- `1`: axial
- `2`: sagittal

One current limitation is that several output folders need to be created manually. Please create the folders listed below first. If any additional folders are reported as missing during execution, create them manually and rerun the script.

All input and output paths are specified in the corresponding `.sh` files.

---

## Data

- 7T ground truth: `T1_7T_processed/`
- 3T ground truth: `jake__20240405_172444___t1___brain___fs/processed_reg/`
- 3T test set and inference results from all models: `to_seg/`
- Patient information: `patient_info2.csv`

Other intermediate data are described below together with the corresponding code.

---

# Bianca

## Slicing

Scripts:

```text
Unet/slicing.py
Unet/slicing.sh
```

Outputs:

```text
Unet/contents_0
Unet/contents_1
Unet/contents_2
```

## Merge batches

Scripts:

```text
Unet/merge_batches.py
Unet/merge_batches.sh
```

Outputs:

```text
Unet/contexts_0_merged
Unet/contexts_1_merged
Unet/contexts_2_merged
```

Manually copy:

```text
patient_info_anon0
patient_info_anon1
patient_info_anon2
```

to the corresponding merged folders.

---

# Berzelius

The top-level folder is also:

```text
super-resolution
```

Transfer the following folders from Bianca:

```text
contexts_0_merged
contexts_1_merged
contexts_2_merged
```

## Train single-view models

Scripts:

```text
Loading_Unet_no_diag_0.py
Loading_Unet_no_diag_1.py
Loading_Unet_no_diag_2.py
GPU_submit_Unet.sh
```

`GPU_submit_Unet.sh` needs to be manually modified to select which axis to train.

Inputs:

```text
contents_ano_0
contents_ano_1
contents_ano_2
```

The generated `images` are only for testing whether the code runs correctly and have no practical use. They can be deleted after generation.

Outputs:

```text
resultsUnet/trial0
resultsUnet/trial1
resultsUnet/trial2
```

## Train CNN fusion model

Scripts:

```text
Unet/fusion_CNN_unet_1.py
Unet/fusion_CNN_unet.sh
```

Inputs:

```text
contents_ano_0
contents_ano_1
contents_ano_2
resultsUnet/trial0
resultsUnet/trial1
resultsUnet/trial2
```

Output:

```text
fusion_results
```

---

# Back to Bianca

Transfer the trained single-view models from Berzelius:

```text
resultsUnet/trial0
resultsUnet/trial1
resultsUnet/trial2
```

to:

```text
Unet/results/unet_test12/trial0
Unet/results/unet_test12/trial1
Unet/results/unet_test12/trial2
```

Transfer the CNN model:

```text
fusion_results
```

to:

```text
to_seg/fusion_results/test7
```

---

## Inference

### Single-view models

Scripts:

```text
Unet/inferer2.py
Unet/inferer_test12.sh
Unet/inferer_test13.sh
Unet/inferer_test14.sh
```

Input:

```text
Unet/results/unet_test12
```

Output:

```text
to_seg/test12_full/axis0,1,2_on_UnetModel0,1,2
```

### Average model

Scripts:

```text
Unet/average.py
Unet/average.sh
```

Output:

```text
to_seg/test12_full/average_UnetModel
```

### CNN fusion model

Scripts:

```text
Unet/inferer_CNN.py
Unet/inferer_CNN.sh
```

Output:

```text
to_seg/test12_full/fusionCNN
```

---

## 3D Evaluation

For the three single-view models, 3D evaluation is handled directly by their own inferer.

For the average model:

```text
Unet/evaluation.py
Unet/evaluation_average.sh
```

Rename the output file to:

```text
metrics_average_UnetModel.csv
```

For the CNN fusion model, 3D evaluation is handled by its own inferer.

All CSV files are output to:

```text
to_seg/test12_full/
```

---

## 2D Evaluation

Scripts:

```text
Unet/evaluation_2d.py
Unet/evaluation_LPIPS.py
```

Related `.sh` files are under:

```text
Unet/evaluation_code/
```

All CSV files are output to:

```text
to_seg/test12_full/
```

---

## 3T Baseline Evaluation

### 2D

Scripts:

```text
Unet/evaluation_2d.py
Unet/evaluation_2d_3T_0.sh
Unet/evaluation_2d_3T_1.sh
Unet/evaluation_2d_3T_2.sh
```

### 3D

SSIM and PSNR:

```text
Unet/evaluation.py
Unet/evaluation_3T.sh
```

`Unet/evaluation_3T.sh` may have been deleted and could not be found.

LPIPS:

```text
Unet/evaluation_LPIPS.py
Unet/evaluation_code/evaluation_LPIPS_3T_0.sh
Unet/evaluation_code/evaluation_LPIPS_3T_1.sh
Unet/evaluation_code/evaluation_LPIPS_3T_2.sh
```

All CSV files are output to:

```text
to_seg/3T_baseline/
```

---

## Plotting

Scripts:

```text
Unet/evaluation_code/summary1.py
Unet/evaluation_code/summary.sh
```

Output:

```text
to_seg/test12_full/evaluation_results
```


## Contributor

Ruizhen Shen  
Developed the multi-view super-resolution framework for the Master's thesis project, building on the original super-resolution framework developed by Malo Gicquel.