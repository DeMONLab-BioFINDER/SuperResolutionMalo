print("start")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import nibabel as nib
from generative.losses import PerceptualLoss
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import json
import os

with open('params.json', 'r') as file:
    params_meth = json.load(file)

df = {"psnr":[],"ssim":[],"psnr_brain":[],"ssim_brain":[],"psnr_no_corrupt":[],"ssim_no_corrupt":[],"perc_10":[],"perc_50":[],"patient":[]}

model10 = PerceptualLoss(spatial_dims=3,is_fake_3d=False, network_type="medicalnet_resnet10_23datasets")
model50 = PerceptualLoss(spatial_dims=3,is_fake_3d=False, network_type="medicalnet_resnet50_23datasets")


path = params_meth["path_inference"]+"processed/infered/"

paths = os.listdir(path)
info = pd.read_csv(params_meth["path_inference_info"])

for patient in paths:
    id_p = patient.split("/")[-1][:6]
    
    if patient[-7:]==".nii.gz":
        
          
        im = nib.load(path+patient).get_fdata()
        im2 =  nib.load(path+patient.replace("/infered/","/7T/")).get_fdata()
        
        v3T = im2.flatten()
        v3T = v3T[v3T>0]
        
        pmax3T = np.percentile(v3T, 99)
        pmin3T = np.percentile(v3T, 1)
        
        v3T = np.clip(im2,pmin3T,pmax3T)
    
        vmin3T = np.min(v3T)
        vmax3T = np.max(v3T)
        im2 = (im2-vmin3T)/(vmax3T-vmin3T)
    
        df["perc_10"].append(model10(torch.from_numpy(im)[None,None,:,:,:].float(),torch.from_numpy(im2)[None,None,:,:,:].float()).item())
        df["perc_50"].append(model50(torch.from_numpy(im)[None,None,:,:,:].float(),torch.from_numpy(im2)[None,None,:,:,:].float()).item())
        
        df["psnr"].append(peak_signal_noise_ratio(im2,im,data_range=1))
        df["ssim"].append(structural_similarity(im2,im,data_range=1))
        df["patient"].append(id_p)
        
        f_true = im2.flatten()
        f_gen = im.flatten()
        f_gen = f_gen[f_true>0]
        print(len(f_gen)/len(f_true))
        f_true = f_true[f_true>0]

        df["psnr_brain"].append(peak_signal_noise_ratio(f_true,f_gen,data_range=1))
        df["ssim_brain"].append(structural_similarity(f_true,f_gen,data_range=1))
        
        shifting = info.loc[info["ID"]==id_p,"corruption"].iloc[0]
        
        ress_s = im[:,shifting:,:]
        im2_s = im2[:,shifting:,:]
        
        f_true = im2_s.flatten()
        f_gen = ress_s.flatten()
        f_gen = f_gen[f_true>0]
        f_true = f_true[f_true>0]
    
        df["psnr_no_corrupt"].append(peak_signal_noise_ratio(f_true,f_gen,data_range=1))
        df["ssim_no_corrupt"].append(structural_similarity(f_true,f_gen,data_range=1))


df["psnr"].append(np.mean(np.array(df["psnr"])))
df["ssim"].append(np.mean(np.array(df["ssim"])))

df["psnr_brain"].append(np.mean(np.array(df["psnr_brain"])))
df["ssim_brain"].append(np.mean(np.array(df["ssim_brain"])))

df["perc_10"].append(np.mean(np.array(df["perc_10"])))
df["perc_50"].append(np.mean(np.array(df["perc_50"])))

df["psnr_no_corrupt"].append(np.mean(np.array(df["psnr_no_corrupt"])))
df["ssim_no_corrupt"].append(np.mean(np.array(df["ssim_no_corrupt"])))

df["patient"].append("Means")


df = pd.DataFrame(data=df)
df.to_csv(path+"/comparisons.csv")
