print("start")
from WarvitoCodes.models_unet_v2_conditioned_AdaDM_2D import UNetModel
from typing import Tuple,Mapping
import torch
import numpy as np
from torch import nn
import nibabel as nib
from generative.losses import PerceptualLoss
from torch.nn import L1Loss

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from medicalnet_models.models.resnet import medicalnet_resnet10_23datasets,medicalnet_resnet50_23datasets
import json
import ast
import sys


path_model = "models/my_unet_no_diag.pt"
path_settings = "models/params_no_diag.txt"

with open(path_settings) as f:
    #print(f.read().replace("\'","\""))
    a = f.read().replace("\'","\"")
    settings = ast.literal_eval(a)
print(settings)

big_path = "/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/"

#path to the input and output images
parent_3T = big_path + "jake__20240405_172444___t1___brain___fs/processed_reg/"
parent_7T = big_path + "T1_7T_processed/"
generated_7T = big_path + "to_seg/UNet_no_diag/"


info = pd.read_csv(big_path+"patient_info2.csv")

n_slices = settings["n_slices"]
n_condi = len(settings["conditions"])
n_colors = settings["n_in_slices"]
checkpoint = torch.load(path_model)

#Calculates the number of downsamples
n_down = len(settings["channel_mult"])-1+int(settings["use_initial_down"])

#The last dimensions of x,y need to be divisible by 2^(n_down) where n_d is the number of downsamples
padder = None

border_bot = [50,61,26]
border_top = [337,302,247]
test_padder = (26,24,60,48,50,14)

n_d1 = 352
n_d2 = 272
#Automatic padding
if padder is None :
    if n_d1%(2**n_down):
        r1 = 2**n_down - n_d1%(2**n_down)
    else :
        r1 = 0
    if n_d2%(2**n_down):
        r2 = 2**n_down - n_d2%(2**n_down)
    else :
        r2 = 0
print(r1,r2)

my_unet = UNetModel(image_size = settings["slice_size"], in_channels = n_colors, out_channels = 1, model_channels = settings["n_channels"],
                           num_res_blocks = settings["n_res_block"], attention_resolutions = settings["attention_res"], 
                            context_dim=settings["c_dim"],
                          dropout=0.,  # Not great for SR
                            use_ada = settings["use_ada"],
                          channel_mult = settings["channel_mult"], # len = nbr down op, value by which model_channels is multipled
                          use_spatial_transformer = settings["use_spatial_transformer"],
                           use_mid_attention = settings["use_mid_attention"],
                            use_initial_downsample = settings["use_initial_down"],
                            num_groups = settings["num_groups"])


my_unet = nn.DataParallel(my_unet)

my_unet.load_state_dict(checkpoint["model_state_dict"])
my_unet.cuda()
my_unet.eval()


df = {"psnr":[],"ssim":[],"psnr_brain":[],"ssim_brain":[],"psnr_no_corrupt":[],"ssim_no_corrupt":[],"perc_10":[],"perc_50":[],"patient":[]}

i_missing = [203,204,223,248,274,288,300]+[184,192]
idx_7T = "7T_015_BioFINDER_"

for i in range(126,305):
    info_i = info.loc[info["7T_idx"]==idx_7T+str(i)]
    
    if i not in i_missing and info_i["usage"].iloc[0]=="test":
        print(i)
        bot0 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(0)].iloc[0]
        bot1 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(1)].iloc[0]
        bot2 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(2)].iloc[0]
        
        top0 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(0)].iloc[0]
        top1 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(1)].iloc[0]
        top2 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(2)].iloc[0]
        mid = (bot1+top1)/2
        
        date = info_i["closest_3T"].iloc[0]
        dates = date.split("-")
        
        #Load the image
        id_3T = info_i["mid"].iloc[0]
        id_3T += "__"+dates[0]+dates[1]+dates[2]
                
        im = nib.load(big_path+"Data_norm/3T/"+str(i)+"_normalized.nii.gz")
        im2 = nib.load(big_path+"Data_norm/7T/"+str(i)+"_normalized.nii.gz").get_fdata()
        
        
        
        hd = im.header
        af = im.affine
        im = im.get_fdata()

        #v3T = im.flatten()
        #v3T = v3T[v3T>0]
        
        #v7T = Img7T.flatten()
        #v7T = v7T[v7T>0]
        
        #pmax3T = np.percentile(v3T, cmax)
        #pmax7T = np.percentile(v7T, cmax)
        
        #v3T = np.clip(v3T,0,pmax3T)
        #v7T = np.clip(v7T,0,pmax7T)
        #im = (im-vmin3T)/(vmax3T-vmin3T)

        #pmin3T = np.percentile(im7, 1)
        #pmax3T = np.percentile(im7, 99)
        #v3T = np.clip(im7,pmin3T,pmax3T)

        #vmin3T = np.min(v3T)
        #vmax3T = np.max(v3T)
        #im2 = (im2-vmin3T)/(vmax3T-vmin3T)

        n_tt = 352
        im = np.transpose(im,(1,0,2))

        x = torch.from_numpy(im)
        
        del im

        padder = (r2//2,r2//2+r2%2,r1//2,r1//2+r1%2)
        x = F.pad(x, padder, "constant", 0)

        
        age = float(info.loc[info["7T_idx"]==idx_7T+str(i),"age"].iloc[0])
        sex = float(info.loc[info["7T_idx"]==idx_7T+str(i),"gender_baseline_variable"].iloc[0])

        ress = torch.zeros(352,352,272)
        tps = torch.ones(1).cuda()
        contexts_i = torch.ones(1,3,1) #torch.ones(1,4,1)
        contexts_i[0,0,0] = age
        contexts_i[0,1,0] = sex
        #contexts_i[0,2,0] = diag
        
        with torch.no_grad():
            for j in range(n_colors//2,n_tt-(n_colors//2)):
                contexts_i[0,2,0] = torch.ones(1)*2*(j-mid)/n_slices
                #contexts_i[0,3,0] = torch.ones(1)*2*(j-mid)/n_slices
                contexts_i = contexts_i.cuda()
                print(contexts_i)
                x_i = x[None,j-n_colors//2:j+n_colors//2+1].float().cuda()
                print(x_i.shape)
                res_i = my_unet(x_i,timesteps = tps,context = contexts_i)
                ress[:,j,:] = res_i[0,0]
                

        model10 = PerceptualLoss(spatial_dims=3,is_fake_3d=False, network_type="medicalnet_resnet10_23datasets")
        model50 = PerceptualLoss(spatial_dims=3,is_fake_3d=False, network_type="medicalnet_resnet50_23datasets")

        df["perc_10"].append(model10(ress[None,None,:,:,:].float(),torch.from_numpy(im2)[None,None,:,:,:].float()).item())
        df["perc_50"].append(model50(ress[None,None,:,:,:].float(),torch.from_numpy(im2)[None,None,:,:,:].float()).item())
        
        del model10
        del model50
        
        ress = ress.numpy()

        df["psnr"].append(peak_signal_noise_ratio(im2,ress,data_range=1))
        df["ssim"].append(structural_similarity(im2,ress,data_range=1))
        df["patient"].append(i)        
        
        f_true = im2.flatten()
        f_gen = ress.flatten()
        f_gen = f_gen[f_true>0]
        print(len(f_gen)/len(f_true))
        f_true = f_true[f_true>0]

        df["psnr_brain"].append(peak_signal_noise_ratio(f_true,f_gen,data_range=1))
        df["ssim_brain"].append(structural_similarity(f_true,f_gen,data_range=1))
        
        shifting = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_first_slice"].iloc[0]
        
        ress_s = ress[:,shifting:,:]
        im2_s = im2[:,shifting:,:]
        
        f_true = im2_s.flatten()
        f_gen = ress_s.flatten()
        f_gen = f_gen[f_true>0]
        f_true = f_true[f_true>0]
    
        df["psnr_no_corrupt"].append(peak_signal_noise_ratio(f_true,f_gen,data_range=1))
        df["ssim_no_corrupt"].append(structural_similarity(f_true,f_gen,data_range=1))

        nib.save(nib.Nifti1Image(ress, af, header=hd),generated_7T+str(i)+"_t1_generated.nii.gz")

df["psnr"].append(np.mean(np.array(df["psnr"])))
df["ssim"].append(np.mean(np.array(df["ssim"])))

df["psnr_brain"].append(np.mean(np.array(df["psnr_brain"])))
df["ssim_brain"].append(np.mean(np.array(df["ssim_brain"])))

df["perc_10"].append(np.mean(np.array(df["perc_10"])))
df["perc_50"].append(np.mean(np.array(df["perc_50"])))

df["psnr_no_corrupt"].append(np.mean(np.array(df["psnr_no_corrupt"])))
df["ssim_no_corrupt"].append(np.mean(np.array(df["ssim_no_corrupt"])))

df["patient"].append(0)


df = pd.DataFrame(data=df)
df.to_csv("/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/to_seg/metricsUnet_n_diag.csv")




