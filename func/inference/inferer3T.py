print("start")

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
toc = time.time()
from WarvitoCodes.models_unet_v2_conditioned_AdaDM_2D import UNetModel
from typing import Tuple,Mapping
import torch
import numpy as np
from torch import nn
import nibabel as nib
import torch.nn.functional as F
import os
import pandas as pd
from torch.utils.data import DataLoader
import json
import ast

from datetime import datetime




def brain_location(im):
    a,b,c = im.shape
    padd = (im>0).astype(int)
    
    i = 0
    i_d = False
    ii = a-1
    ii_d = False
    j = 0
    j_d = False
    jj = b-1
    jj_d = False
    k = 0
    k_d = False
    kk = c-1
    kk_d = False

    while not(i_d) and i<ii:
        i_d = (np.max(padd[i]) != 0)
        i += 1

    while not(ii_d) and i<ii:
        ii_d = (np.max(padd[ii]) != 0)
        ii -= 1

    while not(j_d) and j<jj:
        j_d = (np.max(padd[:, j]) != 0)
        j += 1

    while not(jj_d) and j<jj:
        jj_d = (np.max(padd[:, jj]) != 0)
        jj -= 1

    while not(k_d) and k<kk:
        k_d = (np.max(padd[:, :, k]) != 0)
        k += 1

    while not(kk_d) and k<kk:
        kk_d = (np.max(padd[:, :, kk]) != 0)
        kk -= 1

    return i-1, ii+1, j-1, jj+1, k-1, kk+1


def InfSubj(path_model,path_settings,ref_out):
    batch_size = 12
    
    with open(path_settings) as f:
        a = f.read().replace("\'","\"")
        settings = ast.literal_eval(a)

    #path to the input and output images
    with open('params.json', 'r') as file:
        params_meth = json.load(file)
    
    info =  pd.read_csv(params_meth["path_inference_info"])
    ds = [int(params_meth["d1"]), int(params_meth["d2"]), int(params_meth["d3"])]

    d_slice = params_meth["slice_dim"]

    n_slices = ds[d_slice]
    n_colors = settings["n_in_slices"]
    
    
    if params_meth["slice_dim"]==0:
        n_d1,n_d2 = ds[1],ds[2]
    elif params_meth["slice_dim"]==1:
        n_d1,n_d2 = ds[0],ds[2]
    else:
        n_d1,n_d2 = ds[0],ds[1]
    
    
    checkpoint = torch.load(path_model,weights_only=False)
    my_unet = UNetModel(image_size = (n_d1,n_d2), in_channels = n_colors, out_channels = 1, model_channels = settings["n_channels"],
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


    
    #Calculates the number of downsamples
    n_down = len(settings["channel_mult"])-1+int(settings["use_initial_down"])

    #The last dimensions of x,y need to be divisible by 2^(n_down) where n_d is the number of downsamples
    padder = None

    
    path = params_meth["path_inference"]+"processed/7T/"
    paths = os.listdir(path)
    for subj in paths:
        id_p = subj.split("/")[-1][:6]        
    
        print(subj)
        
        info_i = info.loc[info["ID"]==id_p].iloc[0]
        age = info_i["Age"]
        info_sex = info_i["Sex"]
        
        im = nib.load(path+subj)

        hd = im.header
        af = im.affine
        im = im.get_fdata()
        (i,ii,j,jj,k,kk) = brain_location(im)
        if d_slice==1:    
            mid = (jj+j)/2
        elif d_slice==0:
            mid = (ii+i)/2
        else:
            mid = (kk+k)/2
        
        print("brain loca ",brain_location(im),mid)
        
        
        if d_slice==1:
            im = np.transpose(im,(1,0,2))
        elif d_slice==2:
            im = np.transpose(im,(2,0,1))
        
        im = np.concatenate((im[:-2,None,:,:],im[1:-1,None,:,:],im[2:,None,:,:]),axis=1)
        print(im.shape)
        num_inputs,_,_,_ =  im.shape
        x = torch.from_numpy(im)

        del im
        
        if d_slice==0:
            n_d1,n_d2 = ds[1],ds[2]
        elif d_slice==1:
            n_d1,n_d2 = ds[0],ds[2]
        else:
            n_d1,n_d2 = ds[0],ds[1]
            
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
        
        padder = (r2//2,r2//2+r2%2,r1//2,r1//2+r1%2)
        x = F.pad(x, padder, "constant", 0)
        print(x.shape)

        sex = int(info_sex=="F")

        ress = torch.zeros(num_inputs,n_d1,n_d2)
        contexts_i = torch.ones(batch_size,3,1)
        contexts_i[:,0,0] = age
        contexts_i[:,1,0] = sex
        

        tps = torch.ones(batch_size).cuda()
        
        
        with torch.no_grad():
            for jjj in range(num_inputs//batch_size):
                contexts_i[:,2,0] = 2*(torch.arange(jjj*batch_size,(jjj+1)*batch_size)-mid)/n_slices
                contexts_i = contexts_i.cuda()
                x_i = x[jjj*batch_size:(jjj+1)*batch_size].float().cuda()
                res_i = my_unet(x_i,timesteps = tps,context = contexts_i)
                print(jjj)
                
                if r1!=0 and r2!=0:
                    ress[jjj*batch_size:(jjj+1)*batch_size] = res_i[:,0,r1//2:-(r1//2+r1%2),r2//2:-(r2//2+r2%2)]                    
                elif r1!=0:            
                    ress[jjj*batch_size:(jjj+1)*batch_size] = res_i[:,0,r1//2:-(r1//2+r1%2),:]                 
                elif r2!=0:
                    ress[jjj*batch_size:(jjj+1)*batch_size] = res_i[:,0,:,r2//2:-(r2//2+r2%2)]            
                else:
                    ress[jjj*batch_size:(jjj+1)*batch_size] = res_i[:,0,:,:]                    
                
            if (num_inputs%batch_size)!=0:
                contexts_i2 = torch.ones(num_inputs%batch_size,3,1)
                contexts_i2[:,0,0] = age
                contexts_i2[:,1,0] = sex
                contexts_i2[:,2,0] = 2*(torch.arange((num_inputs//batch_size)*batch_size,num_inputs)-mid)/n_slices
                contexts_i2 = contexts_i2.cuda()
                
                x_i = x[(num_inputs//batch_size)*batch_size:].float().cuda()
                
                tps2 = torch.ones(num_inputs%batch_size).cuda()
                print(x_i.shape,contexts_i2.shape,tps2.shape)
                res_i = my_unet(x_i,timesteps = tps2,context = contexts_i2)
                
                if r1!=0 and r2!=0:
                    ress[(num_inputs//batch_size)*batch_size:] = res_i[:,0,r1//2:-(r1//2+r1%2),r2//2:-(r2//2+r2%2)]                    
                elif r1!=0:            
                    ress[(num_inputs//batch_size)*batch_size:] = res_i[:,0,r1//2:-(r1//2+r1%2),:]                 
                elif r2!=0:
                    ress[(num_inputs//batch_size)*batch_size:] = res_i[:,0,:,r2//2:-(r2//2+r2%2)]            
                else:
                    ress[(num_inputs//batch_size)*batch_size:] = res_i[:,0,:,:]   
                        
        if d_slice==1:
            ress = torch.transpose(ress,0,1)
        elif d_slice==2:
            ress = torch.transpose(torch.transpose(ress,0,1),1,2)


        nib.save(nib.Nifti1Image(ress, af, header=hd),(path+subj.replace(".nii.gz",ref_out+".nii.gz")).replace("/3T/","/infered/"))
        print(time.time()-toc)




if __name__=="__main__":
    globals()[sys.argv[1]](sys.argv[2],sys.argv[3],sys.argv[4])
