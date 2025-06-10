import nibabel as nib
import time
import sys
import os
import json
import numpy as np
import pandas as pd

def loca(im):
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





def lims(usage,datas):
    
    with open('params.json', 'r') as file:
        params_meth = json.load(file)
            
    info = pd.read_csv(params_meth[usage]).to_dict("list")
    info["bot_0"] = [] 
    info["top_0"] = [] 
    info["bot_1"] = [] 
    info["top_1"] = [] 
    info["bot_2"] = [] 
    info["top_2"] = [] 
    info["corruption"] = []

    path = params_meth[datas]+"processed/7T/"

    paths = os.listdir(path)
    
    
    i, ii, j, jj, k, kk = 10000,0,10000,0,10000,0
    for p in paths:

        if p[-7:]==".nii.gz":
            I = nib.load(path+p).get_fdata()
            i2, ii2, j2, jj2, k2, kk2 = loca(I)
            
            info["bot_0"].append(i2)
            info["top_0"].append(ii2)
            info["bot_1"].append(j2) 
            info["top_1"].append(jj2)
            info["bot_2"].append(k2)
            info["top_2"].append(kk2)
            info["corruption"].append(0)
            
            if i>i2:
                i = i2
            if ii<ii2:
                ii = ii2
            if j>j2:
                j = j2
            if jj<jj2:
                jj = jj2
            if k>k2:
                k = k2
            if kk<kk2:
               kk = kk2
               
    path = params_meth["path_inference"]+"processed/7T/"

    paths = os.listdir(path)
    i, ii, j, jj, k, kk = 10000,0,10000,0,10000,0
    for p in paths:
 
     if p[-7:]==".nii.gz":
         I = nib.load(path+p).get_fdata()
         i2, ii2, j2, jj2, k2, kk2 = loca(I)
         
         info["bot_0"].append(i2)
         info["top_0"].append(ii2)
         info["bot_1"].append(j2) 
         info["top_1"].append(jj2)
         info["bot_2"].append(k2)
         info["top_2"].append(kk2)
         info["corruption"].append(0)
         
         if i>i2:
             i = i2
         if ii<ii2:
             ii = ii2
         if j>j2:
             j = j2
         if jj<jj2:
             jj = jj2
         if k>k2:
             k = k2
         if kk<kk2:
            kk = kk2
   
            
    
    a,b,c = I.shape
    
    data = {"i": i, "ii": ii, "j": j, "jj": jj, "k": k, "kk": kk,
    "a": a, "b": b, "c": c }
    print(info)    
    df = pd.DataFrame(info)
    df.to_csv(params_meth[usage],index=False)

    with open(params_meth[datas]+"lims.json", 'w') as f:
        json.dump(data, f)






if __name__=="__main__":
	globals()[sys.argv[1]](sys.argv[2],sys.argv[3])