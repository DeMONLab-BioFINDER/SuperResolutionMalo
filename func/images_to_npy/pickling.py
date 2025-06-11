import numpy as np
import nibabel as nib
import pandas as pd
import json
import os 
import sys
settings = {
"n_in_slices" : 3,  #Number of slices in the input, default is 1.
"conditions" : ["Sex","Age","Diagnostic","position"], 
"normalization_method" : "min_max", #"std_mean" or "min_max"
"clip" : (1,99), #None or tuple of 2 percentiles (recommandation : (1,99)

"c_dim" : 1, #Dimension of the context
}


with open('params.json', 'r') as file:
    params_meth = json.load(file)

n_test = int(params_meth["test_size"])
n_train = int(params_meth["train_size"])

ds = [int(params_meth["d1"]), int(params_meth["d2"]), int(params_meth["d3"])]



"""
with open(params_meth["path_data"]+"lims.json", 'r') as file:
    data = json.load(file)
    i = data["i"]
    ii = data["ii"]
    j = data["j"]
    jj = data["jj"]
    k = data["k"]
    kk = data["kk"]
    a = data["a"]
    b = data["b"]
    c = data["c"]
"""    


big_path = params_meth[sys.argv[1]]+"processed/"

#path to the input and output images
parent_7T = big_path + "7T/"
parent_3T = big_path + "3T/"

info = pd.read_csv(params_meth["path_patient_info"])
print(info)
print(params_meth["path_patient_info"])

n_tot = n_test+n_train


n_condi = len(settings["conditions"])

if params_meth["slice_dim"]==0:
    n_d1,n_d2 = ds[1],ds[2]
elif params_meth["slice_dim"]==1:
    n_d1,n_d2 = ds[0],ds[2]
else:
    n_d1,n_d2 = ds[0],ds[1]


d_slice = params_meth["slice_dim"]
n_colors = settings["n_in_slices"]

n_slices = ds[d_slice]

stepper = 0
i_train = 0
print("loading context")
contexts = np.zeros((n_slices*n_tot,n_condi,1),dtype=np.float16)

stepper = 0

path = params_meth[sys.argv[1]]+"processed/7T/"

paths = os.listdir(path)

for patient in paths:
    id_p = patient.split("/")[-1][:6]
    bot1 = info.loc[info["ID"]==id_p,"bot_"+str(d_slice)].iloc[0]
    top1 = info.loc[info["ID"]==id_p,"top_"+str(d_slice)].iloc[0]
    mid = (bot1+top1)/2

    contexts[stepper*n_slices:(stepper+1)*n_slices,0,0] = np.ones(n_slices)*float(info.loc[info["ID"]==id_p,"Age"].iloc[0])
    contexts[stepper*n_slices:(stepper+1)*n_slices,1,0] = np.ones(n_slices)*float(info.loc[info["ID"]==id_p,"Sex"].iloc[0]=="F")

    if "cognitive_status_baseline_variable" in info.keys():
        contexts[stepper*n_slices:(stepper+1)*n_slices,2,0] = np.ones(n_slices)*int((info.loc[info["ID"]==id_p,"cognitive_status_baseline_variable"].iloc[0] in ["SCD","Normal"]))
    else:
        contexts[stepper*n_slices:(stepper+1)*n_slices,2,0] = np.ones(n_slices)*0

    contexts[stepper*n_slices:(stepper+1)*n_slices,3,0] = 2*(np.arange(0,n_slices)-mid)/n_slices

    stepper += 1

contexts = contexts.astype(np.float16)
print(type(contexts[0,0,0]))
np.save(params_meth[sys.argv[1]]+"npy_files/contexts.npy",contexts,allow_pickle=True)

del contexts


x = np.zeros((n_tot*(n_slices),n_colors,n_d1,n_d2),dtype=np.float16)
y = np.zeros((n_tot*n_slices,1,n_d1,n_d2),dtype=np.float16)

stepper = 0
print("loading the normalized images")
for patient in paths:
    info_i = info.loc[info["ID"]==id_p]

    #Load the image
    Img3T = nib.load(path+patient).get_fdata()
    Img7T = nib.load((path+patient).replace("/3T/","/7T/")).get_fdata()

    Img7T = np.multiply(Img7T,(Img3T!=0))
    
    
    #Removing extreme values, very useful with min_max normalization
    if settings["clip"] is not None:
        cmin,cmax = settings["clip"]        
        
        v3T = Img3T.flatten()
        v3T = v3T[v3T>0]
        
        v7T = Img7T.flatten()
        v7T = v7T[v7T>0]
        
        pmin3T = np.percentile(v3T, cmin)
        pmin7T = np.percentile(v7T, cmin)
        
        pmax3T = np.percentile(v3T, cmax)
        pmax7T = np.percentile(v7T, cmax)
        
        v3T = np.clip(v3T,pmin3T,pmax3T)
        v7T = np.clip(v7T,pmin7T,pmax7T)

    else :
        
        v3T = Img3T.copy()
        v7T = Img7T.copy()

    vmin3T = np.min(v3T)
    vmax3T = np.max(v3T)
    vmin7T = np.min(v7T)
    vmax7T = np.max(v7T)

    Img_n_3T = (Img3T-vmin3T)/(vmax3T-vmin3T)
    Img_n_3T = Img_n_3T.astype(np.float16)
    
    Img_n_7T = (Img7T-vmin7T)/(vmax7T-vmin7T)
    Img_n_7T = Img_n_7T.astype(np.float16)


    #Storing of the normalized images, with the n_in_slices-1 neighbor slices and slicing in the dimension d_slice
    if d_slice==0:
        for j in range(n_slices-n_colors//2-1):
            x[n_slices*stepper+j,:] = Img_n_3T[(j):(j+n_colors)]
        y[n_slices*stepper:n_slices*(stepper+1),0] = Img_n_7T

    elif d_slice==1:
        for j in range(n_slices-n_colors//2-1):
            #print(j,stepper,n_colors,n_slices)
            x[n_slices*stepper+j,:] = np.transpose(Img_n_3T[:,(j):(j+n_colors)],axes=[1,0,2])
        y[n_slices*stepper:n_slices*(stepper+1),0] = np.transpose(Img_n_7T,axes=[1,0,2])

    else :	
        for j in range(n_slices-n_colors//2-1):
            x[n_slices*stepper+j,:] = np.transpose(Img_n_3T[:,:,(j):(j+n_colors)],axes=[2,0,1])
        y[n_slices*stepper:n_slices*(stepper+1),0] = np.transpose(Img_n_7T,axes=[2,0,1])

    stepper += 1


np.save(params_meth[sys.argv[1]]+"npy_files/inputs"+str(n_colors)+".npy",x,allow_pickle=True)
np.save(params_meth[sys.argv[1]]+"npy_files/outputs.npy",y,allow_pickle=True)
