import numpy as np
import nibabel as nib
import pandas as pd
import time


settings = {
"n_in_slices" : 3,  #Number of slices in the input, default is 1.
"conditions" : ["Sex","Age","Diagnostic","position"], 
"block_method" : "slices",  #"slices" is the only option, need to coregister the images
"normalization_method" : "min_max", #"std_mean" or "min_max"
"clip" : (1,99), #None or tuple of 2 percentiles (recommandation : (1,99)

"test_size" : 14, 
"train_size"  : 124,
"n_slices" : 244, 
"c_dim" : 1, #Dimension of the context
"slice_dim" : 1, #Dimension along which to divide the image in slices
"slice_size" : (288,222), #Image of the input and output images, before padding

"padding" : None, #Your own padder, of None the padding will be the smallest possible
"n_background" : 12}


#Indexes that indicate which parts of the images are allways background
border_bot = [50,61,26]
border_top = [337,302,247]

big_path = "/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/"

#path to the input and output images
parent_7T = big_path + "T1_7T_processed/"
parent_3T = big_path + "jake__20240405_172444___t1___brain___fs/processed_reg/"


idx_7T = "7T_015_BioFINDER_"

info = pd.read_csv(big_path+"patient_info.csv")

n_test = settings["test_size"]
n_train = settings["train_size"]
n_tot = n_test+n_train
n_slices = settings["n_slices"]
n_bg = settings["n_background"]
n_condi = len(settings["conditions"])

i_missing = [203,204,223,248,274,288,300]

n_d1,n_d2 = settings["slice_size"]
d_slice = settings["slice_dim"]
n_colors = settings["n_in_slices"]

stepper = 0
i_train = 0
print("loading context")
contexts = np.zeros((n_slices*n_tot,n_condi,1),dtype=np.float16)

bot = border_bot[1]-1
top = border_top[1]+1

stepper = 0
for i in range(126,126):
    if (i not in i_missing) and info.loc[info["7T_idx"]==idx_7T+str(i),"usage"].iloc[0]=="train":
        bot1 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(1)].iloc[0]
        top1 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(1)].iloc[0]
        mid = (bot1+top1)/2
        
        
        contexts[stepper*n_slices:(stepper+1)*n_slices,0,0] = np.ones(n_slices)*float(info.loc[info["7T_idx"]==idx_7T+str(i),"age"].iloc[0])
        contexts[stepper*n_slices:(stepper+1)*n_slices,1,0] = np.ones(n_slices)*float(info.loc[info["7T_idx"]==idx_7T+str(i),"gender_baseline_variable"].iloc[0])
        contexts[stepper*n_slices:(stepper+1)*n_slices,2,0] = np.ones(n_slices)*int((info.loc[info["7T_idx"]==idx_7T+str(i),"cognitive_status_baseline_variable"].iloc[0] in ["SCD","Normal"]))
        contexts[stepper*n_slices:(stepper+1)*n_slices,3,0] = 2*(np.arange(0,352)-mid)[bot:top+1]/n_slices

        stepper += 1

contexts = contexts.astype(np.float16)
print(type(contexts[0,0,0]))
#np.save(big_path+"Unet/contents/contexts.npy",contexts,allow_pickle=True)

del contexts


x = np.zeros((n_tot*(n_slices),n_colors,n_d1,n_d2),dtype=np.float16)
y = np.zeros((n_tot*n_slices,1,n_d1,n_d2),dtype=np.float16)
ranges = np.zeros(n_tot,dtype=np.float16)

stepper = 0
print("loading the normalized images")
for i in range(126,305):
    print(i)
    info_i = info.loc[info["7T_idx"]==idx_7T+str(i)]
    if i not in i_missing and info_i["usage"].iloc[0]=="train":
        
        bot0 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(0)].iloc[0]
        bot1 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(1)].iloc[0]
        bot2 = info.loc[info["7T_idx"]==idx_7T+str(i),"bot_"+str(2)].iloc[0]
        
        top0 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(0)].iloc[0]
        top1 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(1)].iloc[0]
        top2 = info.loc[info["7T_idx"]==idx_7T+str(i),"top_"+str(2)].iloc[0]
        print(bot0,bot1,bot2,top0,top1,top2)
        
        date = info_i["closest_3T"].iloc[0]
        dates = date.split("-")
        
        #Load the image
        id_3T = info_i["mid"].iloc[0]
        id_3T += "__"+dates[0]+dates[1]+dates[2]
        tic = time.time()
        Img3T = nib.load(parent_3T+id_3T+"/brain_corrected_registered.nii.gz").get_fdata()
        Img7T = nib.load(parent_7T+str(i)+"_t1_brain3.nii.gz").get_fdata()
        
        Img7T = np.multiply(Img7T,(Img3T!=0))
        
        
        
        print(np.sum(Img7T)-np.sum(Img7T_small),np.sum(Img3T)-np.sum(Img3T_small))
        
        Img3T = Img3T[border_bot[0]:border_top[0]+1,bot-n_colors//2:top+1+n_colors//2,border_bot[2]:border_top[2]+1]
        Img7T = Img7T[border_bot[0]:border_top[0]+1,bot:top+1,border_bot[2]:border_top[2]+1]

        

        #print("mult time : ",time.time()-tiic)
        
        
        
        toc = time.time()
        print("loading image time : ",toc-tic)
        
        #Removing extreme values, very useful with min_max normalization
        if settings["clip"] is not None:
            cmin,cmax = settings["clip"]        
            
            v3T = Img3T.flatten()
            v3T = v3T[v3T>0]
            
            v7T = Img7T.flatten()
            v7T = v7T[v7T>0]
            
            pmax3T = np.percentile(v3T, cmax)
            pmax7T = np.percentile(v7T, cmax)
            
            v3T = np.clip(v3T,0,pmax3T)
            v7T = np.clip(v7T,0,pmax7T)

        else :
            v3T = Img3T_small.copy()
            v7T = Img7T_small.copy()
    
        if settings["normalization_method"]=="min_max":
            vmin3T = 0
            vmax3T = np.max(v3T)
            vmin7T = 0
            vmax7T = np.max(v7T)
            Img_n_3T = (Img3T-vmin3T)/(vmax3T-vmin3T)
            Img_n_3T = Img_n_3T.astype(np.float16)
            
            Img_n_7T = (Img7T-vmin7T)/(vmax7T-vmin7T)
            Img_n_7T = Img_n_7T.astype(np.float16)
            
        elif settings["normalization_method"]=="std_mean":
            vmean3T = np.mean(v3T)
            vstd3T = np.std(v3T)
            vmean7T = np.mean(v7T)
            vstd7T = np.std(v7T)
            Img_n_3T = (Img3T-vmean3T)/vstd3T
            
            Img_n_3T = Img_n_3T.astype(np.float16)

            Img_n_7T = (Img7T-vmean7T)/vstd7T
            Img_n_7T = Img_n_7T.astype(np.float16)
            
            
        tac = time.time()
        print("normalization time : ",tac-toc)
        #Storing of the normalized images, with the n_in_slices-1 neighbor slices and slicing in the dimension d_slice
        ranges[stepper] = np.max(Img_n_7T)-np.min(Img_n_7T)

        if d_slice==0:
            for j in range(n_slices):
                x[n_slices*stepper+j,:] = Img_n_3T[(j):(j+n_colors)]
            y[n_slices*stepper:n_slices*(stepper+1),0] = Img_n_7T

        elif d_slice==1:
            for j in range(n_slices):
                print(j,stepper)
                x[n_slices*stepper+j,:] = np.transpose(Img_n_3T[:,(j):(j+n_colors)],axes=[1,0,2])
            y[n_slices*stepper:n_slices*(stepper+1),0] = np.transpose(Img_n_7T,axes=[1,0,2])

        else :
            for j in range(n_slices):
                x[n_slices*stepper+j,:] = np.transpose(Img_n_3T[:,:,(j):(j+n_colors)],axes=[2,0,1])
            y[n_slices*stepper:n_slices*(stepper+1),0] = np.transpose(Img_n_7T,axes=[2,0,1])

        
        print("slicing time : ",time.time()-tac)
        print("total loading time : ",time.time()-tic)      
        stepper += 1
        
np.save(big_path+"Unet/contents/ranges.npy",ranges,allow_pickle=True)
np.save(big_path+"Unet/contents/inputs3.npy",x,allow_pickle=True)
np.save(big_path+"Unet/contents/outputs.npy",y,allow_pickle=True)
