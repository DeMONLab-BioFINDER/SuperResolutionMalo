print("start")
from WarvitoCodes.models_unet_v2_conditioned_AdaDM_2D import UNetModel
from WarvitoCodes.wganGP import compute_gradient_penalty
from typing import Tuple,Mapping
import torch
import numpy as np
from torch import nn
import nibabel as nib
from generative.losses import PerceptualLoss,PatchAdversarialLoss
from torch.nn import L1Loss
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from medicalnet_models.models.resnet import medicalnet_resnet10_23datasets,medicalnet_resnet50_23datasets
from generative.networks.nets import PatchDiscriminator


tc = time.time()

n_trial = 2 #create a folder "results" with folders "triali" where i=n_trial to save loss curves, "params.txt"  saves the settings

print(torch.cuda.device_count(),n_trial)

RAM_opti = True #Use it if you don't have enough RAM -- calculates the losses on the cpu

retrain = False #use if want to resume a training, please indicate the path below :
if retrain:
    path_old_model = "models/my_unet.pt"

settings = {
"n_epochs" : 5,
"saved_epoch": [1,3,5],

"n_channels" : 128, #number of channels after initial convolution
"num_groups" : 32, #size of batch in the group normalization

"use_mid_attention" : False, #Self attention at the middle of the Unet, can take a lot of RAM if the image is big at the bottleneck
"n_res_block" : 3, #Number of res blocks in each stage of the U net
"channel_mult" : (1,2,2,2), # the length of this tuple indicates the number of downsample op, the value by which model_channels is multiplied
                        # at the current stage
"attention_res" : [8], #Indicate at which channel multiplication to include a cross attention layer, if empty then no cross attention. Can take a lot of RAM if used in high stages. 

"size_batch" : 8, 
"num_workers" : 4, 
"n_in_slices" : 1,  #Number of input slices/channels, default is 1.

"lr" : 5e-4,  
"scheduler" :  None, #None, "decay" or "reducePlateau" or None
"decay" : (1000,0.5), #learning rate decay
"betas" : (0.9, 0.999), #betas of the optimizer, beta1=0.9 nice for sparse gradients
"dropout" : 0, # not good for SR 

"conditions" : ["Sex","Age","Diagnostic","position"],
"block_method" : "slices",  #"slices" is the only option, need to coregister the images
"normalization_method" : "min_max", #"std_mean" or "min_max", min_max is required if percetual loss
"clip" : (1,99), #None or tuple of 2 percentiles (recommandation : (1,99)

"test_size" : 13, 
"train_size"  : 123,
"n_slices" : 244, 
"c_dim" : 1, #Dimension of the context
"slice_dim" : 1, #Dimension along which to divide the image in slices
"slice_size" : (288,222), #Image of the input and output images, before padding

"use_initial_down" : False, #use if the image is too big -- replaces initial conv by a downsampling conv, adds upsampling before the last layer
"use_perceptual_loss" : True,
"perceptual_network_type" : "radimagenet_resnet50", #"medicalnet_resnet10_23datasets" or "medicalnet_resnet50_23datasets" for 3D , radimagenet_resnet50 for 2D
"use_gan" : True, #Unused
"use_ada" : True, #Layer used to reduce the flaws of batch normalization in SR -- sharpens the edges
"perceptual_weight" : 0.005, 
"warm_up": False,

"padding" : None, #How to padd the input, if None the padding will be as small as possible
"n_background" : 0, #not used
"opti_generator" : "adam", #Can only be adam
"opti_discriminator" : None, #Used only with a GAN
"use_spatial_transformer" : True, #c_dim needs to not be none
"save_loss_ratio": 25, #save the test and train losses every save_loss_ratio step
"save_res_ratio" : 50000, #steps at which to save the output and perceptual metrics, must be a multiple of save_loss_ratio
"save_model" : True} #save the model at the end or not

settings2 = {
"n_crit": 3,
"lr" : 1e-4,
"gan_weight" : 1e-1,

"n_channels" : 96, #Number of channels after initial convolution

"num_layers_d" : 4, #Number of layers and number of downsamples
"gp_weight" : 10,
"wgp" : True
}

wgp = settings2["wgp"]
n_crit = settings2["n_crit"]

save_loss_ratio = settings["save_loss_ratio"]
save_res_ratio = settings["save_res_ratio"]
saved_epoch = settings["saved_epoch"]

border_bot = [50,61,26]
bott = border_bot[1]
border_top = [337,302,247]
topp = border_top[1]

test_padder = (26,24,60,48,50,14)
#Save the parameters
with open("resultsGAN/trial"+str(n_trial)+"/params.txt", 'w') as f:
    f.write(str(settings))

with open("resultsGAN/trial"+str(n_trial)+"/paramsGAN.txt", 'w') as f:
    f.write(str(settings2))


big_path = "/proj/nobackup/sens2023026/wharf/ma2730gi/ma2730gi-sens2023026/"

#path to the input and output images
parent_7T = big_path + "T1_7T_processed/"
parent_3T = big_path + "jake__20240405_172444___t1___brain___fs/processed_reg/"


idx_7T = "7T_015_BioFINDER_"

info = pd.read_csv(big_path+"patient_info.csv") #patient variables

n_test = settings["test_size"]
n_train = settings["train_size"]
n_tot = n_test+n_train
n_slices = settings["n_slices"]
n_bg = settings["n_background"]
n_condi = len(settings["conditions"])

if retrain:
    checkpoint = torch.load(path_old_model)
elif settings["warm_up"]:
    checkpoint = torch.load(big_path+"Unet/resultsUNC/trial1/my_unet.pt")

#Select the train and test sets
i_missing = [203,204,223,248,274,288,300]+[184,192]

idx_test = np.sort(np.random.choice(136,13,replace=False))

if retrain:
    idx_test = np.array(list(checkpoint["idx_test"]))

idx_slices_train = np.zeros((1,1))
idx_slices_test = np.zeros((1,13*n_slices))

step_test = 0
step_train = 0
i_subj = 126
patient_test_idx = []
#Selects the slices index
bots_test = []
for i in range(136):
    if i in idx_test:
        idx_slices_test[0,step_test*n_slices:(step_test+1)*n_slices] = np.arange(i*n_slices,(i+1)*n_slices)
        step_test += 1
        
        bot1 = info.loc[info["7T_idx"]==idx_7T+str(i_subj),"bot_"+str(1)].iloc[0]
        bots_test.append(max(bot1,info.loc[info["7T_idx"]==idx_7T+str(i_subj),"bot_first_slice"].iloc[0]))
        patient_test_idx.append(i_subj)
    else :
        bot1 = info.loc[info["7T_idx"]==idx_7T+str(i_subj),"bot_"+str(1)].iloc[0]
        top1 = info.loc[info["7T_idx"]==idx_7T+str(i_subj),"top_"+str(1)].iloc[0]
        
        bot1 = max(bot1,info.loc[info["7T_idx"]==idx_7T+str(i_subj),"bot_first_slice"].iloc[0])
        
        
        
        idx_slices_train_i = np.zeros((1,top1-topp+n_slices+bott-bot1))
        #idx_slices_train_i = np.zeros((1,n_slices))

        idx_slices_train_i[0] = np.arange(i*n_slices+bot1-bott,(i+1)*n_slices+top1-topp)
        #idx_slices_train_i[0] = np.arange(i*n_slices,(i+1)*n_slices)
        
        idx_slices_train = np.concatenate((idx_slices_train,idx_slices_train_i),axis=1)
        step_train += 1
        
    i_subj+=1
    while i_subj in i_missing:
        i_subj+=1

idx_slices_train = idx_slices_train[:,1:]     

idx_slices_test = idx_slices_test[0].astype(int)
idx_slices_train = idx_slices_train[0].astype(int)

print(idx_slices_test[:250],idx_slices_train[:250])

for i in idx_slices_test:
    if i in idx_slices_train:
        print("plantage")

n_d1,n_d2 = settings["slice_size"]
d_slice = settings["slice_dim"]
n_colors = settings["n_in_slices"]

#Now we get the right input, context and output slices from the npy files generated by pickling.py

contexts = np.load(big_path+"Unet/contents/contexts.npy")
contexts_test = torch.from_numpy(contexts[idx_slices_test]).float()
contexts_train = torch.from_numpy(contexts[idx_slices_train]).float()
n_slices_train = len(idx_slices_train)
print("n_slices_train :",n_slices_train)

l = []
for i in range(136):
    if not(i in idx_test):
        l.append(i)
        
idx_train = np.array(l)

ranges = np.load(big_path+"Unet/contents/ranges.npy")
ranges_test = ranges[idx_test]
ranges_train = ranges[idx_train]


print(contexts_train.shape,contexts_test.shape,contexts.shape)
del contexts
del ranges


hd_test = []
af_test = []

test_subj_list=[]

stepper = 0
#Getting the affine and header for synthetic image saving purposes
for i in range(126,305):
    info_i = info.loc[info["7T_idx"]==idx_7T+str(i)]
    if i not in i_missing and info_i["usage"].iloc[0]=="train":
        if stepper in idx_test:
            date = info_i["closest_3T"].iloc[0]
            dates = date.split("-")

            #Load the image
            id_3T = info_i["mid"].iloc[0]
            id_3T += "__"+dates[0]+dates[1]+dates[2]
            I3T = nib.load(parent_3T+id_3T+"/brain_corrected_registered.nii.gz")
            
            test_subj_list.append(i)
            hd_test.append(I3T.header)
            af_test.append(I3T.affine)
        stepper+=1


df_name_test = {"idx_test" : patient_test_idx,"patients_7T_test" : test_subj_list}
df_name_test = pd.DataFrame(data = df_name_test)
df_name_test.to_csv("resultsGAN/trial"+str(n_trial)+"/test_patients.csv",index=False)


del I3T

if n_colors==1:
    x = np.load(big_path+"Unet/contents/inputs.npy")
else :
    x = np.load(big_path+"Unet/contents/inputs"+str(n_colors)+".npy")

x_test = torch.from_numpy(x[idx_slices_test]).float()
x_train = torch.from_numpy(x[idx_slices_train]).float()

print("Shape of the dataset : ",x.shape)

print("Shape of the test dataset : ",x_test.shape)

print("Shape of the train dataset : ",x_train.shape)

del x

y = np.load(big_path+"Unet/contents/outputs.npy")
y[y>1.5] = 1.5
y_test_no_padd = torch.from_numpy(y[idx_slices_test]).float()
y_train = torch.from_numpy(y[idx_slices_train]).float()


print("Shape of the dataset : ",y.shape)

print("Shape of the test dataset : ",y_test_no_padd.shape)

print("Shape of the train dataset : ",y_train.shape)

del y

#Calculates the number of downsamples
n_down = len(settings["channel_mult"])-1+int(settings["use_initial_down"])

#The last dimensions of x,y need to be divisible by 2^(n_down) where n_d is the number of downsamples
padder = settings["padding"]

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

padder = (r2//2,r2//2+r2%2,r1//2,r1//2+r1%2)

x_train = F.pad(x_train, padder, "constant", 0)
y_train = F.pad(y_train, padder, "constant", 0)

x_test = F.pad(x_test, padder, "constant", 0)
y_test = F.pad(y_test_no_padd, padder, "constant", 0)


#Definition of the model
my_unet = UNetModel(image_size = settings["slice_size"], in_channels = n_colors, out_channels = 1, model_channels = settings["n_channels"],
                    num_res_blocks = settings["n_res_block"], attention_resolutions = settings["attention_res"], 
                    context_dim=settings["c_dim"],
                   dropout=settings["dropout"],
                    use_ada = settings["use_ada"],
                  channel_mult = settings["channel_mult"], # len = nbr down op, value by which model_channels is multipled
                  use_spatial_transformer = settings["use_spatial_transformer"],
                   use_mid_attention = settings["use_mid_attention"],
                    use_initial_downsample = settings["use_initial_down"],
                    num_groups = settings["num_groups"])


my_unet = nn.DataParallel(my_unet)


if retrain:
    my_unet.load_state_dict(checkpoint["model_state_dict"])
elif settings["warm_up"]:
    my_unet.load_state_dict(checkpoint["model_state_dict"])

my_unet.cuda()
#Discriminator
discriminator = PatchDiscriminator(spatial_dims=2, in_channels=n_colors, num_layers_d=settings2["num_layers_d"], num_channels=settings2["n_channels"])
discriminator = nn.DataParallel(discriminator)
discriminator.cuda()

#Definition of the losses and optimizers
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=2, network_type=settings["perceptual_network_type"])
if not(RAM_opti): 
    loss_perceptual.cuda()

perceptual_weight = settings["perceptual_weight"]
gan_weight = settings2["gan_weight"]
gp_weight = settings2["gp_weight"]

optimizer = torch.optim.Adam(params=my_unet.parameters(), lr=settings["lr"], betas=settings["betas"])
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=settings2["lr"],betas=(0,0.9))

if retrain:
    optimizer.load_state_dict(checkpoint["opti_state_dict"])

#learning rate decay
n_decay_steps,rate_decay = settings["decay"]
if settings["scheduler"]=="decay":
    scheduler = StepLR(optimizer,step_size = n_decay_steps,gamma=rate_decay)
    scheduler_d = StepLR(optimizer_d,step_size = n_decay_steps,gamma=rate_decay)
else:
    scheduler = ReduceLROnPlateau(optimizer, patience=5,cooldown = 3,threshold = 5e-4,mode='min')
    scheduler_d = ReduceLROnPlateau(optimizer_d, patience=15,cooldown = 5,threshold = 5e-4,mode='min')

n_epochs = settings["n_epochs"]

if retrain:
    saved_losses = checkpoint["saved_losses"]
    epoch_recon_loss_list = list(saved_losses["train_l1"])
    epoch_recon_perc_loss_list = list(saved_losses["train_perc"])
    epoch_test_loss_list = list(saved_losses["test_l1"])
    epoch_test_perc_loss_list = list(saved_losses["test_perc"])
    past_epoch = checkpoint["epoch"]

else :
    epoch_recon_loss_list = []
    epoch_recon_perc_loss_list = []
    epoch_test_loss_list = []
    epoch_test_perc_loss_list = []
    past_epoch = 0

ssim_test = []
psnr_test = []
psnr_brain_test = []
ssim_brain_test = []
med10 = []
med50 = []
psnr_no_corrupt_test = []
ssim_no_corrupt_test = []
subj = []
steps_t = []
size_batch = settings["size_batch"]


num_work = settings["num_workers"]
batch_per_worker = int(size_batch/num_work)

train_loader = DataLoader(list(zip(x_train,y_train,contexts_train)), batch_size=size_batch, shuffle=True,num_workers=num_work)
test_loader = DataLoader(list(zip(x_test,y_test,contexts_test)), batch_size=size_batch, shuffle=False,num_workers=num_work)

toutou = []
toutou2 = []
toutou3 = []
toutoud = []
toutoud2 = []
moumou = []
moumou2 = []
moumou3 = []
moumoud = []
moumoud2 = []
lrs = []
lrsd = []

del x_train
del y_train
del contexts_train
del x_test
del y_test
del contexts_test
del info

model10 = PerceptualLoss(spatial_dims=3,is_fake_3d=False, network_type="medicalnet_resnet10_23datasets")
model50 = PerceptualLoss(spatial_dims=3,is_fake_3d=False, network_type="medicalnet_resnet50_23datasets")


tic = time.time()
print("start, it took me that many seconds : ",tic-tc)
iii = 1

epoch_loss = 0
epoch_perc_loss = 0
epoch_fool_loss = 0
epoch_discr_loss = 0
epoch_discr_loss_real = 0
tot_step = n_slices_train//size_batch+float((n_slices_train%size_batch)>0)
print("n_steps per epoch : ",tot_step)
for epoch in range(n_epochs):
    
    my_unet.train()
    discriminator.train()

    toc = time.time()
    
    batches = enumerate(train_loader)
    for step,batch in batches:
        tac = time.time()
        in_img,out_img,context = batch
        
        
        sb,_,_,_ = in_img.shape
        
        #in_img = in_img.float()
        #out_img = out_img.float()
        #context = context.float()
        
        in_img = in_img.cuda()
        context = context.cuda()
        timest = torch.ones(sb,dtype=torch.float32).cuda()
        optimizer_d.zero_grad(set_to_none=True)
        
        reconstruction = my_unet(in_img,timesteps = timest,context=context)
        
        del in_img
        del context 
        
        #out_img.cuda()

        rc = reconstruction.contiguous().detach().cuda()
        oc = out_img.contiguous().detach().cuda()
        
        logits_fake = discriminator(rc)[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(oc)[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        
        discriminator_loss = loss_d_fake + loss_d_real
        dl = (loss_d_fake.item())
        dlr = (loss_d_real.item())

        if wgp:
            penalite = compute_gradient_penalty(discriminator,oc,rc)
            discriminator_loss += gp_weight*penalite
        
        del rc
        del oc
        del timest
        
        loss_d = gan_weight * discriminator_loss
        #print("discr loss",dl,loss_d_fake.item(),loss_d_real.item(),penalite.item())
        epoch_discr_loss += dl
        epoch_discr_loss_real += dlr
        
        
        toutoud.append(dl)
        toutoud2.append(dlr)
        
        loss_d.backward()
        optimizer_d.step()
        
        if not((step+1)%n_crit) or (step<50 and epoch==0):
            iii += 1
            optimizer.zero_grad(set_to_none=True)

            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            
            if RAM_opti:
                reconstruction = reconstruction.to("cpu")
            else:
                out_img.cuda()

            recons_loss = l1_loss(reconstruction, out_img)
            p_loss = loss_perceptual(reconstruction, out_img)
            
            del out_img 
            del reconstruction
            
            loss_sr = recons_loss + perceptual_weight * p_loss
        
            if RAM_opti:
                loss_sr = loss_sr.cuda()
        
            loss_sr = loss_sr + gan_weight*generator_loss
        
            rl = recons_loss.item()
            pl = p_loss.item()
            gl = generator_loss.item()
        
            loss_sr.backward()
        
            optimizer.step()

            toutou.append(rl)
            toutou2.append(pl)
            toutou3.append(gl)
            
        
            epoch_loss += rl
            epoch_perc_loss += pl
            epoch_fool_loss += gl
            
        
            if settings["scheduler"]=="decay":
                scheduler.step()
                scheduler_d.step()
        
            if not(iii%save_loss_ratio):
                moumou.append(np.sum(np.array(toutou[-save_loss_ratio:]))/save_loss_ratio)
                moumou2.append(np.sum(np.array(toutou2[-save_loss_ratio:]))/save_loss_ratio)
                moumou3.append(np.sum(np.array(toutou3[-save_loss_ratio:]))/save_loss_ratio)
                moumoud.append(np.sum(np.array(toutoud[-save_loss_ratio:]))/save_loss_ratio/n_crit)
                moumoud2.append(np.sum(np.array(toutoud2[-save_loss_ratio:]))/save_loss_ratio/n_crit)
            
                print("epoch : ",epoch+1," step : ",step," mean25 train l1 loss : ",round(moumou[-1],4)," mean25 train perc loss : ",round(moumou2[-1],4)," mean25 train fool loss : ",round(moumou3[-1],4)," mean25 train discr fake loss : ",round(moumoud[-1],4)," mean25 train discr real loss : ",round(moumoud2[-1],4))
                print("step training time",time.time()-tac)
                if settings["scheduler"]=="reducePlateau":
                    scheduler.step(moumou[-1]+perceptual_weight*moumou2[-1]+gan_weight*moumou3[-1])
                    scheduler_d.step(moumoud[-1]+moumoud2[-1])
                print("gen lr : ",scheduler.get_last_lr()[0],"discr lr : ",scheduler_d.get_last_lr()[0])
                lrs.append(scheduler.get_last_lr()[0])
                lrsd.append(scheduler_d.get_last_lr()[0])
                
            if np.isnan(rl):
                print("Nan crash, reduce the learning rate")
                exit()
    
    toutous = pd.DataFrame(data={"moumou : " : moumou, "moumou2 : " : moumou2,"moumou3 : " : moumou3, "moumoud : " : moumoud,"moumoud2 : " : moumoud2})
    toutous.to_csv("resultsGAN/trial"+str(n_trial)+"/moutous.csv",index=False)        
    
    if (epoch+1) in saved_epoch:
        print("save loss step",step)
        tooc = time.time()
    
        test_l1_loss = 0
        test_perc_loss = 0
        my_unet.eval()

        with torch.no_grad():
            print("saving image step ",step)
            saved_res = torch.zeros(n_test,n_slices,n_d1,n_d2,dtype=torch.float16)
            batches_test = enumerate(test_loader)
            location = 0
            for step_t,batch_test in batches_test:
        
                x_t,y_t,contexts_t = batch_test

                #x_t = x_t.float()
                #y_t = y_t.float()
                #contexts_t = contexts_t.float()
                
                sb,_,_,_ = x_t.shape
                
                x_t = x_t.cuda()
                contexts_t = contexts_t.cuda()

                recons_t = my_unet(x_t,timesteps = torch.ones(sb,dtype=torch.float32).cuda(),context=contexts_t)
        
        
                del x_t
                del contexts_t
        
                if RAM_opti:
                    recons_t = recons_t.to("cpu")
                else :
                    y_t = y_t.cuda()
        
                test_l1_loss += l1_loss(recons_t, y_t).item()
                test_perc_loss += loss_perceptual(recons_t, y_t).item()
        
                recons_t = recons_t.half()
        
    
        
                jj = sb
                if (location%n_slices)+sb>=n_slices:
                   jj = n_slices-location%n_slices
                   print("saving test depassing ",jj,location,sb)
            
                if r1!=0 and r2!=0:
                    saved_res[location//n_slices,((location)%n_slices):((location)%n_slices)+jj] = recons_t[:jj,0,r1//2:-(r1//2+r1%2),r2//2:-(r2//2+r2%2)]
                    if jj<sb and jj>0:
                        saved_res[location//n_slices+1,:(sb-jj)] = recons_t[jj:,0,r1//2:-(r1//2+r1%2),r2//2:-(r2//2+r2%2)]
                elif r1!=0:            
                    saved_res[location//n_slices,((location)%n_slices):((location)%n_slices)+jj] = recons_t[:jj,0,r1//2:-(r1//2+r1%2),:]
                    if jj<sb and jj>0:
                        saved_res[location//n_slices+1,:(sb-jj)] = recons_t[jj:,0,r1//2:-(r1//2+r1%2),:]                
                elif r2!=0:
                    saved_res[location//n_slices,((location)%n_slices):((location)%n_slices)+jj] = recons_t[:jj,0,:,r2//2:-(r2//2+r2%2)]
                    if jj<sb and jj>0:
                        saved_res[location//n_slices+1,:(sb-jj)] = recons_t[jj:,0,:,r2//2:-(r2//2+r2%2)]                
                else:
                    saved_res[location//n_slices,((location)%n_slices):((location)%n_slices)+jj] = recons_t[:jj,0,:,:]
                    if jj<sb and jj>0:
                        saved_res[location//n_slices+1,:(sb-jj)] = recons_t[jj:,0,:,:]
        
                location += sb

                del y_t
                del recons_t
        

            tiic = time.time()
            for i_test in range(n_test):
                res_i_test = saved_res[i_test].float()
                I_t = torch.squeeze(y_test_no_padd[i_test*n_slices:(i_test+1)*n_slices]).float()
            
                I_t2 = I_t[None,None,:,:,:]
                res_i_test2 = res_i_test[None,None,:,:,:]
                    
      
                med10.append(model10(I_t2,res_i_test2).item())
                med50.append(model50(I_t2,res_i_test2).item())
                del res_i_test2
                del I_t2      
            
                shifting = bots_test[i_test]-bott
                I_t_no_corruption = I_t[shifting:]
                res_i_test_shifted = res_i_test[shifting:]
            
                f_true = I_t.flatten()
                f_gen = res_i_test.flatten()
                f_gen = f_gen[f_true>0]
                f_true = f_true[f_true>0]
            
                      
                range_data = ranges_test[i_test]
            
                psnr_test.append(peak_signal_noise_ratio(I_t.numpy(),res_i_test.numpy(),data_range=range_data))
                ssim_test.append(structural_similarity(I_t.numpy(),res_i_test.numpy(),data_range=range_data))
            
                psnr_brain_test.append(peak_signal_noise_ratio(f_true.numpy(),f_gen.numpy(),data_range=range_data))
                ssim_brain_test.append(structural_similarity(f_true.numpy(),f_gen.numpy(),data_range=range_data))
            
                f_true = I_t_no_corruption.flatten()
                f_gen = res_i_test_shifted.flatten()
                f_gen = f_gen[f_true>0]
                f_true = f_true[f_true>0]
            
            
                range_data = (torch.max(f_true)).item()
            
                psnr_no_corrupt_test.append(peak_signal_noise_ratio(f_true.numpy(),f_gen.numpy(),data_range=range_data))
                ssim_no_corrupt_test.append(structural_similarity(f_true.numpy(),f_gen.numpy(),data_range=range_data))
            
                subj.append(df_name_test["patients_7T_test"][i_test])
                steps_t.append(step+1)
            
                df_test = {"SSIM" : ssim_test,"PSNR" : psnr_test,"SSIM_brain" : ssim_brain_test,"PSNR_brain" : psnr_brain_test,
                    "PSNR_no_corruption" :psnr_no_corrupt_test,"SSIM_no_corruption" : ssim_no_corrupt_test,
                    "medicalnet_10" : med10,"medical_50" : med50,
                    "subject": subj,"n step":steps_t}
                df_test = pd.DataFrame(data=df_test)
                df_test.to_csv("resultsGAN/trial"+str(n_trial)+"/test_metrics.csv",index=False)
                    
                res_i_test = torch.transpose(res_i_test,0,1)
                res_i_test = F.pad(res_i_test, test_padder, "constant", 0)
                nib.save(nib.Nifti1Image(res_i_test, af_test[i_test], header=hd_test[i_test]),"resultsGAN/trial"+str(n_trial)+"/images/"+str(df_name_test["patients_7T_test"][i_test])+"_7T_step_"+str(iii)+".nii.gz")

            ssim_test.append(np.mean(np.array(ssim_test)[-(i_test+1):]))
            psnr_test.append(np.mean(np.array(psnr_test)[-(i_test+1):]))
            ssim_brain_test.append(np.mean(np.array(ssim_brain_test)[-(i_test+1):]))
            psnr_brain_test.append(np.mean(np.array(psnr_brain_test)[-(i_test+1):]))
            psnr_no_corrupt_test.append(np.mean(np.array(psnr_no_corrupt_test)[-(i_test+1):]))
            ssim_no_corrupt_test.append(np.mean(np.array(ssim_no_corrupt_test)[-(i_test+1):]))
            med10.append(np.mean(np.array(med10)[-(i_test+1):]))
            med50.append(np.mean(np.array(med50)[-(i_test+1):]))
            subj.append(0)
            steps_t.append((step+1)*(epoch+1))
        
            df_test = {"SSIM" : ssim_test,"PSNR" : psnr_test,"SSIM_brain" : ssim_brain_test,"PSNR_brain" : psnr_brain_test,
                    "PSNR_no_corruption" :psnr_no_corrupt_test,"SSIM_no_corruption" : ssim_no_corrupt_test,
                    "medicalnet_10" : med10,"medical_50" : med50,
                    "subject": subj,"n step":steps_t}
            df_test = pd.DataFrame(data=df_test)
            df_test.to_csv("resultsGAN/trial"+str(n_trial)+"/test_metrics.csv",index=False)
                
            print("perc metric time : ",time.time()-tiic)
            print("testing time :",time.time()-tooc)
               
            del saved_res
            del I_t
            del I_t_no_corruption
            del res_i_test
            del res_i_test_shifted
            del f_true 
            del f_gen
        
        epoch_test_loss_list.append(test_l1_loss/(step_t+1))
        epoch_test_perc_loss_list.append(test_perc_loss/(step_t+1))

        epoch_recon_loss_list.append((epoch_loss/(step+1)))
        epoch_recon_perc_loss_list.append((epoch_perc_loss/(step+1)))

        df = {"train_l1" : epoch_recon_loss_list, "test_l1" : epoch_test_loss_list,"train_perc" : epoch_recon_perc_loss_list,"test_perc" : epoch_test_perc_loss_list}
        df = pd.DataFrame(data=df)
        df.to_csv("resultsGAN/trial"+str(n_trial)+"/losses.csv",index=False)

        print("time : ", time.time()-toc," nb steps : ",past_epoch*n_slices_train+iii*n_crit,"/",(n_epochs+past_epoch)*tot_step, "  train L1 loss : ", round((epoch_loss*n_crit/(step+1)),4), " train perc loss : ", round((epoch_perc_loss*n_crit/(step+1)),4),"  test L1 loss : ", round(test_l1_loss/(step_t+1),4), " test perc loss : ", round(test_perc_loss/(step_t+1),4))

        my_unet.train()

    epoch_loss = 0
    epoch_perc_loss = 0
    epoch_fool_loss = 0
    epoch_discr_loss = 0
    epoch_discr_loss_real = 0
        
        
      

with open("resultsGAN/trial"+str(n_trial)+"/training_time.txt", 'w') as f:
    f.write(str(time.time()-tic))

if settings["save_model"]:
    torch.save({"epoch" : past_epoch+epoch+1,"model_state_dict" : my_unet.state_dict(),"opti_state_dict" : optimizer.state_dict(),"saved_losses" : df,"idx_test" : idx_test}, "resultsGAN/trial"+str(n_trial)+"/my_unet.pt")


  
#plt.plot(save_loss_ratio*np.array(range(1,len(epoch_recon_loss_list)+1)),epoch_recon_loss_list) 
#plt.title("Training L1 loss, trial number "+str(n_trial))
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.savefig("resultsGAN/trial"+str(n_trial)+"/Training_L1.png")
#plt.close()


#plt.plot(save_loss_ratio*np.array(range(1,len(epoch_recon_loss_list)+1)),epoch_recon_perc_loss_list)
#plt.title("Training perceptual loss, trial number "+str(n_trial))
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.savefig("resultsGAN/trial"+str(n_trial)+"/Training_perc.png")
#plt.close()


#plt.plot(save_loss_ratio*np.array(range(1,len(epoch_recon_loss_list)+1)),epoch_test_loss_list)
#plt.title("Testing L1 loss, trial number "+str(n_trial))
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.savefig("resultsGAN/trial"+str(n_trial)+"/Test_L1.png")
#plt.close()


#plt.plot(save_loss_ratio*np.array(range(1,len(epoch_recon_loss_list)+1)),epoch_test_perc_loss_list)
#plt.title("Testing perceptual loss, trial number "+str(n_trial))
#plt.xlabel("step")
#plt.ylabel("loss")
#plt.savefig("resultsGAN/trial"+str(n_trial)+"/Test_perc.png")
#plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(toutou)+1)),toutou) 
plt.title("Training L1 loss at each batch, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/toutou.png")
plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(toutou2)+1)),toutou2) 
plt.title("Training perceptual loss at each batch, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/toutou2.png")
plt.close()

toutous = pd.DataFrame(data={"toutou : " : toutou, "toutou2 : " : toutou2})
toutous.to_csv("resultsGAN/trial"+str(n_trial)+"/toutous.csv",index=False)

plt.plot(save_loss_ratio*np.array(range(1,len(moumou)+1)),moumou) 
plt.title("Mean training L1 loss over 25 batches, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/moutou.png")
plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(moumou2)+1)),moumou2) 
plt.title("Mean training perceptual loss over 25 batches, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/moutou2.png")
plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(moumou3)+1)),moumou3)
plt.title("Mean training fooling loss over 25 batches, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/moutou3.png")
plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(moumoud)+1)),moumoud,label="fake") 
plt.plot(save_loss_ratio*np.array(range(1,len(moumoud)+1)),moumoud2,label="real")
plt.title("Mean training discriminator losses over 25 batches, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("resultsGAN/trial"+str(n_trial)+"/moutoud.png")
plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(lrs)+1)),-np.log(np.array(lrs))) 
plt.title("Learning rate generator, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("-log(lr)")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/lrs.png")
plt.close()

plt.plot(save_loss_ratio*np.array(range(1,len(lrsd)+1)),-np.log10(np.array(lrsd))) 
plt.title("Learning rate discriminator, trial number "+str(n_trial))
plt.xlabel("step")
plt.ylabel("-log(lr)")
plt.savefig("resultsGAN/trial"+str(n_trial)+"/lrs_discr.png")
plt.close()




print("done python")

#torch.cuda.empty_cache()
