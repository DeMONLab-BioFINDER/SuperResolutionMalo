import ants
import time
import sys
import os
import json




def reg(path,path_out):
    tic = time.time()
    with open('params.json', 'r') as file:
        params_meth = json.load(file)
        
    I3T = ants.image_read(path)
    path2 = path.replace("/3T/","/7T/")
    if not(os.path.isfile(path2)):
        path_ref="models/template_3T_to_7T.mat"
        path_ref_3T="models/template_3T.nii.gz"
        path_ref_7T="models/template_7T.nii.gz"
        
        template_I3T = ants.image_read(path_ref_3T)
        reg = ants.registration(fixed=template_I3T,moving=I3T,type_of_transform="Rigid")

        I7T = ants.image_read(path_ref_7T)
        I_moved = reg["warpedmovout"]
        I_moved = ants.apply_transforms(fixed=I7T,moving=I_moved,transformlist=[path_ref])
    else:
        I7T = ants.image_read(path2)
        reg = ants.registration(fixed=I7T,moving=I3T)
        I_moved = reg["warpedmovout"]
    print("success !")
    
    
    ants.image_write(I_moved,path_out)
    
    print(time.time()-tic)




if __name__=="__main__":
	globals()[sys.argv[1]](sys.argv[2],sys.argv[3])