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
    I7T = ants.image_read(path2)
    reg = ants.registration(fixed=I7T,moving=I3T)
    
    print("success !")
    I_moved = reg["warpedmovout"]
    
    ants.image_write(I_moved,path_out)
    
    print(time.time()-tic)




if __name__=="__main__":
	globals()[sys.argv[1]](sys.argv[2],sys.argv[3])