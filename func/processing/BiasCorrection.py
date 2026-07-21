import ants
import sys
import time
import os




def correc(path,path_mask,path_out):
    shrink_fact = 2

    raw_I = ants.image_read(path)

    
    mask = ants.get_mask(raw_I)
    
    corrected = ants.n4_bias_field_correction(image=raw_I, mask=mask,
    shrink_factor=int(shrink_fact),
    convergence={'iters': [150,150,150,150], 'tol': 1e-7})
  
    ants.image_write(corrected, path_out)
    print(path, " corrected ")
    




if __name__=="__main__":
    globals()[sys.argv[1]](sys.argv[2],sys.argv[3],sys.argv[4])