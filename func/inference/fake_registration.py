import ants
import sys




def reg(path,path_out,path_ref_t,path_ref_im):
    
    I3T = ants.image_read(path)
    I7T = ants.image_read(path_ref_im)
    I_moved = ants.apply_transforms(fixed=I7T,moving=I3T,transformlist=[path_ref_t])
    
    ants.image_write(I_moved,path_out)





if __name__=="__main__":
	globals()[sys.argv[1]](sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])