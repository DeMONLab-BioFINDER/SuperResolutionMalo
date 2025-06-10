import SimpleITK as sitk



def correc7T(path,path_mask,path_out):
    shrink_fact = 2

    raw_I = sitk.ReadImage(path,sitk.sitkFloat32)
    
    mask = sitk.ReadImage(path_mask,sitk.sitkFloat32) #sitk.LiThreshold(raw_I,30,15000)
    
    input_I = sitk.Shrink(raw_I,[shrink_fact]*raw_I.GetDimension())
    mask = sitk.Shrink(mask,[shrink_fact]*raw_I.GetDimension())
    
    bias_corr = sitk.N4BiasFieldCorrectionImageFilter()
    #bias_corr.SetConvergenceThreshold(1e-7)
    #bias_corr.SetMaximumNumberOfIterations([150,150,150,150])
    #bias_corr.SetBiasFieldFullWidthAtHalfMaximum(0.18)
    #bias_corr.SetWienerFilterNoise(0.2)
    
    I_c = bias_corr.Execute(input_I,mask)
    
    log_bias = bias_corr.GetLogBiasFieldAsImage(raw_I)	
    corrected_I = raw_I / sitk.Exp(log_bias)
    path_o = path.split(".")
    path_o = path_o[0]+"_corrected.nii.gz"
    sitk.WriteImage(corrected_I,path_o)
    print(path, " corrected ")

def correc7T(path,path_mask,path_out):
    shrink_fact = 2

    raw_I = sitk.ReadImage(path,sitk.sitkFloat32)
    
    mask = sitk.LiThreshold(raw_I,30,15000)
    
    input_I = sitk.Shrink(raw_I,[shrink_fact]*raw_I.GetDimension())
    mask = sitk.Shrink(mask,[shrink_fact]*raw_I.GetDimension())
    
    bias_corr = sitk.N4BiasFieldCorrectionImageFilter()
    #bias_corr.SetConvergenceThreshold(1e-7)
    #bias_corr.SetMaximumNumberOfIterations([150,150,150,150])
    #bias_corr.SetBiasFieldFullWidthAtHalfMaximum(0.18)
    #bias_corr.SetWienerFilterNoise(0.2)
    
    I_c = bias_corr.Execute(input_I,mask)
    
    log_bias = bias_corr.GetLogBiasFieldAsImage(raw_I)	
    corrected_I = raw_I / sitk.Exp(log_bias)
    path_o = path.split(".")
    path_o = path_o[0]+"_corrected.nii.gz"
    sitk.WriteImage(corrected_I,path_o)
    print(path, " corrected ")



if __name__=="__main__":
    globals()[sys.argv[1]](sys.argv[2],sys.argv[3])