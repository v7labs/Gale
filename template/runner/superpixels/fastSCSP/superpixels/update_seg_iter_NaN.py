#!/usr/bin/env python
"""
Authors:
Oren Freifeld, freifeld@csail.mit.edu
Yixin Li, Email: liyixin@mit.edu
"""

import numpy as np
from of.utils import *
from of.gpu.KernelThinWrapper import KernelThinWrapper

from gpu import dirname_of_cuda_files
cuda_filename = os.path.join(dirname_of_cuda_files,'update_seg_NaN.cu')


FilesDirs.raise_if_file_does_not_exist(cuda_filename)
with open(cuda_filename,'r') as cuda_file:
    _gpu_kernel = cuda_file.read()

include_dirs=[dirname_of_cuda_files]


class _UpdateSegIter(KernelThinWrapper):    
    def __init__(self):
        super(type(self),self).__init__(gpu_kernel=_gpu_kernel,
        include_dirs=include_dirs)
        self._get_function_from_src_module('find_border_pixels')
        self._get_function_from_src_module('update_seg_subset')
    def __call__(self,img_gpu, img_isNaN_gpu, seg_gpu, border_gpu,
                 counts_gpu, log_count_gpu,
                 superpixels, nIters = 10, nPts=None,         
                 threads_per_block=1024/4,
                 do_input_checks=False,
                 calculate_cov = True,
                 prior_prob_weight = 0.5): 
        
        
        dim_i = superpixels.dim_i      
        
        sp = superpixels # shorter name
        del superpixels  

        mu_s_gpu=sp.params.mu_s.gpu
        mu_i_gpu=sp.params.mu_i.gpu
        J_s_gpu=sp.params.J_s.gpu
        J_i_gpu=sp.params.J_i.gpu      
        logdet_Sigma_i_gpu = sp.params.logdet_Sigma_i.gpu
        logdet_Sigma_s_gpu = sp.params.logdet_Sigma_s.gpu
        
        s_std = sp.s_std 
        i_std = sp.i_std  
        
        # Part 1: input checks         
        if do_input_checks:
            # types
            for arg in [img_gpu, img_isNaN_gpu, seg_gpu,border_subset_gpu, mu_s_gpu, mu_i_gpu]:
                if not isinstance(arg,gpuarray.GPUArray):
                    raise TypeError(type(arg))
                    
            if dim_i not in [1,3]:
                raise ValueError(dim_i) 
            
            for arg in [border_gpu, border_subset_gpu]:
                # 2d arrays, dtypes
                if len(arg.shape) != 2:
                    raise ValueError(arg.shape)                   
                if arg.dtype != np.bool:
                    raise ValueError(arg.dtype)                           
            # same shape
            if border_gpu.shape != border_subset_gpu.shape:
               raise ValueError(border_gpu.shape, border_gpu.shape)        
        

        # Part 2: preparing for the gpu call
        ydim,xdim = border_gpu.shape[:2]  
        nSuperpixels = np.int32(sp.nSuperpixels)       

        if nPts is None:
            nPts = seg_gpu.shape[0] * seg_gpu.shape[1]        
        
        nBlocks = int(np.ceil(float(nPts) / float(threads_per_block)))               


        # Part 3: The actual work       
        for iter in range(nIters):
            for xmod3 in range(3):
                for ymod3 in range(3):  
                    #find the border pixels
                    self._gpu_find_border_pixels(seg_gpu, border_gpu, 
                        np.int32(nPts), 
                        np.int32(xdim), np.int32(ydim),
                        np.int32(0),
                        grid=(nBlocks,1,1),  
                        block=(threads_per_block,1,1)
                    );
                    
                    self._gpu_update_seg_subset(img_gpu, img_isNaN_gpu, seg_gpu, border_gpu,
                        counts_gpu,  log_count_gpu, 
                        mu_i_gpu,  mu_s_gpu, J_i_gpu,  J_s_gpu, logdet_Sigma_i_gpu, logdet_Sigma_s_gpu,  
                        np.int32(nPts), 
                        np.int32(xdim), np.int32(ydim), 
                        np.int32(xmod3), np.int32(ymod3), 
                        nSuperpixels,
                        np.int32(calculate_cov), 
                        np.int32(s_std), np.int32(i_std), 
                        np.double(prior_prob_weight),
                        grid=(nBlocks,1,1),  
                        block=(threads_per_block,1,1)
                    );

update_seg_iter = _UpdateSegIter()


if __name__ == "__main__":
    pass