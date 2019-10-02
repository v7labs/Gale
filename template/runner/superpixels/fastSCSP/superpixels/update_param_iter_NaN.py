#!/usr/bin/env python
"""
Authors:
Oren Freifeld, freifeld@csail.mit.edu
Yixin Li, Email: liyixin@mit.edu
"""

import numpy as np

from of.utils import *
from of.gpu.KernelThinWrapper import KernelThinWrapper
from of.gpu import CpuGpuArray
from gpu import dirname_of_cuda_files

cuda_filename = os.path.join(dirname_of_cuda_files,'update_param_NaN.cu')
FilesDirs.raise_if_file_does_not_exist(cuda_filename)
with open(cuda_filename,'r') as cuda_file:
    _gpu_kernel = cuda_file.read()
include_dirs=[dirname_of_cuda_files]


class _UpdateParamIter(KernelThinWrapper):    
    def __init__(self):
        super(type(self),self).__init__(gpu_kernel=_gpu_kernel,
        include_dirs=include_dirs)
        self._get_function_from_src_module('clear_fields')
        self._get_function_from_src_module('sum_by_label') 
        self._get_function_from_src_module('calculate_mu_and_sigma')
        self._get_function_from_src_module('clear_fields_2')
        self._get_function_from_src_module('sum_by_label_2') 
        self._get_function_from_src_module('calculate_mu')

    def __call__(self,img_gpu, img_isNaN_gpu, seg_gpu,sp, 
                    calculate_s_cov=True, 
                    threads_per_block = 1024/4,
                    nsuperpixel=None, 
                    do_input_checks=False): 
        # Part 1: input checks       
        if do_input_checks:
            for arg in [img_gpu,img_isNaN_gpu, seg_gpu]:
                if not isinstance(arg,CpuGpuArray):
                    raise TypeError(type(arg))
            # same shape
            if img_gpu.shape != seg_gpu.shape:
               raise ValueError(img_gpu.shape, seg_gpu.shape) 
            
        # Part 2: preparing for the gpu call
        if nsuperpixel is None:
            nsuperpixel= sp.nSuperpixels
        xdim = np.int32(seg_gpu.shape[1])
        ydim = np.int32(seg_gpu.shape[0] )      
        nPts = np.int32(seg_gpu.shape[0] * seg_gpu.shape[1])       
        
        num_block1 = int ( np.ceil(nPts / float(threads_per_block)) ) 
        num_block2 = int ( np.ceil(nsuperpixel / float(threads_per_block)) )    

        dim_i = np.int32(sp.dim_i)
        prior_count = np.int32(sp.prior_count)
        nsuperpixel = np.int32(nsuperpixel)

        mu_i = sp.params.mu_i.gpu 
        mu_s = sp.params.mu_s.gpu
        count = sp.params.counts.gpu
        NaN_count = sp.gpu_helper.NaN_count
        non_NaN_count = sp.gpu_helper.non_NaN_count
        mu_i_h = sp.gpu_helper.mu_i_helper
        mu_s_h = sp.gpu_helper.mu_s_helper
        log_count = sp.gpu_helper.log_count_helper     

        # Part 3: Actual work
        if calculate_s_cov:
            sigma_s = sp.params.Sigma_s.gpu
            sigma_s_h = sp.gpu_helper.sigma_s_helper
            prior_sigma_s_sum =  sp.gpu_helper.prior_sigma_s
            logdet_s = sp.params.logdet_Sigma_s.gpu
            J_s = sp.params.J_s.gpu
                
            # initialize all the arrays, parallelize over superpixels
            self._gpu_clear_fields(count, non_NaN_count, NaN_count, log_count, mu_i_h, mu_s_h, mu_i, mu_s, sigma_s_h,
                    dim_i, nsuperpixel,
                    grid=(num_block2,1,1), 
                    block=(threads_per_block,1,1)
                    )
               
            #calculate all the needed sums: count, mu_i_h, mu_s_h, sigma_s
            #parallelize over pixels
            self._gpu_sum_by_label (img_gpu, img_isNaN_gpu, seg_gpu, 
                    count, non_NaN_count, NaN_count,
                    mu_i_h, mu_s_h, sigma_s_h,
                    xdim, ydim, dim_i, nPts,
                    grid=(num_block1,1,1), 
                    block=(threads_per_block,1,1)
                    )

            #normalization and calculation of mu_i, mu_s, sigma_s, logdet_s, J_s
            #parallelize over superpixels
            self._gpu_calculate_mu_and_sigma(count, non_NaN_count, log_count, mu_i_h, mu_s_h, 
                    mu_i, mu_s, sigma_s_h, prior_sigma_s_sum, sigma_s, logdet_s, J_s, 
                    prior_count, dim_i, nsuperpixel,   
                    grid=(num_block2,1,1), 
                    block=(threads_per_block,1,1)
                    ) 
        else:
            # initialize the arrays count, mu_i_h, mu_s_h, parallelize over superpixels
            self._gpu_clear_fields_2(count, non_NaN_count, mu_i_h, mu_s_h,
                    dim_i, nsuperpixel,
                    grid=(num_block2,1,1), 
                    block=(threads_per_block,1,1)
                   )         
            #calculate all the needed sums: count, mu_i_h, mu_s_h, parallelize over pixels
            self._gpu_sum_by_label_2(img_gpu, img_isNaN_gpu, seg_gpu, 
                    count, non_NaN_count, mu_i_h, mu_s_h,
                    xdim, ydim, dim_i, nPts, 
                    grid=(num_block1,1,1), 
                    block=(threads_per_block,1,1)
                    )
            #normalization and calculation of mu_i, mu_s, parallelize over superpixels
            self._gpu_calculate_mu(count, non_NaN_count, mu_i_h, mu_s_h, mu_i, mu_s,
                    dim_i, nsuperpixel,
                    grid=(num_block2,1,1), 
                    block=(threads_per_block,1,1)
                    ) 


update_param_iter = _UpdateParamIter()

if __name__ == "__main__": 
    pass