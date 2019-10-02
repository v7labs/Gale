#!/usr/bin/env python
"""
Author:
Yixin Li
Email: liyixin@mit.edu
"""

import numpy as np
from template.runner.superpixels.fastSCSP.of.utils import *
from template.runner.superpixels.fastSCSP.of.gpu.KernelThinWrapper import KernelThinWrapper
from .gpu import dirname_of_cuda_files
cuda_filename = os.path.join(dirname_of_cuda_files,'rgb_to_lab.cu')
FilesDirs.raise_if_file_does_not_exist(cuda_filename)
with open(cuda_filename,'r') as cuda_file:
    _gpu_kernel = cuda_file.read()
include_dirs=[dirname_of_cuda_files]



class _RgbToLab(KernelThinWrapper): 

    def __init__(self):
        super(type(self),self).__init__(gpu_kernel=_gpu_kernel,
        include_dirs=include_dirs)
        self._get_function_from_src_module('rgb_to_lab')

    def __call__(self,img_gpu, threads_per_block = 1024, do_input_checks=False):  
        
        if do_input_checks:
            if not isinstance(img_gpu,gpuarray.GPUArray):
                raise TypeError(type(img_gpu)) 

        nPts = img_gpu.shape[0] * img_gpu.shape[1]           
        num_block = int ( np.ceil(nPts / float(threads_per_block)) )
               
        self._gpu_rgb_to_lab(img_gpu, 
                            np.int32(nPts), 
                            grid=(num_block,1,1),  
                            block=(threads_per_block,1,1))          
            
rgb_to_lab = _RgbToLab()