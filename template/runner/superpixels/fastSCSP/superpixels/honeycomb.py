#!/usr/bin/env python
"""
Created on Fri Jan  2 11:40:38 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from template.runner.superpixels.fastSCSP.of.utils import *
import numpy as np
from template.runner.superpixels.fastSCSP.of.gpu.KernelThinWrapper import KernelThinWrapper
from pycuda import gpuarray

from template.runner.superpixels.fastSCSP.superpixels.gpu import dirname_of_cuda_files

cuda_filename = os.path.join(dirname_of_cuda_files,'honeycomb.cu')
include_dirs=[dirname_of_cuda_files]
FilesDirs.raise_if_file_does_not_exist(cuda_filename)
with open(cuda_filename,'r') as cuda_file:
    _gpu_kernel = cuda_file.read()


   

class _Honeycomb(KernelThinWrapper):    
    def __init__(self):
        super(type(self),self).__init__(gpu_kernel=_gpu_kernel,
                                        include_dirs=include_dirs)
        self._get_function_from_src_module('honeycomb')
        
    def __call__(self,seg_gpu, centers_gpu,
                 nPts=None, threadsPerBlock=1024, do_input_checks=False): 
                
        #Part 1: input checks    
        if do_input_checks:         
            for arg in [seg_gpu, centers_gpu]:
              # types
              if not isinstance(arg,gpuarray.GPUArray):
                  raise TypeError(type(arg))  
              # 2d arrays            
              if len(arg.shape) !=2:
                  raise ValueError(arg.shape)
              # dtypes
              if arg.dtype != np.int32:
                  raise ValueError(arg.dtype)

                                    
        # Part 2: preparing for the gpu call
        ydim,xdim= seg_gpu.shape[:2]                 
        if nPts is None:
            nPts = seg_gpu.shape[0] * seg_gpu.shape[1]            
            
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))               
        
        K = centers_gpu.shape[0]


        # Part 3: The actual work   
        self._gpu_honeycomb(seg_gpu, 
              centers_gpu,
              np.int32(K),                    
              np.int32(nPts),
              np.int32(xdim),
              np.int32(ydim),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1))      

honeycomb = _Honeycomb()
