#!/usr/bin/env python
"""
Created on Wed Sep  3 11:12:25 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from template.runner.superpixels.fastSCSP.of.utils import *
import numpy as np
from template.runner.superpixels.fastSCSP.of.gpu.KernelThinWrapper import KernelThinWrapper
from pycuda import gpuarray

from .gpu import dirname_of_cuda_files

cuda_filename = os.path.join(dirname_of_cuda_files,'update_seg.cu')
include_dirs=[dirname_of_cuda_files]
FilesDirs.raise_if_file_does_not_exist(cuda_filename)
with open(cuda_filename,'r') as cuda_file:
    _gpu_kernel = cuda_file.read()



class _FindBorderPixels(KernelThinWrapper):    
    def __init__(self):
        super(type(self),self).__init__(gpu_kernel=_gpu_kernel,
                                        include_dirs=include_dirs)
        self._get_function_from_src_module('find_border_pixels')

    def __call__(self,seg_gpu,border_gpu,single_border = 0, nPts=None,
            threadsPerBlock=1024,do_input_checks=False): 
        """
        Update border_gpu such that 
        the pixels on the superpixel boundary takes value of True.

        Arguments:
        single_border: whether the find the two-pixels-border or one-pixel-border

        """        
        # Part 1: input checks    
        if do_input_checks:          
            for arg in [seg_gpu, border_gpu]:
                if not isinstance(arg,gpuarray.GPUArray):
                    raise TypeError(type(arg)) 
                if len(arg.shape) !=2:
                    raise ValueError(arg.shape)                     
                if arg.dtype != np.int32:
                    raise ValueError(arg.dtype)
            
            # same shape
            if seg_gpu.shape != border_gpu.shape:
                raise ValueError(seg.shape , border_gpu.shape)        
        
        # Part 2: preparing for the gpu call
        ydim,xdim= seg_gpu.shape[:2]                 
        if nPts is None:
            nPts = seg_gpu.shape[0] * seg_gpu.shape[1]            
            
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock)))               
        
        # Part 3: The actual work   
        self._gpu_find_border_pixels(seg_gpu, 
              border_gpu,                          
              np.int32(nPts),
              np.int32(xdim),
              np.int32(ydim),
              np.int32(single_border),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1))      

find_border_pixels = _FindBorderPixels()
