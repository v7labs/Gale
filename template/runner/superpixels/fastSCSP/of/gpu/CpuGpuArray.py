#!/usr/bin/env python
"""
Created on Thu Sep 18 09:42:30 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray

class CpuGpuArray(object):
    """
    The wrapper class for an array that could be used in both cpu and gpu
    """
    def __init__(self,arr):
        """
        arr is either a numpy array or a pycuda array
        """
        if isinstance(arr,CpuGpuArray):
            raise TypeError('arr is already of type CpuGpuArray')
        if isinstance(arr,np.ndarray):
            self.verify_is_c_contiguous_and_is_not_fortran(arr)
            self.cpu = arr
            self.gpu = gpuarray.to_gpu(arr)            
        elif isinstance(arr,gpuarray.GPUArray):
            self.gpu = arr
            self.cpu = arr.get() 
        for prop in ['shape','dtype','ndim','size']:
            self.__dict__[prop] = getattr(self.cpu,prop)

    @classmethod        
    def verify_is_c_contiguous_and_is_not_fortran(cls,arr):
        """
        arr is a numpy array.
        """
        if np.isfortran(arr):
            raise ValueError("Must be 'C' order")  
        if not arr.flags.c_contiguous:
            msg="Must be 'C'-contiguous. Consider passing arr.copy() instead."
            raise ValueError(msg)   
       

        
    def gpu2cpu(self):
        self.gpu.get(ary=self.cpu)
                   
    def cpu2gpu(self):
        self.gpu.set(ary=self.cpu)
    
    def __len__(self):
        return len(self.cpu)
    def max(self,mem,*kw,**kwargs):
        if mem not in ['cpu','gpu']:
            raise ValueError(mem)
        if mem=='cpu':
            return self.cpu.max(*kw,**kwargs)
        elif mem == 'gpu':
            raise NotImplementedError
        else:
            raise ValueError(mem)
    def min(self,mem,*kw,**kwargs):
        if mem not in ['cpu','gpu']:
            raise ValueError(mem)
        if mem=='cpu':
            return self.cpu.min(*kw,**kwargs)
        elif mem == 'gpu':
            raise NotImplementedError
        else:
            raise ValueError(mem)            

    def __repr__(self):
        return 'cpu:\n{0}\ngpu:\n{1}:'.format(repr(self.cpu),repr(self.gpu))
        
    
    def astype(self,dtype,mem='gpu'):
        if mem not in ['cpu','gpu']:
            raise ValueError(mem)
        if mem == 'cpu':
            return CpuGpuArray(self.cpu.astype(dtype))
        elif mem == 'gpu':
            return CpuGpuArray(self.gpu.astype(dtype))
        
    @staticmethod
    def zeros(shape,dtype=np.float64):
        return CpuGpuArray(np.zeros(shape=shape,dtype=dtype))
    @staticmethod
    def zeros_like(cpugpu_arr):
        if not isinstance(cpugpu_arr,CpuGpuArray):
            msg = "Expected a CpuGpuArray. Got {0}".format(type(cpugpu_arr))
            raise TypeError(msg)
        return CpuGpuArray(np.zeros_like(cpugpu_arr.cpu))
    @staticmethod
    def ones(shape,dtype=np.float64):
        return CpuGpuArray(np.ones(shape=shape,dtype=dtype))        
    @staticmethod
    def ones_like(cpugpu_arr):
        if not isinstance(cpugpu_arr,CpuGpuArray):
            msg = "Expected a CpuGpuArray. Got {0}".format(type(cpugpu_arr))
            raise TypeError(msg)
        return CpuGpuArray(np.ones_like(cpugpu_arr.cpu))        
    @staticmethod
    def empty(shape,dtype=np.float64):
        return CpuGpuArray(np.empty(shape=shape,dtype=dtype))
    @staticmethod
    def empty_like(cpugpu_arr):
        if not isinstance(cpugpu_arr,CpuGpuArray):
            msg = "Expected a CpuGpuArray. Got {0}".format(type(cpugpu_arr))
            raise TypeError(msg)       
        return CpuGpuArray(np.empty_like(cpugpu_arr.cpu))

    def __lt__(self, other):
        raise Exception("Use self.cpu or self.gpu instead")
    def __le__(self, other):
        raise Exception("Use self.cpu or self.gpu instead")
    def __eq__(self, other):
        raise Exception("Use self.cpu or self.gpu instead")
    def __ne__(self, other):
        raise Exception("Use self.cpu or self.gpu instead")
    def __gt__(self, other):
        raise Exception("Use self.cpu or self.gpu instead")
    def __ge__(self, other):
        raise Exception("Use self.cpu or self.gpu instead")        
 
   

if __name__ == "__main__":
    print(CpuGpuArray.zeros((3,3)))  
    print(CpuGpuArray.ones((3,3)))
