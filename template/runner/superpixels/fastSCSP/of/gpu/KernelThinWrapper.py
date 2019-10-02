#!/usr/bin/env python
"""
Created on Wed Sep  3 11:08:37 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from pycuda.compiler import SourceModule
from pycuda.driver import Context


try:            
    Context.get_device() 
except:
    import pycuda.autoinit

class KernelThinWrapper(object):
    def __init__(self,gpu_kernel,include_dirs=[]):
        self._gpu_kernel = gpu_kernel
        self._src_module = SourceModule(gpu_kernel,include_dirs=include_dirs) 
    def _get_function_from_src_module(self,func_name):
        self.__dict__['_gpu_'+func_name] = self._src_module.get_function(func_name) 
    def __call__(self,*args,**kwargs):
        msg="""
        You need to customize this method in the derived class.
        
        The customized method will usually have 3 parts:
        # Part 1: input checks (optional)
        # Part 2: preparing for the gpu call (e.g. defining nBlocks, etc.)
        # Part 3: The actual work (calling the gpu function) 
        """                 
        raise Exception(msg)
 


if __name__ == "__main__":
    pass















