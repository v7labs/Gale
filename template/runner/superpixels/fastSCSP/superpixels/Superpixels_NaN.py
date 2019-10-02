#!/usr/bin/env python
"""
Authors:
Oren Freifeld, freifeld@csail.mit.edu
Yixin Li, Email: liyixin@mit.edu

"""

import numpy as np
from scipy.linalg import inv
from numpy.linalg import slogdet

from pycuda import gpuarray
from of.gpu import CpuGpuArray

from of.utils import *


class Superpixels(object):
    def __init__(self, nSuperpixels, s_std, i_std, prior_count, nChannels):
        """
        Initilize the parameters for the superpixels:

        The means are set to zeros at this point, 
        and will be set later in the first M step.
        The space/color covariances (and their inverse), however, are being set 
        to initial values here.    
        We use a Inverse-Wishart prior on the space covariance

        Arguments:
        nSuperpixels: the number of superpixels to generate
        s_std: should be fixed as nPixels_on_side            
        i_std: control the relative importance between RGB and location.
               The smaller it is, bigger the RGB effect is / more irregular the superpixels are.
        prior_count: determines the weight of Inverse-Wishart prior of space covariance(ex:1,5,10)
        nChannels: the number of channels of the input image (gray:1, LAB/RGB: 3)

        """
        if nChannels not in (1,3):
            raise NotImplementedError(nChannels)
        dim_i=nChannels
        dim_s=2 
        self.dim_i=dim_i
        self.dim_s=dim_s

        self.nSuperpixels=nSuperpixels 
        
        self.s_std, self.i_std, self.prior_count = s_std,i_std,prior_count
   
        
        mu_s = CpuGpuArray.zeros((nSuperpixels,dim_s)) 
        mu_i = CpuGpuArray.zeros((nSuperpixels,dim_i)) 
     
        Sigma_s = CpuGpuArray.zeros(shape = (nSuperpixels,dim_s,dim_s))
        J_s = CpuGpuArray.zeros_like(Sigma_s)
        
        Sigma_i = CpuGpuArray.zeros((nSuperpixels,dim_i,dim_i))      
        J_i = CpuGpuArray.zeros_like(Sigma_i)

        logdet_Sigma_i = CpuGpuArray.zeros((nSuperpixels,1)) # scalars
        logdet_Sigma_s = CpuGpuArray.zeros((nSuperpixels,1)) 

        # start with unnormalized counts (uniform)        
        counts = np.ones(nSuperpixels,dtype=np.int32)
        counts = CpuGpuArray(counts)
        
        self.params = Bunch()
        self.params.mu_i = mu_i
        self.params.mu_s = mu_s
        self.params.Sigma_i = Sigma_i
        self.params.Sigma_s = Sigma_s  
        self.params.prior_sigma_s_sum = Sigma_s
        self.params.J_i = J_i
        self.params.J_s = J_s      
        self.params.logdet_Sigma_i = logdet_Sigma_i
        self.params.logdet_Sigma_s = logdet_Sigma_s
        self.params.counts = counts   
        
        # set those parameters related to covariance
        self.initialize_params()
        
        # intermediate arrays needed for the Gaussian parameter calculation on GPU
        self.gpu_helper = Bunch()
        
        self.gpu_helper.mu_i_helper = gpuarray.zeros((nSuperpixels,dim_i),dtype=np.int64)
        self.gpu_helper.mu_s_helper = gpuarray.zeros((nSuperpixels,dim_s),dtype=np.int64)               
        self.gpu_helper.prior_sigma_s = self.params.prior_sigma_s_sum.gpu.copy() 
        self.gpu_helper.sigma_s_helper = gpuarray.zeros((nSuperpixels,3),dtype=np.int64)

        self.gpu_helper.log_count_helper = gpuarray.zeros(nSuperpixels,dtype=np.double)
        self.gpu_helper.non_NaN_count = gpuarray.zeros(nSuperpixels,dtype=np.int32)
        self.gpu_helper.NaN_count = gpuarray.zeros(nSuperpixels,dtype=np.int32)

        
       
    def initialize_params(self):
        """ 
        Initialize the params.Sigma_s/Sigma_i, 
        params.J_i/J_s, params.logdet_Sigma_i/logdet_Sigma_s,
        based on i_std and s_std

        """
        params = self.params
        dim_s = self.dim_s
        dim_i = self.dim_i
        s_std = self.s_std
        i_std = self.i_std
        nSuperpixels = self.nSuperpixels

        # Covariance for each superpixel is a diagonal matrix
        for i in range(dim_s):
            params.Sigma_s.cpu[:,i,i].fill(s_std**2)  
            params.prior_sigma_s_sum.cpu[:,i,i].fill(s_std**4) 

        for i in range(dim_i):
            params.Sigma_i.cpu[:,i,i].fill((i_std)**2)
        params.Sigma_i.cpu[:,1,1].fill((i_std/2)**2) # To account for scale differences between the L,A,B

        #calculate the inverse of covariance
        params.J_i.cpu[:]=map(inv,params.Sigma_i.cpu)
        params.J_s.cpu[:]=map(inv,params.Sigma_s.cpu)
        
        # calculate the log of the determinant of covriance
        for i in range(nSuperpixels):
            junk,params.logdet_Sigma_i.cpu[i] = slogdet(params.Sigma_i.cpu[i])
            junk,params.logdet_Sigma_s.cpu[i] = slogdet(params.Sigma_s.cpu[i])
        del junk

        self.update_params_cpu2gpu()
    
    def update_params_cpu2gpu(self):
        [arr.cpu2gpu() for arr in self.params.itervalues()]   

    def update_params_gpu2cpu(self):
        [arr.gpu2cpu() for arr in self.params.itervalues()]   

    def get_params_cpu(self):
        b = Bunch()
        for (k,arr) in self.params.iteritems():
            b[k]=arr.cpu       
        return b

    def initialize_params_from_cpu_values(self,params_cpu):
        for (k,arr) in self.params.iteritems():
            np.copyto(dst=arr.cpu,src=params_cpu[k])
        self.update_params_cpu2gpu() 


if __name__ == "__main__":
    nSuperpixels = 5
    sp = Superpixels(nSuperpixels,s_std=20,i_std=100,nChannels=3)
    
    params_cpu = sp.get_params_cpu()
    Pkl.dump(HOME+'/tmp.pkl',params_cpu,verbose=1,override=1)
    sp.initialize_params_from_cpu_values(params_cpu=params_cpu)