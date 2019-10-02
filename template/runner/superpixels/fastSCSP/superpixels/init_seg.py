#!/usr/bin/env python
"""
Created on Tue Sep  2 07:14:28 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import logging
import numpy as np
from template.runner.superpixels.fastSCSP.of.utils import *
from template.runner.superpixels.fastSCSP.of.gpu import CpuGpuArray
from pylab import plt 
from .honeycomb import honeycomb

dirname_of_this_file = os.path.dirname(inspect.getfile(inspect.currentframe()))
dirname_precomputed_hex_inits = os.path.join(dirname_of_this_file,'precomputed_hex_inits')
FilesDirs.mkdirs_if_needed(dirname_precomputed_hex_inits)

def create_string(dimx,dimy,nPixels_in_square_side):
   s = 'dimx_{0}_dimy_{1}_n_{2}.npy'.format(dimx,dimy,nPixels_in_square_side) 
   return s


def get_init_seg(dimy,dimx,nPixels_in_square_side,use_hex):
    """
    """
    M=nPixels_in_square_side 

    if use_hex:
        s = create_string(dimx,dimy,nPixels_in_square_side)
#        print s
        fname = os.path.join(dirname_precomputed_hex_inits,s)
        try:
            FilesDirs.raise_if_file_does_not_exist(fname)
            logging.debug("Loading",fname)
            seg = np.load(fname)
            
            return seg
        except FileDoesNotExistError:
            pass
        msg = """
        I could not find a precomputed (image-independent)
        honeycomb initilization for this image size and this values of n.     
        So I will compute it from scratch and we will save the result in
        {}
        Next time you will run the code for an image of size
        nRows={}, nCols={}, with n = {},
        it will be faster.
        """.format(fname,dimy,dimx,nPixels_in_square_side)
        print(msg)
        seg = CpuGpuArray.zeros((dimy,dimx),dtype=np.int32)     

        # length of each side   
        a = np.sqrt(M ** 2 / (  1.5 * np.sqrt(3)  ))      
        H =   a
        W = np.sqrt(3)*H
        # XX and YY need to be float
        YY,XX = np.mgrid[0:float(dimy)+0*1.5*H:1.5*H,0:float(dimx)+0*W:W]

        XX[::2]+= float(W)/2
        centers = np.vstack([XX.ravel(),YY.ravel()]).T.copy()
        centers = CpuGpuArray(centers)    

        honeycomb(seg.gpu,centers.gpu,seg.size) 

        seg.gpu2cpu()
        np.save(fname,seg.cpu)
        return seg.cpu

    else:
        seg_cpu = np.zeros((dimy,dimx),dtype=np.int32)  
        yy,xx = np.mgrid[:dimy,:dimx] 
        xx = xx.astype(np.float)
        yy = yy.astype(np.float)
       
        dimx = float(dimx)
        dimy=float(dimy)        
        nTimesInX = np.floor(xx / M).max() + 1
 
        seg_cpu = np.floor(yy / M)  * nTimesInX + np.floor(xx / M)
        seg_cpu = seg_cpu.astype(np.int32)
        return seg_cpu


def random_permute_seg(seg):
    p=np.random.permutation(seg.max()+1)   
    seg2 = np.zeros_like(seg)
    for c in range(seg.max()+1):             
        seg2[seg==c]=p[c]
    return seg2.astype(np.int32)


if __name__ == "__main__":  
    tic = time.clock()
    seg= get_init_seg(500, 500,17,True)      
#    seg= get_init_seg(512, 512,50,False)  
    toc = time.clock()
    print(toc-tic)
    print('k = ', seg.max()+1)
    plt.figure(1)
    plt.clf()  
    plt.imshow(seg,interpolation="Nearest")
    plt.axis('scaled') 