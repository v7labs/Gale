#!/usr/bin/env python
"""
Authors:
Oren Freifeld, freifeld@csail.mit.edu
Yixin Li, Email: liyixin@mit.edu
"""
import numpy as np
from pycuda import gpuarray
from of.gpu import CpuGpuArray

from superpixels.Superpixels_NaN import Superpixels 

from superpixels.init_seg import get_init_seg,random_permute_seg
from superpixels.find_border_pixels import find_border_pixels
from superpixels.rgb_to_lab import rgb_to_lab 
from superpixels.lab_to_rgb import lab_to_rgb
from superpixels.get_cartoon import get_cartoon
from superpixels.update_seg_iter_NaN import update_seg_iter as _update_seg_iter
from superpixels.update_param_iter_NaN import update_param_iter as _update_param_iter



class SuperpixelsWrapper(object):
 
    _calc_seg = _update_seg_iter
    _calc_param = _update_param_iter

    def __init__(self, dimy, dimx, nPixels_in_square_side=None, permute_seg=False,
                 s_std = 20, i_std=20, prior_count = 1,
                 calc_s_cov=True,
                 use_hex=False):
        """
        arguments:
        nPixels_in_square_side: number of the pixels on the side of a superpixels
        permute_seg: only for display purpose, whether to permute the superpixels labling, 
        s_std: fixed as nPixels_on_side            
        i_std: control the relative importance between RGB and location.
               The smaller it is, bigger the RGB effect is / more irregular the superpixels are.
        prior_count: determines the weight of Inverse-Wishart prior of space covariance(ex:1,5,10)
        calc_s_cov:  whether or not to update the covariance of color component of Gaussians
        use_hex: initialize the superpixels as hexagons or squares
        
        """

        # init the field "nPixels_in_square_side"
        if nPixels_in_square_side is None:
            raise NotImplementedError
        self.nPixels_in_square_side=nPixels_in_square_side
        if not (3<nPixels_in_square_side<min(dimx,dimy)):
            msg = """
            Expected
            3<nPixels_in_square_side<min(dimx,dimy)={1}
            but nPixels_in_square_side={0}
            """.format(nPixels_in_square_side,min(dimx,dimy))
            raise ValueError(msg)

        self.dimx=dimx
        self.dimy=dimy
        self.nChannels = 3 # RGB/LAB

        #init the field "seg"
        self.use_hex = use_hex
        self.permute_seg = permute_seg
        seg = get_init_seg(dimy=dimy,dimx=dimx,
                          nPixels_in_square_side=nPixels_in_square_side, use_hex = use_hex)   
        if permute_seg:
            seg = random_permute_seg(seg)  
        self.seg = CpuGpuArray(arr=seg)     
             
        # keep the initial segmentation for use in images of the same size 
        # so that we won't re-calculate the initial segmenation
        self.seg_ini = self.seg.cpu.copy() 

        #init the field nSuperpixels
        self.nSuperpixels = self.seg.cpu.max()+1 
        if self.nSuperpixels <= 1:
            raise ValueError(self.nSuperpixels)     

        #init the field "superpixels"
        self.superpixels = Superpixels(self.nSuperpixels,
                            s_std=s_std, i_std=i_std, prior_count = prior_count, 
                            nChannels=self.nChannels)

        #init the field "border"(bool array, true for pixels on the superpixel boundary)
        border = gpuarray.zeros((dimy,dimx),dtype=np.bool)
        self.border = CpuGpuArray(arr=border)
        find_border_pixels(seg_gpu=self.seg.gpu, border_gpu=self.border.gpu) 
        self.border.gpu2cpu()        
        self.border_ini = self.border.cpu.copy() 
        
        print 'dimy,dimx=',dimy,dimx
        print 'nSuperpixels =',self.nSuperpixels
        

    def set_img(self,img):   
        """
        read an rgb image, set the gpu copy to be the lab image
        """
        if img.shape[0] != self.dimy or img.shape[1] != self.dimx:
            raise ValueError(img.shape,self.dimy,self.dimx)
        if img.ndim == 1:
            nChannels = 1
            isNaN = np.isnan( img)
        elif img.ndim == 3:
            nChannels = 3
            img_isNaN_r = np.isnan( img[:,:,0] )
            img_isNaN_g = np.isnan( img[:,:,1] )
            img_isNaN_b = np.isnan( img[:,:,2] )
            isNaN =  np.logical_or(img_isNaN_r, np.logical_or(img_isNaN_g,img_isNaN_b))
        else:
            raise NotImplementedError(nChannels)         

        self.img = CpuGpuArray(arr=img)
        self.img_isNaN = CpuGpuArray(arr=isNaN)

        print 'self.img',self.img
        print 'self.img_isNaN',self.img_isNaN

        if nChannels==3:
            rgb_to_lab(img_gpu=self.img.gpu)

        
        
    def initialize_seg(self):
        """ 
        set the self.seg/border to the initial segmentation/border 
        """
        np.copyto(dst=self.seg.cpu,src=self.seg_ini)     
        np.copyto(dst=self.border.cpu,src=self.border_ini)
        self.seg.cpu2gpu()
        self.border.cpu2gpu() 

    def set_superpixels(self, s_std, i_std, prior_count):
        """ 
        set the self.superpixels 
        """  
        self.superpixels = Superpixels(self.nSuperpixels,
                                s_std=s_std, i_std=i_std, prior_count = prior_count,
                                nChannels=self.nChannels)

        
    def calc_seg(self, nEMIters, nItersInner=10, calc_s_cov=True, prior_weight=0.5, verbose=False):
        """ 
        Inference:
        hard-EM (Expectation & Maximization)
        Alternate between the E step (update superpixel assignments) 
        and the M step (update parameters)

        Arguments:
        calc_s_cov: whether or not to update the covariance of color component of Gaussians
        prior_weight: the weight placed on the prior probability of a superpixel

        """
        if verbose:
            print 'start'
            
        for i in range(nEMIters): 
            if verbose:
                print 'iteration',i
            "M step"
            self._calc_param(img_gpu=self.img.gpu, img_isNaN_gpu =self.img_isNaN.gpu, seg_gpu=self.seg.gpu, 
                            sp=self.superpixels, 
                            calculate_s_cov=calc_s_cov)
            "(Hard) E step"
            self._calc_seg(img_gpu=self.img.gpu,  img_isNaN_gpu =self.img_isNaN.gpu,  seg_gpu=self.seg.gpu,
                            border_gpu=self.border.gpu,
                            counts_gpu=self.superpixels.params.counts.gpu,
                            log_count_gpu = self.superpixels.gpu_helper.log_count_helper,
                            superpixels=self.superpixels,
                            nIters=nItersInner,
                            calculate_cov=calc_s_cov,
                            prior_prob_weight = prior_weight)       
        # update the border for display the single border image
        find_border_pixels(seg_gpu=self.seg.gpu, border_gpu=self.border.gpu, single_border=1)


    def gpu2cpu(self):
        """
        Transfer the needed parameters from gpu to cpu.
        Note this is usually slow (when the image and/or nSuperpixels is large)
        """
        if self.img.ndim == 3:
            lab_to_rgb(self.superpixels.params.mu_i.gpu) #convert mu_i.gpu from lab to rgb
            lab_to_rgb(self.img.gpu) # convert img.gpu from lab to rgb
        
        params = self.superpixels.params
        for arg in [self.seg, self.border, self.img, params.counts, params.mu_s, params.mu_i, params.Sigma_s]:
            arg.gpu2cpu()


    def get_img_overlaid(self):
        """
        Set the pixels on the superpixel boundary to red

        """
        img = self.img.cpu    
        border = self.border.cpu    
        nChannels = self.nChannels                 
        if nChannels==1: 
            img_disp = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
            img_disp[:,:,0]=img_disp[:,:,1]=img_disp[:,:,2]=img
        elif nChannels==3:
            img_disp = img.copy().astype(np.uint8)
        img_disp[:,:,0][border] = 255
        img_disp[:,:,1][border] = 0
        img_disp[:,:,2][border] = 0 
        return img_disp             
             

    def get_cartoon(self):
        """
        Replace pixels with superpixels means.

        """   
        img = self.img.cpu   
        nChannels = self.nChannels             
        img_disp = CpuGpuArray.zeros((img.shape[0],img.shape[1],3),dtype=np.int32)     
        get_cartoon(seg_gpu = self.seg.gpu, mu_i_gpu = self.superpixels.params.mu_i.gpu, 
                    img_gpu= img_disp.gpu, nChannels=nChannels)
        img_disp.gpu2cpu()
        return img_disp.cpu
    

if __name__ == '__main__':
    import demo