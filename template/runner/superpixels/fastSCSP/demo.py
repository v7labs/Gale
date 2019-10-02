#!/usr/bin/env python
"""
Authors:

1)
Oren Freifeld
Email: freifeld@csail.mit.edu

2)
Yixin Li
Email: liyixin@mit.edu

Example usage of the superpixels code on an image
"""
import os
from sys import path
if '.'+os.sep not in path:
    path.insert(0,'.'+os.sep)



from scipy import misc
import scipy.io as sio
from matplotlib.pyplot import imread, imsave
import numpy as np
from template.runner.superpixels.fastSCSP.superpixels.SuperpixelsWrapper import SuperpixelsWrapper
from template.runner.superpixels.fastSCSP.of.utils import *


def main(img_filename = os.path.join('image','1.jpg'),
        nPixels_on_side = 15,
        i_std = 20, # std dev for color Gaussian
        **kwargs,
        ):
    if img_filename is None:
        raise ValueError("img_filename cannot be None")

     
    #img_filename = os.path.join(HOME,'Desktop','stairs2.jpg')
    FilesDirs.raise_if_file_does_not_exist(img_filename)
    
    # Part 1: Specify the parameters:
    prior_count_const = 5  # the weight of Inverse-Wishart prior of space covariance(ex:1,5,10)
    use_hex = False # toggle between hexagons and squares (for the init)
    prior_weight = 0.5 # in the segmentation, 
                       # we do argmax w * log_prior + (1-w) *log_likelihood.
                       # Keeping w (i.e., prior_weight) at 0.5 means we are trying 
                       # to maximize the true posterior. 
                       # We keep the paramter here in case the user will like 
                       # to tweak it. 
                        
    calc_s_cov = True # If this is False, then we avoid estimating the spatial cov.
    num_EM_iters = nPixels_on_side
    num_inner_iters = 10


    # Part 2 : prepare for segmentation

    sp_size = nPixels_on_side*nPixels_on_side 
    img = imread(img_filename)
    
    dimx=img.shape[1]
    dimy=img.shape[0]

    tic = time.process_time()
    sw = SuperpixelsWrapper(dimy=dimy,dimx=dimx, nPixels_in_square_side=nPixels_on_side,
                             i_std = i_std , s_std = nPixels_on_side, 
                             prior_count = prior_count_const*sp_size,
                             use_hex = use_hex 
                            )
    toc = time.process_time()
    print('init time = {}'.format(toc-tic))
    print('nSuperpixels ={}'.format(sw.nSuperpixels) )   
    sw.set_img(img) 
    # you can use the same SuperpixelsWrapper object with different imgs and/or, 
    # i_std, s_std, prior_count. 
    # Just call sw.set_img(new_img), sw.initialize_seg(), and/or
    # sw.set_superpixels(i_std=..., s_std = ..., prior_count = ...)
    # again and recompute the seg.
    # Please see demo_for_direc for an example


    # Part 3: Do the superpixel segmentation

    print("Actual works starts now")
    tic  = time.process_time()
    #actual work
    sw.calc_seg(nEMIters=num_EM_iters, nItersInner=num_inner_iters, calc_s_cov=calc_s_cov, prior_weight=prior_weight)
    # Copy the parameters from gpu to cpu
    sw.gpu2cpu() 
    toc  = time.process_time()
    print('superpixel calculation time = {}'.format(toc-tic))



    # Part 4: Save the mean/boundary image and the resulting parameters

    # Results will be save in the /result directory. You can change it.
    root_slash_img_num = os.path.splitext(img_filename)[0]
    image_direc = os.path.split(root_slash_img_num)[0]  
    img_num = os.path.split(root_slash_img_num)[1]   

    save_path_root = os.path.join(image_direc , 'result')
    print("I am going to save results into {}".format(save_path_root))
    FilesDirs.mkdirs_if_needed(save_path_root)

    img_overlaid = sw.get_img_overlaid()   # get the boundary image   
 
    img_cartoon = sw.get_cartoon()         # get the cartoon image
    grid = ['square','hex'][sw.use_hex]

    fname_res_border = os.path.join(save_path_root, '_'.join([img_num,'n','{0:03}'.format(nPixels_on_side), 'std', '{0:02}'.format(i_std), 'border', grid+'.png']))
    fname_res_cartoon = os.path.join(save_path_root, '_'.join([img_num, 'n','{0:03}'.format(nPixels_on_side), 'std', '{0:02}'.format(i_std), 'mean', grid+'.png']))

    print('saving {}'.format(fname_res_border))
    imsave(fname_res_border, img_overlaid)
    print('saving {}'.format(fname_res_cartoon))
    imsave(fname_res_cartoon, img_cartoon)

    #save the resulting parameters to MATLAB .mat file 
    mat_filename = os.path.join(save_path_root#, img_num + '_std_'+str(i_std)+'.mat')
                                              ,'_'.join([img_num,'n','{0:03}'.format(nPixels_on_side), 'std', '{0:02}'.format(i_std)+'.mat']))
    print('Saving params to {}'.format(mat_filename))
    
    pm = sw.superpixels.params    

    sio.savemat(mat_filename,{'pixel_to_super':sw.seg.cpu, 'count':pm.counts.cpu,
                              'mu_i': pm.mu_i.cpu, 'mu_s':pm.mu_s.cpu, 
                              'sigma_s':pm.Sigma_s.cpu, 'sigma_i':pm.Sigma_i.cpu})




if __name__ == "__main__":  
    import argparse
    
    default_fname=os.path.join('image','2.jpg')
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--img_filename",
                        nargs='?',
                        const=default_fname,default=default_fname)
    parser.add_argument("-n","--nPixels_on_side", type=int, help="the desired number of pixels on the side of a superpixel",
                        nargs='?',
                        const=15,default=15)
    parser.add_argument("--i_std", type=int, help="std dev for color Gaussians, should be 5<= value <=40. A smaller value leads to more irregular superpixels",
                        nargs='?',
                        const=20,default=20)
    args = parser.parse_args()
    
     

    args = parser.parse_args()    
    main(**args.__dict__)   
