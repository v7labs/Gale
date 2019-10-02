# Utils
import logging
import os
import time
from os.path import isfile
import cv2
import numpy as np
from matplotlib.pyplot import imread

# DeepDIVA
from tqdm import tqdm
from template.runner.base import AbstractRunner

# Delegated
from template.runner.superpixels.fastSCSP.superpixels.SuperpixelsWrapper import SuperpixelsWrapper
from util.TB_writer import TBWriter


class Superpixels(AbstractRunner):

    @classmethod
    def single_run(cls, input_folder, img_filename, sartorious_cell, **kwargs):
        """
        This is the main routine where I compute superpixels

        input folder : str
            Path to the data
        img_filename : str
            If specified, only that image will be processed
        sartorious_cell : bool
            Flag on whether the data should be preprocessed as a cell image or not

        Returns
        -------
        dummy values
        """
        # If single file goahead, otherwise process whole folder
        if img_filename is not None:
            # Load input image
            img_filename = os.path.join(input_folder, img_filename)
            img = imread(img_filename)

            if sartorious_cell:
                img, heatmap = cls.prepare_for_sartorious(img, log=True, **kwargs)
                img_global = cls.process_image(heatmap, **kwargs)
                output = np.maximum(np.stack((img_global,) * 3, axis=-1), Superpixels.normalize(img))
                TBWriter().save_image(tag='all/output', image=output)
            else:
                output = cls.process_image(img, **kwargs)
                img_bins, img_overlaid, img_cartoon, _ = cls.superpixelize(img, **kwargs)
                TBWriter().save_image(tag='superpixel/bins', image=img_bins)
                TBWriter().save_image(tag='superpixel/overlay', image=img_overlaid)
                TBWriter().save_image(tag='superpixel/cartoon', image=img_cartoon)
                TBWriter().save_image(tag='all/input', image=img)
                TBWriter().save_image(tag='all/output', image=output)
            return 0, 0 ,0

        for file in os.listdir(input_folder):
            path = os.path.join(input_folder, file)
            if isfile(path):
                img = imread(path)
                # Run with original parameters
                img_bins = cls.superpixelize(img, **kwargs)[0]
                TBWriter().save_image(tag=f'superpixel/bins/{file}', image=img_bins)
                # Create the structural analysis
                img_struct = cls.process_image(img, log=False, **kwargs)
                TBWriter().save_image(tag=f'superpixel/structure/{file}', image=img_struct)
        return 0, 0 ,0

    @classmethod
    def process_image(cls, img, nPixels_on_side, i_std, log=False,**kwargs):
        """
        Process one image by aggregating several super-pixels borders into one collective one

        Parameters
        ----------
        img : ndarray
            An RGB image to process
        nPixels_on_side : int
            Parameter for the superpixel algorithm. See documentation
        i_std : int
            Parameter for the superpixel algorithm. See documentation

        Returns
        -------
        global_border_sum : ndarray
            A 1-channel image containing the cropped sum of borders highlighted by the superpixel algorithm
        """
        assert len(img.shape) == 3

        # Pad the image, run superpixel algorithm and aggregate results
        global_border_sum = np.zeros([img.shape[0], img.shape[1]])
        margin = 0.3
        step = int((nPixels_on_side * 2 * margin) // 4)
        for nps in tqdm(range(int(nPixels_on_side * (1 - margin)), int(nPixels_on_side * (1 + margin)), step), leave=False):
            local_border_sum = np.zeros([img.shape[0], img.shape[1]])
            # Pad image
            for h_pad in range(1, nps, nps//3):
                for v_pad in range(1, nps, nps//3):
                    # Pad image
                    padded = np.zeros([img.shape[0] + h_pad, img.shape[1] + v_pad, 3])
                    padded[h_pad:img.shape[0] + h_pad, v_pad:img.shape[1] + v_pad, :] = img
                    # Run and remove padding
                    _, _, _, tmp = cls.superpixelize(padded, nps, i_std, **kwargs)
                    img_border = tmp[h_pad:img.shape[0] + h_pad, v_pad:img.shape[1] + v_pad]
                    # Merge border
                    local_border_sum[img_border] += 1
            if log: TBWriter().save_image(tag=f'superpixel/local_sum_{i_std}', image=local_border_sum, normalize=True)
            global_border_sum += local_border_sum
        if log: TBWriter().save_image(tag='superpixel/sum_global', image=global_border_sum, normalize=True)

        # Remove the weakest % of borders
        global_border_sum -= global_border_sum.max() * 0.25
        global_border_sum[global_border_sum < 0] = 0
        global_border_sum = Superpixels.normalize(global_border_sum)
        if log: TBWriter().save_image(tag='superpixel/sum_global_cropped', image=global_border_sum)

        return global_border_sum

    @classmethod
    def superpixelize(cls, img, nPixels_on_side, i_std, **kwargs):
        """ Process the image by computing the superpixels"""

        tic = time.time()

        # Part 1: Specify the parameters:
        # the weight of Inverse-Wishart prior of space covariance(ex:1,5,10)
        prior_count_const = 5
        # in the segmentation, we do argmax w * log_prior + (1-w) *log_likelihood.
        # Keeping w (i.e., prior_weight) at 0.5 means we are trying to maximize the true posterior.
        # We keep the paramter here in case the user will like to tweak it.
        prior_weight = 0.5

        calc_s_cov = True  # If this is False, then we avoid estimating the spatial cov.
        num_EM_iters = nPixels_on_side
        num_inner_iters = 10

        # Part 2 : prepare for segmentation
        sp_size = nPixels_on_side * nPixels_on_side


        sw = SuperpixelsWrapper(dimy=img.shape[0], dimx=img.shape[1], nPixels_in_square_side=nPixels_on_side,
                                i_std=i_std, s_std=nPixels_on_side,
                                prior_count=prior_count_const * sp_size,
                                use_hex=False)

        logging.debug(f'Image size={img.shape[1]}x{img.shape[0]} nSuperpixels={sw.nSuperpixels}')
        sw.set_img(img)

        # Part 3: Do the superpixel segmentation
        # Actual work
        sw.calc_seg(nEMIters=num_EM_iters, nItersInner=num_inner_iters, calc_s_cov=calc_s_cov, prior_weight=prior_weight)
        # Copy the parameters from gpu to cpu
        sw.gpu2cpu()
        toc = time.time()
        logging.debug(f'Superpixel calculation time ={(toc - tic)*1000:.0f}[ms]')

        # Part 4: Get, log and return the outcome
        img_overlaid = sw.get_img_overlaid()
        img_cartoon = sw.get_cartoon()
        img_bins = cls.encode_bin_to_RGB(sw.get_segmentation_bins())

        return img_bins, img_overlaid, img_cartoon, sw.get_border_mask()

    @classmethod
    def prepare_for_sartorious(cls, img, log=False, **kwargs):
        """
        Prepares a grayscale image from Sartorious cell dataset to be preprocessed with superpixels

        Parameters
        ----------
        img : ndarray
            Input image, potentially grayscale
        log : bool
            Log to TB the intermediate steps

        Returns
        -------
        img : ndarray
            Input image (now in RGB) with white pixels highlighted and noise removal applied
        heatmap : ndarray
            Heatmap generated by openCV on the noise filtered image
        """
        # If its grayscale stack it
        if len(img.shape) == 2:  # 2D matrix (W x H)
            img = np.stack((img,) * 3, axis=-1)
        img = np.copy(img)  # For safety
        _mean = np.mean(img)
        if log: TBWriter().save_image(tag='all/original', image=img)

        # Canny edge detection
        canny = Superpixels.normalize(img)
        canny = cv2.Canny(canny, 100, 250)
        if log: TBWriter().save_image(tag='filtered/canny_detection', image=canny)

        # Filter input with canny edge blurred map
        img[cv2.blur(canny, (7, 7)) <= 10] = _mean
        if log: TBWriter().save_image(tag='filtered/canny', image=img)

        # Compute heatmap
        heatmap = Superpixels.normalize(np.copy(img))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        if log: TBWriter().save_image(tag='all/heatmap', image=heatmap)

        # Select white enough pixels
        _sum = np.sum(img, axis=2)
        img = np.copy(img)
        img[_sum < 2.0] = _mean
        if log: TBWriter().save_image(tag='filtered/white_highlight', image=img)

        # Remove noisy pixel
        # _tmp = Superpixels.normalize(img)
        # # _tmp[_tmp > 0] = 255  # Binarize
        # _tmp[cv2.blur(_tmp, (4, 4)) < 65] = 0
        # img[_tmp < 255] = _mean
        # if log: TBWriter().save_image(tag='filtered/white_highlight_noise_removed', image=img)

        # Standardize image
        img = (img - np.mean(img)) / np.std(img)
        if log: TBWriter().save_image(tag='filtered/standardized', image=img)

        return img, heatmap

    @staticmethod
    def normalize(img):
        return np.array(((img - img.min()) / (img.max() - img.min())) * 255, dtype=np.uint8)

    @staticmethod
    def overlay_border(img, border):
        """
        Set the pixels on the superpixel boundary to red

        """
        img_disp = img.copy().astype(np.uint8)
        img_disp[:,:,0][border] = 255
        img_disp[:,:,1][border] = 0
        img_disp[:,:,2][border] = 0
        return img_disp

    @staticmethod
    def encode_bin_to_RGB(bin_image):
        """
        This method goes from the mask where each pixel value represent
        the index of the superpixel it belong to, to a RGB encoded version
        where the index of the bin is encoded as: R + 255*G + 255^2*B

        Parameters
        ----------
        bin_image : ndarray
           a mask (W x H) where each pixel value represent the index of the superpixel it belong to

        Returns
        -------
        img : ndarray
            An image (W x H x 3) where the index of the superpixel a pixel belongs to is encoded. See above for details.
        """
        img = np.zeros([bin_image.shape[0], bin_image.shape[1], 3])  # The encoding is RGB
        img[:, :, 0] = bin_image % 255
        img[:, :, 1] = bin_image // 255
        img[:, :, 2] = bin_image // (255**2)
        return img

    @staticmethod
    def decode_RGB_to_bin(image):
        """ This method decoded the encoding perofmed in encode_bin_to_RGB()

        Parameters
        ----------
        image : ndarray
           An image (W x H x 3) where the index of the superpixel a pixel belongs to is encoded. See encode_bin_to_RGB() for details.

        Returns
        -------
        ndarray
            A mask (W x H) where each pixel value represent the index of the superpixel it belong to.
        """
        return image [:, :, 0] + image[:, :, 1] * 255 + image[:, :, 2] * (255**2)

    @staticmethod
    def rgb2gray(rgb):
        """Converts an RGB image to grayscale"""
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])