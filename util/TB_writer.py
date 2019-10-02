# Utils
import os
import logging
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class TBWriter(metaclass=Singleton):

    def __init__(self, log_dir):
        self.init(log_dir)

    def init(self, log_dir):
        """Init the output folder and the SummaryWriter object"""
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Wrapper around SummaryWriter method"""
        self.writer.add_text(tag=tag, text_string=text_string, global_step=global_step, walltime=walltime)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Wrapper around SummaryWriter method"""
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step, walltime=walltime)

    def close(self):
        """Wrapper around SummaryWriter method"""
        self.writer.close()

    def _tensor_to_image(self, image, normalize=False):
        """
        Tries to reshape, convert and do operations necessary to bring the image
        in a format friendly to be saved and logged to Tensorboard by
        save_image_and_log_to_tensorboard()

        Parameters
        ----------
        image : ?
            Image to be converted
        normalize : bool
            Specify wwhetherthe image should be normalized or not

        Returns
        -------
        image : ndarray [W x H x C]
            Image, as format friendly to be saved and logged to Tensorboard.

        """
        # Check if the data is still a Variable()
        if 'variable' in str(type(image)):
            image = image.data

        # Check if the data is still on CUDA
        if 'cuda' in str(type(image)):
            image = image.cpu()

        # Check if the data is still on a Tensor
        if 'Tensor' in str(type(image)):
            image = image.numpy()
        assert ('ndarray' in str(type(image)))  # Its an ndarray

        # Make a deep copy of the image
        image = np.copy(image)

        # Check that it does not have anymore the 4th dimension (from the mini-batch)
        if len(image.shape) > 3:
            assert (len(image.shape) == 4)
            image = np.squeeze(image)

        # Check that it is not single channel grayscale
        if len(image.shape) == 2:  # 2D matrix (W x H)
            image = np.stack((image,)*3, axis=-1)
        assert (len(image.shape) == 3)  # 3D matrix (W x H x C)

        # Check that the last channel is of size 3 for RGB
        if image.shape[2] != 3:
            assert (image.shape[0] == 3)
            image = np.transpose(image, (1, 2, 0))
        assert (image.shape[2] == 3)  # Last channel is of size 3 for RGB

        # Check that the range is [0:255]
        if image.min() < 0 or normalize:
            image = (image - image.min()) / (image.max() - image.min())
        if np.mean(image) < 1:
            image *= 255
        assert (image.min() >= 0)  # Data should be in range [0:255]

        return image.astype(np.uint8)

    def save_image(self, tag=None, image=None, global_step=None, normalize=False):
        """Utility function to save image in the output folder and also log it to Tensorboard.

        Parameters
        ----------
        writer : tensorboardX.writer.SummaryWriter object
            The writer object for Tensorboard
        tag : str
            Name of the image.
        image : ndarray [W x H x C]
            Image to be saved and logged to Tensorboard.
        global_step : int
            Epoch/Mini-batch counter.
        normalize : bool
            Specify wwhetherthe image should be normalized or not

        Returns
        -------
        None

        """
        # Ensuring the data passed as parameter is healthy
        image = self._tensor_to_image(image, normalize)

        # Log image to Tensorboard
        self.writer.add_image(tag=tag, img_tensor=image, global_step=global_step, dataformats='HWC')

        # Get output folder using the FileHandler from the logger.
        # (Assumes the file handler is the last one)
        output_folder = os.path.dirname(logging.getLogger().handlers[-1].baseFilename)

        if global_step is not None:
            dest_filename = os.path.join(output_folder, 'images', tag + '_{}.png'.format(global_step))
        else:
            dest_filename = os.path.join(output_folder, 'images', tag + '.png')

        if not os.path.exists(os.path.dirname(dest_filename)):
            os.makedirs(os.path.dirname(dest_filename))

        # Write image to output folder
        self.save_numpy_image(dest_filename, image)

        return

    def save_numpy_image(self, dest_filename, image):
        img = Image.fromarray(image)
        img.save(dest_filename)
        return

    def load_numpy_image(self, dest_filename):
        img = Image.open(dest_filename).convert('RGB')
        img = np.array(img)
        return img
