# Utils
import logging
import os
from multiprocessing import Pool

import numpy as np
# Torch related stuff
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image


def compute_mean_std(input_folder, inmem, workers, **kwargs):
    """
    Computes mean and std of all images present at target folder.

    Parameters
    ----------
    input_folder : String (path)
        Path to the dataset folder (see above for details)
    inmem : Boolean
        Specifies whether is should be computed i nan online of offline fashion.
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    mean : float
        Mean value of all pixels of the images in the input folder
    std : float
        Standard deviation of all pixels of the images in the input folder
    """
    # Sanity check on the folder
    if not os.path.isdir(input_folder):
        logging.error(f"Folder {input_folder} does not exist")
        raise FileNotFoundError

    # Load the dataset
    ds = datasets.ImageFolder(input_folder, transform=transforms.Compose([transforms.ToTensor()]))

    # Extract the actual file names and labels as entries
    file_names = np.asarray([item[0] for item in ds.imgs])

    # Compute mean and std
    mean, std = _cms_inmem(file_names) if inmem else _cms_online(file_names, workers)
    return mean, std


def _cms_online(file_names, workers):
    """
    Computes mean and image_classification deviation in an online fashion.
    This is useful when the dataset is too big to be allocated in memory.

    Parameters
    ----------
    file_names : List of String
        List of file names of the dataset
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    mean : double
    std : double
    """
    logging.info('Begin computing the mean')

    # Set up a pool of workers
    pool = Pool(workers+1)

    # Online mean
    results = pool.map(_return_mean, file_names)
    mean_sum = np.sum(np.array(results), axis=0)

    # Divide by number of samples in train set
    mean = mean_sum / file_names.size

    logging.info('Finished computing the mean')
    logging.info('Begin computing the std')

    # Online image_classification deviation
    results = pool.starmap(_return_std, [[item, mean] for item in file_names])
    std_sum = np.sum(np.array([item[0] for item in results]), axis=0)
    total_pixel_count = np.sum(np.array([item[1] for item in results]))
    std = np.sqrt(std_sum / total_pixel_count)
    logging.info('Finished computing the std')

    # Shut down the pool
    pool.close()

    return mean, std


# Loads an image with OpenCV and returns the channel wise means of the image.
def _return_mean(image_path):
    img = np.array(Image.open(image_path).convert('RGB'))
    mean = np.array([np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])]) / 255.0
    return mean


# Loads an image with OpenCV and returns the channel wise std of the image.
def _return_std(image_path, mean):
    img = np.array(Image.open(image_path).convert('RGB')) / 255.0
    m2 = np.square(np.array([img[:, :, 0] - mean[0], img[:, :, 1] - mean[1], img[:, :, 2] - mean[2]]))
    return np.sum(np.sum(m2, axis=1), 1), m2.size / 3.0


def _cms_inmem(file_names):
    """
    Computes mean and image_classification deviation in an offline fashion. This is possible only when the dataset can
    be allocated in memory.

    Parameters
    ----------
    file_names: List of String
        List of file names of the dataset
    Returns
    -------
    mean : double
    std : double
    """
    img = np.zeros([file_names.size] + list(np.array(Image.open(file_names[0]).convert('RGB')).shape))

    # Load all samples
    for i, sample in enumerate(file_names):
        img[i] = np.array(Image.open(sample).convert('RGB'))

    mean = np.array([np.mean(img[:, :, :, 0]), np.mean(img[:, :, :, 1]), np.mean(img[:, :, :, 2])]) / 255.0
    std = np.array([np.std(img[:, :, :, 0]), np.std(img[:, :, :, 1]), np.std(img[:, :, :, 2])]) / 255.0

    return mean, std


def get_class_weights(input_folder, workers, **kwargs):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    input_folder : String (path)
        Path to the dataset folder (see above for details)
    workers : int
        Number of workers to use for the mean/std computation

    Returns
    -------
    ndarray[double] of size (num_classes)
        The weights vector as a 1D array normalized (sum up to 1)
    """
    # Sanity check on the folder
    if not os.path.isdir(input_folder):
        logging.error(f"Folder {input_folder} does not exist")
        raise FileNotFoundError

    # Load the dataset
    ds = datasets.ImageFolder(input_folder, transform=transforms.Compose([transforms.ToTensor()]))

    logging.info('Begin computing class frequencies weights')

    if hasattr(ds, 'targets'):
        labels = ds.targets
    elif hasattr(ds, 'labels'):
        labels = ds.labels
    else:
        # This is a fail-safe net in case a custom dataset changed the name of the internal variables
        data_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=workers)
        labels = []
        for target, label in data_loader:
            labels.append(label)
        labels = np.concatenate(labels).reshape(len(ds))

    class_support = np.unique(labels, return_counts=True)[1]
    class_frequencies = class_support / len(labels)
    # Class weights are the inverse of the class frequencies
    class_weights = 1 / class_frequencies
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    class_weights /= class_weights.sum()

    logging.info('Finished computing class frequencies weights ')
    logging.info(f'Class frequencies (rounded): {np.around(class_frequencies * 100, decimals=2)}')
    logging.info(f'Class weights (rounded): {np.around(class_weights * 100, decimals=2)}')

    return class_weights

def compute_mean_std_graphs(dataset, **kwargs):
    """
    Computes mean and std of all node and edge features present in the given ParsedGxlDataset (see gxl_parser.py).

    Parameters
    ----------
    input_folder : ParsedGxlDataset
        Dataset object (see above for details)

    # TODO implement online version

    Returns
    -------
    node_features : {"mean": list, "std": list}
        Mean and std value of all node features in the input dataset
    edge_features : {"mean": list, "std": list}
        Mean and std value of all edge features in the input dataset
    """
    if dataset.data.x is not None:
        logging.info('Begin computing the node feature mean and std')
        nodes = _get_feature_mean_std(dataset.data.x)
        logging.info('Finished computing the node feature mean and std')
    else:
        nodes = {}
        logging.info('No node features present')

    if dataset.data.edge_attr is not None:
        logging.info('Begin computing the edge feature mean and std')
        edges = _get_feature_mean_std(dataset.data.edge_attr)
        logging.info('Finished computing the edge feature mean and std')
    else:
        edges = {}
        logging.info('No edge features present')

    return nodes, edges


def _get_feature_mean_std(torch_array):
    array = np.array(torch_array)
    return {'mean': [np.mean(col) for col in array.T], 'std': [np.std(col) for col in array.T]}


def get_class_weights_graphs(dataset, **kwargs):
    """
    Get the weights proportional to the inverse of their class frequencies.
    The vector sums up to 1

    Parameters
    ----------
    input_folder : ParsedGxlDataset
        Dataset object (see above for details)

    # TODO implement online version

    Returns
    -------
    ndarray[double] of size (num_classes)
        The weights vector as a 1D array normalized (sum up to 1)
    """
    logging.info('Begin computing class frequencies weights')

    class_frequencies = np.array(dataset.config['class_freq'][1])
    # Class weights are the inverse of the class frequencies
    class_weights = 1 / class_frequencies
    # Normalize vector to sum up to 1.0 (in case the Loss function does not do it)
    class_weights /= class_weights.sum()

    logging.info('Finished computing class frequencies weights ')
    logging.info(f'Class frequencies (rounded): {np.around(class_frequencies * 100, decimals=2)}')
    logging.info(f'Class weights (rounded): {np.around(class_weights * 100, decimals=2)}')

    return class_weights