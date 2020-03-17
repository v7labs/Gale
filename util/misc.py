"""
General purpose utility functions.

"""

# Utils
import os
import os.path
import string
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


def to_capital_camel_case(s):
    """Converts a string to camel case.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Input string `s` converted to camel case.

    """
    return s[0].capitalize() + string.capwords(s, sep='_').replace('_', '')[1:] if s else s


def get_all_files_in_folders_and_subfolders(root_dir=None):
    """Get all the files in a folder and sub-folders.

    Parameters
    ----------
    root_dir : str
        All files in this directory and it's sub-folders will be returned by this method.

    Returns
    -------
    paths : list of str
        List of paths to all files in this folder and it's subfolders.
    """
    paths = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths


def has_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py.

    Parameters
    ----------
    filename : string
        path to a file
    extensions : list
        extensions to match against
    Returns
    -------
    bool
        True if the filename ends with one of given extensions, false otherwise.
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pil_loader(path, to_rgb=True):
    pic = Image.open(path)
    if to_rgb:
        pic = convert_to_rgb(pic)
    return pic


def convert_to_rgb(pic):
    if pic.mode == "RGB":
        pass
    elif pic.mode in ("CMYK", "RGBA", "P"):
        pic = pic.convert('RGB')
    elif pic.mode == "I":
        img = (np.divide(np.array(pic, np.int32), 2**16-1)*255).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    elif pic.mode == "I;16":
        img = (np.divide(np.array(pic, np.int16), 2**8-1)*255).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    elif pic.mode == "L":
        img = np.array(pic).astype(np.uint8)
        pic = Image.fromarray(np.stack((img, img, img), axis=2))
    else:
        raise TypeError(f"unsupported image type {pic.mode}")
    return pic


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


# The following function is important for COCO segmentation tasks.
# It's a legacy from V7, and can come in handy if a COCO runner is
# implemented in gale.
# from pycocotools import mask as coco_mask
# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         rles = coco_mask.frPyObjects(polygons, height, width)
#         mask = coco_mask.decode(rles)
#         if len(mask.shape) < 3:
#             mask = mask[..., None]
#         mask = torch.as_tensor(mask, dtype=torch.uint8)
#         mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks


def convert_to_polygon_sequence(polygon):
    path = []
    if len(polygon) == 0:
        return path
    elif isinstance(polygon[0], dict):
        for e in polygon:
            path.append(e['x'])
            path.append(e['y'])
        return path
    else:
        return polygon


def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def visualize_mask_output(image, masks, classes, colors, threshold=0.5):
    W, H = image.size
    outmask = np.zeros((H, W, 4), dtype=np.uint8)
    for mask, c in zip(masks, classes):
        color = np.append(colors[c], 128)
        outmask[mask > threshold] = color

    outmask = Image.fromarray(outmask)
    image.paste(outmask, (0, 0), outmask)
    return image


def get_colors(n, name='hsv'):
    '''
    It return n distintc RGB colors using a mapping function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    colors = list(map(plt.cm.get_cmap(name, n), range(n)))
    colors = list(map(lambda c: (np.array(c[:3])*255).astype(np.uint8), colors))
    return colors
