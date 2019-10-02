# Utils
import logging
import numpy as np
from collections import defaultdict
from itertools import chain
from operator import methodcaller

# Torch related stuff
import torch

def _cat_tensors(tensors):
    """
    Concatenate a list of of tensors into a single tensor of size:
    len(list) x D x max(W) x max(H)

    Parameters
    ----------
    tensors : list(torch.Tensor)
        List containing all the tensor to be concatenated

    Returns
    -------
    torch.Tensor : images[0].dtype
        A tensor of size `batch_shape` containing all the images stacked
    """
    def _cat_regular_tensor(tensors):
        # Batch shape is: b_size x Channels (e.g RBG) x W x H
        batch_shape = [len(tensors), *tensors[0].shape]
        batch = torch.zeros(batch_shape, dtype=tensors[0].dtype)
        for t, batch_t in zip(tensors, batch):
            batch_t.copy_(t)
        return batch

    def _cat_padded_tensor(tensors):
        # Batch shape is: b_size x Channels (e.g RBG) x W x H
        batch_shape = [len(tensors), *np.array([img.shape for img in tensors]).max(axis=0)]
        batch = torch.zeros(batch_shape, dtype=tensors[0].dtype)
        for img, pad_img in zip(tensors, batch):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        return batch

    # Check if padding is needed
    if len(np.unique(np.array([t.shape for t in tensors]), axis=0)) > 1:
        return _cat_padded_tensor(tensors)
    else:
        return _cat_regular_tensor(tensors)


def _cat_dict(dicts):
    """
    Concatenate a list of dicts into a single dict by concatenating its content recursively

    Parameters
    ----------
    dicts : list(dict)
        Python dictionary containing all the target to be concatenated

    Returns
    -------
    batch : dict
        A dict which aggregates the list of dict receive as input
    """
    # Initialise defaultdict of lists
    batch = defaultdict(list)
    # Iterate dictionary items
    dict_items = map(methodcaller('items'), dicts)
    for k, v in chain.from_iterable(dict_items):
        batch[k].append(v)
    # Recursively process the values
    for k, v in batch.items():
        batch[k] = _cat(v)
    return batch


def _cat(cat_list):
    """
    This method is a dispatcher for the other cat_* methods

    Parameters
    ----------
    cat_list : list
        List of things to concatenate

    Returns
    -------
    Delegated to dispatched method
    """
    # Check and detect type
    t = np.unique(np.array([str(type(c)) for c in cat_list]))
    if len(t) != 1:
        logging.error("Multiple types found in the same list. This is not supported. Aborting.")
        raise SystemError
    else:
        t = t[0]

    # Dispatch
    if 'Tensor' in t:
        return _cat_tensors(cat_list)
    if 'dict' in t:
        return _cat_dict(cat_list)

    # When all else fails...
    return torch._utils.collate.default_collate(cat_list)

def collate_fn(batch):
    """
    This function collates all the dataloder __get_item__ into a single coherent batch.
    The functionality is dispatched to _cat

    Parameters
    ----------
    batch : list(tuples)
        This is the list of tuples composed by aggregating the dataloader __get_item__ for the entire minibatch.

    Returns
    -------
         An aggregated form of the batch received as input, split into separate items

    """

    return tuple(_cat(list(x)) for x in zip(*batch))