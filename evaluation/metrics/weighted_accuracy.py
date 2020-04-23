# Utils
import logging
import torch
import numpy as np

def weighted_accuracy(predicted, target, class_weights):
    """TODO: Computes the weighted accuracy

    Parameters
    ----------
    predicted : torch.FloatTensor
        The predicted output values of the model.
        The size is batch_size x num_classes
    target : torch.LongTensor
        The ground truth for the corresponding output.
        The size is batch_size x 1
    class_weights : ?
        class weights (like used by the criterion) by which the accuracy will be weight by

    Returns
    -------
    res : list
        List containing the computed accuracy (list because accuracy metric also returns a list)

    """
    with torch.no_grad():
        if len(predicted.shape) != 2:
            logging.error('Invalid input shape for prediction: predicted.shape={}'
                          .format(predicted.shape))
            return None
        if len(target.shape) != 1:
            logging.error('Invalid input shape for target: target.shape={}'
                          .format(target.shape))
            return None

        if len(predicted) == 0 or len(target) == 0 or len(predicted) != len(target):
            logging.error('Invalid input for accuracy: len(predicted)={}, len(target)={}'
                          .format(len(predicted), len(target)))
            return None

    topk = (1,)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = predicted.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res