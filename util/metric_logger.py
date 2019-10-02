# Utils
import logging
import traceback
from abc import abstractmethod
from collections import deque

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import seaborn as sns
import torch
from PIL import Image

mpl.use('Agg')  # To facilitate plotting on a headless server


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MetricLogger(metaclass=Singleton):

    def __init__(self):
        self._meters = None
        self._postfix = None
        self.reset()

    def reset(self, postfix=""):
        self._meters = {}
        self._postfix = postfix

    def set_postfix(self, string):
        self._postfix = string

    def add_scalar_meter(self, tag: str):
        self.add_meter(tag, ScalarValue())

    def add_confusion_matrix_meter(self, tag: str, num_classes: int):
        self.add_meter(tag, ConfusionMatrix(num_classes))

    def add_meter(self, tag, meter):
        assert isinstance(meter, Meter)
        self._meters[tag + self._postfix] = meter

    def update(self, key, **kwargs):
        self._get_meter(key).update(**kwargs)

    def _get_meter(self, key):
        if key in self._meters:
            return self._meters[key]
        if key + self._postfix in self._meters:
            return self._meters[key + self._postfix]
        logging.error(f"\nMeter '{key}' not existing. Current postfix={self._postfix}.")
        logging.warning(f"Create a meter before using it with add_meter().")
        traceback.print_exc()
        raise SystemExit

    def __iter__(self):
        return self._meters.items().__iter__()

    def __next__(self):
        return self._meters.items().__next__()

    def __getitem__(self, item):
        return self._get_meter(item)

    def __contains__(self, item):
        return item in self._meters

    def __len__(self):
        return len(self._meters.items())


class Meter(object):
    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError


class ScalarValue(Meter):
    """
    Track a series of values and provide access to smoothed values
    over a window or the global series average.
    """

    def __init__(self, window_size=1):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.item()
        assert isinstance(value, (float, int))
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]


class ConfusionMatrix(Meter):

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, p, t):
        """
        Update the confusion matrix with the new entries

        Parameters
        ----------
        p : ndarray[int]
        t : ndarray[int]
            Prediction and target arrays of integers
        """
        # Better safe than sorry ;)
        assert isinstance(p, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert p.size == t.size
        assert all((p >= 0) & (p < self.num_classes))
        assert all((t >= 0) & (t < self.num_classes))
        n = self.num_classes
        self.mat += np.bincount(n * p + t, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.mat
        sup = h.sum(0)  # Support
        accuracy = np.diag(h).sum() / sup.sum()
        recall = np.diag(h) / h.sum(1)
        precision = np.diag(h) / sup
        iou = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
        return accuracy, precision, recall, iou, sup

    def make_heatmap(self, class_names=None):
        """
        This function produces a heatmap of confusion matrix.

        Adapted from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

        Parameters
        ----------
        class_names: List[str]
            Names of the classes, sorted

        Returns
        -------
        data : numpy.ndarray
            Contains an RGB image of the plotted confusion matrix
        """
        if class_names is None:
            class_names = list(range(0, self.num_classes))
        df_cm = pd.DataFrame(self.mat, index=class_names, columns=class_names)

        plt.style.use(['seaborn-white', 'seaborn-paper'])
        fig = plt.figure(figsize=(15, 15))
        plt.tight_layout()

        # Disable class labels if there are too many rows/columns in the confusion matrix.
        annot = False if self.mat.size > 100 else True
        try:
            heatmap = sns.heatmap(df_cm, annot=annot, fmt="d", cmap=plt.get_cmap('Blues'), annot_kws={"size": 14})
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        plt.close()
        return data

    def get_classification_report(self, class_names=None):
        """
        This routine computes and prints on Tensorboard TEXT a classification
        report with F1 score, Precision, Recall and similar metrics computed
        per-class.

        Parameters
        ----------
        class_names: List[str]
            Names of the classes, sorted

        Returns
        -------
            None
        """
        if class_names is None:
            class_names = list(range(0,self.num_classes))

        acc, pre, rec, iou, sup = self.compute()
        # The weird formatting is a fix for TB writer.
        # Its an ugly workaround to have it printed nicely in the TEXT section of TB.
        s = (
            f"\n\n       "
            f"{'{:>20}'.format(' ')} \t"                + f"{''.join(['{:^20}'.format(i) for i in class_names])}\n\n       "
            f"{'{:>20}'.format('support')}:\t"          + f"{''.join(['{:^20}'.format(i) for i in sup])}\n\n       "
            f"{'{:>20}'.format('precision')}:\t"        + f"{''.join(['{:^20.1f}'.format(i) for i in pre * 100])}\n\n       "
            f"{'{:>20}'.format('recall')}:\t"           + f"{''.join(['{:^20.1f}'.format(i) for i in rec * 100])}\n\n       "
            f"{'{:>20}'.format('IoU')}:\t"              + f"{''.join(['{:^20.1f}'.format(i) for i in iou * 100])}\n\n       "
            f"{'{:>20}'.format('mean IoU')}:\t"         + f"{iou.mean() * 100:.1f}\n\n       "
            f"{'{:>20}'.format('accuracy')}:\t"   + f"{acc * 100:.1f}\n\n       "
        )
        return s









# TODO: syncronize with Michele
def visualize_in_tensorboard(writer, items_dic, start_iter, mode, losses_dic=None):
    # im, out_v, out_e, polygon, pred_coords,

    # ve_maps = torch.cat([torch.stack([v_map, e_map]) for v_map, e_map in zip(out_v, out_e)])

    # imsv = ims.detach().cpu().numpy().squeeze().transpose((2, 3, 1))
    # visualize only the first image for now

    im0 = items_dic["image"]
    # print(im0.shape)
    imv0 = im0[0][:3, :, :].detach().cpu()  # .numpy().squeeze().transpose((1, 2, 0))
    imv0 = transforms.Unnormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(imv0)
    imv0 = transforms.ToPILImage()(imv0)
    imv0 = np.array(imv0)

    im1 = items_dic["target_mask"]
    # print(im1.shape)
    imv1 = im1[0].detach().cpu().numpy()
    imv1 = np.asarray(imv1, dtype=np.uint8).copy()
    imv1[imv1 == 255] = 128
    imv1[imv1 == 1] = 255

    im2 = items_dic["output"]["out"]
    # print(torch.unique(im2))
    im2 = torch.nn.functional.softmax(im2, 0)
    # print(torch.unique(im2))
    # print(im2.shape)
    imv2 = im2[0].detach().cpu().numpy()
    imv2 = (imv2[1] > 0.55) * 255
    imv2[imv2 == 128] = 128
    # print(np.unique(imv2))

    im0_title = "input image"
    im1_title = "target mask"
    im2_title = "output mask"
    # im3_title = 'prediction'

    # _imv = imv.copy()
    # _imv = np.ascontiguousarray(imv, dtype=np.int32)

    # mask = Image.fromarray(mask)
    imv2 = Image.fromarray(imv2.astype(np.uint8))
    imv2 = np.array(imv2)

    # print(imv0.size)
    # print(imv1.shape)
    # print(imv2.shape)

    writer.add_image(
        im0_title, imv0, global_step=start_iter, dataformats="HWC"
    )  # _imv[:,:,::-1]
    writer.add_image(
        im1_title, imv1, global_step=start_iter, dataformats="HW"
    )  # _imv[:,:,::-1]
    writer.add_image(
        im2_title, imv2, global_step=start_iter, dataformats="HW"
    )  # _imv[:,:,::-1]

    # grid = torchvision.utils.make_grid(ims)
    # writer.add_image(map_title, grid)

    if losses_dic:
        for loss_name, loss_value_list in losses_dic.items():
            avg = False
            if avg:
                loss_value = np.mean(loss_value_list)
                writer.add_scalar(loss_name, loss_value)
            else:
                [
                    writer.add_scalar(
                        loss_name + "/" + mode, loss_value, global_step=start_iter + i
                    )
                    for i, loss_value in enumerate(loss_value_list)
                ]


def visualize_graph_in_tensorboard(writer, model, model_input):
    with writer as w:
        w.add_graph(model, model_input)
